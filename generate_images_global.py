# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the given model."""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
curr_dir = os.path.dirname(os.path.realpath(__file__))
os.environ['HF_HOME'] = os.path.join(curr_dir, 'hf_cache')

import re
import warnings
import click
import pickle
import xarray as xr
import numpy as np
import torch
import cv2
import sys
import logging
import dnnlib
import math
import xarray as xr
import zarr
from matplotlib import pyplot as plt
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from diffusers.models import AutoencoderKL

from data.training_data import get_c2_training_pair_templates

warnings.filterwarnings('ignore', '`resume_download` is deprecated')
warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')
warnings.filterwarnings('ignore', '1Torch was not compiled with flash attention')

def get_logger(name):
    logger_name = f'_logger_{name}'
    if logger_name in sys.modules:
        return sys.modules[logger_name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(filename)s %(lineno)d %(levelname)s]: %(message)s'

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    sys.modules[logger_name] = logger
    return logger

def get_era5_data(query_time, file_tpl):
    year = query_time.year
    month = query_time.month
    hour = query_time.hour

    start_date = datetime(year, month, 1)
    end_date = start_date + relativedelta(months=1) - relativedelta(days=1)

    start_ts = start_date.strftime('%Y%m%d')
    end_ts = end_date.strftime('%Y%m%d')
    era5_ts = f'{start_ts}-{end_ts}'

    inp_tpl, pl = file_tpl[:2]

    inp_file = inp_tpl.format(era5_year=year, era5_ts=era5_ts)
    q_time = query_time.strftime('%Y-%m-%dT%H:00:00')
    with xr.open_dataset(inp_file) as ds:
        var_name = list(ds.keys())[0]

        t_idx = np.argmin(np.abs(ds['time'].values - np.datetime64(q_time)))

        indexer = {
            'time': ds['time'].values[t_idx],
        }
        if pl is not None:
            indexer['level'] = pl

        ds = ds.sel(indexer)

        inp = ds[var_name].values[::-1]
        if np.any(np.isnan(inp)):
            inp = impute_nan(inp)
        inp[np.isnan(inp)] = 0

        return inp

def plot(image1, image2, image1_title, image2_title, output_file_path):
    H, W = image1.shape[-2:]
    dpi = 300
    fig_h = H / dpi
    fig_w = W / dpi
    cmap = 'viridis'

    fig = plt.figure(figsize=(fig_w, fig_h*2), dpi=dpi)
    ax1 = fig.add_subplot(211)
    xr.DataArray(image1).plot.imshow(ax=ax1, robust=True, cmap=cmap, add_colorbar=False)
    plt.title(image1_title, fontsize=26)
    plt.axis('off')

    ax2 = fig.add_subplot(212)
    xr.DataArray(image2).plot.imshow(ax=ax2, robust=True, cmap=cmap, add_colorbar=False)
    plt.title(image2_title, fontsize=26)
    plt.axis('off')

    plt.subplots_adjust(wspace=0.025, hspace=0.05)

    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close(fig)

def impute_nan(image):
    H, W = image.shape[-2:]
    
    patch_size = 16
    
    num_h_patches = math.ceil((H - patch_size) / patch_size + 1)
    num_w_patches = math.ceil((W - patch_size) / patch_size + 1)

    for ih in range(num_h_patches):
        hh_start = ih * patch_size
        hh_end = hh_start + patch_size
        if hh_end > H:
            diff = hh_end - H
            hh_start -= diff
            hh_end -= diff

        for iw in range(num_w_patches):
            ww_start = iw * patch_size
            ww_end = ww_start + patch_size
            if ww_end > W:
                diff = ww_end - H
                ww_start -= diff
                ww_end -= diff
                
            patch = image[..., hh_start:hh_end, ww_start:ww_end]
            nan_mask = np.isnan(patch)
            
            while np.all(nan_mask):
                hh_start -= 4
                hh_end += 4
                
                if hh_start < 0:
                    hh_start = 0
                
                if hh_end > H:
                    diff = hh_end - H
                    hh_start -= diff
                    hh_end -= diff
                
                ww_start -= 4
                ww_end += 4
                
                if ww_start < 0:
                    ww_start = 0
                
                if ww_end > W:
                    diff = ww_end - H
                    ww_start -= diff
                    ww_end -= diff
                    
                patch = image[..., hh_start:hh_end, ww_start:ww_end]
                nan_mask = np.isnan(patch)
            
                if hh_start == 0 and hh_end >= H:
                    break
                    
                if ww_start == 0 and ww_end >= W:
                    break
            
            if np.any(nan_mask):
                means = np.nanmedian(patch, axis=(-2, -1), keepdims=True)
                patch[nan_mask] = means.squeeze()
                image[..., hh_start:hh_end, ww_start:ww_end] = patch

    return image

def blend_h(a, b, overlap):
    w = torch.arange(overlap, dtype=a.dtype, device=a.device) / (overlap - 1)
    w = w.view(1, 1, 1, -1)
    blend = a[..., -overlap:] * (1 - w) + b[..., :overlap] * w
    return torch.cat([blend, b[..., overlap:]], dim=-1)
    
def blend_v(a, b, overlap):
    w = torch.arange(overlap, dtype=a.dtype, device=a.device) / (overlap - 1)
    w = w.view(1, 1, -1, 1)
    blend = a[..., -overlap:, :] * (1 - w) + b[..., :overlap, :] * w
    return torch.cat([blend, b[..., overlap:, :]], dim=-2)

#----------------------------------------------------------------------------
# EDM sampler from the paper
# "Elucidating the Design Space of Diffusion-Based Generative Models",
# extended to support classifier-free guidance.

def edm_sampler(
    net, noise, cond, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    # Guided denoiser.
    def denoise(x, cond, t):
        Dx = net(x, cond, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        if gnet is None:
            ref_Dx = net(x, cond*0, t, labels).to(dtype)
        else:
            ref_Dx = gnet(x, cond, t, labels).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    #def denoise(x, cond, t):
    #    guidance = 2
    #    Dx = net(x, cond, t, labels).to(dtype)
    #    ref_Dx = net(x, cond*0, t, labels).to(dtype)
    #    return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        d_cur = (x_hat - denoise(x_hat, cond, t_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, cond, t_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

class RandomGenerator:
    def __init__(self, seed, device):
        super().__init__()
        self.device = device
        self.generator = torch.Generator(device).manual_seed(int(seed) % (1 << 32))

    def randn(self, *shape):
        return torch.randn(shape, device=self.device, generator=self.generator)
        
    def randn_like(self, inp):
        return torch.randn(inp.shape, dtype=inp.dtype, layout=inp.layout, device=inp.device, generator=self.generator)

def generate_images(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    ema_length,
    var_name,
    stats_root,
    outdir,                                     # Where to save the output images. None = do not save.
    patch_overlap       = 64,                   # Patch overlap
    seed                = 0,                    # List of random seeds.
    device              = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    assert sampler_kwargs['guidance'] >= 1.0
    
    logger = get_logger(__name__)

    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").eval().to(device)

    query_time = datetime(2022, 1, 1, 1, 0, 0)

    training_templates = get_c2_training_pair_templates()
    
    file_template = None
    for tpl in training_templates:
        inp_file, inp_level, tgt_file = tpl

        inp_v_name = inp_file.split('/')[-1].split('_')[0]
        v_name = tgt_file.split('/')[-1].split('_')[0]
        
        if v_name == var_name:
            file_template = tpl
            break
        
    assert file_template is not None, f'variable {var_name} not found'

    inp_stats_path = os.path.join(stats_root, f'{var_name}_inp.pt')
    assert os.path.exists(inp_stats_path)

    tgt_stats_path = os.path.join(stats_root, f'{var_name}_tgt.pt')
    assert os.path.exists(tgt_stats_path)

    inp_state_dict = torch.load(inp_stats_path)
    tgt_state_dict = torch.load(tgt_stats_path)

    stats_shape = (1, -1, 1, 1)
    stats = {
        'inp_mean': inp_state_dict['running_mean'].to(device=device).view(stats_shape),          
        'inp_std': inp_state_dict['running_var'].to(device=device).sqrt().view(stats_shape),
        'tgt_mean': tgt_state_dict['running_mean'].to(device=device).view(stats_shape),
        'tgt_std': tgt_state_dict['running_var'].to(device=device).sqrt().view(stats_shape),
    }

    inp_data = get_era5_data(query_time, file_template)
    
    inp_upsample_h = 6.283950617283951
    inp_upsample_w = 6.2105263157894735
    H = math.ceil(inp_data.shape[-2] * inp_upsample_h)
    W = math.ceil(inp_data.shape[-1] * inp_upsample_w)

    inp_data = cv2.resize(inp_data, (W, H), interpolation=cv2.INTER_CUBIC)
    input_image = np.concatenate([inp_data[None],]*3, axis=0)

    patch_size = 256
    stride = patch_size - patch_overlap
    downsample_factor = 8
    q = torch.tensor([0.005, 0.995], device=device)

    num_h_tiles = math.ceil((H - patch_size) / stride + 1)
    num_w_tiles = math.ceil((W - patch_size) / stride + 1)
    
    vae_recon = np.zeros_like(input_image)

    latent_H = math.ceil((num_h_tiles * stride + patch_size) / downsample_factor)
    latent_W = math.ceil((num_w_tiles * stride + patch_size) / downsample_factor)
    rnd = RandomGenerator(seed, device)
    noise = rnd.randn(1, 4, latent_H, latent_W)
    
    patches = {}
    patch_coords = {}
    for hh in range(num_h_tiles):
        hh_start = hh * stride
        hh_end = hh_start + patch_size
        
        if hh_end > H:
            diff = hh_end - H
            hh_end -= diff
            hh_start -= diff

        for ww in range(num_w_tiles):
            ww_start = ww * stride
            ww_end = ww_start + patch_size
            
            if ww_end > W:
                diff = ww_end - W
                ww_end -= diff
                ww_start -= diff
            
            input_patch = input_image[..., hh_start:hh_end, ww_start:ww_end] 
            with torch.inference_mode():
                inp = torch.from_numpy(input_patch).to(device)
                min_val, max_val = inp.quantile(q)
                range_val = max_val - min_val
                
                inp = (inp - min_val) / (range_val + 1e-8)
                inp.clamp_(min=0, max=1)
                inp = inp[None]
                
                posterior = vae.encode(inp).latent_dist
                z = posterior.mode()
                z = (z - stats['inp_mean']) / stats['inp_std'] * 0.5
                
                latent_hh_start = hh_start // downsample_factor
                latent_hh_end = latent_hh_start + z.shape[-2]
                latent_ww_start = ww_start // downsample_factor
                latent_ww_end = latent_ww_start + z.shape[-1]
                noise_patch = noise[..., latent_hh_start:latent_hh_end, latent_ww_start:latent_ww_end]
                
                z = edm_sampler(net=net, noise=noise_patch, cond=z, randn_like=rnd.randn_like, **sampler_kwargs)
                z = z * stats['tgt_std'] * 2 + stats['tgt_mean']

                out = vae.decode(z).sample.mean(dim=1, keepdims=True)
                out.clamp_(min=0, max=1)
                out = out * range_val + min_val
            
                if (hh, ww-1) in patches:
                    overlap = patch_coords[(hh, ww-1)][-1] - ww_start
                    if overlap > 0:
                        out = blend_h(patches[(hh, ww-1)], out, overlap)
            
                if (hh-1, ww) in patches:
                    overlap = patch_coords[(hh-1, ww)][1] - hh_start
                    if overlap > 0:
                        out = blend_v(patches[(hh-1, ww)], out, overlap)
                    
                patches[(hh, ww)] = out
                patch_coords[(hh, ww)] = (hh_start, hh_end, ww_start, ww_end)
            
            log_msg = f'h={hh+1}/{num_h_tiles}, w={ww+1}/{num_w_tiles}' + \
                f', h_start={hh_start+1}, h_end={hh_end}/{H}, w_start={ww_start+1}, w_end={ww_end}/{W}'
            logger.info(log_msg)
    
    for (hh, ww), patch in patches.items():
        hh_start = hh * stride
        hh_end = hh_start + patch_size
        
        if hh_end > H:
            diff = hh_end - H
            hh_end -= diff
            hh_start -= diff
        
        ww_start = ww * stride
        ww_end = ww_start + patch_size

        if ww_end > W:
            diff = ww_end - W
            ww_end -= diff
            ww_start -= diff
        
        vae_recon[..., hh_start:hh_end, ww_start:ww_end] = patch.cpu().numpy()

    if inp_level is not None:
        inp_v_name = f'{inp_v_name}{inp_level}'

    cfg_scale = sampler_kwargs['guidance']

    pred_file_path = os.path.join(outdir, f'{var_name}_global_inp_vs_ldm-cfg{cfg_scale:.1f}-ema{ema_length}.pth')
    save_data = {
        'input': input_image[0],
        'prediction': vae_recon[0],
        'time': query_time,
        'barra2_var': var_name,
        'era5_var': inp_v_name,
    }
    torch.save(save_data, pred_file_path)

    pred_plot_file_path = os.path.join(outdir, f'{var_name}_global_inp_vs_ldm-cfg{cfg_scale:.1f}-ema{ema_length}.png')
    plot(input_image[0], vae_recon[0], f'LDM input (ERA5 {inp_v_name})', f'LDM prediction (BARRA2 AUST-04 {var_name})', pred_plot_file_path)
    
    pred_plot_file_path = os.path.join(outdir, f'{var_name}_global_inp_vs_ldm-cfg{cfg_scale:.1f}-ema{ema_length}.jpg')
    plot(input_image[0], vae_recon[0], f'LDM input (ERA5 {inp_v_name})', f'LDM prediction (BARRA2 AUST-04 {var_name})', pred_plot_file_path)


#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--net',                      help='Main network pickle filename', metavar='PATH|URL',                type=str, default=None)
@click.option('--ema-length',               help='Main network EMA length', metavar='FLOAT',                        type=float, default=0.05)
@click.option('--var-name',                 help='BARRA2 variable name', metavar='STR',                              type=str, required=True)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--stats-root',               help='Where to load the means and stds', metavar='DIR',                  type=str, required=True)
@click.option('--seed',                     help='random seed', metavar='INT',            type=int, default=0, show_default=True)
@click.option('--patch-overlap',            help='Patch overlap', metavar='Int',   type=int, default=64)
@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--guidance',                 help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT',   type=float, default=1.0)
@click.option('--sigma_min',                help='Lowest noise level', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',                help='Highest noise level', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',                         type=float, default=1, show_default=True)

def cmdline(**opts):
    """Generate random images using the given model.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python generate_images.py --preset=edm2-img512-s-guid-dino --outdir=out

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img64-s-fid --outdir=out --subdirs --seeds=0-49999
    """
    opts = dnnlib.EasyDict(opts)

    # Validate options.
    if opts.net is None:
        raise click.ClickException('Please specify --net')

    if os.path.isdir(opts.net):
        run_dir = opts.net
        ema_length = f'{opts.ema_length:.3f}'
        pattern=r'network-snapshot-(\d+)-' + f'{ema_length}.pkl'
        fnames = [entry.name for entry in os.scandir(run_dir) if entry.is_file() and re.fullmatch(pattern, entry.name)]
        if len(fnames) == 0:
            raise Exception(f'net checkpoint is not found {run_dir}')
        pt_path = os.path.join(run_dir, max(fnames, key=lambda x: float(re.fullmatch(pattern, x).group(1))))

    logger = get_logger(__name__)
    logger.info(opts)
    
    logger.info(f'loading checkpoint {pt_path}')

    with dnnlib.util.open_url(pt_path, verbose=True) as f:
        data = pickle.load(f)
        
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    opts.net = data['ema'].eval().to(device)
    assert opts.net is not None

    logger.info(f'output directory {opts.outdir}')
    if not os.path.exists(opts.outdir):
        os.makedirs(opts.outdir)

    generate_images(**opts)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
