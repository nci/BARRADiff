import os
curr_dir = os.path.dirname(os.path.realpath(__file__))
os.environ['HF_HOME'] = os.path.join(curr_dir, 'hf_cache')

import math
import numpy as np
import xarray as xr
import torch
import zarr
from matplotlib import pyplot as plt
import time
from datetime import datetime
from diffusers.models import AutoencoderKL

from data.training_data import get_c2_training_pair_templates, get_target_shape, get_training_sample

def plot(org, vae, output_file_path):
    rmse = np.sqrt(np.mean((org - vae) ** 2))

    fig = plt.figure(figsize=(22, 11), dpi=80)
    ax1 = fig.add_subplot(121)
    org_image = np.moveaxis(org, 0, -1)
    xr.DataArray(org_image).plot.imshow(ax=ax1, robust=True)
    plt.title(f'original', fontsize=22)
    plt.axis('off')

    ax2 = fig.add_subplot(122)
    vae_image = np.moveaxis(vae, 0, -1)
    xr.DataArray(vae_image).plot.imshow(ax=ax2, robust=True)
    plt.title(f'vae (rmse={rmse:.5f})', fontsize=22)
    plt.axis('off')

    plt.subplots_adjust(wspace=0.025, hspace=0.05)

    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close(fig)

def blend_h(a, b, overlap):
    w = torch.arange(overlap) / (overlap - 1)
    w = w.view(1, 1, 1, -1)
    blend = a[..., -overlap:] * (1 - w) + b[..., :overlap] * w
    return torch.cat([blend, b[..., overlap:]], dim=-1)
    
def blend_v(a, b, overlap):
    w = torch.arange(overlap) / (overlap - 1)
    w = w.view(1, 1, -1, 1)
    blend = a[..., -overlap:, :] * (1 - w) + b[..., :overlap, :] * w
    return torch.cat([blend, b[..., overlap:, :]], dim=-2)

if __name__ == '__main__':
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    
    file_templates = get_c2_training_pair_templates()[:1]
    query_time = datetime(2022, 1, 1, 1, 0, 0)
    
    target_shapes = get_target_shape(query_time, file_templates)
    
    H, W = target_shapes[0][-2:]

    patch_size = 256
    stride = patch_size - 20
    q = torch.tensor([0.005, 0.995])

    num_h_tiles = math.ceil((H - patch_size) / stride + 1)
    num_w_tiles = math.ceil((W - patch_size) / stride + 1)
    
    target_image = np.zeros((3, H, W), dtype=np.float32)
    vae_recon = np.zeros_like(target_image)

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
            
            sample = get_training_sample(query_time, ww_start, hh_start, patch_size, file_templates)
            target_patch = sample['target'][:1]
            input_patch = sample['input'][:1]
            
            target_image[..., hh_start:hh_end, ww_start:ww_end] = np.concatenate([target_patch,]*3, axis=0)
            
            inp = input_patch
            x = target_patch
            
            with torch.inference_mode():
                min_val, max_val = torch.from_numpy(inp).quantile(q)
                range_val = max_val - min_val
                
                x = torch.from_numpy(x)
                x = (x - min_val) / range_val
                x.clamp_(min=0, max=1)
                x = torch.cat([x[None],]*3, dim=1)
                
                posterior = vae.encode(x).latent_dist
                z = posterior.mode()
                out = vae.decode(z).sample
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
            
            log_msg = f'h={hh+1}/{num_h_tiles}, w={ww+1}/{num_w_tiles+1}' + \
                f', h_start={hh_start+1}, h_end={hh_end}/{H}, w_start={ww_start+1}, w_end={ww_end}/{W}'
            print(log_msg)
    
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

    plot(target_image, vae_recon, 'tiled_plot.png')

