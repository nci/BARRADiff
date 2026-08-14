import torch
import cv2
import numpy as np
import xarray as xr
import math
from matplotlib import pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta


from data.training_data import get_c2_training_pair_templates

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
    #image1 = np.moveaxis(image1, 0, -1)
    xr.DataArray(image1).plot.imshow(ax=ax1, robust=True, cmap=cmap, add_colorbar=False)
    plt.title(image1_title, fontsize=26)
    plt.axis('off')

    ax2 = fig.add_subplot(212)
    #image2 = np.moveaxis(image2, 0, -1)
    xr.DataArray(image2).plot.imshow(ax=ax2, robust=True, cmap=cmap, add_colorbar=False)
    plt.title(image2_title, fontsize=26)
    plt.axis('off')

    plt.subplots_adjust(wspace=0.025, hspace=0.05)

    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    data = torch.load('../barra2_fsdp_inference_v3_more_no_var_embed/00000-edm2-img512-xxl/tas_global_inp_vs_ldm-cfg2.5-ema0.05.pth', weights_only=False)

    """
    training_templates = get_c2_training_pair_templates()
    var_name = 'tas'
    
    file_template = None
    for tpl in training_templates:
        inp_file, inp_level, tgt_file = tpl

        inp_v_name = inp_file.split('/')[-1].split('_')[0]
        v_name = tgt_file.split('/')[-1].split('_')[0]
        
        if v_name == var_name:
            file_template = tpl
            break
        
    assert file_template is not None, f'variable {var_name} not found'

    query_time = datetime(2022, 1, 1, 1, 0, 0)
    inp_data = get_era5_data(query_time, file_template)
    
    inp_upsample_h = 6.283950617283951
    inp_upsample_w = 6.2105263157894735
    H = math.ceil(inp_data.shape[-2] * inp_upsample_h)
    W = math.ceil(inp_data.shape[-1] * inp_upsample_w)

    inp_data = cv2.resize(inp_data, (W, H), interpolation=cv2.INTER_CUBIC)
    input_image = np.concatenate([inp_data[None],]*3, axis=0)
    inp_v_name = '2t'

    save_data = {
        'input': input_image[0],
        'prediction': pred[0],
        'time': query_time,
        'barra2_var': var_name,
        'era5_var': inp_v_name,
    }
    torch.save(save_data, 'aaaa.pth')
    """
    
    input_image = data['input']
    pred = data['prediction']
    
    plot(input_image, pred, 'LDM input (ERA5 2t)', 'LDM prediction (BARRA2 AUST-04 tas)', 'tas_global_inp_vs_ldm-cfg2.5-ema0.05.png')
    
    plot(input_image, pred, 'LDM input (ERA5 2t)', 'LDM prediction (BARRA2 AUST-04 tas)', 'tas_global_inp_vs_ldm-cfg2.5-ema0.05.jpg')
