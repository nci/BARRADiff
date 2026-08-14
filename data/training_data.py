import xarray as xr
import cv2
import math
#import torch
#from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta

def get_r2_training_pair_templates():
    return [
    ('/g/data/rt52/era5/single-levels/reanalysis/2t/{era5_year}/2t_era5_oper_sfc_{era5_ts}.nc', None,
    '/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/tas/v20231001/tas_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{barra_ts}.nc'),
    
    ('/g/data/rt52/era5/pressure-levels/reanalysis/z/{era5_year}/z_era5_oper_pl_{era5_ts}.nc', 500,
     '/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/zg500/v20231001/zg500_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{barra_ts}.nc'),
    
    ('/g/data/rt52/era5/single-levels/reanalysis/10u/{era5_year}/10u_era5_oper_sfc_{era5_ts}.nc', None,
     '/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/uas/v20231001/uas_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{barra_ts}.nc'),
    
    ('/g/data/rt52/era5/single-levels/reanalysis/10v/{era5_year}/10v_era5_oper_sfc_{era5_ts}.nc', None,
     '/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/vas/v20231001/vas_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{barra_ts}.nc'),
  ]

def get_c2_training_pair_templates():
    sfc_era5_root = '/g/data/rt52/era5/single-levels/reanalysis'
    pl_era5_root = '/g/data/rt52/era5/pressure-levels/reanalysis'
    barra2_root = '/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr'
    
    templates = []
    
    sfc_vars = [
        ('2t', 'tas'),
        ('10u', 'uas'),
        ('10v', 'vas'),
        ('sp', 'ps'), # Surface pressure
        ('msl', 'psl'), # Sea level pressure
        ('tcwv', 'prw'), # Total column vertically-integrated water vapour
        ('tp',  'pr'), # total precipitation
        ('lsrr', 'prra'), # Large Scale Rainfall Rate (kg m-2 s-1)
        ('tcc', 'clt'), # Total Cloud Cover Percentage (%)
        ('strdc', 'rldscs'), # Surface Downwelling Clear-Sky Longwave Radiation (W m-2)
        ('strd', 'rlds'), # Surface Downwelling Longwave Radiation (W m-2)

        # https://opus.nci.org.au/spaces/NDP/pages/264241304/Known+Issues+-+BOM+BARRA2+ob53
         ('cape', 'CAPE'), # Convective Available Potential Energy (J kg-1)
    ]

    for (era5_var, barra2_var) in sfc_vars:
        templates.append( (
            f'{sfc_era5_root}/{era5_var}/{{era5_year}}/{era5_var}_era5_oper_sfc_{{era5_ts}}.nc', 
            None,
            f'{barra2_root}/{barra2_var}/latest/{barra2_var}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{{barra_ts}}.nc',
        ) )
        
    pl_vars = [
        ('t', 'ta'),
        ('u', 'ua'),
        ('v', 'va'),
        ('w', 'wap'), # vertical wind
        ('z', 'zg'), # geopotential
        ('q', 'hus'),  # specific humidity
    ]
    
    for level in [200, 500, 700, 850]:
        for (era5_var, barra2_var) in pl_vars:
            barra2_var = f'{barra2_var}{level}'
            templates.append( (
                f'{pl_era5_root}/{era5_var}/{{era5_year}}/{era5_var}_era5_oper_pl_{{era5_ts}}.nc', 
                level,
                f'{barra2_root}/{barra2_var}/latest/{barra2_var}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{{barra_ts}}.nc',
            ) )

    return templates


def get_target_shape(query_time, templates):
    year = query_time.year
    month = query_time.month
    hour = query_time.hour

    start_date = datetime(year, month, 1)
    end_date = start_date + relativedelta(months=1) - relativedelta(days=1)

    start_ts = start_date.strftime('%Y%m%d')
    end_ts = end_date.strftime('%Y%m%d')
    era5_ts = f'{start_ts}-{end_ts}'

    barra_ts = start_date.strftime('%Y%m')
    barra_ts = f'{barra_ts}-{barra_ts}'

    tgt_vars = []

    for t in templates:
        assert t[-1] is not None

    q_time = query_time.strftime('%Y-%m-%dT%H:00:00')
    for _, tpl in enumerate(templates):
        inp_tpl, pl, tgt_tpl = tpl

        tgt_file = tgt_tpl.format(barra_ts=barra_ts)
        with xr.open_dataset(tgt_file) as ds:
            var_name = list(ds.keys())[0]
            tgt_vars.append(ds[var_name].shape)

    return tgt_vars

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

def get_training_sample(query_time, crop_x_start, crop_y_start, crop_x_size, crop_y_size, templates):
    year = query_time.year
    month = query_time.month
    hour = query_time.hour

    start_date = datetime(year, month, 1)
    end_date = start_date + relativedelta(months=1) - relativedelta(days=1)

    start_ts = start_date.strftime('%Y%m%d')
    end_ts = end_date.strftime('%Y%m%d')
    era5_ts = f'{start_ts}-{end_ts}'

    barra_ts = start_date.strftime('%Y%m')
    barra_ts = f'{barra_ts}-{barra_ts}'

    inp_vars = []
    tgt_vars = []

    num_inp_vars = len([t for t in templates if t[0] is not None])
    num_tgt_vars = len([t for t in templates if t[-1] is not None])

    inp_arr = np.empty((num_inp_vars, crop_y_size, crop_x_size), dtype=np.float32)
    tgt_arr = np.empty((num_tgt_vars, crop_y_size, crop_x_size), dtype=np.float32)

    inp_idx = 0
    tgt_idx = 0

    tgt_lonlat = None
    q_time = query_time.strftime('%Y-%m-%dT%H:00:00')
    for _, tpl in enumerate(templates):
        inp_tpl, pl, tgt_tpl = tpl

        if tgt_tpl is not None:
            tgt_file = tgt_tpl.format(barra_ts=barra_ts)
            with xr.open_dataset(tgt_file) as ds:
                var_name = list(ds.keys())[0]
                tgt_vars.append(var_name)

                tgt_lon_start = ds['lon'].values[crop_x_start]
                tgt_lon_end = ds['lon'].values[crop_x_start+crop_x_size-1]

                tgt_lat_start = ds['lat'].values[crop_y_start]
                tgt_lat_end = ds['lat'].values[crop_y_start+crop_y_size-1]

                t_idx = np.argmin(np.abs(ds['time'].values - np.datetime64(q_time)))

                indexer = {
                  'time': ds['time'].values[t_idx],
                  'lon': slice(tgt_lon_start, tgt_lon_end),
                  'lat': slice(tgt_lat_start, tgt_lat_end),
                }
                ds = ds.sel(indexer)
                val = ds[var_name].values
                if np.any(np.isnan(val)):
                    val = impute_nan(val)
                val[np.isnan(val)] = 0
                tgt_arr[tgt_idx] = val
                tgt_idx += 1

            if tgt_lonlat is None:
                tgt_lon = ds['lon'].values.astype(np.float32)
                tgt_lat = ds['lat'].values.astype(np.float32)
                tgt_lonlat = np.meshgrid(tgt_lon, tgt_lat, indexing='xy')
                tgt_lonlat = np.stack(tgt_lonlat)

        if inp_tpl is not None:
            inp_file = inp_tpl.format(era5_year=year, era5_ts=era5_ts)
            with xr.open_dataset(inp_file) as ds:
                var_name = list(ds.keys())[0]
                var_name_repr = var_name
                if pl is not None:
                    var_name_repr = f'{var_name}{pl}'
                inp_vars.append(var_name_repr)

                inp_lon_start = ds['longitude'][ds['longitude'] < tgt_lon_start].values[-1]
                inp_lon_end = ds['longitude'][ds['longitude'] > tgt_lon_end].values[0]

                ## ERA 5 has latitude flipped compared to BARRA
                lat = -ds['latitude']
                inp_lat_start = lat[lat < tgt_lat_end].values[-1]
                inp_lat_end = lat[lat > tgt_lat_start].values[0]

                t_idx = np.argmin(np.abs(ds['time'].values - np.datetime64(q_time)))

                indexer = {
                    'time': ds['time'].values[t_idx],
                    'longitude': slice(inp_lon_start, inp_lon_end),
                    'latitude': slice(inp_lat_start, inp_lat_end),
                }
                if pl is not None:
                    indexer['level'] = pl

                ds = ds.sel(indexer)

                inp = ds[var_name].values[::-1]
                if np.any(np.isnan(inp)):
                    inp = impute_nan(inp)
                inp[np.isnan(inp)] = 0
                inp_arr[inp_idx] = cv2.resize(inp, (crop_x_size, crop_y_size), interpolation=cv2.INTER_CUBIC)
                #inp_arr[inp_idx] = F.interpolate(torch.from_numpy(inp.copy()[None, None]),
                #        (crop_y_size, crop_x_size), mode='bicubic', align_corners=True).squeeze().numpy()
                inp_idx += 1

    return {
        'input': inp_arr,
        'target': tgt_arr,
        'inp_vars': inp_vars,
        'tgt_vars': tgt_vars,
        'tgt_lonlat': tgt_lonlat,
    }

def plot(image1, image2, image1_title, image2_title, output_file_path):
    fig = plt.figure(figsize=(22, 11), dpi=80)
    ax1 = fig.add_subplot(121)
    image1 = np.moveaxis(image1, 0, -1)
    xr.DataArray(image1).plot.imshow(ax=ax1, robust=True)
    plt.title(image1_title, fontsize=22)
    plt.axis('off')

    ax2 = fig.add_subplot(122)
    image2 = np.moveaxis(image2, 0, -1)
    xr.DataArray(image2).plot.imshow(ax=ax2, robust=True)
    plt.title(image2_title, fontsize=22)
    plt.axis('off')

    plt.subplots_adjust(wspace=0.025, hspace=0.05)

    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    query_time = datetime(2022, 1, 1, 1, 0, 0)
    templates = get_c2_training_pair_templates()
    
    import os
    import shutil
    import torch
    
    output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_plots')
    
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    q = torch.tensor([0.005, 0.995])

    for it, tpl in enumerate(templates):
        #print(f'{it+1}/{len(templates)}    {tpl}')

        era5_var_name = tpl[0].split('/')[-1].split('_')[0]
        barra2_var_name = tpl[-1].split('/')[-1].split('_')[0]

        target_shape = get_target_shape(query_time, [tpl,])[0]
        H, W = target_shape[-2:]

        data = get_training_sample(query_time, 0, 0, W, H, [tpl,])

        inp = np.concatenate([data['input'][:1],]*3, axis=0)
        target = np.concatenate([data['target'][:1],]*3, axis=0)

        inp_t = torch.from_numpy(inp)
        inp_min, inp_max = inp_t.quantile(q)

        target_t = torch.from_numpy(target)
        tgt_min, tgt_max = target_t.quantile(q)

        print(f'{it+1}/{len(templates)}', era5_var_name, inp_min.item(), inp_max.item(), np.mean(inp), \
              barra2_var_name, tgt_min.item(), tgt_max.item(), np.mean(target), \
              'barra2/era5', np.mean(target) / np.mean(inp))

        output_file = os.path.join(output_root, f'{barra2_var_name}.png')

        inp = (inp - inp_min.item()) / (inp_max.item() - inp_min.item() + 1e-8)
        target = (target - tgt_min.item()) / (tgt_max.item() - tgt_min.item() + 1e-8)
        plot(target, inp, f'BARRA2 {barra2_var_name}', f'ERA5 {era5_var_name}', output_file)
