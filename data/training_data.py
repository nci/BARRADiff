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


# ----------------------------------------------------------------------------------------------
# ERA5 -> BARRA2 unit conversions
#
# Every (era5_var_name, barra2_var_name) pair below maps to the scalar coefficient `c`
# such that
#
#     <ERA5 variable value> * c ~= <BARRA2 variable value>
#
# in the "numerically close" sense: the two datasets never match exactly (different
# models / resolutions), but after applying `c` both fields live on the same scale of
# measurement.  Coefficients other than 1.0 are only introduced where the units of
# measurement are *drastically* different; pairs with identical (or equivalent) units
# always use 1.0, even when the two datasets show small relative gaps (e.g. 0.97, 1.08).
#
# The coefficients were worked out from the `units` attributes of the underlying
# netCDF files and verified empirically against full-domain patches returned by
# `get_training_sample()` (see unit_conversion_tests/ for the verification suite).
#
# Non-trivial conversions:
#   ('tp',   'pr')     ERA5 tp is metres of liquid-water equivalent *accumulated over the
#                      hour ending at the valid time* (units "m"); BARRA-C2 pr is the
#                      hourly-mean precipitation flux (kg m-2 s-1).  Converting an
#                      accumulated depth of water into a mass flux needs the density of
#                      water and the length of the accumulation window:
#                          c = 1000 kg m-3 / 3600 s = 0.27778
#                      Empirical check (2020-06, full AUST-04 domain): era5 tp mean
#                      9.20e-05 m, barra2 pr mean 3.43e-05 kg m-2 s-1, so
#                      pr / (tp * 1000/3600) = 1.34 (a ~34% cross-dataset gap), whereas
#                      dividing by 3600 alone (i.e. ignoring the water density factor)
#                      would leave a ~4800x gap.  ERA5 is internally consistent with
#                      this conversion: domain-mean(lsrr + crr) = 2.563e-05 kg m-2 s-1
#                      ~= domain-mean(tp) * 1000/3600 = 2.557e-05 kg m-2 s-1.
#   ('strd', 'rlds')   ERA5 strd / strdc are surface (clear-sky) downwelling longwave
#   ('strdc','rldscs') radiation *accumulated over the hour* (units "J m**-2");
#                      BARRA2 rlds / rldscs are mean fluxes (W m-2 = J s-1 m-2).  A
#                      per-hour energy amount becomes a mean flux when divided by the
#                      3600 s window: c = 1/3600 = 2.778e-04.
#                      (Empirical tgt/inp: 2.77e-04..2.87e-04.)
#   ('tcc',  'clt')    ERA5 tcc is a fraction in [0, 1] (units "(0 - 1)"); BARRA2 clt is
#                      a percentage (units "%"): c = 100.
#                      (Empirical tgt/inp: 101.6..122.4.)
#   ('z',    'zg*')    ERA5 z is geopotential (m**2 s**-2); BARRA2 zg is geopotential
#                      height (m).  Height = geopotential / g0 with the WMO standard
#                      gravity g0 = 9.80665 m s-2: c = 1/9.80665 = 0.10197.
#                      (Empirical tgt/inp: 0.10182..0.10214.)
#
# Documented same-unit exceptions (coefficient stays 1.0 - the observed gaps are *not*
# unit differences and correspond to known issues documented for BARRA2 on NCI Opus):
#   ('cape', 'CAPE')   both J kg-1.  BARRA-C2 CAPE is biased relative to ERA5 (observed
#                      tgt/inp mean ratio ~2.4..8.1) due to "inaccuracies in parcel
#                      ascent computation in Unified Model UM v13.0" (BARRA2 Known
#                      Issues page, NCI Confluence space NDP).
#   ('lsrr', 'prra')   both kg m-2 s-1.  BARRA-C2 1hr prra is the *maximum* over the
#                      1-hour window (cell_methods "time: maximum (interval: 1 hour)";
#                      BARRA2 FAQ), unlike ERA5's large-scale rain rate; in addition
#                      ERA5 puts most tropical rainfall into the convective flux (crr)
#                      while the convection-permitting BARRA-C2 resolves nearly all
#                      rainfall as "large-scale" (observed tgt/inp mean ratio ~7.6..18.6).
# ----------------------------------------------------------------------------------------------

_WATER_DENSITY = 1000.0      # kg m-3, density of liquid water
_SECONDS_PER_HOUR = 3600.0   # s, length of the ERA5 hourly accumulation window
_STANDARD_GRAVITY = 9.80665  # m s-2, WMO standard gravity (geopotential -> height)


def _build_unit_conversion_lookup():
    """Pre-compute the (era5_var_name, barra2_var_name) -> coefficient lookup table.

    Covers every mapping tuple returned by `get_c2_training_pair_templates()`, the same
    pairs with the pressure level appended to the ERA5 name (e.g. ('t500', 'ta500'), as
    used by the `inp_vars` entries of `get_training_sample()`), and the remaining
    BARRA-C2 hourly pressure levels for the same variables so that plausible template
    extensions keep working (the coefficients do not depend on the level).  The
    BARRA-R2 pairs from `get_r2_training_pair_templates()` - ('2t', 'tas'),
    ('z', 'zg500'), ('10u', 'uas'), ('10v', 'vas') - are covered by the same entries
    because BARRA-R2 uses identical variable names and units.
    """
    one = 1.0
    lookup = {
        # --- ERA5 single-level -> BARRA-C2 surface variables -----------------------
        ('2t', 'tas'): one,       # K -> K
        ('10u', 'uas'): one,      # m s-1 -> m s-1
        ('10v', 'vas'): one,      # m s-1 -> m s-1
        ('sp', 'ps'): one,        # Pa -> Pa
        ('msl', 'psl'): one,      # Pa -> Pa
        ('tcwv', 'prw'): one,     # kg m-2 -> kg m-2
        ('tp', 'pr'): _WATER_DENSITY / _SECONDS_PER_HOUR,  # m per hour -> kg m-2 s-1
        ('lsrr', 'prra'): one,    # kg m-2 s-1 -> kg m-2 s-1 (see note above)
        ('tcc', 'clt'): 100.0,    # fraction (0 - 1) -> %
        ('strdc', 'rldscs'): 1.0 / _SECONDS_PER_HOUR,      # J m-2 per hour -> W m-2
        ('strd', 'rlds'): 1.0 / _SECONDS_PER_HOUR,         # J m-2 per hour -> W m-2
        ('cape', 'CAPE'): one,    # J kg-1 -> J kg-1 (see note above)
    }

    # ERA5 stores a few single-level variables under different names inside its
    # netCDF files (the c2 template short name -> the variable name in the file):
    # alias them so the coefficient is also resolvable from the `inp_vars` names
    # that get_training_sample() reports.
    era5_sfc_file_name_aliases = {
        ('2t', 'tas'): 't2m',
        ('10u', 'uas'): 'u10',
        ('10v', 'vas'): 'v10',
    }
    for (era5_var, barra2_var), era5_file_var in era5_sfc_file_name_aliases.items():
        lookup[(era5_file_var, barra2_var)] = lookup[(era5_var, barra2_var)]

    # --- ERA5 pressure-level -> BARRA-C2 pressure-level variables -----------------
    pl_coeffs = {
        ('t', 'ta'): one,         # K -> K
        ('u', 'ua'): one,         # m s-1 -> m s-1
        ('v', 'va'): one,         # m s-1 -> m s-1
        ('w', 'wap'): one,        # Pa s-1 -> Pa s-1
        ('z', 'zg'): 1.0 / _STANDARD_GRAVITY,  # m**2 s**-2 -> m (geopotential height)
        ('q', 'hus'): one,        # kg kg-1 -> 1 (dimensionless mass fraction)
    }
    c2_levels = [200, 500, 700, 850]                              # levels used by the c2 templates
    extra_barra_c2_levels = [300, 400, 600, 750, 800, 900, 925, 950, 975, 1000]
    for (era5_base, barra2_base), coeff in pl_coeffs.items():
        for level in c2_levels + extra_barra_c2_levels:
            barra2_var = f'{barra2_base}{level}'
            lookup[(era5_base, barra2_var)] = coeff
            # alias with the level appended to the ERA5 name, as used in `inp_vars`
            lookup[(f'{era5_base}{level}', barra2_var)] = coeff

    return lookup


conversion_lookup = _build_unit_conversion_lookup()


def _var_names_from_pair_template(tpl):
    """Extract the (era5_var_name, barra2_var_name) pair from a training-pair template.

    The variable names are taken from the template file names in the same way as the
    module's `__main__` block, e.g.
        ('.../tp/{era5_year}/tp_era5_oper_sfc_{era5_ts}.nc', None,
         '.../pr/latest/pr_AUST-04_..._1hr_{barra_ts}.nc')  ->  ('tp', 'pr')
    """
    inp_tpl, pl, tgt_tpl = tpl
    era5_var_name = inp_tpl.split('/')[-1].split('_')[0]
    barra2_var_name = tgt_tpl.split('/')[-1].split('_')[0]
    return era5_var_name, barra2_var_name


def get_unit_conversion_coefficient(era5_var_name, barra2_var_name):
    """Return the unit conversion coefficient for an ERA5 -> BARRA2 variable pair.

    The returned scalar `c` satisfies `<ERA5 variable value> * c ~= <BARRA2 variable
    value>` (numerically close; the two datasets never match exactly).  It is looked up
    from the pre-computed `conversion_lookup` table, which contains every mapping tuple
    returned by `get_c2_training_pair_templates()`.

    Raises:
        KeyError: if the (era5_var_name, barra2_var_name) pair is not in the lookup
            table (unknown variable pair).
    """
    key = (era5_var_name, barra2_var_name)
    if key not in conversion_lookup:
        raise KeyError(
            f'No unit conversion coefficient is known for the ERA5 -> BARRA2 variable '
            f'pair {key!r}. Known pairs: {sorted(conversion_lookup.keys())}')
    return conversion_lookup[key]


def _validate_unit_conversion_lookup():
    """Guard: every c2 mapping tuple must have a pre-computed conversion coefficient.

    Runs once at import time (no I/O - `get_c2_training_pair_templates()` only builds
    path templates) so that adding a new template without working out its unit
    conversion fails loudly at import instead of silently producing mis-scaled
    training data.
    """
    for tpl in get_c2_training_pair_templates():
        era5_var_name, barra2_var_name = _var_names_from_pair_template(tpl)
        get_unit_conversion_coefficient(era5_var_name, barra2_var_name)


_validate_unit_conversion_lookup()


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
    """Load an ERA5 (input) / BARRA2 (target) training patch pair.

    Returns a dict with:
        'input':  (num_inp_vars, crop_y_size, crop_x_size) float32 array of raw ERA5 values
        'target': (num_tgt_vars, crop_y_size, crop_x_size) float32 array of raw BARRA2 values
        'inp_vars': list of ERA5 variable names (one per input channel)
        'tgt_vars': list of BARRA2 variable names (one per target channel)
        'tgt_lonlat': (2, crop_y_size, crop_x_size) meshgrid of target lon/lat, or None
        'unit_conversion': list with one entry per template, aligned with the order of
            `templates` (and hence with `inp_vars`/`tgt_vars` for the c2 pair
            templates).  Entry i is
            `get_unit_conversion_coefficient(era5_var, barra2_var)` for template i, or
            None if that template does not form an (ERA5, BARRA2) pair (missing input
            or target side).  For a single-template call it is a 1-element list.

    NOTE: the conversion coefficients are *not* applied to the returned arrays - the
    raw values of both datasets are returned unchanged, so that callers can decide
    whether/when to rescale.
    """
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

    # Unit conversion coefficients for every template, resolved up-front (before any
    # file I/O) so that an unknown variable pair fails fast with a clear error.
    unit_conversion_coefficient = []
    for tpl in templates:
        inp_tpl, pl, tgt_tpl = tpl
        if inp_tpl is None or tgt_tpl is None:
            unit_conversion_coefficient.append(None)
            continue
        era5_var_name, barra2_var_name = _var_names_from_pair_template(tpl)
        unit_conversion_coefficient.append(
            get_unit_conversion_coefficient(era5_var_name, barra2_var_name))

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
        'unit_conversion': unit_conversion_coefficient,
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
