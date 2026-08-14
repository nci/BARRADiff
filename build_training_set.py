# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the given model."""

import os
curr_dir = os.path.dirname(os.path.realpath(__file__))

if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = os.path.join(curr_dir, 'hf_cache')

import argparse
import ray
import numpy as np
import torch
import math
import xarray as xr
import zarr
import sys
import shutil
import logging
from matplotlib import pyplot as plt
import time
from datetime import datetime
from diffusers.models import AutoencoderKL

from data.training_data import get_c2_training_pair_templates, get_target_shape, get_training_sample
from data.var_embedder import VarEmbed
from data.sampler import TrainingSampler

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

def init_schema(output_path, num_samples, z_channels, H, W):
    if not os.path.exists(output_path):
        root = zarr.open(output_path, 'w')
    else:
        root = zarr.open(output_path, 'a')

    if not os.path.exists(os.path.join(output_path, f'input')):
        root.create('input',
          shape=(num_samples, z_channels, H, W),
          chunks=(z_channels, H, W),
          fill_value=np.nan,
          dtype=np.float32)
    else:
        root['input'].resize(num_samples, z_channels, H, W)

    if not os.path.exists(os.path.join(output_path, f'target')):
        root.create(f'target',
          shape=(num_samples, z_channels, H, W),
          chunks=(z_channels, H, W),
          fill_value=np.nan,
          dtype=np.float32)
    else:
        root['target'].resize(num_samples, z_channels, H, W)
        
    if not os.path.exists(os.path.join(output_path, f'var_embedding')):
        root.create(f'var_embedding',
          shape=(1024,),
          chunks=(1024,),
          fill_value=np.nan,
          dtype=np.float32)

@ray.remote(num_cpus=1, scheduling_strategy='SPREAD', max_restarts=-1, max_task_retries=3)
class VaeWorker:
    def __init__(self):
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

    def get_training_data(self, sample_idx, query_time, ww_start, hh_start, patch_size, file_template):
        target_shapes = get_target_shape(query_time, [file_template,])
        H, W = target_shapes[0][-2:]

        # we load the entire region all at once to void the tiling artifacts caused by cv2.resize
        sample = get_training_sample(query_time, 0, 0, W, H, [file_template,])
        
        ww_end = ww_start + patch_size
        if ww_end > W:
            diff = ww_end - W
            ww_end -= diff
            ww_start -= diff
            
        hh_end = hh_start + patch_size
        if hh_end > H:
            diff = hh_end - H
            hh_end -= diff
            hh_start -= diff

        target_patch = sample['target'][..., hh_start:hh_end, ww_start:ww_end]
        input_patch = sample['input'][..., hh_start:hh_end, ww_start:ww_end]
        
        q = torch.tensor([0.005, 0.995])
        
        with torch.inference_mode():
            inp = torch.from_numpy(input_patch)
            min_val, max_val = inp.quantile(q)
            range_val = max_val - min_val

            inp = (inp - min_val) / range_val
            inp.clamp_(min=0, max=1)
            inp = torch.cat([inp[None],]*3, dim=1)

            posterior = self.vae.encode(inp).latent_dist
            z_inp = posterior.mode()
            
            tgt = torch.from_numpy(target_patch)
            min_val, max_val = tgt.quantile(q)
            range_val = max_val - min_val
            tgt = (tgt - min_val) / range_val
            tgt.clamp_(min=0, max=1)
            tgt = torch.cat([tgt[None],]*3, dim=1)

            posterior = self.vae.encode(tgt).latent_dist
            z_tgt = posterior.mode()
            
        data = {
            'sample_idx': sample_idx,
            'input': z_inp.detach().cpu().numpy(),
            'target': z_tgt.detach().cpu().numpy(),
        }
        return data
            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--var', required=True)
    parser.add_argument('--total-samples', type=int, required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--max-workers', type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    args = parser.parse_args()
    
    logger = get_logger(__name__)
    logger.info(args)

    total_samples = args.total_samples
    assert total_samples > 0
    
    training_templates = get_c2_training_pair_templates()
    
    file_template = None
    for tpl in training_templates:
        _, _, tgt_file = tpl
        var_name = tgt_file.split('/')[-1].split('_')[0]
        if var_name == args.var:
            file_template = tpl
            break
        
    assert file_template is not None, f'variable {args.var} not found'

    output_path = os.path.join(args.output_root, f'data/{var_name}')
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)
   
    query_time = datetime(2022, 1, 1, 1, 0, 0)
    target_shape = get_target_shape(query_time, [file_template,])[0]
    logger.info(f'variable shape: {target_shape}')
    
    H, W = target_shape[-2:]
    patch_size = 256
    latent_size = patch_size // 8
    z_channels = 4

    init_schema(output_path, total_samples, z_channels, latent_size, latent_size)

    logger.info(f'embedding variable text: {var_name}')
    var_embed = VarEmbed()
    var_name = args.var
    var_level = str(file_template[1])
    var_embedding = var_embed.embed(var_name, var_level)
    
    with zarr.open(output_path, 'a') as ds:
        ds['var_embedding'][:] = var_embedding.cpu().numpy()[0]
    
    sampler = TrainingSampler(
        start_time=datetime(2000, 1, 1, 0, 0, 0),
        end_time=datetime(2022, 12, 31, 23, 0, 0),
        max_crop_x_start=W-patch_size,
        max_crop_y_start=H-patch_size,
        seed=args.seed,
    )
    
    running_stats_root = os.path.join(args.output_root, 'stats')
    if not os.path.exists(running_stats_root):
        os.makedirs(running_stats_root)
    inp_running_stats_path = os.path.join(running_stats_root, f'{var_name}_inp.pt')
    tgt_running_stats_path = os.path.join(running_stats_root, f'{var_name}_tgt.pt')
    
    inp_running_stats = torch.nn.BatchNorm2d(z_channels, affine=False, track_running_stats=True)
    tgt_running_stats = torch.nn.BatchNorm2d(z_channels, affine=False, track_running_stats=True)
    if os.path.exists(inp_running_stats_path):
        inp_stats_state_dict = torch.load(inp_running_stats_path, weights_only=False)
        inp_running_stats.load_state_dict(inp_stats_state_dict)

        tgt_stats_state_dict = torch.load(tgt_running_stats_path, weights_only=False)
        tgt_running_stats.load_state_dict(tgt_stats_state_dict)

        logger.info(f"resume building training set, sample index: {inp_stats_state_dict['num_batches_tracked'].item()}")

    task_state_root = os.path.join(args.output_root, 'tasks')
    if not os.path.exists(task_state_root):
        os.makedirs(task_state_root)
    task_state_path = os.path.join(task_state_root, f'{var_name}.pt')

    if os.path.exists(task_state_path):
        task_checkpoint = torch.load(task_state_path)
    else:
        task_checkpoint = []
 
    try:
        ray.init(address='auto')
    except Exception as exc:
        ray.init(address=None, num_cpus=args.max_workers)

    cluster_resources = ray.cluster_resources()
    logger.info(cluster_resources)

    max_workers = int(cluster_resources['CPU'])
    workers = [VaeWorker.remote() for _ in range(max_workers)]
    num_tasks = len(task_checkpoint)
    
    display_progress = min(max(math.ceil(total_samples * 0.05), 1), 1000)

    obj_refs = []
    ref_task_id = {}
    task_attempts = {}
    
    def submit_task(task_id):
        worker_ref = workers[idx % len(workers)]
        ref = worker_ref.get_training_data.remote(*task_id)
        obj_refs.append(ref)
        ref_task_id[ref] = task_id
    
        if task_id not in task_attempts:
            task_attempts[task_id[0]] = 0
            
        task_attempts[task_id[0]] += 1
        
        if task_attempts[task_id[0]] >= 30:
            raise Exception(f'task {task_id} failed too many times')
    
    for idx, (ts, crop_x_start, crop_y_start) in enumerate(sampler):
        if idx in task_checkpoint:
            continue

        if num_tasks >= total_samples:
            break

        task_id = (idx, ts, crop_x_start, crop_y_start, patch_size, file_template)
        submit_task(task_id)

        is_last = idx+1 >= total_samples
    
        while is_last or len(obj_refs) >= max_workers * 2:
            ready_refs, obj_refs = ray.wait(obj_refs, num_returns=1)
            for ref in ready_refs:
                try:
                    data = ray.get(ref)
                except Exception as exc:
                    _task_id = ref_task_id[ref]
                    del ref_task_id[ref]
                    new_task_id = _task_id[:1] + sampler.__next__() + _task_id[-2:]
                    if args.verbose > 0:
                        logger.info(f'task {_task_id} failed, exception: {exc}, re-attempting {new_task_id}')
                    submit_task(new_task_id)
                    continue

                if np.any(np.isnan(data['input'])) or np.any(np.isnan(data['target'])):
                    _task_id = ref_task_id[ref]
                    del ref_task_id[ref]
                    new_task_id = _task_id[:1] + sampler.__next__() + _task_id[-2:]
                    if args.verbose > 0:
                        logger.info(f'task {_task_id} found nan, re-attempting task {new_task_id}')
                    submit_task(new_task_id)
                    continue
                
                task_id = ref_task_id[ref]
                del task_attempts[task_id[0]]
                del ref_task_id[ref]
                num_tasks += 1

                with zarr.open(output_path, 'a') as ds:
                    ds[f'input'][data['sample_idx']] = data['input'][0]
                    ds[f'target'][data['sample_idx']] = data['target'][0]

                inp_running_stats(torch.from_numpy(data['input']))
                torch.save(inp_running_stats.state_dict(), inp_running_stats_path)

                tgt_running_stats(torch.from_numpy(data['target']))
                torch.save(tgt_running_stats.state_dict(), tgt_running_stats_path)

                if task_id[0] not in task_checkpoint:
                    task_checkpoint.append(task_id[0]) 
                torch.save(task_checkpoint, task_state_path)

                if num_tasks == 1 or num_tasks % display_progress == 0 or is_last:
                    logger.info(f'{num_tasks}/{total_samples} done')

            if num_tasks >= total_samples:
                is_last = False


