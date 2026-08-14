# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import sys
import gc
import logging
import time
import copy
import pickle
import psutil
import math
import numpy as np
import torch
import dnnlib

from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch_utils import distributed as dist
from torch_utils.checkpointer import Checkpointer
from torch.distributed.checkpoint.state_dict import (
    _init_optim_state,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch_utils import training_stats
from torch_utils import persistence
from torch_utils import misc
from torch_utils.logger import get_logger

#----------------------------------------------------------------------------
# Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
# paper "Analyzing and Improving the Training Dynamics of Diffusion Models".

@persistence.persistent_class
class EDM2Loss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5, cfg_dropout=0.1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.cfg_dropout = cfg_dropout

    def __call__(self, net, inp_images, tgt_images, labels=None):
        rnd_normal = torch.randn([inp_images.shape[0], 1, 1, 1], device=inp_images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(inp_images) * sigma

        cfg_w = torch.ones_like(rnd_normal)
        if self.cfg_dropout > 0:
            rnd_uniform = torch.rand([inp_images.shape[0], 1, 1, 1], device=inp_images.device)
            cfg_w[rnd_uniform < self.cfg_dropout] = 0.0

        denoised, logvar = net(tgt_images + noise, inp_images*cfg_w, sigma, labels, return_logvar=True)
        loss = (weight / logvar.exp()) * ((denoised - tgt_images) ** 2) + logvar
        return loss

#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= math.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

def print0(msg):
    if dist.get_rank() == 0:
        logger = get_logger(__name__)
        logger.info(msg)

#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    dataset_kwargs      = dict(class_name='training.dataset.ImageFolderDataset', path=None),
    encoder_kwargs      = dict(class_name='training.encoders.StabilityVAEEncoder'),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2),
    network_kwargs      = dict(class_name='training.networks_edm2.Precond'),
    loss_kwargs         = dict(class_name='training.training_loop.EDM2Loss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99), foreach=False, fused=False),
    lr_kwargs           = dict(func_name='training.training_loop.learning_rate_schedule'),
    ema_kwargs          = dict(class_name='training.phema.PowerFunctionEMA'),

    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    batch_size          = 2048,     # Total batch size for one training iteration.
    batch_gpu           = 1,        # Limit batch size per GPU. None = no limit.
    grad_accumulation   = 1,        # Gradient accumulation
    model_shards        = 2,        # Number of model shards
    total_nimg          = 8<<30,    # Train for a total of N training images.
    slice_nimg          = None,     # Train for a maximum of N training images in one invocation. None = no limit.
    status_nimg         = 128<<10,  # Report status every N training images. None = disable.
    snapshot_nimg       = 8<<20,    # Save network snapshot every N training images. None = disable.
    checkpoint_nimg     = 128<<20,  # Save state checkpoint every N training images. None = disable.

    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
):
    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate batch size.
    #batch_gpu_total = batch_size // dist.get_world_size()
    #if batch_gpu is None or batch_gpu > batch_gpu_total:
    #    batch_gpu = batch_gpu_total
    #num_accumulation_rounds = batch_gpu_total // batch_gpu
    #assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    
    num_accumulation_rounds = grad_accumulation
    loss_scaling = loss_scaling / (batch_gpu * grad_accumulation)
    assert total_nimg % batch_size == 0
    assert slice_nimg is None or slice_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or (snapshot_nimg % batch_size == 0 and snapshot_nimg % 1024 == 0)
    assert checkpoint_nimg is None or (checkpoint_nimg % batch_size == 0 and checkpoint_nimg % 1024 == 0)

    # Setup dataset, encoder, and network.
    print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    print0(f'dataset info: {dataset_obj.dataset_shape}, {dataset_obj.var_paths}')
    ref_image, cond_image, ref_label = dataset_obj[0]

    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).unsqueeze(0))
    cond_image = encoder.encode_latents(torch.as_tensor(cond_image).unsqueeze(0))
    
    # The network architecture assumes 64x64 latent input size, or 512x512 pixel input size
    interface_kwargs = dict(img_resolution=64, img_channels=ref_image.shape[1], 
        cond_img_channels=cond_image.shape[1], label_dim=ref_label.shape[-1])
    
    print0(network_kwargs)
    print0(interface_kwargs)
    print0('Constructing network...')
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    mesh = np.arange(dist.get_world_size())
    if mesh.size == model_shards:
        mesh = mesh.tolist()
        mesh_dim_names = None
    elif mesh.size > model_shards:
        mesh = mesh.reshape(-1, model_shards).tolist()
        mesh_dim_names = ['replicate', 'shard']
    else:
        raise Exception(f'model_shards {model_shards} larger than world_size {dist.get_world_size()}')

    device_mesh = DeviceMesh(device_type=device_type, mesh=mesh, mesh_dim_names=mesh_dim_names)
    print0(device_mesh)

    fsdp_kwargs = {
        'mesh': device_mesh,         
    }
    
    #if args.mixed_precision:
    #    fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
    #        param_dtype=torch.bfloat16,
    #        reduce_dtype=torch.float32,
    #    )
    
    for model in [net, *ema.emas]:
        for name, stage in model.unet.enc.items():
            fully_shard(stage, **fsdp_kwargs)

        for name, stage in model.unet.dec.items():
            fully_shard(stage, **fsdp_kwargs)

        fully_shard(model, **fsdp_kwargs)
    
        for param in model.parameters():
            assert isinstance(param, DTensor)
            assert param.placements[-1] == Shard(0)
            # inspect sharded parameters with param.to_local()

    print0(model.parameters().__next__().placements)
            
    net.train().requires_grad_(True).to(device)
    
    for model in ema.emas:
        model.eval().requires_grad_(False).to(device)

    # Setup training state.
    state = dnnlib.EasyDict(cur_nimg=0)
    state['network_args'] = network_kwargs
    state['interface_args'] = interface_kwargs

    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    
    checkpoint = Checkpointer(run_dir, False)
    if not checkpoint.is_empty():
        checkpoint.load(model=net, emas=ema.emas, optim=optimizer, training_states=state)

    # Print network summary.
    rows = misc.get_module_summary(net, [
        torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device),
        torch.zeros([batch_gpu, net.cond_img_channels, net.img_resolution, net.img_resolution], device=device),
        torch.ones([batch_gpu], device=device),
        torch.zeros([batch_gpu, net.label_dim], device=device),
    ], max_nesting=2)
    
    if dist.get_rank() == 0:
        # Print table.
        widths = [max(len(cell) for cell in column) for column in zip(*rows)]
        print()
        for row in rows:
            print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
        print()

    stop_at_nimg = total_nimg
    if slice_nimg is not None:
        granularity = checkpoint_nimg if checkpoint_nimg is not None else snapshot_nimg if snapshot_nimg is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity # round down
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg
    print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg:')

    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), 
                                           seed=seed, start_idx=state.cur_nimg)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, 
                                                                batch_size=batch_gpu, **data_loader_kwargs))
    prev_status_nimg = state.cur_nimg
    start_nimg = state.cur_nimg

    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Report status.
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            prev_status_nimg = state.cur_nimg

            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < total_nimg:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True

        # Done?
        if done:
            break

            
        # Evaluate loss and accumulate gradients.
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)

        for round_idx in range(num_accumulation_rounds):
            is_last_backward = round_idx == num_accumulation_rounds - 1
            for name, stage in net.unet.enc.items():
                stage.set_is_last_backward(is_last_backward)

            for name, stage in net.unet.dec.items():
                stage.set_is_last_backward(is_last_backward)

            net.set_is_last_backward(is_last_backward)
            
            inp_images, tgt_images, _ = next(dataset_iterator)
            inp_images = inp_images.to(device)
            tgt_images = tgt_images.to(device)
            loss = loss_fn(net=net, inp_images=inp_images, tgt_images=tgt_images)
            loss = loss.sum().mul(loss_scaling)
            loss.backward()
            
            #if dist.get_rank() == 0 and state.cur_nimg == 0 :
            #    msg = f'grad accum {round_idx}/{num_accumulation_rounds}'
            #    print0(msg)

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        
        for g in optimizer.param_groups:
            g['lr'] = lr

        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num_(param.grad, nan=0, posinf=0, neginf=0)

        optimizer.step()
        optimizer.zero_grad()

        # Update EMA and training state.
        state.cur_nimg += batch_size
        ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)

        '''
        loss_val = loss.item()
        if dist.get_rank() == 0:
            curr_step = state.cur_nimg // batch_size
            total_steps = total_nimg // batch_size
            msg = f'{curr_step}/{total_steps} ({state.cur_nimg}/{total_nimg}), lr={lr:.5f}, loss={loss_val:.5f}'
            print0(msg)
        '''

        if status_nimg is not None and \
            (done or (state.cur_nimg-batch_size) % status_nimg == 0):
            #state.cur_nimg != start_nimg or start_nimg == 0

            loss_val = loss.item()

            if dist.get_rank() == 0:
                curr_step = state.cur_nimg // batch_size
                total_steps = total_nimg // batch_size
                msg = f'{curr_step}/{total_steps} ({state.cur_nimg}/{total_nimg}), lr={lr:.5f}, loss={loss_val:.5f}'
                print0(msg)
            
            if np.isnan(loss_val):
                raise Exception('loss is nan')
                
        if checkpoint_nimg is not None \
            and state.cur_nimg % checkpoint_nimg == 0 \
            and (state.cur_nimg != start_nimg or start_nimg == 0):

            emas = [e[0] for e in ema.get()]
            checkpoint.save(step=state.cur_nimg, model=net, emas=emas, optim=optimizer, training_states=state)

            gc.collect()

#----------------------------------------------------------------------------
