import os
import pickle
import argparse
import torch
import dnnlib

from torch_utils import checkpointer as ckp
from torch_utils.logger import get_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-root', type=str, default='')
    parser.add_argument('--training-time', type=str, default='latest')
    args = parser.parse_args()

    args.input_root = os.path.abspath(args.input_root)
    assert os.path.exists(args.input_root), f'{args.input_root} is not accessible'

    if len(args.output_root) > 0:
        args.output_root = os.path.abspath(args.output_root)
    else:
        args.output_root = os.path.join(args.input_root, 'snapshots')

    logger = get_logger(__name__)
    logger.info(args)

    checkpoint = ckp.Checkpointer(args.input_root)

    if args.training_time == 'latest':
        training_time = checkpoint.last_training_time
    else:
        training_time = int(args.training_time)

    last_states_checkpoint = (
        f"{checkpoint.folder}/{'dcp_api' if checkpoint.dcp_api else 'dtensor_api'}"
        f"/{training_time}/{ckp.STATES_CHECKPOINT}"
    )
    logger.info(f'loading states checkpoint {last_states_checkpoint}')
    training_states = torch.load(
        last_states_checkpoint, weights_only=False, map_location="cpu"
    )

    logger.info(training_states['network_args'])
    logger.info(training_states['interface_args'])
    net = dnnlib.util.construct_class_by_name(**training_states['network_args'], 
            **training_states['interface_args'])

    os.makedirs(args.output_root, exist_ok=True)
    
    ema_stds = [0.050, 0.100]

    last_model_dir = (
        f"{checkpoint.folder}/{'dcp_api' if checkpoint.dcp_api else 'dtensor_api'}"
        f"/{training_time}"
    )
    for name in os.listdir(last_model_dir):
        if name.startswith(ckp.EMA_CHECKPOINT):
            ema_idx = int(name.split('.')[-1])
            ema_checkpoint_path = os.path.join(last_model_dir, name)
            logger.info(f'loading EMA checkpoint: {ema_checkpoint_path}')

            full_sd = torch.load(
                ema_checkpoint_path, mmap=True, weights_only=True, map_location="cpu"
            )
            net.load_state_dict(full_sd)
            data = dnnlib.EasyDict(ema=net)

            ema_suffix = f'-{ema_stds[ema_idx]:.3f}'
            fname = f'network-snapshot-{training_time//1000:07d}{ema_suffix}.pkl'
            snapshot_file_path = os.path.join(args.output_root, fname)
            logger.info(f'saving EMA snapshot {snapshot_file_path}')

            with open(snapshot_file_path, 'wb') as f:
                pickle.dump(data, f)

