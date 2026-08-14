import os
curr_dir = os.path.dirname(os.path.realpath(__file__))
os.environ['HF_HOME'] = os.path.join(curr_dir, 'hf_cache')

import argparse
import sys
import logging
import math
import numpy as np
import xarray as xr
import torch
import zarr
from matplotlib import pyplot as plt

from diffusers.models import AutoencoderKL

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


def plot(inp, tgt, output_file_path):
    fig = plt.figure(figsize=(22, 11), dpi=80)
    ax1 = fig.add_subplot(121)
    inp_image = np.moveaxis(inp, 0, -1)
    xr.DataArray(inp_image).plot.imshow(ax=ax1, robust=True)
    plt.title(f'input', fontsize=22)
    plt.axis('off')

    ax2 = fig.add_subplot(122)
    tgt_image = np.moveaxis(tgt, 0, -1)
    xr.DataArray(tgt_image).plot.imshow(ax=ax2, robust=True)
    plt.title(f'target', fontsize=22)
    plt.axis('off')

    plt.subplots_adjust(wspace=0.025, hspace=0.05)

    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--num-samples', type=int, default=20)
    args = parser.parse_args()

    assert args.num_samples > 0

    logger = get_logger(__name__)
    logger.info(args)

    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    np.random.seed(0)
    with zarr.open(args.data_root) as ds:
        sample_indices = np.random.choice(ds['input'].shape[0], size=args.num_samples)

    for ii, sample_idx in enumerate(sample_indices):
        with zarr.open(args.data_root) as ds:
            inp = ds['input'][sample_idx]
            tgt = ds['target'][sample_idx]
            
        with torch.inference_mode():
            inp = torch.from_numpy(inp[None])
            inp_out = vae.decode(inp).sample.cpu()[0].numpy()

            tgt = torch.from_numpy(tgt[None])
            tgt_out = vae.decode(tgt).sample.cpu()[0].numpy()

        output_file_path = os.path.join(args.output_root, f'training_sample_{sample_idx}.png')
        plot(inp_out, tgt_out, output_file_path)

        logger.info(f'{ii+1} / {args.num_samples} done')

