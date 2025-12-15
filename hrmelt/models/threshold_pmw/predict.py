"""
Predict surface meltwater fraction as a function of
PMW observations
"""

import yaml
import argparse
import logging
import datetime
from typing import List
import numpy as np
from tqdm import tqdm
from functools import partial
from pathlib import Path
from osgeo import gdal # rasterio in dataloader uses
    # gdal. Need to import gdal to suppress warning msg. 
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from hrmelt.models.threshold_pmw.model import PmwThreshold
from hrmelt.dataset import HRMeltDataset
from hrmelt.utils.utils import lookup_torch_dtype
from hrmelt.utils.utils import save_tensor_as_tif
from hrmelt.utils.utils import init_sweep_config
from hrmelt.utils.utils import set_all_seeds
from hrmelt.utils.utils import MaskedLoss
from hrmelt.utils.utils import _worker_init_fn

def predict(cfg, save_pred=False, split='train', device='cuda', compress=False, verbose=False):
    """
    Applies the PmwThreshold model on every image in the dataset. Designed
     to be used on full images. Calculates the loss function on every image and 
     saves it to disk. 
    
    Args:
        cfg: config dictionary loaded from cfg_path
        save_pred: If true, saves predictions as .tif to cfg['path_predictions']
        split: Which dataset split to use, e.g., train, val, test
        device: Which device; cpu, cuda, etc
        compress: If True and save_pred, saves predictions as .png to cfg['path_predictions']. 
            Same functionality as save_png in other predict() functions.
    Returns:
        loss: average loss per image
    """

    # Create dataset and train, val, test partition.
    dtype = lookup_torch_dtype(cfg['dtype'])

    # init_fn needs to be global func so it can be Pickeled
    init_fn_with_cfg = partial(_worker_init_fn, seed=cfg['seed'])

    # Initialize dataset from .csv file with filenames
    dataset = HRMeltDataset(cfg=cfg, split=split, verbose=False)
    dataset.sort()
    loader_args = dict(batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                    pin_memory=True,
                    worker_init_fn=init_fn_with_cfg)
    dataloader = DataLoader(dataset, shuffle=False, drop_last=False, **loader_args)
    criterion = MaskedLoss(cfg['loss_function'])

    # Only apply the landmask only when the prediction is saved, to minimize 
    #  computation during hyperparameter sweep
    apply_landmask = save_pred

    model = PmwThreshold(
        mask_threshold=cfg['mask_threshold'],
        apply_landmask=apply_landmask
    )

    average_loss = 0.
    with tqdm(total=len(dataloader.dataset), unit='img',
                #disable=(sweep==False) # disable tqdm if printing to log instead of console
                ) as pbar:
        for batch in dataloader:
            inputs, targets, targets_mask, meta = batch
            n_imgs_in_current_batch = inputs.shape[0]

            inputs = inputs.to(device=device, dtype=dtype, memory_format=torch.channels_last)
            targets = targets.to(device=device, dtype=dtype)
            targets_mask = targets_mask.to(device=device, dtype=dtype)

            prediction = model(inputs)

            average_loss += criterion(prediction, targets, targets_mask) * n_imgs_in_current_batch

            if save_pred:
                for i, pred in enumerate(prediction):
                    if 'path_melt' in meta:
                        tif_path = Path(meta['path_melt'][i])
                    else:
                        tif_path = Path(cfg['data_root']) / Path(cfg['path_melt_reference'])

                    # Save prediction to file using metadata of one of the latest training tif. 
                    if dataloader.dataset.split == 'deploy':
                        new_tif_path = Path(cfg['path_deploy']) / Path(meta['filename'][i])
                    else:
                        new_tif_path = Path(cfg['path_predictions']) / Path(meta['filename'][i])

                    # prediction = prediction.squeeze(0) # Remove batch dimension to reshape to (1, height_tif, width_tif)
                    save_tensor_as_tif(pred, tif_path=str(tif_path), new_tif_path=str(new_tif_path), verbose=verbose)

                    if compress:
                        # Save compressed image
                        new_png_path = Path(new_tif_path).with_suffix('.' + 'png')
                        if verbose: print(f'Saving pred png at {new_png_path}')
                        save_image(pred, str(new_png_path))

            pbar.update(inputs.shape[0])

    loss = average_loss / len(dataset)

    return loss

def get_args():
    parser = argparse.ArgumentParser(description='Create threshold_pmw predictions')
    parser.add_argument('--cfg_path', type=str, default='runs/threshold_pmw/data_v1_4/config/config.yaml', help='Path to config yaml')
    parser.add_argument('--data_split', type=str, default='val', help='Split [train, val, or test] for which'\
                        ' predictions will be calculated')
    parser.add_argument('--parallel', action='store_true', default=False, help='Enable parallel training')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print verbose logs')
    parser.add_argument('--save_png', action='store_true', default=False, help='Save predictions also as png')
    return parser.parse_args()

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    gdal.UseExceptions() # Enable gdal error messages (otherwise a warning is raise)

    # Init cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Import cfg and set seeds
    cfg = yaml.safe_load(open(args.cfg_path, 'r'))
    cfg['in_channels'] = len(cfg['in_keys']) + len(cfg['in_keys_static']) + len(cfg['in_keys_aux'])
    set_all_seeds(cfg['seed'], device=device.type, 
                  use_deterministic_algorithms=cfg['use_deterministic_algorithms'])
    
    loss = predict(cfg=cfg,
            save_pred=cfg['save_predictions'], 
            split=args.data_split,
            device=device, 
            compress=args.save_png,
            verbose=args.verbose)

    print(f"{cfg['loss_function']} loss on {args.data_split} set is: {loss}/img")