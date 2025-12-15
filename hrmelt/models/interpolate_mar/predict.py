"""
Create prediction for every image in the validation set
by interpolation of the MAR data
"""

import yaml
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import wandb
from osgeo import gdal
import torch
from functools import partial
from tqdm import tqdm

from hrmelt.models.interpolate_mar.model import MarInterpolation
from hrmelt.dataset import HRMeltDataset
from hrmelt.utils.utils import save_tensor_as_tif, init_sweep_config
from hrmelt.utils.utils import MaskedLoss
from hrmelt.utils.utils import lookup_torch_dtype
from hrmelt.utils.utils import _worker_init_fn
from torchvision.utils import save_image

def predict(cfg, save_pred=False, split='val', device='cpu', save_png=False, verbose=False):
    """
    Applies the MarInterpolation model on every image in the dataset. Designed
     to be used on full images. Calculates the loss function on every image and 
     saves it to disk. 
    
    Args:
        cfg: config dictionary loaded from cfg_path
        save_pred: If true, saves predictions to cfg['path_predictions'] or cfg['path_deploy']
        split: Which dataset split to use, e.g., train, val, test
        save_png: If True, saves predictions also as png
    
    Returns:
        loss per image
    """
    assert not (cfg['batch_size'] > 1 and cfg['use_cv'] == True), 'MarInterpolation with use_cv=True is only tested with batch size of 1'
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
        
    model = MarInterpolation(
        blur_type=cfg['blur'],
        kernel_size=cfg['kernel_size'],
        sigma=cfg['sigma'],
        mask_threshold=cfg['mask_threshold'],
        use_cv=cfg['use_cv'],
        gamma=cfg['gamma'],
        brightness_factor=cfg['brightness_factor'],
        apply_landmask=apply_landmask
    )

    average_loss = 0.
    with tqdm(total=len(dataloader.dataset), unit='img',
                #disable=(sweep==False) # disable tqdm if printing to log instead of console
                ) as pbar:
        for batch in dataloader:
            inputs, targets, targets_mask, meta = batch

            inputs = inputs.to(device=device, dtype=dtype, memory_format=torch.channels_last)
            targets = targets.to(device=device, dtype=dtype)
            targets_mask = targets_mask.to(device=device, dtype=dtype)

            prediction = model(inputs)

            average_loss += criterion(prediction, targets, targets_mask)
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

                    print(f'Saving pred at {new_tif_path}')

                    save_tensor_as_tif(pred, tif_path=str(tif_path), new_tif_path=str(new_tif_path), verbose=verbose)

                    if save_png:
                        # Save prediction also as .png
                        new_png_path = new_tif_path.with_suffix('.' + 'png')
                        Path(new_png_path).parent.mkdir(parents=True,exist_ok=True)
                        save_image(pred, str(new_png_path))
            
            pbar.update(inputs.shape[0])

    return average_loss / len(dataloader)

def get_args():
    parser = argparse.ArgumentParser(description='Create interpolation predictions on validation set')
    parser.add_argument('--cfg_path', type=str, default='runs/interpolate_mar/sample/config/config.yaml',
                        help='Path to config yaml')
    parser.add_argument('--data_split', type=str, default='val', help='Split [train, val, or test] for '\
                        'which the predictions will be calculated')
    parser.add_argument('--verbose', type=bool, default=False, help='Set true to print verbose logs')
    parser.add_argument('--sweep', action='store_true', default=False,
                        help='If true, indicates that program is running a hyperparameter sweep')
    parser.add_argument('--save_png', action='store_true', default=False, help='Save predictions also as png')

    return parser.parse_args()

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()
    gdal.UseExceptions() # Enable gdal error messages (otherwise a warning is raise)

    # Import cfg and set seeds
    cfg = yaml.safe_load(open(args.cfg_path, 'r'))
    # Init cpu or gpu
    if cfg['use_gpu']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Using device {device}')

    # Initialize hyperparameter sweep
    if args.sweep:
        cfg = init_sweep_config(cfg, cfg['path_sweep_cfg'], args.task_id, args.num_tasks)
        wandb_run = wandb.init(project=cfg['wandb_project_name'],
                               resume='allow', anonymous='must',
                               dir=cfg['path_wandb'])
        wandb_run.config.update(
            dict(
                 img_size=cfg['img_size'][0],
                 optimizer=cfg['optimizer'],
                 loss_function=cfg['loss_function'],
                 blur=cfg['blur'],
                 kernel_size=cfg['kernel_size'],
                 sigma=cfg['sigma'],
            )
        )
        wandb.save(args.cfg_path)

    loss = predict(cfg, save_pred=cfg['save_predictions'], split=args.data_split, device=device, save_png=args.save_png,
                   verbose=args.verbose)

    print(f"{cfg['loss_function']} loss on {args.data_split} set is: {loss}")