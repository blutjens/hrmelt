"""
Create prediction for every image in the validation set
by predicting the weighted average of all valid pixels
in the surrounding train images
"""

import yaml
import argparse
import logging
import datetime
from typing import List
import numpy as np
from tqdm import tqdm
from pathlib import Path
from osgeo import gdal # rasterio in dataloader uses
    # gdal. Need to import gdal to suppress warning msg. 
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from hrmelt.dataset import HRMeltDataset
from hrmelt.utils.utils import lookup_torch_dtype
from hrmelt.utils.utils import save_tensor_as_tif

def get_args():
    parser = argparse.ArgumentParser(description='Create time_interpolate_sar predictions on validation set')
    parser.add_argument('--cfg_path', type=str, default='runs/time_interpolate_sar/data_v1_3/config/config.yaml', help='Path to config yaml')
    parser.add_argument('--target_split', type=str, default='val', help='Split [train, val, or test] for the'\
                        ' time_interpolate_sar predictions will be calculated based on previous images in train set')
    parser.add_argument('--parallel', action='store_true', default=False, help='Enable parallel training')
    parser.add_argument('--save_png', action='store_true', default=False, help='Save predictions also as png')
    parser.add_argument('--verbose', type=bool, default=False, help='Set true to print verbose logs')
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

    # Create dataset and train, val, test partition.
    dtype = lookup_torch_dtype(cfg['dtype'])

    # The train dataset, in this case, contains all inputs to the predictions
    train_set = HRMeltDataset(cfg=cfg, split='train', verbose=False)
    # List of dates for which predictions should be created
    # if args.target_split == 'deploy':
    # todo: code currently needs in_keys because the dataloader assumes there's retrieve meta data. But, that could
    #  be changed. 
    # cfg['in_keys'] = ['pmw', 'mar_wa1'] 
    # cfg['path_pmw'] = 'raw/Helheim_data/reprojected_100m/PMWv1.0/'
    # cfg['path_mar_wa1'] = 'raw/Helheim_data/reprojected_100m/MAR/MARv3.14_nearest/WA1/'

    target_set = HRMeltDataset(cfg=cfg, split=args.target_split, verbose=False)
    train_set.sort()
    target_set.sort()
    
    def return_img_idcs_around_target_date(list_of_dates: List[str],
                                target_date: str='2019_01_01', 
                                n_previous_dates: int=3,
                                n_posterior_dates: int=0):
        """
        Returns the indices from list_of_dates of the n_previous_dates 
         dates before and n_posterior after the target date

        Args:
            list_of_dates: list of dates in the format "YYYY_MM_DD"
            target_date: the target date in the format "YYYY_MM_DD"

        """
        # Convert target_date into a datetime object
        target_date = datetime.datetime.strptime(target_date, "%Y_%m_%d")

        # Iterate over all indices in train_loader and return all indices 
        #  before target date
        previous_indices = []
        posterior_indices = []
        for i, date in enumerate(list_of_dates):
            date = datetime.datetime.strptime(date, "%Y_%m_%d")
            if date.year != target_date.year:
                # Skip dates from different years to avoid wrapping 
                #  images eg from september to march.
                continue
            if date < target_date:
                previous_indices.append(i)
            elif date > target_date:
                posterior_indices.append(i)
        # If no previous dates are found: return empty list
        n_previous_dates = min(len(previous_indices), n_previous_dates)
        previous_indices = previous_indices[-n_previous_dates:]

        # Get all indices after target date.
        n_posterior_dates = min(len(posterior_indices), n_posterior_dates)
        posterior_indices = posterior_indices[:n_posterior_dates]

        surrounding_indices = previous_indices + posterior_indices
        return surrounding_indices

    # Create data loader
    num_workers = 1 #set_num_workers(num_workers, parallel=parallel)

    loader_args = dict(batch_size=cfg['prediction_batch_size'], num_workers=num_workers, 
                       pin_memory=True)
    # train_loader = DataLoader(train_set, shuffle=False, **loader_args)
    target_loader = DataLoader(target_set, shuffle=False, drop_last=False, **loader_args)

    # Get list of dates in train_set
    list_of_dates = [Path(x).stem for x in train_set.filenames]

    for i, batch in tqdm(enumerate(target_loader)):
        _, targets, targets_mask, meta = batch
        batch_size = len(targets)
        assert batch_size == 1, 'Time interpolate SAR model has only been tested with batch size of 1'
    
        # Send to GPU and remove batch dimension
        targets = targets.to(device=device, dtype=dtype)[0,...]
        targets_mask = targets_mask.to(device=device, dtype=dtype)[0,...]

        # Get the indices of the last N dates before the validation date
        target_filename = Path(meta['filename'][0])
        target_date = target_filename.stem
        surrounding_img_idcs = return_img_idcs_around_target_date(list_of_dates, target_date, 
                                n_previous_dates=cfg['n_previous_dates'],
                                n_posterior_dates=cfg['n_posterior_dates'])
        if cfg['verbose']:
            print(f'\nPredicting target at {target_date} from {[list_of_dates[i] for i in surrounding_img_idcs]}')
            if len(surrounding_img_idcs) == 0:
                print(f'No previous or posterior dates found for {target_date}. Predicting image with all zeros.')


        # Create equally weighted average of all valid pixels in the N previous train images
        for i, idx in enumerate(surrounding_img_idcs):
            # Get the image from the train set
            _, targets_train, targets_mask_train, meta_train = train_set[idx]
            targets_train = targets_train.to(device=device, dtype=dtype)[0,...]
            targets_mask_train = targets_mask_train.to(device=device, dtype=dtype)[0,...]

            if i==0:
                # Init prediction and count of valid pixels
                prediction = torch.zeros(((1,)+ targets_train.shape), device=device)
                counts = torch.zeros(((1,)+ targets_train.shape), device=device)

            # Add the image to the prediction. Mask has 1 value for invalid pixels.
            prediction += torch.mul(targets_train, (1 - targets_mask_train))
            counts += (1 - targets_mask_train)

            # smooth-interpolate-sar
            # train_date = Path(meta_train['filename'][0]).stem
            # target_timestamp = datetime.strptime(target_date, "%Y_%m_%d") 
            # train_timestamp = datetime.strptime(train_date, "%Y_%m_%d") 
            # diff = np.abs((train_timestamp - target_timestamp).days)
            # weights[i] = 1. / diff

        # Divide by the number of valid predictions on each pixel to get the average
        counts[counts == 0.] = 1. # Set all zero counts to ones to avoid division by zero
        prediction = prediction.div(counts)

        # Save prediction to file using metadata of one of the latest training tif. 
        if args.target_split == 'deploy':
            new_tif_path = Path(cfg['path_deploy']) / target_filename.name
        else:
            new_tif_path = Path(cfg['path_predictions']) / target_filename.name
        
        save_tensor_as_tif(prediction, tif_path=meta_train['path_melt'],
                            new_tif_path=str(new_tif_path))
        if args.save_png:
            # Save prediction also as .png
            # new_tif_path = Path(cfg['path_predictions']) / target_filename.name
            new_png_path = new_tif_path.with_suffix('.' + 'png')
            Path(new_png_path).parent.mkdir(parents=True,exist_ok=True)
            save_image(prediction, str(new_png_path))
            # print('Saving png at ', new_png_path)