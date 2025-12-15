import yaml
import argparse
import random
import numpy as np
import logging
import torch
from osgeo import gdal

from hrmelt.utils.utils import set_all_seeds
from hrmelt.dataset import HRMeltDataset


def get_args():
    parser = argparse.ArgumentParser(description='Determine the variance in evaluation metrics on the test set')
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed')
    parser.add_argument('--cfg_path', type=str, default='runs/unet/data_v1_4/config/config.yaml', 
                        help='Pass a default config, mainly for retrieving the filenames to build the dataloader.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print verbose')
    return parser.parse_args()

if __name__ == "__main__":
    # Get command line arguments
    args = get_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    gdal.UseExceptions()

    # Init cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Load config and overwrite relevant parameters
    cfg = yaml.safe_load(open(args.cfg_path, 'r'))
    cfg['seed'] = 42
    cfg['use_deterministic_algorithms'] = True
    cfg['create_csv_w_split_cfg'] = True
    cfg['path_train_split_csv'] = 'runs/all/data_variance/config/train.csv'
    cfg['path_val_split_csv'] = 'runs/all/data_variance/config/val.csv'
    cfg['path_test_split_csv'] = 'runs/all/data_variance/config/test.csv'
    cfg['path_all_split_csv'] = 'runs/all/data_variance/config/all.csv'
    cfg['path_deploy'] = None
    cfg['path_eval'] = None
    cfg['n_imgs_per_month'] = 2

    set_all_seeds(cfg['seed'], device=device.type, 
                    use_deterministic_algorithms=cfg['use_deterministic_algorithms'])

    # Load dataset
    cfg['split_cfg'] = 'csv'
    dataset = HRMeltDataset(cfg=cfg, split='all', verbose=args.verbose)

    k_folds = 10
    for k in range(k_folds):
        # Draw a new stratified data split
        cfg['split_cfg'] = 'stratified_time'
        dataset.verbose = args.verbose
        train_set, val_set, test_set = dataset.create_data_splits(
            split_cfg=cfg['split_cfg'],
            val_percent=None,
            test_percent=None,
            seed=cfg['seed'],
            create_csv_w_split_cfg=cfg['create_csv_w_split_cfg'],
            path_train_split_csv=cfg['path_train_split_csv'],
            path_val_split_csv=cfg['path_val_split_csv'],
            path_test_split_csv=cfg['path_test_split_csv'],
            path_all_split_csv=cfg['path_all_split_csv'],
            n_imgs_per_month=cfg['n_imgs_per_month']
            )
        print(f'Index {k} - 1st filepath in test set - {test_set[0]["melt"].name}')

        # (Fit model -> First try with threshold_pmw only)
        import pdb;pdb.set_trace()
        # Evaluate model
        ## Load model

        ## Create predictions on test_set
        # python hrmelt/models/threshold_pmw/predict.py --cfg_path runs/threshold_pmw/data_v1_4/config/config.yaml --data_split test

        ## Calculate metrics

        # Plot test performance