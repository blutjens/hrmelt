"""
This file will evaluate the deployment predictions of all models.
The code assumes that predictions have already been generated 
using, e.g., predict.py --target_split deploy

Author: Björn Lütjens
"""
import argparse
import os
from typing import Sequence

import numpy as np
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
from osgeo import gdal # rasterio in dataloader uses
    # gdal. Need to import gdal to suppress warning msg. 
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime
import pandas as pd
from functools import partial

import hrmelt.eval.metrics as metrics
from hrmelt.dataset import HRMeltDataset
from hrmelt.dataset import get_filepaths_from_csv
from hrmelt.utils.utils import get_size_of_tif
from hrmelt.utils.utils import MaskedLoss
from hrmelt.utils.utils import set_num_workers
from hrmelt.utils.utils import lookup_torch_dtype
from hrmelt.utils.utils import _worker_init_fn
from hrmelt.eval.metrics import MaskedSSIM
from hrmelt.eval.metrics import MaskedR2
from hrmelt.eval.metrics import CountValidPx


from hrmelt.eval.benchmark import HRMeltDatasetPredictions

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the quality of all predictions from different models.')
    parser.add_argument('--parallel', action='store_true', default=False, help='Enable parallel training')
    parser.add_argument('--verbose', type=bool, default=False, help='Set true to print verbose logs')
    parser.add_argument('--data_split', type=str, default='deploy', help='Split for which the'\
                        'evaluation will be conducted')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for parallel processing')
    parser.add_argument('--melt_cutoff', type=float, default=0.5, help='Cutoff value for binary classification of meltwater from surface meltwater fraction. ')
    parser.add_argument('--unet', action='store_true', default=False, help='Add unet to evaluation')
    parser.add_argument('--unet_smp', action='store_true', default=False, help='Add unet_smp to evaluation')
    parser.add_argument('--time_interpolate_sar', action='store_true', default=False, help='Add time_interpolate_sar to evaluation')
    parser.add_argument('--interpolate_mar', action='store_true', default=False, help='Add interpolate_mar to evaluation')
    parser.add_argument('--linear_dem', action='store_true', default=False, help='Add linear_dem to evaluation')
    parser.add_argument('--deeplabv3', action='store_true', default=False, help='Add deeplabv3 to evaluation')
    parser.add_argument('--threshold_pmw', action='store_true', default=False, help='Add threshold_pmw to evaluation')
    return parser.parse_args()

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()
    gdal.UseExceptions() # Enable gdal error messages (otherwise a warning is raised)
    
    
    # Initialize config, mainly with dataset paths
    cfg = {}
    cfg['verbose'] = args.verbose
    cfg['data_root'] = '/home/gridsan/lutjens/EarthIntelligence_shared/datasets/hrmelt/raw/Helheim_data/reprojected_100m'
    cfg['path_deploy_split_csv'] = f'./runs/unet/data_v1_4/config/deploy.csv' # Image filenames used during test
    cfg['split_cfg'] = 'csv'
    cfg['create_csv_w_split_cfg'] = False
    cfg['path_landmask'] = 'GIMP_Mask/landMask_100m.tif'
    cfg['dtype'] = 'float32' # datatype for in-, output and model weights
    cfg['path_benchmark_figures'] = './references/figures/deploy/' # Store all plots from benchmarking the models
    cfg['prediction_batch_size'] = args.batch_size
    cfg['melt_cutoff'] = args.melt_cutoff
    dir_figures = './references/figures/deploy/meltwater_extent_per_day/'
    if args.unet_smp:
        dir_figures = dir_figures + 'unet_smp/'
    # Assign colors to each model_key
    model_colors = {'unet': 'tab:blue',
              'time_interpolate_sar': 'tab:olive',
              'interpolate_mar': 'tab:brown',
              'linear_dem': 'tab:orange',
              'deeplabv3': 'tab:red',
              'threshold_pmw': 'lightgray',
              'unet_smp': 'tab:blue',
              }

    # Set directories of every model's predictions
    model_keys = []
    if args.threshold_pmw:
        model_keys.append('threshold_pmw')
        cfg['path_predictions_threshold_pmw'] = Path('/home/gridsan/lutjens/EarthIntelligence_shared/datasets/hrmelt/interim/runs/threshold_pmw/data_v1_4/deploy/')
    if args.interpolate_mar:
        model_keys.append('interpolate_mar')
        cfg['path_predictions_interpolate_mar'] = Path('/home/gridsan/lutjens/EarthIntelligence_shared/datasets/hrmelt/interim/runs/interpolate_mar/data_v1_4/deploy/')
    if args.linear_dem:
        model_keys.append('linear_dem')
        cfg['path_predictions_linear_dem'] = Path('/home/gridsan/lutjens/EarthIntelligence_shared/datasets/hrmelt/interim/runs/linear_dem/data_v1_4/deploy/')
    if args.deeplabv3:
        model_keys.append('deeplabv3')
        cfg['path_predictions_deeplabv3'] = Path('/home/gridsan/lutjens/EarthIntelligence_shared/datasets/hrmelt/interim/runs/deeplabv3/data_v1_4/deploy/')
    if args.unet:
        model_keys.append('unet')
        cfg['path_predictions_unet'] = Path('/home/gridsan/lutjens/EarthIntelligence_shared/datasets/hrmelt/interim/runs/unet/data_v1_4/deploy/')
    if args.unet_smp:
        model_keys.append('unet_smp')
        cfg['path_predictions_unet_smp'] = Path('/home/gridsan/lutjens/EarthIntelligence_shared/datasets/hrmelt/interim/runs/unet_smp/data_v1_4/deploy/')
    if args.time_interpolate_sar:
        model_keys.append('time_interpolate_sar')
        # cfg['path_predictions_time_interpolate_sar'] = Path('./runs/time_interpolate_sar/data_v1_4/predictions/')
        cfg['path_predictions_time_interpolate_sar'] = Path('/home/gridsan/lutjens/EarthIntelligence_shared/datasets/hrmelt/interim/runs/time_interpolate_sar/data_v1_4/deploy/')
    if not model_keys:
        print('Warning: need to supply at least one model key as command line argument, e.g., --unet. For now, we evaluate only unet.')
        model_keys.append('unet')
    cfg['paths_deploy'] = []
    for model_key in model_keys:
        print(f'Retrieving {model_key} predictions from {cfg[f"path_predictions_{model_key}"]}')
        cfg['paths_deploy'].append(cfg[f'path_predictions_{model_key}'])

    # Init compute parameters, e.g., cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg['num_workers'] = set_num_workers(num_workers=None, parallel=True)
    cfg['batch_size'] = args.batch_size
    cfg['prediction_batch_size'] = args.batch_size
    cfg['seed'] = 42
    print(f'Using device {device}, num workers: {cfg["num_workers"]}, batch size: {cfg["batch_size"]}')

    # Set dataloader config
    init_fn_with_cfg = partial(_worker_init_fn, seed=cfg['seed'])
    dataloader_args = dict(shuffle=False, 
                    drop_last=False,
                    batch_size=cfg['prediction_batch_size'], 
                    num_workers=cfg['num_workers'],
                    pin_memory=True,
                    worker_init_fn=init_fn_with_cfg,
                    )
    dtype = lookup_torch_dtype(cfg['dtype'])
    cfg['in_keys_static'] = ['landmask']
    total_meltwaters = {}
    
    # Iterate over each model:
    for model_key, path_predictions in zip(model_keys, cfg['paths_deploy']):
        # import pdb;pdb.set_trace()
        print(f'Computing metrics for model: {model_key}')
        # Initialize dataset
        dataset = HRMeltDatasetPredictions(cfg=cfg,
                    split=args.data_split,
                    verbose=args.verbose,
                    model_keys=[model_key],
                    paths_predictions=[path_predictions],)
        # Create data loader
        dataloader = DataLoader(dataset, **dataloader_args)

        total_meltwaters[model_key] = np.zeros((len(dataset)), dtype=cfg['dtype'])
        # Compute meltwater extent on every image
        for i, batch in tqdm(enumerate(dataloader), disable=False):
            with torch.no_grad():
                inputs, _, targets_mask, meta = batch
                batch_size = len(inputs)

                inputs = inputs.to(device=device, dtype=dtype, memory_format=torch.channels_last)
                targets_mask = targets_mask.to(device=device, dtype=dtype) # targets_mask includes invalid SAR pixels and landmask
                # Get the model prediction from the inputs
                pred_model_ch_idx = dataset.get_channel_idx(meta, f'predictions_{model_key}')
                pred_model = inputs[:,pred_model_ch_idx:pred_model_ch_idx+1,...]
                
                landmask_ch_idx = dataset.get_channel_idx(meta, f'landmask')
                landmask = inputs[:,landmask_ch_idx:landmask_ch_idx+1,...]

                # Apply landmask to predictions
                #  Set all values in predictions to zero where landmask is one
                pred_model = pred_model * (1-landmask)

                # Threshold into binary meltwater classification
                # pred_model = torch.where(pred_model > cfg['melt_cutoff'], 1., 0.)

                # Compute total meltwater per day, i.e., sum of all fractional meltwater values
                # ->> No need to threshold, bc models are predicting surface meltwater fraction
                total_meltwater = torch.sum(pred_model[:,0,...], axis=(-2,-1)) # dims: (batch_size, 1); in 100m^2
                
                # Convert from 100m^2 to 1km^2
                total_meltwater /= 100.

                # Add to list
                total_meltwaters[model_key][i*batch_size:i*batch_size+batch_size] = total_meltwater.cpu().numpy()

                if i == 0:
                    total_area = (torch.sum(landmask[0,0,...], axis=(-2,-1)) / 100.).cpu().numpy() # dims: (,); in 1km^2

                plot = False
                if plot:
                    # Init plot
                    fig, axs = plt.subplots(1, 1)   
                    ax = axs.imshow(landmask[0,0,...].cpu().numpy())
                    cbar = plt.colorbar(ax, orientation='horizontal', fraction=0.05, pad=0.01 , spacing='proportional') # ticks=ticks[i]
                    axs.axis('off')
                        
                    plt.tight_layout()
                    # Save figure
                    Path(dir_figures).mkdir(parents=True, exist_ok=True)
                    plt.savefig(f"{dir_figures}{model_key}.png")
                    plt.close()

        print(model_key, total_meltwaters[model_key])

    """
    Plot meltwater extent per day over time
    """
    import matplotlib.dates as mdates
    import pandas as pd
    from hrmelt.utils.plotting import split_axes

    # Convert to pandas dataframe for easier handling of time stamps
    timestamps = np.array([datetime.strptime(filename, "%Y_%m_%d.tif") for filename in dataset.filenames]) # Convert from List('YYYY_MM_DD') to np.array(DatetimeIndex)
    df = pd.DataFrame(index=timestamps)

    key = 'total_meltwater_per_day'
    filepath_to_save = f'{dir_figures}total_meltwater_per_day.png'

    years = np.unique([timestamp.year for timestamp in timestamps])
    n_years = len(years)

    fig, axs = plt.subplots(1, n_years, sharey=True, facecolor='w', figsize=(13,4), dpi=200)
    for model_key in model_keys:
        df[f'total_meltwater_per_day_{model_key}'] = total_meltwaters[model_key]
        # Plot the data of each year
        for ax, yr in zip(axs, years):
            df_yr = df[df.index.year == yr]
            if key == 'total_meltwater_per_day':
                ax.plot(df_yr.index, df_yr[f'total_meltwater_per_day_{model_key}'], color=model_colors[model_key], label=model_key, linewidth=1.)
            else:
                raise NotImplementedError(f'Given key ({key}) is invalid.')
        axs = split_axes(axs)

    # Modify axis settings
    for i, (ax, yr) in enumerate(zip(axs,years)):
        if i == 0:
            plt.legend(loc='upper right', bbox_to_anchor=(-4.6,0.9)) # (offset_from_right,
        ax.set_title(yr, y=1.0, pad=-14)
        # Remove ticks outside of data range
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(df.index.month.unique().min(),df.index.month.unique().max()+1,1)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_title(yr, y=1.0, pad=-14)
        # Remove ticks outside of data range
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(df.index.month.unique().min(),df.index.month.unique().max()+1,1)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_ylim((0.,32000.))

    axs[1].set_xlabel('Time (1st of month)', fontsize='x-large') # Time in YYYY-MM
    if key == 'total_meltwater_per_day':
        axs[0].set_ylabel('Total inferred surface melt-\nwater over study area in '+r'km$^2$', fontsize='x-large')

    if filepath_to_save is not None:
        Path(filepath_to_save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath_to_save)

    plt.show()
    plt.close()

    #import pdb;pdb.set_trace()

