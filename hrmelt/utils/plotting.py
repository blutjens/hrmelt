import random
import time
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from hrmelt.dataset import HRMeltDataset
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors # plot_tas_annual_local_err_map
from hrmelt.utils.utils import set_num_workers
from osgeo import gdal

def plot_dataset_mar_over_melt(cfg):
    """
    Iterate through all images in dataloader and plot the full
    tif with MAR overlayed onto meltwater 
    """
    # Initialize dataset from .csv file with filenames
    dataset = HRMeltDataset(cfg=cfg, split='train', verbose=False)

    # Initialize dataloader
    num_workers = 1
    loader_args = dict(batch_size=cfg['batch_size'], num_workers=num_workers, pin_memory=True)
    dataloader = DataLoader(dataset, shuffle=True, **loader_args)

    print(f'Loaded dataset with {len(dataloader)} batches and {len(dataset)} images. \n'\
          f"Num_workers {num_workers}. Batch size {cfg['batch_size']}.")
    
    for i, batch in tqdm(enumerate(dataloader)):
        inputs, targets, targets_mask, meta = batch        
        # dataset.verbose=verbose
        fig, axs = plt.subplots(1, 2, figsize=(10,4))
        columns = 4 # len(dataloader.dataset.keys)
        imgs = torch.cat((inputs[0], targets_mask[0], targets[0]),dim=0)

        mar = inputs[0,0] # .cpu().numpy()
        plt.subplot(1, columns, 1)
        plt.imshow(mar.cpu().numpy())
        plt.title('mar')
        plt.axis('off')

        dem = inputs[0,1] # .cpu().numpy()
        plt.subplot(1, columns, 2)
        plt.imshow(dem.cpu().numpy())
        plt.title('dem')
        plt.axis('off')
        
        mask = targets_mask[0,0]
        plt.subplot(1, columns, 3)
        plt.imshow(mask.cpu().numpy())
        plt.title('mask')
        plt.axis('off')

        mar = inputs[0,0]
        melt = targets[0,0]
        plt.subplot(1, columns, columns)
        plt.imshow(melt.cpu().numpy(), cmap='viridis', interpolation='nearest')
        plt.imshow(mar.cpu().numpy(), cmap='Greys', interpolation='nearest', alpha=0.5)
        plt.title('melt + mar')
        plt.axis('off')

        plt.tight_layout()
        dir_figures = 'references/figures/all_data_mar_over_melt/'
        Path(dir_figures).mkdir(parents=True, exist_ok=True)
        timestamp = Path(meta['path_melt'][0]).stem # e.g., 2017_08_23
        plt.savefig(f"{dir_figures}{timestamp}.png")
    return 1

def plot_dataset(cfg, split='train', plot_raw_data=False, dpi=50, plot_time_interpolate_sar=True, plot_mask=False):
    """
    Iterate through all images in dataloader and plot the full
    tif of every in-/output channel. If cfg['split_cfg'] != csv,
    this function will just plot the full dataset without doing
    any train/val/test splits

    plot_raw_data: bool If true, plots raw data as it is loaded 
        from disk. If False, plots data as it is returned by dataloader.
    dpi int: resolution of plot. Default is very low to plot quickly
    plot_time_interpolate_sar: If False, removes time_interpolate_sar from in_keys 
        to avoid plotting it.
    plot_mask: If True, add plot of the binary mask
    """
    plot_titles = {'pmw': 'PMW',
                    'mar_wa1': 'MAR WA1',
                    'time_interpolate_sar': 'Time interpolate SAR',
                    'dem': 'DEM',
                    'masked_melt': 'Masked targets',
                    'melt': 'Targets'}

    if plot_raw_data:
        # Disable normalization and blurring of inputs
        cfg['normalize_inputs'] = False
        cfg['pmw_GaussianBlur_kernel_size'] = 1.
        cfg['pmw_GaussianBlur_sigma'] = 1.
        cfg['mar_wa1_GaussianBlur_kernel_size'] = 1.
        cfg['mar_wa1_GaussianBlur_sigma'] = 1.

    if not plot_time_interpolate_sar:
        if 'time_interpolate_sar' in cfg['in_keys']:
            cfg['in_keys'].remove('time_interpolate_sar')
    
    # Initialize dataset from .csv file with filenames
    dataset = HRMeltDataset(cfg=cfg, split=split, verbose=False)
    # Initialize dataloader
    num_workers = 1 # Increasing num_workers does not accelerate plotting, if 
        # the plotting is not implemented in parallel across CPU
    loader_args = dict(batch_size=cfg['batch_size'], num_workers=num_workers, pin_memory=True)
    dataloader = DataLoader(dataset, shuffle=False, **loader_args)

    print(f'Loaded dataset with {len(dataloader)} batches and {len(dataset)} images. \n'\
          f"Num_workers {num_workers}. Batch size {cfg['batch_size']}.")

    ## Init colormaps
    # load sample image, to get approximate ranges for colormap.
    # inputs, targets, targets_mask, meta = dataset.__getitem__(0)
    inputs, targets, targets_mask, meta = next(iter(dataloader))
    imgs = torch.cat((inputs[0], targets[0]),dim=0)
    plot_keys = cfg['in_keys'] + cfg['in_keys_static'] + ['masked_melt']
    titles = [plot_titles[plot_key] for plot_key in plot_keys]
    if plot_mask:
        imgs = torch.cat((imgs,targets_mask[0]),dim=0)
        titles.append('melt_mask')
    plot_position = {plot_key: i for i, plot_key in enumerate(plot_keys)} # order of plots
    if plot_time_interpolate_sar:
        # Define custom plot order
        plot_position = {'pmw': 1,
                         'mar_wa1': 0,
                         'dem': 2,
                         'time_interpolate_sar':3,
                         'masked_melt':4}
    # titles = cfg['in_keys'] + cfg['in_keys_static'] + ['melt', 'melt_mask']
    # Init colormaps
    cmaps = len(imgs) * ['viridis'] # init colormaps
    ticks = len(imgs) * [None] # init ticks
    cnorms = len(imgs) * [None] # init colornorms
    xlabels = len(imgs) * [None] # init xlabels
    
    # Set PMW ticks and colormap
    if 'pmw' in cfg['in_keys']:
        pmw_idx = dataset.get_channel_idx(meta,'pmw')
        pmw_min = 150. # in Kelvin
        pmw_max = 273. # in Kelvin
        pmw_vcenter = 225. # in Kelvin
        if cfg['normalize_inputs']:
            pmw_min = (pmw_min- cfg['mean_pmw']) / cfg['std_pmw'] # in Kelvin and then normalized
            pmw_max = (pmw_max- cfg['mean_pmw']) / cfg['std_pmw'] # in Kelvin and then normalized
            pmw_vcenter = (pmw_vcenter- cfg['mean_pmw']) / cfg['std_pmw'] # in Kelvin and then normalized
            print(f'PMW is scaled around: \n'
                f'\t\t normalized: {pmw_min}, {pmw_vcenter}, {pmw_max} \n'
                f'\t\t unnormalized: {np.array([pmw_min, pmw_vcenter, pmw_max])*cfg["std_pmw"]-cfg["mean_pmw"]}')
        ticks[pmw_idx] = np.concatenate(
            (np.linspace(pmw_min, pmw_vcenter, 2, endpoint=False), 
            np.linspace(pmw_vcenter, pmw_max, 2))).astype(int)
        cnorms[pmw_idx] = colors.TwoSlopeNorm(vmin=pmw_min, vcenter=pmw_vcenter, vmax=pmw_max) # center colorbar around zero
        cmaps[pmw_idx] = 'coolwarm'
        xlabels[pmw_idx] = 'Brightness tempe-\nrature in K'

    # Set MAR ticks and colormap
    if 'mar_wa1' in cfg['in_keys']:
        mar_wa1_idx = dataset.get_channel_idx(meta,'mar_wa1')
        mar_wa1_min = 0.
        mar_wa1_max = 0.01 # max_mar_wa1 is usually 1.
        if cfg['normalize_inputs']:
            mar_wa1_min = (0. - cfg['mean_mar_wa1']) / cfg['std_mar_wa1'] # mar is originally on range [0,1]
            mar_wa1_max = (cfg['max_mar_wa1'] - cfg['mean_mar_wa1']) / 0.25 # max_mar_wa1 is usually 1.
            print(f'MAR WA1 is scaled around: \n'
                f'\t\t normalized: {mar_wa1_min}, {mar_wa1_max} \n'
                f'\t\t unnormalized: {np.array([mar_wa1_min, mar_wa1_max])*cfg["std_mar_wa1"]-cfg["mean_mar_wa1"]}')
        cnorms[mar_wa1_idx] = colors.Normalize(vmin=mar_wa1_min, vmax=mar_wa1_max)
        ticks[mar_wa1_idx] = np.linspace(mar_wa1_min, mar_wa1_max, 3)
        xlabels[mar_wa1_idx] = 'Liquid water content \nin kg/kg' #top \nmeter of snow in 

    # Set DEM colormap
    if 'time_interpolate_sar' in cfg['in_keys']:
        time_interpolate_sar_idx = dataset.get_channel_idx(meta,'time_interpolate_sar')
        xlabels[time_interpolate_sar_idx] = 'Surface meltwater \nfraction per 100m'

    # Set DEM colormap
    if 'dem' in cfg['in_keys_static']:
        dem_idx = dataset.get_channel_idx(meta,'dem')
        cmaps[dem_idx] = 'terrain'
        xlabels[dem_idx] = 'Elevation in m'

    xlabels[-1] = 'Surface meltwater \nfraction per 100m'

    # iterate over all images
    for i, batch in tqdm(enumerate(dataloader)):
        inputs, targets, targets_mask, meta = batch

        # dataset.verbose=verbose
        columns = len(plot_keys)
        fig, axs = plt.subplots(1, columns, figsize=(12,4), dpi=dpi)
        imgs = torch.cat((inputs[0], targets[0]),dim=0)
        if plot_mask:
            imgs = torch.cat((imgs,targets_mask[0]),dim=0)

        for i, img in enumerate(imgs):
            axs_id = plot_position[plot_keys[i]]
            mappable = axs[axs_id].imshow(img.cpu().numpy(), cmap=cmaps[i], norm=cnorms[i])
            if plot_keys[i] == 'masked_melt':
                # Plot mask overlayed onto targets
                axs[axs_id].imshow(targets_mask[0,0].cpu().numpy(), cmap='Set1', interpolation='nearest', alpha=0.8*targets_mask[0,0].cpu().numpy())
            cbar = fig.colorbar(mappable=mappable, ax=axs[axs_id], orientation='horizontal', ticks=ticks[i], fraction=0.025, pad=0.01, label=xlabels[i]) # 
            if plot_keys[i] == 'mar_wa1':
                axs[axs_id].xaxis.set_major_formatter('{x:.1f}')
            axs[axs_id].set_title(titles[i])
            axs[axs_id].axes.set_axis_off()
            #axs[axs_id].axes.get_xaxis().set_visible(False)
            #axs[axs_id].axes.get_yaxis().set_visible(False)
            # cbar.ax.tick_params(axis='x', labelrotation=45)

        plt.tight_layout()
        #plt.subplots_adjust(bottom=.2, hspace=0.0) 
        #plt.subplots_adjust(wspace=0.4, 
        #                    hspace=-1.0)
        dir_figures = f'references/figures/all_data/{cfg["data_key"]}/{split}/'
        Path(dir_figures).mkdir(parents=True, exist_ok=True)
        timestamp = Path(meta['path_melt'][0]).stem # e.g., 2017_08_23
        plt.savefig(f"{dir_figures}{timestamp}.png")
        plt.close()

    return 1

def split_axes(axs):
    # Hide spines and ticks between axes
    for ax in axs[:-1]:
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelright=False)
        ax.tick_params(right = False) 
    for ax in axs[1:]:
        ax.spines['left'].set_visible(False)
        ax.tick_params(left = False) 
    axs[-1].yaxis.tick_right()

    # Add cut-out diagonal lines
    d = .015  # size of diagonal lines
    for ax in axs[:-1]:
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((1-d, 1+d), (-d, +d), **kwargs)
        ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    for ax in axs[1:]:
        kwargs.update(transform=ax.transAxes)
        ax.plot((-d, +d), (1-d, 1+d), **kwargs)
        ax.plot((-d, +d), (-d, +d), **kwargs)
    
    # Adjust spacing between plots
    plt.subplots_adjust(wspace=0.02)

    return axs

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--parallel', action='store_true', default=False, help='Enable parallel')
    parser.add_argument('--cfg_path', type=str, default='runs/unet/sample/config/config.yaml', help='Path to config yaml')
    parser.add_argument('--plot_dataset', action='store_true', default=False, help='Plot every image in the dataset')
    parser.add_argument('--split', type=str, default='val', help='Split for which the data is plotted. Use train, val, or test.')
    parser.add_argument('--dpi', type=int, default=500, help='Resolution for creating plots. Can be set to, e.g., 50 for rapid prototyping')
    return parser.parse_args()

if __name__ == "__main__":
    """
    """
    gdal.UseExceptions() # Enable gdal error messages. 

    args = get_args()
    random.seed(0)
    torch.manual_seed(0)
    #-------------
    # Plot MAR overlayed onto melt on all images in the dataset
    #-------------
    """
    cfg = yaml.safe_load(open('runs/unet/data_2017_to_20/config/config.yaml', 'r'))
    cfg['img_size'] = [2863,1633]
    cfg['batch_size'] = 1
    cfg['in_keys'] = ['mar_wa1']

    plot_dataset_mar_over_melt(cfg)
    """

    #-------------
    # Plot all images in the dataset
    #-------------
    if args.plot_dataset:
        cfg = yaml.safe_load(open(args.cfg_path, 'r'))
        cfg['img_size'] = [2863,1633]
        cfg['batch_size'] = 1
        plot_dataset(cfg, split=args.split, 
                     plot_raw_data=True, dpi=args.dpi,
                     plot_time_interpolate_sar=True,
                     )