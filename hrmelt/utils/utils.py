import os
import random
import yaml
import numpy as np
from typing import Callable, Dict, List, Any, Union, Tuple, Sequence, Optional
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
import rasterio # to save tifs
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from osgeo import gdal

'''
Python utils file with loss functions, and misc 
'''

def apply_mask(x, mask, threshold=0.5):
    """
    Sets all values in x where the mask is below
     a threshold to zero.

    Args:
        x torch or np array of any shape.
        mask torch or np.array with same shape as x. E.g., landmask.     
        threshold: real value
    """
    # Ensure both arrays have the same shape
    assert x.shape == mask.shape, "Arrays must have the same shape"

    # Set values to 0. where the threshold is exceeded
    x[mask > threshold] = 0.
    return x

def lookup_torch_activation(activation: str) -> Any:
    """
    Returns torch activation function given a string
    """
    if activation == 'gelu':
        return nn.GELU()
    elif activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'linear':
        return nn.Identity()
    elif activation == 'sigmoidcutoff':
        return SigmoidCutoff()
    elif activation == 'tanhshelf':
        return TanhShelf()
    else:
        return nn.Identity()

class SigmoidCutoff(nn.Module):
    def __init__(self):
        super().__init__()
        self.sharpness = torch.tensor(10.) # 
        # self.sharpness = nn.Parameter(torch.tensor(10.))

    def forward(self, x):
        return torch.sigmoid(-torch.pow(x,6.) + self.sharpness)

class TanhShelf(nn.Module):
    def __init__(self):
        super().__init__()
        # self.sharpness = torch.tensor(4.) # nn.Parameter(torch.tensor(4.))
        self.sharpness = torch.tensor(4.)

    def forward(self, x):
        return 0.5 * (torch.tanh(self.sharpness + x) + torch.tanh(self.sharpness - x))

def _worker_init_fn(worker_id, seed):  # Set different seed in every worker.
    # Note: this will reset numpy and random seeds on every epoch.
    # So, we recommend to use torch.random instead of np.random or
    # random. This function exists to ensure that 3rd party libraries,
    # such as upsample deterministic
    # Enable gdal error messages (otherwise each CPU raises a warning message at the start of each epoch)
    gdal.UseExceptions()

    worker_seed = seed + worker_id
    os.environ['PYTHONHASHSEED'] = str(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_size_of_tif(path):
    """
    # Get the size of the tif
    path: str path to full-scale tif. Only used to get dimensions
    """
    ds = gdal.Open(path)
    width_tif = ds.RasterXSize
    height_tif = ds.RasterYSize
    del ds
    return height_tif, width_tif

def save_tensor_as_tif(tensor, tif_path, new_tif_path, verbose=True):
    """
    Saves a torch tensor as tif with the same metadata from 
    an existing tif.
    Args:
        tensor torch.Tensor: tested with shape (1, height, width)
        tif_path str: Path to .tif file that contains desired metadata
        new_tif_path: Path to new .tif file that will be created
    """
    # Extract the metadata from the existing tif file
    with rasterio.open(tif_path) as src: 
        meta = src.meta

    # Create directory
    Path(new_tif_path).parent.mkdir(parents=True,exist_ok=True)

    # Write the array to a new tif file with the same metadata
    with rasterio.open(new_tif_path, 'w', **meta) as dst: 
        dst.write(tensor.cpu().numpy())
    
    # Save the new tif file
    if verbose:
        print(f"Saved {new_tif_path}")

def calculate_means_stds_across_dataset(dataset, verbose=False):
    # Initialize running sums
    n_channels = len(dataset.keys)
    sum_of_means = torch.zeros(n_channels, dtype = dataset.dtype)
    sum_of_stds = torch.zeros(n_channels, dtype = dataset.dtype)

    # Deactivate any normalization in dataloader 
    dataset.normalize = False
    
    # Calculate running sum
    for idx in tqdm(range(len(dataset))):
        inputs, targets, targets_mask, meta = dataset.__getitem__(idx=idx)
        imgs = torch.cat((inputs, targets_mask, targets),dim=0)
        assert imgs.shape[0] == len(dataset.keys), 'Dataset.keys does not match dataset.__getitem__ output.'
        assert imgs.dtype == dataset.dtype, 'dataset.dtype does not match __getitem__.dtype'

        # modis_mean = ...
        
        # melt_mean = np.ma.array(melt, mask=mask_nans).mean()
        # melt_std = np.ma.array(melt, mask=mask_nans).std()

        # Add channel wise mean and std to running sum
        sum_of_means += imgs.mean(dim=(1,2))
        sum_of_stds += imgs.std(dim=(1,2))
    dataset.normalize = True

    # Divide totol sum by number of entries in dataset
    means = sum_of_means / torch.tensor(len(dataset), dtype=dataset.dtype)
    stds = sum_of_stds / torch.tensor(len(dataset), dtype=dataset.dtype)
    if verbose:
        print('means', means)
        print('stds', stds)

    # Print out the means and stds to be copied into experiments.default.config.config.yaml
    np.set_printoptions(precision=100)

    means = means.cpu().numpy()
    stds = stds.cpu().numpy()
    for (key, mean) in zip(dataset.keys, means):
        print(f'mean_{key}: {mean}')
    for (key, std) in zip(dataset.keys, stds):
        print(f'std_{key}: {std}')
            
    print(f'')
    return means, stds

def calculate_target_histogram(dataloader=None, n_bins=10, 
    binarize_data=False,
    mask_invalid_pixels=True):
    """
    Calculates and plots a histogram over all (valid pixels) 
    in the targets from the dataloader. Assumes that targets 
    are in the shape [batch_size, 1, h, w]. 
    
    Args:
        dataloader torch.data.utils.Dataset with all images
        n_bins : number of bins in histogram
        binarize_data : if True, rounds up every value in dataset to the closest integer. 
            Use this and pass n_bins = 2 to calculate binary class frequencies.
        mask_invalid_pixels : if True, will mask invalid pixels and not exclude them 
            from the calculation of histograms
    """
    # Deactivate any normalization in dataloader 
    try:
        dataloader.dataset.normalize = False
    except:
        pass
    
    # Initialize histogram with empty bins that will count the pixel values
    hist = torch.zeros(n_bins)
        
    for i, batch in tqdm(enumerate(dataloader)):
        # Load one tile
        _, targets, targets_mask, meta = batch

        # Calculate histogram only over valid pixels.
        if mask_invalid_pixels:
            pixel_values = targets[targets_mask==False]

        if binarize_data:
            pixel_values = pixel_values.ceil()

        # Calculate histogram values of all images in one batch
        hist_freqs_in_batch, hist_bin_values = torch.histogram(pixel_values, bins=n_bins)
        
        # Add histogram values to running sum
        hist += hist_freqs_in_batch # .sum(dim=0)

    # Normalize the histogram to get the relative frequency per pixel value
    hist = hist / hist.sum()
    
    try:
        dataloader.dataset.normalize = True
    except:
        pass
    
    return hist

def set_all_seeds(seed, device='cpu',
                  use_deterministic_algorithms=False,
                  warn_only=False):
    """
    sets all seeds. 
    See src: https://github.com/pytorch/pytorch/issues/7068
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        print('in utils.py -> set_all_seeds cuda')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic=True
    # sets, e.g., nn.ConvTranspose2d to deterministic
    torch.use_deterministic_algorithms(mode=use_deterministic_algorithms, warn_only=warn_only)

def set_num_workers(num_workers=None, parallel=False):
    '''
    Sets the number of workers. Highest priority has optional 
    argument. If that is not given, the number of 
    workers is set the number of CPUs.
    Args:
        num_workers int: desired number of workers
        parallel bool: if true, model is run in parallel
    '''
    if num_workers:
        num_workers = num_workers
    elif parallel:
        num_workers = os.cpu_count()
    else:
        num_workers = 0
    return num_workers

def plot_effective_receptive_field(model, inputs, device):
    """
    Plot the effective receptive field of a model by plotting
      the model output for an input that is all zeroes except for 
      the center pixel

    model torch module
    inputs torch.shape(batch_size, n_channels, img_size[0], img_size[1])
    device torch.device
    """
    import math
    img_size = inputs.shape[2:]
    tst_inputs = torch.zeros(inputs.shape).to(device=device)
    center = [math.floor(i / 2.) for i in img_size]
    tst_input = tst_inputs[:1,...]
    tst_input[:,:,center[0],center[1]] = 1000000*torch.ones(tst_input[:,:,center[0],center[1]].shape)
    tst_pred = model(tst_input)
    import matplotlib.pyplot as plt
    receptive_field = torch.where(tst_pred > 0., 1., 0.).squeeze()
    fig, axs = plt.subplots(1, 1, dpi=300)
    axs.imshow(receptive_field.cpu().numpy())
    dir_figures = 'references/figures/receptive_field/'
    if model.__class__.__name__ == 'DeepLabV3Plus':
        figname = 'deeplabv3'
    else:
        figname = 'model'
    Path(dir_figures).mkdir(parents=True, exist_ok=True)
    figpath = f"{dir_figures}{figname}.png"
    plt.savefig(figpath)
    print(f'Plotted effective receptive field in {figpath}')
    plt.close()

def lookup_loss_function(loss_name: str):
    """
    Returns loss function given a string
    """
    if loss_name == 'l1':
        return nn.L1Loss
    elif loss_name == 'bceloss':
        return nn.BCELoss
    elif loss_name == 'nllloss':
        return nn.NLLLoss
    elif loss_name == 'accuracy':
        from hrmelt.eval.metrics import NCorrectPreds
        return NCorrectPreds
    else:
        return nn.MSELoss

class MaskedLoss(torch.nn.Module):
    '''
    A loss function wrapper that calculates the loss on unmasked values only.

    This module wraps around standard loss functions (such as L1 or L2) and applies them to the model predictions and
    ground truth targets, while excluding masked values from the loss calculation.
    Args:
        loss_name (str): loss functions to be used either 'l1', 'l2'
        reduction (str): loss reduction to apply to the batch of loss values either 'none', 'mean', 'sum'.
        **kwargs: miscellaneous arguments that might be required by the loss function
    '''
    def __init__(self, loss_name: str = 'l1',
                 reduction: str = 'mean',
                 **kwargs):
        super(MaskedLoss, self).__init__()
        Criterion = lookup_loss_function(loss_name)
        self.criterion = Criterion(reduction = 'none', **kwargs)
        self.reduction = reduction
        
    def forward(self, input, target, mask):
        """
        Input:
            input torch.Tensor(batch_size, out_channels, height, width): Model prediction
            target torch.Tensor(batch_size, out_channels, height, width): Ground-truth target
            mask torch.Tensor(batch_size, out_channels, height, width, dtype=float32):
                with 1. for masked values and 0. for unmasked values
        Returns:
            loss: torch.Tensor(batch_size)
        """
        loss = self.criterion(input, target) # dims: (batch_size, out_ch, height, width)
        # First calculate the loss over all, but the batch dimension.
        all_dims_but_first = tuple(range(1, len(loss.shape)))
        # Cumulative loss over all filled, unmasked pixels.
        loss = (loss * (1.-mask)).sum(dim=all_dims_but_first) # dims: (batch_size)
        # Total number of valid, unmasked pixels
        num_valid_pixels = (1.-mask).sum(dim=all_dims_but_first) # dims: (batch_size)
        # Set to one, in case all pixels in an image are masked, to avoid division by zero
        num_valid_pixels[num_valid_pixels==0] = 1. 
        # Calculate average loss per valid pixel
        loss_pixelwise = loss / num_valid_pixels # dims: (batch_size)

        # Optionally reduce loss over batch dimension
        if self.reduction == 'mean':
            loss_pixelwise = loss_pixelwise.mean() # dims: ()
        elif self.reduction == 'sum':
            loss_pixelwise = loss_pixelwise.sum()
        return loss_pixelwise

def lookup_torch_dtype(dtype_name: str) -> Any:
    """
    Returns torch dtype given a string
    """
    if dtype_name == 'float16' or dtype_name == 'half':
        return torch.float16
    elif dtype_name == 'float32' or dtype_name == 'float':
        return torch.float32
    elif dtype_name == 'float64' or dtype_name == 'double':
        return torch.float64
    else:
        raise NotImplementedError('only float32 implemented')

def generate_dicts_recursive(input_dict, current_dict=None, depth=0):
    '''
    Recursively generates a list of dictionaries containing
    all possible combinations of the values
    source: chat gpt-3.5
    Args:
        keys list(): List of dictionary keys
        input_dict dict(): Input dictionary
    '''
    keys = list(input_dict.keys())

    if current_dict is None:
        current_dict = {}

    if depth == len(keys):
        return [current_dict]

    key = keys[depth]
    values = input_dict[key]
    result_dicts = []

    for value in values:
        new_dict = current_dict.copy()
        new_dict[key] = value
        result_dicts.extend(generate_dicts_recursive(input_dict, new_dict, depth + 1))

    return result_dicts

def init_sweep_config(cfg, path_sweep_cfg, task_id=1, num_tasks=1, task_id_offset=0):
    '''
    Updates the cfg with a randomly drawn combination of 
    hyperparameters from the sweep config.

    task_id_offset int: offset to task ids
    '''
    # Update logging paths
    cfg['path_wandb'] = Path(cfg['path_sweep'] / Path(f'task-{task_id+task_id_offset}'))
    cfg['path_checkpoints'] = Path(cfg['path_sweep'] / Path(f'task-{task_id+task_id_offset}/checkpoints'))
    cfg['path_eval'] = Path(cfg['path_sweep'] / Path(f'task-{task_id+task_id_offset}/eval_during_training'))
    Path(cfg['path_wandb']).mkdir(parents=True, exist_ok=True)
    Path(cfg['path_checkpoints']).mkdir(parents=True, exist_ok=True)
    Path(cfg['path_eval']).mkdir(parents=True, exist_ok=True)

    # Update config with sweep parameters
    sweep_cfg = yaml.safe_load(open(cfg['path_sweep_cfg'], 'r'))
    # Initialize list of all possible cfg combinations
    list_of_sweep_cfgs = generate_dicts_recursive(sweep_cfg)
    print(f'Running {num_tasks}/{len(list_of_sweep_cfgs)} random sweep configurations on all tasks.')
    # Randomly shuffle the combinations and then draw the element with index
    # task.id. This is necessary because all tasks run on different GPUs, but
    # share the same random seed.
    random.shuffle(list_of_sweep_cfgs)
    current_sweep_cfg = list_of_sweep_cfgs[task_id-1] # minus 1 switches from 1 to zero indexing

    # Update the main config with the parameters chosen for this sweep
    cfg.update(current_sweep_cfg)
    print('Choosing sweep configuration:')
    pprint(current_sweep_cfg)

    return cfg