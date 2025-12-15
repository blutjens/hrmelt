"""
Uses the trained model to apply it to across all images
 in the dataset and save predictions in a results folder.
"""
import yaml
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
from osgeo import gdal # rasterio in dataloader uses
    # gdal. Need to import gdal to suppress warning msg. 
from pathlib import Path
from pprint import pprint
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from typing import Callable, Dict, List, Any
import torchvision.transforms.functional as F

from hrmelt.evaluate import evaluate
from hrmelt.dataset import HRMeltDataset
from hrmelt.utils.utils import lookup_torch_dtype
from hrmelt.utils.utils import MaskedLoss
from hrmelt.utils.utils import set_num_workers
from hrmelt.utils.utils import set_all_seeds
from hrmelt.utils.utils import save_tensor_as_tif
from hrmelt.utils.utils import get_size_of_tif

def create_list_of_yx_offsets_in_tif(
    tif_size: [int,int], 
    tile_size: [int,int], 
    stride: int,
    ) -> np.ndarray:
    """
    # Creates a list of y- and x- offsets of each tile in the full-scale
    #  image. This is done to allow loading tiles in batches. y
    Args:
        tif_size: size of the full tif [height,width] in px
        tile_size: size of each crop in the full tif [height,width] in px
        stride: Stride is the number of pixels between every tile's top
            left corner +1. If the combination of tile_size and stride
            would have skipped over the pixels at the boundary, this algorithm
            will add an extra tile at the boundary.
    Returns:
        offsets_list: List of y-, x-offsets of shape (n_tiles_in_tif, 2).
            Iterating over offsets_list will iterate over rows first; 
            then columns. offsets_list[:,0] is y-dim and offsets_list[:,1] is 
            x-dim.
    """
    height_tif, width_tif = tif_size

    # Specify the size of the tile
    height_tile = min(tile_size[0], height_tif) 
    width_tile = min(tile_size[1], width_tif)

    # Calculate the maximum offset to prevent going out of bounds
    max_y_offset = height_tif - height_tile
    max_x_offset = width_tif - width_tile

    # Compute x-, and y-offsets for every tile in the full-scale tif
    #  (we add +1. s.t. e.g., if the tif and tile are size 1, the list of 
    #   offsets is [0])
    y_offsets = np.arange(0, max_y_offset+1, stride)
    if y_offsets[-1] != max_y_offset:
        # Add last row, if the combination of tile_height and stride
        #  would have skipped over the pixels at the boundary
        y_offsets = np.concatenate((y_offsets, np.array([max_y_offset])))

    x_offsets = np.arange(0, max_x_offset+1, stride)
    if x_offsets[-1] != max_x_offset:
        x_offsets = np.concatenate((x_offsets, np.array([max_x_offset])))

    # Create a 2D grid of offsets
    y_grid, x_grid = np.meshgrid(y_offsets, x_offsets)
    # Stack the grid into a 2D array of shape (n, 2). 
    offsets_list = np.block([y_grid.reshape(-1, 1), x_grid.reshape(-1, 1)])

    return offsets_list

class HRMeltDatasetConvolution(HRMeltDataset):
    def __init__(self, cfg, split='val', verbose=False, stride=None):
        '''
            Child class of HRMeltDataset that will sweep over 
            all tiles in each tif. To do, this class will create
            a list of offsets of each tile that fits into a full
            scale tif, given a stride. Upon calling __getitem__
            this function will iterate first over every tile within
            a big tif and then over every big tif.
            
            There is only one list of offsets for every tif, so we 
            assume that each tif in dataset has the same size. 
        Args:
            cfg, split, verbose: see parent class
            stride int: Stride is the number of pixels+1 between every tile's top-left
             corner that is loaded into memory for prediction. If the stride
             is smaller than the img_size, each pixel in the predicted 
             image will be a weighted average of all predictions at that
             pixel.
        '''
        # Call the parent class constructor
        super().__init__(cfg=cfg, split=split, verbose=verbose)

        if stride is None:
            stride = cfg['img_size'][0]
        else:
            if stride > cfg['img_size'][0] or stride > cfg['img_size'][1]:
                raise ValueError(f'Configuration stride of {stride} is larger than img size of {cfg["img_size"]}.')
            if 'erode_size' in cfg:
                if stride > (cfg['img_size'][0] - cfg['erode_size']) or stride > (cfg['img_size'][1] - cfg['erode_size']):
                    raise ValueError(f'Configuration stride of {stride} is larger than img size, {cfg["img_size"]}, minus erode_size, {cfg["erode_size"]}.')
            self.cfg['prediction_stride'] = stride

        # Get the path of one sample tif
        idx_tif = 0 # Index of each full-scale tif in the dataset
        sample_path = self.data[idx_tif][list(self.data[idx_tif].keys())[0]]
        self.tif_size = get_size_of_tif(str(sample_path))

        self.offsets_list = create_list_of_yx_offsets_in_tif(
            self.tif_size,
            tile_size=cfg['img_size'],
            stride=stride
        )

    def __len__(self):
        '''
            Returns the length of the dataset, which is the number of tiles
        '''
        n_tifs = len(self.data)
        n_tiles_in_tif = len(self.offsets_list)
        return n_tifs * n_tiles_in_tif
    
    def __getitem__(self, idx):
        """
        Args:
            idx Index into all tiles in dataset
        """
        # offsets_list contains the position of the top-left corner of each tile. 
        #   Each tif uses the same offsets_list
        n_tiles_in_tif = len(self.offsets_list)

        idx_tif = np.floor(idx / n_tiles_in_tif).astype(int) # Index of each full-scale tif in dataset

        idx_offset = idx % n_tiles_in_tif # Index of each tile in the full-scale tif

        # Get offset position of one tile from list
        offsets = self.offsets_list[idx_offset]
        offsets = [offsets[0].item(), offsets[1].item()] # convert numpy int to python int

        return super().__getitem__(idx_tif, offsets)

class Prediction(object):
    def __init__(self, tif_size, img_size, offsets_list, device, dtype, erode_size=0):
        """
        Class that holds a running sum of all predictions on the current tif.
        This can used, e.g., to convolve the trained model across the image and to
        create a new full-scale tif.
        """
        height_tif, width_tif = tif_size # size of predicted tif
        self.shape = (1, height_tif, width_tif) # shape of prediction
        self.height_tile = min(img_size[0], height_tif) # size of each tile
        self.width_tile = min(img_size[1], width_tif)
        self.device = device # cpu or gpu
        self.dtype = dtype # datatype of prediction
        self.offsets_list = offsets_list # list with offsets of each tile. see Dataset class

        # Declare full-scale .tif prediction
        self.pred_sum = None
        self.counts = None
        self.tile_idx_in_tif_counter = 0

        self.erode_size = erode_size
        # Instantiate full-scale tif predictions
        self.reset_counters()

    def reset_counters(self):
        """
        resets counters for convolution of model across full-scale tif
        """
        # Running sum of predictions in each pixel of the full-scale tif
        self.pred_sum = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        # Running sum of how many predictions were made per pixel
        self.counts = torch.zeros(self.shape, device=self.device, dtype=int)
        # Running sum of how many tiles have been computed in current tif.
        #  indexes next tile in tif; resets after every tif
        self.tile_idx_in_tif_counter = 0 

        return 1
    
    def compute_pred_sum(self, pred):
        """
        This function takes in predicted tiles and adds them to the full sized image array.
        It also keeps track of how many predictions there are for any given pixel to later average them out.
        Args:
            pred torch.Tensor(batch_size, n_ch, h, w) Batch of predicted tiles with different offsets wrt. full-size tif
        """
        # Add each predicted tile onto pred_sum
        # todo: parallelize this
        for j, pred_tile in enumerate(pred):  # (n_ch, h, w)
            # get the offsets to know where tiles should be added
            y_offset, x_offset = self.offsets_list[self.tile_idx_in_tif_counter]

            # get the min and max offset values which are at the beginning and end of the offset list
            y_max, x_max = self.offsets_list[-1]
            y_min, x_min = self.offsets_list[0]

            # define erosion if the tile is not on the edge of the full sized img
            y_erosion_top = self.erode_size
            y_erosion_bottom = -self.erode_size
            x_erosion_left = self.erode_size
            x_erosion_right = -self.erode_size

            # remove erosion if the tile is on a edge of the full sized tif. This will make sure the size
            # of the full image stays the same
            if y_offset == y_min:
                y_erosion_top = 0

            if y_offset == y_max:
                y_erosion_bottom = 0

            if x_offset == x_min:
                x_erosion_left = 0

            if x_offset == x_max:
                x_erosion_right = 0

            # if there is no erosion the added tile should not be eroded on at the end of the axis
            tile_y_bottom = y_erosion_bottom if y_erosion_bottom != 0 else None
            tile_x_right = x_erosion_right if x_erosion_right != 0 else None

            # here all predictions get added to a array that will later make up the full sized prediction
            # 1. access where the tile should be added with erosion decreasing accessed tile size
            # 2. adding pixel values of the predicted tile while removing eroded pixels from the prediction
            self.pred_sum[:,
            y_offset + y_erosion_top:y_offset + self.height_tile + y_erosion_bottom,
            x_offset + x_erosion_left:x_offset + self.width_tile + x_erosion_right] \
                += pred_tile[:, y_erosion_top:tile_y_bottom, x_erosion_left:tile_x_right]

            # this array keeps count of where pixel values have been added
            # to later average out where more than one pixel value has been added
            # accesses the same tile size and offset as above and adds 1 to the selected values
            self.counts[:,
            y_offset + y_erosion_top:y_offset + self.height_tile + y_erosion_bottom,
            x_offset + x_erosion_left:x_offset + self.width_tile + x_erosion_right] += 1

            # keeps count of tiles added to know when a full sized tif has been created
            self.tile_idx_in_tif_counter += 1
        return 1

    def compute_pred_avg(self):
        """
        Calculates pred_sum / counts to get the average prediction
        Returns:
            pred_avg torch.Tensor((self.shape)): average prediction of full-scale tif
        """
        # Current tif ends with tiles in current batch. Only occurs if n_tiles_in_tif is truly divisible by batch_size
        assert not torch.any(self.counts==0), 'There was no predictions for some area of prediction.'

        # Calculate average prediction
        pred_avg = self.pred_sum / self.counts

        return pred_avg

@torch.inference_mode()
def predict(model, 
        dataloader, 
        device, 
        cfg=None,
        compress=False,
        verbose=False,
        specific_pred_path=None,
        ):
    '''
    Uses the model to create and store
    predictions of large-scale tifs. The large-scale tif
    is loaded img-by-img via the dataloader.

    Args:
        model torch.nn.Module
        dataloader torch.utils.data.dataloader.DataLoader
        device torch.device: device, e.g., cpu or gpu
        cfg dict: Config file with all hyperparameters.
        compress: this will also save the predictions as compressed png.
        specific_pred_path path: path used to save prediction. If not specified the cfg pred_path is used.
    Returns:
        a array of paths to the predicted images
    '''
    model.eval()
    dtype = lookup_torch_dtype(cfg['dtype'])

    n_tiles = len(dataloader.dataset) # number of tiles in dataset
    n_batches = len(dataloader) # number of batches in dataset
    n_tiles_in_tif = len(dataloader.dataset.offsets_list) # number of tiles per tif
    n_batches_in_tif = float(n_tiles_in_tif) / float(dataloader.batch_size) # number of batches per tif
    if verbose:
        print('# of tifs: ', len(dataloader.dataset.data))
        print('# of tiles per tif: ', n_tiles_in_tif)
        print('# of tiles in full dataset (n_tifs * n_tiles_per_tif): ', n_tiles)

    if True:
        # Initialize the full-scale .tif prediction
        prediction = Prediction(tif_size=dataloader.dataset.tif_size, 
                                img_size=dataloader.dataset.cfg['img_size'],
                                offsets_list=dataloader.dataset.offsets_list, 
                                device=device, 
                                dtype=dtype,
                                erode_size=cfg['erode_size']
                                )

        pred_paths = []
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=cfg['amp']):
            with tqdm(total=n_tiles, desc='prediction', unit='tile') as pbar:
                for i, batch in enumerate(dataloader):
                    inputs, _, targets_mask, meta = batch
                    batch_size = inputs.shape[0] # batch_size can vary with dataloader.drop_last = False

                    if batch_size > 1 and inputs.shape[-2:] == (2863,1633):
                        raise NotImplementedError(f'Batch size of {batch_size} too large. The '\
                            'predict.py currently assumes that a batch can only contain tiles from '\
                            ' <2 large-scale tifs. Reduce batch of img size.')

                    inputs = inputs.to(device=device, dtype=dtype, memory_format=torch.channels_last)
                    targets_mask = targets_mask.to(device=device, dtype=dtype)
                    
                    pred = model(inputs)

                    # clip output to min, max values
                    pred = torch.clip(pred, cfg['min_melt'], cfg['max_melt'])
                    # todo: denormalize. Meltwater is currently not normalized, but need to denormalize if it is.
                    
                    # todo: add option to apply landmask
                    if cfg['apply_landmask_to_predictions']:
                        if dataloader.dataset.split == 'deploy':
                            # Mask has 1 value for invalid pixels.
                            pred = torch.mul(pred, (1 - targets_mask))
                        else:
                            raise ValueError('cfg[apply_landmask_to_predictions]=True only implemented if'\
                                             'split is "deploy"')
                    
                    # Number of tiles in current batch that belong to current prediction tif.
                    #  The last batch of the current tif might contain tiles of two tifs. In that case,
                    #  n_tiles_in_current_tif will be the number of leftover tiles
                    n_leftover_tiles = n_tiles_in_tif - prediction.tile_idx_in_tif_counter
                    n_tiles_in_batch_of_current_tif = min(batch_size, n_leftover_tiles)
                    
                    # Add the current batch of predictions to the current prediction tif
                    prediction.compute_pred_sum(pred[:n_tiles_in_batch_of_current_tif,...])

                    # If all tiles in tif have successfully been predicted:
                    if prediction.tile_idx_in_tif_counter == n_tiles_in_tif:
                        
                        # Compute the average prediction for every pixel in the tif.
                        pred_avg = prediction.compute_pred_avg()

                        # Postprocess here? eg add landmask.
                        # pred_avg = postprocess(pred_avg, cfg)
                        # Get path of current tif and storage location of predicted tif 
                        if 'path_melt' in meta:
                            tif_path = Path(meta['path_melt'][0])
                        else:
                            tif_path = Path(cfg['data_root']) / Path(cfg['path_melt_reference'])
                        
                        if specific_pred_path is not None:
                            new_tif_path = str(Path(specific_pred_path) / Path(meta['filename'][0]))
                        elif dataloader.dataset.split == 'deploy':
                            new_tif_path = str(Path(cfg['path_deploy']) / Path(meta['filename'][0]))
                        else:
                            new_tif_path = str(Path(cfg['path_predictions']) / Path(meta['filename'][0]))

                        if compress:
                            # Save compressed image
                            path = Path(new_tif_path)
                            new_png_path = path.with_suffix('.' + 'png')

                            # Create directory
                            Path(new_png_path).parent.mkdir(parents=True,exist_ok=True)

                            save_image(pred_avg, str(new_png_path))
                            if verbose:
                                logging.info(f"\nSaved: {new_png_path}")

                        # Uncompressed images are stored as .tifs, s.t., evaluation metrics can be computed
                        save_tensor_as_tif(pred_avg, tif_path=str(tif_path), new_tif_path=str(new_tif_path),verbose=verbose)

                        # Init new prediction.
                        prediction.reset_counters()

                        # If the current batch contains tiles from next tif, we add them to the new prediction:
                        if n_tiles_in_batch_of_current_tif < batch_size:
                            prediction.compute_pred_sum(pred[n_tiles_in_batch_of_current_tif:,...])

                        pred_paths.append(new_tif_path)
                    pbar.update(batch_size)

                    # Plot the effect receptive field size.
                    #from hrmelt.utils.utils import plot_effective_receptive_field
                    #plot_effective_receptive_field(model,inputs=inputs,device=device)

    return pred_paths

def get_args():
    parser = argparse.ArgumentParser(description='Use the UNet to create full-scale predictions on validation set')
    parser.add_argument('--load', '-f', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--cfg_path', type=str, default='runs/unet/default/config/config.yaml', help='Path to config yaml')
    parser.add_argument('--parallel', action='store_true', default=False, help='Enable parallel training')
    parser.add_argument('--data_split', type=str, default='val', help='Split [train, val, or test] for which the'\
                         'predictions will be calculated')
    # parser.add_argument('--no_wandb', action='store_true', default=False, help='Disable wandb logs')
    parser.add_argument('--verbose', action='store_true', default=False, help='Set true to print verbose logs')
    parser.add_argument('--prediction_stride', type=int, default=None, help='Overwrite cfg[prediction_stride] argument')
    parser.add_argument('--erode_size', type=int, default=None, help='Overwrite cfg[erode_size] argument')
    parser.add_argument('--prediction_batch_size', type=int, default=None, help='Overwrite cfg[prediction_batch_size] argument')
    parser.add_argument('--apply_landmask_to_predictions', type=bool, default=None, help='Overwrite cfg[apply_landmask_to_predictions] argument')
    parser.add_argument('--path_time_interpolate_sar', type=str, default=None, help='Overwrite cfg[path_time_interpolate_sar].'\
                        'used to specify distinct path for deploy without modifying config.yaml')
    # parser.add_argument('--task_id', type=int, default=1, help='SLURM task id, when script is called in job array')
    # parser.add_argument('--num_tasks',type=int, default=1, help='Total number of SLURM tasks when script is called in job array')
    return parser.parse_args()

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    gdal.UseExceptions() # Enable gdal error messages (otherwise a warning is raise)

    # Import cfg
    cfg = yaml.safe_load(open(args.cfg_path, 'r'))
    
    # Init device
    if cfg['device'] == 'gpu':
        if torch.cuda.is_available():
            device_type = 'cuda'
        elif torch.backends.mps.is_available():
            device_type = 'mps'
        else:
            device_type = 'cpu'
    else:
        device_type = 'cpu'
    device = torch.device(torch.device(device_type))
    logging.info(f'Using device {device}')

    # Set seeds
    set_all_seeds(cfg['seed'], device=device.type, 
                  use_deterministic_algorithms=cfg['use_deterministic_algorithms'])

    # Update config with cmd line arguments
    cfg['in_channels'] = len(cfg['in_keys']) + len(cfg['in_keys_static']) + len(cfg['in_keys_aux'])
    if args.prediction_stride is not None:
        cfg['prediction_stride'] = args.prediction_stride
    if args.erode_size is not None:
        cfg['erode_size'] = args.erode_size
    if args.path_time_interpolate_sar is not None:
        cfg['path_time_interpolate_sar'] = args.path_time_interpolate_sar
    if args.apply_landmask_to_predictions is not None:
        cfg['apply_landmask_to_predictions'] = args.apply_landmask_to_predictions
    elif 'apply_landmask_to_predictions' not in cfg:
        cfg['apply_landmask_to_predictions'] = False
    if args.prediction_batch_size is not None:
        cfg['prediction_batch_size'] = args.prediction_batch_size
    
    if args.verbose:
        print('Default model configuration:')
        pprint(cfg)

    # Create dataset from validation set
    dtype = lookup_torch_dtype(cfg['dtype'])
    assert cfg['split_cfg'] == 'csv', ('predict.py currently only'
        'accepts image paths in .csv format')

    # Initialize dataset from .csv file with filenames
    dataset = HRMeltDatasetConvolution(cfg=cfg, 
                                       split=args.data_split,
                                       verbose=False,
                                       stride=cfg['prediction_stride'])

    # Create data loader
    num_workers = set_num_workers(cfg['num_workers'], parallel=args.parallel)
    print('num_workers', num_workers)
    def _init_fn(worker_id): # Set different seed in every worker. 
        # Note: this will reset numpy and random seeds on every epoch. 
        # So, we recommend to use torch.random instead of np.random or 
        # random. This function exists to ensure that 3rd party libraries,
        # such as upsample deterministic
        seed = cfg['seed'] + worker_id
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
    loader_args = dict(batch_size=cfg['prediction_batch_size'], num_workers=num_workers, 
                       pin_memory=True, worker_init_fn=_init_fn)
    dataloader = DataLoader(dataset, shuffle=False, drop_last=False, **loader_args)

    # Load model
    if cfg['model_key']=='unet':
        from hrmelt.models.unet.unet_model import UNet
        cfg['model_args'] = {
            'in_channels': cfg['in_channels'],
            'activation': cfg['activation'],
            'out_activation': cfg['out_activation'],
            'out_channels': cfg['out_channels'],
            'bilinear': cfg['bilinear'],
            'num_extra_convs': cfg['num_extra_convs'],
        }
        model = UNet(**cfg['model_args'])
    elif cfg['model_key']=='unet_smp':
        from hrmelt.models.unet_smp.model import UNet_smp
        cfg['model_args'] = {
            'in_channels': cfg['in_channels'],
            'out_activation': cfg['out_activation'],
            'encoder_name': cfg['encoder_name'],
            'encoder_depth': cfg['encoder_depth'],
            'encoder_weights': cfg['encoder_weights'],
            'decoder_use_batchnorm': cfg['decoder_use_batchnorm'],
            'out_channels': cfg['out_channels'],
        }
        model = UNet_smp(**cfg['model_args'])
    elif cfg['model_key']=='deeplabv3':
        from hrmelt.models.deeplabv3.deeplabv3_model import DeepLabV3Plus
        # rename some model hyperparameters
        cfg['classes'] = cfg['out_channels']
        # List all model hyperparameters
        cfg['model_args'] = {
            'in_channels': cfg['in_channels'],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            'activation': cfg['activation'],  # activation for decoder layers
            'out_activation': cfg['out_activation'],  # output activation function
            'encoder_name': cfg['encoder_name'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            'encoder_depth': cfg['encoder_depth'],  # input size has to be at least dividable by 2 encoder_depth times
            'encoder_weights': cfg['encoder_weights'],  # use `imagenet` pre-trained weights for encoder initialization
            'encoder_output_stride': cfg['encoder_output_stride'],
            'decoder_channels': cfg['decoder_channels'],  # if encoder depth or image size is changed this has to be updated
            'decoder_atrous_rates': cfg['decoder_atrous_rates'],
            'classes': cfg['classes'],  # model output channels (number of classes in your dataset)
            'batch_norm': cfg['batch_norm'],  # batch normalization active or not
            'upsampling': cfg['upsampling']
        }
        model = DeepLabV3Plus(**cfg['model_args'])
    elif cfg['model_key'] == 'linear_dem':
        from hrmelt.models.linear.linear_model import Linear
        cfg['model_args'] = {
            'in_channels': cfg['in_channels'],
            'out_activation': cfg['out_activation'],
            'out_channels': cfg['out_channels'],
            'model_key': cfg['model_key'],
            'n_models': cfg['n_models'],
            'time_channel_idx': cfg['time_channel_idx']
        }
        model = Linear(**cfg['model_args'])

    model = model.to(memory_format=torch.channels_last)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        try:
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from {args.load}')
        except:
            raise ValueError('Error loading model. Verify that hyperparameters in config.yaml '\
                             'match the hyperparameters of the loaded model.')
    model.to(device=device)

    if args.verbose:
        from torchsummary import summary
        print('Calculating # of model weights.')
        summary(model, (cfg['in_channels'],) + tuple(cfg['img_size']))

    predict_args = {   
        'model' : model,
        'dataloader': dataloader,
        'device' : device,
        'compress' : True,
        'cfg' : cfg,
        'verbose' : True
    }
    predict(**predict_args)

    # Calculate metrics on stored tifs --> write another function for this.