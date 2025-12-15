"""
Computes common metrics
"""
import numpy as np
import torch
# For R2

# For MaskedSSIM
from typing import Callable, Dict, List, Any, Union, Tuple, Sequence, Optional
from torch import Tensor
from torch.nn import functional as F

class MaskedSSIM(torch.nn.Module):
    def __init__(
            self,
            device,
            **kwargs: Any,
        )-> None:
        """Calculate Masked SSIM
        To calculate the Masked SSIM, we set all invalid values in the targets and input to zero
        before computing a full image of SSIM values, using torchmetrics. Then, The SSIM map is 
        averaged across all valid pixels. This implementation is not perfect: the SSIM value of valid 
        pixels that border invalid pixels will be slightly better than other valid pixels, 
        because the size of the Gaussian kernel is > 1, but we accept that.

        Args:
            device: torch.device
            **kwargs: See https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html
        """
        super(MaskedSSIM, self).__init__()

        from torchmetrics.image import StructuralSimilarityIndexMeasure

        # Need to return full image to add mask on it.
        kwargs['return_full_image'] = True
        kwargs['reduction'] = None
        self.ssim = StructuralSimilarityIndexMeasure(**kwargs)
        self.ssim = self.ssim.to(device)

    def forward(self,
            input: Tensor,
            target: Tensor,
            mask: Tensor = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            input: estimated image. Shape (B, C, H, W) or (B, C, D, H, W)
            target: ground truth image. Shape (B, C, H, W) or (B, C, D, H, W)
            mask: masked values that should not be considered in the metric calculation. 
              Assumes 0 for valid and 1 for invalid pixels.
        Returns:
            mean_ssim: The SSIM score between input and target factoring out mask. The SSIM score ranges from -1 to 1,
            where 1 indicates perfect similarity and -1 indicates perfect dissimilarity. Shape is (B,)
        """
        # Set invalid pixels in input and target to zero
        if mask is None:
            mask = torch.zeros(input.shape, dtype=input.dtype, device=input.device)

        masked_input = input.clone()
        masked_target = target.clone()

        masked_input.masked_fill_(mask.bool(), 0)
        masked_target.masked_fill_(mask.bool(), 0)

        # Compute SSIM full image
        _, ssim_idx_full_image = self.ssim(masked_input, masked_target)

        # Set invalid pixels to zero in full image output; this does not
        #  impact the mean_ssim, because masked pixels are excluded from
        #  the mean. But, it will make the visualiztion more interpretable.
        ssim_idx_full_image = ssim_idx_full_image * (1. - mask)
        
        # Remove the padded pixels from the SSIM full image.
        if self.ssim.gaussian_kernel:
            if isinstance(self.ssim.sigma, Sequence):
                raise NotImplementedError('Anisotropic kernels are not yet supported for MaskedSSIM')
            gauss_kernel_size = int(3.5 * self.ssim.sigma + 0.5) * 2 + 1
            pad_h = (gauss_kernel_size - 1) // 2
            pad_w = pad_h
        else:
            raise NotImplementedError('MaskedSSIM has not been tested with uniform kernels')
        mask = mask[...,pad_h:-pad_h, pad_w:-pad_w]
        ssim_idx_full_image = ssim_idx_full_image[...,pad_h:-pad_h, pad_w:-pad_w]
        
        mask_MaskedLoss = mask
        loss = ssim_idx_full_image

        ## Calculate the average SSIM over all valid pixels (code copied from MaskedLoss)
        all_dims_but_first = tuple(range(1, len(loss.shape)))
        # Cumulative loss over all filled, unmasked pixels.
        loss = (loss * (1.-mask_MaskedLoss)).sum(dim=all_dims_but_first) # dims: (batch_size)
        # Total number of valid, unmasked pixels
        num_valid_pixels = (1.-mask_MaskedLoss).sum(dim=all_dims_but_first)
        # Calculate average loss per valid pixel
        num_valid_pixels[num_valid_pixels==0] = 1. # set zeros to one to avoid division by zero 
        # In case that all pixels are masked the loss is zero and we divide zero by one.
        loss_pixelwise = loss / num_valid_pixels # dims: (batch_size)
        mean_ssim = loss_pixelwise
        
        plot = False
        if plot:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors

            fig, axs = plt.subplots(1, 1, dpi=300)
            cmaps = ['coolwarm'] # init colormaps
            bounds = np.array([-1., -0.9, -0.1, 0.1, 0.9, 1.])
            cnorms = [colors.BoundaryNorm(boundaries=bounds, ncolors=256)]
            ax = axs.imshow(ssim_idx_full_image[4,0].cpu().numpy(), cmap=cmaps[0], norm=cnorms[0])
            cbar = plt.colorbar(ax, orientation='horizontal', fraction=0.05, pad=0.01 , spacing='proportional') # ticks=ticks[i]
            plt.tight_layout()
            plt.savefig('references/figures/benchmark/plot_ssim/delete_image.png')
            plt.close()

        # Clean memory in ssim object to avoid memory leak.
        self.ssim.reset()
            
        return mean_ssim

class MaskedR2(torch.nn.Module):
    def __init__(self, reduction: str = 'none'):
        """
        Computes R2 with the help of 
        https://pytorch.org/torcheval/main/generated/torcheval.metrics.functional.r2_score.html#torcheval.metrics.functional.r2_score

        Args:
            reduction: loss reduction to apply to the batch of loss values either 'none', 'mean', 'sum'.
        """
        super(MaskedR2, self).__init__()
        
        self.reduction = reduction
    
    def forward(self, input, target, mask):
        """
        Input:
            input torch.Tensor(batch_size, out_channels, height, width): Model prediction
            target torch.Tensor(batch_size, out_channels, height, width): Ground-truth target
            mask torch.Tensor(batch_size, out_channels, height, width, dtype=float32):
                with 1. for invalid values and 0. for valid values
        Returns:
            loss: torch.Tensor(batch_size)
        """
        from torcheval.metrics.functional import r2_score
        # Compute r2 score for each tile. 
        #  Todo: parallelize across batch, but difficult because masked_select does not 
        #  feature keep_dims parameter and even if I could still not stack tiles of different flattened sizes.
        batch_size = len(input)
        r2_scores = torch.zeros(batch_size)
        for tile_idx in range(batch_size):
            # Select all pixels in image that are not masked.
            target_masked = torch.masked_select(target[tile_idx,...], ~mask[tile_idx,...].bool())
            input_masked = torch.masked_select(input[tile_idx,...], ~mask[tile_idx,...].bool())
            
            # If all pixels are masked, set r2 score to 0.
            if len(target_masked) == 0:
                r2_scores[tile_idx] = torch.tensor(0, device=input.device)
                continue
            else:
                # Compute R2
                r2_scores[tile_idx] = r2_score(input_masked, target_masked)

        # Clip the r2 score to [-1,1], to avoid large outliers that skew the mean.
        r2_scores = torch.clamp(r2_scores, min=-1.)
        return r2_scores

class NCorrectPreds(torch.nn.Module):
    def __init__(self, reduction: str = 'none',
                 threshold: float = 0.1) -> None:
        """
        Returns a binary image that contains 1. if predictions and targets match,
         0. otherwise. In combination with MaskedLoss, this can be used to calculate 
         accuracy on masked images.

        Args:
            reduction: Not implemented.
            threshold: threshold to binarize the target and predictions
        """
        super().__init__()
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.reduction == 'none'

        # Binarize the predictions and targets
        preds_binary = torch.where(input > self.threshold, 1., 0.)
        target_binary = torch.where(target > self.threshold, 1., 0.)
        # Create binary image with 1. for correct predictions
        correct_preds = torch.where(preds_binary == target_binary, 1., 0.) 

        return correct_preds

class MaskedPrecision(torch.nn.Module):
    def __init__(self, reduction: str = 'none', threshold: float = 0.1):
        """
        Computes precision on a batch of single-channel images with a loss mask.

        Args:
            reduction: loss reduction to apply to the batch of precision values: 'none', 'mean', or 'sum'.
            threshold: threshold to binarize the predictions and targets.
        """
        super(MaskedPrecision, self).__init__()
        
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, input, target, mask):
        """
        Input:
            input torch.Tensor(batch_size, out_channels, height, width): Model prediction
            target torch.Tensor(batch_size, out_channels, height, width): Ground-truth target
            mask torch.Tensor(batch_size, out_channels, height, width, dtype=float32):
                with 1. for invalid values and 0. for valid values
        Returns:
            precision: torch.Tensor(batch_size) or scalar depending on reduction
        """
        all_dims_but_first = tuple(range(1, len(input.shape)))

        # Binarize the predictions and targets
        predicted_positives = torch.where(input > self.threshold, 1., 0.)
        target_positives = torch.where(target > self.threshold, 1., 0.)

        # Count the number of true and predicted positives over all valid pixels
        true_positives = (predicted_positives * target_positives)
        num_true_positives = (true_positives * (1. - mask)).sum(dim=all_dims_but_first)
        num_predicted_positives = (predicted_positives * (1. - mask)).sum(dim=all_dims_but_first) # same as true + false positives.

        # Compute precision accounting for division by zero
        precision = torch.where(
            num_predicted_positives > 0,
            num_true_positives / num_predicted_positives,
            torch.tensor(1.0, device=input.device)
        )

        # Apply reduction if specified
        if self.reduction == 'mean':
            precision = precision.mean()
        elif self.reduction == 'sum':
            precision = precision.sum()

        return precision

class MaskedRecall(torch.nn.Module):
    def __init__(self, reduction: str = 'none', threshold: float = 0.1):
        """
        Computes recall on a batch of single-channel images with a loss mask.

        Args:
            reduction: loss reduction to apply to the batch of recall values: 'none', 'mean', or 'sum'.
            threshold: threshold to binarize the predictions and targets.
        """
        super(MaskedRecall, self).__init__()
        
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, input, target, mask):
        """
        Input:
            input torch.Tensor(batch_size, out_channels, height, width): Model prediction
            target torch.Tensor(batch_size, out_channels, height, width): Ground-truth target
            mask torch.Tensor(batch_size, out_channels, height, width, dtype=float32):
                with 1. for invalid values and 0. for valid values
        Returns:
            recall: torch.Tensor(batch_size) or scalar depending on reduction
        """
        all_dims_but_first = tuple(range(1, len(input.shape)))

        # Binarize the predictions and targets
        predicted_positives = torch.where(input > self.threshold, 1., 0.)
        target_positives = torch.where(target > self.threshold, 1., 0.)

        # Count the number of true and predicted positives over all valid pixels
        true_positives = (predicted_positives * target_positives)
        num_true_positives = (true_positives * (1. - mask)).sum(dim=all_dims_but_first)
        num_target_positives = (target_positives * (1. - mask)).sum(dim=all_dims_but_first) # same as true positive + false negative

        # Compute recall accounting for division by zero
        recall = torch.where(
            num_target_positives > 0,
            num_true_positives / num_target_positives,
            torch.tensor(1.0, device=input.device)
        )

        # Apply reduction if specified
        if self.reduction == 'mean':
            recall = recall.mean()
        elif self.reduction == 'sum':
            recall = recall.sum()

        return recall
    

class CountValidPx(torch.nn.Module):
    def __init__(self, reduction: str = 'none'):
        """
        Returns the number of valid pixels 
        Args:
            reduction: loss reduction to apply to the batch of loss values either 'none', 'mean', 'sum'.
        """
        super(CountValidPx, self).__init__()
        self.reduction = reduction
    
    def forward(self, mask, input=None, target=None):
        """
        Counts the number of valid pixels, i.e., zero values in the mask. Input and target are arguments for compatibility with other metrics,but not used.
        Args:
            mask torch.Tensor(batch_size, out_channels, height, width, dtype=float32):
                with 1. for invalid values and 0. for valid values
        Returns:
            n_valid_pixels: torch.Tensor(batch_size)
        """
        n_valid_pixels = torch.count_nonzero(~mask.bool(), dim=tuple(np.arange(1,len(mask.shape))))
        
        return n_valid_pixels

"""
Shapes of pred and true are expected to be (height, width)
for all _np functions
"""
def r2_score_np(pred: np.ndarray, true: np.ndarray):
    r2 = r2_score(true.flatten(), pred.flatten()) 
    return r2

def MSE_np(pred: np.ndarray, true: np.ndarray, mask: np.ndarray):
    mse = np.mean((pred - true) ** 2)
    return mse

def RMSE_np(pred: np.ndarray, true: np.ndarray):
    return np.mean(np.sqrt(MSE_np(pred, true)))

def MAE_np(pred: np.ndarray, true: np.ndarray):
    return np.mean(np.abs(pred - true))
