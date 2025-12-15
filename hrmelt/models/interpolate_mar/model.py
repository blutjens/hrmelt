from torch import nn
from torchvision import transforms as T
from torchvision.transforms import functional as F

from hrmelt.utils.utils import apply_mask

def get_blur_func(blur_type, kernel_size, sigma):

    def gaus(x):
        return T.GaussianBlur(kernel_size, sigma)(x)

    if blur_type == 'gaussian':
        return gaus
    else:
        raise NotImplementedError(f'Attempting to apply blur {blur_type}, but when using torch instead of cv2 we only have gaussian blur implemented')

class MarInterpolation(nn.Module):

    def __init__(
            self,
            blur_type=None,
            kernel_size=1,
            sigma=1,
            apply_landmask=False,
            use_cv=False,
            brightness_factor=1.,
            gamma=1.,
            mask_threshold=0.5
    ):
        """
        Creates meltwater predictions by blurring the MAR WA1 data.
         Also modifies gamma and brightness to match scale of meltwater.

        Args:
            blur_type (str): Type of blur to use. e.g., 'gaussian', 'biliteral', 'average', 'median'
            kernel_size (int): Size of the kernel to use for blurring or filtering
            sigma (int): Standard deviation of the kernel, only used in gaussian blur
            apply_landmask (bool): If true, sets all values in the landmask to zero after blurring.
            use_cv (bool): If true, uses opencv2 for blurring. Otherwise uses torch
            brightness_factor (float): Brightness factor to use for brightness adjustment
            gamma (float): Gamma to use for gamma adjustment
            mask_threshold (float): Threshold to use for applying the landmask
        """
        super(MarInterpolation, self).__init__()

        self.threshold = mask_threshold
        self.apply_landmask = apply_landmask
        self.brightness_factor = brightness_factor
        self.gamma = gamma
        self.use_cv = use_cv
        if self.use_cv:
            from hrmelt.models.interpolate_mar.cv_blur_functions import get_blur_func_cv
            self.blur = get_blur_func_cv(blur_type, kernel_size, sigma)
        else:
            self.blur = get_blur_func(blur_type, kernel_size,  sigma)

    def forward(self, x):
        """
        Runs the Interpolation model forward
        Args:
            x torch.Tensor([batch_size, in_channels, height, width]):
                assumes that MAR is stored in first and landmask in second in_channel
        """
        # Grab channel that stores landmask
        landmask = x[:, 1, ...]

        # Grab channel that stores MAR 
        x = x[:,0,...]

        if self.use_cv:
            # Remove batch dimension for openCV
            x = x.unsqueeze(0)
            landmask = landmask.unsqueeze(0)
        else:
            # Add channel dimension back in for torch
            x = x[:,None,...]
            landmask = landmask[:,None,...]

        # Blur image
        x = self.blur(x)
        if self.apply_landmask:
            x = apply_mask(x, landmask, self.threshold)
        # Here we adjust brightness and gamma to match the scale of meltwater. One could
        #  also try sigmoid activation, but brightness and gamma worked really well.
        x = F.adjust_brightness(x, self.brightness_factor)
        # Adjust gamma, e.g., raise brightness of dark regions and decrease brightness in light regions
        x = F.adjust_gamma(x, self.gamma)
        return x