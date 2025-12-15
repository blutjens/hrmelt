import torch
from torch import nn
from hrmelt.utils.utils import apply_mask

class PmwThreshold(nn.Module):

    def __init__(
            self,
            apply_landmask=False,
            mask_threshold=0.5
    ):
        """
        Predict surface meltwater fraction as a function of
        PMW observations using a threshold based approach,
        based on https://doi.org/10.5194/tc-15-2623-2021 eq. 3

        Args:
            apply_landmask (bool): If true, sets all values in the landmask to zero
            mask_threshold (float): Threshold to use for applying the landmask
        """
        super(PmwThreshold, self).__init__()

        self.threshold = mask_threshold
        self.apply_landmask = apply_landmask

    def forward(self, x):
        """
        Runs the model
        Args:
            x torch.Tensor([batch_size, in_channels, height, width]):
                assumes that PMW is stored in 1st, landmask in 2nd,
                and pmw_winter_mean in last in_channel
        Returns:
            pred torch.Tensor([batch_size, 1, height, width]):
                prediction of surface meltwater in range [0.,1.]
        """
        # Grab channel that stores PMW 
        pmw = x[:,0:1,...]

        # Grab channel that stores landmask
        landmask = x[:, 1:2, ...]

        # Grab channel that stores PMW winter mean
        pmw_winter_mean = x[:,-1:,...]

        # Apply https://doi.org/10.5194/tc-15-2623-2021 eq. 3
        slope = 0.48 # linear regression slope; gamma in eq 3
        intercept = 128. # linear regression intercept in Kelvin; omega in eq 3
        pmw_threshold = slope * pmw_winter_mean + intercept

        # Predict surface meltwater (1.) for any pixel with brightness
        #  temperature above the threshold
        pred = torch.where(pmw > pmw_threshold, 1.0, 0.0)

        if self.apply_landmask:
            pred = apply_mask(pred, landmask, self.threshold)

        return pred
