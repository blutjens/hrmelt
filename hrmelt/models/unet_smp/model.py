from segmentation_models_pytorch import Unet

class UNet_smp(Unet):
    def __init__(self, out_channels=1, out_activation=None, **kwargs):
        """
        Thin wrapper around segmentation_models_pytorch.Unet to 
            rename attributes.
        """
        super().__init__(
            activation=out_activation,
            classes=out_channels,
            **kwargs
        )
    def get_in_channels(self):
        return self.encoder._in_channels