""" Full assembly of the parts to form the complete network 
Src: https://github.com/milesial/Pytorch-UNet/
"""
import logging
from hrmelt.models.unet.unet_parts import *

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, 
        bilinear=False, 
        activation='relu',
        num_extra_convs=0,
        out_activation=None,
        verbose=False):

        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.activation = activation
        self.num_extra_convs = num_extra_convs

        # todo: add MaskAttentionLayer
        # self.maskAttention = MaskAttentionLayer(?)
        self.inc = DoubleConv(in_channels, 64, activation=activation)
        self.extraincs = nn.ModuleList(num_extra_convs * [ResConv(64, 64, activation=activation)])
        self.down1 = Down(64, 128, activation=activation)
        self.down2 = Down(128, 256, activation=activation)
        self.down3 = Down(256, 512, activation=activation)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, activation=activation)
        self.up1 = Up(1024, 512 // factor, bilinear, activation=activation)
        self.up2 = Up(512, 256 // factor, bilinear, activation=activation)
        self.up3 = Up(256, 128 // factor, bilinear, activation=activation)
        self.up4 = Up(128, 64, bilinear, activation=activation)
        self.extraoutcs = nn.ModuleList(num_extra_convs * [ResConv(64, 64, activation=activation)])
        self.outc = OutConv(64, out_channels, activation=out_activation)

        logging.info(f'Network:\n'
            f'\t{self.in_channels} input channels\n'
            f'\t{self.out_channels} output channels\n')

    def get_in_channels(self):
        return self.in_channels

    def forward(self, x):
        '''
        Runs the UNet model forward
        Args:
            x torch.Tensor([batch_size, in_channels, img_size, img_size])
        '''
        # x, mask = x
        # x = self.maskAttention(x, mask)
        x1 = self.inc(x) # [batch_size, 64, img_size, img_size]
        for extrainc in self.extraincs: # 
            x1 = extrainc(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        for extraoutc in self.extraoutcs:
            x = extraoutc(x)
        logits = self.outc(x)
        return logits
