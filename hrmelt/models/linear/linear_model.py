import logging
import torch
import torch.nn as nn
from hrmelt.utils.utils import lookup_torch_activation

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, 
        out_activation=None,
        verbose=False,
        model_key='linear',
        n_models=1,
        time_channel_idx=None):
        """
        Args:
            out_activation str:
                Choose 'sigmoidcutoff' or 'tanhshelf' to get a model that finds a low and high cut-off
                Choose 'sigmoid' to fit a logistic regression model
            model_key str: Key that specifies the configuration of this linear model.
                'linear_dem': Fits one linear model per month. 
                'linear': Standard linear model
            n_models int: Number of separate linear regression models that are trained at 
             the same time. E.g., n_models = 12 if one linear regression model should be trained
             per month.
            time_channel_idx int: Channel index that corresponds to the time dimension. Used for 
             fitting one linear regression model per month. Expected format of time input is
             label encoding (1,2,3,4,...) for (Jan., Feb., Mar., Apr.,...).
        """
        super(Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
                
        self.time_channel_idx = time_channel_idx
        if model_key == 'linear_dem':
            assert n_models == 12, f'Expected n_models=12 for 12 months, but got {n_models}.'
            assert self.time_channel_idx is not None, 'Expected time_channel_idx when model_key==linear_dem'
            # Remove time channel from inputs:
            model_in_channels = in_channels - 1
        else:
            assert n_models == 1, f'Expected n_models=1 when model_key!=linear_dem'\
                f', but got {n_models}.'
            model_in_channels = in_channels
        
        # Initialize multiple independent linear layers to, e.g., fit a model that 
        #  desires a separate linear regression model per month.
        self.n_models = n_models
        linear_layer_list = [nn.Conv2d(model_in_channels, out_channels, kernel_size=1, padding=0, bias=True) for _ in range(n_models)]
        self.linear_layers = nn.ParameterList(linear_layer_list)
        self.out_activation = out_activation
        self.out_act = lookup_torch_activation(out_activation)

        logging.info(f'Linear model:\n'
            f'\t{self.in_channels} input channels\n'
            f'\t{self.out_channels} output channels\n')

    def get_in_channels(self):
        return self.in_channels

    def forward(self, x):
        '''
        Runs the Linear model forward
        Args:
            x torch.Tensor([batch_size, in_channels, img_size, img_size])
        Returns:
            preds torch.Tensor([batch_size, out_channels, img_size, img_size])
        '''
        if self.n_models > 1:
            logits = torch.empty((x.shape[0], self.out_channels)+ x.shape[2:], device=x.device, dtype=x.dtype)

            batch_size = x.shape[0]
            time_input = x[:,self.time_channel_idx,...]
            other_inputs = torch.cat([x[:,:self.time_channel_idx,...],x[:,self.time_channel_idx+1:,...]],dim=1)

            # Extract unique ID of each month in the input
            months = time_input.unique(dim=-1).squeeze(-1).unique(dim=-1).squeeze(-1).int() # dim: (batch_size)
            assert len(months.flatten()) == batch_size, 'Expected one month id per sample in batch. The config might '\
                f'contain the wrong time_channel_idx which is {self.time_channel_idx}.'

            for i in range(batch_size):
                # Feed each sample into the linear model of the corresponding month
                logits[i:i+1,...] = self.linear_layers[months[i]-1](other_inputs[i:i+1,...])

            preds = self.out_act(logits)
        else:
            logits = self.linear_layers[0](x)
            preds = self.out_act(logits)
        return preds