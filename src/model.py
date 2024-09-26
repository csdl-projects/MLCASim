##################################################################
## 1. CircuitMixer : Mixer based prediction model               ##
## 2. RidgeModel : Linear Regression based prediction model     ##
## 3. IHE_CLSTM : CNN-LSTM based prediction model               ## 
##################################################################
## Author: Jaeseung Lee                                         ##
## Date: July 2024                                              ##
## Affiliation: POSTECH CSDL, South Korea                       ##
##################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import (
    ConditionalMixerLayer,
    ConditionalFeatureMixing,
    CircuitNorm2d,
    feature_to_data,
    data_to_feature,
)

# params : i, num_scan_pixels, j, num_data_pixels, float(res), cap, tw_s, VDH
class CircuitMixer(nn.Module):
    def __init__(
        self,
        data_length : int,
        pred_length : int,
        activation : str = "gelu",
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        input_channel: int = 1,
        hidden_channel: int = 10,
        param_length: int = 10,
        hidden_size: int = 256,
        output_channel: int = None,
        is_norm_before: bool = False,
        norm_type: str = "layer",
    ):
        super().__init__()

        if hasattr(F, activation):
            activation = getattr(F, activation)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type = CircuitNorm2d if norm_type == "batch" else nn.LayerNorm

        self.fc1 = nn.Linear(data_length, pred_length)
        self.fc_out = nn.Linear(hidden_channel, output_channel or input_channel)

        self.feature_mixing = ConditionalFeatureMixing(
            data_length=pred_length,
            input_channel=input_channel,
            output_channel=hidden_channel,
            param_length=param_length,
            hidden_size=hidden_size,
            activation=activation,
            dropout_rate=dropout_rate,
            is_norm_before=is_norm_before,
            norm_type=norm_type,
        )
        
        self.conditional_mixer = self._build_mixer(
            num_blocks,
            hidden_channel,
            pred_length,
            hidden_size = hidden_size,
            param_length = param_length,
            activation = activation,
            dropout_rate = dropout_rate,
            is_norm_before = is_norm_before,
            norm_type = norm_type,
        )

    @staticmethod
    def _build_mixer(
        num_blocks: int,
        hidden_channel: int,
        pred_length: int,
        **kwargs,
    ):
        channels = (num_blocks) * [hidden_channel]
        return nn.ModuleList([
            ConditionalMixerLayer(
                data_length = pred_length,
                input_channel = in_channel,
                output_channel = out_channel,
                **kwargs,
            )
            for in_channel, out_channel in zip(channels[:-1], channels[1:])
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        x_param: torch.Tensor,
    ) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = feature_to_data(x)
        x = self.fc1(x)
        x = data_to_feature(x)
        x, _ = self.feature_mixing(x, x_param)
        for mixer in self.conditional_mixer:
            x = mixer(x, x_param)

        x = self.fc_out(x)
        return x
        
class RidgeModel(nn.Module):
    def __init__(
        self, 
        hidden_size : int,
        num_param : int,
        num_output :int,
    ):
        super(RidgeModel, self).__init__()
        self.lstm_hidden = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(num_param, int(self.lstm_hidden), dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(int(self.lstm_hidden), num_output, dtype=torch.float32)
        ).cuda()
        
    def forward(self, x, params):
        r = self.encoder(params).squeeze()
        return r

class IHE_CLSTM(nn.Module):
    def __init__(
        self,
        hidden_size : int,
        window_size : int,
        lstm_num_layers : int,
        batch_size : int,
        num_param : int,
        num_output : int
    ):
        super(IHE_CLSTM, self).__init__()
        self.lstm_hidden = hidden_size
        self.seq_len = window_size
        self.n_layers = lstm_num_layers
        self.batch_size = batch_size

        self.cnn = nn.Conv1d(in_channels=1, out_channels=1, kernel_size = 2, stride = 1, dtype=torch.float32)
        self.lstm = nn.LSTM(input_size=2, hidden_size=self.lstm_hidden, num_layers=self.n_layers, batch_first = True, dtype=torch.float32)
        self.encoder = nn.Linear(num_param, int(self.seq_len-1), dtype=torch.float32)
        self.decoder = nn.Linear((self.seq_len-1) * self.lstm_hidden, num_output, dtype=torch.float32)

    def set_initial_hidden_state(self):
        hidden = torch.zeros([self.n_layers, self.batch_size, self.lstm_hidden]).cuda()
        self.hidden = (hidden, hidden)

    def forward(self, x, params):
        self.set_initial_hidden_state()
        x = self.cnn(x.unsqueeze(1))
        self.lstm.flatten_parameters()
        px = self.encoder(params).unsqueeze(1)
        # print(x.shape, px.shape)
        x = torch.cat([x, px], dim=1)
        lstm_out, self.hidden = self.lstm(
            x.reshape(self.batch_size, self.seq_len-1, -1),
            self.hidden
        )
        # print(lstm_out.shape)
        lstm_out = lstm_out.reshape(self.batch_size, -1)
        # print(lstm_out.shape)
        result = self.decoder(lstm_out).squeeze()
        return result
    