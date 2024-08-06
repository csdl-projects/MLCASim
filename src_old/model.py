### model.py

import math
import torch
import torch.nn as nn
from torch.multiprocessing import Pool

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
        # print(x.shape)
        x = x.unsqueeze(-1)
        x = feature_to_data(x)
        x = self.fc1(x)
        x = data_to_feature(x)
        x, _ = self.feature_mixing(x, x_param)
        # print("Feature Mixing", x.shape)
        for mixer in self.conditional_mixer:
            x = mixer(x, x_param)

        # print("FC Out", x.shape)        
        x = self.fc_out(x)
        return x
        

class FastModel(nn.Module):
    def __init__(self, args, num_param, num_output, PE=0, mode=0):
        super(FastModel, self).__init__()
        self.lstm_hidden = int(args['hidden_size'])


        self.encoder = nn.Sequential(
            nn.Linear(num_param, int(self.lstm_hidden/2), dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(int(self.lstm_hidden/2), int(self.lstm_hidden/2), dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(int(self.lstm_hidden/2), int(self.lstm_hidden), dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(int(self.lstm_hidden), num_output, dtype=torch.float32)
        ).cuda()
        
    def forward(self, x, params):
        r = self.encoder(params).squeeze()
        return r

class FastCNNLSTM(nn.Module):
    def __init__(self, args, num_param, num_output, PE=0, mode = 0):
        super(FastCNNLSTM, self).__init__()
        self.lstm_hidden = int(args['hidden_size'])
        # self.seq_len = int(args['window_size'] - 1)
        self.seq_len = int(args['window_size'])
        self.n_layers = int(args['lstm_num_layers'])
        self.batch_size = args['batch_size']
        self.PE = PE
        self.mode = mode

        # self.cnn = nn.Conv1d(in_channels=1, out_channels=1, kernel_size = 2, stride = 1, dtype=torch.float32).cuda()
        self.lstm = nn.LSTM(input_size=2, hidden_size=self.lstm_hidden, num_layers=self.n_layers, batch_first = True, dtype=torch.float32).cuda()
        # self.encoder = nn.Linear(num_param, self.seq_len, dtype=torch.float32).cuda()
        self.encoder = nn.Sequential(
            nn.Linear(num_param, int(self.seq_len/2), dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(int(self.seq_len/2), int(self.seq_len/2), dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(int(self.seq_len/2), int(self.seq_len), dtype=torch.float32)
        ).cuda()
        self.decoder = nn.Linear(self.seq_len * self.n_layers * self.lstm_hidden, num_output, dtype=torch.float32).cuda()

    def set_initial_hidden_state(self):
        hidden = torch.zeros([self.n_layers, self.batch_size, self.lstm_hidden]).cuda()
        self.hidden = (hidden, hidden)

    def forward(self, x, params):
        self.set_initial_hidden_state()
        # print(x.shape, params.shape)
        # x = self.cnn(x.unsqueeze(1))
        self.lstm.flatten_parameters()
        px = self.encoder(params)
        # print(px.shape, x.shape)
        x = torch.cat([x, px], dim=1).unsqueeze(1)
        lstm_out, self.hidden = self.lstm(
            x.reshape(self.batch_size, self.seq_len, 2),
            self.hidden
        )
        # print(lstm_out.shape)
        lstm_out = lstm_out.reshape(self.batch_size, -1)
        result = self.decoder(lstm_out).squeeze()
        return result
    
# params : i, num_scan_pixels, j, num_data_pixels, float(res), cap, tw_s, VDH
class Model(nn.Module):
    def __init__(self, args, num_param, PE=0, mode=0):
        super(Model, self).__init__()
        self.CNNLSTMs = CNNLSTM(args, num_param, 1, PE, mode).cuda()
        
    def forward(self, x, params):
        r = self.CNNLSTMs(x, params)
        return r
            
class CNNLSTM(nn.Module):
    def __init__(self, args, num_param, num_output, PE=0, mode = 0):
        super(CNNLSTM, self).__init__()
        self.lstm_hidden = int(args['hidden_size'])
        self.seq_len = int(args['window_size'] - 1)
        self.n_layers = int(args['lstm_num_layers'])
        self.batch_size = args['batch_size']
        self.PE = PE
        self.mode = mode

        self.cnn = nn.Conv1d(in_channels=1, out_channels=1, kernel_size = 2, stride = 1, dtype=torch.float32).cuda()
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.lstm_hidden, num_layers=self.n_layers, batch_first = True, dtype=torch.float32).cuda()
        self.decoder = nn.Linear(self.seq_len * self.n_layers * self.lstm_hidden, num_output, dtype=torch.float32).cuda()

        if PE == 1:
            if mode == 0:
                self.PositionalEncoding = PositionalEncoding1D(int(self.lstm_hidden), 0.1)

            elif mode == 1:
                self.PositionalEncoding = PositionalEncoding1D(int(self.lstm_hidden/2), 0.1)
            
        elif PE == 2:
            if mode == 0:
                self.PositionalEncoding = PositionalEncoding1D(int(self.lstm_hidden/2), 0.1)
            elif mode == 1:
                self.PositionalEncoding = PositionalEncoding1D(int(self.lstm_hidden/4), 0.1)

        if PE != 0 and mode == 1:
            self.hiddenUpdater = nn.Sequential(
                nn.Linear(num_param, int(self.lstm_hidden/4), dtype=torch.float32),
                nn.ReLU(),
                nn.Linear(int(self.lstm_hidden/4), int(self.lstm_hidden/4), dtype=torch.float32),
                nn.ReLU(),
                nn.Linear(int(self.lstm_hidden/4), int(self.lstm_hidden/2), dtype=torch.float32)
            ).cuda()
        else :
            self.hiddenUpdater = nn.Sequential(
                nn.Linear(num_param, int(self.lstm_hidden/2), dtype=torch.float32),
                nn.ReLU(),
                nn.Linear(int(self.lstm_hidden/2), int(self.lstm_hidden/2), dtype=torch.float32),
                nn.ReLU(),
                nn.Linear(int(self.lstm_hidden/2), int(self.lstm_hidden), dtype=torch.float32)
            ).cuda()

    def set_initial_hidden_state(self, params):
        hidden = self.hiddenUpdater(params)
        if self.PE == 1:
            if self.mode == 0:
                hidden += self.PositionalEncoding(params[:, -1])
            elif self.mode == 1:
                hidden = torch.cat([hidden, self.PositionalEncoding(params[:, -1])], dim=1)

        elif self.PE == 2:
            if self.mode == 0:
                hidden += torch.cat([self.PositionalEncoding(params[:, 0]), self.PositionalEncoding(params[:, 2])], dim=1)
            
            elif self.mode == 1:
                hidden = torch.cat([hidden, self.PositionalEncoding(params[:, 0]), self.PositionalEncoding(params[:, 2])], dim=1)

        hidden = hidden.reshape([self.batch_size, self.lstm_hidden])
        hidden = hidden.expand(self.n_layers, self.batch_size, self.lstm_hidden).contiguous()
        self.hidden = (hidden, hidden)

    def forward(self, x, params):
        self.set_initial_hidden_state(params)
        x = self.cnn(x.unsqueeze(1))
        self.lstm.flatten_parameters()
        lstm_out, self.hidden = self.lstm(
            x.reshape(self.batch_size, self.seq_len, 1),
            self.hidden
        )
        lstm_out = lstm_out.reshape(self.batch_size, -1)
        result = self.decoder(lstm_out).squeeze()
        return result
    
# params : i, num_scan_pixels, j, num_data_pixels, float(res), cap, tw_s, VDH, load_ratio
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout).cuda()
        self.encoding = torch.zeros(max_len, d_model).cuda()
        self.encoding.requires_grad = False
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.int64)
        x = self.encoding[x]

        return self.dropout(x)
    

