### model.py

import math
import torch
import torch.nn as nn
from torch.multiprocessing import Pool

class FastModel(nn.Module):
    def __init__(self, args, num_param):
        super(FastModel, self).__init__()        
        self.start = args['start']
        self.end = args['end']
        self.resolution = args['resolution']
        self.total = int((self.end - self.start) / self.resolution)
        self.lstm_window_size = args["lstm_window_size"]
        self.CNNLSTMs = CNNLSTM(args, num_param, self.total - self.lstm_window_size).cuda()

    def forward(self, x, params):
        r = self.CNNLSTMs(x, params)
        return r
    
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
        self.lstm_hidden = int(args['lstm_hidden_size'])
        self.seq_len = int(args['lstm_window_size'] - 1)
        self.n_layers = int(args['lstm_num_layers'])
        self.batch_size = args['batch_size']
        self.PE = PE
        self.mode = mode

        self.cnn = nn.Conv1d(in_channels=1, out_channels=1, kernel_size = 2, stride = 1, dtype=torch.float64).cuda()
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.lstm_hidden, num_layers=self.n_layers, batch_first = True, dtype=torch.float64).cuda()
        self.decoder = nn.Linear(self.seq_len * self.n_layers * self.lstm_hidden, num_output, dtype=torch.float64).cuda()

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
                nn.Linear(num_param, int(self.lstm_hidden/4), dtype=torch.float64),
                nn.ReLU(),
                nn.Linear(int(self.lstm_hidden/4), int(self.lstm_hidden/4), dtype=torch.float64),
                nn.ReLU(),
                nn.Linear(int(self.lstm_hidden/4), int(self.lstm_hidden/2), dtype=torch.float64)
            ).cuda()
        else :
            self.hiddenUpdater = nn.Sequential(
                nn.Linear(num_param, int(self.lstm_hidden/2), dtype=torch.float64),
                nn.ReLU(),
                nn.Linear(int(self.lstm_hidden/2), int(self.lstm_hidden/2), dtype=torch.float64),
                nn.ReLU(),
                nn.Linear(int(self.lstm_hidden/2), int(self.lstm_hidden), dtype=torch.float64)
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
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4000):
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
    