### model.py

import torch
import torch.nn as nn
from torch.multiprocessing import Pool


class Model(nn.Module):
    def __init__(self, args, count, num_param, num_batch):
        super(Model, self).__init__()
        self.lstm_window_size = args["lstm_window_size"]
        self.batch_size = num_batch
        self.LSTMBlocks = nn.ModuleList([LSTMBlock(args, num_param, num_batch).cuda() for _ in range(count)])
        self.channel = int((args['end'] - args['start'])*args['resolution'] - args['lstm_window_size'] - 1) 

    def forward(self, x, params, mode = 'train'):
        result = torch.empty((self.batch_size, self.channel, 1, 1)).cuda()
        if mode == 'train':
            for index, block in enumerate(self.LSTMBlocks):
                r = block(x[:, :, :, index], params, mode = mode, inference_num = -1)
                result = torch.cat([result, r], dim = 3)
            return result[:,:,:,1:]
            
        else:
            for index, block in enumerate(self.LSTMBlocks):
                r = self.parallel_forward(index, block, x, params)
                result = torch.cat([result, r], dim = 3)
            return result[:,:,:,1:]
    
    def parallel_forward(self, index, block, x, params):
        r = torch.empty(self.batch_size, 1, 1).cuda()
        for i in range(x.shape[1]):
            in_x = x[:, :, :, index]
            if i != 0:
                if i < self.lstm_window_size:
                    in_x[:, i, -i:] = r[:, -i:, :].squeeze(2)
                else:
                    in_x[:, i, :] = r[:, -self.lstm_window_size:, :].squeeze(2)

            t = block(in_x, params, mode = 'inference', inference_num = i)
            r = torch.cat([r, t], dim = 1)
        return r[:, 1:, :].unsqueeze(-1)


class LSTMBlock(nn.Module):
    def __init__(self, args, num_param, num_batch):
        super(LSTMBlock, self).__init__()
        self.lstm_hidden = int(args['lstm_hidden_size'])
        self.seq_len = int(args['lstm_window_size'] - 1)
        self.batch_size = num_batch
        self.n_layers = int(args['lstm_num_layers'])
        self.start = args['start']
        self.end = args['end']
        self.resolution = args['resolution']
        self.channel = int((args['end'] - args['start'])*args['resolution'] - args['lstm_window_size'] - 1) 

        self.cnn = nn.Conv1d(in_channels=self.channel, out_channels=self.channel, kernel_size = 2, stride = 1, dtype=torch.float64).cuda()
        self.lstm = nn.LSTM(input_size=self.seq_len, hidden_size=self.lstm_hidden, num_layers=self.n_layers, batch_first = True, dtype=torch.float64).cuda()
        self.decoder = nn.Linear(self.lstm_hidden, 1, dtype=torch.float64).cuda()
        
        self.hiddenUpdater = nn.Sequential(
            nn.Linear(num_param, int(self.lstm_hidden/2), dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(int(self.lstm_hidden/2), self.lstm_hidden, dtype=torch.float64)
        ).cuda()

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.batch_size, self.lstm_hidden, dtype=torch.float64).cuda(),
            torch.zeros(self.n_layers, self.batch_size, self.lstm_hidden, dtype=torch.float64).cuda()
        )
    
    def set_hidden_state(self, params):
        hidden = self.hiddenUpdater(params)
        hidden = hidden.reshape([self.batch_size, -1])
        hidden = hidden.expand(self.n_layers, self.batch_size, self.lstm_hidden).contiguous()
        self.hidden = (hidden, hidden)

    # Batch, total, window, drg/drs/diode current
    def forward(self, x, params, mode = 'train', inference_num = -1):
        self.reset_hidden_state()
        self.set_hidden_state(params)
        x = self.cnn(x)
        self.lstm.flatten_parameters()
        lstm_out, self.hidden = self.lstm(
            x.reshape(self.batch_size, -1, self.seq_len),
            self.hidden
        )        
        result = self.decoder(lstm_out).unsqueeze(-1)
        if mode == 'inference':   
            return result[:, inference_num, :, :]
        else:
            return result
