### metric.py

import torch

def MSE(output, target):
    loss_fn = torch.nn.MSELoss()
    return loss_fn(output, target)

def R2Score(output, target):
    mse = MSE(output, target)
    var = torch.var(output)
    return 1 - mse/var

def MAPE(output, target):
    return torch.mean(torch.abs((output - target)/(output+0.001)))

def MAE(output, target):
    loss_fn = torch.nn.L1Loss()
    return loss_fn(output, target)

