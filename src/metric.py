### metric.py

import torch

# def MSE(output, target):
#     loss_fn = torch.nn.MSELoss()
#     return loss_fn(output, target)

# def R2Score(output, target):
#     mse = MSE(output, target)
#     var = torch.var(output)
#     return 1 - mse/var

# def MAPE(output, target):
#     return torch.mean(torch.abs((output - target)/(output+0.001)))

# def MAE(output, target):
#     loss_fn = torch.nn.L1Loss()
#     return loss_fn(output, target)

def R2Score(y_pred, y_true):
    """
    Calculate the R² score using PyTorch.
    """
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def MAE(y_pred, y_true):
    """
    Calculate Mean Absolute Error (MAE) using PyTorch.
    """
    return torch.mean(torch.abs(y_true - y_pred))

def MAPE(y_pred, y_true):
    """
    Calculate Mean Absolute Percentage Error (MAPE) using PyTorch.
    """
    # Avoid division by zero
    epsilon = 1e-10
    y_true = torch.clamp(y_true, min=epsilon)
    abs_percent_error = torch.abs((y_true - y_pred) / y_true)
    return torch.mean(abs_percent_error) * 100

def MSE(y_pred, y_true):
    loss_fn = torch.nn.MSELoss()
    return loss_fn(y_pred, y_true)