##################################################################
## metric.py                                                    ##
## This file is used to define the evaluation metrics.          ##
##################################################################
## Author: Jaeseung Lee                                         ##
## Date: September 2024                                         ##
## Affiliation: POSTECH CSDL, South Korea                       ##
##################################################################
import torch


## Calculate the R² score using PyTorch
def R2Score(y_pred, y_true):
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

## Calculate Mean Absolute Error (MAE) using PyTorch
def MAE(y_pred, y_true):
    return torch.mean(torch.abs(y_true - y_pred))

## Calculate Mean Absolute Percentage Error (MAPE) using PyTorch.
# Avoid division by zero
def MAPE(y_pred, y_true):
    epsilon = 1e-10
    y_true = torch.clamp(y_true, min=epsilon)
    abs_percent_error = torch.abs((y_true - y_pred) / y_true)
    return torch.mean(abs_percent_error) * 100

## Calculate Mean Squared Error (MSE) using PyTorch
def MSE(y_pred, y_true):
    loss_fn = torch.nn.MSELoss()
    return loss_fn(y_pred, y_true)