### transform.py

import torch

## Parse the lis file and save the raw data and labels



def Z_Normalization(input, mean, std):
    return (input-mean)/std

def MM_Normalization(input, _min, _max):
    return (input - _min)/(_max - _min)
