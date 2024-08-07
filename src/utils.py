##################################################################
## 1. CreateDirectory : Create a directory if it does not exist ##
## 2. ExecuteCommand : Execute a command                        ##
## 3. _float :  Convert a string to a float                     ##
## 4. plot : Plot a graph                                       ##
## 5. dual_plot : Plot two data into graphs                     ##
##################################################################
## Author: Jaeseung Lee                                         ##
## Date: July 2024                                              ##
## Affiliation: POSTECH CSDL, South Korea                       ##
##################################################################

import os
import subprocess as sp
import decimal
import errno
import numpy as np
import matplotlib.pyplot as plt
import torch


def CreateDirectory( dirName ):
    try:
        if not(os.path.isdir(dirName)):
            os.makedirs(os.path.join(dirName))
            # print( dirName )
    except OSError as e:
        if e.errno != errno.EEXIST:
            print( "Failed to create directory." )
            print( "  Directory: %s" % (dirName) )
            raise

def ExecuteCommand( curCmd ):
    print( curCmd )
    sp.call( curCmd, shell=True )

def _float(f):
    Prefixes = {'k':'1E3', 'M':'1E6', 'G':'1E9',
                'T':'1E12', 'P':'1E15', 'E':'1E18',
                'Z':'1E21' , 'Y':'1E24',
                'm':'1E-3', 'u':'1E-6', 'n':'1E-9',
                'p':'1E-12', 'f':'1E-15', 'a':'1E-18',
                'z':'1E-21', 'y':'1E-24'}

    if f[-1] in Prefixes:
        #v = decimal.Decimal(f[:-1])
        v = float(f[:-1])
        return v*float(Prefixes[f[-1]])
        #return v*decimal.Decimal(Prefixes[f[-1]])
    else:
        #return decimal.Decimal(f)
        return float(f)

def plot(y, path, name, color):
    plt.clf()
    x = range(len(y))
    png_file = os.path.join(path, f'{name}.png')
    plt.ylim(0,1)
    plt.plot(x, y, color, label=name)    
    plt.legend()
    plt.savefig(png_file)

def dual_plot(y1, y2, png_path, name, legend1, legend2):
    plt.clf()
    x = range(len(y1))
    os.makedirs(png_path, exist_ok =True)
    png_file = os.path.join(png_path, f'{name}.png')
    plt.plot(x, y1, 'b', label=legend1)
    plt.plot(x, y2, 'g', label=legend2)
    plt.legend()
    plt.gcf().set_dpi(400)
    plt.savefig(png_file)

def condition_function(label_name, query):
    return query in label_name

def getCircuitName(circuitName, config):
    number_scanline_pixel, step_x, number_dataline_pixel, step_y, res, cap, tw_s, VDH, load_ratio = config
    name = f'{circuitName}_{number_scanline_pixel}_{step_x}_{number_dataline_pixel}_{step_y}_{res}_{cap}_{tw_s}_{VDH:.1f}_{load_ratio}'
    return name

def getModelName(base_name, window_size, lstm_num_layers, hidden_size, hidden_channel, type_index):
    return f'{base_name}_{window_size}_{lstm_num_layers}_{hidden_size}_{hidden_channel}_{type_index}'

def load_maxmin(min_max_dir):
    max_file, min_file = os.path.join(min_max_dir, f'max.np'), os.path.join(min_max_dir, f'min.np')
    v_max = np.array([-1e10, -1e10, -1e10, -1e10, -1e10])
    v_min = np.array([1e10, 1e10, 1e10, 1e10, 1e10])
    if os.path.exists(max_file) and os.path.exists(min_file):
        v_max = np.loadtxt(os.path.join(min_max_dir, f'max.np'), dtype=float)
        v_min = np.loadtxt(os.path.join(min_max_dir, f'min.np'), dtype=float)

    return v_max, v_min

def convert_param_to_tensor(index_scan, number_scanline_pixel, index_data, number_dataline_pixel, res, cap, tw_s, VDH, load_ratio):
    return torch.tensor([index_scan/2000.0, number_scanline_pixel/2000.0, index_data/2000.0, number_dataline_pixel/2000.0, float(res)/8, cap, float(tw_s)/2.0, float(VDH)/10.0, float(load_ratio)/10.0])

def convert_tensor_to_param(param):
    index_scan, number_scanline_pixel, index_data, number_dataline_pixel, res, cap, tw_s, VDH, load_ratio, _ = tuple(param)
    return np.array([index_scan*2000.0, number_scanline_pixel*2000.0, index_data*2000.0, number_dataline_pixel*2000.0, float(res)*8, cap, float(tw_s)*2.0, float(VDH)*10.0, float(load_ratio)*10.0])

def convert_param_to_name(param):
    print(param)
    index_scan, number_scanline_pixel, index_data, number_dataline_pixel, res, cap, tw_s, VDH, load_ratio = tuple(param)
    return f'{index_scan}_{number_scanline_pixel}_{index_data}_{number_dataline_pixel}_{res}_{cap}_{tw_s}_{VDH}_{load_ratio}'

def linear_normalization(x, max, min):
    return (x - min) / (max - min)

def linear_denormalization(x, max, min):
    return x * (max - min) + min

def convert_wandb_yaml_to_dict(yaml_data):
    yaml_data = yaml_data['parameters']
    args = {}
    for key, value in yaml_data.items():
        value = value['value'] if 'value' in value else value        
        args[key] = value
    return args