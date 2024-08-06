### userutil.py
### user utilization tools

import os
import subprocess as sp
import decimal
import errno
import numpy as np
import matplotlib.pyplot as plt

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