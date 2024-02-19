### transform.py

import torch
import multiprocessing
import time
import os
from argparse import ArgumentParser


def lis_file_parse(lis_file, raw_file, label_file):
    labels = []
    voltage_data = torch.empty([1,1])
    with open(lis_file, 'r') as file:
        lines = file.readlines()
        start_index = -1
        label_line_indices = []

        for index, line in enumerate(lines):
            if ' transient analysis' in line:
                start_index = index + 1
            if start_index != -1:
                if 'x' == line[0]:
                    label_line_indices.append(index+3)
        s = 0
        prev = 0
        prec = 0
        for index in label_line_indices:
            labels.extend(lines[index].strip().split())
            label_num = len(lines[index].strip().split())
            volt = torch.empty([label_num,1])
            time = torch.empty([1])
            for line in lines[index+1:]:
                if line.strip() == '': 
                    continue
                if line.strip() == 'y':
                    if s == 0:
                        voltage_data = time[1:].unsqueeze(0)
                        s = 1
                    voltage_data = torch.cat([voltage_data, volt[:, 1:]], dim=0)                    
                    break

                values = line.strip().split()
                voltages = torch.tensor([float(value) for value in values[1:]]).unsqueeze(1)
                if s == 0:
                    if prev == float(values[0]):
                        prec = prec + 2e-10
                        if prec >= 1e-9:
                            prec = 0

                    t = torch.tensor([float(values[0]) + prec])
                    time = torch.cat([time,t], dim=0)
                    prev = float(values[0])
                    
                volt = torch.cat([volt, voltages], dim=1)

    with open(label_file, 'w') as file:
        file.write(' '.join(labels))
    torch.save(voltage_data, raw_file)
    return voltage_data.shape


def Z_Normalization(input, mean, std):
    return (input-mean)/std

def MM_Normalization(input, _min, _max):
    return (input - _min)/(_max - _min)
