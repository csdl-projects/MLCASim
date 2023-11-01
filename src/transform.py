### transform.py

import torch
import time
import os
from argparse import ArgumentParser


def lis_file_parse(file_path, output_name):
    labels = []
    voltage_data = torch.empty([1,1])
    with open(file_path, 'r') as file:
        lines = file.readlines()
        start_index = -1
        label_line_indices = []

        for index, line in enumerate(lines):
            if ' transient analysis' in line:
                start_index = index + 3
            if start_index != -1:
                if 'xpixel' in line:
                    label_line_indices.append(index+1)
                    labels.extend(line.split())
        s = 0
        prev = 0
        prec = 0
        for index in label_line_indices:
            label_num = len(lines[index-1].strip().split())
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
                    
                volt= torch.cat([volt, voltages], dim=1)

    torch.save(voltage_data,output_name)
    return voltage_data.shape

def extract_scan_rising_time(times, datas, max, min):    
    isFalling = 0
    isStart, isEnd = 0, 0
    targetY_s, targetY_e = 0.90 * min + 0.1 * max, 0.1 * min + 0.9 * max

    def interpolate(x1, x2, y1, y2, target_y):
        if y2 - y1 == 0:
            return x1
        delta_y = y2 - y1
        delta_x = x2 - x1
        delta_target_y = target_y - y1
        delta_ratio = delta_target_y / delta_y
        predicted_x = x1 + delta_ratio * delta_x
        return predicted_x

    for index, data in enumerate(datas):
        if targetY_s < data and isStart == 0:
            risingX_s = interpolate(times[index-1], times[index], datas[index-1], datas[index], targetY_s)
            isStart = 1
        if targetY_e < data and isEnd == 0 :
            risingX_e = interpolate(times[index-1], times[index], datas[index-1], datas[index], targetY_e)
            isEnd = 1
        
        if isStart and isEnd:
            if datas[index - 1] - data > 0.01:
                isFalling = 1
            
        if isFalling:
            if targetY_e < data and isStart == 1:
                fallingX_s = interpolate(times[index-1], times[index], datas[index-1], datas[index], targetY_e)
                isStart = 2

            if targetY_s < data and isEnd == 1:
                fallingX_e = interpolate(times[index-1], times[index], datas[index-1], datas[index], targetY_s)
                isEnd = 2
                break
                
    return risingX_s, risingX_e, fallingX_s, fallingX_e

def find_maxmin(datas):
    max_ = torch.full((22,), -1e20)
    min_ = torch.full((22,), 1e20)
    for index in range(datas.shape[0]):
        s_index = index % 22
        if max_[s_index] < torch.max(datas[index, :]).item():
            max_[s_index] = torch.max(datas[index, :]).item()

        if min_[s_index] > torch.min(datas[index, :]).item():
            min_[s_index] = torch.min(datas[index, :]).item()

    return max_, min_
                                
def Z_Normalization(input, mean, std):
    return (input-mean)/std

def MM_Normalization(input, _min, _max):
    return (input - _min)/(_max - _min)

if __name__ == '__main__':
    s = time.time()
    parser = ArgumentParser(description='SPICE')
    parser.add_argument('-s', '--number_scanline_pixel', required=True, type=int, default = '10', help='number')
    parser.add_argument('-d', '--number_dataline_pixel', required=True, type=int, default = '10', help='number')
    parser.add_argument('-r', '--res', required=False, type=float, default = '8', help='number')
    parser.add_argument('-c', '--cap', required=False, type=float, default = '0.008', help='number')
    parser.add_argument('-t', '--tw_s', required=False, type=int, default = '2', help='number')
    parser.add_argument('-v', '--VDH', required=False, type=float, default = '5.8', help='number')

    args = parser.parse_args()

    number_scanline_pixel = args.number_scanline_pixel
    number_dataline_pixel = args.number_dataline_pixel
    res = args.res
    cap = args.cap
    tw_s = 'TH*2'
    if tw_s == 1:
        args.tw_s = 'TH*1'
    VDH = args.VDH
    pixel_name = f'{number_scanline_pixel}_{number_dataline_pixel}_{res}_{cap}_{args.tw_s}_{VDH}'
    rawdata_home = '/project/common/LGD/data/subpixel/raw/'

    rawfile = os.path.join(rawdata_home, f'raw{pixel_name}.raw')
    # if not os.path.exists(rawfile):
    #     print("RAWFILE")
    #     print(lis_file_parse(f'/project/common/LGD/data/subpixel/circuit/pixel/data/Array{pixel_name}.lis', rawfile))

    print(lis_file_parse(f'/project/common/LGD/data/subpixel/circuit/pixel/data/Array{pixel_name}.lis', rawfile))
    print(pixel_name, time.time()-s)