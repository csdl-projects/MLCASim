import sys
import statistics
import time
import os
import shutil
import csv
from argparse import ArgumentParser
from unittest import result
import torch
import transform
import metric

import numpy as np
import torch.multiprocessing as mp
from dataset import get_datas, get_datasets
from model import Model

from userutil import plot
from userutil import dual_plot
from itertools import product
import matplotlib.pyplot as plt

# from mp_inference import model_inference

config = {
    "cases":[(400, 1, 8.0, 0.008, 2, 5.8)],

    "learning_rate": 0.001,
    "hidden_size": 128,
    "lstm_window_size": 20,
    "lstm_num_layers": 1,
    "lstm_hidden_size": 128,
    "start": 0,
    "end": 40000,
    "resolution": 1e-2,
    "plot_number": 1,
}

def get_model(args, count, num_param, num_batch):
    model = Model(args, count, num_param, num_batch).cuda()
    return model
  

if __name__ == '__main__':
    parser = ArgumentParser(description='LGD Spice Version 2.0')
    parser.add_argument('-n', '--name', required=False, type=str, default = 't', help='Name of model')
    parser.add_argument('-d', '--gpu_ids', required=True, type=int, nargs='+', help='GPU device IDs to use')
    parser.add_argument('-v', '--verbose', required=False, type=int, default = 1, help='True when verbose mode')
    parser.add_argument('-p', '--plot', required=False, type=int, default = 0, help='True when plot mode')

    args = parser.parse_args()
    args = vars(args)
    args.update(config)
    device = [torch.device(f"cuda:{gpu_id}") for gpu_id in args['gpu_ids']]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu_id) for gpu_id in args['gpu_ids'])
    torch.multiprocessing.set_start_method('spawn')

    CHECKPOINT_PATH = f'../checkpoint/'
    name = args['name']

    start = time.time()

    def gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a
    cases = args['cases']
    sum = 0
    datas = []
    for c in cases:
        sum += c[0] * c[1]
        d = get_datas(args, c, 'train')
        datas += d
        
    num_train, num_test = int(0.8 * sum), int(0.2 * sum)
    # num_batch = gcd(num_train, num_test)        
    # num_batch = 72        
    index_to_name = ['R_DRG', 'R_DRS', 'R_DIODE', 'W_DRG', 'W_DRS', 'W_DIODE', 'B_DRG', 'B_DRS', 'B_DIODE', 'G_DRG', 'G_DRS', 'G_DIODE']

    num_batch = 100
    # num_batch = 164
    plot_dataset = get_datasets(args, args['plot_number'] * num_batch, num_train, datas, 'plot')
    plot_loader = torch.utils.data.DataLoader(plot_dataset, batch_size = num_batch, shuffle=False, pin_memory=(torch.cuda.is_available()), num_workers = 15)
    end_dataset = time.time()
    print(f"Dataset generated\tTIME : {(end_dataset- start):.2f}, Total : {len(datas)}, Batch : {num_batch}")
    model = get_model(args, 12, 8, num_batch)
    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, name + '_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.share_memory()

    start = time.time()
    R2_list, MAPE_list, MAE_list, MSE_list = [], [], [], []


    min_max_dir = '/project/common/LGD/data/subpixel/raw/max_min'
    plt.clf()

    with torch.no_grad():
        model.eval()
        for j, (x, y_true, params) in enumerate(plot_loader):
            x = x.to(torch.float64).cuda()
            y_true = y_true.to(torch.float64).cuda()            
            params = params.to(torch.float64).cuda() 

            s = time.time()        
            y_pred = model(x, params, 'inference')
            
            R2   = metric.R2Score(y_pred, y_true)
            MAPE = metric.MAPE(y_pred, y_true)
            MAE  = metric.MAE(y_pred, y_true)
            MSE  = metric.MSE(y_pred, y_true)

            R2_list.append(float(R2))
            MAPE_list.append(float(MAPE))
            MAE_list.append(float(MAE))
            MSE_list.append(float(MSE))

            print(f"One Step : {time.time()-s}")    
            inx, iny = y_pred.detach().cpu().numpy().flatten(), y_true.detach().cpu().numpy().flatten()
            inx = [val if val >= 0 else 0 for val in inx]
            inx = [val if val <= 1 else 1 for val in inx]
            plt.scatter(inx, iny, c='b', s=0.3, alpha=0.1) 
            
            if args['plot'] == 1:
                v_max, v_min = torch.empty([1, y_true.shape[1], y_true.shape[2], y_true.shape[3]]).cuda(),torch.empty([1, y_true.shape[1], y_true.shape[2], y_true.shape[3]]).cuda()
                for i in range(y_pred.shape[0]):                
                    np_v_max = np.loadtxt(os.path.join(min_max_dir, f'max{int(params[i][1])}_{int(params[i][3])}_{float(params[i][4]):.1f}_{float(params[i][5]):.3f}_2_5.8.np'), dtype=float)
                    np_v_min = np.loadtxt(os.path.join(min_max_dir, f'min{int(params[i][1])}_{int(params[i][3])}_{float(params[i][4]):.1f}_{float(params[i][5]):.3f}_2_5.8.np'), dtype=float)
                    t_v_max, t_v_min = torch.from_numpy(np_v_max).to(torch.float64).cuda(), torch.from_numpy(np_v_min).to(torch.float64).cuda()
                    t_v_max, t_v_min = t_v_max[10:], t_v_min[10:]
                    t_v_max, t_v_min = t_v_max.unsqueeze(0).unsqueeze(0).unsqueeze(0), t_v_min.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    t_v_max, t_v_min = t_v_max.repeat(1, y_true.shape[1], 1, 1), t_v_min.repeat(1, y_true.shape[1], 1, 1)
                    v_max, v_min = torch.cat([v_max, t_v_max], dim=0), torch.cat([v_min, t_v_min], dim=0)

                v_max, v_min = v_max[1:,], v_min[1:,]

                x = x * (v_max - v_min) + v_min
                y_true = y_true * (v_max - v_min) + v_min
                y_pred = y_pred * (v_max - v_min) + v_min
                # print(y_pred.shape)
                x = x.detach().cpu().numpy()
                y_true = y_true.detach().cpu().numpy()
                y_pred = y_pred.detach().cpu().numpy()      
                for i in range(y_true.shape[0]):
                    for index, type in enumerate(index_to_name):
                        plot_dir = f'../plot/{name}/{type}'
                        os.makedirs(plot_dir, exist_ok=True)
                        # print(y_pred.shape)
                        pred = np.concatenate([x[i, 0, :, index].flatten(), y_pred[i, :, :, index].flatten()])
                        true = np.concatenate([x[i, 0, :, index].flatten(), y_true[i, :, :, index].flatten()])        
                        dual_plot(pred, true, plot_dir, f'{j*y_pred.shape[0] + i}', 'pred', 'golden')
                        p = params[i].detach().cpu().numpy()
                        with open(f'../plot/{name}/{type}/{j*y_pred.shape[0] + i}.csv', 'w') as f:
                            wr = csv.writer(f)
                            wr.writerow(p)
                            wr.writerow(['PREDICTED VALUE', 'GOLDEN'])
                            wr.writerows(zip(pred, true))          

    R2   = statistics.mean(R2_list)
    MAPE = statistics.mean(MAPE_list)
    MAE  = statistics.mean(MAE_list)
    MSE  = statistics.mean(MSE_list)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot([0, 1], [0, 1], color='k')
    plt.savefig(f'r2__4.png')
            
    print(f"TEST R2 score: {R2:4f}\t MAPE: {MAPE:4f}\t MAE: {MAE:4f}\t MSE: {MSE:4f}\tTIME : {(time.time()-start):.2f}")
