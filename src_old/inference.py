import statistics
import time
import os
from argparse import ArgumentParser
import torch
import metric
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from dataset import load_datasets
from model import *
from utils import *

# from mp_inference import model_inference

config = {
    # "cases":[[1920, 1, 8.0, 0.008, 2, 5.9, 2], [1920, 1, 8.0, 0.008, 2, 4.3, 2], [1920, 1, 8.0, 0.008, 2, 3.6, 2], [1920, 1, 8.0, 0.008, 2, 3.3, 2]],
    # "cases":[[1, 1080, 8.0, 0.008, 2, 5.8, 2], [1, 1080, 8.0, 0.008, 2, 4.8, 2], [1, 1080, 8.0, 0.008, 2, 3.9, 2], [1, 1080, 8.0, 0.008, 2, 3.1, 2]],
    "cases": [[64, 10, 36, 10, 8.0, 0.008, 2, 3.0, 2],],
    # "cases": [[2, 2, 1080, 50, 8.0, 0.008, 2, 3.0, 2],],

    "window_size": 10,
    "lstm_num_layers": 1,
    "hidden_size": 50,
    "input_size": 3,
    "start": 0,
    # "end": 15000,
    "end": 200000,
    "resolution": 1e-2,
    # "resolution": 4e-2,
    "batch_size": 2,
}

def get_model(args, num_param, PE = 0, mode = 0, option=0):
    if option == 0:
        model = Model(args, num_param, PE, mode).cuda()
        
    if option == 1:
        model = RidgeModel(args, num_param, PE).cuda()

    return model


  
if __name__ == '__main__':
    parser = ArgumentParser(description='LGD Spice Version 2.0')
    parser.add_argument('-n', '--name', required=False, type=str, default = 't', help='Name of model')
    parser.add_argument('-d', '--device', required=True, type=str, help='gpu-id')
    # parser.add_argument('-d', '--device', required=True, type=int, nargs='+', help='GPU device IDs to use')
    parser.add_argument('-v', '--verbose', required=False, type=int, default = 0, help='True when verbose mode')
    parser.add_argument('-p', '--plot', required=False, type=int, default = 0, help='True when plot mode')
    # parser.add_argument('-s', '--sample_num', required=False, type=int, default = 30, help='Number of sample hop') 
    parser.add_argument('-o', '--model_option', required=False, type=int, default = '0', help='Number of model type')
    parser.add_argument('-pe', '--PE', required=False, type=int, default = '0', help='Positional Encoding')
    parser.add_argument('-m', '--mode', required=False, type=int, default = 0, help='Positional Encoding Sum or Concatenate')

    args = parser.parse_args()
    args = vars(args)
    args.update(config)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu_id) for gpu_id in args['gpu_ids'])
    os.environ["CUDA_VISIBLE_DEVICES"] = args['device']

    CHECKPOINT_PATH = f'../checkpoint/'
    
    base_name = args['name']
    window_size = args['window_size']
    lstm_num_layers = args['lstm_num_layers']
    hidden_size = args['hidden_size']
    hidden_channel = args['hidden_channel']

    name = f'{base_name}_{window_size}_{lstm_num_layers}_{hidden_size}_{hidden_channel}'
    start = time.time()
    cases = args['cases']
    if args['verbose']:
        print('SPICE Data Loading Completed')
    index_to_name = ['DRG', 'DRS', 'DIODE']


    # for type in range(3):
    for type in [2]:
        model = get_model(args, 11, args['PE'], args['mode'], args['model_option'])
        plot_loader = load_datasets(args, type, 'plot')
        checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, f'{name}_{index_to_name[type]}_best.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.share_memory()

        start = time.time()
        R2_list, MAPE_list, MAE_list, MSE_list = [], [], [], []
        min_max_dir = '/project/common/LGD/spice_data/raw/max_min'

        # print(len(plot_loader.dataset))
        final = {}
        with torch.no_grad():
            model.eval()
            for j, (x, y_true, params) in enumerate(plot_loader):
                # print(x.shape, y_true.shape, params.shape)
                
                x = x.to(torch.float32).cuda()
                y_true = y_true.to(torch.float32).cuda()            
                params = params.to(torch.float32).cuda() 
                window_size = args['window_size']

                s = time.time()
                input = x.clone()
                result = x.clone()
                for index in range(y_true.shape[1] - window_size):
                    t = torch.full((args['batch_size'], 1), index/1000.0, dtype=torch.float32).cuda()
                    t_params = torch.cat((params, t), dim=1)
                    y_pred = model(input, t_params).unsqueeze(1)
                    input = torch.cat((input[:, 1:], y_pred), dim=1)
                    result = torch.cat((result, y_pred), dim=1)

                R2   = metric.R2Score(result, y_true)
                MAPE = metric.MAPE(result, y_true)
                MAE  = metric.MAE(result, y_true)
                MSE  = metric.MSE(result, y_true)

                R2_list.append(float(R2))
                MAPE_list.append(float(MAPE))
                MAE_list.append(float(MAE))
                MSE_list.append(float(MSE))
                
                # print(f"One Step : {time.time()-s}")    

                #	params = torch.tensor([i, num_scan_pixels, j, num_data_pixels, float(res), cap, tw_s, VDH, load_ratio, type])
                for index in range(x.shape[0]):
                    list_param = params[index].tolist()
                    # print(list_param)
                    np_result = result[index].clone().detach().cpu().numpy()
                    final[tuple(list_param)] = np_result
                    if args['plot'] == 1:
                        plt.clf()                    
                        x = range(y_true.shape[1])
                        plt.plot(x, np_result, 'b')    
                        plt.plot(x, y_true[index].clone().detach().cpu().numpy(), 'g')   
                        plt.ylim(0, 1)

                        plot_dir = f'../plot/{name}/{index_to_name[int(list_param[-1])]}'
                        os.makedirs(plot_dir, exist_ok=True)
                        t_name = f'{int(list_param[0]*2000)}_{int(list_param[1]*2000)}_{int(list_param[2]*2000)}_{int(list_param[3]*2000)}_{int(list_param[4]*8)}_{list_param[5]:.3f}_{int(list_param[6]*2)}_{float(list_param[7]*10):.1f}_{int(list_param[8]*10)}'
                        plt.savefig(os.path.join(plot_dir, f'{t_name}.png'))          

        R2   = statistics.mean(R2_list)
        MAPE = statistics.mean(MAPE_list)
        MAE  = statistics.mean(MAE_list)
        MSE  = statistics.mean(MSE_list)


                
        print(f"TEST R2 score: {R2:4f}\t MAPE: {MAPE:4f}\t MAE: {MAE:4f}\t MSE: {MSE:4f}\tTIME : {(time.time()-start):.2f}")
