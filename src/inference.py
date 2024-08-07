import statistics
import time
import os
from argparse import ArgumentParser
import torch
import metric
import numpy as np
import matplotlib.pyplot as plt
import copy
import yaml

from dataset import load_datasets
from getModel import get_model
import utils


if __name__ == '__main__':
    torch.set_printoptions(precision=6)
    torch.set_default_dtype(torch.float32)

    torch.manual_seed(42)
    np.random.seed(42)    
    parser = ArgumentParser(description='ML Circuit Array Simulation Version 3.0')
    parser.add_argument('--config_dir', required=False, type=str, default = '../configs', help='Config directory')
    parser.add_argument('-c','--config_file', required=True, type=str, default = '../configs', help='Config File')
    pargs = parser.parse_args()
    with open(os.path.join(pargs.config_dir, f'{pargs.config_file}.yaml')) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        args = utils.convert_wandb_yaml_to_dict(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args['device']
    CHECKPOINT_PATH = f'../checkpoint/'    
    base_name = args['name']
    window_size = args['window_size']
    num_layers = args['num_layers']
    hidden_size = args['hidden_size']
    hidden_channel = args['hidden_channel']
    type_index = args['type_index']

    name = utils.getModelName(base_name, window_size, num_layers, hidden_size, hidden_channel, type_index)
    print(name)
    start = time.time()

    cases = args['cases']
    if args['verbose']:
        print('SPICE Data Loading Completed')

    index_to_name = ['DRG', 'DRS', 'DIODE']
    model = get_model(args, 11, args['model_option'])
    inference_loader = load_datasets(args, type_index, 'inference')
    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, f'{name}_{index_to_name[type_index]}_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.share_memory()
    min_max_dir = '/project/common/LGD/spice_data/raw/max_min'
    v_max, v_min = utils.load_maxmin(min_max_dir)

    start = time.time()
    R2_list, MAPE_list, MAE_list, MSE_list = [], [], [], []
    min_max_dir = '/project/common/LGD/spice_data/raw/max_min'
    final = {}
    total = int((args['end'] - args['start']) * args['resolution'])
    results, y_trues, params = torch.zeros([1, total]).cuda(), torch.zeros([1, total]).cuda(), torch.zeros([1, 10]).cuda()

    with torch.no_grad():
        model.eval()
        for x, y_true, param in inference_loader:
            x = x.to(torch.float32).cuda()
            y_true = y_true.to(torch.float32).cuda()            
            param = param.to(torch.float32).cuda()             
            model_input = x.clone()
            result = x.clone()
            if args['model_option'] != 3:
                for index in range(total - args["window_size"]): 
                    t = torch.full((args['batch_size'], 1), index/1000.0, dtype=torch.float32).cuda()
                    t_param = torch.cat((param, t), dim=1)
                    r = model(model_input, t_param).unsqueeze(1)
                    result = torch.cat([result, r], dim=1)
                    model_input = torch.cat([model_input[:, 1:], r], dim=1)
            else:
                result = model(x, param)
                result = torch.cat([x, result.squeeze()], dim=1)

            results = torch.cat([results, result], dim=0)
            y_trues = torch.cat([y_trues, y_true], dim=0)
            params = torch.cat([params, param], dim=0)
        
    results = utils.linear_denormalization(results[1:], v_max[type_index], v_min[type_index])
    y_trues = utils.linear_denormalization(y_trues[1:], v_max[type_index], v_min[type_index])
    params = params[1:]

    R2   = metric.R2Score(results, y_trues)
    MAPE = metric.MAPE(results, y_trues)
    MAE  = metric.MAE(results, y_trues)
    MSE  = metric.MSE(results, y_trues)
            
    print(f"Inference Time : {time.time()-start}")    
    print(f"TEST R2 score: {R2:4f}\t MAPE: {MAPE:4f}\t MAE: {MAE:4f}\t MSE: {MSE:4f}\tTIME : {(time.time()-start):.2f}")

    np_results = results.clone().detach().cpu().numpy()
    np_y_trues = y_trues.clone().detach().cpu().numpy()
    np_params = params.clone().detach().cpu().numpy()
    #	params = torch.tensor([i, num_scan_pixels, j, num_data_pixels, float(res), cap, tw_s, VDH, load_ratio, type])
    for index in range(np_results.shape[0]):
        np_param = utils.convert_tensor_to_param(np_params[index])
        np_result = np_results[index]
        # final[tuple(list_param)] = np_result
        if args['test_plot'] == 1:
            plt.clf()                    
            x = range(np_result.shape[0])
            plt.plot(x, np_result, 'b')    
            plt.plot(x, np_y_trues[index], 'g')   
            plt.ylim(0, 1)

            plot_dir = f'../plot/{name}/{index_to_name[int(np_param[-1])]}'
            os.makedirs(plot_dir, exist_ok=True)
            t_name = utils.convert_param_to_name(np_param)
            plt.savefig(os.path.join(plot_dir, f'{t_name}.png'))          

    print('Inference Completed')