## main.py
## parameter sweep script

import statistics
import time
import os
import shutil
from argparse import ArgumentParser
import torch
import metric
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt


from dataset import load_datasets
from model import *

sweep_config = {
    "project": "MLCASim",
    "method": "grid",
    # "method": "bayes",
    "metric": {
        "goal": "minimize",
        "name": "MSE"
    },
    "parameters": {
        "cases": {
            "values": [[(1920, 1, 8.0, 0.008, 2, 4.8, 2), (1920, 1, 8.0, 0.008, 2, 4.3, 2), (1920, 1, 8.0, 0.008, 2, 3.9, 2), (1920, 1, 8.0, 0.008, 2, 3.6, 2), (1920, 1, 8.0, 0.008, 2, 3.3, 2), (1920, 1, 8.0, 0.008, 2, 3.1, 2)]],
            # "values": [[(1, 1080, 8.0, 0.008, 2, 5.9, 2), (1, 1080, 8.0, 0.008, 2, 4.8, 2), (1, 1080, 8.0, 0.008, 2, 4.3, 2), (1, 1080, 8.0, 0.008, 2, 3.9, 2), (1, 1080, 8.0, 0.008, 2, 3.6, 2), (1, 1080, 8.0, 0.008, 2, 3.3, 2), (1, 1080, 8.0, 0.008, 2, 3.1, 2)]],
        },
        "learning_rate": {
            "values": [0.001]
        },
        "input_size": {
            "values" : [3]
        },
        "lstm_window_size": {
            "values" : [50]
        },
        "lstm_num_layers": {
            "values" : [1]
        },
        "lstm_hidden_size": {
            "values" : [100]
        },
        "start": {
            "values":[0]
        },
        "end": {
            "values":[20000]
            # "values":[200000]
        },
        "resolution": {
            "values":[5e-2]
            # "values":[1e-3]
        },
        "batch_size": {
            "values":[200]
            # "values":[50]
        },
    }
}

def get_model(args, num_param, PE = 0, mode = 0, option=0):
    if option == 0:
        model = Model(args, num_param, PE, mode)
        
    if option == 1:
        model = FastModel(args, num_param, PE)

    return model


def train(args, model, train_loader, loss_function, optimizer, epoch, mode):
    train_losses, times = [], []
    model.train()

    start = time.time()
    total = int((args['end'] - args['start']) * args['resolution'])

    for x, y_true, params in train_loader:
        # torch.cuda.empty_cache()
        start = time.time()
        x = x.to(torch.float32).cuda()
        y_true = y_true.to(torch.float32).cuda()       
        params = params.to(torch.float32).cuda()
        # print(x.shape, y_true.shape, params.shape)

        optimizer.zero_grad()
        model_input = x.clone()
        result = x.clone()
        for index in range(total - args["lstm_window_size"]): 
            t = torch.full((args['batch_size'], 1), index, dtype=torch.float32).cuda()
            t_params = torch.cat((params, t), dim=1)
            r = model(model_input, t_params).unsqueeze(1)
            result = torch.cat([result, r], dim=1)
            model_input = torch.cat([model_input[:, 1:], r], dim=1)

        loss = loss_function(result, y_true)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss))            
        times.append(time.time()-start)

    train_loss = max(train_losses)
    
    if args['verbose'] and epoch%10 == 0:
        print(f'EPOCH {epoch}\t LOSS : {train_loss:.6f}\tTIME : {(time.time()-start):.2f}')

    wandb.log({
        f'Train Loss' : train_loss,
        f'One Epoch Time' : max(times),
    })
    return model, optimizer

def plot(args, model, plot_loader, name, best_loss):    
    index_to_name = ['W_DRG', 'W_DRS', 'W_DIODE']
    R2_list, MAPE_list, MAE_list, MSE_list = [], [], [], []
    min_max_dir = '/project/common/LGD/spice_data/raw/max_min'
    final = {}
    total = int((args['end'] - args['start']) * args['resolution'])

    with torch.no_grad():        
        model.eval()
        for x, y_true, params in plot_loader:
            x = x.to(torch.float32).cuda()
            y_true = y_true.to(torch.float32).cuda()       
            params = params.to(torch.float32).cuda()

            model_input = x.clone()
            result = x.clone()
            for index in range(total - args["lstm_window_size"]): 
                t = torch.full((args['batch_size'], 1), index, dtype=torch.float32).cuda()
                t_params = torch.cat((params, t), dim=1)
                r = model(model_input, t_params).unsqueeze(1)
                result = torch.cat([result, r], dim=1)
                model_input = torch.cat([model_input[:, 1:], r], dim=1)

            R2   = metric.R2Score(result, y_true)
            MAPE = metric.MAPE(result, y_true)
            MAE  = metric.MAE(result, y_true)
            MSE  = metric.MSE(result, y_true)

            R2_list.append(float(R2))
            MAPE_list.append(float(MAPE))
            MAE_list.append(float(MAE))
            MSE_list.append(float(MSE))

            if best_loss > MSE:
                for index in range(args['batch_size']):
                    list_param = params[index].tolist()

                    np_result = result[index].clone().detach().cpu().numpy()
                    np_y_true = y_true[index].clone().detach().cpu().numpy()
                    final[tuple(list_param)] = np_result
                    if args['plot'] == 1:
                        plt.clf()                    
                        x = range(y_true.shape[1])
                        plt.plot(x, np_result, 'b')    
                        plt.plot(x, np_y_true, 'g')   
                        plt.ylim(0, 1)

                        plot_dir = f'../plot/{name}/{index_to_name[int(list_param[-1])]}'
                        os.makedirs(plot_dir, exist_ok=True)
                        t_name = f'{int(list_param[0]*2000)}_{int(list_param[1]*2000)}_{int(list_param[2]*2000)}_{int(list_param[3]*2000)}_{int(list_param[4]*8)}_{list_param[5]:.3f}_{int(list_param[6]*2)}_{float(list_param[7]*10):.1f}_{int(list_param[8]*10)}'
                        plt.savefig(os.path.join(plot_dir, f'{t_name}.png'))

    R2   = statistics.mean(R2_list)
    MAPE = statistics.mean(MAPE_list)
    MAE  = statistics.mean(MAE_list)
    MSE  = statistics.mean(MSE_list)
    
    wandb.log({
        f"R2" : R2,
        f"MAPE" : MAPE,
        f"MAE": MAE,
        f"MSE": MSE
    })

    return [R2, MAPE, MAE, MSE]

def process(args, train_loader, plot_loader, CHECKPOINT_PATH, name, maxepoch):
    best_loss = 1e9
    start_epoch = 0

    if args['verbose']:
        start = time.time()       

    model = get_model(args, 11, args['PE'], args['mode'], args['model_option'])
    optimizer = torch.optim.Adam(model.parameters(), lr = args['learning_rate'])    

    if args['verbose']:
        end_model = time.time()
        print(f"Model generated\t\tTIME : {(end_model - start):.2f}")

    if args['resume']:
        checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, name + '.pth'))
        model.load_state_dict(checkpoint[f'model_state_dict'])
        optimizer.load_state_dict(checkpoint[f'model_opt_state_dict'])
        best_loss = checkpoint['loss'][3]
        start_epoch= checkpoint['epoch'] + 1

    loss_function = torch.nn.MSELoss()
    wandb.watch(model, loss_function, log="all", log_freq=10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
    scheduler.last_epoch = start_epoch - 1
    if args['verbose']:
        print(f'Training starts, epoch : {scheduler.last_epoch + 1}' )

    for epoch in tqdm(range(start_epoch, maxepoch), desc="Training",leave=True):
        model, optimizer = train(args, model, train_loader, loss_function, optimizer, epoch, 'm')
        loss = plot(args, model, plot_loader, name, best_loss)

        state = {
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'model_opt_state_dict' : optimizer.state_dict(),
            'loss' : loss,
        }
        torch.save(state, os.path.join(CHECKPOINT_PATH, name + '.pth'))
        if best_loss > loss[3]:
            best_loss = loss[3]
            shutil.copyfile(os.path.join(CHECKPOINT_PATH, name + '.pth'),
                            os.path.join(CHECKPOINT_PATH, name + '_best.pth'))
            
    if args['verbose']:        
        end_train = time.time()
        print(f"Train completed\t\tTIME : {(end_train - end_model):.2f}")


def main():
    torch.set_printoptions(precision=6)
    parser = ArgumentParser(description='ML Circuit Array Simulation Version 3.0')
    parser.add_argument('-n', '--name', required=False, type=str, default = 't', help='Name of model')
    parser.add_argument('-e', '--epoch', required=True, type=int, default = '100', help='Number of training epoch')
    parser.add_argument('-o', '--model_option', required=False, type=int, default = '0', help='Number of model type')
    parser.add_argument('-pe', '--PE', required=False, type=int, default = 0, help='Positional Encoding')
    parser.add_argument('-m', '--mode', required=False, type=int, default = 0, help='Positional Encoding Sum or Concatenate')
    parser.add_argument('-d', '--device', required=True, type=str, help='gpu-id')
    parser.add_argument('-r', '--resume', required=False, type=int, default = 0, help='True when resume')
    parser.add_argument('-v', '--verbose', required=False, type=int, default = 0, help='True when verbose mode')
    # parser.add_argument('-t', '--test', required=False, type=int, default = 0, help='True when want FINAL Test')
    parser.add_argument('-si', '--sample_num', required=False, type=int, default = 30, help='Number of sample hop') 
    parser.add_argument('-p', '--plot', required=False, type=int, default = 1, help='True when plot mode')
    args = parser.parse_args()

    wandb.init(project = "MLCASim")
    wandb.config.update(args)
    args = wandb.config
    os.environ["CUDA_VISIBLE_DEVICES"] = args['device']

    CHECKPOINT_PATH = f'../checkpoint/'
    base_name = args['name']
    lstm_window_size = args['lstm_window_size']
    lstm_num_layers = args['lstm_num_layers']
    lstm_hidden_size = args['lstm_hidden_size']

    name = f'{base_name}_{lstm_window_size}_{lstm_num_layers}_{lstm_hidden_size}'
    if not os.path.isdir(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    start = time.time()
    if args['verbose']:
        print('SPICE Data Loading Completed')
    
    train_loader, _ = load_datasets(args)
    # plot_loader = load_datasets(args, 'plot')

    end_dataset = time.time()
    
    if args['verbose']:
        print(f"Dataset generated\tTIME : {(end_dataset- start):.2f}, Batch : {args['batch_size']}") 

    # Dataset generation
    if args['epoch'] > 0:
        process(args, train_loader, train_loader, CHECKPOINT_PATH, name, args['epoch'])

    print("time (Total)   : ", time.time() - end_dataset)

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config['project'])    
    wandb.agent(sweep_id, function = main)
