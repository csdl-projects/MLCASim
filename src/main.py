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
            # "values": [[(1920, 1, 8.0, 0.008, 2, 5.8, 2), (1920, 1, 8.0, 0.008, 2, 4.8, 2), (1920, 1, 8.0, 0.008, 2, 3.9, 2), (1920, 1, 8.0, 0.008, 2, 3.1, 2)]],
            "values": [[(1920, 1, 8.0, 0.008, 2, 5.9, 2), (1920, 1, 8.0, 0.008, 2, 4.8, 2), (1920, 1, 8.0, 0.008, 2, 4.3, 2), (1920, 1, 8.0, 0.008, 2, 3.9, 2), (1920, 1, 8.0, 0.008, 2, 3.6, 2), (1920, 1, 8.0, 0.008, 2, 3.3, 2), (1920, 1, 8.0, 0.008, 2, 3.1, 2)]],
            # "values": [[(1, 1080, 8.0, 0.008, 2, 5.9, 2), (1, 1080, 8.0, 0.008, 2, 4.8, 2), (1, 1080, 8.0, 0.008, 2, 4.3, 2), (1, 1080, 8.0, 0.008, 2, 3.9, 2), (1, 1080, 8.0, 0.008, 2, 3.6, 2), (1, 1080, 8.0, 0.008, 2, 3.3, 2), (1, 1080, 8.0, 0.008, 2, 3.1, 2)]],
        },
        "learning_rate": {
            "values": [0.001]
        },
        "input_size": {
            "values" : [3]
        },
        "lstm_window_size": {
            "values" : [70]
        },
        # "lstm_window_size": {
        #     "max" : 100,
        #     "min" : 10,
        # },
        "lstm_num_layers": {
            "values" : [1]
        },
        "lstm_hidden_size": {
            "values" : [200]
            # "values" : [500, 1000]
        },
        # "lstm_hidden_size": {
        #     "max" : 200,
        #     "min" : 10,
        # },
        "start": {
            "values":[0]
        },
        "end": {
            # "values":[15000]
            "values":[20000]
        },
        "resolution": {
            # "values":[1e-2]
            "values":[4e-2]
        },
        "batch_size": {
            "values":[200]
        },
    }
}

def get_model(args, num_param, PE = 0, mode = 0, option=0):
    if option == 0:
        model = Model(args, num_param, PE, mode).cuda()
        
    if option == 1:
        model = FastModel(args, num_param, PE).cuda()

    return model


def train(args, model, train_loader, loss_function, optimizer, epoch, mode):
    train_losses = []
    model.train()
    start = time.time()

    for x, y_true, params in train_loader:        
        x = x.to(torch.float64).cuda()
        y_true = y_true.to(torch.float64).cuda()
        params = params.to(torch.float64).cuda() 
        optimizer.zero_grad()
        y_pred = model(x, params)
        train_loss = loss_function(y_pred, y_true)

        train_losses.append(float(train_loss))
        train_loss.backward()
        optimizer.step()

    train_loss = statistics.mean(train_losses)
    
    if args['verbose'] and epoch%10 == 0:
        print(f'EPOCH {epoch}\t LOSS : {train_loss:.6f}\tTIME : {(time.time()-start):.2f}')

    wandb.log({
        f'Train Loss' : train_loss,
    })
    return model, optimizer


def test(args, model, test_loader, epoch, mode):
    start = time.time()
    R2_list, MAPE_list, MAE_list, MSE_list = [], [], [], []
    model.eval()
    with torch.no_grad():
        for i, (x, y_true, params) in enumerate(test_loader):
            x = x.to(torch.float64).cuda()
            y_true = y_true.to(torch.float64).cuda() 
            params = params.to(torch.float64).cuda() 

            y_pred = model(x, params)

            R2   = metric.R2Score(y_pred, y_true)
            MAPE = metric.MAPE(y_pred, y_true)
            MAE  = metric.MAE(y_pred, y_true)
            MSE  = metric.MSE(y_pred, y_true)
            
            R2_list.append(float(R2))
            MAPE_list.append(float(MAPE))
            MAE_list.append(float(MAE))
            MSE_list.append(float(MSE))
    
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
    
    if args['verbose'] and epoch%10 == 0:
    # # overall average of metrics
        print(f"TEST R2 score: {R2:4f}\t MAPE: {MAPE:4f}\t MAE: {MAE:4f}\t MSE: {MSE:4f}\tTIME : {(time.time()-start):.2f}")
        
    return [R2, MAPE, MAE, MSE]


def process(args, train_loader, test_loader, CHECKPOINT_PATH, name, maxepoch):
    name = name

    # best_loss, best_loss_, best_loss__ = 1e9, 1e9, 1e9
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
        start_epoch= checkpoint['epoch'] + 1
        best_loss = checkpoint['loss'][3]

    loss_function = torch.nn.MSELoss()  
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
    scheduler.last_epoch = start_epoch - 1
    if args['verbose']:
        print(f'Training starts, epoch : {scheduler.last_epoch + 1}' )

    for epoch in tqdm(range(start_epoch, maxepoch), desc="Training",leave=True):
        model, optimizer = train(args, model, train_loader, loss_function, optimizer, epoch, 'm')
        loss = test(args, model, test_loader, epoch, 'm')

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
            
    wandb.watch(model)
    if args['verbose']:        
        end_train = time.time()
        print(f"Train completed\t\tTIME : {(end_train - end_model):.2f}")


def main():
    torch.set_printoptions(precision=6)
    parser = ArgumentParser(description='ML Circuit Array Simulation Version 3.0')
    parser.add_argument('-n', '--name', required=False, type=str, default = 't', help='Name of model')
    parser.add_argument('-e', '--epoch', required=False, type=int, default = '0', help='Number of training epoch')
    parser.add_argument('-o', '--model_option', required=False, type=int, default = '0', help='Number of model type')
    parser.add_argument('-pe', '--PE', required=False, type=int, default = 0, help='Positional Encoding')
    parser.add_argument('-m', '--mode', required=False, type=int, default = 0, help='Positional Encoding Sum or Concatenate')
    parser.add_argument('-d', '--device', required=True, type=str, help='gpu-id')
    parser.add_argument('-r', '--resume', required=False, type=int, default = 0, help='True when resume')
    parser.add_argument('-v', '--verbose', required=False, type=int, default = 0, help='True when verbose mode')
    parser.add_argument('-t', '--test', required=False, type=int, default = 0, help='True when want FINAL Test')
    parser.add_argument('-si', '--sample_num', required=False, type=int, default = 30, help='Number of sample hop') 
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
    
    train_loader, test_loader = load_datasets(args)
    end_dataset = time.time()
    
    if args['verbose']:
        print(f"Dataset generated\tTIME : {(end_dataset- start):.2f}, Batch : {args['batch_size']}") 

    # Dataset generation
    if args['epoch'] > 0:
        process(args, train_loader, test_loader, CHECKPOINT_PATH, name, args['epoch'])

    print("time (Total)   : ", time.time() - end_dataset)

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config['project'])    
    wandb.agent(sweep_id, function = main)
