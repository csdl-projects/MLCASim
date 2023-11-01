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

from construct_dataset import get_datas, get_datasets
from model import Model

sweep_config = {
    "project": "MLCASim",
    "method": "grid",
    "metric": {
        "goal": "minimize",
        "name": "MSE"
    },
    "parameters": {
        "cases": {
            "values": [[(101, 1, 8.0, 0.008, 2, 5.8), ]],
        },
        "learning_rate": {
            "values": [0.001]
        },
        "hidden_size": {
            "values" : [128]
        },
        "lstm_window_size": {
            "values" : [20]
        },
        "lstm_num_layers": {
            "values" : [1]
        },
        "lstm_hidden_size": {
            "values" : [128]
        },
        "start": {
            "values":[0]
        },
        "end": {
            "values":[20000]
        },
        "resolution": {
            "values":[1e-2]
        },
        "batch": {
            "values":[100]
        },
    }
}

def get_model(args, count, num_param, batch):
    model = Model(args, count, num_param, batch).cuda()
    return model


def train(args, model, train_loader, loss_function, optimizer, epoch, mode):
    train_losses = []
    model.train()
    start = time.time()

    for _, (x, y_true, params) in enumerate(train_loader):        
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
        # for i, (x, y_true, params, scanline, dataline) in enumerate(test_loader):
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


def process(args, batch, train_loader, test_loader, CHECKPOINT_PATH, name, maxepoch):
    name = name

    # best_loss, best_loss_, best_loss__ = 1e9, 1e9, 1e9
    best_loss = 1e9
    start_epoch = 0

    if args['verbose']:
        start = time.time()       

    model = get_model(args, 12, 8, batch)
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
    parser.add_argument('-e', '--epoch', required=False, type=int, default = '0', help='Number tof training epoch')
    parser.add_argument('-d', '--device', required=True, type=str, help='gpu-id')
    parser.add_argument('-b', '--batch', required=False, type=int, default=1, help='batch size')
    parser.add_argument('-r', '--resume', required=False, type=int, default = 0, help='True when resume')
    parser.add_argument('-v', '--verbose', required=False, type=int, default = 1, help='True when verbose mode')
    parser.add_argument('-t', '--test', required=False, type=int, default = 0, help='`True when want FINAL Test')
    args = parser.parse_args()

    wandb.init(project = "MLCASim")
    wandb.config.update(args)
    args = wandb.config
    os.environ["CUDA_VISIBLE_DEVICES"] = args['device']

    CHECKPOINT_PATH = f'../checkpoint/'
    name = args['name']
    if not os.path.isdir(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    start = time.time()

    cases = args['cases']
    sum = 0
    datas = []
    for c in cases:
        sum += c[0] * c[1]
        if c[2] == 8:
            c[2] = '8.0'
        d = get_datas(args, c, 'train')
        datas += d
        
    num_train, num_test = int(0.8 * sum), int(0.2 * sum)
    if args['verbose']:
        print('SPICE Data Loading Completed')
    train_dataset, test_dataset = get_datasets(args, num_train, num_test, datas, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args['batch'], shuffle=False, pin_memory=(torch.cuda.is_available()), num_workers = 15)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args['batch'], shuffle=False, pin_memory=(torch.cuda.is_available()), num_workers = 15)
    end_dataset = time.time()
    
    if args['verbose']:
        print(f"Dataset generated\tTIME : {(end_dataset- start):.2f}, Total : {len(datas)}, Batch : {args['batch']}") 

    # Dataset generation
    if args['epoch'] > 0:
        process(args, args['batch'], train_loader, test_loader, CHECKPOINT_PATH, name, args['epoch'])

    print("time (Total)   : ", time.time() - end_dataset)

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config['project'])    
    wandb.agent(sweep_id, function = main)
