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
import yaml


from dataset import load_datasets
from get_model import get_model


def train(args, model, train_loader, loss_function, optimizer, schedular, epoch, mode):
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

        optimizer.zero_grad()
        model_input = x.clone()
        result = x.clone()
        if args['model_option'] != 3:
            for index in range(total - args["window_size"]): 
                t = torch.full((args['batch_size'], 1), index/1000.0, dtype=torch.float32).cuda()
                t_params = torch.cat((params, t), dim=1)
                r = model(model_input, t_params).unsqueeze(1)
                result = torch.cat([result, r], dim=1)
                model_input = torch.cat([model_input[:, 1:], r], dim=1)

        else:
            result = model(x, params)
            result = torch.cat([x, result.squeeze()], dim=1)

        loss = loss_function(result, y_true)
        loss.backward()
        optimizer.step()
        schedular.step()
        
        train_losses.append(float(loss))            
        times.append(time.time()-start)

    train_loss = max(train_losses)
    
    if args['verbose'] and epoch%10 == 0:
        print(f'EPOCH {epoch}\t LOSS : {train_loss:.6f}\tTIME : {(time.time()-start):.2f}')

    wandb.log({
        f'Train Loss' : train_loss,
        f'One Epoch Time' : max(times),
    }, step = epoch)
    return model, optimizer

def plot(args, model, plot_loader, name, best_loss, epoch):    
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
            if args['model_option'] != 3:
                for index in range(total - args["window_size"]): 
                    t = torch.full((args['batch_size'], 1), index/1000.0, dtype=torch.float32).cuda()
                    t_params = torch.cat((params, t), dim=1)
                    r = model(model_input, t_params).unsqueeze(1)
                    result = torch.cat([result, r], dim=1)
                    model_input = torch.cat([model_input[:, 1:], r], dim=1)
            else:
                result = model(x, params)
                result = torch.cat([x, result.squeeze()], dim=1)
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
    }, step = epoch)

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

    # loss_function = torch.nn.MSELoss()
    loss_function = torch.nn.HuberLoss(reduction='mean', delta=0.5)
    wandb.watch(model, loss_function, log="all", log_freq=10)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, mode='min', factor=0.5, patience = 100)
    scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    scheduler.last_epoch = start_epoch - 1
    if args['verbose']:
        print(f'Training starts, epoch : {scheduler.last_epoch + 1}' )

    for epoch in tqdm(range(start_epoch, maxepoch), desc="Training",leave=True):
        model, optimizer = train(args, model, train_loader, loss_function, optimizer, scheduler, epoch, 'm')
        loss = plot(args, model, plot_loader, name, best_loss, epoch)

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
    torch.set_default_dtype(torch.float32)
    os.environ["WANDB_SILENT"] = "true"

    run = wandb.init(project = "MLCASim")
    args = wandb.config
    os.environ["CUDA_VISIBLE_DEVICES"] = args['device']

    CHECKPOINT_PATH = f'../checkpoint/'
    base_name = args['name']
    window_size = args['window_size']
    num_layers = args['num_layers']
    hidden_size = args['hidden_size']
    hidden_channel = args['hidden_channel']

    name = f'{base_name}_{window_size}_{num_layers}_{hidden_size}_{hidden_channel}'
    run.name = name

    torch.manual_seed(42)
    
    if not os.path.isdir(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    start = time.time()
    if args['verbose']:
        print('SPICE Data Loading Completed')
    
    end_dataset = time.time()
    
    index_to_name = ['DRG', 'DRS', 'DIODE']
    # Dataset generation
    if args['epoch'] > 0:
        # for type in range(args['input_size']):
        for type in [2]:
            train_loader, _ = load_datasets(args, type, 'train')
            process(args, train_loader, train_loader, CHECKPOINT_PATH, f'{name}_{index_to_name[type]}', args['epoch'])

    print("time (Total)   : ", time.time() - end_dataset)

if __name__ == '__main__':    
    parser = ArgumentParser(description='ML Circuit Array Simulation Version 3.0')
    parser.add_argument('--config_dir', required=False, type=str, default = '../configs', help='Config directory')
    parser.add_argument('-c','--config_file', required=False, type=str, default = '../configs', help='Config directory')
    args = parser.parse_args()

    with open(os.path.join(args.config_dir, f'{args.config_file}.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep=config, project=config['project'])
    wandb.agent(sweep_id, function = main)