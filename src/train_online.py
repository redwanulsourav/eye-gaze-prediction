import torch
import torchvision
import torchvision.transforms as transforms
import logging

from torch.utils.data import DataLoader

from datetime import datetime
import yaml
import argparse
import pathlib
import os
import shutil
import time
import json
import glob

from dataset import GazeDataset
from models import GazeNet2

logger = logging.getLogger(__name__)

def train_one_epoch(train_loader, optim, loss_fn, model, dev):
    loss_sum = 0
    
    data_len = len(train_loader)

    H = {
        "running_loss": []
    }
    gt_xs = []
    gt_ys = []
    pred_xs = []
    pred_ys = []

    for i, data in enumerate(train_loader):
        feat_x = data['features'].to(dev)
        gaze_x = data['gaze_x'].to(dev)
        gaze_y = data['gaze_y'].to(dev)
        
        optim.zero_grad()
        outputs = model(feat_x, gaze_x)
        
        gt_xs.append(gaze_y[0].tolist()[0][0])
        gt_ys.append(gaze_y[0].tolist()[0][1])
        pred_xs.append(outputs[0].tolist()[0][0])
        pred_ys.append(outputs[0].tolist()[0][1])
        
        loss = loss_fn(outputs, gaze_y)
        loss.backward()

        optim.step()

        loss_sum += loss.item()
        H["running_loss"].append(loss.item())

        print(f'{i} / {data_len} Loss: {loss.item()}')
        logger.info(f'{i} / {data_len} Loss: {loss.item()}')

    H['gt_xs'] = gt_xs
    H['gt_ys'] = gt_ys
    H['pred_xs'] = pred_xs
    H['pred_ys'] = pred_ys
    return loss_sum / data_len, H['running_loss'], H['gt_xs'], H['gt_ys'], H['pred_xs'], H['pred_ys']


def prepare_dirs(output_path: str):
    os.makedirs(f'{output_path}/', exist_ok = True)
    run_id = len(os.listdir(f'{output_path}/'))
    
    os.makedirs(f'{output_path}/{run_id}/')
    os.makedirs(f'{output_path}/{run_id}/history')
    os.makedirs(f'{output_path}/{run_id}/weights')
    os.makedirs(f'{output_path}/{run_id}/epochs')
    os.makedirs(f'{output_path}/{run_id}/src')
    os.system(f'cp models.py {output_path}/{run_id}/src/models.py')
    os.system(f'cp train_online.py {output_path}/{run_id}/src/train_online.py')   # TODO: Make this dynamic
    os.system(f'cp {ap.cfg} {output_path}/{run_id}/config.yaml')
    
    return run_id


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--cfg', help = 'Train config [.yaml]', required = True)
    ap.add_argument('-o', '--output', help = 'Path to store output', default = 'runs')
    ap = ap.parse_args()

    with open(ap.cfg) as f:
        config = yaml.safe_load(f)

    run_id = prepare_dirs(ap.output)

    logging.basicConfig(level = logging.DEBUG, filename = f'{ap.output}/{run_id}/log')
    logger = logging.getLogger(__name__)
    
    
    gaze_dataset = GazeDataset(stride=          config['stride'], 
                               length =         config['length'],
                               videos=          config['videos'] if 'videos' in config else [0],
                               basePath =       config['base_path'],
                               viewers =        config['viewers'] if 'viewers' in config else [0])
    
    training_loader = DataLoader(gaze_dataset,  config['batch_size'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GazeNet2(
                    length =        config['length'], 
                    model_type =    config['model_type'],
                    stride =        config['stride']).to(device)
    
    if 'weights' in config:
        if os.path.exists(run_configuration['weights']['weights_path']) == True:
            model.load_state_dict(torch.load(run_configuration['weights']['weights_path']))
            print('Pretrained model loaded')
        else:
            print('Pretrained weights not found')
        
    model.train()

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adagrad(model.parameters(), lr=config['lr'])

    print(f'Run ID: {run_id}')
    
    avg_loss, running_losses, gt_xs, gt_ys, pred_xs, pred_ys = \
                    train_one_epoch(training_loader, optim, loss_fn, model, device)
    H = {
        "avg_loss": avg_loss,
        "running_losses": running_losses,
        "gt_xs": train_xs, 
        "gt_ys": train_ys,
        "pred_xs": pred_xs,
        "pred_ys": pred_ys
    }    
        
    with open(f'{ap.output}/{run_id}/history.json', 'w') as f:
        json.dump(H, f)

    print(f'Average Loss: {avg_loss}')
        
    torch.save(model.state_dict(), f'{ap.output}/{run_id}/model_state.pt')
    torch.save(optim.state_dict(), f'{ap.output}/{run_id}/optim_state.pt')
