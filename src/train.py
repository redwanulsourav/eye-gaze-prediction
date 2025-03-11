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

torch.manual_seed(0)

def train_one_epoch(epoch, train_loader, optim, loss_fn, model, dev, output_path, logger = None):
    startTime = time.time()
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
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
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
        end_event.record()
        torch.cuda.synchronize()
        print(f'{i} / {data_len} Loss: {loss.item()}, took {start_event.elapsed_time(end_event)} seconds')
        if logger is not None:
            logger.info(f'Epoch {epoch}: {i} / {data_len} Loss: {loss.item()}, took {start_event.elapsed_time(end_event)} seconds')

    H = {
        "avg_loss": loss_sum / data_len,
        "running_losses": running_losses,
        "gt_xs": gt_xs, 
        "gt_ys": gt_ys,
        "pred_xs": pred_xs,
        "pred_ys": pred_ys
    }    
    
    endTime = time.time()
    seconds = endTime - startTime
    os.makedirs(f'{output_path}/{run_id}/epochs/{epoch}')
    with open(f'{output_path}/{run_id}/epochs/{epoch}/history.json', 'w') as f:
        json.dump(H, f)

    torch.save(model.state_dict(), f'{output_path}/{run_id}/epochs/{epoch}/model_state.pt')
    torch.save(optim.state_dict(), f'{output_path}/{run_id}/epochs/{epoch}/optim_state.pt')
    
    return loss_sum / data_len, H['running_loss'], H['gt_xs'], H['gt_ys'], H['pred_xs'], H['pred_ys'], seconds


def prepare_dirs(output_path: str, cfg_path):
    os.makedirs(f'{output_path}/', exist_ok = True)
    run_id = len(os.listdir(f'{output_path}/'))
    
    os.makedirs(f'{output_path}/{run_id}/')
    os.makedirs(f'{output_path}/{run_id}/history')
    os.makedirs(f'{output_path}/{run_id}/weights')
    os.makedirs(f'{output_path}/{run_id}/epochs')
    os.makedirs(f'{output_path}/{run_id}/src')
    os.system(f'cp models.py {output_path}/{run_id}/src/models.py')
    os.system(f'cp train_online.py {output_path}/{run_id}/src/train.py')   # TODO: Make this dynamic
    os.system(f'cp {cfg_path} {output_path}/{run_id}/config.yaml')
    
    return run_id


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--cfg', help = 'Train config [.yaml]', required = True)
    # ap.add_argument('-o', '--output', help = 'Path to store output', default = 'runs')
    ap = ap.parse_args()

    with open(ap.cfg) as f:
        config = yaml.safe_load(f)

    run_id = prepare_dirs(config['output_dir'], ap.cfg)

    logging.basicConfig(level = logging.DEBUG, filename = f'{config["output_dir"]}/{run_id}/log')
    logger = logging.getLogger(__name__)
    
    
    gaze_dataset = GazeDataset(stride=          config['stride'], 
                               length =         config['length'],
                               videos=          config['videos'] if 'videos' in config else [0],
                               rootPath =       config['base_path'],
                               viewers =        config['viewers'] if 'viewers' in config else [0])
    
    # print(f"Batch Size: {config['batch_size']}")
    train_loader = DataLoader(
            gaze_dataset,  
            batch_size = config['batch_size'], 
            shuffle = config['shuffle'] if 'shuffle' in config else True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GazeNet2(
                    length =        config['length'], 
                    model_type =    config['model_type'],
                    stride =        config['stride']).to(device)
    
    epochs = config['epochs'] if 'epochs' in config else 2
    # if 'weights' in config:
    #     if os.path.exists(run_configuration['weights']['weights_path']) == True:
    #         model.load_state_dict(torch.load(run_configuration['weights']['weights_path']))
    #         print('Pretrained model loaded')
    #     else:
    #         print('Pretrained weights not found')
        
    model.train()

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adagrad(model.parameters(), lr=config['lr'])

    print(f'Run ID: {run_id}')
    
    for i in range(0, epochs):
        avg_loss, running_losses, gt_xs, gt_ys, pred_xs, pred_ys, seconds = \
                    train_one_epoch(i, train_loader, optim, loss_fn, model, device, logger)

        print(f'[{i}/{epochs}] Average Loss: {avg_loss}, took {seconds} seconds')
        logger.info(f'[{i}/{epochs}] Average Loss: {avg_loss} took {seconds} seconds')

        
    
