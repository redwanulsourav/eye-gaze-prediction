import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")


import torch
import torchvision
import logging
from datetime import datetime
import yaml
import argparse
import pathlib
import shutil
import time
import json

import torch.nn.functional as F

from torch.utils.data import DataLoader

from dfg_dataset import DFG_GTEA_Dataset
from models import FrameGenerator, TemporalSaliencyPredictor, Discriminator

def trainGazePredictor(generator, g_predictor, gt_map, x_video):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(f'gt_map.shape {gt_map.shape}')
    g_predictor['optim'].zero_grad()

    gen_out = generator['model'](x_video) # (batch_size, 3, 32, 64, 64)
    p_map = g_predictor['model'](gen_out) # Predicted map
    print(f'p_map.shape {p_map.shape}')
    loss_KL_div = torch.nn.KLDivLoss(reduction = 'batchmean', log_target = True)

    B, C, T, H, W = p_map.shape

    # Prepare the output and gt for loss calculation.
    p_map = p_map.squeeze()
    gt_map = gt_map.squeeze()

    p_map = p_map.view(B, T, -1)
    gt_map = gt_map.view(B, T, -1)

    mask = (gt_map != 0).to(dev)
    gt_map = torch.where(mask, gt_map, torch.tensor(-1e9).to(dev))
    
    gt_map = F.log_softmax(gt_map, dim = -1)
    p_map = F.log_softmax(p_map, dim = -1)

    loss = loss_KL_div(p_map, gt_map)
    loss.backward()

    g_predictor['optim'].step()

    return loss.item()


def train_one_epoch(epoch, trainLoader, generator, gazePredictor, dev, output_path, logger = None):
    data_len = len(trainLoader)

    H = {
        'losses': [],
        'avg_loss': 0
    }

    for i, data in enumerate(trainLoader):
        input_frame = data['input_frame'].to(dev)
        tgt_map = data['tgt_gaze'].to(dev)
        
        loss = trainGazePredictor(generator, gazePredictor, tgt_map, input_frame)

        H['avg_loss'] += loss
        H['losses'].append(loss)
        
        print(f'[{i} / {data_len}]: loss: {loss}')
        
        if logger is not None:
            logger.info(f'Epoch {epoch}: {i} / {data_len} loss: {loss}')

        if i % 50 == 0:
            torch.save(gazePredictor['model'].state_dict(), f'{output_path}/{run_id}/tmp/gazePredictor_model_{epoch}_{i}.pt')
            torch.save(gazePredictor['optim'].state_dict(), f'{output_path}/{run_id}/tmp/gazePredictor_optim_{epoch}_{i}.pt')

    H['avg_loss'] /= data_len
        
    os.makedirs(f'{output_path}/{run_id}/epochs/{epoch}')
    with open(f'{output_path}/{run_id}/epochs/{epoch}/history.json', 'w') as f:
        json.dump(H, f)

    """ Save Temporal Gaze Predictor state """
    torch.save(gazePredictor['model'].state_dict(), f'{output_path}/{run_id}/epochs/{epoch}/model_state.pt')
    torch.save(gazePredictor['optim'].state_dict(), f'{output_path}/{run_id}/epochs/{epoch}/optim_state.pt')
    
    return H['avg_loss']


def prepare_dirs(output_path: str, cfg_path):
    os.makedirs(f'{output_path}/', exist_ok = True)
    run_id = len(os.listdir(f'{output_path}/'))
    
    os.makedirs(f'{output_path}/{run_id}/')
    os.makedirs(f'{output_path}/{run_id}/history')
    os.makedirs(f'{output_path}/{run_id}/weights')
    os.makedirs(f'{output_path}/{run_id}/epochs')
    os.makedirs(f'{output_path}/{run_id}/src')
    os.makedirs(f'{output_path}/{run_id}/tmp')

    os.system(f'cp models.py {output_path}/{run_id}/src/models.py')
    os.system(f'cp {os.path.abspath(__file__)} {output_path}/{run_id}/src/train.py')   # TODO: Make this dynamic
    os.system(f'cp {cfg_path} {output_path}/{run_id}/config.yaml')
    
    return run_id


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--cfg', help = 'Train config [.yaml]', required = True)
    
    ap = ap.parse_args()

    with open(ap.cfg) as f:
        config = yaml.safe_load(f)

    run_id = prepare_dirs(config['output_dir'], ap.cfg)

    logging.basicConfig(level = logging.DEBUG, filename = f'{config["output_dir"]}/{run_id}/log')
    logger = logging.getLogger(__name__)
    
    
    trainData = DFG_GTEA_Dataset(
                               videos =          config['videos'] if 'videos' in config else [0],
                               data_root =       config['base_path'],
                               t_sample = 20)
    
    trainLoader = DataLoader(trainData, batch_size = config['batch_size'], shuffle = config['shuffle'] if 'shuffle' in config else True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = {}
    generator['model'] = FrameGenerator().to(device).eval()
    generator['model'].load_state_dict(torch.load(config['generator_saved_model']))
    generator['model'].eval()
    # generator['optim'] = torch.optim.Adam(generator['model'].parameters(), lr = config['lr'], betas = (config['momentum'], 0.999))

    gazePredictor = {}
    gazePredictor['model'] = TemporalSaliencyPredictor().to(device).train()
    gazePredictor['optim'] = torch.optim.Adam(gazePredictor['model'].parameters(), lr = config['lr'], betas = (config['momentum'], 0.999))

    epochs = config['epochs'] if 'epochs' in config else 2
        
    for i in range(0, epochs):
        avg_loss = train_one_epoch(i, trainLoader, generator, gazePredictor, device, config['output_dir'], logger)

        print(f'[{i}/{epochs}] Average Loss: Discriminator: {avg_loss_discriminator}, Generator: {avg_loss_generator}')
        logger.info(f'[{i}/{epochs}] Average Loss: Discriminator: {avg_loss_discriminator}, Generator: {avg_loss_generator}')

        
    
