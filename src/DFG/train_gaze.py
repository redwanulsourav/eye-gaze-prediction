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
from torch.utils.data import DataLoader

from dfg_dataset import DFG_GTEA_Dataset
from models import FrameGenerator, TemporalSaliencyPredictor, Discriminator

def trainGazePredictor(generator, gazePredictor, fixationMap, video):
    gOut = generator['model'](video[:, :, 0, :, :]) # (batch_size, 3, 32, 64, 64)
    sMap = gazePredictor['model'](gOut)

    criterionKLDiv = torch.nn.KLDivLoss(reduction = 'batchmean', log_target = True)

    loss = criterionKLDiv(sMap.log(), F.log_softmax(fixationMap))
    loss.backward()

    gazePredictor['optim'].step()


def train_one_epoch(epoch, trainLoader, generator, gazePredictor, dev, output_path, logger = None):
    
    data_len = len(trainLoader)

    H = {
        'losses': [],
        'avg_loss': 0
    }

    for i, data in enumerate(trainLoader):
        video = data['frames'].to(dev)
        fixationMap = data['temporal_fixation_map'].to(dev)
        
        loss = trainGazePredictor(generator, gazePredictor, fixationMap, video)

        H['avg_loss'] += loss
        H['losses'].append(lD)
        
        print(f'[{i} / {data_len}]: loss: {loss}')
        
        if logger is not None:
            logger.info(f'Epoch {epoch}: {i} / {data_len} loss: {loss}')

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
    os.system(f'cp models.py {output_path}/{run_id}/src/models.py')
    os.system(f'cp train.py {output_path}/{run_id}/src/train.py')   # TODO: Make this dynamic
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
                               length =         config['length'],
                               videos=          config['videos'] if 'videos' in config else [0],
                               rootPath =       config['base_path'])
    
    trainLoader = DataLoader(trainData, batch_size = config['batch_size'], shuffle = config['shuffle'] if 'shuffle' in config else True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = {}
    generator['model'] = FrameGenerator().load_state_dict(torch.load(config['generator_saved_model'])).to(device).eval()
    # generator['optim'] = torch.optim.Adam(generator['model'].parameters(), lr = config['lr'], betas = (config['momentum'], 0.999))

    gazePredictor = {}
    gazePredictor['model'] = TemporalSaliencyPredictor().to(device).eval()
    gazePredictor['optim'] = torch.optim.Adam(gazePredictor['model'].parameters(), lr = config['lr'], betas = (config['momentum'], 0.999))

    epochs = config['epochs'] if 'epochs' in config else 2
        
    for i in range(0, epochs):
        avg_loss = train_one_epoch(i, trainLoader, generator, gazePredictor, device, config['output_dir'], logger)

        print(f'[{i}/{epochs}] Average Loss: Discriminator: {avg_loss_discriminator}, Generator: {avg_loss_generator}')
        logger.info(f'[{i}/{epochs}] Average Loss: Discriminator: {avg_loss_discriminator}, Generator: {avg_loss_generator}')

        
    
