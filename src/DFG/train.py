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

def trainDiscriminator(discriminator, generator, video, tgt_video, dev):
    discriminator['optim'].zero_grad()
    # generator['optim'].zero_grad()
    gOut = generator['model'](video)   # (batch_size, 3, 32, 64, 64)
    tgt_video = tgt_video.permute(0, 2, 1, 3, 4)
    dOut0 = discriminator['model'](tgt_video) # (batch_size, 2)
    dOut1 = discriminator['model'](gOut)

    realLabel = torch.Tensor([1, 0]).expand(dOut0.shape[0], -1).to(dev)
    fakeLabel = torch.Tensor([0, 1]).expand(dOut0.shape[0], -1).to(dev)

    criterion = torch.nn.BCELoss()
    loss = criterion(dOut0, realLabel) + criterion(dOut1, fakeLabel)
    
    loss.backward()

    discriminator['optim'].step()
    return loss.item()

def trainGenerator(discriminator, generator, video, dev):
    generator['optim'].zero_grad()

    gOut = generator['model'](video) # (batch_size, 3, 32, 64, 64)
    dOut = discriminator['model'](gOut)

    realLabel = torch.Tensor([1, 0]).expand(dOut.shape[0], -1).to(dev)
    
    criterionBCE = torch.nn.BCELoss()
    criterionAbs = torch.nn.L1Loss()

    loss = criterionBCE(dOut, realLabel) + 0.1 * criterionAbs(video[:, :, :, :], gOut[:, :, 0, :, :])
    loss.backward()

    generator['optim'].step()

    return loss.item()

def trainGazePredictor(generator, gazePredictor, fixationMap, video):
    gOut = generator['model'](video) # (batch_size, 3, 32, 64, 64)
    sMap = gazePredictor['model'](gOut)

    criterionKLDiv = torch.nn.KLDivLoss(reduction = 'batchmean', log_target = True)

    loss = criterionKLDiv(sMap.log(), F.log_softmax(fixationMap))
    loss.backward()

    gazePredictor['optim'].step()


def train_one_epoch(epoch, trainLoader, generator, discriminator, dev, output_path, logger = None):
    data_len = len(trainLoader)

    H = {
        'discriminator_losses': [],
        'generator_losses': [],
        'avg_loss_discriminator': 0,
        'avg_loss_generator': 0
    }

    for i, data in enumerate(trainLoader):
        video = data['input_frame'].to(dev)
        tgt_video = data['tgt_frames'].to(dev)

        lD = trainDiscriminator(discriminator, generator, video, tgt_video, dev)
        lG = trainGenerator(discriminator, generator, video, dev)

        H['avg_loss_discriminator'] += lD
        H['avg_loss_generator'] += lG
        
        H['discriminator_losses'].append(lD)
        H['generator_losses'].append(lG)

        print(f'[{i} / {data_len}]: losses: discriminator: {lD}, generator: {lG}')
        
        if logger is not None:
            logger.info(f'Epoch {epoch}: {i} / {data_len} losses: discriminator: {lD}, generator: {lG}')
        
        if i % 50 == 0:
            """ Save discriminator state """
            torch.save(discriminator['model'].state_dict(), f'{output_path}/{run_id}/tmp/discriminator_model_state_{epoch}_{i}.pt')
            torch.save(discriminator['optim'].state_dict(), f'{output_path}/{run_id}/tmp/discriminator_optim_state_{epoch}_{i}.pt')
            
            """ Save Generator state """
            torch.save(generator['model'].state_dict(), f'{output_path}/{run_id}/tmp/generator_model_state_{epoch}_{i}.pt')
            torch.save(generator['optim'].state_dict(), f'{output_path}/{run_id}/tmp/generator_optim_state_{epoch}_{i}.pt')
        
       
    H['avg_loss_discriminator'] /= data_len
    H['avg_loss_generator'] /= data_len
        
    os.makedirs(f'{output_path}/{run_id}/epochs/{epoch}')
    with open(f'{output_path}/{run_id}/epochs/{epoch}/history.json', 'w') as f:
        json.dump(H, f)

    """ Save discriminator state """
    torch.save(discriminator['model'].state_dict(), f'{output_path}/{run_id}/epochs/{epoch}/discriminator_model_state.pt')
    torch.save(discriminator['optim'].state_dict(), f'{output_path}/{run_id}/epochs/{epoch}/discriminator_optim_state.pt')
    
    """ Save Generator state """
    torch.save(generator['model'].state_dict(), f'{output_path}/{run_id}/epochs/{epoch}/generator_model_state.pt')
    torch.save(generator['optim'].state_dict(), f'{output_path}/{run_id}/epochs/{epoch}/generator_optim_state.pt')
    
    return H['avg_loss_discriminator'], H['avg_loss_generator']


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
    
    
    train_data = DFG_GTEA_Dataset(
                               videos=          config['videos'] if 'videos' in config else [0],
                               data_root =       config['base_path'],
                               t_sample = 20)
    
    train_loader = DataLoader(train_data, batch_size = config['batch_size'], shuffle = config['shuffle'] if 'shuffle' in config else True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    discriminator = {}
    discriminator['model'] = Discriminator().to(device).train()
    discriminator['optim'] = torch.optim.Adam(discriminator['model'].parameters(), lr = config['lr'], betas = (config['momentum'], 0.999))
    
    if 'pretrained_discriminator_model' in list(config.keys()):
        discriminator['model'].load_state_dict(torch.load(config['pretrained_discriminator_model']))

    if 'pretrained_discriminator_optim' in list(config.keys()):
        discriminator['optim'].load_state_dict(torch.load(config['pretrained_discriminator_optim']))


    generator = {}
    generator['model'] = FrameGenerator().to(device).train()
    generator['optim'] = torch.optim.Adam(generator['model'].parameters(), lr = config['lr'], betas = (config['momentum'], 0.999))
    
    if 'pretrained_generator_model' in list(config.keys()):
        generator['model'].load_state_dict(torch.load(config['pretrained_generator_model']))
    
    if 'pretrained_generator_optim' in list(config.keys()):
        generator['optim'].load_state_dict(torch.load(config['pretrained_generator_optim']))

    epochs = config['epochs'] if 'epochs' in config else 2
       
    if 'starting_epoch' in list(config.keys()):
        starting_epoch = config['starting_epoch']
    else:
        starting_epoch = 0

    for i in range(starting_epoch, epochs):
        avg_loss_discriminator, avg_loss_generator = \
                    train_one_epoch(i, train_loader, generator, discriminator, device, config['output_dir'], logger)

        print(f'[{i}/{epochs}] Average Loss: Discriminator: {avg_loss_discriminator}, Generator: {avg_loss_generator}')
        logger.info(f'[{i}/{epochs}] Average Loss: Discriminator: {avg_loss_discriminator}, Generator: {avg_loss_generator}')

        
    
