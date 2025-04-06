import torch
import torchvision
import torchvision.transforms as transforms
import logging
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datetime import datetime
import yaml
import argparse
import pathlib
import os
import shutil
import time
import json
import glob
import pickle

from dataset import GazeDataset
from models import GazeNet2
from utils import Utils



def getConfig(configPath):
    """ Load eval config """
    with open(configPath, 'r') as f:
        evalConfig = yaml.safe_load(f)
    
    """ Verify eval config """
    Utils.Eval.verifyEvalConfig(evalConfig)

    runPath = os.path.join(evalConfig['base_path_model'], str(evalConfig['run_id']))
    trainConfigPath = os.path.join(runPath, 'config.yaml')

    with open(trainConfigPath, 'r') as f:
        trainConfig = yaml.safe_load(f)

    """ Verify train config """
    # Utils.Train.verifyTrainConfig(trainConfig)

    config = Utils.Eval.mergeTrainEvalConfig(evalConfig, trainConfig)
    return config

def main(configPath):
    config = getConfig(configPath)
    dataset = Utils.Common.loadDataset(config)
    dataLoader = DataLoader(dataset, batch_size = 1, shuffle = False)
    model = Utils.Common.loadModel(config)
    lossFn = torch.nn.MSELoss()
    boundaries = [0, config['length'] - 1]

    evalPath, picklesPath, pngsPath = Utils.Eval.createEvalDirs(config['base_path_model'], str(config['run_id']))
    
    model.eval()

    history = {}

    for i, data in enumerate(dataLoader):
        print(f'Evaluating {i}/{len(dataLoader)}')
        feat_x = data['features']
        gaze_x = data['gaze_x']
        gaze_y = data['gaze_y']
        
        assert torch.isnan(feat_x).any() == False
        assert torch.isnan(gaze_x).any() == False
        assert torch.isnan(gaze_y).any() == False, f'{gaze_y} has nan'

        outputs = model(feat_x, gaze_x)

        loss = lossFn(outputs, gaze_y)
        
        history[f'{boundaries[0]}-{boundaries[1]}'] = {}
        history[f'{boundaries[0]}-{boundaries[1]}']['loss'] = loss.item()
        history[f'{boundaries[0]}-{boundaries[1]}']['ground'] = gaze_y[0].tolist()
        history[f'{boundaries[0]}-{boundaries[1]}']['prediction'] = outputs[0].tolist()

        frameLosses = []

        """ Get loss for individual frames """
        for k in range(config['stride']):
            groundTruth = gaze_y[0, k].unsqueeze(0)
            prediction = outputs[0, k].unsqueeze(0)

            assert groundTruth.shape == (1, 2)
            assert prediction.shape == (1, 2)

            frameLoss = lossFn(groundTruth, prediction)
            frameLosses.append(frameLoss)
       
        x = list(range(boundaries[1] + 1, boundaries[1] + config['stride'] + 1))
        y = frameLosses

        """ Create plot """
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title(f'{boundaries[0]}-{boundaries[1]} -> {boundaries[1] + 1}-{boundaries[1] + config["stride"]}')
        ax.set_xlabel('Frame Idx')
        ax.set_ylabel('Normalized Error')
        
        """ Save pickle """
        with open(os.path.join(picklesPath,f'{boundaries[1] + 1}-{boundaries[1] + config["stride"]}.pkl'), 'wb') as f:
            pickle.dump(fig, f)

        """ Save png """
        fig.savefig(os.path.join(pngsPath, f'{boundaries[1] + 1}-{boundaries[1] + config["stride"]}.png'))

        plt.clf()
        plt.close(fig)

        boundaries[0] += 1
        boundaries[1] += 1

    with open(os.path.join(evalPath, 'history.yml'), 'w') as f:
        yaml.dump(history, f, default_flow_style = False)
    
    with open(os.path.join(evalPath, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style = False)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--cfg', required = True)
    ap = ap.parse_args()

    main(ap.cfg)