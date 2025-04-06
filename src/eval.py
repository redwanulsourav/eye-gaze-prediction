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
from utils import Utils



def getConfig(configPath):
    """ Load eval config """
    with open(configPath, 'r') as f:
        evalConfig = yaml.safe_load(f)
    
    """ Verify eval config """
    Utils.Eval.verifyEvalConfig(evalConfig)

    runPath = os.path.join(evalConfig['base_path_models'], evalConfig['run_id'])
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

    model.eval()

    history = {}

    for i, data in enumerate(dataLoader):
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

        """ Create plot """
        fig, ax = plt.subplots()
        ax.plot(x, y, label='y = x^2')
        ax.set_title("Sample Plot")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.legend()
        
        """ Save pickle """
        with open("sample_plot.pkl", "wb") as f:
            pickle.dump(fig, f)

        """ Save png """
        fig.savefig("sample_plot.png")

        plt.clf()
        plt.close(fig)


