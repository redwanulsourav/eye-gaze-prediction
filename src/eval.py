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
    


