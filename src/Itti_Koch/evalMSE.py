"""
    Run the Itti Koch model against a video or picture
"""

import cv2
import numpy as np
import os
import sys
import logging
import math
import json
from sklearn.metrics import roc_auc_score
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from itti_koch import IttiKochModel
from dataset_interface.dataset_interface import DatasetInterface

model = IttiKochModel()

def getImageLoss(img: np.ndarray, groundTruth: tuple, model: IttiKochModel, normalized, logger):
    """
        Get the error between ground truth and predicted fixation point for a single image.
    """
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (1024, 1024))
    sMap = model.saliencyMap(img)
    # assert sMap.shape[2] == 1, f'Expected sMap to have a single color channel, found {sMap.shape[0]}'
    
    logger.info(f'[getImageFixation] sMap has single channel')
    
    fixationPt = np.argmax(sMap)
    
    # assert type(fixationPt) == np.int64, f'Expected fixationPt to be of type int, found {type(fixationPt)}'

    # Get 2D index instead of a flat index.
    fixationPt = np.unravel_index(fixationPt, sMap.shape)
    
    # assert type(fixationPt) == tuple, 'unravelling index didn\'t return a 2d index'
    # assert len(fixationPt) == 2, f'expected the index to have 2 values, but got {len(fixation_pt)}'
    # assert sMap.max() == sMap[fixationPt[0], fixationPt[1]], 'sMap max do not match with argmax'
    
    fixationPt = list(fixationPt)
    groundTruth = list(groundTruth)
    
    # print(f'fixationPt: {fixationPt[0]}, {fixationPt[1]}')
    # print(f'groundTruth: {groundTruth[0]}, {groundTruth[1]}')

    if normalized:
        fixationPt[0] /= sMap.shape[0]
        fixationPt[1] /= sMap.shape[1]

        # groundTruth[0] /= img.shape[0]
        # groundTruth[1] /= img.shape[1]
    
    # print(f'fixationPt: {fixationPt[0]}, {fixationPt[1]}')
    # print(f'groundTruth: {groundTruth[0]}, {groundTruth[1]}')
    return np.hypot(fixationPt[0] - groundTruth[0], fixationPt[1] - groundTruth[1])


def compute_auc(saliency_map, fixation_map):
    # Flatten the maps
    saliency_values = saliency_map.flatten()
    fixation_values = fixation_map.flatten()
    
    # Binary ground truth: 1 for fixations, 0 otherwise
    labels = (fixation_values > 0).astype(int)
    
    return roc_auc_score(labels, saliency_values)

    # print(img.shape)
    # assert sMap.shape[2] == 1, f'Expected sMap to have a single color channel, found {sMap.shape[0]}'
    # logger.info(f'[getImageFixation] sMap has single channel')
    
    # fixationPt = np.argmax(sMap)
    
    # assert type(fixationPt) == np.int64, f'Expected fixationPt to be of type int, found {type(fixationPt)}'

    # # Get 2D index instead of a flat index.
    # fixationPt = np.unravel_index(fixationPt, sMap.shape)
    
    # assert type(fixationPt) == tuple, 'unravelling index didn\'t return a 2d index'
    # assert len(fixationPt) == 2, f'expected the index to have 2 values, but got {len(fixation_pt)}'
    # assert sMap.max() == sMap[fixationPt[0], fixationPt[1]], 'sMap max do not match with argmax'
    
    # fixationPt = list(fixationPt)
    # groundTruth = list(groundTruth)
    
    # # print(f'fixationPt: {fixationPt[0]}, {fixationPt[1]}')
    # # print(f'groundTruth: {groundTruth[0]}, {groundTruth[1]}')

    # if normalized:
    #     fixationPt[0] /= sMap.shape[0]
    #     fixationPt[1] /= sMap.shape[1]

    #     groundTruth[0] /= img.shape[0]
    #     groundTruth[1] /= img.shape[1]
    
    # # print(f'fixationPt: {fixationPt[0]}, {fixationPt[1]}')
    # # print(f'groundTruth: {groundTruth[0]}, {groundTruth[1]}')
    # return np.hypot(fixationPt[0] - groundTruth[0], fixationPt[1] - groundTruth[1])

def getVideoLoss(
    frames: list, 
    groundTruths: list, 
    model: IttiKochModel, 
    normalized: bool, 
    logger):
    """
        Get the average error over all frames in the video
    """

    errorSum = 0
    n = len(frames)
    result = {}
    result['frames'] = {}
    for i, (frame, groundTruth) in enumerate(zip(frames, groundTruths)):
        loss =  getImageLoss(frame, groundTruth, model, normalized, logger)
        print(f'Processing frame {i + 1}/{n}... loss: {loss}')
        result['frames'][i] = loss
        errorSum += loss

    result['avg'] = errorSum / len(frames)
    return result

def evalVideo(videoId: int, personId: int, datasetInterface: DatasetInterface, model: IttiKochModel, normalized: bool, logger):
    videoFrames = datasetInterface.getAllFrames(videoIdx = videoId)
    gazeLocations = datasetInterface.getAllGazeOfSingleViewer(
                    videoIdx = videoId,
                    viewerIdx = personId)
    
    return getVideoLoss(videoFrames, gazeLocations, model, normalized, logger)

def prepareOutputDirs(outputPath):
    try:
        os.mkdir(f'{outputPath}/saved_models/Itti_Koch/')
    except:
        print('Path already exists')
    
    
    try:
        os.mkdir(f'{outputPath}/saved_models/Itti_Koch/EGTEA')
    except:
        print('Path already exists')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output_path', required = True)
    ap.add_argument('-r', '--dataset_root', required = True)
    ap = ap.parse_args()

    model = IttiKochModel()
    datasetInterface = DatasetInterface(ap.dataset_root)
    
    logger = logging.getLogger(__name__)  
    result = evalVideo(9, 0, datasetInterface, model, True, logger)
    
    with open(f'{ap.output_path}/{9}-{0}.json', 'w')  as f:   # videoId-personId
        f.write(json.dumps(result))


