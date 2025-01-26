"""
    Run the Itti Koch model against a video or picture
"""

import cv2
import numpy as np
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from itti_koch import Itti_Koch_Model
from dataset.dataset_interface import DatasetInterface

model = Itti_Koch_Model()

def getImageLoss(
    img: np.ndarray, 
    groundTruth: tuple, 
    model: Itti_Koch_Model, 
    normalized: bool, 
    logger):
    """
        Get the error between ground truth and predicted fixation point for a single image.
    """

    sMap = model.get_saliency_map(img)
    assert sMap.shape[0] == 1, f'Expected sMap to have a single color channel, found {s_map.shape[0]}'
    
    logger.info(f'[getImageFixation] sMap has single channel')
    
    fixationPt = np.argmax(s_map)
    
    assert type(fixationPt) == int

    # Get 2D index instead of a flat index.
    fixationPt = np.unravel_index(fixationPt, sMap.shape)
    
    assert type(fixationPt) == tuple, 'unravelling index didn\'t return a 2d index'
    assert len(fixationPt) == 2, f'expected the index to have 2 values, but got {len(fixation_pt)}'
    assert sMap.max() == sMap[fixationPt[0], fixationPt[1]], 'sMap max do not match with argmax'
    
    fixationPt = list(fixationPt)
    groundTruth = list(groundTruth)

    if normalized:
        fixationPt[0] /= sMap.shape[0]
        fixationPt[1] /= sMap.shape[1]

        groundTruth[0] /= sMap.shape[0]
        groundTruth[1] /= sMap.shape[1]
    
    return np.hypot(np.array(fixationPt), np.array(groundTruth))

def getVideoLoss(
    frames: list, 
    groundTruths: list, 
    model: Itti_Koch_Model, 
    normalized: bool, 
    logger):
    """
        Get the average error over all frames in the video
    """

    errorSum = 0
    print(frames)
    for frame, groundTruth in zip(frames, groundTruths):
        errorSum += getImageLoss(frame, groundTruth, model, normalized, logger)

    return errorSum / len(frames)

def evalVideo(
    videoId: int,
    personId: int,
    datasetInterface: DatasetInterface,
    model: Itti_Koch_Model,
    normalized: bool,
    logger
):
    videoFrames = datasetInterface.getAllFrames(videoIdx = videoId)
    gazeLocations = datasetInterface.getAllGazeOfSingleViewer(
                    videoIdx = videoId,
                    viewerIdx = personId)
    
    return getVideoLoss(videoFrames, gazeLocations, model, normalized, logger)

if __name__ == '__main__':
    model = Itti_Koch_Model()
    datasetInterface = DatasetInterface('/data/rsourave/datasets/Coutrot/')
    logger = logging.getLogger(__name__)

    loss = evalVideo(
        0,
        0,
        datasetInterface,
        model,
        True,
        logger
    )
