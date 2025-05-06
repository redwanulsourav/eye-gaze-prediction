"""
    Run the Itti Koch model against a video or picture
"""

import cv2
import numpy as np
import os
import sys
import logging
import math
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from itti_koch import Itti_Koch_Model
from dataset_interface.dataset_interface import DatasetInterface

model = Itti_Koch_Model()

def getImageLoss(
    img: np.ndarray, 
    groundTruth: tuple, 
    model: Itti_Koch_Model, 
    normalized, 
    logger):
    """
        Get the error between ground truth and predicted fixation point for a single image.
    """

    sMap = model.saliencyMap(img)
    # assert sMap.shape[2] == 1, f'Expected sMap to have a single color channel, found {sMap.shape[0]}'
    
    logger.info(f'[getImageFixation] sMap has single channel')
    
    fixationPt = np.argmax(sMap)
    
    assert type(fixationPt) == np.int64, f'Expected fixationPt to be of type int, found {type(fixationPt)}'

    # Get 2D index instead of a flat index.
    fixationPt = np.unravel_index(fixationPt, sMap.shape)
    
    assert type(fixationPt) == tuple, 'unravelling index didn\'t return a 2d index'
    assert len(fixationPt) == 2, f'expected the index to have 2 values, but got {len(fixation_pt)}'
    assert sMap.max() == sMap[fixationPt[0], fixationPt[1]], 'sMap max do not match with argmax'
    
    fixationPt = list(fixationPt)
    groundTruth = list(groundTruth)
    
    # print(f'fixationPt: {fixationPt[0]}, {fixationPt[1]}')
    # print(f'groundTruth: {groundTruth[0]}, {groundTruth[1]}')

    if normalized:
        fixationPt[0] /= sMap.shape[0]
        fixationPt[1] /= sMap.shape[1]

        groundTruth[0] /= img.shape[0]
        groundTruth[1] /= img.shape[1]
    
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

def getImageLossAuc(
    img: np.ndarray, 
    dsInterface,
    groundTruths: list, 
    model: Itti_Koch_Model, 
    normalized, 
    logger):
    """
        Get the error between ground truth and predicted fixation point for a single image.
    """

    sMap = model.saliencyMap(img)
    sMap = cv2.resize(sMap, img.shape[:-1])
    viewerCnt = dsInterface.getViewerCount(0)
    print(viewerCnt)
    grndTruth = np.zeros_like(sMap)
    print(sMap.shape)
    print(grndTruth.shape)
    truePts = []
    for i in range(viewerCnt):
        truePt = dsInterface.getGazeLocation(0, 0, i)
        print(truePt)
        grndTruth[round(truePt[1]), round(truePt[0])] = 255
    
    print(compute_auc(sMap, grndTruth))
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

# def getVideoLoss(
#     frames: list, 
#     groundTruths: list, 
#     model: Itti_Koch_Model, 
#     normalized: bool, 
#     logger):
#     """
#         Get the average error over all frames in the video
#     """

#     # First see if any NaNs are there
#     for i in range(0, len(groundTruths)):
#         if math.isnan(groundTruths[i][0]) or math.isnan(groundTruths[i][1]):
#             previousExists = None 
#             if i > 0: 
#                 previousExists = True 
#             else: 
#                 previousExists = False

#             nextExists = None
#             if i < len(groundTruths) - 1: 
#                 nextExists = True 
#             else: 
#                 nextExists = False
            
#             if previousExists == False and nextExists == False:
#                 groundTruths[i][0] = 0
#                 groundTruths[i][1] = 0
#             elif previousExists == False and nextExists == True:
#                 groundTruths[i] = groundTruths[iruths[i - 1][0] + groundTruths[i + 1][0]) / 2 + 1]
#             elif previousExists == True and nextExists == False:
#                 groundTruths[i] = groundTruths[i - 1]
#             else:
#                 groundTruths[i][0] = (groundTruths[i-1][0] + groundTruths[i + 1][0]) / 2
#                 groundTruths[i][1] = (groundTruths[i - 1][0] + groundTruths[i + 1][0]) / 2
            
#     errorSum = 0
#     n = len(frames)

#     for i, (frame, groundTruth) in enumerate(zip(frames, groundTruths)):
#         loss =  getImageLoss(frame, groundTruth, model, normalized, logger)
#         print(f'Processing frame {i + 1}/{n}... loss: {loss}')

#         errorSum += loss
        

#     return errorSum / len(frames)

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
    img = datasetInterface.getFrame(0, 0)
    img = np.array(img)
    loss = getImageLossAuc(
        img,
        datasetInterface,
        [],
        model,
        True,
        logger
    )

    print(f'loss: {loss}')
