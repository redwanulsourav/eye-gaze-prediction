import os
from torch.utils.data import Dataset
import torch
from torchvision.io import read_image
from PIL import Image
import sys
import numpy as np
import cv2
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from feature_extractor import ExtractFeatures    # Should put this in model.
from dataset_interface.dataset_interface import DatasetInterface

class GazeDataset(Dataset):
    def __init__(
        self, rootPath = '', 
        viewers = [0], 
        videos = [0], 
        length = 1):
        """
            Inputs:
                rootPath (str) -> Folder containing the `raw` and `processed` folders
                viewers (list) -> Indices of viewers of the video
                stride (list) -> Prediction distance
                videos (list) -> Indices of the videos
                length (int) -> Past history frame length
                startFrame (int) -> Which frame to start from.
        """

        self.rootPath = rootPath
        # self.featureExtractor = ExtractFeatures()
        # self.startFrame = startFrame
        self.datasetInterface = DatasetInterface(rootPath)

        self.index = []

        for p in viewers:
            for idx, video in enumerate(videos):
                frameCount = self.datasetInterface.getFrameCount(videoIdx = video)
                for i in range(frameCount - 1 + 1):
                    if i + length - 1 + 32 >= frameCount:
                        break
                    
                    self.index.append({
                        'start_frame': i,
                        'end_frame': i + 1 - 1,  # Inclusive
                        'video_idx': video,
                        'start_gaze': i,
                        'end_gaze': i + 1 - 1,    # Inclusive
                        'viewer_id': p,
                        'target_gaze_start': (i + 1 - 1) + 1,
                        'target_gaze_end': (i + 1 - 1) + 1  # inclusive
                    })
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        videoIdx = self.index[idx]['video_idx']
        startFrameIdx = self.index[idx]['start_frame']
        endFrameIdx = self.index[idx]['end_frame']
        
        startGazeIdx = self.index[idx]['start_gaze']
        endGazeIdx = self.index[idx]['end_gaze']
        viewerIdx = self.index[idx]['viewer_id']
        targetGazeStart = self.index[idx]['target_gaze_start']
        targetGazeEnd = self.index[idx]['target_gaze_end']
        
        videoFrame = self.datasetInterface.getRangeFrames(videoIdx = videoIdx, start = startFrameIdx, end = endFrameIdx + 1)[0].convert('RGB')
        videoFrame = np.array(videoFrame)
        videoFrame = cv2.cvtColor(videoFrame, cv2.COLOR_RGB2BGR)
        videoFrame = cv2.resize(videoFrame, (224, 224))
        videoWidth, videoHeight = videoFrame.shape[1], videoFrame.shape[0]  # Video Frames is an array of PIL images! .size works
        # ??videoFrames = [self.featureExtractor.get_features(frame) for frame in videoFrames]
        # ideoFrames = torch.stack(videoFrames)
        
        gazeData = self.datasetInterface.getAllGazeOfSingleViewer(videoIdx = videoIdx, viewerIdx = viewerIdx)
        gazeData = gazeData[startGazeIdx: endGazeIdx + 1][0]
        gazeData = (min(gazeData[0] / videoWidth, 1), min(gazeData[1] / videoHeight, 1))
        
        gazeY = self.datasetInterface.getAllGazeOfSingleViewer(videoIdx = videoIdx, viewerIdx = viewerIdx)[targetGazeStart: targetGazeEnd + 1][0]
        gazeY = (min(gazeY[0] / videoWidth, 1), min(gazeY[1] / videoHeight, 1))
        
        result_dict = {
            'features': videoFrame,
            'gaze_x': gazeData,
            'gaze_y': gazeY
        }
        return result_dict
