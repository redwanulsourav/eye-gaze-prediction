import os
from torch.utils.data import Dataset
import torch
from torchvision.io import read_image
from PIL import Image
import sys
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from dataset_interface.dataset_interface import DatasetInterface

class DFG_GTEA_Dataset(Dataset):
    def __init__(self, rootPath = '',  videos = [0], length = 32):
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
        self.datasetInterface = DatasetInterface(rootPath)

        self.index = []


        for idx, video in enumerate(videos):
            frameCount = self.datasetInterface.getFrameCount(videoIdx = video)
            for i in range(frameCount - length + 1):
                if i + length - 1 + 32 >= frameCount:
                    break
                
                self.index.append({
                    'start_frame': i,
                    'end_frame': i + length - 1,  # Inclusive
                    'video_idx': video,
                    'start_gaze': i,
                    'end_gaze': i + length - 1,    # Inclusive
                    'target_gaze_start': (i + length - 1) + 1,
                    'target_gaze_end': (i + length - 1) + 32  # inclusive
                })
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        videoIdx = self.index[idx]['video_idx']
        startFrameIdx = self.index[idx]['start_frame']
        endFrameIdx = self.index[idx]['end_frame']
        
        startGazeIdx = self.index[idx]['start_gaze']
        endGazeIdx = self.index[idx]['end_gaze']
        targetGazeStart = self.index[idx]['target_gaze_start']
        targetGazeEnd = self.index[idx]['target_gaze_end']
        
        """ Load videos """
        """ Video output shape should be (3, 32, 64, 64) """

        videoWidth, videoHeight = None, None
        videoFrames = self.datasetInterface.getRangeFrames(videoIdx = videoIdx, start = startFrameIdx, end = endFrameIdx + 1)
        for i in range(32):
            frame = videoFrames[i]
            frame = np.array(frame)
            frame = frame[:, :, ::-1].copy()
            videoWidth, videoHeight = frame.shape[1], frame.shape[0]
            frame = cv2.resize(frame, (64, 64)) # (64, 64, 3)
            frame = np.transpose(frame, (2, 0, 1))  # (3, 64, 64)
            videoFrames[i] = torch.from_numpy(frame)
            
        videoFrames = torch.stack(videoFrames)  # (32, 3, 64, 64)
        print(videoFrames.shape)
        """ Reshape to bring the color channel first """
        videoFrames = torch.transpose(videoFrames, 1, 0)
        print(videoFrames.shape)

        """ Get all viewers gaze data """
        viewerCount = self.datasetInterface.getViewerCount(videoIdx = videoIdx)
        flattenedTimeIdx = []
        flattenedRowIdx = []
        flattenedColIdx = []
        for i in range(startGazeIdx, endGazeIdx + 1):
            for viewerIdx in range(viewerCount):
                gazeData = self.datasetInterface.getAllGazeOfSingleViewer(videoIdx = videoIdx, viewerIdx = viewerIdx)
                # gazeData = gazeData[i]
                gazeData_ = (round(min(gazeData[i][0] / videoWidth, 1) * 63), round(min(gazeData[i][1] / videoHeight, 1) * 63))
                flattenedTimeIdx.append(i-startGazeIdx)
                flattenedRowIdx.append(gazeData_[0])
                flattenedColIdx.append(gazeData_[1])
                # gazeData = torch.stack(gazeData)
        
        temporalFixationMap = torch.zeros(1, 32, 64, 64)
        temporalFixationMap[0, flattenedTimeIdx, flattenedRowIdx, flattenedColIdx] = 1

        result_dict = {
            'frames': videoFrames.float(),
            'temporal_fixation_map': temporalFixationMap.float(),
        }

        return result_dict
