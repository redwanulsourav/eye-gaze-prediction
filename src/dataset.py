import os
from torch.utils.data import Dataset
import torch
from torchvision.io import read_image
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from feature_extractor import ExtractFeatures
from dataset_interface.dataset_interface import DatasetInterface

class GazeDataset(Dataset):
    def __init__(
        self, rootPath = '', 
        viewers = [0], 
        stride = 0, 
        videos = [0], 
        length = 3, 
        startFrame = 0):
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
        self.featureExtractor = ExtractFeatures()
        self.startFrame = startFrame
        self.datasetInterface = DatasetInterface(rootPath)

        self.index = []

        for p in viewers:
            for idx, video in enumerate(videos):
                frameCount = self.datasetInterface.getFrameCount(videoIdx = video)
                for i in range(frameCount - length + 1):
                    if i + length - 1 + stride >= frameCount:
                        break
                    
                    self.index.append({
                        'start_frame': i,
                        'end_frame': i + length - 1,  # Inclusive
                        'video_idx': video,
                        'start_gaze': i,
                        'end_gaze': i + length - 1,    # Inclusive
                        'viewer_id': p,
                        'target_gaze_start': (i + length - 1) + 1,
                        'target_gaze_end': (i + length - 1) + stride  # inclusive
                    })
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if idx < self.startFrame:
            return torch.tensor([0])
        
        videoIdx = self.index[idx]['video_idx']
        startFrameIdx = self.index[idx]['start_frame']
        endFrameIdx = self.index[idx]['end_frame']
        
        startGazeIdx = self.index[idx]['start_gaze']
        endGazeIdx = self.index[idx]['end_gaze']
        viewerIdx = self.index[idx]['viewer_id']
        targetGazeStart = self.index[idx]['target_gaze_start']
        targetGazeEnd = self.index[idx]['target_gaze_end']
        
        videoFrames = self.datasetInterface.getRangeFrames(videoIdx = videoIdx, start = startFrameIdx, end = endFrameIdx + 1)
        videoWidth, videoHeight = videoFrames[0].size   # Video Frames is an array of PIL images! .size works
        videoFrames = [self.featureExtractor.get_features(frame) for frame in videoFrames]
        videoFrames = torch.stack(videoFrames)
        
        gazeData = self.datasetInterface.getAllGazeOfSingleViewer(videoIdx = videoIdx, viewerIdx = viewerIdx)
        gazeData = gazeData[startGazeIdx: endGazeIdx + 1]
        gazeData = [torch.tensor([x / videoHeight, y / videoWidth]) for (x, y) in gazeData]
        gazeData = torch.stack(gazeData)
        
        gazeY = self.datasetInterface.getAllGazeOfSingleViewer(videoIdx = videoIdx, viewerIdx = viewerIdx)[targetGazeStart: targetGazeEnd + 1]
        gazeY = [torch.tensor([x[0] / videoHeight, x[1] / videoWidth]) for x in gazeY]
        gazeY = torch.stack(gazeY)
        
        result_dict = {
            'features': videoFrames.float(),
            'gaze_x': gazeData.float(),
            'gaze_y': gazeY.float()
        }
        return result_dict
