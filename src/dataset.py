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
        length = 30, 
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
                        'frame_start_frame': i,
                        'frame_end_frame': i + length - 1,  # Inclusive
                        'video_idx': video,
                        'gaze_start_frame': i,
                        'gaze_end_frame': i + length - 1,    # Inclusive
                        'viewer_id': p,
                        'gaze_y_idx': i + length - 1 + stride
                    })
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if idx < self.startSample:
            return torch.tensor([0])
        
        videoIdx = self.index[idx]['video_idx']
        startFrameIdx = self.index[idx]['frame_start_frame']
        endFrameIdx = self.index[idx]['frame_end_frame']
        
        startGazeIdx = self.index[idx]['gaze_start_frame']
        endGazeIdx = self.index[idx]['gaze_end_frame']
        viewerIdx = self.index[idx]['viewer_id']
        gazeYIdx = self.index[idx]['gaze_y_idx']
        
        videoFrames = self.datasetInterface.getAllFrames(videoIdx = videoIdx)[startFrameIdx: endFrameIdx + 1]
        videoWidth, videoHeight = videoFrames[0].size
        videoFrames = [self.featureExtractor.get_features(frame) for frame in videoFrames]
        videoFrames = torch.stack(videoFrames)
        
        gazeData = self.datasetInterface.getAllGazeOfSingleViewer(videoIdx = videoIdx, viewerIdx = viewerIdx)
        gazeData = gazeData[startGazeIdx: endGazeIdx + 1]
        gazeData = [torch.tensor([x / videoHeight, y / videoWidth]) for (x, y) in gazeData]
        gazeData = torch.stack(gazeData)
        
        gazeY = self.datasetInterface.getGazeLocation(videoIdx = videoIdx, frameIdx = gazeYIdx, viewerIdx = viewerIdx)
        gazeY = torch.tensor([gazeY[0] / videoHeight, gazeY[1] / videoWidth])
        
        result_dict = {
            'features': videoFrames.float(),
            'gaze_x': gazeData.float(),
            'gaze_y': gazeY.float()
        }
        return result_dict
