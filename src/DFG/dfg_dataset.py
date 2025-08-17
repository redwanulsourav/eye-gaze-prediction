import os
from torch.utils.data import Dataset
import torch
from torchvision.io import read_image
from PIL import Image
import sys
import numpy as np
import cv2
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from dataset_interface.dataset_interface import DatasetInterface

class DFG_GTEA_Dataset(Dataset):
    def __init__(self, data_root = '',  videos = [0], out_dim = (64, 64)):
        """
            Inputs:
                rootPath (str) -> Folder containing the `raw` and `processed` folders
                videos (list) -> Indices of the videos
                t_sample (int) -> Temporal sampling rate.
                out_dim (int, int) -> Output (width, height) shape.
        """

        self.data_root = data_root
        self.index = []
        self.video_order_dir = os.path.join(self.data_root, 'processed', 'videos', 'video_order.json')
        self.gaze_order_dir = os.path.join(self.data_root, 'processed', 'gaze', 'gaze_order.json')

        self.video_json = None
        self.gaze_json = None

        with open(self.video_order_dir, 'r') as f:
            self.video_json = json.load(f)

        with open(self.gaze_order_dir, 'r') as f:
            self.gaze_json = json.load(f)


        for idx, video in enumerate(videos):
            video_name = self.video_json[str(video)].split('.')[0]

            frames_dir = os.path.join(self.data_root, 'processed', 'frames', video_name, 'm1')
            n_frames = len(os.listdir(frames_dir))

            for i in range(0, n_frames, t_sample):
                if i + 32 >= n_frames:
                    break
                
                self.index.append({
                    'frame': i,
                    'video': video,
                    'gaze': i,
                    'tgt_begin': i,
                    'tgt_end': i + 32  # inclusive
                })
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Returns a dictionary with training data, and target data
            Dictionary:
                input_frame: An PyTorch Tensor of shape (C, H, W), in RGB format.
                tgt_frames: An PyTorch Tensor of shape (32, C, H, W) in RGB format
                tgt_gaze: An Pytorch tensor of shape (32, 1, H, W).
        """

        video = self.index[idx]['video']
        frame = self.index[idx]['frame']
        
        tgt_begin = self.index[idx]['tgt_begin']
        tgt_end = self.index[idx]['tgt_end']

        video_name = self.video_json[str(video)].split('.')[0]
        frame_dir = os.path.join(self.data_root, 'processed', 'frames', video_name, 'm1')
        input_frame = cv2.imread(os.path.join(frame_dir, f'{frame}.png')) 

        
        """ Load target video """
        """ Video output shape should be (3, 32, 64, 64) """
        
        w, h = None, None
        tgt_frames = [None for i in range(32)]
        tgt_gaze = torch.zeros((32, 1, self.out_dim[1], self.out_dim[0]))

        for i in range(32):
            tgt_frames[i] = cv2.imread(os.path.join(frame_dir, f'{frame + i}.png'))
            tgt_frames[i] = cv2.cvtColor(tgt_frames[i], cv2.BGR2RGB)
            w, h = frame.shape[1], frame.shape[0]
            tgt_frames[i] = cv2.resize(tgt_frames[i], self.out_dim) # (64, 64, 3)
            tgt_frames[i] = np.transpose(tgt_frames[i], (2, 0, 1))  # (3, 64, 64)
            tgt_frames[i] = torch.from_numpy(tgt_frames[i])
            x = self.gaze_json[str(video)][str(0)][f'{frame + i}']['x']
            y = self.gaze_json[str(video)][str(0)][f'{frame + i}']['y']

            x = round(min(x, 1) * (self.out_dim[1] - 1))
            y = round(min(y, 1) * (self.out_dim[0] - 1))

            tgt_gaze[i, 0, y, x] = 1

        # Should really convolve here? Or no.
        # My variable names are shit.

        tgt_frames = torch.stack(tgt_frames)  # (32, 3, 64, 64)
        
        result_dict = {
            'input_frame': input_frame.float(),
            'tgt_frames': tgt_frames.float(),
            'tgt_gaze': tgt_gaze.float()
        }

        return result_dict
