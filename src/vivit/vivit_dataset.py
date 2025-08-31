import os
from torch.utils.data import Dataset
import torch
import sys
import numpy as np
import cv2
import json
import torch.nn.functional as F
import torchvision.transforms as T
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

class ViViT_EGTEA_Dataset(Dataset):
    def __init__(self, 
        data_root = '',  
        videos = [0], 
        length = 32, 
        t_sample = 1, 
        out_dim = (64, 64), 
        stride = 32,
        k = 10):
        """
            Inputs:
                rootPath (str) -> Folder containing the `raw` and `processed` folders
                videos (list) -> Indices of the videos
                length (int) -> Past history frame length
                t_sample (int) -> Temporal sampling rate.
                out_dim: (int, int) -> (width, height) shape
                stride: int -> prediction distance
                K: int -> clips per video
        """

        self.data_root = data_root
        self.out_dim = out_dim
        self.stride = stride
        self.t_sample = t_sample
        self.length = length
        self.k = k;

        self.index = []

        self.video_order_dir = os.path.join(self.data_root, 'processed', 'videos', 'video_order.json')
        self.gaze_order_dir = os.path.join(self.data_root, 'processed', 'gaze', 'gaze_order.json')

        self.video_json = None
        self.gaze_json = None

        with open(self.video_order_dir, 'r') as f:
            self.video_json = json.load(f)
        
        with open(self.gaze_order_dir, 'r') as f:
            self.gaze_json = json.load(f)

        self.indices = []
        for idx, video in enumerate(videos):
            self.indices.extend([video] * self.k)

        self.transform = T.Compose([
            T.resize((64, 64)),
            T.ToTensor()
        ]) 
    
    def __len__(self):
        return len(self.indices)
    
    def _get_gaussian_(self, dim, channels):
        x = torch.arange(dim) - dim // 2
        gauss = torch.exp(-(x**2) / (2 * 1.0**2))
        gauss = gauss / gauss.sum()

        kernel_2d = gauss[:, None] * gauss[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()

        kernel_2d = kernel_2d.expand(channels, 1, dim, dim)
        return kernel_2d
    
    def __getitem__(self, idx):
        """
            Returns frame in (T, C, H, W) shape
            Image is in RGB format
        """

        video = self.indices[idx]
        video_name = self.indices[idx].split('.')[0]

        frames_path = os.path.join(self.data_root, 'processed', 'frame', video_name, 'm1')
        n_frames = sum(1 for _ in os.scandir(folder))

        tau = (n_frames - self.stride) // (self.k * self.length)
        max_start = self.n - (self.length - 1) * tau - 1
        
        begin = random.randint(0, max_start)
        frame_indices = [begin + i * tau for i in range(self.length)]
        frame_names = [f'{i}.png' for i in frame_indices]
        rgb_clip = []
        for f in frame_names:
            img = Image.open(os.path.join(frames_path, f)).convert('RGB')
            img = self.transform(img)
            rgb_clip.append(img)
        
        rgb_clip = torch.stack(rgb_clip, dim = 0)
        tgt_gaze = torch.zeros((self.stride, 1, self.out_dim[0], self.out_dim[1]))

        last_frame = frame_indices[-1] 
        for i in range(1, self.stride + 1):
            x = self.gaze_json[str(video)]['0'][str(last_frame + i)]['x']
            y = self.gaze_json[str(video)]['0'][str(last_frame + i)]['y']
            x = round(min(x, 1) * (self.out_dim[1] - 1))
            y = round(min(y, 1) * (self.out_dim[0] - 1))
            tgt_gaze[i, 0, y, x] = 1
            
        kernel = _get_gaussian_(7, 1)
        tgt_gaze = F.conv2d(tgt_gaze, kernel, padding = 3, groups = 1)


        result_dict = {
            'input_frames': rgb_clip.float(),
            'tgt_gaze': tgt_gaze.float()
        }

        return result_dict