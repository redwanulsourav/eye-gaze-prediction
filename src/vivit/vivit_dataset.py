import os
from torch.utils.data import Dataset
import torch
import sys
import numpy as np
import cv2
import json
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

class ViViT_EGTEA_Dataset(Dataset):
    def __init__(self, data_root = '',  videos = [0], length = 32, t_sample = 1, out_dim = (64, 64), stride = 32):
        """
            Inputs:
                rootPath (str) -> Folder containing the `raw` and `processed` folders
                videos (list) -> Indices of the videos
                length (int) -> Past history frame length
                t_sample (int) -> Temporal sampling rate.
                out_dim: (int, int) -> (width, height) shape
                stride: int -> prediction distance
        """

        self.data_root = data_root
        self.out_dim = out_dim
        self.stride = stride
        self.t_sample = t_sample
        self.length = length

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
            
            for i in range(n_frames):
                last_frame_idx = length * t_sample + i
                
                if last_frame_idx + self.stride >= n_frames:
                    break

                self.index.append({
                    'begin': i,
                    'video': video,
                    'tgt_begin': (i + length - 1) + 1,
                    'tgt_end': (i + length - 1) + stride  # inclusive
                })
    
    def __len__(self):
        return len(self.index)
    
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

        video = self.index[idx]['video']
        begin = self.index[idx]['begin']

        g_begin = self.index[idx]['begin']
        tgt_g_begin = self.index[idx]['tgt_begin']
        tgt_g_end = self.index[idx]['tgt_end']
        
        """ Load videos """

        v_height, v_width = None, None
        video_frames = [None for i in range(self.length)]
        g_data = torch.zeros((self.length, 1, self.out_dim[0], self.out_dim[1]))
        tgt_g_data = torch.zeros((self.stride, 1, self.out_dim[0], self.out_dim[1]))

        for i in range(self.length):
            video_name = self.video_json[str(video)].split('.')[0]
            frame_idx = i * self.t_sample + begin
            frame_dir = os.path.join(self.data_root, 'processed', 'frames', video_name, 'm1', f'{frame_idx}.png')
            frame = cv2.imread(frame_dir)
            v_height, v_width = frame.shape[0], frame.shape[1]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.out_dim[1], self.out_dim[0])) # (64, 64, 3)
            frame = np.transpose(frame, (2, 0, 1))  # (3, 64, 64)
            video_frames[i] = torch.from_numpy(frame)
            x = self.gaze_json[str(video)][str(0)][str(frame_idx)]['x']
            y = self.gaze_json[str(video)][str(0)][str(frame_idx)]['y']
            x = round(min(x, 1) * (self.out_dim[1] - 1))
            y = round(min(y, 1) * (self.out_dim[0] - 1))
            g_data[i, 0, y, x] = 1
            
        video_frames = torch.stack(video_frames)  # (T, 3, 64, 64)
        
        """ Get all viewers gaze data """
        for i in range(self.stride):
            frame_idx = begin + self.length * self.t_sample + (i + 1)
            x = self.gaze_json[str(video)][str(0)][str(frame_idx)]['x']
            y = self.gaze_json[str(video)][str(0)][str(frame_idx)]['y']
            x = round(min(x, 1) * (self.out_dim[1] - 1))
            y = round(min(y, 1) * (self.out_dim[0] - 1))

            tgt_g_data[i, 0, y, x] = 1

        kernel = _get_gaussian_(7, 1)
        tgt_g_data = F.conv2d(tgt_g_data, kernel, padding = 3, groups = 1)


        result_dict = {
            'input_frames': video_frames.float(),
            'g_data': g_data.float(),
            'tgt_g_data': tgt_g_data.float()
        }

        return result_dict

if __name__ == '__main__':
    dataset = ViViT_EGTEA_Dataset(data_root = '/data/rsourave/datasets/EGTEA', videos = [0, 1, 2, 3], length = 16, t_sample = 3, out_dim = (64, 64), stride = 32)
    print(len(dataset))
    print('dataset init done')
    print(list(dataset[0].keys()))
    print(dataset[0]['input_frames'].shape)
    print(dataset[0]['g_data'].shape)
    print(dataset[0]['tgt_g_data'].shape)
