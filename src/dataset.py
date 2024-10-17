import os

from torch.utils.data import Dataset
import torch
from torchvision.io import read_image

from PIL import Image

from feature_extractor import ExtractFeatures

class GazeDataset(Dataset):
    def __init__(self, base_path='ERB3_Stimuli_Extracted', persons = [0], stride = 0, videos = ['clip_3'], length = 30, start_sample = 0):
        self.index = []
        self.base_path = base_path
        self.feature_extractor = ExtractFeatures()
        self.start_sample = start_sample
        for p in persons:
            for idx, video in enumerate(videos):
                frame_count = len(os.listdir(f'{base_path}/{video}/frames/'))
                print(f'{base_path}/{video}/frames/')
                print(frame_count)
                for i in range(frame_count - length + 1):
                    if i + length - 1 + stride >= frame_count:
                        break
                    
                    self.index.append({
                        'frame_start_frame': i,
                        'frame_end_frame': i + length - 1,  # Inclusive
                        'video_name': video,
                        'gaze_start_frame': i,
                        'gaze_end_frame': i + length - 1,    # Inclusive
                        'person_id': p,
                        'gaze_y_idx': i + length - 1 + stride
                    })
        
        # self.index = self.index[start_sample:]
                
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if idx < self.start_sample:
            return torch.tensor([0])
        video_name = self.index[idx]['video_name']
        start_frame_no = self.index[idx]['frame_start_frame']
        end_frame_no = self.index[idx]['frame_end_frame']
        
        start_gaze_no = self.index[idx]['gaze_start_frame']
        end_gaze_no = self.index[idx]['gaze_end_frame']
        person_id = self.index[idx]['person_id']
        gaze_y_idx = self.index[idx]['gaze_y_idx']
        video_frames = []
        
        for i in range(start_frame_no, end_frame_no + 1):
            frame_path = f'{self.base_path}/{video_name}/frames/{i}.jpg'
            img = Image.open(frame_path)
            img = self.feature_extractor.get_features(img)
            video_frames.append(img)
        
        result_frame = torch.stack(video_frames) # (length, channel, height, width)

        gaze_data = []
        for i in range(start_gaze_no, end_gaze_no + 1):
            file_path = f'{self.base_path}/{video_name}/gaze/{person_id}/{i}.txt'
            f = open(file_path, 'r')
            contents = f.read()
            f.close()
            x, y = contents.split(',')
            x, y = int(x), int(y)
            x, y = x / 576, y / 720
            gaze_data.append(torch.tensor([x, y]))
        
        result_gaze = torch.stack(gaze_data) # (length, 1, 2)

        file_path = f'{self.base_path}/{video_name}/gaze/{person_id}/{gaze_y_idx}.txt'
        f = open(file_path, 'r')
        contents = f.read()
        f.close()
        x, y = contents.split(',')
        x, y = int(x), int(y)
        x, y = x / 576, y / 720
        gaze_y = torch.tensor([x, y])


        result_dict = {
            'features': result_frame.float(),
            'gaze_x': result_gaze.float(),
            'gaze_y': gaze_y.float()
        }
        return result_dict
