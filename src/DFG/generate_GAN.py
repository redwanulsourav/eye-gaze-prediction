import torch
from torch.utils.data import DataLoader
import cv2
import argparse
import yaml
import numpy as np

from dfg_dataset import DFG_GTEA_Dataset
from models import FrameGenerator

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--cfg', help = 'Config [.yaml]', required = True)
    ap = ap.parse_args()

    with open(ap.cfg) as f:
        config = yaml.safe_load(f)

    train_data = DFG_GTEA_Dataset(
                length = config['length'],
                videos = config['videos'],
                root = config['base_path'])
    train_loader = DataLoader(train_data, batch_size = config['batch_size'], shuffle = False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = {}
    generator['model'] = FrameGenerator().to(device).eval()
    generator['model'].load_state_dict(torch.load('/data/rsourave/DFG_Saved/GTEA/59/epochs/2/generator_model_state.pt'))

    for idx, data in enumerate(train_loader):
        input_frames = data['input_frames'].to(device)
        output = generator['model'](input_frames[:, 0, :, :, :])
        output = output.squeeze()
        for i in range(32):
            frame = output[:, i, :, :].permute(1, 2, 0)
            frame_np = frame.cpu().detach().numpy()
            frame_np = (frame_np - frame_np.min())/(frame_np.max() - frame_np.min()) 
            assert frame_np.max() <= 1
            assert frame_np.min() >= 0
            frame_np = frame_np * 255
            frame_np = frame_np.astype(np.uint8)
            print(frame_np.shape)
            print(frame_np.dtype)            
            cv2.imwrite(f'{i}.png', frame_np)
        print(output.shape)
        frame = input_frames[:, 0, :, :, :]
        print(frame.shape)
        frame = frame[0, :, :, :].permute(1, 2, 0)
        frame_np = frame.cpu().detach().numpy()
        frame_np = (frame_np - frame_np.min()) / (frame_np.max() - frame_np.min())
        frame_np = frame_np * 255
        frame_np = frame_np.astype(np.uint8)
        print(frame_np.shape)
        cv2.imwrite('0-true.png', frame_np)
        break


