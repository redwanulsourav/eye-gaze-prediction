import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from datetime import datetime
import yaml
import argparse
import pathlib
import os
import shutil
import time
import json
import glob

from dataset import GazeDataset
from models import GazeNet


def train_one_epoch(epoch_index, training_loader, optimizer, loss_fn, model, device):
    loss_sum = 0
    batch_loss_sum = 0

    H = {
        "running_loss": []
    }
    
    for i, data in enumerate(training_loader):
        feature_x = data['features'].to(device)
        gaze_x = data['gaze_x'].to(device)
        gaze_y = data['gaze_y'].to(device)
        
        optimizer.zero_grad()

        outputs = model(feature_x, gaze_x)

        loss = loss_fn(outputs, gaze_y)
        loss.backward()

        optimizer.step()

        loss_sum += loss.item()
        H["running_loss"].append(loss.item())

    return loss_sum / len(training_loader), H['running_loss']

def prepare_dirs():
    os.makedirs(f'runs/', exist_ok = True)
    run_id = len(os.listdir('runs/'))
    
    os.makedirs(f'runs/{run_id}/')
    os.makedirs(f'runs/{run_id}/history')
    os.makedirs(f'runs/{run_id}/weights')
    os.makedirs(f'runs/{run_id}/epochs')

    os.system(f'cp {ap.cfg} runs/{run_id}/run_config.yaml')
    
    return run_id

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--cfg', help='Train run cfg', required = True)
    ap = ap.parse_args()
    
    run_id = prepare_dirs()

    with open(ap.cfg) as f:
        run_cfg = yaml.safe_load(f)
    
    
    start_epoch = 0
    end_epoch = 50
    batch_size = 1
 
    gaze_dataset = GazeDataset(
                               persons=run_cfg['dataset']['persons'], 
                               stride=run_cfg['dataset']['stride'], 
                               length = run_cfg['dataset']['length'],
                               start_sample = 0,
                               videos=['1'],
                               base_path = '/data/rsourave/datasets/extracted/ERB3_Stimuli_Extracted')
    training_loader = DataLoader(gaze_dataset, batch_size=batch_size, shuffle = True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GazeNet(length = run_cfg['dataset']['length'], model_type = run_cfg['model']['type']).to(device)
    model.train()

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adagrad(model.parameters(), lr=run_cfg['optimizer']['lr'])

    print(f'Run ID: {run_id}')
    
    running_losses = []
    
    for i in range(start_epoch, end_epoch + 1):
        model.train()
        avg_loss, current_running_loss = train_one_epoch(i, training_loader, optim, loss_fn, model, device)
        running_losses += current_running_loss
        
        H = {
            "average_loss": avg_loss,
            "running_losses": running_losses,
        }
        os.makedirs(f'runs/{run_id}/epochs/{i}/')
        with open(f'runs/{run_id}/epochs/{i}/history.json', 'w') as f:
            json.dump(H, f)
        
        torch.save(model.state_dict(), f'runs/{run_id}/epochs/{i}/weights.pt')

        print(f'Epoch {i}/{end_epoch + 1}: Loss: {avg_loss}')


