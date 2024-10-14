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


def train_one_epoch(epoch_index, run_id, training_loader, 
                    optimizer, loss_fn, model, device, starting_sample):
    loss_sum = 0
    batch_loss_sum = 0

    data_length = len(training_loader)

    H = {
        "running_loss": [],
        "batch_losses": []
    }
    print(len(training_loader))
    for i, data in enumerate(training_loader):
        if i < starting_sample:
            print(f'Skipping {i}')
            continue
        start_time = time.time()
        
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
        batch_loss_sum += loss.item()
        print(f'Epoch {epoch_index}: [{i}/{data_length}] Time: {time.time() - start_time} Loss: {loss.item()}')

        if i % 100 == 99:
            torch.save(model.state_dict(), f'runs/{run_id}/weights/epoch_{epoch_index}_run_{i}.pt')
            H["batch_losses"].append(batch_loss_sum / 100)
            with open(f'runs/{run_id}/history/epoch_{epoch_index}_run_{i}.json', 'w') as f:
                json.dump(H, f)
            batch_loss_sum = 0

    return loss_sum / data_length, H['running_loss'], H['batch_losses']

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--cfg', help='Train run cfg', required = True)

    ap = ap.parse_args()

    with open(ap.cfg) as f:
        run_cfg = yaml.safe_load(f)
    
    # Folder to save checkpoints
    os.makedirs(f'runs/', exist_ok = True)
    
    run_id = len(os.listdir(f'runs/'))
    pathlib.Path(f'runs/{run_id}/history').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'runs/{run_id}/weights').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'runs/{run_id}/epochs').mkdir(parents=True, exist_ok=True)
    
    # Copy the cfg file.
    shutil.copy(ap.cfg, f'runs/{run_id}/run_config.yaml')

    model = GazeNet(length = run_cfg['dataset']['length'], model_type=run_cfg['model']['type'])
    
    start_epoch = 0
    starting_sample = 0
    # Check if pretrained data needs to be loaded.
    end_epoch = 50
    if 'weights' in run_cfg:
        if os.path.exists(run_cfg['weights']['weights_path']) == True:
            model.load_state_dict(torch.load(run_cfg['weights']['weights_path']))
            print('Pretrained model loaded')
        else:
            print(f'{run_cfg["weights"]["weights_path"]} not found')
        
    if 'epochs' in run_cfg:
        if 'start_epoch' in run_cfg['epochs']:
            start_epoch = run_cfg['epochs']['start_epoch']
        else:
            start_epoch = 0
        if 'end_epoch' in run_cfg['epochs']:
            end_epoch = run_cfg['epochs']['end_epoch']
        else:
            end_epoch = start_epoch + 50

    gaze_dataset = GazeDataset(
                               persons=run_cfg['dataset']['persons'], 
                               stride=run_cfg['dataset']['stride'], 
                               length = run_cfg['dataset']['length'],
                               start_sample=starting_sample,
                               videos=['57'],
                               base_path = '/data/rsourave/datasets/extracted/ERB3_Stimuli_Extracted')
    
    batch_size = 1
    if 'batch_size' in run_cfg['dataset']:
        batch_size = run_cfg['dataset']['batch_size']

    training_loader = DataLoader(gaze_dataset, batch_size=batch_size, shuffle = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    model.to(device)
    
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adagrad(model.parameters(), lr=run_cfg['optimizer']['lr'])

    print(f'Run ID: {run_id}')
    
    timestamp = datetime.now().strftime(f'%Y%m%d_%H%M%S')
    running_losses = []
    batch_losses = []
    
    
    for i in range(start_epoch, end_epoch + 1):
        model.train()
        start_time = time.time()
        avg_loss, current_running_loss, current_batch_loss = train_one_epoch(i, run_id, training_loader, optim, loss_fn, model, device, starting_sample)
        print('epoch done') 
        start_sample = 0
        running_losses += current_running_loss
        batch_losses += current_batch_loss
        
        H = {
            "time": time.time() - start_time,
            "average_loss": avg_loss,
            "running_losses": running_losses,
            "batch_losses": batch_losses
        }

        pathlib.Path(f'runs/{run_id}/epochs/{i}/').mkdir(parents=True, exist_ok=True)
        with open(f'runs/{run_id}/epochs/{i}/history.json', 'w') as f:
            json.dump(H, f)
        
        torch.save(model.state_dict(), f'runs/{run_id}/epochs/{i}/weights.pt')

        print(f'Epoch {i}/{end_epoch + 1}: Time: {H["time"]} Loss: {avg_loss}')

        weight_files = glob.glob(f'runs/{run_id}/weights/*.pt')
        history_files = glob.glob(f'runs/{run_id}/history/*.json')

        for f in weight_files:
            os.remove(f)
        for f in history_files:
            os.remove(f)

        if starting_sample != 0:
            gaze_dataset = GazeDataset(persons=run_cfg['dataset']['persons'], 
                               stride=run_cfg['dataset']['stride'], 
                               length = run_cfg['dataset']['length'],
                               start_sample=0,
                               base_path = '/data/rsourave/datasets/extracted/ERB3_Stimuli_Extracted', 
                               videos = ['clip_10'])
            print(len(gaze_dataset)) 
            training_loader = DataLoader(gaze_dataset, batch_size=batch_size, shuffle=True)
            starting_sample = 0

