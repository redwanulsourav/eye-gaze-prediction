import torch
import torchvision
import torchvision.transforms as transforms

# PyTorch TensorBoard support
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


def train_one_epoch(training_loader, optimizer, loss_fn, model, device):
    loss_sum = 0
    batch_loss_sum = 0

    data_length = len(training_loader)

    H = {
        "running_loss": []
    }
    train_xs = []
    train_ys = []
    predicted_xs = []
    predicted_ys = []

    for i, data in enumerate(training_loader):
        feature_x = data['features'].to(device)
        gaze_x = data['gaze_x'].to(device)
        gaze_y = data['gaze_y'].to(device)
        
        optimizer.zero_grad()
        outputs = model(feature_x, gaze_x)
        
        train_xs.append(gaze_y[0].tolist()[0])
        train_ys.append(gaze_y[0].tolist()[1])
        predicted_xs.append(outputs[0].tolist()[0])
        predicted_ys.append(outputs[0].tolist()[1])
        
        loss = loss_fn(outputs, gaze_y)
        loss.backward()

        optimizer.step()

        loss_sum += loss.item()
        H["running_loss"].append(loss.item())

        print(f'{i} / {data_length} Loss: {loss.item()}')
    
    H['train_xs'] = train_xs
    H['train_ys'] = train_ys
    H['predicted_xs'] = predicted_xs
    H['predicted_ys'] = predicted_ys
    return loss_sum / data_length, H['running_loss'], H['train_xs'], H['train_ys'], H['predicted_xs'], H['predicted_ys']


def prepare_dirs(output_path: str):
    os.makedirs(f'{output_path}/', exist_ok = True)
    run_id = len(os.listdir(f'{output_path}/'))
    
    os.makedirs(f'{output_path}/{run_id}/')
    os.makedirs(f'{output_path}/{run_id}/history')
    os.makedirs(f'{output_path}/{run_id}/weights')
    os.makedirs(f'{output_path}/{run_id}/epochs')

    os.system(f'cp {ap.cfg} {output_path}/{run_id}/run_config.yaml')
    
    return run_id


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--cfg', help = 'Train run cfg', required = True)
    ap.add_argument('-o', '--output', help = 'Path to store output', default = 'runs')
    ap = ap.parse_args()

    with open(ap.cfg) as f:
        run_configuration = yaml.safe_load(f)
    run_id = prepare_dirs(ap.output)
    
    gaze_dataset = GazeDataset(persons=        run_configuration['dataset']['persons'], 
                               stride=         run_configuration['dataset']['stride'], 
                               length =        run_configuration['dataset']['length'],
                               start_sample=   0,
                               videos=         ['1'],
                               base_path =     '/data/rsourave/datasets/extracted/ERB3_Stimuli_Extracted')
    training_loader = DataLoader(gaze_dataset, batch_size=1)

   
                        
    training_loader = DataLoader(gaze_dataset, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GazeNet(length =        run_configuration['dataset']['length'], 
                    model_type =    run_configuration['model']['type']).to(device)
    
    if 'weights' in run_configuration:
        if os.path.exists(run_configuration['weights']['weights_path']) == True:
            model.load_state_dict(torch.load(run_configuration['weights']['weights_path']))
            print('Pretrained model loaded')
        else:
            print('Pretrained weights not found')
        
    model.train()

    loss_fn = torch.nn.MSELoss()
    print(run_configuration['optimizer']['lr'])
    optim = torch.optim.Adagrad(model.parameters(), lr=run_configuration['optimizer']['lr'])

    print(f'Run ID: {run_id}')
    
    avg_loss, running_losses, train_xs, train_ys, predicted_xs, predicted_ys = \
                    train_one_epoch(training_loader, optim, loss_fn, model, device)
    H = {
        "average_loss": avg_loss,
        "running_losses": running_losses,
        "train_xs": train_xs, 
        "train_ys": train_ys,
        "predicted_xs": predicted_xs,
        "predicted_ys": predicted_ys
    }    
        
    with open(f'{ap.output}/{run_id}/history.json', 'w') as f:
        json.dump(H, f)

    print(f'Average Loss: {avg_loss}')
        
    torch.save(model.state_dict(), f'{ap.output}/{run_id}/weights.pt')
