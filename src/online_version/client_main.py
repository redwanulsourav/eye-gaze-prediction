import os
import sys
import socket
import threading

import yaml
import json
import argparse
import random
import time
import posix_ipc

from torch.utils.data import DataLoader
import cv2

from dataset import GazeDataset

def handleNewPrediction(conn, addr):
    print(f'New connection from {addr}')
    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            data = data.decode().rstrip('X')
            data = json.loads(data)
            
            updatePredictedGaze(data)
            



def prepareDirs(outputPath: str, cfgPath):
    os.makedirs(f'{outputPath}/', exist_ok = True)
    runId = len(os.listdir(f'{outputPath}/'))
    
    runPath = os.path.join(outputPath, str(runId))
    historyPath = os.path.join(runPath, 'history')
    weightsPath = os.path.join(runPath, 'weights')
    epochsPath = os.path.join(runPath, 'epochs')
    srcPath = os.path.join(runPath, 'src')

    os.system(f'cp models.py {srcPath}/models.py')
    os.system(f'cp client_main.py {srcPath}/client_main.py')   
    os.system(f'cp {cfgPath} {runPath}/config.yaml')
    
    return run_id

def findFPS(config):
    videoJSONPath = os.path.join(config['base_path'], 'processed', 'videos', 'video_order.json')

    with open(videoJSONPath) as f:
        contents = f.read()
    
    videoJSON = json.loads(contents)
    videoName = videoJSON[str(config['videos'][0])]
    print(videoName)
    videoPath = os.path.join(config['base_path'], 'processed', 'videos', videoName)
    
    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return float(fps)

def parseConfig(configPath):
    with open(ap.cfg) as f:
        config = yaml.safe_load(f)
    return config


def main(config):
    """
        Wait for x seconds, then send the data.
    """
    fps = findFPS(config)
    frameRenderTime = float(1.0) / fps

    gazeDataset = GazeDataset(  stride=          config['stride'], 
                                length =         config['length'],
                                videos=          config['videos'] if 'videos' in config else [0],
                                rootPath =       config['base_path'],
                                viewers =        config['viewers'] if 'viewers' in config else [0])
    trainLoader = DataLoader(gazeDataset,  
                            batch_size = config['batch_size'], 
                            shuffle = config['shuffle'] if 'shuffle' in config else True)
    
    for i, data in enumerate(trainLoader):
        # Blocking wait for x seconds.
        rtt = random.gauss(40, 10)
        print(i)
        time.sleep(rtt / 1000 + frameRenderTime)
        print(data['gaze_x'].shape)
        tosend = {
            'type': 'RecGL',
            'gazeLocations': {
                'length': 1,
                'data': [{
                    'frame': {
                        'idx': i,
                        'x': data['gaze_x'][0, 0, 0].item(),
                        'y': data['gaze_x'][0, 0, 1].item()
                    }
                }
                ]
            }
        }

        mq.send(json.dumps(tosend).encode())
        print('data sent')
    





if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--cfg', help = 'Train config [.yaml]', required = True)
    ap = ap.parse_args()

    config = parseConfig(ap.cfg)
    try:
        main(config)
    except Exception as e:
        print(e)
        mq.close()
        posix_ipc.unlink_message_queue('/gp3_mq')
    finally:
        mq.close()
        posix_ipc.unlink_message_queue('/gp3_mq')