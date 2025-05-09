import socket
import threading
from queue import Queue

from ObservableDict import ObservableDict

import os
import sys
import socket
import threading

import yaml
import json
import argparse
import random
import time
# import posix_ipc
import base64
import torch

from PIL import Image
import numpy as np

from torch.utils.data import DataLoader
import cv2

from gaze_dataset import GazeDataset

import torchvision.transforms as transforms


predictedGaze = ObservableDict('predictions')
actualGaze = ObservableDict('ground truths')

serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
serverSocket.bind(('', 12345))

clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
clientSocket.bind(('', 12346))

recvQueue = Queue()
sendQueue = Queue()

stopEvent = threading.Event()    

def blockingServer():
    print('Client server is up')
    serverSocket.listen()
    while not stopEvent.is_set():
        conn, addr = serverSocket.accept()
        while True:
            try:
                data = conn.recv(4096)
                if not data:
                    break
                if stopEvent.is_set():
                    break
                data = data.decode().rstrip('X')
                print(data)
                predictedData = np.frombuffer(base64.b64decode(data['predicted']))
                print(predictedData.shape)
                # data = json.loads(data)
                for frameIdx in range(predictedData['startIdx'], predictedData['startIdx'] + 64):
                    print(f'{frameIdx} inserted')
                    predictedGaze.updateKey(f'{frameIdx}x', gazeX)
                    predictedGaze.updateKey(f'{frameIdx}y', gazeY)
                print(f'Received: {data}')
            except Exception as e:
                print(e) 
    serverSocket.shutdown(socket.SHUT_RDWR)

def blockingClient():
    while True:
        try:
            clientSocket.connect(('', 12347))
            print('Connected with server')
            break
        except Exception as e:
            print('Server is not up yet')
        
    while not stopEvent.is_set():
        if sendQueue.empty() == False:
            # predictedGaze.changed = False
            data = sendQueue.get()
            data = data + '__'
            # print(f'sent {len(data.encode())} bytes')
            clientSocket.sendall(data.encode())
    try:
        clientSocket.shutdown(socket.SHUT_RDWR)
    except Exception as e:
        pass
    clientSocket.close()

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


def mediaPlayer(config):
    """ 
        Set up data.
    """
    fps = findFPS(config)
    frameRenderTime = float(1.0) / fps

    # gazeDataset = GazeDataset(
    #                             length =         config['length'],
    #                             videos =          config['videos'] if 'videos' in config else [0],
    #                             rootPath =       config['base_path'],
    #                             viewers =        config['viewers'] if 'viewers' in config else [0])
    # trainLoader = DataLoader(gazeDataset,  
                            # batch_size = config['batch_size'], 
                            # shuffle = config['shuffle'] if 'shuffle' in config else True)
            
    for i in range(0, 609):
        predictedGaze.updateKey(f'{i}at', 0)
        predictedGaze.updateKey(f'{i}x', 0.5)
        predictedGaze.updateKey(f'{i}y', 0.5)
    
    videoJSONPath = os.path.join(config['base_path'], 'processed', 'videos', 'video_order.json')

    with open(videoJSONPath) as f:
        contents = f.read()
    
    videoJSON = json.loads(contents)
    videoName = videoJSON[str(config['videos'][0])]
    cap = cv2.VideoCapture(os.path.join(config['base_path'], 'processed', 'videos', videoName))
    read, frame = cap.read()
    # systemTime = 0
    frame_delay = 1 / fps
    # n = len(gazeDataset)
    gazeData = json.load(open(os.path.join(config['base_path'], 'processed', 'gaze', 'gaze_order.json')))[str(config['videos'][0])]['0']
    # print(gazeData.keys())
    frameIndex = 0
    while read:
        # data = gazeDataset[i]
        start_time = time.time()
        print(f'FrameIdx: {frameIndex}')
        # print(gazeData[str(frameIndex)]['x'])
        currentGazeX, currentGazeY = round(gazeData[str(frameIndex)]['x'] * frame.shape[1]), round(gazeData[str(frameIndex)]['y'] * frame.shape[0])
        # print(currentGazeX, currentGazeY)
        """ Play i-th frame """
        # frame = data['features']
        predictedGazeX, predictedGazeY = round(predictedGaze[f'{frameIndex}x'] * frame.shape[1]), round(predictedGaze[f'{frameIndex}y'] * frame.shape[0])
        cv2.circle(frame, (currentGazeX, currentGazeY), 10, (255, 0, 0), 2)
        cv2.circle(frame, (predictedGazeX, predictedGazeY), 10, (0, 255, 0), 2)
        # print(frame.dtype)
        # print(type(frame))
        # print(frame.shape)
        cv2.imshow('Video', frame)
        elapsed_time = time.time() - start_time
        wait_time = max(1, int(frame_delay * 1000 - elapsed_time * 1000))
        # print(wait_time)
        frame = cv2.resize(frame, (224, 224))
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
        
        preprocess = transforms.Compose([
            transforms.Resize(256),              # Resize shorter side to 256
            transforms.CenterCrop(224),          # Then crop to 224x224
            transforms.ToTensor(),               # Convert to tensor [0,1]
            transforms.Normalize(                # Normalize using ImageNet stats
                mean=[0.485, 0.456, 0.406],       # RGB means
                std=[0.229, 0.224, 0.225]         # RGB stds
            )
        ])
        frame = preprocess(Image.fromarray(frame)).detach().numpy().astype(np.float16)

        tosend = {
            'gaze': (currentGazeX, currentGazeY),
            'frame': base64.b64encode(frame.tobytes()).decode('utf-8'),
            'frameIdx': i
        }
        # print(f'Sent {len(json.dumps(tosend))} bytes')
        sendQueue.put(json.dumps(tosend))
        # frameEncoded = base64.b64encode(frame.tobytes())
        # print(frameEncoded.find('__'))
        read, frame = cap.read()
        frameIndex += 1
        
        

        # """ Find the predicted gaze location """

        """ Send the server actual data """
        
        """ Calculate the error """

    cv2.destroyAllWindows()


    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--cfg', help = 'Train config [.yaml]', required = True)
    ap = ap.parse_args()
    config = parseConfig(ap.cfg)

    """ 
        Set up data.
    """


    th0 = threading.Thread(target = blockingServer)
    th1 = threading.Thread(target = blockingClient)
    try:
        th0.start()
        th1.start()
        mediaPlayer(config)
        while True:
            pass
    except Exception as e:
        print(f'Error: {e}')
        stopEvent.set()
        th0.join()
        th1.join()
        serverSocket.close()

    
        