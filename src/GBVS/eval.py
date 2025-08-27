"""
    Generates saliency maps for frames.
"""

# sys hack to import metrics
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'metrics'))

import torch
import argparse
import yaml
import auc_borji
import json
import cv2
import shutil

from gbvs import GBVS

def parse_yaml(yaml_path):
    """
        Parses the config .yaml
        Required Keys:
            - base_path
            - eval_name
            - output_path
            - videos
            - tsampling_rate

        Parameters:
            yaml_path (str): Path to the .yaml config
        Returns:
            dict: Dictionary containing the values of the keys.
    """

    if not isinstance(yaml_path, str):
        raise TypeError('yaml_path should be of type str')

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    required_keys = ['base_path', 'eval_name', 'output_path', 'videos', 'tsampling_rate']

    for key in required_keys:
        if key not in list(config.keys()):
            raise KeyError(f'Value of {key} not found [Required]')

    return config

def prepare_output_dirs(config):
    """
        Prepares the output dirs for storing generated frames

        Parameters:
            config (dict): The configuration dictionary
        Returns:
            None
    """

    try:
        os.mkdir(config['output_path'])
    except FileExistsError:
        # Report in a log
        pass

    try:
        os.mkdir(os.path.join(config['output_path'], config['eval_name']))
    except FileExistsError:
        # Report in a log
        pass
    
    for vidx in config['videos']:
        try:
            os.mkdir(os.path.join(config['output_path'], config['eval_name'], str(vidx)))
        except:
            # Report in a log
            pass

def get_model_outputs(config, config_path):
    """
        Generate model outputs and saves them.
        Input:
            config (dict): The config with required values
        Returns:
            None
    """

    prepare_output_dirs(config)
    
    model = GBVS()

    frames_root_dir = os.path.join(config['base_path'], 'processed', 'frames')
    videos_json_path = os.path.join(config['base_path'], 'processed', 'videos', 'video_order.json')
    shutil.copy(config_path, os.path.join(config['output_path'], config['eval_name'], 'config.yaml'))

    with open(videos_json_path, 'r') as f:
        video_order = json.load(f)
    
    for vidx in config['videos']:
        video_name = video_order[str(vidx)].split('.')[0]
        frames_path = os.path.join(frames_root_dir, video_name, f'm{str(config["tsampling_rate"])}')
        frames_output_path = os.path.join(config['output_path'], config['eval_name'], str(vidx))
        frame_cnt = len(os.listdir(frames_path))
        for fidx in range(frame_cnt):
            img = cv2.imread(os.path.join(frames_path, f'{str(fidx)}.png'))
            img = cv2.resize(img, (1024, 1024))
            model_output = model.forward(img)
            cv2.imwrite(os.path.join(frames_output_path, f'{fidx}.png'), model_output)
            print(f'Video [{vidx}]/[{len(config["videos"])}], Frame [{fidx}]/[{frame_cnt}] saved.')
    return None 

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--cfg', help = 'Path to eval cfg', required = True)
    ap = ap.parse_args()

    config = parse_yaml(ap.cfg)

    get_model_outputs(config, ap.cfg)

