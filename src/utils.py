import os
import random

from models import GazeNet2
from dataset import GazeDataset

class Utils:
    class Eval:
        def createEvalDirs(savedModelsRoot: str, runId: str) -> str:
            runPath = os.path.join(savedModelsRoot, runId)

            if os.path.isdir(runPath) == False:
                raise FileNotFoundError
            
            evalPath = os.path.join(savedModelsRoot, runId, 'evals', 'eval0')
            os.makedirs(evalPath, exist_ok = True)
            evalId = len(os.listdir(evalPath))
            evalPath = os.path.join(evalPath, str(evalId))
            
            picklesPath = os.path.join(evalPath, 'pickles')
            pngsPath = os.path.join(evalPath, 'pngs')

            os.makedirs(evalPath, exist_ok = True)
            os.makedirs(picklesPath)
            os.makedirs(pngsPath)

            return evalPath, picklesPath, pngsPath
        
        def verifyEvalConfig(config: dict):
            requiredKeys = ('run_id', 'base_path_model', 'base_path_dataset')
            for key in requiredKeys:
                if key not in config:
                    raise KeyError(f'{key} is not in evalConfig')
                
        def mergeTrainEvalConfig(evalConfig: dict, trainConfig: dict):
            mergedConfig = {}

            mergedConfig['base_path_model'] = evalConfig['base_path_model']
            mergedConfig['base_path_dataset'] = evalConfig['base_path_dataset']
            mergedConfig['run_id'] = evalConfig['run_id']

            if 'videos' not in evalConfig:
                mergedConfig['videos'] = [random.choice(trainConfig['videos'])]
            else:
                mergedConfig['videos'] = evalConfig['videos']
            
            if 'viewers' not in evalConfig:
                mergedConfig['viewers'] = [random.choice(trainConfig['viewers'])]
            else:
                mergedConfig['viewers'] = evalConfig['viewers']
            
            mergedConfig['length'] = trainConfig['length']
            mergedConfig['model_type'] = trainConfig['model_type']
            mergedConfig['stride'] = trainConfig['stride']
            
            return mergedConfig
        
    class Common:
        def loadModel(config: dict):
            return GazeNet2(length = config['length'], 
                            model_type = config['model_type'],
                            stride = config['stride'])
            
        def loadDataset(config: dict):
            return GazeDataset(stride = config['stride'], 
                               length = config['length'],
                               videos = config['videos'] if 'videos' in config else [0],
                               rootPath = config['base_path_dataset'],
                               viewers = config['viewers'] if 'viewers' in config else [0])