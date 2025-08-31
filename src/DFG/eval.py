"""
    Script to evaluate a model.
    Input: A trained model and a dataset
    Output: Score according to the eval metric.
"""
import models
import torch
import argparse
import yaml
from dfg_dataset import DFG_GTEA_Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def aucBorji(pred1, gt1):
    nFrame = pred1.shape[0]
    totalMean = 0
    print(f'gt mean: {gt1.min()}')
    print(f'gt max: {gt1.max()}')
    print(f'pred1 max: {pred1.max()}')
    print(f'pred1 min: {pred1.min()}')
    for i in range(nFrame):
        pred = pred1[i]
        gt = gt1[i]

        nTotal = gt.shape[0] * gt.shape[1]
        nPve = len(torch.where(gt > 0)[0])
        print('nPve: ', nPve)
        a = torch.tensor(pred[gt > 0].flatten()).to(device)
        aucMean = 0
        for kk in range(100):
            randomPts = torch.randint(0, nTotal, (nPve, )).to(device)
            values = []
            for j in randomPts:
                x = j // gt.shape[1]
                y = j % gt.shape[0]
                values.append(pred[x, y])
            values = torch.tensor(values)

            rocValues = []
            rocValues.append((0.0, 0.0))
            thresh = 0
            while thresh <= 1:
                # print(thresh)
                x = len(torch.where(values > thresh)[0]) / nPve
                y = len(torch.where(a > thresh)[0]) / nPve
                rocValues.append((y, x))
                thresh += 0.01
            rocValues.append((1.0, 1.0))
            rocValues.sort(key = lambda x: x[0])
            rocValuesX = torch.tensor([x[1] for x in rocValues]).to(device)
            rocValuesY = torch.tensor([x[0] for x in rocValues]).to(device)

            auc = torch.trapz(rocValuesY, rocValuesX)
            aucMean += auc
        aucMean /= 100
        totalMean += aucMean

    return totalMean / nFrame

def evalBorji(config: dict):
    # Initialize model.
    generator = models.FrameGenerator().to(device)
    salPredictor = models.TemporalSaliencyPredictor().to(device)

    # Load pretrained models.
    generator.load_state_dict(torch.load(config['pretrained_generator']))
    salPredictor.load_state_dict(torch.load(config['pretrained_sal_predictor']))
    
    generator.eval()
    salPredictor.eval()

    # Initialize dataset
    testData = DFG_GTEA_Dataset(
                                    videos=          config['videos'] if 'videos' in config else [0],
                                    data_root =       config['base_path'], t_sample = 20)

    testLoader = DataLoader(testData, batch_size = config['batch_size'], shuffle = config['shuffle'] if 'shuffle' in config else True)

    scoreSum = 0
    for i, data in enumerate(testLoader):
        frames = data['input_frame'].to(device)
        fixationMap = data['tgt_gaze'].to(device)
        genOut = generator(frames)
        salOut = salPredictor(genOut).squeeze()
        fixationMap = fixationMap.squeeze()
        T, H, W = fixationMap.shape
        salOut = salOut.view(T, H * W)
        fixationMap = fixationMap.view(T, H * W) 

        fixationMap = fixationMap.masked_fill(fixationMap == 0, float('-inf'))
        salOut = salOut.masked_fill(salOut == 0, float('-inf'))    
        fixationMap = torch.nn.functional.softmax(fixationMap, dim = -1)
        salOut = torch.nn.functional.softmax(salOut, dim = -1)
        fixationMap = fixationMap.view(T, H, W)
        salOut = salOut.view(T, H, W)

        
        score = aucBorji(salOut, fixationMap)
        scoreSum += score
        print(f'Sample {i}: {score}')
    scoreSum /= len(testLoader)
    print(f'AUCBorji: {scoreSum}')

    return scoreSum

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--cfg', help = 'Eval config [.yaml]', required = True)
    
    ap = ap.parse_args()

    with open(ap.cfg) as f:
        config = yaml.safe_load(f)

    """
        Assumption: `config` contains the dataset description and the model path, and the eval metric
    """

    if config['eval_method'] == 'AUCBorji':
        evalBorji(config)
    
