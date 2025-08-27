import torch
import numpy as np

def auc_borji(predicted, ground_truth, device = torch.device('cpu')):
    """
        Calculates the AUC-Borji score of `predicted`.
        Inputs:
            predicted: A `torch.tensor` object that holds predicted saliency map. It should have (1, h, w) dim. Otherwise throws `ValueError`.
            ground_truth: A `torch.tensor` object that holds ground truth saliency map. It should have (1, h, w) dim. Otherwise throws `ValueError`.

        Output:
            Outputs the AUC-Borji score for single frame.
    """
    
    if not isinstance(predicted, (torch.tensor)):
        raise TypeError('The first parameter should be a pytorch tensor')

    if not isinstance(ground_truth, (torch.tensor)):
        raise TypeError('The second parameter should be a pytorch tensor')

    if len(predicted.shape) != 3 or len(ground_truth.shape) != 3:
        raise ValueError(f'Both parameters should have dim of length 3, found {len(predicted.shape)} and {len(ground_truth.shape)}')

    if predicted.shape[0] != 1 or ground_truth.shape[0] != 1:
        raise ValueError(f'expected the first dim to be channels, expected 1 channel')
    
    if predicted.shape != ground_truth.shape:
        raise ValueError('Both inputs should have same shape')

    # nFrame = pred1.shape[0]
    # totalMean = 0
    # for i in range(nFrame):
        # pred = pred1[i]
        # gt = gt1[i]
    n_total = ground_truth.shape[0] * ground_truth.shape[1]
    n_pve = len(torch.where(gt > 0))
    # print('nPve: ', nPve)
    a = torch.tensor(predicted[ground_truth > 0].flatten()).to(device)
    auc_mean = 0
    for kk in range(100):
        random_pts = torch.randint(0, nTotal, (n_pve, )).to(device)
        values = []
        for j in random_pts:
            x = j // ground_truth.shape[1]
            y = j % ground_truth.shape[0]
            values.append(predicted[x, y])
        values = torch.tensor(values)

        roc_values = []
        roc_values.append((0.0, 0.0))
        thresh = 0
        while thresh <= 1:
            x = len(torch.where(values > thresh)[0]) / n_pve
            y = len(torch.where(a > thresh)[0]) / n_pve
            roc_values.append((y, x))
            thresh += 0.01
        roc_values.append((1.0, 1.0))
        roc_values.sort(key = lambda x: x[0])
        roc_valuesX = torch.tensor([x[1] for x in roc_values]).to(device)
        roc_valuesY = torch.tensor([x[0] for x in roc_values]).to(device)

        auc = torch.trapz(rocValuesY, rocValuesX)
        auc_mean += auc
    auc_mean /= 100

    return auc_mean

