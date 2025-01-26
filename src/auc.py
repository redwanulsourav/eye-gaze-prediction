import numpy as np
from sklearn.metrics import roc_auc_score

def compute_auc(saliency_map, fixation_map):
    # Flatten the maps
    saliency_values = saliency_map.flatten()
    fixation_values = fixation_map.flatten()
    
    # Binary ground truth: 1 for fixations, 0 otherwise
    labels = (fixation_values > 0).astype(int)
    
    return roc_auc_score(labels, saliency_values)
    