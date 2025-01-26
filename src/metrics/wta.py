import numpy as np

def wta(saliency_map):
    """
        Returns the (x, y) with highest probability
    """

    return np.argmax(saliency_map)