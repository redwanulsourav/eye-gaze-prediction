import torch
import torch.nn as nn
import numpy as np


if __name__ == '__main__':
    pe = torch.zeros(5, 11)
    x = torch.arange(0, 5).unsqueeze(1)
    print(pe.size(1))
    pe[:, 0::2] = 1
    pe[:, 1::2] = 0
    print(pe.shape)
    print(pe)

    # x = x.transpose(2, 1)

    # print(x.shape)
    # print(x)

    # x = nn.functional.softmax(x, dim = 1)

    # print(x)
