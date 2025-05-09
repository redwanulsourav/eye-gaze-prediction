import torch.nn as nn
import torch
from torchvision import models

class CustomLSTMCell(nn.Module):
    def __init__(self, hiddenDim):
        super(CustomLSTMCell, self).__init__()
        resnet18 = models.resnet18(pretrained = True)
        self.feature_extractor = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4 
        )
        self.conv = nn.Conv2d(512, 1, 1)
        self.linear1 = nn.Linear(in_features = 49, out_features = 32)
        self.relu = nn.ReLU(inplace = True)
        self.layerNorm = nn.LayerNorm(32,)
        self.linear2 = nn.Linear(in_features = 2, out_features = 32)
        self.layerNorm2 = nn.LayerNorm(32,)
        self.linear3 = nn.Linear(64, hiddenDim)

    def forward(self, frame, gaze):
        B = frame.shape[0]
        # print(frame.shape)
        frame = self.feature_extractor(frame)   # (B, 512, 7, 7)
        
        frame = self.conv(frame)    # (B, 1, 7, 7)
        frame = frame.view(B, 49)
        
        x = self.linear1(frame) # (B, 32)
        x = self.layerNorm(x) # (B, 32)
        x = self.relu(x)    # (B, 32)

        xx = self.linear2(gaze)
        xx = self.layerNorm2(xx)
        xx = self.relu(xx)  # (B, 32)

        x = torch.concat([x, xx], dim = 1) # (B, 64)

        x = self.linear3(x) # (B, hiddenDim)
        return x

class CustomLSTM(nn.Module):
    def __init__(self, hiddenDim):
        super(CustomLSTM, self).__init__()
        self.ccell = CustomLSTMCell(hiddenDim)
        self.lstm = nn.LSTM(hiddenDim, 64, num_layers = 3, batch_first = True)
        self.linear = nn.Linear(64, 128)
    
    def forward(self, xFrames, xGaze):

        """ xFrames.shape = (B, T, C, H, W) 
            xGaze.shape = (B, T, 2) """
        
        T = xFrames.shape[1]
        self.hiddenOuts = []
        # print(f'xFrames.shape: {xFrames.shape}')
        # print(f'xGaze.shape: {xGaze.shape}')

        for i in range(T):
            h = self.ccell(xFrames[:, i, :, :, :], xGaze[:, i, :])  # (B, hiddenDim)
            self.hiddenOuts.append(h)
            # print(h.shape)
        self.hiddenOuts = torch.stack(self.hiddenOuts)
        self.hiddenOuts = self.hiddenOuts.permute(1, 0, 2)
        # print(f'hiddenOuts.shape: {self.hiddenOuts.shape}')
        out, (h_n, c_n) = self.lstm(self.hiddenOuts)
        out = self.linear(out)
        # print(f'outshape: {out.shape}')
        # print(f'outshape2: {out[:, -1, :].unsqueeze(1).shape}')
        out = out[:, -1, :].unsqueeze(1)
        out = out.view(1, 64, 2)
        # out = out
        out = torch.sigmoid(out)
        # print(f'outshape: {out.shape}')
        return out

if __name__ == '__main__':
    xFrames = torch.randn(1, 64, 3, 224, 224)
    xGaze = torch.randn(1, 64, 2)

    model = CustomLSTM(32)
    out = model(xFrames, xGaze)