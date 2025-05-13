import torch.nn as nn
import torch.nn.functional as F
import torch


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet18FeatureExtractor, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet18.children())[:-1]) 

    def forward(self, x):
        x = self.features(x)  # (batch_size, 512, 1, 1)
        x = torch.flatten(x, 1)  # (batch_size, 512)
        return x


class TemporalFeatureNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 32)
        self.relu1 = nn.ReLU()
        self.layerNorm1 = nn.LayerNorm((32,))
    def forward(self, x):
        x = torch.flatten(x, start_dim = 1)
        x = self.fc1(x)
        x = self.layerNorm1(x)
        x = torch.tanh(x)
        return x


class TemporalGazeNet2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=36, batch_first = True)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(36, 16)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.layerNorm1 = nn.LayerNorm((8,))
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        h_n = h_n.squeeze(0)
        x = self.relu1(h_n)
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        x = self.layerNorm1(x)
        x = torch.tanh(x)
        return x

class GazePredictor2(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
        self.fc1 = nn.Linear(in_features=32+8, out_features=32)
        self.layerNorm1 = nn.LayerNorm((32,))
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.layerNorm2 = nn.LayerNorm((16,))
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=16, out_features=8)
        self.layerNorm3 = nn.LayerNorm((8,))
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(in_features=8, out_features=2 * stride)
        self.layerNorm4 = nn.LayerNorm((2 * stride,))
        self.relu4 = nn.ReLU()
        
    def forward(self, x):
        x = self.relu1(self.layerNorm1(self.fc1(x)))
        x = self.relu2(self.layerNorm2(self.fc2(x)))
        x = self.relu3(self.layerNorm3(self.fc3(x)))
        x = self.layerNorm4(self.fc4(x))
        x = x.view(-1, self.stride, 2)
        x = torch.sigmoid(x)
        return x


class GazeNet(nn.Module):
    def __init__(self, length, model_type: int, stride: int):
        super(GazeNet, self).__init__()
        self.temporalFrameNet = TemporalFeatureNet()
        self.temporalGazeNet = TemporalGazeNet()
        self.gazePredictor = GazePredictor()

    def forward(self, xFrame, xGaze):
        xFrame = self.temporalFrameNet(xFrame)
        xGaze = self.temporalGazeNet(xGaze)
        x = torch.cat([x, x_gaze], dim = 1)
        x = self.gazePredictor(x)
        return x