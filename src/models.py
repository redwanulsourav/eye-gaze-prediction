import torch.nn as nn
import torch.nn.functional as F
import torch

class TemporalFeatureNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 32)
        self.relu1 = nn.ReLU()
        
    def forward(self, x):
        x = torch.flatten(x)
        x = x.unsqueeze(0)
        x = self.fc1(x)
        return x


class TemporalGazeNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=36, batch_first = True)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(36, 16)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        x = self.relu1(h_n)
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x

class GazePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=32+8, out_features=32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=16, out_features=8)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(in_features=8, out_features=4)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        return x


class GazeNet(nn.Module):
    def __init__(self, length, model_type: int):
        super().__init__()
        self.temporal_feature_net = TemporalFeatureNet(100352*length)
        self.temporal_gaze_net = TemporalGazeNet()
        self.gaze_predictor = GazePredictor()

        self.model_type = model_type

        if self.model_type == 0:
            pass
        if self.model_type == 1:    # Only image data
            for param in self.temporal_gaze_net.parameters():
                nn.init.constant(param, 0)
                param.requires_grad = False
        if self.model_type == 2:
            for param in self.temporal_feature_net.parameters():
                nn.init.constant(param, 0)
                param.requires_grad = False

    def forward(self, x_features, x_gaze):
        x_features = torch.flatten(self.temporal_feature_net(x_features))
        x_gaze = torch.flatten(self.temporal_gaze_net(x_gaze))
        x = torch.cat([x_features, x_gaze])
        x = x.unsqueeze(0)
        x = self.gaze_predictor(x)
        return x

class GazeEncoder(nn.Module):
    def __init__(self, out_dim):
        super(GazeEncoder, self).__init__()
        self.out_dim = out_dim

        self.linear_layers = nn.ModuleList([
            nn.ModuleList([nn.Linear(in_features = 2, out_features = out_dim[1]), nn.ReLU()]) for i in range(out_dim[0])])
    
    def forward(self, x):
        output = []
        for i in range(self.out_dim[0]):
            x_ = self.linear_layers[i][0](x)
            x_ = self.linear_layers[i][1](x_)
            output.append(x_)
        output = torch.stack(output, dim=1)
        return output

class GazeDecoder(nn.Module):
    def __init__(self, in_dim):
        super(GazeDecoder, self).__init__()
        self.in_dim = in_dim

        self.linear_layers = nn.ModuleList([
            nn.ModuleList([nn.Linear(in_features=in_dim[0], out_features = 1), nn.ReLU()]) for i in range(in_dim[1])
        ])

        self.readout = nn.ModuleList((nn.Linear(in_features=in_dim[1], out_features = 2), nn.ReLU()))

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        arr = []
        for i in range(self.in_dim[1]):
            arr.append(self.linear_layers[i][1](self.linear_layers[i][0](x[i])))

        assert(len(arr) == self.in_dim[0]) 
        x = torch.cat(arr, dim=1)
        return self.readout[1](self.readout[0](x)) 

class GazeEncoderDecoder(nn.Module):
    def __init__(self, dim):
        super(GazeEncoderDecoder, self).__init__()
        self.encoder = GazeEncoder(dim)
        self.decoder = GazeDecoder(dim)

    def forward(self, x):
        x = self.encoder.forward(x)
        x = self.decoder.forward(x)

        return x


























class TemporalFeatureNet2(nn.Module):
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


class GazeNet2(nn.Module):
    def __init__(self, length, model_type: int, stride: int):
        super().__init__()
        self.temporal_feature_net = TemporalFeatureNet2(100352*length)
        self.temporal_gaze_net = TemporalGazeNet2()
        self.gaze_predictor = GazePredictor2(stride)

        self.model_type = model_type

        if self.model_type == 0:
            pass
        if self.model_type == 1:    # Only image data
            for param in self.temporal_gaze_net.parameters():
                nn.init.constant(param, 0)
                param.requires_grad = False
        if self.model_type == 2:
            for param in self.temporal_feature_net.parameters():
                nn.init.constant(param, 0)
                param.requires_grad = False

    def forward(self, x_features, x_gaze):
        x = self.temporal_feature_net(x_features)
        x_gaze = self.temporal_gaze_net(x_gaze)
        x = torch.cat([x, x_gaze], dim = 1)
        x = self.gaze_predictor(x)
        return x