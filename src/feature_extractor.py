from torchvision import models
from collections import namedtuple
import torch
import torch.nn as nn
from torchvision import transforms

class vgg19_features(nn.Module):
    def __init__(self):
        super(vgg19_features, self).__init__()
        self.features = models.vgg19(pretrained=True).features
        self.feature_list = [nn.Sequential(x) for x in self.features]

        for parameter in self.parameters():
            parameter.requires_grad = False
    
    def forward(self, x):
        x = x.unsqueeze(0)
        conv1 = self.feature_list[0](x)
        relu1 = self.feature_list[1](conv1)
        conv2 = self.feature_list[2](relu1)
        relu2 = self.feature_list[3](conv2)
        pool1 = self.feature_list[4](relu2)

        conv3 = self.feature_list[5](pool1)
        relu3 = self.feature_list[6](conv3)
        conv4 = self.feature_list[7](relu3)
        relu4 = self.feature_list[8](conv4)
        pool2 = self.feature_list[9](relu4)

        conv4 = self.feature_list[10](pool2)
        relu5 = self.feature_list[11](conv4)
        conv5 = self.feature_list[12](relu5)
        relu5 = self.feature_list[13](conv5)
        conv6 = self.feature_list[14](relu5)
        relu6 = self.feature_list[15](conv6)
        conv7 = self.feature_list[16](relu6)
        relu7 = self.feature_list[17](conv7)
        pool3 = self.feature_list[18](relu7)

        conv8 = self.feature_list[19](pool3)
        relu8 = self.feature_list[20](conv8)
        conv9 = self.feature_list[21](relu8)
        relu9 = self.feature_list[22](conv9)
        conv10 = self.feature_list[23](relu9)
        relu10 = self.feature_list[24](conv10)
        conv11 = self.feature_list[25](relu10)
        relu11 = self.feature_list[26](conv11)
        pool4 = self.feature_list[27](relu11)

        return pool4
    

class ExtractFeatures():
    def __init__(self):
        self.feature_extractor_model = vgg19_features()
        self.transformation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def get_features(self, x):
        x = self.transformation(x)
        x = self.feature_extractor_model.forward(x)
        # print(f'{x.shape}')
        x = torch.flatten(x)
        return x
