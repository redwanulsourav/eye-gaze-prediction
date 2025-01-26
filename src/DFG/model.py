import torch
import torch.nn as nn

class FrameGenerator(nn.Module):
    def __init__(self):
        super(FrameGenerator, self).__init__()

        self.latent_representation_generator = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(256, eps=1e-3),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(512, eps=1e-3),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 1024, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(1024, eps = 1e-3),
            nn.ReLU(inplace = True)
        )

        self.background_generator = nn.Sequential(
            nn.ConvTranspose3d(1024, 1024, (2, 1, 1)),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace = True),
            nn.ConvTranspose3d(1024, 512, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace = True),
            nn.ConvTranspose3d(512, 256, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace = True),
            nn.ConvTranspose3d(256, 128, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace = True),
            nn.ConvTranspose3d(128, 3, kernel_size = 4, stride = 2, padding = 1),
            # nn.BatchNorm3d(3),
            # nn.ReLU(inplace = True),
            nn.Tanh()
        )

        self.foreground_backbone = nn.Sequential(
            nn.ConvTranspose3d(1024, 1024, (2, 1, 1)),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace = True),
            nn.ConvTranspose3d(1024, 512, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace = True),
            nn.ConvTranspose3d(512, 256, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace = True),
            nn.ConvTranspose3d(256, 128, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace = True)
        )

        self.foreground_mask_generator = nn.Sequential(
            nn.ConvTranspose3d(128, 1, kernel_size = 4, stride = 2, padding = 1),
            nn.Sigmoid()
        )

        self.foreground_frame_generator = nn.Sequential(
            nn.ConvTranspose3d(128, 3, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )


    def forward(self, x):
        latent = self.latent_representation_generator(x)
        print(latent.shape)
        latent = latent.view(-1, 1024, 1, 4, 4)
        print(f'latent.shape: {latent.shape}')
        background = self.background_generator(latent)
        foreground_backbone_output = self.foreground_backbone(latent)
        foreground_mask = self.foreground_mask_generator(foreground_backbone_output)
        print(f'foreground_mask.shape: {foreground_mask.shape}')
        background_mask = 1 - foreground_mask
        print(f'background_mask.shape: {background_mask.shape}')
        print(f'background.shape: {background.shape}')
        background = background * background_mask
        foreground = self.foreground_frame_generator(foreground_backbone_output)
        print(f'foreground.shape: {foreground.shape}')
        print(f'foreground_mask.shape: {foreground_mask.shape}')
        foreground = foreground * foreground_mask
        
        output = foreground + background

        return output
    
class TemporalSaliencyPredictor(nn.Module):
    def __init__(self):
        super(TemporalSaliencyPredictor, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv3d(3, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv3d(128, 256, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv3d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv3d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.ConvTranspose3d(256, 1, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(inplace = True)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        frame_size = x.shape[2]
        img_width = x.shape[4]
        img_height = x.shape[3]
        print(x.shape)
        x = self.layers(x)
        print(x.shape)
        x = x.squeeze()
        print(x.shape)
        
        x = x.view(batch_size, frame_size, -1)
        print(x.shape)
        
        x = torch.transpose(x, 0, 2)
        # x = nn.functional.softmax(x)
        print(x.shape)
        
        x = nn.functional.log_softmax(x)
        print(x.shape)
        
        x = torch.transpose(x, 0, 2)
        print(x.shape)
        x = x.view(batch_size, 1, frame_size, img_width, img_height)

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv3d(3, 128, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv3d(128, 256, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(256, eps = 1e-3),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv3d(256, 512, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(512, eps = 1e-3),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv3d(512, 1024, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(1024, eps = 1e-3),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv3d(1024, 2, kernel_size = (2, 4, 4), stride = 1, padding = 0)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        frame_size = x.shape[2]
        img_width = x.shape[4]
        img_height = x.shape[3]
        print(x.shape)
        x = self.layers(x)
        print(x.shape)
        x = x.view(2,-1)

        return x

class DFG(nn.Module):
    def __init__(self):
        super(DFG, self).__init__()

        self.frame_generators = FrameGenerator()
        self.saliency_predictor = TemporalSaliencyPredictor()
        self.discriminator = Discriminator()
    
    def forward_generate(self, x):
        rgb_frames = self.frame_generators(x)
        saliency_maps = self.saliency_predictor(rgb_frames)

        return (rgb_frames, saliency_maps)
    
    def forward_discriminate(self, x):
        x = self.discriminator(x)
        return x
