import torch
import torch.nn as nn

class FrameGenerator(nn.Module):
    def __init__(self):
        super(FrameGenerator, self).__init__()

        self.latent_representation_generator = nn.Sequential(
            # (batcn, 3, 64, 64)
            nn.Conv2d(3, 128, kernel_size = 4, stride = 2, padding = 1),    
            # (batch, 128, 32, 32)
            nn.ReLU(inplace = True),
            # (batch, 128, 32, 32)
            nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 256, 16, 16)
            nn.BatchNorm2d(256, eps=1e-3),
            # (batch, 256, 16, 16)
            nn.ReLU(inplace = True),
            # (batch, 256, 16, 16)
            nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 512, 8, 8)
            nn.BatchNorm2d(512, eps=1e-3),
            # (batch, 512, 8, 8)
            nn.ReLU(inplace = True),
            # (batch, 512, 8, 8)
            nn.Conv2d(512, 1024, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 1024, 4, 4)
            nn.BatchNorm2d(1024, eps = 1e-3),
            # (batch, 1024, 4, 4)
            nn.ReLU(inplace = True)
            # (batch, 1024, 4, 4)
        )

        self.background_generator = nn.Sequential(
            # (batch, 1024, 1, 4, 4)
            nn.ConvTranspose3d(1024, 1024, (2, 1, 1)),
            # (batch, 1024, 2, 4, 4)
            nn.BatchNorm3d(1024),
            # (batch, 1024, 2, 4, 4)
            nn.ReLU(inplace = True),
            # (batch, 1024, 2, 4, 4)
            nn.ConvTranspose3d(1024, 512, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 512, 2, 8, 8)
            nn.BatchNorm3d(512),
            # (batch, 512, 2, 8, 8)
            nn.ReLU(inplace = True),
            # (batch, 512, 2, 8, 8)
            nn.ConvTranspose3d(512, 256, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 256, 4, 16, 16)
            nn.BatchNorm3d(256),
            # (batch, 256, 4, 16, 16)
            nn.ReLU(inplace = True),
            # (batch, 256, 4, 16, 16)
            nn.ConvTranspose3d(256, 128, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 128, 8, 32, 32)
            nn.BatchNorm3d(128),
             # (batch, 128, 8, 32, 32)
            nn.ReLU(inplace = True),
             # (batch, 128, 8, 32, 32)
            nn.ConvTranspose3d(128, 3, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 3, 32, 64, 64)
            nn.Tanh()
        )

        self.foreground_backbone = nn.Sequential(
            # (batch, 1024, 1, 4, 4)
            nn.ConvTranspose3d(1024, 1024, (2, 1, 1)),
            # (batch, 1024, 2, 4, 4)
            nn.BatchNorm3d(1024),
            # (batch, 1024, 2, 4, 4)
            nn.ReLU(inplace = True),
            # (batch, 1024, 2, 4, 4)
            nn.ConvTranspose3d(1024, 512, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 512, 4, 8, 8)
            nn.BatchNorm3d(512),
            # (batch, 512, 4, 8, 8)
            nn.ReLU(inplace = True),
            # (batch, 512, 4, 8, 8)
            nn.ConvTranspose3d(512, 256, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 256, 8, 16, 16)
            nn.BatchNorm3d(256),
            # (batch, 256, 8, 16, 16)
            nn.ReLU(inplace = True),
            # (batch, 256, 8, 16, 16)
            nn.ConvTranspose3d(256, 128, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 256, 16, 32, 32)
            nn.BatchNorm3d(128),
            # (batch, 256, 16, 32, 32)
            nn.ReLU(inplace = True)
            # (batch, 256, 16, 32, 32)
        )

        self.foreground_mask_generator = nn.Sequential(
            # (batch, 256, 16, 32, 32)
            nn.ConvTranspose3d(128, 1, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 1, 32, 64, 64)
            nn.Sigmoid()
            # (batch, 1, 32, 64, 64)
        )

        self.foreground_frame_generator = nn.Sequential(
            # (batch, 256, 16, 32, 32)
            nn.ConvTranspose3d(128, 3, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 3, 32, 64, 64)
            nn.Tanh()
            # (batch, 3, 32, 64, 64)
        )


    def forward(self, x):
        # (batch, 3, 64, 64)
        latent = self.latent_representation_generator(x)
        # (batch, 1, 1024, 4, 4)
        latent = latent.view(-1, 1024, 1, 4, 4)
        # (batch, 1024, 1, 4, 4)
        background = self.background_generator(latent)
        # (batch, 3, 32, 64, 64)
        foreground_backbone_output = self.foreground_backbone(latent)
        # (batch, 256, 16, 32, 32)
        foreground_mask = self.foreground_mask_generator(foreground_backbone_output)
        # (batch, 1, 32, 64, 64)
        
        background_mask = 1 - foreground_mask
        # (batch, 1, 32, 64, 64)

        background = background * background_mask
        # (batch, 3, 32, 64, 64)
        
        foreground = self.foreground_frame_generator(foreground_backbone_output)
        # (batch, 3, 32, 64, 64)

        foreground = foreground * foreground_mask
        # (batch, 3, 32, 64, 64)

        output = foreground + background
        # (batch, 3, 32, 64, 64)
        
        return output
    
class TemporalSaliencyPredictor(nn.Module):
    def __init__(self):
        super(TemporalSaliencyPredictor, self).__init__()

        self.layers = nn.Sequential(
            # (batch, 3, 32, 64, 64)
            nn.Conv3d(3, 128, kernel_size = 3, stride = 1, padding = 1),
            # (batch, 128, 32, 64, 64)
            nn.ReLU(inplace = True),
            # (batch, 128, 32, 64, 64)
            nn.Conv3d(128, 256, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 256, 16, 32, 32)
            nn.ReLU(inplace = True),
            # (batch, 256, 16, 32, 32)
            nn.Conv3d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            # (batch, 256, 16, 32, 32)
            nn.ReLU(inplace = True),
            # (batch, 256, 16, 32, 32)
            nn.Conv3d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            # (batch, 256, 16, 32, 32)
            nn.ReLU(inplace = True),
            # (batch, 256, 16, 32, 32)
            nn.ConvTranspose3d(256, 1, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 1, 32, 64, 64)
            nn.ReLU(inplace = True)
            # (batch, 1, 32, 64, 64)
        )
    
    def forward(self, x):
        # (batch, 3, 32, 64, 64)
        batch_size = x.shape[0]
        frame_size = x.shape[2]
        img_width = x.shape[4]
        img_height = x.shape[3]
        
        x = self.layers(x)
        # (batch, 1, 32, 64, 64)
        x = x.squeeze()
        # (batch, 32, 64, 64)
        
        x = x.view(batch_size, frame_size, -1)
        # (batch, 32, 4096)
        x = torch.transpose(x, 0, 2)
        # (4096, 32, batch)
        
        x = nn.functional.log_softmax(x)
        # (4096, 32, batch)
        
        x = torch.transpose(x, 0, 2)
        # (batch, 32, 4096)
        
        x = x.view(batch_size, 1, frame_size, img_width, img_height)
        # (batch, 1, 32, 64, 64)
        
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            # (batch, 1, 32, 64, 64)
            nn.Conv3d(3, 128, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 128, 16, 32, 32)
            nn.LeakyReLU(0.2, inplace = True),
            # (batch, 128, 16, 32, 32)
            nn.Conv3d(128, 256, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 256, 8, 16, 16)
            nn.BatchNorm3d(256, eps = 1e-3),
            # (batch, 256, 8, 16, 16)
            nn.LeakyReLU(0.2, inplace = True),
            # (batch, 256, 8, 16, 16)
            nn.Conv3d(256, 512, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 512, 4, 8, 8)
            nn.BatchNorm3d(512, eps = 1e-3),
            # (batch, 512, 4, 8, 8)
            nn.LeakyReLU(0.2, inplace = True),
            # (batch, 512, 4, 8, 8)
            nn.Conv3d(512, 1024, kernel_size = 4, stride = 2, padding = 1),
            # (batch, 1024, 2, 4, 4)
            nn.BatchNorm3d(1024, eps = 1e-3),
            # (batch, 1024, 2, 4, 4)
            nn.LeakyReLU(0.2, inplace = True),
            # (batch, 1024, 2, 4, 4)
            nn.Conv3d(1024, 2, kernel_size = (2, 4, 4), stride = 1, padding = 0)
            # (batch, 2, 1, 1, 1)
        )
    
    def forward(self, x):
        # (batch, 3, 32, 64, 64)
        batch_size = x.shape[0]
        frame_size = x.shape[2]
        img_width = x.shape[4]
        img_height = x.shape[3]
        
        x = self.layers(x)
        # (batch, 2, 1, 1, 1)
        
        x = x.view(batch_size, 2)
        x = torch.nn.functional.softmax(x, dim = 1)
        # (batch, 2)

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
