import torch
import torch.nn as nn

class ViViT(nn.Module):
    def __init__(self, p, embed_dim):
        self.p = p
        self.linear = nn.Linear(in_features = p * p, out_features = embed_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.reshape(B, T, C, H // self.p, self.p, W // self.p, self.p)   # (B, T, C, H // P, P, W // P, P)
        x = x.permute(0, 1, 2, 3, 5, 4, 6)
        x = x.reshape(B, T, C, (H // self.p) * (W // self.p), P, P)    # (B, C, (H*W)/(P^2), P, P)
        x = self.linear(x)  # (B, T, C, (H * W)/(P^2) P, embed_dim)


