import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # assert 
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)

    def attention(query, key, value, mask = None, dropout = None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, 1e-9)
        
        p_attn = scores.softmax(dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask = None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linear, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, mask = mask, dropout = self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )

        del query
        del key
        del value
        return self.linears[-1](x)

class ViViT(nn.Module):
    def __init__(self, embed_dim, p, c = 3, t = 32):
        super(ViVit, self).__init__()
        self.p = p
        self.c = c
        self.t = t
        self.linear = nn.Linear(in_features = c * p * p, out_features = embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, ))
    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.reshape(B, T, C, H // self.p, self.p, W // self.p, self.p)   # (B, T, C, H // P, P, W // P, P)
        x = x.permute(0, 1, 2, 3, 5, 4, 6)
        x = x.reshape(B, T, C, (H // self.p) * (W // self.p), P, P)    # (B, T, (H*W)/(P^2), P, P)
        x = x.permute(0, 1, 3, 2, 4, 5)  # (B, T, N, C, P, P)
        x = x.reshape(B, T, (H // self.p) * (w // self.p), -1)   # (B, T, (H*W)/(P^2), C * P ^ 2)
        x = x.permute(0, 2, 1, 3) 
        # N = (H * W) / (P ^ 2)
        x = self.linear(x)  # (B, N, T, embed_dim)





