import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, d_v):
        super(MultiHeadAttention, self).__init__()
        # assert 
        # NOTE: I don't think I will need mask for this.
        self.d_k = d_model // h
        self.h = h
        self.dropout = nn.Dropout(p = dropout)
        self.d_v = d_v
        self.W_q = nn.Linear(in_features = d_model, out_features = h * d_k)
        self.W_k = nn.Linear(in_features = d_model, out_features = h * d_k)
        self.W_v = nn.Linear(in_features = d_model, out_features = h * d_v)
        self.W_o = nn.Linear(in_features = h * d_v, out_features = d_model)

    def forward(self, x):
        B, T, F = x.shape
        query_embed = self.W_q(x)   # B, T, h * d_k
        key_embed = self.W_k(x)     # B, T, h * d_k
        value_embed = self.W_v(x)   # B, T, h * d_v

        # Split into `h` heads
        query = query_embed.view(B, T, self.h, self.d_k)
        query = query.permute(0, 2, 1, 3)   # B, h, T, d_k
        key = key_embed.view(B, T, self.h, self.d_k)
        key = key.permute(0, 2, 1, 3)   # B, h, T, d_k
        value = value_embed.view(B, T, self.h, d_v)
        value = key.permute(0, 2, 1, 3) # B, h, T, d_v

        # Calculate weights.
        attn_weights = torch.matmul(query, key.transpose(-2, -1))   # B, h, T, T
        attn_weights = attn_weights / torch.sqrt(d_k) # B, h, T, T

        attn_weights = attn_weights.softmax(dim = -1) # B, h, T, T

        encodings = torch.matmul(attn_weights, value) # B, h, T, d_v
        encodings = encodings.permute(0, 2, 1, 3) # B, T, h, d_v
        encodings = encodings.view(0, 1, -1); # B, T, h * d_v
        encodings = self.w_o(encodings) # B, T, d_model
        return encodings
        
class LayerNorm(nn.Module):
    def __init__(self, features, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MLP(nn.Module):
    def __init__(self, embed_dim, inner_layer):
        super(MLP, self).__init__()
        self.linear0 = nn.Linear(in_features = embed_dim, out_features = inner_layer)
        self.linear1 = nn.Linear(in_features = inner_layer, out_features = embed_dim)

    def forward(self, x):
        return self.linear1(self.linear0(x))    # TODO: Prolly should apply dropout here.
        

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(TransformerEncoder, self).__init__()
        """
            Each transformer encoder contains two sub-layers, a multihead attention followed by an MLP.
        """
        self.embed_dim = embed_dim
        self.multi_head_attention = MultiHeadAttention(embed_dim, 8, 512) # TODO
        self.feed_forward = MLP(embed_dim, 1024)
        self.layer_norm0 = LayerNorm(embed_dim)
        self.layer_norm1 = LayerNorm(embed_dim)

    def forward(self, x):
        B, N, embed_dim = x.shape
        x = self.multi_head_attention(x) + x  # B, N, embed_dim
        x = self.layer_norm0(x) # B, N, embed_dim
        x = self.feed_forward(x) + x  # B, N, embed_dim
        x = self.layer_norm1(x) # B, N, embed_dim
        return x # B, N, embed_dim

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim


class SMapEncoder(nn.Module):
    def __init__(self, latent_dim = 512, p = 8, h = 64, w = 64):
        super(SMapEncoder, self).__init__()
        self.p = p
        self.latent_dim = latent_dim
        self.transformer = torch.Sequential([TransformerEncoder(latent_dim) for i in range(6)])
        self.out_mlp = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, x):
        B, H, W = x.shape
        x = x.view(B, H // self.p, self.p, W // self.p, self.p) # shape (B, H // self.p, p, W // self.p, p)
        x = x.permute(0, 1, 3, 2, 4) # shape (B, T, H //self.p, W // self.p, p, p)
        x = x.view(B, H // self.p * W // self.p, self.p * self.p) # shape (B, T, N, P ^ 2)
        # TODO: Append label encoding here. 
        x = self.transformer(x) # shape (B, N + 1, d_v)
        y = out_mlp(x[:, 0, :])
        return y

class SMapDecoder(nn.Module):
    def __init__(self, latent_dim = 512, p = 8, h = 64, w = 64):
        super(SMapDecoder, self).__init__()

        
        
class ViViT(nn.Module):
    def __init__(self, embed_dim, p, c = 3, t = 32, h = 64, w = 64):
        super(ViViT, self).__init__()
        self.p = p
        self.c = c
        self.t = t
        self.h = h
        self.w = w
        self.linear = nn.Linear(in_features = c * p * p, out_features = embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, h//p * w //p * t, embed_dim))
        self.transformer_encoders = torch.Sequential([TransformerEncoder(embed_dim) for i in range(6)])

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B, T, C, H // self.p, self.p, W // self.p, self.p)   # (B, T, C, H // P, P, W // P, P)
        x = x.permute(0, 1, 2, 3, 5, 4, 6) # (B, T, C, H // self.p, W // self.p, p, p)
        x = x.view(B, T, C, (H // self.p) * (W // self.p), P, P)    # (B, T, C, (H*W)/(P^2), P, P)
        x = x.permute(0, 1, 3, 2, 4, 5)  # (B, T, N, C, P, P)
        x = x.view(B, T, (H // self.p) * (W // self.p), -1)   # (B, T, N, C * P ^ 2)
        x = x.permute(0, 2, 1, 3) # (B, N, T, C * P^2)
        x = x.view(B, T * (H // self.p) * (W // self.p), -1) # (B, T * N, C * P^2)
        # N = (H * W) / (P ^ 2)
        x = self.linear(x)  # (B, T * N, embed_dim)
        # TODO: Put the target encoding here.
        x = x + self.position_embedding         # (B, T * N, embed_dim)

        x = self.transformer_encoders(x) # (B, T * N, embed_dim)




