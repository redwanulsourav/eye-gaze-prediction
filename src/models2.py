import torch.nn as nn
import torch
from torchvision import transforms
from torchsummary import summary
from torchvision import models

import numpy as np

class VGG19_FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19_FeatureExtractor, self).__init__()
        self.features = models.vgg19(pretrained=True).features
        self.feature_list = [nn.Sequential(x) for x in self.features]

        for parameter in self.parameters():
            parameter.requires_grad = False
    
    def forward(self, x):
        # x = x.unsqueeze(0)
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
        # Shape is 28x28x256
        conv8 = self.feature_list[19](pool3)
        relu8 = self.feature_list[20](conv8)
        conv9 = self.feature_list[21](relu8)
        relu9 = self.feature_list[22](conv9)
        conv10 = self.feature_list[23](relu9)
        relu10 = self.feature_list[24](conv10)
        conv11 = self.feature_list[25](relu10)
        relu11 = self.feature_list[26](conv11)
        pool4 = self.feature_list[27](relu11)

        return pool3

class Vectorizer(nn.Module):
    """
        Module responsible for outputting a single vector, for an RGB image
    """
    
    def __init__(self, d_model):
        super(Vectorizer, self).__init__()
        self.transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.vgg19_feature_extractor = VGG19_FeatureExtractor()
        self.vectorizer_head = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Conv2d(1, 1, (28, 1)),
            nn.ReLU(),
            nn.Linear(in_features = 28, out_features = d_model)
        )

    def forward(self, x):
        x = self.transformer(x)     # [batch, 3, 224, 224]
        x = self.vgg19_feature_extractor(x) # [batch, 256, 28, 28]
        x = self.vectorizer_head(x) # [batch, 1, 1, d_model]
        x = x.squeeze(1)    # [batch, 1, d_model]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, nHeads, dModel, dK, dV):
        super(MultiHeadAttention, self).__init__()
        self.dModel = dModel
        self.nHeads = nHeads
        self.dK = dK
        self.dV = dV

        """
            For converting a [batch, seq_length, d_model] into 
            [batch, seq_length, n_heads * d_model], so that, each head can 
            receive a vector with dimension d_model
        """
        self.W_q = nn.Linear(self.dModel, self.dK, bias = False)
        self.W_k = nn.Lienar(self.dModel, self.dk, bias = False)
        self.W_v = nn.Linear(self.dModel, self.dV, bias = False)

        self.queryHeads = nn.Linear(self.dK, self.nHeads * self.dK, bias = False)
        self.kHeads = nn.Linear(self.dK, self.nHeads * self.dK, bias = False)
        self.vHeads = nn.Linear(self.dV, self.nHeads * self.dV, bias = False)
        
        """
            After merging tensors from each head, we will have
            n_heads * d_model tensor, which will need to be transformed into 
            d_model tensor.
        """
        self.vOut = nn.Linear(self.nHeads * self.dV, self.dModel, bias = False)

    def forward(self, query, key, value):
        # For each head
        seqLength = query.shape[1]
        batch = query.shape[0]

        query = self.W_q(self.queryHeads(query)) # [batch, seqLength, nHeads * dK]
        query = query.view(batch, seqLength, self.nHeads, self.dK) # [batch, seqLength, n_heads, dK]
        query = key.transpose(2, 1)   #[batch, n_heads, sequence,  dK]

        key = self.keyHeads(query) # [batch, sequence, n_heads * dK]
        key = key.view(batch, seqLength, nHeads, dModel) # [batch, sequence, n_heads, dK]
        key = value.transpose(2, 1) # [batch, n_heads, sequence, dK]
        
        value = self.valueHeads(value) # [batch, sequence, n_heads * dV]
        value = value.view(batch, seqLength, nHeads, dModel) # [batch, sequence, n_heads, dV]
        value = query.transpose(2, 1)   # [batch, n_heads, seq_length, dV]
        
        nopeak = (1 - torch.triu(torch.ones(1, seqLength, seqLength), diagonal = 1)).bool()
        
        attnScores = query @ key.transpose(3, 2) 
        # [batch, n_heads, sequence, dK]   @ [batch, n_heads, dK, sequence]
        # = [batch, n_heads, sequence, sequence]
        attnScores = attnScores.masked_fill(nopeak == 0, -1e9)  # [batch, n_heads, sequence, sequence]
        attnScores = torch.softmax(attnScores, dim = 2) # [batch, n_heads, sequence, sequence]

        value = attnScores @ value # [batch, n_heads, sequence, sequence]   @ [batch, n_heads, sequence, dV] = 
        # [batch, n_heads, sequence, dV]
        
        value = value.transpose(1, 2)   # [batch, sequence, n_heads, dV]
        value = value.view(batch, seq_length, n_heads * d_model)    # [batch, seq_length, n_heads * dV]
        value = self.project_v_out(value)   # [batch, seq_length, d_model]
        return value

class Encoder(nn.Module):
    def __init__(self, dModel, dK, dV, ff):
        self.dModel = dModel
        self.dK = dk
        self.dV = dV
        
        self.multiHeadAttention = MultiHeadAttention(dModel, dK, dV)
        self.layerNorm0 = nn.LayerNorm() # TODO:
        self.linear0 = nn.Linear(dModel, ff)
        self.linear1 = nn.Linear(ff, dModel)
        self.layerNorm1 = nn.LayerNorm() # TODO:

    def forward(self, x):
        # x: [batch, seqLength, dModel]
        value = multiHeadAttention(x, x, x)   # [batch, seq_length, dModel]
        value = self.layerNorm0(value + x)  # [batch, seq_length, dModel]
        
        ffOut = self.linear1(self.linear0(value))   #[batch, seq_Length, dModel]
        out = self.layerNorm1(value + ffOut)    # [batch, seqLength, dModel]

        return out

class DiscriminatorOutputHead(nn.Module):
    def __init__(self, dModel, dK, dV):
        self.dModel = dModel
        self.dK = dK
        self.dV = dV

        self.projectFrames = nn.Linear(self.dModel, 1)
        self.projectGazes = nn.Linear(self.dModel, 1)
        self.conv = nn.Conv2d(1, 2, dModel)

    def forward(self, encodedFrames, encodedGazes):
        # encodedFrames: [batch, seqLength, dModel]
        # encodedGazes: [batch, seqLength, dModel]

        encodedFrames = self.projectFrames(encodedFrames).transpose(2, 1)   # [batch, dModel, seqLength]
        encodedGazes = self.projectGazes(encodedGazes)  # [batch, seqLength, dModel]

        result = encodedFrames @ encodedGazes
        result = self.conv(result)  # [batch, 2, 1]
        result = result.view(-1, 1, 2)
        result = nn.Softmax(result, dim = 2)
        return result

class Discriminator(nn.Module):
    def __init__(self, dModel, dK, dV):
        self.dModel = dModel
        self.dK = dK
        self.dV = dV

        self.frameEncoders = [Encoder(dModel, dK, dV) for i in range(4)]
        self.gazeEncoders = [Encoder(dModel, dK, dV) for i in range(4)]
        self.W_q = nn.Linear(in_features = d_model, out_features = d_k, bias = False)
        self.W_k = nn.Linear(in_features = d_model, out_features = d_k, bias = False)
        self.W_v = nn.Linear(in_features = d_model, out_features = d_k, bias = False)
        
        self.frame_vectorizer = Vectorizer(d_model)
        self.gaze_vectorizer = Vectorizer(d_model)
        self.frame_multi_head_attention = MultiHeadAttention(d_model)
        self.gaze_multi_head_attention = MultiHeadAttention(d_model)

        self.gaze_enc_processor0 = nn.Linear(in_features = d_model, out_features = 1)
        self.gaze_enc_processor1 = nn.Linear(in_features = d_model, out_features = 1)
        self.relu = nn.ReLU()

    def forward(self, frame_seq, gaze_seq):
        # 
        # frame_sequence: [batch, seq_length, 3, frames_row, frames_column]
        # gaze_sequence: [batch, seq_length, 1, frames_row, frames_column]
        framesEmbedding = self.frame_vectorizer(frameSeq)   # [batch, seqLength, 1, dModel]
        gazeEmbedding = self.gaze_vectorizer(gazeSeq)   # [batch, seqLength, 1, dModel]

        encodedFrames = framesEmbedding
        for encoder in frameEncoders:
            encodedFrames = encoder(encodedFrames)  #[batch, seqLength, 1, dModel]
        
        encodedGazes = gazeEmbedding
        for encoder in gazeEncoders:
            encodedGazes = encoder(encodedGazes)    #[batch, seqLength, 1, dModel]

        
        probs = outputHead(encodedFrames, encodedGazes)   #[batch, 1]
        return probs

class Generator(nn.Module):
    def __init__(self, dModel, dK, dV):
        self.dModel = dModel
        self.dK = dk
        self.dV = dV
        
        self.frameVectorizer = Vectorizer(dModel)
        self.encoders = [Encoder(dModel, dK, dV, 8) for i in range(4)]

    def forward(self, frameSeq):
        # frameSeq: [batch, seqLen, 3, frameRow, frameCol]
        frameEmbedding = self.frameVectorizer(frameSeq) # [batch, seqLen, dModel]
        # TODO: Positional Encoding.

        frameEnc = frameEmbedding
        for enc in encoders:
            frameEnc = encoders(frameEnc)   # [batch seqLen, dModel]
        

        

def main():
    dk = 5
    dv = 3
    dmodel = 7

    module = MultiHeadAttention(8, dk, dv, dmodel)
    
    key = torch.Tensor(np.random.rand(2, 5, dmodel))
    query = torch.Tensor(np.random.rand(2, 5, dmodel))
    value = torch.Tensor(np.random.rand(2, 5, dmodel))

    result = module.forward(key, query, value)

    print(result.shape) 
    
if __name__ == '__main__':
    initial = Vectorizer(5).to(torch.device("cuda:0"))
    summary(initial, (3, 224, 224))