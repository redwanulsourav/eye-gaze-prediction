import os
import unittest
import torch
import math

from ..src import models

class TestTemporalFeatureNet(unittest.TestCase):
    def __init__(self):
        self.net = models.TemporalFeatureNet(3 * 4 * 5 * 5)

    def test(self):
        x = torch.rand((3, 4, 5, 5))
        y1 = self.net(x)
        
        x = x.flatten().tolist()
        linear_weights = list(self.net.parameters())[0].tolist()
        biases = list(self.net.parameters())[1].tolist()
        
        y2 = []
        for j, p in enumerate(linear_weights):
            sum_ = 0
            for i in range(len(x)):
                sum_ += x[i] * p[i]
            sum_ += biases[j]
            y2.append(sum_)

        y1 = y1[0].tolist()
        assert len(y1) == len(y2), f'Length of y1 and y2 do not match, {len(y1)} != {len(y2)}'
        for i in range(len(y1)):
            assert abs(y1[i] - y2[i]) < 1e-6, f'y1 and y2 do not match at idx {i}, {y1[i]} != {y2[i]}'

class TestTemporalGazeNet(unittest.TestCase):
    def __init__(self):
        self.net = models.TemporalGazeNet()
        self.w_ii = list(self.net.parameters())[0][0:36,:].tolist()
        self.w_if = list(self.net.parameters())[0][36:72, :].tolist()
        self.w_ig = list(self.net.parameters())[0][72:108, :].tolist()
        self.w_io = list(self.net.parameters())[0][108:144, :].tolist()
    
        self.w_hi = list(self.net.parameters())[1][0:36, : ].tolist()
        self.w_hf = list(self.net.parameters())[1][36: 72, :].tolist()
        self.w_hg = list(self.net.parameters())[1][72: 108, :].tolist()
        self.w_ho = list(self.net.parameters())[1][108: 144, :].tolist()

        self.b_ii = list(self.net.parameters())[2][0: 36].tolist()
        self.b_if = list(self.net.parameters())[2][36: 72].tolist()
        self.b_ig = list(self.net.parameters())[2][72: 108].tolist()
        self.b_io = list(self.net.parameters())[2][108: 144].tolist()

        self.b_hi = list(self.net.parameters())[3][0: 36].tolist()
        self.b_hf = list(self.net.parameters())[3][36: 72].tolist()
        self.b_hg = list(self.net.parameters())[3][72: 108].tolist()
        self.b_ho = list(self.net.parameters())[3][108: 144].tolist()
        
        self.linear1_w = list(self.net.parameters())[4].tolist()
        self.linear1_b = list(self.net.parameters())[5].tolist()

        self.linear2_w = list(self.net.parameters())[6].tolist()
        self.linear2_b = list(self.net.parameters())[7].tolist()


    def sigmoid(self,val):
        return 1 / (1 + math.exp(-val))

    def tanh(self, val):
        return math.tanh(val)
    def relu(self, val):
        return max((0, val))


    def test(self):
        x = torch.rand(1,3,2)
        y1 = self.net(x)
        
        x = x.tolist()[0]
        
        c = [0 for i in range(36)]
        h = [0 for i in range(36)]
        
        for k in range(3): 
            f = [0 for i in range(36)]
            o = [0 for i in range(36)]
            p = [0 for i in range(36)]
            g = [0 for i in range(36)]
            

            for i in range(36):
                for j in range(2):
                    p[i] += self.w_ii[i][j] * x[k][j]
                    f[i] += self.w_if[i][j] * x[k][j]
                    g[i] += self.w_ig[i][j] * x[k][j]
                    o[i] += self.w_io[i][j] * x[k][j]

                p[i] += self.b_ii[i]
                f[i] += self.b_if[i]
                g[i] += self.b_ig[i]
                o[i] += self.b_io[i]

            for i in range(36):
                for j in range(36):
                    p[i] += self.w_hi[i][j] * h[j]
                    f[i] += self.w_hf[i][j] * h[j]
                    g[i] += self.w_hg[i][j] * h[j]
                    o[i] += self.w_ho[i][j] * h[j]

                p[i] += self.b_hi[i]
                f[i] += self.b_hf[i]
                g[i] += self.b_hg[i]
                o[i] += self.b_ho[i]


            p = [self.sigmoid(p[i]) for i in range(36)]
            f = [self.sigmoid(f[i]) for i in range(36)]
            o = [self.sigmoid(o[i]) for i in range(36)]
            g = [self.tanh(g[i]) for i in range(36)]
            c = [f[i] * c[i] + p[i] * g[i] for i in range(36)]
            h = [o[i] * self.tanh(c[i]) for i in range(36)]

        h = [self.relu(h[i]) for i in range(36)]
        
        h1 = [0 for i in range(16)]
        for i in range(16):
            for j in range(36):
                h1[i] += self.linear1_w[i][j] * h[j]
            h1[i] += self.linear1_b[i]
        h1 = [self.relu(h1[i]) for i in range(16)]

        h2 = [0 for i in range(8)]
        for i in range(8):
            for j in range(16):
                h2[i] += self.linear2_w[i][j] * h1[j]
            h2[i] += self.linear2_b[i]
        y1 = y1[0, 0, :].tolist() 
        assert len(y1) == len(h2), f'Lengths do not math, {len(y1)} != {len(h2)}'
        for i in range(len(y1)):
            assert abs(y1[i] - h2[i]) < 1e-6, f'Value mismatch at idx {i}'

obj = TestTemporalFeatureNet()
obj2 = TestTemporalGazeNet()
obj.test()
obj2.test()
print('models test passed')
