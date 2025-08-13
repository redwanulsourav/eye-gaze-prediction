import numpy as np
import cv2
from scipy.signal import convolve2d
import argparse
import torch
import torch.nn as nn

class GBVS():
    def __init__(self, dim = (32, 32)):
        self.dim = dim
        self.dev = torch.device('gpu' if torch.cuda.is_avialable() else 'cpu')
    def forward(self, x):
        """
            Input: 
                x (numpy.ndarray) -> An OpenCV image in BGR format
            Output:
                Saliency map
        """
        x = torch.from_numpy(x).to(self.dev) # (H, W, 3)
        x = x.permute(2, 1, 0) # (3, H, W)
        x = x.unsqueeze(0) # (1, 3, H, W)
        # xOrig = x.copy()
        
        x = torch.interpolate(x, self.dim)
        x = x / 255.0
        x = x.float()
        
        # Extract visual and orientation features.
        visualFeatures = self.extractVisualFeatures(x)
        orientation = self.extractOrientationFeatures(x, 4)
        resultMap = torch.zeros(self.dim).to(self.dev)
        
        for featureName, feature in visualFeatures.items():
            tempMap = self.normalize(self.calcActivation(feature))
            resultMap += tempMap

        for angleIdx, feature in orientation.items():
            tempMap = self.normalize(self.calcActivation(feature))
            resultMap += tempMap

        resultMap = (resultMap - resultMap.min()) / (resultMap.max() - resultMap.min())
        resultMap = resultMap * 255
        resultMap = resultMap.astype(np.uint8)
        resultMap = cv2.resize(resultMap, (xOrig.shape[1], xOrig.shape[0]))
        return resultMap

    
    def calcActivation(self, x):
        x[x == 0] = 1e-5
        nNodes = x.shape[0] * x.shape[1]
        xx = torch.arange(nNodes).float().to(self.dev)
        yy = torch.arange(nNodes).float().to(self.dev)
        X, Y = torch.meshgrid(xx, yy)

        rowX = X // dim[1]
        rowY = Y // dim[1]

        colX = X % dim[1]
        colY = Y % dim[1]

        

        xxRow = xx // self.dim[1]
        xxCol = xx % self.dim[1]


        adjMatrix = torch.zeros((nNodes, nNodes), dtype = np.float32)
        for index, value in np.ndenumerate(adjMatrix):
            i, j = index
            rowI = i // self.dim[1] # Y
            colI = i % self.dim[1] # X

            rowJ = j // self.dim[1] # Y
            colJ = j % self.dim[1] # X
            # print(i, j, rowI, colI, rowJ, colJ)
            adjMatrix[i, j] = adjMatrix[j, i] = torch.abs(torch.log(x[rowI, colI] / x[rowJ, colJ])) * \
                                np.exp(-(np.square(rowI - rowJ) + np.square(colI - colJ))/(2 * 6.4))        

        rowSums = adjMatrix.sum(axis = 1, keepdims = True)
        transitionMatrix = adjMatrix / rowSums

        pi = np.full(nNodes, 1.0 / nNodes)
        tolerance = 1e-8
        maxiter = 1000

        for i in range(maxiter):
            nextPi = pi @ transitionMatrix
            print(np.abs(nextPi - pi).max())
            if np.allclose(nextPi, pi, atol = tolerance):
                break
            pi = nextPi
        print('activation done')
        return pi.reshape(self.dim[0], self.dim[1])

    def normalize(self, x):
        print(f'normalize {x.shape}')
        x[x == 0] = 1e-5
        nNodes = x.shape[0] * x.shape[1]
        adjMatrix = np.zeros((nNodes, nNodes), dtype = np.float32)
        for index, value in np.ndenumerate(adjMatrix):
            # 2D coordinates of node i.
            i, j = index
            rowI = i // self.dim[1] # Y
            colI = i % self.dim[1] # X

            rowJ = j // self.dim[1] # Y
            colJ = j % self.dim[1] # X

            adjMatrix[i, j] = x[rowJ, colJ] * \
                    np.exp(-(np.square(rowI - rowJ) + np.square(colI - colJ))/(2 * 6.4))

        rowSums = adjMatrix.sum(axis = 1, keepdims = True)
        transitionMatrix = adjMatrix / rowSums

        pi = np.full(nNodes, 1.0 / nNodes)
        tolerance = 1e-8
        maxiter = 100000

        for i in range(maxiter):
            nextPi = pi @ transitionMatrix
            print(np.abs(nextPi - pi).max())
            if np.allclose(nextPi, pi, atol = tolerance):
                break
            pi = nextPi
        print('normalization done')
        return pi.reshape(self.dim[0], self.dim[1])
    
    def fsdLaplacian(self, img, n: int):
        lpf = torch.zeros(5).float().to(self.dev)
        lpf[2] = 3.0/8.0      # Middle
        lpf[1] = 0.25
        lpf[3] = 0.25
        lpf[0] = 1.0/16
        lpf[4] = 1.0/16
        
        lpf = lpf[:, None] * lpf[None, :]
        lpf = lpf.unsqueeze(0).unsqueeze(0)
        # print(lpf)
        # lpf = np.matmul(lpf, lpf.transpose())

        fsdLowPassedPyr = {}
        fsdLaplacianPyr = {}
        fsdLowPassedPyr[0] = img   # At scale 0, we have the original image.

        for i in range(1, n+1):
            # padding = kernelSize // 2
            # paddedImg = 
            g0 = F.conv2d(F.pad(fsdLowPassedPyr[i-1], (2, 2, 2, 2), mode = 'reflect'), lpf)
            fsdLaplacianPyr[i-1] = fsdLowPassedPyr[i-1] - g0  # The difference is the laplacian at previous scale.
            newDim = (fsdLaplacianPyr[i-1].shape[3] // 2, fsdLaplacianPyr[i-1].shape[2] // 2) # The gaussian at current scale will be downsampled to this dim.
            fsdLowPassedPyr[i] = F.interpolate(g0, newDim)   # Downsample and store.
            # print(g0.shape)
            # print(fsdLowPassedPyr[i].shape)
        
        return (fsdLowPassedPyr, fsdLaplacianPyr)

    def getOrientedFeatures(self, img, anglesN, pyramidDepth): 
        lpf = torch.zeros(5).float().to(self.dev)
        lpf[2] = 3.0/8.0      
        lpf[1] = 0.25
        lpf[3] = 0.25
        lpf[0] = 1.0/16.0
        lpf[4] = 1.0/16.0
        lpf = lpf[:, None] * lpf[None, :]
        lpf = lpf.unsqueeze(0).unsqueeze(0)
        
        r = img[0, 2, :, :].unsqueeze(0).unsqueeze(0)
        g = img[0, 1, :, :].unsqueeze(0).unsqueeze(0)
        b = img[0, 0, :, :].unsqueeze(0).unsqueeze(0)

        intensity = (r + g + b) / 3

        orientedFeatures = {}

        fsdLowPassed, laplacian = self.fsdLaplacian(intensity, pyramidDepth)
        
        for p, img in laplacian.items():
            orientedFeatures[p] = {}
            for alpha in range(1, anglesN + 1):
                H, W = img.shape[2], img.shape[3]
                xx = torch.arange(W) - W // 2
                yy = torch.arange(H) - H // 2
                X, Y = torch.meshgrid(xx, yy)

                theta = torch.pi / 4 * (alpha - 1)
                k = (torch.pi / 2) * torch.tensor([torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))])
                multiplier = torch.complex(torch.zeros(img.shape[2], img.shape[3]), k[0] * X + k[1] * Y)
                multiplier = torch.exp(multiplier).to(self.dev).unsqueeze(0).unsqueeze(0)
        
                imgI = img * multiplier
        
                imgI = F.pad(imgI, (2, 2, 2, 2), mode = 'reflect')
                real = imgI.real
                imag = imgI.imag
                convolved = torch.complex(F.conv2d(real, lpf), F.conv2d(imag, lpf))
                imgM = torch.abs(convolved)
                orientedFeatures[p][alpha] = imgM
        return orientedFeatures[0]

    def extractVisualFeatures(self, img):
        r = img[0, 2, :, :].unsqueeze(0).unsqueeze(0)
        g = img[0, 1, :, :].unsqueeze(0).unsqueeze(0)
        b = img[0, 0, :, :].unsqueeze(0).unsqueeze(0)
        # print(r.shape)
        intensity = (r + g + b) / 3
        
        maxI = intensity.max()
        maxI = maxI / 10
        mask = intensity > maxI
        
        r[mask] = r[mask] / intensity[mask]
        g[mask] = g[mask] / intensity[mask]
        b[mask] = b[mask] / intensity[mask]

        red = (r - (g + b) / 2)
        green = (g - (r + b) / 2)
        blue = (b - (r + g) / 2)
        yellow = (r + g) / 2 - torch.abs(r - g) / 2 - b
        
        intensity[intensity < 0] = 0
        red[red < 0] = 0
        green[green < 0] = 0
        blue[blue < 0] = 0
        yellow[yellow < 0] = 0
        
        # iSigma = self.gaussianPyramid(intensity)
        # rSigma = self.gaussianPyramid(red)
        # gSigma = self.gaussianPyramid(green)
        # bSigma = self.gaussianPyramid(blue)
        # ySigma = self.gaussianPyramid(yellow)

        features = {}
        features['intensity'] = intensity
        features['red'] = red
        features['green'] = green
        features['blue'] = blue
        features['yellow'] = yellow

        return features


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input')

    ap = ap.parse_args()
    img = cv2.imread(ap.input)
    model = GBVS()
    saliencyMap = model.forward(img)
    cv2.imwrite('out.jpg', saliencyMap)
