import numpy as np
import cv2
from scipy.signal import convolve2d
import argparse

class GBVS():
    def __init__(self, dim = (64, 64)):
        self.dim = dim

    def forward(self, x):
        """
            Input: 
                x (numpy.ndarray) -> An OpenCV image in BGR format
            Output:
                Saliency map
        """
        
        xOrig = x.copy()
        x = cv2.resize(x, self.dim)
        x = x / 255.0
        x = x.astype(np.float32)
        
        # Extract visual and orientation features.
        visualFeatures = self.extractVisualFeatures(x)
        orientation = self.extractOrientationFeatures(x, 4)
        resultMap = np.zeros(self.dim, np.float32)
        
        for featureName, scaleFeatures in visualFeatures.items():
            tempMap = self.normalize(self.calcActivation(scaleFeatures[0]))
            resultMap += tempMap

        # for scale, angleMap in orientation.items():
        for angleIdx, feature in orientation[0].items():
            # feature = cv2.resize(feature, self.dim)
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
        adjMatrix = np.zeros((nNodes, nNodes), dtype = np.float32)
        for index, value in np.ndenumerate(adjMatrix):
            i, j = index
            rowI = i // self.dim[1] # Y
            colI = i % self.dim[1] # X

            rowJ = j // self.dim[1] # Y
            colJ = j % self.dim[1] # X
            # print(i, j, rowI, colI, rowJ, colJ)
            adjMatrix[i, j] = adjMatrix[j, i] = np.abs(np.log(x[rowI, colI] / x[rowJ, colJ])) * \
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
    
    def fsdLaplacian(self, img: np.ndarray, n: int):
        lpf = np.zeros((5,1), dtype=np.float64)
        lpf[2, 0] = 3.0/8.0      # Middle
        lpf[1, 0] = 0.25
        lpf[3, 0] = 0.25
        lpf[0, 0] = 1.0/16
        lpf[4, 0] = 1.0/16
        lpf = np.matmul(lpf, lpf.transpose())

        fsdLowPassedPyr = {}
        fsdLaplacianPyr = {}
        fsdLowPassedPyr[0] = img   # At scale 0, we have the original image.

        for i in range(1, n+1):
            g0 = cv2.filter2D(fsdLowPassedPyr[i-1], -1, lpf, borderType = cv2.BORDER_REFLECT)
            fsdLaplacianPyr[i-1] = fsdLowPassedPyr[i-1] - g0  # The difference is the laplacian at previous scale.
            newDim = (fsdLaplacianPyr[i-1].shape[1] // 2, fsdLaplacianPyr[i-1].shape[0] // 2) # The gaussian at current scale will be downsampled to this dim.
            fsdLowPassedPyr[i] = cv2.resize(g0, newDim)   # Downsample and store.
        
        return (fsdLowPassedPyr, fsdLaplacianPyr)

    def extractOrientationFeatures(self, img: np.ndarray, anglesN): 
        lpf = np.zeros((5,1), dtype=np.float64)
        lpf[2, 0] = 3.0/8.0      # Middle
        lpf[1, 0] = 0.25
        lpf[3, 0] = 0.25
        lpf[0, 0] = 1.0/16.0
        lpf[4, 0] = 1.0/16.0
        lpf = np.matmul(lpf, lpf.transpose())
        
        r = img[:, :, 2].astype(np.float64)
        g = img[:, :, 1].astype(np.float64)
        b = img[:, :, 0].astype(np.float64)
        intensity = (r + g + b) / 3

        orientedFeatures = {}

        fsdLowPassed, laplacian = self.fsdLaplacian(intensity, 4)
        
        for p, img in laplacian.items():
            orientedFeatures[p] = {}
            for alpha in range(1, anglesN + 1):
                imgI = img.astype(np.complex128)
                H, W = img.shape[0], img.shape[1]
                xx = np.arange(W) - W // 2
                yy = np.arange(H) - H // 2
                X, Y = np.meshgrid(xx, yy)

                theta = np.pi / 4 * (alpha - 1)
                k = (np.pi / 2) * np.array([np.cos(theta), np.sin(theta)])
                multiplier = k[0] * X + k[1] * Y
                imgI = img * np.exp(1j * multiplier)
                convolved = convolve2d(imgI, lpf, mode='same', boundary= 'symm')
                imgM = np.abs(convolved)
                orientedFeatures[p][alpha] = imgM

        return orientedFeatures

    def extractVisualFeatures(self, img: np.ndarray):
        r = img[:, :, 2].astype(np.float64)
        g = img[:, :, 1].astype(np.float64)
        b = img[:, :, 0].astype(np.float64)
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
        yellow = (r + g) / 2 - np.abs(r - g) / 2 - b
        
        intensity[intensity < 0] = 0
        red[red < 0] = 0
        green[green < 0] = 0
        blue[blue < 0] = 0
        yellow[yellow < 0] = 0

        features = {}
        features['intensity'] = {}
        features['red'] = {}
        features['green'] = {}
        features['blue'] = {}
        features['yellow'] = {}

        features['intensity'][0] = intensity
        features['red'][0] = red
        features['green'][0] = green
        features['blue'][0] = blue
        features['yellow'][0] = yellow

        for featureName in ['intensity', 'red', 'green', 'blue', 'yellow']:
            for i in range(1, 4):
                features[featureName][i] = None
                features[featureName][i] = cv2.pyrDown(features[featureName][i - 1])

        return features


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input')

    ap = ap.parse_args()
    img = cv2.imread(ap.input)
    model = GBVS()
    saliencyMap = model.forward(img)
    cv2.imwrite('out.jpg', saliencyMap)
