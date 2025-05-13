import numpy as np
import cv2
from scipy.signal import convolve2d
# Implementation for Itti-Koch model.

class IttiKochModel():
    def __init__(self):
        pass
        
    def downsample(self, img: np.ndarray):
        img = cv2.GaussianBlur(img, (3,3), 2)
        newDim = (img.shape[1] // 2, img.shape[0] // 2)
        downsampled = cv2.resize(img, newDim)
        return downsampled
    
    def gaussianPyramid(self, img: np.ndarray):
        result = {}
        result[0] = img 
        for level in list(range(1,9)):
            result[level] = self.downsample(result[level-1])

        return result

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
        
        iSigma = self.gaussianPyramid(intensity)
        rSigma = self.gaussianPyramid(red)
        gSigma = self.gaussianPyramid(green)
        bSigma = self.gaussianPyramid(blue)
        ySigma = self.gaussianPyramid(yellow)

        pyramids = {}
        pyramids['intensity_pyr'] = iSigma
        pyramids['red_pyr'] = rSigma
        pyramids['green_pyr'] = gSigma
        pyramids['blue_pyr'] = bSigma
        pyramids['yellow_pyr'] = ySigma

        return pyramids

    def upscaleImage(self, img, coarserScale: int, finerScale: int):
        scaleDiff = coarserScale - finerScale
        upscaleDim = (img.shape[1] * (1 << scaleDiff), img.shape[0] * (1 << scaleDiff))
        upscaled = cv2.resize(img, upscaleDim)
        return upscaled

    def acrossScaleDiff(self, pyramids: dict):
        """
            Calculate the difference between a coarser scale and a finerscale across a pyramid.
            Need to upsample the coarser scaled image to finer scale first, then subtract.

            Inputs:
                `pyramid`: A `dict` containing multiple feature pyramids, with each pyramid containing
                (scale, image) pair.
            
            Returns:
                Another nested dictionary, containing (c, f, img) for color features, or
                (c, f, theta, img) for orientation feature.
        """

        # Extract each pyramids first.
        iSigma = pyramids['intensity_pyr']
        rSigma = pyramids['red_pyr']
        gSigma = pyramids['green_pyr']
        bSigma = pyramids['blue_pyr']
        ySigma = pyramids['yellow_pyr']
        oSigmaTheta = pyramids['orientation_pyr']
        
        iCS = {}
        byCS = {}
        rgCS = {}
        oCSTheta = {}

        for i in (2, 3, 4):   # Loop over finer scale.
            iCS[i] = {}
            byCS[i] = {}
            rgCS[i] = {}
            oCSTheta[i] = {}
            for j in (3, 4):   # Loop over the difference between finer and coarser scale.
                _s = i + j  # The coarser scale.

                tempI = self.upscaleImage(iSigma[_s], _s, i)    # Upscale the coarser scaled image, to finer scale.
                tempI = np.abs(iSigma[i] - tempI)   # Take the absolute difference.
                iCS[i][_s] = tempI    # Store.
                # Calculate BY(c,s)
                
                # upscaled_B_S = self.upscaleImage(BSigma[_s], _s, i) # Upscale the coarser scaled image, to finer scale.
                # upscaled_Y_S = self.upscaleImage(YSigma[_s], _s, i)    # Upscale the coarser scaled image, to finer scale.
                diff = np.abs(ySigma[_s] - bSigma[_s])
                temp = np.abs((bSigma[i] - ySigma[i]) - self.upscaleImage(diff, _s, i))  # Take the absolute difference.
                byCS[i][_s] = temp    # Store.

                # upscaled_G_s = self.upscaleImage(GSigma[_s], _s, i)   
                # upscaled_R_s = self.upscaleImage(RSigma[_s], _s, i)
                diff = np.abs(gSigma[_s] - rSigma[_s])
                temp = np.abs((rSigma[i] - gSigma[i]) - self.upscaleImage(diff, _s,i))
                rgCS[i][_s] = temp

                angles = list(oSigmaTheta[list(oSigmaTheta.keys())[0]].keys())  # Retrieve the list of angles
                
                oCSTheta[i][_s] = {}
                for k in angles:
                    upscaled_Os_Theta = self.upscaleImage(oSigmaTheta[_s][k], _s, i).astype(np.float64)   # Upscale
                    oCSTheta[i][_s][k] = np.abs(oSigmaTheta[i][k].astype(np.float64) - upscaled_Os_Theta)   # Absolute Diff
        
        
        result = {}
        result['I_C_S'] = iCS
        result['BY_C_S'] = byCS
        result['RG_C_S'] = rgCS
        result['O_C_S_Theta'] = oCSTheta
        return result

    def getLPF(self):
        """
            Get the separable low pass filter.
            Input:
                Empty
            Result:
                The (5 x 5) low pass filter
        """

        w = np.zeros((5,1), dtype=np.float64)
        
        w[2, 0] = 3/8      # Middle
        
        w[1, 0] = 0.25
        w[3, 0] = 0.25
        
        w[0, 0] = 1/16
        w[4, 0] = 1/16

        lpf = np.matmul(w, w.transpose())

        return lpf

    def convolution(self, w: np.ndarray, img: np.ndarray):
        result = cv2.filter2D(img, -1, w, borderType = cv2.BORDER_REFLECT)
        assert result.shape == img.shape

        return result

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

    def orientedGaborPyramid(self, img: np.ndarray, anglesN, pyramidDepth): 
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

        fsdLowPassed, laplacian = self.fsdLaplacian(intensity, pyramidDepth)
        
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
    
    def mapNorm(self, img: np.ndarray):
        for i in range(10):
            minValue = np.min(img)
            maxValue = np.max(img)
            img = (img - minValue) / (maxValue - minValue)
            kernel_size = (151, 151)    # The paper mentions a "big" kernel.
            xx = np.arange(kernel_size[1]) - kernel_size[1] // 2
            yy = np.arange(kernel_size[0]) - kernel_size[0] // 2

            X, Y = np.meshgrid(xx, yy)
            pwr = (-np.square(X) - np.square(Y))/(2 * 0.02 * img.shape[1])

            gauss_1 = (np.exp(pwr) * (0.5) * (0.5))/(2 * np.pi * np.square(0.02*img.shape[1]))
            # X, Y = np.meshgrid(xx, yy)
            pwr = (-np.square(X) - np.square(Y))/(2 * 0.25 * img.shape[1])
            gauss_2 = (np.exp(pwr) * (1.5) * (1.5))/(2 * np.pi * np.square(0.25*img.shape[1]))

            dogFilter = gauss_1 - gauss_2
            tempImg = img.copy()
            img = tempImg + cv2.filter2D(img, -1, dogFilter, borderType = cv2.BORDER_REFLECT) - 0.02
            img[img < 0] = 0
        # minValue = np.min(img)
        # maxValue = np.max(img)
        # img = (img - minValue) / (maxValue - minValue)
        return img

    def localMaximas(self, img: np.ndarray):
        """
            Find the average of local maximas
            Input:
                `img`: The input image.
            Output:
                The average of local maximas in a (5 x 5) windows.
        """

        sums_ = 0
        count = 0

        for i in range(0, img.shape[0], 1):
            if i + 5 > img.shape[0]:
                break
            for j in range(0, img.shape[1], 1):
                if j + 5 > img.shape[1]:
                    break
                
                window = img[i : i + 5, j : j + 5].copy()        
                sums_ += window.max()
                count += 1
                assert window.shape[0] == 5 and window.shape[1] == 5

        return sums_ / count
    
    def mergeIntensityMaps(self, I_C_S: dict):
        result = None

        for c, tempDict in I_C_S.items():   # Loop over finer scales.
            for s, img in tempDict.items(): # Loop over scale scales.
                img = self.mapNorm(img)
                if c < 4:
                    newDim = (img.shape[1] // (1 << (4 - c)), img.shape[0] // (1 << (4 - c)))
                    img = cv2.resize(img, newDim)
                
                assert c <= 4

                if result is None:
                    result = img
                else:
                    result += img
        result = self.mapNorm(result)
        return result
    
    def mergeColorMaps(self, BY_C_S: dict, RG_C_S: dict):
        result = None

        for c in (2, 3, 4):
            for _delta in (3, 4):
                s = c + _delta
                BYMap = self.mapNorm(BY_C_S[c][s])
                RGMap = self.mapNorm(RG_C_S[c][s])
                img = BYMap + RGMap
                if c < 4:
                    newDim = (img.shape[1] // (1 << (4 - c)), img.shape[0] // (1 << (4 - c)))
                    img = cv2.resize(img, newDim)
            
                assert c <= 4

                if result is None:
                    result = img
                else:
                    result += img
        result = self.mapNorm(result)
        return result
    
    def mergeOrientationMaps(self, O_C_S_Theta: dict):
        tempResult = None

        for angle in (1, 2, 3, 4):
            temp = None
            for c in (2, 3, 4):
                for _delta in (3, 4):
                    s = c + _delta

                    img = self.mapNorm(O_C_S_Theta[c][s][angle])

                    if c < 4:
                        newDim = (img.shape[1] // (1 << (4 - c)), img.shape[0] // (1 << (4 - c)))
                        img = cv2.resize(img, newDim)
            
                    assert c <= 4

                    if temp is None:
                        temp = img
                    else:
                        temp += img
            temp = self.mapNorm(temp)

            if tempResult is None:
                tempResult = temp
            else:
                tempResult += temp

        tempResult = self.mapNorm(tempResult)
        return tempResult
    
    def minMaxNormalize(self, img):
        imgMin = np.min(img)
        imgMax = np.max(img)
        return (img - imgMin)  / (imgMax - imgMin)

    def saliencyMap(self, img: np.ndarray):
        img = img.astype(np.float32)/255.0
        pyramids = self.extractVisualFeatures(img)
        orientation = self.orientedGaborPyramid(img, 4, 9)        
        
        pyramids['orientation_pyr'] = orientation

        # for scale, pyr in orientation.items():
        #     print(np.max(orientation[scale][1] - orientation[scale][2]))
        #     for angle, img in pyr.items():
        #         cv2.imwrite(f'{scale}-{angle}.jpg', self.minMaxNormalize(img)*255)

        result = {}
        result['pyramids'] = pyramids
        
        pyramids = self.acrossScaleDiff(pyramids)
        result['merged_pyramids'] = pyramids

        IBar = self.mergeIntensityMaps(pyramids['I_C_S'])
        CBar = self.mergeColorMaps(pyramids['BY_C_S'], pyramids['RG_C_S'])
        OBar = self.mergeOrientationMaps(pyramids['O_C_S_Theta'])
        result['I_Bar'] = IBar
        result['C_Bar'] = CBar
        result['O_Bar'] = OBar
        # cv2.imwrite('IBar.jpg', (self.minMaxNormalize(IBar)*255).astype(np.uint8))
        # cv2.imwrite('CBar.jpg', (self.minMaxNormalize(CBar)*255).astype(np.uint8))
        # cv2.imwrite('OBar.jpg', (self.minMaxNormalize(OBar)*255).astype(np.uint8))
        
        # print(OBar.shape)
        # cv2.imwrite('obar.jpg', OBar)
        sMap = (IBar + CBar + OBar) 
        # sMap = cv2.normalize(sMap, sMap, 0, 255, cv2.NORM_MINMAX)
        sMapMin = np.min(sMap)
        sMapMax = np.max(sMap)
        sMap = (sMap - sMapMin) / (sMapMax - sMapMin)
        return sMap * 255

