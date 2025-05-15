import numpy as np
import cv2
from scipy.signal import convolve2d
import torch 
import torch.nn as nn
import torch.nn.functional as F
# Implementation for Itti-Koch model.

class IttiKochModel():
    """
        IMPORTANT: Use cv2.BORDER_DEFAULT / cv2.BORDER_REFLECT_101 for everything
    """
    def __init__(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def gaussianBlur(self, img, kernelSize = 5, sigma = 1.0):
        """
            Perform a gaussian blur
            Input:
                img: torch.Tensor -> img.shape: (1, 1, H, W)
            Returns:
                result: torch.Tensor -> result.shape: (1, 1, H, W)
        """

        channels = img.shape[1]
        x = torch.arange(kernelSize).float() - kernelSize // 2
        gauss = torch.exp(-x**2 / (2 * sigma ** 2))
        kernel1d = gauss / gauss.sum()
        kernel2d = kernel1d[:, None] * kernel1d[None, :]    # (5, 5)
        kernel2d = kernel2d.unsqueeze(0).unsqueeze(0).to(self.dev)    # (1, 1, 5, 5)
        
        padding = kernelSize // 2
        paddedImg = F.pad(img, (padding, padding, padding, padding), mode = 'reflect')
        return F.conv2d(img, kernel2d)

    def pyrDown(self, img):
        blurred = self.gaussianBlur(img, kernelSize = 5, sigma = 1.0)
        return F.interpolate(img, scale_factor = 0.5, mode = 'bilinear', align_corners = False)

    
    def gaussianPyramid(self, img):
        result = {}
        result[0] = img 
        for level in list(range(1,9)):
            result[level] = self.pyrDown(result[level-1])

        return result

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
        upscaleDim = (img.shape[3] * (1 << scaleDiff), img.shape[2] * (1 << scaleDiff))
        upscaled = F.interpolate(img, upscaleDim, mode = 'bilinear')
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
                tempI = torch.abs(iSigma[i] - tempI)   # Take the absolute difference.
                iCS[i][_s] = tempI    # Store.
                # Calculate BY(c,s)
                
                # upscaled_B_S = self.upscaleImage(BSigma[_s], _s, i) # Upscale the coarser scaled image, to finer scale.
                # upscaled_Y_S = self.upscaleImage(YSigma[_s], _s, i)    # Upscale the coarser scaled image, to finer scale.
                diff = torch.abs(ySigma[_s] - bSigma[_s])
                temp = torch.abs((bSigma[i] - ySigma[i]) - self.upscaleImage(diff, _s, i))  # Take the absolute difference.
                byCS[i][_s] = temp    # Store.

                # upscaled_G_s = self.upscaleImage(GSigma[_s], _s, i)   
                # upscaled_R_s = self.upscaleImage(RSigma[_s], _s, i)
                diff = torch.abs(gSigma[_s] - rSigma[_s])
                temp = torch.abs((rSigma[i] - gSigma[i]) - self.upscaleImage(diff, _s,i))
                rgCS[i][_s] = temp

                angles = list(oSigmaTheta[list(oSigmaTheta.keys())[0]].keys())  # Retrieve the list of angles
                
                oCSTheta[i][_s] = {}
                for k in angles:
                    upscaled_Os_Theta = self.upscaleImage(oSigmaTheta[_s][k], _s, i)   # Upscale
                    oCSTheta[i][_s][k] = torch.abs(oSigmaTheta[i][k] - upscaled_Os_Theta)   # Absolute Diff
        
        
        result = {}
        result['I_C_S'] = iCS
        result['BY_C_S'] = byCS
        result['RG_C_S'] = rgCS
        result['O_C_S_Theta'] = oCSTheta
        return result

    def convolution(self, w: np.ndarray, img: np.ndarray):
        result = cv2.filter2D(img, -1, w, borderType = cv2.BORDER_REFLECT)
        assert result.shape == img.shape

        return result

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

    def orientedGaborPyramid(self, img, anglesN, pyramidDepth): 
        lpf = torch.zeros(5).float().to(self.dev)
        lpf[2] = 3.0/8.0      # Middle
        lpf[1] = 0.25
        lpf[3] = 0.25
        lpf[0] = 1.0/16.0
        lpf[4] = 1.0/16.0
        lpf = lpf[:, None] * lpf[None, :]
        lpf = lpf.unsqueeze(0).unsqueeze(0)
        # lpf = torch.complex(lpf, lpf)
        # print(lpf)
        # print(lpf.shape)
        # lpf = np.matmul(lpf, lpf.transpose())
        
        r = img[0, 2, :, :].unsqueeze(0).unsqueeze(0)
        g = img[0, 1, :, :].unsqueeze(0).unsqueeze(0)
        b = img[0, 0, :, :].unsqueeze(0).unsqueeze(0)

        intensity = (r + g + b) / 3

        orientedFeatures = {}

        fsdLowPassed, laplacian = self.fsdLaplacian(intensity, pyramidDepth)
        
        for p, img in laplacian.items():
            orientedFeatures[p] = {}
            for alpha in range(1, anglesN + 1):
                # imgI = img.astype(np.complex128)
                H, W = img.shape[2], img.shape[3]
                xx = torch.arange(W) - W // 2
                yy = torch.arange(H) - H // 2
                X, Y = torch.meshgrid(xx, yy)

                theta = torch.pi / 4 * (alpha - 1)
                k = (torch.pi / 2) * torch.tensor([torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))])
                multiplier = torch.complex(torch.zeros(img.shape[2], img.shape[3]), k[0] * X + k[1] * Y)
                multiplier = torch.exp(multiplier).to(self.dev).unsqueeze(0).unsqueeze(0)
                # print(type(multiplier))
                # print(multiplier.dtype)
                # print(multiplier.shape) # (H, W)
                imgI = img * multiplier
                # F.pad(img, (padding, padding, padding, padding)
                imgI = F.pad(imgI, (2, 2, 2, 2), mode = 'reflect')
                real = imgI.real
                imag = imgI.imag
                convolved = torch.complex(F.conv2d(real, lpf), F.conv2d(imag, lpf))
                # convolved = F.conv2d(imgI, lpf)
                imgM = torch.abs(convolved)
                # print(imgM.shape)
                orientedFeatures[p][alpha] = imgM
        return orientedFeatures
    
    def mapNorm(self, img):
        for i in range(10):
            minValue = img.min()
            maxValue = img.max()
            img = (img - minValue) / (maxValue - minValue)
            kernel_size = (75, 75)    # The paper mentions a "big" kernel.
            xx = torch.arange(kernel_size[1]) - kernel_size[1] // 2
            yy = torch.arange(kernel_size[0]) - kernel_size[0] // 2

            X, Y = torch.meshgrid(xx, yy)
            pwr = (-torch.square(X) - torch.square(Y))/(2 * 0.02 * img.shape[3])

            gauss_1 = (torch.exp(pwr) * (0.5) * (0.5))/(2 * torch.pi * torch.square(torch.tensor(0.02 * img.shape[3])))
            # X, Y = np.meshgrid(xx, yy)
            pwr = (-torch.square(X) - torch.square(Y))/(2 * 0.25 * img.shape[3])
            gauss_2 = (torch.exp(pwr) * (1.5) * (1.5))/(2 * torch.pi * torch.square(torch.tensor(0.25*img.shape[3])))

            dogFilter = (gauss_1 - gauss_2).unsqueeze(0).unsqueeze(0).to(self.dev)
            # tempImg = img.copy()
            img = img + F.conv2d(F.pad(img, (kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[0] // 2, kernel_size[0] // 2), mode = 'reflect'), 
            dogFilter) - 0.02
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
                    newDim = (img.shape[3] // (1 << (4 - c)), img.shape[2] // (1 << (4 - c)))
                    img = F.interpolate(img, newDim, mode = 'bilinear', align_corners = False)
                
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
                    newDim = (img.shape[3] // (1 << (4 - c)), img.shape[2] // (1 << (4 - c)))
                    img = F.interpolate(img, newDim, mode = 'bilinear', align_corners = False)
            
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
                        newDim = (img.shape[3] // (1 << (4 - c)), img.shape[2] // (1 << (4 - c)))
                        img = F.interpolate(img, newDim, mode = 'bilinear', align_corners = False)
            
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
        img = torch.from_numpy(img).float().to(self.dev)
        img = img / 255.0
        img = img.unsqueeze(0)  # (1, H, W, 3)
        img = img.permute(0, 3, 1, 2)   # (1, 3, H, W)
        print(img.shape)
        pyramids = self.extractVisualFeatures(img)
        orientation = self.orientedGaborPyramid(img, 4, 9)        
        pyramids['orientation_pyr'] = orientation
        pyramids = self.acrossScaleDiff(pyramids)
        IBar = self.mergeIntensityMaps(pyramids['I_C_S'])
        CBar = self.mergeColorMaps(pyramids['BY_C_S'], pyramids['RG_C_S'])
        OBar = self.mergeOrientationMaps(pyramids['O_C_S_Theta'])
        sMap = (IBar + CBar + OBar) 
        sMapMin = sMap.min()
        sMapMax = sMap.max()
        sMap = (sMap - sMapMin) / (sMapMax - sMapMin)
        sMap = sMap.permute(0, 2, 3, 1).squeeze(0)
        sMap = sMap.detach().cpu().numpy()
        sMap = (sMap * 255).astype(np.uint8)
        return sMap

        # img = img.astype(np.float32)/255.0
        # pyramids = self.extractVisualFeatures(img)
        

        # # for scale, pyr in orientation.items():
        # #     print(np.max(orientation[scale][1] - orientation[scale][2]))
        # #     for angle, img in pyr.items():
        # #         cv2.imwrite(f'{scale}-{angle}.jpg', self.minMaxNormalize(img)*255)

        # result = {}
        # result['pyramids'] = pyramids
        
        # result['merged_pyramids'] = pyramids

        # result['I_Bar'] = IBar
        # result['C_Bar'] = CBar
        # result['O_Bar'] = OBar
        # # cv2.imwrite('IBar.jpg', (self.minMaxNormalize(IBar)*255).astype(np.uint8))
        # # cv2.imwrite('CBar.jpg', (self.minMaxNormalize(CBar)*255).astype(np.uint8))
        # # cv2.imwrite('OBar.jpg', (self.minMaxNormalize(OBar)*255).astype(np.uint8))
        
        # # print(OBar.shape)
        # # cv2.imwrite('obar.jpg', OBar)
        # # sMap = cv2.normalize(sMap, sMap, 0, 255, cv2.NORM_MINMAX)
        # sMapMax = np.max(sMap)
        # return sMap * 255

