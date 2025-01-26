import numpy as np
import cv2
# Implementation for Itti-Koch model.

class Itti_Koch_Model():
    def __init__(self):
        self.BLUE_CHANNEL = 0
        self.GREEN_CHANNEL = 1
        self.RED_CHANNEL = 2
        self._c = (2, 3, 4)
        self._delta = (3, 4)
        self.center_surround_maps = None
        self.I_sigma = None
        self.R_sigma = None
        self.G_sigma = None
        self.B_sigma = None
        self.Y_sigma = None
        
    def downsample(self, img: np.ndarray, level: int):
        """
            Smooths the image using a (3x3) Gaussian kernel and then 
            downsamples it by a factor of (2^level)

            Inputs: 
                `img`: The input image to downsample
                `level`: Downsample the image to have a dimension of original_dimension/(2^level) 
            Output:
                Returns the downsampled image.
        """
        
        # Both grayscale and color images are accepted.
        assert len(img.shape) == 2 or len(img.shape) == 3
        assert type(img) == np.ndarray and type(level) == int

        # Low pass filter.
        img = cv2.GaussianBlur(img, (3,3), 2)

        divFactor = 1 << level    # Divide the original dimension by this value.

        # OpenCV takes the dimension along x-axis (columns) first, then y-axis (rows)
        newDim = (img.shape[1] // divFactor, img.shape[0] // divFactor)
        
        # Subsample
        downsampled = cv2.resize(img, newDim)
        return downsampled
    
    def gaussianPyramid(self, img: np.ndarray, levels: list):
        """
            Calculates a gaussian pyramid of progressively downsized
            image.
            Inputs:
                `img`: The input image
                `levels`: A list of integer specifying the scales of 
                          images in the pyramid.
            Returns:
                A `dict` containing the (level, image) pairs.
        """
        assert type(img) == np.ndarray, f'Expected the type of img to be numpy.ndarray, found {type(img)}'
        assert type(levels) == list, f'Expected the type of levels to be list, found {type(levels)}'

        result = {}
        for level in levels:
            # Downsample image to scale
            result[level] = self.downsample(img, level)

        return result

    def extractVisualFeatures(self, img: np.ndarray, levels: list):
        """
            Extract 1. intensity, 2. red channel, 3. green channel, 4. blue channel, 5. yellow channel features.
            Create a gaussian pyramid of each of the feature at scale given by the second paramters.
            Return each of the pyarmids in a dictionary.

            Inputs:
                `img`: The input color image
                `levels`: The scales of downsampling.
            
            Returns:
                A `dict` containing the (featureName, pyramid) features.
        """

        assert type(img) == np.ndarray, f'Expected the type of img to be numpy.ndarray, found {type(img)}'
        assert type(levels) == list, f'Expected the type of levels to be list, found {type(levels)}'
        
        # TODO (!!Urgent!!): Need extra requirements on dimension of the image.

        r = img[:, :, self.RED_CHANNEL].astype(np.float64)
        g = img[:, :, self.GREEN_CHANNEL].astype(np.float64)
        b = img[:, :, self.BLUE_CHANNEL].astype(np.float64)

        # Extract featuers according to the paper.
        I = (r + g + b) / 3
        R = (r - (g + b) / 2)
        G = (g - (r + b) / 2)
        B = (b - (r + g) / 2)
        Y = (r + g) / 2 - np.abs(r - g) / 2 - b
        
        # Set the negatives to zero.
        I[I < 0] = 0
        R[R < 0] = 0
        G[G < 0] = 0
        B[B < 0] = 0
        Y[Y < 0] = 0


        ISigma = self.gaussianPyramid(I, levels)
        RSigma = self.gaussianPyramid(R, levels)
        GSigma = self.gaussianPyramid(G, levels)
        BSigma = self.gaussianPyramid(B, levels)
        YSigma = self.gaussianPyramid(Y, levels)

        pyramids = {}
        pyramids['intensity_pyr'] = ISigma
        pyramids['red_pyr'] = RSigma
        pyramids['green_pyr'] = GSigma
        pyramids['blue_pyr'] = BSigma
        pyramids['yellow_pyr'] = YSigma

        return pyramids

    def upscaleImage(self, img, coarserScale: int, finerScale: int):
        """
            Upsample an image from coarserScale (downsampled scale) to finerScale (target scale, upsampled)
            Inputs:
                `img`: The input image.
                `coarserScale`: The current scale of the input.
                `finerScale`: The target (upsampled) scale of input
            Returns:
                A `numpy.ndarray` with same feature channels but with upsmapled image.
        """
        assert finerScale < coarserScale f'Finerscale should be lower than coarser scale'
        assert type(img) == np.ndarray
        assert type(coarserScale) == int
        assert type(finerScale) == int

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
        ISigma = pyramids['intensity_pyr']
        RSigma = pyramids['red_pyr']
        GSigma = pyramids['green_pyr']
        BSigma = pyramids['blue_pyr']
        YSigma = pyramids['yellow_pyr']
        OSigmaTheta = pyramids['orientation_pyr']

        I_C_S = {}
        BY_C_S = {}
        RG_C_S = {}
        O_C_S_Theta = {}

        for i in self._c:   # Loop over finer scale.
            I_C_S[i] = {}
            BY_C_S[i] = {}
            RG_C_S[i] = {}
            O_C_S_Theta[i] = {}
            for j in self._delta:   # Loop over the difference between finer and coarser scale.
                _s = i + j  # The coarser scale.

                tempI = self.upscaleImage(ISigma[_s], _s, i)    # Upscale the coarser scaled image, to finer scale.
                tempI = np.abs(ISigma[i] - tempI)   # Take the absolute difference.
                I_C_S[i][_s] = tempI    # Store.
                # Calculate BY(c,s)
                
                upscaled_B_S = self.upscaleImage(BSigma[_s], _s, i) # Upscale the coarser scaled image, to finer scale.
                upscaled_Y_S = self.upscaleImage(YSigma[_s], _s, i)    # Upscale the coarser scaled image, to finer scale.
                temp = np.abs((BSigma[i] - YSigma[i]) - (upscaled_Y_S - upscaled_B_S))  # Take the absolute difference.
                BY_C_S[i][_s] = temp    # Store.

                upscaled_G_s = self.upscale_image(G_sigma[_s], _s, i)   
                upscaled_R_s = self.upscale_image(R_sigma[_s], _s, i)
                temp = np.abs((R_sigma[i] - G_sigma[i]) - (upscaled_G_s - upscaled_R_s))
                RG_C_S[i][_s] = temp

                angles = list(OSigmaTheta[list(OSigmaTheta.keys())[0]].keys())  # Retrieve the list of angles
                
                O_C_S_Theta[i][_s] = {}
                for k in angles:
                    upscaled_Os_Theta = self.upscaleImage(O_Sigma_Theta[_s][k], _s, i).astype(np.float64)   # Upscale
                    O_C_S_Theta[i][_s][k] = np.abs(O_Sigma_Theta[i][k].astype(np.float64) - upscaled_Os_theta)   # Absolute Diff
        
        
        result = {}
        result['I_C_S'] = I_C_S
        result['BY_C_S'] = BY_C_S
        result['RG_C_S'] = RG_C_S
        result['O_C_S_Theta'] = O_C_S_Theta
        return result
        
    def deg2Rad(self, angle: float):
        """
            Convert a degree to radian
            Input:
                `angle`: The input angle in degrees
            Result:
                `angle` in radians
        """
        assert type(angle) == float

        return  (np.pi * angle / 180)   # pi * x / 180.
    

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
        result = img.copy()

        grayscale = False

        if len(result.shape) == 2:
            grayscale = True
        
        for i in range(2, img.shape[0] - 2):
            for j in range(2, img.shape[1] - 2):
                if grayscale == True:
                    result[i,j] = 0
                    for m in range(-2, 3):
                        for n in range(-2, 3):
                            result[i,j] += w[m + 2, n + 2] * img[i + m, j + n]
                else:
                    result[i, j, ...] = 0
                    for k in range(3):
                        for m in range(-2, 3):
                            for n in range(-2, 3):
                                result[i, j, k] += w[m + 2, n + 2] * img[i + m, j + n, k]

        return result

    def fsd_laplacian(self, img: np.ndarray, n: int):
        fsd_low_passed = {}
        fsd_laplacian = {}
        fsd_low_passed[0] = img

        for i in range(1, n):
            g_0 = self.convolution(self.get_lpf(), fsd_low_passed[i-1])
            fsd_laplacian[i-1] = fsd_low_passed[i-1] - g_0

            new_dim = (fsd_laplacian[i-1].shape[1] // 2, fsd_laplacian[i-1].shape[0] // 2)

            fsd_low_passed[i] = cv2.resize(g_0, new_dim)
        
        fsd_laplacian[n-1] = fsd_low_passed[n-1]
        return (fsd_low_passed, fsd_laplacian)

    def orientedGaborPyramid(self, img: np.ndarray, anglesN, pyramidDepth): 
        """
            Returns a oriented gabor pyramid of `img` and angles specified by angles_n.
            Input:
                `img`: The input image.
                `angleN`: How many equally spaced angles
                `pyramidDepth: Max depth or scale of a pyramid

        """
        r = img[:, :, self.RED_CHANNEL].astype(np.float64)
        g = img[:, :, self.GREEN_CHANNEL].astype(np.float64)
        b = img[:, :, self.BLUE_CHANNEL].astype(np.float64)
        I = (r + g + b) / 3

        oriented_features = {}

        fsdLowPassed, laplacian = self.fsdLaplacian(I, pyramidDepth)
        
        for p, img in laplacian.items():
            orientedFeatures[p] = {}
            for alpha in range(1, angles_n + 1):
                img_i = img.astype(np.complex128)
                for j in range(img.shape[1]):
                    for i in range(img.shape[0]):
                        x = j - img.shape[1] // 2
                        y = i - img.shape[0] // 2

                        r = np.array([x, y])
                        theta = np.pi/4 * (alpha - 1)

                        k = np.pi / 2 * np.array([np.cos(theta), np.sin(theta)])

                        multiplier = np.dot(k, r)

                        img_i[i, j] = img[i, j] * np.exp(1j * multiplier)
                        # print(img_i.shape)
                oriented_features[p][alpha] = self.convolution(self.get_lpf(), img_i.real)
        return oriented_features
    
    def map_normalization(self, img: np.ndarray):
        """
            TODO: Write description
        """
        assert len(img.shape) == 2, f'The input image should have single channel'

        # Step 1. Normalize the map in [0, 255] range
        img = img.astype(np.float64)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        
        # Step 2. Finding global maxima and average of local maxima
        global_maxima = img.max()
        local_maxima = self.local_maximas(img)

        # Step 3. Multiplication.
        img = img * np.square(global_maxima-local_maxima)
        return img

    def local_maximas(self, img: np.ndarray):
        """
            TODO: Write description.
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
    
    def merge_intensity_maps(self, I_cs: dict):
        result = None
        for c, temp_dict in I_cs.items():
            for s, img in temp_dict.items():
                # result = None
                img = self.map_normalization(img)

                if c < 4:
                    new_dim = (img.shape[1] // (1 << (4 - c)), img.shape[0] // (1 << (4 - c)))
                    img = cv2.resize(img, new_dim)
                
                assert c <= 4

                if result is None:
                    result = img
                else:
                    result += img
        return result
    
    def merge_color_maps(self, BY_cs: dict, RG_cs: dict):
        result = None
        for c in (2, 3, 4):
            for _delta in (3, 4):
                s = c + _delta

                BY_map = self.map_normalization(BY_cs[c][s])
                RG_map = self.map_normalization(RG_cs[c][s])

                img = BY_map + RG_map
                # print(f'\nc: {c}, s: {s}')
                # print(img.shape)
                
                if c < 4:
                    new_dim = (img.shape[1] // (1 << (4 - c)), img.shape[0] // (1 << (4 - c)))
                    img = cv2.resize(img, new_dim)
                    # print(f'After resize: {img.shape}')
            
                assert c <= 4
                if result is None:
                    result = img
                else:
                    # print(c, s)
                    # print(img.shape)
                    # print(result.shape)
                    result += img
        return result
    
    def merge_orientation_maps(self, O_cs_theta: dict):
        temp_result = None

        for angle in (1, 2, 3, 4):
            temp = None
            for c in (2, 3, 4):
                for _delta in (3, 4):
                    s = c + _delta

                    img = self.map_normalization(O_cs_theta[c][s][angle])

                    if c < 4:
                        new_dim = (img.shape[1] // (1 << (4 - c)), img.shape[0] // (1 << (4 - c)))
                        img = cv2.resize(img, new_dim)
            
                    assert c <= 4

                    if temp is None:
                        temp = img
                    else:
                        temp += img
            temp = self.map_normalization(temp)
            if temp_result is None:
                temp_result = temp
            else:
                temp_result += temp
        return temp_result
    
    def get_saliency_map(self, img: np.ndarray):
        pyramids = self.extractVisualFeatures(img, list(range(0, 9)))
        orientation = self.get_oriented_gabor_pyramid(img, 4, 9)
        
        pyramids['orientation_pyr'] = orientation

        intensity = pyramids['intensity_pyr']
        # colors = visual_features[1:5]

        (I_cs, BY_cs, RG_cs, O_cs_theta) = self.calculate_across_scale_differences(pyramids)
        I_bar = self.merge_intensity_maps(I_cs)
        C_bar = self.merge_color_maps(BY_cs, RG_cs)
        O_bar = self.merge_orientation_maps(O_cs_theta)

        saliency_map = (I_bar + C_bar + O_bar) / 3
        saliency_map = cv2.normalize(saliency_map, saliency_map, 0, 255, cv2.NORM_MINMAX)

        return saliency_map

