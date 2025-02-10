import cv2
import unittest
import os
import numpy as np


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import itti_koch

class TestIttiKoch(unittest.TestCase):
    def __init__(self, pathToRes):
        self.model = itti_koch.Itti_Koch_Model()
        self.resPath = pathToRes    # Resource path
        
    def testDownsample(self):
        img = cv2.imread(self.resPath)
        assert img is not None

        level0 = self.model.downsample(img, 0)
        assert img.shape == level_0.shape

        level1 = self.model.downsample(level0, 1)
        assert (img.shape[0] // 2, img.shape[1] // 2, img.shape[2]) == level1.shape
        level10 = self.model.downsample(img, 1)
        assert (level1.shape == level10.shape)

        level2 = self.model.downsample(level1, 1)
        assert (img.shape[0] // 4, img.shape[1] // 4, img.shape[2]) == level2.shape
        level20 = self.model.downsample(img, 2)
        assert(level2.shape == level20.shape)

        level3 = self.model.downsample(level2, 1)
        assert (img.shape[0] // 8, img.shape[1] // 8, img.shape[2]) == level3.shape
        level30 = self.model.downsample(img, 3)
        assert(level3.shape == level30.shape)

        level4 = self.model.downsample(level3, 1)
        assert (img.shape[0] // 16, img.shape[1] // 16, img.shape[2]) == level4.shape
        level40 = self.model.downsample(img, 4)
        assert(level4.shape == level40.shape)
    
    def testGaussianPyramid(self):
        img = cv2.imread(self.resPath)
        assert img is not None

        img = cv2.resize(img, (512, 512))

        scales = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        pyramid = self.model.gaussianPyramid(img, scales)

        assert len(scales) == len(pyramid)
        
    
        for scale, downsampled in pyramid.items():
            assert (img.shape[0] // (1 << scale), img.shape[1] // (1 << scale), img.shape[2]) == downsampled.shape
            # cv2.imwrite(f'{self.path_to_res}/outputs/{scale}_i.png', downsampled)
    
    def testExtractVisualFeatures(self):
        img = cv2.imread(self.resPath)
        assert img is not None

        scales = [0, 1, 2, 3, 4]
        pyramids = self.model.extract_visual_features(img, scales)

        for i, scale in enumerate(scales):
            downsampled = pyramids['intensity_pyr'][scale]
            assert (img.shape[0] // (1 << scale), img.shape[1] // (1 << scale)) == downsampled.shape, \
                f'Shape mismatch, {(img.shape[0] // (1 << scale), img.shape[1] // (1 << scale))} != {downsampled.shape}'

            downsampled = pyramids['red_pyr'][scale]
            assert (img.shape[0] // (1 << scale), img.shape[1] // (1 << scale)) == downsampled.shape, \
                f'Shape mismatch, {(img.shape[0] // (1 << scale), img.shape[1] // (1 << scale))} != {downsampled.shape}'

            downsampled = pyramids['green_pyr'][scale]
            assert (img.shape[0] // (1 << scale), img.shape[1] // (1 << scale)) == downsampled.shape, \
                f'Shape mismatch, {(img.shape[0] // (1 << scale), img.shape[1] // (1 << scale))} != {downsampled.shape}'

            downsampled = pyramids['blue_pyr'][scale]
            assert (img.shape[0] // (1 << scale), img.shape[1] // (1 << scale)) == downsampled.shape, \
                f'Shape mismatch, {(img.shape[0] // (1 << scale), img.shape[1] // (1 << scale))} != {downsampled.shape}'

            downsampled = pyramids['yellow_pyr'][scale]
            assert (img.shape[0] // (1 << scale), img.shape[1] // (1 << scale)) == downsampled.shape, \
                f'Shape mismatch, {(img.shape[0] // (1 << scale), img.shape[1] // (1 << scale))} != {downsampled.shape}'
    
    def test_upscale_image(self):
        img = cv2.imread(f'{self.path_to_res}/images/0.png')
        assert img is not None

        upscaled = self.model.upscale_image(img, 4, 2)
        assert (upscaled.shape[0], upscaled.shape[1]) == (img.shape[0] * 4, img.shape[1] * 4)

    def test_degrees_to_radian(self):
        radian = self.model.degrees_to_radian(0)
        assert radian == 0

        radian = self.model.degrees_to_radian(90)
        assert radian == np.pi/2

        radian = self.model.degrees_to_radian(180)
        assert radian == np.pi

    def test_get_oriented_gabor_pyramids(self):
        # Check if all angles are present as keys.
        # Check if all the C,S values are present as keys.
        img = cv2.imread(f'{self.path_to_res}/images/0.png')
        assert img is not None

        scales = [0, 1, 2, 3]
        angles = [0, 45, 90, 135]
        pyramid = self.model.get_oriented_gabor_pyramid(img, len(angles), len(scales))

        
        for i in range(len(scales)):
            for j in range(1, len(angles) + 1):
                assert scales[i] in list(pyramid.keys())
                # assert angles[j] in list(pyramid[scale].keys())
                assert pyramid[i][j] is not None
                downscaled_shape = (img.shape[0] // (1 << scales[i]), img.shape[1] // (1 << scales[i]))
                assert pyramid[i][j].shape == downscaled_shape, f'{pyramid[i][j].shape} != {downscaled_shape}'
                        
    def test_across_scale_differences(self):
        img = cv2.imread(f'{self.path_to_res}/images/lena.png')
        assert img is not None

        scales = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        pyramids = self.model.extract_visual_features(img, scales)
        
        angles = [0, 45, 90, 135]
        gabor_pyramid = self.model.get_oriented_gabor_pyramid(img, 4, 9)

        # all_pyramids = (color_pyramids + (gabor_pyramid,))
        pyramids['orientation_pyr'] = gabor_pyramid
        assert len(pyramids) == 6

        for key, pyramid in pyramids.items():
            keys = list(pyramid.keys())
            for scale in scales:
                assert scale in keys, f'{scale} is not in pyramid'

        (I_cs, BY_cs, RG_cs, O_cs_theta) = self.model.calculate_across_scale_differences(pyramids)

        # Verify I_cs
        for i, temp_dict in I_cs.items():
            for j in self.model._delta:
                _s = i + j
                assert _s in list(temp_dict.keys())

                map_ = I_cs[i][_s]

                # Verify dimensions
                expected_dim = (img.shape[0] // (1 << i), img.shape[1] // (1 << i))
                assert expected_dim == map_.shape, f'{expected_dim} != {img.shape} for scale {i}'

        # Verify BY_cs
        for i, temp_dict in BY_cs.items():
            for j in self.model._delta:
                _s = i + j
                assert _s in list(temp_dict.keys())

                map_ = BY_cs[i][_s]

                # Verify dimensions
                expected_dim = (img.shape[0] // (1 << i), img.shape[1] // (1 << i))
                assert expected_dim == map_.shape, f'{expected_dim} != {img.shape} for scale {i}'
        
        # Verify RG_cs
        for i, temp_dict in RG_cs.items():
            for j in self.model._delta:
                _s = i + j
                assert _s in list(temp_dict.keys())

                map_ = RG_cs[i][_s]

                # Verify dimensions
                expected_dim = (img.shape[0] // (1 << i), img.shape[1] // (1 << i))
                assert expected_dim == map_.shape, f'{expected_dim} != {img.shape} for scale {i}'

        # Verify O_cs_theta
        for i, temp_dict in O_cs_theta.items():
            for j in self.model._delta:
                _s = i + j
                assert _s in list(temp_dict.keys())
                for k, temp_dict2 in O_cs_theta[i][_s].items():
                    map_ = O_cs_theta[i][_s][k]
                    # Verify dimensions
                    expected_dim = (img.shape[0] // (1 << i), img.shape[1] // (1 << i))
                    assert expected_dim == map_.shape, f'{expected_dim} != {img.shape} for scale {i}'
            
    def test_local_maximas(self):
        img = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [2, 1, 9, 3, 4, 5],
                [1, 9, 4, 5, 1, 3],
                [9, 1, 5, 2, 3, 5],
                [0, 2, 3, 9, 1, 5],
                [9, 8, 1, 5, 3, 1],
            ]
        )
        # sum_ = (9 + 9 + 9 + 6) + (9 + 9 + 9 + 5) + (9 + 9 + 9 + 9) + (9 + 9 + 9 + 9)
        sum_ = (9 + 9) + (9 + 9) # + (9 + 9 + 9 + 9) + (9 + 9 + 9 + 9)

        assert self.model.local_maximas(img) == sum_/4, f'{self.model.local_maximas(img)} != {sum_/16}'


    def test_map_normalization(self):
        img = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [2, 1, 9, 3, 4, 5],
                [1, 9, 4, 5, 1, 3],
                [9, 1, 5, 2, 3, 5],
                [0, 2, 3, 9, 1, 5],
                [9, 8, 1, 5, 3, 1],
            ]
        )
        img = img.astype(np.float64)
        normalized_map = self.model.map_normalization(img)
        assert normalized_map.shape == img.shape

        # TODO: Add test for value check.
    
    def test_merge_intensity_maps(self):
        img = cv2.imread(f'{self.path_to_res}/images/lena.png')
        assert img is not None

        img = cv2.resize(img, (512, 512))

        scales = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        all_pyramids = self.model.extract_visual_features(img, scales)
        
        angles = [0, 45, 90, 135]
        gabor_pyramid = self.model.get_oriented_gabor_pyramid(img, 4, 9)

        # all_pyramids = (color_pyramids + (gabor_pyramid,))
        all_pyramids['orientation_pyr'] = gabor_pyramid
        assert len(all_pyramids) == 6

        for key, pyramid in all_pyramids.items():
            keys = list(pyramid.keys())
            for scale in scales:
                assert scale in keys, f'{scale} is not in pyramid'

        (I_cs, BY_cs, RG_cs, O_cs_theta) = self.model.calculate_across_scale_differences(all_pyramids)

        I_bar = self.model.merge_intensity_maps(I_cs)

        # Verify shape
        expected_dim = (img.shape[0] // (1 << 4), img.shape[1] // (1 << 4))
        assert expected_dim == I_bar.shape, f'{expected_shape} != {I_bar.shape}'

        I_bar = cv2.normalize(I_bar, -1, 0, 255, cv2.NORM_MINMAX)
        I_bar = cv2.resize(I_bar, (512, 512))
        cv2.imwrite(f'{self.path_to_res}/outputs/i_bar.png', I_bar)


    def test_merge_color_maps(self):
        img = cv2.imread(f'{self.path_to_res}/images/paper_input.png')
        assert img is not None

        img = cv2.resize(img, (512, 512))
        scales = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        all_pyramids = self.model.extract_visual_features(img, scales)
        
        angles = [0, 45, 90, 135]
        gabor_pyramid = self.model.get_oriented_gabor_pyramid(img, 4, 9)

        # all_pyramids = (color_pyramids + (gabor_pyramid,))
        all_pyramids['orientation_pyr'] = gabor_pyramid
        assert len(all_pyramids) == 6

        for key, pyramid in all_pyramids.items():
            keys = list(pyramid.keys())
            for scale in scales:
                assert scale in keys, f'{scale} is not in pyramid'

        (I_cs, BY_cs, RG_cs, O_cs_theta) = self.model.calculate_across_scale_differences(all_pyramids)

        C_bar = self.model.merge_color_maps(BY_cs, RG_cs)

        # Verify shape
        expected_dim = (img.shape[0] // (1 << 4), img.shape[1] // (1 << 4))
        assert expected_dim == C_bar.shape, f'{expected_shape} != {C_bar.shape}'
        
        C_bar = cv2.normalize(C_bar, -1, 0, 255, cv2.NORM_MINMAX)
        C_bar = cv2.resize(C_bar, (512, 512))
        cv2.imwrite(f'{self.path_to_res}/outputs/c_bar.png', C_bar)

    def test_merge_orientation_maps(self):
        img = cv2.imread(f'{self.path_to_res}/images/lena.png')
        assert img is not None

        scales = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        all_pyramids = self.model.extract_visual_features(img, scales)
        
        angles = [0, 45, 90, 135]
        gabor_pyramid = self.model.get_oriented_gabor_pyramid(img, 4, 9)

        # all_pyramids = (color_pyramids + (gabor_pyramid,))
        all_pyramids['orientation_pyr'] = gabor_pyramid
        assert len(all_pyramids) == 6

        for key, pyramid in all_pyramids.items():
            keys = list(pyramid.keys())
            for scale in scales:
                assert scale in keys, f'{scale} is not in pyramid'

        (I_cs, BY_cs, RG_cs, O_cs_theta) = self.model.calculate_across_scale_differences(all_pyramids)

        O_bar = self.model.merge_orientation_maps(O_cs_theta)

        # Verify shape
        expected_dim = (img.shape[0] // (1 << 4), img.shape[1] // (1 << 4))
        assert expected_dim == O_bar.shape, f'{expected_shape} != {C_bar.shape}'

    
    def test_get_saliency_map(self):
        img = cv2.imread(f'{self.path_to_res}/images/paper_input.png')
        img = cv2.resize(img, (512, 512))
        assert img is not None

        saliency_map = self.model.get_saliency_map(img)

        expected_dim = (img.shape[0] // (1 << 4), img.shape[1] // (1 << 4))
        assert expected_dim == saliency_map.shape, f'{expected_dim} != {saliency_map.shape}'

        # saliency_map = saliency_map.astype(np.uint8)
        saliency_map = cv2.normalize(saliency_map, -1, 0, 255, cv2.NORM_MINMAX)
        saliency_map = cv2.resize(saliency_map, (512, 512))
        cv2.imwrite(f'{self.path_to_res}/outputs/output.png', saliency_map)
        
    def write_all_images(self):
        img = cv2.imread(f'{self.path_to_res}/images/paper_input.png')
        img = cv2.resize(img, (512, 512))
        assert img is not None

        pyramids = self.model.extract_visual_features(img, list(range(0, 9)))
        # visual_features = (I_sigma, R_sigma, G_sigma, B_sigma, Y_sigma)
        intensity = pyramids['intensity_pyr']
        colors = (pyramids['red_pyr'], pyramids['green_pyr'], pyramids['blue_pyr'], pyramids['yellow_pyr'])
        orientation = self.model.get_oriented_gabor_pyramid(img, 4, 9)
        pyramids['orientation_pyr'] = orientation
        

        (I_cs, BY_cs, RG_cs, O_cs_theta) = self.model.calculate_across_scale_differences(pyramids)

        I_bar = self.model.merge_intensity_maps(I_cs)
        C_bar = self.model.merge_color_maps(BY_cs, RG_cs)
        O_bar = self.model.merge_orientation_maps(O_cs_theta)

        for scale, img in pyramids['intensity_pyr'].items():
            cv2.imwrite(f'{self.path_to_res}/outputs/I_sigma_{scale}.png', img)
        for scale, img in pyramids['red_pyr'].items():
            cv2.imwrite(f'{self.path_to_res}/outputs/R_sigma_{scale}.png', img)
        for scale, img in pyramids['green_pyr'].items():
            cv2.imwrite(f'{self.path_to_res}/outputs/G_sigma_{scale}.png', img)
        for scale, img in pyramids['yellow_pyr'].items():
            cv2.imwrite(f'{self.path_to_res}/outputs/Y_sigma_{scale}.png', img)
        for scale, temp_dict in orientation.items():
            for angle, img in temp_dict.items():
                temp = cv2.normalize(img, -1, 0, 255, cv2.NORM_MINMAX)
                cv2.imwrite(f'{self.path_to_res}/outputs/orientation_sigma_{scale}_{angle}.png', img)
        
        for c, temp_dict in I_cs.items():
            for s, img in temp_dict.items():
                cv2.imwrite(f'{self.path_to_res}/outputs/I_c={c},s={s}.png', img)
        for c, temp_dict in BY_cs.items():
            for s, img in temp_dict.items():
                cv2.imwrite(f'{self.path_to_res}/outputs/BY_c={c},s={s}.png', img)
        for c, temp_dict in RG_cs.items():
            for s, img in temp_dict.items():
                cv2.imwrite(f'{self.path_to_res}/outputs/RG_c={c},s={s}.png', img)
        for c, temp_dict in O_cs_theta.items():
            for s, temp_dict2 in temp_dict.items():
                for theta, img in temp_dict2.items():
                    cv2.imwrite(f'{self.path_to_res}/outputs/O_c={c},s={s},theta={theta}.png', img)
        
        I_bar_temp = cv2.normalize(I_bar, -1, 0, 255, cv2.NORM_MINMAX)
        I_bar_temp = cv2.resize(I_bar_temp, (512, 512))
        cv2.imwrite(f'{self.path_to_res}/outputs/I_bar.png', I_bar_temp)

        C_bar_temp = cv2.normalize(C_bar, -1, 0, 255, cv2.NORM_MINMAX)
        C_bar_temp = cv2.resize(C_bar_temp, (512, 512))
        cv2.imwrite(f'{self.path_to_res}/outputs/C_bar.png', C_bar_temp)

        O_bar_temp = cv2.normalize(O_bar, -1, 0, 255, cv2.NORM_MINMAX)
        O_bar_temp = cv2.resize(O_bar_temp, (512, 512))
        cv2.imwrite(f'{self.path_to_res}/outputs/O_bar.png', O_bar_temp)

    def test_all(self):
        # print('Running tests for Itti&Koch model')
        # print('[1 / 13] Running test_gaussian_downsample...', end=' ')
        # self.test_gaussian_downsample()
        # print('passed.')

        # print('[2 / 13] Running test_gaussian_pyramid...', end=' ')
        # self.test_gaussian_pyramid()
        # print('passed.')

        # print('[3 / 13] Running test_extract_visual_features...', end=' ')
        # self.test_extract_visual_features()
        # print('passed.')

        # print('[4 / 13] Running test_upscale_image...', end=' ')
        # self.test_upscale_image()
        # print('passed.')

        # print('[5 / 13] Running test_degrees_to_radian...', end=' ')
        # self.test_degrees_to_radian()
        # print('passed.')

        # print('[6 / 13] Running test_local_maximas...', end=' ')
        # self.test_local_maximas()
        # print('passed.')

        # print('[7 / 13] Running test_map_normalization...', end=' ')
        # self.test_map_normalization()
        # print('passed.')

        # print('[8 / 13] Running test_merge_intensity_maps...', end=' ')
        # self.test_merge_intensity_maps()
        # print('passed.')

        # print('[9 / 13] Running test_merge_color_maps...', end=' ')
        # self.test_merge_color_maps()
        # print('passed.')

        # print('[10 / 13] Running test_merge_orientation_maps...', end=' ')
        # self.test_merge_orientation_maps()
        # print('passed.')

        # print('[11 / 13] Running test_get_oriented_gabor_pyramids...', end=' ')
        # self.test_get_oriented_gabor_pyramids()
        # print('passed.')

        # print('[12 / 13] Running test_across_scale_differences...', end=' ')
        # self.test_across_scale_differences()
        # print('passed.')

        print('[13 / 13] Running test_get_saliency_map...', end=' ')
        self.test_get_saliency_map()
        print('passed.')

        print('Saving all debug images...', end=' ')
        self.write_all_images()
        print('finished.')

        print('All tests passed.')



if __name__ == '__main__':
    obj = TestIttiKoch()
    obj.test_all()
