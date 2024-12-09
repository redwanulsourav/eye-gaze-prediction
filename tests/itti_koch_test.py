import cv2
import unittest
import os

from ..src import itti_koch

class TestIttiKoch(unittest.TestCase):
    def __init__(self):
        self.path_to_res = f'{os.path.dirname(os.path.abspath(__file__))}/res/'
        
    def test_gaussian_pyramid(self):
        img = cv2.imread(f'{self.path_to_res}/images/0.png')
        assert img is not None

        level_0 = itti_koch.gaussian_pyramid(img, 0)
        assert img.shape == level_0.shape

        level_1 = itti_koch.gaussian_pyramid(level_0, 1)
        assert (img.shape[0] // 2, img.shape[1] // 2, img.shape[2]) == level_1.shape
        level_1_1 = itti_koch.gaussian_pyramid(img, 1)
        assert (level_1.shape == level_1_1.shape)

        level_2 = itti_koch.gaussian_pyramid(level_1, 1)
        assert (img.shape[0] // 4, img.shape[1] // 4, img.shape[2]) == level_2.shape
        level_2_1 = itti_koch.gaussian_pyramid(img, 2)
        assert(level_2.shape == level_2_1.shape)

        level_3 = itti_koch.gaussian_pyramid(level_2, 1)
        assert (img.shape[0] // 8, img.shape[1] // 8, img.shape[2]) == level_3.shape
        level_3_1 = itti_koch.gaussian_pyramid(img, 3)
        assert(level_3.shape == level_3_1.shape)

        level_4 = itti_koch.gaussian_pyramid(level_3, 1)
        assert (img.shape[0] // 16, img.shape[1] // 16, img.shape[2]) == level_4.shape
        level_4_1 = itti_koch.gaussian_pyramid(img, 4)
        assert(level_4.shape == level_4_1.shape)


obj = TestIttiKoch()
obj.test_gaussian_pyramid()