import cv2
from itti_koch import IttiKochModel
import numpy as np

model = IttiKochModel()
img = cv2.imread('../shreelock-gbvs/images/4.jpg')
# expected = cv2.imread('STIMautobahn/DBtest/cB10_K027.ppm')
img = cv2.resize(img, (1024, 1024))
smap = model.saliencyMap(img)
print(smap.shape)
cv2.imwrite('outsmap.jpg', smap)
# smap = smap.astype(np.uint8)
# smap2 = cv2.resize(smap, (1024, 1024))
# smap3 = smap2.astype(np.float64)/255.0
# out3 = img.copy()
# out3[:,:,0] = out3[:, :, 0] * smap3
# out3[:,:,1] = out3[:, :, 1] * smap3
# out3[:,:,2] = out3[:, :, 2] * smap3

# # out3 = (img * smap3).astype(np.uint8)

# cv2.imwrite('smap.jpg', smap2)
# cv2.imwrite('out3.jpg', out3)
