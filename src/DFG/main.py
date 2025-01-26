import torch
import cv2
import numpy as np

from model import FrameGenerator, TemporalSaliencyPredictor, Discriminator

def main():
    img = cv2.imread('/data/rsourave/datasets/extracted/ERB3_Stimuli_Extracted/1/frames/0.jpg')
    img = cv2.resize(img, (64, 64))
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis = 0)
    print(img.shape)
    model = FrameGenerator()

    out = model(torch.from_numpy(img).float())

    print(out.shape)

    model2 = TemporalSaliencyPredictor()
    out2 = model2(out)

    print(out2.shape)

    dis = Discriminator()

    y = dis(out)
    print(f'y: {y}')

if __name__ == '__main__':
    main()