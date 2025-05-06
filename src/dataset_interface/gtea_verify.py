import cv2
import argparse
import os

def main(rootPath, videoName):
    # allVideos = os.listdir(videoPath)

    # for video in allVideos:
    videoPath = os.path.join(rootPath, 'raw', 'Videos')
    videoPath = os.path.join(videoPath, f'{videoName}.mpg')
    print(f'Playing {videoName}')
    cap = cv2.VideoCapture(videoPath)
    read, frame = cap.read()

    while read:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        read, frame = cap.read()
    # cap.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-r', '--root', required = True)
    ap.add_argument('-v', '--video', required = True)
    ap = ap.parse_args()

    main(ap.root, ap.video)