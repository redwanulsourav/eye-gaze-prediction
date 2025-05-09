import cv2
import argparse
import os
import time

def main(rootPath, videoName):
    # allVideos = os.listdir(videoPath)

    # for video in allVideos:
    videoPath = os.path.join(rootPath, 'raw', 'Videos')
    videoPath = os.path.join(videoPath, f'{videoName}.mpg')
    # print(videoPath)
    print(f'Playing {videoName}')
    
    
    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1 / fps
    
    read, frame = cap.read()

    while read:
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow('Video', frame)

        elapsed_time = time.time() - start_time
        wait_time = max(1, int(frame_delay * 1000 - elapsed_time * 1000))
    
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-r', '--root', required = True)
    ap.add_argument('-v', '--video', required = True)
    ap = ap.parse_args()

    main(ap.root, ap.video)