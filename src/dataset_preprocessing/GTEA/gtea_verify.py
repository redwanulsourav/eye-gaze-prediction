import cv2
import argparse
import os
import json
import time

def main(rootPath, videoName):
    # allVideos = os.listdir(videoPath)

    # for video in allVideos:
    videoPath = os.path.join(rootPath, 'raw', 'Videos')
    videoPath = os.path.join(videoPath, f'{videoName}.mpg')
    print(f'Playing {videoName}')
    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1 / fps

    read, frame = cap.read()

    gazeJsonLocation = os.path.join(f'{rootPath}', 'processed', 'gaze', 'gaze_order.json')
    videoJsonLocation = os.path.join(f'{rootPath}', 'processed', 'videos', 'video_order.json')
    videoData = json.load(open(videoJsonLocation, 'r'))
    gazeData = json.load(open(gazeJsonLocation, 'r'))

    videoIdx = None
    for idx, name in videoData.items():
        if name == f'{videoName}.mpg':
            videoIdx = idx
            break

    gazeData = gazeData[videoIdx]['0']
    # print(gazeData)
    frameIdx = 0
    cv2.circle(frame, (int(gazeData[str(frameIdx)]['x'] * frame.shape[1]), int(gazeData[str(frameIdx)]['y'] * frame.shape[0])), 3, (0, 255, 0))

    while read:
        start_time = time.time()
        cv2.imshow('frame', frame)
        elapsed_time = time.time() - start_time
        wait_time = max(1, int(frame_delay * 1000 - elapsed_time * 1000))
        if cv2.waitKey(wait_time) == ord('q'):
            break
        
        read, frame = cap.read()
        cv2.circle(frame, (int(gazeData[str(frameIdx)]['x'] * frame.shape[1]), int(gazeData[str(frameIdx)]['y'] * frame.shape[0])), 3, (0, 255, 0)) 
        frameIdx+=1
    # cap.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-r', '--root', required = True)
    ap.add_argument('-v', '--video', required = True)
    ap = ap.parse_args()

    main(ap.root, ap.video)