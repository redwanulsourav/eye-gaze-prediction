import cv2
import json
import os
import shutil

if __name__ == '__main__':
    try:
        os.mkdir(f'/data/rsourave/datasets/GTEA/processed')
    except:
        print('`processed` directory already exists')

    try:
        os.mkdir(f'/data/rsourave/datasets/GTEA/processed/videos')
    except:
        print('`videos` directory already exists')
    
    try:
        os.mkdir(f'/data/rsourave/datasets/GTEA/processed/gaze')
    except:
        print('`gaze` directory already exists')

    allVideos = os.listdir(f'/data/rsourave/datasets/GTEA/raw/Videos')
    videoJson = {}

    for i, video in enumerate(allVideos):
        srcVideo = f'/data/rsourave/datasets/GTEA/raw/Videos/{video}'
        dstVideo = f'/data/rsourave/datasets/GTEA/processed/videos/{video}'

        shutil.copy(srcVideo, dstVideo)
        videoJson[i] = video
    
    with open(f'/data/rsourave/datasets/GTEA/processed/videos/video_order.json', 'w') as f:
        f.write(json.dumps(videoJson))

    gazeJson = {}
    """ Assumptions:
        1. The Tobii eye tracker is synchronized with the video, that is the Tobii
        eye tracker  starts recording as soon as the video starts.
        2. 
    """

    for key, videoName in videoJson.items():
        viewerCount = 1
        cap = cv2.VideoCapture(f'/data/rsourave/datasets/GTEA/processed/videos/{videoName}')
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # cap.close()
        gazeJson[key] = {}
        gazeJson[key][0] = {}
        videoFileName = videoName.split('.')[0]
        f = open(f'/data/rsourave/datasets/GTEA/raw/Gaze/{videoFileName}-Glasses-Data.tsv', 'r')
        contents = f.read()
        contents = contents.split('\n')
        contents = contents[22:]
        contents = [x.split('\t') for x in contents]
        f.close()
        
        videoDuration = frameCount / 30
        if contents[-1][0] == '':
            contents = contents[: -1]
        lastTimestamp = int(contents[-1][0])
        secondsPerTick = videoDuration / lastTimestamp
        tickPerFrame = lastTimestamp / frameCount 
        """
            Each line in the gaze file contains a timestamp.
            Divide the final timestamp in the file with framecount to get
            timestamp `tick` per frame.

            To find out which frame a given timestamp belongs to, divide the
            timestamp with number of `ticks` per frame. It should give the 
            index of the frame.
        """
        gazeXs = [[] for i in range(frameCount)]
        gazeYs = [[] for i in range(frameCount)]
        print()
        for line in contents:
            timestamp = int(line[0])
            frameIndex = timestamp // round(tickPerFrame)
            try:
                x = int(line[1]) / 640
            except:
                x = 0.5

            try:
                y = int(line[2]) / 480
            except:
                y = 0.5

            try:
                gazeXs[frameIndex].append(min(max(x, 0), 1))
                gazeYs[frameIndex].append(min(max(y, 0), 1))
            except IndexError:
                print(f'{timestamp} made frameIndex = {frameIndex}, with frameCount = {frameCount}')

        for i in range(frameCount):
            gazeJson[key][0][i] = {}
            
            try:
                gazeJson[key][0][i]['x'] = sum(gazeXs[i]) / len(gazeXs[i])
                gazeJson[key][0][i]['y'] = sum(gazeYs[i]) / len(gazeYs[i])
            except ZeroDivisionError:
                print(f'Zero division error at index = {i}')
                
    with open(f'/data/rsourave/datasets/GTEA/processed/gaze/gaze_order.json', 'w') as f:
        f.write(json.dumps(gazeJson))

