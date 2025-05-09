import argparse
import scipy.io
import os
import shutil
import json
import math

# TODO: Write some echo to update about progress.

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-r', '--dataset_root', help = 'Path to dataset root', required = True, type = str)
    ap = ap.parse_args()

    # Create the folders 'videos' and 'gaze'
    try:
        os.mkdir(f'{ap.dataset_root}/processed/videos')
        os.mkdir(f'{ap.dataset_root}/processed/gaze')
    except FileExistsError:
        pass

    # Copy all the video files from raw

    allVideos = os.listdir(f'{ap.dataset_root}/raw/ERB3_Stimuli/')
    videoJson = {}

    for i, video in enumerate(allVideos):
        srcPath = f'{ap.dataset_root}/raw/ERB3_Stimuli/{video}'
        dstPath = f'{ap.dataset_root}/processed/videos/{video}'

        shutil.copy(srcPath, dstPath)
        videoJson[i] = video

    with open(f'{ap.dataset_root}/processed/videos/video_order.json', 'w') as f:
        f.write(json.dumps(videoJson))

    # Gaze Data
    mat = scipy.io.loadmat(f'{ap.dataset_root}/raw/coutrot_database1.mat')
    gazeJson = {}

    for key, videoName in videoJson.items():
        print(videoName)
        data = mat['Coutrot_Database1'][0][0]['OriginalSounds'][videoName.split('.')[0]][0][0]['data'][0][0]
        info = mat['Coutrot_Database1'][0][0]['OriginalSounds'][videoName.split('.')[0]][0][0]['info'][0][0]
        videoWidth = float(info['vidwidth'][0][0][0][0])
        videoHeight = float(info['vidheight'][0][0][0][0])
        nFrame = int(info['nframe'][0][0][0][0])
        # Viewer Count
        viewerCount = data.shape[2]
        # frameCount = data.shape[1]
        gazeJson[key] = {}

        for i in range(viewerCount):
            gazeJson[key][i] = {}

            for j in range(nFrame):
                gazeJson[key][i][j] = {}

                x = float(data[0][j][i])
                y = float(data[1][j][i])

                if math.isnan(x):
                    x = videoWidth / 2
                if math.isnan(y):
                    y = videoHeight / 2
                
                # assert math.isnan(x)
                gazeJson[key][i][j]['x'] = x / 720
                gazeJson[key][i][j]['y'] = y / 576

    with open(f'{ap.dataset_root}/processed/gaze/gaze_order.json', 'w') as f:
        f.write(json.dumps(gazeJson))


