import argparse
import scipy.io
import os
import shutil
import json
import math

# TODO: Write some echo to update about progress.

def processVideos(datasetRoot):
    # Copy all the video files from raw

    allVideos = os.listdir(f'{datasetRoot}/raw/ERB3_Stimuli/')
    videoJson = {}

    for i, video in enumerate(allVideos):
        srcPath = f'{datasetRoot}/raw/ERB3_Stimuli/{video}'
        dstPath = f'{datasetRoot}/processed/videos/{video}'

        shutil.copy(srcPath, dstPath)
        videoJson[i] = video

    with open(f'{datasetRoot}/processed/videos/video_order.json', 'w') as f:
        f.write(json.dumps(videoJson))

    return videoJson

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

    videoJson = processVideos(ap.dataset_root)
    
    # Gaze Data
    mat = scipy.io.loadmat(f'{ap.dataset_root}/raw/coutrot_database1.mat')
    gazeJson = {}

    for key, videoName in videoJson.items():
        gazeData = mat['Coutrot_Database1']['OriginalSounds'][0, 0][videoName.split('.')[0]][0, 0]['data'][0, 0]
        metaData = mat['Coutrot_Database1']['OriginalSounds'][0, 0][videoName.split('.')[0]][0, 0]['info'][0, 0]

        vidWidth = metaData['vidwidth'][0, 0][0, 0]
        vidHeight = metaData['vidheight'][0, 0][0, 0]
        frameCount = metaData['nframe'][0, 0][0, 0]

        viewerCount = gazeData.shape[2]  # Viewer Count
        gazeJson[key] = {}

        for i in range(viewerCount):
            gazeJson[key][i] = {}

            for j in range(frameCount):
                gazeJson[key][i][j] = {}

                x = gazeData[0][j][i]
                y = gazeData[1][j][i]

                if math.isnan(x):
                    x = 0.5
                if math.isnan(y):
                    y = 0.5
                
                # assert math.isnan(x)
                gazeJson[key][i][j]['x'] = min(x / vidWidth, 1)
                gazeJson[key][i][j]['y'] = min(y / vidHeight, 1)

    with open(f'{ap.dataset_root}/processed/gaze/gaze_order.json', 'w') as f:
        f.write(json.dumps(gazeJson))


