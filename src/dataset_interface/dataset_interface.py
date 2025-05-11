import cv2
import json
from PIL import Image

class DatasetInterface():
    def __init__(self, datasetPath: str):
        """
            Initialize the dataset interface
            
            Parameters:
                `datasetPath` (str) -> Path to dataset root. (containing `raw` and `processed`)
        """
        self.datasetPath = datasetPath
        self.videoJsonPath = f'{datasetPath}/processed/videos/video_order.json'
        self.gazeJsonPath = f'{datasetPath}/processed/gaze/gaze_order.json'
        self.videoWidth = None
        self.videoHeight = None
        self.videoJson = None
        self.gazeJson = None
        
        with open(self.videoJsonPath, 'r') as f:
            contents = f.read()
            self.videoJson = json.loads(contents)

        with open(self.gazeJsonPath, 'r') as f:
            contents = f.read()
            self.gazeJson = json.loads(contents)

    def getFrameCount(self, videoIdx: int):
        """
            TODO: change video json format to include this,
                so that we don't have load video each time this
                function is called.
        """
        # # Cache frames
        # if self.cached == False or self.cachedVideoId != videoIdx:
        #     print('cache miss')
        #     self.getAllFrames(videoIdx)

        # if self.cached == True and self.cachedVideoId == videoIdx:
        #     return len(self.cachedFrames)

        videoPath = self.videoJson[str(videoIdx)]
        
        cap = cv2.VideoCapture(f'{self.datasetPath}/processed/videos/{videoPath}')
        totalFrames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return totalFrames
    
    def getFrame(self, videoIdx: int, frameIdx: int):
        # # Cache frames
        # if self.cached == False or self.cachedVideoId != videoIdx:
        #     print('cache miss')
        #     self.getAllFrames(videoIdx)

        # if self.cached == True and self.cachedVideoId == videoIdx:
        #     return self.cachedFrames[frameIdx]

        videoPath = self.videoJson[str(videoIdx)]

        cap = cv2.VideoCapture(f'{self.datasetPath}/processed/videos/{videoPath}')
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        assert (frameIdx < totalFrames), f'0 <= frameIdx < {totalFrames} is not satisfied.'

        cap.set(cv2.CAP_PROP_POS_FRAMES, frameIdx)
        read, frame = cap.read()

        assert (read == True), f'Unknown error, frame read failed'

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        return frame
    
    def getRangeFrames(self, videoIdx: int, start: int, end: int):
        videoPath = self.videoJson[str(videoIdx)]

        cap = cv2.VideoCapture(f'{self.datasetPath}/processed/videos/{videoPath}')
        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        assert (start < totalFrames) and (end <= totalFrames), f'0 <= {frameIdx} < {totalFrames} is not satisfied.'

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = start
        result = []

        for i in range(start, end):
            read, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            result.append(frame)
        return result

    def getAllFrames(self, videoIdx: int):
        # if self.cached == True and self.cachedVideoId == videoIdx:
        #     return self.cachedFrames

        videoPath = self.videoJson[str(videoIdx)]

        cap = cv2.VideoCapture(f'{self.datasetPath}/processed/videos/{videoPath}')
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        allFrames = []

        read, frame = cap.read()
        while read:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            allFrames.append(frame)
            read, frame = cap.read()
        
        self.cached = True
        self.cachedVideoId = videoIdx
        self.cachedFrames = allFrames
        return allFrames
    
    def getGazeLocation(self, videoIdx: int, frameIdx: int, viewerIdx: int):
        # Cache frames
        # if self.cached == False or self.cachedVideoId != videoIdx:
        #     print('cache miss')
        #     self.getAllFrames(videoIdx)

        gaze = self.gazeJson[str(videoIdx)][str(viewerIdx)][str(frameIdx)]
        return (gaze['x'], gaze['y'])
    
    def getViewerCount(self, videoIdx: int):
        # Cache frames
        # if self.cached == False or self.cachedVideoId != videoIdx:
        #     self.getAllFrames(videoIdx)

        gaze = self.gazeJson[str(videoIdx)]
        return len(gaze)
    
    def getAllGazeOfSingleViewer(self, videoIdx: int, viewerIdx: int):
        # Cache frames
        # if self.cached == False or self.cachedVideoId != videoIdx:
        #     print('cache miss')
        #     self.getAllFrames(videoIdx)

        frameCount = self.getFrameCount(videoIdx)
        
        allGaze = []

        for i in range(int(frameCount)):
            allGaze.append(self.getGazeLocation(videoIdx, i, viewerIdx))
        
        return allGaze
