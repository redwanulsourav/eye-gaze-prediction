import cv2
import json

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
        videoPath = self.videoJson[str(videoIdx)]
        
        cap = cv2.VideoCapture(f'{self.datasetPath}/processed/videos/{videoPath}')
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        return totalFrames
    
    def getFrame(self, videoIdx: int, frameIdx: int):
        videoPath = self.videoJson[str(videoIdx)]

        cap = cv2.VideoCapture(f'{self.datasetPath}/processed/videos/{videoPath}')
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        assert (frameIdx < totalFrames), f'0 <= frameIdx < {totalFrames} is not satisfied.'

        cap.set(cv2.CAP_PROP_POS_FRAMES, frameIdx)
        read, frame = cap.read()

        assert (read == True), f'Unknown error, frame read failed'

        return frame
    
    def getAllFrames(self, videoIdx: int):
        print(self.videoJson)
        videoPath = self.videoJson[str(videoIdx)]

        cap = cv2.VideoCapture(f'{self.datasetPath}/processed/videos/{videoPath}')
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        allFrames = []

        read, frame = cap.read()
        while read:
            allFrames.append(frame)
            read, frame = cap.read()
        
        return allFrames
    
    def getGazeLocation(self, videoIdx: int, frameIdx: int, viewerIdx: int):
        gaze = self.gazeJson[str(videoIdx)][str(viewerIdx)][str(frameIdx)]
        return (gaze['x'], gaze['y'])
    
    def getViewerCount(self, videoIdx: int):
        gaze = self.gazeJson[str(videoIdx)]
        return len(gaze)
    
    def getAllGazeOfSingleViewer(self, videoIdx: int, viewerIdx: int):
        frameCount = self.getFrameCount(videoIdx)
        
        allGaze = []

        for i in range(int(frameCount)):
            allGaze.append(self.getGazeLocation(videoIdx, i, viewerIdx))
        
        return allGaze
