import json
import posix_ipc
import time

class Message:
    def __init__(self, msgType: str):
        self.type = msgType
        self.index = None
        self.gazeLocations = {}
    
    def __str__(self):
        data = f'{
            "type": "RecGL", 
            "gaze_locations":  
        }'