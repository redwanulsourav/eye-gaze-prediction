class ObservableDict():
    def __init__(self, name: str):
        self.data = {}
        self.name = name
        self.changed = False
    
    def updateKey(self, key, value):
        self.data[key] = value
        self.changed = True
    
    
    