import threading

class ObservableDict:
    def __init__(self, name="observable"):
        self._dict = {}
        self._lock = threading.Lock()
        self.name = name
        self.changed = False  # You can customize this behavior

    def updateKey(self, key, value):
        with self._lock:
            self._dict[key] = value
            self.changed = True
            print(f"[{self.name}] Updated: {key} = {value}")

    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)

    def items(self):
        with self._lock:
            return list(self._dict.items())  # Return a snapshot

    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]

    def __setitem__(self, key, value):
        self.updateKey(key, value)

    def __contains__(self, key):
        with self._lock:
            return key in self._dict

    def __str__(self):
        with self._lock:
            return str(self._dict)