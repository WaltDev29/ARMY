from threading import Lock
from copy import deepcopy

class ImageStore:
    def __init__(self):
        self._lock = Lock()
        self._store = {}
        self._image_key = 0

    @property
    def store(self) -> dict:
        with self._lock:
            return deepcopy(self._store)
        
    def add_image(self, image:list) -> None:
        with self._lock:
            self._store[self._image_key] = image
            self._image_key += 1

    def get_image(self, key:str) -> list|None:
        with self._lock:
            return self._store.get(key, None).copy()
        
    def get_image_key(self) -> int:
        with self._lock:
            return self._image_key


image_store = ImageStore()