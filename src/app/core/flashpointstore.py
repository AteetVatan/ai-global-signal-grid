# flashpoint_dataset_singleton.py

import threading
from .flashpoint import FlashpointDataset, FlashpointItem

class FlashpointStore:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst.dataset = FlashpointDataset()
                    cls._instance = inst
        return cls._instance

    def clear(self):
        self.dataset = FlashpointDataset()

    def add_items(self, items: list[FlashpointItem]):
        self.dataset.extend(items)

    def get_items(self) -> FlashpointDataset:
        return self.dataset

flashpoint_store = FlashpointStore()
