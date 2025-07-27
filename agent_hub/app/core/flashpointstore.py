# ┌───────────────────────────────────────────────────────────────┐
# │  Copyright (c) 2025 Ateet Vatan Bahmani                      │
# │  Project: MASX AI – Strategic Agentic AI System              │
# │  All rights reserved.                                        │
# └───────────────────────────────────────────────────────────────┘
#
# MASX AI is a proprietary software system developed and owned by Ateet Vatan Bahmani.
# The source code, documentation, workflows, designs, and naming (including "MASX AI")
# are protected by applicable copyright and trademark laws.
#
# Redistribution, modification, commercial use, or publication of any portion of this
# project without explicit written consent is strictly prohibited.
#
# This project is not open-source and is intended solely for internal, research,
# or demonstration use by the author.
#
# Contact: ab@masxai.com | MASXAI.com

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
