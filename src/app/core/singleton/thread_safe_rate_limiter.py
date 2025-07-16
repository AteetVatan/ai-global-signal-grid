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

import threading
import time


class ThreadSafeRateLimiter:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, max_calls_per_sec: int):
        self.max_calls = max_calls_per_sec
        self.time_window = 1.0
        self.call_times = []
        self.lock = threading.Lock()

    @classmethod
    def get_instance(cls, max_calls_per_sec: int = 5):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(max_calls_per_sec)
            return cls._instance

    def acquire(self):
        with self.lock:
            now = time.time()
            self.call_times = [t for t in self.call_times if now - t < self.time_window]
            if len(self.call_times) >= self.max_calls:
                wait_time = self.time_window - (now - self.call_times[0])
                time.sleep(wait_time)
            self.call_times.append(time.time())
