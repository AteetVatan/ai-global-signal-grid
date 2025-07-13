"""
Model-aware spaCy singleton manager for MASX AI.
Keeps one loaded instance per unique spaCy model name.
Automatically downloads missing models.
"""

import threading
import spacy
import sys
import subprocess
import importlib
from typing import Dict, Union
from ..enums import SpaCyModelName


class SpacySingleton:
    _lock = threading.Lock()
    _instances: Dict[str, spacy.Language] = {}

    @classmethod
    def get(cls, model_name: Union[SpaCyModelName, str] = SpaCyModelName.EN_CORE_WEB_SM) -> spacy.Language:
        """
        Get a spaCy NLP pipeline instance for the given model name.
        Downloads the model if not already installed.

        Args:
            model_name: spaCy model enum or string

        Returns:
            spacy.Language: Loaded spaCy NLP pipeline
        """
        model_str = model_name.value if isinstance(model_name, SpaCyModelName) else model_name

        if model_str not in cls._instances:
            with cls._lock:  # to avoid loading the same model simultaneously from multiple threads :) Cool Man
                if model_str not in cls._instances:
                    cls._ensure_model_installed(model_str)
                    cls._instances[model_str] = spacy.load(model_str)
        return cls._instances[model_str]

    @classmethod
    def clear(cls):
        """Clear all loaded spaCy model instances (for test teardown)."""
        with cls._lock:
            cls._instances.clear()

    @staticmethod
    def _ensure_model_installed(model_name: str):
        """
        Ensure a spaCy model is installed and downloadable if missing.
        """
        try:
            importlib.import_module(model_name)
        except ImportError:
            print(f"[SpacySingleton] Model '{model_name}' not found. Auto-downloading...")
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=True
            )
