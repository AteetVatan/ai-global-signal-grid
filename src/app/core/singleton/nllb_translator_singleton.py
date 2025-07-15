# file: nllb_translator.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from threading import Lock
from typing import Dict


class NLLBTranslatorSingleton:
    """
    Singleton class for loading and serving the NLLB-200 multilingual translation model.
    Supports efficient reuse across threads, agents, or pipelines.
    """

    _instance = None
    _lock: Lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(NLLBTranslatorSingleton, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._initialized = True

        # Cache translation pipelines to avoid re-init
        self.pipelines: Dict[str, pipeline] = {}

    def _get_pipeline(self, src_lang: str, tgt_lang: str):
        """
        Returns a cached pipeline for the given src→tgt translation.
        """
        key = f"{src_lang}->{tgt_lang}"
        if key not in self.pipelines:
            self.pipelines[key] = pipeline(
                "translation",
                model=self.model,
                tokenizer=self.tokenizer,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                max_length=512,
            )
        return self.pipelines[key]

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate text from source to target language.
        """
        pipe = self._get_pipeline(src_lang, tgt_lang)
        result = pipe(text)
        return result[0]["translation_text"]
