"""Kokoro local TTS backend."""

import logging

import numpy as np
from kokoro import KPipeline

from ..config import Config
from .base import TTSBackend

log = logging.getLogger(__name__)


class KokoroBackend(TTSBackend):
    def __init__(self, config: Config):
        super().__init__(config)
        log.info("Initializing Kokoro TTS (voice=%s)", config.voice)
        self._pipeline = KPipeline(lang_code="a")
        self._voice = config.voice

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text via Kokoro's generator (handles internal chunking)."""
        segments: list[np.ndarray] = []
        for _, _, audio in self._pipeline(text, voice=self._voice):
            if audio is not None:
                segments.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))

        if not segments:
            log.warning("Kokoro produced no audio for text: %.50s...", text)
            return np.array([], dtype=np.float32)

        return np.concatenate(segments)
