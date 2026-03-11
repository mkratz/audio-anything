"""OpenAI cloud TTS backend."""

import io
import logging
import os

import numpy as np
import soundfile as sf

from ..config import Config
from .base import TTSBackend

log = logging.getLogger(__name__)


class OpenAIBackend(TTSBackend):
    def __init__(self, config: Config):
        super().__init__(config)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required")

        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._voice = config.voice
        log.info("Initialized OpenAI TTS (voice=%s)", self._voice)

    def synthesize(self, text: str) -> np.ndarray:
        try:
            response = self._client.audio.speech.create(
                model="tts-1",
                voice=self._voice,
                input=text,
                response_format="pcm",
            )
            pcm_bytes = response.content
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return samples
        except Exception:
            log.warning("OpenAI TTS synthesis failed, returning silence", exc_info=True)
            return np.array([], dtype=np.float32)
