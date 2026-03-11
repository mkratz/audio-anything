"""ElevenLabs cloud TTS backend."""

import io
import logging
import os

import numpy as np
import soundfile as sf

from ..config import Config
from .base import TTSBackend

log = logging.getLogger(__name__)


class ElevenLabsBackend(TTSBackend):
    def __init__(self, config: Config):
        super().__init__(config)
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY environment variable is required")

        from elevenlabs import ElevenLabs
        self._client = ElevenLabs(api_key=api_key)
        self._voice = config.voice
        log.info("Initialized ElevenLabs TTS (voice=%s)", self._voice)

    def synthesize(self, text: str) -> np.ndarray:
        try:
            audio_iter = self._client.text_to_speech.convert(
                voice_id=self._voice,
                text=text,
                model_id="eleven_monolingual_v1",
                output_format="pcm_24000",
            )
            pcm_bytes = b"".join(audio_iter)
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return samples
        except Exception:
            log.warning("ElevenLabs synthesis failed, returning silence", exc_info=True)
            return np.array([], dtype=np.float32)
