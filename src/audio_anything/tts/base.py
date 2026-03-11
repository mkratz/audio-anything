"""Abstract TTS backend interface."""

from abc import ABC, abstractmethod

import numpy as np

from ..config import Config


class TTSBackend(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text into audio samples (float32, mono, at config.sample_rate)."""
        ...
