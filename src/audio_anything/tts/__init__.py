"""TTS backend factory."""

from ..config import Config
from .base import TTSBackend


def get_tts_backend(config: Config) -> TTSBackend:
    """Return the appropriate TTS backend based on config."""
    match config.tts_backend:
        case "kokoro":
            from .kokoro import KokoroBackend
            return KokoroBackend(config)
        case "elevenlabs":
            from .elevenlabs import ElevenLabsBackend
            return ElevenLabsBackend(config)
        case "openai":
            from .openai_tts import OpenAIBackend
            return OpenAIBackend(config)
        case _:
            raise ValueError(f"Unknown TTS backend: {config.tts_backend}")
