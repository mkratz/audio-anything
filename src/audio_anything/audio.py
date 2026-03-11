"""Sentence splitting, TTS synthesis loop, and MP3 export."""

import logging
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import Config
from .tts.base import TTSBackend

log = logging.getLogger(__name__)

STRUCTURAL_CUE = re.compile(r"^(Chapter|Section):\s", re.MULTILINE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_into_segments(transcript: str, max_chars: int) -> list[str]:
    """Split transcript into segments at sentence boundaries, respecting max_chars."""
    sentences = SENTENCE_END.split(transcript)
    segments: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            segments.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence

    if current:
        segments.append(current)

    return segments


def synthesize_audio(transcript: str, backend: TTSBackend, config: Config) -> np.ndarray:
    """Synthesize full transcript into a single audio array."""
    segments = _split_into_segments(transcript, config.segment_max_chars)
    log.info("Synthesizing %d text segments via %s", len(segments), config.tts_backend)

    silence = np.zeros(int(config.silence_duration * config.sample_rate), dtype=np.float32)
    audio_parts: list[np.ndarray] = []

    for i, segment in enumerate(segments, 1):
        if STRUCTURAL_CUE.match(segment):
            audio_parts.append(silence)

        log.info("Synthesizing segment %d/%d (%d chars)", i, len(segments), len(segment))
        try:
            samples = backend.synthesize(segment)
            if len(samples) > 0:
                audio_parts.append(samples)
        except Exception:
            log.warning("TTS failed for segment %d, skipping", i, exc_info=True)

    if not audio_parts:
        log.error("No audio produced")
        return np.array([], dtype=np.float32)

    return np.concatenate(audio_parts)


def export_mp3(audio: np.ndarray, output_path: Path, config: Config) -> Path:
    """Export audio array to MP3 via WAV intermediate."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path = output_path.with_suffix(".wav")

    log.info("Writing WAV to %s", wav_path)
    sf.write(str(wav_path), audio, config.sample_rate)

    log.info("Converting to MP3 at %s bitrate", config.mp3_bitrate)
    mp3_path = output_path.with_suffix(".mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-b:a", config.mp3_bitrate, str(mp3_path)],
        capture_output=True, check=True,
    )

    wav_path.unlink()
    log.info("Exported %s", mp3_path)
    return mp3_path
