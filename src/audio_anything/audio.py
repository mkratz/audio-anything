"""Sentence splitting, TTS synthesis loop, and MP3/M4B export."""

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import numpy as np
import soundfile as sf

from .checkpoint import CheckpointManager
from .config import Config
from .tts.base import TTSBackend

log = logging.getLogger(__name__)

STRUCTURAL_CUE = re.compile(r"^(Chapter|Section):\s", re.MULTILINE)
CHAPTER_CUE = re.compile(r"^Chapter:\s+(.+)")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChapterMarker:
    title: str
    start_sample: int


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


def synthesize_audio(
    transcript: str, backend: TTSBackend, config: Config,
    ckpt: CheckpointManager | None = None,
) -> tuple[np.ndarray, list[ChapterMarker]]:
    """Synthesize full transcript into a single audio array with chapter markers."""
    segments = _split_into_segments(transcript, config.segment_max_chars)
    log.info("Synthesizing %d text segments via %s", len(segments), config.tts_backend)

    silence = np.zeros(int(config.silence_duration * config.sample_rate), dtype=np.float32)
    chapters: list[ChapterMarker] = []
    segment_audio: list[np.ndarray | None] = [None] * len(segments)

    skipped = 0
    for i, segment in enumerate(tqdm(segments, desc="Synthesizing", unit="seg")):
        # Check checkpoint — skip if already done
        if ckpt and ckpt.is_segment_done(i):
            skipped += 1
            continue

        log.info("Synthesizing segment %d/%d (%d chars)", i + 1, len(segments), len(segment))
        try:
            samples = backend.synthesize(segment)
            if len(samples) > 0:
                if ckpt:
                    ckpt.save_segment(i, samples)
                segment_audio[i] = samples
            else:
                # Save empty segment to checkpoint so we don't retry
                if ckpt:
                    ckpt.save_segment(i, np.array([], dtype=np.float32))
        except Exception:
            log.warning("TTS failed for segment %d, skipping", i + 1, exc_info=True)
            if ckpt:
                ckpt.save_segment(i, np.array([], dtype=np.float32))

    if skipped:
        log.info("Resumed: skipped %d already-completed segments", skipped)

    # Collect all audio and compute chapter markers
    audio_parts: list[np.ndarray] = []
    total_samples = 0

    for i, segment in enumerate(segments):
        if STRUCTURAL_CUE.match(segment):
            m = CHAPTER_CUE.match(segment)
            if m:
                chapters.append(ChapterMarker(
                    title=m.group(1).strip(),
                    start_sample=total_samples,
                ))
            audio_parts.append(silence)
            total_samples += len(silence)

        # Load audio: from memory if just synthesized, from checkpoint if resumed
        if segment_audio[i] is not None:
            audio = segment_audio[i]
        elif ckpt and ckpt.is_segment_done(i):
            audio = ckpt.load_segment(i)
        else:
            audio = np.array([], dtype=np.float32)

        if len(audio) > 0:
            audio_parts.append(audio)
            total_samples += len(audio)

    if not audio_parts:
        log.error("No audio produced")
        return np.array([], dtype=np.float32), []

    return np.concatenate(audio_parts), chapters


def synthesize_audio_streaming(
    chunk_iter,
    backend: TTSBackend,
    config: Config,
    ckpt: CheckpointManager | None = None,
) -> tuple[np.ndarray, list[ChapterMarker]]:
    """Synthesize audio from a streaming iterable of cleaned text chunks.

    Each chunk is split into segments and synthesized incrementally.
    No checkpoint resume support in streaming mode (use sequential for resume).
    """
    silence = np.zeros(int(config.silence_duration * config.sample_rate), dtype=np.float32)
    chapters: list[ChapterMarker] = []
    audio_parts: list[np.ndarray] = []
    total_samples = 0
    seg_index = 0

    for chunk_text in chunk_iter:
        segments = _split_into_segments(chunk_text, config.segment_max_chars)

        for segment in segments:
            if STRUCTURAL_CUE.match(segment):
                m = CHAPTER_CUE.match(segment)
                if m:
                    chapters.append(ChapterMarker(
                        title=m.group(1).strip(),
                        start_sample=total_samples,
                    ))
                audio_parts.append(silence)
                total_samples += len(silence)

            log.info("Synthesizing segment %d (%d chars)", seg_index + 1, len(segment))
            try:
                samples = backend.synthesize(segment)
                if len(samples) > 0:
                    if ckpt:
                        ckpt.save_segment(seg_index, samples)
                    audio_parts.append(samples)
                    total_samples += len(samples)
            except Exception:
                log.warning("TTS failed for segment %d, skipping", seg_index + 1, exc_info=True)

            seg_index += 1

    if not audio_parts:
        log.error("No audio produced")
        return np.array([], dtype=np.float32), []

    return np.concatenate(audio_parts), chapters


def _write_wav(audio: np.ndarray, wav_path: Path, sample_rate: int) -> None:
    """Write audio samples to a WAV file."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing WAV to %s", wav_path)
    sf.write(str(wav_path), audio, sample_rate)


def export_mp3(audio: np.ndarray, output_path: Path, config: Config) -> Path:
    """Export audio array to MP3 via WAV intermediate."""
    wav_path = output_path.with_suffix(".wav")
    _write_wav(audio, wav_path, config.sample_rate)

    log.info("Converting to MP3 at %s bitrate", config.mp3_bitrate)
    mp3_path = output_path.with_suffix(".mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-b:a", config.mp3_bitrate, str(mp3_path)],
        capture_output=True, check=True,
    )

    wav_path.unlink()
    log.info("Exported %s", mp3_path)
    return mp3_path


def export_m4b(
    audio: np.ndarray, output_path: Path, config: Config,
    chapters: list[ChapterMarker],
) -> Path:
    """Export audio array to M4B with chapter markers via WAV + ffmpeg."""
    wav_path = output_path.with_suffix(".wav")
    _write_wav(audio, wav_path, config.sample_rate)

    total_ms = len(audio) * 1000 // config.sample_rate

    # Build FFMETADATA with chapter entries
    meta_lines = [";FFMETADATA1"]

    if chapters:
        # Add preamble chapter if first marker isn't at the start
        if chapters[0].start_sample > 0:
            chapters = [ChapterMarker(title="Preamble", start_sample=0)] + chapters

        for i, chapter in enumerate(chapters):
            start_ms = chapter.start_sample * 1000 // config.sample_rate
            end_ms = (
                chapters[i + 1].start_sample * 1000 // config.sample_rate
                if i + 1 < len(chapters)
                else total_ms
            )
            meta_lines.extend([
                "",
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start_ms}",
                f"END={end_ms}",
                f"title={chapter.title}",
            ])

    meta_path = output_path.with_suffix(".ffmeta")
    meta_path.write_text("\n".join(meta_lines))

    log.info("Converting to M4B with %d chapters at %s bitrate", len(chapters), config.mp3_bitrate)
    m4b_path = output_path.with_suffix(".m4b")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-i", str(meta_path),
            "-map_metadata", "1",
            "-c:a", "aac",
            "-b:a", config.mp3_bitrate,
            str(m4b_path),
        ],
        capture_output=True, check=True,
    )

    wav_path.unlink()
    meta_path.unlink()
    log.info("Exported %s", m4b_path)
    return m4b_path
