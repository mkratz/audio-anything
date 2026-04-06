from audio_anything.audio import _split_into_segments


def test_split_respects_max_chars():
    text = "Hello world. " * 100  # ~1300 chars
    segments = _split_into_segments(text, max_chars=1500)
    assert len(segments) == 1
    assert len(segments[0]) <= 1500


def test_split_breaks_at_sentence_boundary():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    segments = _split_into_segments(text, max_chars=40)
    for seg in segments:
        assert len(seg) <= 40 or " " not in seg  # may exceed if single sentence > max


def test_split_old_default_produces_more_segments():
    text = "Hello world. " * 100  # ~1300 chars
    old_segments = _split_into_segments(text, max_chars=500)
    new_segments = _split_into_segments(text, max_chars=1500)
    assert len(old_segments) > len(new_segments)


import numpy as np
from unittest.mock import MagicMock
from audio_anything.audio import synthesize_audio_streaming
from audio_anything.config import Config
from pathlib import Path


def test_synthesize_audio_streaming_basic():
    """Streaming synthesis consumes an iterable of text chunks."""
    chunks = iter(["Chapter: Intro\n\nHello world. This is a test.", "More content here. End of text."])
    config = Config(pdf_path=Path("/tmp/fake.pdf"), segment_max_chars=1500)

    backend = MagicMock()
    backend.synthesize.return_value = np.ones(100, dtype=np.float32)

    audio, chapters = synthesize_audio_streaming(chunks, backend, config)

    assert len(audio) > 0
    assert backend.synthesize.call_count >= 2
    assert len(chapters) >= 1  # "Intro" chapter detected
