from unittest.mock import patch
from pathlib import Path

from audio_anything.clean import clean_transcript, clean_and_yield
from audio_anything.config import Config
from audio_anything.extract import PageChunk


def _make_config(**overrides):
    defaults = dict(
        pdf_path=Path("/tmp/fake.pdf"),
        preprocessing=False,
        ollama_parallel=2,
    )
    defaults.update(overrides)
    return Config(**defaults)


def test_concurrent_cleaning_calls_all_chunks():
    """Verify all chunks get cleaned when using concurrent Ollama calls."""
    pages = [
        PageChunk(page_number=i, text=f"Page {i} content. " * 50)
        for i in range(1, 5)
    ]
    config = _make_config(max_chunk_chars=500)

    call_count = 0

    def mock_clean_chunk(text, cfg):
        nonlocal call_count
        call_count += 1
        return text  # pass-through

    with patch("audio_anything.clean._clean_chunk", side_effect=mock_clean_chunk):
        result = clean_transcript(pages, config)

    assert call_count > 1  # multiple chunks were cleaned
    assert "Page 1 content" in result
    assert "Page 4 content" in result


def test_sequential_when_parallel_is_1():
    """With ollama_parallel=1, chunks are still cleaned correctly."""
    pages = [
        PageChunk(page_number=1, text="Hello world. " * 50),
        PageChunk(page_number=2, text="Goodbye world. " * 50),
    ]
    config = _make_config(max_chunk_chars=300, ollama_parallel=1)

    with patch("audio_anything.clean._clean_chunk", side_effect=lambda t, c: t):
        result = clean_transcript(pages, config)

    assert "Hello world" in result
    assert "Goodbye world" in result


def test_clean_and_yield_produces_all_chunks():
    """Streaming generator yields one cleaned chunk per input chunk group."""
    pages = [
        PageChunk(page_number=i, text=f"Page {i} content. " * 30)
        for i in range(1, 5)
    ]
    config = _make_config(max_chunk_chars=300, ollama_parallel=1)

    with patch("audio_anything.clean._clean_chunk", side_effect=lambda t, c: t):
        chunks = list(clean_and_yield(pages, config))

    assert len(chunks) >= 2  # multiple chunks yielded
    combined = "\n\n".join(chunks)
    assert "Page 1 content" in combined
    assert "Page 4 content" in combined
