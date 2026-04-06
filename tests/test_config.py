from pathlib import Path
from audio_anything.config import Config


def test_default_segment_max_chars():
    cfg = Config(pdf_path=Path("/tmp/fake.pdf"))
    assert cfg.segment_max_chars == 1500


def test_default_ollama_parallel():
    cfg = Config(pdf_path=Path("/tmp/fake.pdf"))
    assert cfg.ollama_parallel == 2


def test_ollama_parallel_override():
    cfg = Config(pdf_path=Path("/tmp/fake.pdf"), ollama_parallel=4)
    assert cfg.ollama_parallel == 4
