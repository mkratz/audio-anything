import numpy as np
import soundfile as sf
from pathlib import Path

from audio_anything.audio import StreamingAudioWriter


def test_streaming_writer_creates_valid_wav(tmp_path):
    wav_path = tmp_path / "test.wav"
    writer = StreamingAudioWriter(wav_path, sample_rate=24000)

    chunk1 = np.ones(1000, dtype=np.float32) * 0.5
    chunk2 = np.ones(500, dtype=np.float32) * -0.3

    writer.write(chunk1)
    writer.write(chunk2)
    writer.close()

    # Read back and verify
    data, sr = sf.read(str(wav_path))
    assert sr == 24000
    assert len(data) == 1500
    np.testing.assert_allclose(data[:1000], 0.5, atol=1e-6)
    np.testing.assert_allclose(data[1000:], -0.3, atol=1e-6)


def test_streaming_writer_total_samples(tmp_path):
    wav_path = tmp_path / "test.wav"
    writer = StreamingAudioWriter(wav_path, sample_rate=24000)

    writer.write(np.zeros(100, dtype=np.float32))
    writer.write(np.zeros(200, dtype=np.float32))

    assert writer.total_samples == 300
    writer.close()


def test_streaming_writer_context_manager(tmp_path):
    wav_path = tmp_path / "test.wav"
    with StreamingAudioWriter(wav_path, sample_rate=24000) as writer:
        writer.write(np.ones(100, dtype=np.float32))

    data, sr = sf.read(str(wav_path))
    assert len(data) == 100
