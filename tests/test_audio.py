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
