from audio_anything.estimate import estimate_stats

def test_estimate_basic():
    transcript = "Hello world. " * 100  # 1300 chars
    stats = estimate_stats(transcript, segment_max_chars=500)
    assert stats["char_count"] == len(transcript.strip())
    assert stats["segment_count"] >= 2
    assert stats["chapter_count"] == 0
    assert stats["estimated_duration_min"] > 0

def test_estimate_with_chapters():
    transcript = "Chapter: Introduction\n\nSome text here.\n\nChapter: Conclusion\n\nMore text."
    stats = estimate_stats(transcript, segment_max_chars=500)
    assert stats["chapter_count"] == 2

def test_estimate_empty():
    stats = estimate_stats("", segment_max_chars=500)
    assert stats["segment_count"] == 0
    assert stats["estimated_duration_min"] == 0.0
