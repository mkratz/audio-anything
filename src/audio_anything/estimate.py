"""Pre-run estimation: segment counts, chapter counts, projected duration."""

import re

from .audio import CHAPTER_CUE, _split_into_segments

# ~150 words/min ≈ ~900 chars/min for English narration
CHARS_PER_MINUTE = 900


def estimate_stats(transcript: str, segment_max_chars: int) -> dict:
    """Compute stats from a transcript without running TTS."""
    transcript = transcript.strip()
    if not transcript:
        return {
            "char_count": 0,
            "segment_count": 0,
            "chapter_count": 0,
            "estimated_duration_min": 0.0,
        }

    segments = _split_into_segments(transcript, segment_max_chars)
    chapters = [
        line for line in transcript.splitlines()
        if CHAPTER_CUE.match(line.strip())
    ]

    char_count = len(transcript)
    duration_min = char_count / CHARS_PER_MINUTE

    return {
        "char_count": char_count,
        "segment_count": len(segments),
        "chapter_count": len(chapters),
        "estimated_duration_min": round(duration_min, 1),
    }


def print_estimate(stats: dict, pdf_name: str) -> None:
    """Print a human-readable estimate summary."""
    dur = stats["estimated_duration_min"]
    hours = int(dur // 60)
    mins = int(dur % 60)
    dur_str = f"{hours}h {mins}m" if hours else f"{mins}m"

    print(f"\n{'=' * 50}")
    print(f"  Estimate: {pdf_name}")
    print(f"{'=' * 50}")
    print(f"  Characters:     {stats['char_count']:,}")
    print(f"  TTS segments:   {stats['segment_count']}")
    print(f"  Chapters:       {stats['chapter_count']}")
    print(f"  Est. duration:  ~{dur_str}")
    print(f"{'=' * 50}\n")
