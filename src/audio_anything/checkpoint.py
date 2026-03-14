"""Checkpoint/resume support for TTS synthesis.

Saves each audio segment to disk as it completes, enabling crash recovery.
On restart, if a matching checkpoint exists, synthesis resumes from the
last completed segment.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def transcript_hash(text: str) -> str:
    """Deterministic hash of transcript text for checkpoint validation."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class CheckpointManager:
    """Manages segment-level checkpointing for TTS synthesis."""

    def __init__(self, ckpt_dir: Path, transcript_hash: str, total_segments: int):
        self._dir = ckpt_dir
        self._manifest_path = ckpt_dir / "checkpoint.json"
        self._hash = transcript_hash
        self._total = total_segments

        if self._manifest_path.exists():
            manifest = json.loads(self._manifest_path.read_text())
            if manifest.get("transcript_hash") == self._hash:
                log.info(
                    "Resuming from checkpoint: %d/%d segments complete",
                    manifest["completed"], self._total,
                )
                self._completed: set[int] = set(manifest.get("completed_indices", []))
                return
            else:
                log.warning("Transcript changed since last run — discarding checkpoint")
                self.cleanup()

        self._completed = set()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    @property
    def is_complete(self) -> bool:
        return self.completed_count >= self._total

    def is_segment_done(self, index: int) -> bool:
        return index in self._completed

    def save_segment(self, index: int, audio: np.ndarray) -> None:
        """Save a single segment's audio to disk."""
        np.save(str(self._seg_path(index)), audio)
        self._completed.add(index)
        self._write_manifest()

    def load_segment(self, index: int) -> np.ndarray:
        """Load a previously saved segment."""
        return np.load(str(self._seg_path(index)))

    def collect_all(self) -> np.ndarray:
        """Load and concatenate all segments in order."""
        arrays = [self.load_segment(i) for i in range(self._total)]
        return np.concatenate(arrays)

    def cleanup(self) -> None:
        """Remove checkpoint directory entirely."""
        if self._dir.exists():
            shutil.rmtree(self._dir)

    def _seg_path(self, index: int) -> Path:
        return self._dir / f"seg_{index:04d}.npy"

    def _write_manifest(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "transcript_hash": self._hash,
            "total_segments": self._total,
            "completed": self.completed_count,
            "completed_indices": sorted(self._completed),
        }
        self._manifest_path.write_text(json.dumps(manifest, indent=2))
