import json
import numpy as np
import pytest
from pathlib import Path

from audio_anything.checkpoint import CheckpointManager


@pytest.fixture
def ckpt_dir(tmp_path):
    return tmp_path / "checkpoint"


def test_new_checkpoint_has_no_progress(ckpt_dir):
    mgr = CheckpointManager(ckpt_dir, transcript_hash="abc123", total_segments=10)
    assert mgr.completed_count == 0
    assert not mgr.is_complete


def test_save_and_load_segment(ckpt_dir):
    mgr = CheckpointManager(ckpt_dir, transcript_hash="abc123", total_segments=3)
    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mgr.save_segment(0, audio)
    assert mgr.completed_count == 1

    loaded = mgr.load_segment(0)
    np.testing.assert_array_equal(loaded, audio)


def test_resume_from_existing_checkpoint(ckpt_dir):
    mgr1 = CheckpointManager(ckpt_dir, transcript_hash="abc123", total_segments=5)
    mgr1.save_segment(0, np.zeros(100, dtype=np.float32))
    mgr1.save_segment(1, np.ones(100, dtype=np.float32))

    mgr2 = CheckpointManager(ckpt_dir, transcript_hash="abc123", total_segments=5)
    assert mgr2.completed_count == 2


def test_hash_mismatch_resets_checkpoint(ckpt_dir):
    mgr1 = CheckpointManager(ckpt_dir, transcript_hash="abc123", total_segments=5)
    mgr1.save_segment(0, np.zeros(100, dtype=np.float32))

    mgr2 = CheckpointManager(ckpt_dir, transcript_hash="different", total_segments=5)
    assert mgr2.completed_count == 0


def test_collect_all_segments(ckpt_dir):
    mgr = CheckpointManager(ckpt_dir, transcript_hash="abc123", total_segments=2)
    mgr.save_segment(0, np.array([1.0, 2.0], dtype=np.float32))
    mgr.save_segment(1, np.array([3.0, 4.0], dtype=np.float32))
    assert mgr.is_complete

    all_audio = mgr.collect_all()
    np.testing.assert_array_equal(all_audio, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


def test_cleanup_removes_directory(ckpt_dir):
    mgr = CheckpointManager(ckpt_dir, transcript_hash="abc123", total_segments=1)
    mgr.save_segment(0, np.zeros(10, dtype=np.float32))
    assert ckpt_dir.exists()

    mgr.cleanup()
    assert not ckpt_dir.exists()
