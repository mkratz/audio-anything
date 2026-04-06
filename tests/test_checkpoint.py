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


def test_batch_writes_reduces_manifest_updates(ckpt_dir):
    """With batch_interval=5, manifest is only written every 5th segment."""
    mgr = CheckpointManager(ckpt_dir, transcript_hash="abc123", total_segments=10, batch_interval=5)
    manifest_path = ckpt_dir / "checkpoint.json"

    for i in range(4):
        mgr.save_segment(i, np.zeros(10, dtype=np.float32))

    # After 4 segments, in-memory state should be correct
    assert mgr.completed_count == 4
    assert mgr.is_segment_done(3)

    # Read manifest from disk — should lag behind in-memory state
    disk_manifest = json.loads(manifest_path.read_text())
    assert disk_manifest["completed"] == 0  # not yet flushed

    # 5th segment triggers a write
    mgr.save_segment(4, np.zeros(10, dtype=np.float32))
    disk_manifest = json.loads(manifest_path.read_text())
    assert disk_manifest["completed"] == 5


def test_flush_writes_final_manifest(ckpt_dir):
    """flush() writes manifest regardless of batch interval."""
    mgr = CheckpointManager(ckpt_dir, transcript_hash="abc123", total_segments=10, batch_interval=100)
    mgr.save_segment(0, np.zeros(10, dtype=np.float32))
    mgr.save_segment(1, np.zeros(10, dtype=np.float32))
    mgr.flush()

    disk_manifest = json.loads((ckpt_dir / "checkpoint.json").read_text())
    assert disk_manifest["completed"] == 2


def test_default_batch_interval_writes_every_segment(ckpt_dir):
    """Default batch_interval=1 preserves existing behavior."""
    mgr = CheckpointManager(ckpt_dir, transcript_hash="abc123", total_segments=3)
    mgr.save_segment(0, np.zeros(10, dtype=np.float32))

    disk_manifest = json.loads((ckpt_dir / "checkpoint.json").read_text())
    assert disk_manifest["completed"] == 1
