"""
Tests for digest state persistence.
"""

import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.object_detection.consumers.digest_state import (
    DigestPeriodState,
    DigestStateManager,
    STALE_THRESHOLD,
)


class TestDigestPeriodState(unittest.TestCase):
    """Test DigestPeriodState dataclass."""

    def test_new_period(self):
        """Test creating a new period."""
        state = DigestPeriodState.new_period("daily_traffic", 1440)

        self.assertEqual(state.digest_id, "daily_traffic")
        self.assertEqual(state.event_count, 0)
        self.assertFalse(state.digest_sent)
        self.assertEqual(state.period_end, state.period_start + timedelta(minutes=1440))

    def test_add_event(self):
        """Test adding events."""
        state = DigestPeriodState.new_period("test", 60)

        state.add_event({
            'object_class_name': 'car',
            'line_description': 'driveway'
        })
        state.add_event({
            'object_class_name': 'car',
            'zone_description': 'parking'
        })
        state.add_event({
            'object_class_name': 'truck',
            'line_description': 'driveway'
        })

        self.assertEqual(state.event_count, 3)
        self.assertEqual(state.by_class, {'car': 2, 'truck': 1})
        self.assertEqual(state.by_line, {'driveway': 2})
        self.assertEqual(state.by_zone, {'parking': 1})

    def test_serialization(self):
        """Test to_dict and from_dict."""
        state = DigestPeriodState.new_period("test", 60)
        state.add_event({'object_class_name': 'cat'})
        state.add_frame_ref("/tmp/frame.jpg")

        # Round-trip
        data = state.to_dict()
        recovered = DigestPeriodState.from_dict(data)

        self.assertEqual(recovered.digest_id, state.digest_id)
        self.assertEqual(recovered.event_count, state.event_count)
        self.assertEqual(recovered.by_class, state.by_class)
        self.assertEqual(recovered.frame_refs, state.frame_refs)


class TestDigestStateManager(unittest.TestCase):
    """Test DigestStateManager."""

    def setUp(self):
        """Create temp directory for state."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_dir = Path(self.temp_dir) / "state"
        self.frames_dir = Path(self.temp_dir) / "frames"
        self.frames_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_init_creates_directories(self):
        """Test that init creates state directory."""
        _manager = DigestStateManager(
            state_dir=str(self.state_dir),
            frames_dir=str(self.frames_dir)
        )
        self.assertTrue(self.state_dir.exists())

    def test_initialize_new_digest(self):
        """Test initializing a new digest."""
        manager = DigestStateManager(
            state_dir=str(self.state_dir),
            frames_dir=str(self.frames_dir)
        )

        state = manager.initialize_digest(
            "daily_traffic",
            period_minutes=1440,
            config_digest_ids={"daily_traffic"}
        )

        self.assertIsNotNone(state)
        self.assertEqual(state.digest_id, "daily_traffic")
        self.assertEqual(state.event_count, 0)

    def test_add_event_and_checkpoint(self):
        """Test adding events triggers checkpoint."""
        manager = DigestStateManager(
            state_dir=str(self.state_dir),
            frames_dir=str(self.frames_dir)
        )

        manager.initialize_digest("test", 60, {"test"})

        # Add events
        for i in range(15):
            manager.add_event("test", {'object_class_name': 'car'})

        # State should be persisted
        state_file = self.state_dir / "digests.json"
        self.assertTrue(state_file.exists())

        with open(state_file) as f:
            data = json.load(f)

        self.assertEqual(data['digests']['test']['event_count'], 15)

    def test_recovery_after_restart(self):
        """Test state recovery after simulated restart."""
        # First run - accumulate events
        manager1 = DigestStateManager(
            state_dir=str(self.state_dir),
            frames_dir=str(self.frames_dir)
        )
        manager1.initialize_digest("test", 1440, {"test"})
        for i in range(5):
            manager1.add_event("test", {'object_class_name': 'car'})
        manager1._save_state()  # Force checkpoint

        # "Restart" - new manager instance
        manager2 = DigestStateManager(
            state_dir=str(self.state_dir),
            frames_dir=str(self.frames_dir)
        )

        state = manager2.get_state("test")
        self.assertIsNotNone(state)
        self.assertEqual(state.event_count, 5)

    def test_mark_sent_deletes_state(self):
        """Test that mark_sent removes state."""
        manager = DigestStateManager(
            state_dir=str(self.state_dir),
            frames_dir=str(self.frames_dir)
        )

        manager.initialize_digest("test", 60, {"test"})
        manager.add_event("test", {'object_class_name': 'car'})

        self.assertIsNotNone(manager.get_state("test"))

        manager.mark_sent("test")

        self.assertIsNone(manager.get_state("test"))

    def test_gc_removes_stale_state(self):
        """Test GC removes state older than 2 weeks."""
        manager = DigestStateManager(
            state_dir=str(self.state_dir),
            frames_dir=str(self.frames_dir)
        )

        # Manually create stale state
        stale_state = DigestPeriodState(
            digest_id="stale_digest",
            period_start=datetime.now(timezone.utc) - timedelta(weeks=3),
            period_end=datetime.now(timezone.utc) - timedelta(weeks=3) + timedelta(days=1),
        )
        manager.states["stale_digest"] = stale_state
        manager._save_state()

        # New manager runs GC on init
        manager2 = DigestStateManager(
            state_dir=str(self.state_dir),
            frames_dir=str(self.frames_dir)
        )

        self.assertIsNone(manager2.get_state("stale_digest"))

    def test_gc_removes_sent_state(self):
        """Test GC removes state with digest_sent=true."""
        manager = DigestStateManager(
            state_dir=str(self.state_dir),
            frames_dir=str(self.frames_dir)
        )

        # Create state and mark as sent but don't delete (simulating crash)
        state = DigestPeriodState.new_period("test", 60)
        state.digest_sent = True
        manager.states["test"] = state
        manager._save_state()

        # New manager runs GC on init
        manager2 = DigestStateManager(
            state_dir=str(self.state_dir),
            frames_dir=str(self.frames_dir)
        )

        self.assertIsNone(manager2.get_state("test"))

    def test_cleanup_orphaned_frames(self):
        """Test orphaned frame cleanup."""
        import time

        manager = DigestStateManager(
            state_dir=str(self.state_dir),
            frames_dir=str(self.frames_dir)
        )

        # Create some frames
        referenced_frame = self.frames_dir / "referenced.jpg"
        orphan_frame = self.frames_dir / "orphan.jpg"
        referenced_frame.write_text("fake image")
        orphan_frame.write_text("fake image")

        # Make orphan frame old
        old_time = time.time() - 86400 * 2  # 2 days old
        import os
        os.utime(orphan_frame, (old_time, old_time))

        # Create state referencing one frame
        manager.initialize_digest("test", 60, {"test"})
        manager.add_event("test", {}, frame_path=str(referenced_frame))

        # Clean up
        removed = manager.cleanup_orphaned_frames(max_age_hours=24)

        self.assertEqual(removed, 1)
        self.assertTrue(referenced_frame.exists())
        self.assertFalse(orphan_frame.exists())


if __name__ == '__main__':
    unittest.main()
