"""
Digest State Persistence

Manages checkpointing and recovery of digest accumulation state.
Ensures digests survive system restarts without losing accumulated events.

Lifecycle:
1. Period starts → Create state entry
2. Events accumulate → Update state, checkpoint periodically
3. Period ends → Send digest
4. Send succeeds → Delete state entry

GC runs on startup and periodically to clean:
- Orphaned states (digest removed from config)
- Stale states (period ended > 2 weeks ago)
- Sent states (digest_sent=true but not cleaned up)
- Dangling frame references
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

logger = logging.getLogger(__name__)

# How long to keep stale state before GC deletes it
STALE_THRESHOLD = timedelta(weeks=2)

# How often to checkpoint (events or seconds)
CHECKPOINT_INTERVAL_EVENTS = 10
CHECKPOINT_INTERVAL_SECONDS = 60

# How often to run GC
GC_INTERVAL_SECONDS = 3600  # 1 hour


@dataclass
class DigestPeriodState:
    """State for a single digest's current period."""
    digest_id: str
    period_start: datetime
    period_end: datetime
    digest_sent: bool = False
    event_count: int = 0
    by_class: Dict[str, int] = field(default_factory=dict)
    by_zone: Dict[str, int] = field(default_factory=dict)
    by_line: Dict[str, int] = field(default_factory=dict)
    frame_refs: List[str] = field(default_factory=list)
    last_checkpoint: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_event(self, event: Dict[str, Any]) -> None:
        """Add an event to the accumulation."""
        self.event_count += 1

        # Track by class
        class_name = event.get('object_class_name', 'unknown')
        self.by_class[class_name] = self.by_class.get(class_name, 0) + 1

        # Track by zone
        zone = event.get('zone_description')
        if zone:
            self.by_zone[zone] = self.by_zone.get(zone, 0) + 1

        # Track by line
        line = event.get('line_description')
        if line:
            self.by_line[line] = self.by_line.get(line, 0) + 1

    def add_frame_ref(self, frame_path: str) -> None:
        """Add a captured frame reference."""
        self.frame_refs.append(frame_path)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            'digest_id': self.digest_id,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'digest_sent': self.digest_sent,
            'event_count': self.event_count,
            'by_class': self.by_class,
            'by_zone': self.by_zone,
            'by_line': self.by_line,
            'frame_refs': self.frame_refs,
            'last_checkpoint': self.last_checkpoint.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DigestPeriodState':
        """Deserialize from dictionary."""
        return cls(
            digest_id=data['digest_id'],
            period_start=datetime.fromisoformat(data['period_start']),
            period_end=datetime.fromisoformat(data['period_end']),
            digest_sent=data.get('digest_sent', False),
            event_count=data.get('event_count', 0),
            by_class=data.get('by_class', {}),
            by_zone=data.get('by_zone', {}),
            by_line=data.get('by_line', {}),
            frame_refs=data.get('frame_refs', []),
            last_checkpoint=datetime.fromisoformat(data['last_checkpoint']),
        )

    @classmethod
    def new_period(cls, digest_id: str, period_minutes: int) -> 'DigestPeriodState':
        """Create state for a new period."""
        now = datetime.now(timezone.utc)
        return cls(
            digest_id=digest_id,
            period_start=now,
            period_end=now + timedelta(minutes=period_minutes),
        )


class DigestStateManager:
    """
    Manages digest state persistence and recovery.

    Thread-safe for single-writer, multiple-reader access.
    """

    def __init__(self, state_dir: str = "data/state", frames_dir: str = "frames"):
        self.state_dir = Path(state_dir)
        self.frames_dir = Path(frames_dir)
        self.state_file = self.state_dir / "digests.json"

        # In-memory state
        self.states: Dict[str, DigestPeriodState] = {}

        # Checkpoint tracking
        self.events_since_checkpoint = 0
        self.last_checkpoint_time = time.time()
        self.last_gc_time = time.time()

        # Ensure directories exist
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Run GC and load state on init
        self._garbage_collect()
        self._load_state()

    def _load_state(self) -> None:
        """Load state from disk."""
        if not self.state_file.exists():
            logger.info("No prior digest state found, starting fresh")
            return

        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)

            for digest_id, state_data in data.get('digests', {}).items():
                self.states[digest_id] = DigestPeriodState.from_dict(state_data)
                logger.info(
                    f"Loaded state for '{digest_id}': "
                    f"{self.states[digest_id].event_count} events accumulated"
                )

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load state file: {e}")
            # Corrupted state - back it up and start fresh
            backup = self.state_file.with_suffix('.json.corrupted')
            self.state_file.rename(backup)
            logger.warning(f"Backed up corrupted state to {backup}")

    def _save_state(self) -> None:
        """Atomically save state to disk."""
        data = {
            'version': 1,
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'digests': {
                digest_id: state.to_dict()
                for digest_id, state in self.states.items()
            }
        }

        # Atomic write: write to temp, then rename
        temp_file = self.state_file.with_suffix('.json.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)

        temp_file.rename(self.state_file)
        self.last_checkpoint_time = time.time()
        self.events_since_checkpoint = 0

    def _garbage_collect(self) -> None:
        """Clean up stale state and temp files."""
        now = datetime.now(timezone.utc)
        logger.info("Running digest state garbage collection")

        # 1. Clean up temp files from interrupted checkpoints
        for tmp_file in self.state_dir.glob("*.tmp"):
            logger.info(f"GC: Removing incomplete checkpoint {tmp_file.name}")
            tmp_file.unlink()

        # 2. Clean up corrupted backups older than 7 days
        for backup in self.state_dir.glob("*.corrupted"):
            if backup.stat().st_mtime < time.time() - 604800:
                logger.info(f"GC: Removing old corrupted backup {backup.name}")
                backup.unlink()

        # 3. Load and clean state entries
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return

        to_delete = []
        digests = data.get('digests', {})

        for digest_id, state_data in digests.items():
            try:
                period_end = datetime.fromisoformat(state_data['period_end'])
                digest_sent = state_data.get('digest_sent', False)

                # Already sent: cleanup didn't complete
                if digest_sent:
                    logger.info(f"GC: Removing already-sent state for '{digest_id}'")
                    to_delete.append(digest_id)
                    continue

                # Stale: period ended more than 2 weeks ago
                if period_end + STALE_THRESHOLD < now:
                    logger.warning(
                        f"GC: Removing stale state for '{digest_id}' "
                        f"(ended {period_end.date()}, > 2 weeks ago)"
                    )
                    to_delete.append(digest_id)
                    continue

                # Validate frame refs - remove dangling
                valid_frames = []
                for frame_ref in state_data.get('frame_refs', []):
                    frame_path = Path(frame_ref)
                    if frame_path.exists():
                        valid_frames.append(frame_ref)
                    else:
                        logger.debug(f"GC: Removing dangling frame ref: {frame_ref}")
                state_data['frame_refs'] = valid_frames

            except (KeyError, ValueError) as e:
                logger.warning(f"GC: Removing malformed state for '{digest_id}': {e}")
                to_delete.append(digest_id)

        # Apply deletions
        for digest_id in to_delete:
            del digests[digest_id]

        if to_delete:
            # Save cleaned state
            data['digests'] = digests
            data['last_updated'] = now.isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"GC: Cleaned {len(to_delete)} stale/invalid entries")

        self.last_gc_time = time.time()

    def initialize_digest(self, digest_id: str, period_minutes: int,
                          config_digest_ids: Set[str]) -> Optional[DigestPeriodState]:
        """
        Initialize or recover state for a digest.

        Returns the state to use (recovered or new).
        """
        now = datetime.now(timezone.utc)

        # Check if we have existing state
        if digest_id in self.states:
            state = self.states[digest_id]

            # Is the digest still in config?
            if digest_id not in config_digest_ids:
                logger.info(f"Digest '{digest_id}' removed from config, discarding state")
                del self.states[digest_id]
                self._save_state()
                return None

            # Is the period still active?
            if state.period_end > now:
                logger.info(
                    f"Resuming digest '{digest_id}': "
                    f"{state.event_count} events, ends {state.period_end}"
                )
                return state

            # Period ended - this will be handled by caller (send delayed digest)
            logger.info(
                f"Digest '{digest_id}' period ended while down, "
                f"returning state for delayed send"
            )
            return state

        # No existing state - create new
        if digest_id in config_digest_ids:
            state = DigestPeriodState.new_period(digest_id, period_minutes)
            self.states[digest_id] = state
            self._save_state()
            logger.info(f"Created new period for '{digest_id}', ends {state.period_end}")
            return state

        return None

    def add_event(self, digest_id: str, event: Dict[str, Any],
                  frame_path: Optional[str] = None) -> None:
        """Add an event to a digest's accumulation."""
        if digest_id not in self.states:
            logger.warning(f"No state for digest '{digest_id}', event ignored")
            return

        state = self.states[digest_id]
        state.add_event(event)

        if frame_path:
            state.add_frame_ref(frame_path)

        self.events_since_checkpoint += 1
        self._maybe_checkpoint()

    def _maybe_checkpoint(self) -> None:
        """Checkpoint if thresholds exceeded."""
        should_checkpoint = (
            self.events_since_checkpoint >= CHECKPOINT_INTERVAL_EVENTS or
            time.time() - self.last_checkpoint_time > CHECKPOINT_INTERVAL_SECONDS
        )

        if should_checkpoint:
            self._save_state()
            logger.debug(f"Checkpointed digest state ({self.events_since_checkpoint} events)")

    def maybe_gc(self) -> None:
        """Run GC if interval exceeded."""
        if time.time() - self.last_gc_time > GC_INTERVAL_SECONDS:
            self._garbage_collect()

    def mark_sent(self, digest_id: str) -> None:
        """Mark a digest as sent and delete its state."""
        if digest_id in self.states:
            logger.info(f"Digest '{digest_id}' sent, deleting state")
            del self.states[digest_id]
            self._save_state()

    def get_state(self, digest_id: str) -> Optional[DigestPeriodState]:
        """Get current state for a digest."""
        return self.states.get(digest_id)

    def get_pending_sends(self) -> List[DigestPeriodState]:
        """Get digests whose period has ended but haven't been sent."""
        now = datetime.now(timezone.utc)
        pending = []

        for state in self.states.values():
            if state.period_end <= now and not state.digest_sent:
                pending.append(state)

        return pending

    def start_new_period(self, digest_id: str, period_minutes: int) -> DigestPeriodState:
        """Start a new period for a digest (after successful send)."""
        state = DigestPeriodState.new_period(digest_id, period_minutes)
        self.states[digest_id] = state
        self._save_state()
        logger.info(f"Started new period for '{digest_id}', ends {state.period_end}")
        return state

    def cleanup_orphaned_frames(self, max_age_hours: int = 24) -> int:
        """Remove frames not referenced by any state and older than max_age."""
        if not self.frames_dir.exists():
            return 0

        # Collect all referenced frames
        referenced = set()
        for state in self.states.values():
            referenced.update(state.frame_refs)

        # Find and remove orphans
        cutoff = time.time() - (max_age_hours * 3600)
        removed = 0

        for frame_path in self.frames_dir.glob("*.jpg"):
            if str(frame_path) not in referenced:
                if frame_path.stat().st_mtime < cutoff:
                    logger.debug(f"Removing orphaned frame: {frame_path.name}")
                    frame_path.unlink()
                    removed += 1

        if removed:
            logger.info(f"Cleaned up {removed} orphaned frames")

        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about current state."""
        return {
            'active_digests': len(self.states),
            'total_events': sum(s.event_count for s in self.states.values()),
            'total_frames': sum(len(s.frame_refs) for s in self.states.values()),
            'digests': {
                digest_id: {
                    'event_count': state.event_count,
                    'period_end': state.period_end.isoformat(),
                    'frames': len(state.frame_refs),
                }
                for digest_id, state in self.states.items()
            }
        }
