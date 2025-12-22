"""
Frame Service
Handles local frame storage and metadata management.
"""

import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FrameService:
    """Manages local frame storage and retrieval."""

    def __init__(self, config: Dict):
        """Initialize frame service.

        Args:
            config: Configuration with storage settings
        """
        self.local_dir = config.get('storage', {}).get('local_dir', 'frames')
        self.metadata_file = os.path.join(self.local_dir, 'metadata.json')

        # Ensure local directory exists
        os.makedirs(self.local_dir, exist_ok=True)

        # Load or initialize metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load frame metadata from disk."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                logger.warning("Failed to load metadata, starting fresh")
        return {}

    def _save_metadata(self):
        """Save frame metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def save_event_frame(self, event: Dict, temp_frame_path: str) -> Optional[str]:
        """
        Save frame permanently with event metadata.

        Args:
            event: Enriched event dictionary
            temp_frame_path: Path to temporary frame file

        Returns:
            Local path if successful, None otherwise
        """
        if not os.path.exists(temp_frame_path):
            logger.warning(f"Temp frame not found: {temp_frame_path}")
            return None

        # Generate permanent filename
        event_id = f"{event['timestamp']}_{event['track_id']}"
        event_type = event['event_type']
        obj_class = event.get('object_class_name', 'unknown')

        # Clean filename
        safe_timestamp = event['timestamp'].replace(':', '-').replace('.', '-')
        filename = f"{safe_timestamp}_{event_type}_{obj_class}_{event['track_id']}.jpg"

        try:
            local_path = os.path.join(self.local_dir, filename)

            # Copy temp frame to permanent location
            shutil.copy2(temp_frame_path, local_path)

            # Store metadata
            self.metadata[event_id] = {
                'event': event,
                'local_path': local_path,
                'timestamp': event['timestamp']
            }
            self._save_metadata()

            logger.info(f"Saved frame: {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
            return None

    def get_frame_path(self, event_id: str) -> Optional[str]:
        """Get local path for a frame by event ID."""
        frame_data = self.metadata.get(event_id)
        if frame_data:
            return frame_data.get('local_path')
        return None

    def get_frame_paths_for_events(self, events: List[Dict]) -> Dict[str, str]:
        """
        Get frame paths for a list of events.

        Args:
            events: List of enriched event dictionaries

        Returns:
            Dictionary mapping event_id to local file path
        """
        result = {}

        for event in events:
            event_id = f"{event['timestamp']}_{event['track_id']}"

            if event_id not in self.metadata:
                continue

            frame_data = self.metadata[event_id]
            local_path = frame_data.get('local_path')

            # Verify file exists
            if local_path and os.path.exists(local_path):
                result[event_id] = local_path

        return result

    def read_frame_bytes(self, event_id: str) -> Optional[bytes]:
        """
        Read frame file as bytes (for email embedding).

        Args:
            event_id: Event identifier

        Returns:
            Frame bytes if found, None otherwise
        """
        local_path = self.get_frame_path(event_id)
        if local_path and os.path.exists(local_path):
            try:
                with open(local_path, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to read frame {local_path}: {e}")
        return None

    def cleanup_old_frames(self, days: int = 7) -> int:
        """
        Remove frames and metadata older than specified days.

        Args:
            days: Number of days to retain

        Returns:
            Number of frames removed
        """
        cutoff = datetime.now() - timedelta(days=days)
        removed = 0

        to_remove = []
        for event_id, data in self.metadata.items():
            try:
                timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                timestamp = timestamp.replace(tzinfo=None)

                if timestamp < cutoff:
                    # Delete the file
                    local_path = data.get('local_path')
                    if local_path and os.path.exists(local_path):
                        os.unlink(local_path)
                        removed += 1

                    to_remove.append(event_id)
            except Exception:
                continue

        for event_id in to_remove:
            del self.metadata[event_id]

        if to_remove:
            self._save_metadata()
            logger.info(f"Cleaned up {removed} old frames")

        return removed

    def get_frame_info(self, event_id: str) -> Optional[Dict]:
        """Get frame metadata for an event ID."""
        return self.metadata.get(event_id)
