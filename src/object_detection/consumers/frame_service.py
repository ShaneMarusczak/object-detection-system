"""
Frame Service
Handles frame storage, S3 uploads, and metadata management.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FrameService:
    """Manages frame storage and retrieval with S3 support."""

    def __init__(self, config: Dict):
        """Initialize frame service.

        Args:
            config: Configuration with storage settings
        """
        self.storage_type = config.get('storage', {}).get('type', 'local')
        self.local_dir = config.get('storage', {}).get('local_dir', 'frames')
        self.metadata_file = os.path.join(self.local_dir, 'metadata.json')

        # S3 configuration
        if self.storage_type == 's3':
            s3_config = config.get('storage', {}).get('s3', {})
            self.s3_bucket = s3_config.get('bucket')
            self.s3_region = s3_config.get('region', 'us-east-1')
            self.presigned_expires = s3_config.get('presigned_url_expires', 604800)  # 7 days

            # Initialize S3 client (using IAM role or credentials)
            try:
                import boto3
                self.s3_client = boto3.client('s3', region_name=self.s3_region)
                logger.info(f"S3 client initialized for bucket: {self.s3_bucket}")
            except ImportError:
                logger.error("boto3 not installed. Install with: pip install boto3")
                self.s3_client = None
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                self.s3_client = None
        else:
            self.s3_client = None

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
            except:
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
            S3 key or local path if successful, None otherwise
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
            if self.storage_type == 's3' and self.s3_client:
                # Upload to S3
                s3_key = f"frames/{filename}"
                self.s3_client.upload_file(temp_frame_path, self.s3_bucket, s3_key)

                # Store metadata
                self.metadata[event_id] = {
                    'event': event,
                    's3_key': s3_key,
                    'bucket': self.s3_bucket,
                    'timestamp': event['timestamp']
                }
                self._save_metadata()

                logger.info(f"Uploaded frame to S3: {s3_key}")
                return s3_key

            else:
                # Local storage
                local_path = os.path.join(self.local_dir, filename)

                # Copy temp frame to permanent location
                import shutil
                shutil.copy2(temp_frame_path, local_path)

                # Store metadata
                self.metadata[event_id] = {
                    'event': event,
                    'local_path': local_path,
                    'timestamp': event['timestamp']
                }
                self._save_metadata()

                logger.info(f"Saved frame locally: {local_path}")
                return local_path

        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
            return None

    def get_frames_for_events(self, events: List[Dict]) -> Dict[str, str]:
        """
        Get frame URLs/paths for a list of events.

        Args:
            events: List of enriched event dictionaries

        Returns:
            Dictionary mapping event_id to URL/path
        """
        result = {}

        for event in events:
            event_id = f"{event['timestamp']}_{event['track_id']}"

            if event_id not in self.metadata:
                continue

            frame_data = self.metadata[event_id]

            if self.storage_type == 's3' and self.s3_client:
                # Generate presigned URL
                s3_key = frame_data['s3_key']
                try:
                    url = self.s3_client.generate_presigned_url(
                        'get_object',
                        Params={
                            'Bucket': self.s3_bucket,
                            'Key': s3_key
                        },
                        ExpiresIn=self.presigned_expires
                    )
                    result[event_id] = url
                except Exception as e:
                    logger.error(f"Failed to generate presigned URL: {e}")
            else:
                # Local file path
                result[event_id] = frame_data['local_path']

        return result

    def cleanup_old_metadata(self, days: int = 7):
        """
        Remove metadata for frames older than specified days.

        Args:
            days: Number of days to retain
        """
        cutoff = datetime.now() - timedelta(days=days)

        to_remove = []
        for event_id, data in self.metadata.items():
            try:
                timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                timestamp = timestamp.replace(tzinfo=None)

                if timestamp < cutoff:
                    to_remove.append(event_id)
            except:
                continue

        for event_id in to_remove:
            del self.metadata[event_id]

        if to_remove:
            self._save_metadata()
            logger.info(f"Cleaned up {len(to_remove)} old frame metadata entries")

    def get_frame_info(self, event_id: str) -> Optional[Dict]:
        """Get frame metadata for an event ID."""
        return self.metadata.get(event_id)
