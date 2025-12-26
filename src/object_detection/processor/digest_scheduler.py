"""
Digest Scheduler

Lightweight scheduler for periodic email digests using threading.Event
for efficient sleep/wake with clean shutdown support.

Config format:
    digests:
      - id: daily_summary
        events: [car_detected]
        schedule:
          interval_hours: 24
        on_shutdown: true
        photos: true
"""

import logging
import threading
from datetime import datetime, timezone

from .email_digest import generate_email_digest

logger = logging.getLogger(__name__)


class DigestScheduler:
    """
    Schedules periodic email digests using interval-based timing.

    Uses threading.Event for efficient blocking that can be interrupted
    immediately on shutdown - no polling, no hanging.
    """

    def __init__(
        self,
        json_dir: str,
        digest_configs: list[dict],
        notification_config: dict,
        frame_storage_config: dict,
    ):
        """
        Initialize the digest scheduler.

        Args:
            json_dir: Directory containing JSON log files
            digest_configs: List of digest configurations with schedules
            notification_config: Email notification settings
            frame_storage_config: Frame storage settings for photos
        """
        self._json_dir = json_dir
        self._notification_config = notification_config
        self._frame_storage_config = frame_storage_config

        # Filter to only digests with schedules
        self._scheduled_digests = [
            d for d in digest_configs
            if d.get("schedule", {}).get("interval_hours")
        ]

        # Track last send time per digest (for interval calculation)
        self._last_sent: dict[str, datetime] = {}

        # Shutdown signal (like a CancellationToken)
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None

        # Track start time for digest time windows
        self._start_time = datetime.now(timezone.utc)

    def start(self) -> None:
        """Start the scheduler thread if there are scheduled digests."""
        if not self._scheduled_digests:
            logger.debug("No scheduled digests configured")
            return

        count = len(self._scheduled_digests)
        logger.info(f"Starting digest scheduler with {count} digest(s)")
        for digest in self._scheduled_digests:
            hours = digest.get("schedule", {}).get("interval_hours", 24)
            logger.info(f"  - {digest.get('id')}: every {hours} hours")

        self._thread = threading.Thread(
            target=self._run,
            name="DigestScheduler",
            daemon=True,  # Dies with parent process
        )
        self._thread.start()

    def stop(self) -> None:
        """
        Stop the scheduler cleanly.

        Signals shutdown and waits briefly for thread to exit.
        Thread is daemon, so it will die with process regardless.
        """
        if not self._thread:
            return

        logger.debug("Stopping digest scheduler...")
        self._shutdown.set()  # Wakes thread immediately from wait()

        # Wait up to 2 seconds for clean exit
        self._thread.join(timeout=2.0)

        if self._thread.is_alive():
            logger.warning("Digest scheduler did not stop cleanly")
        else:
            logger.debug("Digest scheduler stopped")

    def _run(self) -> None:
        """
        Main scheduler loop.

        Calculates time until next digest, sleeps efficiently,
        wakes on timeout (send digest) or shutdown signal (exit).
        """
        # Initialize last sent times to now (first digest after interval)
        now = datetime.now(timezone.utc)
        for digest in self._scheduled_digests:
            digest_id = digest.get("id", "digest")
            self._last_sent[digest_id] = now

        while not self._shutdown.is_set():
            # Find the next digest that needs to be sent
            seconds_until_next = self._calculate_seconds_until_next()

            if seconds_until_next <= 0:
                # A digest is due now
                self._send_due_digests()
                continue

            # Wait for timeout OR shutdown signal
            # Returns True if shutdown was signaled, False if timeout
            if self._shutdown.wait(timeout=seconds_until_next):
                break  # Shutdown requested, exit loop

            # Timeout expired - send any due digests
            self._send_due_digests()

        logger.debug("Digest scheduler loop exited")

    def _calculate_seconds_until_next(self) -> float:
        """Calculate seconds until the next scheduled digest."""
        now = datetime.now(timezone.utc)
        min_seconds = float("inf")

        for digest in self._scheduled_digests:
            digest_id = digest.get("id", "digest")
            interval_hours = digest.get("schedule", {}).get("interval_hours", 24)
            interval_seconds = interval_hours * 3600

            last_sent = self._last_sent.get(digest_id, now)
            next_send = last_sent.timestamp() + interval_seconds
            seconds_until = next_send - now.timestamp()

            min_seconds = min(min_seconds, seconds_until)

        # Clamp to reasonable range
        return max(0, min(min_seconds, 86400))  # Max 24 hours

    def _send_due_digests(self) -> None:
        """Send any digests that are due based on their intervals."""
        now = datetime.now(timezone.utc)

        for digest in self._scheduled_digests:
            digest_id = digest.get("id", "digest")
            interval_hours = digest.get("schedule", {}).get("interval_hours", 24)
            interval_seconds = interval_hours * 3600

            last_sent = self._last_sent.get(digest_id, self._start_time)
            elapsed = (now - last_sent).total_seconds()

            if elapsed >= interval_seconds:
                logger.info(f"Scheduled digest '{digest_id}' triggered")
                self._send_single_digest(digest, last_sent, now)
                self._last_sent[digest_id] = now

    def _send_single_digest(
        self,
        digest_config: dict,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Send a single digest for the given time window."""
        try:
            # Build config matching generate_email_digest expectations
            config = {
                "digests": [digest_config],
                "notification_config": self._notification_config,
                "frame_service_config": {"storage": self._frame_storage_config},
            }
            generate_email_digest(self._json_dir, config, start_time)
        except Exception as e:
            logger.error(f"Error sending scheduled digest: {e}", exc_info=True)
