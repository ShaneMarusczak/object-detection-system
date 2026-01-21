"""
VLM Analyzer Consumer - Sends frames to remote VLM for analysis.

This consumer:
1. Receives events with vlm_analyze actions
2. Encodes the frame as base64
3. Sends to the configured analyzer endpoint (e.g., Orin 2)
4. Receives analysis text
5. Fans out to configured notifiers
"""

import base64
import logging
from multiprocessing import Queue
from typing import Any

import requests

from .notifiers import Notifier, create_notifiers

logger = logging.getLogger(__name__)


def call_analyzer(
    image_path: str,
    prompt: str,
    analyzer_url: str,
    timeout: int = 60,
    event: dict[str, Any] | None = None,
) -> str | None:
    """
    Call the VLM analyzer endpoint with an image and prompt.

    Args:
        image_path: Path to the image file
        prompt: Analysis prompt
        analyzer_url: URL of the analyzer endpoint
        timeout: Request timeout in seconds
        event: Optional event dict for context

    Returns:
        Analysis text or None if failed
    """
    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to read image: {e}")
        return None

    # Build request payload
    payload = {
        "image": image_b64,
        "prompt": prompt,
    }

    # Include event context if available
    if event:
        payload["context"] = {
            "event": event.get("_event_name"),
            "object_class": event.get("object_class_name"),
            "confidence": event.get("confidence"),
            "timestamp": event.get("timestamp"),
        }

    try:
        response = requests.post(
            analyzer_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )

        if not response.ok:
            logger.warning(
                f"Analyzer returned {response.status_code}: {response.text[:100]}"
            )
            return None

        result = response.json()
        analysis = result.get("analysis")

        if not analysis:
            logger.warning("Analyzer returned empty analysis")
            return None

        logger.debug(f"Analysis received: {analysis[:50]}...")
        return analysis

    except requests.Timeout:
        logger.warning(f"Analyzer timeout after {timeout}s")
        return None
    except requests.RequestException as e:
        logger.error(f"Analyzer request failed: {e}")
        return None
    except ValueError as e:
        logger.error(f"Invalid JSON response: {e}")
        return None


def process_vlm_event(
    event: dict[str, Any],
    notifiers: dict[str, Notifier],
    temp_frame_dir: str | None = None,
) -> None:
    """
    Process a single event with vlm_analyze action.

    Args:
        event: Event dictionary with _vlm_config
        notifiers: Dictionary of notifier ID -> Notifier
        temp_frame_dir: Temp frame directory (for fallback)
    """
    vlm_config = event.get("_vlm_config", {})
    if not vlm_config:
        return

    # Get frame path
    frame_path = event.get("_frame_path")
    if not frame_path:
        logger.warning("No frame path in event, skipping VLM analysis")
        return

    # Get analyzer config
    analyzer_url = vlm_config.get("_analyzer_url")
    if not analyzer_url:
        logger.error("No analyzer URL configured")
        return

    prompt = vlm_config.get("prompt", "Describe this image.")
    timeout = vlm_config.get("_analyzer_timeout", 60)

    # Call analyzer
    logger.info(f"Calling VLM analyzer for event {event.get('_event_name')}")
    analysis = call_analyzer(
        image_path=frame_path,
        prompt=prompt,
        analyzer_url=analyzer_url,
        timeout=timeout,
        event=event,
    )

    # Use fallback if analyzer failed
    if not analysis:
        obj_class = event.get("object_class_name", "object")
        confidence = event.get("confidence", 0)
        analysis = (
            f"Detection: {obj_class} (confidence: {confidence:.2f}). "
            "Automated analysis unavailable."
        )
        logger.warning("Using fallback analysis message")

    # Fan out to notifiers
    notify_ids = vlm_config.get("notify", [])
    for notify_id in notify_ids:
        notifier = notifiers.get(notify_id)
        if notifier:
            try:
                success = notifier.send(
                    analysis=analysis,
                    image_path=frame_path,
                    event=event,
                )
                if success:
                    logger.debug(f"Notification sent via {notify_id}")
                else:
                    logger.warning(f"Notification failed for {notify_id}")
            except Exception as e:
                logger.error(f"Notifier {notify_id} error: {e}")
        else:
            logger.warning(f"Notifier not found: {notify_id}")


def vlm_analyzer_consumer(
    event_queue: Queue,
    config: dict[str, Any],
) -> None:
    """
    VLM analyzer consumer process.

    Receives events from the queue, calls the analyzer, and notifies.

    Args:
        event_queue: Queue of events to process
        config: Configuration dictionary with notifiers list
    """
    logger.info("VLM Analyzer consumer started")

    # Create notifiers from config
    notifier_configs = config.get("notifiers", [])
    notifiers = create_notifiers(notifier_configs)
    logger.info(f"Initialized {len(notifiers)} notifier(s)")

    temp_frame_dir = config.get("temp_frame_dir")

    try:
        while True:
            event = event_queue.get()

            # Shutdown signal
            if event is None:
                logger.info("VLM Analyzer consumer received shutdown signal")
                break

            try:
                process_vlm_event(event, notifiers, temp_frame_dir)
            except Exception as e:
                logger.error(f"Error processing VLM event: {e}", exc_info=True)

    except KeyboardInterrupt:
        logger.info("VLM Analyzer consumer interrupted")
    finally:
        logger.info("VLM Analyzer consumer shutdown")
