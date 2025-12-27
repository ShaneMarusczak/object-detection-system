"""
On-demand snapshot server - grab current camera frame via HTTP.

Runs in a daemon thread alongside detection. Browse to the URL
to see what the camera sees right now.
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2


class SnapshotHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures and returns current camera frame."""

    camera_url: str = ""

    def do_GET(self):
        """Handle GET request - capture frame and return as JPEG."""
        if self.path not in ("/", "/snapshot"):
            self.send_error(404, "Not found. Try /snapshot")
            return

        try:
            cap = cv2.VideoCapture(self.camera_url)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                self.send_error(503, "Failed to capture frame from camera")
                return

            _, jpg = cv2.imencode(".jpg", frame)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(jpg.tobytes())

        except Exception as e:
            self.send_error(500, f"Error capturing frame: {e}")

    def log_message(self, format, *args):
        """Suppress default logging to avoid cluttering detection output."""
        pass


def start_snapshot_server(camera_url: str, port: int = 8085) -> str:
    """
    Start snapshot server in daemon thread.

    Args:
        camera_url: Camera URL to capture from
        port: Port to listen on (default 8085)

    Returns:
        URL string for accessing snapshots
    """
    SnapshotHandler.camera_url = camera_url

    server = HTTPServer(("0.0.0.0", port), SnapshotHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    return f"http://localhost:{port}/snapshot"
