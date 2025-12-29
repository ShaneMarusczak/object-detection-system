"""
Snapshot server - serves latest camera frame via HTTP.

Detector saves latest.jpg periodically. This server serves that file.
Runs in a daemon thread alongside detection.
"""

import os
import socket
import subprocess
import sys
import threading

from .constants import SNAPSHOT_DIR


def _get_local_ip() -> str:
    """Get local IP address for remote access."""
    try:
        # Connect to external address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def start_snapshot_server(port: int = 8085) -> tuple[str, subprocess.Popen | None]:
    """
    Start static file server for snapshots.

    Args:
        port: Port to listen on (default 8085)

    Returns:
        Tuple of (URL string, server process or None if failed)
    """
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    local_ip = _get_local_ip()
    url = f"http://{local_ip}:{port}/latest.jpg"

    # Check if port is already in use
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        if result == 0:
            # Port already in use, assume server is running
            return url, None
    except Exception:
        pass

    # Start simple HTTP server serving snapshot directory
    try:
        server = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(port), "--bind", "0.0.0.0"],
            cwd=SNAPSHOT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Give it a moment to start
        def check_server():
            import time

            time.sleep(0.5)

        thread = threading.Thread(target=check_server, daemon=True)
        thread.start()

        return url, server

    except Exception:
        return url, None
