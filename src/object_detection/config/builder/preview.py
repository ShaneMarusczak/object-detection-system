"""
Preview service for the config builder.

Handles camera connection, HTTP preview server, and frame capture with annotations.
"""

import os
import socket
import subprocess
import sys

import cv2

from .prompts import Colors


def get_local_ip() -> str:
    """Get local IP address for remote access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def start_preview_server(preview_dir: str) -> subprocess.Popen | None:
    """
    Start HTTP server in preview folder for preview images.

    Args:
        preview_dir: Directory to serve files from

    Returns:
        Popen process if started, None if server already running or failed
    """
    os.makedirs(preview_dir, exist_ok=True)

    local_ip = get_local_ip()
    preview_url = f"http://{local_ip}:8000/preview.jpg"

    # Check if something is already running on port 8000
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("localhost", 8000))
        sock.close()
        if result == 0:
            print(
                f"{Colors.GREEN}Preview server already running:{Colors.RESET} {preview_url}"
            )
            return None
    except Exception:
        pass

    # Start server (bind to all interfaces for remote access)
    try:
        http_server = subprocess.Popen(
            [sys.executable, "-m", "http.server", "8000", "--bind", "0.0.0.0"],
            cwd=preview_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"{Colors.GREEN}Preview server started:{Colors.RESET} {preview_url}")
        print(f"{Colors.GRAY}(Open in browser to see captured frames){Colors.RESET}")
        return http_server
    except Exception as e:
        print(f"{Colors.YELLOW}Could not start preview server: {e}{Colors.RESET}")
        print(
            f"{Colors.GRAY}Frames will still be saved to {preview_dir}/{Colors.RESET}"
        )
        return None


def stop_preview_server(http_server: subprocess.Popen | None) -> None:
    """Stop the HTTP preview server."""
    if http_server is not None:
        http_server.terminate()
        try:
            http_server.wait(timeout=2)
        except Exception:
            http_server.kill()


def capture_preview(
    cap: cv2.VideoCapture | None,
    preview_dir: str,
    frame=None,
) -> None:
    """Capture a raw preview frame."""
    if frame is None:
        if cap is None:
            return
        ret, frame = cap.read()
        if not ret:
            return

    preview_path = os.path.join(preview_dir, "preview.jpg")
    cv2.imwrite(preview_path, frame)


def capture_annotated_preview(
    cap: cv2.VideoCapture | None,
    preview_dir: str,
    lines: list[dict] | None = None,
    zones: list[dict] | None = None,
) -> None:
    """Capture preview with line and zone annotations."""
    if cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        return

    height, width = frame.shape[:2]

    # Draw lines (yellow)
    if lines:
        for line in lines:
            line_type = line.get("type", "vertical")
            position_pct = line.get("position_pct", 50)
            description = line.get("description", "")

            if line_type == "vertical":
                x = int(width * position_pct / 100)
                cv2.line(frame, (x, 0), (x, height), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    description,
                    (x + 5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{position_pct}%",
                    (x + 5, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
            else:
                y = int(height * position_pct / 100)
                cv2.line(frame, (0, y), (width, y), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    description,
                    (10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{position_pct}%",
                    (10, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

    # Draw zones (cyan)
    if zones:
        for zone in zones:
            x1 = int(width * zone.get("x1_pct", 0) / 100)
            y1 = int(height * zone.get("y1_pct", 0) / 100)
            x2 = int(width * zone.get("x2_pct", 100) / 100)
            y2 = int(height * zone.get("y2_pct", 100) / 100)
            description = zone.get("description", "")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                frame,
                description,
                (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

    # Save
    preview_path = os.path.join(preview_dir, "preview.jpg")
    cv2.imwrite(preview_path, frame)
