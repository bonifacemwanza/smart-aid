"""Camera service for video capture and frame management."""

import asyncio
import threading
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


class CameraService:
    """Manages camera lifecycle and frame capture.

    Runs capture in a background thread to avoid blocking async operations.
    """

    def __init__(self):
        self._camera: cv2.VideoCapture | None = None
        self._source: int | str = 0
        self._current_frame: NDArray[np.uint8] | None = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._capture_thread: threading.Thread | None = None

    def set_source(self, source: int | str) -> None:
        """Set the video source before initialization."""
        self._source = source

    async def initialize(self) -> bool:
        """Initialize the camera and start capture thread."""
        print(f"Opening camera: {self._source}")

        self._camera = cv2.VideoCapture(self._source)
        if not self._camera.isOpened():
            print(f"Failed to open camera: {self._source}")
            return False

        print("Camera ready!")

        # Start background capture thread
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        return True

    async def cleanup(self) -> None:
        """Release camera resources."""
        self._running = False

        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)

        if self._camera:
            self._camera.release()
            self._camera = None

        print("Camera released.")

    def _capture_loop(self) -> None:
        """Background thread for continuous frame capture."""
        while self._running and self._camera and self._camera.isOpened():
            ret, frame = self._camera.read()
            if ret:
                with self._frame_lock:
                    self._current_frame = frame.copy()

            # Small delay to prevent CPU hogging
            threading.Event().wait(0.01)

    def get_current_frame(self) -> NDArray[np.uint8] | None:
        """Get the most recent captured frame."""
        with self._frame_lock:
            if self._current_frame is not None:
                return self._current_frame.copy()
        return None

    def is_ready(self) -> bool:
        """Check if camera is initialized and capturing."""
        return self._camera is not None and self._camera.isOpened() and self._running


# Global singleton instance
camera_service = CameraService()
