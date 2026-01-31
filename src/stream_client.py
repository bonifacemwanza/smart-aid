import time
from urllib.request import urlopen

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import StreamConfig


class StreamClient:
    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self.stream = None
        self.bytes_buffer = b""

    def connect(self) -> bool:
        for attempt in range(self.config.reconnect_attempts):
            try:
                self.stream = urlopen(self.config.pi_url, timeout=self.config.timeout)
                return True
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.config.reconnect_attempts - 1:
                    time.sleep(self.config.reconnect_delay)
        return False

    def read(self) -> NDArray[np.uint8] | None:
        if self.stream is None:
            return None

        try:
            self.bytes_buffer += self.stream.read(1024)

            start = self.bytes_buffer.find(b"\xff\xd8")
            end = self.bytes_buffer.find(b"\xff\xd9")

            if start != -1 and end != -1:
                jpg = self.bytes_buffer[start : end + 2]
                self.bytes_buffer = self.bytes_buffer[end + 2 :]

                frame = cv2.imdecode(
                    np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                return frame

        except Exception:
            return None

        return None

    def disconnect(self) -> None:
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.bytes_buffer = b""

    def __enter__(self) -> "StreamClient":
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self.stream is not None


class StreamClientCV:
    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self.cap: cv2.VideoCapture | None = None

    def connect(self) -> bool:
        for attempt in range(self.config.reconnect_attempts):
            try:
                self.cap = cv2.VideoCapture(self.config.pi_url)
                if self.cap.isOpened():
                    return True
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < self.config.reconnect_attempts - 1:
                time.sleep(self.config.reconnect_delay)
        return False

    def read(self) -> NDArray[np.uint8] | None:
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return frame

    def disconnect(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "StreamClientCV":
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.isOpened()


if __name__ == "__main__":
    from src.utils import FPSCounter, draw_fps

    config = StreamConfig(pi_url="http://192.168.1.100:5000/video_feed")
    fps_counter = FPSCounter()

    print(f"Connecting to: {config.pi_url}")

    with StreamClientCV(config) as client:
        if not client.is_connected:
            print("Failed to connect to Pi stream")
            print("Make sure:")
            print("  1. Pi is running stream_server.py")
            print("  2. Pi IP address is correct")
            print("  3. Both devices are on same network")
            exit(1)

        print("Connected! Press 'q' to quit, 's' to save frame")

        frame_count = 0
        while True:
            frame = client.read()
            if frame is None:
                continue

            fps = fps_counter.update()
            frame = draw_fps(frame, fps)

            cv2.imshow("Pi Stream", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = f"data/captures/capture_{frame_count:04d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
                frame_count += 1

    cv2.destroyAllWindows()
