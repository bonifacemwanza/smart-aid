import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import CameraConfig


class Camera:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.config.device)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        return True

    def read(self) -> NDArray[np.uint8] | None:
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)
        if self.config.flip_vertical:
            frame = cv2.flip(frame, 0)

        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()


if __name__ == "__main__":
    from src.utils import FPSCounter, draw_fps

    config = CameraConfig()
    fps_counter = FPSCounter()

    with Camera(config) as camera:
        if not camera.is_opened:
            print("Failed to open camera")
            exit(1)

        print(f"Camera opened: {config.width}x{config.height} @ {config.fps}fps")
        print("Press 'q' to quit")

        while True:
            frame = camera.read()
            if frame is None:
                continue

            fps = fps_counter.update()
            frame = draw_fps(frame, fps)

            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
