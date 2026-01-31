import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray


class FPSCounter:
    def __init__(self, window_size: int = 30) -> None:
        self.timestamps: deque[float] = deque(maxlen=window_size)

    def update(self) -> float:
        now = time.perf_counter()
        self.timestamps.append(now)
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed

    def reset(self) -> None:
        self.timestamps.clear()


@dataclass
class Timer:
    name: str = ""
    times: list[float] = field(default_factory=list)
    _start: float = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        elapsed = time.perf_counter() - self._start
        self.times.append(elapsed)
        return elapsed

    @property
    def avg_ms(self) -> float:
        if not self.times:
            return 0.0
        return sum(self.times) / len(self.times) * 1000

    def reset(self) -> None:
        self.times.clear()


def draw_fps(frame: NDArray[np.uint8], fps: float) -> NDArray[np.uint8]:
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def resize_with_aspect(
    frame: NDArray[np.uint8], target_width: int
) -> NDArray[np.uint8]:
    h, w = frame.shape[:2]
    aspect = w / h
    new_width = target_width
    new_height = int(target_width / aspect)
    return cv2.resize(frame, (new_width, new_height))


def create_side_by_side(
    frame1: NDArray[np.uint8],
    frame2: NDArray[np.uint8],
    labels: tuple[str, str] = ("Original", "Processed"),
) -> NDArray[np.uint8]:
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]

    if len(frame2.shape) == 2:
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

    if h1 != h2:
        scale = h1 / h2
        frame2 = cv2.resize(frame2, (int(w2 * scale), h1))
        h2, w2 = frame2.shape[:2]

    combined = np.hstack([frame1, frame2])

    cv2.putText(
        combined,
        labels[0],
        (10, h1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        combined,
        labels[1],
        (w1 + 10, h2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return combined


def save_frame(frame: NDArray[np.uint8], path: str) -> bool:
    try:
        cv2.imwrite(path, frame)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    fps_counter = FPSCounter()
    for _ in range(100):
        fps_counter.update()
        time.sleep(0.033)
    print(f"FPS: {fps_counter.update():.1f}")
