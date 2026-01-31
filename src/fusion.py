from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import FusionConfig
from src.detector import Detection


@dataclass
class Obstacle:
    detection: Detection
    distance: float
    position: Literal["left", "center", "right"]
    priority: int

    @property
    def class_name(self) -> str:
        return self.detection.class_name

    @property
    def confidence(self) -> float:
        return self.detection.confidence

    @property
    def is_danger(self) -> bool:
        return self.distance < 1.5

    @property
    def is_warning(self) -> bool:
        return self.distance < 3.0

    def to_alert_text(self) -> str:
        return f"{self.class_name}, {self.distance:.1f} meters, {self.position}"


class FusionEngine:
    PRIORITY_CLASSES = {
        "person": 30,
        "car": 25,
        "bicycle": 20,
        "motorcycle": 20,
        "bus": 25,
        "truck": 25,
        "dog": 15,
        "cat": 10,
        "chair": 10,
        "door": 5,
    }

    def __init__(self, config: FusionConfig) -> None:
        self.config = config

    def process(
        self,
        detections: list[Detection],
        depth_map: NDArray[np.float32],
        frame_width: int,
    ) -> list[Obstacle]:
        obstacles: list[Obstacle] = []

        for det in detections:
            h, w = depth_map.shape[:2]
            scale_x = w / frame_width
            scaled_bbox = (
                int(det.x1 * scale_x),
                int(det.y1 * scale_x),
                int(det.x2 * scale_x),
                int(det.y2 * scale_x),
            )

            distance = self._get_distance(depth_map, scaled_bbox)
            position = self._get_position(det.center[0], frame_width)
            priority = self._calculate_priority(det, distance)

            obstacles.append(
                Obstacle(
                    detection=det,
                    distance=distance,
                    position=position,
                    priority=priority,
                )
            )

        obstacles.sort(key=lambda o: o.priority, reverse=True)
        return obstacles

    def _get_distance(
        self, depth_map: NDArray[np.float32], bbox: tuple[int, int, int, int]
    ) -> float:
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return 10.0

        region = depth_map[y1:y2, x1:x2]

        cy = (y2 - y1) // 2
        cx = (x2 - x1) // 2
        my = max(1, (y2 - y1) // 4)
        mx = max(1, (x2 - x1) // 4)

        center_region = region[cy - my : cy + my, cx - mx : cx + mx]

        if center_region.size > 0:
            avg_depth = float(center_region.mean())
        else:
            avg_depth = float(region.mean())

        distance = (1.0 - avg_depth) * 10.0
        return max(0.1, distance)

    def _get_position(
        self, center_x: int, frame_width: int
    ) -> Literal["left", "center", "right"]:
        relative_x = center_x / frame_width

        if relative_x < self.config.position_threshold:
            return "left"
        elif relative_x > (1.0 - self.config.position_threshold):
            return "right"
        else:
            return "center"

    def _calculate_priority(self, detection: Detection, distance: float) -> int:
        priority = 0

        if distance < self.config.danger_zone:
            priority += 100
        elif distance < self.config.warning_zone:
            priority += 50
        else:
            priority += 10

        class_bonus = self.PRIORITY_CLASSES.get(detection.class_name, 0)
        priority += class_bonus

        priority += int(detection.confidence * 10)

        return priority

    def draw_obstacles(
        self,
        frame: NDArray[np.uint8],
        obstacles: list[Obstacle],
        thickness: int = 2,
    ) -> NDArray[np.uint8]:
        frame = frame.copy()

        for obs in obstacles:
            det = obs.detection

            if obs.is_danger:
                color = (0, 0, 255)
            elif obs.is_warning:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, thickness)

            label = f"{obs.class_name}: {obs.distance:.1f}m ({obs.position})"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            cv2.rectangle(
                frame,
                (det.x1, det.y1 - label_h - baseline - 5),
                (det.x1 + label_w, det.y1),
                color,
                -1,
            )

            cv2.putText(
                frame,
                label,
                (det.x1, det.y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        return frame


if __name__ == "__main__":
    config = FusionConfig()
    fusion = FusionEngine(config)

    test_detections = [
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox=(100, 100, 200, 300),
            center=(150, 200),
        ),
        Detection(
            class_id=2,
            class_name="car",
            confidence=0.72,
            bbox=(400, 150, 550, 280),
            center=(475, 215),
        ),
    ]

    test_depth = np.random.rand(480, 640).astype(np.float32)
    test_depth[100:300, 100:200] = 0.8
    test_depth[150:280, 400:550] = 0.5

    obstacles = fusion.process(test_detections, test_depth, 640)

    print("Detected Obstacles:")
    for obs in obstacles:
        print(f"  - {obs.to_alert_text()} (priority: {obs.priority})")

    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = fusion.draw_obstacles(test_frame, obstacles)
    cv2.imshow("Fusion Test", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
