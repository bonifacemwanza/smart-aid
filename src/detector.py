from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import DetectionConfig


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height


class Detector:
    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.model = None
        self.class_names: list[str] = []
        self.use_world_model = False

    def load(self) -> bool:
        try:
            # Check if using YOLO-World model
            if "world" in self.config.model.lower():
                from ultralytics import YOLOWorld

                self.model = YOLOWorld(self.config.model)
                self.use_world_model = True

                # Set custom classes if provided
                if self.config.classes:
                    self.model.set_classes(self.config.classes)
                    self.class_names = self.config.classes
                else:
                    # Default classes for visually impaired navigation
                    default_classes = [
                        "person",
                        "door",
                        "chair",
                        "table",
                        "car",
                        "bicycle",
                        "stairs",
                        "wall",
                        "tree",
                        "pole",
                        "bench",
                        "trash can",
                        "fire hydrant",
                        "curb",
                    ]
                    self.model.set_classes(default_classes)
                    self.class_names = default_classes
            else:
                from ultralytics import YOLO

                self.model = YOLO(self.config.model)
                self.class_names = list(self.model.names.values())
                self.use_world_model = False

            return True
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return False

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        if self.model is None:
            return []

        results = self.model(
            frame,
            conf=self.config.confidence,
            iou=self.config.iou_threshold,
            device=self.config.device,
            verbose=False,
        )

        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]

                if self.config.classes and class_name not in self.config.classes:
                    continue

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        center=center,
                    )
                )

        return detections

    def draw_detections(
        self,
        frame: NDArray[np.uint8],
        detections: list[Detection],
        thickness: int = 2,
        font_scale: float = 0.6,
    ) -> NDArray[np.uint8]:
        frame = frame.copy()

        for det in detections:
            color = (0, 255, 0)
            if det.confidence < 0.7:
                color = (0, 255, 255)
            if det.confidence < 0.5:
                color = (0, 165, 255)

            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, thickness)

            label = f"{det.class_name}: {det.confidence:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
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
                font_scale,
                (0, 0, 0),
                thickness,
            )

            cv2.circle(frame, det.center, 4, color, -1)

        return frame


if __name__ == "__main__":
    import sys

    config = DetectionConfig()
    detector = Detector(config)

    print("Loading YOLOv8 model...")
    if not detector.load():
        print("Failed to load model")
        sys.exit(1)
    print(f"Loaded model with {len(detector.class_names)} classes")

    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (128, 128, 128)

    cv2.putText(
        test_image,
        "Test Image - No detections expected",
        (100, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    detections = detector.detect(test_image)
    print(f"Detections: {len(detections)}")

    result = detector.draw_detections(test_image, detections)
    cv2.imshow("Detection Test", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
