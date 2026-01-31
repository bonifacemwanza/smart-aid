import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import DepthConfig


class DepthEstimator:
    def __init__(self, config: DepthConfig) -> None:
        self.config = config
        self.pipe = None

    def load(self) -> bool:
        try:
            from transformers import pipeline
            import torch

            model_map = {
                "vits": "depth-anything/Depth-Anything-V2-Small-hf",
                "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
                "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
            }

            model_name = model_map.get(self.config.model)
            if model_name is None:
                print(f"Unknown model: {self.config.model}")
                return False

            # Force CPU to avoid MPS compatibility issues with bicubic upsampling
            device = "cpu"
            print(f"Loading depth model: {model_name} (device: {device})")
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=device)
            return True

        except Exception as e:
            print(f"Failed to load depth model: {e}")
            return False

    def estimate(self, frame: NDArray[np.uint8]) -> NDArray[np.float32] | None:
        if self.pipe is None:
            return None

        try:
            from PIL import Image

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            result = self.pipe(pil_image)
            depth_pil = result["depth"]
            depth_map = np.array(depth_pil, dtype=np.float32)

            depth_normalized = (depth_map - depth_map.min()) / (
                depth_map.max() - depth_map.min() + 1e-8
            )

            return depth_normalized

        except Exception as e:
            print(f"Depth estimation error: {e}")
            return None

    def get_distance_at(
        self, depth_map: NDArray[np.float32], x: int, y: int
    ) -> float:
        h, w = depth_map.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        depth_value = depth_map[y, x]

        # Use indoor-calibrated max distance if enabled
        max_dist = self.config.indoor_max_distance if self.config.indoor_mode else self.config.max_distance

        # Non-linear mapping for better indoor accuracy
        # Depth model gives relative depth where higher = closer
        # Use exponential mapping for more realistic indoor distances
        if self.config.indoor_mode:
            # Exponential mapping: closer objects get more accurate distances
            distance = max_dist * (1.0 - depth_value) ** 1.5
        else:
            distance = (1.0 - depth_value) * max_dist

        return float(distance)

    def get_distance_in_region(
        self,
        depth_map: NDArray[np.float32],
        bbox: tuple[int, int, int, int],
    ) -> float:
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            max_dist = self.config.indoor_max_distance if self.config.indoor_mode else self.config.max_distance
            return max_dist

        center_y = (y2 - y1) // 2
        center_x = (x2 - x1) // 2
        margin_y = max(1, (y2 - y1) // 4)
        margin_x = max(1, (x2 - x1) // 4)

        center_region = region[
            center_y - margin_y : center_y + margin_y,
            center_x - margin_x : center_x + margin_x,
        ]

        if center_region.size == 0:
            avg_depth = float(region.mean())
        else:
            avg_depth = float(center_region.mean())

        # Use indoor-calibrated max distance if enabled
        max_dist = self.config.indoor_max_distance if self.config.indoor_mode else self.config.max_distance

        # Non-linear mapping for better indoor accuracy
        if self.config.indoor_mode:
            # Exponential mapping: closer objects get more accurate distances
            distance = max_dist * (1.0 - avg_depth) ** 1.5
        else:
            distance = (1.0 - avg_depth) * max_dist

        return distance

    def get_qualitative_distance(self, distance: float) -> str:
        """Convert distance to qualitative description for blind users.

        Returns descriptions like 'within reach', 'very close', 'nearby'.
        """
        if distance < self.config.very_close:
            return "within reach"
        elif distance < self.config.close:
            return "very close"
        elif distance < self.config.nearby:
            return "nearby"
        else:
            return "across the room"

    def get_steps(self, distance: float) -> int:
        """Convert distance to approximate step count."""
        return max(1, int(distance / self.config.step_length))

    def colorize(
        self, depth_map: NDArray[np.float32], colormap: str | None = None
    ) -> NDArray[np.uint8]:
        if colormap is None:
            colormap = self.config.colormap

        colormap_map = {
            "inferno": cv2.COLORMAP_INFERNO,
            "jet": cv2.COLORMAP_JET,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma": cv2.COLORMAP_PLASMA,
            "magma": cv2.COLORMAP_MAGMA,
            "turbo": cv2.COLORMAP_TURBO,
        }

        cv_colormap = colormap_map.get(colormap, cv2.COLORMAP_INFERNO)

        depth_uint8 = (depth_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_uint8, cv_colormap)

        return colored


if __name__ == "__main__":
    import sys

    config = DepthConfig(model="vits")
    estimator = DepthEstimator(config)

    print("Loading Depth Anything V2 model...")
    if not estimator.load():
        print("Failed to load model")
        sys.exit(1)
    print("Model loaded successfully!")

    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.rectangle(test_image, (400, 200), (550, 350), (128, 128, 128), -1)

    print("Running depth estimation...")
    depth_map = estimator.estimate(test_image)

    if depth_map is not None:
        print(f"Depth map shape: {depth_map.shape}")
        print(f"Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")

        distance = estimator.get_distance_at(depth_map, 200, 200)
        print(f"Distance at (200, 200): {distance:.2f}m")

        colored = estimator.colorize(depth_map)
        cv2.imshow("Test Image", test_image)
        cv2.imshow("Depth Map", colored)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
