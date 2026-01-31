"""Smart navigator using Florence-2 for intelligent guidance.

This provides actual navigation help for blind users:
- Simple body-relative directions: "on your left", "straight ahead", "on your right"
- Distance in steps (not meters) - easier to understand
- Obstacle warnings with priority
- Finds specific objects when asked
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import re

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import Config
from src.depth import DepthEstimator
from src.detector import Detector
from src.florence import FlorenceModel
from src.fusion import FusionEngine, Obstacle


class SmartNavigator:
    """Navigation assistant using Florence-2 + YOLO + Depth.

    Provides:
    1. Quick safety scans with obstacle warnings
    2. Object search with directions
    3. Scene understanding for navigation
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.florence: FlorenceModel | None = None
        self.detector: Detector | None = None
        self.depth_estimator: DepthEstimator | None = None
        self.fusion_engine: FusionEngine | None = None
        self._loaded = False

    def load(self) -> bool:
        """Load all models."""
        success = True

        # Florence-2 for understanding
        if self.config.pipeline.use_florence:
            print("Loading Florence-2...")
            self.florence = FlorenceModel(self.config.florence)
            if not self.florence.load():
                print("Failed to load Florence-2")
                success = False

        # YOLO for fast detection
        if self.config.pipeline.use_yolo_world:
            print("Loading YOLO-World...")
            self.detector = Detector(self.config.detection)
            if not self.detector.load():
                print("Failed to load YOLO-World")
                success = False

        # Depth for distances
        if self.config.pipeline.use_depth:
            print("Loading Depth Estimator...")
            self.depth_estimator = DepthEstimator(self.config.depth)
            if not self.depth_estimator.load():
                print("Failed to load Depth Estimator")

        if self.depth_estimator:
            self.fusion_engine = FusionEngine(self.config.fusion)

        self._loaded = success
        return success

    def quick_scan(self, frame: NDArray[np.uint8]) -> str:
        """Quick safety scan - check for immediate obstacles.

        Returns a short, urgent message if obstacles are close.
        """
        if not self._loaded:
            return "System loading..."

        obstacles = self._get_obstacles(frame)

        if not obstacles:
            return "Clear path ahead."

        # Sort by distance
        obstacles = sorted(obstacles, key=lambda o: o.distance)

        # Check for immediate danger (< 0.5m - within reach)
        dangers = [o for o in obstacles if o.distance < 0.5]
        if dangers:
            obs = dangers[0]
            direction = self._to_direction(obs.position)
            return f"STOP! {obs.detection.class_name} {direction}, within reach!"

        # Check for very close (0.5-1.5m)
        very_close = [o for o in obstacles if o.distance < 1.5]
        if very_close:
            obs = very_close[0]
            direction = self._to_direction(obs.position)
            distance_desc = self._distance_to_words(obs.distance)
            return f"Caution: {obs.detection.class_name} {direction}, {distance_desc}."

        # Check for nearby (1.5-3m)
        nearby = [o for o in obstacles if o.distance < 3.0]
        if nearby:
            obs = nearby[0]
            direction = self._to_direction(obs.position)
            distance_desc = self._distance_to_words(obs.distance)
            return f"{obs.detection.class_name} {direction}, {distance_desc}."

        # Further obstacles
        obs = obstacles[0]
        direction = self._to_direction(obs.position)
        return f"Clear nearby. {obs.detection.class_name} {direction}, across the room."

    def _distance_to_words(self, distance: float) -> str:
        """Convert distance to natural language for blind users.

        Uses qualitative descriptions instead of exact measurements.
        """
        if distance < 0.5:
            return "within reach"
        elif distance < 1.0:
            return "one step away"
        elif distance < 1.5:
            return "very close"
        elif distance < 2.5:
            return "a few steps away"
        elif distance < 4.0:
            return "nearby"
        else:
            return "across the room"

    def navigate(self, query: str, frame: NDArray[np.uint8]) -> str:
        """Process any navigation query intelligently."""
        if not self._loaded:
            return "System loading..."

        query_lower = query.lower().strip()

        # Determine what the user wants
        if self._is_search_query(query_lower):
            target = self._extract_target(query_lower)
            if target:
                return self._find_object(target, frame)

        if self._is_safety_query(query_lower):
            return self._describe_path(frame)

        if self._is_describe_query(query_lower):
            return self._describe_scene(frame)

        # Default: describe scene with obstacles
        return self._describe_scene(frame)

    def _is_search_query(self, query: str) -> bool:
        """Check if query is looking for something."""
        search_words = ["where", "find", "locate", "search", "look for",
                       "get to", "navigate to", "go to", "how do i get"]
        return any(w in query for w in search_words)

    def _is_safety_query(self, query: str) -> bool:
        """Check if query is about safety/path."""
        safety_words = ["safe", "clear", "obstacle", "block", "path",
                       "walk", "forward", "ahead"]
        return any(w in query for w in safety_words)

    def _is_describe_query(self, query: str) -> bool:
        """Check if query wants description."""
        describe_words = ["what", "describe", "tell", "see", "look"]
        return any(w in query for w in describe_words)

    def _extract_target(self, query: str) -> str | None:
        """Extract what the user is looking for."""
        patterns = [
            r"(?:where|find|locate|get to|navigate to|go to)\s+(?:is\s+)?(?:the\s+|my\s+|a\s+)?(.+?)(?:\?|$)",
            r"how\s+(?:do\s+i\s+)?(?:get|go)\s+to\s+(?:the\s+|my\s+)?(.+?)(?:\?|$)",
            r"look\s+for\s+(?:the\s+|my\s+|a\s+)?(.+?)(?:\?|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                target = match.group(1).strip()
                # Clean up
                target = re.sub(r"\b(please|now|quickly)\b", "", target).strip()
                if target:
                    return target
        return None

    def _find_object(self, target: str, frame: NDArray[np.uint8]) -> str:
        """Find a specific object and give directions."""
        # Try Florence-2 grounding first (more accurate)
        if self.florence:
            results = self.florence.ground(frame, target)
            if results:
                result = results[0]
                position = self._get_position(result.center[0], frame.shape[1])
                direction = self._to_direction(position)

                # Get distance with qualitative description
                distance_str = ""
                if self.depth_estimator:
                    depth_map = self.depth_estimator.estimate(frame)
                    if depth_map is not None:
                        dist = self.depth_estimator.get_distance_in_region(depth_map, result.bbox)
                        if dist:
                            distance_str = f", {self._distance_to_words(dist)}"

                if len(results) > 1:
                    return f"Found {len(results)} {target}s. Nearest is {direction}{distance_str}."
                return f"Found {target} {direction}{distance_str}."

        # Try YOLO fallback
        if self.detector:
            detections = self.detector.detect(frame)
            matching = [d for d in detections if target.lower() in d.class_name.lower()]
            if matching:
                det = matching[0]
                position = self._get_position(det.center[0], frame.shape[1])
                direction = self._to_direction(position)

                distance_str = ""
                if self.depth_estimator:
                    depth_map = self.depth_estimator.estimate(frame)
                    if depth_map is not None:
                        dist = self.depth_estimator.get_distance_in_region(depth_map, det.bbox)
                        if dist:
                            distance_str = f", {self._distance_to_words(dist)}"

                return f"Found {target} {direction}{distance_str}."

        return f"Cannot find {target}. Try turning slowly to scan the room."

    def _describe_path(self, frame: NDArray[np.uint8]) -> str:
        """Describe the path ahead with obstacles."""
        obstacles = self._get_obstacles(frame)

        if not obstacles:
            return "Path ahead is clear. Safe to walk forward."

        # Group by position
        left = [o for o in obstacles if o.position == "left"]
        center = [o for o in obstacles if o.position == "center"]
        right = [o for o in obstacles if o.position == "right"]

        parts = []

        # Check center first (most important)
        if center:
            closest = min(center, key=lambda o: o.distance)
            distance_desc = self._distance_to_words(closest.distance)
            if closest.distance < 1.5:
                parts.append(f"BLOCKED ahead! {closest.detection.class_name} {distance_desc}.")
            else:
                parts.append(f"{closest.detection.class_name} ahead, {distance_desc}.")

        # Check sides
        if left:
            closest = min(left, key=lambda o: o.distance)
            distance_desc = self._distance_to_words(closest.distance)
            parts.append(f"{closest.detection.class_name} on left, {distance_desc}.")

        if right:
            closest = min(right, key=lambda o: o.distance)
            distance_desc = self._distance_to_words(closest.distance)
            parts.append(f"{closest.detection.class_name} on right, {distance_desc}.")

        # Give advice
        if center and not left:
            parts.append("Go left to avoid.")
        elif center and not right:
            parts.append("Go right to avoid.")
        elif not center:
            parts.append("Center path is clear.")

        return " ".join(parts)

    def _describe_scene(self, frame: NDArray[np.uint8]) -> str:
        """Describe the scene for navigation context."""
        parts = []

        # Get caption from Florence-2
        if self.florence:
            caption = self.florence.caption(frame, detailed=False)
            if caption:
                # Shorten for blind user
                if len(caption) > 100:
                    caption = caption[:100] + "..."
                parts.append(caption)

        # Add obstacle info with qualitative distances
        obstacles = self._get_obstacles(frame)
        if obstacles:
            # Top 3 closest
            closest = sorted(obstacles, key=lambda o: o.distance)[:3]
            obstacle_parts = []
            for obs in closest:
                distance_desc = self._distance_to_words(obs.distance)
                direction = self._to_direction(obs.position)
                obstacle_parts.append(f"{obs.detection.class_name} {direction}, {distance_desc}")
            parts.append("Nearby: " + "; ".join(obstacle_parts) + ".")
        else:
            parts.append("No obstacles detected.")

        return " ".join(parts) if parts else "Cannot analyze scene."

    def _get_obstacles(self, frame: NDArray[np.uint8]) -> list[Obstacle]:
        """Get obstacles with distances."""
        if self.detector is None or self.depth_estimator is None:
            return []

        detections = self.detector.detect(frame)
        if not detections:
            return []

        depth_map = self.depth_estimator.estimate(frame)
        if depth_map is None:
            return []

        return self.fusion_engine.process(detections, depth_map, frame.shape[1])

    def _get_position(self, x: float, frame_width: int) -> str:
        """Get position from x coordinate."""
        rel_x = x / frame_width
        if rel_x < 0.33:
            return "left"
        elif rel_x > 0.67:
            return "right"
        return "center"

    def _to_direction(self, position: str) -> str:
        """Convert position to simple body-relative direction.

        Uses clear directions a blind person understands:
        - "straight ahead" / "in front of you"
        - "on your left" / "to your left"
        - "on your right" / "to your right"
        """
        mapping = {
            "left": "on your left",
            "center": "straight ahead",
            "right": "on your right",
        }
        return mapping.get(position, "ahead")


if __name__ == "__main__":
    import cv2

    config = Config()
    nav = SmartNavigator(config)

    print("Loading models...")
    nav.load()

    # Test with captured image
    frame = cv2.imread("data/captures/capture_20260130_063141_00.jpg")
    if frame is None:
        print("No test image, creating dummy")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    print("\n" + "=" * 50)
    print("SMART NAVIGATOR TEST")
    print("=" * 50)

    # Test quick scan
    print("\n[Quick Scan]")
    print(nav.quick_scan(frame))

    # Test queries
    queries = [
        "What's in front of me?",
        "Is it safe to walk forward?",
        "Where is the door?",
        "How do I get to the TV?",
        "Find the chair",
    ]

    for q in queries:
        print(f"\n[Q] {q}")
        print(f"[A] {nav.navigate(q, frame)}")
