"""Smart query handler using Florence-2's full capabilities.

Instead of rigid pattern matching, this module uses Florence-2's
visual question answering and grounding capabilities to handle
any natural language query intelligently.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import re

import numpy as np
from numpy.typing import NDArray

from src.config import Config, FlorenceConfig
from src.depth import DepthEstimator
from src.detector import Detector
from src.florence import FlorenceModel
from src.fusion import FusionEngine


class SmartQueryHandler:
    """Intelligent query handler using VLM capabilities.

    This handler uses Florence-2's various tasks to answer queries:
    - <CAPTION> / <DETAILED_CAPTION> for scene descriptions
    - <OD> for object detection
    - <CAPTION_TO_PHRASE_GROUNDING> for finding specific objects
    - <OCR> for reading text

    It also combines with depth estimation for distance information.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.florence: FlorenceModel | None = None
        self.detector: Detector | None = None
        self.depth_estimator: DepthEstimator | None = None
        self.fusion_engine: FusionEngine | None = None
        self._loaded = False

    def load(self) -> bool:
        """Load required models."""
        success = True

        # Florence-2 is required for smart queries
        print("Loading Florence-2 for smart queries...")
        self.florence = FlorenceModel(self.config.florence)
        if not self.florence.load():
            print("Failed to load Florence-2")
            success = False

        # Depth for distance estimation
        if self.config.pipeline.use_depth:
            print("Loading Depth Estimator...")
            self.depth_estimator = DepthEstimator(self.config.depth)
            if not self.depth_estimator.load():
                print("Failed to load Depth Estimator")

        # YOLO for fast detection fallback
        if self.config.pipeline.use_yolo_world:
            print("Loading YOLO-World...")
            self.detector = Detector(self.config.detection)
            if not self.detector.load():
                print("Failed to load YOLO-World")

        # Fusion engine
        if self.depth_estimator:
            self.fusion_engine = FusionEngine(self.config.fusion)

        self._loaded = success
        return success

    def process(self, query: str, frame: NDArray[np.uint8]) -> str:
        """Process any natural language query intelligently.

        Args:
            query: User's natural language query (anything they say).
            frame: Current camera frame.

        Returns:
            Natural language response.
        """
        if not self._loaded or self.florence is None:
            return "System not ready. Please wait for models to load."

        query_lower = query.lower().strip()

        # Analyze query intent
        intent = self._analyze_intent(query_lower)

        if intent == "search":
            return self._handle_search(query_lower, frame)
        elif intent == "count":
            return self._handle_count(query_lower, frame)
        elif intent == "read":
            return self._handle_read(frame)
        elif intent == "direction":
            return self._handle_direction(query_lower, frame)
        else:
            # Default: comprehensive scene description
            return self._handle_describe(frame)

    def _analyze_intent(self, query: str) -> str:
        """Analyze query to determine intent."""
        # Search intent
        search_words = ["where", "find", "locate", "look for", "search", "spot"]
        if any(word in query for word in search_words):
            return "search"

        # Count intent
        count_words = ["how many", "count", "number of"]
        if any(word in query for word in count_words):
            return "count"

        # Read intent (OCR)
        read_words = ["read", "text", "sign", "label", "written", "says"]
        if any(word in query for word in read_words):
            return "read"

        # Direction intent
        direction_words = ["left", "right", "ahead", "behind", "next to", "near"]
        if any(word in query for word in direction_words):
            return "direction"

        # Default to describe
        return "describe"

    def _extract_target(self, query: str) -> str | None:
        """Extract target object from query."""
        # Patterns to extract object names
        patterns = [
            r"where (?:is |are )?(?:my |the |a |an )?(.+?)(?:\?|$)",
            r"find (?:my |the |a |an )?(.+?)(?:\?|$)",
            r"locate (?:my |the |a |an )?(.+?)(?:\?|$)",
            r"look for (?:my |the |a |an )?(.+?)(?:\?|$)",
            r"how many (.+?)(?:\?|$)",
            r"count (?:the |all )?(.+?)(?:\?|$)",
            r"is there (?:a |an |any )?(.+?)(?:\?|$)",
            r"do you see (?:a |an |any |my )?(.+?)(?:\?|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                target = match.group(1).strip()
                # Clean up common words
                target = re.sub(r"\b(here|there|around|nearby)\b", "", target).strip()
                if target:
                    return target

        return None

    def _handle_describe(self, frame: NDArray[np.uint8]) -> str:
        """Generate comprehensive scene description."""
        parts = []

        # Get detailed caption from Florence-2
        caption = self.florence.caption(frame, detailed=True)
        if caption:
            parts.append(caption)

        # Add obstacle information with qualitative distances
        if self.detector and self.depth_estimator:
            detections = self.detector.detect(frame)
            if detections:
                depth_map = self.depth_estimator.estimate(frame)
                if depth_map is not None and self.fusion_engine:
                    obstacles = self.fusion_engine.process(
                        detections, depth_map, frame.shape[1]
                    )
                    if obstacles:
                        # Top 3 most important obstacles
                        notable = sorted(obstacles, key=lambda o: o.priority, reverse=True)[:3]
                        obstacle_parts = []
                        for obs in notable:
                            pos = self._position_to_text(obs.position)
                            dist_desc = self._distance_to_words(obs.distance)
                            obstacle_parts.append(f"{obs.detection.class_name} {pos}, {dist_desc}")
                        if obstacle_parts:
                            parts.append("I can see: " + "; ".join(obstacle_parts))

        return " ".join(parts) if parts else "I cannot clearly see the surroundings."

    def _distance_to_words(self, distance: float) -> str:
        """Convert distance to natural language for blind users."""
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

    def _handle_search(self, query: str, frame: NDArray[np.uint8]) -> str:
        """Search for a specific object."""
        target = self._extract_target(query)
        if not target:
            # If no target found, describe what's visible
            return self._handle_describe(frame)

        # Use Florence-2 grounding to find the object
        results = self.florence.ground(frame, target)

        if not results:
            # Try with YOLO as fallback
            if self.detector:
                detections = self.detector.detect(frame)
                matching = [d for d in detections if target.lower() in d.class_name.lower()]
                if matching:
                    det = matching[0]
                    distance_str = ""
                    position = self._get_position(det.center[0], frame.shape[1])

                    if self.depth_estimator:
                        depth_map = self.depth_estimator.estimate(frame)
                        if depth_map is not None:
                            dist = self.depth_estimator.get_distance_in_region(depth_map, det.bbox)
                            if dist:
                                distance_str = f", {self._distance_to_words(dist)}"

                    return f"I found {target} {self._position_to_text(position)}{distance_str}."

            return f"I cannot find {target} in the current view. Try moving the camera or asking about something else."

        # Found with Florence-2
        result = results[0]
        position = self._get_position(result.center[0], frame.shape[1])
        distance_str = ""

        if self.depth_estimator:
            depth_map = self.depth_estimator.estimate(frame)
            if depth_map is not None:
                dist = self.depth_estimator.get_distance_in_region(depth_map, result.bbox)
                if dist:
                    distance_str = f", {self._distance_to_words(dist)}"

        if len(results) > 1:
            return f"I found {len(results)} {target}s. The closest one is {self._position_to_text(position)}{distance_str}."
        else:
            return f"I found {target} {self._position_to_text(position)}{distance_str}."

    def _handle_count(self, query: str, frame: NDArray[np.uint8]) -> str:
        """Count specific objects."""
        target = self._extract_target(query)
        if not target:
            return "What would you like me to count?"

        # Clean target (remove plural 's')
        target_singular = target.rstrip("s") if target.endswith("s") else target

        count = 0

        # Use Florence-2 grounding
        results = self.florence.ground(frame, target_singular)
        count = len(results)

        # Also check YOLO
        if count == 0 and self.detector:
            detections = self.detector.detect(frame)
            count = sum(1 for d in detections if target_singular.lower() in d.class_name.lower())

        if count == 0:
            return f"I don't see any {target} in the current view."
        elif count == 1:
            return f"I can see 1 {target_singular}."
        else:
            return f"I can see {count} {target}."

    def _handle_read(self, frame: NDArray[np.uint8]) -> str:
        """Read text in the image using OCR."""
        # Use Florence-2 OCR task
        try:
            text = self.florence.ocr(frame)
            if text:
                return f"I can read: {text}"
            else:
                return "I cannot see any readable text in the current view."
        except Exception:
            return "Text reading is not available."

    def _handle_direction(self, query: str, frame: NDArray[np.uint8]) -> str:
        """Answer questions about directions/positions."""
        # Extract what the user is asking about
        target = self._extract_target(query)

        if target:
            return self._handle_search(query, frame)

        # General direction query - describe what's in each direction
        return self._handle_describe(frame)

    def _get_position(self, x: float, frame_width: int) -> str:
        """Determine position from x coordinate."""
        relative_x = x / frame_width
        if relative_x < 0.33:
            return "left"
        elif relative_x > 0.67:
            return "right"
        else:
            return "center"

    def _position_to_text(self, position: str) -> str:
        """Convert position to natural text."""
        mapping = {
            "left": "to your left",
            "center": "directly ahead",
            "right": "to your right",
        }
        return mapping.get(position, position)


# Convenience function
def create_smart_handler(config: Config | None = None) -> SmartQueryHandler:
    """Create and load a smart query handler."""
    if config is None:
        config = Config()
        config.pipeline.use_florence = True
        config.pipeline.use_depth = True
        config.pipeline.use_yolo_world = True

    handler = SmartQueryHandler(config)
    handler.load()
    return handler


if __name__ == "__main__":
    import cv2

    # Test the smart query handler
    config = Config()
    handler = SmartQueryHandler(config)

    print("Loading models...")
    handler.load()

    # Test with sample image
    test_frame = cv2.imread("data/captures/capture_20260130_063141_00.jpg")
    if test_frame is None:
        print("Creating test frame...")
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:] = (200, 200, 200)

    test_queries = [
        "What do you see?",
        "What is in front of me?",
        "Where is the door?",
        "Is there a chair?",
        "How many people?",
        "Help me navigate",
        "What are you looking at?",
        "Find my phone",
        "Describe the room",
        "What's happening?",
    ]

    print("\n" + "=" * 60)
    print("SMART QUERY HANDLER TEST")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQ: {query}")
        response = handler.process(query, test_frame)
        print(f"A: {response}")
