"""Main Smart Aid pipeline with configurable features."""

import numpy as np
from numpy.typing import NDArray

from src.config import Config
from src.depth import DepthEstimator
from src.detector import Detection, Detector
from src.feedback import FeedbackManager
from src.florence import FlorenceModel, GroundingResult
from src.fusion import FusionEngine, Obstacle
from src.intent import Intent, IntentParser, IntentType
from src.response import ResponseBuilder, SearchResult
from src.voice import VoiceInput


class SmartAidPipeline:
    """Main pipeline orchestrating all Smart Aid components.

    Features can be toggled on/off via configuration:
    - YOLO-World detection (fast object detection)
    - Florence-2 (visual grounding + captioning)
    - Depth estimation
    - Voice input (Whisper)
    - Voice output (TTS)
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        # Components (initialized in load())
        self.detector: Detector | None = None
        self.florence: FlorenceModel | None = None
        self.depth_estimator: DepthEstimator | None = None
        self.fusion_engine: FusionEngine | None = None
        self.voice_input: VoiceInput | None = None
        self.feedback: FeedbackManager | None = None

        # Helpers
        self.intent_parser = IntentParser()
        self.response_builder = ResponseBuilder()

        # State
        self._loaded = False

    def load(self) -> bool:
        """Load all enabled components.

        Returns:
            True if all enabled components loaded successfully.
        """
        success = True
        pipeline_config = self.config.pipeline

        # Load YOLO-World detector
        if pipeline_config.use_yolo_world:
            print("Loading YOLO-World detector...")
            self.detector = Detector(self.config.detection)
            if not self.detector.load():
                print("Failed to load YOLO-World detector")
                success = False

        # Load Florence-2
        if pipeline_config.use_florence:
            print("Loading Florence-2 model...")
            self.florence = FlorenceModel(self.config.florence)
            if not self.florence.load():
                print("Failed to load Florence-2 model")
                success = False

        # Load Depth Estimator
        if pipeline_config.use_depth:
            print("Loading Depth Estimator...")
            self.depth_estimator = DepthEstimator(self.config.depth)
            if not self.depth_estimator.load():
                print("Failed to load Depth Estimator")
                success = False

        # Load Voice Input
        if pipeline_config.use_voice_input:
            print("Loading Voice Input (Whisper)...")
            self.voice_input = VoiceInput(self.config.voice)
            if not self.voice_input.load():
                print("Failed to load Whisper model")
                success = False

        # Load Feedback (TTS)
        if pipeline_config.use_voice_output:
            print("Loading Feedback (TTS)...")
            self.feedback = FeedbackManager(self.config.feedback)

        # Fusion engine (always available if depth is loaded)
        if self.depth_estimator is not None:
            self.fusion_engine = FusionEngine(self.config.fusion)

        self._loaded = success
        print(f"Pipeline loaded: {'success' if success else 'with errors'}")
        return success

    def process_query(self, query: str, frame: NDArray[np.uint8]) -> str:
        """Process a voice/text query and return a response.

        Args:
            query: User's natural language query.
            frame: Current camera frame.

        Returns:
            Natural language response string.
        """
        # Parse intent
        intent = self.intent_parser.parse(query)

        # Route to appropriate handler
        if intent.type == IntentType.DESCRIBE:
            return self._handle_describe(frame)
        elif intent.type == IntentType.SEARCH:
            return self._handle_search(frame, intent.target)
        elif intent.type == IntentType.DETECT:
            return self._handle_detect(frame)
        elif intent.type == IntentType.COUNT:
            return self._handle_count(frame, intent.target)
        else:
            # For unknown queries, default to scene description
            return self._handle_describe(frame)

    def _handle_describe(self, frame: NDArray[np.uint8]) -> str:
        """Handle scene description query."""
        caption = ""
        obstacles = None

        # Get caption from Florence-2
        if self.florence is not None:
            caption = self.florence.caption(frame, detailed=True)

        # Get obstacles if depth is available
        if self.detector is not None and self.depth_estimator is not None:
            detections = self.detector.detect(frame)
            depth_map = self.depth_estimator.estimate(frame)
            if depth_map is not None and detections:
                obstacles = self.fusion_engine.process(
                    detections, depth_map, frame.shape[1]
                )

        return self.response_builder.build_scene_description(caption, obstacles)

    def _handle_search(self, frame: NDArray[np.uint8], target: str | None) -> str:
        """Handle object search query."""
        if not target:
            return "What would you like me to find?"

        results: list[GroundingResult] = []
        depth_map = None

        # Use Florence-2 for grounding
        if self.florence is not None:
            results = self.florence.ground(frame, target)

        # Fallback to YOLO-World if Florence didn't find anything
        if not results and self.detector is not None:
            detections = self.detector.detect(frame)
            # Filter for target
            matching = [d for d in detections if target.lower() in d.class_name.lower()]
            for det in matching:
                results.append(
                    GroundingResult(
                        phrase=det.class_name,
                        bbox=det.bbox,
                        center=det.center,
                    )
                )

        # Get depth for distance
        if self.depth_estimator is not None:
            depth_map = self.depth_estimator.estimate(frame)

        if not results:
            return self.response_builder.build_search_response(
                SearchResult(found=False, target=target)
            )

        # Get distance and position for closest result
        closest = results[0]
        distance = None
        position = None

        if depth_map is not None:
            distance = self.depth_estimator.get_distance_in_region(depth_map, closest.bbox)
            # Determine position
            frame_width = frame.shape[1]
            relative_x = closest.center[0] / frame_width
            if relative_x < 0.33:
                position = "left"
            elif relative_x > 0.67:
                position = "right"
            else:
                position = "center"

        return self.response_builder.build_search_response(
            SearchResult(
                found=True,
                target=target,
                distance=distance,
                position=position,
                count=len(results),
            )
        )

    def _handle_detect(self, frame: NDArray[np.uint8]) -> str:
        """Handle object detection query."""
        all_detections: list[Detection | GroundingResult] = []

        # Use YOLO-World for detection
        if self.detector is not None:
            detections = self.detector.detect(frame)
            all_detections.extend(detections)

        # Or use Florence-2
        elif self.florence is not None:
            results = self.florence.detect(frame)
            all_detections.extend(results)

        return self.response_builder.build_detection_response(all_detections)

    def _handle_count(self, frame: NDArray[np.uint8], target: str | None) -> str:
        """Handle count query."""
        if not target:
            return "What would you like me to count?"

        # Normalize target (remove trailing 's' for plurals)
        target_singular = target.lower().rstrip("s")
        count = 0

        # Use YOLO-World
        if self.detector is not None:
            detections = self.detector.detect(frame)
            count = sum(
                1 for d in detections
                if target_singular in d.class_name.lower() or target.lower() in d.class_name.lower()
            )

        # Also use Florence-2 grounding if YOLO didn't find anything
        if count == 0 and self.florence is not None:
            results = self.florence.ground(frame, target_singular)
            count = len(results)

        return self.response_builder.build_count_response(target, count)

    def run_alerts(self, frame: NDArray[np.uint8]) -> list[Obstacle]:
        """Run passive alert mode (existing behavior).

        Args:
            frame: Current camera frame.

        Returns:
            List of detected obstacles sorted by priority.
        """
        if self.detector is None or self.depth_estimator is None:
            return []

        # Detect objects
        detections = self.detector.detect(frame)
        if not detections:
            return []

        # Estimate depth
        depth_map = self.depth_estimator.estimate(frame)
        if depth_map is None:
            return []

        # Fuse detection + depth
        obstacles = self.fusion_engine.process(detections, depth_map, frame.shape[1])

        # Speak alert for highest priority obstacle
        if obstacles and self.feedback is not None:
            top_obstacle = obstacles[0]
            if top_obstacle.distance < self.config.fusion.warning_zone:
                alert_text = self.response_builder.build_alert_response(top_obstacle)
                self.feedback.speak(alert_text)

        return obstacles

    def transcribe_audio(self, audio: NDArray[np.float32]) -> str | None:
        """Transcribe audio to text using Whisper.

        Args:
            audio: Audio data as float32 numpy array.

        Returns:
            Transcribed text or None.
        """
        if self.voice_input is None:
            return None
        return self.voice_input.transcribe(audio)

    def speak(self, text: str) -> None:
        """Speak text using TTS.

        Args:
            text: Text to speak.
        """
        if self.feedback is not None:
            self.feedback.speak(text)

    # Convenience methods for accessing individual components

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Run YOLO-World detection."""
        if self.detector is None:
            return []
        return self.detector.detect(frame)

    def caption(self, frame: NDArray[np.uint8]) -> str:
        """Get scene caption from Florence-2."""
        if self.florence is None:
            return ""
        return self.florence.caption(frame)

    def ground(self, frame: NDArray[np.uint8], phrase: str) -> list[GroundingResult]:
        """Visual grounding with Florence-2."""
        if self.florence is None:
            return []
        return self.florence.ground(frame, phrase)

    def estimate_depth(self, frame: NDArray[np.uint8]) -> NDArray[np.float32] | None:
        """Estimate depth map."""
        if self.depth_estimator is None:
            return None
        return self.depth_estimator.estimate(frame)


if __name__ == "__main__":
    import cv2
    import sys

    # Load config
    config = Config()

    # For testing, disable features to speed up loading
    config.pipeline.use_voice_input = False  # Skip Whisper
    config.pipeline.use_florence = True
    config.pipeline.use_yolo_world = True
    config.pipeline.use_depth = True

    # Create pipeline
    pipeline = SmartAidPipeline(config)

    print("Loading pipeline...")
    if not pipeline.load():
        print("Pipeline load failed")
        sys.exit(1)

    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (200, 200, 200)
    cv2.rectangle(test_image, (100, 100), (250, 350), (139, 69, 19), -1)
    cv2.putText(
        test_image,
        "Test Object",
        (110, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )

    # Test queries
    test_queries = [
        "What is in front of me?",
        "Where is the door?",
        "What objects are here?",
        "How many chairs?",
    ]

    print("\n=== Testing Queries ===")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        response = pipeline.process_query(query, test_image)
        print(f"Response: {response}")

    # Test alert mode
    print("\n=== Testing Alert Mode ===")
    obstacles = pipeline.run_alerts(test_image)
    print(f"Detected {len(obstacles)} obstacles")
    for obs in obstacles:
        print(f"  - {obs.to_alert_text()}")
