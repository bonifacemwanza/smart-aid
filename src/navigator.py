"""Smart navigation guidance using local LLM (Ollama).

This module uses a local vision-language model to provide actual
navigation guidance - not just descriptions, but step-by-step
instructions to help blind users navigate to objects.

Requires: Ollama running locally with a vision model (llava, llama3.2-vision)
Install: brew install ollama && ollama pull llava
"""

import base64
import json
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
import requests
from numpy.typing import NDArray

from src.config import Config
from src.depth import DepthEstimator
from src.detector import Detector
from src.fusion import FusionEngine


class NavigationAssistant:
    """Navigation assistant using local LLM for intelligent guidance.

    This assistant provides:
    1. Step-by-step navigation instructions
    2. Real-time obstacle warnings
    3. Contextual understanding of user intent
    4. Spatial reasoning about the environment
    """

    OLLAMA_URL = "http://localhost:11434/api/generate"

    # System prompt for navigation assistance
    SYSTEM_PROMPT = """You are a navigation assistant for a blind person. Your job is to help them navigate safely.

IMPORTANT RULES:
1. Be CONCISE - blind users need quick, clear instructions
2. Use CLOCK DIRECTIONS (12 o'clock = straight ahead, 3 o'clock = right, 9 o'clock = left)
3. Give DISTANCES in steps (1 step ≈ 0.7 meters)
4. WARN about obstacles first, then give directions
5. Be SPECIFIC about what you see and where

FORMAT YOUR RESPONSES LIKE THIS:
- Start with any WARNINGS about close obstacles
- Then give the navigation instruction
- Keep it under 3 sentences

Example good responses:
- "Stop! Chair 2 steps ahead at 12 o'clock. Step left to go around it."
- "TV is at 2 o'clock, about 8 steps away. Walk forward, slight right."
- "Clear path ahead. Door is at 11 o'clock, 5 steps away."

Example bad responses (too long):
- "I can see a living room with various furniture including a sofa, coffee table, and TV mounted on the wall..." (TOO DESCRIPTIVE)
"""

    def __init__(self, config: Config, model: str = "llava") -> None:
        """Initialize the navigation assistant.

        Args:
            config: Application configuration.
            model: Ollama model to use (llava, llama3.2-vision, etc.)
        """
        self.config = config
        self.model = model

        # Components for obstacle detection
        self.detector: Detector | None = None
        self.depth_estimator: DepthEstimator | None = None
        self.fusion_engine: FusionEngine | None = None

        self._loaded = False
        self._ollama_available = False

    def load(self) -> bool:
        """Load required components."""
        success = True

        # Check Ollama availability
        self._ollama_available = self._check_ollama()
        if not self._ollama_available:
            print("WARNING: Ollama not available. Run: ollama serve")
            print("         Then: ollama pull llava")

        # Load YOLO for fast obstacle detection
        if self.config.pipeline.use_yolo_world:
            print("Loading YOLO-World for obstacle detection...")
            self.detector = Detector(self.config.detection)
            if not self.detector.load():
                print("Failed to load YOLO-World")
                success = False

        # Load depth estimator
        if self.config.pipeline.use_depth:
            print("Loading Depth Estimator...")
            self.depth_estimator = DepthEstimator(self.config.depth)
            if not self.depth_estimator.load():
                print("Failed to load Depth Estimator")

        # Fusion engine
        if self.depth_estimator:
            self.fusion_engine = FusionEngine(self.config.fusion)

        self._loaded = success
        return success

    def _check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _frame_to_base64(self, frame: NDArray[np.uint8]) -> str:
        """Convert frame to base64 for Ollama."""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def _get_obstacle_context(self, frame: NDArray[np.uint8]) -> str:
        """Get obstacle information as context for the LLM."""
        if self.detector is None or self.depth_estimator is None:
            return ""

        detections = self.detector.detect(frame)
        if not detections:
            return "No obstacles detected nearby."

        depth_map = self.depth_estimator.estimate(frame)
        if depth_map is None:
            return ""

        obstacles = self.fusion_engine.process(detections, depth_map, frame.shape[1])
        if not obstacles:
            return "Path appears clear."

        # Build context string
        context_parts = ["DETECTED OBSTACLES:"]
        for obs in sorted(obstacles, key=lambda o: o.distance)[:5]:
            steps = int(obs.distance / 0.7)  # Convert meters to steps
            clock = self._position_to_clock(obs.position)
            context_parts.append(f"- {obs.detection.class_name}: {steps} steps at {clock}")

        return "\n".join(context_parts)

    def _position_to_clock(self, position: str) -> str:
        """Convert position to clock direction."""
        mapping = {
            "left": "9 o'clock",
            "center": "12 o'clock",
            "right": "3 o'clock",
        }
        return mapping.get(position, "12 o'clock")

    def navigate(self, query: str, frame: NDArray[np.uint8]) -> str:
        """Process a navigation query.

        Args:
            query: User's question (e.g., "How do I get to the TV?")
            frame: Current camera frame.

        Returns:
            Navigation instructions.
        """
        if not self._ollama_available:
            return self._fallback_navigate(query, frame)

        # Get obstacle context
        obstacle_context = self._get_obstacle_context(frame)

        # Build prompt
        prompt = f"""{self.SYSTEM_PROMPT}

{obstacle_context}

USER'S QUESTION: {query}

Give a brief, helpful navigation response:"""

        # Convert frame to base64
        image_b64 = self._frame_to_base64(frame)

        try:
            response = requests.post(
                self.OLLAMA_URL,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 150,  # Keep responses short
                    }
                },
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return self._fallback_navigate(query, frame)

        except Exception as e:
            print(f"Ollama error: {e}")
            return self._fallback_navigate(query, frame)

    def _fallback_navigate(self, query: str, frame: NDArray[np.uint8]) -> str:
        """Fallback navigation when Ollama is not available."""
        # Get obstacle info
        obstacle_context = self._get_obstacle_context(frame)

        # Simple rule-based response
        query_lower = query.lower()

        # Check for obstacles first
        warnings = []
        if self.detector and self.depth_estimator:
            detections = self.detector.detect(frame)
            depth_map = self.depth_estimator.estimate(frame)
            if depth_map is not None and detections:
                obstacles = self.fusion_engine.process(detections, depth_map, frame.shape[1])
                for obs in obstacles:
                    if obs.distance < 1.5:  # Close obstacle
                        steps = max(1, int(obs.distance / 0.7))
                        clock = self._position_to_clock(obs.position)
                        warnings.append(f"Warning: {obs.detection.class_name} {steps} steps at {clock}")

        # Build response
        response_parts = []

        if warnings:
            response_parts.extend(warnings[:2])  # Max 2 warnings

        # Extract target from query
        target = self._extract_target(query_lower)
        if target:
            response_parts.append(f"Looking for {target}. Use voice command 'where is {target}' for location.")
        else:
            response_parts.append("Path ahead. Tell me where you want to go.")

        return " ".join(response_parts)

    def _extract_target(self, query: str) -> str | None:
        """Extract navigation target from query."""
        import re
        patterns = [
            r"(?:navigate|go|get|walk|find).+?(?:to|the)\s+(.+?)(?:\?|$)",
            r"where.+?(?:is|are)\s+(?:the\s+)?(.+?)(?:\?|$)",
            r"(?:find|locate)\s+(?:the\s+)?(.+?)(?:\?|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def get_quick_scan(self, frame: NDArray[np.uint8]) -> str:
        """Get a quick safety scan of the environment.

        Returns short warnings about immediate obstacles.
        """
        if self.detector is None or self.depth_estimator is None:
            return "System not ready."

        detections = self.detector.detect(frame)
        if not detections:
            return "Clear. No obstacles detected."

        depth_map = self.depth_estimator.estimate(frame)
        if depth_map is None:
            return "Unable to estimate distances."

        obstacles = self.fusion_engine.process(detections, depth_map, frame.shape[1])

        # Check for immediate dangers (< 1 meter)
        dangers = [o for o in obstacles if o.distance < 1.0]
        warnings = [o for o in obstacles if 1.0 <= o.distance < 2.0]

        if dangers:
            obs = dangers[0]
            steps = max(1, int(obs.distance / 0.7))
            clock = self._position_to_clock(obs.position)
            return f"STOP! {obs.detection.class_name} {steps} step{'s' if steps > 1 else ''} at {clock}!"

        if warnings:
            obs = warnings[0]
            steps = int(obs.distance / 0.7)
            clock = self._position_to_clock(obs.position)
            return f"Caution: {obs.detection.class_name} {steps} steps at {clock}."

        return "Clear path ahead."


def check_ollama_models() -> list[str]:
    """Check which models are available in Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return models
    except Exception:
        pass
    return []


if __name__ == "__main__":
    print("=" * 50)
    print("Navigation Assistant Test")
    print("=" * 50)

    # Check Ollama
    print("\nChecking Ollama...")
    models = check_ollama_models()
    if models:
        print(f"Available models: {models}")
        vision_models = [m for m in models if any(v in m.lower() for v in ["llava", "vision", "llama3.2"])]
        if vision_models:
            print(f"Vision models: {vision_models}")
        else:
            print("No vision models found. Run: ollama pull llava")
    else:
        print("Ollama not running. Start it with: ollama serve")

    # Test with sample image
    config = Config()
    assistant = NavigationAssistant(config)
    print("\nLoading models...")
    assistant.load()

    # Load test image
    test_frame = cv2.imread("data/captures/capture_20260130_063141_00.jpg")
    if test_frame is None:
        print("Creating test frame...")
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:] = (200, 200, 200)

    # Test queries
    test_queries = [
        "What's in front of me?",
        "How do I get to the door?",
        "Is it safe to walk forward?",
        "Where is the TV?",
    ]

    print("\n" + "-" * 50)
    for query in test_queries:
        print(f"\nQ: {query}")
        response = assistant.navigate(query, test_frame)
        print(f"A: {response}")

    # Test quick scan
    print("\n" + "-" * 50)
    print("\nQuick Scan:")
    print(assistant.get_quick_scan(test_frame))
