"""Real-time detection UI with video streaming and highlighted detections."""

import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

# Add project root to path for direct execution
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import Config
from src.depth import DepthEstimator
from src.detector import Detection, Detector
from src.florence import FlorenceModel, GroundingResult
from src.fusion import FusionEngine, Obstacle
from src.utils import FPSCounter


class DisplayMode(Enum):
    """Display modes for the UI."""

    DETECTION = "detection"
    DEPTH = "depth"
    FUSION = "fusion"
    SPLIT = "split"


@dataclass
class UIState:
    """Current state of the UI."""

    mode: DisplayMode = DisplayMode.FUSION
    show_detections: bool = True
    show_depth: bool = True
    show_labels: bool = True
    show_distances: bool = True
    show_fps: bool = True
    show_help: bool = False
    paused: bool = False
    last_query: str = ""
    last_response: str = ""


class SmartAidUI:
    """Real-time detection UI with video streaming."""

    # Colors (BGR)
    COLOR_DANGER = (0, 0, 255)  # Red
    COLOR_WARNING = (0, 165, 255)  # Orange
    COLOR_SAFE = (0, 255, 0)  # Green
    COLOR_INFO = (255, 255, 0)  # Cyan
    COLOR_TEXT = (255, 255, 255)  # White
    COLOR_BG = (40, 40, 40)  # Dark gray

    def __init__(self, config: Config) -> None:
        self.config = config
        self.state = UIState()
        self.fps_counter = FPSCounter()

        # Components
        self.detector: Detector | None = None
        self.depth_estimator: DepthEstimator | None = None
        self.florence: FlorenceModel | None = None
        self.fusion_engine: FusionEngine | None = None

        # Cached results
        self._detections: list[Detection] = []
        self._depth_map: NDArray[np.float32] | None = None
        self._obstacles: list[Obstacle] = []
        self._florence_results: list[GroundingResult] = []

        # Window name
        self.window_name = "Smart Aid - Real-time Detection"

    def load(self) -> bool:
        """Load all components."""
        success = True

        # Load detector
        if self.config.pipeline.use_yolo_world:
            print("Loading YOLO-World detector...")
            self.detector = Detector(self.config.detection)
            if not self.detector.load():
                print("Failed to load detector")
                success = False

        # Load depth estimator
        if self.config.pipeline.use_depth:
            print("Loading Depth Estimator...")
            self.depth_estimator = DepthEstimator(self.config.depth)
            if not self.depth_estimator.load():
                print("Failed to load depth estimator")
                success = False

        # Load Florence-2 (optional for queries)
        if self.config.pipeline.use_florence:
            print("Loading Florence-2...")
            self.florence = FlorenceModel(self.config.florence)
            if not self.florence.load():
                print("Failed to load Florence-2")
                # Don't fail - Florence is optional

        # Fusion engine
        if self.depth_estimator is not None:
            self.fusion_engine = FusionEngine(self.config.fusion)

        return success

    def process_frame(self, frame: NDArray[np.uint8]) -> None:
        """Process a frame and update cached results."""
        if self.state.paused:
            return

        # Run detection
        if self.detector is not None and self.state.show_detections:
            self._detections = self.detector.detect(frame)

        # Run depth estimation
        if self.depth_estimator is not None and self.state.show_depth:
            self._depth_map = self.depth_estimator.estimate(frame)

        # Run fusion
        if (
            self.fusion_engine is not None
            and self._detections
            and self._depth_map is not None
        ):
            self._obstacles = self.fusion_engine.process(
                self._detections, self._depth_map, frame.shape[1]
            )
        else:
            self._obstacles = []

    def render(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Render the UI overlay on the frame."""
        self.fps_counter.update()

        if self.state.mode == DisplayMode.SPLIT:
            return self._render_split_view(frame)
        elif self.state.mode == DisplayMode.DEPTH:
            return self._render_depth_view(frame)
        else:
            return self._render_detection_view(frame)

    def _render_detection_view(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Render detection/fusion view."""
        output = frame.copy()

        # Draw detections/obstacles
        if self.state.show_detections:
            if self._obstacles:
                output = self._draw_obstacles(output, self._obstacles)
            elif self._detections:
                output = self._draw_detections(output, self._detections)

        # Draw Florence results if any
        if self._florence_results:
            output = self._draw_grounding_results(output, self._florence_results)

        # Draw overlays
        output = self._draw_status_bar(output)

        if self.state.show_help:
            output = self._draw_help(output)

        if self.state.last_response:
            output = self._draw_response(output)

        return output

    def _render_depth_view(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Render depth-only view."""
        if self._depth_map is not None:
            output = self.depth_estimator.colorize(self._depth_map)
            # Resize to match frame if needed
            if output.shape[:2] != frame.shape[:2]:
                output = cv2.resize(output, (frame.shape[1], frame.shape[0]))
        else:
            output = frame.copy()
            cv2.putText(
                output,
                "Depth estimation disabled or unavailable",
                (50, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                self.COLOR_TEXT,
                2,
            )

        output = self._draw_status_bar(output)
        return output

    def _render_split_view(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Render split view with detection and depth side by side."""
        h, w = frame.shape[:2]
        half_w = w // 2

        # Left: Detection view
        left = frame.copy()
        if self._obstacles:
            left = self._draw_obstacles(left, self._obstacles)
        elif self._detections:
            left = self._draw_detections(left, self._detections)

        # Right: Depth view
        if self._depth_map is not None:
            right = self.depth_estimator.colorize(self._depth_map)
            if right.shape[:2] != frame.shape[:2]:
                right = cv2.resize(right, (w, h))
        else:
            right = np.zeros_like(frame)
            cv2.putText(
                right,
                "No depth",
                (w // 4, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                self.COLOR_TEXT,
                2,
            )

        # Resize both to half width
        left_resized = cv2.resize(left, (half_w, h))
        right_resized = cv2.resize(right, (half_w, h))

        # Combine
        output = np.hstack([left_resized, right_resized])

        # Add labels
        cv2.putText(output, "Detection", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_INFO, 2)
        cv2.putText(output, "Depth", (half_w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_INFO, 2)

        # Draw divider line
        cv2.line(output, (half_w, 0), (half_w, h), self.COLOR_INFO, 2)

        output = self._draw_status_bar(output)
        return output

    def _draw_obstacles(
        self, frame: NDArray[np.uint8], obstacles: list[Obstacle]
    ) -> NDArray[np.uint8]:
        """Draw obstacles with distance-based coloring."""
        for obs in obstacles:
            det = obs.detection

            # Color based on distance
            if obs.distance < self.config.fusion.danger_zone:
                color = self.COLOR_DANGER
            elif obs.distance < self.config.fusion.warning_zone:
                color = self.COLOR_WARNING
            else:
                color = self.COLOR_SAFE

            # Draw bounding box
            thickness = 3 if obs.distance < self.config.fusion.danger_zone else 2
            cv2.rectangle(
                frame, (det.x1, det.y1), (det.x2, det.y2), color, thickness
            )

            # Draw label with distance
            if self.state.show_labels:
                if self.state.show_distances:
                    label = f"{obs.class_name}: {obs.distance:.1f}m ({obs.position})"
                else:
                    label = f"{obs.class_name}"

                self._draw_label(frame, label, det.x1, det.y1, color)

            # Draw center point
            cv2.circle(frame, det.center, 5, color, -1)

            # Draw direction indicator for close objects
            if obs.distance < self.config.fusion.warning_zone:
                self._draw_direction_indicator(frame, obs)

        return frame

    def _draw_detections(
        self, frame: NDArray[np.uint8], detections: list[Detection]
    ) -> NDArray[np.uint8]:
        """Draw detections without depth info."""
        for det in detections:
            # Color based on confidence
            if det.confidence >= 0.7:
                color = self.COLOR_SAFE
            elif det.confidence >= 0.5:
                color = self.COLOR_WARNING
            else:
                color = self.COLOR_DANGER

            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)

            if self.state.show_labels:
                label = f"{det.class_name}: {det.confidence:.0%}"
                self._draw_label(frame, label, det.x1, det.y1, color)

            cv2.circle(frame, det.center, 4, color, -1)

        return frame

    def _draw_grounding_results(
        self, frame: NDArray[np.uint8], results: list[GroundingResult]
    ) -> NDArray[np.uint8]:
        """Draw Florence-2 grounding results."""
        for result in results:
            color = self.COLOR_INFO

            cv2.rectangle(
                frame, (result.x1, result.y1), (result.x2, result.y2), color, 3
            )

            if self.state.show_labels:
                label = f"[FOUND] {result.phrase}"
                self._draw_label(frame, label, result.x1, result.y1, color)

            cv2.circle(frame, result.center, 6, color, -1)

        return frame

    def _draw_label(
        self,
        frame: NDArray[np.uint8],
        label: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a label with background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (label_w, label_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Draw background
        cv2.rectangle(
            frame,
            (x, y - label_h - baseline - 5),
            (x + label_w + 5, y),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (x + 2, y - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    def _draw_direction_indicator(
        self, frame: NDArray[np.uint8], obstacle: Obstacle
    ) -> None:
        """Draw directional warning indicator."""
        h, w = frame.shape[:2]
        center_x = w // 2

        # Arrow parameters
        arrow_y = h - 50
        arrow_length = 60

        if obstacle.position == "left":
            start = (center_x, arrow_y)
            end = (center_x - arrow_length, arrow_y)
            cv2.arrowedLine(frame, start, end, self.COLOR_DANGER, 3, tipLength=0.3)
            cv2.putText(
                frame, "LEFT", (center_x - arrow_length - 50, arrow_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_DANGER, 2
            )
        elif obstacle.position == "right":
            start = (center_x, arrow_y)
            end = (center_x + arrow_length, arrow_y)
            cv2.arrowedLine(frame, start, end, self.COLOR_DANGER, 3, tipLength=0.3)
            cv2.putText(
                frame, "RIGHT", (center_x + arrow_length + 10, arrow_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_DANGER, 2
            )
        else:
            # Center - warning triangle
            pts = np.array([
                [center_x, arrow_y - 30],
                [center_x - 25, arrow_y + 10],
                [center_x + 25, arrow_y + 10],
            ], np.int32)
            cv2.fillPoly(frame, [pts], self.COLOR_DANGER)
            cv2.putText(
                frame, "!", (center_x - 5, arrow_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

    def _draw_status_bar(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Draw status bar at the top."""
        h, w = frame.shape[:2]
        bar_height = 35

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Status text
        status_parts = []

        if self.state.show_fps:
            # FPSCounter.update() returns current FPS
            fps = self.fps_counter.update()
            status_parts.append(f"FPS: {fps:.1f}")

        status_parts.append(f"Mode: {self.state.mode.value}")

        if self._detections:
            status_parts.append(f"Objects: {len(self._detections)}")

        if self._obstacles:
            danger_count = sum(1 for o in self._obstacles if o.distance < self.config.fusion.danger_zone)
            if danger_count > 0:
                status_parts.append(f"DANGER: {danger_count}")

        status_text = " | ".join(status_parts)
        cv2.putText(
            frame, status_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 1
        )

        # Help hint
        cv2.putText(
            frame, "Press 'h' for help", (w - 150, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_INFO, 1
        )

        return frame

    def _draw_help(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Draw help overlay."""
        h, w = frame.shape[:2]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (w - 50, h - 50), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

        # Help text
        help_lines = [
            "KEYBOARD CONTROLS",
            "",
            "h - Toggle this help",
            "q / ESC - Quit",
            "SPACE - Pause/Resume",
            "",
            "d - Toggle detections",
            "z - Toggle depth display",
            "l - Toggle labels",
            "f - Toggle FPS",
            "",
            "1 - Detection mode",
            "2 - Depth mode",
            "3 - Fusion mode",
            "4 - Split view",
            "",
            "s - Search (type object name)",
            "c - Caption scene",
        ]

        y = 80
        for line in help_lines:
            if line == "KEYBOARD CONTROLS":
                cv2.putText(
                    frame, line, (70, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_INFO, 2
                )
            else:
                cv2.putText(
                    frame, line, (70, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1
                )
            y += 25

        return frame

    def _draw_response(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Draw query response at the bottom."""
        h, w = frame.shape[:2]

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Query
        if self.state.last_query:
            cv2.putText(
                frame, f"> {self.state.last_query}", (10, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_INFO, 1
            )

        # Response (truncate if too long)
        response = self.state.last_response
        if len(response) > 80:
            response = response[:77] + "..."

        cv2.putText(
            frame, response, (10, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1
        )

        return frame

    def handle_key(self, key: int) -> bool:
        """Handle keyboard input. Returns False if should quit."""
        if key == ord("q") or key == 27:  # q or ESC
            return False

        elif key == ord("h"):
            self.state.show_help = not self.state.show_help

        elif key == ord(" "):  # Space
            self.state.paused = not self.state.paused

        elif key == ord("d"):
            self.state.show_detections = not self.state.show_detections

        elif key == ord("z"):
            self.state.show_depth = not self.state.show_depth

        elif key == ord("l"):
            self.state.show_labels = not self.state.show_labels

        elif key == ord("f"):
            self.state.show_fps = not self.state.show_fps

        elif key == ord("1"):
            self.state.mode = DisplayMode.DETECTION

        elif key == ord("2"):
            self.state.mode = DisplayMode.DEPTH

        elif key == ord("3"):
            self.state.mode = DisplayMode.FUSION

        elif key == ord("4"):
            self.state.mode = DisplayMode.SPLIT

        return True

    def search(self, frame: NDArray[np.uint8], query: str) -> str:
        """Search for an object using Florence-2."""
        self.state.last_query = f"Where is {query}?"

        if self.florence is None:
            self.state.last_response = "Florence-2 not loaded"
            return self.state.last_response

        # Run grounding
        self._florence_results = self.florence.ground(frame, query)

        if self._florence_results:
            result = self._florence_results[0]

            # Get distance if available
            distance_str = ""
            if self._depth_map is not None and self.depth_estimator is not None:
                distance = self.depth_estimator.get_distance_in_region(
                    self._depth_map, result.bbox
                )
                distance_str = f", {distance:.1f}m away"

            # Determine position
            frame_width = frame.shape[1]
            rel_x = result.center[0] / frame_width
            if rel_x < 0.33:
                position = "to your left"
            elif rel_x > 0.67:
                position = "to your right"
            else:
                position = "ahead"

            self.state.last_response = f"Found {query} {position}{distance_str}"
        else:
            self.state.last_response = f"Could not find {query}"
            self._florence_results = []

        return self.state.last_response

    def caption(self, frame: NDArray[np.uint8]) -> str:
        """Get scene caption using Florence-2."""
        self.state.last_query = "What is in front of me?"

        if self.florence is None:
            self.state.last_response = "Florence-2 not loaded"
            return self.state.last_response

        caption = self.florence.caption(frame, detailed=True)
        self.state.last_response = caption if caption else "Could not describe scene"

        return self.state.last_response

    def clear_response(self) -> None:
        """Clear the query/response display."""
        self.state.last_query = ""
        self.state.last_response = ""
        self._florence_results = []


def run_ui(source: str | int = 0) -> None:
    """Run the Smart Aid UI.

    Args:
        source: Video source (0 for webcam, URL for stream, path for file).
    """
    # Load config
    config = Config()
    config.pipeline.use_florence = True
    config.pipeline.use_yolo_world = True
    config.pipeline.use_depth = True

    # Create UI
    ui = SmartAidUI(config)

    print("Loading models...")
    if not ui.load():
        print("Failed to load models")
        return

    # Open video source
    if isinstance(source, str) and source.startswith("http"):
        print(f"Connecting to stream: {source}")
    else:
        print(f"Opening video source: {source}")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return

    print("\nUI started. Press 'h' for help, 'q' to quit.")
    cv2.namedWindow(ui.window_name, cv2.WINDOW_NORMAL)

    input_buffer = ""
    input_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Process frame
        ui.process_frame(frame)

        # Render UI
        output = ui.render(frame)

        # Show input prompt if in input mode
        if input_mode:
            cv2.putText(
                output, f"Search: {input_buffer}_", (10, output.shape[0] - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ui.COLOR_INFO, 2
            )

        cv2.imshow(ui.window_name, output)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF

        if input_mode:
            if key == 27:  # ESC - cancel
                input_mode = False
                input_buffer = ""
            elif key == 13:  # Enter - submit
                if input_buffer:
                    ui.search(frame, input_buffer)
                input_mode = False
                input_buffer = ""
            elif key == 8:  # Backspace
                input_buffer = input_buffer[:-1]
            elif 32 <= key <= 126:  # Printable characters
                input_buffer += chr(key)
        else:
            if key == ord("s"):
                input_mode = True
                input_buffer = ""
                ui.clear_response()
            elif key == ord("c"):
                ui.caption(frame)
            elif not ui.handle_key(key):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        source = sys.argv[1]
        if source.isdigit():
            source = int(source)
    else:
        source = 0  # Default webcam

    run_ui(source)
