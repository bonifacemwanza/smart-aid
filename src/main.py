import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera
from src.config import Config
from src.depth import DepthEstimator
from src.detector import Detector
from src.feedback import FeedbackManager
from src.fusion import FusionEngine
from src.stream_client import StreamClientCV
from src.utils import FPSCounter, create_side_by_side, draw_fps, save_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Aid - Navigation Assistant")
    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        help="Video source: 'webcam', 'stream', or path to video/image",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results",
        help="Output directory for saved frames",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Disable audio feedback",
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Disable depth estimation",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save processed frames",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = Config.from_yaml(args.config)

    if args.no_feedback:
        config.feedback.enabled = False

    print("Loading models...")
    detector = Detector(config.detection)
    if not detector.load():
        print("Failed to load detector")
        return 1
    print(f"  Detector: {config.detection.model}")

    depth_estimator: DepthEstimator | None = None
    if not args.no_depth:
        depth_estimator = DepthEstimator(config.depth)
        if not depth_estimator.load():
            print("Failed to load depth estimator")
            return 1
        print(f"  Depth: {config.depth.model}")

    fusion = FusionEngine(config.fusion)
    feedback = FeedbackManager(config.feedback)
    fps_counter = FPSCounter()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0

    if args.source == "webcam":
        print("Using webcam...")
        source = Camera(config.camera)
        if not source.open():
            print("Failed to open webcam")
            return 1
    elif args.source == "stream":
        print(f"Connecting to stream: {config.stream.pi_url}")
        source = StreamClientCV(config.stream)
        if not source.connect():
            print("Failed to connect to stream")
            return 1
    else:
        source_path = Path(args.source)
        if source_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            print(f"Processing image: {args.source}")
            frame = cv2.imread(args.source)
            if frame is None:
                print("Failed to load image")
                return 1
            process_single_frame(
                frame, detector, depth_estimator, fusion, feedback, args
            )
            return 0
        else:
            print(f"Opening video: {args.source}")
            source = cv2.VideoCapture(args.source)
            if not source.isOpened():
                print("Failed to open video")
                return 1

    print("Starting pipeline...")
    print("Press 'q' to quit, 's' to save frame")

    try:
        while True:
            if hasattr(source, "read"):
                frame = source.read()
            else:
                ret, frame = source.read()
                if not ret:
                    frame = None

            if frame is None:
                continue

            detections = detector.detect(frame)

            depth_map = None
            if depth_estimator is not None:
                depth_map = depth_estimator.estimate(frame)

            if depth_map is not None:
                obstacles = fusion.process(detections, depth_map, frame.shape[1])
                vis_frame = fusion.draw_obstacles(frame, obstacles)

                if obstacles and config.feedback.enabled:
                    feedback.alert_danger(obstacles)

                depth_colored = depth_estimator.colorize(depth_map)
                display = create_side_by_side(
                    vis_frame, depth_colored, ("Detection", "Depth")
                )
            else:
                vis_frame = detector.draw_detections(frame, detections)
                display = vis_frame

            fps = fps_counter.update()
            display = draw_fps(display, fps)

            cv2.imshow("Smart Aid", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") or args.save_frames:
                filename = output_dir / f"frame_{frame_count:04d}.jpg"
                save_frame(display, str(filename))
                print(f"Saved: {filename}")
                frame_count += 1

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        if hasattr(source, "close"):
            source.close()
        elif hasattr(source, "disconnect"):
            source.disconnect()
        elif hasattr(source, "release"):
            source.release()
        cv2.destroyAllWindows()

    return 0


def process_single_frame(
    frame: np.ndarray,
    detector: Detector,
    depth_estimator: DepthEstimator | None,
    fusion: FusionEngine,
    feedback: FeedbackManager,
    args: argparse.Namespace,
) -> None:
    detections = detector.detect(frame)
    print(f"Detections: {len(detections)}")
    for det in detections:
        print(f"  - {det.class_name}: {det.confidence:.2f}")

    if depth_estimator is not None:
        depth_map = depth_estimator.estimate(frame)
        if depth_map is not None:
            obstacles = fusion.process(detections, depth_map, frame.shape[1])
            print(f"\nObstacles: {len(obstacles)}")
            for obs in obstacles:
                print(f"  - {obs.to_alert_text()}")

            vis_frame = fusion.draw_obstacles(frame, obstacles)
            depth_colored = depth_estimator.colorize(depth_map)
            display = create_side_by_side(
                vis_frame, depth_colored, ("Detection", "Depth")
            )
        else:
            vis_frame = detector.draw_detections(frame, detections)
            display = vis_frame
    else:
        vis_frame = detector.draw_detections(frame, detections)
        display = vis_frame

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.jpg"
    save_frame(display, str(output_path))
    print(f"\nSaved result to: {output_path}")

    cv2.imshow("Result", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
