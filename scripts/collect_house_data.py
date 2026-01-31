"""
Data Collection Script for House-Specific Fine-Tuning

This script helps capture images from different rooms in your house
for creating a custom dataset to fine-tune YOLO models.

Usage:
    python scripts/collect_house_data.py --room living_room --num 50
    python scripts/collect_house_data.py --room kitchen --num 50 --camera 1

Controls:
    SPACE - Capture image
    A     - Auto-capture mode (captures every 2 seconds)
    Q     - Quit and move to next room
"""

import argparse
import cv2
import os
import time
from datetime import datetime
from pathlib import Path


# Room configurations with target objects
ROOM_CONFIGS = {
    "living_room": [
        "sofa", "TV", "coffee table", "lamp", "bookshelf",
        "chair", "plant", "picture frame", "window", "door"
    ],
    "kitchen": [
        "refrigerator", "stove", "microwave", "sink", "cabinet",
        "table", "chair", "dishwasher", "toaster", "door"
    ],
    "bedroom": [
        "bed", "wardrobe", "nightstand", "dresser", "lamp",
        "mirror", "chair", "window", "door", "closet"
    ],
    "bathroom": [
        "toilet", "sink", "bathtub", "shower", "mirror",
        "towel rack", "cabinet", "door"
    ],
    "hallway": [
        "door", "stairs", "coat rack", "shoe rack", "mirror",
        "light", "window", "plant"
    ],
    "office": [
        "desk", "chair", "computer", "monitor", "bookshelf",
        "lamp", "window", "door", "printer"
    ]
}


def capture_room_images(
    room_name: str,
    output_dir: Path,
    num_images: int = 50,
    camera_id: int = 0,
    auto_interval: float = 2.0
) -> int:
    """Capture images from a room for dataset creation.

    Args:
        room_name: Name of the room (e.g., 'living_room')
        output_dir: Directory to save images
        num_images: Target number of images to capture
        camera_id: Camera device ID (0 for default)
        auto_interval: Interval for auto-capture mode (seconds)

    Returns:
        Number of images captured
    """
    room_dir = output_dir / room_name / "images"
    room_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return 0

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    count = 0
    auto_mode = False
    last_capture_time = 0

    target_objects = ROOM_CONFIGS.get(room_name, ["object"])

    print(f"\n{'='*60}")
    print(f"Capturing images for: {room_name.upper()}")
    print(f"{'='*60}")
    print(f"Target: {num_images} images")
    print(f"Target objects: {', '.join(target_objects[:5])}...")
    print(f"\nControls:")
    print(f"  SPACE - Capture image")
    print(f"  A     - Toggle auto-capture mode")
    print(f"  Q     - Quit")
    print(f"{'='*60}\n")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Create display frame with info overlay
        display = frame.copy()

        # Add info text
        mode_text = "AUTO" if auto_mode else "MANUAL"
        cv2.putText(display, f"Room: {room_name} | Mode: {mode_text}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Captured: {count}/{num_images}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=capture | A=auto | Q=quit",
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(f"Capture: {room_name}", display)

        # Auto-capture mode
        current_time = time.time()
        if auto_mode and (current_time - last_capture_time) >= auto_interval:
            save_image(frame, room_dir, room_name, count)
            count += 1
            last_capture_time = current_time
            print(f"  [AUTO] Captured {count}/{num_images}")

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Space to capture
            save_image(frame, room_dir, room_name, count)
            count += 1
            print(f"  Captured {count}/{num_images}")

        elif key == ord('a'):  # Toggle auto mode
            auto_mode = not auto_mode
            last_capture_time = current_time
            print(f"  Auto-capture: {'ON' if auto_mode else 'OFF'}")

        elif key == ord('q'):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nCaptured {count} images for {room_name}")
    return count


def save_image(frame, output_dir: Path, room_name: str, index: int) -> str:
    """Save a captured frame with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{room_name}_{timestamp}_{index:04d}.jpg"
    filepath = output_dir / filename
    cv2.imwrite(str(filepath), frame)
    return str(filepath)


def capture_from_stream(
    room_name: str,
    output_dir: Path,
    stream_url: str,
    num_images: int = 50
) -> int:
    """Capture images from a network stream (e.g., Pi camera).

    Args:
        room_name: Name of the room
        output_dir: Directory to save images
        stream_url: URL of the MJPEG stream
        num_images: Target number of images

    Returns:
        Number of images captured
    """
    import urllib.request

    room_dir = output_dir / room_name / "images"
    room_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    print(f"\nCapturing from stream: {stream_url}")
    print(f"Press Ctrl+C to stop\n")

    try:
        while count < num_images:
            try:
                # Fetch frame from stream
                with urllib.request.urlopen(stream_url, timeout=5) as response:
                    img_array = bytearray(response.read())

                frame = cv2.imdecode(
                    cv2.np.frombuffer(img_array, dtype=cv2.np.uint8),
                    cv2.IMREAD_COLOR
                )

                if frame is not None:
                    save_image(frame, room_dir, room_name, count)
                    count += 1
                    print(f"  Captured {count}/{num_images}")
                    time.sleep(1)  # Wait between captures

            except Exception as e:
                print(f"  Error fetching frame: {e}")
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nCapture interrupted")

    print(f"\nCaptured {count} images from stream")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Capture house images for fine-tuning"
    )
    parser.add_argument(
        "--room", "-r",
        type=str,
        default="living_room",
        choices=list(ROOM_CONFIGS.keys()),
        help="Room to capture"
    )
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=50,
        help="Number of images to capture"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device ID"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/house",
        help="Output directory"
    )
    parser.add_argument(
        "--stream", "-s",
        type=str,
        default=None,
        help="Stream URL (for Pi camera)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Capture all rooms sequentially"
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.all:
        # Capture all rooms
        total = 0
        for room in ROOM_CONFIGS.keys():
            print(f"\n\nStarting capture for: {room}")
            input("Press Enter when ready...")
            count = capture_room_images(room, output_dir, args.num, args.camera)
            total += count
        print(f"\n\nTotal images captured: {total}")
    elif args.stream:
        capture_from_stream(args.room, output_dir, args.stream, args.num)
    else:
        capture_room_images(args.room, output_dir, args.num, args.camera)


if __name__ == "__main__":
    main()
