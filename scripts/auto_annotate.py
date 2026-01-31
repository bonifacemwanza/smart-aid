"""
Auto-Annotation Script using YOLO-World

This script uses YOLO-World's zero-shot detection to automatically
annotate house images for fine-tuning.

Usage:
    python scripts/auto_annotate.py --input data/house/living_room/images
    python scripts/auto_annotate.py --input data/house --all-rooms
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


# Default classes for house environments
HOUSE_CLASSES = [
    # Furniture
    "sofa", "chair", "table", "desk", "bed", "wardrobe", "dresser",
    "nightstand", "bookshelf", "shelf", "cabinet", "drawer",
    # Electronics
    "TV", "television", "computer", "monitor", "laptop", "printer",
    # Kitchen
    "refrigerator", "stove", "oven", "microwave", "sink", "dishwasher",
    "toaster", "coffee maker",
    # Bathroom
    "toilet", "bathtub", "shower",
    # Structure
    "door", "window", "stairs", "wall",
    # Decor
    "lamp", "mirror", "plant", "picture frame", "rug", "curtain",
    # Other
    "person", "bag", "bottle", "cup", "book"
]


def auto_annotate_images(
    image_dir: Path,
    output_dir: Path,
    classes: list[str] = None,
    confidence: float = 0.25,
    visualize: bool = True
) -> dict:
    """Auto-annotate images using YOLO-World.

    Args:
        image_dir: Directory containing images
        output_dir: Directory to save labels
        classes: List of classes to detect
        confidence: Confidence threshold
        visualize: Save visualization images

    Returns:
        Statistics about annotations
    """
    if classes is None:
        classes = HOUSE_CLASSES

    # Load YOLO-World model
    print("Loading YOLO-World model...")
    model = YOLO("yolov8s-world.pt")
    model.set_classes(classes)

    # Create output directories
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    if visualize:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images to annotate")
    print(f"Classes: {len(classes)} categories")
    print(f"Confidence threshold: {confidence}")

    stats = {
        "total_images": len(image_files),
        "annotated_images": 0,
        "total_annotations": 0,
        "class_counts": {},
        "images_without_detections": []
    }

    for i, img_path in enumerate(image_files):
        print(f"  Processing {i+1}/{len(image_files)}: {img_path.name}", end="")

        # Run detection
        results = model.predict(
            source=str(img_path),
            conf=confidence,
            verbose=False
        )

        if len(results) == 0 or len(results[0].boxes) == 0:
            print(" - No detections")
            stats["images_without_detections"].append(img_path.name)
            continue

        result = results[0]
        boxes = result.boxes

        # Save YOLO format labels
        label_path = labels_dir / f"{img_path.stem}.txt"
        img_h, img_w = result.orig_shape

        with open(label_path, "w") as f:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Get normalized coordinates (YOLO format)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                # Update stats
                cls_name = classes[cls_id]
                stats["class_counts"][cls_name] = stats["class_counts"].get(cls_name, 0) + 1
                stats["total_annotations"] += 1

        stats["annotated_images"] += 1
        print(f" - {len(boxes)} detections")

        # Save visualization
        if visualize:
            vis_img = result.plot()
            vis_path = vis_dir / f"{img_path.stem}_annotated.jpg"
            cv2.imwrite(str(vis_path), vis_img)

    return stats


def create_dataset_yaml(
    dataset_dir: Path,
    classes: list[str],
    train_ratio: float = 0.8
) -> Path:
    """Create YOLO dataset configuration file.

    Args:
        dataset_dir: Root directory of dataset
        classes: List of class names
        train_ratio: Ratio of training images

    Returns:
        Path to created YAML file
    """
    # Split images into train/val
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return None

    # Get annotated images (those with label files)
    annotated = []
    for img_path in images_dir.iterdir():
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            annotated.append(img_path.name)

    if len(annotated) == 0:
        print("Error: No annotated images found")
        return None

    # Shuffle and split
    import random
    random.shuffle(annotated)
    split_idx = int(len(annotated) * train_ratio)
    train_images = annotated[:split_idx]
    val_images = annotated[split_idx:]

    # Create train/val directories
    for split in ["train", "val"]:
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy files to train/val
    for img_name in train_images:
        src_img = images_dir / img_name
        dst_img = dataset_dir / "train" / "images" / img_name
        shutil.copy2(src_img, dst_img)

        label_name = f"{Path(img_name).stem}.txt"
        src_label = labels_dir / label_name
        dst_label = dataset_dir / "train" / "labels" / label_name
        if src_label.exists():
            shutil.copy2(src_label, dst_label)

    for img_name in val_images:
        src_img = images_dir / img_name
        dst_img = dataset_dir / "val" / "images" / img_name
        shutil.copy2(src_img, dst_img)

        label_name = f"{Path(img_name).stem}.txt"
        src_label = labels_dir / label_name
        dst_label = dataset_dir / "val" / "labels" / label_name
        if src_label.exists():
            shutil.copy2(src_label, dst_label)

    # Create YAML config
    yaml_content = {
        "path": str(dataset_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "names": {i: name for i, name in enumerate(classes)}
    }

    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print(f"\nDataset created:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")
    print(f"  Config: {yaml_path}")

    return yaml_path


def annotate_all_rooms(
    house_dir: Path,
    classes: list[str] = None,
    confidence: float = 0.25
) -> dict:
    """Annotate images from all rooms.

    Args:
        house_dir: Directory containing room subdirectories
        classes: Classes to detect
        confidence: Confidence threshold

    Returns:
        Combined statistics
    """
    if classes is None:
        classes = HOUSE_CLASSES

    total_stats = {
        "total_images": 0,
        "annotated_images": 0,
        "total_annotations": 0,
        "class_counts": {},
        "rooms": {}
    }

    # Find room directories
    room_dirs = [d for d in house_dir.iterdir() if d.is_dir() and (d / "images").exists()]

    if len(room_dirs) == 0:
        print(f"No room directories found in {house_dir}")
        return total_stats

    print(f"Found {len(room_dirs)} rooms to annotate")

    for room_dir in room_dirs:
        room_name = room_dir.name
        print(f"\n{'='*60}")
        print(f"Annotating: {room_name}")
        print(f"{'='*60}")

        images_dir = room_dir / "images"
        stats = auto_annotate_images(images_dir, room_dir, classes, confidence)

        total_stats["rooms"][room_name] = stats
        total_stats["total_images"] += stats["total_images"]
        total_stats["annotated_images"] += stats["annotated_images"]
        total_stats["total_annotations"] += stats["total_annotations"]

        for cls, count in stats["class_counts"].items():
            total_stats["class_counts"][cls] = total_stats["class_counts"].get(cls, 0) + count

    return total_stats


def merge_room_datasets(house_dir: Path, output_dir: Path, classes: list[str] = None) -> Path:
    """Merge all room datasets into a single dataset.

    Args:
        house_dir: Directory containing room subdirectories
        output_dir: Output directory for merged dataset
        classes: Class names

    Returns:
        Path to merged dataset YAML
    """
    if classes is None:
        classes = HOUSE_CLASSES

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)

    # Find and merge all room data
    room_dirs = [d for d in house_dir.iterdir() if d.is_dir()]
    total_images = 0

    for room_dir in room_dirs:
        room_name = room_dir.name
        images_dir = room_dir / "images"
        labels_dir = room_dir / "labels"

        if not images_dir.exists():
            continue

        for img_path in images_dir.iterdir():
            # Copy image with room prefix
            new_name = f"{room_name}_{img_path.name}"
            dst_img = output_dir / "images" / new_name
            shutil.copy2(img_path, dst_img)

            # Copy label if exists
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                dst_label = output_dir / "labels" / f"{room_name}_{img_path.stem}.txt"
                shutil.copy2(label_path, dst_label)

            total_images += 1

    print(f"Merged {total_images} images from {len(room_dirs)} rooms")

    # Create dataset YAML
    return create_dataset_yaml(output_dir, classes)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-annotate house images using YOLO-World"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory (images or house root)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.25,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--all-rooms",
        action="store_true",
        help="Process all room subdirectories"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all rooms into single dataset"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip saving visualization images"
    )

    args = parser.parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir

    if args.all_rooms:
        stats = annotate_all_rooms(input_dir, confidence=args.confidence)

        # Save statistics
        stats_path = input_dir / "annotation_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {stats_path}")

        if args.merge:
            merged_dir = input_dir / "merged"
            merge_room_datasets(input_dir, merged_dir)

    else:
        stats = auto_annotate_images(
            input_dir,
            output_dir,
            confidence=args.confidence,
            visualize=not args.no_visualize
        )

        # Create dataset YAML if needed
        if (output_dir / "labels").exists():
            create_dataset_yaml(output_dir, HOUSE_CLASSES)

        print(f"\nAnnotation complete:")
        print(f"  Images: {stats['annotated_images']}/{stats['total_images']}")
        print(f"  Annotations: {stats['total_annotations']}")
        print(f"\nTop classes:")
        sorted_classes = sorted(
            stats["class_counts"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for cls, count in sorted_classes[:10]:
            print(f"    {cls}: {count}")


if __name__ == "__main__":
    main()
