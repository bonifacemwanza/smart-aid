"""Florence-2 Vision-Language Model wrapper for visual grounding and captioning."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import FlorenceConfig


@dataclass
class GroundingResult:
    """Result from visual grounding."""

    phrase: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
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


class FlorenceModel:
    """Florence-2 model wrapper for visual grounding, detection, and captioning."""

    # Florence-2 task prompts
    TASK_CAPTION = "<CAPTION>"
    TASK_DETAILED_CAPTION = "<DETAILED_CAPTION>"
    TASK_MORE_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"
    TASK_OD = "<OD>"  # Object detection
    TASK_DENSE_REGION_CAPTION = "<DENSE_REGION_CAPTION>"
    TASK_CAPTION_TO_PHRASE_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
    TASK_REFERRING_EXPRESSION_SEGMENTATION = "<REFERRING_EXPRESSION_SEGMENTATION>"
    TASK_REGION_TO_CATEGORY = "<REGION_TO_CATEGORY>"
    TASK_OPEN_VOCABULARY_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    TASK_OCR = "<OCR>"
    TASK_OCR_WITH_REGION = "<OCR_WITH_REGION>"

    def __init__(self, config: FlorenceConfig) -> None:
        self.config = config
        self.model = None
        self.processor = None

    def load(self) -> bool:
        """Load Florence-2 model.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self.config.enabled:
            print("Florence-2 disabled in config")
            return True

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            print(f"Loading Florence-2 model: {self.config.model}")
            print(f"Device: {self.config.device}")

            self.processor = AutoProcessor.from_pretrained(
                self.config.model,
                trust_remote_code=True,
            )

            import torch
            # Load model with attn_implementation to avoid SDPA issues
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )

            if self.config.device != "cpu":
                self.model = self.model.to(self.config.device)

            self.model.eval()
            print("Florence-2 model loaded successfully")
            return True

        except ImportError as e:
            print(f"Missing dependency: {e}")
            print("Run: pip install transformers torch")
            return False
        except Exception as e:
            print(f"Failed to load Florence-2 model: {e}")
            return False

    def _run_task(
        self, frame: NDArray[np.uint8], task_prompt: str, text_input: str = ""
    ) -> dict:
        """Run a Florence-2 task on an image.

        Args:
            frame: Input image as BGR numpy array.
            task_prompt: Florence-2 task prompt (e.g., "<CAPTION>").
            text_input: Optional text input for grounding tasks.

        Returns:
            Dictionary with task results.
        """
        if self.model is None or self.processor is None:
            return {}

        try:
            import torch
            from PIL import Image

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Prepare prompt
            if text_input:
                prompt = f"{task_prompt}{text_input}"
            else:
                prompt = task_prompt

            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt",
            )

            if self.config.device != "cpu":
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                )

            # Decode
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            # Post-process
            parsed = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(pil_image.width, pil_image.height),
            )

            return parsed

        except Exception as e:
            print(f"Florence-2 task error: {e}")
            return {}

    def caption(self, frame: NDArray[np.uint8], detailed: bool = True) -> str:
        """Generate a caption for the image.

        Args:
            frame: Input image as BGR numpy array.
            detailed: If True, generate more detailed caption.

        Returns:
            Caption string.
        """
        if not self.config.enabled or self.model is None:
            return ""

        task = self.TASK_DETAILED_CAPTION if detailed else self.TASK_CAPTION
        result = self._run_task(frame, task)

        if task in result:
            return result[task]
        return ""

    def detect(self, frame: NDArray[np.uint8]) -> list[GroundingResult]:
        """Detect objects in the image.

        Args:
            frame: Input image as BGR numpy array.

        Returns:
            List of GroundingResult objects.
        """
        if not self.config.enabled or self.model is None:
            return []

        result = self._run_task(frame, self.TASK_OD)
        return self._parse_detection_results(result)

    def ground(self, frame: NDArray[np.uint8], phrase: str) -> list[GroundingResult]:
        """Find objects matching the phrase (visual grounding).

        Args:
            frame: Input image as BGR numpy array.
            phrase: Text description of object to find (e.g., "phone", "red chair").

        Returns:
            List of GroundingResult objects for matching objects.
        """
        if not self.config.enabled or self.model is None:
            return []

        result = self._run_task(frame, self.TASK_CAPTION_TO_PHRASE_GROUNDING, phrase)
        return self._parse_grounding_results(result, phrase)

    def detect_open_vocabulary(
        self, frame: NDArray[np.uint8], classes: list[str]
    ) -> list[GroundingResult]:
        """Detect objects from a list of class names (open vocabulary).

        Args:
            frame: Input image as BGR numpy array.
            classes: List of class names to detect.

        Returns:
            List of GroundingResult objects.
        """
        if not self.config.enabled or self.model is None:
            return []

        # Join classes for the prompt
        class_text = ", ".join(classes)
        result = self._run_task(frame, self.TASK_OPEN_VOCABULARY_DETECTION, class_text)
        return self._parse_detection_results(result)

    def ocr(self, frame: NDArray[np.uint8]) -> str:
        """Extract text from the image using OCR.

        Args:
            frame: Input image as BGR numpy array.

        Returns:
            Extracted text string.
        """
        if not self.config.enabled or self.model is None:
            return ""

        result = self._run_task(frame, self.TASK_OCR)

        if self.TASK_OCR in result:
            return result[self.TASK_OCR]
        return ""

    def _parse_detection_results(self, result: dict) -> list[GroundingResult]:
        """Parse detection results into GroundingResult list."""
        detections = []

        # Handle <OD> task results
        if self.TASK_OD in result:
            data = result[self.TASK_OD]
            if "bboxes" in data and "labels" in data:
                for bbox, label in zip(data["bboxes"], data["labels"]):
                    x1, y1, x2, y2 = map(int, bbox)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    detections.append(
                        GroundingResult(
                            phrase=label,
                            bbox=(x1, y1, x2, y2),
                            center=center,
                        )
                    )

        # Handle open vocabulary detection results
        if self.TASK_OPEN_VOCABULARY_DETECTION in result:
            data = result[self.TASK_OPEN_VOCABULARY_DETECTION]
            if "bboxes" in data and "bboxes_labels" in data:
                for bbox, label in zip(data["bboxes"], data["bboxes_labels"]):
                    x1, y1, x2, y2 = map(int, bbox)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    detections.append(
                        GroundingResult(
                            phrase=label,
                            bbox=(x1, y1, x2, y2),
                            center=center,
                        )
                    )

        return detections

    def _parse_grounding_results(
        self, result: dict, phrase: str
    ) -> list[GroundingResult]:
        """Parse grounding results into GroundingResult list."""
        detections = []

        if self.TASK_CAPTION_TO_PHRASE_GROUNDING in result:
            data = result[self.TASK_CAPTION_TO_PHRASE_GROUNDING]
            if "bboxes" in data:
                labels = data.get("labels", [phrase] * len(data["bboxes"]))
                for bbox, label in zip(data["bboxes"], labels):
                    x1, y1, x2, y2 = map(int, bbox)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    detections.append(
                        GroundingResult(
                            phrase=label if label else phrase,
                            bbox=(x1, y1, x2, y2),
                            center=center,
                        )
                    )

        return detections

    def draw_results(
        self,
        frame: NDArray[np.uint8],
        results: list[GroundingResult],
        thickness: int = 2,
        font_scale: float = 0.6,
    ) -> NDArray[np.uint8]:
        """Draw grounding results on frame.

        Args:
            frame: Input image.
            results: List of GroundingResult objects.
            thickness: Line thickness.
            font_scale: Font scale for labels.

        Returns:
            Frame with drawn results.
        """
        frame = frame.copy()

        for result in results:
            color = (0, 255, 0)  # Green

            # Draw bounding box
            cv2.rectangle(
                frame, (result.x1, result.y1), (result.x2, result.y2), color, thickness
            )

            # Draw label
            label = result.phrase
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            cv2.rectangle(
                frame,
                (result.x1, result.y1 - label_h - baseline - 5),
                (result.x1 + label_w, result.y1),
                color,
                -1,
            )

            cv2.putText(
                frame,
                label,
                (result.x1, result.y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
            )

            # Draw center point
            cv2.circle(frame, result.center, 4, color, -1)

        return frame


if __name__ == "__main__":
    import sys

    config = FlorenceConfig()
    model = FlorenceModel(config)

    print("Loading Florence-2 model...")
    if not model.load():
        print("Failed to load model")
        sys.exit(1)
    print("Model loaded successfully!")

    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (200, 200, 200)  # Gray background
    cv2.rectangle(test_image, (100, 100), (250, 300), (139, 69, 19), -1)  # Brown rectangle (door-like)
    cv2.rectangle(test_image, (400, 200), (550, 400), (50, 50, 50), -1)  # Dark rectangle (chair-like)

    print("\nTesting caption...")
    caption = model.caption(test_image)
    print(f"Caption: {caption}")

    print("\nTesting object detection...")
    detections = model.detect(test_image)
    print(f"Detections: {len(detections)}")
    for d in detections:
        print(f"  - {d.phrase}: {d.bbox}")

    print("\nTesting visual grounding for 'rectangle'...")
    results = model.ground(test_image, "rectangle")
    print(f"Found: {len(results)}")
    for r in results:
        print(f"  - {r.phrase}: {r.bbox}")

    # Visualize
    result_img = model.draw_results(test_image, detections)
    cv2.imshow("Florence-2 Test", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
