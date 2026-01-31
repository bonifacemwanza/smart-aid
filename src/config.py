from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class StreamConfig(BaseModel):
    pi_url: str = "http://192.168.1.100:5000/video_feed"
    timeout: float = 5.0
    reconnect_attempts: int = 3
    reconnect_delay: float = 1.0


class CameraConfig(BaseModel):
    width: int = 640
    height: int = 480
    fps: int = 30
    device: int = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False


class DetectionConfig(BaseModel):
    model: str = "yolov8s-world.pt"  # YOLO-World for zero-shot detection
    confidence: float = 0.2  # Lower threshold for YOLO-World
    iou_threshold: float = 0.45
    classes: list[str] | None = None  # None means detect all classes
    device: str = "cpu"


class DepthConfig(BaseModel):
    model: Literal["vits", "vitb", "vitl"] = "vits"
    max_distance: float = 10.0
    colormap: str = "inferno"
    # Indoor mode - more realistic for rooms/houses
    indoor_mode: bool = True
    indoor_max_distance: float = 5.0  # Typical room is 3-5 meters
    # Distance zones for qualitative descriptions
    very_close: float = 0.5   # Within arm's reach
    close: float = 1.5        # A couple of steps
    nearby: float = 3.0       # A few steps away
    # Step calibration (average step length)
    step_length: float = 0.6  # More conservative for indoor


class FusionConfig(BaseModel):
    danger_zone: float = 1.5
    warning_zone: float = 3.0
    position_threshold: float = 0.33


class FeedbackConfig(BaseModel):
    enabled: bool = True
    min_interval: float = 1.5
    voice_rate: int = 150
    voice_volume: float = 1.0


class VisualizationConfig(BaseModel):
    show_fps: bool = True
    show_detections: bool = True
    show_depth: bool = True
    bbox_thickness: int = 2
    font_scale: float = 0.6


class VoiceConfig(BaseModel):
    """Configuration for voice input (Whisper STT)."""

    enabled: bool = True
    model: Literal["tiny", "base", "small", "medium"] = "base"
    language: str = "en"
    device: str = "cpu"


class FlorenceConfig(BaseModel):
    """Configuration for Florence-2 vision-language model."""

    enabled: bool = True
    model: str = "microsoft/Florence-2-base"
    device: str = "cpu"


class PipelineConfig(BaseModel):
    """Configuration for the main processing pipeline with feature toggles."""

    # Feature toggles
    use_yolo_world: bool = True
    use_florence: bool = True
    use_depth: bool = True
    use_voice_input: bool = True
    use_voice_output: bool = True

    # Operating mode
    mode: Literal["alert", "interactive", "both"] = "both"


class Config(BaseModel):
    stream: StreamConfig = Field(default_factory=StreamConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    florence: FlorenceConfig = Field(default_factory=FlorenceConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    config = Config()
    print("Default config loaded:")
    print(f"  Stream URL: {config.stream.pi_url}")
    print(f"  Camera: {config.camera.width}x{config.camera.height}")
    print(f"  Detection model: {config.detection.model}")
    print(f"  Depth model: {config.depth.model}")
    print(f"  Feedback enabled: {config.feedback.enabled}")
    print(f"  Voice input enabled: {config.voice.enabled}")
    print(f"  Florence-2 enabled: {config.florence.enabled}")
    print(f"  Pipeline mode: {config.pipeline.mode}")
