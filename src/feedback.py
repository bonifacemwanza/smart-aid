import time
from threading import Lock, Thread

from src.config import FeedbackConfig
from src.fusion import Obstacle


class FeedbackManager:
    def __init__(self, config: FeedbackConfig) -> None:
        self.config = config
        self.last_alert_time = 0.0
        self.engine = None
        self.lock = Lock()
        self._init_tts()

    def _init_tts(self) -> None:
        if not self.config.enabled:
            return

        try:
            import pyttsx3

            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", self.config.voice_rate)
            self.engine.setProperty("volume", self.config.voice_volume)
        except Exception as e:
            print(f"TTS initialization failed: {e}")
            print("Falling back to console output")
            self.engine = None

    def alert(self, obstacle: Obstacle) -> bool:
        if not self.config.enabled:
            return False

        now = time.time()
        if now - self.last_alert_time < self.config.min_interval:
            return False

        text = obstacle.to_alert_text()

        with self.lock:
            self.last_alert_time = now

        Thread(target=self._speak, args=(text,), daemon=True).start()
        return True

    def _speak(self, text: str) -> None:
        print(f"[ALERT] {text}")

        if self.engine is not None:
            try:
                with self.lock:
                    self.engine.say(text)
                    self.engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")

    def alert_text(self, text: str) -> bool:
        if not self.config.enabled:
            return False

        now = time.time()
        if now - self.last_alert_time < self.config.min_interval:
            return False

        with self.lock:
            self.last_alert_time = now

        Thread(target=self._speak, args=(text,), daemon=True).start()
        return True

    def alert_danger(self, obstacles: list[Obstacle]) -> bool:
        danger_obstacles = [o for o in obstacles if o.is_danger]
        if not danger_obstacles:
            return False

        top = danger_obstacles[0]
        return self.alert(top)

    def alert_top(self, obstacles: list[Obstacle]) -> bool:
        if not obstacles:
            return False

        return self.alert(obstacles[0])


if __name__ == "__main__":
    from src.detector import Detection

    config = FeedbackConfig(enabled=True, min_interval=1.0)
    feedback = FeedbackManager(config)

    test_detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.85,
        bbox=(100, 100, 200, 300),
        center=(150, 200),
    )

    test_obstacle = Obstacle(
        detection=test_detection,
        distance=1.2,
        position="center",
        priority=100,
    )

    print("Testing feedback system...")
    feedback.alert(test_obstacle)

    time.sleep(2)

    test_obstacle2 = Obstacle(
        detection=Detection(
            class_id=2,
            class_name="car",
            confidence=0.9,
            bbox=(300, 100, 500, 300),
            center=(400, 200),
        ),
        distance=2.5,
        position="right",
        priority=75,
    )

    feedback.alert(test_obstacle2)
    time.sleep(2)
