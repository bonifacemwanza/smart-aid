"""Voice input module using OpenAI Whisper for speech-to-text."""

import numpy as np
from numpy.typing import NDArray

from src.config import VoiceConfig


class VoiceInput:
    """Whisper-based speech-to-text for voice commands."""

    def __init__(self, config: VoiceConfig) -> None:
        self.config = config
        self.model = None

    def load(self) -> bool:
        """Load Whisper model.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self.config.enabled:
            print("Voice input disabled in config")
            return True

        try:
            import whisper

            print(f"Loading Whisper model: {self.config.model}")
            self.model = whisper.load_model(self.config.model, device=self.config.device)
            print("Whisper model loaded successfully")
            return True

        except ImportError:
            print("Whisper not installed. Run: pip install openai-whisper")
            return False
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            return False

    def transcribe(self, audio: NDArray[np.float32]) -> str | None:
        """Transcribe audio to text.

        Args:
            audio: Audio data as float32 numpy array (16kHz sample rate).

        Returns:
            Transcribed text or None on error.
        """
        if not self.config.enabled or self.model is None:
            return None

        try:
            result = self.model.transcribe(
                audio,
                language=self.config.language,
                fp16=False,
            )
            text = result["text"].strip()
            return text if text else None

        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def transcribe_file(self, audio_path: str) -> str | None:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.).

        Returns:
            Transcribed text or None on error.
        """
        if not self.config.enabled or self.model is None:
            return None

        try:
            result = self.model.transcribe(
                audio_path,
                language=self.config.language,
                fp16=False,
            )
            text = result["text"].strip()
            return text if text else None

        except Exception as e:
            print(f"Transcription error: {e}")
            return None


class AudioRecorder:
    """Simple audio recorder for voice input."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._recording = False
        self._audio_data: list[NDArray[np.float32]] = []

    def start_recording(self) -> bool:
        """Start recording audio from microphone.

        Returns:
            True if recording started, False otherwise.
        """
        try:
            import sounddevice as sd

            self._recording = True
            self._audio_data = []

            def callback(indata: NDArray, frames: int, time_info: dict, status: int) -> None:
                if self._recording:
                    self._audio_data.append(indata.copy())

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=callback,
            )
            self._stream.start()
            return True

        except ImportError:
            print("sounddevice not installed. Run: pip install sounddevice")
            return False
        except Exception as e:
            print(f"Failed to start recording: {e}")
            return False

    def stop_recording(self) -> NDArray[np.float32] | None:
        """Stop recording and return audio data.

        Returns:
            Audio data as float32 array or None if no recording.
        """
        if not self._recording:
            return None

        self._recording = False

        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

        if not self._audio_data:
            return None

        audio = np.concatenate(self._audio_data, axis=0).flatten()
        self._audio_data = []
        return audio

    def record_for_duration(self, duration: float) -> NDArray[np.float32] | None:
        """Record audio for specified duration.

        Args:
            duration: Recording duration in seconds.

        Returns:
            Audio data as float32 array or None on error.
        """
        try:
            import sounddevice as sd

            print(f"Recording for {duration} seconds...")
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
            )
            sd.wait()
            return audio.flatten()

        except ImportError:
            print("sounddevice not installed. Run: pip install sounddevice")
            return None
        except Exception as e:
            print(f"Recording error: {e}")
            return None


if __name__ == "__main__":
    config = VoiceConfig()
    voice = VoiceInput(config)

    print("Loading Whisper model...")
    if not voice.load():
        print("Failed to load model")
        exit(1)

    recorder = AudioRecorder()
    print("\nRecording 3 seconds of audio...")
    audio = recorder.record_for_duration(3.0)

    if audio is not None:
        print("Transcribing...")
        text = voice.transcribe(audio)
        if text:
            print(f"Transcription: {text}")
        else:
            print("No speech detected")
