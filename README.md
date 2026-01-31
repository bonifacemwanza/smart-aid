# Smart Aid

**AI-Powered Navigation Assistant for Visually Impaired Users**

Smart Aid is a real-time navigation system that helps visually impaired users navigate indoor spaces using computer vision and natural language interaction.

**Author:** Boniface Mwanza
**Supervisor:** dr inż. Tomasz Siemek
**Institution:** AGH University

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Zero-shot Object Detection** - YOLO-World detects custom objects (doors, stairs) without retraining
- **Depth Estimation** - Depth Anything V2 with indoor calibration (5m max)
- **Voice Interaction** - Wake word activation ("Hey Manso") with voice calibration
- **Natural Language** - Florence-2 VLM for scene understanding and queries
- **Local LLM Support** - Ollama integration for privacy-focused navigation instructions
- **Qualitative Feedback** - "within reach", "nearby" instead of numeric distances
- **Body-Relative Directions** - "on your left" instead of clock positions

## Quick Start

```bash
# Clone the repository
git clone https://github.com/bonifacemwanza/smart-aid.git
cd smart-aid

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the web UI
python src/nav_ui.py

# Open http://localhost:8080 in Chrome
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      WEB BROWSER                             │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐ │
│  │  Wake Word    │  │    Speech     │  │  Text-to-Speech │ │
│  │  "Hey Manso"  │──│  Recognition  │──│     Output      │ │
│  └───────────────┘  └───────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     PYTHON BACKEND                           │
│                                                              │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐           │
│  │YOLO-World │    │Florence-2 │    │   Depth   │           │
│  │  249ms    │    │  3938ms   │    │  1402ms   │           │
│  └─────┬─────┘    └─────┬─────┘    └─────┬─────┘           │
│        │                │                │                  │
│        └────────────────┼────────────────┘                  │
│                         ▼                                   │
│              ┌──────────────────────┐                       │
│              │   Smart Navigator    │                       │
│              │ + Qualitative Dist.  │                       │
│              │ + Body-Rel. Dirs.    │                       │
│              └──────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
smart-aid/
├── src/
│   ├── detector.py         # YOLO-World object detection
│   ├── depth.py            # Depth Anything V2 with indoor calibration
│   ├── florence.py         # Florence-2 VLM
│   ├── navigator.py        # Ollama/local LLM navigation
│   ├── smart_navigator.py  # Query processing
│   ├── nav_ui.py           # Web interface
│   └── config.py           # Configuration
├── scripts/
│   ├── collect_house_data.py  # Data collection for fine-tuning
│   └── auto_annotate.py       # Auto-annotation with YOLO-World
├── notebooks/
│   └── 07_thesis_experiments.ipynb  # Benchmarks and experiments
├── thesis/
│   ├── THESIS.md           # Full thesis document
│   ├── RESULTS_SUMMARY.md  # Experiment results
│   └── figures/            # Diagrams and screenshots
└── requirements.txt
```

## Performance

| Component | Inference Time |
|-----------|---------------|
| YOLO-World | 249 ± 14 ms |
| Depth Anything V2 | 1,402 ± 28 ms |
| Florence-2 | 3,938 ± 191 ms |
| **Full Pipeline** | **~6 seconds** |
| Fast Path (no VLM) | ~1.8 seconds |

*Tested on MacBook (Intel i7), CPU-only*

## Key Innovations

1. **Indoor Depth Calibration** - 5m max distance (vs 10m default) for realistic room-scale estimates
2. **Qualitative Distances** - "within reach", "very close", "nearby", "across the room"
3. **Body-Relative Directions** - "on your left" instead of "9 o'clock"
4. **Voice Calibration** - System learns user's pronunciation of wake word
5. **House-Specific Fine-Tuning** - Framework for adapting to specific homes

## Usage Examples

**Voice Commands:**
- "Hey Manso, what's in front of me?"
- "Hey Manso, where is the door?"
- "Hey Manso, is it safe to walk forward?"
- "Hey Manso, describe the room"

**Example Responses:**
- "A room with a table and two chairs. Door on your right, nearby."
- "Clear path ahead. Chair on your left, within reach."
- "Warning: obstacle directly ahead, very close. Step right to avoid."

## Screenshots

| Ready State | Voice Calibration |
|-------------|-------------------|
| ![Ready](thesis/figures/ui_ready_state.png) | ![Calibration](thesis/figures/ui_calibration_complete.png) |

| Query Response | Safety Scan |
|----------------|-------------|
| ![Query](thesis/figures/ui_query_response.png) | ![Scan](thesis/figures/ui_safety_scan.png) |

## Requirements

- Python 3.11+
- Chrome browser (for Web Speech API)
- Webcam or Raspberry Pi Camera
- (Optional) Ollama for local LLM support

## Optional: Ollama Setup

For smarter navigation instructions without cloud dependencies:

```bash
# Install Ollama
brew install ollama

# Start the server
ollama serve

# Pull a vision model
ollama pull llava
```

## Raspberry Pi Setup

```bash
# Install dependencies
sudo apt update
sudo apt install python3-opencv python3-flask python3-picamera2

# Start streaming
python pi_scripts/stream_server.py
```

## Thesis

This project is part of a thesis on AI-powered assistive technology. Full documentation:
- [THESIS.md](thesis/THESIS.md) - Complete thesis document
- [RESULTS_SUMMARY.md](thesis/RESULTS_SUMMARY.md) - Experimental results

## Cost Comparison

| System | Price |
|--------|-------|
| **Smart Aid (ours)** | ~$250 |
| Envision Glasses | $1,899 |
| OrCam MyEye | $4,500 |

## References

- [YOLO-World](https://github.com/AILab-CVC/YOLO-World) - Zero-shot object detection
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) - Monocular depth estimation
- [Florence-2](https://huggingface.co/microsoft/Florence-2-base) - Vision-language model
- [Ollama](https://ollama.ai/) - Local LLM inference

## License

MIT License - See [LICENSE](LICENSE) for details.
