"""Lightweight browser-based UI for Smart Aid.

This is much faster than the OpenCV UI because:
1. Models load once at startup
2. Detection runs on-demand (button click), not every frame
3. Video streaming is separate from processing
"""

import sys
import threading
import time
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

from src.config import Config
from src.smart_query import SmartQueryHandler

app = Flask(__name__)

# Global state
query_handler: SmartQueryHandler | None = None
camera: cv2.VideoCapture | None = None
current_frame: np.ndarray | None = None
frame_lock = threading.Lock()
last_response: str = ""
is_processing: bool = False


# HTML Template - lightweight and accessible
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Aid - Vision Assistant</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #00d4ff;
        }
        .video-container {
            position: relative;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        #video-feed {
            width: 100%;
            display: block;
        }
        .status-bar {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            display: flex;
            justify-content: space-between;
        }
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        button {
            padding: 20px;
            font-size: 18px;
            font-weight: bold;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        button:hover { transform: scale(1.02); }
        button:active { transform: scale(0.98); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }

        .btn-primary { background: #00d4ff; color: #000; }
        .btn-success { background: #00ff88; color: #000; }
        .btn-warning { background: #ffaa00; color: #000; }
        .btn-danger { background: #ff4444; color: #fff; }

        .voice-btn {
            background: #ff4444;
            color: white;
            position: relative;
        }
        .voice-btn.listening {
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(255,68,68,0.7); }
            50% { box-shadow: 0 0 0 20px rgba(255,68,68,0); }
        }

        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .search-box input {
            flex: 1;
            padding: 15px;
            font-size: 18px;
            border: 2px solid #333;
            border-radius: 12px;
            background: #2a2a4e;
            color: #fff;
        }
        .search-box input:focus {
            outline: none;
            border-color: #00d4ff;
        }

        .response-box {
            background: #2a2a4e;
            border-radius: 12px;
            padding: 20px;
            min-height: 100px;
        }
        .response-box h3 {
            color: #00d4ff;
            margin-bottom: 10px;
        }
        #response-text {
            font-size: 20px;
            line-height: 1.5;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #333;
            border-top-color: #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-ready { background: #00ff88; }
        .status-processing { background: #ffaa00; }
        .status-error { background: #ff4444; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🦯 Smart Aid Vision Assistant</h1>

        <div class="video-container">
            <img id="video-feed" src="/video_feed" alt="Camera feed">
            <div class="status-bar">
                <span><span id="status-dot" class="status-indicator status-ready"></span><span id="status-text">Ready</span></span>
                <span id="fps">--</span>
            </div>
        </div>

        <div class="controls">
            <button class="btn-primary" onclick="describeScene()" id="btn-describe">
                👁️ What's in front?
            </button>
            <button class="btn-success" onclick="detectObjects()" id="btn-detect">
                📦 Detect Objects
            </button>
            <button class="voice-btn" onclick="toggleVoice()" id="btn-voice">
                🎤 Voice Command
            </button>
        </div>

        <div class="search-box">
            <input type="text" id="search-input" placeholder="Search for something... (e.g., 'door', 'phone', 'chair')"
                   onkeypress="if(event.key==='Enter')searchObject()">
            <button class="btn-warning" onclick="searchObject()">🔍 Find</button>
        </div>

        <div class="response-box">
            <h3>Response</h3>
            <p id="response-text">Click a button or use voice to ask a question.</p>
        </div>
    </div>

    <script>
        let isListening = false;
        let recognition = null;

        // Initialize speech recognition
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';

            recognition.onresult = function(event) {
                const text = event.results[0][0].transcript;
                document.getElementById('search-input').value = text;
                processQuery(text);
            };

            recognition.onend = function() {
                isListening = false;
                document.getElementById('btn-voice').classList.remove('listening');
                document.getElementById('btn-voice').textContent = '🎤 Voice Command';
            };

            recognition.onerror = function(event) {
                console.error('Speech recognition error:', event.error);
                isListening = false;
                document.getElementById('btn-voice').classList.remove('listening');
            };
        }

        function toggleVoice() {
            if (!recognition) {
                alert('Speech recognition not supported in this browser. Try Chrome.');
                return;
            }

            if (isListening) {
                recognition.stop();
            } else {
                isListening = true;
                document.getElementById('btn-voice').classList.add('listening');
                document.getElementById('btn-voice').textContent = '🔴 Listening...';
                recognition.start();
            }
        }

        function setStatus(status, text) {
            const dot = document.getElementById('status-dot');
            const statusText = document.getElementById('status-text');
            dot.className = 'status-indicator status-' + status;
            statusText.textContent = text;
        }

        function setResponse(text) {
            document.getElementById('response-text').textContent = text;
            // Also speak the response
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 1.0;
                speechSynthesis.speak(utterance);
            }
        }

        function setLoading(loading) {
            const buttons = document.querySelectorAll('button');
            buttons.forEach(btn => btn.disabled = loading);
            if (loading) {
                setStatus('processing', 'Processing...');
                document.getElementById('response-text').innerHTML = '<span class="loading"></span> Processing...';
            }
        }

        async function describeScene() {
            setLoading(true);
            try {
                const response = await fetch('/api/describe');
                const data = await response.json();
                setResponse(data.response);
                setStatus('ready', 'Ready');
            } catch (error) {
                setResponse('Error: ' + error.message);
                setStatus('error', 'Error');
            }
            setLoading(false);
        }

        async function detectObjects() {
            setLoading(true);
            try {
                const response = await fetch('/api/detect');
                const data = await response.json();
                setResponse(data.response);
                setStatus('ready', 'Ready');
            } catch (error) {
                setResponse('Error: ' + error.message);
                setStatus('error', 'Error');
            }
            setLoading(false);
        }

        async function searchObject() {
            const query = document.getElementById('search-input').value.trim();
            if (!query) {
                setResponse('Please enter something to search for.');
                return;
            }

            setLoading(true);
            try {
                const response = await fetch('/api/search?q=' + encodeURIComponent(query));
                const data = await response.json();
                setResponse(data.response);
                setStatus('ready', 'Ready');
            } catch (error) {
                setResponse('Error: ' + error.message);
                setStatus('error', 'Error');
            }
            setLoading(false);
        }

        async function processQuery(text) {
            setLoading(true);
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: text})
                });
                const data = await response.json();
                setResponse(data.response);
                setStatus('ready', 'Ready');
            } catch (error) {
                setResponse('Error: ' + error.message);
                setStatus('error', 'Error');
            }
            setLoading(false);
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT') return;

            if (e.key === 'd' || e.key === 'D') describeScene();
            if (e.key === 'o' || e.key === 'O') detectObjects();
            if (e.key === 'v' || e.key === 'V') toggleVoice();
            if (e.key === 's' || e.key === 'S') document.getElementById('search-input').focus();
        });
    </script>
</body>
</html>
"""


def generate_frames():
    """Generate video frames for streaming."""
    global current_frame

    while True:
        if camera is None or not camera.isOpened():
            time.sleep(0.1)
            continue

        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue

        with frame_lock:
            current_frame = frame.copy()

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.033)  # ~30 FPS


@app.route('/')
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    """Video streaming endpoint."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/describe')
def api_describe():
    """Get scene description."""
    global current_frame, query_handler

    if current_frame is None:
        return jsonify({'response': 'No camera feed available.'})

    if query_handler is None:
        return jsonify({'response': 'System not loaded.'})

    with frame_lock:
        frame = current_frame.copy()

    response = query_handler.process("What is in front of me?", frame)
    return jsonify({'response': response})


@app.route('/api/detect')
def api_detect():
    """Detect objects in current frame."""
    global current_frame, query_handler

    if current_frame is None:
        return jsonify({'response': 'No camera feed available.'})

    if query_handler is None:
        return jsonify({'response': 'System not loaded.'})

    with frame_lock:
        frame = current_frame.copy()

    response = query_handler.process("What objects are here?", frame)
    return jsonify({'response': response})


@app.route('/api/search')
def api_search():
    """Search for an object."""
    global current_frame, query_handler

    query = request.args.get('q', '')
    if not query:
        return jsonify({'response': 'Please specify what to search for.'})

    if current_frame is None:
        return jsonify({'response': 'No camera feed available.'})

    if query_handler is None:
        return jsonify({'response': 'System not loaded.'})

    with frame_lock:
        frame = current_frame.copy()

    response = query_handler.process(f"Where is the {query}?", frame)
    return jsonify({'response': response})


@app.route('/api/query', methods=['POST'])
def api_query():
    """Process a natural language query."""
    global current_frame, query_handler

    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({'response': 'Please ask a question.'})

    if current_frame is None:
        return jsonify({'response': 'No camera feed available.'})

    if query_handler is None:
        return jsonify({'response': 'System not loaded.'})

    with frame_lock:
        frame = current_frame.copy()

    response = query_handler.process(query, frame)
    return jsonify({'response': response})


@app.route('/api/count')
def api_count():
    """Count objects."""
    global current_frame, query_handler

    target = request.args.get('target', '')
    if not target:
        return jsonify({'response': 'Please specify what to count.'})

    if current_frame is None:
        return jsonify({'response': 'No camera feed available.'})

    if query_handler is None:
        return jsonify({'response': 'System not loaded.'})

    with frame_lock:
        frame = current_frame.copy()

    response = query_handler.process(f"How many {target}?", frame)
    return jsonify({'response': response})


def init_query_handler(use_florence: bool = True):
    """Initialize the smart query handler."""
    global query_handler

    config = Config()
    config.pipeline.use_florence = use_florence
    config.pipeline.use_yolo_world = True
    config.pipeline.use_depth = True

    query_handler = SmartQueryHandler(config)
    print("Loading models...")
    query_handler.load()
    print("Smart query handler ready!")


def init_camera(source: str | int = 0):
    """Initialize the camera."""
    global camera

    print(f"Opening camera: {source}")
    camera = cv2.VideoCapture(source)

    if not camera.isOpened():
        print(f"Failed to open camera: {source}")
        return False

    print("Camera ready!")
    return True


def run_server(
    source: str | int = 0,
    host: str = "0.0.0.0",
    port: int = 8080,
    use_florence: bool = True,
):
    """Run the web server.

    Args:
        source: Video source (0 for webcam, URL for stream).
        host: Server host.
        port: Server port.
        use_florence: Whether to load Florence-2 (slower but more accurate).
    """
    print("=" * 50)
    print("Smart Aid Web UI")
    print("=" * 50)

    # Initialize components
    init_query_handler(use_florence=use_florence)
    init_camera(source)

    print(f"\nStarting server at http://{host}:{port}")
    print("Open this URL in your browser to use the interface.")
    print("\nKeyboard shortcuts:")
    print("  D - Describe scene")
    print("  O - Detect objects")
    print("  V - Voice command")
    print("  S - Focus search box")
    print("=" * 50)

    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smart Aid Web UI")
    parser.add_argument("--source", default="0", help="Video source (0 for webcam, URL for stream)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--no-florence", action="store_true", help="Disable Florence-2 (faster startup)")

    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    run_server(
        source=source,
        host=args.host,
        port=args.port,
        use_florence=not args.no_florence,
    )
