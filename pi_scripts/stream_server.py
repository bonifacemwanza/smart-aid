#!/usr/bin/env python3
"""
Stream Server for Raspberry Pi

This script runs on the Raspberry Pi and streams camera frames
over HTTP/MJPEG to be received by the MacBook client.

Usage:
    python stream_server.py [--port 5000] [--width 640] [--height 480]

Requirements (install on Pi):
    sudo apt update
    sudo apt install python3-opencv python3-flask python3-picamera2
"""

import argparse
import sys
from typing import Generator

import cv2
from flask import Flask, Response, render_template_string

app = Flask(__name__)

camera = None
config = {
    "width": 640,
    "height": 480,
    "fps": 30,
}


def init_camera_opencv() -> bool:
    global camera
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return False
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
        camera.set(cv2.CAP_PROP_FPS, config["fps"])
        return True
    except Exception as e:
        print(f"OpenCV camera init failed: {e}")
        return False


def init_camera_picamera2() -> bool:
    global camera
    try:
        from picamera2 import Picamera2

        camera = Picamera2()
        camera.configure(
            camera.create_preview_configuration(
                main={
                    "format": "RGB888",
                    "size": (config["width"], config["height"]),
                }
            )
        )
        camera.start()
        return True
    except Exception as e:
        print(f"Picamera2 init failed: {e}")
        return False


def get_frame_opencv():
    global camera
    if camera is None:
        return None
    ret, frame = camera.read()
    if not ret:
        return None
    return frame


def get_frame_picamera2():
    global camera
    if camera is None:
        return None
    try:
        frame = camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    except Exception:
        return None


get_frame = get_frame_opencv


def generate_frames() -> Generator[bytes, None, None]:
    while True:
        frame = get_frame()
        if frame is None:
            continue

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Pi Camera Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: #fff;
            margin: 0;
            padding: 20px;
            text-align: center;
        }
        h1 { color: #4CAF50; }
        img {
            max-width: 100%;
            border: 2px solid #4CAF50;
            border-radius: 8px;
        }
        .info {
            margin-top: 20px;
            padding: 10px;
            background: #333;
            border-radius: 4px;
            display: inline-block;
        }
        code {
            background: #222;
            padding: 2px 6px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <h1>Smart Aid - Pi Camera Stream</h1>
    <img src="/video_feed" alt="Camera Stream">
    <div class="info">
        <p>Resolution: {{ width }}x{{ height }}</p>
        <p>Stream URL: <code>http://{{ host }}:{{ port }}/video_feed</code></p>
    </div>
</body>
</html>
"""


@app.route("/")
def index():
    import socket

    hostname = socket.gethostname()
    try:
        ip = socket.gethostbyname(hostname)
    except Exception:
        ip = "localhost"

    return render_template_string(
        INDEX_HTML,
        width=config["width"],
        height=config["height"],
        host=ip,
        port=config.get("port", 5000),
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/health")
def health():
    return {"status": "ok", "camera": camera is not None}


def main() -> int:
    global get_frame, config

    parser = argparse.ArgumentParser(description="Pi Camera Stream Server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--camera",
        choices=["opencv", "picamera2"],
        default="picamera2",
        help="Camera backend to use",
    )
    args = parser.parse_args()

    config["width"] = args.width
    config["height"] = args.height
    config["fps"] = args.fps
    config["port"] = args.port

    print(f"Initializing camera ({args.camera})...")

    if args.camera == "picamera2":
        if init_camera_picamera2():
            get_frame = get_frame_picamera2
            print("Using Picamera2")
        elif init_camera_opencv():
            print("Picamera2 failed, falling back to OpenCV")
        else:
            print("Failed to initialize camera")
            return 1
    else:
        if not init_camera_opencv():
            print("Failed to initialize OpenCV camera")
            return 1
        print("Using OpenCV")

    print(f"Starting server on port {args.port}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Stream URL: http://0.0.0.0:{args.port}/video_feed")
    print("Press Ctrl+C to stop")

    try:
        app.run(host="0.0.0.0", port=args.port, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if camera is not None:
            if hasattr(camera, "release"):
                camera.release()
            elif hasattr(camera, "stop"):
                camera.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
