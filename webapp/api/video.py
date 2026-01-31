"""Video streaming endpoint."""

import asyncio
from typing import AsyncGenerator

import cv2
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from webapp.services.camera import camera_service

router = APIRouter()


async def generate_frames() -> AsyncGenerator[bytes, None]:
    """Generate MJPEG frames for video streaming.

    Yields frames at approximately 30 FPS with JPEG compression.
    """
    while True:
        frame = camera_service.get_current_frame()

        if frame is None:
            await asyncio.sleep(0.01)
            continue

        # Encode frame as JPEG
        _, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        # Yield as multipart frame
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

        # Target ~30 FPS
        await asyncio.sleep(0.033)


@router.get("/video_feed")
async def video_feed():
    """Stream MJPEG video from camera."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
