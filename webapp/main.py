"""Smart Aid Navigator - FastAPI Web Application.

A navigation assistant for visually impaired users featuring:
- Real-time video streaming
- Voice-activated commands ("Hey Manso")
- Object detection and scene description
- Depth-based distance estimation
"""

import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webapp.api import routes, video
from webapp.services.camera import camera_service
from webapp.services.navigator import navigator_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    print("=" * 50)
    print("Smart Aid Navigator")
    print("=" * 50)

    # Initialize services
    await navigator_service.initialize()
    await camera_service.initialize()

    print(f"\n-> Open http://localhost:8080 in your browser")
    print("=" * 50)

    yield

    # Shutdown
    print("\nShutting down...")
    await camera_service.cleanup()


app = FastAPI(
    title="Smart Aid Navigator",
    description="AI-powered navigation assistant for visually impaired users",
    version="1.0.0",
    lifespan=lifespan,
)

# Include API routers
app.include_router(routes.router, prefix="/api", tags=["Navigation"])
app.include_router(video.router, tags=["Video"])

# Serve static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the main UI."""
    return FileResponse(str(static_dir / "index.html"))


def run_server(source: int | str = 0, host: str = "0.0.0.0", port: int = 8080):
    """Run the navigation server."""
    import uvicorn

    # Set camera source before starting
    camera_service.set_source(source)

    uvicorn.run(
        "webapp.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Aid Navigator")
    parser.add_argument("--source", default="0", help="Video source (0 for webcam, or URL)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to run on")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run_server(source=source, host=args.host, port=args.port)
