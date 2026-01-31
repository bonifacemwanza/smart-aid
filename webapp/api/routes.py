"""API routes for navigation and queries."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from webapp.services.camera import camera_service
from webapp.services.navigator import navigator_service

router = APIRouter()


class NavigateRequest(BaseModel):
    """Request body for navigation query."""

    query: str


class NavigateResponse(BaseModel):
    """Response for navigation query."""

    response: str


class StatusResponse(BaseModel):
    """Response for system status."""

    ready: bool
    model: str


@router.get("/scan", response_model=NavigateResponse)
async def scan():
    """Perform a quick safety scan of the environment."""
    frame = camera_service.get_current_frame()
    if frame is None:
        return NavigateResponse(response="No camera feed available.")

    if not navigator_service.is_ready():
        return NavigateResponse(response="System is loading...")

    response = await navigator_service.quick_scan(frame)
    return NavigateResponse(response=response)


@router.post("/navigate", response_model=NavigateResponse)
async def navigate(request: NavigateRequest):
    """Process a navigation query."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    frame = camera_service.get_current_frame()
    if frame is None:
        return NavigateResponse(response="No camera feed available.")

    if not navigator_service.is_ready():
        return NavigateResponse(response="System is loading...")

    response = await navigator_service.navigate(request.query, frame)
    return NavigateResponse(response=response)


@router.get("/status", response_model=StatusResponse)
async def status():
    """Check system status."""
    return StatusResponse(
        ready=navigator_service.is_ready(),
        model="Florence-2 + YOLO + Depth",
    )
