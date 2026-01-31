"""Navigator service wrapping SmartNavigator for async operations."""

import asyncio
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.config import Config
from src.smart_navigator import SmartNavigator


class NavigatorService:
    """Async wrapper for SmartNavigator.

    Runs blocking ML operations in a thread pool to avoid blocking
    the event loop.
    """

    def __init__(self):
        self._navigator: SmartNavigator | None = None
        self._config: Config | None = None
        self._loaded = False

    async def initialize(self) -> bool:
        """Initialize the navigator with models."""
        print("Loading models...")

        self._config = Config()
        self._config.pipeline.use_florence = True
        self._config.pipeline.use_yolo_world = True
        self._config.pipeline.use_depth = True

        self._navigator = SmartNavigator(self._config)

        # Load models in thread pool (blocking operation)
        await asyncio.to_thread(self._navigator.load)

        self._loaded = True
        print("Navigator ready!")

        return True

    def is_ready(self) -> bool:
        """Check if navigator is loaded and ready."""
        return self._loaded and self._navigator is not None

    async def quick_scan(self, frame: NDArray[np.uint8]) -> str:
        """Perform quick safety scan.

        Args:
            frame: Current camera frame

        Returns:
            Safety scan result as text
        """
        if not self.is_ready():
            return "System is loading..."

        # Run blocking operation in thread pool
        return await asyncio.to_thread(self._navigator.quick_scan, frame)

    async def navigate(self, query: str, frame: NDArray[np.uint8]) -> str:
        """Process navigation query.

        Args:
            query: User's query text
            frame: Current camera frame

        Returns:
            Navigation response as text
        """
        if not self.is_ready():
            return "System is loading..."

        # Run blocking operation in thread pool
        return await asyncio.to_thread(self._navigator.navigate, query, frame)


# Global singleton instance
navigator_service = NavigatorService()
