"""Natural language response builder for Smart Aid."""

from dataclasses import dataclass

from src.detector import Detection
from src.florence import GroundingResult
from src.fusion import Obstacle


@dataclass
class SearchResult:
    """Result from object search."""

    found: bool
    target: str
    distance: float | None = None
    position: str | None = None
    count: int = 0


class ResponseBuilder:
    """Build natural language responses for various queries."""

    # Position descriptions
    POSITION_MAP = {
        "left": "to your left",
        "center": "directly ahead",
        "right": "to your right",
    }

    # Distance descriptions
    DISTANCE_THRESHOLDS = [
        (1.0, "very close"),
        (2.0, "nearby"),
        (3.0, "a few steps away"),
        (5.0, "several steps away"),
        (float("inf"), "far away"),
    ]

    def build_scene_description(
        self,
        caption: str,
        obstacles: list[Obstacle] | None = None,
    ) -> str:
        """Build a scene description response.

        Args:
            caption: Caption from Florence-2.
            obstacles: Optional list of detected obstacles with distances.

        Returns:
            Natural language description.
        """
        if not caption and not obstacles:
            return "I cannot see anything clearly."

        parts = []

        # Add caption if available
        if caption:
            parts.append(caption)

        # Add notable obstacles
        if obstacles:
            # Sort by priority (most urgent first)
            sorted_obstacles = sorted(obstacles, key=lambda o: -o.priority)
            notable = sorted_obstacles[:3]  # Top 3

            if notable:
                obstacle_descriptions = []
                for obs in notable:
                    desc = self._describe_obstacle(obs)
                    obstacle_descriptions.append(desc)

                if obstacle_descriptions:
                    parts.append("I can see: " + "; ".join(obstacle_descriptions))

        return " ".join(parts) if parts else "I cannot see anything clearly."

    def build_search_response(self, result: SearchResult) -> str:
        """Build a response for object search.

        Args:
            result: SearchResult with found status and location info.

        Returns:
            Natural language response.
        """
        if not result.found:
            return f"I could not find {result.target}."

        if result.count > 1:
            return self._build_multiple_found_response(result)

        return self._build_single_found_response(result)

    def _build_single_found_response(self, result: SearchResult) -> str:
        """Build response for single object found."""
        parts = [f"I found {result.target}"]

        if result.distance is not None:
            dist_desc = self._describe_distance(result.distance)
            parts.append(f"{result.distance:.1f} meters away ({dist_desc})")

        if result.position:
            pos_desc = self.POSITION_MAP.get(result.position, result.position)
            parts.append(pos_desc)

        return ", ".join(parts) + "."

    def _build_multiple_found_response(self, result: SearchResult) -> str:
        """Build response for multiple objects found."""
        return f"I found {result.count} {result.target}s. The closest one is {result.distance:.1f} meters {self.POSITION_MAP.get(result.position, result.position)}."

    def build_detection_response(
        self,
        detections: list[Detection] | list[GroundingResult],
        with_distances: dict[int, float] | None = None,
    ) -> str:
        """Build a response listing detected objects.

        Args:
            detections: List of detections.
            with_distances: Optional dict mapping detection index to distance.

        Returns:
            Natural language response.
        """
        if not detections:
            return "I did not detect any objects."

        # Group by class name
        class_counts: dict[str, int] = {}
        for det in detections:
            if isinstance(det, Detection):
                name = det.class_name
            else:
                name = det.phrase
            class_counts[name] = class_counts.get(name, 0) + 1

        # Build description
        items = []
        for name, count in class_counts.items():
            if count > 1:
                items.append(f"{count} {name}s")
            else:
                items.append(f"a {name}")

        if len(items) == 1:
            return f"I detected {items[0]}."
        elif len(items) == 2:
            return f"I detected {items[0]} and {items[1]}."
        else:
            return f"I detected {', '.join(items[:-1])}, and {items[-1]}."

    def build_count_response(self, target: str, count: int) -> str:
        """Build a response for count query.

        Args:
            target: Object being counted.
            count: Number found.

        Returns:
            Natural language response.
        """
        # Normalize target name (remove trailing 's' for singular)
        target_singular = target.rstrip("s") if target.endswith("s") else target
        target_plural = target_singular + "s"

        if count == 0:
            return f"I did not find any {target_plural}."
        elif count == 1:
            return f"I found 1 {target_singular}."
        else:
            return f"I found {count} {target}s."

    def build_alert_response(self, obstacle: Obstacle) -> str:
        """Build an alert response for an obstacle.

        Args:
            obstacle: Obstacle to alert about.

        Returns:
            Alert message.
        """
        pos_desc = self.POSITION_MAP.get(obstacle.position, obstacle.position)
        return f"{obstacle.detection.class_name}, {obstacle.distance:.1f} meters, {pos_desc}"

    def _describe_obstacle(self, obstacle: Obstacle) -> str:
        """Create a short description of an obstacle."""
        pos_desc = self.POSITION_MAP.get(obstacle.position, obstacle.position)
        dist_desc = self._describe_distance(obstacle.distance)
        return f"{obstacle.detection.class_name} {pos_desc} ({dist_desc}, {obstacle.distance:.1f}m)"

    def _describe_distance(self, distance: float) -> str:
        """Get qualitative distance description."""
        for threshold, description in self.DISTANCE_THRESHOLDS:
            if distance < threshold:
                return description
        return "far away"


def build_response(
    query_type: str,
    caption: str = "",
    obstacles: list[Obstacle] | None = None,
    search_result: SearchResult | None = None,
    detections: list[Detection] | list[GroundingResult] | None = None,
    target: str = "",
    count: int = 0,
) -> str:
    """Convenience function to build responses.

    Args:
        query_type: Type of query (describe, search, detect, count).
        caption: Scene caption for describe queries.
        obstacles: Detected obstacles for describe queries.
        search_result: Result for search queries.
        detections: Detections for detect queries.
        target: Target object for count queries.
        count: Count for count queries.

    Returns:
        Natural language response.
    """
    builder = ResponseBuilder()

    if query_type == "describe":
        return builder.build_scene_description(caption, obstacles)
    elif query_type == "search" and search_result:
        return builder.build_search_response(search_result)
    elif query_type == "detect" and detections:
        return builder.build_detection_response(detections)
    elif query_type == "count":
        return builder.build_count_response(target, count)
    else:
        return "I'm not sure how to respond to that."


if __name__ == "__main__":
    from src.detector import Detection
    from src.fusion import Obstacle

    builder = ResponseBuilder()

    # Test scene description
    print("=== Scene Description ===")
    caption = "A living room with a sofa and coffee table."
    print(f"Caption only: {builder.build_scene_description(caption)}")
    print()

    # Test search response
    print("=== Search Response ===")
    result = SearchResult(found=True, target="phone", distance=1.5, position="left")
    print(f"Found: {builder.build_search_response(result)}")

    result = SearchResult(found=False, target="keys")
    print(f"Not found: {builder.build_search_response(result)}")

    result = SearchResult(found=True, target="chair", distance=2.0, position="center", count=3)
    print(f"Multiple: {builder.build_search_response(result)}")
    print()

    # Test detection response
    print("=== Detection Response ===")
    detections = [
        Detection(class_id=0, class_name="person", confidence=0.9, bbox=(0, 0, 100, 200), center=(50, 100)),
        Detection(class_id=1, class_name="chair", confidence=0.8, bbox=(200, 100, 300, 300), center=(250, 200)),
        Detection(class_id=1, class_name="chair", confidence=0.7, bbox=(400, 100, 500, 300), center=(450, 200)),
    ]
    print(f"Detections: {builder.build_detection_response(detections)}")
    print()

    # Test count response
    print("=== Count Response ===")
    print(f"Zero: {builder.build_count_response('door', 0)}")
    print(f"One: {builder.build_count_response('door', 1)}")
    print(f"Many: {builder.build_count_response('chair', 5)}")
