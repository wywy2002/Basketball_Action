from __future__ import annotations


def point_in_normalized_polygon(
    point: tuple[float, float], image_size: tuple[int, int],
    polygon: list[tuple[float, float]],
) -> bool:
    """Return whether an image point falls inside a normalized polygon."""
    width, height = image_size
    x, y = point[0] / width, point[1] / height
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        crosses = (y1 > y) != (y2 > y)
        if crosses and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        previous = current
    return inside


def pose_foot_point(
    keypoints: list[tuple[float, float, float]],
    bbox_xyxy: tuple[float, float, float, float],
    confidence_threshold: float = 0.3,
) -> tuple[float, float]:
    ankles = [keypoints[index] for index in (15, 16) if keypoints[index][2] >= confidence_threshold]
    if ankles:
        return (
            sum(point[0] for point in ankles) / len(ankles),
            max(point[1] for point in ankles),
        )
    return ((bbox_xyxy[0] + bbox_xyxy[2]) / 2.0, bbox_xyxy[3])

