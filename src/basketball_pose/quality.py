from __future__ import annotations

from math import hypot

from .schema import Pose2DRecord

ACTION_KEYPOINT_INDICES = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)


def inspect_track(
    track: list[Pose2DRecord], confidence_threshold: float = 0.3,
    jump_threshold: float = 0.35,
) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    for record in track:
        x1, y1, x2, y2 = record.bbox_xyxy
        low = sum(
            record.keypoints_2d[index][2] < confidence_threshold
            for index in ACTION_KEYPOINT_INDICES
        )
        outside = sum(
            score >= confidence_threshold and not (x1 <= x <= x2 and y1 <= y <= y2)
            for x, y, score in record.keypoints_2d
        )
        if low:
            issues.append({"frame_id": record.frame_id, "type": "low_confidence", "count": low})
        if outside:
            issues.append({"frame_id": record.frame_id, "type": "outside_bbox", "count": outside})

    for previous, current in zip(track, track[1:]):
        diagonal = hypot(
            previous.bbox_xyxy[2] - previous.bbox_xyxy[0],
            previous.bbox_xyxy[3] - previous.bbox_xyxy[1],
        )
        if diagonal == 0:
            continue
        visible_jumps = [
            hypot(b[0] - a[0], b[1] - a[1]) / diagonal
            for a, b in zip(previous.keypoints_2d, current.keypoints_2d)
            if a[2] >= confidence_threshold and b[2] >= confidence_threshold
        ]
        if not visible_jumps:
            continue
        largest_jump = max(visible_jumps)
        if largest_jump > jump_threshold:
            issues.append({
                "frame_id": current.frame_id,
                "type": "keypoint_jump",
                "normalized_distance": round(largest_jump, 4),
            })
    return issues
