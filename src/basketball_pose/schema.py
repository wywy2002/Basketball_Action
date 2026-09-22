from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

COCO17_COUNT = 17


@dataclass(frozen=True)
class Pose2DRecord:
    video_id: str
    shot_id: str
    frame_id: int
    timestamp: float
    track_id: str
    image_size: tuple[int, int]
    bbox_xyxy: tuple[float, float, float, float]
    keypoints_2d: tuple[tuple[float, float, float], ...]
    pose_schema: str = "coco17"

    def __post_init__(self) -> None:
        if not self.video_id or not self.shot_id or not self.track_id:
            raise ValueError("video_id, shot_id and track_id must be non-empty")
        if self.frame_id < 0 or self.timestamp < 0:
            raise ValueError("frame_id and timestamp must be non-negative")
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must be positive")
        x1, y1, x2, y2 = self.bbox_xyxy
        if not all(isfinite(value) for value in self.bbox_xyxy) or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must be finite with x2>x1 and y2>y1")
        if self.pose_schema != "coco17" or len(self.keypoints_2d) != COCO17_COUNT:
            raise ValueError("keypoints_2d must contain exactly 17 COCO keypoints")
        for x, y, score in self.keypoints_2d:
            if not all(isfinite(value) for value in (x, y, score)):
                raise ValueError("keypoints must be finite")
            if not 0.0 <= score <= 1.0:
                raise ValueError("keypoint confidence must be within [0, 1]")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Pose2DRecord":
        return cls(
            video_id=str(value["video_id"]),
            shot_id=str(value["shot_id"]),
            frame_id=int(value["frame_id"]),
            timestamp=float(value["timestamp"]),
            track_id=str(value["track_id"]),
            image_size=tuple(int(v) for v in value["image_size"]),
            bbox_xyxy=tuple(float(v) for v in value["bbox_xyxy"]),
            keypoints_2d=tuple(
                tuple(float(v) for v in keypoint) for keypoint in value["keypoints_2d"]
            ),
            pose_schema=str(value.get("pose_schema", "coco17")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "shot_id": self.shot_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "track_id": self.track_id,
            "image_size": list(self.image_size),
            "bbox_xyxy": list(self.bbox_xyxy),
            "keypoints_2d": [list(keypoint) for keypoint in self.keypoints_2d],
            "pose_schema": self.pose_schema,
        }

