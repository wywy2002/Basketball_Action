from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, hypot

Keypoint = tuple[float, float, float]
BBox = tuple[float, float, float, float]


@dataclass
class PoseDetection:
    bbox_xyxy: BBox
    keypoints: tuple[Keypoint, ...]
    score: float
    appearance_hsv: tuple[float, float, float] | None = None
    role: str = "unknown"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class TrackState:
    track_id: int
    bbox_xyxy: BBox
    keypoints: tuple[Keypoint, ...]
    appearance_hsv: tuple[float, float, float] | None
    missed: int = 0
    age: int = 1


def bbox_iou(left: BBox, right: BBox) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _pose_similarity(left: TrackState, right: PoseDetection) -> float:
    diagonal = hypot(
        left.bbox_xyxy[2] - left.bbox_xyxy[0],
        left.bbox_xyxy[3] - left.bbox_xyxy[1],
    )
    distances = [
        hypot(a[0] - b[0], a[1] - b[1]) / diagonal
        for a, b in zip(left.keypoints, right.keypoints)
        if a[2] >= 0.3 and b[2] >= 0.3
    ]
    return exp(-4.0 * sum(distances) / len(distances)) if distances and diagonal else 0.0


def _appearance_similarity(
    left: tuple[float, float, float] | None,
    right: tuple[float, float, float] | None,
) -> float:
    if left is None or right is None:
        return 0.5
    hue_distance = min(abs(left[0] - right[0]), 180.0 - abs(left[0] - right[0])) / 90.0
    sv_distance = (abs(left[1] - right[1]) + abs(left[2] - right[2])) / 510.0
    return max(0.0, 1.0 - (hue_distance + sv_distance) / 2.0)


class OcclusionAwareTracker:
    """Small pose tracker that keeps unmatched identities through short occlusions."""

    def __init__(self, max_missed: int = 8, match_threshold: float = 0.25):
        self.max_missed = max_missed
        self.match_threshold = match_threshold
        self.next_id = 0
        self.tracks: dict[int, TrackState] = {}

    def update(self, detections: list[PoseDetection]) -> list[int]:
        candidates = []
        for track_id, track in self.tracks.items():
            for detection_id, detection in enumerate(detections):
                score = (
                    0.35 * bbox_iou(track.bbox_xyxy, detection.bbox_xyxy)
                    + 0.45 * _pose_similarity(track, detection)
                    + 0.20 * _appearance_similarity(track.appearance_hsv, detection.appearance_hsv)
                )
                if score >= self.match_threshold:
                    candidates.append((score, track_id, detection_id))

        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, track_id, detection_id in sorted(candidates, reverse=True):
            if track_id not in used_tracks and detection_id not in assignments:
                assignments[detection_id] = track_id
                used_tracks.add(track_id)

        for track in self.tracks.values():
            track.missed += 1
        for detection_id, detection in enumerate(detections):
            track_id = assignments.get(detection_id)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                assignments[detection_id] = track_id
            self.tracks[track_id] = TrackState(
                track_id=track_id,
                bbox_xyxy=detection.bbox_xyxy,
                keypoints=detection.keypoints,
                appearance_hsv=detection.appearance_hsv,
                missed=0,
                age=self.tracks.get(track_id, TrackState(
                    track_id, detection.bbox_xyxy, detection.keypoints,
                    detection.appearance_hsv,
                )).age + 1,
            )
        self.tracks = {
            track_id: track for track_id, track in self.tracks.items()
            if track.missed <= self.max_missed
        }
        return [assignments[index] for index in range(len(detections))]


def mark_overlaps(detections: list[PoseDetection], threshold: float = 0.15) -> list[bool]:
    overlaps = [False] * len(detections)
    for left in range(len(detections)):
        for right in range(left + 1, len(detections)):
            if bbox_iou(detections[left].bbox_xyxy, detections[right].bbox_xyxy) >= threshold:
                overlaps[left] = overlaps[right] = True
    return overlaps

