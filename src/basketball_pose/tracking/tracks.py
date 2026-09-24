from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from ..schema import Pose2DRecord

TrackKey = tuple[str, str, str]


def group_tracks(records: Iterable[Pose2DRecord]) -> dict[TrackKey, list[Pose2DRecord]]:
    tracks: dict[TrackKey, list[Pose2DRecord]] = defaultdict(list)
    for record in records:
        tracks[(record.video_id, record.shot_id, record.track_id)].append(record)
    for key, track in tracks.items():
        track.sort(key=lambda item: item.frame_id)
        frame_ids = [item.frame_id for item in track]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError(f"duplicate frame in track {key}")
    return dict(tracks)


def interpolate_short_gaps(
    track: list[Pose2DRecord], max_missing_frames: int = 2
) -> list[Pose2DRecord]:
    if max_missing_frames < 0:
        raise ValueError("max_missing_frames must be non-negative")
    if not track:
        return []
    result = [track[0]]
    for previous, current in zip(track, track[1:]):
        missing = current.frame_id - previous.frame_id - 1
        if 0 < missing <= max_missing_frames:
            for offset in range(1, missing + 1):
                ratio = offset / (missing + 1)
                result.append(_interpolate_record(previous, current, ratio))
        result.append(current)
    return result


def _interpolate_record(
    left: Pose2DRecord, right: Pose2DRecord, ratio: float
) -> Pose2DRecord:
    def lerp(a: float, b: float) -> float:
        return a + (b - a) * ratio

    frame_id = round(lerp(left.frame_id, right.frame_id))
    bbox = tuple(lerp(a, b) for a, b in zip(left.bbox_xyxy, right.bbox_xyxy))
    keypoints = tuple(
        (lerp(a[0], b[0]), lerp(a[1], b[1]), min(a[2], b[2]))
        for a, b in zip(left.keypoints_2d, right.keypoints_2d)
    )
    return Pose2DRecord(
        video_id=left.video_id,
        shot_id=left.shot_id,
        frame_id=frame_id,
        timestamp=lerp(left.timestamp, right.timestamp),
        track_id=left.track_id,
        image_size=left.image_size,
        bbox_xyxy=bbox,
        keypoints_2d=keypoints,
    )

