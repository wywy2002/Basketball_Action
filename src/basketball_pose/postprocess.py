from __future__ import annotations

from collections import defaultdict
from math import cos, hypot, pi, sin
from statistics import median


def hsv_team_feature(hsv: tuple[float, float, float]) -> tuple[float, float, float]:
    hue, saturation, value = hsv
    chroma = saturation / 255.0
    angle = 2.0 * pi * hue / 180.0
    return chroma * cos(angle), chroma * sin(angle), value / 255.0


def median_feature(
    values: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    return tuple(median(value[index] for value in values) for index in range(3))


def is_usable_team_color_sample(record: dict[str, object]) -> bool:
    return bool(
        record.get("keep_after_roi")
        and not record.get("overlap")
        and record.get("uniform_hsv") is not None
        and record.get("final_role") != "referee_candidate"
    )


def _appearance_changed(
    current: tuple[float, float, float], reference: tuple[float, float, float],
) -> bool:
    hue_gap = min(abs(current[0] - reference[0]), 180.0 - abs(current[0] - reference[0]))
    saturation_gap = abs(current[1] - reference[1])
    # White/blue-team material is low saturation; a move to a saturated red shirt
    # is strong identity-switch evidence even when hue itself is unreliable.
    if saturation_gap >= 65.0 and min(current[1], reference[1]) <= 70.0:
        return True
    return min(current[1], reference[1]) >= 80.0 and hue_gap >= 35.0


def assign_appearance_segments(records: list[dict[str, object]]) -> None:
    """Split a track only when clean uniform evidence makes an ID switch likely."""
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record.get("track_id") is not None:
            grouped[(str(record["shot_id"]), str(record["track_id"]))].append(record)

    for track_records in grouped.values():
        track_records.sort(key=lambda item: int(item["frame_id"]))
        segment = 0
        segment_samples: list[tuple[float, float, float]] = []
        for record in track_records:
            if is_usable_team_color_sample(record):
                sample = tuple(float(value) for value in record["uniform_hsv"])
                reference = median_feature(segment_samples) if segment_samples else None
                if reference is not None and len(segment_samples) >= 3 and _appearance_changed(sample, reference):
                    segment += 1
                    segment_samples = []
                    record["identity_review_reason"] = "appearance_shift_possible_id_switch"
                segment_samples.append(sample)
            record["appearance_segment"] = segment


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return hypot(*(a - b for a, b in zip(left, right)))


def cluster_two_teams(
    track_features: dict[int, tuple[float, float, float]],
    confidence_threshold: float = 0.55,
    minimum_cluster_tracks: int = 2,
    maximum_imbalance_ratio: float = 8.0,
) -> dict[int, tuple[str, float]]:
    if len(track_features) < 2:
        return {track_id: ("unknown", 0.0) for track_id in track_features}

    ordered = sorted(track_features.items())
    first = ordered[0][1]
    second = max((feature for _, feature in ordered), key=lambda item: _distance(first, item))
    if _distance(first, second) < 1e-6:
        return {track_id: ("unknown", 0.0) for track_id in track_features}
    centers = [first, second]

    for _ in range(12):
        groups: list[list[tuple[float, float, float]]] = [[], []]
        for feature in track_features.values():
            group = int(_distance(feature, centers[1]) < _distance(feature, centers[0]))
            groups[group].append(feature)
        if not all(groups):
            return {track_id: ("unknown", 0.0) for track_id in track_features}
        updated = [median_feature(group) for group in groups]
        if max(_distance(a, b) for a, b in zip(centers, updated)) < 1e-5:
            centers = updated
            break
        centers = updated

    group_sizes = [0, 0]
    for feature in track_features.values():
        group_sizes[int(_distance(feature, centers[1]) < _distance(feature, centers[0]))] += 1
    smallest, largest = min(group_sizes), max(group_sizes)
    if (
        smallest < minimum_cluster_tracks
        or largest / max(smallest, 1) > maximum_imbalance_ratio
    ):
        return {track_id: ("unknown", 0.0) for track_id in track_features}

    if centers[1] < centers[0]:
        centers.reverse()
    result = {}
    for track_id, feature in track_features.items():
        distances = [_distance(feature, center) for center in centers]
        group = int(distances[1] < distances[0])
        confidence = distances[1 - group] / max(sum(distances), 1e-6)
        team_id = f"team_{group}" if confidence >= confidence_threshold else "unknown"
        result[track_id] = (team_id, round(confidence, 4))
    return result


def annotate_roi_tracks(
    records: list[dict[str, object]],
    max_edge_gap: int = 8,
    min_inside_run: int = 2,
) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record.get("track_id") is not None:
            grouped[str(record["track_id"])].append(record)

    for track_records in grouped.values():
        track_records.sort(key=lambda item: int(item["frame_id"]))
        inside_frames = [
            int(record["frame_id"]) for record in track_records if record["on_court"]
        ]
        longest_run = 0
        current_run = 0
        previous = None
        for frame_id in inside_frames:
            current_run = current_run + 1 if previous is not None and frame_id == previous + 1 else 1
            longest_run = max(longest_run, current_run)
            previous = frame_id

        for record in track_records:
            record["track_length"] = len(track_records)
            record["edge_track_recovery"] = False
            record["edge_recovery_reason"] = None
            record["edge_recovery_confidence"] = 0.0
            record["rejection_reason"] = None
            if record["on_court"]:
                record["roi_status"] = "inside"
                record["keep_after_roi"] = True
                continue

            frame_id = int(record["frame_id"])
            nearest_gap = min((abs(frame_id - value) for value in inside_frames), default=None)
            if longest_run >= min_inside_run and nearest_gap is not None and nearest_gap <= max_edge_gap:
                record["roi_status"] = "outside_recovered"
                record["edge_track_recovery"] = True
                record["edge_recovery_reason"] = "connected_to_inside_track"
                record["edge_recovery_confidence"] = round(
                    1.0 - nearest_gap / (max_edge_gap + 1.0), 4
                )
                record["keep_after_roi"] = True
            else:
                record["roi_status"] = "outside_pending"
                record["keep_after_roi"] = False
                if len(track_records) == 1:
                    record["rejection_reason"] = "single_frame_outside_track"
                elif not inside_frames:
                    record["rejection_reason"] = "outside_track_without_inside_evidence"
                else:
                    record["rejection_reason"] = "outside_track_without_recent_inside_evidence"


def mark_single_frame_uncertain(records: list[dict[str, object]]) -> None:
    for record in records:
        if record.get("track_length") == 1 and record.get("keep_after_roi"):
            record["keep_after_roi"] = False
            record["review_status"] = "uncertain"
            record["rejection_reason"] = "single_frame_inside_uncertain"
        else:
            record["review_status"] = (
                "accepted" if record.get("keep_after_roi") else "rejected"
            )


def mark_short_stationary_boundary_tracks(
    records: list[dict[str, object]],
    image_size: tuple[int, int],
    max_track_length: int = 15,
    max_motion_ratio: float = 0.04,
    max_height_width_ratio: float = 1.1,
    boundary_margin: float = 0.20,
) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["track_id"])].append(record)
    image_width, image_height = image_size
    diagonal = hypot(image_width, image_height)

    for track_records in grouped.values():
        if not 1 < len(track_records) <= max_track_length:
            continue
        boxes = [record["bbox_xyxy"] for record in track_records]
        centers = [((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0) for box in boxes]
        motion = max(hypot(x - centers[0][0], y - centers[0][1]) for x, y in centers)
        shape = median((box[3] - box[1]) / max(box[2] - box[0], 1.0) for box in boxes)
        near_boundary = all(
            x <= image_width * boundary_margin
            or x >= image_width * (1.0 - boundary_margin)
            or y >= image_height * (1.0 - boundary_margin)
            for x, y in centers
        )
        if near_boundary and motion / diagonal <= max_motion_ratio and shape <= max_height_width_ratio:
            for record in track_records:
                record["keep_after_roi"] = False
                record["review_status"] = "rejected"
                record["rejection_reason"] = "short_stationary_boundary_track"
