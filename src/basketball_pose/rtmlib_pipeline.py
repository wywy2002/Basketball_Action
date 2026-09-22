from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .court import point_in_normalized_polygon, pose_foot_point
from .io import write_json
from .postprocess import (
    annotate_roi_tracks,
    cluster_two_teams,
    hsv_team_feature,
    mark_short_stationary_boundary_tracks,
    mark_single_frame_uncertain,
    median_feature,
)
from .roles import UniformRules, classify_uniform
from .sportsmot_detector import SportsMOTDetector
from .tracking import OcclusionAwareTracker, PoseDetection, mark_overlaps

COCO_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)
_DLL_HANDLES: list[object] = []


def configure_cuda_dlls() -> None:
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia_root.is_dir():
        return
    bin_dirs = [path / "bin" for path in nvidia_root.iterdir() if (path / "bin").is_dir()]
    os.environ["PATH"] = os.pathsep.join(map(str, bin_dirs)) + os.pathsep + os.environ["PATH"]
    if hasattr(os, "add_dll_directory"):
        _DLL_HANDLES.extend(os.add_dll_directory(str(path)) for path in bin_dirs)


def _bbox_from_pose(keypoints, scores, width: int, height: int):
    visible = [
        (float(point[0]), float(point[1]))
        for point, score in zip(keypoints, scores) if float(score) >= 0.3
    ]
    if len(visible) < 3:
        return None
    xs, ys = zip(*visible)
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    pad_x, pad_y = max(4.0, (x2 - x1) * 0.15), max(4.0, (y2 - y1) * 0.10)
    return (
        max(0.0, x1 - pad_x), max(0.0, y1 - pad_y),
        min(float(width - 1), x2 + pad_x), min(float(height - 1), y2 + pad_y),
    )


def _uniform_hsv(frame, keypoints, bbox):
    import cv2
    import numpy as np

    torso = [keypoints[index] for index in (5, 6, 12, 11)]
    if all(point[2] >= 0.3 for point in torso):
        polygon = np.array([[point[0], point[1]] for point in torso], dtype=np.int32)
        x1, y1, width, height = cv2.boundingRect(polygon)
        x2, y2 = x1 + width, y1 + height
    else:
        x1, y1, x2, y2 = bbox
        y2 = y1 + (y2 - y1) * 0.55
        x1 += (x2 - x1) * 0.25
        x2 -= (x2 - x1) * 0.25
    x1, x2 = int(max(0, x1)), int(min(frame.shape[1], x2 + 1))
    y1, y2 = int(max(0, y1)), int(min(frame.shape[0], y2 + 1))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    if all(point[2] >= 0.3 for point in torso):
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        local_polygon = polygon - np.array([x1, y1])
        cv2.fillConvexPoly(mask, local_polygon, 255)
        pixels = crop[mask > 0]
    else:
        pixels = crop.reshape(-1, 3)
    if not len(pixels):
        return None
    median = np.median(pixels, axis=0)
    return tuple(float(value) for value in median)


def _draw_record(frame, record: dict[str, object]) -> None:
    import cv2

    roi_status = record["roi_status"]
    if roi_status == "outside_pending":
        color = (0, 0, 255)
        state_label = "OUTSIDE_PENDING"
    elif record["final_role"] == "referee_candidate":
        color = (0, 165, 255)
        state_label = "REFEREE"
    elif record["rejection_reason"]:
        color = (160, 160, 160)
        state_label = "UNCERTAIN" if record["review_status"] == "uncertain" else "REJECTED"
    elif record["team_id"] == "team_0":
        color = (255, 120, 0)
        state_label = "team_0"
    elif record["team_id"] == "team_1":
        color = (0, 220, 255)
        state_label = "team_1"
    else:
        color = (0, 220, 0)
        state_label = "unknown"
    keypoints = record["keypoints_2d"]
    for left, right in COCO_EDGES:
        a, b = keypoints[left], keypoints[right]
        if a[2] >= 0.3 and b[2] >= 0.3:
            cv2.line(frame, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])), color, 2)
    x1, y1, x2, y2 = map(round, record["bbox_xyxy"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    confidence = float(record["team_confidence"])
    edge = " EDGE" if record["edge_track_recovery"] else ""
    label = f"ID {record['track_id']} {state_label}{edge} {confidence:.2f}"
    cv2.putText(frame, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def run_video(
    input_path: str | Path, output_dir: str | Path, config_path: str | Path,
    model_name: str = "rtmo", device: str = "cpu", max_frames: int | None = None,
) -> dict[str, object]:
    import cv2
    import numpy as np
    import onnxruntime as ort

    project_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / "checkpoints"))
    sportsmot = model_name == "sportsmot-rtmpose"
    if sportsmot:
        # Load PyTorch's bundled CUDA libraries before adding the ONNX Runtime
        # NVIDIA wheels to PATH; otherwise Windows can select an incompatible cuDNN.
        import torch  # noqa: F401
    configure_cuda_dlls()
    ort.preload_dlls(directory="")
    from rtmlib import Body, RTMPose

    source = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    polygon = [tuple(point) for point in config["court_polygon_normalized"]]
    rules = UniformRules(**config.get("uniform_rules", {}))
    exclude_referee_by_uniform = config.get("exclude_referee_by_uniform", True)

    if sportsmot:
        detector = SportsMOTDetector(
            project_root,
            project_root / "checkpoints" / "sportsmot" / "yolox_x_sports_mix.pth.tar",
            device=device,
        )
        model = RTMPose(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
            model_input_size=(288, 384), backend="onnxruntime", device=device,
        )
    else:
        detector = None
        model = Body(
            pose="rtmo" if model_name == "rtmo" else None,
            mode="balanced", backend="onnxruntime", device=device,
        )
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = OcclusionAwareTracker(max_missed=8)
    raw_records: list[dict[str, object]] = []
    occlusion_events: list[dict[str, object]] = []
    role_votes: dict[int, Counter[str]] = defaultdict(Counter)
    frame_id = 0
    shot_index = 0
    previous_cut_frame = None
    cut_threshold = float(config.get("shot_cut_threshold", 50.0))
    cut_cooldown = int(config.get("shot_cut_cooldown", 15))
    last_cut_frame = -cut_cooldown

    while max_frames is None or frame_id < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        cut_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        if (
            previous_cut_frame is not None
            and float(np.mean(cv2.absdiff(cut_frame, previous_cut_frame))) >= cut_threshold
            and frame_id - last_cut_frame >= cut_cooldown
        ):
            next_track_id = tracker.next_id
            tracker = OcclusionAwareTracker(max_missed=8)
            tracker.next_id = next_track_id
            shot_index += 1
            last_cut_frame = frame_id
        previous_cut_frame = cut_frame
        if sportsmot:
            player_candidates = detector(frame)
            candidate_boxes = [bbox for bbox, _ in player_candidates]
            if candidate_boxes:
                keypoints_array, scores_array = model(frame, bboxes=candidate_boxes)
            else:
                keypoints_array, scores_array = [], []
        else:
            player_candidates = []
            keypoints_array, scores_array = model(frame)
        detections: list[PoseDetection] = []
        for pose_index, (keypoints, scores) in enumerate(zip(keypoints_array, scores_array)):
            if sportsmot:
                bbox, detector_score = player_candidates[pose_index]
            else:
                bbox = _bbox_from_pose(keypoints, scores, width, height)
                detector_score = None
            if bbox is None:
                continue
            points = tuple(
                (float(point[0]), float(point[1]), float(np.clip(score, 0.0, 1.0)))
                for point, score in zip(keypoints, scores)
            )
            hsv = _uniform_hsv(frame, points, bbox)
            foot_x, foot_y = pose_foot_point(list(points), bbox)
            visible_foot = (
                float(np.clip(foot_x, 0.0, width - 1.0)),
                float(np.clip(foot_y, 0.0, height - 1.0)),
            )
            on_court = point_in_normalized_polygon(
                visible_foot, (width, height), polygon
            )
            detections.append(PoseDetection(
                bbox_xyxy=bbox,
                keypoints=points,
                score=float(np.mean([point[2] for point in points])),
                appearance_hsv=hsv,
                role=classify_uniform(hsv, rules),
                metadata={"on_court": on_court, "detector_score": detector_score},
            ))

        track_ids = tracker.update(detections)
        track_by_detection = dict(enumerate(track_ids))
        overlaps = mark_overlaps(detections)
        overlap_by_detection = dict(enumerate(overlaps))
        missing = [track.track_id for track in tracker.tracks.values() if track.missed > 0]
        if missing:
            occlusion_events.append({"frame_id": frame_id, "missing_track_ids": missing})

        for index, detection in enumerate(detections):
            track_id = track_by_detection.get(index)
            recovered_low_confidence = bool(
                track_id is not None
                and detection.metadata.get("detector_score") is not None
                and detection.metadata["detector_score"] < 0.30
                and tracker.tracks[track_id].age > 2
            )
            if track_id is not None:
                role_votes[track_id][detection.role] += 1
            raw_records.append({
                "video_id": source.stem,
                "shot_id": f"shot_{shot_index:04d}",
                "frame_id": frame_id,
                "timestamp": frame_id / fps,
                "track_id": str(track_id) if track_id is not None else None,
                "image_size": [width, height],
                "bbox_xyxy": list(detection.bbox_xyxy),
                "keypoints_2d": [list(point) for point in detection.keypoints],
                "pose_schema": "coco17",
                "pose_score": detection.score,
                "player_detector_score": detection.metadata.get("detector_score"),
                "recovered_low_confidence": recovered_low_confidence,
                "uniform_hsv": list(detection.appearance_hsv) if detection.appearance_hsv else None,
                "observed_role": detection.role,
                "on_court": detection.metadata["on_court"],
                "roi_status": "inside" if detection.metadata["on_court"] else "outside_pending",
                "overlap": overlap_by_detection.get(index, False),
                "filter_reasons": [],
            })
        frame_id += 1
        if frame_id % 30 == 0:
            print(f"processed {frame_id} frames")

    capture.release()

    final_roles = {
        track_id: votes.most_common(1)[0][0] for track_id, votes in role_votes.items()
    }
    annotate_roi_tracks(
        raw_records,
        max_edge_gap=int(config.get("edge_max_frame_gap", 8)),
        min_inside_run=int(config.get("edge_min_inside_run", 2)),
    )
    mark_single_frame_uncertain(raw_records)
    mark_short_stationary_boundary_tracks(
        raw_records,
        (width, height),
        max_track_length=int(config.get("nonplayer_max_track_length", 15)),
    )

    team_samples: dict[str, dict[int, list[tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in raw_records:
        track_id = int(record["track_id"])
        final_role = final_roles.get(track_id)
        hsv = record["uniform_hsv"]
        if record["keep_after_roi"] and final_role != "referee_candidate" and hsv is not None:
            team_samples[str(record["shot_id"])][track_id].append(hsv_team_feature(tuple(hsv)))
    minimum_team_samples = int(config.get("team_min_feature_frames", 3))
    team_features: dict[tuple[str, int], tuple[float, float, float]] = {}
    team_assignments: dict[tuple[str, int], tuple[str, float]] = {}
    for shot_id, shot_samples in team_samples.items():
        shot_features = {
            track_id: median_feature(samples)
            for track_id, samples in shot_samples.items()
            if len(samples) >= minimum_team_samples
        }
        shot_assignments = cluster_two_teams(
            shot_features,
            confidence_threshold=float(config.get("team_confidence_threshold", 0.55)),
        )
        team_features.update(
            {(shot_id, track_id): feature for track_id, feature in shot_features.items()}
        )
        team_assignments.update(
            {(shot_id, track_id): value for track_id, value in shot_assignments.items()}
        )

    player_records = []
    per_frame_kept: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in raw_records:
        track_id = record["track_id"]
        final_role = final_roles.get(int(track_id)) if track_id is not None else None
        record["final_role"] = final_role
        track_key = (str(record["shot_id"]), int(track_id))
        team_id, team_confidence = team_assignments.get(track_key, ("unknown", 0.0))
        record["team_id"] = team_id
        record["team_confidence"] = team_confidence
        record["team_feature"] = list(team_features[track_key]) if track_key in team_features else None
        if exclude_referee_by_uniform and final_role == "referee_candidate":
            record["keep_after_roi"] = False
            record["review_status"] = "rejected"
            record["rejection_reason"] = "referee_candidate"
        record["filter_reasons"] = (
            [record["rejection_reason"]] if record["rejection_reason"] else []
        )
        if record["keep_after_roi"]:
            per_frame_kept[int(record["frame_id"])].append(record)
    for records in per_frame_kept.values():
        player_records.extend(sorted(records, key=lambda item: item["pose_score"], reverse=True)[:10])

    records_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in raw_records:
        records_by_frame[int(record["frame_id"])].append(record)
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(
        str(destination / "pose2d_audit.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )
    for audit_frame_id in range(frame_id):
        ok, frame = capture.read()
        if not ok:
            break
        for record in records_by_frame[audit_frame_id]:
            _draw_record(frame, record)
        writer.write(frame)
    capture.release()
    writer.release()

    write_json(destination / "detections_audit.json", {"records": raw_records})
    write_json(destination / "pose2d_players.json", {"records": player_records})
    kept_track_ids = {
        (str(record["shot_id"]), int(record["track_id"])) for record in player_records
    }
    edge_track_ids = {
        int(record["track_id"])
        for record in player_records if record["edge_track_recovery"]
    }
    summary = {
        "video": str(source),
        "model": model_name,
        "device": device,
        "frames": frame_id,
        "raw_detections": len(raw_records),
        "player_records": len(player_records),
        "tracks": len(role_votes),
        "final_roles": {str(key): value for key, value in final_roles.items()},
        "occlusion_event_count": len(occlusion_events),
        "low_confidence_recoveries": sum(
            bool(record["recovered_low_confidence"]) for record in raw_records
        ),
        "outside_roi_records": sum(not bool(record["on_court"]) for record in raw_records),
        "edge_track_recoveries": sum(
            bool(record["edge_track_recovery"]) for record in raw_records
        ),
        "edge_recovered_tracks": len(edge_track_ids),
        "single_frame_outside_exclusions": sum(
            record["rejection_reason"] == "single_frame_outside_track"
            for record in raw_records
        ),
        "single_frame_inside_uncertain": sum(
            record["rejection_reason"] == "single_frame_inside_uncertain"
            for record in raw_records
        ),
        "short_stationary_boundary_exclusions": len({
            record["track_id"]
            for record in raw_records
            if record["rejection_reason"] == "short_stationary_boundary_track"
        }),
        "referee_excluded_tracks": sum(
            role == "referee_candidate" for role in final_roles.values()
        ) if exclude_referee_by_uniform else 0,
        "team_tracks": dict(Counter(
            team_assignments.get(track_id, ("unknown", 0.0))[0]
            for track_id in kept_track_ids
        )),
        "shots": shot_index + 1,
        "exclude_referee_by_uniform": exclude_referee_by_uniform,
        "occlusion_events": occlusion_events,
    }
    write_json(destination / "summary.json", summary)
    return summary
