from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..config.auto_config import shot_for_frame
from ..io import write_json
from ..postprocess.court import point_in_normalized_polygon, pose_foot_point
from ..postprocess.postprocess import (
    annotate_roi_tracks,
    assign_appearance_segments,
    cluster_two_teams,
    hsv_team_feature,
    is_usable_team_color_sample,
    mark_short_stationary_boundary_tracks,
    mark_single_frame_uncertain,
    median_feature,
)
from ..postprocess.roles import UniformRules, classify_uniform
from ..detection.sportsmot_detector import SportsMOTDetector
from ..tracking.tracking import OcclusionAwareTracker, PoseDetection, mark_overlaps

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


def _infer_frame(model, detector, sportsmot: bool, frame, width: int, height: int):
    """Return pose data and the detector boxes used for one frame."""
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
    result = []
    for pose_index, (keypoints, scores) in enumerate(zip(keypoints_array, scores_array)):
        if sportsmot:
            bbox, detector_score = player_candidates[pose_index]
        else:
            bbox = _bbox_from_pose(keypoints, scores, width, height)
            detector_score = None
        if bbox is not None:
            result.append((keypoints, scores, bbox, detector_score))
    return result


def _expand_polygon(polygon: list[tuple[float, float]], amount: float = 0.04):
    return [
        (round(min(1.0, max(0.0, x + (amount if x >= 0.5 else -amount))), 4),
         round(min(1.0, max(1e-4, y + (amount if y >= 0.5 else -amount))), 4))
        for x, y in polygon
    ]


def _trial_auto_config(
    source: Path, config: dict[str, object], model, detector, sportsmot: bool,
    width: int, height: int,
) -> dict[str, object]:
    """Probe distributed representative frames and apply at most one cautious correction."""
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(source))
    diagnoses = []
    corrections = []
    for shot in config.get("shots", []):
        polygon = [tuple(point) for point in shot["court_polygon_normalized"]]
        total, outside = 0, 0
        for frame_id in shot.get("representative_frame_ids", []):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, frame = capture.read()
            if not ok:
                continue
            for keypoints, scores, bbox, _ in _infer_frame(
                model, detector, sportsmot, frame, width, height
            ):
                points = tuple(
                    (float(point[0]), float(point[1]), float(np.clip(score, 0.0, 1.0)))
                    for point, score in zip(keypoints, scores)
                )
                foot = pose_foot_point(list(points), bbox)
                total += 1
                outside += not point_in_normalized_polygon(foot, (width, height), polygon)
        outside_fraction = outside / total if total else None
        diagnosis = {
            "shot_id": shot["shot_id"],
            "representative_frame_ids": shot.get("representative_frame_ids", []),
            "trial_pose_candidates": total,
            "outside_roi_candidates": outside,
            "outside_roi_fraction": outside_fraction,
            "status": "ok" if total else "unknown",
            "reason": None if total else "no pose candidates in representative frames",
        }
        # Correct only a strongly evidenced, too-tight ROI.  A broad expansion is
        # intentionally avoided because it would admit spectators/camera crew.
        if (
            not corrections and total >= 6 and outside_fraction is not None
            and outside_fraction >= 0.70
        ):
            expanded = _expand_polygon(polygon)
            shot["court_polygon_normalized"] = [list(point) for point in expanded]
            diagnosis["status"] = "corrected_once"
            diagnosis["reason"] = "representative multi-shot trial indicated a too-tight ROI"
            corrections.append({"shot_id": shot["shot_id"], "action": "expand_roi_0.04"})
        diagnoses.append(diagnosis)
    capture.release()
    return {
        "scope": "representative_frames_from_every_detected_shot",
        "automatic_corrections": corrections,
        "correction_limit": 1,
        "shots": diagnoses,
    }


def run_video(
    input_path: str | Path, output_dir: str | Path, config_path: str | Path,
    model_name: str = "rtmo", device: str = "cpu", max_frames: int | None = None,
    automatic_config: bool = False,
) -> dict[str, object]:
    import cv2
    import numpy as np
    import onnxruntime as ort

    project_root = Path(__file__).resolve().parents[3]
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
    if automatic_config:
        trial = _trial_auto_config(source, config, model, detector, sportsmot, width, height)
        write_json(destination.parent.parent / "configs" / source.stem / "trial_diagnosis.json", trial)
        write_json(destination.parent.parent / "configs" / source.stem / "auto_config.json", config)
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
    current_auto_shot_id = None

    while max_frames is None or frame_id < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        configured_shot = shot_for_frame(config, frame_id) if automatic_config else None
        if configured_shot is not None:
            active_polygon = [tuple(point) for point in configured_shot["court_polygon_normalized"]]
            active_shot_id = str(configured_shot["shot_id"])
        else:
            active_polygon = polygon
            active_shot_id = f"shot_{shot_index:04d}"
        cut_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        detected_cut = (
            not automatic_config
            and previous_cut_frame is not None
            and float(np.mean(cv2.absdiff(cut_frame, previous_cut_frame))) >= cut_threshold
            and frame_id - last_cut_frame >= cut_cooldown
        )
        configured_cut = automatic_config and current_auto_shot_id not in (None, active_shot_id)
        if detected_cut or configured_cut:
            next_track_id = tracker.next_id
            tracker = OcclusionAwareTracker(max_missed=8)
            tracker.next_id = next_track_id
            shot_index += 1
            last_cut_frame = frame_id
        current_auto_shot_id = active_shot_id
        previous_cut_frame = cut_frame
        detections: list[PoseDetection] = []
        for keypoints, scores, bbox, detector_score in _infer_frame(
            model, detector, sportsmot, frame, width, height
        ):
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
                visible_foot, (width, height), active_polygon
            )
            detections.append(PoseDetection(
                bbox_xyxy=bbox,
                keypoints=points,
                score=float(np.mean([point[2] for point in points])),
                appearance_hsv=hsv,
                role=classify_uniform(hsv, rules),
                metadata={"on_court": on_court, "detector_score": detector_score},
            ))

        overlaps = mark_overlaps(detections)
        for detection, overlap in zip(detections, overlaps):
            detection.metadata["overlap"] = overlap
        track_ids = tracker.update(detections)
        track_by_detection = dict(enumerate(track_ids))
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
                "shot_id": active_shot_id,
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
                "roi_config_confidence": (
                    configured_shot.get("roi_estimation", {}).get("confidence", "manual")
                    if configured_shot is not None else "manual"
                ),
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
    assign_appearance_segments(raw_records)

    team_samples: dict[tuple[str, int, int], list[tuple[float, float, float]]] = defaultdict(list)
    for record in raw_records:
        track_id = int(record["track_id"])
        hsv = record["uniform_hsv"]
        record["final_role"] = final_roles.get(track_id)
        if is_usable_team_color_sample(record):
            team_samples[(
                str(record["shot_id"]), track_id, int(record["appearance_segment"])
            )].append(hsv_team_feature(tuple(hsv)))
    minimum_team_samples = int(config.get("team_min_feature_frames", 3))
    team_features: dict[tuple[str, int, int], tuple[float, float, float]] = {}
    team_assignments: dict[tuple[str, int, int], tuple[str, float]] = {}
    team_features = {
        track_key: median_feature(samples)
        for track_key, samples in team_samples.items()
        if len(samples) >= minimum_team_samples
    }
    team_assignments = cluster_two_teams(
        team_features,
        confidence_threshold=float(config.get("team_confidence_threshold", 0.55)),
        minimum_cluster_tracks=int(config.get("team_min_cluster_tracks", 2)),
        maximum_imbalance_ratio=float(config.get("team_max_imbalance_ratio", 8.0)),
    )

    player_records = []
    per_frame_kept: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in raw_records:
        track_id = record["track_id"]
        final_role = final_roles.get(int(track_id)) if track_id is not None else None
        record["final_role"] = final_role
        track_key = (
            str(record["shot_id"]), int(track_id), int(record["appearance_segment"])
        )
        team_id, team_confidence = team_assignments.get(track_key, ("unknown", 0.0))
        record["team_id"] = team_id
        record["team_confidence"] = team_confidence
        record["team_feature"] = list(team_features[track_key]) if track_key in team_features else None
        record["team_assignment_reason"] = (
            "global_cross_shot_cluster_segmented" if team_id != "unknown" else (
                "insufficient_feature_frames" if track_key not in team_features
                else "ambiguous_or_imbalanced_global_clusters"
            )
        )
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
        (
            str(record["shot_id"]), int(record["track_id"]), int(record["appearance_segment"])
        ) for record in player_records
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
        "unknown_team_assignment_reasons": dict(Counter(
            record["team_assignment_reason"] for record in player_records
            if record["team_id"] == "unknown"
        )),
        "shots": shot_index + 1,
        "automatic_config": automatic_config,
        "exclude_referee_by_uniform": exclude_referee_by_uniform,
        "occlusion_events": occlusion_events,
    }
    write_json(destination / "summary.json", summary)
    return summary
