from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .court import point_in_normalized_polygon, pose_foot_point
from .io import write_json
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

    torso = [keypoints[index] for index in (5, 6, 11, 12) if keypoints[index][2] >= 0.3]
    if len(torso) >= 3:
        xs, ys = [point[0] for point in torso], [point[1] for point in torso]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    else:
        x1, y1, x2, y2 = bbox
        y2 = y1 + (y2 - y1) * 0.55
        x1 += (x2 - x1) * 0.25
        x2 -= (x2 - x1) * 0.25
    x1, x2 = int(max(0, x1)), int(min(frame.shape[1], x2 + 1))
    y1, y2 = int(max(0, y1)), int(min(frame.shape[0], y2 + 1))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    pixels = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    if not len(pixels):
        return None
    median = np.median(pixels, axis=0)
    return tuple(float(value) for value in median)


def _draw_pose(frame, detection: PoseDetection, track_id: int | None) -> None:
    import cv2

    if not detection.metadata["on_court"]:
        color = (0, 0, 255)
    elif detection.role == "referee_candidate":
        color = (0, 165, 255)
    else:
        color = (0, 220, 0)
    for left, right in COCO_EDGES:
        a, b = detection.keypoints[left], detection.keypoints[right]
        if a[2] >= 0.3 and b[2] >= 0.3:
            cv2.line(frame, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])), color, 2)
    x1, y1, x2, y2 = map(round, detection.bbox_xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    label = detection.role if track_id is None else f"ID {track_id} {detection.role}"
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
    writer = cv2.VideoWriter(
        str(destination / "pose2d_audit.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )
    tracker = OcclusionAwareTracker(max_missed=8)
    raw_records: list[dict[str, object]] = []
    occlusion_events: list[dict[str, object]] = []
    role_votes: dict[int, Counter[str]] = defaultdict(Counter)
    frame_id = 0

    while max_frames is None or frame_id < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
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

        court_indices = [index for index, item in enumerate(detections) if item.metadata["on_court"]]
        court_detections = [detections[index] for index in court_indices]
        track_ids = tracker.update(court_detections)
        track_by_detection = dict(zip(court_indices, track_ids))
        overlaps = mark_overlaps(court_detections)
        overlap_by_detection = dict(zip(court_indices, overlaps))
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
            reasons = []
            if not detection.metadata["on_court"]:
                reasons.append("outside_court")
            if exclude_referee_by_uniform and detection.role == "referee_candidate":
                reasons.append("referee_candidate")
            if track_id is not None:
                role_votes[track_id][detection.role] += 1
            raw_records.append({
                "video_id": source.stem,
                "shot_id": "shot_0000",
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
                "overlap": overlap_by_detection.get(index, False),
                "filter_reasons": reasons,
            })
            _draw_pose(frame, detection, track_id)
        writer.write(frame)
        frame_id += 1
        if frame_id % 30 == 0:
            print(f"processed {frame_id} frames")

    capture.release()
    writer.release()

    final_roles = {
        track_id: votes.most_common(1)[0][0] for track_id, votes in role_votes.items()
    }
    player_records = []
    per_frame_kept: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in raw_records:
        track_id = record["track_id"]
        final_role = final_roles.get(int(track_id)) if track_id is not None else None
        record["final_role"] = final_role
        if record["on_court"] and (
            not exclude_referee_by_uniform or final_role != "referee_candidate"
        ):
            per_frame_kept[int(record["frame_id"])].append(record)
    for records in per_frame_kept.values():
        player_records.extend(sorted(records, key=lambda item: item["pose_score"], reverse=True)[:10])

    write_json(destination / "detections_audit.json", {"records": raw_records})
    write_json(destination / "pose2d_players.json", {"records": player_records})
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
        "exclude_referee_by_uniform": exclude_referee_by_uniform,
        "occlusion_events": occlusion_events,
    }
    write_json(destination / "summary.json", summary)
    return summary
