from __future__ import annotations

"""Full-video, video-local configuration analysis for basketball broadcasts."""

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median


@dataclass(frozen=True)
class Shot:
    start_frame: int
    end_frame: int
    representative_frame_ids: tuple[int, ...]


def _normalized_polygon_from_frame(frame) -> tuple[list[list[float]], dict[str, object]]:
    """Estimate a conservative court trapezoid from this frame only.

    The estimate looks for long near-horizontal court/paint lines.  It deliberately
    reports low confidence instead of borrowing an ROI from a different video.
    """
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=max(35, width // 12),
        minLineLength=max(50, width // 4), maxLineGap=max(12, width // 30),
    )
    candidates: list[float] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if abs(y2 - y1) <= max(3, height * 0.025) and abs(x2 - x1) >= width * 0.30:
                y = (y1 + y2) / 2.0 / height
                if 0.14 <= y <= 0.78:
                    candidates.append(float(y))
    # A line found in the upper-middle picture is useful evidence for the far
    # boundary.  Bottom remains deliberately conservative to retain edge players.
    upper = min(candidates) if candidates else 0.32
    upper = min(0.58, max(0.16, upper - 0.035))
    lower = 0.94
    inset = min(0.20, max(0.04, 0.15 - (upper - 0.20) * 0.20))
    polygon = [
        [round(inset, 4), round(upper, 4)],
        [round(1.0 - inset, 4), round(upper, 4)],
        [1.0, lower], [0.0, lower],
    ]
    confidence = "medium" if len(candidates) >= 2 else "low"
    return polygon, {
        "method": "horizontal_line_geometry",
        "confidence": confidence,
        "reason": (
            "court boundary candidates detected" if candidates
            else "no reliable court boundary line; conservative video-local fallback"
        ),
        "horizontal_line_candidates": len(candidates),
        "frame_size": [width, height],
    }


def _scan_shots(source: Path) -> tuple[int, float, int, int, list[Shot]]:
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    differences: list[float] = []
    previous = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        if previous is not None:
            differences.append(float(np.mean(cv2.absdiff(gray, previous))))
        previous = gray
    capture.release()
    frame_count = len(differences) + (1 if previous is not None else 0)
    if frame_count == 0:
        raise RuntimeError(f"no frames in {source}")
    baseline = median(differences) if differences else 0.0
    mad = median(abs(value - baseline) for value in differences) if differences else 0.0
    threshold = max(25.0, baseline + 8.0 * max(mad, 1.0))
    minimum_shot_frames = max(15, round(fps * 0.5))
    cuts = [
        index + 1 for index, value in enumerate(differences)
        if value >= threshold
        and (index + 1) >= minimum_shot_frames
    ]
    accepted_cuts: list[int] = []
    for cut in cuts:
        if not accepted_cuts or cut - accepted_cuts[-1] >= minimum_shot_frames:
            accepted_cuts.append(cut)
    boundaries = [0, *accepted_cuts, frame_count]
    shots = []
    for start, end in zip(boundaries, boundaries[1:]):
        # Three well-spaced representatives are enough for geometry while keeping
        # trial inference bounded and distributed across every detected shot.
        span = max(end - start, 1)
        representatives = tuple(sorted({
            min(end - 1, start + round(span * fraction))
            for fraction in (0.2, 0.5, 0.8)
        }))
        shots.append(Shot(start, end - 1, representatives))
    return frame_count, fps, width, height, shots


def build_auto_config(source: str | Path) -> dict[str, object]:
    """Scan the whole video and return a self-contained, per-shot configuration."""
    import cv2

    path = Path(source)
    frame_count, fps, width, height, shots = _scan_shots(path)
    capture = cv2.VideoCapture(str(path))
    config_shots = []
    for index, shot in enumerate(shots):
        polygons = []
        evidence = []
        for frame_id in shot.representative_frame_ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok:
                continue
            polygon, detail = _normalized_polygon_from_frame(frame)
            polygons.append(polygon)
            evidence.append({"frame_id": frame_id, **detail})
        if not polygons:
            raise RuntimeError(f"could not extract representative frame for shot {index}")
        polygon = [
            [round(float(median(point[column] for point in sample)), 4) for column in (0, 1)]
            for sample in zip(*polygons)
        ]
        confidence = (
            "medium" if all(item["confidence"] == "medium" for item in evidence) else "low"
        )
        config_shots.append({
            "shot_id": f"shot_{index:04d}",
            "start_frame": shot.start_frame,
            "end_frame": shot.end_frame,
            "representative_frame_ids": list(shot.representative_frame_ids),
            "court_polygon_normalized": polygon,
            "roi_estimation": {
                "confidence": confidence,
                "evidence": evidence,
                "requires_review": confidence != "medium",
            },
        })
    capture.release()
    return {
        "auto_config_version": 1,
        "source_video": str(path),
        "analysis": {
            "scan_scope": "complete_video",
            "frame_count": frame_count,
            "fps": fps,
            "image_size": [width, height],
            "shot_count": len(config_shots),
            "roi_source": "video_local_analysis_only",
        },
        "court_polygon_normalized": config_shots[0]["court_polygon_normalized"],
        "shots": config_shots,
        "uniform_rules": {},
        "exclude_referee_by_uniform": True,
        "edge_max_frame_gap": 8,
        "edge_min_inside_run": 2,
        "team_min_feature_frames": 3,
        "team_confidence_threshold": 0.60,
        "team_min_cluster_tracks": 2,
        "team_max_imbalance_ratio": 8.0,
        "nonplayer_max_track_length": 15,
    }


def write_auto_config(source: str | Path, destination: str | Path) -> dict[str, object]:
    config = build_auto_config(source)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def shot_for_frame(config: dict[str, object], frame_id: int) -> dict[str, object] | None:
    for shot in config.get("shots", []):
        if int(shot["start_frame"]) <= frame_id <= int(shot["end_frame"]):
            return shot
    return None
