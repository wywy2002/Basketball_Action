from __future__ import annotations

import argparse
from pathlib import Path

from .config.auto_config import write_auto_config
from .io import load_records, write_json
from .pose.keypoints import H36M17_NAMES, coco17_to_h36m17
from .pose.mmpose_adapter import run_inference
from .pose.quality import inspect_track
from .pipeline.rtmlib_pipeline import run_video
from .tracking.tracks import group_tracks, interpolate_short_gaps


def _validate(input_path: str) -> int:
    records = load_records(input_path)
    tracks = group_tracks(records)
    reports = []
    for (video_id, shot_id, track_id), track in tracks.items():
        reports.append({
            "video_id": video_id,
            "shot_id": shot_id,
            "track_id": track_id,
            "frame_count": len(track),
            "issues": inspect_track(track),
        })
    issue_count = sum(len(report["issues"]) for report in reports)
    print(f"records={len(records)} tracks={len(tracks)} issues={issue_count}")
    for report in reports:
        if report["issues"]:
            print(report)
    return 0


def _preprocess(input_path: str, output_path: str, max_gap: int) -> int:
    tracks = group_tracks(load_records(input_path))
    payload = {"pose2d_tracks": [], "h36m17_names": list(H36M17_NAMES)}
    for key, track in tracks.items():
        prepared = interpolate_short_gaps(track, max_gap)
        payload["pose2d_tracks"].append({
            "video_id": key[0],
            "shot_id": key[1],
            "track_id": key[2],
            "frames": [
                {
                    **record.to_dict(),
                    "keypoints_h36m17": [
                        list(point) for point in coco17_to_h36m17(record.keypoints_2d)
                    ],
                }
                for record in prepared
            ],
            "quality_issues": inspect_track(prepared),
        })
    write_json(output_path, payload)
    print(f"wrote {len(payload['pose2d_tracks'])} tracks to {Path(output_path)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Basketball 2D-to-3D pose baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate pose JSON")
    validate.add_argument("input")

    preprocess = subparsers.add_parser("preprocess", help="prepare tracks for 3D lifting")
    preprocess.add_argument("input")
    preprocess.add_argument("output")
    preprocess.add_argument("--max-gap", type=int, default=2)

    infer = subparsers.add_parser("infer", help="run the official MMPose inferencer")
    infer.add_argument("input")
    infer.add_argument("output_dir")
    infer.add_argument("--task", choices=("2d", "3d"), default="2d")
    infer.add_argument("--device", default=None)

    video = subparsers.add_parser("run-video", help="run basketball-aware 2D pose pipeline")
    video.add_argument("input")
    video.add_argument("output_dir", nargs="?")
    video.add_argument("--config")
    video.add_argument(
        "--auto-config", action="store_true",
        help="force a fresh full-video configuration; ignores any existing configs/<video>.json",
    )
    video.add_argument(
        "--model", choices=("rtmpose", "rtmo", "sportsmot-rtmpose"), default="rtmo"
    )
    video.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    video.add_argument("--max-frames", type=int, default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "validate":
        return _validate(args.input)
    if args.command == "preprocess":
        return _preprocess(args.input, args.output, args.max_gap)
    if args.command == "run-video":
        source = Path(args.input)
        project_root = Path(__file__).resolve().parents[2]
        existing_video_config = project_root / "configs" / f"{source.stem}.json"
        automatic = args.auto_config or (args.config is None and not existing_video_config.exists())
        if automatic:
            config_dir = project_root / "experiments" / "auto-video-config" / "configs" / source.stem
            output_dir = project_root / "experiments" / "auto-video-config" / "outputs" / source.stem
            if args.output_dir and Path(args.output_dir).resolve() != output_dir.resolve():
                parser.error(
                    "automatic runs always write to experiments/auto-video-config/outputs/<video>; "
                    "omit output_dir"
                )
            config_path = config_dir / "auto_config.json"
            config = write_auto_config(source, config_path)
            write_json(config_dir / "auto_config_analysis.json", config["analysis"] | {
                "shots": config["shots"],
            })
        else:
            if args.output_dir is None:
                parser.error("output_dir is required when using an explicit or existing config")
            output_dir = Path(args.output_dir)
            config_path = Path(args.config) if args.config else existing_video_config
        summary = run_video(
            source, output_dir, config_path, args.model, args.device,
            args.max_frames, automatic,
        )
        print(summary)
        return 0
    frames = run_inference(args.input, args.output_dir, args.task, args.device)
    print(f"processed {frames} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
