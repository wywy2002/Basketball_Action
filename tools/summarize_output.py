from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from basketball_pose.io import load_records


def main() -> None:
    output_root = Path(__file__).resolve().parents[1] / "output"
    for pose_path in sorted(output_root.glob("video_*/pose2d_players.json")):
        summary = json.loads((pose_path.parent / "summary.json").read_text(encoding="utf-8"))
        records = json.loads(pose_path.read_text(encoding="utf-8"))["records"]
        audit_records = json.loads(
            (pose_path.parent / "detections_audit.json").read_text(encoding="utf-8")
        )["records"]
        parsed_records = load_records(pose_path)
        by_frame = Counter(record["frame_id"] for record in records)
        counts = [by_frame[frame_id] for frame_id in range(summary["frames"])]
        result = {
            "frames": summary["frames"],
            "raw_detections": summary["raw_detections"],
            "player_records": len(records),
            "parsed_records": len(parsed_records),
            "referee_candidates_kept": sum(
                record.get("final_role") == "referee_candidate" for record in records
            ),
            "keypoint_confidence_range": [
                min(point[2] for record in parsed_records for point in record.keypoints_2d),
                max(point[2] for record in parsed_records for point in record.keypoints_2d),
            ],
            "tracks": summary["tracks"],
            "low_confidence_recoveries": summary["low_confidence_recoveries"],
            "shots": summary.get("shots", 1),
            "outside_roi_records": summary["outside_roi_records"],
            "edge_track_recoveries": summary["edge_track_recoveries"],
            "single_frame_outside_exclusions": summary["single_frame_outside_exclusions"],
            "short_stationary_boundary_exclusions": summary[
                "short_stationary_boundary_exclusions"
            ],
            "team_tracks": summary["team_tracks"],
            "missing_required_fields": sum(
                not all(field in record for field in (
                    "team_id", "team_confidence", "team_feature", "roi_status",
                    "edge_track_recovery", "edge_recovery_reason", "track_length",
                    "rejection_reason",
                ))
                for record in records
            ),
            "audit_missing_required_fields": sum(
                not all(field in record for field in (
                    "team_id", "team_confidence", "team_feature", "roi_status",
                    "edge_track_recovery", "edge_recovery_reason", "track_length",
                    "rejection_reason",
                ))
                for record in audit_records
            ),
            "players_per_frame": {
                "minimum": min(counts),
                "median": median(counts),
                "maximum": max(counts),
                "mean": round(mean(counts), 2),
            },
        }
        print(pose_path.parent.name, json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
