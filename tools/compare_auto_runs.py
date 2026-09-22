from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(directory: Path) -> dict[str, object]:
    summary = _load(directory / "summary.json")
    records = _load(directory / "pose2d_players.json")["records"]
    audit = _load(directory / "detections_audit.json")["records"]
    return {
        "frames": summary["frames"],
        "player_records": len(records),
        "raw_detections": summary["raw_detections"],
        "tracks": summary["tracks"],
        "shots": summary.get("shots"),
        "unknown_player_records": sum(record.get("team_id") == "unknown" for record in records),
        "edge_recoveries": sum(bool(record.get("edge_track_recovery")) for record in audit),
        "rejected_audit_records": sum(bool(record.get("rejection_reason")) for record in audit),
        "team_records": dict(Counter(record.get("team_id", "unknown") for record in records)),
    }


def _difference(baseline: dict[str, object], automatic: dict[str, object]) -> dict[str, int]:
    return {
        key: int(automatic[key]) - int(baseline[key])
        for key in ("player_records", "raw_detections", "tracks", "unknown_player_records",
                    "edge_recoveries", "rejected_audit_records")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare manual-baseline and automatic video runs")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", default=["video_1", "video_2", "video_3"])
    args = parser.parse_args()

    reports = {}
    for video in args.videos:
        baseline = _metrics(args.baseline_root / video)
        automatic = _metrics(args.automatic_root / video)
        report = {
            "video": video,
            "baseline": baseline,
            "automatic": automatic,
            "difference_automatic_minus_baseline": _difference(baseline, automatic),
            "interpretation": (
                "This is an output difference report, not an accuracy measurement. "
                "The baseline was produced by older code and no human ground truth is supplied."
            ),
        }
        reports[video] = report
        target = args.automatic_root / video
        (target / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (target / "comparison.md").write_text(
            f"# {video} automatic-run comparison\n\n"
            f"{report['interpretation']}\n\n"
            f"- baseline player records: {baseline['player_records']}\n"
            f"- automatic player records: {automatic['player_records']}\n"
            f"- automatic minus baseline: {report['difference_automatic_minus_baseline']}\n",
            encoding="utf-8",
        )
    combined = {
        "videos": reports,
        "interpretation": (
            "No accuracy claim is made: these are counts and audit-state differences only."
        ),
    }
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination / "comparison_summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["# Automatic configuration comparison summary", "", combined["interpretation"], ""]
    for video, report in reports.items():
        delta = report["difference_automatic_minus_baseline"]
        lines.append(f"- {video}: player-record delta {delta['player_records']}, "
                     f"unknown-record delta {delta['unknown_player_records']}")
    (args.destination / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
