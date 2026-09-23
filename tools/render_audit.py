from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

from basketball_pose.rtmlib_pipeline import _draw_record


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: render_audit.py INPUT_VIDEO AUDIT_JSON OUTPUT_VIDEO")
    input_path, audit_path, output_path = map(Path, sys.argv[1:])
    records = json.loads(audit_path.read_text(encoding="utf-8"))["records"]
    by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_frame[int(record["frame_id"])].append(record)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    frame_id = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        for record in by_frame[frame_id]:
            _draw_record(frame, record)
        writer.write(frame)
        frame_id += 1
    capture.release()
    writer.release()
    print(output_path)


if __name__ == "__main__":
    main()
