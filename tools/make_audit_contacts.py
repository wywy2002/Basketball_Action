from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_contact(video_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_ids = (0, frame_count // 2, max(0, frame_count - 1))
    panels: list[np.ndarray] = []
    for frame_id in sample_ids:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {frame_id} from {video_path}")
        scale = min(640 / frame.shape[1], 360 / frame.shape[0])
        panel = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.putText(
            panel,
            f"frame {frame_id}",
            (12, panel.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    capture.release()
    destination = video_path.with_name("pose2d_audit_contact.jpg")
    if not cv2.imwrite(str(destination), np.hstack(panels)):
        raise RuntimeError(f"cannot write {destination}")
    print(destination)


def main() -> None:
    output_root = Path(__file__).resolve().parents[1] / "output"
    for video_path in sorted(output_root.glob("video_*/pose2d_audit.mp4")):
        make_contact(video_path)


if __name__ == "__main__":
    main()
