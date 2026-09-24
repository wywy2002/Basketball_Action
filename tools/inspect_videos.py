from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def inspect_video(path: Path, output_dir: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_ids = sorted({0, max(0, frame_count // 2), max(0, frame_count - 1)})
    samples = []
    for frame_id in sample_ids:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        if not ok:
            continue
        scale = min(640 / width, 360 / height)
        resized = cv2.resize(frame, (round(width * scale), round(height * scale)))
        canvas = np.zeros((390, 640, 3), dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        cv2.putText(
            canvas, f"{path.name} frame={frame_id}", (10, 382),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        samples.append(canvas)
    capture.release()
    if samples:
        cv2.imwrite(str(output_dir / f"{path.stem}_contact.jpg"), np.hstack(samples))
    return {
        "name": path.name,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / fps if fps else None,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "output" / "inspection"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = [
        inspect_video(path, output_dir)
        for path in sorted((root / "input_videos").glob("*"))
        if path.is_file()
    ]
    (output_dir / "videos.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

