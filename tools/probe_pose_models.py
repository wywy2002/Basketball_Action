from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import onnxruntime as ort


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "outputs" / "probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    ort.preload_dlls(directory="")
    print("providers:", ort.get_available_providers())

    capture = cv2.VideoCapture(str(root / "input_videos" / "video_1.mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 58)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("could not read video_1.mp4 frame 58")

    from rtmlib import Body, draw_skeleton

    variants = {
        "rtmpose": Body(mode="balanced", backend="onnxruntime", device="cuda"),
        "rtmo": Body(
            pose="rtmo", mode="balanced", backend="onnxruntime", device="cuda"
        ),
    }
    for name, model in variants.items():
        started = perf_counter()
        keypoints, scores = model(frame)
        elapsed = perf_counter() - started
        result = draw_skeleton(
            frame.copy(), keypoints, scores, openpose_skeleton=False,
            kpt_thr=0.3, line_width=2,
        )
        cv2.imwrite(str(output_dir / f"video_1_frame58_{name}.jpg"), result)
        print(f"{name}: people={len(keypoints)} seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()

