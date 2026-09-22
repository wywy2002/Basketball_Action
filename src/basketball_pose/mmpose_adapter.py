from __future__ import annotations

from pathlib import Path


def run_inference(
    input_path: str, output_dir: str, task: str = "2d", device: str | None = None
) -> int:
    """Run the official MMPose inferencer without hiding its model downloads."""
    try:
        from mmpose.apis import MMPoseInferencer
    except ImportError as error:
        raise RuntimeError(
            "MMPose is not installed. Create the documented Python 3.10/3.11 "
            "environment and follow the official OpenMMLab installation guide."
        ) from error

    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if task == "2d":
        inferencer = MMPoseInferencer(pose2d="human", device=device)
    elif task == "3d":
        inferencer = MMPoseInferencer(pose3d="human3d", device=device)
    else:
        raise ValueError("task must be '2d' or '3d'")

    frame_count = 0
    for _ in inferencer(str(source), out_dir=str(destination), draw_bbox=True):
        frame_count += 1
    return frame_count

