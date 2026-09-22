from __future__ import annotations

from collections.abc import Sequence

Keypoint = tuple[float, float, float]

H36M17_NAMES = (
    "pelvis", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle", "spine", "thorax",
    "neck", "head", "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
)


def _midpoint(left: Keypoint, right: Keypoint) -> Keypoint:
    return (
        (left[0] + right[0]) / 2.0,
        (left[1] + right[1]) / 2.0,
        min(left[2], right[2]),
    )


def coco17_to_h36m17(keypoints: Sequence[Sequence[float]]) -> tuple[Keypoint, ...]:
    """Convert COCO-17 order to an H36M-17 lifting order.

    Virtual torso joints are geometric midpoints. The mapping is deterministic,
    but it is not ground truth and must be versioned with exported predictions.
    """
    if len(keypoints) != 17 or any(len(keypoint) != 3 for keypoint in keypoints):
        raise ValueError("expected 17 keypoints shaped [x, y, confidence]")
    coco = tuple(tuple(float(v) for v in point) for point in keypoints)
    pelvis = _midpoint(coco[11], coco[12])
    thorax = _midpoint(coco[5], coco[6])
    spine = _midpoint(pelvis, thorax)
    face = _midpoint(coco[3], coco[4])
    head = face if face[2] > 0 else coco[0]
    neck = _midpoint(thorax, head)
    return (
        pelvis,
        coco[12], coco[14], coco[16],
        coco[11], coco[13], coco[15],
        spine, thorax, neck, head,
        coco[5], coco[7], coco[9],
        coco[6], coco[8], coco[10],
    )

