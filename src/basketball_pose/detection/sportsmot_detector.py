from __future__ import annotations

import sys
from pathlib import Path


class SportsMOTDetector:
    """YOLOX-X player detector trained on the official SportsMOT data."""

    def __init__(
        self,
        project_root: Path,
        checkpoint: Path,
        device: str = "cuda",
        confidence: float = 0.08,
        nms: float = 0.7,
    ) -> None:
        import torch

        mixsort_root = project_root / "third_party" / "MixSort"
        if str(mixsort_root) not in sys.path:
            sys.path.insert(0, str(mixsort_root))

        from yolox.exp import get_exp

        experiment = get_exp(
            str(mixsort_root / "exps" / "example" / "mot" / "yolox_x_sportsmot.py"),
            None,
        )
        experiment.test_conf = confidence
        experiment.nmsthre = nms
        self.test_size = experiment.test_size
        self.confidence = confidence
        self.nms = nms
        self.device = torch.device(device)
        self.model = experiment.get_model()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.model.to(self.device).eval()
        self.fp16 = self.device.type == "cuda"
        if self.fp16:
            self.model.half()

    def __call__(self, frame) -> list[tuple[tuple[float, float, float, float], float]]:
        import torch
        from yolox.data.data_augment import preproc
        from yolox.utils import postprocess

        image, ratio = preproc(
            frame,
            self.test_size,
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        )
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        tensor = tensor.half() if self.fp16 else tensor.float()
        with torch.inference_mode():
            output = self.model(tensor)
            output = postprocess(output, 1, self.confidence, self.nms)[0]
        if output is None:
            return []

        detections = []
        for row in output.detach().float().cpu().numpy():
            x1, y1, x2, y2 = (float(value / ratio) for value in row[:4])
            score = float(row[4] * row[5])
            detections.append(((x1, y1, x2, y2), score))
        return detections
