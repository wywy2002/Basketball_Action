from __future__ import annotations

import json
from pathlib import Path

from .schema import Pose2DRecord


def load_records(path: str | Path) -> list[Pose2DRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload["records"] if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("input JSON must be a list or an object containing records")
    return [Pose2DRecord.from_dict(value) for value in values]


def write_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

