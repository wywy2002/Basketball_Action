from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UniformRules:
    referee_max_saturation: float = 55.0
    referee_min_value: float = 45.0
    referee_max_value: float = 120.0
    light_team_max_saturation: float = 70.0
    light_team_min_value: float = 120.0


def classify_uniform(
    hsv: tuple[float, float, float] | None, rules: UniformRules
) -> str:
    """Classify a track conservatively; unknown objects are never hard-filtered."""
    if hsv is None:
        return "unknown"
    _, saturation, value = hsv
    if (
        saturation <= rules.referee_max_saturation
        and rules.referee_min_value <= value <= rules.referee_max_value
    ):
        return "referee_candidate"
    if saturation <= rules.light_team_max_saturation and value >= rules.light_team_min_value:
        return "team_light"
    return "team_dark"
