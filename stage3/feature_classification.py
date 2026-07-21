import math
from collections.abc import Iterable

COIL_TIGHTNESS_RATIO_THRESHOLD = 1.2
INNER_TIGHT_OUTER_LOOSE = "inner volutions are more tightly coiled than outer ones"
UNIFORM_COILING = "volutions are uniformly coiled"


def classify_size(area_mm2: float) -> str:
    """Classify shell size from its alpha-mask area in square millimetres."""
    if area_mm2 < 1.0:
        return "minute"
    if area_mm2 < 4.5:
        return "small"
    if area_mm2 < 16:
        return "medium"
    if area_mm2 < 40:
        return "large"
    return "gigantic"


def classify_coil_tightness(heights_of_volutions: Iterable[float]) -> tuple[str, float | None]:
    """Classify coiling from the ratio of outer-third to middle-third mean height."""
    heights = [float(height) for height in heights_of_volutions]
    if len(heights) < 3 or any(not math.isfinite(height) or height < 0 for height in heights):
        return INNER_TIGHT_OUTER_LOOSE, None

    third_size, remainder = divmod(len(heights), 3)
    inner_size = third_size + (remainder > 0)
    middle_size = third_size + (remainder > 1)
    middle = heights[inner_size : inner_size + middle_size]
    outer = heights[inner_size + middle_size :]

    middle_mean = math.fsum(middle) / len(middle)
    if middle_mean == 0:
        return INNER_TIGHT_OUTER_LOOSE, None

    outer_mean = math.fsum(outer) / len(outer)
    ratio = outer_mean / middle_mean
    if ratio < COIL_TIGHTNESS_RATIO_THRESHOLD:
        return UNIFORM_COILING, ratio
    return INNER_TIGHT_OUTER_LOOSE, ratio
