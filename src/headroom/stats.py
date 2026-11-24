import math
from collections.abc import Sequence


def nearest_rank(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in (0, 1]). Always returns an observed value."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]
