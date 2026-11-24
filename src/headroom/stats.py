import math
import random
import statistics
from collections.abc import Callable, Sequence

BOOTSTRAP_RESAMPLES = 10_000


def nearest_rank(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in (0, 1]). Always returns an observed value."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("no values")
    return statistics.median(values)


def cv(values: Sequence[float]) -> float | None:
    """Coefficient of variation (sample stdev / mean). None with fewer than 2 values."""
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else None


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = median,
    *,
    level: float = 0.95,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval of stat(values), resampling values with replacement.
    Seeded, so the same data always gives the same interval."""
    if not values:
        raise ValueError("no values")
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    lo = stats[int((1 - level) / 2 * resamples)]
    hi = stats[min(resamples - 1, int((1 + level) / 2 * resamples))]
    return lo, hi
