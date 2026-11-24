import pytest

from headroom.stats import bootstrap_ci, cv, median, nearest_rank


def test_median_and_cv():
    assert median([3, 1, 2]) == 2
    assert median([1, 2, 3, 4]) == 2.5
    assert cv([2.0, 2.0, 2.0]) == 0
    assert cv([1.0]) is None
    assert cv([1.0, 3.0]) == pytest.approx(1.4142 / 2, rel=1e-3)


def test_nearest_rank_returns_an_observed_value():
    assert nearest_rank(list(range(1, 21)), 0.95) == 19
    assert nearest_rank([5.0], 0.95) == 5.0


def test_bootstrap_is_reproducible_and_brackets_the_statistic():
    values = [1.0, 1.2, 0.9, 1.1, 1.3, 1.0, 0.95, 1.05, 1.15, 1.25]
    lo, hi = bootstrap_ci(values)
    assert (lo, hi) == bootstrap_ci(values)  # seeded
    assert lo <= median(values) <= hi
    assert bootstrap_ci([2.0] * 8) == (2.0, 2.0)  # no spread, no interval


def test_bootstrap_interval_narrows_with_more_data():
    small = [1.0, 2.0, 1.5, 1.2, 1.8]
    big = small * 20
    w_small = bootstrap_ci(small)[1] - bootstrap_ci(small)[0]
    w_big = bootstrap_ci(big)[1] - bootstrap_ci(big)[0]
    assert w_big < w_small
