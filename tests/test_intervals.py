import numpy as np
import pytest

from vgsales.evaluation.intervals import (
    INTERVAL_METRICS,
    coverage,
    interval_metrics,
    interval_score,
    pinball_loss,
)

Y = np.array([0.1, 0.5, 1.0, 2.0, 4.0])
LOWER = np.array([0.0, 0.6, 0.5, 1.0, 1.0])
UPPER = np.array([0.2, 0.9, 1.5, 1.5, 5.0])


def test_pinball_loss_by_hand() -> None:
    assert pinball_loss([1.0], [0.0], 0.9) == pytest.approx(0.9)
    assert pinball_loss([0.0], [1.0], 0.9) == pytest.approx(0.1)
    assert pinball_loss([2.0, 2.0], [2.0, 2.0], 0.3) == 0.0
    assert pinball_loss([1.0, 3.0], [2.0, 2.0], 0.5) == pytest.approx(0.5)


def test_pinball_loss_is_minimized_by_the_quantile() -> None:
    rng = np.random.default_rng(0)
    sample = rng.lognormal(size=4000)
    for tau in (0.1, 0.5, 0.9):
        best = np.quantile(sample, tau)
        loss = pinball_loss(sample, np.full(sample.size, best), tau)
        for shift in (-0.2, 0.2):
            shifted = np.full(sample.size, best + shift)
            assert loss < pinball_loss(sample, shifted, tau)


def test_coverage_and_tails() -> None:
    metrics = interval_metrics(Y, LOWER, UPPER, alpha=0.2)
    assert tuple(metrics) == INTERVAL_METRICS
    assert coverage(Y, LOWER, UPPER) == pytest.approx(0.6)
    assert metrics["coverage"] == pytest.approx(0.6)
    assert metrics["below"] == pytest.approx(0.2)
    assert metrics["above"] == pytest.approx(0.2)
    assert metrics["width"] == pytest.approx(np.mean(UPPER - LOWER))
    assert metrics["width_log"] == pytest.approx(
        np.mean(np.log1p(UPPER) - np.log1p(LOWER))
    )
    assert coverage([1.0], [1.0], [1.0]) == 1.0


def test_interval_score_by_hand() -> None:
    alpha = 0.2
    # The second observation lies 0.1 below its interval, the fourth 0.5 above.
    width = np.mean(UPPER - LOWER)
    penalty = (2 / alpha) * (0.1 + 0.5) / Y.size
    assert interval_score(Y, LOWER, UPPER, alpha) == pytest.approx(width + penalty)


def test_interval_score_equals_scaled_pinball_losses() -> None:
    rng = np.random.default_rng(1)
    y = rng.lognormal(size=200)
    lower = np.quantile(y, 0.1) * rng.uniform(0.5, 1.5, 200)
    upper = lower + rng.uniform(0.0, 3.0, 200)
    alpha = 0.2
    pinball = pinball_loss(y, lower, alpha / 2) + pinball_loss(y, upper, 1 - alpha / 2)
    assert interval_score(y, lower, upper, alpha) == pytest.approx(
        (2 / alpha) * pinball
    )


def test_coverage_is_invariant_to_the_log_scale() -> None:
    rng = np.random.default_rng(2)
    y = rng.lognormal(size=300)
    z_low = np.log1p(y) - rng.uniform(-0.5, 1.0, 300)
    z_high = z_low + rng.uniform(0.0, 1.5, 300)
    z = np.log1p(y)
    on_log = np.mean((z >= z_low) & (z <= z_high))
    assert 0.0 < on_log < 1.0
    assert coverage(y, np.expm1(z_low), np.expm1(z_high)) == pytest.approx(on_log)


def test_negative_bounds_are_clipped() -> None:
    metrics = interval_metrics([0.0, 1.0], [-0.5, 0.5], [0.5, 2.0], alpha=0.1)
    assert metrics["coverage"] == 1.0
    assert metrics["width"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (([1.0, 2.0], [0.0], [3.0]), "same length"),
        (([1.0], [2.0], [1.5]), "must not exceed"),
        (([-1.0], [0.0], [1.0]), "non-negative"),
        (([np.nan], [0.0], [1.0]), "finite"),
        (([], [], []), "non-empty"),
    ],
)
def test_invalid_intervals(args: tuple, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        interval_metrics(*args, alpha=0.1)


def test_invalid_levels() -> None:
    with pytest.raises(ValueError, match="alpha"):
        interval_score([1.0], [0.0], [2.0], alpha=1.0)
    with pytest.raises(ValueError, match="tau"):
        pinball_loss([1.0], [1.0], tau=0.0)
    with pytest.raises(ValueError, match="same length"):
        pinball_loss([1.0, 2.0], [1.0], tau=0.5)
