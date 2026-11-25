import numpy as np
import pandas as pd
import pytest

from vgsales.config import load_config
from vgsales.data.cleaning import clean
from vgsales.data.series import aggregate_series, to_wide
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.data.temporal import apply_release_scope
from vgsales.features.windows import (
    N_CHANNELS,
    SeriesScaler,
    last_windows,
    make_windows,
    month_channels,
    split_origins,
)


def toy_wide(n_months: int = 30, n_segments: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2001-01-01", periods=n_months, freq="MS")
    values = rng.gamma(2.0, 1.0, size=(n_months, n_segments))
    columns = [f"seg{k}" for k in range(n_segments)]
    return pd.DataFrame(values, index=index, columns=columns)


def test_scaler_standardizes_and_inverts() -> None:
    wide = toy_wide()
    scaler = SeriesScaler().fit(wide)
    scaled = scaler.transform(wide)
    np.testing.assert_allclose(scaled.mean().to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(scaled.std(ddof=0).to_numpy(), 1.0, atol=1e-12)
    restored = scaler.inverse_transform(scaled.to_numpy())
    np.testing.assert_allclose(restored, wide.to_numpy(), rtol=1e-10)

    rows = np.array([2, 0, 0])
    by_row = scaler.inverse_transform(scaled.to_numpy()[:3, rows].T, segment=rows)
    np.testing.assert_allclose(by_row, wide.to_numpy()[:3, rows].T, rtol=1e-10)


def test_scaler_uses_only_the_fitting_months() -> None:
    wide = toy_wide()
    scaler = SeriesScaler().fit(wide.iloc[:20])
    shifted = wide.copy()
    shifted.iloc[20:] *= 100.0
    assert np.allclose(SeriesScaler().fit(shifted.iloc[:20]).mean_, scaler.mean_)
    constant = wide.assign(seg0=1.0)
    fitted = SeriesScaler(min_scale=0.5).fit(constant)
    assert fitted.scale_[0] == 0.5
    assert np.allclose(fitted.transform(constant)["seg0"], 0.0)


def test_scaler_rejects_bad_inputs() -> None:
    wide = toy_wide()
    with pytest.raises(RuntimeError, match="not fitted"):
        SeriesScaler().transform(wide)
    scaler = SeriesScaler().fit(wide)
    with pytest.raises(ValueError, match="missing fitted segments"):
        scaler.transform(wide.drop(columns="seg1"))
    with pytest.raises(ValueError, match="non-negative"):
        SeriesScaler().fit(wide - 10.0)
    with pytest.raises(ValueError, match="one position per row"):
        scaler.inverse_transform(np.zeros((2, 4)), segment=np.array([0]))


def test_windows_hold_the_right_months() -> None:
    wide = toy_wide(n_months=30, n_segments=3)
    scaled = SeriesScaler().fit(wide).transform(wide)
    windows = make_windows(scaled, input_length=6, horizon=4)
    n_origins = 30 - 6 - 4 + 1
    assert len(windows) == n_origins * 3
    assert windows.inputs.shape == (n_origins * 3, 6, N_CHANNELS)
    assert windows.target.shape == (n_origins * 3, 4)
    assert windows.inputs.dtype == np.float32
    assert windows.input_length == 6
    assert windows.horizon == 4

    i = 7 * 3 + 2
    position = scaled.index.get_loc(windows.origin[i])
    assert windows.segment[i] == 2
    np.testing.assert_allclose(
        windows.inputs[i, :, 0], scaled.iloc[position - 5 : position + 1, 2], atol=1e-6
    )
    np.testing.assert_allclose(
        windows.target[i], scaled.iloc[position + 1 : position + 5, 2], atol=1e-6
    )
    months = month_channels(scaled.index[position - 5 : position + 1])
    np.testing.assert_allclose(windows.inputs[i, :, 1:], months, atol=1e-6)


def test_month_channels_are_cyclic() -> None:
    channels = month_channels(pd.date_range("2000-01-01", periods=13, freq="MS"))
    np.testing.assert_allclose(channels[0], channels[12], atol=1e-12)
    np.testing.assert_allclose(np.hypot(channels[:, 0], channels[:, 1]), 1.0)


def test_split_origins_separate_target_months() -> None:
    wide = toy_wide(n_months=40)
    scaled = SeriesScaler().fit(wide).transform(wide)
    horizon, validation = 4, 8
    last_train, first_val = split_origins(scaled.index, horizon, validation)
    windows = make_windows(scaled, 6, horizon)
    train = windows.subset(windows.origin <= last_train)
    val = windows.subset(windows.origin >= first_val)
    last_train_target = train.origin.max() + pd.DateOffset(months=horizon)
    first_val_target = val.origin.min() + pd.DateOffset(months=1)
    assert last_train_target < first_val_target
    assert first_val_target == scaled.index[-validation]
    assert val.origin.max() == scaled.index[-1 - horizon]
    with pytest.raises(ValueError, match=">= horizon"):
        split_origins(scaled.index, horizon=6, validation_months=4)


def test_selected_origins_and_last_windows() -> None:
    wide = toy_wide(n_months=24)
    scaled = SeriesScaler().fit(wide).transform(wide)
    chosen = scaled.index[[5, 9]]
    windows = make_windows(scaled, 6, 3, origins=chosen)
    assert list(pd.unique(windows.origin)) == list(chosen)
    with pytest.raises(ValueError, match="history"):
        make_windows(scaled, 6, 3, origins=scaled.index[[2]])
    with pytest.raises(ValueError, match="too few"):
        make_windows(scaled.iloc[:5], 6, 3)

    last = last_windows(scaled, 6)
    assert last.target is None
    assert last.horizon is None
    assert len(last) == scaled.shape[1]
    np.testing.assert_allclose(last.inputs[1, :, 0], scaled.iloc[-6:, 1], atol=1e-6)
    assert (last.origin == scaled.index[-1]).all()


def test_incomplete_grid_is_rejected() -> None:
    wide = toy_wide(n_months=24).drop(index=pd.Timestamp("2001-05-01"))
    with pytest.raises(ValueError, match="complete monthly grid"):
        make_windows(wide, 3, 2)


def test_windows_from_synthetic_series() -> None:
    raw = make_synthetic(SyntheticSpec(n_titles=1200), seed=3)
    titles = apply_release_scope(clean(raw)[0], load_config().data)[0]
    wide = to_wide(aggregate_series(titles, by=("genre",)))
    scaled = SeriesScaler().fit(wide).transform(wide)
    windows = make_windows(scaled, 12, 6)
    assert np.isfinite(windows.inputs).all()
    assert np.isfinite(windows.target).all()
    assert set(windows.segment) == set(range(wide.shape[1]))
