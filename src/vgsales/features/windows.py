r"""Scaling and sliding windows for the monthly aggregate series.

The sequence models of :mod:`vgsales.models.neural.sequence` are trained on
fixed-length windows cut from the wide series of
:func:`vgsales.data.series.to_wide`. For segment :math:`k` and month
:math:`t`, the monthly sales :math:`S_{k,t}` are first moved to the log scale
and standardized per segment,

.. math::

    u_{k,t} = \frac{\log(1 + S_{k,t}) - \mu_k}{\sigma_k},

where :math:`\mu_k` and :math:`\sigma_k` are the mean and standard deviation
of :math:`\log(1 + S_{k,t})` over the fitting months only
(:class:`SeriesScaler`). Standardizing per segment puts large and small
segments on a common scale, so that one network can be shared across them.

A window with origin :math:`t` (the last observed month), input length
:math:`L` and horizon :math:`H` holds

.. math::

    x_{k,t} = \big(u_{k,s}, \sin(2\pi m_s / 12), \cos(2\pi m_s / 12)
              \big)_{s = t-L+1}^{t} \in \mathbb{R}^{L \times 3},
    \qquad
    y_{k,t} = (u_{k,t+1}, \dots, u_{k,t+H}) \in \mathbb{R}^{H},

with :math:`m_s \in \{0, \dots, 11\}` the calendar month of :math:`s`. The
cyclic encoding of the month lets the network place December next to
January. The target is the whole horizon, so the model forecasts all
:math:`H` months at once (direct multi-horizon forecasting) and errors do not
compound as they do when one-step forecasts are fed back as inputs.

Two windows of the same segment overlap whenever their origins are less than
:math:`L + H` months apart. A validation set that only excludes the training
origins would therefore share target months with the training set.
:func:`split_origins` instead separates windows by the months of their
targets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

N_CHANNELS = 3


def _as_values(wide: pd.DataFrame) -> np.ndarray:
    values = wide.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("wide must be a non-empty months by segments frame")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("series values must be finite and non-negative")
    return values


def _check_index(wide: pd.DataFrame) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(wide.index)
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("the index of wide must hold increasing, unique months")
    expected = pd.period_range(index[0], periods=len(index), freq="M")
    if not (index.to_period("M") == expected).all():
        raise ValueError("the index of wide must be a complete monthly grid")
    return index


class SeriesScaler:
    """Per-segment standardization of ``log1p`` monthly sales.

    Parameters
    ----------
    min_scale : float, default 1e-3
        Lower bound on :math:`\\sigma_k`, so that a segment that is constant
        over the fitting months is centred but not divided by zero.

    Attributes
    ----------
    segments_ : pandas.Index
        Segment labels seen in :meth:`fit`, in column order.
    mean_, scale_ : numpy.ndarray
        :math:`\\mu_k` and :math:`\\sigma_k` of each segment.
    """

    def __init__(self, min_scale: float = 1e-3) -> None:
        if not min_scale > 0:
            raise ValueError("min_scale must be positive")
        self.min_scale = min_scale

    def fit(self, wide: pd.DataFrame) -> SeriesScaler:
        """Estimate the location and scale of each segment from ``wide``."""
        z = np.log1p(_as_values(wide))
        self.segments_ = pd.Index(wide.columns)
        self.mean_ = z.mean(axis=0)
        self.scale_ = np.maximum(z.std(axis=0), self.min_scale)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "segments_"):
            raise RuntimeError("the scaler is not fitted")

    def transform(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Return :math:`u_{k,t}` for the fitted segments, in fitted order."""
        self._check_fitted()
        missing = self.segments_.difference(wide.columns)
        if len(missing):
            raise ValueError(f"wide is missing fitted segments: {list(missing)}")
        values = _as_values(wide[self.segments_])
        scaled = (np.log1p(values) - self.mean_) / self.scale_
        return pd.DataFrame(scaled, index=wide.index, columns=self.segments_)

    def inverse_transform(
        self, scaled: np.ndarray, segment: np.ndarray | None = None
    ) -> np.ndarray:
        """Map standardized values back to non-negative sales.

        Parameters
        ----------
        scaled : numpy.ndarray
            Standardized values. Without ``segment`` the last axis runs over
            the fitted segments; with it, the first axis runs over rows.
        segment : numpy.ndarray of int, optional
            Segment position of each row of ``scaled``, for example
            :attr:`SequenceWindows.segment`.

        Returns
        -------
        numpy.ndarray
            :math:`\\max(0, \\exp(\\sigma_k u + \\mu_k) - 1)`, same shape.
        """
        self._check_fitted()
        values = np.asarray(scaled, dtype=np.float64)
        if segment is None:
            if values.shape[-1] != len(self.segments_):
                raise ValueError("the last axis must run over the fitted segments")
            z = values * self.scale_ + self.mean_
        else:
            rows = np.asarray(segment, dtype=np.int64)
            if values.ndim == 0 or rows.shape != (values.shape[0],):
                raise ValueError("segment must hold one position per row")
            shape = (-1,) + (1,) * (values.ndim - 1)
            z = values * self.scale_[rows].reshape(shape) + self.mean_[rows].reshape(
                shape
            )
        return np.maximum(np.expm1(z), 0.0)


@dataclass(frozen=True)
class SequenceWindows:
    """Windows cut from standardized monthly series.

    Attributes
    ----------
    inputs : numpy.ndarray of float32, shape (n, L, 3)
        Standardized value and cyclic month encoding of each input month.
    target : numpy.ndarray of float32, shape (n, H), or None
        Standardized values of the ``H`` months after the origin; None for
        windows built for forecasting.
    segment : numpy.ndarray of int64, shape (n,)
        Column position of the window's segment.
    origin : pandas.DatetimeIndex, length n
        Last input month of each window.
    segments : pandas.Index
        Segment labels; ``segment`` indexes into it.
    """

    inputs: np.ndarray
    target: np.ndarray | None
    segment: np.ndarray
    origin: pd.DatetimeIndex
    segments: pd.Index

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def input_length(self) -> int:
        """Number of input months :math:`L`."""
        return int(self.inputs.shape[1])

    @property
    def horizon(self) -> int | None:
        """Number of target months :math:`H`, None without a target."""
        return None if self.target is None else int(self.target.shape[1])

    def subset(self, mask: np.ndarray) -> SequenceWindows:
        """Return the windows selected by a boolean mask."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self),):
            raise ValueError("mask must hold one value per window")
        return SequenceWindows(
            inputs=self.inputs[mask],
            target=None if self.target is None else self.target[mask],
            segment=self.segment[mask],
            origin=self.origin[mask],
            segments=self.segments,
        )


def month_channels(index: pd.DatetimeIndex) -> np.ndarray:
    """Return the sine and cosine of the calendar month, shape ``(len(index), 2)``."""
    angle = 2.0 * np.pi * (pd.DatetimeIndex(index).month.to_numpy() - 1) / 12.0
    return np.column_stack([np.sin(angle), np.cos(angle)])


def make_windows(
    scaled: pd.DataFrame,
    input_length: int,
    horizon: int,
    *,
    origins: Sequence[pd.Timestamp] | pd.DatetimeIndex | None = None,
) -> SequenceWindows:
    """Cut training windows from standardized wide series.

    Parameters
    ----------
    scaled : pandas.DataFrame
        Output of :meth:`SeriesScaler.transform`: a complete monthly grid as
        index and one column per segment.
    input_length : int
        Number of input months :math:`L`.
    horizon : int
        Number of target months :math:`H`.
    origins : sequence of timestamps, optional
        Origins to use. Defaults to every month with :math:`L` months of
        history and :math:`H` months of future in ``scaled``.

    Returns
    -------
    SequenceWindows
        One window per segment and origin, ordered by origin and then by
        segment.
    """
    if input_length < 1 or horizon < 1:
        raise ValueError("input_length and horizon must be >= 1")
    index = _check_index(scaled)
    values = scaled.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("scaled values must be finite")
    n_months, n_segments = values.shape
    valid = np.arange(input_length - 1, n_months - horizon)
    if origins is None:
        positions = valid
    else:
        positions = index.get_indexer(pd.DatetimeIndex(origins))
        if (positions < 0).any() or not np.isin(positions, valid).all():
            raise ValueError(
                "every origin needs input_length months of history and "
                "horizon months of future in scaled"
            )
    if positions.size == 0:
        raise ValueError(
            f"{n_months} months are too few for input_length={input_length} "
            f"and horizon={horizon}"
        )
    calendar = month_channels(index)
    steps = np.arange(-input_length + 1, 1)
    ahead = np.arange(1, horizon + 1)
    inputs_idx = positions[:, None] + steps[None, :]
    target_idx = positions[:, None] + ahead[None, :]

    # (origin, step, segment) to (origin, segment, step)
    level = values[inputs_idx].transpose(0, 2, 1)
    month = np.broadcast_to(
        calendar[inputs_idx][:, None, :, :],
        (positions.size, n_segments, input_length, 2),
    )
    inputs = np.concatenate([level[..., None], month], axis=-1)
    target = values[target_idx].transpose(0, 2, 1)
    return SequenceWindows(
        inputs=inputs.reshape(-1, input_length, N_CHANNELS).astype(np.float32),
        target=target.reshape(-1, horizon).astype(np.float32),
        segment=np.tile(np.arange(n_segments, dtype=np.int64), positions.size),
        origin=index[np.repeat(positions, n_segments)],
        segments=pd.Index(scaled.columns),
    )


def last_windows(scaled: pd.DataFrame, input_length: int) -> SequenceWindows:
    """Return the window that ends at the last month of each segment.

    These are the inputs of a forecast for the months after ``scaled``.
    """
    if input_length < 1:
        raise ValueError("input_length must be >= 1")
    index = _check_index(scaled)
    if len(index) < input_length:
        raise ValueError(
            f"{len(index)} months are fewer than input_length={input_length}"
        )
    values = scaled.to_numpy(dtype=np.float64)[-input_length:]
    calendar = month_channels(index[-input_length:])
    n_segments = values.shape[1]
    level = values.T[..., None]
    month = np.broadcast_to(calendar, (n_segments, input_length, 2))
    return SequenceWindows(
        inputs=np.concatenate([level, month], axis=-1).astype(np.float32),
        target=None,
        segment=np.arange(n_segments, dtype=np.int64),
        origin=pd.DatetimeIndex([index[-1]] * n_segments),
        segments=pd.Index(scaled.columns),
    )


def split_origins(
    index: pd.DatetimeIndex, horizon: int, validation_months: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the last training origin and the first validation origin.

    The validation period is the last ``validation_months`` months of
    ``index``. Training windows end their targets before it, and validation
    windows start their targets in it, so no target month is shared.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Monthly grid of the fitting data.
    horizon : int
        Number of target months :math:`H` of a window.
    validation_months : int
        Length :math:`v` of the validation period; must be at least
        ``horizon`` so that a validation window fits inside it.

    Returns
    -------
    tuple of pandas.Timestamp
        Last origin :math:`T - v - H` of a training window and first origin
        :math:`T - v` of a validation window, where :math:`T` indexes the
        last month and positions count from zero.
    """
    if validation_months < horizon:
        raise ValueError("validation_months must be >= horizon")
    n_months = len(index)
    first_val = n_months - validation_months - 1
    last_train = first_val - horizon
    if last_train < 0:
        raise ValueError(
            f"{n_months} months are too few for a {validation_months}-month "
            "validation period"
        )
    return index[last_train], index[first_val]
