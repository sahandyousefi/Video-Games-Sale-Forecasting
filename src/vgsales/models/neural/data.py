"""Tensor dataset and batch loader for encoded tabular rows.

:class:`TabularDataset` holds the output of
:class:`vgsales.features.tabular.TabularEncoder` as tensors and is indexed by
lists of row positions, so a whole batch is gathered with one indexing
operation instead of collating single rows. :func:`make_loader` wraps it in a
``DataLoader`` driven by a batch sampler. Shuffling draws from a dedicated
``torch.Generator`` seeded by the caller, so the sequence of batches over
successive epochs is reproducible and independent of the global random state.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)

from vgsales.features.tabular import TabularArrays

Batch = tuple[torch.Tensor, ...]


class TabularDataset(Dataset):
    """Embedding indices, numeric inputs and an optional target as tensors.

    Parameters
    ----------
    arrays : TabularArrays
        Encoded rows.
    target : array-like of float, optional
        Target on the modelling scale, one value per row.

    Notes
    -----
    Indexing with an integer returns the tensors of one row; indexing with a
    sequence of positions returns the stacked tensors of those rows. The
    items are ``(categorical, numeric)`` without a target and
    ``(categorical, numeric, target)`` with one.
    """

    def __init__(self, arrays: TabularArrays, target: ArrayLike | None = None):
        self.categorical = torch.as_tensor(arrays.categorical, dtype=torch.long)
        self.numeric = torch.as_tensor(arrays.numeric, dtype=torch.float32)
        self.target: torch.Tensor | None = None
        if target is not None:
            values = np.asarray(target, dtype=np.float32)
            if values.shape != (len(arrays),):
                raise ValueError(
                    f"target must hold one value per row ({len(arrays)}), "
                    f"got shape {values.shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError("target must be finite")
            self.target = torch.from_numpy(values)

    def __len__(self) -> int:
        return int(self.categorical.shape[0])

    @property
    def has_target(self) -> bool:
        """Whether the items carry the target."""
        return self.target is not None

    def __getitem__(self, index: int | Sequence[int] | torch.Tensor) -> Batch:
        if self.target is None:
            return self.categorical[index], self.numeric[index]
        return self.categorical[index], self.numeric[index], self.target[index]


def make_loader(
    dataset: Dataset,
    batch_size: int = 256,
    *,
    shuffle: bool = False,
    seed: int | None = None,
    drop_last: bool = False,
) -> DataLoader:
    """Build a loader that yields whole batches of ``dataset``.

    Parameters
    ----------
    dataset : Dataset
        Map-style dataset that accepts a list of positions as index, such as
        :class:`TabularDataset` or ``torch.utils.data.TensorDataset``.
    batch_size : int, default 256
        Rows per batch. The last batch is smaller unless ``drop_last``.
    shuffle : bool, default False
        Draw a new permutation of the rows at the start of every epoch.
    seed : int, optional
        Seed of the generator that draws the permutations. Required when
        ``shuffle`` is True.
    drop_last : bool, default False
        Drop the last batch when it is smaller than ``batch_size``.

    Returns
    -------
    torch.utils.data.DataLoader
        Loader whose items are the batches returned by ``dataset``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if shuffle:
        if seed is None:
            raise ValueError("a seed is required when shuffle is True")
        generator = torch.Generator().manual_seed(int(seed))
        rows = RandomSampler(dataset, generator=generator)
    else:
        rows = SequentialSampler(dataset)
    sampler = BatchSampler(rows, batch_size=batch_size, drop_last=drop_last)
    # batch_size=None turns off per-row collation; the dataset receives the
    # list of positions of each batch.
    return DataLoader(dataset, sampler=sampler, batch_size=None)
