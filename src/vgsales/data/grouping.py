"""Grouping keys that identify a title across platforms.

The extract has one row per (title, console) pair, so a title released on
several platforms yields several rows with correlated targets. Splits use the
key returned by :func:`title_key` as a group label to keep these rows on one
side of every split.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

_NON_ALNUM = re.compile(r"[\W_]+")


def _key(title: str) -> str:
    decomposed = unicodedata.normalize("NFKD", title)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NON_ALNUM.sub(" ", stripped.casefold()).strip()


def title_key(titles: pd.Series) -> pd.Series:
    """Return a normalised grouping key for each title.

    The key is case-folded, has diacritics removed and has every run of
    punctuation, underscores and whitespace replaced by a single space. Thus
    ``"Halo: Reach"`` and ``"HALO Reach"`` share a key, while edition
    suffixes such as ``"Remastered"`` are kept, because they usually denote a
    separate release.

    Parameters
    ----------
    titles : pandas.Series
        Title strings with ``NaN`` for missing values.

    Returns
    -------
    pandas.Series
        Object-dtype keys, ``NaN`` where the title is missing or has no
        alphanumeric characters.
    """
    out = titles.astype(object)
    present = out.notna()
    out[present] = [_key(str(t)) for t in out[present]]
    return out.where(present & out.ne(""), np.nan).rename("title_key")
