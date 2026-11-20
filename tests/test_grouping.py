import numpy as np
import pandas as pd

from vgsales.data.grouping import title_key


def test_case_punctuation_and_whitespace_are_ignored() -> None:
    keys = title_key(pd.Series(["Halo: Reach", "HALO   Reach", "halo_reach "]))
    assert keys.nunique() == 1
    assert keys[0] == "halo reach"


def test_diacritics_are_removed() -> None:
    keys = title_key(pd.Series(["Pokémon Red", "Pokemon Red"]))
    assert keys[0] == keys[1] == "pokemon red"


def test_edition_suffixes_are_kept() -> None:
    keys = title_key(pd.Series(["The Last of Us", "The Last of Us Remastered"]))
    assert keys[0] != keys[1]


def test_missing_and_symbol_only_titles_are_missing() -> None:
    keys = title_key(pd.Series([np.nan, "!!!", "Tetris"]))
    assert keys.isna().tolist() == [True, True, False]
    assert keys.name == "title_key"
