import json
from pathlib import Path

import pytest

from vgsales.config import DEFAULT_CONFIG_PATH, load_config

ROOT = DEFAULT_CONFIG_PATH.parents[1]
EDA = ROOT / "notebooks" / "01_eda.ipynb"


def _cells(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def test_eda_notebook_is_committed_without_outputs() -> None:
    for cell in _cells(EDA):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_eda_notebook_runs_on_synthetic_data() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (ROOT / load_config().data.raw_path).is_file():
        pytest.skip("the real extract is present; run the notebook directly")
    namespace: dict[str, object] = {}
    for index, cell in enumerate(_cells(EDA)):
        if cell["cell_type"] == "code":
            code = compile("".join(cell["source"]), f"01_eda.ipynb[{index}]", "exec")
            exec(code, namespace)
    plt.close("all")
    assert namespace["SOURCE"] == "synthetic"
    assert len(namespace["frame"]) > 0
