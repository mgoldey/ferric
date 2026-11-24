"""The Colab notebooks under examples/notebooks/.

Always on (no kernel needed):
  * each notebook is valid JSON in nbformat 4, stored without outputs (a
    stale output would show numbers the code no longer produces);
  * the first code cell installs ferric from PyPI, which is what makes the
    notebook run on a fresh Colab runtime, and the header carries a Colab
    badge that points at the notebook's own path;
  * every `ferric.<name>` and `viz.<name>` a code cell uses exists in the
    module under test, so a renamed or removed binding fails here and not on
    a user's runtime.

Opt-in (`FERRIC_RUN_NOTEBOOKS=1`, needs nbclient, ipykernel and py3Dmol): each
notebook is executed top to bottom with the install cell dropped (the build
under test is already importable). The notebooks assert the numbers they
print, so a clean run is the evidence.
"""

import json
import os
import re
from pathlib import Path

import pytest

import ferric

NOTEBOOK_DIR = Path(__file__).resolve().parents[3] / "examples" / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
COLAB = "https://colab.research.google.com/github/mgoldey/ferric/blob/main/examples/notebooks/"


def _load(path):
    nb = json.loads(path.read_text())
    assert nb["nbformat"] == 4
    return nb


def _code(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_there_are_notebooks():
    assert len(NOTEBOOKS) >= 3


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_is_clean_and_installs_ferric(path):
    nb = _load(path)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], "notebooks are stored without outputs"
            assert cell["execution_count"] is None
    first = _code(nb)[0]
    assert re.match(r"%pip install\b.*\bferric\b", first), first
    header = "".join(nb["cells"][0]["source"])
    assert f"({COLAB}{path.name})" in header, "the Colab badge must open this notebook"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_calls_only_existing_api(path):
    sources = "\n".join(_code(_load(path)))
    ferric_names = set(re.findall(r"\bferric\.([A-Za-z_]\w*)", sources)) - {"viz"}
    missing = sorted(n for n in ferric_names if not hasattr(ferric, n))
    assert not missing, f"{path.name} uses ferric names that do not exist: {missing}"
    viz_names = set(re.findall(r"\bviz\.([A-Za-z_]\w*)", sources))
    if viz_names:
        viz = pytest.importorskip("ferric.viz")
        missing = sorted(n for n in viz_names if not hasattr(viz, n))
        assert not missing, f"{path.name} uses viz names that do not exist: {missing}"


@pytest.mark.skipif(
    not os.environ.get("FERRIC_RUN_NOTEBOOKS"),
    reason="set FERRIC_RUN_NOTEBOOKS=1 to execute the notebooks",
)
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_executes(path, tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("py3Dmol")
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.source.startswith("%pip"):
            cell.source = "\n".join(
                line for line in cell.source.splitlines() if not line.startswith("%pip")
            )
    nbclient.NotebookClient(
        nb, timeout=300, resources={"metadata": {"path": str(tmp_path)}}
    ).execute()
