"""Generated ferric inputs run under the real CLI parser; the metadata Avogadro reads is complete."""

from __future__ import annotations

import subprocess
import sys

import pytest
from avoferric_testlib import OH, PLUGIN_DIR, WATER, make_cjson, run_plugin

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = pytest.importorskip("tomli")

H2 = [(1, 0.0, 0.0, 0.0), (1, 0.0, 0.0, 0.74)]


def _gen(env, atoms, charge=0, mult=1, **opts):
    options = {
        "Title": "test job",
        "Filename Base": "job",
        "Calculation Type": "Single Point",
        "Theory": "HF",
        "Basis": "sto-3g",
        "Charge": charge,
        "Multiplicity": mult,
        "Processor Cores": 0,
        "Memory": 0.0,
        "Dispersion": "none",
    }
    options.update(opts)
    out, _ = run_plugin(
        env, ["ferric_input"], {"options": options, "cjson": make_cjson(atoms)}
    )
    return out


def _run_cli(tmp_path, files):
    for f in files:
        (tmp_path / f["filename"]).write_text(f["contents"], encoding="utf-8")
    # `ferric` console script == ferric._cli_main with sys.argv.
    code = "import sys, ferric; sys.argv = ['ferric', '--no-json', 'job.toml']; ferric._cli_main()"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.parametrize(
    "atoms,mult,opts,expect",
    [
        (WATER, 1, {}, "RHF"),
        (OH, 2, {}, "UHF"),
        (H2, 1, {"Theory": "PBE", "Calculation Type": "Equilibrium Geometry"}, "ptimi"),
        (H2, 1, {"Calculation Type": "Frequencies"}, "requenc"),
        (WATER, 1, {"Theory": "PBE", "Dispersion": "d3bj", "Memory": 1.0}, "D3"),
    ],
)
def test_generated_input_runs_with_the_ferric_cli(
    plugin_env, tmp_path, atoms, mult, opts, expect
):
    env = plugin_env()
    out = _gen(env, atoms, mult=mult, **opts)
    assert out["mainFile"] == "job.toml"
    files = {f["filename"]: f for f in out["files"]}
    assert set(files) == {"job.toml", "job.xyz"}
    assert files["job.toml"]["highlightStyles"] == ["ferric"]
    tomllib.loads(files["job.toml"]["contents"])  # valid TOML at all
    proc = _run_cli(tmp_path, out["files"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert expect.lower() in (proc.stdout + proc.stderr).lower()


@pytest.mark.parametrize("brk", ["\n", "\r\n", "\r"])
def test_a_multiline_title_stays_a_comment(plugin_env, brk):
    """A line break in the Title must not start a live TOML line (CodeRabbit,
    PR #632): `Title = "water<LF>unknown = 1"` used to emit `unknown = 1`."""
    env = plugin_env()
    out = _gen(env, WATER, Title=f"water{brk}unknown = 1")
    files = {f["filename"]: f["contents"] for f in out["files"]}
    parsed = tomllib.loads(files["job.toml"])
    assert "unknown" not in parsed
    assert files["job.toml"].splitlines()[0] == "# water unknown = 1"
    assert files["job.xyz"].splitlines()[1] == "water unknown = 1"


def test_the_cli_check_is_strict(plugin_env, tmp_path):
    """The parse gate can fail: an unknown key in a generated section is refused."""
    env = plugin_env()
    out = _gen(env, WATER)
    for f in out["files"]:
        if f["filename"] == "job.toml":
            f["contents"] = f["contents"].replace("[method]", "[method]\nbogus_key = 1")
    proc = _run_cli(tmp_path, out["files"])
    assert proc.returncode != 0
    assert "bogus_key" in proc.stderr


def test_bad_option_comes_back_as_a_warning(plugin_env):
    env = plugin_env()
    out = _gen(env, WATER, **{"Calculation Type": "Transition State"})
    assert out["warnings"] and "calculation type" in out["warnings"][0]


# ------------------------------------------------------------------ metadata


def _pyproject():
    return tomllib.loads((PLUGIN_DIR / "pyproject.toml").read_text(encoding="utf-8"))


def test_package_metadata_satisfies_avogadro():
    """What avogadrolibs 2.1.0 packagemanager.cpp parsePackage() requires."""
    meta = _pyproject()
    name = meta["project"]["name"]
    assert name.startswith("avogadro-") and "_" not in name
    scripts = meta["project"]["scripts"]
    assert list(scripts) == [name]
    assert meta["tool"]["pixi"]["workspace"]["channels"] == ["conda-forge"]
    assert meta["tool"]["pixi"]["pypi-dependencies"][name]["editable"] is True
    assert any(d.startswith("ferric") for d in meta["project"]["dependencies"])
    avo = meta["tool"]["avogadro"]
    for kind in [
        "energy-models",
        "electrostatic-models",
        "input-generators",
        "menu-commands",
    ]:
        assert avo[kind], kind
        for feat in avo[kind]:
            assert feat["identifier"]
            assert feat["input-format"] == "cjson"
            elements = feat.get("support", {}).get("elements")
            if kind in ("energy-models", "electrostatic-models"):
                # scriptchargemodel.cpp drops everything after a range token
                assert elements and elements.count("-") <= 1 and "," not in elements


def test_every_declared_identifier_is_dispatched():
    from avogadro_ferric import build_parser

    parser = build_parser()
    avo = _pyproject()["tool"]["avogadro"]
    for kind, feats in avo.items():
        if not isinstance(feats, list):
            continue
        for feat in feats:
            flags = []
            if kind == "electrostatic-models":
                flags = ["--charges"]
            args = parser.parse_args([feat["identifier"], *flags, "--lang", "en_US"])
            assert args.feature == feat["identifier"]
    # the flags Avogadro actually sends
    parser.parse_args(["ferric_lowdin", "--potentials", "--lang", "de_DE"])
    parser.parse_args(["config", "--user-options"])
    parser.parse_args(["sp", "--debug", "--lang", "en_US"])


def test_option_and_highlight_files_load():
    meta = _pyproject()
    gen = meta["tool"]["avogadro"]["input-generators"][0]
    opts = tomllib.loads((PLUGIN_DIR / gen["user-options"]).read_text(encoding="utf-8"))
    for key, spec in opts.items():
        assert "type" in spec, key  # jsonwidget.cpp builds no widget without one
        if spec["type"] == "stringList":
            assert 0 <= spec["default"] < len(spec["values"])
    from avogadro_ferric import config, inputgen

    assert opts["Theory"]["values"] == config.METHODS
    assert opts["Basis"]["values"] == config.BASES
    assert set(opts["Calculation Type"]["values"]) == set(inputgen.TASKS)
    styles = tomllib.loads(
        (PLUGIN_DIR / gen["highlight-styles"]).read_text(encoding="utf-8")
    )
    assert list(styles) == ["ferric"]
    import re

    for rule in styles["ferric"]["rules"]:
        assert rule["format"]["preset"] in {
            "title",
            "keyword",
            "property",
            "literal",
            "comment",
        }
        for pat in rule["patterns"]:
            re.compile(pat["regexp"])
