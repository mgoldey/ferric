"""ferric.configure_precision and the precision provenance record.

The GPU/precision settings are process-global, so every configure case runs in
a fresh subprocess with the FERRIC_GPU* environment cleared. No device is used:
the [gpu] mode stays off.
"""

import json
import os
import subprocess
import sys

import ferric

F64_RECORD = {"precision": "f64", "counter_scope": "process", "kernels": []}

_WATER = (
    "m = ferric.Molecule.from_xyz_string("
    "'3\\n\\nO 0 0 0.1173\\nH 0 0.7572 -0.4692\\nH 0 -0.7572 -0.4692\\n'); "
)


def _run(code, **extra_env):
    env = {k: v for k, v in os.environ.items() if not k.startswith("FERRIC_GPU")}
    env.update(extra_env)
    p = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    lines = p.stdout.strip().splitlines()
    return p.returncode, (lines[-1] if lines else ""), p.stderr


def _ok(code):
    rc, out, err = _run(code)
    assert rc == 0, err
    return json.loads(out)


def _refusal(code, exc="ValueError", **extra_env):
    rc, _, err = _run(code, **extra_env)
    assert rc != 0
    last = err.strip().splitlines()[-1]
    assert last.startswith(exc + ": "), last
    return last[len(exc) + 2 :]


# ── the record on results (this process: nothing configured, all f64) ──


def _water():
    return ferric.Molecule.from_xyz_string(
        "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"
    )


def test_f64_results_carry_the_explicit_f64_record():
    m = _water()
    bs = ferric.BasisSet.bundled("sto-3g")
    assert ferric.run_rhf(m, bs).precision == F64_RECORD
    mp2 = ferric.run_rimp2(m, bs, ferric.BasisSet.bundled("cc-pvdz-ri"))
    assert mp2.precision == F64_RECORD
    assert mp2.scf_precision == F64_RECORD
    oh = ferric.Molecule.from_xyz_string("2\n\nO 0 0 0\nH 0 0 0.97\n", 0, 2)
    assert ferric.run_uhf(oh, bs).precision == F64_RECORD


def test_a_dft_result_carries_the_record():
    r = ferric.run_dft(_water(), ferric.BasisSet.bundled("sto-3g"), "LDA")
    assert r.precision == F64_RECORD


def test_reading_the_record_does_not_freeze_the_gpu_settings():
    # An f64 run reports its record without initialising the settings, so a
    # configure call after it still works (the record peeks, it never reads).
    rc, out, err = _run(
        "import json, ferric; "
        + _WATER
        + "r = ferric.run_rhf(m, ferric.BasisSet.bundled('sto-3g')); "
        "p = r.precision; s = ferric.configure_gpu(preset='auto'); "
        "print(json.dumps([p, s['mode']]))"
    )
    assert rc == 0, err
    assert json.loads(out) == [F64_RECORD, "auto"]


# ── configure_precision ──


def test_configure_precision_returns_the_staged_request():
    assert _ok(
        "import json, ferric; print(json.dumps(ferric.configure_precision(mode='f64')))"
    ) == {"mode": "f64", "mixed_kernels": None}
    assert _ok(
        "import json, ferric; print(json.dumps(ferric.configure_precision("
        "mode='mixed', mixed_kernels=['rimp2-energy'])))"
    ) == {"mode": "mixed", "mixed_kernels": ["rimp2-energy"]}


def test_configure_precision_is_strict():
    assert _refusal("import ferric; ferric.configure_precision(mode='half')") == (
        '[precision] mode: invalid GPU precision "half" (expected f64 or mixed)'
    )
    assert _refusal(
        "import ferric; ferric.configure_precision(mixed_kernels=['diis'])"
    ) == (
        '[precision] mixed_kernels: unknown mixed-precision kernel "diis" '
        "(valid: rimp2-energy, ccsd-amplitudes, dfk-occ, cosx-kern, dfj-pack, xc-vxc)"
    )
    assert _refusal("import ferric; ferric.configure_precision(mixed_kernels=[])") == (
        '[precision] mixed_kernels: unknown mixed-precision kernel "" '
        "(valid: rimp2-energy, ccsd-amplitudes, dfk-occ, cosx-kern, dfj-pack, xc-vxc)"
    )
    assert _refusal(
        "import ferric; ferric.configure_precision(mixed_kernels=['rimp2-energy,xc-vxc'])"
    ) == (
        "[precision] mixed_kernels: kernel names are separate list items, "
        'got "rimp2-energy,xc-vxc"'
    )
    assert _refusal(
        "import ferric; ferric.configure_precision(mode='mixed'); "
        "ferric.configure_precision(mode='f64')"
    ) == (
        "[precision] mode = mixed conflicts with the same key staged again = f64; "
        "set it once"
    )


def test_configure_precision_then_configure_gpu_installs_both():
    st = _ok(
        "import json, ferric; ferric.configure_precision(mode='f64'); "
        "print(json.dumps(ferric.configure_gpu(mode='auto')))"
    )
    assert st["mode"] == "auto" and st["precision"] == "f64"


def test_the_configure_gpu_alias_must_agree_with_the_staged_request():
    msg = _refusal(
        "import ferric; ferric.configure_precision(mode='mixed'); "
        "ferric.configure_gpu(mode='auto', precision='f64')"
    )
    assert msg == (
        "[precision] mode = mixed and [gpu] precision = f64 disagree ([gpu] precision "
        "is a deprecated alias of [precision] mode); set only [precision] mode"
    )


def test_configure_precision_after_install_is_a_runtime_error():
    msg = _refusal(
        "import ferric; ferric.configure_gpu(preset='off'); "
        "ferric.configure_precision(mode='f64')",
        exc="RuntimeError",
    )
    assert msg.startswith("[gpu] settings already installed")
    assert "call configure_precision before configure_gpu" in msg


def test_cpu_only_mixed_names_the_device_kernels():
    # No kernel this build admits has a CPU mixed path, so CPU-only mixed is
    # refused with the reason when the settings are installed.
    msg = _refusal(
        "import ferric; ferric.configure_precision(mode='mixed'); ferric.configure_gpu()"
    )
    assert msg == (
        "[precision] mode = mixed requires mode = auto or on (FERRIC_GPU / [gpu] mode "
        "is off): rimp2-energy runs only on a device; CPU mixed-precision kernels in "
        "this build: none"
    )


def test_a_staged_f64_request_keeps_the_record_f64():
    rec = _ok(
        "import json, ferric; ferric.configure_precision(mode='f64'); "
        "ferric.configure_gpu(); "
        + _WATER
        + "print(json.dumps(ferric.run_rhf(m, ferric.BasisSet.bundled('sto-3g')).precision))"
    )
    assert rec == F64_RECORD


STAGED_NOT_INSTALLED = (
    "configure_precision staged a request; call configure_gpu() to install it"
)


def test_an_uninstalled_staged_request_raises_at_the_first_calculation():
    # Staged but never installed: the calculation refuses to start rather
    # than run with the request silently dropped (or silently resolved).
    for request in ("mode='f64'", "mode='mixed', mixed_kernels=['rimp2-energy']"):
        msg = _refusal(
            f"import ferric; ferric.configure_precision({request}); "
            + _WATER
            + "ferric.run_rhf(m, ferric.BasisSet.bundled('sto-3g'))",
            exc="RuntimeError",
        )
        assert STAGED_NOT_INSTALLED in msg, msg
    msg = _refusal(
        "import ferric; ferric.configure_precision(mode='mixed'); "
        + _WATER
        + "ferric.run_rimp2(m, ferric.BasisSet.bundled('sto-3g'), "
        "ferric.BasisSet.bundled('cc-pvdz-ri'))",
        exc="RuntimeError",
    )
    assert STAGED_NOT_INSTALLED in msg, msg


def test_an_invalid_staged_request_never_turns_a_user_enabled_gpu_off():
    # FERRIC_GPU=auto enables the GPU; the staged request is invalid
    # (cosx-kern is not shipped). The lazy settings read must not resolve the
    # staged request, which used to fail and degrade the whole GPU to `off`:
    # the env's mode stands, and the calculation raises instead.
    rc, out, err = _run(
        "import json, ferric; "
        "ferric.configure_precision(mode='mixed', mixed_kernels=['cosx-kern']); "
        "s = ferric.gpu_status(); print(json.dumps([s['mode'], s['precision']]))",
        FERRIC_GPU="auto",
    )
    assert rc == 0, err
    assert json.loads(out) == ["auto", "f64"]
    assert "GPU disabled" not in err, err
    msg = _refusal(
        "import ferric; "
        "ferric.configure_precision(mode='mixed', mixed_kernels=['cosx-kern']); "
        + _WATER
        + "ferric.run_rhf(m, ferric.BasisSet.bundled('sto-3g'))",
        exc="RuntimeError",
        FERRIC_GPU="auto",
    )
    assert STAGED_NOT_INSTALLED in msg, msg


def test_a_mixed_rimp2_request_on_the_cpu_build_is_recorded():
    # The request side of the record: mixed requested and installed, no
    # device, so the MP2 energy runs in f64 on the CPU and says so.
    rec = _ok(
        "import json, ferric; "
        "ferric.configure_precision(mode='mixed', mixed_kernels=['rimp2-energy']); "
        "ferric.configure_gpu(mode='auto'); "
        + _WATER
        + "r = ferric.run_rimp2(m, ferric.BasisSet.bundled('sto-3g'), "
        "ferric.BasisSet.bundled('cc-pvdz-ri')); "
        "print(json.dumps([r.precision, r.scf_precision]))"
    )
    mp2, scf = rec
    assert mp2["precision"] == "f64", mp2
    assert mp2["counter_scope"] == "process", mp2
    assert len(mp2["kernels"]) == 1, mp2
    k = mp2["kernels"][0]
    assert k["kernel"] == "rimp2-energy"
    assert k["requested"] == "mixed", k
    assert k["mode"] == "f64", k
    assert k["calls_mixed"] == 0, k
    assert k["device"] is None, k
    assert k["calls_f64_fallback"]["cpu_f64"] > 0, k
    assert scf == F64_RECORD, scf
