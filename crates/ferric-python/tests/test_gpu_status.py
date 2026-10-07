"""ferric.gpu_status(): shape, types, and the per-mode contract.

The state is a process-wide OnceLock resolved from FERRIC_GPU at first use, so
every env-dependent case runs in a fresh subprocess.
"""

import json
import os
import subprocess
import sys

import ferric

KEYS = {"compiled", "mode", "status", "reason", "device", "precision", "mixed_kernels"}
DEVICE_KEYS = {"ordinal", "name", "cc", "free_bytes", "total_bytes"}
COMPILED = ferric.gpu_status()["compiled"]


def _check_types(st):
    assert set(st) == KEYS
    assert type(st["compiled"]) is bool
    assert st["mode"] in {"off", "auto", "on"}
    assert st["status"] in {"not_compiled", "unavailable", "ready"}
    assert st["precision"] in {"f64", "mixed"}
    assert isinstance(st["mixed_kernels"], list) and all(
        isinstance(k, str) for k in st["mixed_kernels"]
    )
    assert st["reason"] is None or isinstance(st["reason"], str)
    if st["device"] is None:
        assert st["status"] != "ready"
    else:
        dev = st["device"]
        assert st["status"] == "ready"
        assert set(dev) == DEVICE_KEYS
        assert type(dev["ordinal"]) is int
        assert isinstance(dev["name"], str) and dev["name"]
        major, minor = dev["cc"].split(".")
        assert major.isdigit() and minor.isdigit()
        assert type(dev["free_bytes"]) is int and type(dev["total_bytes"]) is int
        assert 0 < dev["free_bytes"] <= dev["total_bytes"]


def _in_subprocess(value):
    """gpu_status() in a fresh process with FERRIC_GPU=value (None = unset)."""
    env = dict(os.environ)
    env.pop("FERRIC_GPU", None)
    env.pop("FERRIC_GPU_PRECISION", None)
    env.pop("FERRIC_GPU_MIXED_KERNELS", None)
    if value is not None:
        env["FERRIC_GPU"] = value
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, ferric; print(json.dumps(ferric.gpu_status()))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    st = json.loads(out.stdout.strip().splitlines()[-1])
    _check_types(st)
    return st


def _skip_or_fail(reason):
    print(f"skipping: {reason}")
    assert os.environ.get("FERRIC_GPU_TESTS_REQUIRED") != "1", reason


def test_gpu_status_shape_and_types_in_this_process():
    _check_types(ferric.gpu_status())


def test_default_is_off_and_touches_no_device():
    st = _in_subprocess(None)
    assert st["compiled"] is COMPILED and st["mode"] == "off"
    if COMPILED:
        assert st["status"] == "unavailable" and st["reason"] == "mode off"
    else:
        assert st["status"] == "not_compiled" and st["reason"] is None
    assert st["device"] is None


def test_explicit_off():
    st = _in_subprocess("off")
    assert st["mode"] == "off" and st["device"] is None
    assert st["status"] == ("unavailable" if COMPILED else "not_compiled")


def test_default_build_contract():
    if COMPILED:
        return _skip_or_fail_not_required(
            "gpu build; default-build contract not applicable"
        )
    st = _in_subprocess(None)
    assert st == {
        "compiled": False,
        "mode": "off",
        "status": "not_compiled",
        "reason": None,
        "device": None,
        "precision": "f64",
        "mixed_kernels": ["rimp2-energy"],
    }


def test_precision_keys_present_and_default_f64():
    st = _in_subprocess(None)
    assert st["precision"] == "f64" and st["mixed_kernels"] == ["rimp2-energy"]


def test_mixed_without_a_device_mode_degrades_to_off_and_says_why():
    env = dict(os.environ)
    env.pop("FERRIC_GPU", None)
    env["FERRIC_GPU_PRECISION"] = "mixed"
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, ferric; print(json.dumps(ferric.gpu_status()))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    st = json.loads(out.stdout.strip().splitlines()[-1])
    _check_types(st)
    # library path: an unsatisfiable knob set degrades to off with the reason
    assert st["mode"] == "off" and st["precision"] == "f64"
    if COMPILED:
        assert st["status"] == "unavailable" and "precision" in st["reason"]


def _skip_or_fail_not_required(reason):
    # Absence of the feature is not a device skip; never a failure.
    print(f"skipping: {reason}")


def test_auto_degrades_never_errors():
    st = _in_subprocess("auto")
    assert st["mode"] == "auto"
    if not COMPILED:
        assert st["status"] == "not_compiled" and st["reason"] is None
    elif st["status"] == "unavailable":
        assert st["reason"]
        _skip_or_fail("no usable device for FERRIC_GPU=auto")
    else:
        assert st["status"] == "ready"


def test_on_without_the_feature_is_unavailable_with_a_reason():
    if COMPILED:
        return _skip_or_fail_not_required("gpu build")
    st = _in_subprocess("on")
    assert st["mode"] == "on" and st["status"] == "unavailable"
    assert "without the gpu feature" in st["reason"]
    assert st["device"] is None


def test_on_with_the_feature_is_ready_or_names_the_reason():
    if not COMPILED:
        return _skip_or_fail_not_required("default build")
    st = _in_subprocess("on")
    assert st["mode"] == "on"
    if st["status"] == "ready":
        assert st["device"] is not None and st["reason"] is None
        assert st["device"]["total_bytes"] > 0
    else:
        assert st["status"] == "unavailable"
        assert "mode = on but no usable device" in st["reason"]
        _skip_or_fail("no usable device for FERRIC_GPU=on")


def test_ready_branch_populates_the_device():
    if not COMPILED:
        return _skip_or_fail_not_required("default build")
    st = _in_subprocess("auto")
    if st["status"] != "ready":
        return _skip_or_fail("no usable device")
    assert st["reason"] is None
    assert set(st["device"]) == DEVICE_KEYS


def test_malformed_value_degrades_to_off_and_says_why():
    st = _in_subprocess("bogus")
    assert st["mode"] == "off" and st["device"] is None
    if COMPILED:
        assert st["status"] == "unavailable"
        assert "FERRIC_GPU" in st["reason"]
    else:
        assert st["status"] == "not_compiled" and st["reason"] is None


# ── ferric.configure_gpu(): process-global, so every case is a subprocess ──


def _configure(code, **env_extra):
    """Run `code` in a fresh process with a cleared GPU environment.

    Returns (returncode, last stdout line, stderr)."""
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("FERRIC_GPU"):
            del env[k]
    env.update(env_extra)
    p = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    lines = p.stdout.strip().splitlines()
    return p.returncode, (lines[-1] if lines else ""), p.stderr


def _configured(kwargs):
    rc, out, err = _configure(
        f"import json, ferric; print(json.dumps(ferric.configure_gpu(**{kwargs!r})))"
    )
    assert rc == 0, err
    st = json.loads(out)
    _check_types(st)
    return st


def _refusal(code, exc="ValueError", **env_extra):
    """The message of the exception `code` must raise (its last stderr line)."""
    rc, _, err = _configure(code, **env_extra)
    assert rc != 0
    last = err.strip().splitlines()[-1]
    assert last.startswith(exc + ": "), last
    return last[len(exc) + 2 :]


def test_configure_gpu_preset_off_equals_the_untouched_default():
    st = _configured({"preset": "off"})
    assert st == _in_subprocess(None)
    assert st["mode"] == "off" and st["precision"] == "f64"
    assert st["mixed_kernels"] == ["rimp2-energy"]


def test_configure_gpu_returns_what_gpu_status_reports_afterwards():
    rc, out, err = _configure(
        "import json, ferric; a = ferric.configure_gpu(preset='auto'); "
        "print(json.dumps([a, ferric.gpu_status()]))"
    )
    assert rc == 0, err
    a, b = json.loads(out)
    assert a == b and a["mode"] == "auto" and a["precision"] == "f64"


def test_configure_gpu_auto_mixed_degrades_without_a_device():
    st = _configured({"preset": "auto-mixed"})
    assert st["mode"] == "auto"
    if st["status"] == "ready":
        assert st["precision"] == "mixed" and st["mixed_kernels"] == ["rimp2-energy"]
    else:
        # an unusable device never leaves a run labelled mixed
        assert st["status"] in ("not_compiled", "unavailable")
        assert st["device"] is None
        if COMPILED:
            _skip_or_fail("no usable device for preset auto-mixed")


def test_configure_gpu_mixed_preset_on_the_device_or_a_named_refusal():
    rc, out, err = _configure(
        "import json, ferric; print(json.dumps(ferric.configure_gpu(preset='mixed')))"
    )
    if rc == 0:
        st = json.loads(out)
        assert st["mode"] == "on" and st["status"] == "ready"
        assert st["precision"] == "mixed" and st["mixed_kernels"] == ["rimp2-energy"]
    elif COMPILED:
        assert "[gpu] mode = on but no usable device: " in err
        _skip_or_fail("no usable device for preset mixed")
    else:
        assert err.strip().splitlines()[-1] == (
            "ValueError: [gpu] mode = on but this binary was built without the gpu "
            "feature; rebuild with `--features ferric-cli/gpu`"
        )


def test_configure_gpu_fine_keys_that_agree_with_the_preset_are_kept():
    st = _configured({"preset": "off", "mode": "off", "precision": "f64"})
    assert st["mode"] == "off" and st["precision"] == "f64"
    st = _configured({"preset": "auto", "device": 0, "min_flops": 1, "memory_gb": 1.0})
    assert st["mode"] == "auto"


def test_configure_gpu_preset_vs_precision_conflict_names_both_keys():
    msg = _refusal(
        "import ferric; ferric.configure_gpu(preset='mixed', precision='f64')"
    )
    assert msg == (
        "[gpu] preset = mixed (explicit (config/TOML/kwarg)) implies precision = mixed "
        "but FERRIC_GPU_PRECISION / [gpu] precision = f64 (explicit (config/TOML/kwarg)); "
        "set one of them"
    )


def test_configure_gpu_preset_vs_mode_conflict_names_both_keys():
    msg = _refusal("import ferric; ferric.configure_gpu(preset='off', mode='on')")
    assert msg == (
        "[gpu] preset = off (explicit (config/TOML/kwarg)) implies mode = off "
        "but FERRIC_GPU / [gpu] mode = on (explicit (config/TOML/kwarg)); set one of them"
    )


def test_configure_gpu_kwarg_preset_against_env_mode_names_both_sources():
    msg = _refusal("import ferric; ferric.configure_gpu(preset='off')", FERRIC_GPU="on")
    assert msg == (
        "[gpu] preset = off (explicit (config/TOML/kwarg)) implies mode = off "
        "but FERRIC_GPU / [gpu] mode = on (env); set one of them"
    )


def test_configure_gpu_unknown_values_are_value_errors_naming_the_key():
    assert _refusal("import ferric; ferric.configure_gpu(preset='bogus')") == (
        '[gpu] preset: invalid GPU preset "bogus" '
        "(expected off, auto, on, mixed or auto-mixed)"
    )
    assert _refusal("import ferric; ferric.configure_gpu(precision='half')") == (
        '[gpu] precision: invalid GPU precision "half" (expected f64 or mixed)'
    )
    assert _refusal("import ferric; ferric.configure_gpu(mode='sometimes')").startswith(
        "[gpu] mode: "
    )
    msg = _refusal("import ferric; ferric.configure_gpu(mixed_kernels=['nope'])")
    assert msg == (
        '[gpu] mixed_kernels: unknown mixed-precision kernel "nope" '
        "(valid: rimp2-energy, ccsd-amplitudes, dfk-occ, cosx-kern)"
    )


def test_configure_gpu_mixed_needs_a_device_mode():
    msg = _refusal("import ferric; ferric.configure_gpu(precision='mixed')")
    assert msg == (
        "[gpu] precision = mixed requires mode = auto or on "
        "(FERRIC_GPU / [gpu] mode is off)"
    )


def test_configure_gpu_kernel_list_under_f64_is_refused():
    msg = _refusal(
        "import ferric; ferric.configure_gpu(mixed_kernels=['rimp2-energy'])"
    )
    assert msg == (
        "FERRIC_GPU_MIXED_KERNELS / [gpu] mixed_kernels = rimp2-energy given but the "
        'precision is f64; set [gpu] precision = "mixed" (or use preset = "mixed") '
        "or drop the list"
    )


def test_configure_gpu_on_without_the_feature_is_refused():
    if COMPILED:
        return _skip_or_fail_not_required("gpu build")
    msg = _refusal("import ferric; ferric.configure_gpu(mode='on')")
    assert msg == (
        "[gpu] mode = on but this binary was built without the gpu feature; "
        "rebuild with `--features ferric-cli/gpu`"
    )


def test_configure_gpu_twice_identical_is_ok_and_different_is_a_runtime_error():
    rc, out, err = _configure(
        "import json, ferric; a = ferric.configure_gpu(preset='off'); "
        "b = ferric.configure_gpu(preset='off'); print(json.dumps(a == b))"
    )
    assert rc == 0, err
    assert out == "true"
    assert err.count("FERRIC_GPU_PRESET: off") == 1  # audited once
    msg = _refusal(
        "import ferric; ferric.configure_gpu(preset='off'); "
        "ferric.configure_gpu(preset='auto')",
        exc="RuntimeError",
    )
    assert msg.startswith("[gpu] settings already installed (")
    assert "requested GpuSettings" in msg
    assert "GPU settings are process-global" in msg


def test_configure_gpu_after_gpu_status_read_the_settings_is_refused():
    msg = _refusal(
        "import ferric; ferric.gpu_status(); ferric.configure_gpu(preset='auto')",
        exc="RuntimeError",
    )
    assert "already installed" in msg and "process-global" in msg
