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
        "mixed_kernels": [],
    }


def test_precision_keys_present_and_default_f64():
    st = _in_subprocess(None)
    assert st["precision"] == "f64" and st["mixed_kernels"] == []


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
