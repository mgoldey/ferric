"""`python -m tools.tox` WITHOUT `--offline`: the exit contract under outages.

The defect this file pins (2026-09-23): the online mode exited 2 on EVERY run.
ADMETlab's documented path 404'd, and the ProTox stub's by-design "no JSON
API" note was counted as a provider failure, so even a healthy ADMETlab could
never produce 0. Exit 2 was also argparse's usage-error status. An
infrastructure outage therefore looked exactly like a broken screen or a
typo'd flag.

Every test here mocks the two network seams in `tools.tox.web`
(`_post_json` for ADMETlab, `_get_status` for ProTox); none touches the live
network.
"""

from __future__ import annotations

import json
import time
import urllib.error

import pytest

pytest.importorskip("rdkit", reason="the local alert screen is RDKit-based")

import tools.tox.web as web  # noqa: E402
from tools.tox.__main__ import (  # noqa: E402
    EXIT_ALERTS,
    EXIT_CLEAN,
    EXIT_INPUT_ERROR,
    EXIT_ONLINE_UNAVAILABLE,
    EXIT_REQUIRED_CHECK_FAILED,
    main,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"  # 2 structural alerts
ETHANOL = "CCO"  # 0 structural alerts

# The `/api/single/admet` shape, trimmed to the columns the client maps.
_GOOD_ADMETLAB = {
    "status": "success",
    "code": 200,
    "data": {
        "smiles": "X",
        "data": {
            "toxicity": [
                {"name": "DILI", "value": 0.12},
                {"name": "hERG", "value": 0.03},
                {"name": "Ames", "value": 0.2},
            ],
            "absorption": [{"name": "f20", "value": 0.9}],
        },
    },
}


def _admetlab_ok(url, payload, timeout):
    return _GOOD_ADMETLAB


def _protox_ok(url, timeout):
    return 200


def _raiser(exc):
    def f(*a, **k):
        raise exc

    return f


_OUTAGES = {
    "http_404": urllib.error.HTTPError("u", 404, "Not Found", {}, None),
    "http_429": urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None),
    "refused": urllib.error.URLError(ConnectionRefusedError(111, "refused")),
    "dns": urllib.error.URLError("[Errno -2] Name or service not known"),
}


@pytest.fixture
def net(monkeypatch):
    """Healthy providers by default; tests override one seam at a time."""

    def set_(admetlab=_admetlab_ok, protox=_protox_ok):
        monkeypatch.setattr(web, "_post_json", admetlab)
        monkeypatch.setattr(web, "_get_status", protox)

    set_()
    return set_


def _json_run(argv, capsys):
    rc = main(["--json", *argv])
    cap = capsys.readouterr()
    return rc, json.loads(cap.out), cap.err


# ── an outage on each provider ──


@pytest.mark.parametrize("kind", sorted(_OUTAGES))
def test_admetlab_outage_is_exit_3_with_offline_results_and_provider_named(
    net, capsys, kind
):
    """Catches: an outage exiting 2 (the reported defect) or 0 (folded into
    success), and the offline screen being dropped when a web provider fails.
    """
    net(admetlab=_raiser(_OUTAGES[kind]))
    rc = main([ASPIRIN])
    cap = capsys.readouterr()
    assert rc == EXIT_ONLINE_UNAVAILABLE
    # Offline results still present -- the VALUES, not just the labels.
    assert "180.2" in cap.out and "63.6" in cap.out
    # The unavailable provider is named, with its status and why.
    assert "admetlab3 [unavailable]" in cap.out
    assert "online provider admetlab3 unavailable" in cap.err
    if kind.startswith("http_"):
        assert kind.split("_")[1] in cap.err


def test_admetlab_outage_is_machine_readable(net, capsys):
    """Catches: JSON consumers being unable to tell WHICH provider was down
    (only a free-text error, or no per-provider status at all)."""
    net(admetlab=_raiser(_OUTAGES["http_404"]))
    rc, data, _ = _json_run([ASPIRIN], capsys)
    assert rc == EXIT_ONLINE_UNAVAILABLE
    st = data[0]["provider_status"]
    assert st == {
        "rdkit-alerts": "ok",
        "admetlab3": "unavailable",
        "protox3": "unsupported",
    }
    assert "404" in data[0]["provider_errors"]["admetlab3"]
    assert any(e["name"] == "desc_mw" for e in data[0]["endpoints"])


def test_admetlab_response_shape_change_is_an_outage_not_a_verdict(net, capsys):
    """Catches: a moved/changed API (200 with an unexpected body) reading as a
    clean molecule. It is an unavailable online check, exit 3."""
    net(admetlab=lambda *a: {"detail": "moved"})
    rc = main([ETHANOL])
    err = capsys.readouterr().err
    assert rc == EXIT_ONLINE_UNAVAILABLE
    assert "unexpected response shape" in err


def test_protox_outage_does_not_change_the_status(net, capsys):
    """ProTox never returns endpoints (no JSON API), so its outage cannot make
    a screen less complete. Catches: its by-design note (or its outage)
    being counted as a failure again. That is half of why the online mode
    could never exit 0. It is still NAMED, so it is not hidden."""
    net(protox=_raiser(_OUTAGES["refused"]))
    rc, data, _ = _json_run([ETHANOL], capsys)
    assert rc == EXIT_CLEAN
    assert data[0]["provider_status"]["protox3"] == "unsupported"
    assert "unreachable" in data[0]["provider_errors"]["protox3"]


def test_every_web_provider_down_still_reports_the_offline_screen(net, capsys):
    """Catches: a total network outage losing the local results or exiting
    with the input-error / required-failure codes."""
    net(admetlab=_raiser(_OUTAGES["dns"]), protox=_raiser(_OUTAGES["dns"]))
    rc, data, err = _json_run([ASPIRIN, ETHANOL], capsys)
    assert rc == EXIT_ONLINE_UNAVAILABLE
    for mol in data:
        assert mol["provider_status"]["rdkit-alerts"] == "ok"
        assert any(e["name"] == "alert_total_count" for e in mol["endpoints"])
    assert "admetlab3" in err


# ── healthy providers ──


def test_all_providers_ok_and_a_clean_molecule_exits_0(net, capsys):
    """Catches: the ProTox note or any other by-design message turning a fully
    healthy online run into a failure (the original always-2 defect)."""
    rc, data, _ = _json_run(["--fail-on-alerts", ETHANOL], capsys)
    assert rc == EXIT_CLEAN
    names = {e["name"] for e in data[0]["endpoints"]}
    assert {"dili", "herg", "ames_mutagenicity"} <= names
    assert data[0]["provider_status"]["admetlab3"] == "ok"


def test_alerts_exit_4_only_when_asked(net, capsys):
    """Catches: alerts changing the default status (which would break every
    existing `--offline` caller: aspirin has always exited 0), and
    `--fail-on-alerts` failing to signal them."""
    assert main([ASPIRIN]) == EXIT_CLEAN
    assert main(["--fail-on-alerts", ASPIRIN]) == EXIT_ALERTS
    assert main(["--offline", "--fail-on-alerts", ASPIRIN]) == EXIT_ALERTS
    assert main(["--offline", "--fail-on-alerts", ETHANOL]) == EXIT_CLEAN


def test_alerts_outrank_an_online_outage(net, capsys):
    """Precedence 4 > 3: the local screen's positive finding is valid without
    the web providers. Catches: an outage hiding a found alert."""
    net(admetlab=_raiser(_OUTAGES["http_404"]))
    assert main(["--fail-on-alerts", ASPIRIN]) == EXIT_ALERTS


# ── --require-online ──


def test_require_online_makes_an_outage_fatal(net, capsys):
    """Catches: a caller who demanded online predictions getting the
    degraded-but-usable 3 (or a 0) when they did not run."""
    net(admetlab=_raiser(_OUTAGES["refused"]))
    rc = main(["--require-online", ASPIRIN])
    cap = capsys.readouterr()
    assert rc == EXIT_REQUIRED_CHECK_FAILED
    assert "error: online provider admetlab3 unavailable" in cap.err
    assert "180.2" in cap.out, "results are still printed on a hard failure"


def test_require_online_with_healthy_providers_exits_0(net, capsys):
    """Catches: --require-online treating the ProTox stub as a missing check,
    which would make the flag fail on every run."""
    assert main(["--require-online", ETHANOL]) == EXIT_CLEAN


def test_require_online_and_offline_together_is_a_usage_error(net, capsys):
    """Catches: contradictory flags silently picking one."""
    assert main(["--offline", "--require-online", ETHANOL]) == EXIT_INPUT_ERROR


def test_a_bad_flag_is_a_usage_error_not_exit_2(capsys):
    """argparse's default usage-error status is 2, which this CLI uses for
    "a required check did not run". Catches: the two meanings colliding
    again."""
    assert main(["--no-such-flag", ETHANOL]) == EXIT_INPUT_ERROR
    assert main(["--timeout", "0", ETHANOL]) == EXIT_INPUT_ERROR


def test_an_unparseable_smiles_online_is_still_an_input_error(net, capsys):
    """Once web providers run, a service may return value=None rows for a
    SMILES RDKit rejected, so "no endpoints at all" no longer identifies it.
    Catches: a bad SMILES being reported as an outage (3) or as clean (0)."""
    invalid = {"data": {"data": {"toxicity": [{"name": "DILI", "value": "Invalid"}]}}}
    net(admetlab=lambda *a: invalid)
    assert main(["not!a!smiles"]) == EXIT_INPUT_ERROR
    assert "could not assess" in capsys.readouterr().err


# ── bounded timeouts ──


def _slow(seconds):
    def f(*a, **k):
        time.sleep(seconds)
        return _GOOD_ADMETLAB

    return f


def test_a_hung_admetlab_is_cut_off_at_the_deadline(net, capsys):
    """The mock ignores urllib's socket timeout entirely (as a slow-drip server
    effectively does). Catches: the CLI waiting on the provider's own pace, so
    an outage hangs it."""
    net(admetlab=_slow(30))
    t0 = time.monotonic()
    rc = main(["--timeout", "0.5", ETHANOL])
    elapsed = time.monotonic() - t0
    err = capsys.readouterr().err
    assert rc == EXIT_ONLINE_UNAVAILABLE
    assert elapsed < 10, f"took {elapsed:.1f} s against a 0.5 s deadline"
    assert "timed out" in err


def test_a_hung_provider_costs_one_timeout_per_run_not_per_molecule(
    net, capsys, monkeypatch
):
    """Catches: the batch re-trying a dead service for every molecule, i.e. N
    timeouts. The skipped molecules must still be marked unavailable."""
    calls = {"admetlab": 0, "protox": 0}

    def slow_admetlab(*a, **k):
        calls["admetlab"] += 1
        time.sleep(30)

    def slow_protox(*a, **k):
        calls["protox"] += 1
        time.sleep(30)

    net(admetlab=slow_admetlab, protox=slow_protox)
    t0 = time.monotonic()
    rc, data, _ = _json_run(["--timeout", "0.3", ETHANOL, ASPIRIN, "CCCCO"], capsys)
    elapsed = time.monotonic() - t0
    assert rc == EXIT_ONLINE_UNAVAILABLE
    assert calls == {"admetlab": 1, "protox": 1}
    assert elapsed < 10, f"took {elapsed:.1f} s; 5 skipped calls were not skipped"
    for mol in data[1:]:
        assert mol["provider_status"]["admetlab3"] == "unavailable"
        assert "earlier in this run" in mol["provider_errors"]["admetlab3"]


def test_an_http_error_does_not_trip_the_breaker(net, capsys):
    """A 429/404 answers quickly, so it is re-tried per molecule (a rate limit
    may clear). Catches: one transient 429 silently dropping ADMETlab for the
    rest of a batch."""
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _OUTAGES["http_429"]
        return _GOOD_ADMETLAB

    net(admetlab=flaky)
    rc, data, _ = _json_run([ETHANOL, "CCCCO"], capsys)
    assert calls["n"] == 2
    assert data[1]["provider_status"]["admetlab3"] == "ok"
    assert rc == EXIT_ONLINE_UNAVAILABLE, "the first molecule still lacked it"


# ── the revived ADMETlab client ──


def test_admetlab_single_endpoint_payload_and_parsing(monkeypatch):
    """Catches: the client posting the batch body (a list) to the
    single-molecule path, which the live service rejects with HTTP 422, or
    failing to flatten the category-grouped response."""
    seen = {}

    def capture(url, payload, timeout):
        seen["url"], seen["payload"] = url, payload
        return _GOOD_ADMETLAB

    monkeypatch.setattr(web, "_post_json", capture)
    p = web.AdmetlabProvider()
    by_name = {e.name: e for e in p.fetch(ETHANOL)}
    assert seen["url"].endswith("/api/single/admet")
    assert seen["payload"] == {"SMILES": ETHANOL}
    assert by_name["dili"].value == pytest.approx(0.12)
    assert p.last_error is None


def test_admetlab_low_bioavailability_is_higher_is_worse(monkeypatch):
    """ADMETlab's f20 is the probability that bioavailability is BELOW 20%.
    Catches: the old `higher_is_worse=False` mapping, which ranked a
    compound predicted to be poorly absorbed as SAFER."""
    monkeypatch.setattr(web, "_post_json", _admetlab_ok)
    e = {e.name: e for e in web.AdmetlabProvider().fetch(ETHANOL)}
    f20 = e["low_bioavailability_20pct"]
    assert f20.value == pytest.approx(0.9)
    assert f20.higher_is_worse is True
