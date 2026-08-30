"""Contract tests for every toxicity provider.

THE INVARIANT THIS FILE EXISTS FOR: a provider that cannot answer must return
NO endpoint, or an endpoint with `value=None`. It must never return `0.0`.

Why that specific bug is worth a dedicated test file: every toxicity endpoint
here is polarity `higher_is_worse=True`, so a fabricated 0.0 reads as
"confidently predicted non-toxic". A dead web service would then silently
promote every candidate to the top of the safety ranking — the exact inversion
that would make this whole experiment worse than not running it. It is also a
completely plausible bug: `dict.get(key, 0.0)` and `float(x or 0)` both produce
it, and both look reasonable in review.

The second invariant: providers must not RAISE for a service failure. A batch
over 20 analogues must not abort because one HTTP call timed out.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tools.tox.alerts import RdkitAlertsProvider
from tools.tox.assess import assess_smiles
from tools.tox.model import ToxAssessment, ToxEndpoint
from tools.tox.web import AdmetlabProvider, ProToxProvider

ETHANOL = "CCO"
DANUGLIPRON = (
    "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)O)N=C2CN4CCC(CC4)"
    "C5=NC(=CC=C5)OCC6=C(C=C(C=C6)C#N)F"
)
# A port that nothing is listening on, so the client takes the connection-refused
# path immediately rather than waiting on a timeout.
UNREACHABLE = "http://127.0.0.1:9"


# ── the core invariant, per provider ──

def test_offline_provider_never_returns_none_score_for_valid_input():
    """The baseline must always produce a usable score -- it is the reason the
    experiment survives every web service being down."""
    a = assess_smiles(DANUGLIPRON, providers=[RdkitAlertsProvider()])
    assert a.liability_score is not None
    assert a.known_endpoints


def test_offline_provider_returns_empty_for_unparseable_smiles():
    """Bad input -> no endpoints, no exception (batch must continue)."""
    assert RdkitAlertsProvider().fetch("this is not a smiles ((((") == []


def test_unreachable_admetlab_yields_no_endpoints_and_no_exception():
    """The headline case: a dead service must not raise, and must not
    fabricate."""
    p = AdmetlabProvider(base_url=UNREACHABLE, timeout=2.0)
    endpoints = p.fetch(ETHANOL)
    assert endpoints == []
    assert p.last_error is not None, "a failure must be recorded, not silent"


def test_unreachable_admetlab_gives_none_score_not_zero():
    """THE fabrication test. With only a dead provider configured, the
    aggregate must be None -- NOT 0.0, which would mean 'maximally safe'.
    """
    a = assess_smiles(ETHANOL, providers=[AdmetlabProvider(base_url=UNREACHABLE, timeout=2.0)])
    assert a.liability_score is None, (
        "a molecule no provider could score must be UNRANKED (None); a 0.0 here "
        "would rank it as the safest compound in the set"
    )
    assert a.known_endpoints == []
    assert "admetlab3" in a.provider_errors


def test_protox_reports_its_limitation_rather_than_scraping():
    """ProTox is an honest stub: no endpoints, and an error string that tells a
    human what to do. Asserting this pins the design decision so nobody
    'improves' it into a silent HTML scraper."""
    p = ProToxProvider(timeout=5.0)
    try:
        endpoints = p.fetch(ETHANOL)
    except Exception as e:  # pragma: no cover
        pytest.fail(f"ProToxProvider.fetch raised {type(e).__name__}: {e}")
    assert endpoints == []
    assert p.last_error is not None
    assert "no documented JSON API" in p.last_error or "unreachable" in p.last_error


# ── malformed-response handling, via a local mock server ──

class _MockHandler(BaseHTTPRequestHandler):
    """Serves whatever `payload`/`status` the test class attribute holds."""
    payload: object = {}
    status: int = 200

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        body = (
            self.payload if isinstance(self.payload, (bytes, bytearray))
            else json.dumps(self.payload).encode()
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence the test log
        pass


class _MockServer:
    def __init__(self, payload, status=200):
        _MockHandler.payload = payload
        _MockHandler.status = status
        self.httpd = HTTPServer(("127.0.0.1", 0), _MockHandler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def test_admetlab_parses_a_well_formed_response():
    """Proves the client is real, not just a graceful-failure shell -- without
    this, every other admetlab test would pass on a stub that always returns
    []. This is the reachable-pass-condition check.
    """
    payload = {"data": {"data": [{"DILI": 0.87, "hERG": "0.42", "Ames": 0.05}]}}
    with _MockServer(payload) as base:
        p = AdmetlabProvider(base_url=base, timeout=10.0)
        endpoints = p.fetch(DANUGLIPRON)

    by_name = {e.name: e for e in endpoints}
    assert by_name["dili"].value == pytest.approx(0.87)
    assert by_name["herg"].value == pytest.approx(0.42), "string values must parse"
    assert by_name["ames_mutagenicity"].value == pytest.approx(0.05)
    assert all(e.source == "admetlab3" for e in endpoints)
    assert p.last_error is None


def test_admetlab_non_numeric_value_becomes_none_not_zero():
    """A category label ('Low', '-', 'NA') must become unknown, not safe."""
    payload = {"data": {"data": [{"DILI": "Low risk", "hERG": None, "Ames": 0.9}]}}
    with _MockServer(payload) as base:
        endpoints = AdmetlabProvider(base_url=base, timeout=10.0).fetch(ETHANOL)

    by_name = {e.name: e for e in endpoints}
    assert by_name["dili"].value is None, "'Low risk' must not become 0.0"
    assert by_name["herg"].value is None
    assert by_name["ames_mutagenicity"].value == pytest.approx(0.9)
    # And the aggregate must ignore the unknowns rather than treating them as 0.
    a = ToxAssessment(smiles=ETHANOL, endpoints=endpoints)
    assert a.liability_score == pytest.approx(0.9)


def test_admetlab_http_error_is_recorded_not_raised():
    with _MockServer({}, status=404) as base:
        p = AdmetlabProvider(base_url=base, timeout=10.0)
        assert p.fetch(ETHANOL) == []
        assert "404" in (p.last_error or "")


def test_admetlab_unexpected_shape_is_recorded_not_guessed():
    with _MockServer({"unexpected": "shape"}) as base:
        p = AdmetlabProvider(base_url=base, timeout=10.0)
        assert p.fetch(ETHANOL) == []
        assert "unexpected response shape" in (p.last_error or "")


def test_admetlab_response_without_tox_columns_is_flagged():
    """A 200 with only ADME columns must not read as 'no toxicity found'."""
    payload = {"data": {"data": [{"logP": 1.2, "MW": 46.0}]}}
    with _MockServer(payload) as base:
        p = AdmetlabProvider(base_url=base, timeout=10.0)
        assert p.fetch(ETHANOL) == []
        assert "none of the expected tox" in (p.last_error or "")


# ── aggregation semantics ──

def test_liability_score_respects_polarity():
    """`higher_is_worse=False` endpoints must be inverted before averaging; a
    dropped inversion would rank high-bioavailability compounds as toxic."""
    good = ToxEndpoint("bioavailability_20pct", 0.9, False, "t")  # good -> 0.1
    bad = ToxEndpoint("dili", 0.9, True, "t")                     # bad  -> 0.9
    assert ToxAssessment("X", [good]).liability_score == pytest.approx(0.1)
    assert ToxAssessment("X", [bad]).liability_score == pytest.approx(0.9)
    assert ToxAssessment("X", [good, bad]).liability_score == pytest.approx(0.5)


def test_liability_score_excludes_non_probability_units():
    """Averaging an LD50 in mg/kg against a probability is meaningless, so
    non-probability endpoints are reported but never aggregated."""
    prob = ToxEndpoint("dili", 0.4, True, "t", units="probability")
    mw = ToxEndpoint("desc_mw", 555.6, True, "t", units="Da")
    a = ToxAssessment("X", [prob, mw])
    assert a.liability_score == pytest.approx(0.4), (
        "a 555.6 Da descriptor leaking into the average would swamp every "
        "probability endpoint"
    )


def test_assessment_with_no_endpoints_at_all_is_none():
    assert ToxAssessment("X", []).liability_score is None


def test_provider_that_raises_is_recorded_as_contract_violation():
    """The driver must survive a misbehaving provider and say what happened."""
    class Exploding:
        name = "exploding"

        def fetch(self, smiles):
            raise RuntimeError("boom")

    a = assess_smiles(ETHANOL, providers=[Exploding(), RdkitAlertsProvider()])
    assert "contract violation" in a.provider_errors["exploding"]
    # ...and the good provider still contributed.
    assert a.liability_score is not None


def test_non_string_input_raises_for_every_provider():
    """Programmer error SHOULD raise (unlike service failure)."""
    for p in (RdkitAlertsProvider(), AdmetlabProvider(), ProToxProvider()):
        with pytest.raises(TypeError):
            p.fetch(None)


def test_providers_expose_a_stable_name_used_in_every_endpoint():
    p = RdkitAlertsProvider()
    assert p.name == "rdkit-alerts"
    assert {e.source for e in p.fetch(ETHANOL)} == {p.name}
