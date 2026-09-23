"""Web-service toxicity providers (ADMETlab 3.0, ProTox-3.0).

These are the *preferred* sources — real trained ADMET/tox models over large
curated datasets, which is not something this repo should be reinventing. They
are also third-party HTTP endpoints, so they are treated as unreliable by
construction:

- every network failure, timeout, HTTP error, and unparseable body becomes an
  empty endpoint list (never an exception, never a fabricated value);
- the reason is recorded on the provider's `last_error` so the driver can put it
  in `ToxAssessment.provider_errors` and the user can see the source was absent
  rather than reading its absence as safety.

**Status as measured 2026-08-29 (re-probed 2026-09-23):** ADMETlab 3.0's
documented `POST /api/admet`
(base `https://admetlab3.scbdd.com`, payload `{"SMILES": [...]}`, as used by the
published API tutorial and community clients) returns **HTTP 404** on every path
variant tried (`/api/admet`, `/api/admet/`, `/api/v1/admet`, `/admet/api`), while
the site root returns 200. The service's own API manual (`/apis/two.md`) still documents that path.

Re-probed 2026-09-23: `/api/admet` and `/api/admetCSV` still 404, but
`POST /api/single/admet` with `{"SMILES": "<one smiles>"}` answers HTTP 200 in
~1 s with the full prediction table, grouped by category
(`{"data": {"data": {"toxicity": [{"name": "DILI", "value": ...}, ...], ...}}}`)
under the SAME column keys the batch API documented (DILI, hERG, H-HT, Ames,
...; bioavailability is `f20`/`f30` rather than `F(20%)`/`F(30%)`). It is
undocumented, so it is the default only because it is the one path on the
host that works; the documented batch shape is still parsed, and
`AdmetlabProvider(endpoint_path="/api/admet", batch_payload=True)` restores it.
The host rate-limits hard (HTTP 429 after a handful of requests in a minute),
which surfaces as an ordinary "unavailable" provider error.

Every web call runs under a WALL-CLOCK deadline (`_call_with_deadline`), not
only urllib's per-socket-operation timeout: a server that trickles bytes can
keep a socket timeout from ever firing, and an outage must not hang the CLI.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request

from .model import STATUS_UNAVAILABLE, STATUS_UNSUPPORTED, ToxEndpoint

_UA = "Mozilla/5.0 (ferric tox-assessment; research use)"

# ADMETlab endpoint keys -> (our endpoint name, higher_is_worse). Only the
# tox/liability subset is mapped; the service returns ~119 columns and pulling
# all of them into a liability average would drown the tox signal in ADME
# descriptors. Names follow ADMETlab 3.0's published column keys.
_ADMETLAB_TOX_KEYS: dict[str, tuple[str, bool]] = {
    "DILI": ("dili", True),
    "hERG": ("herg", True),
    "H-HT": ("hepatotoxicity", True),
    "Ames": ("ames_mutagenicity", True),
    "Carcinogenicity": ("carcinogenicity", True),
    "SkinSen": ("skin_sensitization", True),
    "Respiratory": ("respiratory_toxicity", True),
    # POLARITY: ADMETlab's own column explanation (live response, 2026-09-23)
    # is "Category 1: F20%+ (bioavailability < 20%) ... The output value is the
    # probability of being F20%+". A HIGH value means POOR bioavailability, so
    # higher is worse. These were mapped `higher_is_worse=False` as
    # "bioavailability_20pct" until the endpoint came back -- latent while it
    # 404'd, and a ranking inversion the moment it answered.
    "F(20%)": ("low_bioavailability_20pct", True),
    "F(30%)": ("low_bioavailability_30pct", True),
    # `/api/single/admet` spells the same columns this way; a row carries one
    # spelling or the other, never both.
    "f20": ("low_bioavailability_20pct", True),
    "f30": ("low_bioavailability_30pct", True),
}


def _call_with_deadline(fn, timeout: float):
    """Run `fn()` and return its result, or raise `TimeoutError` after `timeout` s.

    urllib's `timeout=` bounds each socket operation, not the request, so a
    slow-drip server (or DNS stall) can exceed it by any amount. The call runs
    in a daemon thread; on expiry it is abandoned (a daemon thread cannot keep
    the process alive), which is the price of a hard bound without asyncio.
    """
    box: dict[str, object] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"no complete response within {timeout:g} s")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("value")


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": _UA},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310 -- base_url defaults are fixed https:// endpoints; callers are local scripts
        return json.loads(r.read().decode())


class AdmetlabProvider:
    """ADMETlab 3.0 REST client (`POST {base}/api/single/admet`).

    `endpoint_path` is injectable so the tests can point it at a local mock,
    and so a future path change is a one-argument fix at the call site rather
    than an edit here. `batch_payload=True` sends the documented batch body
    (`{"SMILES": [smiles]}`) instead of the single-molecule one.

    After every `fetch`, `last_error` is `None` or the reason, and
    `last_error_kind` is `None` or `model.STATUS_UNAVAILABLE`.
    `last_error_unreachable` is True when the service did not answer at all
    (timeout, refused, DNS) -- the case where retrying later in the same run
    would only cost another full timeout.
    """

    name = "admetlab3"
    online = True

    def __init__(
        self,
        base_url: str = "https://admetlab3.scbdd.com",
        endpoint_path: str = "/api/single/admet",
        timeout: float = 60.0,
        batch_payload: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint_path = endpoint_path
        self.timeout = timeout
        self.batch_payload = batch_payload
        self.last_error: str | None = None
        self.last_error_kind: str | None = None
        self.last_error_unreachable = False

    def _fail(self, msg: str, unreachable: bool = False) -> list[ToxEndpoint]:
        self.last_error = msg
        self.last_error_kind = STATUS_UNAVAILABLE
        self.last_error_unreachable = unreachable
        return []

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.endpoint_path}"

    def fetch(self, smiles: str) -> list[ToxEndpoint]:
        if not isinstance(smiles, str):
            raise TypeError(f"smiles must be str, got {type(smiles).__name__}")
        self.last_error = None
        self.last_error_kind = None
        self.last_error_unreachable = False
        payload = {"SMILES": [smiles] if self.batch_payload else smiles}
        try:
            data = _call_with_deadline(
                lambda: _post_json(self.url, payload, self.timeout), self.timeout
            )
        except urllib.error.HTTPError as e:
            why = " (rate limited)" if e.code == 429 else ""
            return self._fail(f"HTTP {e.code}{why} from {self.url}")
        except TimeoutError as e:
            # Before OSError: TimeoutError is a subclass of it.
            return self._fail(f"timed out contacting {self.url}: {e}", True)
        except (urllib.error.URLError, OSError) as e:
            return self._fail(f"network error contacting {self.url}: {e}", True)
        except json.JSONDecodeError as e:
            return self._fail(f"unparseable JSON from {self.url}: {e}")

        row = _admetlab_row(data)
        if row is None:
            return self._fail(
                f"unexpected response shape from {self.url}: {str(data)[:200]}"
            )

        out: list[ToxEndpoint] = []
        for key, (endpoint_name, worse) in _ADMETLAB_TOX_KEYS.items():
            if key not in row:
                continue
            out.append(
                ToxEndpoint(
                    name=endpoint_name,
                    value=_coerce_float(row[key]),
                    higher_is_worse=worse,
                    source=self.name,
                    units="probability",
                    note=f"ADMETlab 3.0 column {key!r} (raw: {row[key]!r})",
                )
            )
        if not out:
            self._fail(
                f"response from {self.url} contained none of the expected tox "
                f"columns {sorted(_ADMETLAB_TOX_KEYS)}; got {sorted(row)[:12]}"
            )
        return out


def _admetlab_row(data) -> dict | None:
    """Flatten either ADMETlab response shape to one `{column: value}` row.

    Batch (documented): `{"data": {"data": [ {col: val, ...} ]}}`, or a bare
    list. Single (`/api/single/admet`): `{"data": {"data": {category:
    [{"name": col, "value": val, ...}, ...]}}}`. Anything else is `None`:
    a wrapper change is reported, never guessed at.
    """
    rows = None
    if isinstance(data, dict):
        inner = data.get("data")
        if isinstance(inner, dict):
            rows = inner.get("data")
        elif isinstance(inner, list):
            rows = inner
    elif isinstance(data, list):
        rows = data

    if isinstance(rows, list):
        if rows and isinstance(rows[0], dict):
            return rows[0]
        return None
    if isinstance(rows, dict):
        row: dict = {}
        for items in rows.values():
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict) and isinstance(it.get("name"), str):
                    row.setdefault(it["name"], it.get("value"))
        return row or None
    return None


def _get_status(url: str, timeout: float) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": _UA}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310 -- fixed https:// base_url
        return r.status


def _coerce_float(v) -> float | None:
    """Parse a service value to float, or `None` if it isn't numeric.

    Returns `None` rather than 0.0 for an unparseable value -- the whole point
    of this package's contract. ADMETlab returns some columns as strings and
    occasionally as category labels; a label must become "unknown", not "safe".
    """
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


class ProToxProvider:
    """ProTox-3.0 (`tox.charite.de/protox3`) organ-toxicity / LD50.

    ProTox exposes no documented JSON API — it is a form-driven web app whose
    results page is HTML. Rather than screen-scrape a page whose layout is not
    a contract (a scraper that silently returns "no toxicity found" when the
    markup changes is precisely the fabrication this package forbids), this
    provider performs a REACHABILITY check only and reports no endpoints,
    recording in `last_error` what a human needs to do.

    This is deliberately an honest stub, not a placeholder to be filled in with
    a fragile scraper. If ProTox endpoints are needed, the right move is a
    manual/batch submission whose CSV export is committed as reference data —
    the same pattern `testdata/reference/` already uses for PySCF numbers.
    """

    name = "protox3"
    online = True

    def __init__(
        self,
        base_url: str = "https://tox.charite.de/protox3",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.last_error: str | None = None
        # ALWAYS `unsupported`, reachable or not: this provider can never
        # contribute an endpoint, so its outage cannot make a screen less
        # complete, and counting its by-design "no API" note as a failure made
        # every online run exit non-zero (the 2026-09-23 defect).
        self.last_error_kind: str | None = None
        self.last_error_unreachable = False

    def fetch(self, smiles: str) -> list[ToxEndpoint]:
        if not isinstance(smiles, str):
            raise TypeError(f"smiles must be str, got {type(smiles).__name__}")
        self.last_error_kind = STATUS_UNSUPPORTED
        self.last_error_unreachable = False
        url = f"{self.base_url}/"
        try:
            status = _call_with_deadline(
                lambda: _get_status(url, self.timeout), self.timeout
            )
            reachable = status == 200
        except Exception as e:  # noqa: BLE001 - reachability probe only
            self.last_error = (
                f"ProTox-3.0 unreachable ({e}); it contributes no endpoints "
                "either way (no documented JSON API)"
            )
            self.last_error_unreachable = True
            return []

        self.last_error = (
            f"ProTox-3.0 at {self.base_url} is "
            + ("reachable" if reachable else "not reachable")
            + " but exposes no documented JSON API; this provider does not "
            "screen-scrape its HTML results page by design (a layout change "
            "would silently read as 'non-toxic'). Submit the SMILES set "
            "manually and commit the CSV export as reference data if these "
            "endpoints are needed."
        )
        return []
