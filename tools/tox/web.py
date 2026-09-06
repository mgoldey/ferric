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

**Status as measured 2026-08-29:** ADMETlab 3.0's documented `POST /api/admet`
(base `https://admetlab3.scbdd.com`, payload `{"SMILES": [...]}`, as used by the
published API tutorial and community clients) returns **HTTP 404** on every path
variant tried (`/api/admet`, `/api/admet/`, `/api/v1/admet`, `/admet/api`), while
the site root returns 200. The endpoint has evidently moved or been withdrawn.
The client is kept, exercised against a mock in the tests, and degrades cleanly;
if the service returns, `AdmetlabProvider` starts contributing with no code
change. Do not delete it on the assumption it never worked -- and do not
"fix" it by inventing a new path without re-probing first.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from .model import ToxEndpoint

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
    "F(20%)": ("bioavailability_20pct", False),
    "F(30%)": ("bioavailability_30pct", False),
}


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "User-Agent": _UA},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310 -- base_url defaults are fixed https:// endpoints; callers are local scripts
        return json.loads(r.read().decode())


class AdmetlabProvider:
    """ADMETlab 3.0 REST client (`POST {base}/api/admet`).

    `endpoint_path` is injectable purely so the tests can point it at a local
    mock, and so a future path change is a one-argument fix at the call site
    rather than an edit here.
    """

    name = "admetlab3"

    def __init__(
        self,
        base_url: str = "https://admetlab3.scbdd.com",
        endpoint_path: str = "/api/admet",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint_path = endpoint_path
        self.timeout = timeout
        self.last_error: str | None = None

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.endpoint_path}"

    def fetch(self, smiles: str) -> list[ToxEndpoint]:
        if not isinstance(smiles, str):
            raise TypeError(f"smiles must be str, got {type(smiles).__name__}")
        self.last_error = None
        try:
            data = _post_json(self.url, {"SMILES": [smiles]}, self.timeout)
        except urllib.error.HTTPError as e:
            self.last_error = f"HTTP {e.code} from {self.url}"
            return []
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            self.last_error = f"network error contacting {self.url}: {e}"
            return []
        except json.JSONDecodeError as e:
            self.last_error = f"unparseable JSON from {self.url}: {e}"
            return []

        # Documented shape: {"data": {"data": [ {col: val, ...} ]}}. Tolerate a
        # bare list too, rather than dropping a good response over a wrapper
        # change -- but never guess at values.
        rows = None
        if isinstance(data, dict):
            inner = data.get("data")
            if isinstance(inner, dict):
                rows = inner.get("data")
            elif isinstance(inner, list):
                rows = inner
        elif isinstance(data, list):
            rows = data
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            self.last_error = (
                f"unexpected response shape from {self.url}: "
                f"{str(data)[:200]}"
            )
            return []

        row = rows[0]
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
            self.last_error = (
                f"response from {self.url} contained none of the expected tox "
                f"columns {sorted(_ADMETLAB_TOX_KEYS)}; got {sorted(row)[:12]}"
            )
        return out


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

    def __init__(
        self,
        base_url: str = "https://tox.charite.de/protox3",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.last_error: str | None = None

    def fetch(self, smiles: str) -> list[ToxEndpoint]:
        if not isinstance(smiles, str):
            raise TypeError(f"smiles must be str, got {type(smiles).__name__}")
        try:
            req = urllib.request.Request(
                f"{self.base_url}/", headers={"User-Agent": _UA}, method="GET"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # nosec B310 -- same fixed https:// base_url
                reachable = r.status == 200
        except Exception as e:  # noqa: BLE001 - reachability probe only
            self.last_error = f"ProTox-3.0 unreachable: {e}"
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
