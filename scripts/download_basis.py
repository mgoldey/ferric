#!/usr/bin/env python3
"""Download a basis set from the Basis Set Exchange and save to bundled sets."""

import json
import sys
import tempfile
from pathlib import Path

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
    import requests


def download_basis(basis_name: str) -> None:
    """Download basis set from BSE and save to bundled directory."""
    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    bundled_dir = script_dir.parent / "crates" / "ferric-core" / "src" / "basis" / "bundled"
    output_file = bundled_dir / f"{basis_name}.json"

    # Fetch from BSE
    url = f"https://www.basissetexchange.org/api/basis/{basis_name}/format/json"
    print(f"Downloading {basis_name} from {url}...", file=sys.stderr)

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {basis_name}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        print(f"Error parsing response as JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Write atomically via temp file
    bundled_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=bundled_dir,
        delete=False,
        suffix=".json",
    ) as tmp:
        json.dump(data, tmp, indent=2)
        tmp_path = Path(tmp.name)

    tmp_path.replace(output_file)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: download_basis.py <basis_name>", file=sys.stderr)
        print("Example: download_basis.py aug-cc-pvdz", file=sys.stderr)
        sys.exit(1)

    download_basis(sys.argv[1])
