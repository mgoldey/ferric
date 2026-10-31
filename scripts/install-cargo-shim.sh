#!/usr/bin/env bash
# Install scripts/cargo-shim and scripts/box-lease into the shim directory (/tmp, so it is gone
# after a reboot: re-run this). Then: export PATH=/tmp/ferric-cargo-shim:$PATH
#
# Atomic on purpose. Agents and the gate use the shim live, and bash reads a script as it runs, so
# overwriting it in place would corrupt every running shim. A rename leaves those on the old inode.
# They keep the old behaviour until they exit, which is safe: a lease still holds them out through
# the compile slots. The previous version stays beside it as NAME.prev for rollback.
set -euo pipefail

src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dest="${FERRIC_SHIM_DIR:-/tmp/ferric-cargo-shim}"
mkdir -p "$dest"

install_one() { # SOURCE NAME
    local tmp="$dest/.$2.new.$$"
    install -m 755 "$src/$1" "$tmp"
    if [[ -e "$dest/$2" ]]; then cp -p "$dest/$2" "$dest/$2.prev"; fi
    mv -f "$tmp" "$dest/$2"
    echo "installed $dest/$2"
}

install_one box-lease box-lease
install_one cargo-shim cargo
