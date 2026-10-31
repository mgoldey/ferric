#!/usr/bin/env bash
# Install scripts/cargo-shim and scripts/box-lease into the shim directory, which survives a reboot.
# Then: export PATH="$HOME/.local/share/ferric-cargo-shim:$PATH". The name must keep "ferric-cargo-shim":
# the shim skips PATH entries matching it when looking for the real cargo.
#
# Atomic on purpose. Agents and the gate use the shim live, and bash reads a script as it runs, so
# overwriting it in place would corrupt every running shim. A rename leaves those on the old inode.
# They keep the old behaviour until they exit, which is safe: a lease still holds them out through
# the compile slots. The previous DIFFERENT version stays beside it as NAME.prev for rollback, so
# re-installing the same content does not overwrite a rollback with itself.
set -euo pipefail

src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
home="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"
dest="${FERRIC_SHIM_DIR:-$home/.local/share/ferric-cargo-shim}"
mkdir -p "$dest/under-lease"

install_one() { # SOURCE NAME
    local tmp="$dest/$2.new.$$"
    install -m 755 "$src/$1" "$tmp"
    if [[ -e "$dest/$2" ]] && ! cmp -s "$tmp" "$dest/$2"; then cp -p "$dest/$2" "$dest/$2.prev"; fi
    mv -f "$tmp" "$dest/$2"
    echo "installed $dest/$2"
}

# Dependencies first: the moment `cargo`/`box-lease` land, they must find these.
install_one cargo-lock-lib.sh cargo-lock-lib.sh
install_one under-lease/flock under-lease/flock
install_one box-lease box-lease
install_one cargo-shim cargo
