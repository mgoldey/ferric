#!/usr/bin/env bash
# One-shot cuEST check for a GPU tester (RTX 30xx/40xx, A100, H100, ...).
#
#   scripts/cuest/run_gpu_check.sh [cuda12|cuda13]
#
# 1. installs cuEST 0.3.0 into ~/.local/cuest-0.3.0 if CUEST_ROOT is unset
#    and nothing is there yet;
# 2. builds ferric-cuest with the `cuest` feature (compiles the ABI check
#    against the real cuest.h);
# 3. runs the CPU tests, then the GPU tests with FERRIC_CUEST_REQUIRE_DEVICE=1
#    (so an unsupported or missing GPU FAILS instead of skipping):
#      - one-electron S, T, V vs libint (ferric-cuest tests/gpu.rs);
#      - full SCF on cuEST vs ferric's CPU SCF: RHF, PBE, PBE0, UHF, UKS on
#        water (ferric-scf tests/cuest_gpu.rs);
# 4. builds the CLI with cuEST and runs examples/water-rhf-cuest.toml;
# 5. leaves cuest-gpu-check.log, cuest-oe-report.json and
#    cuest-scf-report.json in the repo root.
#
# Send all three files back, pass or fail.
set -uo pipefail
cd "$(dirname "$0")/../.."
flavor="${1:-cuda12}"
log=cuest-gpu-check.log
: > "$log"
say() { echo "== $*" | tee -a "$log"; }

say "host"
{ uname -srm; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv 2>&1;
  rustc --version; cargo --version; git rev-parse HEAD; } 2>&1 | tee -a "$log"

if [[ -z "${CUEST_ROOT:-}" ]]; then
  export CUEST_ROOT="$HOME/.local/cuest-0.3.0"
  if [[ ! -f "$CUEST_ROOT/include/cuest.h" ]]; then
    say "installing cuEST ($flavor) into $CUEST_ROOT"
    scripts/cuest/fetch_cuest.sh "$flavor" "$CUEST_ROOT" 2>&1 | tee -a "$log"
  fi
fi
say "CUEST_ROOT=$CUEST_ROOT"
# The SCF steps compile ferric-integrals, which links libint2. The installer
# puts conda-forge's prebuilt 2.13.1 at build.rs's default prefix (seconds).
if [[ -z "${LIBINT2_PREFIX:-}" && ! -f "$HOME/.local/libint2-2.13.1/include/libint2.hpp" ]]; then
  say "installing libint2 2.13.1 into ~/.local/libint2-2.13.1"
  scripts/install-libint.sh 2>&1 | tee -a "$log"
fi
ldd "$CUEST_ROOT/lib/libcuest.so" 2>&1 | grep -E 'not found|cuda|cublas|cusolver|cusparse|nvrtc|nvJitLink' | tee -a "$log"

export OPENBLAS_NUM_THREADS=1
# Tests write their reports under CARGO_TARGET_TMPDIR = <target dir>/tmp.
tgt="${CARGO_TARGET_DIR:-target}"
# Binaries outside ferric-cuest (ferric-scf tests, the CLI) find libcuest here.
export LD_LIBRARY_PATH="$CUEST_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
say "build + CPU tests"
cargo test -p ferric-cuest --features cuest --release 2>&1 | grep -vE '^\s+Compiling' | tee -a "$log"
cpu=${PIPESTATUS[0]}

say "GPU tests (device required)"
rm -f "$tgt/tmp/cuest-oe-report.json" cuest-oe-report.json
FERRIC_CUEST_REQUIRE_DEVICE=1 cargo test -p ferric-cuest --features cuest --release --test gpu \
  -- --nocapture --test-threads=1 2>&1 | grep -vE '^\s+Compiling' | tee -a "$log"
gpu=${PIPESTATUS[0]}

say "SCF route: CPU anchor (no GPU needed)"
cargo test -p ferric-scf --features cuest --release --test cuest_route \
  -- --nocapture --test-threads=1 2>&1 | grep -vE '^\s+Compiling' | tee -a "$log"
scf_cpu=${PIPESTATUS[0]}

say "SCF on cuEST vs ferric CPU (device required)"
rm -f "$tgt/tmp/cuest-scf-report.json" cuest-scf-report.json
FERRIC_CUEST_REQUIRE_DEVICE=1 cargo test -p ferric-scf --features cuest --release --test cuest_gpu \
  -- --nocapture --test-threads=1 2>&1 | grep -vE '^\s+Compiling' | tee -a "$log"
scf_gpu=${PIPESTATUS[0]}

say "CLI: examples/water-rhf-cuest.toml"
cargo build -p ferric-cli --features cuest --release 2>&1 | grep -vE '^\s+Compiling' | tail -3 | tee -a "$log"
"$tgt/release/ferric" examples/water-rhf-cuest.toml 2>&1 | tail -25 | tee -a "$log"
cli=${PIPESTATUS[0]}

missing=0
for r in cuest-oe-report.json cuest-scf-report.json; do
  if [[ -f "$tgt/tmp/$r" ]]; then
    cp "$tgt/tmp/$r" "$r"
  else
    say "MISSING report $tgt/tmp/$r (the test did not get far enough to write it)"
    missing=1
  fi
done
say "exit codes: oe_cpu=$cpu oe_gpu=$gpu scf_cpu=$scf_cpu scf_gpu=$scf_gpu cli=$cli"
say "send back: $log cuest-oe-report.json cuest-scf-report.json"
exit $(( cpu || gpu || scf_cpu || scf_gpu || cli || missing ))
