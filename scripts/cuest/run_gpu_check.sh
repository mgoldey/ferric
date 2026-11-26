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
#    (so an unsupported or missing GPU FAILS instead of skipping);
# 4. leaves cuest-gpu-check.log and cuest-oe-report.json in the repo root.
#
# Send both files back, pass or fail.
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
ldd "$CUEST_ROOT/lib/libcuest.so" 2>&1 | grep -E 'not found|cuda|cublas|cusolver|cusparse|nvrtc|nvJitLink' | tee -a "$log"

export OPENBLAS_NUM_THREADS=1
say "build + CPU tests"
cargo test -p ferric-cuest --features cuest --release 2>&1 | grep -vE '^\s+Compiling' | tee -a "$log"
cpu=${PIPESTATUS[0]}

say "GPU tests (device required)"
rm -f target/tmp/cuest-oe-report.json cuest-oe-report.json
FERRIC_CUEST_REQUIRE_DEVICE=1 cargo test -p ferric-cuest --features cuest --release --test gpu \
  -- --nocapture --test-threads=1 2>&1 | grep -vE '^\s+Compiling' | tee -a "$log"
gpu=${PIPESTATUS[0]}

report=none
if [[ -f target/tmp/cuest-oe-report.json ]]; then
  cp target/tmp/cuest-oe-report.json cuest-oe-report.json
  report=cuest-oe-report.json
fi
say "cpu_tests_exit=$cpu gpu_tests_exit=$gpu report=$report"
say "send back: $log and cuest-oe-report.json"
exit $(( cpu || gpu ))
