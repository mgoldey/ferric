# ferric-cuest

Optional NVIDIA [cuEST](https://docs.nvidia.com/cuda/cuest/) backend for
ferric. **Status: first GPU check.** Nothing in ferric's SCF calls cuEST yet.
This crate builds the one-electron matrices S, T and V with cuEST and compares
them against libint references stored in `testdata/reference/cuest/`. That
settles the basis-normalization and AO-ordering questions before any J/K or
XC work.

## What is verified without a supported GPU

On any box (`cargo test -p ferric-cuest`):

- converting ferric shells into cuEST's pre-normalized coefficients;
- the cuEST basis-compatibility audit;
- the AO-order inference used to read a GPU run (`src/layout.rs`). Its tests
  recover a planted foreign order with sign flips, and flag a planted 5%
  normalization error with the factor.

On a box with cuEST installed (`--features cuest`), even an unsupported GPU:

- `csrc/abi_check.c` compiles against the real `cuest.h`. Every prototype and
  enum value in `src/ffi.rs` is checked, so a mismatch is a build error.
  Mutation-tested: a wrong integer width, enum value and argument count are
  each rejected.
- the binary links and loads `libcuest`. An unsupported device is reported
  cleanly: a GTX 1080 (sm_61) gets `UNSUPPORTED_ARCHITECTURE`, and the tests
  SKIP.

## What only a supported GPU (sm_86+: RTX 30xx/40xx, A100, H100, ...) can verify

`tests/gpu.rs::one_electron_matches_libint` runs on NH3 in a generic
orientation with STO-3G, 6-31G* (Cartesian d), cc-pVDZ (pure d, so p is sent
pure) and cc-pVTZ (pure f). For each basis it reports:

- max |cuEST − libint| for S, T and V under the guessed AO map (bars 1e-10
  for S and T, 1e-9 for V);
- the AO order, phases and any normalization factor inferred from the data,
  plus the errors under the inferred map;
- diag(S) range and timings.

## Running it (GPU tester)

Prerequisites:

- Linux x86_64 with an NVIDIA driver for CUDA 12.x (or 13.x);
- the CUDA toolkit runtime libraries (cudart, cublas, cusolver, cusparse,
  nvrtc, nvJitLink);
- Rust 1.87+;
- `libopenblas-dev` and `liblapack-dev` (ferric-core links them);
- `curl`, `unzip`, and `tar` with zstd.

libint is **not** needed.

```
git clone https://github.com/mgoldey/ferric && cd ferric
git checkout feat/cuest-nvidia-integration
scripts/cuest/run_gpu_check.sh            # or: scripts/cuest/run_gpu_check.sh cuda13
```

Send back `cuest-gpu-check.log` and `cuest-oe-report.json`, pass or fail.
A failure report is just as useful: it records which AO order and
normalization cuEST really uses.

The runner installs cuEST 0.3.0 from conda-forge into `~/.local/cuest-0.3.0`
(`scripts/cuest/fetch_cuest.sh`, sha256-pinned) unless `CUEST_ROOT` is set.
With conda instead:
`conda install -c conda-forge libcuest-dev cuda-version=12 cuda-cudart-dev`,
then `CUEST_ROOT=$CONDA_PREFIX`. With several GPUs, select one with
`CUDA_VISIBLE_DEVICES`.

## Licensing

cuEST is NVIDIA proprietary software. No cuEST header or library text is in
this repository. `ffi.rs` declarations are hand-written, and the ABI check
reads the header from the user's own install at build time.
