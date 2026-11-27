# ferric-cuest

Optional NVIDIA [cuEST](https://docs.nvidia.com/cuda/cuest/) backend for
ferric. **Status: wired into the SCF, not yet run on a supported GPU.**

With `[gpu] backend = "cuest"` (CLI) or `ferric_scf::cuest_route::solve_{rhf,uhf}_cuest`
(library), an SCF gets S, T, V and exact integral-direct J and K from cuEST on
GPU 0; ferric runs the SCF loop (DIIS, guess, convergence) and, for KS-DFT, the
XC on its own CPU grid. Wired: `task = "energy"` for RHF, UHF, and RKS/UKS with
pure or global-hybrid functionals. Hard errors: range-separated hybrids, ECPs,
ghost atoms, optimize/frequencies, correlated methods, and any `[scf]` J/K key
(`jk`, `jk_aux`, `df_j_aux`, `df_k_aux`, `k_builder`), which cuEST's exact J/K
would replace.

Before every SCF the route calibrates cuEST's AO order: it computes S, T, V
through cuEST and through libint on a symmetry-free probe copy of the molecule,
infers the per-shell-class permutation and phases, and refuses to run unless
they are a clean signed permutation and the molecule's own mapped matrices
match libint to 1e-8. A wrong guess about cuEST's AO conventions can therefore
not produce a wrong energy; it is either corrected or refused. Calibration does
not cover J and K, which the SCF energy test below does.

cuEST documents its direct J/K as functional but slow on consumer GPUs with
reduced fp64 throughput (RTX cards). Small molecules are the target here:
correctness first, speed later (density-fitted J/K is the next step).

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

CPU anchor for the SCF route (`cargo test -p ferric-scf --test cuest_route`,
no GPU, no cuEST): a stand-in backend serves libint S, T, V and ferric's exact
J/K in a scrambled AO order (every l>=1 shell reversed, alternating signs). The
route must calibrate it away and reproduce ferric's own SCF. Measured
2026-11-26: |ΔE| <= 9e-14 Ha for water RHF (6-31G*, cc-pVDZ), PBE, PBE0, and
H2O+ UHF and UKS/PBE0. Mutations caught: J/K in a different order than S/T/V
(energy off by > 1e-3 Ha), and a 5% d-shell normalization error (refused with
the factor named).

## What only a supported GPU (sm_80+: A100, RTX 30xx/40xx, H100, ...) can verify

`tests/gpu.rs::one_electron_matches_libint` runs on NH3 in a generic
orientation with STO-3G, 6-31G* (Cartesian d), cc-pVDZ (pure d, so p is sent
pure) and cc-pVTZ (pure f). For each basis it reports:

- max |cuEST − libint| for S, T and V under the guessed AO map (bars 1e-10
  for S and T, 1e-9 for V). The guess is cuEST 0.3's documented order:
  Cartesian components lexical as in libint; pure components m = 0, +1, −1,
  +2, −2, … (pure p is z, x, y) against libint's m = −l…+l, all signs +1
  (both use Racah-normalized real solid harmonics without the Condon–Shortley
  phase);
- the AO order, phases and any normalization factor inferred from the data,
  plus the errors under the inferred map;
- diag(S) range and timings.

`ferric-scf/tests/cuest_gpu.rs::scf_on_cuest_matches_ferric` runs full SCFs
on cuEST and compares each against ferric's CPU SCF with exact J/K (bar 1e-8
Ha): water RHF in STO-3G, 6-31G*, cc-pVDZ and cc-pVTZ; RKS PBE and PBE0;
H2O+ UHF and UKS/PBE0. It writes `cuest-scf-report.json` with both energies,
iteration counts, the AO calibration and cuEST timings.

The runner then builds the CLI with cuEST and runs
`examples/water-rhf-cuest.toml` end to end.

## Running it (GPU tester)

Prerequisites:

- Linux x86_64 with an NVIDIA driver for CUDA 12.x (or 13.x);
- the CUDA toolkit runtime libraries (cudart, cublas, cusolver, cusparse,
  nvrtc, nvJitLink);
- Rust 1.87+;
- `build-essential`, `libeigen3-dev`, `libopenblas-dev`, `liblapack-dev`;
- `curl`, `unzip`, `python3`, `zstd`, `patchelf`, and `tar` with zstd.

On Debian/Ubuntu: `sudo apt install build-essential libeigen3-dev
libopenblas-dev liblapack-dev curl unzip python3 zstd patchelf`.

The runner installs libint2 (conda-forge prebuilt 2.13.1, a few seconds, via
`scripts/install-libint.sh`) into `~/.local/libint2-2.13.1` unless
`LIBINT2_PREFIX` is set. The SCF comparison needs it.

```
git clone https://github.com/mgoldey/ferric && cd ferric
git checkout feat/cuest-nvidia-integration
scripts/cuest/run_gpu_check.sh            # or: scripts/cuest/run_gpu_check.sh cuda13
```

Send back `cuest-gpu-check.log`, `cuest-oe-report.json` and
`cuest-scf-report.json`, pass or fail. A failure report is just as useful: it
records which AO order and normalization cuEST really uses.

To run your own input afterwards:

```
export LD_LIBRARY_PATH=$HOME/.local/cuest-0.3.0/lib:$LD_LIBRARY_PATH
target/release/ferric examples/water-rhf-cuest.toml
```

Any `rhf`/`uhf`/`ksdft` energy input works with `[gpu] backend = "cuest"` added.
Binaries outside this crate need `LD_LIBRARY_PATH` to find `libcuest.so`
(cargo does not pass this crate's rpath on to them).

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
