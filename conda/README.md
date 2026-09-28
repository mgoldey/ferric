# conda package

`recipe.yaml` builds a conda package of ferric (the Python module and the
`ferric` command) with [rattler-build](https://rattler.build).

## Build locally

```bash
rattler-build build --recipe conda/recipe.yaml -c conda-forge
```

`variants.yaml` builds one package per CPython 3.10–3.13. For a single one,
add `--variant python=3.12`. The package lands in `output/`; the recipe's
tests (import, water/STO-3G analytic frequencies, `ferric --help`) run after
the build.

## What it links

All native libraries come from conda-forge and are shared, not bundled:

| Library | conda-forge package |
|---|---|
| libint2 | `libint 2.13.1` (second-derivative integrals) |
| libxc | `libxc-c` (CPU build) |
| OpenBLAS (BLAS and LAPACK) | `libopenblas` |
| Eigen, Boost headers | `eigen`, `libboost-headers` (build time only) |

With conda-forge's libint, analytic Hessians cover the same angular-momentum
range as a source build against `scripts/install-libint.sh` (see
`site/src/using/installation.md`, "What the libint2 build carries").

The PyPI wheel instead carries a smaller libint2 generated for the wheel, so
its integral limits (maximum angular momentum, and which derivative orders
are available) are narrower; outside them, frequencies fall back to finite
differences of the analytic gradient.

## Publishing

`.github/workflows/conda.yml` builds and tests the package on pull requests
and uploads it as a workflow artifact. Publishing to a channel is a separate
step that this workflow does not perform.
