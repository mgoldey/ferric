# Architecture

A Cargo workspace of focused crates, layered so that method crates depend on
integrals and core, not on each other.

```text
                          +------------------+
                          |   ferric-cli     |   TOML config -> all methods
                          +--------+---------+   (+ ferric-python: pyo3 bindings)
                                   |
   +-----------+-----------+-------+------+-----------+------------+
   |           |           |              |           |            |
+--v----+ +----v----+ +----v----+   +-----v----+ +----v-----+ +---v------+
|ferric | |ferric   | |ferric   |   |ferric    | |ferric    | |ferric    |
|-scf   | |-mp2     | |-dft     |   |-rpa      | |-gw       | |-cc       |
|RHF/UHF| |RI-MP2,  | |RKS/UKS/ |   |PDEP-RPA, | |G0W0,     | |CCD/CCSD/ |
|/ROHF, | |OO,att,  | |ROKS,    |   |U-PDEP,   | |COHSEX,   | |(T)       |
|KS-DFT,| |SCS,     | |libxc,   |   |response  | |evGW,     | +----------+
|DIIS,  | |2terfc,  | |Becke    |   |props,    | |U-GW      |
|MOM,AH,| |Laplace  | |grids,   |   |ESP/Hirsh/| +-----+----+
|cDFT,  | +----+----+ |VV10     |   |NPZ export|       |
|grads  |      |      +----+----+   +-----+----+       |
+---+---+      |           |              |            |
    |          +-----+-----+------+-------+------------+
    |                |     |      |
    |   +------------v--+ +v------v-----+   ferric-tensors (sparse),
    |   |ferric-export | |ferric-      |   ferric-quadrature (Laplace/grid roots)
    |   |cube,NPZ,GTO  | |integrals    |   support crates
    |   +--------------+ |libint2 FFI  |
    |                    |shim/shim.cc |   Coulomb/erf/erfc, 1e/2e/3c/2c, derivs
    +--------+-----------+------+------+
             |                  |
        +----v------------------v----+
        |        ferric-core         |   Molecule, BasisSet, Shell, elements,
        |                            |   BSE-JSON / G94 parsers, bundled bases
        +----------------------------+
```

## Layers

**`ferric-core`** — molecular structure, basis sets, shells, elements, and the
BSE-JSON / Gaussian-94 parsers. Also the home of shared infrastructure:
configuration (`ConfigVar`), memory budgets (`MemoryPlan`), the BLAS-thread
hazard model, and MPI context.

**`ferric-integrals`** — the libint2 FFI and its C++ shim. Coulomb, erf and erfc
operators; 1-electron, 2-electron, 3-center and 2-center integrals; first
derivatives. The shim wraps every libint2 call in `try`/`catch` and returns a
sentinel, so a C++ exception never unwinds across the FFI boundary.

**Method crates** — `ferric-scf`, `ferric-mp2`, `ferric-dft`, `ferric-rpa`,
`ferric-gw`, `ferric-cc`, `ferric-ci`, `ferric-tddft`, `ferric-pcm`,
`ferric-mm`, `ferric-xtb`.

**Support** — `ferric-tensors` (the `einsum!` contraction macro),
`ferric-quadrature` (Lebedev grids, minimax Laplace roots), `ferric-export`
(cube files, NPZ, GTO evaluation).

**Interfaces** — `ferric-cli` (TOML-driven) and `ferric-python` (pyo3).

## Cross-cutting conventions

**Threading.** rayon owns outer parallelism; BLAS is pinned to one thread inside
any rayon worker, enforced at runtime rather than by convention. Throughput
across many jobs comes from many single-threaded processes, not one wide job.

**Determinism.** Reductions fold in a fixed ascending order independent of
thread count, so results are bit-identical across `RAYON_NUM_THREADS`. This is
pinned by tests. A different-but-deterministic order — a tree-fold, say — would
*not* be acceptable, because floating-point addition is not associative.

**Memory.** `MemoryPlan` expresses what a path will allocate and when, so an
oversized job is refused before allocating, with a breakdown naming the dominant
term. Guards are tested in both directions: a starved budget must be refused,
and an ample budget must still run.

**Errors.** Methods return `Result`; iterative solvers additionally carry a
`converged` flag, since non-convergence is a result rather than an error.
Callers are expected to check it.
