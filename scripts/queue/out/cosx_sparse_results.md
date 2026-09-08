# Shell-sparse COSX half transforms — results

Date: 2026-09-08. Branch `feat/cosx-sparse-gemm` (rebased onto main e2cd1898).
Pre-registration: `cosx_sparse_prereg.md`, committed BEFORE the implementation
and before any number here.

## Protocol as run

* Release, ONE thread everywhere (`OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1`;
  the harness additionally installs a 1-thread rayon pool around each timed
  segment). `cpu_s == wall_s` to <= 0.2% on every row kept below.
* `scripts/ferric-limited --max=4G --high=3600M`, `FERRIC_MEM_BUDGET_GB=2`,
  foreground under `timeout 480`, ONE process per timed cell.
* PSI `/proc/pressure/memory` full avg10 printed before AND after every cell;
  every row below is 0.00/0.00 except where noted. Two cells were re-taken for
  nonzero after-PSI (C8-sparse first take 28.917 s with after=0.22 -> retake
  30.069 s at 0.00; C12-dense first take 61.888 s at 0.00/0.47 -> retake
  67.325 s but cpu 65.42 != wall 67.33, so the 61.888 s cell — cpu==wall, PSI
  clean before — is the one tabulated; the two differ by 8%, which bounds this
  measurement's run-to-run noise).
* Harness `crates/ferric-scf/tests/cosx_full_k.rs` (`COSX_FK_HALF=sparse|dense`
  added this session), md3c1e backend, (50,110) grid, overlap fit ON,
  density-driven screen at the default `t = 1e-7`. First build per process, so
  each includes the `S_num = X X^T` accumulation and its Cholesky.
* Densities: C4/C8/C12 from a DF-JK SCF (def2-universal-jkfit, default
  convergence); C16 from a direct SCF (density_conv 1e-5, converged, final
  dp_rms 1.28e-6); C20 from a direct SCF chained over two foreground windows
  (final dp_rms 1.70e-6). Both half transforms in a size row contract the SAME
  density file, so provenance cancels out of every sparse-vs-dense comparison.

## def2-SVP series, sparse vs dense (one thread)

| system | natoms | nbf | npts | total s sparse | total s dense | A-build s sparse | GEMMs s sparse | GEMMs s dense | GEMM % sparse | GEMM % dense | GEMM speedup | \|A\|/nbf (min..max) | \|Λ\|/nbf | \|B\|/nbf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alkane_4  | 14 | 106 | 77000  | 8.039   | 8.257   | 7.098   | 0.156 | 0.172  | 1.9 | 2.1  | 1.1x  | 0.8175 (0.236..1.000) | 1.0000 | 1.0000 |
| alkane_8  | 26 | 202 | 143000 | 30.069  | 30.055  | 26.466  | 0.615 | 1.477  | 2.0 | 4.9  | 2.4x  | 0.5857 (0.317..0.896) | 1.0000 | 0.9994 |
| alkane_12 | 38 | 298 | 209000 | 58.518  | 61.888  | 51.297  | 1.251 | 6.246  | 2.1 | 10.1 | 5.0x  | 0.4358 (0.030..0.701) | 1.0000 | 0.9832 |
| alkane_16 | 50 | 394 | 275000 | 91.815  | 108.313 | 79.447  | 2.169 | 23.004 | 2.4 | 21.2 | 10.6x | 0.3463 (0.162..0.543) | 1.0000 | 0.9521 |
| alkane_20 | 62 | 490 | 341000 | 125.903 | 170.582 | 107.643 | 3.263 | 55.263 | 2.6 | 32.4 | 16.9x | 0.2856 (0.008..0.437) | 1.0000 | 0.8935 |

Sparse GEMM split at C20 (s): `F = D[Λ,A] X[A]` 1.571, `Ktilde[A,B] += X[A] G[B]^T`
1.293, `S_num[A,A]` 0.399, mask+gather 0.176. Dense at C20: 9.288 / 16.271 /
29.704. Other segments are unchanged between paths within noise (C20 ao_eval
8.354 vs 8.352; per-point fold 5.704 vs 5.848; A-build 107.6 vs 100.4, a 7%
spread that is run-to-run variation on identical work — the pair counters are
IDENTICAL at every size, `kept/total` and `geom` to the digit).

### Tail exponents (log-log least squares, LAST THREE sizes, N = atoms)

| quantity | tail exponent | baseline (main, dense) |
|---|---|---|
| **full COSX K, sparse** | **1.57** | 2.16 (this session's dense re-take: 2.07) |
| full COSX K, dense (re-taken here) | 2.07 | 2.16 |
| COSX A-build, sparse | 1.52 | 1.54 |
| **COSX block GEMMs, sparse** | **1.96** | 4.55 (this session's dense re-take: 4.47) |
| COSX block GEMMs, dense (re-taken here) | 4.47 | 4.55 |

Pairwise sparse total (C4->C8->C12->C16->C20): 2.13, 1.75, 1.64, 1.47 — bending,
like the A-build it now tracks. Pairwise sparse GEMM: 2.22, 1.87, 2.01, 1.90.

## Reading against the pre-registration

* **GEMM seconds at C20: predicted 15-30 s, MEASURED 3.26 s** — better than
  pre-registered by 5-9x. The prediction assumed `|A| ~ 0.5 nbf`; the measured
  `|A|/nbf` at C20 is 0.2856, and the cost goes as `|A| x |B|`, so the
  0.29 x 0.89 product beats the assumed 0.5 x 1.0 by 1.9x, with the rest from
  the smaller operands recovering dgemm rate (the pre-registered
  cache-effect mechanism, now confirmed in the direction predicted).
* **GEMM fraction at C20: predicted 10-20%, MEASURED 2.6%** — same reason.
* **Full-K tail exponent: predicted 1.6-1.9, MEASURED 1.57** — just below the
  predicted band, i.e. the change did slightly better than pre-registered.
  The floor argued in the prereg was "the A-build's 1.54 plus the still-dense
  O(N^2) terms"; measured A-build 1.52, full K 1.57, so the remaining gap is
  ~0.05 and the GEMMs are no longer the scaling bottleneck. THE STATED
  OBJECTIVE (full K scaling like its A-build) IS MET.
* **GEMM tail exponent: predicted <= 2.5, MEASURED 1.96.**
* **Butane/def2-QZVP unchanged: predicted, CONFIRMED** — sparse 92.649 s vs
  dense 89.810 s in the same session (+3.2%), with the GEMMs 2.516 vs 3.554 s.
  At 7 Bohr `|A|/nbf` is 0.7065 and `|B|/nbf` 1.0000, so there is little to
  save and the gather is pure overhead; +3% sits inside the A-build's own
  run-to-run spread (81.5 vs 78.1 s on identical work). NOTE the old 137.04 s
  QZVP figure is NOT the comparison used here — it predates intervening main
  changes; both numbers above were taken this session on this branch.
* **Artifact hypotheses: none observed.** Exponent did change (2.07 -> 1.57);
  `|A|/nbf` is well below 1 and FALLS monotonically with N (0.82, 0.59, 0.44,
  0.35, 0.29); it VARIES strongly across blocks at every size (C20 min 0.008,
  max 0.437), which is the pre-registered too-clean check. The gather counter
  stayed negligible (0.176 s of 125.9 s at C20), ruling out the
  "gather/scatter dominates" artifact.
* **`|Λ|/nbf = 1.0000 at every alkane size**, exactly as pre-registered: the
  def2-SVP density matrix has no element below 1e-10 anywhere on a <= 50 Bohr
  chain, so the D-sparsity restriction is entirely pre-onset on this series.
  It DOES bite where the geometry allows it (water dimer at 28 Bohr: 0.5152),
  which is how the anchor keeps it honest. On this series the win therefore
  comes from `A` and `B` alone; `Λ` is machinery that will only pay at larger
  N (or with a screened/attenuated kernel).

## Accuracy and anchors (details in the commit messages / the test file)

* Trivial limit (eps_ao = eps_d = 0): BITWISE equal to dense on water/cc-pVDZ
  and butane/def2-SVP, fit on and off, at full scale and on a 1e-8-scaled
  density, screen off and at 1e-7 — `max|dK| = 0.000e0` in all 10 combinations,
  with counters `|A| = |Λ| = 1.0000` and the pair-screen decisions identical
  (90512912/114345000 both paths).
* Production eps: `max|K_sparse - K_dense|` = 1.38e-18 (water/cc-pVDZ),
  3.46e-10 (butane/def2-SVP), 7.38e-10 (alkane_8/def2-SVP), 1.78e-15 (water
  dimer) — bar 1e-6, grid errors 5.6e-5 / 2.9e-4.
* SCF: water/cc-pVDZ COSX-RHF dense -76.0267671912 (14 iters) vs sparse
  -76.0267671912 (14 iters), dE = -5.7e-14 Ha — bar 1e-7.
* Legacy suites (`cosx_k_anchors`, `cosx_scf`, `cosx_a_anchor`, 15 tests) pass
  on BOTH paths (`COSX_ANCHOR_HALF=dense|sparse`).

## Mutation proofs (each applied by hand, run, reverted)

| mutant | expected | observed |
|---|---|---|
| M1 drop shell 0 from `A` | (a) and (b) RED | RED both: with the fit on, `S_num` is no longer positive definite (hard error); with the fit off, `max\|dK\| = 9.767e0` = the full magnitude of K, and eps=0 is not bitwise |
| M2 `Λ` = all shells (skip the D restriction) | K unchanged, counters RED | exactly that: `max\|dK\|` still 1.776e-15 on the dimer, `\|Λ\|/nbf` 0.5152 -> 1.0000, counters anchor RED |
| M3 shift the `Ktilde` column map by one AO | (a) and (b) RED at O(1) | RED both, `max\|dK\| = 9.304e0` — O(1), distinguishable by size from an eps-too-loose failure as pre-registered |

## Gates

`cargo check -p ferric-scf --all-targets`, `cargo clippy -p ferric-scf -p
ferric-cli --all-targets` (clean), `RUSTDOCFLAGS="-D warnings" cargo doc`
(clean), `python3 scripts/complexity_gate.py` (PASS, 6453 functions, no
regressions), `cargo test -p ferric-cli --lib config` (43 pass).
