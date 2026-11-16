# Coupled cluster

"RI-CC" here means the two-electron integrals in the amplitude equations come
from density fitting (three-centre B tensors from an RI auxiliary basis), not
from exact four-centre integrals. Everything below needs an orbital basis and
an RI auxiliary basis. Formal cost is O(N⁶) for CCSD and O(N⁷) for (T).

## CCSD and CCSD(T)

**What it is.** RI-CCSD and the perturbative triples correction (T), in two
implementations:

- **Spin-adapted, closed shell** (`ccsd_closed_shell`, `ccsd_t_closed_shell`):
  amplitudes over spatial orbitals, the algorithm PySCF's `cc.CCSD` uses. This
  is what both the CLI and Python run for an RHF reference. Its VVVV block is
  16× smaller than the spin-orbital one; measured about 8–10× faster than the
  spin-orbital CCSD at cc-pVDZ, and the (T) step 8.7–41× faster (H2O/aug-cc-pVDZ
  and ethane/cc-pVDZ, 12 threads). The (T) loop runs over virtual triples
  a ≤ b ≤ c with [n<sub>o</sub>,n<sub>o</sub>,n<sub>o</sub>] blocks, as PySCF's
  `ccsd_t.c` does. Measured 2026-11-14, 12 threads, 3 interleaved runs each:
  the (T) loop takes 6.2–6.8 s on butane/cc-pVDZ and 86–93 s on hexane/cc-pVDZ.
  PySCF 2.13 takes 9.4 s and 110 s for the same (T) step on the same machine.
- **Spin-orbital** (`ccsd`, `ccsd_t`): built from the same RHF spatial
  orbitals and kept as the cross-check of the spin-adapted solvers, so it is
  closed shell only as well. Its (T) streams one occupied triple at a time, so
  memory is O(n<sub>o</sub>·n<sub>v</sub>³)-class rather than the dense
  six-index tensor.

No open-shell CCD, CCSD or CCSD(T) exists in ferric: every CC solver reads
restricted orbitals and refuses an unrestricted reference. The only open-shell
coupled-cluster-type method is LinLCCD(hh): the CLI `linlccd` kind is
closed shell, and its open-shell (UHF) version is library-only
(`ferric_cc::linlccd_u::u_linlccd`).

**Run it.**

- CCSD: `method.kind = "ccsd"` (`examples/water-ccsd.toml`, water/cc-pVDZ); Python
  `ferric.run_ccsd(mol, bs, aux)`.
- CCSD(T): `method.kind = "ccsd(t)"` (`examples/water-ccsd-t.toml`); Python
  `ferric.run_ccsd_t(mol, bs, aux)`.
- CCD: `method.kind = "ccd"` (`examples/water-ccd.toml`); Python `ferric.run_ccd(mol, bs, aux)`.
- All three kinds are closed shell only and refuse `multiplicity > 1`.

**Aux basis.** The RI error is not negligible at CC accuracy: on water /
cc-pVDZ with `cc-pvdz-ri`, RI-CCSD differs from exact-integral CCSD by about
1.3e-4 Ha. Choose the aux basis for the accuracy you need.

**Accuracy.** `ccsd` is Proven.

| Quantity | System / basis | Reference | Agreement | Pinned by |
|---|---|---|---|---|
| CCSD correlation energy | H2 / STO-3G | exact-integral numpy | −0.02052453 Ha | `ferric-cc/src/ccsd.rs::test_ccsd_h2_sto3g` |
| Closed-shell (T) | H2O / cc-pVDZ | PySCF `ccsd_t()` | ~1e-6 Ha (test asserts 1e-4) | `ferric-cc/src/ccsd_t_closed_shell.rs::closed_shell_t_h2o_ccpvdz_matches_pyscf` |
| Streaming vs dense spin-orbital (T) | H2O / cc-pVDZ | ferric's former dense path | 5e-16 Ha | `ferric-cc/src/ccsd_t.rs::streaming_matches_dense_h2o_ccpvdz` |

**Limits.** Closed-shell entry points only in the CLI and Python. The
spin-adapted (T) rejects spin-orbital amplitudes with a typed error rather
than mixing conventions.

## Particle-particle ladder: dense or RI

The particle–particle ladder Σ<sub>cd</sub> W<sub>abcd</sub> τ<sub>ij</sub><sup>cd</sup>
of closed-shell CCSD can be evaluated two ways, chosen by `[cc] ladder`
(Python `run_ccsd(ladder=)` / `run_ccsd_t(ladder=)`):

- **`dense`** holds the (ac|bd) block in core (one n<sub>v</sub>⁴ buffer, two
  during setup) and contracts the t1 terms of W<sub>abcd</sub> with τ
  separately, so W<sub>abcd</sub> is never formed.
- **`ri`** holds neither the (ab|cd) block nor W<sub>abcd</sub>. It rebuilds
  the (ac|bd) entries from the RI 3-index tensors every iteration in
  a<sub>blk</sub>·n<sub>v</sub>³ panels: at most one n<sub>v</sub>⁴-sized
  panel at full width (dense holds two n<sub>v</sub>⁴ buffers at its peak),
  and narrower panels when the budget requires. It is the same algebra over
  the same tensors, so it is exact: on water/cc-pVDZ |ΔE| and
  max|Δt| are ≲ 3e-17 (round-off), and the alkane energies below agree to
  all 10 printed digits.
- **`auto`** (the default) runs `dense` whenever its memory plan fits the
  budget and `ri` otherwise, so it is never refused where `ri` would run.
  When it falls back to `ri` it says so on stderr. The route taken is
  printed (`ladder = dense (auto)`), logged, and returned
  (`CcResult::ladder_route`; Python `CcResult.ladder_route`).

**Reproducibility.** Under `auto`, a closed-shell CCSD job whose dense plan
exceeds the memory budget runs on the RI ladder instead of being refused.
Because `auto` decides against the budget, a run without an explicit
`[memory] budget_gb` (auto-detected as 0.8 × the RAM available at start-up)
can take different routes on the same input from one run to the next. The
energies agree to round-off either way; the time does not. Set `ladder`
explicitly, or fix the budget, for reproducible timings.

Measured 2026-10-26, alkane/cc-pVDZ (cc-pvdz-ri), `OPENBLAS_NUM_THREADS=1`,
`RAYON_NUM_THREADS=4`, routes interleaved one per process, load average 3–6
from other jobs (`benchmarks/harness/examples/ri_ccsd_memory.rs`). RI ran at
full block width (a<sub>blk</sub> = n<sub>v</sub>, the default-budget
choice). Per-iteration time is (time for k iterations − setup time) / k,
repeated 4× per route for water, 3× for C3H8 and 2× for C4H10/C5H12;
ratios are of the medians. Peak RSS is one converged run per route.

| System | n<sub>v</sub> | Per iteration, dense | Per iteration, RI | RI / dense | Peak RSS, dense | Peak RSS, RI |
|---|---|---|---|---|---|---|
| water | 19 | 7.8 ms | 8.3 ms | 1.06 | — | — |
| C3H8 | 69 | 1.27 s | 1.64 s | 1.30 | 1.50 GB | 0.70 GB |
| C4H10 | 89 | 3.2–3.7 s | 5.2–5.3 s | 1.50 | 3.79 GB | 1.67 GB |
| C5H12 | 109 | 8.8–10.8 s | 14.5–14.6 s | 1.48 | 7.90 GB | 3.40 GB |

RI is 1.3–1.5× slower per iteration from n<sub>v</sub> ≈ 70 up and uses
2.1–2.3× less memory, which is why `auto` takes it only when `dense` does
not fit. The RI ladder now runs one rayon job per virtual index a, and its
result is bit-identical at any thread count. It ran single-threaded when the
table above was measured. On hexane/cc-pVDZ (n<sub>v</sub> = 129, 12 threads,
3 interleaved runs per build, 2026-11-14), the RI pp-ladder step fell from
258–277 s to 57–58 s over 11 iterations, and CCSD from 381–401 s to 175–179 s.
The dense route takes about 140–146 s on the same input. The parallel width
equals the block width a<sub>blk</sub>, so a budget that forces narrow blocks
also narrows the parallelism. The RI route still holds the n<sub>o</sub>·n<sub>v</sub>³
`(ia|bc)` block in core, so it lowers the n<sub>v</sub>⁴ ceiling but does
not remove every large tensor.

## EOM-CCSD excitation energies

**What it is.** Equation-of-motion CCSD for excited states (EE-EOM-CCSD):
the lowest eigenvalues of the similarity-transformed Hamiltonian
H̄ = e<sup>−T</sup> H e<sup>T</sup> − E<sub>CC</sub> in the space of singly
and doubly excited singlets, on top of the closed-shell RI-CCSD ground state.
H̄ is not symmetric, so the roots come from a non-symmetric Davidson solver
(`ferric_cc::eom_davidson`) that keeps every Ritz pair real, applies an Olsen
correction, and starts from unit vectors with a small dense part, so roots in
any point-group irrep are reachable. Only right eigenvectors are computed.

**Run it.** `method.kind = "eom-ccsd"` with an optional [`[eom]`](../reference/input.md#eom)
section (`examples/water-eom-ccsd.toml`); Python
`ferric.run_eom_ccsd(mol, bs, aux, n_roots=5)`, which returns the excitation
energies, each root's residual and singles weight, and `r1(k)` / `r2(k)`.
Closed shell only; refuses `multiplicity > 1`.

**Accuracy.** `eom-ccsd` is Proven (narrow): singlets, closed shell, all
electrons.

| Quantity | System / basis | Reference | Agreement | Pinned by |
|---|---|---|---|---|
| Lowest five singlet ω | H2O, C2H4 / cc-pVDZ (cc-pVDZ-RI) | PySCF `EOMEESinglet` on the same DF integrals | ≤ 2.9e-10 Ha | `ferric-cc/tests/validation_eom_ccsd.rs` |
| Singles block at t = 0 | H2O / 6-31G | CIS (`ferric_gw::bse::run_cis_tda`) | 3.6e-14 Ha | `ferric-cc/tests/eom_ccsd_anchors.rs` |
| Davidson roots | H2 / 6-31G; H2O / 6-31G through subspace collapses | dense H̄ diagonalized by LAPACK | ≤ 1.0e-12 Ha | `ferric-cc/tests/eom_ccsd_anchors.rs` |
| EOM product | H2, H2O / 6-31G, exact integrals | exact CC Jacobian ∂Ω/∂t of PySCF's RCCSD residual (5-point stencil, exact for the quartic residual) | 8.5e-14 relative to ‖σ‖ ≈ 150 | `scripts/prototypes/eom_ccsd_proto.py` |

At water / cc-pVDZ (`examples/water-eom-ccsd.toml`) the lowest singlet is
0.29995 Ha (8.162 eV). The density-fitting error of the ground state carries
into ω: choose the aux basis for the accuracy you need.

**Limits.** Singlets only (no triplets, no spin-flip, no open-shell
reference); no left eigenvectors, so no oscillator strengths or transition
properties; no point-group symmetry, so roots of every irrep are mixed in
the requested count. A target root still complex (a near-degenerate pair)
when `max_iter` runs out is an error, as is an unconverged one.

**Memory and cost.** The EOM product needs the same integral blocks as one
CCSD iteration, plus the Davidson subspace (2 × `max_space` vectors of
length n<sub>o</sub>n<sub>v</sub> + n<sub>o</sub>n<sub>v</sub>(n<sub>o</sub>n<sub>v</sub>+1)/2).
The n<sub>v</sub>⁴ `(ae|bf)` block is held when the memory plan fits the
budget; otherwise it is rebuilt from the RI tensors in panels on every product,
the same choice as the CCSD [ladder](#particle-particle-ladder-dense-or-ri).
A job whose plan cannot fit is refused before the EOM step. Each product
costs about one CCSD iteration.

## LinLCCD and ωB97X-L-V

**LinLCCD(hh)** is linearized coupled-cluster doubles with the hole–hole ladder
kept to all orders, closed shell only. The ladder keeps the correlation energy
finite as the HOMO–LUMO gap closes, where MP2 diverges.
`method.kind = "linlccd"` (`examples/water-linlccd.toml`). Proven (narrow):
no other quantum chemistry code implements LinLCCD(hh), so its energy is
checked against an independent numpy solve of the same equations on PySCF
density-fitted integrals (H2O and NH3 through the closed-shell path, and UHF
OH through the library-only open-shell path; agreement ≤1.2e-12 Ha). With
the ladder off it reduces exactly to RI-MP2, and with exact integrals its
driver terms reproduce canonical MP2.

`[mp2] linlccd_variant` (Python `run_linlccd(variant=...)`) selects the
method's ladder terms: `hh` (default), `drivers-only` (no ladder, equal to
RI-MP2) or `full` (hole–hole plus particle–particle, with CCD-like VVVV
memory). Every variant is computed exactly by default. All three variants
match an independent numpy solve on the same density-fitted integrals
(H2O and NH3, 6-31G and cc-pVDZ, ≤ 4.4e-13 Ha); `drivers-only` also matches
PySCF `DFMP2`.

**Local LinLCCD.** With `[local] scheme = "amplitude-threshold"` and `eps`
(`examples/water-linlccd-local.toml`; Python
`run_linlccd(..., local="amplitude-threshold", eps=1e-4)`) the same variant
is solved in the Boys-localized basis with pair amplitudes at or below `eps`
dropped (see [Exact and local correlation](./index.md#exact-and-local-correlation)).
`eps` has no default and is printed and logged with the kept fraction;
`eps = 0` reproduces the exact LinLCCD of the same variant and the numpy
reference (≤ 4.4e-13 Ha, every variant), which is how the local path is
anchored. `[local] reference = true` also runs the exact
LinLCCD and prints the local error. Closed shell and energy only.

**ωB97X-L-V** is a double-hybrid functional that uses short-range LinLCCD(hh)
instead of MP2 for its correlation term. It converges its own ωB97X-L
Kohn–Sham reference (a non-converged reference is an error), then adds the
LinLCCD(hh) correction on those orbitals. `method.kind = "wb97x-l-v"`
(`examples/water-wb97xlv.toml`). `[dft] lambda` and `omega` override the
published 0.6 and 0.1 Bohr⁻¹; omitting them gives the published values.
λ enters the amplitude equations as well as the energy (the paper's eqn 22), so
the correlation term is quadratic in λ at leading order. Validated against PySCF
with the published parameters plus a numpy LinLCCD(hh) on water and OH, and
against the paper's Be₂ bond energy (2.3 kcal/mol; ferric 2.299).

## MP2-based double hybrids

**B2PLYP** and **DSD-PBEP86**: a KS reference with weighted exchange and
correlation components, plus scaled (SCS-)RI-MP2 correlation.
`method.kind = "b2plyp"` / `"dsd-pbep86"` (`examples/water-b2plyp.toml`);
Python `ferric.run_double_hybrid(mol, bs, aux, kind="b2plyp")`.

**B2PLYP is Proven** (Grimme 2006 parameters: 0.53 HF, 0.47 B88, 0.73 LYP,
0.27 PT2): the total energy matches PySCF built from the definition to
1.3e-11 Ha on H2O, NH3 and CH4 (cc-pVDZ, def2-SVP), at the flat (75, 110) grid
and f64 RI-J storage. The CLI's defaults (pruned grid, f32 RI-J) shift the total
by up to 1.0e-6 Ha.

**DSD-PBEP86 is Proven** as the no-dispersion set of Kozuch and Martin 2013
(0.72 HF, 0.28 PBE exchange, 0.44 P86 correlation, c_os = 0.51, c_ss = 0.36; the
"DSD-PBEP86" of Psi4): the total energy matches PySCF built from that
definition at the same settings as B2PLYP (`examples/water-dsd-pbep86.toml`).
The 2011 DSD-PBEP86-D2 set is not available. Earlier ferric releases used 0.69 HF, 0.31 PBE, c_os = 0.56,
c_ss = 0.29, a set that is none of the published ones; energies shift by
mHa against those.

**D3(BJ) sets.** `[dft] dispersion = "d3bj"` (Python `dispersion="d3bj"`) on
`b2plyp` or `dsd-pbep86` selects the kind's own published D3(BJ) set and adds its
D3(BJ) energy. B2PLYP-D3(BJ) keeps B2PLYP's coefficients and takes the B2PLYP row
of Goerigk and Grimme 2011. DSD-PBEP86-D3(BJ) is the refit set of Kozuch and
Martin 2013 as a whole (0.69 HF, 0.31 PBE, 0.44 P86, c_os = 0.52, c_ss = 0.22;
D3(BJ) s6 = 0.48, a1 = 0, s8 = 0, a2 = 5.6; checked against Psi4's
`dh_functionals.py`), not the no-dispersion set plus a D3 term. A named
functional (`d3bj(pbe)`) and `mbd` are errors for a double hybrid.
(`examples/nh2-dsd-pbep86-d3bj.toml`.)

**Open shell.** With `multiplicity > 1` (CLI) or `mol.multiplicity > 1` /
`unrestricted=True` (Python) both kinds converge a UKS reference with the
double hybrid's functional and add `c_os E_αβ + c_ss (E_αα + E_ββ)` of the
unrestricted RI-MP2 on its orbitals (`examples/nh2-b2plyp.toml`). Energy only:
there is no nuclear gradient, so `optimize` and `frequencies` are errors, and
`wb97x-l-v` stays closed shell. The singlet UKS route equals the RKS route to
1e-13 Ha (water, cc-pVDZ, both PT2 frozen-core settings). Against PySCF
(`dft.UKS` + `mp.dfump2.DFUMP2` with the spin components scaled, same grid, fits
and stable state) the NH2 and CH3 doublets at cc-pVDZ and def2-SVP, three SCF
sets, frozen core 0 and 1, agree to at most 3.1e-13 (E_KS), 7.6e-11 (E_OS),
1.8e-11 (E_SS) and 3.2e-11 Ha (total); the D3(BJ) energy equals simple-dftd3's to
3.6e-16 Ha. The open-shell runs have no CLI default-grid measurement, so a default
run carries no anchor.

## Implementation

The equations each solver implements, with file and line, are in
[`crates/ferric-cc/theory.md`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-cc/theory.md).

All contractions go through `einsum!`, a macro that maps tensor
contractions onto BLAS3 GEMMs. The permutation copies that feed those GEMMs are
parallelized, because for a strided permutation the copy can dominate the
contraction it feeds: measured at 47% at n<sub>v</sub> = 40 and 70% at
n<sub>v</sub> = 80. The copies are bit-identical regardless of thread count,
since a permutation writes each output element once; a test pins this.

## Memory

The amplitude tensors dominate and grow as
\\( n_o^2 n_v^2 \\), or \\( (2n_o)^2 (2n_v)^2 \\) in the spin-orbital
drivers. Memory budgets are enforced: an oversized job is refused with a
breakdown naming the dominant term instead of being OOM-killed partway through.

## Cite

CCSD: Scuseria, Janssen & Schaefer 1988; spin-adapted closed-shell CCSD
equations: Hirata et al. 2004. (T): Raghavachari et al. 1989; closed-shell
(T) algorithm: Rendell, Lee & Komornicki 1991. EOM-CCSD: Stanton & Bartlett 1993; Davidson correction: Olsen, Jørgensen & Simons 1990. LinLCCD(hh): Carter-Fenk 2025.
ωB97X-L-V: Ransford & Carter-Fenk 2026. B2PLYP: Grimme 2006. DSD-PBEP86:
Kozuch & Martin 2013. Review: Bartlett & Musiał 2007. Full entries in
[References](../reference/references.md).
