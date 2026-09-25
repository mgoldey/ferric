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
  spin-orbital CCSD at cc-pVDZ, and the (T) step 9.6–42× faster.
- **Spin-orbital** (`ccsd`, `ccsd_t`): kept for non-restricted references and
  as the cross-check. Its (T) streams one occupied triple at a time, so memory
  is O(n<sub>o</sub>·n<sub>v</sub>³)-class rather than the dense six-index tensor.

**Run it.**

- CCSD: `method.kind = "ccsd"` (`examples/water-ccsd.toml`, water/cc-pVDZ); Python
  `ferric.run_ccsd(mol, bs, aux)`.
- CCSD(T) and CCD: **Python only**, `ferric.run_ccsd_t(mol, bs, aux)` and
  `ferric.run_ccd(mol, bs, aux)`. Not wired into the CLI.

**Aux basis.** The RI error is not negligible at CC accuracy: on water /
cc-pVDZ with `cc-pvdz-ri`, RI-CCSD differs from exact-integral CCSD by about
1.3e-4 Ha. Pick the aux for the accuracy you need, not by habit.

**Accuracy.** `ccsd` is Proven.

| Quantity | System / basis | Reference | Agreement | Pinned by |
|---|---|---|---|---|
| CCSD correlation energy | H2 / STO-3G | exact-integral numpy | −0.02052453 Ha | `ferric-cc/src/ccsd.rs::test_ccsd_h2_sto3g` |
| Closed-shell (T) | H2O / cc-pVDZ | PySCF `ccsd_t()` | ~1e-6 Ha (test asserts 1e-4) | `ferric-cc/src/ccsd_t_closed_shell.rs::closed_shell_t_h2o_ccpvdz_matches_pyscf` |
| Streaming vs dense spin-orbital (T) | H2O / cc-pVDZ | ferric's former dense path | 5e-16 Ha | `ferric-cc/src/ccsd_t.rs::streaming_matches_dense_h2o_ccpvdz` |

**Limits.** Closed-shell entry points only in the CLI and Python. The
spin-adapted (T) rejects spin-orbital amplitudes with a typed error rather
than mixing conventions.

## LinLCCD and ωB97X-L-V

**LinLCCD(hh)** is linearized coupled-cluster doubles with the hole–hole ladder
kept to all orders, closed shell only. The ladder keeps the correlation energy
finite as the HOMO–LUMO gap closes, where MP2 diverges.
`method.kind = "linlccd"` (`examples/water-linlccd.toml`). Proven (narrow,
exact limits only): no external code has a reference for the LinLCCD(hh)
energy; with the ladder off it reduces exactly to RI-MP2, and with exact
integrals its driver terms reproduce canonical MP2.

**ωB97X-L-V** is a double-hybrid functional that uses short-range LinLCCD(hh)
instead of MP2 for its correlation term. It converges its **own** ωB97X-L
Kohn–Sham reference (a non-converged reference is an error), then adds the
LinLCCD(hh) correction on those orbitals. `method.kind = "wb97x-l-v"`
(`examples/water-wb97xlv.toml`). `[dft] lambda` and `omega` override the
published 0.6 and 0.1 Bohr⁻¹; omitting them gives the published values.
Smoke: its pieces and limits are checked, but no reference value for the total
energy exists in ferric.

## MP2-based double hybrids

**B2PLYP** and **DSD-PBEP86**: a KS reference with weighted exchange and
correlation components, plus scaled (SCS-)RI-MP2 correlation.
`method.kind = "b2plyp"` / `"dsd-pbep86"` (`examples/water-b2plyp.toml`);
Python `ferric.run_double_hybrid(mol, bs, aux, kind="b2plyp")`. **Spike**: no
comparison to a reference code yet.

## Implementation

All contractions go through **`einsum!`**, a macro that maps tensor
contractions onto BLAS3 GEMMs. The permutation copies that feed those GEMMs are
parallelized, because for a strided permutation the copy can dominate the
contraction it feeds: measured at 47% at n<sub>v</sub> = 40 and 70% at
n<sub>v</sub> = 80. The copies are bit-identical regardless of thread count,
since a permutation writes each output element exactly once; a test pins this
and has been checked to fail when deliberately broken.

## Memory

The amplitude tensors dominate and grow as
\\( n_o^2 n_v^2 \\), or \\( (2n_o)^2 (2n_v)^2 \\) in the spin-orbital
drivers. Memory budgets are enforced: an oversized job is refused with a
breakdown naming the dominant term instead of being OOM-killed partway through.

## Cite

CCSD: Scuseria, Janssen & Schaefer 1988; spin-adapted closed-shell CCSD
equations: Hirata et al. 2004. (T): Raghavachari et al. 1989; closed-shell
(T) algorithm: Rendell, Lee & Komornicki 1991. LinLCCD(hh): Carter-Fenk 2025.
ωB97X-L-V: Ransford & Carter-Fenk 2026. B2PLYP: Grimme 2006. DSD-PBEP86:
Kozuch & Martin 2011. Review: Bartlett & Musiał 2007. Full entries in
[References](../reference/references.md).
