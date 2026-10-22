# RSH exchange forces on k-point RS-GDF: Python prototype (#492 item 1)

Code: `prototypes/pbc/pbc_kgrad_rsgdf_rsh.py` (library) and
`prototypes/pbc/run_kgrad_rsgdf_rsh.py` (anchors). Both sit next to the
prototypes they import (`pbc_kgrad_gdf`, `pbc_kgrad`, `pbc_supercell`).
Run them from `prototypes/pbc` with `python run_kgrad_rsgdf_rsh.py [anchor] [fd] [total] [mut]`.
The environment variable `ATT_LINDEP` sets the metric cut of the attenuated fit (default `1e-10`).

## What is differentiated

At a fixed density `D_s(k)`:

```
E_x^RSH = c_sr E_K[full fit] + (c_lr − c_sr) E_K[attenuated fit] − (v_M^mix/2N_k) Σ_s Σ_k tr[D_s S D_s S]
v_M^mix = c_sr v_M^full + (c_lr − c_sr) v_M^LR(ω)       (diag(n) supercell constants)
```

The attenuated fit is `klr.rs` `build_attenuated`. It is the LR pass on its own, with
`v_ω(K) = 4π/K² e^{−K²/4ω²}` on every `K = G + q ≠ 0` with `|K| ≤ 2ω√ln(1/prec)`.
It has no SR walks and no `c0`. Its derivative is the plain fit's LR force and metric
passes (`pbc_kgrad_gdf.grad`) with `vlr → v_ω`. The SR bins, the SR metric and the
q = 0 `M_g0` term are all absent. `fit_xgrad` is one code path for both fits. The SR
and G0 pieces run only when the fit carries them.

## Measurements

System: triclinic H2 (TRI2 geometry) with STO-3G, a 1x1x2 mesh, the ET-sp aux (20 Cartesian functions) and `ω0 = 1`. The density is a fixed model density. All numbers are max-abs force components in Ha/Bohr.

| Anchor | Result |
|---|---|
| A0: shared code, `fit_xgrad(J+K)` vs `pbc_kgrad_gdf.grad` | 0.0 (bitwise) |
| A0: `E_K` vs `jk_from_kB` | 6.9e-14 |
| A1: ω = 1e7 (gcut 24) attenuated vs plain fit, energy | 1.8e-13 |
| A1: same, force (\|F\| = 0.167) | 1.6e-13 |
| A1: ΣF of attenuated fit | 6.7e-16 |

Central finite differences of the attenuated exchange alone:

| ω, lindep | h = 1e-3 | 1e-4 | 1e-5 |
|---|---|---|---|
| 0.11, 1e-10 (\|F\| 1.4e-5) | 1.8e-12 | 1.8e-14 | 1.0e-15 |
| 0.33, 1e-10 (\|F\| 3.7e-2) | 3.0e-5 | 6.7e-4 | 3.9e-3 |
| 0.33, 1e-6 | 2.7e-8 | 2.7e-7 | 1.4e-6 |

Total RSH exchange force against finite differences:

| Functional | h = 1e-3 | 1e-4 | 1e-5 |
|---|---|---|---|
| HSE06 (0.25/0/0.11, \|F\| 6.2e-2) | 7.4e-9 | 2.0e-10 | 1.3e-9 |
| CAM-B3LYP (0.19/0.65/0.33, \|F\| 0.11, att lindep 1e-6) | 1.1e-8 | 1.2e-7 | 6.3e-7 |

### Conditioning of the attenuated fit

The CAM rows at lindep 1e-10 grow as 1/h. That is energy noise, not an error in the
derivative. At ω = 0.33 the attenuated metric (72–78 K vectors against 20 aux
functions) keeps eigenvalues of 1.2e-11, 2.1e-11 and 3.9e-10, just above the cut.
Reversing the order of the K summation changes `E_K[erf]` by 4.6e-8 and F by 2.2e-8 at
lindep 1e-10. At lindep 1e-8 the change is 2.3e-9 (E) and 1.5e-9 (F); at 1e-6 it is
1.6e-11 and 1.5e-11. Moving from lindep 1e-10 to 1e-6 changes E by 4.9e-8 and F by
2.2e-8. These are the Python prototype's numbers. The Rust energy does not carry this
noise: re-chunking its K sum and a rigid translation move `E_K[erf]` by at most 7e-17 at
lindep 1e-10, 1e-8 and 1e-6. In Rust it is the analytic FORCE that is roundoff-limited
when eigenvalues near 1e-11 are kept: a rigid translation of 1e-3 at fixed density moves
it by 3.7e-8 / 2.9e-9 / 4.6e-12 at lindep 1e-10 / 1e-8 / 1e-6, and the plain Coulomb
RS-GDF force shows the same limit with a near-duplicate aux shell
(`crates/ferric-pbc/tests/validation_krsh_rsgdf_force.rs`; see #512).

At ω = 0.11 the q = 0 class has no K vectors, because |G|min ≈ 1.3 is greater than the
cut of 1.2. The q ≠ 0 class has 2 K vectors. As a result the attenuated piece of HSE06
on this small cell is about 1e-5.

## Mutation ledger (max |F_mut − F|, attenuated lindep 1e-6)

| Mutant | HSE06 | CAM-B3LYP |
|---|---|---|
| keep_c0 (K=0 / c0 kept in attenuated fit) | 0 (blind: no q=0 K) | 1.9e-4 |
| omega0 (fit built at ω0 = 1 instead of ω) | 3.0e-2 | 3.9e-2 |
| no_scale ((c_lr − c_sr) dropped) | 1.7e-5 | 2.0e-2 |
| madelung_unmixed (c_sr v_M^full only) | 1.0e-2 | 3.8e-2 |
| madelung_full_lr (LR piece with v_M^full) | 2.0e-2 | 1.8e-2 |
| madelung_scaled (c_sr · v_M^mix) | 1.5e-2 | 5.0e-2 |

The floors are about 3e-8 (CAM) and 1e-9 (HSE). Every mutant clears them by more than
1e3 except HSE06 keep_c0, which is exactly 0 on this cell (the q = 0 class has no K
vectors). HSE06 no_scale (1.7e-5) clears the HSE floor by ~1.7e4, but the Rust plan's
mutant bar (1e-5, item 5d) by only 1.7x. Both mutants must therefore be pinned on a
CAM-type ω, or on a cell longer than 2π/(2ω√ln(1/ε)) ≈ 5.2 Bohr at ω = 0.11.

## Rust change plan

1. **`rsgdf/kpoint/klr.rs`.** Add `pub(super) fn attenuated(&self) -> Option<(&KRsGdf, f64 /*ω*/, f64 /*c_lr−c_sr*/)>` on `RshAttached`. Refuse derivatives for `SwappedRoles`.
2. **`rsgdf/kpoint/kderiv.rs`.** Add `enum FitMode { Coulomb, Attenuated { omega: f64 } }`. Pass it as a new argument to `kpoint_fit_gradient`, `kpoint_fit_strain` and `kpoint_fit_core`. Every existing caller passes `Coulomb`. In `Attenuated` mode:
   - Build the `Stage` with `omega = ω` and `gcut = 2ω√ln(1/prec)`, exactly as in `build_attenuated`.
   - Leave out the following: `SplitPlan`, `fit_rotation`, `sr_bins_rotated`, `g0_inputs`, and the SR bin and pair-image ledger lines. Pass `c0 = 0`.
   - `rebuild_q`: no SR fold and no `subtract_g0*`. It keeps `hermitize_pairs` at q = 0, and it must match `build_attenuated`'s `lr_accumulate_q` half sets bit for bit, so that `b_mismatch` passes.
   - `q_class_pass`: apply `fit_block`'s rule. An empty K set, or all eigenvalues at or below lindep with `wsum < lindep`, means the class is skipped with a zero contribution instead of raising today's "every eigenvalue <= lindep" error. HSE06 q = 0 hits this rule.
   - Split the q = 0 `Z` Hermitisation out of `coulomb_q0`. Under `Attenuated`, run the Hermitisation only, with no ρ term and no `M_g0`.
   - `k_lr_force_unsplit` and `k_lr_strain_q` stay unchanged, because they read `st.omega`. Skip `k_sr_force_parts`. In `finish_strain`, skip the SR terms and the G0 volume terms.
   - Return `orb_sr`, `aux_sr` and `metric_sr` as zeros, `mg0` as zeros, and `smooth_g0` as `None`.
3. **`kgrad.rs`.** In `check_inputs_rsh`, replace the RS-GDF refusal at lines 801–806 with `check_dense_rsh`-style consistency (`s.gdf.rsh()` must equal `rsh`). Allow it only on the closed-shell RKS entry (`kpoint_rks_gradient`), the same scope as dense.
   - `assemble`: the full fit keeps `exch_fit` scaled by `c_sr` (`exact_exchange_fraction`). If `attenuated()` is `Some`, add a second `kpoint_fit_gradient(lr, …, scaled_exch_terms(spins, c_lr − c_sr), FitMode::Attenuated{ω})` and add its `orb_lr`, `aux_lr` and `metric_lr`.
   - `rsh_and_m_term_vm`: for RS-GDF return `(gdf.rsh(), gdf.madelung)`. That value is already mixed, so the M-term must not scale it by `c_sr` again (madelung_scaled).
4. **`kstress.rs`.** Lift the refusal at line 1079 in the same way. The attenuated strain is `kpoint_fit_strain(FitMode::Attenuated)` plus `RshParams::madelung_strain` (already present).
5. **Tests.** Add a new `validation_krsh_rsgdf_force.rs`. Every test in it is `#[ignore = "validation: ..."]`.
   - (a) ω = 1e7 attenuated force equals the plain fit, bar 1e-11 (measured 1.6e-13).
   - (b) Attenuated alone vs FD at h = 1e-3 and ω = 0.33, lindep 1e-6, bar 1e-6 (measured 2.7e-8).
   - (c) Total CAM-B3LYP and HSE06 RKS force vs FD, bar 1e-6. For HSE06, use a cell with a > 6 Bohr so that q = 0 carries K vectors.
   - (d) Mutants keep_c0, omega0, no_scale, madelung_unmixed and madelung_scaled. Each must exceed 1e-5; the smallest measured is 1.9e-4.
   - Also: the `ΣF = 0` guard, and a per-commit fast test for the empty-class skip.
   - Grep for the old refusal text (`has no derivative here`) in tests, docs and `site/`.
