# LNO-MP2 lane (2026-07-28 .. 07-31): archived measurements

Written 2026-10-23. This is a record for future per-LMO / fragment work, not
a feature description. None of the code below is on `main`.

**Source.** Branch `worktree-linlccd-hh` (local worktree
`.claude/worktrees/linlccd-hh`). About 95 commits from before the history
rewrite, so they share no history with `main`; the SHAs below resolve only on
that branch. The modules (`lno.rs`, `lno_driver.rs`, `lno_coupled.rs`,
`lno_fragment.rs`, `lno_laplace.rs`, `lno_pno.rs`, `lno_aux_fit.rs`,
`bp_domain.rs`, `pao.rs`, `aux_domain.rs`, `pair_screen.rs`, about 10k lines in
`crates/ferric-mp2/src`) were never merged. `main` went to
`dlpno_localized.rs` (pair screen on localized occupieds) and `lmp2_direct`
instead.

**What was ported.** One idea from the lane: add-back of the screened pairs'
estimated energy (`80b99e508`, `c3cace821`), now
`LocalizedMp2Config::add_back_truncation` in `dlpno_localized.rs`. It is
re-measured on `main`'s own path in
`crates/ferric-mp2/tests/validation_dlpno_localized.rs` (gains of 4.1x to 26x
on C4–C8; see that file). The estimators differ: the branch added back a
dipole-dipole ORIENTED estimate from its own pair screen, while the port adds
back the full diagonal-f MP2 pair estimate that `main`'s screen already
computes. The branch's 2.2x–4.5x gain and "no time cost" describe the branch's
estimator, not the ported one.

**What was checked and not ported.** The amplitude `conv_thresh` loosening
(`0330d11b7`, `fe09cf88c`: 1e-12 to 1e-9 gave 1.63x on the LNO DIIS solve at
0.0000 uEh). `main`'s Jacobi solve already defaults to 1e-10. Going to 1e-9
there saves 3 of 23 iterations and moves the energy by up to 5.9e-11 Ha, so
the default is unchanged (numbers in the same validation file). `fe09cf88c`
also corrects `0330d11b7`'s reasoning: the energy error is LINEAR in the
threshold (err/thresh flat at ~0.03), not quadratic.

Everything below was measured on the branch. The numbers are as they were
recorded. The verdicts are dated and provisional.

## 1. Fragment LNO-MP2 (pyscf/lno port: one-hole MP1 densities, spectral truncation)

Construction: each Boys LMO is a fragment. A one-hole MP1 occupied density
gives natural orbitals, which are kept if `|occ| > thresh` and then
semicanonicalized. The virtual space comes from a one-hole virtual density.
The fragment MP2 is closed-form, with the projector `u_loc u_locᵀ`.

| SHA | Measured |
|---|---|
| `a9f5df5db` | B layout `(naux, nocc·nvir)`, column `i·nvir+a`, reproduces production `ri_mp2` to < 1e-12 (water/cc-pVDZ) |
| `d807b6692` | one-hole density matches the Python prototype to 8.9e-9 relative |
| `d5b0d09c3` | **zero-threshold anchor**: Σ fragments − canonical = 3.33e-15 (water/cc-pVDZ), 1.54e-13 (butane/cc-pVDZ, nocc 17). The first attempt had a flipped sign: +0.204014809927 against −0.204014809927 |
| `a909ea245` | occ thresh 1e-5: water keeps 96% of occ (err −0.454 kcal/mol), butane 69% (err +0.157). At 1e-3 butane is off by 23.6 kcal/mol |
| `3071539ae` | RETRACTION: the first virtual density was particle-side (the wrong projector, two terms instead of four). With the correct one-hole form, nv_frag ~ nocc^0.249 at pno 1e-6, and the error per occupied is flat at 0.0436 / 0.0470 / 0.0451 kcal/mol (C4/C6/C8) |
| `a831100d4` | butane, occ 1e-6: pno 1e-5 / 1e-6 / 1e-7 / 1e-9 keeps 0.354 / 0.557 / 0.737 / 0.919 of the virtuals, err 3.85 / 0.41 / 0.063 / 0.0003 kcal/mol |
| `179879fdd` | driver, localized occupied Fock (1.003 Ha off-diagonal on water): 4.44e-16 vs RI-MP2 at zero thresholds; 0.023 kcal/mol at 1e-6/1e-6 |
| `d810134d1` | **NEGATIVE, wall clock** (cc-pVDZ, release, min of 3): C2 0.92x, C4 0.94x, C6 0.78x, C8 0.67x against canonical. Phase exponents: occ_density nocc^5.07, vir_density nocc^4.93, energy nocc^3.99. Both density builds run at the order of canonical RI-MP2, because truncation is the OUTPUT of that step |
| `dbaa52d93` | PAO-domain kernels (Python first): m ~ nocc^0.20; predicted occ_density nocc^5.00 → nocc^3.41; C8 at t_bp 0.999 gives 6.249 vs 6.229 kcal/mol full-virtual (0.02 kcal/mol). Rust butane m/nvir 0.675 vs Python 0.661 |
| `40539d0fb` | C8 phase trace (occ/pno 1e-6): no domain 23.0 s, PAO 0.999 12.0 s (1.78x of the fragment phases). Exponents MISSED the prediction: occ 4.61 (predicted 3.41), vir 4.39, energy 3.49. `rotate_vir_axis` costs nocc²·nvir·m·naux ~ nocc^4.27 |
| `51a918818` | integral-direct (domain folded into `c_right`): occ 4.39 / vir 4.17 / ene 3.32, C8 fragment phases 9.6 s (2.15x vs full-virtual); equal to the global-B path to 3.05e-16. The AO half-transform (~nocc^3.93) now dominates |
| `f693db070` | one shared half-transform for all fragments: FLOPs 2.53x / 3.09x / 3.67x (C4/C6/C8). **No observable wall-clock change**, because RHF is 77% of the C8 job |

Interpretation (2026-07-31, provisional): the fragment construction is
correct and exactly anchored. At C2–C8/cc-pVDZ it is slower than canonical
RI-MP2 end to end, and correlation is under 2% of the job there. The PAO
domain fixes the density-phase order only partly (4.4 measured against 3.4
predicted). No size past the locality onset (~30 Bohr) was measured.

## 2. Per-LMO LNO pair lane (BP domains, coupled occupied solve)

| SHA | Measured |
|---|---|
| `bf39b2e83` | domain-local aux fit routed through the coupled path: aux retention 0.5733, energies agree to 6e-7 Ha, **speedup 1.01x** (the coupled amplitude iteration is 21.1 s of 21.4 s). Hoisting `ThreeIndexSource` out of the batch loop: C8 25.27 → 21.36 s |
| `80b99e508` | add-back on the neopentane dimer (34 atoms, 260 bf; canonical 25.88 s): pair 1e-4 err 2.37 → 0.67 kcal/mol, pair 1e-3 15.10 → 3.33, no time cost. The 1e-3 + add-back setting runs at 2.32x but carries 3.33 kcal/mol: on 3D systems the trade is accuracy XOR speed |
| `c3cace821` | add-back, 5 of 5 cases improved with no overshoot: C6 1e-3 3.08x, C8 1e-4 2.15x, C8 1e-3 3.15x, 3D 1e-4 3.56x, 3D 1e-3 4.54x. The dipole-dipole ORIENTED estimate was used (NOT the diagonal-f estimate the `main` port uses, so these gains do not carry over); the orientation-free BOUND over-corrected 5.56x |
| `0330d11b7` | coupled DIIS conv 1e-12 → 1e-9: 23 → 12 iterations, 1.63x, 0.0000 uEh. DIIS depth 8 is optimal (deeper is worse); f_cut 1e-5 gives 1.08x at −0.003 uEh |

Memory: the localized-coupled run breached `budget_gb` (alkane_10, 7.3 GB peak
against a 0.055 GB prediction), because the coupled amplitudes were never
budgeted.

## 3. Laplace energy over LNO domains

| SHA | Measured |
|---|---|
| `4e4b5b687` | Laplace identity in a localized basis: 8.3e-8 Ha at n_quad 7 |
| `e6195801b` | against the scalar-denominator oracle (diagonal-F toy): n_quad 3 / 5 / 7 gives 8.2e-6 / 1.1e-8 / < 1e-8. Flipping the virtual exponent sign fails both tests |
| `125db783e` | diagonal-Fock mode: C6 err 2.2e-4 / 4.2e-6 / 6.1e-8 Ha (nq 3/5/7), C8 nq 7 8.6e-8. Slower by design: 0.94x / 0.86x / 0.78x |
| `56a9befaa` | **NEGATIVE, coupled mode** (n_quad 7, pair 1e-3) against Sylvester: C6 1.75x, C8 1.10x, C10 0.91x. Peak-memory ratio 0.80 / 0.98 / 1.08. The B̃ rotation, O(nocc·\|lmos\|²) per point, grows ~2.0x per rung against ~1.4x for the solve, so the crossover is behind, not ahead |

## 4. Why this was archived, not ported

* The lane's own end-to-end measurements are negative or neutral at every
  measured size: fragment 0.67x at C8, getting worse; local aux fit 1.01x;
  Laplace 0.91x at C10, getting worse.
* `wiki/local-ri-robust-domain-fitting.md` §7 (the wiki is untracked and is
  not in clones; it lives in the main checkout only) lists "Per-LMO local
  correlation (LNO gap): REFUTED as stated" (cost and integration, not
  accuracy).
* The fragment and PAO kernels need the per-LMO driver, the BP domains and
  `pao.rs`, none of which exist on `main`. Porting them would mean landing a
  second local-MP2 stack next to `dlpno_localized` and `lmp2_direct`, with no
  measured win.

If per-LMO work reopens, start from these facts. The fragment zero-threshold
anchor (3e-15) is the exactness test to reuse. Price every phase, including
the transform INTO the domain: three of this lane's cost models missed exactly
that term. And measure past ~30 Bohr before concluding anything.
