# Sizing: #290 analytic DF (RI-J/RI-K) RHF/UHF Hessian, #291 analytic ROHF Hessian

Sizing only; no production code. Size scale: S = 1-2 days, M = 3-5 days, L = 1-2 weeks, XL = more.
Only one probe was run (PySCF, below). Every ferric-side statement comes from reading source.

## Summary

| | Size | Days | Wheel (2.7.2 small libint) | Biggest risk |
|---|---|---|---|---|
| #290 DF Hessian | L (RI-J only: M) | RI-J RHF 4-5, RI-K 4-5, UHF 2-3, wiring/validation 2 = about 13-15 | Works if the 3c/2c second derivatives come from a stencil of first-derivative blocks (recommended). The issue's analytic deriv2 route is source-build-only | RI-K truncated metric, and the matrix-valued RI-K `F^x` (not the skeleton) |
| #291 ROHF Hessian | L, low end (M-L) | matvec FD check 1, generator + skeleton 2, response + W^y 3-4, wiring/validation/docs 2 = about 8-10 | Same as RHF/UHF today: needs 4-centre deriv2, so source build only, and Auto falls back to FD | The response/W^y derivation, and mutants that are identities at convergence |

The two issues are independent. Do #291 first if only one is funded: it needs no new integral code and reuses the UHF skeleton.

## Facts verified in the code

- **Refusals:** `hessian.rs` refuses RI in `config_refusal` (about line 550, `df_j_aux`/`df_k_aux`) and in `check_supported` (line 415, `active_df_route`). It refuses ROHF in `spin_refusal` (line 428). `frequencies.rs:567` refuses `FrequencyReference::Rohf`.
- **Recent extensions:** the field, smeared and ECP Hessians use one pattern: a 4th-order stencil of first-derivative blocks, step 2e-3, rigid per-atom displacement, with `dV/dx` joined into `F^x` through `first_order_ao`.
- **DF first-derivative engines exist:** `new_3center_deriv`, `new_2center_deriv`, `compute_eri3_deriv`, `compute_eri2_deriv`, and shifted variants `compute_eri3_deriv_shifted(obs, dfbs, sh_p, sh1, sh2, [s_P, s_1, s_2])` and `compute_eri2_deriv_shifted(dfbs, sh_p, sh_q, s_Q)` (engine.rs 1640 and 1797). There is no `new_3center_deriv2` or `new_2center_deriv2`. The 3c shifted engine takes a shift per centre. The 2c engine shifts only Q; a rigid-atom move should be fine for a 2c integral (it depends only on the relative position), but this needs a check in PR 1.
- **Prior sizing report** (`research/hessian-sizing:benchmarks/hessian-sizing/REPORT.md`): the wheel's 2.7.2 small export has `LIBINT2_DERIV_ERI3_ORDER 1`, so a deriv-2 3c engine compiles and then aborts on first use. The 2.13.1 source build has deriv 2 with an effective l <= 3. Rigid-atom displacement is mandatory: moving one shell of a same-atom pair is off by 1.4e-3.
- **DF gradient** (`df_gradient.rs::df_jk_gradient`): RI-J uses the full Cholesky inverse. RI-K uses a per-spin-channel `V^-1/2 V^-1/2` metric with an eigenvalue cut and a Daleckii-Krein derivative with a 1e-7 noise floor. It is the template for the skeleton contractions, but it only contracts to a gradient vector. The Hessian needs the matrix `F^x` (virtual-occupied block per coordinate), which does not exist for DF.
- **ROHF:** `rohf_gradient` uses `W = D_a F_a D_a + D_a F_b D_b`, symmetrized, deliberately not `C diag(2 eps) C^T` (which misses FD by 1e-2 Ha/Bohr on HO2). `rohf_newton.rs::hessian_matvec(ctx, inp, k_vc, k_vo, k_oc, pool)` is `pub(crate)`; `gradient_blocks_from` returns half the true orbital gradient. `tests/rohf_newton_smoke.rs` only compares Newton against DIIS and has no FD check of the matvec.
- **PySCF 2.13.1** has `pyscf/df/hessian/{rhf,rks,uhf,uks}.py` and no `hessian/rohf.py` (checked against the installed package).

## Probe (the only thing run)

PySCF 2.13.1, water STO-3G at a non-stationary geometry, `density_fit(auxbasis="def2-universal-jkfit")`, conv_tol 1e-13, OPENBLAS=1.

- PySCF's analytic DF Hessian vs a central FD (h = 1e-3 Bohr) of its analytic DF gradient: max difference 5.5e-7 (max |H| = 0.989). The analytic Hessian took 1.9 s. So PySCF DF is a usable anchor at a bar of about 5e-6; FD of ferric's own DF gradient works at the same floor.
- PySCF's DF Hessian vs its exact-integral Hessian: max difference 5.2e-5. The issue's negative control (exact-J Hessian must miss the DF reference by >= 10x the bar) is marginal at STO-3G/jkfit against a 5e-6 bar; rerun at cc-pVDZ and set the bar from the measured max.
- DF stencil accuracy was NOT run in ferric. The nearest evidence is the prior report's smeared-charge 3c stencil: agreement with libint analytic second derivatives to 9e-11 at worst (l = 3,3), and with value mixed differences to 4e-13..7e-10; error falls as h^4. The 2c stencil has no such measurement, so PR 1 must measure it first.

## #290 DF Hessian

### Terms (RHF, RI-J, exact K)

Energy `E_J = 1/2 d^T M d`, `d_P = sum (P|mu nu) D`, `M = V^-1` (mirror the SCF's Cholesky solve), `c = M d`.

Skeleton, at fixed D and W:
1. V_nn, overlap/W and one-electron terms are unchanged.
2. Skeleton J is `(d_xy d)^T c - 1/2 c^T (d_xy V) c + (d_y d)^T M (d_x d) - (d_x d)^T M (d_y V) c - (d_y d)^T M (d_x V) c + c^T (d_x V) M (d_y V) c`. It needs the 3c second derivative, the 2c second derivative and the cross terms. The first-order vectors are naux x 3N (small).
3. Skeleton K (`df_k_aux`) uses per-channel `L^x_{P,ij}` (naux x m^2 x 3N) and the truncated `M`. With dropped modes (`DFK_LINDEP_THRESH`) it needs a Daleckii-Krein second derivative.
4. For RI-J with exact K, reuse the 4-centre skeleton (`skeleton_hess_2e`) with only the K part of Gamma.

First-order Fock: `F^x = h^x + J_DF^x(D) - 1/2 K^x(D)` as MO virtual-occupied blocks. J: `J^x_mn = d_x(mn|P) c_P + (mn|P)[M(d_x d - d_x V c)]_P`. K needs the matrix-valued derivative of the fit in the MO basis (as `pyscf/df/hessian/rhf.py` builds with ip1 3c integrals and aux-centre terms). This is the largest new code, larger than the skeleton.

CPHF response: replace `TwoElectronBuilder` (exact four-centre `build_jk_with_pool`) with a DF builder using the same `DfJ`/`DfK` objects the SCF used. The plumbing exists (`DfJ::from_source`, `new_for_jk`).

Reusable unchanged: terms 1, 2(1e) and 3; the CPHF PCG (`pcg`, `solve_cphf_one`, `assemble_response`); the frequency/dispatch layer and the UHF coupled solve; the RHF validation scaffolding.

### Stencil viability

A stencil of first-derivative DF integrals is viable and recommended: the wheel then works, there is no l <= 3 cap, and no shim work. Stencil `compute_eri3_deriv_shifted` and `compute_eri2_deriv_shifted` with 4 displaced evaluations per atom-coordinate (as `hessian_smeared.rs::pair_stencil`), shifting P, mu and nu rigidly with the atom. Cost is about 4x the DF-gradient integral work per coordinate, quadratic in N. Expected accuracy 1e-10..1e-9 on the integrals (smeared precedent), below the 5e-7 anchor floor; the 2c case and the Cholesky-solved `M` are unmeasured.

A cheap bring-up oracle: a stencil of the finished DF Fock/energy builders at fixed AO density (12N DF builds, no new integral code). It is worse for RI-K because of the near-cut noise band (the df_gradient doc records 5e-6 from modes in [1e-10, 1e-7)); use it as a test oracle only.

### Anchors, controls, mutants

Anchors: (a) trivial limit: an RI-J Hessian with an aux basis spanning the products exactly (a minimal system) equals the exact-J `rhf_hessian` term by term; (b) FD of ferric's own DF gradient (independent of PySCF); (c) DF off bit-identical to today; (d) row sums (translational and rotational invariance); (e) PySCF DF Hessian on ferric's bundled aux JSON on both sides, bar at 10x the measured max.

Negative controls: the exact-J Hessian must miss the DF reference by >= 10x the bar (marginal at STO-3G, see the probe); DF-K with dropped metric modes is refused with a typed error naming the count (option b of the issue).

Mutants (each must fail): drop `-1/2 c^T d_xy V c`; drop the 3c-2c cross terms; use the Coulomb metric for K or the K metric for J; use an explicit inverse where the SCF used Cholesky (the dfj-explicit-inverse-jitter trap); omit the aux-centre part of `J^x`; multiply the stencil step by 10; displace one shell instead of the atom (known 1.4e-3 miss); swap `d_x d` and `d_y d` (symmetric, may be an identity: check the algebra first).

### Risks

- Aux conditioning: `V` for def2-universal-jkfit has small eigenvalues; the RI-J Cholesky inverse amplifies stencil noise by cond(V), and the second derivative of `M` multiplies two such terms.
- Robust (Dunlap) fit is a different functional (RPA/MP2); keep the SCF aux separate from the RPA aux.
- Lindep: RI-K truncation makes the energy non-smooth at cut crossings, and a mode can cross the cut inside the stencil width. Refuse near-cut boundary eigenvalues, and refuse dropped modes altogether in the first version.
- Memory: `L^x` is 3N x naux x m^2; needs the gradient's budget gating. Avoid the "rebuilt per pass" trap (build per perturbation batch once, not per CPHF iteration).
- Range-separated RI-K: out of scope.
- CI cost: `#[ignore]` validation plus a few water/STO-3G FD tests (seconds); the stencil is quadratic in atoms, so keep default-run systems to 3-4 atoms; the PySCF generator takes about 2 s per system; no cold-shim cost unless the analytic deriv2 route is taken (then at least 8G MemoryMax).

### PR sequence

1. DF integral-stencil module (3c/2c) with an accuracy test against value mixed differences (and, on a source build, against libint deriv-2), plus the anchor (a) harness (S-M, 3 d).
2. RHF RI-J with exact K: skeleton J, `J^x` in `F^x`, DF `G` in CPHF, anchors (a)-(e), mutants (M, 4-5 d).
3. RI-K for the no-dropped-mode case only, typed refusal otherwise: skeleton K, matrix `K^x`, K in CPHF (M, 4-5 d). Optional follow-up: the Daleckii-Krein second derivative once shown correct by FD.
4. UHF per-spin channels and coupled CPHF (S-M, 2-3 d).
5. DF checks in `analytic_hessian_available` and `analytic_uhf_hessian_available`, an Auto/FD fallback test, validation row, scope doc, `gen_df_hessian.py`, ruff (S, 1-2 d).

Total about 13-15 working days: L, trending to XL if RI-K must handle dropped modes. RI-J alone (PRs 1, 2, 5) is M, about 8 days. The issue assumes analytic deriv2 engines (shim work, source-build-only); the stencil route avoids both, and analytic second derivatives are a later optimization.

## #291 ROHF Hessian

### Terms

Setup: `C_a = C_b`, MOs split into closed (c), open (o), virtual (v); alpha occupies c+o, beta occupies c.

Skeleton terms 1-4 are the UHF skeleton evaluated with the ROHF `D_a`, `D_b` and the ROHF `W` (`rohf_energy_weighted_density`). Reusable: `skeleton_hess_1e`, `skeleton_hess_overlap`, `skeleton_hess_2e` with `gamma_uhf`, `hess_nuclear_repulsion`. Hazard: `uhf_hessian_parts` builds `W` from UHF eigenvalues (`build_energy_weighted_density_uhf`), so the ROHF branch must take `W` from the spin Focks.

The response is new. The rotation parameters kappa live in the vc, vo and oc blocks. Closed-closed and open-open rotations are energy-invariant and fixed to `-1/2 S^x` by orthonormality, as in RHF/UHF.
- `U_a = [kappa_vc ; kappa_vo]` (virtual rows, c+o columns).
- `U_b = [kappa_vc ; kappa_oc]` (open orbitals act as beta virtuals). This is the UHF response with the two spins tied.
- The right-hand side uses the per-spin `F^x_s` from `first_order_ao`, projected to the three blocks via `gradient_blocks_from`, plus overlap terms built from the spin Focks (not a diagonal eps), plus `G` on the occupied-occupied density change `D_oo^x`.
- The operator is `hessian_matvec`; pin its normalization (`gradient_blocks_from` is half the true gradient).
- Hessian contribution: `sum_s Tr[D_s^y F^x_s(fixed D)] - Tr[W^y S^x]`, with `W^y = d_y(D_a F_a D_a + D_a F_b D_b)` needing `F_s^y = F_s^y(fixed) + G_s(D^y)` (`G(D^y)` is already a by-product of the matvec).
- This replaces the UHF `assemble_response`, which assumes spin-diagonal Focks (the `eps_ij^x` term) and is not reusable. This is the derivation to write in the module doc.

Reusable unchanged: `pcg`, `FirstOrderAo`/`first_order_ao`, `deriv1_g_matrices` with two `FockSpec`s, the refusal helpers, the `frequencies` dispatch.

### Anchors

PySCF has no ROHF Hessian, so all anchors are FD-based:
1. A closed-shell ROHF (singlet water) equals `rhf_hessian` term by term.
2. Each skeleton piece matches an FD of the corresponding `rohf_gradient` piece (pattern `uhf_hessian_fd.rs::check_skeleton_*`).
3. The total matches ferric's own central FD of `rohf_gradient`.
4. The total matches a PySCF central FD of PySCF's analytic ROHF gradient (`scf.ROHF(...).nuc_grad_method()`), built with the `gen_uhf_hessian.py` `hessian_fd` construction.

The floor measured on the analogous PySCF DF case was 5.5e-7, so a bar of about 5e-6 is plausible but unmeasured for ROHF.

Prerequisite check in PR 1: an FD of `gradient_blocks` along kappa (the `uhf_newton_smoke.rs` pattern) must show `hessian_matvec` is an exact Hessian and not just a preconditioner. If it fails in the closed-open block, expect 2-3 extra days.

### Controls and mutants

- Negative control: the UHF Hessian evaluated at ROHF orbitals must miss by >= 10x the bar on OH, NH2 and CH2 (plausible, unmeasured).
- Mutants (at least 5, as the issue asks): drop the oc block from the response; use a wrong `W` (the UHF eigenvalue form, or the `C diag(2 eps, eps_open)` Roothaan form); swap the alpha and beta densities; drop the 1/2 on an open-shell occupation; use a skeleton 2e Gamma with the wrong exchange. Extra: drop `-Tr[W^y S^x]`; use one spin's `F^x` for both.
- Trap: `D_a F_b D_b -> D_b F_b D_b` is an identity at convergence (the closed-open block of `F_b` vanishes). Compute the mutant minus the original algebraically before counting it.

### Risks

- Near-degenerate SOMO systems such as OH make the orbital Hessian near-singular and the PCG may stall (a typed error exists for that).
- PySCF ROHF can land on a non-lowest state; the generator needs the stability check from `gen_uhf_rohf.py`.
- ROHF orbitals are not unique, so compare invariants only (Hessian and frequencies).
- Needs 4-centre deriv2, the same source-build limit as RHF/UHF; Auto falls back to FD.

### PR sequence

1. FD test of `hessian_matvec` and `gradient_blocks` (S, 1 d). Decides the rest.
2. `gen_rohf_hessian.py`, references, the `Reference::Rohf` refusal set, the skeleton with ROHF `D`/`W`, per-term skeleton FD tests (S-M, 2 d).
3. Rotation-block CPHF solver, `W^y` assembly, the closed-shell anchor and the FD of `rohf_gradient` (M, 3-4 d).
4. `analytic_rohf_hessian_available`, Auto/Analytic for ROHF in `frequencies.rs` (ROKS stays FD), Python/CLI acceptance, `validation_rohf_hessian.rs`, validation.md row, updating the FD-only ROHF text (S-M, 2 d).

Total 8-10 working days: L at the low end.

## Not verified

- No ferric code was built or run. No stencil timing was taken. DF stencil accuracy for the 2c block and the Cholesky-solved metric is unmeasured, as is wall-clock against FD frequencies.
- The probe is PySCF only (water, STO-3G, jkfit): no ferric-vs-PySCF agreement, no cc-pVDZ negative-control margin, no RI-J-only difference, no UHF.
- ORCA was not checked for an analytic ROHF Hessian.
- Whether `hessian_matvec` is an exact Hessian is unestablished, as is its normalization against the gradient blocks (the factor 2).
- The wheel's ability to run the 4-centre deriv-2 skeleton was not re-checked (assumed to behave like the existing RHF/UHF tests, which skip on a first-derivative libint).
- The l <= 3 cap on 3c deriv-2 comes from the prior report and was not re-measured. Rigid-atom displacement with the 2c shifted engine (only Q shiftable) is unchecked.
- Cut crossings inside a stencil width were reasoned about, not tested. The wheel size effect of eri3/eri2 deriv 2 was not measured.
