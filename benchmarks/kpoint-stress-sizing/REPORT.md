# Sizing: k-point stress tensor, RHF/UHF (#236; related #353)

Paths are under `crates/ferric-pbc/src` unless noted. Sizing only; the probe is `probe_kstress.rs.txt` with output `probe_run.log` (74 s; run it as a release test in `crates/ferric-pbc/tests/`).

## Convention (verified in code)
`Cell::strained` (lattice.rs:129-160) moves lattice rows and atoms by (1+eps) and freezes every G sphere and image list as integer indices of the reference cell: G(eps) = (1+eps)^-T G0 at fixed Miller index (stress.rs:8-12). k-points are `k = sum f_i b_i` with integer numerators (kpts.rs:1-12), so `KPointMesh::gamma_centred(&strained, n)` gives k(eps) = (1+eps)^-T k0 automatically. The Bloch phase depends only on `n_i mod M_i` and the integer numerators (`phase`, kpts.rs:263) and the residue binning is invariant, so there is no "k strain" term: every k weight built for the forces (Mt(L), Delta(L), Yt_r, Zbin[r_L,r_T], the Wm bins, M_g0(k)) is reused unchanged and the stress multiplies the same per-image or per-triplet derivative by the pair vector, as the Gamma stress does with the Gamma force walk. PySCF `krks_stress.py:301-309` uses the same convention.

## Term inventory
| # | Term | Gamma stress | k force analogue | Needed at k |
|---|---|---|---|---|
| 1,2 | overlap Pulay (incl. c0, Madelung S) and kinetic | stress.rs:999-1050 `add_pair_virial` | kgrad.rs:864-903, weights `w_s`/`w_t` (810-840) | same loop with `w_s[r(L)]`, `w_t[r(L)]` times the pair vector |
| 3 | V_SR | `sr_attraction_strain` hcore.rs:1640 | `sr_attraction_deriv_visit` kgrad.rs:905-935, weight `w_d` | swap `d` for `w_d[residue(t.l)]` in the closure (hcore.rs:1664-1700) |
| 4 | V_ECP | ecp.rs:1390 | none (k energy supports ECP, k forces refuse it, kgrad.rs:534) | out of first scope |
| 5 | V_LR (full G sphere) | `pair_ft_strain_chunked` pair_ft.rs:1129 (half sphere) | `pair_ft_deriv_residues_chunked` residues.rs:439 | NEW residue-bucketed copy of `pair_ft_strain_block` (pair_ft.rs:1171) giving P plus 9 dP per residue |
| 6,7 | c0 volume; Ewald E_nn | stress.rs:1155; `ewald_nuclear_strain` | `ewald_nuclear_gradient_parts` kgrad.rs:1043 | identical to Gamma, N = (1/Nk) sum_k tr D(k)S(k) |
| 8 | Madelung | `madelung_strain` ewald.rs:507 | v_M = `mesh.madelung` kpts.rs:342 (Madelung constant of the diag(N) supercell lattice) | `madelung_strain` of the supercell Cell weighted by -1/2 (1/Nk) sum_s sum_k tr[D_s S D_s S] |
| 9,15,16 | dense-AFT J and K | stress.rs:1374 | `dense_two_electron` kgrad.rs:1144-1254 | per-q strain on K = G+q (dv/dK^2 with -2 K_a K_b, -delta E volume) with the new kernel |
| 10,11 | RS-GDF SR3, J2 | `sr3_strain`/`sr2_strain` strain.rs:456/553 | `sr3_force`/`sr2_force` closures `z(l,t)`, `wm(t)`, kderiv.rs:1004-1058 | give the strain walks the same closures (addend at strain.rs:493-551 already receives `l`, `t`) |
| 12,13 | RS-GDF LR unsplit, G=0 | `lr_strain_unsplit` strain.rs:330; strain.rs:298 | `k_lr_force_unsplit` kderiv.rs:918; `mg0`/`w_g0` kgrad.rs:857 | per-q residue strain with complex Yt_r (`aux_ft_strain_shells` takes any G list); q=0 dS pass plus J3/J2 volume terms |
| 14 | range split | split/deriv.rs:1295-1500, 1644 | `k_lr_force_moved/smooth` split/ksplit.rs:463/531; `smooth_g0_force` kgrad.rs:1116 | per-q v_SR(|K|) slopes (`kernel_slopes` split/deriv.rs:116), smooth-pair strain, kept (q_c, S - S_ss) volume terms |

The pair-symmetry map M(b) acts only in the energy-side bin construction; the derivative and strain walks use ordered walks, so M(b) is not touched.

## Anchors
1. 1x1x1 mesh equals the Gamma stress (independent construction: half sphere real vs full sphere complex, blind to phases).
2. k-mesh equals one supercell copy: build the cell with `supercell()` from `tests/pbc_kgrad.rs:252` (rows N_i a_i); de_deps(k, per cell) = de_deps(supercell Gamma)/N and sigma_k = sigma_sc since Omega_sc = N Omega. Needs a NON-TRIM mesh (some n_i >= 3): ConjD and WrongQ are identities on meshes with all n_i <= 2 (kgrad.rs:62-69). Valid for Gamma-centred meshes only (a shifted MP mesh is a twisted supercell, so only FD anchors it). Agreement is to truncation precision (forces: 1e-12 dense, 1e-11 RS-GDF).
3. FD of the k energy: `cell.strained(eps)`, `KPointMesh::gamma_centred(&strained, n)` with the same integer numerators, fresh SCF. 18 SCFs for all nine components (6 for the symmetric set); keep RS-GDF 2x2x2 `#[ignore]`.
4. PySCF 2.13.1 has `pyscf.pbc.grad.krks_stress` / `kuks_stress` (FFTDF only; `NotImplementedError('Stress tensor for hybrid DFT')` at krks_stress.py:289-290; no KRHF/KUHF stress), so it cannot anchor exchange, RS-GDF or the Madelung term; it could anchor non-exchange terms and the KS-DFT XC part (#353) at LDA/PBE.

## Probe (H2, a = 4 cubic, off-axis, STO-3G, dense AFT precision 1e-12, hcore omega 0.8, 1x1x3 Gamma-centred mesh vs the explicit 6-H supercell at Gamma; 74 s)
Hypotheses before probing: H1: with k fixed in fractional coordinates strain does not touch any Bloch phase, so E_k(eps) on a strained cell equals E_supercell(eps)/N pointwise; H2: sigma_k = sigma_supercell. Artifact hypothesis: a non-smooth k energy under strain (sets not actually frozen) would show FD(h) and FD(2h) disagreeing faster than h^2.
- Unstrained E_k - E_sc/3 = -7.1e-14 (both exxdiv). Strained cells (eps_zz, xz, zx, xx at h = +-1e-4, exxdiv none and ewald): pointwise E_k - E_sc/3 = -6.9e-14..-5.9e-14 for every printed point. H1 confirmed. FD(h) vs FD(2h) differ at the h^2 level (the ~1e-8 on (z,z) is h^2 truncation, identical in the supercell FD); the artifact hypothesis is rejected for this system.
- After Richardson, the analytic Gamma-supercell stress / 3 matches the k FD to <= 7.1e-10 (floor an analytic anchor can reach; bar about 1e-8 at h = 1e-4): zz 7.1e-10, xz -3.4e-12/-6.6e-13, zx 2.1e-12/1.9e-12, xx 2.8e-10; FD_k - FD_sc/3 = 2e-10 (zz), 7-8e-12 (xz, zx), 4e-12 (xx). H2 holds.
- Probe limits: dense AFT only, one tiny s-only system, one mesh; RS-GDF and range-split smoothness under strain, shifted meshes, p shells, UHF and the K-set boundary (R3) were not probed.

## Sizing (S = 1-2 days, M = 3-5 days, L = 1-2 weeks)
| Term | Exists at k? | Size | Anchor | Risk |
|---|---|---|---|---|
| overlap, kinetic, c0, Madelung-S pair virial (1,2) | force weights yes; strain loop no | S | 1x1x1 = Gamma; supercell/N | M(k) carries the mesh v_M (GammaMadelung is the model mutant) |
| V_SR (3) | force walk yes | S | supercell/N | rotation |
| c0 volume, E_nn, Madelung strain (6,7,8) | Gamma functions plus supercell lattice | S | FD of `mesh.madelung`; supercell/N | unfrozen supercell Cell makes FD noisy |
| residue pair-FT strain kernel (5) | no | M | sum over residues = Gamma kernel; FD of the FT | memory x10 arrays x nr (up to 512 on MP-even meshes); serial |
| V_LR at k | force yes | S after the kernel | supercell/N | full sphere, no half-sphere trick |
| dense-AFT J/K strain (9,15,16) | force yes | M | supercell/N, FD | K-set freezing (R3) |
| RS-GDF SR3/SR2 (10,11) | closures yes | S-M | supercell/N | generalising the `Y3` API; ordered-parallel bit-identity |
| RS-GDF LR and G0 volume (12,13) | force yes | M-L | supercell/N | complex Loewner weights per q; the J2 volume term is constant under atom motion, so forces never see it |
| range split (14) | force yes | L | supercell/N split; 1x1x1 = Gamma split | most ways to be silently wrong |
| UHF | k UHF force yes | S (shared `assemble`) | H3 UHF supercell/N | none beyond RHF |
| ECP at k | no force, no stress | M-L | n/a | separate PR |
| column rotation at k (perf) | energy yes (#377), force no | M | rotated vs unrotated agree to screening | below |
Overall about 4-6 person-weeks for RHF/UHF with dense, RS-GDF and split, unrotated; items 5, 12 and 14 dominate.

Dependency order: (1) the residue strain kernel and the `(l,t)` closure form of `sr3_strain`/`sr2_strain`; (2) hcore, Ewald, Madelung and overlap/kinetic assembly; (3) dense-AFT J/K; (4) RS-GDF unsplit; (5) range split; (6) ECP and rotation perf, optional.

Mutation tests (modelled on `StressMutation` and `KGradMutation`, kgrad.rs:79-132), each must miss the anchor by more than 1e-4: Gamma-style NoPulay, NoSrImages, NoVolume, GUnstrained, FtNoCentres, NoC0Volume, NoMadelung, MadelungS, NoEwaldBg/Lr, NoAuxFt3/NoAuxFt, NoMetric, NoJ2G0Vol, NoJ3G0Vol; k-specific NoPhase, ConjD, WrongQ, NoInvNk, GammaMadelung (blind at exxdiv none and 1x1x1), FitAuxPhase; split SplitNoSrKernelStrain, SplitNoSmoothPair, SplitFullG0. Assert the volume-only mutants miss only on the diagonal; ConjD and WrongQ are identities on TRIM meshes, so assert that blind spot and require 1x1x3.

## Risks
- R1 non-TRIM complex bins: every gate needs a 1x1x3 mesh, plus 2x2x2 for the blind-spot assertion.
- R2 Madelung/exxdiv: v_M is the supercell constant summed over k; the MP-even shifted mesh has no supercell anchor; `ExxDiv::None` hides the whole term.
- R3 K-set freezing: `lr_kvectors` (rsgdf/kpoint.rs:1132-1137) and `dense_two_electron` (kgrad.rs:1182-1190) filter |K| <= gcut in the CURRENT frame although `cell.gvectors` is frozen, so under strain a K at the cut can flip and the strained k energy is piecewise (Gamma's frozen `half_gvectors` is not); the effect is at precision level (~precision/h, ~1e-10), not seen in the probe; if it matters filter by the reference-frame norm (the analytic derivative is unaffected).
- R4 M(b): no new exposure if the strain code uses the ordered walks.
- R5 RS-GDF smoothness under strain: unprobed.
- R6 memory: `pair_ft_strain_bytes_per_g` is much larger than the deriv kernel's, times nr.
- R7 speed: the Gamma strain and force kernels are serial, so the k kernel is not a regression but bounds wall time on large meshes.

## SR column rotation
Gamma (#370): force and stress differentiate the rotated walk with W' = T^T W T (sr_rotation.rs "Forces and stress"; `SrRotatedDeriv` strain.rs:177; `hcore_rotation` stress.rs:1061). k (#377): the energy builds rotate; k forces resolve Auto to off and refuse an explicit On (kgrad.rs:526-531, 579-583; `refuse_column_rotation` rsgdf/kpoint.rs:361). Missing at k: a per-residue `T^T W[r_L,r_T] T` weight for the rotated SR3 walk. Recommendation: the first PR does not rotate (unrotated walk, refuse an explicit On, as kgrad does; set rotation Off in the energy for FD anchors); the rotated stress lands with the rotated k force, since one transform serves both.

## Recommended PR sequence (each stage has its own anchor and a clean refusal for the part not yet built)
1. PR-1 dense RHF/UHF: items 1-3, 5-9, 15, 16, the residue strain kernel and `kpoint_rhf_stress`/`kpoint_uhf_stress` on `KDenseAftEri`. Anchors: 1x1x1 = Gamma; H2 1x1x3 and H3 UHF = supercell/N for both exxdiv; FD at ~1e-8; the mutants; |sigma - sigma^T| <= 1e-10. An hcore-only FD of tr[hD] at fixed D can gate the one-electron pieces before the ERI terms land.
2. PR-2 RS-GDF unsplit: items 10-13. Anchors: 1x1x1 = Gamma RS-GDF; 1x1x3 supercell/N; FD.
3. PR-3 range split: item 14. Anchors: 1x1x1 = Gamma split; 1x1x3 supercell/N split; k split = k unsplit at converged omega.
4. PR-4 optional: ECP at k (force and stress together) and rotation at k (force and stress together).
CLI and Python wiring can ride with PR-1.

## KS-DFT (#353) would add
XC grid strain (AO strain on Bloch AOs, phases still invariant, weight strain over image atoms; `xc_stress` stress.rs:1508 is the model; an atom-centred grid is not rotation invariant, so sigma has a grid-size antisymmetric part); range-separated hybrids (screened-exchange kernel strain at K = G+q); meta-GGA (tau strain and the AO Hessian); VV10 (the nonlocal grid term); ROHF/ROKS at k. k KS-DFT forces do not exist yet either, so they come first. PySCF provides a pure-functional FFTDF k stress, so the XC part can be pinned at LDA/PBE (the HF part cannot).
