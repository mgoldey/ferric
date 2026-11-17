# Book plan: ferric, an implementer's reference

The book explains how each equation in ferric becomes code. A reader should be
able to start from any equation, find the function that contracts it, and find
the script and test that check it. The audience is someone implementing or
auditing an electronic-structure method, not a student meeting the method for
the first time.

This plan is written against commit `81486b984`. It was drafted without
building or running any code.

Companion files: `book/notation.md` (drafted notation chapter),
`book/BUILD.md` (build proposal), `book/sample-ri-mp2.md` (one section drafted
to the proposed standard).

## 1. Sources and what each one is good for

| Source | Size | Role in the book | Trust |
|---|---|---|---|
| `crates/*/theory.md` | 17 files, 5,189 lines | The derivations of record. Uneven: `ferric-ci` 1,233 lines, `ferric-mp2` 633, `ferric-scf` 196, `ferric-integrals` 66, `ferric-tensors` 21. | High where a proof script and a code reference stand next to the equation. Line references inside them drift (see `book/notation.md`, section 10). |
| `scripts/*_sympy.py`, `scripts/mp2_variant_identities.py` | 21 scripts at this commit | The machine checks behind verified-equation boxes. | High. Only two of the 21 are run by a test at this commit (`oomp2_mp3_identities_sympy.py`, `ump3_spin_blocks_sympy.py`); the runner for the rest is on `docs/derivations-backfill`. |
| `testdata/derivations/registry.json` (branch `docs/derivations-backfill`) | 108 routes, 39 proven methods, 25 recorded gaps | The source of truth for coverage. Section statuses below are derived from it wherever a method id exists. | High. It is checked by a test and can only shrink its gap list. |
| `site/src` mdBook | methods pages 3,162 lines | User-facing "what runs and how to run it", measured numbers, validation grades. The book links to it and does not repeat it. | High for capabilities and numbers. |
| Dissertation (`dissertation.txt`, local only) | 8,420 lines | Chapter 4 for the RI-MP2 algorithm. Chapters 2, 3, 5 and 6 for the attenuated and spin-component-scaled methods, their parameters and their training. Chapter 1 for the framing of range separation. | High for the methods it defines. Its notation differs from the code's (`book/notation.md`). |
| `wiki/sympy_derivations.md` (local only) | 1,532 lines, 25 sections | Leads for sections that have no `theory.md` text yet. | Low. Working notes. Every item used is marked `verify` below and must be re-derived and machine-checked first. |
| `wiki/papers/sr-mp2-lr-rpa-derivation.tex`, `sr-mp2-lr-rpa.tex` (local only) | 735 and 1,387 lines | The capstone chapter's derivation and classification. | Medium. Unpublished. Marked `verify`. |

Snapshot of the registry used here: branch `docs/derivations-backfill` at
`2d955d5a8` plus its uncommitted working tree. That branch forked before
`81486b984`, so it lacks `scripts/ump3_spin_blocks_sympy.py` and the
"6.1 Spin-blocked unrestricted MP3" section, and it adds sections to
`ferric-scf`, `ferric-mp2` and `ferric-rpa` theory files that this commit does
not have. Sources that exist only there are marked †.

## 2. Editorial model

**One home for each equation.** The per-crate `theory.md` file stays the home
of a derivation, as `CONTRIBUTING.md` requires on `docs/derivations-backfill`.
A book chapter is the reading order, the shared notation, the algorithm and
the connective text; it pulls each derivation section from `theory.md` by an
include and adds what `theory.md` does not carry: why the equation has this
form, how the algorithm is laid out, and what the dissertation or the
literature says. A method change then updates the book by updating
`theory.md`, which contributors already do.

This needs one pass over the `theory.md` files to bring them to the notation
of `book/notation.md` and to add include anchors. That pass belongs to the
derivation backfill, which owns those files. Until it happens, book sections
are written as standalone text, as the sample section is. Question 1 in
section 8 asks Matt to confirm this model.

**Section template.** Every section has, in order:

1. a header table of index ranges and entry points (Rust, CLI, Python);
2. the equations, each working equation in a verified or unverified box;
3. the algorithm as a table of steps with cost, memory and the function for
   each step;
4. a code excerpt of the innermost kernel with a symbol-to-variable table;
5. reproducibility and memory rules where they affect a result;
6. options that change the equation;
7. numerical anchors, quoted from the validation page;
8. a sources table.

**Statuses.**

| Status | Meaning | Work remaining |
|---|---|---|
| `ready` | A committed derivation with function names, and a proof script with a runner (at this commit or on `docs/derivations-backfill`). | Notation pass and section template. |
| `edit` | A committed derivation or dissertation text exists, but it is thin, has no runner-backed proof for some equations, or needs restructuring. | Rewrite plus one or two proof claims. |
| `gap` | No committed derivation. | Derive from the code and the literature, write the proof script, then write the section. |
| `verify` | The only source is a working note or an unpublished draft, or the registry records the committed text as stale or unproven. | Re-derive against the code and machine-check before any text is reused. |

## 3. Table of contents

Source abbreviations: **T** `crates/ferric-<crate>/theory.md` section;
**S** script under `scripts/` († only on `docs/derivations-backfill`);
**R** registry method id (*gap* when the registry lists it under `gaps`);
**D** dissertation chapter or section; **W** `wiki/sympy_derivations.md`
section; **P** `wiki/papers/` ("derivation" is `sr-mp2-lr-rpa-derivation.tex`
with its sections numbered in file order, "paper" is `sr-mp2-lr-rpa.tex`); **M** `site/src/methods/` page.
Code references are `crate/src/file.rs::item` under `crates/`.

### Part 0. Front matter

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 0.1 | How to read this book: boxes, code references, grades | `book/BUILD.md`; `site/src/reference/validation.md` | none | edit |
| 0.2 | Notation and conventions | `book/notation.md` | see that file | ready |

### Part I. Foundations

**Chapter 1. Second quantization and the reference state**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 1.1 | Hamiltonian, spin orbitals, normal ordering | P derivation §1–3; W §1; D 1.1 | none | edit |
| 1.2 | Wick contractions as the proof scripts run them | S `oomp2_mp3_identities_sympy.py`; W §1–2 | none | edit |
| 1.3 | Spin adaptation: from spin orbitals to spatial orbitals | T mp2 "Conventions", §2; T cc header; S `mp2_variant_identities.py` claims 1–2 | `ferric-mp2/src/spinorbital.rs::{spin,spat,asym_oovv,asym_phys}`, `ferric-cc/src/ccsd_closed_shell.rs::expand_amplitudes_to_spin_orbital` | edit |

**Chapter 2. Molecules, basis sets and numerical ground rules**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 2.1 | Molecule, charge, multiplicity, units | T core §1 | `ferric-core/src/mol.rs::{Molecule,parse_xyz,nelec,nuclear_repulsion,apply_ecp}` | edit |
| 2.2 | Contracted Gaussians, normalization, parsers | T core §3; T integrals "Basis Sets" | `ferric-core/src/basis.rs::{BasisSet,Shell,load_bse_json,load_g94}`, `ferric-integrals/src/basis_bridge.rs::PreparedBasis` | edit |
| 2.3 | Determinism, threads and the memory budget | `site/src/reference/architecture.md`; code comments | `ferric-core/src/blas_threads.rs::{init_threading,with_blas_threads}`, `ferric-core/src/memory.rs::resolve_budget_bytes` | edit |
| 2.4 | Dense tensors and the einsum macro | T tensors; `site/src/reference/einsum.md` | `ferric-tensors/src/tensor.rs::Tensor` | edit |

### Part II. Integrals, screening and density fitting

**Chapter 3. Gaussian integrals**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 3.1 | One-electron integrals and multipoles | T integrals "One-Electron Integrals" | `ferric-integrals/src/oneelectron.rs::{overlap,kinetic,nuclear,hcore,dipole,second_moment}` | edit |
| 3.2 | Two-, three- and four-center integrals through libint2 and the C shim | T integrals "ERI", "Three-Center", "Two-Center" | `ferric-integrals/src/engine.rs::{new_2e,compute_quartet,new_3center,compute_eri3,new_2center,compute_eri2}` | edit |
| 3.3 | Derivative integrals and buffer sizing | none | `ferric-integrals/src/engine.rs::{new_1e_deriv,compute_1e_deriv_block_n,new_2e_deriv,compute_eri_deriv_quartet,new_3center_deriv}` | edit |
| 3.4 | A McMurchie–Davidson engine for three-center one-electron integrals | none | `ferric-integrals/src/md3c1e.rs::{e_table,boys,pair_block,a_matrix_at_point}` | edit |
| 3.5 | Effective core potentials | T integrals "Effective Core Potentials" | `ferric-integrals/src/ecp.rs::{ecp_matrix_spherical,ecp_matrix_deriv_spherical}`, `ferric-integrals/src/oneelectron.rs::hcore_ecp` | edit |
| 3.6 | The pVp integral and spin-free X2C-1e | S `pvp_ss_sympy.py`; R pvp-integrals *gap*; M scf "Scalar relativity" | `ferric-integrals/src/pvp.rs::pvp_nuclear`, `ferric-scf/src/x2c.rs::{x2c1e_transform,x2c1e_hcore}` | edit |
| 3.7 | Basis functions and their derivatives on a grid | none | `ferric-integrals/src/ao_grid.rs::{eval_basis_on_points,eval_basis_and_grad_on_points,eval_basis_grad_hess_on_points}` | edit |

**Chapter 4. Range-separated and attenuated operators**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 4.1 | erf and erfc; terf and terfc; the curvature constraint | D 2 (introduction), D 4.1 Eqs. 4.1–4.2; T mp2 §4 (kernel); S `mp2_family_sympy.py`† part D | `ferric-integrals/src/operator.rs::{Operator,erf,erfc,terf,terfc,terfc_with_omega,terf_with_omega}` | edit |
| 4.2 | The terf table engine and its far-field fallback | none committed; code comments in `crates/ferric-integrals/shim/shim.cc` | `ferric-integrals/src/engine.rs::{new_terf_libint2,is_table_engine}` | edit |
| 4.3 | Yukawa, Slater geminal and fitted geminal kernels | none | `ferric-integrals/src/operator.rs::{yukawa,slater_geminal,stg}`, `ferric-integrals/src/engine.rs::new_2e_geminal` | gap |

**Chapter 5. Screening**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 5.1 | Schwarz bounds, and why they do not see attenuation | none | `ferric-integrals/src/schwarz.rs::{schwarz,bound_inflation}`, `ferric-scf/src/screening.rs::{Bound,SchwarzBounds}` | edit |
| 5.2 | Distance-dependent bounds (QQR) for four- and three-index integrals | none | `ferric-scf/src/qqr.rs::QqrBounds`, `ferric-integrals/src/qqr3.rs::{QqrBounds3,estimate3}` | edit |
| 5.3 | Significant pair lists | none | `ferric-scf/src/pairs.rs::{SignificantPairs,DensityPairs,intersect_sorted}` | edit |

**Chapter 6. Density fitting**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 6.1 | The factorization, the metric, Cholesky and eigendecomposition factors | T mp2 §1; T gw §1; D 1.4.1, D 4.1 Eq. 4.5 | `ferric-integrals/src/threeindex.rs::coulomb_metric_2c`, `ferric-mp2/src/rimp2.rs::{cholesky_inverse_sqrt,eigh_inverse_sqrt,metric_inverse_sqrt}` | edit |
| 6.2 | Three-index sources: in core, packed, spilled, recomputed | none; `site/src/reference/input.md` (`jk_storage`) | `ferric-integrals/src/three_index_source.rs::{ThreeIndexSource,build_band_screened,build_dressed,for_each_block}` | edit |
| 6.3 | Mixed metrics and robust (Dunlap) fitting | T mp2 §1 (last paragraph) | `ferric-mp2/src/rimp2.rs::{ri_mp2_robust_attenuated_metric,robust_fit_coulomb_parts}` | ready |
| 6.4 | Complementary auxiliary basis (CABS) | T mp2 §11; S `mp2_variant_identities.py` claim 7 | `ferric-mp2/src/f12.rs::build_cabs`, `ferric-integrals/src/cabs.rs::cross_overlap` | edit |

### Part III. Self-consistent field

**Chapter 7. Hartree–Fock**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 7.1 | RHF, UHF and ROHF energy functionals | T scf "Hartree-Fock energy functionals"†; S `hf_energy_sympy.py`†; R rhf, uhf, rohf; D 1.1.2 | `ferric-scf/src/rhf.rs::solve_rhf`, `ferric-scf/src/uhf.rs::solve_uhf`, `ferric-scf/src/rohf.rs::solve_rohf` | ready |
| 7.2 | Orthogonalization and starting guesses | none | `ferric-scf/src/rhf.rs::canonical_orthogonalizer`, `ferric-scf/src/guess.rs::{hcore_guess,sad_guess,minao_projection_guess}` | edit |
| 7.3 | Direct and incremental J/K builds | none | `ferric-scf/src/rhf.rs::build_jk`, `ferric-scf/src/fock.rs::{JBuilder,KBuilder}`, `ferric-scf/src/direct_jk.rs::build_incremental` | ready |
| 7.4 | RI-J, RI-K and Split-RI-J | M scf "Choosing how exchange is built"; module docs of `df_j.rs`, `df_k.rs` | `ferric-scf/src/df_j.rs::DfJ`, `ferric-scf/src/df_k.rs::DfK`, `ferric-scf/src/split_rij.rs::SplitRiJ` | ready |
| 7.5 | LinK, seminumerical exchange (COSX) and CFMM | T scf "CFMM" (10 lines); M scf | `ferric-scf/src/link_k.rs::LinkK`, `ferric-scf/src/cosx_k.rs::CosxK`, `ferric-scf/src/cfmm.rs::CfmmJ` | ready |
| 7.6 | Fock hooks | T scf "Fock hooks and their nuclear gradient" | `ferric-scf/src/rhf.rs::{solve_rhf_with_hook,validate_fock_hook}` | edit |

**Chapter 8. Convergence**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 8.1 | DIIS and its variants | T scf "DIIS" (6 lines) | `ferric-scf/src/diis.rs::{Diis,DiisFlavor}` | ready |
| 8.2 | Level shift, MOM and smearing | none | `ferric-scf/src/mom.rs::mom_reorder`, `ferric-scf/src/smearing.rs::{fermi_dirac,solve_fermi_level}` | ready |
| 8.3 | Second-order steps: Newton, augmented Hessian, trust region | none | `ferric-scf/src/rhf_newton.rs::{rhf_newton_step,hessian_matvec}`, `ferric-scf/src/uhf_newton.rs::uhf_newton_step`, `ferric-scf/src/rohf_newton.rs::rohf_newton_step`, `ferric-scf/src/rohf_ah.rs::rohf_ah_step`, `ferric-scf/src/trah.rs::{solve_trust_region,rhf_trah_step}`, `ferric-scf/src/cayley.rs::cayley_unitary` | ready |
| 8.4 | Stability analysis | none | `ferric-scf/src/stability.rs::{rhf_internal_stability,rhf_external_stability,uhf_internal_stability}` | ready |

### Part IV. Density functional theory

**Chapter 9. Kohn–Sham**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 9.1 | Kohn–Sham equations and functional families | T dft "Kohn-Sham Equations", "Functionals"; R ks-semilocal | `ferric-scf/src/rhf.rs::ks_vxc_closed`, `ferric-dft/src/ks.rs::KsXc` | ready |
| 9.2 | Radial and angular grids, the Becke partition, pruning | T dft "Integration Grid", "Becke partition"; T quadrature §3; S `dft_becke_weight_grad_sympy.py`; R ks-grid | `ferric-dft/src/becke.rs::{becke_weights_all,becke_weights_and_grad}`, `ferric-dft/src/grid.rs::build_atomic_grid`, `ferric-dft/src/prune.rs::angular_orders_for_atom`, `ferric-dft/src/radial.rs::treutler_ahlrichs_m4`, `ferric-quadrature/src/lebedev.rs::lebedev` | ready |
| 9.3 | $V_{xc}$ matrix elements: LDA, GGA, meta-GGA | T dft "Semilocal $V_{xc}$ matrix elements"; S `dft_vxc_matrix_elements_sympy.py`; R ks-semilocal, ks-mgga | `ferric-dft/src/vxc.rs::{semilocal_vxc_closed,semilocal_vxc_polarized}`, `ferric-dft/src/density_on_grid.rs::{eval_density_closed,eval_tau_closed}` | ready |
| 9.4 | Global and range-separated hybrids; tuning $\omega$ | R ks-hybrid-rsh *gap* (a proof script is in progress on `docs/derivations-backfill`); D 1.3.2 | `ferric-scf/src/rsh_response.rs::exchange_response`, `ferric-scf/src/omega_tuning.rs::tune_omega` | ready |
| 9.5 | VV10 nonlocal correlation | T dft "VV10 nonlocal correlation"; S `dft_vv10_potential_sympy.py`; R ks-vv10 | `ferric-dft/src/vv10.rs::{compute_vv10_energy_and_potentials,add_vv10_scratch}` | ready |
| 9.6 | $f_{xc}$ response kernels | T dft "$f_{xc}$ response kernels"; S `dft_fxc_kernel_sympy.py` | `ferric-dft/src/fxc.rs::{LdaFxcKernel,GgaFxcKernel,MggaFxcKernel}`, `ferric-dft/src/lr_kernel.rs::{fxc_ov_block,singlet_fxc_ov_block}` | ready |
| 9.7 | Constrained DFT and electron-transfer coupling | M cdft; R cdft *gap* | `ferric-dft/src/cdft.rs::build_weight_matrix`, `ferric-scf/src/cdft_driver.rs::solve_cdft_uhf`, `ferric-scf/src/cdft_coupling.rs::coupling_hab` | ready |

### Part V. The MP2 family

**Chapter 10. MP2 and its variants** (dissertation chapters 2, 3, 5 and 6)

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 10.1 | Closed-shell RI-MP2 energy | `book/sample-ri-mp2.md`; T mp2 §1–2; D 4.1–4.2; S `mp2_variant_identities.py` claim 1; R ri-mp2 | `ferric-mp2/src/rimp2.rs::{ri_mp2,ri_mp2_spin_components,pair_energy,spin_components_from_b_ov_kappa_cpu}` | edit |
| 10.2 | Unrestricted RI-MP2 | T mp2 §3; S `mp2_variant_identities.py` claim 2; R u-ri-mp2 | `ferric-mp2/src/u_rimp2.rs::{u_ri_mp2,same_spin_pair_kernel,opposite_spin_pair_kernel}` | ready |
| 10.3 | Spin-component scaling and $\kappa$-regularization | T mp2 §2 (last paragraph), §4; S `mp2_family_sympy.py`† part C; R scs-mp2 | `ferric-mp2/src/scs.rs::scs_mp2`, `ferric-mp2/src/rimp2.rs::spin_components_from_b_ov_kappa` | ready |
| 10.4 | Attenuated MP2 with erfc and terfc | T mp2 §5; D 2, D 3; R attenuated-mp2 | `ferric-mp2/src/attenuated.rs::{attenuated_ri_mp2,attenuated_ri_mp2_long_range}` | edit |
| 10.5 | SCS-MP2(2terfc): two attenuation lengths | T mp2 §4; D 5.2 Eq. 5.6, D 5.3; R scs-mp2-2terfc | `ferric-mp2/src/scs.rs::scs_mp2_2terfc` | ready |
| 10.6 | Attenuation and the basis-set limit | D 6 | none | edit |
| 10.7 | The range split is not additive: the cross term | T mp2 §5 "Range split"; S `mp2_variant_identities.py` claim 4 | `ferric-mp2/src/attenuated.rs::rs_mp2_decomposition` | ready |
| 10.8 | MP2-V: attenuated MP2 with damped VV10 | T mp2 "13. MP2-V"†; S `mp2_family_sympy.py`† part E; R mp2-v | `ferric-mp2/src/att_vv10.rs::{att_mp2_vv10,u_att_mp2_vv10}`, `ferric-dft/src/vv10.rs::compute_vv10_damped_energy_and_potentials` | edit |
| 10.9 | MP3, restricted and spin-blocked unrestricted | T mp2 §6, §6.1; S `oomp2_mp3_identities_sympy.py`, `ump3_spin_blocks_sympy.py`; R mp3. W §2b is superseded by the scripts: its ring prefactor was wrong. | `ferric-mp2/src/mp3.rs::{mp3_energy,mp3_energy_from_spin_orbital_blocks}`, `ferric-mp2/src/u_mp3.rs::u_mp3` | ready |
| 10.10 | MP2-based double hybrids | R double-hybrid *gap* (owned by another branch); M cc "MP2-based double hybrids" | `ferric-mp2/src/double_hybrid.rs::mp2_double_hybrid` | ready |
| 10.11 | Orbital-partitioned Padé MP3 | T mp2 §6.2; S `opmp3_identities_sympy.py`; R opmp3 | `ferric-mp2/src/opmp3.rs::{w_channels,pade,op_mp3,sop_mp3,ssd_opmp3}`, `ferric-mp2/src/opmp3_driver.rs::run_opmp3` | ready |

**Chapter 11. RI-MP2 algorithms and memory** (dissertation chapter 4)

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 11.0 | Chapter overview: the RI-MP2 pipeline and where each piece lives | `book/sample-ri-mp2.md`; T mp2 §1–2 | none | ready |
| 11.1 | The four-step algorithm, 2014 and now | D 4.2 Table 4.1, Functions 1–4; `book/sample-ri-mp2.md` §10.1.3 | `ferric-mp2/src/rimp2.rs::ri_mp2_spin_components` | ready |
| 11.2 | Streaming the MO transform and the dressing | code comments in `rimp2.rs` | `ferric-mp2/src/rimp2.rs::{stream_dressed_mo_band,stream_dressed_mo_band_budgeted,mo_stream_chunk_for}` | ready |
| 11.3 | The pair-energy kernel: batching, symmetry, fixed summation order | code comments in `rimp2.rs`; D 4.2 | `ferric-mp2/src/rimp2.rs::{spin_components_from_b_ov_kappa_cpu,pair_energy}` | ready |
| 11.4 | Memory budget, admission checks and the shared pool | code comments; `site/src/reference/input.md` `[memory]` | `ferric-mp2/src/rimp2.rs::{eri3_budget_bytes,mo_side_alloc_bytes,check_mo_side_alloc}` | verify |
| 11.5 | Parallel performance then and now | D 4.3; `site/src/reference/validation.md` | none | verify |
| 11.6 | Device and mixed-precision kernels | `site/src/reference/validation.md` "Mixed-precision GPU kernels" | `ferric-mp2/src/rimp2_gpu.rs::try_spin_components_on_device`, `ferric-mp2/src/rimp2.rs::mixed_energy_operator` | ready |
| 11.7 | Distributed RI-MP2 by auxiliary bands | none | `ferric-mp2/src/mpi_rimp2.rs::run_mpi_ri_mp2` | verify |

**Chapter 12. Orbital optimization and Laplace forms**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 12.0 | Chapter overview: orbital optimization and Laplace forms | T mp2 §7, §8, §14 | none | verify |
| 12.1 | Orbital-optimized RI-MP2 | T mp2 §7; S `oomp2_mp3_identities_sympy.py`; S `mp2_variant_identities.py` claim 6; R oo-rimp2 | `ferric-mp2/src/oo_rimp2.rs::{oo_ri_mp2,build_oo_mp2_relaxed_density}`, `ferric-mp2/src/orbital_rotation.rs::cayley_rotation` | ready |
| 12.2 | Unrestricted OO-RI-MP2 | T mp2 "14. Unrestricted OO-RI-MP2 orbital gradient"†; R u-oo-rimp2 *gap* | `ferric-mp2/src/u_oo_rimp2.rs::u_oo_ri_mp2`, `ferric-mp2/src/u_rimp2.rs::compute_u_mp2_orbital_gradient` | verify |
| 12.3 | Laplace-transform MP2 and SOS-MP2 | T mp2 §8; T quadrature §1; S `mp2_variant_identities.py` claim 3; R laplace-mp2, laplace-sos-mp2; W §11 | `ferric-mp2/src/laplace.rs::{laplace_ri_mp2,laplace_sos_mp2,build_pseudo_density_occ,build_pseudo_density_vir}`, `ferric-quadrature/src/minimax.rs::select_minimax_points` | ready |
| 12.4 | Local Laplace SOS-MP2 | M mp2 "Laplace formulations" | `ferric-mp2/src/laplace_local.rs::sos_local_energy`, `ferric-mp2/src/boys.rs::boys_localize` | gap |

**Chapter 13. Local MP2**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 13.1 | Amplitude-threshold local MP2 | T mp2 §12 (a pointer); S `lmp2_hylleraas_vs_wshg23_sympy.py`; W §14; R local-mp2 *gap* | `ferric-mp2/src/lmp2_amplitude.rs::{amplitude_lmp2,build_vvhv,hylleraas_energy}`, `ferric-mp2/src/ragged.rs::solve_ragged` | edit |
| 13.2 | The integral-direct path | M mp2 "Exact and local MP2" | `ferric-mp2/src/lmp2_direct.rs::amplitude_lmp2_direct` | ready |
| 13.3 | Pair natural orbitals and DLPNO-MP2 | W §10 | `ferric-mp2/src/dlpno_mp2.rs::dlpno_mp2_from_b_ov`, `ferric-mp2/src/local_pno.rs::build_pno_transforms` | ready |

### Part VI. Derivatives

**Chapter 14. SCF gradients, Hessians and geometry**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 14.1 | Hartree–Fock nuclear gradient | T scf "Hartree-Fock nuclear gradient"†; S `hf_gradient_sympy.py`†; R hf-gradient | `ferric-scf/src/gradient.rs::{rhf_gradient,uhf_gradient,rohf_gradient,build_energy_weighted_density}` | ready |
| 14.2 | Gradients of density-fitted J and K | T scf "Density-fitted J/K nuclear gradient"; S `df_jk_gradient_sympy.py`; R hf-gradient | `ferric-scf/src/df_gradient.rs::df_jk_gradient` | edit |
| 14.3 | XC gradient: AO-derivative and grid-response terms | T dft "XC nuclear gradient"; S `dft_mgga_tau_gradient_sympy.py`; R ks-gradient | `ferric-dft/src/gradient.rs::{xc_gradient_closed_gga_from_density,xc_gradient_closed_mgga_from_density}`, `ferric-scf/src/ks_gradient.rs::ks_gradient_closed` | ready |
| 14.4 | Hartree–Fock Hessian and coupled-perturbed equations | T scf "Hartree-Fock nuclear Hessian"†; S `hf_hessian_sympy.py`†; R hf-hessian | `ferric-scf/src/hessian.rs::{rhf_hessian,uhf_hessian}` | ready |
| 14.5 | Kohn–Sham Hessian on a fixed and a moving grid | T scf "Kohn-Sham XC Hessian terms"; S `ks_fixed_grid_hessian_sympy.py`, `uks_xc_hessian_sympy.py`; R ks-hessian | `ferric-scf/src/ks_fixed_grid.rs::{skeleton_hessian,first_order_fock,skeleton_hessian_moving}`, `ferric-dft/src/becke_hessian.rs::becke_home_weight_hess` | ready |
| 14.6 | Harmonic frequencies | T scf "Harmonic frequencies"†; S `hf_frequencies_sympy.py`†; R harmonic-frequencies | `ferric-scf/src/frequencies.rs::{frequencies_from_cartesian_hessian,harmonic_frequencies_analytic}` | ready |
| 14.7 | Fock-hook gradients | T scf "Fock hooks and their nuclear gradient" | `ferric-scf/src/gradient.rs::with_fock_hook_gradient` | edit |
| 14.8 | Geometry optimization, saddle points and reaction paths | T scf "Geometry searches"; S `geometry_search_steps_sympy.py`; R geometry-search; M scf "Gradients, optimization, frequencies, transition states" | `ferric-scf/src/optimize.rs::optimize_geometry`, `ferric-scf/src/saddle.rs::find_saddle`, `ferric-scf/src/irc.rs::follow_irc` | edit |

**Chapter 15. Correlated gradients and the Z-vector**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 15.1 | Lagrangians and the Z-vector equation | T mp2 §9; S `mp2_variant_identities.py` claim 5 | `ferric-mp2/src/zvector.rs::{solve_zvector,build_relaxed_density_ao}` | ready |
| 15.2 | RI-MP2 gradient with exact and RI-JK references; SCS-MP2 | T mp2 §9, §9.1, §9.2; S `mp2_gradient_sympy.py`†; R ri-mp2-gradient, scs-mp2-gradient | `ferric-mp2/src/gradient.rs::{rimp2_gradient_analytical,scs_mp2_gradient_analytical,total_rimp2_gradient}` | ready |
| 15.3 | OO-RI-MP2 gradient | T mp2 "9.3 OO-RI-MP2 nuclear gradient"†; R oo-rimp2-gradient | `ferric-mp2/src/oo_rimp2_gradient.rs::oo_ri_mp2_gradient` | ready |
| 15.4 | Kohn–Sham Z-vector for post-SCF corrections | none | `ferric-scf/src/zvector_ks.rs::relaxation_gradient_closed` | ready |
| 15.5 | RI-dRPA gradient | T rpa "Analytic nuclear gradient"; S `rpa_gradient_sympy.py`†; R rpa-gradient | `ferric-rpa/src/gradient_analytic.rs::{rpa_analytic_gradient,rpa_ri_densities}` | ready |
| 15.6 | MP2 static polarizability | T mp2 §10; R mp2-polarizability *gap* | `ferric-mp2/src/cpks_polar.rs::{mp2_polarizability_analytic,static_relaxed_density_ao}`, `ferric-mp2/src/ff_polar.rs::mp2_polarizability_static` | ready |
| 15.7 | Unrestricted RI-MP2 gradient | T mp2 §15; S `u_rimp2_gradient_sympy.py`; R u-ri-mp2-gradient | `ferric-mp2/src/u_rimp2_gradient.rs::{u_rimp2_gradient_analytical,u_energy_weights,solve_u_zvector}` | ready |

### Part VII. Coupled cluster

**Chapter 16. Coupled cluster**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 16.1 | CCD and CCSD in spin orbitals | T cc "CCD", "CCSD"; W §5; R ccd *gap*, ccsd *gap* | `ferric-cc/src/ccd.rs::ccd`, `ferric-cc/src/ccsd.rs::ccsd` | ready |
| 16.2 | Closed-shell CCSD and the particle-particle ladder | T cc "CCSD"; M cc "Particle-particle ladder"; R ccsd *gap* | `ferric-cc/src/ccsd_closed_shell.rs::ccsd_closed_shell`, `ferric-cc/src/ri_ladder.rs::ri_ladder_contract` | ready |
| 16.3 | (T) in spin orbitals and in closed-shell form | T cc "CCSD(T), spin-orbital", "CCSD(T), closed shell"; S `cc_triples_weights_and_linlccd_spin_adaptation_sympy.py`; R ccsd-t *gap* (the closed-shell text describes a removed kernel) | `ferric-cc/src/ccsd_t.rs::ccsd_t`, `ferric-cc/src/ccsd_t_closed_shell.rs::ccsd_t_closed_shell` | ready |
| 16.4 | EOM-CCSD singlets | T cc "EE-EOM-CCSD singlets"; R eom-ccsd *gap* | `ferric-cc/src/eom_ccsd.rs::eom_ccsd_singlets`, `ferric-cc/src/eom_davidson.rs::davidson_nonsymmetric` | ready |
| 16.5 | LinLCCD and LinLCCD(hh) | T cc "LinLCCD and LinLCCD(hh)"; S `cc_triples_weights_and_linlccd_spin_adaptation_sympy.py`; R linlccd; W §9 | `ferric-cc/src/linlccd.rs::linlccd`, `ferric-cc/src/linlccd_u.rs::u_linlccd`, `ferric-cc/src/helpers.rs::contract_hh_ladder` | ready |
| 16.6 | The ωB97X-L-V double hybrid | T cc "ωB97X-L-V correlation"; R double-hybrid *gap* | `ferric-cc/src/double_hybrid.rs::run_wb97x_l_v` | ready |
| 16.7 | Local coupled cluster: amplitude threshold and DLPNO | T cc "Amplitude-threshold LinLCCD", "DLPNO-…" (three sections); W §10, §13 | `ferric-cc/src/linlccd_amplitude.rs::amplitude_linlccd`, `ferric-cc/src/dlpno_linlccd.rs::dlpno_linlccd_hh`, `ferric-cc/src/dlpno_ccsd_virtual.rs::pno_ccsd_energy`, `ferric-cc/src/dlpno_ccsd_t.rs::screened_triple_energy` | edit |
| 16.8 | Unrestricted CCSD and CCSD(T) | T cc "Unrestricted CCSD and CCSD(T)"; S `cc_uccsd_spin_blocks_sympy.py`; R u-ccsd, u-ccsd-t | `ferric-cc/src/u_ccsd.rs::u_ccsd`, `ferric-cc/src/u_ccsd_t.rs::u_ccsd_t` | ready |

### Part VIII. Multireference

**Chapter 17. CAS-CI, CASSCF and NEVPT2**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 17.1 | Determinants, strings and the phase convention | T ci "Determinants and the phase convention", "Hamiltonian" | `ferric-ci/src/strings.rs::{enumerate_strings,phase_below}` | ready |
| 17.2 | Slater–Condon rules as implemented | T ci "Slater–Condon rules as implemented"; S `casci_slater_condon_phases_sympy.py`; R casci | `ferric-ci/src/hamiltonian.rs::{general_element,diagonal_element}` | ready |
| 17.3 | Direct sigma and the Davidson solver | T ci "Direct sigma" | `ferric-ci/src/sigma.rs::DirectSigma`, `ferric-ci/src/davidson.rs::davidson_lowest`, `ferric-ci/src/lib.rs::run_cas_ci` | ready |
| 17.4 | Reduced density matrices | T ci "Reduced density matrices", "NEVPT RDMs" | `ferric-ci/src/rdm.rs::{spin_summed_rdms,rdm_energy}`, `ferric-ci/src/rdm3.rs::nevpt_rdms` | ready |
| 17.5 | Two-step CASSCF | T ci "Two-step CASSCF" (835 lines, mostly measurement records to be cut); R casscf *gap* | `ferric-ci/src/casscf.rs::{run_casscf,optimize,expm_antisymmetric}`, `ferric-ci/src/casscf_newton.rs::{orbital_hessian,ah_davidson}`, `ferric-ci/src/casscf_stability.rs::stability` | ready |
| 17.6 | SC-NEVPT2 | T ci "SC-NEVPT2"; R nevpt2 *gap* | `ferric-ci/src/nevpt2.rs::sc_nevpt2` | ready |

### Part IX. RPA, PDEP and GW

**Chapter 18. The random phase approximation**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 18.1 | dRPA three ways: frequency integral, plasmon formula, ring amplitudes | T rpa "RPA Correlation Energy" (extended†); S `rpa_drpa_pdep_sympy.py`†; R drpa; P derivation §7–9 | `ferric-rpa/src/energy.rs::rpa_correlation_energy`, `ferric-rpa/src/diagnostics.rs::ri_drpa_energy`, `ferric-mp2/src/drpa_amplitude.rs::canonical_plasmon_drpa` | ready |
| 18.2 | Imaginary-frequency quadratures | T quadrature §2 | `ferric-rpa/src/quadrature.rs::{build_quadrature,chebyshev_tan_nodes,gauss_legendre_nodes}` | ready |
| 18.3 | Projective dielectric eigenpotentials | T rpa "The PDEP Eigenpotential Approach"; S `rpa_drpa_pdep_sympy.py`†; R pdep-rpa; W §7 | `ferric-rpa/src/lib.rs::run_pdep_rpa`, `ferric-rpa/src/sternheimer.rs::{dielectric_matrix,dielectric_apply}`, `ferric-rpa/src/lanczos.rs::run_lanczos_seeded` | ready |
| 18.4 | Open-shell dielectric | T rpa "Open-Shell (Spin-Summed) Dielectric"; R u-pdep-rpa | `ferric-rpa/src/lib.rs::run_u_pdep_rpa`, `ferric-rpa/src/sternheimer.rs::dielectric_matrix_unrestricted` | ready |
| 18.5 | Imaginary-time and AO forms of $\chi_0$ | T quadrature §1 | `ferric-rpa/src/laplace_chi0.rs::dielectric_matrix_laplace`, `ferric-rpa/src/ao_rpa.rs::ao_rpa_correlation_energy` | ready |
| 18.6 | Ring amplitudes with a threshold: local dRPA | W §12; M rpa-gw "Exact and local dRPA" | `ferric-mp2/src/drpa_amplitude.rs::{amplitude_drpa,amplitude_drpa_direct}`, `ferric-mp2/src/rccd_family.rs::{rccd,sosex}` | verify |

**Chapter 19. GW and the Bethe–Salpeter equation**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 19.1 | RI factors and the exchange self-energy | T gw §1 | `ferric-gw/src/mo_b.rs::build_full_b`, `ferric-gw/src/cohsex.rs::sigma_x_diag` | ready |
| 19.2 | The screened interaction on the imaginary axis; PDEP as $W$ | T gw §2, §3, §10; S `gw_self_energy_sympy.py` | `ferric-gw/src/w_pdep.rs::static_weights`, `ferric-gw/src/cohsex.rs::project_b_into_pdep` | ready |
| 19.3 | Correlation self-energy | T gw §4; S `gw_self_energy_sympy.py`; R gw | `ferric-gw/src/sigma.rs::{sigma_c_at_z,run_g0w0}` | ready |
| 19.4 | COHSEX | T gw §5 | `ferric-gw/src/cohsex.rs::run_cohsex` | ready |
| 19.5 | Analytic continuation and the quasiparticle equation | T gw §6, §7 | `ferric-gw/src/pade.rs::{fit,eval}`, `ferric-gw/src/sigma.rs::solve_qp_for_mo` | ready |
| 19.6 | Eigenvalue self-consistency | T gw §8 | `ferric-gw/src/sigma.rs::{run_evgw0,run_evgw}` | ready |
| 19.7 | Open-shell GW | T gw §9; R u-gw *gap* | `ferric-gw/src/u_sigma.rs::run_u_g0w0`, `ferric-gw/src/u_cohsex.rs::run_u_cohsex` | ready |
| 19.8 | BSE in the Tamm–Dancoff approximation | T gw §11; R bse-tda *gap* | `ferric-gw/src/bse.rs::run_bse_tda` | ready |
| 19.9 | Statically screened TDHF polarizability | T gw §12; R tdhf-static-polarizability *gap* | `ferric-gw/src/bse.rs::run_rpax_static_polarizability` | ready |

### Part X. Response and excited states

**Chapter 20. TDDFT**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 20.1 | Response matrices, TDA and the Casida equations | T tddft §1–3, §6; R tddft *gap* (text exists, no proof; a script is in progress on `docs/derivations-backfill`) | `ferric-tddft/src/tddft.rs::run_tddft`, `ferric-tddft/src/davidson.rs::{davidson_tda,davidson_casida}` | ready |
| 20.2 | Transition dipoles and oscillator strengths | T tddft §4 | `ferric-tddft/src/tddft.rs::run_tddft` | ready |
| 20.3 | The kernel and the functional gate | T tddft §5; T dft "$f_{xc}$ response kernels" | `ferric-dft/src/lr_kernel.rs::resolve_response_xc` | ready |
| 20.4 | Range-separated hybrids | T tddft §8 | `ferric-scf/src/rsh_response.rs::exchange_response` | ready |
| 20.5 | Triplets | T tddft §9 | `ferric-scf/src/rhf_newton.rs::triplet_hessian_matvec` | ready |
| 20.6 | Unrestricted references and $\langle S^2\rangle$ | T tddft §10, §11 | `ferric-tddft/src/tddft.rs::run_tddft` | ready |

**Chapter 21. Polarizabilities, charges and export**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 21.1 | Static and dynamic polarizabilities from the dielectric modes | T rpa "Dispersion" (first part); S `rpa_c6_polarizability_sympy.py`†; R rpa-polarizability-c6 | `ferric-rpa/src/properties.rs::{pdep_polarizability_static,molecular_dynamic_polarizability}`, `ferric-rpa/src/response_modes.rs::polarizability_from_modes` | ready |
| 21.2 | Atomic partitions: Becke, Hirshfeld, Löwdin, ESP fits | T scf "Atomic charges"; S `atomic_charges_sympy.py`; R atomic-charges | `ferric-scf/src/properties.rs::{lowdin_charges,becke_charges,chelpg_charges,resp_charges,esp_at_atoms}` | edit |
| 21.3 | Export schemas | T export §1–2 | `ferric-export/src/ml.rs::export_npz`, `ferric-export/src/cube.rs::export_cube` | edit |

### Part XI. Environment

**Chapter 22. Implicit solvation**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 22.1 | Cavity and boundary-element matrices | T pcm §1–2 | `ferric-pcm/src/cavity.rs::build_cavity`, `ferric-pcm/src/matrices.rs::{build_s_d,build_k_r}` | ready |
| 22.2 | The IEF-PCM operators, the symmetrized solve and the sphere limits | T pcm §3–5; S `pcm_ief_sympy.py`; R pcm | `ferric-pcm/src/solver.rs::solve_pcm_charges` | ready |
| 22.3 | The reaction field in the Fock matrix | T pcm §7 | `ferric-pcm/src/potential.rs::{solute_potential_at_tesserae,build_reaction_field_operator}` | ready |
| 22.4 | Nuclear gradient | T pcm §8; S `pcm_ief_sympy.py`; R pcm-gradient | `ferric-pcm/src/gradient.rs::pcm_gradient` | ready |
| 22.5 | The conductor limit and COSMO | T pcm §6, T scf "COSMO reaction field"; S `cosmo_sympy.py`; R cosmo | `ferric-scf/src/cosmo.rs::{cosmo_reaction_field,f_epsilon}` | edit |

**Chapter 23. Embedding and QM/MM**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 23.1 | Point charges, smeared charges and uniform fields | T scf "External point charges and fields"; S `external_potential_sympy.py` | `ferric-core/src/external_potential.rs::ExternalPotential`, `ferric-integrals/src/oneelectron.rs::hcore_with_external` | edit |
| 23.2 | QM/MM boundaries, link atoms and the full gradient | T scf "QM/MM boundary and the full gradient"; S `qmmm_boundary_sympy.py`; M `site/src/using/qmmm.md`; R qmmm | `ferric-scf/src/qmmm.rs::{mm_forces,full_gradient,optimize_qmmm}` | edit |
| 23.3 | Polarizable embedding | T scf "Thole polarizable embedding"; S `thole_polarizable_sympy.py`; R qmmm | `ferric-scf/src/polarizable.rs::{induce,site_gradient}` | edit |
| 23.4 | The molecular-mechanics force field | T mm §1–8; S `mm_dihedral_gradient_sympy.py`, `mm_cutoff_sympy.py`; R mm | `ferric-mm/src/energy.rs::{energy,gradient}`, `ferric-mm/src/cutoff.rs::{energy_with,gradient_with}` | ready |

### Part XII. Periodic systems

**Chapter 24. Gaussian-basis periodic calculations**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 24.1 | Lattices, Ewald summation and the Madelung constant | T pbc §1; S `pbc_ewald_pairft_stress_sympy.py`; R pbc-ewald-pairft-stress | `ferric-pbc/src/ewald.rs::{ewald_nuclear_repulsion,madelung_constant}` | ready |
| 24.2 | Fourier transform of lattice-summed Gaussian pairs | T pbc §2 | `ferric-pbc/src/pair_ft.rs::pair_ft` | ready |
| 24.3 | Ewald-split nuclear attraction | T pbc §3 | `ferric-pbc/src/hcore.rs::periodic_hcore` | ready |
| 24.4 | The exchange divergence | T pbc §4; R pbc-hf *gap* | `ferric-pbc/src/kpts.rs::madelung` | ready |
| 24.5 | Range-separated Gaussian density fitting for J and K | none; R pbc-hf *gap* | `ferric-pbc/src/rsgdf.rs::{j_builder,k_builder}` | ready |
| 24.6 | k-point Hartree–Fock and Kohn–Sham | none; R pbc-hf *gap*, pbc-ks *gap* | `ferric-pbc/src/kpts.rs::monkhorst_pack`, `ferric-pbc/src/kscf.rs::solve_krhf`, `ferric-pbc/src/kuscf.rs::solve_kuhf`, `ferric-pbc/src/kdft.rs::solve_krks` | edit |
| 24.7 | Forces | T pbc §1 "Gradient", §6; R pbc-ks *gap* | `ferric-pbc/src/grad.rs::gamma_rhf_gradient`, `ferric-pbc/src/kgrad.rs::kpoint_rhf_gradient` | edit |
| 24.8 | Stress | T pbc §5; S `pbc_ewald_pairft_stress_sympy.py` | `ferric-pbc/src/stress.rs::gamma_rhf_stress`, `ferric-pbc/src/ewald.rs::ewald_nuclear_strain`, `ferric-pbc/src/pair_ft.rs::pair_ft_strain_chunked` | ready |
| 24.9 | Periodic MP2 and dRPA | none; R pbc-mp2 *gap*, pbc-drpa *gap* | `ferric-pbc/src/mp2.rs::gamma_mp2`, `ferric-pbc/src/drpa.rs::gamma_drpa`, `ferric-pbc/src/kcorr.rs::{kpoint_mp2,kpoint_drpa}` | ready |

### Part XIII. Dispersion

**Chapter 25. Dispersion**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 25.1 | D3(BJ): energy, coordination numbers, gradient, three-body term | T d3 (all); S `d3_cn_chain_rule_gradient_sympy.py`, `d3_atm_gradient_sympy.py`; R d3 | `ferric-d3/src/lib.rs::{d3bj_energy,d3bj_gradient}` | ready |
| 25.2 | Casimir–Polder $C_6$ from dynamic polarizabilities | T rpa "Dispersion: Casimir-Polder C6"; S `rpa_c6_polarizability_sympy.py`†; D 1.4.3 Eqs. 1.49–1.51 | `ferric-rpa/src/dispersion.rs::{casimir_polder_c6,ts_dynamic_polarizability,pdep_dynamic_polarizability}` | ready |
| 25.3 | Many-body dispersion | M scf "Dispersion correction: MBD@rsSCS"; R mbd *gap* | `ferric-rpa/src/dispersion/mbd.rs::{mbd_screen,mbd_dynamic_polarizability,dipole_coupling_tensor}` | edit |

### Part XIV. Capstone

**Chapter 26. Short-range MP2 with long-range RPA**

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 26.1 | Second order in a split interaction: the cross term again | §10.7; P derivation §5–6; P paper "Second order and the cross term" | `ferric-mp2/src/attenuated.rs::rs_mp2_decomposition` | edit |
| 26.2 | The ring series by three routes | §18.1; P derivation §7–9 | `ferric-rpa/src/energy.rs::rpa_correlation_energy` | gap |
| 26.3 | Coloring the rings: which diagrams each scheme keeps | P derivation §10–16; P paper "range-coloring classification", Appendix A | none | gap |
| 26.4 | The two formulations in the code | T rpa "Range-separated MP2 + RPA"†; S `rpa_rs_mp2_rpa_sympy.py`†; R rs-mp2-rpa | `ferric-rpa/src/rs_mp2_rpa.rs::{rs_mp2_lr_rpa,RsMp2RpaFormulation}` | ready |
| 26.5 | The $\omega$-scale law | P derivation §17; P paper "The $\omega$-scale law" | none | gap |
| 26.6 | Shared intermediates across the two kernels | P derivation §19; code comments | `ferric-mp2/src/rimp2.rs::{compute_rpa_intermediates,spin_components_from_b_ov}`, `ferric-rpa/src/rs_mp2_rpa.rs::second_intermediate_bov_bytes` | edit |
| 26.7 | What was measured, and what does not work | P paper "Results", P derivation §18, §20; D 7.2 | none | gap |

### Part XV. Interfaces

**Chapter 27. ML integration touchpoints**

For an implementer wiring ferric outputs into ML models. Chapter 21 derives the
properties; this chapter states what each exported array is, its frame behaviour,
its evidence and its use as a feature or label.

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 27.1 | The NPZ feature bundle | T export §1; R pdep-rpa; `site/src/reference/validation.md` NPZ rows | `ferric-export/src/ml.rs::export_npz`, `ferric-rpa/src/property_bundle.rs::{compute_pdep_properties,write_pdep_npz}` | edit |
| 27.2 | Electrostatic features and atomic charges | none (no theory text); validation anchors | `ferric-scf/src/properties.rs::{esp_at_atoms,electric_field_at_atoms,esp_at_points,lowdin_charges,mulliken_charges,chelpg_charges,resp_charges}`, `ferric-scf/src/properties/hirshfeld_grid.rs::hirshfeld_charges` | gap |
| 27.3 | Polarizabilities, PDEP modes and $C_6$ | T rpa "Dispersion: Casimir-Polder C6"; S `rpa_c6_polarizability_sympy.py` claims 1, 2, 3, 5; R rpa-polarizability-c6 | `ferric-rpa/src/properties/per_atom.rs::{closed_per_atom_alpha,charge_transfer_remainder}`, `ferric-rpa/src/dispersion.rs::{casimir_polder_c6,ts_dynamic_polarizability,pdep_dynamic_polarizability}` | ready |
| 27.4 | The Python surface, cube files and provenance | T export §2; `ferric.pyi`; M `site/src/using/{ase,qcschema,export}.md` | `ferric-export/src/cube.rs::export_cube`, `ferric-core/src/certificate.rs::Certificate` | edit |
| 27.5 | Reproducibility and validation grades | `site/src/reference/{architecture,validation,input}.md`; `site/src/using/sharp-bits.md` | `ferric-core/src/blas_threads.rs::{with_blas_threads,opt_in_blas_threads}` | edit |
| 27.6 | Energies, gradients, Hessians and response as labels | validation matrix and anchors; R hf-gradient, ks-gradient, ri-mp2-gradient, rpa-gradient, hf-hessian, ks-hessian, harmonic-frequencies | `ferric-rpa/src/gradient.rs::total_rpa_gradient`, `ferric-scf/src/noise_floor.rs::de_noise_floor` | edit |
| 27.7 | A worked consumer: esp-conditioning | the esp-conditioning builder code (interface only) | none | edit |

### Appendices

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| A | Index of verified equations and proof scripts (generated) | registry; box metadata | none | ready |
| B | Index of functions cited (generated) | code-reference metadata | none | ready |
| C | Validation grades (generated from the validation page) | `site/src/reference/validation.md` | `ferric-core/src/certificate/anchors.rs::EnergyAnchor` | ready |
| D | The semiempirical interface (xTB) | R xtb *gap* | `ferric-xtb/src/calculator.rs::XtbCalculator` | gap |

## 4. Coverage

Counts of sections by status are in the final table of this section. The
statuses in the tables above follow each section file's header comment, and
`scripts/book_build.py --write-plan` rewrites both the statuses and the final
table (between the `status-table` markers) from those headers; `--check`
fails when they differ.

Registry methods and where they land:

| Registry state | Count | Book status |
|---|---|---|
| Proven (`methods`) | 39 | `ready`, except attenuated-mp2 and mp2-v, which are `edit` because the dissertation narrative has to be merged |
| Gap (`gaps`) | 25 | `gap`, `verify` or `edit`, by what text already exists |

Every registry method id appears in a Sources cell above. The book adds
sections the registry does not track because they are not methods: integrals,
screening, Fock builders, convergence, memory and determinism. Those have no
registry row and are the largest block of `gap` sections.

<!-- status-table:begin -->
| Part | ready | edit | gap | verify | Sections |
|---|---|---|---|---|---|
| 0. Front matter | 1 | 1 | 0 | 0 | 2 |
| I. Foundations | 0 | 7 | 0 | 0 | 7 |
| II. Integrals, screening and density fitting | 1 | 15 | 1 | 0 | 17 |
| III. Self-consistent field | 8 | 2 | 0 | 0 | 10 |
| IV. Density functional theory | 7 | 0 | 0 | 0 | 7 |
| V. The MP2 family | 16 | 5 | 1 | 5 | 27 |
| VI. Derivatives | 12 | 3 | 0 | 0 | 15 |
| VII. Coupled cluster | 7 | 1 | 0 | 0 | 8 |
| VIII. Multireference | 6 | 0 | 0 | 0 | 6 |
| IX. RPA, PDEP and GW | 14 | 0 | 0 | 1 | 15 |
| X. Response and excited states | 7 | 2 | 0 | 0 | 9 |
| XI. Environment | 5 | 4 | 0 | 0 | 9 |
| XII. Periodic systems | 7 | 2 | 0 | 0 | 9 |
| XIII. Dispersion | 2 | 1 | 0 | 0 | 3 |
| XIV. Capstone | 1 | 2 | 4 | 0 | 7 |
| XV. Interfaces | 1 | 5 | 1 | 0 | 7 |
| Appendices | 3 | 0 | 1 | 0 | 4 |
| **Total** | **98** | **50** | **8** | **6** | **162** |

162 sections: 60% `ready`, 31% `edit`, 5% `gap` and 4% `verify`.
<!-- status-table:end -->

## 5. Gaps, ranked by how much rests on them

Rank reflects how many other chapters or user-facing results depend on the
missing text or proof.

| Rank | Gap | Sections | Why it ranks here | What closes it |
|---|---|---|---|---|
| 1 | Fock builders: direct and incremental J/K, RI-J, RI-K, Split-RI-J, LinK, COSX, CFMM | 7.3–7.5 | Every calculation in every later chapter runs through them. The only committed text is 10 lines on CFMM. | Derivations in `ferric-scf/theory.md`, a proof script for the RI-J and RI-K algebra and the COSX quadrature identity, then the sections. |
| 2 | Integrals, operators and screening | 3.1–3.7, 4.2–4.3, 5.1–5.3 | `ferric-integrals/theory.md` is 66 lines. The attenuated operators, their table engine and the bounds are what the MP2 and capstone chapters stand on. | A full `ferric-integrals/theory.md`: kernels, the terf tables and their fallback, Schwarz and QQR bounds with proofs of the bound inequalities. |
| 3 | SCF convergence: DIIS variants, second-order steps, stability | 8.1–8.4 | Decides whether a result exists at all. 6 lines of committed text. | Derivation of the orbital Hessian-vector products for RHF, UHF and ROHF, checked against finite differences of the gradient. |
| 4 | CCD and CCSD working equations | 16.1–16.2 | Graded, user-facing methods with no committed derivation and no proof (registry gaps `ccd`, `ccsd`). | Spin-orbital residuals by `sympy.physics.secondquant`, then the spin-adapted form checked against them. |
| 5 | Closed-shell (T) text describes a removed kernel | 16.3 | The committed section is wrong about the current code (registry gap `ccsd-t`). | Rewrite the section from the current `triple_energy` kernel; the script's claim 4b already ports it. |
| 6 | Hybrid and range-separated exchange in Kohn–Sham | 9.4 | The default functionals for most users are hybrids; registry gap `ks-hybrid-rsh`. In progress on `docs/derivations-backfill`. | Land the derivation and its script. |
| 7 | TDDFT has text and no proof | 20.1–20.6 | 441 lines of derivation with no machine check (registry gap `tddft`). In progress on `docs/derivations-backfill`. | Land the script; the sections then move to `ready`. |
| 8 | The capstone rests on unpublished notes | 26.2, 26.3, 26.5, 26.7 | The book's final chapter. Only §26.4 is committed and proven. | Move the range-coloring classification and the $\omega$-scale law into `ferric-rpa/theory.md` with proof scripts (the paper's Appendix A already describes a symbolic check). |
| 9 | CASSCF and NEVPT2 have no proof | 17.5–17.6 | Graded methods; registry gaps `casscf`, `nevpt2`. The CASSCF text is long and is mostly measurement records. | A proof of the generalized-Fock gradient and of the NEVPT2 class energies on a small active space; cut the section to the derivation. |
| 10 | Periodic Hartree–Fock, RS-GDF and periodic correlation | 24.5, 24.6, 24.9 | Four registry gaps (`pbc-hf`, `pbc-ks`, `pbc-mp2`, `pbc-drpa`). The Ewald, pair-FT and stress pieces are proven; the energy functional and the fitting are not. | Derivation of the k-point energy functional, the exchange-divergence correction and the RS-GDF split. |
| 11 | Local correlation | 13.1–13.3, 16.7, 18.6 | Five sections whose only derivations are working notes. | Re-derive from the code; one proof per method of the zero-threshold limit. |
| 12 | Embedding: external potentials, QM/MM, polarizable sites, cDFT, COSMO | 9.7, 22.5, 23.1–23.3 | Registry gaps `qmmm`, `cdft`, `cosmo`. Validated numerically, not derived. | Derivations of the link-atom gradient projection and of the stationary polarization functional. |
| 13 | Density-fitted gradients and the Kohn–Sham Z-vector | 14.2, 15.4 | Used by every optimization with fitted J/K. | Derivation plus a finite-difference check of the three-index and two-index terms. |
| 14 | BSE-TDA, the statically screened TDHF polarizability and open-shell GW | 19.7–19.9 | Three registry gaps. | Proofs of the per-spin dielectric and of the BSE kernel. |
| 15 | Unverified statements inside otherwise ready sections | 10.1 (Eq. 10.5), 12.2, 15.6 | Small, but each is an equation the text asserts without a check. | One claim per statement in an existing script. |
| 16 | Many-body dispersion | 25.3 | Registry gap `mbd`. | Derivation of the range-separated self-consistent screening and its gradient. |
| 17 | RI-MP2 at scale: memory, device kernels, distributed bands | 11.2, 11.4, 11.6, 11.7 | Algorithmic text with no source besides code comments. | Written from the code; no proof needed beyond the existing bit-identity tests. |
| 18 | Notation defects in committed files | `book/notation.md` §10 | They mislead a reader who cross-checks the book against `theory.md`. | Five small fixes, owned by the derivation backfill. |

## 6. Effort by part

A work stream ("lane") here is one agent taking a chapter from sources to a
committed draft that passes the book checker, including any proof claims the
chapter needs. "Proof work" counts the scripts or claims that must exist
before the text can be `ready`; it belongs with the derivation backfill, not
with the book lanes.

| Part | Chapters | Book lanes | Proof work first | Can run in parallel with | Needs Matt |
|---|---|---|---|---|---|
| 0, I | 0–2 | 1 | none | everything | Confirm the notation choices (section 8, question 2). |
| II | 3–6 | 3 | 3 scripts (bounds, terf identities, robust fit) | III, IV | No. |
| III | 7–8 | 3 | 3 scripts (RI-J/RI-K algebra, orbital Hessians, DIIS error vector) | II, IV | No. |
| IV | 9 | 1 | 1 script, in progress | II, III | No. |
| V | 10–13 | 4 | 2 claims; local MP2 re-derivation | VI onward | Yes: how much of dissertation chapters 2, 3, 5 and 6 to carry, and whether figures and tables may be reproduced (questions 3 and 4). |
| VI | 14–15 | 2 | 2 scripts (DF gradient, KS Z-vector) | V, VII | No. |
| VII | 16 | 2 | 2 scripts (CCSD residuals, EOM) | VI, VIII | No. |
| VIII | 17 | 1 | 2 scripts (CASSCF gradient, NEVPT2 classes) | VII, IX | Whether the long CASSCF measurement record stays in `theory.md` or moves out (question 6). |
| IX | 18–19 | 2 | 2 scripts (BSE, per-spin GW) | VIII, X | No. |
| X | 20–21 | 1 | 1 script, in progress | IX | No. |
| XI | 22–23 | 2 | 2 scripts (QM/MM gradient, polarization functional) | XII | No. |
| XII | 24 | 2 | 2 scripts (k-point functional, RS-GDF) | XI | No. |
| XIII | 25 | 1 | 1 script (MBD) | any | No. |
| XIV | 26 | 2 | 2 scripts (range coloring, $\omega$-scale law) | after V and IX | Yes: the chapter is Matt's unpublished work (question 5). |
| Build | tooling | 1 | none | everything | Format choice (question 7). |

Total: about 28 book lanes and about 25 proof scripts or claims. The
dependency order is 0 → I → II → III → (IV, V) → (VI, VII, VIII, IX) →
(X, XI, XII, XIII) → XIV. Parts II and III are the critical path because they
have the least committed text and the most dependents.

A first release that is useful on its own: Parts 0, I, V (chapters 10–12),
VI (chapter 15) and IX (chapter 18), plus §6.1 and §4.1. Those sections are
`ready` or `edit` today and lead directly to the capstone.

## 7. Working with the derivation backfill

- The registry is the coverage ledger. A book section lists the registry ids
  it covers. The book checker fails when a section marked `ready` covers an
  id that the registry lists under `gaps`, and when a registry id is covered
  by no section.
- The book does not edit `theory.md` or `scripts/`. A defect found while
  writing a section is filed against the backfill (the first five are in
  `book/notation.md`, section 10).
- When the backfill merges, three things change here: the † marks drop, the
  section numbers of `ferric-mp2/theory.md` shift (its §6.1 does not exist on
  that branch and §13–§14 do not exist on this one), and the 19 scripts
  without a runner gain one. The book cites `theory.md` sections by heading
  text, not by number, for that reason.

## 8. Questions only Matt can answer

1. **One text or two.** Should the book include derivation sections from
   `theory.md` (section 2), or own its text and let `theory.md` shrink to a
   function map? The plan assumes the first. It costs a notation pass over the
   17 files and keeps one home per equation.
2. **Notation.** `book/notation.md` picks a positive denominator
   $\Delta_{ij}^{ab}$, overbars for $\beta$ orbitals, $u$ for imaginary
   frequency so that $\omega$ stays the range-separation parameter, and
   $\xi, \eta$ for PDEP modes. Each choice changes existing `theory.md` text.
   Accept, or name the ones to reverse.
3. **How much dissertation.** Chapters 2, 3, 5 and 6 are mostly training and
   transferability results. The plan carries the method definitions, the
   parameters and one summary table per method (§10.4–10.6), and leaves the
   benchmark tables in the dissertation. Is that the right cut for an
   implementer's reference?
4. **Reuse rights.** Dissertation chapters 2–6 were published as journal
   articles. May their figures and tables be reproduced in a public book, or
   should the book redraw from ferric's own runs?
5. **The capstone and the paper.** Chapter 26 draws on an unpublished
   manuscript. Should the chapter wait for the paper, appear with only the
   committed formulations (§26.4), or carry the full classification?
6. **Negative results and measurement records.** `theory.md` files and the
   wiki hold withdrawn and negative results (local RPA, the AO-Laplace
   locality history, the CASSCF measurement log). Does the book state only
   what the code does today, or does it keep a short "what does not work"
   section per chapter, as §26.7 does?
7. **Format and audience of the output.** HTML next to the user guide, a PDF
   with a version number, or both (`book/BUILD.md` recommends both from one
   source)? Is the book public from the first chapter, or after a complete
   first release?
8. **Title, authorship and license.** The dissertation text is Matt's; the
   repository is dual MIT/Apache. Does the book text take the same license?
9. **Scope edges.** xTB (an interface to an external library) and the
   Avogadro, ASE and QCSchema integrations are left to the user guide, with
   xTB as a one-page appendix. Confirm.
