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
| `crates/*/theory.md` | 17 generated files | Indexes of the book pages that cite each crate's code, written by `scripts/book_build.py --write-theory-index`. They carry no derivations; the book does. | Generated; never edited by hand. |
| `scripts/*_sympy.py`, `scripts/mp2_variant_identities.py` | 21 scripts at this commit | The machine checks behind verified-equation boxes. | High. Only two of the 21 are run by a test at this commit (`oomp2_mp3_identities_sympy.py`, `ump3_spin_blocks_sympy.py`); the runner for the rest is on `docs/derivations-backfill`. |
| `testdata/derivations/registry.json` (branch `docs/derivations-backfill`) | 108 routes, 39 proven methods, 25 recorded gaps | The source of truth for coverage. Section statuses below are derived from it wherever a method id exists. | High. It is checked by a test and can only shrink its gap list. |
| `site/src` mdBook | methods pages 3,162 lines | User-facing "what runs and how to run it", measured numbers, validation grades. The book links to it and does not repeat it. | High for capabilities and numbers. |
| Dissertation (`dissertation.txt`, local only) | 8,420 lines | Chapter 4 for the RI-MP2 algorithm. Chapters 2, 3, 5 and 6 for the attenuated and spin-component-scaled methods, their parameters and their training. Chapter 1 for the framing of range separation. | High for the methods it defines. Its notation differs from the code's (`book/notation.md`). |
| `wiki/sympy_derivations.md` (local only) | 1,532 lines, 25 sections | Leads for sections that have no committed derivation yet. | Low. Working notes. Every item used is marked `verify` below and must be re-derived and machine-checked first. |
| `wiki/papers/sr-mp2-lr-rpa-derivation.tex`, `sr-mp2-lr-rpa.tex` (local only) | 735 and 1,387 lines | The capstone chapter's derivation and classification. | Medium. Unpublished. Marked `verify`. |

Snapshot of the registry used here: branch `docs/derivations-backfill` at
`2d955d5a8` plus its uncommitted working tree. That branch forked before
`81486b984`, so it lacks `scripts/ump3_spin_blocks_sympy.py`. Sources that
exist only there are marked †.

## 2. Editorial model

**One home for each equation.** The book is the home of every derivation. A
book chapter is the reading order, the shared notation, the derivation, the
algorithm and the connective text. Each `crates/<crate>/theory.md` is a
generated index of the book pages that cite the crate's code
(`scripts/book_build.py --write-theory-index`), so a method change updates the
book section, and the index follows.

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

Source abbreviations: **S** script under `scripts/` († only on `docs/derivations-backfill`);
**R** registry method id (*gap* when the registry lists it under `gaps`);
**D** dissertation chapter or section; **W** `wiki/sympy_derivations.md`
section; **P** `wiki/papers/` ("derivation" is `sr-mp2-lr-rpa-derivation.tex`
with its sections numbered in file order, "paper" is `sr-mp2-lr-rpa.tex`); **M** `site/src/methods/` page.
Code references are `crate/src/file.rs::item` under `crates/`.

### Part 0. Front matter

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `how-to-read` | How to read this book: boxes, code references, grades | `book/BUILD.md`; `site/src/reference/validation.md` | none | edit |
| `notation` | Notation and conventions | `book/notation.md` | see that file | ready |

### Part I. Foundations

**Second quantization and the reference state** (chapter `second-quantization`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `hamiltonian` | Hamiltonian, spin orbitals, normal ordering | P derivation §1–3; W §1; D 1.1 | none | edit |
| `wick` | Wick contractions as the proof scripts run them | S `oomp2_mp3_identities_sympy.py`; W §1–2 | none | edit |
| `spin-adaptation` | Spin adaptation: from spin orbitals to spatial orbitals | S `mp2_variant_identities.py` claims 1–2 | `ferric-mp2/src/spinorbital.rs::{spin,spat,asym_oovv,asym_phys}`, `ferric-cc/src/ccsd_closed_shell.rs::expand_amplitudes_to_spin_orbital` | edit |

**Molecules, basis sets and numerical ground rules** (chapter `molecules-and-numerics`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `molecule` | Molecule, charge, multiplicity, units | none | `ferric-core/src/mol.rs::{Molecule,parse_xyz,nelec,nuclear_repulsion,apply_ecp}` | edit |
| `basis-sets` | Contracted Gaussians, normalization, parsers | none | `ferric-core/src/basis.rs::{BasisSet,Shell,load_bse_json,load_g94}`, `ferric-integrals/src/basis_bridge.rs::PreparedBasis` | edit |
| `determinism-and-memory` | Determinism, threads and the memory budget | `site/src/reference/architecture.md`; code comments | `ferric-core/src/blas_threads.rs::{init_threading,with_blas_threads}`, `ferric-core/src/memory.rs::resolve_budget_bytes` | edit |
| `tensors` | Dense tensors and the einsum macro | `site/src/reference/einsum.md` | `ferric-tensors/src/tensor.rs::Tensor` | edit |

### Part II. Integrals, screening and density fitting

**Gaussian integrals** (chapter `integrals`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `one-electron` | One-electron integrals and multipoles | none | `ferric-integrals/src/oneelectron.rs::{overlap,kinetic,nuclear,hcore,dipole,second_moment}` | edit |
| `two-electron` | Two-, three- and four-center integrals through libint2 and the C shim | none | `ferric-integrals/src/engine.rs::{new_2e,compute_quartet,new_3center,compute_eri3,new_2center,compute_eri2}` | edit |
| `integrals-derivatives` | Derivative integrals and buffer sizing | none | `ferric-integrals/src/engine.rs::{new_1e_deriv,compute_1e_deriv_block_n,new_2e_deriv,compute_eri_deriv_quartet,new_3center_deriv}` | edit |
| `md-three-center` | A McMurchie–Davidson engine for three-center one-electron integrals | none | `ferric-integrals/src/md3c1e.rs::{e_table,boys,pair_block,a_matrix_at_point}` | edit |
| `ecp` | Effective core potentials | none | `ferric-integrals/src/ecp.rs::{ecp_matrix_spherical,ecp_matrix_deriv_spherical}`, `ferric-integrals/src/oneelectron.rs::hcore_ecp` | edit |
| `pvp-x2c` | The pVp integral and spin-free X2C-1e | S `pvp_ss_sympy.py`; R pvp-integrals *gap*; M scf "Scalar relativity" | `ferric-integrals/src/pvp.rs::pvp_nuclear`, `ferric-scf/src/x2c.rs::{x2c1e_transform,x2c1e_hcore}` | edit |
| `ao-grid` | Basis functions and their derivatives on a grid | none | `ferric-integrals/src/ao_grid.rs::{eval_basis_on_points,eval_basis_and_grad_on_points,eval_basis_grad_hess_on_points}` | edit |

**Range-separated and attenuated operators** (chapter `operators`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `erf-terf` | erf and erfc; terf and terfc; the curvature constraint | D 2 (introduction), D 4.1 Eqs. 4.1–4.2; S `mp2_family_sympy.py`† part D | `ferric-integrals/src/operator.rs::{Operator,erf,erfc,terf,terfc,terfc_with_omega,terf_with_omega}` | edit |
| `terf-table-engine` | The terf table engine and its far-field fallback | none committed; code comments in `crates/ferric-integrals/shim/shim.cc` | `ferric-integrals/src/engine.rs::{new_terf_libint2,is_table_engine}` | edit |
| `yukawa-geminal` | Yukawa, Slater geminal and fitted geminal kernels | none | `ferric-integrals/src/operator.rs::{yukawa,slater_geminal,stg}`, `ferric-integrals/src/engine.rs::new_2e_geminal` | gap |

**Screening** (chapter `screening`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `schwarz` | Schwarz bounds, and why they do not see attenuation | none | `ferric-integrals/src/schwarz.rs::{schwarz,bound_inflation}`, `ferric-scf/src/screening.rs::{Bound,SchwarzBounds}` | edit |
| `qqr` | Distance-dependent bounds (QQR) for four- and three-index integrals | none | `ferric-scf/src/qqr.rs::QqrBounds`, `ferric-integrals/src/qqr3.rs::{QqrBounds3,estimate3}` | edit |
| `pair-lists` | Significant pair lists | none | `ferric-scf/src/pairs.rs::{SignificantPairs,DensityPairs,intersect_sorted}` | edit |

**Density fitting** (chapter `density-fitting`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `factorization-and-metric` | The factorization, the metric, Cholesky and eigendecomposition factors | D 1.4.1, D 4.1 Eq. 4.5 | `ferric-integrals/src/threeindex.rs::coulomb_metric_2c`, `ferric-mp2/src/rimp2.rs::{cholesky_inverse_sqrt,eigh_inverse_sqrt,metric_inverse_sqrt}` | edit |
| `three-index-sources` | Three-index sources: in core, packed, spilled, recomputed | none; `site/src/reference/input.md` (`jk_storage`) | `ferric-integrals/src/three_index_source.rs::{ThreeIndexSource,build_band_screened,build_dressed,for_each_block}` | edit |
| `robust-fit` | Mixed metrics and robust (Dunlap) fitting | none | `ferric-mp2/src/rimp2.rs::{ri_mp2_robust_attenuated_metric,robust_fit_coulomb_parts}` | ready |
| `cabs` | Complementary auxiliary basis (CABS) | S `mp2_variant_identities.py` claim 7 | `ferric-mp2/src/f12.rs::build_cabs`, `ferric-integrals/src/cabs.rs::cross_overlap` | edit |

### Part III. Self-consistent field

**Hartree–Fock** (chapter `hartree-fock`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `energy-functionals` | RHF, UHF and ROHF energy functionals | S `hf_energy_sympy.py`†; R rhf, uhf, rohf; D 1.1.2 | `ferric-scf/src/rhf.rs::solve_rhf`, `ferric-scf/src/uhf.rs::solve_uhf`, `ferric-scf/src/rohf.rs::solve_rohf` | ready |
| `orthogonalization-and-guesses` | Orthogonalization and starting guesses | none | `ferric-scf/src/rhf.rs::canonical_orthogonalizer`, `ferric-scf/src/guess.rs::{hcore_guess,sad_guess,minao_projection_guess}` | edit |
| `direct-and-incremental` | Direct and incremental J/K builds | none | `ferric-scf/src/rhf.rs::build_jk`, `ferric-scf/src/fock.rs::{JBuilder,KBuilder}`, `ferric-scf/src/direct_jk.rs::build_incremental` | ready |
| `ri-j-ri-k-split-ri-j` | RI-J, RI-K and Split-RI-J | M scf "Choosing how exchange is built"; module docs of `df_j.rs`, `df_k.rs` | `ferric-scf/src/df_j.rs::DfJ`, `ferric-scf/src/df_k.rs::DfK`, `ferric-scf/src/split_rij.rs::SplitRiJ` | ready |
| `link-cosx-cfmm` | LinK, seminumerical exchange (COSX) and CFMM | M scf | `ferric-scf/src/link_k.rs::LinkK`, `ferric-scf/src/cosx_k.rs::CosxK`, `ferric-scf/src/cfmm.rs::CfmmJ` | ready |
| `fock-hooks` | Fock hooks | none | `ferric-scf/src/rhf.rs::{solve_rhf_with_hook,validate_fock_hook}` | edit |

**Convergence** (chapter `convergence`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `diis` | DIIS and its variants | none | `ferric-scf/src/diis.rs::{Diis,DiisFlavor}` | ready |
| `level-shift-mom-smearing` | Level shift, MOM and smearing | none | `ferric-scf/src/mom.rs::mom_reorder`, `ferric-scf/src/smearing.rs::{fermi_dirac,solve_fermi_level}` | ready |
| `second-order` | Second-order steps: Newton, augmented Hessian, trust region | none | `ferric-scf/src/rhf_newton.rs::{rhf_newton_step,hessian_matvec}`, `ferric-scf/src/uhf_newton.rs::uhf_newton_step`, `ferric-scf/src/rohf_newton.rs::rohf_newton_step`, `ferric-scf/src/rohf_ah.rs::rohf_ah_step`, `ferric-scf/src/trah.rs::{solve_trust_region,rhf_trah_step}`, `ferric-scf/src/cayley.rs::cayley_unitary` | ready |
| `stability` | Stability analysis | none | `ferric-scf/src/stability.rs::{rhf_internal_stability,rhf_external_stability,uhf_internal_stability}` | ready |
| `scf-exits` | Exits, the ladder and opt-in solvers | none | `ferric-scf/src/rhf.rs::{scf_converged,veto_stalled_energy_diis,stall_detected}`, `ferric-scf/src/ladder.rs::{solve_rhf_ladder,default_ladder_from,ksdft_ladder}`, `ferric-scf/src/symmetric.rs::solve_rhf_symmetric` | edit |

### Part IV. Density functional theory

**Kohn–Sham** (chapter `kohn-sham`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `ks-equations` | Kohn–Sham equations and functional families | R ks-semilocal | `ferric-scf/src/rhf.rs::ks_vxc_closed`, `ferric-dft/src/ks.rs::KsXc` | ready |
| `grids` | Radial and angular grids, the Becke partition, pruning | S `dft_becke_weight_grad_sympy.py`; R ks-grid | `ferric-dft/src/becke.rs::{becke_weights_all,becke_weights_and_grad}`, `ferric-dft/src/grid.rs::build_atomic_grid`, `ferric-dft/src/prune.rs::angular_orders_for_atom`, `ferric-dft/src/radial.rs::treutler_ahlrichs_m4`, `ferric-quadrature/src/lebedev.rs::lebedev` | ready |
| `vxc` | $V_{xc}$ matrix elements: LDA, GGA, meta-GGA | S `dft_vxc_matrix_elements_sympy.py`; R ks-semilocal, ks-mgga | `ferric-dft/src/vxc.rs::{semilocal_vxc_closed,semilocal_vxc_polarized}`, `ferric-dft/src/density_on_grid.rs::{eval_density_closed,eval_tau_closed}` | ready |
| `hybrids` | Global and range-separated hybrids; tuning $\omega$ | R ks-hybrid-rsh *gap* (a proof script is in progress on `docs/derivations-backfill`); D 1.3.2 | `ferric-scf/src/rsh_response.rs::exchange_response`, `ferric-scf/src/omega_tuning.rs::tune_omega` | ready |
| `vv10` | VV10 nonlocal correlation | S `dft_vv10_potential_sympy.py`; R ks-vv10 | `ferric-dft/src/vv10.rs::{compute_vv10_energy_and_potentials,add_vv10_scratch}` | ready |
| `fxc` | $f_{xc}$ response kernels | S `dft_fxc_kernel_sympy.py` | `ferric-dft/src/fxc.rs::{LdaFxcKernel,GgaFxcKernel,MggaFxcKernel}`, `ferric-dft/src/lr_kernel.rs::{fxc_ov_block,singlet_fxc_ov_block}` | ready |
| `cdft` | Constrained DFT and electron-transfer coupling | M cdft; R cdft *gap* | `ferric-dft/src/cdft.rs::build_weight_matrix`, `ferric-scf/src/cdft_driver.rs::solve_cdft_uhf`, `ferric-scf/src/cdft_coupling.rs::coupling_hab` | ready |

### Part V. The MP2 family

**MP2 and its variants** (chapter `mp2`; dissertation chapters 2, 3, 5 and 6)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `closed-shell-ri-mp2` | Closed-shell RI-MP2 energy | `book/sample-ri-mp2.md`; D 4.1–4.2; S `mp2_variant_identities.py` claim 1; R ri-mp2 | `ferric-mp2/src/rimp2.rs::{ri_mp2,ri_mp2_spin_components,pair_energy,spin_components_from_b_ov_kappa_cpu}` | ready |
| `unrestricted-ri-mp2` | Unrestricted RI-MP2 | S `mp2_variant_identities.py` claim 2; R u-ri-mp2 | `ferric-mp2/src/u_rimp2.rs::{u_ri_mp2,same_spin_pair_kernel,opposite_spin_pair_kernel}` | ready |
| `scs-and-kappa` | Spin-component scaling and $\kappa$-regularization | S `mp2_family_sympy.py`† part C; R scs-mp2 | `ferric-mp2/src/scs.rs::scs_mp2`, `ferric-mp2/src/rimp2.rs::spin_components_from_b_ov_kappa` | ready |
| `attenuated-mp2` | Attenuated MP2 with erfc and terfc | D 2, D 3; R attenuated-mp2 | `ferric-mp2/src/attenuated.rs::{attenuated_ri_mp2,attenuated_ri_mp2_long_range}` | edit |
| `scs-mp2-2terfc` | SCS-MP2(2terfc): two attenuation lengths | D 5.2 Eq. 5.6, D 5.3; R scs-mp2-2terfc | `ferric-mp2/src/scs.rs::scs_mp2_2terfc` | ready |
| `attenuation-basis-limit` | Attenuation and the basis-set limit | D 6 | none | edit |
| `range-split-cross-term` | The range split is not additive: the cross term | S `mp2_variant_identities.py` claim 4 | `ferric-mp2/src/attenuated.rs::rs_mp2_decomposition` | ready |
| `mp2-v` | MP2-V: attenuated MP2 with damped VV10 | S `mp2_family_sympy.py`† part E; R mp2-v | `ferric-mp2/src/att_vv10.rs::{att_mp2_vv10,u_att_mp2_vv10}`, `ferric-dft/src/vv10.rs::compute_vv10_damped_energy_and_potentials` | edit |
| `mp3` | MP3, restricted and spin-blocked unrestricted | S `oomp2_mp3_identities_sympy.py`, `ump3_spin_blocks_sympy.py`; R mp3. W §2b is superseded by the scripts: its ring prefactor was wrong. | `ferric-mp2/src/mp3.rs::{mp3_energy,mp3_energy_from_spin_orbital_blocks}`, `ferric-mp2/src/u_mp3.rs::u_mp3` | ready |
| `double-hybrids` | MP2-based double hybrids | R double-hybrid *gap* (owned by another branch); M cc "MP2-based double hybrids" | `ferric-mp2/src/double_hybrid.rs::mp2_double_hybrid` | ready |
| `opmp3` | Orbital-partitioned Padé MP3 | S `opmp3_identities_sympy.py`; R opmp3 | `ferric-mp2/src/opmp3.rs::{w_channels,pade,op_mp3,sop_mp3,ssd_opmp3}`, `ferric-mp2/src/opmp3_driver.rs::run_opmp3` | ready |

**RI-MP2 algorithms and memory** (chapter `rimp2-algorithms`; dissertation chapter 4)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `rimp2-algorithms` | Chapter overview: the RI-MP2 pipeline and where each piece lives | `book/sample-ri-mp2.md` | none | ready |
| `four-step-algorithm` | The four-step algorithm, 2014 and now | D 4.2 Table 4.1, Functions 1–4; `book/sample-ri-mp2.md` §10.1.3 | `ferric-mp2/src/rimp2.rs::ri_mp2_spin_components` | ready |
| `streaming-transform` | Streaming the MO transform and the dressing | code comments in `rimp2.rs` | `ferric-mp2/src/rimp2.rs::{stream_dressed_mo_band,stream_dressed_mo_band_budgeted,mo_stream_chunk_for}` | ready |
| `pair-kernel` | The pair-energy kernel: batching, symmetry, fixed summation order | code comments in `rimp2.rs`; D 4.2 | `ferric-mp2/src/rimp2.rs::{spin_components_from_b_ov_kappa_cpu,pair_energy}` | ready |
| `memory-budget` | Memory budget, admission checks and the shared pool | code comments; `site/src/reference/input.md` `[memory]` | `ferric-mp2/src/rimp2.rs::{eri3_budget_bytes,mo_side_alloc_bytes,check_mo_side_alloc}` | verify |
| `parallel-performance` | Parallel performance then and now | D 4.3; `site/src/reference/validation.md` | none | verify |
| `device-mixed-precision` | Device and mixed-precision kernels | `site/src/reference/validation.md` "Mixed-precision GPU kernels" | `ferric-mp2/src/rimp2_gpu.rs::try_spin_components_on_device`, `ferric-mp2/src/rimp2.rs::mixed_energy_operator` | ready |
| `distributed-rimp2` | Distributed RI-MP2 by auxiliary bands | none | `ferric-mp2/src/mpi_rimp2.rs::run_mpi_ri_mp2` | verify |

**Orbital optimization and Laplace forms** (chapter `oo-laplace`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `oo-laplace` | Chapter overview: orbital optimization and Laplace forms | none | none | verify |
| `oo-rimp2` | Orbital-optimized RI-MP2 | S `oomp2_mp3_identities_sympy.py`; S `mp2_variant_identities.py` claim 6; R oo-rimp2 | `ferric-mp2/src/oo_rimp2.rs::{oo_ri_mp2,build_oo_mp2_relaxed_density}`, `ferric-mp2/src/orbital_rotation.rs::cayley_rotation` | ready |
| `unrestricted-oo-rimp2` | Unrestricted OO-RI-MP2 | R u-oo-rimp2 *gap* | `ferric-mp2/src/u_oo_rimp2.rs::u_oo_ri_mp2`, `ferric-mp2/src/u_rimp2.rs::compute_u_mp2_orbital_gradient` | verify |
| `laplace-mp2` | Laplace-transform MP2 and SOS-MP2 | S `mp2_variant_identities.py` claim 3; R laplace-mp2, laplace-sos-mp2; W §11 | `ferric-mp2/src/laplace.rs::{laplace_ri_mp2,laplace_sos_mp2,build_pseudo_density_occ,build_pseudo_density_vir}`, `ferric-quadrature/src/minimax.rs::select_minimax_points` | ready |
| `local-laplace-sos` | Local Laplace SOS-MP2 | M mp2 "Laplace formulations" | `ferric-mp2/src/laplace_local.rs::sos_local_energy`, `ferric-mp2/src/boys.rs::boys_localize` | gap |

**Local MP2** (chapter `local-mp2`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `amplitude-lmp2` | Amplitude-threshold local MP2 | S `lmp2_hylleraas_vs_wshg23_sympy.py`; W §14; R local-mp2 *gap* | `ferric-mp2/src/lmp2_amplitude.rs::{amplitude_lmp2,build_vvhv,hylleraas_energy}`, `ferric-mp2/src/ragged.rs::solve_ragged` | ready |
| `integral-direct` | The integral-direct path | M mp2 "Exact and local MP2" | `ferric-mp2/src/lmp2_direct.rs::amplitude_lmp2_direct` | ready |
| `pno-dlpno` | Pair natural orbitals and DLPNO-MP2 | W §10 | `ferric-mp2/src/dlpno_mp2.rs::dlpno_mp2_from_b_ov`, `ferric-mp2/src/local_pno.rs::build_pno_transforms` | ready |
| `local-opmp3` | Amplitude-threshold local OP-MP3 | S `local_opmp3_sympy.py`; P `prototypes/local_opmp3_proto.py`; R local-opmp3 | `ferric-mp2/src/opmp3.rs::{pade_trace,ssd_opmp3}`, `ferric-mp2/src/ragged.rs::solve_ragged` | ready |

### Part VI. Derivatives

**SCF gradients, Hessians and geometry** (chapter `derivatives`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `hf-gradient` | Hartree–Fock nuclear gradient | S `hf_gradient_sympy.py`†; R hf-gradient | `ferric-scf/src/gradient.rs::{rhf_gradient,uhf_gradient,rohf_gradient,build_energy_weighted_density}` | ready |
| `df-jk-gradient` | Gradients of density-fitted J and K | S `df_jk_gradient_sympy.py`; R hf-gradient | `ferric-scf/src/df_gradient.rs::df_jk_gradient` | edit |
| `xc-gradient` | XC gradient: AO-derivative and grid-response terms | S `dft_mgga_tau_gradient_sympy.py`; R ks-gradient | `ferric-dft/src/gradient.rs::{xc_gradient_closed_gga_from_density,xc_gradient_closed_mgga_from_density}`, `ferric-scf/src/ks_gradient.rs::ks_gradient_closed` | ready |
| `hf-hessian` | Hartree–Fock Hessian and coupled-perturbed equations | S `hf_hessian_sympy.py`†; R hf-hessian | `ferric-scf/src/hessian.rs::{rhf_hessian,uhf_hessian}` | ready |
| `ks-hessian` | Kohn–Sham Hessian on a fixed and a moving grid | S `ks_fixed_grid_hessian_sympy.py`, `uks_xc_hessian_sympy.py`; R ks-hessian | `ferric-scf/src/ks_fixed_grid.rs::{skeleton_hessian,first_order_fock,skeleton_hessian_moving}`, `ferric-dft/src/becke_hessian.rs::becke_home_weight_hess` | ready |
| `frequencies` | Harmonic frequencies | S `hf_frequencies_sympy.py`†; R harmonic-frequencies | `ferric-scf/src/frequencies.rs::{frequencies_from_cartesian_hessian,harmonic_frequencies_analytic}` | ready |
| `fock-hook-gradient` | Fock-hook gradients | none | `ferric-scf/src/gradient.rs::with_fock_hook_gradient` | edit |
| `geometry-search` | Geometry optimization, saddle points and reaction paths | S `geometry_search_steps_sympy.py`; R geometry-search; M scf "Gradients, optimization, frequencies, transition states" | `ferric-scf/src/optimize.rs::optimize_geometry`, `ferric-scf/src/saddle.rs::find_saddle`, `ferric-scf/src/irc.rs::follow_irc` | edit |

**Correlated gradients and the Z-vector** (chapter `correlated-gradients`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `lagrangian-zvector` | Lagrangians and the Z-vector equation | S `mp2_variant_identities.py` claim 5 | `ferric-mp2/src/zvector.rs::{solve_zvector,build_relaxed_density_ao}` | ready |
| `rimp2-gradient` | RI-MP2 gradient with exact and RI-JK references; SCS-MP2 | S `mp2_gradient_sympy.py`†; R ri-mp2-gradient, scs-mp2-gradient | `ferric-mp2/src/gradient.rs::{rimp2_gradient_analytical,scs_mp2_gradient_analytical,total_rimp2_gradient}` | ready |
| `oo-rimp2-gradient` | OO-RI-MP2 gradient | R oo-rimp2-gradient | `ferric-mp2/src/oo_rimp2_gradient.rs::oo_ri_mp2_gradient` | ready |
| `ks-zvector` | Kohn–Sham Z-vector for post-SCF corrections | none | `ferric-scf/src/zvector_ks.rs::relaxation_gradient_closed` | ready |
| `rpa-gradient` | RI-dRPA gradient | S `rpa_gradient_sympy.py`†; R rpa-gradient | `ferric-rpa/src/gradient_analytic.rs::{rpa_analytic_gradient,rpa_ri_densities}` | ready |
| `mp2-polarizability` | MP2 static polarizability | R mp2-polarizability *gap* | `ferric-mp2/src/cpks_polar.rs::{mp2_polarizability_analytic,static_relaxed_density_ao}`, `ferric-mp2/src/ff_polar.rs::mp2_polarizability_static` | ready |
| `u-rimp2-gradient` | Unrestricted RI-MP2 gradient | S `u_rimp2_gradient_sympy.py`; R u-ri-mp2-gradient | `ferric-mp2/src/u_rimp2_gradient.rs::{u_rimp2_gradient_analytical,u_energy_weights,solve_u_zvector}` | ready |

### Part VII. Coupled cluster

**Coupled cluster** (chapter `coupled-cluster`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `ccd-ccsd-spin-orbital` | CCD and CCSD in spin orbitals | W §5; R ccd *gap*, ccsd *gap* | `ferric-cc/src/ccd.rs::ccd`, `ferric-cc/src/ccsd.rs::ccsd` | ready |
| `closed-shell-ccsd` | Closed-shell CCSD and the particle-particle ladder | M cc "Particle-particle ladder"; R ccsd *gap* | `ferric-cc/src/ccsd_closed_shell.rs::ccsd_closed_shell`, `ferric-cc/src/ri_ladder.rs::ri_ladder_contract` | ready |
| `triples` | (T) in spin orbitals and in closed-shell form | S `cc_triples_weights_and_linlccd_spin_adaptation_sympy.py`; R ccsd-t *gap* (the closed-shell text describes a removed kernel) | `ferric-cc/src/ccsd_t.rs::ccsd_t`, `ferric-cc/src/ccsd_t_closed_shell.rs::ccsd_t_closed_shell` | ready |
| `eom-ccsd` | EOM-CCSD singlets | R eom-ccsd *gap* | `ferric-cc/src/eom_ccsd.rs::eom_ccsd_singlets`, `ferric-cc/src/eom_davidson.rs::davidson_nonsymmetric` | ready |
| `linlccd` | LinLCCD and LinLCCD(hh) | S `cc_triples_weights_and_linlccd_spin_adaptation_sympy.py`; R linlccd; W §9 | `ferric-cc/src/linlccd.rs::linlccd`, `ferric-cc/src/linlccd_u.rs::u_linlccd`, `ferric-cc/src/helpers.rs::contract_hh_ladder` | ready |
| `wb97x-l-v` | The ωB97X-L-V double hybrid | R double-hybrid *gap* | `ferric-cc/src/double_hybrid.rs::run_wb97x_l_v` | ready |
| `local-cc` | Local coupled cluster: amplitude threshold and DLPNO | W §10, §13 | `ferric-cc/src/linlccd_amplitude.rs::amplitude_linlccd`, `ferric-cc/src/dlpno_linlccd.rs::dlpno_linlccd_hh`, `ferric-cc/src/dlpno_ccsd_virtual.rs::pno_ccsd_energy`, `ferric-cc/src/dlpno_ccsd_t.rs::screened_triple_energy` | edit |
| `uccsd` | Unrestricted CCSD and CCSD(T) | S `cc_uccsd_spin_blocks_sympy.py`; R u-ccsd, u-ccsd-t | `ferric-cc/src/u_ccsd.rs::u_ccsd`, `ferric-cc/src/u_ccsd_t.rs::u_ccsd_t` | ready |

### Part VIII. Multireference

**CAS-CI, CASSCF and NEVPT2** (chapter `multireference`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `determinants` | Determinants, strings and the phase convention | none | `ferric-ci/src/strings.rs::{enumerate_strings,phase_below}` | ready |
| `slater-condon` | Slater–Condon rules as implemented | S `casci_slater_condon_phases_sympy.py`; R casci | `ferric-ci/src/hamiltonian.rs::{general_element,diagonal_element}` | ready |
| `direct-sigma-davidson` | Direct sigma and the Davidson solver | none | `ferric-ci/src/sigma.rs::DirectSigma`, `ferric-ci/src/davidson.rs::davidson_lowest`, `ferric-ci/src/lib.rs::run_cas_ci` | ready |
| `rdms` | Reduced density matrices | none | `ferric-ci/src/rdm.rs::{spin_summed_rdms,rdm_energy}`, `ferric-ci/src/rdm3.rs::nevpt_rdms` | ready |
| `casscf` | Two-step CASSCF | R casscf *gap* | `ferric-ci/src/casscf.rs::{run_casscf,optimize,expm_antisymmetric}`, `ferric-ci/src/casscf_newton.rs::{orbital_hessian,ah_davidson}`, `ferric-ci/src/casscf_stability.rs::stability` | ready |
| `nevpt2` | SC-NEVPT2 | R nevpt2 *gap* | `ferric-ci/src/nevpt2.rs::sc_nevpt2` | ready |

### Part IX. RPA, PDEP and GW

**The random phase approximation** (chapter `rpa`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `drpa-three-ways` | dRPA three ways: frequency integral, plasmon formula, ring amplitudes | S `rpa_drpa_pdep_sympy.py`†; R drpa; P derivation §7–9 | `ferric-rpa/src/energy.rs::rpa_correlation_energy`, `ferric-rpa/src/diagnostics.rs::ri_drpa_energy`, `ferric-mp2/src/drpa_amplitude.rs::canonical_plasmon_drpa` | ready |
| `quadratures` | Imaginary-frequency quadratures | none | `ferric-rpa/src/quadrature.rs::{build_quadrature,chebyshev_tan_nodes,gauss_legendre_nodes}` | ready |
| `pdep` | Projective dielectric eigenpotentials | S `rpa_drpa_pdep_sympy.py`†; R pdep-rpa; W §7 | `ferric-rpa/src/lib.rs::run_pdep_rpa`, `ferric-rpa/src/sternheimer.rs::{dielectric_matrix,dielectric_apply}`, `ferric-rpa/src/lanczos.rs::run_lanczos_seeded` | ready |
| `open-shell-dielectric` | Open-shell dielectric | R u-pdep-rpa | `ferric-rpa/src/lib.rs::run_u_pdep_rpa`, `ferric-rpa/src/sternheimer.rs::dielectric_matrix_unrestricted` | ready |
| `laplace-ao-chi0` | Imaginary-time and AO forms of $\chi_0$ | none | `ferric-rpa/src/laplace_chi0.rs::dielectric_matrix_laplace`, `ferric-rpa/src/ao_rpa.rs::ao_rpa_correlation_energy` | ready |
| `local-drpa` | Ring amplitudes with a threshold: local dRPA | W §12; M rpa-gw "Exact and local dRPA" | `ferric-mp2/src/drpa_amplitude.rs::{amplitude_drpa,amplitude_drpa_direct}`, `ferric-mp2/src/rccd_family.rs::{rccd,sosex}` | verify |

**GW and the Bethe–Salpeter equation** (chapter `gw-bse`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `ri-factors-and-exchange` | RI factors and the exchange self-energy | none | `ferric-gw/src/mo_b.rs::build_full_b`, `ferric-gw/src/cohsex.rs::sigma_x_diag` | ready |
| `screened-interaction` | The screened interaction on the imaginary axis; PDEP as $W$ | S `gw_self_energy_sympy.py` | `ferric-gw/src/w_pdep.rs::static_weights`, `ferric-gw/src/cohsex.rs::project_b_into_pdep` | ready |
| `correlation-self-energy` | Correlation self-energy | S `gw_self_energy_sympy.py`; R gw | `ferric-gw/src/sigma.rs::{sigma_c_at_z,run_g0w0}` | ready |
| `cohsex` | COHSEX | none | `ferric-gw/src/cohsex.rs::run_cohsex` | ready |
| `continuation-and-qp-equation` | Analytic continuation and the quasiparticle equation | none | `ferric-gw/src/pade.rs::{fit,eval}`, `ferric-gw/src/sigma.rs::solve_qp_for_mo` | ready |
| `eigenvalue-self-consistency` | Eigenvalue self-consistency | none | `ferric-gw/src/sigma.rs::{run_evgw0,run_evgw}` | ready |
| `open-shell-gw` | Open-shell GW | R u-gw *gap* | `ferric-gw/src/u_sigma.rs::run_u_g0w0`, `ferric-gw/src/u_cohsex.rs::run_u_cohsex` | ready |
| `bse-tda` | BSE in the Tamm–Dancoff approximation | R bse-tda *gap* | `ferric-gw/src/bse.rs::run_bse_tda` | ready |
| `rpax-static-polarizability` | Statically screened TDHF polarizability | R tdhf-static-polarizability *gap* | `ferric-gw/src/bse.rs::run_rpax_static_polarizability` | ready |

### Part X. Response and excited states

**TDDFT** (chapter `tddft`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `response` | Response matrices, TDA and the Casida equations | R tddft *gap* (text exists, no proof; a script is in progress on `docs/derivations-backfill`) | `ferric-tddft/src/tddft.rs::run_tddft`, `ferric-tddft/src/davidson.rs::{davidson_tda,davidson_casida}` | ready |
| `oscillator-strengths` | Transition dipoles and oscillator strengths | none | `ferric-tddft/src/tddft.rs::run_tddft` | ready |
| `kernel` | The kernel and the functional gate | none | `ferric-dft/src/lr_kernel.rs::resolve_response_xc` | ready |
| `rsh` | Range-separated hybrids | none | `ferric-scf/src/rsh_response.rs::exchange_response` | ready |
| `triplets` | Triplets | none | `ferric-scf/src/rhf_newton.rs::triplet_hessian_matvec` | ready |
| `unrestricted` | Unrestricted references and $\langle S^2\rangle$ | none | `ferric-tddft/src/tddft.rs::run_tddft` | ready |

**Polarizabilities, charges and export** (chapter `polarizabilities`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `polarizabilities-from-dielectric-modes` | Static and dynamic polarizabilities from the dielectric modes | S `rpa_c6_polarizability_sympy.py`†; R rpa-polarizability-c6 | `ferric-rpa/src/properties.rs::{pdep_polarizability_static,molecular_dynamic_polarizability}`, `ferric-rpa/src/response_modes.rs::polarizability_from_modes` | ready |
| `atomic-partitions` | Atomic partitions: Becke, Hirshfeld, Löwdin, ESP fits | S `atomic_charges_sympy.py`; R atomic-charges | `ferric-scf/src/properties.rs::{lowdin_charges,becke_charges,chelpg_charges,resp_charges,esp_at_atoms}` | edit |
| `export-schemas` | Export schemas | none | `ferric-export/src/ml.rs::export_npz`, `ferric-export/src/cube.rs::export_cube` | edit |

### Part XI. Environment

**Implicit solvation** (chapter `implicit-solvation`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `cavity-and-bem` | Cavity and boundary-element matrices | none | `ferric-pcm/src/cavity.rs::build_cavity`, `ferric-pcm/src/matrices.rs::{build_s_d,build_k_r}` | ready |
| `ief-operators` | The IEF-PCM operators, the symmetrized solve and the sphere limits | S `pcm_ief_sympy.py`; R pcm | `ferric-pcm/src/solver.rs::solve_pcm_charges` | ready |
| `reaction-field` | The reaction field in the Fock matrix | none | `ferric-pcm/src/potential.rs::{solute_potential_at_tesserae,build_reaction_field_operator}` | ready |
| `gradient` | Nuclear gradient | S `pcm_ief_sympy.py`; R pcm-gradient | `ferric-pcm/src/gradient.rs::pcm_gradient` | ready |
| `conductor-limit-and-cosmo` | The conductor limit and COSMO | S `cosmo_sympy.py`; R cosmo | `ferric-scf/src/cosmo.rs::{cosmo_reaction_field,f_epsilon}` | edit |

**Embedding and QM/MM** (chapter `embedding`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `external-potentials` | Point charges, smeared charges and uniform fields | S `external_potential_sympy.py` | `ferric-core/src/external_potential.rs::ExternalPotential`, `ferric-integrals/src/oneelectron.rs::hcore_with_external` | edit |
| `qmmm-boundaries` | QM/MM boundaries, link atoms and the full gradient | S `qmmm_boundary_sympy.py`; M `site/src/using/qmmm.md`; R qmmm | `ferric-scf/src/qmmm.rs::{mm_forces,full_gradient,optimize_qmmm}` | edit |
| `polarizable-embedding` | Polarizable embedding | S `thole_polarizable_sympy.py`; R qmmm | `ferric-scf/src/polarizable.rs::{induce,site_gradient}` | edit |
| `molecular-mechanics` | The molecular-mechanics force field | S `mm_dihedral_gradient_sympy.py`, `mm_cutoff_sympy.py`; R mm | `ferric-mm/src/energy.rs::{energy,gradient}`, `ferric-mm/src/cutoff.rs::{energy_with,gradient_with}` | ready |

### Part XII. Periodic systems

**Gaussian-basis periodic calculations** (chapter `periodic`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `ewald` | Lattices, Ewald summation and the Madelung constant | S `pbc_ewald_pairft_stress_sympy.py`; R pbc-ewald-pairft-stress | `ferric-pbc/src/ewald.rs::{ewald_nuclear_repulsion,madelung_constant}` | ready |
| `pair-ft` | Fourier transform of lattice-summed Gaussian pairs | none | `ferric-pbc/src/pair_ft.rs::pair_ft` | ready |
| `nuclear-attraction` | Ewald-split nuclear attraction | none | `ferric-pbc/src/hcore.rs::periodic_hcore` | ready |
| `exchange-divergence` | The exchange divergence | R pbc-hf *gap* | `ferric-pbc/src/kpts.rs::madelung` | ready |
| `rsgdf` | Range-separated Gaussian density fitting for J and K | none; R pbc-hf *gap* | `ferric-pbc/src/rsgdf.rs::{j_builder,k_builder}` | ready |
| `kpoint-scf` | k-point Hartree–Fock and Kohn–Sham | none; R pbc-hf *gap*, pbc-ks *gap* | `ferric-pbc/src/kpts.rs::monkhorst_pack`, `ferric-pbc/src/kscf.rs::solve_krhf`, `ferric-pbc/src/kuscf.rs::solve_kuhf`, `ferric-pbc/src/kdft.rs::solve_krks` | edit |
| `forces` | Forces | R pbc-ks *gap* | `ferric-pbc/src/grad.rs::gamma_rhf_gradient`, `ferric-pbc/src/kgrad.rs::kpoint_rhf_gradient` | edit |
| `stress` | Stress | S `pbc_ewald_pairft_stress_sympy.py` | `ferric-pbc/src/stress.rs::gamma_rhf_stress`, `ferric-pbc/src/ewald.rs::ewald_nuclear_strain`, `ferric-pbc/src/pair_ft.rs::pair_ft_strain_chunked` | edit |
| `mp2-drpa` | Periodic MP2 and dRPA | none; R pbc-mp2 *gap*, pbc-drpa *gap* | `ferric-pbc/src/mp2.rs::gamma_mp2`, `ferric-pbc/src/drpa.rs::gamma_drpa`, `ferric-pbc/src/kcorr.rs::{kpoint_mp2,kpoint_drpa}` | ready |

### Part XIII. Dispersion

**Dispersion** (chapter `dispersion`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `d3bj` | D3(BJ): energy, coordination numbers, gradient, three-body term | S `d3_cn_chain_rule_gradient_sympy.py`, `d3_atm_gradient_sympy.py`; R d3 | `ferric-d3/src/lib.rs::{d3bj_energy,d3bj_gradient}` | ready |
| `casimir-polder-c6` | Casimir–Polder $C_6$ from dynamic polarizabilities | S `rpa_c6_polarizability_sympy.py`†; D 1.4.3 Eqs. 1.49–1.51 | `ferric-rpa/src/dispersion.rs::{casimir_polder_c6,ts_dynamic_polarizability,pdep_dynamic_polarizability}` | ready |
| `mbd` | Many-body dispersion | M scf "Dispersion correction: MBD@rsSCS"; R mbd *gap* | `ferric-rpa/src/dispersion/mbd.rs::{mbd_screen,mbd_dynamic_polarizability,dipole_coupling_tensor}` | edit |

### Part XIV. Capstone

**Short-range MP2 with long-range RPA** (chapter `rs-mp2-rpa`)

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `cross-term` | Second order in a split interaction: the cross term again | §10.7; P derivation §5–6; P paper "Second order and the cross term" | `ferric-mp2/src/attenuated.rs::rs_mp2_decomposition` | edit |
| `ring-series` | The ring series by three routes | §18.1; P derivation §7–9 | `ferric-rpa/src/energy.rs::rpa_correlation_energy` | gap |
| `coloring-the-rings` | Coloring the rings: which diagrams each scheme keeps | P derivation §10–16; P paper "range-coloring classification", Appendix A | none | gap |
| `two-formulations` | The two formulations in the code | S `rpa_rs_mp2_rpa_sympy.py`†; R rs-mp2-rpa | `ferric-rpa/src/rs_mp2_rpa.rs::{rs_mp2_lr_rpa,RsMp2RpaFormulation}` | ready |
| `omega-scale-law` | The $\omega$-scale law | P derivation §17; P paper "The $\omega$-scale law" | none | gap |
| `shared-intermediates` | Shared intermediates across the two kernels | P derivation §19; code comments | `ferric-mp2/src/rimp2.rs::{compute_rpa_intermediates,spin_components_from_b_ov}`, `ferric-rpa/src/rs_mp2_rpa.rs::second_intermediate_bov_bytes` | edit |
| `measurements` | What was measured, and what does not work | P paper "Results", P derivation §18, §20; D 7.2 | none | gap |

### Part XV. Interfaces

**ML integration touchpoints** (chapter `ml`)

For an implementer wiring ferric outputs into ML models. Chapter `polarizabilities` derives the
properties; this chapter states what each exported array is, its frame behaviour,
its evidence and its use as a feature or label.

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `npz-bundle` | The NPZ feature bundle | R pdep-rpa; `site/src/reference/validation.md` NPZ rows | `ferric-export/src/ml.rs::export_npz`, `ferric-rpa/src/property_bundle.rs::{compute_pdep_properties,write_pdep_npz}` | edit |
| `electrostatics-and-charges` | Electrostatic features and atomic charges | none (no derivation); validation anchors | `ferric-scf/src/properties.rs::{esp_at_atoms,electric_field_at_atoms,esp_at_points,lowdin_charges,mulliken_charges,chelpg_charges,resp_charges}`, `ferric-scf/src/properties/hirshfeld_grid.rs::hirshfeld_charges` | gap |
| `polarizability-pdep-c6` | Polarizabilities, PDEP modes and $C_6$ | S `rpa_c6_polarizability_sympy.py` claims 1, 2, 3, 5; R rpa-polarizability-c6 | `ferric-rpa/src/properties/per_atom.rs::{closed_per_atom_alpha,charge_transfer_remainder}`, `ferric-rpa/src/dispersion.rs::{casimir_polder_c6,ts_dynamic_polarizability,pdep_dynamic_polarizability}` | ready |
| `python-cube-provenance` | The Python surface, cube files and provenance | `ferric.pyi`; M `site/src/using/{ase,qcschema,export}.md` | `ferric-export/src/cube.rs::export_cube`, `ferric-core/src/certificate.rs::Certificate` | edit |
| `reproducibility-and-grades` | Reproducibility and validation grades | `site/src/reference/{architecture,validation,input}.md`; `site/src/using/sharp-bits.md` | `ferric-core/src/blas_threads.rs::{with_blas_threads,opt_in_blas_threads}` | edit |
| `training-labels` | Energies, gradients, Hessians and response as labels | validation matrix and anchors; R hf-gradient, ks-gradient, ri-mp2-gradient, rpa-gradient, hf-hessian, ks-hessian, harmonic-frequencies | `ferric-rpa/src/gradient.rs::total_rpa_gradient`, `ferric-scf/src/noise_floor.rs::de_noise_floor` | edit |
| `esp-conditioning` | A worked consumer: esp-conditioning | the esp-conditioning builder code (interface only) | none | edit |

### Part XVI. Dynamics

**Molecular dynamics** (chapter `md`)

Time integration of the nuclei on any gradient source: velocity Verlet, Berendsen weak coupling and Langevin BAOAB.

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `molecular-dynamics` | Integrators and thermostats: velocity Verlet, Berendsen, Langevin BAOAB | S `md_integrators_sympy.py`; R molecular-dynamics | `ferric-md/src/driver.rs::run_md`, `ferric-md/src/config.rs::{MdConfig,Thermostat}`, `ferric-md/src/velocities.rs::{maxwell_boltzmann,remove_com_momentum,kinetic_energy,temperature}`, `ferric-md/src/rng.rs::Xoshiro256pp` | edit |

### Appendices

| Label | Section | Sources | Code | Status |
|---|---|---|---|---|
| `appendix-a` | Index of verified equations and proof scripts (generated) | registry; box metadata | none | ready |
| `appendix-b` | Index of functions cited (generated) | code-reference metadata | none | ready |
| `appendix-c` | Validation grades (generated from the validation page) | `site/src/reference/validation.md` | `ferric-core/src/certificate/anchors.rs::EnergyAnchor` | ready |
| `appendix-d` | The semiempirical interface (xTB) | R xtb *gap* | `ferric-xtb/src/calculator.rs::XtbCalculator` | gap |

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
| III. Self-consistent field | 8 | 3 | 0 | 0 | 11 |
| IV. Density functional theory | 7 | 0 | 0 | 0 | 7 |
| V. The MP2 family | 19 | 3 | 1 | 5 | 28 |
| VI. Derivatives | 12 | 3 | 0 | 0 | 15 |
| VII. Coupled cluster | 7 | 1 | 0 | 0 | 8 |
| VIII. Multireference | 6 | 0 | 0 | 0 | 6 |
| IX. RPA, PDEP and GW | 14 | 0 | 0 | 1 | 15 |
| X. Response and excited states | 7 | 2 | 0 | 0 | 9 |
| XI. Environment | 5 | 4 | 0 | 0 | 9 |
| XII. Periodic systems | 6 | 3 | 0 | 0 | 9 |
| XIII. Dispersion | 2 | 1 | 0 | 0 | 3 |
| XIV. Capstone | 1 | 2 | 4 | 0 | 7 |
| XV. Interfaces | 1 | 5 | 1 | 0 | 7 |
| XVI. Dynamics | 0 | 1 | 0 | 0 | 1 |
| Appendices | 3 | 0 | 1 | 0 | 4 |
| **Total** | **100** | **51** | **8** | **6** | **165** |

165 sections: 61% `ready`, 31% `edit`, 5% `gap` and 4% `verify`.
<!-- status-table:end -->

## 5. Gaps, ranked by how much rests on them

Rank reflects how many other chapters or user-facing results depend on the
missing text or proof.

| Rank | Gap | Sections | Why it ranks here | What closes it |
|---|---|---|---|---|
| 1 | Fock builders: direct and incremental J/K, RI-J, RI-K, Split-RI-J, LinK, COSX, CFMM | `direct-and-incremental`, `ri-j-ri-k-split-ri-j`, `link-cosx-cfmm` | Every calculation in every later chapter runs through them. The only committed text is 10 lines on CFMM. | Derivations in the book, a proof script for the RI-J and RI-K algebra and the COSX quadrature identity, then the sections. |
| 2 | Integrals, operators and screening | chapter `integrals`; `terf-table-engine`, `yukawa-geminal`; chapter `screening` | The attenuated operators, their table engine and the bounds are what the MP2 and capstone chapters stand on. | A full derivation in the book: kernels, the terf tables and their fallback, Schwarz and QQR bounds with proofs of the bound inequalities. |
| 3 | SCF convergence: DIIS variants, second-order steps, stability | chapter `convergence` | Decides whether a result exists at all. 6 lines of committed text. | Derivation of the orbital Hessian-vector products for RHF, UHF and ROHF, checked against finite differences of the gradient. |
| 4 | CCD and CCSD working equations | `ccd-ccsd-spin-orbital`, `closed-shell-ccsd` | Graded, user-facing methods with no committed derivation and no proof (registry gaps `ccd`, `ccsd`). | Spin-orbital residuals by `sympy.physics.secondquant`, then the spin-adapted form checked against them. |
| 5 | Closed-shell (T) text describes a removed kernel | `triples` | The committed section is wrong about the current code (registry gap `ccsd-t`). | Rewrite the section from the current `triple_energy` kernel; the script's claim 4b already ports it. |
| 6 | Hybrid and range-separated exchange in Kohn–Sham | `hybrids` | The default functionals for most users are hybrids; registry gap `ks-hybrid-rsh`. In progress on `docs/derivations-backfill`. | Land the derivation and its script. |
| 7 | TDDFT has text and no proof | chapter `tddft` | 441 lines of derivation with no machine check (registry gap `tddft`). In progress on `docs/derivations-backfill`. | Land the script; the sections then move to `ready`. |
| 8 | The capstone rests on unpublished notes | `ring-series`, `coloring-the-rings`, `omega-scale-law`, `measurements` | The book's final chapter. Only `two-formulations` is committed and proven. | Move the range-coloring classification and the $\omega$-scale law into the book with proof scripts (the paper's Appendix A already describes a symbolic check). |
| 9 | CASSCF and NEVPT2 have no proof | `casscf`, `nevpt2` | Graded methods; registry gaps `casscf`, `nevpt2`. The CASSCF text is long and is mostly measurement records. | A proof of the generalized-Fock gradient and of the NEVPT2 class energies on a small active space; cut the section to the derivation. |
| 10 | Periodic Hartree–Fock, RS-GDF and periodic correlation | `rsgdf`, `kpoint-scf`, `mp2-drpa` | Four registry gaps (`pbc-hf`, `pbc-ks`, `pbc-mp2`, `pbc-drpa`). The Ewald, pair-FT and stress pieces are proven; the energy functional and the fitting are not. | Derivation of the k-point energy functional, the exchange-divergence correction and the RS-GDF split. |
| 11 | Local correlation | chapter `local-mp2`; `local-cc`, `local-drpa` | Five sections whose only derivations are working notes. | Re-derive from the code; one proof per method of the zero-threshold limit. |
| 12 | Embedding: external potentials, QM/MM, polarizable sites, cDFT, COSMO | `cdft`, `conductor-limit-and-cosmo`, `external-potentials`, `qmmm-boundaries`, `polarizable-embedding` | Registry gaps `qmmm`, `cdft`, `cosmo`. Validated numerically, not derived. | Derivations of the link-atom gradient projection and of the stationary polarization functional. |
| 13 | Density-fitted gradients and the Kohn–Sham Z-vector | `df-jk-gradient`, `ks-zvector` | Used by every optimization with fitted J/K. | Derivation plus a finite-difference check of the three-index and two-index terms. |
| 14 | BSE-TDA, the statically screened TDHF polarizability and open-shell GW | `open-shell-gw`, `bse-tda`, `rpax-static-polarizability` | Three registry gaps. | Proofs of the per-spin dielectric and of the BSE kernel. |
| 15 | Unverified statements inside otherwise ready sections | `closed-shell-ri-mp2` (one equation), `oo-rimp2`, `mp2-polarizability` | Small, but each is an equation the text asserts without a check. | One claim per statement in an existing script. |
| 16 | Many-body dispersion | `mbd` | Registry gap `mbd`. | Derivation of the range-separated self-consistent screening and its gradient. |
| 17 | RI-MP2 at scale: memory, device kernels, distributed bands | `four-step-algorithm`, `pair-kernel`, `parallel-performance`, `device-mixed-precision` | Algorithmic text with no source besides code comments. | Written from the code; no proof needed beyond the existing bit-identity tests. |

## 6. Effort by part

A work stream ("lane") here is one agent taking a chapter from sources to a
committed draft that passes the book checker, including any proof claims the
chapter needs. "Proof work" counts the scripts or claims that must exist
before the text can be `ready`; it belongs with the derivation backfill, not
with the book lanes.

| Part | Chapters | Book lanes | Proof work first | Can run in parallel with | Needs Matt |
|---|---|---|---|---|---|
| 0, I | front matter, `second-quantization`, `molecules-and-numerics` | 1 | none | everything | Confirm the notation choices (section 8, question 2). |
| II | `integrals`, `operators`, `screening`, `density-fitting` | 3 | 3 scripts (bounds, terf identities, robust fit) | III, IV | No. |
| III | `hartree-fock`, `convergence` | 3 | 3 scripts (RI-J/RI-K algebra, orbital Hessians, DIIS error vector) | II, IV | No. |
| IV | `kohn-sham` | 1 | 1 script, in progress | II, III | No. |
| V | `mp2`, `rimp2-algorithms`, `oo-laplace`, `local-mp2` | 4 | 2 claims; local MP2 re-derivation | VI onward | Yes: how much of dissertation chapters 2, 3, 5 and 6 to carry, and whether figures and tables may be reproduced (questions 3 and 4). |
| VI | `derivatives`, `correlated-gradients` | 2 | 2 scripts (DF gradient, KS Z-vector) | V, VII | No. |
| VII | `coupled-cluster` | 2 | 2 scripts (CCSD residuals, EOM) | VI, VIII | No. |
| VIII | `multireference` | 1 | 2 scripts (CASSCF gradient, NEVPT2 classes) | VII, IX | Whether the long CASSCF measurement record is carried in the book (question 6). |
| IX | `rpa`, `gw-bse` | 2 | 2 scripts (BSE, per-spin GW) | VIII, X | No. |
| X | `tddft`, `polarizabilities` | 1 | 1 script, in progress | IX | No. |
| XI | `implicit-solvation`, `embedding` | 2 | 2 scripts (QM/MM gradient, polarization functional) | XII | No. |
| XII | `periodic` | 2 | 2 scripts (k-point functional, RS-GDF) | XI | No. |
| XIII | `dispersion` | 1 | 1 script (MBD) | any | No. |
| XIV | `rs-mp2-rpa` | 2 | 2 scripts (range coloring, $\omega$-scale law) | after V and IX | Yes: the chapter is Matt's unpublished work (question 5). |
| Build | tooling | 1 | none | everything | Format choice (question 7). |

Total: about 28 book lanes and about 25 proof scripts or claims. The
dependency order is 0 → I → II → III → (IV, V) → (VI, VII, VIII, IX) →
(X, XI, XII, XIII) → XIV. Parts II and III are the critical path because they
have the least committed text and the most dependents.

A first release that is useful on its own: Parts 0, I, V (chapters `mp2`,
`rimp2-algorithms` and `oo-laplace`), VI (chapter `correlated-gradients`) and
IX (chapter `rpa`), plus `factorization-and-metric` and `erf-terf`. Those sections are
`ready` or `edit` today and lead directly to the capstone.

## 7. Working with the derivation backfill

- The registry is the coverage ledger. A book section lists the registry ids
  it covers. The book checker fails when a section marked `ready` covers an
  id that the registry lists under `gaps`, and when a registry id is covered
  by no section.
- The book does not edit `scripts/`. A defect found while writing a section
  is filed against the backfill.
- When the backfill merges, two things change here: the † marks drop, and
  the 19 scripts without a runner gain one.

## 8. Questions only Matt can answer

1. **One text or two.** Settled: the book owns the text, and each
   `theory.md` is a generated index of the book pages that cite the crate.
2. **Notation.** `book/notation.md` picks a positive denominator
   $\Delta_{ij}^{ab}$, overbars for $\beta$ orbitals, $u$ for imaginary
   frequency so that $\omega$ stays the range-separation parameter, and
   $\xi, \eta$ for PDEP modes. Accept, or name the ones to reverse.
3. **How much dissertation.** Chapters 2, 3, 5 and 6 are mostly training and
   transferability results. The plan carries the method definitions, the
   parameters and one summary table per method (`attenuated-mp2`, `scs-mp2-2terfc`,
   `attenuation-basis-limit`), and leaves the
   benchmark tables in the dissertation. Is that the right cut for an
   implementer's reference?
4. **Reuse rights.** Dissertation chapters 2–6 were published as journal
   articles. May their figures and tables be reproduced in a public book, or
   should the book redraw from ferric's own runs?
5. **The capstone and the paper.** Chapter `rs-mp2-rpa` draws on an unpublished
   manuscript. Should the chapter wait for the paper, appear with only the
   committed formulations (`two-formulations`), or carry the full classification?
6. **Negative results and measurement records.** The wiki holds withdrawn
   and negative results (local RPA, the AO-Laplace
   locality history, the CASSCF measurement log). Does the book state only
   what the code does today, or does it keep a short "what does not work"
   section per chapter, as `measurements` does?
7. **Format and audience of the output.** HTML next to the user guide, a PDF
   with a version number, or both (`book/BUILD.md` recommends both from one
   source)? Is the book public from the first chapter, or after a complete
   first release?
8. **Title, authorship and license.** The dissertation text is Matt's; the
   repository is dual MIT/Apache. Does the book text take the same license?
9. **Scope edges.** xTB (an interface to an external library) and the
   Avogadro, ASE and QCSchema integrations are left to the user guide, with
   xTB as a one-page appendix. Confirm.
