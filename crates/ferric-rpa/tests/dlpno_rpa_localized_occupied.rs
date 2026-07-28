//! Does DLPNO-RPA's reduced virtual basis (and hence its compression) depend on
//! whether the OCCUPIED orbitals are canonical or Boys-localized?
//!
//! # Why this specific test
//!
//! `dlpno_rpa.rs` and `pno.rs` are the two places in ferric that get the
//! non-canonical-orbital rule RIGHT: both build `F` in the reduced virtual basis
//! and RE-DIAGONALIZE (`dlpno_rpa.rs:309-320`, `pno.rs:174-186`), with an
//! explicit comment that "taking the diagonal alone is silently wrong". So the
//! VIRTUAL side is disciplined.
//!
//! But DLPNO-RPA feeds on `compute_rpa_intermediates`, which builds `b_ov` from
//! `rhf.mos_r()` — i.e. CANONICAL occupieds — and `run_pdep_rpa_pno` passes a
//! ZERO-FILLED centres placeholder (`dlpno_rpa.rs:439`) because
//! `complete_pair_domains` applies no distance test. DLPNO-RPA has therefore
//! never been run on localized occupieds. Since "localization-first" is exactly
//! construction #2 of `rpa-locality-wall-lane-closed`, and DLPNO-RPA is the one
//! path with correct virtual-side treatment, this is the cleanest available test
//! of the original claim on existing machinery.
//!
//! # Hypotheses, stated BEFORE measuring
//!
//! * **H_phys** — localized occupieds produce spatially compact pair amplitudes,
//!   so each pair's PNO subspace is smaller and, crucially, DIFFERENT pairs
//!   select OVERLAPPING subspaces. The union `Σ_i Q^i (Q^i)ᵀ` then has lower rank
//!   than in the canonical case ⇒ `n_vir_reduced` DROPS ⇒ real compression that
//!   the canonical measurement hid.
//! * **H_null** — the shared reduced basis is built from a sum of projectors over
//!   ALL occupied orbitals. If that sum is an occupied-rotation invariant, the
//!   reduced basis is numerically identical and localization contributes nothing,
//!   exactly as it does for the AO-time pseudo-densities.
//!
//! These predict opposite observations (`n_vir_reduced` drops vs. is identical),
//! so the experiment discriminates.
//!
//! # How the localized occupieds are injected
//!
//! Under an occupied-space rotation `C_occ → C_occ U`, the RI tensor transforms
//! exactly as `B'^P_{i'a} = Σ_i U_{i i'} B^P_{ia}` — no new integrals are needed.
//! `U` is taken from a REAL Boys localization of the SCF occupied block, and the
//! test asserts `U` is genuinely orthogonal and genuinely non-identity, so it
//! cannot pass vacuously.
//!
//! No timings anywhere. Ranks, dimensions, retention fractions, errors.
//!
//! Run with:
//!   OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo test -p ferric-rpa --test dlpno_rpa_localized_occupied -- --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::{dipole, overlap};
use ferric_integrals::operator::Operator;
use ferric_mp2::pair_domains::complete_pair_domains;
use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config, RpaIntermediates};
use ferric_rpa::dlpno_rpa::build_dlpno_rpa_transform;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{s, Array2};

struct Case {
    inter: RpaIntermediates,
    eps: Vec<f64>,
    /// Boys rotation of the ACTIVE occupied block, (nocc, nocc).
    u_boys: Array2<f64>,
    label: String,
}

fn setup(path: &str, obs_name: &str, aux_name: &str, label: &str) -> Case {
    let mol = Molecule::load_xyz(path).unwrap_or_else(|e| panic!("load {path}: {e}"));
    let bs = basis::bundled(obs_name).unwrap();
    let aux = basis::bundled(aux_name).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let cfg = RiMp2Config::default();
    let inter = compute_rpa_intermediates(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();

    // Boys-localize the ACTIVE occupied block and extract the rotation U.
    let c = rhf.mos_r();
    let c_occ = c
        .slice(s![.., inter.first_occ..inter.first_occ + inter.nocc])
        .to_owned();
    let dip = dipole(&obs, [0.0, 0.0, 0.0]).unwrap();
    let boys = ferric_mp2::boys::boys_localize(&c_occ, &dip, 400);
    // C_loc = C_occ U  ⇒  U = C_occᵀ S C_loc. Using the SCF overlap keeps this
    // exact for a non-orthogonal AO basis.
    let s_ao = overlap(&obs);
    let u = c_occ.t().dot(&s_ao).dot(&boys.c_loc);

    println!(
        "\n=== {label}: nbas={} nocc={} nvir={} naux={} ===",
        obs.nbasis(),
        inter.nocc,
        inter.nvir,
        inter.naux
    );

    Case {
        inter,
        eps: rhf.eps_r().to_vec(),
        u_boys: u,
        label: label.to_string(),
    }
}

/// Apply an occupied-space rotation to `b_ov`, in place of rebuilding integrals:
/// `B'^P_{i'a} = Σ_i U_{i i'} B^P_{ia}`.
fn rotate_b_ov_occupied(b_ov: &Array2<f64>, u: &Array2<f64>, nocc: usize, nvir: usize) -> Array2<f64> {
    let naux = b_ov.nrows();
    assert_eq!(b_ov.ncols(), nocc * nvir);
    assert_eq!(u.dim(), (nocc, nocc));
    let mut out = Array2::<f64>::zeros((naux, nocc * nvir));
    for ip in 0..nocc {
        for i in 0..nocc {
            let uii = u[(i, ip)];
            if uii == 0.0 {
                continue;
            }
            let src = b_ov.slice(s![.., i * nvir..(i + 1) * nvir]);
            let mut dst = out.slice_mut(s![.., ip * nvir..(ip + 1) * nvir]);
            dst.scaled_add(uii, &src);
        }
    }
    out
}

// ===========================================================================
// EXACTNESS ANCHORS — must pass before any measurement is believed
// ===========================================================================

/// ANCHOR 1: the Boys rotation `U` really is orthogonal (so it is a legitimate
/// occupied-space rotation) and really is NOT the identity (so it does something).
#[test]
fn anchor_boys_rotation_is_orthogonal_and_nontrivial() {
    for (path, obs, aux, label) in [
        ("../../testdata/molecules/water.xyz", "sto-3g", "cc-pvdz-ri", "water/STO-3G"),
        ("../../testdata/molecules/alkane_4.xyz", "sto-3g", "cc-pvdz-ri", "alkane_4/STO-3G"),
    ] {
        let c = setup(path, obs, aux, label);
        let n = c.u_boys.nrows();
        let uut = c.u_boys.t().dot(&c.u_boys);
        let mut orth_err = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let target = if i == j { 1.0 } else { 0.0 };
                let e = (uut[(i, j)] - target).abs();
                if e > orth_err {
                    orth_err = e;
                }
            }
        }
        let off_ident = {
            let mut m = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    let t = if i == j { 1.0 } else { 0.0 };
                    let d = (c.u_boys[(i, j)] - t).abs();
                    if d > m {
                        m = d;
                    }
                }
            }
            m
        };
        println!("  {}: |UᵀU − I|max = {:.3e}   |U − I|max = {:.3}", label, orth_err, off_ident);
        assert!(orth_err < 1e-9, "{label}: U is not orthogonal ({orth_err:.3e})");
        assert!(
            off_ident > 0.1,
            "{label}: Boys rotation is essentially the identity (|U−I| = {off_ident:.3e}); \
             every 'localized' measurement below would secretly be the canonical one"
        );
    }
}

/// ANCHOR 2: rotating `b_ov` by the IDENTITY reproduces the original tensor
/// exactly, and rotating by `U` then `Uᵀ` round-trips. This validates
/// `rotate_b_ov_occupied` itself — the only new machinery in this file.
#[test]
fn anchor_b_ov_rotation_helper_is_correct() {
    let c = setup("../../testdata/molecules/alkane_4.xyz", "sto-3g", "cc-pvdz-ri", "alkane_4 anchor-2");
    let (nocc, nvir) = (c.inter.nocc, c.inter.nvir);
    let b = &c.inter.b_ov;

    let ident = Array2::<f64>::eye(nocc);
    let b_id = rotate_b_ov_occupied(b, &ident, nocc, nvir);
    let e_id = (&b_id - b).iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    println!("  identity-rotation error = {e_id:.3e}");
    assert!(e_id < 1e-14, "identity rotation must be exact: {e_id:.3e}");

    let b_rot = rotate_b_ov_occupied(b, &c.u_boys, nocc, nvir);
    let b_back = rotate_b_ov_occupied(&b_rot, &c.u_boys.t().to_owned(), nocc, nvir);
    let scale = b.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let e_rt = (&b_back - b).iter().map(|v| v.abs()).fold(0.0_f64, f64::max) / scale;
    println!("  round-trip (U then Uᵀ) rel error = {e_rt:.3e}");
    assert!(e_rt < 1e-10, "U/Uᵀ round-trip must be exact: {e_rt:.3e}");

    // TEETH: the rotation must actually CHANGE b_ov, else the round-trip above
    // is trivially satisfied and this anchor proves nothing.
    let changed = (&b_rot - b).iter().map(|v| v.abs()).fold(0.0_f64, f64::max) / scale;
    println!("  rotation changes b_ov by rel {changed:.3e}");
    assert!(
        changed > 1e-2,
        "the Boys rotation barely changed b_ov (rel {changed:.3e}); the measurement below \
         would then be comparing a tensor to itself"
    );
}

// ===========================================================================
// THE MEASUREMENT
// ===========================================================================

/// THE QUESTION: with the virtual side already handled correctly by DLPNO-RPA's
/// semicanonicalization, does switching the OCCUPIED orbitals from canonical to
/// Boys-localized change the reduced virtual basis DLPNO-RPA arrives at?
///
/// Reported per threshold: `n_vir_reduced` for both orbital choices, and the
/// principal-angle agreement of the two reduced SUBSPACES.
#[test]
fn dlpno_rpa_reduced_basis_canonical_vs_localized_occupied() {
    let mut any_truncated = false;
    // Water/STO-3G is deliberately EXCLUDED: nvir = 2, so DLPNO-RPA cannot
    // truncate anything and the comparison would be vacuous. That is a
    // pass-condition-reachability constraint, not a result.
    for (path, obs, aux, label) in [
        ("../../testdata/molecules/alkane_2.xyz", "sto-3g", "cc-pvdz-ri", "alkane_2/STO-3G"),
        ("../../testdata/molecules/alkane_4.xyz", "sto-3g", "cc-pvdz-ri", "alkane_4/STO-3G"),
        ("../../testdata/molecules/benzene.xyz", "sto-3g", "cc-pvdz-ri", "benzene/STO-3G"),
        // 6-31G / cc-pVDZ rows: the ONLY regime where DLPNO-RPA truncates at
        // all, hence the only rows where the canonical-vs-localized comparison
        // is non-vacuous on counts. STO-3G is minimal (nvir < nocc) and the
        // union always re-inflates to full rank there.
        ("../../testdata/molecules/water.xyz", "6-31g", "cc-pvdz-ri", "water/6-31G"),
        ("../../testdata/molecules/water.xyz", "cc-pvdz", "cc-pvdz-ri", "water/cc-pVDZ"),
        ("../../testdata/molecules/alkane_2.xyz", "6-31g", "cc-pvdz-ri", "alkane_2/6-31G"),
        ("../../testdata/molecules/alkane_4.xyz", "6-31g", "cc-pvdz-ri", "alkane_4/6-31G"),
    ] {
        let c = setup(path, obs, aux, label);
        let (nocc, nvir) = (c.inter.nocc, c.inter.nvir);
        let domains = complete_pair_domains(&Array2::<f64>::zeros((nocc, 3))).unwrap();

        // The localized-occupied intermediates: same everything, rotated b_ov.
        let b_loc = rotate_b_ov_occupied(&c.inter.b_ov, &c.u_boys, nocc, nvir);
        let inter_loc = RpaIntermediates {
            b_ov: b_loc,
            v_inv_sqrt: c.inter.v_inv_sqrt.clone(),
            nocc: c.inter.nocc,
            nvir: c.inter.nvir,
            nocc_total: c.inter.nocc_total,
            first_occ: c.inter.first_occ,
            naux: c.inter.naux,
        };

        println!(
            "\n  {:<18} {:>10} {:>12} {:>12} {:>10} {:>10} {:>12}",
            c.label, "t_cut", "nvir_can", "nvir_loc", "nvir_full", "Δ", "max|Δeps|"
        );
        // Thresholds span from AGGRESSIVE (1e-1, far looser than any production
        // DLPNO setting) down to tight. The loose end exists to make the pass
        // condition REACHABLE: if even 1e-1 retains every virtual, the union has
        // re-inflated to full rank and no threshold can make this comparison
        // informative — which is itself the finding.
        for &t in &[3e-1, 1e-1, 3e-2, 1e-2, 1e-3, 1e-4] {
            let can = build_dlpno_rpa_transform(&c.inter, &c.eps, &domains, t);
            let loc = build_dlpno_rpa_transform(&inter_loc, &c.eps, &domains, t);
            match (can, loc) {
                (Ok(a), Ok(b)) => {
                    // Equal COUNTS could still hide different SUBSPACES. Compare
                    // the spans via the eigenvalues of the reduced-basis Fock
                    // sets, which are basis-independent invariants of the span.
                    let span_note = if a.n_vir_reduced == b.n_vir_reduced {
                        let mut d = 0.0_f64;
                        for k in 0..a.n_vir_reduced {
                            let x = (a.eps_vir_reduced[k] - b.eps_vir_reduced[k]).abs();
                            if x > d {
                                d = x;
                            }
                        }
                        format!("{d:.2e}")
                    } else {
                        "n/a".to_string()
                    };
                    println!(
                        "  {:<18} {:>10.0e} {:>12} {:>12} {:>10} {:>10} {:>12}",
                        "",
                        t,
                        a.n_vir_reduced,
                        b.n_vir_reduced,
                        nvir,
                        b.n_vir_reduced as i64 - a.n_vir_reduced as i64,
                        span_note
                    );
                }
                (x, y) => {
                    println!("  {:<18} {:>10.0e}  build failed: can={:?} loc={:?}",
                        "", t, x.err(), y.err());
                }
            }
        }

        let a = build_dlpno_rpa_transform(&c.inter, &c.eps, &domains, 1e-1)
            .expect("canonical transform at 1e-1");
        if a.n_vir_reduced < nvir {
            any_truncated = true;
        } else {
            println!(
                "  [{}] NOTE: no truncation even at t_cut=1e-1 ({} of {} virtuals kept) — \
                 the per-orbital PNO union has re-inflated to FULL rank",
                c.label, a.n_vir_reduced, nvir
            );
        }
    }

    // TEETH (global): at least one system must actually have been truncated,
    // otherwise every "Δ = 0" row above is vacuous — two identical full-rank
    // bases trivially agree and say nothing about localization.
    assert!(
        any_truncated,
        "DLPNO-RPA truncated NOTHING on any system even at t_cut=1e-1. The \
         canonical-vs-localized comparison is therefore vacuous on COUNTS (two \
         identical full-rank bases trivially agree). The max|Δeps| column is still \
         a valid invariance measurement, but this assertion fires so the counts \
         are never read as evidence. It ALSO reproduces the union re-inflation of \
         `rpa-cannot-consume-pair-indexed-bases` at full strength."
    );
}

/// The `eps` slice DLPNO-RPA consumes is indexed by CANONICAL orbital number
/// (`eps[nocc_total + a]` at `dlpno_rpa.rs:315`, and `eps_occ` sliced at
/// `dlpno_rpa.rs:444-445`). Under a localized-occupied `b_ov` those occupied
/// energies are no longer the right per-orbital numbers — the localized Fock
/// block is non-diagonal. This test measures the size of that inconsistency so
/// the result above can be interpreted honestly.
#[test]
fn dlpno_rpa_occupied_energies_are_canonical_by_construction() {
    let c = setup("../../testdata/molecules/alkane_4.xyz", "sto-3g", "cc-pvdz-ri", "alkane_4 eps-audit");
    let nocc = c.inter.nocc;
    // F_loc in the localized occupied basis is Uᵀ diag(eps_occ) U.
    let eps_occ: Vec<f64> = c.eps[c.inter.first_occ..c.inter.first_occ + nocc].to_vec();
    let d = Array2::from_diag(&ndarray::Array1::from(eps_occ.clone()));
    let f_loc = c.u_boys.t().dot(&d).dot(&c.u_boys);
    let mut off = 0.0_f64;
    for i in 0..nocc {
        for j in 0..nocc {
            if i != j && f_loc[(i, j)].abs() > off {
                off = f_loc[(i, j)].abs();
            }
        }
    }
    let diag: Vec<f64> = (0..nocc).map(|i| f_loc[(i, i)]).collect();
    let spread = diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - diag.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("  localized occupied Fock: max|offdiag| = {off:.4e}, diag spread = {spread:.4e}, ratio = {:.3}", off / spread);
    assert!(
        off > 1e-3,
        "localized occupied Fock is essentially diagonal ({off:.3e}); then feeding \
         canonical eps_occ alongside a localized b_ov would be harmless and this \
         caveat would not need recording"
    );
}
