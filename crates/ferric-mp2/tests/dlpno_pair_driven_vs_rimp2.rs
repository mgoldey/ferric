//! Pair-driven DLPNO-MP2 on a REAL molecule: same energy as production RI-MP2,
//! without ever allocating the dense `(ia|jb)` matrix.
//!
//! # What this closes
//!
//! `ferric_mp2::dlpno_mp2::dlpno_mp2_spin_components` takes the dense
//! `(nocc*nvir) x (nocc*nvir)` matrix `g` as its INPUT, while production
//! `ri_mp2` never forms that object — `rimp2::spin_components_from_b_ov` streams
//! i-blocked wide GEMMs instead. So DLPNO's screening could never help: the
//! allocation it was supposed to avoid happened before any threshold applied.
//! MEASURED at a drug-scale shape (nocc=170, nvir=760, def2-SVP) the dense `g`
//! is 124 GiB against a 23 GB box, so DLPNO-MP2 could not START on a system
//! production RI-MP2 completes.
//!
//! `dlpno_mp2_from_b_ov` screens pairs first and then builds only each retained
//! pair's `nvir x nvir` block from the SAME `b_ov` tensor `ri_mp2` already
//! returns. This test drives it end to end on genuine integrals — real RHF
//! orbitals, real RI three-index tensors — rather than on a synthetic `b_ov`,
//! because a construction bug is deterministic and would reproduce perfectly on
//! a toy (see the project's "consistency is not corroboration" rule).
//!
//! # The claim, and its limits
//!
//! Asserted here: untruncated pair-driven DLPNO-MP2 reproduces `ri_mp2`'s
//! correlation energy on water/cc-pVDZ to the RI round-off floor, and the
//! pair-energy screen computed from `b_ov` reproduces the one computed from the
//! dense `g`.
//!
//! NOT asserted here: any speedup. The structural/memory result is what this
//! test covers; a wall-clock claim needs a quiet box and a size sweep.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::dlpno_mp2::{
    dense_g_bytes, dlpno_mp2_from_b_ov, dlpno_mp2_spin_components,
    estimate_pair_energies_from_b_ov, pair_driven_working_bytes, DlpnoConfig,
};
use ferric_mp2::pair_domains::complete_pair_domains;
use ferric_mp2::pair_energy_screen::estimate_pair_energies;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// Water at a standard geometry, in Angstrom (the parser converts to Bohr).
fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 -0.757 0.587\nH 0.0 0.757 0.587\n",
        0,
        1,
    )
    .unwrap()
}

/// Everything the pair-driven path needs, from a real SCF + RI transform.
struct Setup {
    b_ov: Array2<f64>,
    eps: Vec<f64>,
    nocc: usize,
    nvir: usize,
    nocc_total: usize,
    /// Production RI-MP2 correlation energy, from `spin_components_from_b_ov`.
    e_corr_ri: f64,
}

fn run_scf_and_ri() -> Setup {
    let ctx = ParallelContext::new();
    let mol = water();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    assert!(rhf.converged, "premise: the RHF reference must converge");

    let cfg = RiMp2Config::default();
    let (sc, b_ov) = ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();

    let nocc_total = mol.nelec() as usize / 2;
    let nocc = nocc_total - cfg.frozen_core;
    let nvir = obs.nbasis() - nocc_total;
    assert_eq!(
        b_ov.ncols(),
        nocc * nvir,
        "premise: b_ov must be (naux, nocc*nvir)"
    );

    Setup {
        b_ov,
        eps: rhf.eps_r().to_vec(),
        nocc,
        nvir,
        nocc_total,
        e_corr_ri: sc.e_total,
    }
}

/// Boys centers are only used for the DISTANCE screen and for diagnostics. With
/// complete domains nothing reads them, so the atom-free placeholder below is
/// honest rather than a stand-in for a localization we did not run.
fn placeholder_centers(nocc: usize) -> Array2<f64> {
    Array2::zeros((nocc, 3))
}

/// THE ANCHOR, on real integrals: untruncated pair-driven DLPNO-MP2 == RI-MP2.
///
/// This is the test that has to pass before any performance work, per the
/// project's experimental protocol. The approximation's trivial limit
/// (`DlpnoConfig::exact()` — every screen off) must do nothing at all.
#[test]
fn pair_driven_dlpno_reproduces_ri_mp2_on_water() {
    let s = run_scf_and_ri();
    let d = complete_pair_domains(&placeholder_centers(s.nocc)).unwrap();
    assert!(d.is_complete(), "premise: domains must be unscreened");

    let (got, diag) = dlpno_mp2_from_b_ov(
        &s.b_ov,
        &s.eps,
        s.nocc,
        s.nvir,
        0,
        s.nocc_total,
        &d,
        &DlpnoConfig::exact(),
    )
    .unwrap();

    let d_e = got.e_total - s.e_corr_ri;
    eprintln!(
        "water/cc-pVDZ  nocc={} nvir={}\n  ri_mp2            E_corr = {:.14} Ha\n  \
         pair-driven DLPNO E_corr = {:.14} Ha\n  difference               = {d_e:+.3e} Ha\n  \
         pair retention {:.3}, virtual retention {:.3}",
        s.nocc, s.nvir, s.e_corr_ri, got.e_total, diag.pair_retention, diag.virtual_retention
    );

    assert_eq!(diag.pair_retention, 1.0, "exact() must retain every pair");
    assert_eq!(
        diag.virtual_retention, 1.0,
        "exact() must retain every virtual"
    );
    // The two routes differ only in GEMM shape (one big contraction vs one per
    // pair), so they agree to double-precision round-off, not bit for bit.
    assert!(
        d_e.abs() < 1e-12,
        "untruncated pair-driven DLPNO must reproduce RI-MP2: {:.14} vs {:.14} (d = {d_e:.3e})",
        got.e_total,
        s.e_corr_ri
    );
    // Guard the guard: the correlation energy must be big enough that 1e-12 is a
    // real bar and not satisfied by two near-zero numbers.
    assert!(
        s.e_corr_ri.abs() > 1e-3,
        "premise: E_corr must be non-trivial, got {:.3e}",
        s.e_corr_ri
    );
}

/// The pair-driven path must agree with the EXISTING dense DLPNO path, on real
/// integrals, at the same truncation.
///
/// `pair_driven_matches_dense` (unit test) pins the kernel-level bit-identity on
/// a synthetic tensor. This is the independent construction: real RHF orbitals
/// and a real RI fit, where the dense `g` is formed only because water/cc-pVDZ is
/// small enough that it still can be.
#[test]
fn pair_driven_agrees_with_dense_dlpno_on_water() {
    let s = run_scf_and_ri();
    let d = complete_pair_domains(&placeholder_centers(s.nocc)).unwrap();
    let cfg = DlpnoConfig::exact();

    // The dense path's input — the object the restructuring exists to avoid.
    let g = s.b_ov.t().dot(&s.b_ov);
    let (dense, _) =
        dlpno_mp2_spin_components(&g, &s.eps, s.nocc, s.nvir, 0, s.nocc_total, &d, &cfg).unwrap();
    let (paired, _) =
        dlpno_mp2_from_b_ov(&s.b_ov, &s.eps, s.nocc, s.nvir, 0, s.nocc_total, &d, &cfg).unwrap();

    eprintln!(
        "dense DLPNO  e_os={:.14} e_ss={:.14}\npaired DLPNO e_os={:.14} e_ss={:.14}\n  \
         dOS={:+.3e} dSS={:+.3e}",
        dense.e_os,
        dense.e_ss,
        paired.e_os,
        paired.e_ss,
        paired.e_os - dense.e_os,
        paired.e_ss - dense.e_ss
    );
    assert!((paired.e_os - dense.e_os).abs() < 1e-12, "OS differs");
    assert!((paired.e_ss - dense.e_ss).abs() < 1e-12, "SS differs");
}

/// The pair-energy screen — the one `DlpnoConfig::default()` enables — must be
/// computable WITHOUT the dense matrix.
///
/// This mattered as much as the energy kernel: `estimate_pair_energies` takes
/// the dense `g`, so deciding which pairs to skip required first paying the
/// allocation the skipping exists to avoid.
#[test]
fn pair_energy_screen_is_computable_from_b_ov_on_water() {
    let s = run_scf_and_ri();
    let g = s.b_ov.t().dot(&s.b_ov);

    let dense = estimate_pair_energies(g.view(), &s.eps, s.nocc, s.nvir, 0, s.nocc_total).unwrap();
    let streamed =
        estimate_pair_energies_from_b_ov(&s.b_ov, &s.eps, s.nocc, s.nvir, 0, s.nocc_total).unwrap();

    let mut worst = 0.0f64;
    for i in 0..s.nocc {
        for j in 0..s.nocc {
            worst = worst.max((streamed.e[(i, j)] - dense.e[(i, j)]).abs());
        }
    }
    eprintln!(
        "pair-energy screen: dense total {:.14}, streamed total {:.14}, worst |de_ij| {worst:.3e}",
        dense.total(),
        streamed.total()
    );
    assert!(
        worst < 1e-12,
        "streamed pair energies differ by {worst:.3e}"
    );
    // The estimator's total IS the MP2 correlation energy, so it also anchors
    // against the production number rather than only against its dense twin.
    assert!(
        (streamed.total() - s.e_corr_ri).abs() < 1e-10,
        "pair energies must sum to E_corr: {:.14} vs {:.14}",
        streamed.total(),
        s.e_corr_ri
    );
}

/// The memory argument, at the shape that motivated the change.
///
/// Reported as a counted allocation rather than peak RSS: the dense `g` at this
/// shape cannot be allocated on this box at all, so the number that matters is
/// the one that says why.
#[test]
fn dense_g_is_unallocatable_where_pair_driven_is_free() {
    // danuglipron + capped SER31 at def2-SVP.
    let (nocc, nvir) = (170usize, 760usize);
    let gib = |b: usize| b as f64 / 1024.0f64.powi(3);
    let dense = gib(dense_g_bytes(nocc, nvir));
    let paired = gib(pair_driven_working_bytes(nvir));
    eprintln!(
        "nocc={nocc} nvir={nvir}: dense g = {dense:.1} GiB, pair-driven per-pair working set \
         = {paired:.4} GiB ({:.0}x smaller). Box is 23 GB.",
        dense / paired
    );
    assert!(
        dense > 23.0,
        "premise: the dense path must exceed this box's RAM ({dense:.1} GiB)"
    );
    assert!(
        paired < 1.0,
        "the pair-driven working set must fit comfortably ({paired:.4} GiB)"
    );

    // And at water/cc-pVDZ, where both still run, the same formulas must agree
    // with the arrays actually built — so the counters are not fiction.
    let s = run_scf_and_ri();
    let g = s.b_ov.t().dot(&s.b_ov);
    assert_eq!(
        g.len() * 8,
        dense_g_bytes(s.nocc, s.nvir),
        "dense_g_bytes must match the array that was really allocated"
    );
}
