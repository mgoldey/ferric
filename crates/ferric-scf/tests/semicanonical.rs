//! ROHF → semi-canonical orbital tests.
//!
//! Semi-canonicalization has sharp, checkable properties, and these tests pin each one:
//!
//!   1. Orthonormality is preserved (the rotation is orthogonal).
//!   2. The occupied SPAN is preserved (block-diagonal ⇒ same reference determinant).
//!   3. F_σ becomes diagonal WITHIN the occ and virt blocks.
//!   4. The occ–virt block does NOT vanish (that is what makes it "semi").
//!   5. Orbital energies equal the diagonal of F_σ in the new basis.
//!
//! Together these are close to a complete specification: a rotation satisfying all five
//! is semi-canonical almost by definition.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::semicanonical::semicanonicalize;
use ndarray::Array2;

/// OH radical, doublet — the standard open-shell smoke system.
fn oh_radical() -> Molecule {
    Molecule::parse_xyz("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap()
}

struct Fixture {
    mol: Molecule,
    obs: PreparedBasis,
    bounds: SchwarzBounds,
    rohf: ferric_scf::result::ScfResult,
    s: Array2<f64>,
}

fn setup(basis_name: &str) -> Fixture {
    let mol = oh_radical();
    let obs = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let rohf = solve_rohf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &RhfConfig { density_conv: 1e-9, max_iter: 200, ..Default::default() },
    )
    .expect("ROHF on OH should converge");
    assert!(rohf.converged);
    let s = ferric_integrals::oneelectron::overlap(&obs);
    Fixture { mol, obs, bounds, rohf, s }
}

fn run(f: &Fixture) -> ferric_scf::semicanonical::SemicanonicalOrbitals {
    semicanonicalize(
        &ParallelContext::default(),
        &f.mol,
        &f.obs,
        &f.bounds,
        &f.rohf,
        1e-12,
        None,
    )
    .expect("semicanonicalization should succeed on a converged ROHF reference")
}

/// PROPERTY 1 — the rotation is orthogonal, so C^T S C = I is preserved.
#[test]
fn preserves_orthonormality() {
    let f = setup("sto-3g");
    let sc = run(&f);

    for (label, c) in [("alpha", &sc.mos_alpha), ("beta", &sc.mos_beta)] {
        let csc = c.t().dot(&f.s).dot(c);
        let n = csc.nrows();
        let mut max_dev = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { 1.0 } else { 0.0 };
                max_dev = max_dev.max((csc[[i, j]] - want).abs());
            }
        }
        eprintln!("{label}: max |C^T S C - I| = {max_dev:.3e}");
        assert!(max_dev < 1e-10, "{label} MOs are not orthonormal (dev {max_dev:.3e})");
    }
}

/// PROPERTY 2 — the occupied SPAN is unchanged, so the reference determinant is the same.
///
/// The rotation is block-diagonal, mixing occupieds only with occupieds. The occupied
/// density D = C_occ C_occ^T is therefore invariant. This is the property that makes
/// semi-canonicalization free: it changes the orbitals without changing the state.
#[test]
fn preserves_the_occupied_span() {
    let f = setup("sto-3g");
    let sc = run(&f);

    let dens = |c: &Array2<f64>, nocc: usize| -> Array2<f64> {
        let occ = c.slice(ndarray::s![.., ..nocc]);
        occ.dot(&occ.t())
    };

    // ROHF stores one spatial MO set; both spins are rotated from it.
    let c_rohf = &f.rohf.mos_alpha;
    for (label, c_new, nocc) in [
        ("alpha", &sc.mos_alpha, sc.nocc_alpha),
        ("beta", &sc.mos_beta, sc.nocc_beta),
    ] {
        let before = dens(c_rohf, nocc);
        let after = dens(c_new, nocc);
        let max_dev =
            before.iter().zip(after.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        eprintln!("{label}: max |D_before - D_after| = {max_dev:.3e}");
        assert!(
            max_dev < 1e-10,
            "{label} occupied span changed (dev {max_dev:.3e}) -- the rotation is not \
             block-diagonal, so the reference determinant was altered"
        );
    }
}

/// PROPERTIES 3, 4, 5 — the defining behavior.
///
/// F_σ must be diagonal WITHIN occ-occ and virt-virt, its diagonal must equal the
/// reported orbital energies, and the occ–virt block must NOT vanish.
#[test]
fn fock_is_block_diagonal_but_ov_survives() {
    let f = setup("sto-3g");
    let sc = run(&f);

    // Rebuild F_alpha/F_beta the same way the routine does, to inspect them.
    let n = f.obs.nbasis();
    let h = ferric_integrals::oneelectron::hcore(&f.obs);
    let d_a = &f.rohf.density_alpha;
    let d_b = f.rohf.density_beta.as_ref().unwrap();
    let d_tot = d_a + d_b;
    let ctx = ParallelContext::default();

    let (mut j_tot, mut scratch) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    ferric_scf::rhf::build_jk(&ctx, &f.obs, &f.bounds, 1e-12, &d_tot, &mut j_tot, &mut scratch)
        .unwrap();
    let (mut js, mut k_a) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    ferric_scf::rhf::build_jk(&ctx, &f.obs, &f.bounds, 1e-12, d_a, &mut js, &mut k_a).unwrap();
    let mut k_b = Array2::zeros((n, n));
    js.fill(0.0);
    ferric_scf::rhf::build_jk(&ctx, &f.obs, &f.bounds, 1e-12, d_b, &mut js, &mut k_b).unwrap();

    let f_a = &h + &j_tot - &k_a;
    let f_b = &h + &j_tot - &k_b;

    for (label, f_ao, c, eps, nocc, reported_ov) in [
        ("alpha", &f_a, &sc.mos_alpha, &sc.eps_alpha, sc.nocc_alpha, sc.max_ov_alpha),
        ("beta", &f_b, &sc.mos_beta, &sc.eps_beta, sc.nocc_beta, sc.max_ov_beta),
    ] {
        let f_mo = c.t().dot(f_ao).dot(c);
        let nmo = f_mo.nrows();

        // PROPERTY 3: off-diagonal elements WITHIN each block must vanish.
        let mut max_off_block = 0.0f64;
        for i in 0..nocc {
            for j in 0..nocc {
                if i != j {
                    max_off_block = max_off_block.max(f_mo[[i, j]].abs());
                }
            }
        }
        for a in nocc..nmo {
            for b in nocc..nmo {
                if a != b {
                    max_off_block = max_off_block.max(f_mo[[a, b]].abs());
                }
            }
        }

        // PROPERTY 5: the diagonal IS the reported orbital energies.
        let mut max_eps_dev = 0.0f64;
        for p in 0..nmo {
            max_eps_dev = max_eps_dev.max((f_mo[[p, p]] - eps[p]).abs());
        }

        // PROPERTY 4: the occ-virt block survives.
        let mut max_ov = 0.0f64;
        for i in 0..nocc {
            for a in nocc..nmo {
                max_ov = max_ov.max(f_mo[[i, a]].abs());
            }
        }

        eprintln!(
            "{label}: max off-block = {max_off_block:.3e}   max |F_pp - eps_p| = \
             {max_eps_dev:.3e}   max |F_ia| = {max_ov:.3e}"
        );

        assert!(
            max_off_block < 1e-10,
            "{label}: F is not block-diagonal (max off-block {max_off_block:.3e}) -- \
             denominators built from these orbital energies would be invalid"
        );
        assert!(
            max_eps_dev < 1e-10,
            "{label}: reported orbital energies do not match diag(F) (dev {max_eps_dev:.3e})"
        );
        assert!(
            (max_ov - reported_ov).abs() < 1e-10,
            "{label}: reported max_ov {reported_ov:.3e} != measured {max_ov:.3e}"
        );
        assert!(
            max_ov > 1e-8,
            "{label}: the occ-virt block VANISHED (max {max_ov:.3e}). For a ROHF \
             reference F_sigma's ov block is non-zero -- if it is zero here, the wrong \
             Fock operator was built (e.g. the effective Roothaan one, whose ov block \
             ROHF stationarity does annihilate)"
        );
    }
}

/// Orbital energies must be ordered within each block, as diagonalization guarantees.
#[test]
fn orbital_energies_are_ordered_within_blocks() {
    let f = setup("cc-pvdz");
    let sc = run(&f);

    for (label, eps, nocc) in
        [("alpha", &sc.eps_alpha, sc.nocc_alpha), ("beta", &sc.eps_beta, sc.nocc_beta)]
    {
        for i in 1..nocc {
            assert!(
                eps[i] >= eps[i - 1] - 1e-12,
                "{label}: occupied energies not ascending at {i}: {} < {}",
                eps[i],
                eps[i - 1]
            );
        }
        for a in nocc + 1..eps.len() {
            assert!(
                eps[a] >= eps[a - 1] - 1e-12,
                "{label}: virtual energies not ascending at {a}: {} < {}",
                eps[a],
                eps[a - 1]
            );
        }
        // The open shell makes alpha and beta genuinely different -- that is the whole
        // point of doing this per spin.
        eprintln!("{label}: HOMO = {:.6}, LUMO = {:.6}", eps[nocc - 1], eps[nocc]);
    }
    assert!(
        sc.nocc_alpha > sc.nocc_beta,
        "OH doublet must have more alpha than beta electrons"
    );
    let differ = sc
        .eps_alpha
        .iter()
        .zip(sc.eps_beta.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        differ,
        "alpha and beta orbital energies are identical -- the spin-dependent exchange \
         (K[D_sigma]) is not reaching the Fock build"
    );
}

/// Bad inputs must be refused, not silently approximated.
#[test]
fn invalid_references_are_rejected() {
    let f = setup("sto-3g");
    let ctx = ParallelContext::default();

    // A restricted (closed-shell) reference has no open shell.
    let mol_h2o = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/water.xyz"
    ))
    .unwrap();
    let obs_h2o = PreparedBasis::new(&mol_h2o, &basis::bundled("sto-3g").unwrap()).unwrap();
    let bounds_h2o = SchwarzBounds::compute(Operator::coulomb(), &obs_h2o).unwrap();
    let rhf = ferric_scf::rhf::solve_rhf(
        &ctx,
        &mol_h2o,
        &obs_h2o,
        Operator::coulomb(),
        &bounds_h2o,
        &RhfConfig { density_conv: 1e-9, ..Default::default() },
    )
    .unwrap();
    assert!(
        semicanonicalize(&ctx, &mol_h2o, &obs_h2o, &bounds_h2o, &rhf, 1e-12, None).is_err(),
        "a restricted reference must be rejected"
    );

    // An unknown functional name must error rather than silently fall back to HF.
    let bad = ferric_scf::semicanonical::XcSpec::new("NOT_A_REAL_FUNCTIONAL");
    assert!(
        semicanonicalize(&ctx, &f.mol, &f.obs, &f.bounds, &f.rohf, 1e-12, Some(&bad)).is_err(),
        "an unresolvable functional name must be an error, not a silent HF fallback"
    );
}

/// The conversion to a UHF-shaped result must carry genuine per-spin data.
///
/// This is the practical payoff: ferric's open-shell post-SCF code detects a ROHF
/// result and falls back to alpha orbitals with the EFFECTIVE Fock's eigenvalues for
/// both spins (`u_rimp2.rs:97`: "ROHF has no eps_beta -- fall back to eps_alpha").
/// After conversion there is a real eps_beta, so that fallback no longer fires.
#[test]
fn converts_to_a_usable_unrestricted_result() {
    let f = setup("sto-3g");
    let sc = run(&f);
    let u = sc.to_unrestricted_result(&f.rohf);

    assert!(matches!(u.spin, ferric_scf::result::Spin::Unrestricted));
    assert!(u.converged);
    assert!(u.mos_beta.is_some(), "converted result must carry beta MOs");
    let eps_b = u.eps_beta.as_ref().expect("converted result must carry beta eigenvalues");

    // The whole point: eps_beta exists AND differs from eps_alpha.
    let differ = u.eps_alpha.iter().zip(eps_b.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        differ,
        "eps_alpha and eps_beta are identical -- the ROHF fallback this conversion \
         exists to remove is still effectively in force"
    );

    // The reference determinant is unchanged, so the energy carries over exactly.
    assert!(
        (u.energy - f.rohf.energy).abs() < 1e-14,
        "the block-diagonal rotation preserves the occupied span, so the SCF energy \
         must be identical: {:.12} vs {:.12}",
        u.energy,
        f.rohf.energy
    );

    // Total electron count must be preserved: tr(D S) = nelec.
    let n_elec: f64 = (0..u.density_total.nrows())
        .map(|i| (0..u.density_total.ncols()).map(|j| u.density_total[[i, j]] * f.s[[j, i]]).sum::<f64>())
        .sum();
    let want = f.mol.nelec() as f64;
    eprintln!("tr(D S) = {n_elec:.10}  (expected {want})");
    assert!(
        (n_elec - want).abs() < 1e-9,
        "converted density has {n_elec:.6} electrons, expected {want}"
    );
}

/// THE XC-THREADING TEST — a Kohn-Sham F_sigma must differ from the HF one.
///
/// Before XC threading this routine always built a Hartree-Fock F_sigma, so a ROKS
/// reference silently received HF-like orbital energies. Passing an `XcSpec` must
/// visibly change the result; if HF and KS agreed, the XC potential would not be
/// reaching the Fock build.
#[test]
fn kohn_sham_fock_differs_from_hartree_fock() {
    use ferric_scf::semicanonical::XcSpec;
    let f = setup("sto-3g");
    let ctx = ParallelContext::default();

    let hf = run(&f);
    for name in ["PBE", "B3LYP", "wB97X-L-V"] {
        let spec = XcSpec::new(name);
        let ks = semicanonicalize(&ctx, &f.mol, &f.obs, &f.bounds, &f.rohf, 1e-12, Some(&spec))
            .unwrap_or_else(|e| panic!("{name} semicanonicalization failed: {e}"));

        let max_d = hf
            .eps_alpha
            .iter()
            .zip(ks.eps_alpha.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "{name:12} HOMO_a = {:>10.6}  (HF {:>10.6})   max |eps_KS - eps_HF| = {max_d:.3e}",
            ks.eps_alpha[ks.nocc_alpha - 1],
            hf.eps_alpha[hf.nocc_alpha - 1]
        );

        assert!(
            max_d > 1e-4,
            "{name}: KS and HF orbital energies agree to {max_d:.3e} -- the XC potential \
             is not reaching the Fock build"
        );
        // All the defining properties must still hold under XC.
        assert!(ks.eps_alpha.iter().all(|v| v.is_finite()), "{name}: non-finite eps");
        assert!(
            ks.max_ov_alpha > 1e-10,
            "{name}: occ-virt block vanished -- wrong Fock operator"
        );
        let differ =
            ks.eps_alpha.iter().zip(ks.eps_beta.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differ, "{name}: alpha and beta energies identical under XC");
    }
}

/// A range-separated functional must take the SPLIT exchange path, not full K.
///
/// wB97X-L-V has omega = 0.1 with c_sr = 0.6, c_lr = 1.0, so its exchange is
/// c_sr*K_SR + c_lr*K_LR -- materially different from both full K (HF) and from a
/// plain hybrid's scaled K. Comparing against a plain hybrid confirms the RSH branch
/// is distinct rather than silently collapsing to the hybrid one.
#[test]
fn range_separated_exchange_takes_its_own_path() {
    use ferric_scf::semicanonical::XcSpec;
    let f = setup("sto-3g");
    let ctx = ParallelContext::default();

    let run_xc = |name: &str| {
        let spec = XcSpec::new(name);
        semicanonicalize(&ctx, &f.mol, &f.obs, &f.bounds, &f.rohf, 1e-12, Some(&spec)).unwrap()
    };

    let rsh = run_xc("wB97X-L-V");
    let hybrid = run_xc("B3LYP");
    let pure = run_xc("PBE");

    let spread = |a: &[f64], b: &[f64]| {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    };
    eprintln!("|RSH - hybrid| = {:.3e}", spread(&rsh.eps_alpha, &hybrid.eps_alpha));
    eprintln!("|RSH - pure|   = {:.3e}", spread(&rsh.eps_alpha, &pure.eps_alpha));

    assert!(
        spread(&rsh.eps_alpha, &hybrid.eps_alpha) > 1e-3,
        "wB97X-L-V and B3LYP gave near-identical orbital energies -- the RSH exchange \
         split (c_sr*K_SR + c_lr*K_LR) is probably collapsing to the plain-hybrid path"
    );
    assert!(
        spread(&rsh.eps_alpha, &pure.eps_alpha) > 1e-3,
        "wB97X-L-V and PBE gave near-identical orbital energies -- exact exchange is \
         not being applied at all"
    );
}
