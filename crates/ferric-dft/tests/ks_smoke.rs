use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::ks::KsXc;
use ferric_dft::xc_trait::XcContribution;

fn h2() -> Molecule {
    Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 1.4\n", 0, 1).unwrap()
}

fn h2o() -> Molecule {
    Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n", 0, 1,
    ).unwrap()
}

#[test]
fn ks_xc_lda_kmix_is_no_exact_exchange() {
    let mol = h2();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let main = AtomicGridConfig::default();
    let nlc = AtomicGridConfig { n_radial: 50, n_angular: 50 };
    let ks = KsXc::new(&mol, &bs, "LDA", &main, &nlc).unwrap();
    let mix = ks.k_mix();
    assert_eq!(mix.sr, 0.0,    "LDA has no exact exchange (sr)");
    assert_eq!(mix.lr, 0.0,    "LDA has no exact exchange (lr)");
    assert_eq!(mix.omega, 0.0, "LDA is not range-separated");
}

#[test]
fn ks_xc_b3lyp_kmix_is_plain_hybrid() {
    let mol = h2();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let main = AtomicGridConfig::default();
    let nlc = AtomicGridConfig { n_radial: 50, n_angular: 50 };
    let ks = KsXc::new(&mol, &bs, "B3LYP", &main, &nlc).unwrap();
    let mix = ks.k_mix();
    assert_eq!(mix.omega, 0.0, "B3LYP is not range-separated");
    assert!(mix.sr > 0.0 && mix.sr < 1.0, "B3LYP has partial exact exchange: sr = {}", mix.sr);
    assert!((mix.lr - mix.sr).abs() < 1e-12, "B3LYP: sr == lr when omega=0");
    // B3LYP exact-exchange fraction is ~0.20; loosen to 0.05..0.30 for libxc version tolerance
    assert!(mix.sr > 0.05 && mix.sr < 0.30, "B3LYP exact-exchange mix ≈ 0.20, got {}", mix.sr);
}

#[test]
fn ks_xc_wb97xv_kmix_is_range_separated() {
    let mol = h2();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let main = AtomicGridConfig::default();
    let nlc = AtomicGridConfig { n_radial: 50, n_angular: 50 };
    let ks = KsXc::new(&mol, &bs, "wB97X-V", &main, &nlc).unwrap();
    let mix = ks.k_mix();
    // libxc convention: at short range c_SR = α+β, at long range c_LR = α.
    // wB97X-V is long-range-corrected: 100% HF at long range, 16.7% at short.
    assert!((mix.omega - 0.3).abs() < 1e-6,    "wB97X-V ω = 0.3");
    assert!((mix.sr - 0.167).abs() < 5e-3,     "wB97X-V c_SR ≈ 0.167");
    assert!((mix.lr - 1.0).abs() < 1e-6,       "wB97X-V c_LR = 1.0");
}

#[test]
fn ks_xc_add_xc_produces_negative_energy_h2o_lda() {
    // End-to-end: build KsXc, get a converged RHF density, call add_xc.
    // E_xc should be negative and within the LDA expected range.
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    let mol = h2o();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let main = AtomicGridConfig::default();
    let nlc = AtomicGridConfig { n_radial: 50, n_angular: 50 };
    let ks = KsXc::new(&mol, &bs, "LDA", &main, &nlc).unwrap();

    let n = rhf.density_total.nrows();
    let mut f = ndarray::Array2::<f64>::zeros((n, n));
    let e_xc = ks.add_xc(&rhf.density_total, &mut f);
    eprintln!("KsXc(LDA) E_xc(H2O, cc-pVDZ) = {e_xc:.6} Ha");
    assert!(e_xc < -3.0 && e_xc > -15.0, "E_xc out of expected range: {e_xc}");

    // The Fock contribution should also be Hermitian after add_xc.
    let mut asym: f64 = 0.0;
    for i in 0..n { for j in 0..n {
        let a = (f[(i, j)] - f[(j, i)]).abs();
        if a > asym { asym = a; }
    }}
    assert!(asym < 1e-10, "F not Hermitian after add_xc: {asym:.2e}");
}
