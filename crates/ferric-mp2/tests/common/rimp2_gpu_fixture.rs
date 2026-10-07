//! Shared by the RI-MP2 device tests: a real RHF + RI-MP2 reference whose
//! `b_ov` and orbital energies every arm compares on.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config, SpinComponents};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// The SCF settings: RI-JK (def2-universal-jkfit) with loose integral threshold
/// and convergence, so the reference SCF is cheap. Every arm of a test compares
/// on the SAME B_ov and orbital energies this SCF produces, so only their mutual
/// consistency matters, not how converged the orbitals are against an
/// exact-integral SCF.
fn scf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 200,
        energy_conv: 1e-8,
        density_conv: 1e-6,
        integral_thresh: 1e-9,
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    }
}

pub struct Prepared {
    pub b_ov: Array2<f64>,
    pub eps: Vec<f64>,
    pub nocc: usize,
    pub nvir: usize,
    pub first_occ: usize,
    pub nocc_total: usize,
}

/// RHF + the CPU RI-MP2 reference for `sys` in `obs_name`/`aux_name`. The CPU
/// reference runs inside a 1-thread rayon pool, where the dispatcher declines
/// the device, so it is the CPU energy whatever the GPU mode is.
pub fn prepare_scf(
    sys: &str,
    obs_name: &str,
    aux_name: &str,
    frozen: usize,
) -> (Prepared, SpinComponents) {
    let root = env!("CARGO_MANIFEST_DIR");
    // testdata/molecules/{sys}.xyz first, then testdata/molecules/validation/
    let dir = format!("{root}/../../testdata/molecules");
    let path = [
        format!("{dir}/{sys}.xyz"),
        format!("{dir}/validation/{sys}.xyz"),
    ]
    .into_iter()
    .find(|p| std::path::Path::new(p).exists())
    .unwrap_or_else(|| panic!("no geometry for {sys} under {dir}"));
    let mol = Molecule::load_xyz(&path).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &scf_config(),
    )
    .unwrap();
    assert!(rhf.converged, "{sys}: RHF did not converge");
    let cfg = RiMp2Config {
        frozen_core: frozen,
        ..Default::default()
    };
    let (cpu, b_ov) = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap());
    let nocc_total = (mol.nelec() as usize) / 2;
    let nocc = nocc_total - frozen;
    let nvir = obs.nbasis() - nocc_total;
    (
        Prepared {
            b_ov,
            eps: rhf.eps_r().to_vec(),
            nocc,
            nvir,
            first_occ: frozen,
            nocc_total,
        },
        cpu,
    )
}
