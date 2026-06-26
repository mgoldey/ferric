//! Energy impact of QQR-3 erfc screening on the SR-MP2 spin-component energy.
//! Compares the screened compute_rpa_intermediates b_ov (and its E_OS) against a
//! dense erfc reference on a small alkane that fires screening (C6/C8). Fast: a
//! single SCF + two integral builds, no decane.
//!
//! Run: cargo run --release -p ferric-mp2 --example sr_screen_energy

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ferric_mp2::mo_transform::transform_3center_ov;
use ferric_mp2::rimp2::{
    compute_rpa_intermediates, metric_inverse_sqrt, spin_components_from_b_ov, RiMp2Config,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn main() {
    for n in [6usize, 8] {
        let path = format!("../../testdata/molecules/alkane_{n}.xyz");
        let mol = match Molecule::load_xyz(&path).or_else(|_| Molecule::load_xyz(&format!("testdata/molecules/alkane_{n}.xyz"))) {
            Ok(m) => m,
            Err(e) => { eprintln!("skip C{n}: {e}"); continue; }
        };
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op_c = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op_c, &obs).unwrap();
        let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op_c, &bounds,
            &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();

        let op = Operator::erfc(0.222);
        let cfg = RiMp2Config::default();
        let nocc = mol.nelec() as usize / 2;
        let nvir = obs.nbasis() - nocc;
        let naux = dfbs.nbasis();
        let c = rhf.mos_r();
        let c_occ = c.slice(ndarray::s![.., ..nocc]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc..]).to_owned();

        // Dense erfc reference b_ov.
        let v2c = threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
        let v_is = metric_inverse_sqrt(&v2c, op).unwrap();
        let eri3_ao = threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let eri3_ov = transform_3center_ov(&eri3_ao, &c_occ, &c_vir);
        let dense = v_is.dot(&eri3_ov.into_shape_with_order((naux, nocc * nvir)).unwrap());

        // Screened (production) b_ov via compute_rpa_intermediates.
        let inter = compute_rpa_intermediates(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();

        let bmax = dense.iter().zip(inter.b_ov.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        let eps = rhf.eps_r();
        let sc_d = spin_components_from_b_ov(&dense, eps, nocc, nvir, 0, nocc);
        let sc_s = spin_components_from_b_ov(&inter.b_ov, eps, nocc, nvir, 0, nocc);
        println!(
            "C{n}: b_ov maxdiff={bmax:.3e}  E_corr[erfc] dense={:.10} screened={:.10}  ΔE={:.3e} Ha",
            sc_d.e_total, sc_s.e_total, sc_s.e_total - sc_d.e_total,
        );
    }
}
