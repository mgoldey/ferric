//! Shared fixture for the localized-occupied DLPNO-MP2 tests (#475): a real
//! SCF + RI transform, Boys-localized (core and valence separately) occupieds,
//! and the rotated RI tensor / Fock block that
//! `ferric_mp2::dlpno_localized` consumes.
#![allow(dead_code)]

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::{dipole, overlap};
use ferric_integrals::operator::Operator;
use ferric_mp2::dlpno_localized::{
    localize_occupied_boys_blocks, localized_occupied_fock, rotate_b_ov_occupied,
};
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{s, Array2};

pub struct LocFixture {
    /// Canonical dressed RI tensor `(naux, nocc*nvir)` and canonical energies.
    pub b_ov: Array2<f64>,
    pub eps_occ: Vec<f64>,
    pub eps_vir: Vec<f64>,
    /// Occupied rotation, rotated tensor, rotated Fock block, Boys centers.
    pub u: Array2<f64>,
    pub b_loc: Array2<f64>,
    pub f_loc: Array2<f64>,
    pub centers: Array2<f64>,
    pub nocc: usize,
    pub nvir: usize,
    /// Production RI-MP2 correlation energy of the same system.
    pub e_ri_mp2: f64,
}

pub fn load(xyz: &str, obs_name: &str, aux_name: &str, frozen_core: bool) -> LocFixture {
    let mol = Molecule::load_xyz(xyz).unwrap_or_else(|e| panic!("load {xyz}: {e}"));
    let ctx = ParallelContext::new();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    assert!(rhf.converged, "premise: RHF must converge");

    let ncore = mol.atoms.iter().filter(|a| a.z > 2).count();
    let fc = if frozen_core { ncore } else { 0 };
    let cfg = RiMp2Config {
        frozen_core: fc,
        ..RiMp2Config::default()
    };
    let (sc, b_ov) = ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let nocc_total = mol.nelec() as usize / 2;
    let nocc = ferric_mp2::rimp2::active_occ(nocc_total, fc).unwrap();
    let nvir = obs.nbasis() - nocc_total;
    assert_eq!(b_ov.ncols(), nocc * nvir);

    let eps = rhf.eps_r().to_vec();
    let c = rhf.mos_r();
    let c_occ = c.slice(s![.., fc..nocc_total]).to_owned();
    let dip = dipole(&obs, [0.0, 0.0, 0.0]).unwrap();
    #[allow(clippy::single_range_in_vec_init)]
    let blocks: Vec<std::ops::Range<usize>> = if frozen_core || ncore == 0 {
        vec![0..nocc]
    } else {
        vec![0..ncore, ncore..nocc]
    };
    let (u, centers) =
        localize_occupied_boys_blocks(&c_occ, &overlap(&obs), &dip, &blocks).unwrap();
    let eps_occ = eps[fc..nocc_total].to_vec();
    let b_loc = rotate_b_ov_occupied(&b_ov, &u, nocc, nvir).unwrap();
    let f_loc = localized_occupied_fock(&eps_occ, &u);
    LocFixture {
        b_ov,
        eps_occ,
        eps_vir: eps[nocc_total..].to_vec(),
        u,
        b_loc,
        f_loc,
        centers,
        nocc,
        nvir,
        e_ri_mp2: sc.e_total,
    }
}
