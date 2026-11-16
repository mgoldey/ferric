//! Open-shell attenuated and spin-component-scaled RI-MP2
//! (`ri_mp2_spin_components_any`; theory.md section 5.1).
//!
//! * Singlet anchor: a closed-shell water RHF written as a UHF (α ≡ β) runs
//!   the unrestricted path of every driver, and each must equal the
//!   restricted result to 1e-10 Ha.
//! * Operator limit ω → 0 of erfc-attenuated MP2 on a genuine open shell
//!   (OH doublet).
//!
//! The open-shell values against PySCF are in
//! `validation_u_attenuated_pyscf.rs`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::attenuated::{
    attenuated_ri_mp2, attenuated_ri_mp2_long_range, AttenuatedMp2Config,
};
use ferric_mp2::rimp2::RiMp2Config;
use ferric_mp2::scs::{scs_mp2, scs_mp2_2terfc, ScsMp2Config, ScsMp2TerfcConfig};
use ferric_mp2::u_rimp2::{ri_mp2_spin_components_any, u_ri_mp2};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::{ScfResult, Spin};

const WATER: &str = "3\nwater\nO 0.000 0.000 0.118\nH 0.000 0.755 -0.471\nH 0.000 -0.755 -0.471\n";
const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";

fn terf_tables_available() -> bool {
    std::env::var("FERRIC_TERF_TABLE_DIR")
        .map(|d| std::path::Path::new(&d).join("16_4_2.bin").exists())
        .unwrap_or(false)
}

/// The RHF `rhf` written as a UHF with identical α and β orbitals, energies,
/// densities and Fock matrices: a UHF stationary point.
fn lift_to_unrestricted(rhf: &ScfResult) -> ScfResult {
    let mut u = rhf.clone();
    u.spin = Spin::Unrestricted;
    u.density_beta = Some(rhf.density_alpha.clone());
    u.mos_beta = Some(rhf.mos_alpha.clone());
    u.eps_beta = Some(rhf.eps_alpha.clone());
    u.fock_beta = Some(rhf.fock_alpha.clone());
    u
}

fn water() -> (Molecule, PreparedBasis, PreparedBasis, ScfResult) {
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-11,
            density_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(rhf.converged);
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    (mol, obs, dfbs, rhf)
}

/// Bar for every singlet comparison. The orbitals are identical on both
/// sides; the U path sums the αα, ββ and αβ blocks with different kernels
/// and in a different order from the restricted kernel, so the difference is
/// rounding, O(ε · Σ|terms|) with Σ|terms| ≈ |E_corr| ≈ 0.3 Ha: about 1e-16 ·
/// 0.3 · (number of partial sums ≲ 1e3) ≈ 3e-14. 1e-10 leaves margin for a
/// different BLAS blocking and stays far below any spin-factor error
/// (≥ 1e-2 Ha here).
const SINGLET_BAR: f64 = 1e-10;

fn close(label: &str, r: f64, u: f64) {
    let d = (r - u).abs();
    eprintln!("{label}: R {r:.12} U {u:.12} |diff| {d:.3e}");
    assert!(
        d <= SINGLET_BAR,
        "{label}: |R - U| = {d:.3e} > {SINGLET_BAR:e}"
    );
}

/// EXACTNESS ANCHOR (singlet limit) for att-rimp2 (erfc and erf), scs-mp2
/// and the terfc spin components, on the unrestricted path.
#[test]
fn open_shell_path_matches_restricted_in_the_singlet_limit() {
    let (mol, obs, dfbs, rhf) = water();
    let uhf = lift_to_unrestricted(&rhf);

    let att = AttenuatedMp2Config::default();
    let (r, u) = (
        attenuated_ri_mp2(&mol, &obs, &dfbs, &rhf, &att).unwrap(),
        attenuated_ri_mp2(&mol, &obs, &dfbs, &uhf, &att).unwrap(),
    );
    close("erfc e_os", r.spin_components.e_os, u.spin_components.e_os);
    close("erfc e_ss", r.spin_components.e_ss, u.spin_components.e_ss);
    close("erfc total", r.total_energy, u.total_energy);

    let (r, u) = (
        attenuated_ri_mp2_long_range(&mol, &obs, &dfbs, &rhf, &att).unwrap(),
        attenuated_ri_mp2_long_range(&mol, &obs, &dfbs, &uhf, &att).unwrap(),
    );
    close("erf e_os", r.spin_components.e_os, u.spin_components.e_os);
    close("erf e_ss", r.spin_components.e_ss, u.spin_components.e_ss);

    let scs = ScsMp2Config::default();
    let (r, u) = (
        scs_mp2(&mol, &obs, &dfbs, &rhf, &scs).unwrap(),
        scs_mp2(&mol, &obs, &dfbs, &uhf, &scs).unwrap(),
    );
    close("scs e_os", r.e_os, u.e_os);
    close("scs e_ss", r.e_ss, u.e_ss);
    close("scs total", r.total_energy, u.total_energy);

    if !terf_tables_available() {
        eprintln!("SKIP terfc half: FERRIC_TERF_TABLE_DIR tables absent");
        return;
    }
    let ri = RiMp2Config::default();
    let op = Operator::terfc(1.55 * ferric_core::units::ANGSTROM_TO_BOHR);
    let (r, u) = (
        ri_mp2_spin_components_any(&mol, &obs, &dfbs, op, &rhf, &ri).unwrap(),
        ri_mp2_spin_components_any(&mol, &obs, &dfbs, op, &uhf, &ri).unwrap(),
    );
    close("terfc e_os", r.e_os, u.e_os);
    close("terfc e_ss", r.e_ss, u.e_ss);
    let cfg = ScsMp2TerfcConfig::default();
    let (r, u) = (
        scs_mp2_2terfc(&mol, &obs, &dfbs, &rhf, &cfg).unwrap(),
        scs_mp2_2terfc(&mol, &obs, &dfbs, &uhf, &cfg).unwrap(),
    );
    close("2terfc e_os", r.e_os, u.e_os);
    close("2terfc e_ss", r.e_ss, u.e_ss);
    close("2terfc total", r.total_energy, u.total_energy);
}

/// The open-shell split is E_OS = E_αβ, E_SS = E_αα + E_ββ, read off the
/// U-RI-MP2 blocks. OH doublet; the SCS combination is assembled by hand.
#[test]
fn open_shell_scs_reads_the_spin_blocks() {
    let (mol, obs, dfbs, uhf) = oh();
    let u = u_ri_mp2(
        &mol,
        &obs,
        &dfbs,
        Operator::coulomb(),
        &uhf,
        &RiMp2Config::default(),
    )
    .unwrap();
    let cfg = ScsMp2Config::default();
    let s = scs_mp2(&mol, &obs, &dfbs, &uhf, &cfg).unwrap();
    let c = &u.components;
    assert_eq!(s.e_os.to_bits(), c.e_ab.to_bits());
    assert_eq!(s.e_ss.to_bits(), (c.e_aa + c.e_bb).to_bits());
    let want = cfg.c_os * c.e_ab + cfg.c_ss * (c.e_aa + c.e_bb);
    assert!(
        (s.scs_corr - want).abs() < 1e-15,
        "{} vs {want}",
        s.scs_corr
    );
    // The blocks differ on a genuine open shell (else the test is blind to a
    // swapped αα/ββ or a closed-shell transfer).
    assert!(
        (c.e_aa - c.e_bb).abs() > 1e-3,
        "aa {} bb {}",
        c.e_aa,
        c.e_bb
    );
}

fn oh() -> (Molecule, PreparedBasis, PreparedBasis, ScfResult) {
    let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let uhf = ferric_scf::uhf::solve_uhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-11,
            density_conv: 1e-9,
            max_iter: 300,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(uhf.converged);
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    (mol, obs, dfbs, uhf)
}

/// ω → 0: erfc(ωr)/r → 1/r, so erfc-attenuated U-MP2 → U-RI-MP2 (OH doublet).
///
/// erfc = 1/r − erf(ωr)/r and erf(ωr)/r = (2ω/√π)(1 − ω²r²/3 + …). The
/// leading term (2ω/√π)·S·S vanishes for (ia|jb) (orthogonal orbitals), but
/// not in the fitting metric, whose aux functions carry charge; the fitted
/// difference is O(ω) times the squared charge error of the Coulomb fit, plus
/// O(ω³) from the dipole term. So the gap must close at least linearly:
/// a tenfold smaller ω must shrink it at least fivefold, and at ω = 1e-4 it
/// must be below 1e-7 Ha. A wrong operator in the U branch (Coulomb, or erf)
/// leaves an ω-independent gap and fails.
#[test]
fn open_shell_erfc_reduces_to_ump2_as_omega_goes_to_zero() {
    let (mol, obs, dfbs, uhf) = oh();
    let ump2 = u_ri_mp2(
        &mol,
        &obs,
        &dfbs,
        Operator::coulomb(),
        &uhf,
        &RiMp2Config::default(),
    )
    .unwrap()
    .mp2_corr;
    let gap = |omega: f64| {
        let r = attenuated_ri_mp2(
            &mol,
            &obs,
            &dfbs,
            &uhf,
            &AttenuatedMp2Config {
                omega,
                ..Default::default()
            },
        )
        .unwrap();
        (r.mp2_corr - ump2).abs()
    };
    let (g3, g4) = (gap(1e-3), gap(1e-4));
    let g_default = gap(AttenuatedMp2Config::default().omega);
    eprintln!("UMP2 {ump2:.12}: |gap| w=1e-3 {g3:.3e}, w=1e-4 {g4:.3e}, w=0.222 {g_default:.3e}");
    assert!(g4 < 1e-7, "omega = 1e-4 gap {g4:.3e}");
    assert!(
        g4 * 5.0 <= g3 || g3 < 1e-12,
        "gap not closing: {g3:.3e} -> {g4:.3e}"
    );
    // Reachability: at the production omega the attenuation is a real effect.
    assert!(g_default > 1e-3, "production omega gap {g_default:.3e}");
}
