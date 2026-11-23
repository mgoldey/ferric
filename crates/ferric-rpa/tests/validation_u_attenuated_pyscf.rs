//! Open-shell attenuated MP2, SCS-MP2 and RS-MP2 + LR-RPA on UHF references,
//! against PySCF and an independent numpy route
//! (`scripts/validation/gen_u_attenuated_pyscf.py`).
//!
//! Systems: OH and NH2 doublets, O2 triplet, cc-pVDZ / cc-pVDZ-RI, at the
//! production ω = 0.420 Å⁻¹. ferric runs on PySCF's UHF orbitals and orbital
//! energies (both spins, written in ferric's AO order), so the SCF is out of
//! the comparison; what is left is the integrals, the metric factorization and
//! the frequency quadrature (both sides 20 Gauss–Legendre nodes, x0 = 0.5).
//!
//! References, per field (see the generator's header):
//! * `numpy_ferric_ints`: numpy U-MP2 blocks and spin-summed U-dRPA on
//!   ferric's own integrals with ferric's metric factorization. Every field.
//! * `pyscf`: PySCF's `DFUMP2` and `URPA` on its own attenuated fit. Only the
//!   fields that use Coulomb and erfc fits (both codes Cholesky-factor those
//!   metrics): att-rimp2 erfc, scs-mp2, and the coupled-rings (T) form.
//!
//! Bars (no measured number is used to set a bar):
//! * vs numpy on the same integrals: `2 fp_floor + FLOOR_ABS`. `fp_floor` is
//!   the field change when every integral moves by a relative 64 ε (the
//!   accuracy of a contracted Gaussian integral), so it bounds the rounding
//!   one code carries; two independent codes carry it twice. `FLOOR_ABS` =
//!   1e-12 Ha covers the dRPA's log-det rounding on ≤ 84 modes × 20 nodes
//!   (84 · 20 · ε · max λ / 2π ≈ 5e-13 for λ ≤ 10).
//! * vs PySCF's code: the bar above plus |integral_shift| (the numpy field on
//!   ferric's integrals minus the same on PySCF's) plus the numpy-vs-PySCF
//!   code residual.
//! * Controls: the closed-shell rule dMP2 = 2 E_αβ carried over must MISS by
//!   more than `MUST_MISS_FACTOR` bars.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::attenuated::{
    attenuated_ri_mp2, attenuated_ri_mp2_long_range, AttenuatedMp2Config,
};
use ferric_mp2::scs::{scs_mp2, ScsMp2Config};
use ferric_rpa::{rs_mp2_lr_rpa, RsMp2RpaConfig, RsMp2RpaFormulation};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::{ScfResult, Spin};
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/u_attenuated_pyscf";
const MOL_DIR: &str = "testdata/molecules/validation";
const FLOOR_ABS: f64 = 1e-12;
const MUST_MISS_FACTOR: f64 = 10.0;
const TOL_ENUC: f64 = 1e-9;

fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("testdata").is_dir() && p.join("crates").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return p.to_path_buf();
            }
            here = p.parent();
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-rpa manifest dir should be <root>/crates/ferric-rpa")
        .to_path_buf()
}

fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_cc-pvdz.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_u_attenuated_pyscf.py",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn num(v: &Value, ptr: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("reference field {ptr} missing or not a number"))
}

/// PySCF's orbitals for spin `tag` in ferric AO order: (C, eps, nocc).
fn spin_orbitals(r: &Value, tag: &str) -> (Array2<f64>, Vec<f64>, usize) {
    let s = &r["uhf_orbitals"][tag];
    let rows = s["mo_coeff_ferric_order"].as_array().expect("mo_coeff");
    let (nao, nmo) = (rows.len(), rows[0].as_array().unwrap().len());
    let c = Array2::from_shape_fn((nao, nmo), |(m, p)| rows[m][p].as_f64().unwrap());
    let eps = s["mo_energy"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    (c, eps, s["nocc"].as_u64().unwrap() as usize)
}

struct Sys {
    label: String,
    r: Value,
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    uhf: ScfResult,
}

fn load(system: &str) -> Sys {
    let r = reference(system);
    let label = format!("{system}/cc-pvdz");
    let mult = r["multiplicity"].as_u64().unwrap() as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    assert!((mol.nuclear_repulsion() - num(&r, "/nuclear_repulsion")).abs() < TOL_ENUC);
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    assert_eq!(
        obs.nbasis() as i64,
        r["nao"].as_i64().unwrap(),
        "{label}: nao"
    );
    assert_eq!(
        dfbs.nbasis() as i64,
        r["naux"].as_i64().unwrap(),
        "{label}: naux"
    );
    assert!(
        (num(&r, "/omega_bohr_inv") - RsMp2RpaConfig::default().omega).abs() < 1e-15
            && (num(&r, "/omega_bohr_inv") - AttenuatedMp2Config::default().omega).abs() < 1e-15,
        "{label}: reference omega is not the production default"
    );
    // ferric's own UHF is only a template for the ScfResult fields the
    // correlation code does not read; orbitals, energies and densities are
    // PySCF's.
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let own = ferric_scf::uhf::solve_uhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        &bounds,
        &RhfConfig::default().with_max_iter(400),
    )
    .unwrap_or_else(|e| panic!("{label}: UHF failed: {e:?}"));
    let (ca, ea, na) = spin_orbitals(&r, "alpha");
    let (cb, eb, nb) = spin_orbitals(&r, "beta");
    let (ne_a, ne_b) = (
        (mol.nelec() as usize + mult - 1) / 2,
        (mol.nelec() as usize + 1 - mult) / 2,
    );
    assert_eq!((na, nb), (ne_a, ne_b), "{label}: occupations");
    let da = ca
        .slice(ndarray::s![.., ..na])
        .dot(&ca.slice(ndarray::s![.., ..na]).t());
    let db = cb
        .slice(ndarray::s![.., ..nb])
        .dot(&cb.slice(ndarray::s![.., ..nb]).t());
    let mut uhf = own;
    assert!(matches!(uhf.spin, Spin::Unrestricted));
    uhf.mos_alpha = ca;
    uhf.eps_alpha = ea;
    uhf.mos_beta = Some(cb);
    uhf.eps_beta = Some(eb);
    uhf.density_total = &da + &db;
    uhf.density_alpha = da;
    uhf.density_beta = Some(db);
    uhf.energy = num(&r, "/uhf/energy");
    // Orthonormal in ferric's overlap: the AO map is right (a wrong map
    // moves elements by O(0.1); the bound is the eigensolver's n² ε / λ_min(S)).
    let s = ferric_integrals::oneelectron::overlap(&obs);
    for c in [&uhf.mos_alpha, uhf.mos_beta.as_ref().unwrap()] {
        let g = c.t().dot(&s).dot(c);
        let dev = g
            .indexed_iter()
            .map(|((i, j), x)| (x - if i == j { 1.0 } else { 0.0 }).abs())
            .fold(0.0, f64::max);
        assert!(
            dev < 1e-8,
            "{label}: PySCF orbitals not orthonormal in ferric's S: {dev:.2e}"
        );
    }
    Sys {
        label,
        r,
        mol,
        obs,
        dfbs,
        uhf,
    }
}

/// Assert `got` against both references for `field`; returns the numpy bar.
fn check(sys: &Sys, field: &str, got: f64) -> f64 {
    let r = &sys.r;
    let fp = num(r, &format!("/fp_floor/{field}"));
    let bar = 2.0 * fp + FLOOR_ABS;
    let want = num(r, &format!("/fields/numpy_ferric_ints/{field}"));
    let d = (got - want).abs();
    eprintln!(
        "{} {field}: ferric {got:.12} numpy(ferric ints) {want:.12} |d| {d:.2e} bar {bar:.2e}",
        sys.label
    );
    assert!(
        d <= bar,
        "{} {field}: |ferric - numpy| {d:.3e} > {bar:.3e}",
        sys.label
    );
    if let Some(p) = r
        .pointer(&format!("/fields/pyscf/{field}"))
        .and_then(Value::as_f64)
    {
        let shift = num(r, &format!("/integral_shift/{field}")).abs();
        let resid = ["coulomb", "erfc"]
            .iter()
            .flat_map(|k| {
                r["kernels"][k]["code_residual"]
                    .as_object()
                    .unwrap()
                    .values()
                    .map(|v| v.as_f64().unwrap())
                    .collect::<Vec<_>>()
            })
            .fold(0.0, f64::max);
        let bar_p = bar + shift + resid;
        let dp = (got - p).abs();
        eprintln!(
            "{} {field}: PySCF {p:.12} |d| {dp:.2e} bar {bar_p:.2e} (shift {shift:.1e}, resid {resid:.1e})",
            sys.label
        );
        assert!(
            dp <= bar_p,
            "{} {field}: |ferric - PySCF| {dp:.3e} > {bar_p:.3e}",
            sys.label
        );
    }
    bar
}

fn run_case(system: &str) {
    let sys = load(system);
    let (mol, obs, dfbs, uhf) = (&sys.mol, &sys.obs, &sys.dfbs, &sys.uhf);

    let att = AttenuatedMp2Config::default();
    let e = attenuated_ri_mp2(mol, obs, dfbs, uhf, &att).unwrap();
    check(&sys, "att_erfc", e.mp2_corr);
    check(&sys, "att_erfc_e_os", e.spin_components.e_os);
    check(&sys, "att_erfc_e_ss", e.spin_components.e_ss);
    let e = attenuated_ri_mp2_long_range(mol, obs, dfbs, uhf, &att).unwrap();
    check(&sys, "att_erf", e.mp2_corr);

    let scs_cfg = ScsMp2Config {
        c_os: num(&sys.r, "/c_os"),
        c_ss: num(&sys.r, "/c_ss"),
        ..Default::default()
    };
    let s = scs_mp2(mol, obs, dfbs, uhf, &scs_cfg).unwrap();
    check(&sys, "scs", s.scs_corr);
    check(&sys, "scs_e_os", s.e_os);
    check(&sys, "scs_e_ss", s.e_ss);

    for form in [
        RsMp2RpaFormulation::DeltaLr,
        RsMp2RpaFormulation::CoupledRings,
    ] {
        let cfg = RsMp2RpaConfig {
            formulation: form,
            ..Default::default()
        };
        let x = rs_mp2_lr_rpa(mol, obs, dfbs, uhf, &cfg).unwrap();
        check(&sys, "e_mp2_full", x.e_mp2_full);
        let (tag, bar) = match form {
            RsMp2RpaFormulation::DeltaLr => {
                check(&sys, "e_dmp2_lr", x.e_dmp2_lr);
                check(&sys, "B_e_drpa_lr", x.e_drpa_lr.unwrap());
                ("B", check(&sys, "B_e_corr", x.e_corr))
            }
            RsMp2RpaFormulation::CoupledRings => {
                check(&sys, "T_e_delta_drpa_full", x.e_delta_drpa_full.unwrap());
                check(&sys, "T_e_delta_drpa_sr", x.e_delta_drpa_sr.unwrap());
                ("T", check(&sys, "T_e_corr", x.e_corr))
            }
        };
        let control = num(
            &sys.r,
            &format!("/fields/numpy_ferric_ints/{tag}_control_2eab"),
        );
        let miss = (x.e_corr - control).abs();
        eprintln!("{} {tag}: |ferric - (2 E_ab rule)| {miss:.2e}", sys.label);
        assert!(
            miss > MUST_MISS_FACTOR * bar,
            "{} {tag}: the closed-shell 2 E_ab rule is not excluded ({miss:.2e})",
            sys.label
        );
    }
}

#[test]
#[ignore = "validation: RS-MP2-RPA vs PySCF"]
fn u_attenuated_oh_vs_pyscf() {
    run_case("oh");
}

#[test]
#[ignore = "validation: RS-MP2-RPA vs PySCF"]
fn u_attenuated_nh2_vs_pyscf() {
    run_case("nh2");
}

#[test]
#[ignore = "validation: RS-MP2-RPA vs PySCF"]
fn u_attenuated_o2_triplet_vs_pyscf() {
    run_case("o2");
}
