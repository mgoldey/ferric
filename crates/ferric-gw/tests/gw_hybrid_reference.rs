//! G0W0 on HYBRID and range-separated-hybrid KS references vs PySCF `gw_ac`
//! (H2O/cc-pVDZ, PBE0 and CAM-B3LYP; `testdata/reference/validation/gw_hybrid/`,
//! generator `scripts/validation/gen_gw_hybrid.py`).
//!
//! The QP shift at a KS reference is Σx − v_mf, where v_mf is everything in
//! the KS Fock besides h + J: the semilocal V_xc AND the exact exchange a
//! hybrid put there (book/src/20-gw-bse/05-continuation-and-qp-equation.md,
//! "Hybrid references"). Σx is
//! the FULL exchange, so a v_mf without the K term counts the exact-exchange
//! fraction twice. `vxc_mo::vxc_diagonal_mo_on_grids` is that v_mf.
//!
//! # Like-for-like recipe
//!
//! gen_gw.py's @PBE recipe (see validation_gw.rs): exact four-centre J and K
//! in the SCF on both sides, the flat (75,110) Becke grid, ferric's 1e-14 XC
//! density floor applied to PySCF, Σx from the same Lpq (`vhf_df = True`),
//! and QP energies against the grid-free `exact` sub-block (dRPA pole sum at
//! the 18 Padé nodes, textbook Thiele, ferric's root rule), which
//! `run_gw`'s product-rule Σc matches.
//!
//! # Anchors
//!
//! * v_mf diagonal vs PySCF `get_veff − get_j` in the MO basis (the static
//!   anchor, no correlation in it), and its exact-exchange part vs PySCF's
//!   `hyb·K` (+ the LR erf term) built separately.
//! * PBE (no exact exchange) and HF are unchanged: the K term is skipped when
//!   `k_mix` is zero (bit-identical, `pure_functional_has_no_k_term`), and HF
//!   references pass no v_mf at all.
//! * Must-miss control: PySCF's own double-counted QP energies
//!   (`exact/eps_qp_hybrid_double_counted`) are MISSED by ≥ 10× the bar.
//!
//! Mutation record (2026-11-17): deleting the two `add_exact_exchange` calls
//! in vxc_mo.rs fails `g0w0_pbe0_h2o_vs_pyscf` and `g0w0_cam_b3lyp_h2o_vs_pyscf`
//! at the v_mf assertion (see the commit message for the measured miss).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_gw::vxc_mo::{vxc_diagonal_mo_on_grids, JkRoute};
use ferric_gw::{run_gw, GwConfig, GwMethod};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use serde_json::Value;

const REF: &str = "testdata/reference/validation/gw_hybrid/h2o_cc-pvdz.json";
const XYZ: &str = "testdata/molecules/validation/h2o.xyz";
const HA_TO_EV: f64 = 27.211_386_245_988;
const N_QUAD: usize = 100;
/// SCF energy (exact ERIs both sides, same grid): validation_gw.rs's bar.
const TOL_E_SCF: f64 = 1e-9;
/// ε_mf and the v_mf diagonal: validation_gw.rs's `TOL_EPS_MF`.
const TOL_EPS_MF: f64 = 3e-8;
/// QP energies vs the grid-free `exact` block: validation_gw.rs's `TOL_QP_PBE`.
const TOL_QP: f64 = 1e-6;
const MUST_MISS_FACTOR: f64 = 10.0;
// CAM-B3LYP bars (in `g0w0_cam_b3lyp_h2o_vs_pyscf`): ferric's RSH SCF
// exchange is RI-K (def2-universal-jkfit) only, PySCF's is exact. MEASURED
// 2026-11-17: E_RKS 1.23e-5, ε_mf 2.3e-4 (high virtuals; window ≤ 2.6e-6),
// v_mf 2.1e-6, Σx 2.6e-6, QP 1.33e-5 Ha (LUMO+2; HOMO/LUMO ≤ 3e-6). The
// double-counted reference is 8.46 eV off at the HOMO. PBE0 (exact K both
// sides): E 2e-13, v_mf 3.1e-10, QP 2.1e-9 Ha.

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
        .unwrap()
        .to_path_buf()
}

fn reference() -> Value {
    let path = workspace_root().join(REF);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing reference {} ({e}); regenerate with scripts/validation/gen_gw_hybrid.py",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap()
}

fn vec_f64(v: &Value, ptr: &str) -> Vec<f64> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("reference field {ptr} missing"))
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn worst(ctx: &str, what: &str, orbs: &[usize], got: &[f64], want: &[f64]) -> f64 {
    let mut w = 0.0_f64;
    for ((&p, &g), &r) in orbs.iter().zip(got).zip(want) {
        let d = (g - r).abs();
        eprintln!(
            "{ctx}: {what}[{p}] ferric {g:+.10} ref {r:+.10} |d| {d:.2e} Ha ({:.4} meV)",
            d * HA_TO_EV * 1e3
        );
        w = w.max(d);
    }
    w
}

struct H2o {
    mol: Molecule,
    obs_bs: basis::BasisSet,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
}

fn h2o(r: &Value) -> H2o {
    let mol =
        Molecule::load_xyz_with_charge(workspace_root().join(XYZ).to_str().unwrap(), 0, 1).unwrap();
    assert!((mol.nuclear_repulsion() - r["nuclear_repulsion"].as_f64().unwrap()).abs() < 1e-9);
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let aux_bs = basis::bundled(r["aux"].as_str().unwrap()).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    assert_eq!(obs.nbasis() as i64, r["nao"].as_i64().unwrap());
    assert_eq!(dfbs.nbasis() as i64, r["naux"].as_i64().unwrap());
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    H2o {
        mol,
        obs_bs,
        obs,
        dfbs,
        bounds,
    }
}

/// Exact J (empty name = opt out of the RI auto-default), as the reference.
/// K: exact for a global hybrid; a range-separated functional's SCF exchange
/// is RI-K only in ferric (driver.rs refuses `df_k_aux = ""` for ω > 0), so
/// there `df_k_aux` stays at its def2-universal-jkfit default and the
/// comparison carries that fitting error (the `Bars` of that case).
fn ks_config(xc: &str, exact_k: bool) -> RhfConfig {
    let mut c = RhfConfig::default()
        .with_xc(xc)
        .with_df_j_aux(String::new())
        .with_max_iter(400)
        .with_energy_conv(1e-11)
        .with_density_conv(1e-10)
        .with_dft_grid(AtomicGridConfig::default());
    c.df_k_aux = exact_k.then(String::new);
    c
}

/// Bars for one functional: SCF energy, v_mf / ε_mf / Σx diagonal, QP.
struct Bars {
    e_scf: f64,
    /// Every mean-field orbital energy.
    eps_mf: f64,
    /// v_mf and Σx over the QP window.
    eps: f64,
    qp: f64,
}

fn hybrid_case(block: &str, exact_k: bool, bars: Bars) {
    let r = reference();
    let sys = h2o(&r);
    let xc = r[block]["xc"].as_str().unwrap().to_string();
    let ctx = format!("h2o/cc-pvdz @{xc}");
    let scf = solve_rhf(
        &ParallelContext::default(),
        &sys.mol,
        &sys.obs,
        Operator::coulomb(),
        &sys.bounds,
        &ks_config(&xc, exact_k),
    )
    .unwrap();
    assert!(scf.converged, "{ctx}: RKS did not converge");
    let e_ref = r[block]["rks_energy"].as_f64().unwrap();
    eprintln!(
        "{ctx}: E_RKS ferric {:.12} ref {e_ref:.12} |d| {:.2e}",
        scf.energy,
        (scf.energy - e_ref).abs()
    );
    assert!((scf.energy - e_ref).abs() < bars.e_scf, "{ctx}: E_RKS");
    let mo_ref = vec_f64(&r, &format!("/{block}/mo_energy"));
    let all: Vec<usize> = (0..mo_ref.len()).collect();
    let d_eps = worst(&ctx, "eps_mf", &all, scf.eps_r(), &mo_ref);
    assert!(d_eps < bars.eps_mf, "{ctx}: ε_mf {d_eps:.2e}");

    let orbs: Vec<usize> = r[block]["orbs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let (vmf, vmf_b) = vxc_diagonal_mo_on_grids(
        &sys.mol,
        &sys.obs_bs,
        &xc,
        &scf,
        &AtomicGridConfig::default(),
        &AtomicGridConfig::default_nlc(),
        JkRoute::default(),
    )
    .unwrap();
    assert_eq!(
        vmf, vmf_b,
        "{ctx}: restricted reference must give equal spins"
    );
    let got: Vec<f64> = orbs.iter().map(|&p| vmf[p]).collect();
    // The static anchor: v_mf == PySCF get_veff − get_j (semilocal + K term).
    let d_vmf = worst(
        &ctx,
        "v_mf",
        &orbs,
        &got,
        &vec_f64(&r, &format!("/{block}/v_mf")),
    );
    // Control: the semilocal-only diagonal (the double-counting defect) misses.
    let d_sl = worst(
        &ctx,
        "v_mf vs v_xc_semilocal",
        &orbs,
        &got,
        &vec_f64(&r, &format!("/{block}/v_xc_semilocal")),
    );
    assert!(
        d_vmf < bars.eps,
        "{ctx}: v_mf diagonal differs from PySCF get_veff − get_j by {d_vmf:.2e} Ha \
         (bar {:.0e}); the semilocal-only diagonal is {d_sl:.2e} Ha away",
        bars.eps
    );
    assert!(
        d_sl > MUST_MISS_FACTOR * bars.eps,
        "{ctx}: K term is negligible here?"
    );

    let qp = orbs[0]..orbs[orbs.len() - 1] + 1;
    let res = run_gw(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        &scf,
        &PdepRpaConfig {
            frozen_core: 0,
            trunc_thresh: 0.0,
            eigensolver: Eigensolver::Lanczos,
            quadrature: QuadratureConfig {
                scheme: QuadratureScheme::GaussLegendre,
                n_points: N_QUAD,
                u0: 0.5,
            },
            need_inv_dielectric_freq: true,
            need_eigenvalues_freq: true,
            ..Default::default()
        },
        &GwConfig {
            method: GwMethod::G0W0,
            qp_mos: Some(qp),
            ..Default::default()
        },
        Some(&vmf),
    )
    .unwrap();
    assert_eq!(res.mo_indices, orbs);
    assert!(res.qp_converged.iter().all(|&c| c), "{ctx}: QP Newton");
    let d_sx = worst(
        &ctx,
        "sigma_x(DF)",
        &orbs,
        res.sigma_x.as_slice().unwrap(),
        &vec_f64(&r, &format!("/{block}/sigma_x_df")),
    );
    assert!(d_sx < bars.eps, "{ctx}: Σx(DF) {d_sx:.2e}");
    let eqp = res.eps_qp.as_slice().unwrap();
    let d_qp = worst(
        &ctx,
        "eps_qp",
        &orbs,
        eqp,
        &vec_f64(&r, &format!("/{block}/exact/eps_qp")),
    );
    let d_dc = worst(
        &ctx,
        "eps_qp vs double-counted",
        &orbs,
        eqp,
        &vec_f64(&r, &format!("/{block}/exact/eps_qp_hybrid_double_counted")),
    );
    let nocc = (sys.mol.nelec() / 2) as usize;
    for (name, p) in [("HOMO", nocc - 1), ("LUMO", nocc)] {
        let k = orbs.iter().position(|&q| q == p).unwrap();
        eprintln!(
            "{ctx}: {name} G0W0 ferric {:.6} eV, ref {:.6} eV, double-counted ref {:.6} eV",
            eqp[k] * HA_TO_EV,
            vec_f64(&r, &format!("/{block}/exact/eps_qp"))[k] * HA_TO_EV,
            vec_f64(&r, &format!("/{block}/exact/eps_qp_hybrid_double_counted"))[k] * HA_TO_EV,
        );
    }
    assert!(
        d_qp < bars.qp,
        "{ctx}: QP energies differ from the grid-free reference by {d_qp:.2e} Ha (bar {:.0e})",
        bars.qp
    );
    assert!(
        d_dc > MUST_MISS_FACTOR * bars.qp,
        "{ctx}: the double-counted reference is not missed ({d_dc:.2e} Ha)"
    );
}

#[test]
fn g0w0_pbe0_h2o_vs_pyscf() {
    hybrid_case(
        "g0w0_pbe0",
        true,
        Bars {
            e_scf: TOL_E_SCF,
            eps_mf: TOL_EPS_MF,
            eps: TOL_EPS_MF,
            qp: TOL_QP,
        },
    );
}

#[test]
fn g0w0_cam_b3lyp_h2o_vs_pyscf() {
    hybrid_case(
        "g0w0_cam_b3lyp",
        false,
        Bars {
            e_scf: 3e-5,
            eps_mf: 5e-4,
            eps: 1e-5,
            qp: 4e-5,
        },
    );
}

/// Exactness anchor: for a pure functional `k_mix` is zero and the K term is
/// not built, so the v_mf diagonal is the semilocal V_xc alone. Checked
/// against the same AO V_xc assembled here by hand (`KsXc::add_xc`), bit for
/// bit.
#[test]
fn pure_functional_has_no_k_term() {
    use ferric_dft::ks::KsXc;
    use ferric_dft::xc_trait::XcContribution;
    let r = reference();
    let sys = h2o(&r);
    let grid = AtomicGridConfig::default();
    let cfg = RhfConfig::default()
        .with_xc("pbe")
        .with_df_j_aux(String::new())
        .with_dft_grid(grid.clone());
    let scf = solve_rhf(
        &ParallelContext::default(),
        &sys.mol,
        &sys.obs,
        Operator::coulomb(),
        &sys.bounds,
        &cfg,
    )
    .unwrap();
    let nlc = AtomicGridConfig::default_nlc();
    let (v, _) = vxc_diagonal_mo_on_grids(
        &sys.mol,
        &sys.obs_bs,
        "pbe",
        &scf,
        &grid,
        &nlc,
        JkRoute::default(),
    )
    .unwrap();
    let ks = KsXc::new(&sys.mol, &sys.obs_bs, "pbe", &grid, &nlc).unwrap();
    let d = 2.0 * &scf.density_alpha;
    let mut vao = ndarray::Array2::<f64>::zeros(d.raw_dim());
    ks.add_xc(&d, &mut vao);
    // The same contraction order as vxc_mo.rs `mo_diagonal`.
    let c = scf.mos_r();
    let ac = vao.dot(c);
    for p in 0..v.len() {
        let mut acc = 0.0;
        for mu in 0..c.nrows() {
            acc += c[(mu, p)] * ac[(mu, p)];
        }
        assert_eq!(v[p].to_bits(), acc.to_bits(), "MO {p}");
    }
}
