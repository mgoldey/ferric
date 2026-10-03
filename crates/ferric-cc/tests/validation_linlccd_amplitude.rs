//! VALIDATION tier — VALIDATION.md row "Amplitude-threshold LinLCCD".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-cc --test validation_linlccd_amplitude \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's amplitude-threshold LinLCCD (`linlccd_amplitude`: VV-HV localized
//! virtuals, Boys-localized occupieds, masked ragged CG on the spatial
//! residual) at ε = 0, for every tier of `LadderVariant` — `DriversOnly`
//! (MP2), `Hh` (LinLCCD(hh), eq. 14) and `Full` (hh + pp ladders, eq. 7) —
//! through BOTH the global-B path (`amplitude_linlccd`) and the
//! integral-direct path (`amplitude_linlccd_direct`, every locality map made
//! trivial), all electrons correlated, cc-pVDZ-RI Coulomb-metric fitting,
//! against an independent numpy reference on PySCF density-fitted integrals
//! (`scripts/validation/gen_linlccd.py` →
//! `testdata/reference/validation/linlccd/<system>_<basis>.json`, fields
//! `/mp2/e_corr`, `/linlccd_hh/e_corr`, `/linlccd_full/e_corr`):
//! H2O and NH3 × 6-31G and cc-pVDZ, exact-J/K RHF.
//!
//! The reference shares nothing with ferric's construction beyond the
//! fitted integrals: canonical orbitals, no localization, no iteration. The
//! hh tier is solved in the occupied-pair eigenbasis; the full tier as the
//! Sylvester equation H_occ T − T H_vir = v in the joint eigenbasis; each is
//! built in spin orbitals AND spin-adapted, and the generator refuses to
//! write unless they agree to 1e-12 (measured ≤ 1.1e-16).
//!
//! # Exactness anchors (asserted first, per system)
//!
//! E_nuc (geometry), nao/naux (basis), E_RHF vs PySCF, then LADDER OFF:
//! `DriversOnly` must equal the reference MP2 (itself proven equal to PySCF
//! `DFMP2` in the generator) before Hh or Full is examined. At ε = 0 the
//! localized amplitude solve is a rotation of the canonical one, so any
//! disagreement above the CG floor is a construction defect (wrong virtual
//! span, a mis-indexed ladder), never "localization error".
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If the ladders are right: every tier agrees with numpy to the CG floor
//!   (`cg_rtol` 1e-12 here) on all four inputs, on both paths.
//! * If a ladder has a wrong factor, sign or index pairing, the error is a
//!   fraction of the ladder's own contribution — |E_hh − E_MP2| is
//!   2.0e-2..3.0e-2 Eh and |E_full − E_hh| 1.4e-2..2.3e-2 Eh here, nine or
//!   more orders above the bars. The MP2 anchor cannot see either.
//! * If the localized virtual space does not span the canonical one, every
//!   tier misses, MP2 included (the dropped-virtual negative control).
//! * If the HARNESS is broken: E_nuc / nao / naux / E_RHF fail first.
//!
//! # TOLERANCES
//!
//! Each bar is ~10× the measured maximum recorded on its const.
//!
//! # NEGATIVE CONTROLS (asserted in-test, always on)
//!
//! * TIER SWAP: Hh must MISS the Full and MP2 references, and Full must MISS
//!   the Hh reference, by more than [`MUST_MISS`].
//! * DROPPED VIRTUAL: `amplitude_linlccd_with_virtuals` with one hard
//!   virtual removed must MISS the reference by more than [`MUST_MISS`]
//!   (`dropped_hard_virtual_misses_every_tier`).
//!
//! # FINITE ε (measurement, NH3/cc-pVDZ, Hh)
//!
//! `finite_eps_error_is_one_sided_and_monotone_nh3_ccpvdz` records the error
//! against the numpy ε = 0 value at ε ∈ {1e-6, 1e-5, 1e-4}; it asserts only
//! the sign (under-correlation) and monotonicity, not a size.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_cc::linlccd::{linlccd, LadderVariant};
use ferric_cc::linlccd_amplitude::{
    amplitude_linlccd, amplitude_linlccd_direct, amplitude_linlccd_with_virtuals,
    AmplitudeLinLccdConfig,
};
use ferric_cc::CcConfig;
use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::{build_vvhv, VvHv};
use ferric_mp2::lmp2_direct::DirectConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::s;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/linlccd";
const MOL_DIR: &str = "testdata/molecules/validation";
const AUX: &str = "cc-pvdz-ri";

/// Geometry check (PySCF E_nuc from ferric's Bohr geometry).
const TOL_ENUC: f64 = 1e-9;
/// RHF energy vs PySCF, exact J/K on both sides.
// Measured max 7.3e-12 (NH3/6-31G).
const TOL_E_SCF: f64 = 1e-9;
/// ε = 0 correlation energy vs the numpy reference, global-B path, per tier.
// Measured max (2026-10-02, cg_rtol 1e-12): DriversOnly 4.4e-13 (H2O/6-31G),
// Hh 2.8e-13 (H2O/6-31G), Full 2.0e-13 (H2O/6-31G). Spec target 1e-10.
const TOL_DRIVERS: f64 = 5e-12;
const TOL_HH: f64 = 3e-12;
const TOL_FULL: f64 = 2e-12;
/// Same comparison through the integral-direct path with trivial maps.
// Measured max 3.0e-13 (DriversOnly, H2O/6-31G); Hh 1.6e-13, Full 1.1e-13.
const TOL_DIRECT: f64 = 3e-12;
/// ferric's EXACT canonical LinLCCD (`linlccd::linlccd`, spin-orbital
/// Jacobi + DIIS) vs the same reference, per tier — the third leg of
/// "ε = 0 local == exact == numpy".
// Measured max 4.4e-13 (DriversOnly, H2O/6-31G); Hh 2.7e-13, Full 1.9e-13.
const TOL_CANONICAL: f64 = 5e-12;
/// A reference ferric must MISS (negative controls). The tiers are
/// separated by |E_hh − E_MP2| = 2.0e-2..3.0e-2 and |E_full − E_hh| =
/// 1.4e-2..2.3e-2 Eh on these inputs (JSON `hh_minus_mp2`, `full_minus_hh`).
const MUST_MISS: f64 = 1e-6;
/// CG relative-residual target: tighter than the library default (1e-11) so
/// the bars measure the construction, not the solver stop.
const CG_RTOL: f64 = 1e-12;

const TIERS: [LadderVariant; 3] = [
    LadderVariant::DriversOnly,
    LadderVariant::Hh,
    LadderVariant::Full,
];

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
        .expect("ferric-cc manifest dir should be <root>/crates/ferric-cc")
        .to_path_buf()
}

fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_linlccd.py — a missing reference is a failure, never a skip",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn num(v: &Value, ptr: &str, ctx: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not a number"))
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<28} ferric {got:+.13} ref {want:+.13} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.13} vs reference {want:.13} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_miss(ctx: &str, what: &str, got: f64, want: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<28} |d| {d:.2e} (must exceed {MUST_MISS:.0e})");
    assert!(
        d > MUST_MISS,
        "{ctx}: negative control {what}: |d| {d:.2e} <= {MUST_MISS:.0e} — the bar cannot \
         tell these apart"
    );
}

/// The reference field for a tier.
fn ref_ptr(variant: LadderVariant) -> &'static str {
    match variant {
        LadderVariant::DriversOnly => "/mp2/e_corr",
        LadderVariant::Hh => "/linlccd_hh/e_corr",
        LadderVariant::Full => "/linlccd_full/e_corr",
    }
}

fn tol_global(variant: LadderVariant) -> f64 {
    match variant {
        LadderVariant::DriversOnly => TOL_DRIVERS,
        LadderVariant::Hh => TOL_HH,
        LadderVariant::Full => TOL_FULL,
    }
}

fn amp_config(eps: f64) -> AmplitudeLinLccdConfig {
    let cfg = AmplitudeLinLccdConfig {
        eps,
        cg_rtol: CG_RTOL,
        cg_max_iter: 2000,
        ..Default::default()
    };
    assert_eq!(cfg.frozen_core, 0, "all-electron expected");
    assert!(cfg.pair_gate_cal.is_none(), "pair gate must be off");
    cfg
}

/// Every locality map of the direct path made inert: global aux fit, all
/// virtuals, no AO-support or Schwarz truncation (`trivial_maps()` in
/// `crates/ferric-mp2/tests/lmp2_direct.rs`, with `schwarz_skip` pinned).
fn trivial_maps() -> DirectConfig {
    DirectConfig {
        aux_radius_bohr: 1e6,
        virt_radius_bohr: None,
        ao_tail: 0.0,
        schwarz_skip: 0.0,
        ..Default::default()
    }
}

struct Case {
    ctx: String,
    r: Value,
    mol: Molecule,
    obs_bs: BasisSet,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    scf: ScfResult,
}

impl Case {
    /// Geometry, basis and RHF anchors — asserted before any correlation.
    fn new(system: &str, basis_name: &str) -> Self {
        let r = reference(system, basis_name);
        let ctx = format!("{system}/{basis_name}");
        assert_eq!(r["basis"].as_str(), Some(basis_name), "{ctx}: basis");
        assert_eq!(r["aux_basis"].as_str(), Some(AUX), "{ctx}: aux basis");
        assert_eq!(r["multiplicity"].as_u64(), Some(1), "{ctx}: closed shell");
        assert_eq!(r["charge"].as_i64(), Some(0), "{ctx}: neutral");

        let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
        let mol = Molecule::load_xyz(xyz.to_str().unwrap())
            .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
        check_close(
            &ctx,
            "E_nuc",
            mol.nuclear_repulsion(),
            num(&r, "/nuclear_repulsion", &ctx),
            TOL_ENUC,
        );
        let obs_bs = basis::bundled(basis_name).unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
        assert_eq!(
            obs.nbasis() as u64,
            r["nao"].as_u64().unwrap(),
            "{ctx}: nao"
        );
        assert_eq!(
            dfbs.nbasis() as u64,
            r["naux"].as_u64().unwrap(),
            "{ctx}: naux"
        );

        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let scf_cfg = RhfConfig {
            max_iter: 400,
            energy_conv: 1e-11,
            density_conv: 1e-10,
            ..Default::default()
        };
        assert!(
            scf_cfg.df_j_aux.is_none() && scf_cfg.df_k_aux.is_none(),
            "exact J/K expected"
        );
        let scf = solve_rhf(
            &ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &scf_cfg,
        )
        .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"));
        assert!(scf.converged, "{ctx}: SCF not converged");
        check_close(
            &ctx,
            "E_RHF",
            scf.energy,
            num(&r, "/e_scf", &ctx),
            TOL_E_SCF,
        );
        Case {
            ctx,
            r,
            mol,
            obs_bs,
            obs,
            dfbs,
            scf,
        }
    }

    fn reference_e(&self, variant: LadderVariant) -> f64 {
        num(&self.r, ref_ptr(variant), &self.ctx)
    }

    fn global(&self, variant: LadderVariant, eps: f64) -> f64 {
        let res = amplitude_linlccd(
            &self.mol,
            &self.obs,
            &self.obs_bs,
            &self.dfbs,
            Operator::coulomb(),
            &self.scf,
            &amp_config(eps),
            variant,
        )
        .unwrap_or_else(|e| panic!("{}: amplitude_linlccd {variant:?}: {e:?}", self.ctx));
        assert!(res.cg_converged, "{}: CG not converged", self.ctx);
        eprintln!(
            "{}: {variant:?} eps={eps:.0e} keep={:.6} cg {} relres {:.1e}",
            self.ctx, res.keep_fraction, res.cg_iterations, res.cg_relres
        );
        if eps == 0.0 {
            assert_eq!(res.keep_fraction, 1.0, "{}: eps=0 must keep all", self.ctx);
        }
        res.e_corr
    }

    fn canonical(&self, variant: LadderVariant) -> f64 {
        let cfg = CcConfig {
            energy_conv: 1e-13,
            max_iter: 300,
            ..Default::default()
        };
        assert_eq!(cfg.frozen_core, 0, "all-electron expected");
        linlccd(
            &self.mol,
            &self.obs,
            &self.dfbs,
            Operator::coulomb(),
            &self.scf,
            &cfg,
            variant,
        )
        .unwrap_or_else(|e| panic!("{}: linlccd {variant:?}: {e:?}", self.ctx))
        .correlation_energy
    }

    fn direct(&self, variant: LadderVariant) -> f64 {
        let (res, stats) = amplitude_linlccd_direct(
            &self.mol,
            &self.obs,
            &self.obs_bs,
            &self.dfbs,
            Operator::coulomb(),
            &self.scf,
            &amp_config(0.0),
            &trivial_maps(),
            variant,
        )
        .unwrap_or_else(|e| panic!("{}: amplitude_linlccd_direct {variant:?}: {e:?}", self.ctx));
        assert!(res.cg_converged, "{}: direct CG not converged", self.ctx);
        assert_eq!(res.keep_fraction, 1.0, "{}: eps=0 must keep all", self.ctx);
        // The maps must actually be trivial: every strip spans the full aux.
        assert_eq!(
            stats.strip_rows_max,
            self.dfbs.nbasis(),
            "{}: trivial maps did not span the aux basis",
            self.ctx
        );
        res.e_corr
    }
}

fn check_system(system: &str, basis_name: &str) {
    let c = Case::new(system, basis_name);
    let ctx = c.ctx.clone();

    // ---- anchor first: ladder off == numpy MP2 (== PySCF DFMP2) ----
    let mut got = Vec::new();
    for variant in TIERS {
        let e = c.global(variant, 0.0);
        check_close(
            &ctx,
            &format!("global {variant:?}"),
            e,
            c.reference_e(variant),
            tol_global(variant),
        );
        got.push(e);
    }
    let (e_drv, e_hh, e_full) = (got[0], got[1], got[2]);

    // ---- exact canonical path, same reference ----
    for variant in TIERS {
        let e = c.canonical(variant);
        check_close(
            &ctx,
            &format!("canonical {variant:?}"),
            e,
            c.reference_e(variant),
            TOL_CANONICAL,
        );
    }

    // ---- integral-direct path, trivial maps ----
    for variant in TIERS {
        let e = c.direct(variant);
        check_close(
            &ctx,
            &format!("direct {variant:?}"),
            e,
            c.reference_e(variant),
            TOL_DIRECT,
        );
    }

    // ---- negative controls: the tiers are distinguishable ----
    let ref_mp2 = c.reference_e(LadderVariant::DriversOnly);
    let ref_hh = c.reference_e(LadderVariant::Hh);
    let ref_full = c.reference_e(LadderVariant::Full);
    check_miss(&ctx, "Hh vs ref Full", e_hh, ref_full);
    check_miss(&ctx, "Hh vs ref MP2", e_hh, ref_mp2);
    check_miss(&ctx, "Full vs ref Hh", e_full, ref_hh);
    check_miss(&ctx, "DriversOnly vs ref Hh", e_drv, ref_hh);
}

#[test]
#[ignore = "validation: Amplitude-threshold LinLCCD"]
fn amplitude_linlccd_h2o_631g_vs_numpy() {
    check_system("h2o", "6-31g");
}

#[test]
#[ignore = "validation: Amplitude-threshold LinLCCD"]
fn amplitude_linlccd_h2o_ccpvdz_vs_numpy() {
    check_system("h2o", "cc-pvdz");
}

#[test]
#[ignore = "validation: Amplitude-threshold LinLCCD"]
fn amplitude_linlccd_nh3_631g_vs_numpy() {
    check_system("nh3", "6-31g");
}

#[test]
#[ignore = "validation: Amplitude-threshold LinLCCD"]
fn amplitude_linlccd_nh3_ccpvdz_vs_numpy() {
    check_system("nh3", "cc-pvdz");
}

/// NEGATIVE CONTROL: a virtual space missing one hard virtual no longer
/// spans the canonical one, so every tier must miss its reference — the
/// ε = 0 anchors above can see a span defect.
#[test]
#[ignore = "validation: Amplitude-threshold LinLCCD"]
fn dropped_hard_virtual_misses_every_tier() {
    let c = Case::new("h2o", "cc-pvdz");
    let vvhv = build_vvhv(&c.mol, &c.obs, &c.obs_bs, &c.scf).unwrap();
    let nvir = vvhv.c_vloc.ncols();
    assert!(vvhv.n_hard > 0, "no hard virtual to drop");
    let broken = VvHv {
        c_vloc: vvhv.c_vloc.slice(s![.., ..nvir - 1]).to_owned(),
        n_valence: vvhv.n_valence,
        n_hard: vvhv.n_hard - 1,
    };
    for variant in TIERS {
        let e = amplitude_linlccd_with_virtuals(
            &c.mol,
            &c.obs,
            &c.dfbs,
            Operator::coulomb(),
            &c.scf,
            &amp_config(0.0),
            variant,
            &broken,
        )
        .unwrap()
        .e_corr;
        check_miss(
            &c.ctx,
            &format!("dropped virtual {variant:?}"),
            e,
            c.reference_e(variant),
        );
    }
}

/// MEASUREMENT (NH3/cc-pVDZ, Hh): the amplitude-threshold error against the
/// numpy ε = 0 value. ε = 1e-6 drops only symmetry-zero integrals (keep
/// 0.950), so its error must sit at the ε = 0 floor (< [`TOL_HH`]); above
/// the floor the error must be one-sided (under-correlation, E(ε) > E(0))
/// and grow with ε. The sizes are recorded, not asserted.
// Measured (2026-10-02): ε = 1e-6 +1.6e-13 (keep 0.9500, the same offset as
// ε = 0 itself), 1e-5 +2.68e-9 (keep 0.9493), 1e-4 +1.07e-5 (keep 0.9281).
#[test]
#[ignore = "validation: Amplitude-threshold LinLCCD"]
fn finite_eps_error_is_one_sided_and_monotone_nh3_ccpvdz() {
    let c = Case::new("nh3", "cc-pvdz");
    let e0 = c.reference_e(LadderVariant::Hh);
    let de_floor = c.global(LadderVariant::Hh, 1e-6) - e0;
    eprintln!(
        "{}: Hh eps=1e-6 E(eps) - E_numpy(0) = {de_floor:+.3e}",
        c.ctx
    );
    assert!(
        de_floor.abs() < TOL_HH,
        "eps=1e-6 drops only symmetry zeros, yet |dE| {de_floor:.2e} >= {TOL_HH:.0e}"
    );
    let mut prev = de_floor;
    for eps in [1e-5, 1e-4] {
        let de = c.global(LadderVariant::Hh, eps) - e0;
        eprintln!(
            "{}: Hh eps={eps:.0e} E(eps) - E_numpy(0) = {de:+.3e}",
            c.ctx
        );
        assert!(
            de > TOL_HH,
            "eps={eps:.0e}: error not one-sided above the floor: {de:+.3e}"
        );
        assert!(
            de > prev,
            "eps={eps:.0e}: error not monotone ({de:+.3e} after {prev:+.3e})"
        );
        prev = de;
    }
}
