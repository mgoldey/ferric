//! The PDEP property bundle (`ferric_rpa::property_bundle`), shared by the CLI
//! NPZ export and the Python `run_pdep_rpa(properties=True)`.
//!
//! Exactness anchors, water / STO-3G with cc-pVDZ-RI:
//!
//! * `alpha_atomic` is the static Becke per-atom polarizability. The dynamic
//!   per-atom path at omega = 0 is an independent construction (different
//!   code: the frequency loop of `pdep_polarizability_becke_dynamic`), so
//!   `dynamic(0) == static` is the trivial-limit anchor for the dynamic
//!   per-atom arrays the bundle exports as `alpha_atomic_dynamic`. Bar: the two
//!   solve the same dielectric system by different linear-algebra routes, so
//!   they agree to a few ulp x condition; 1e-10 absolute (alpha ~ 1-10 a.u.)
//!   is ~1e5 above the measured floor and ~1e5 below any physics difference
//!   (a swapped partition moves alpha_atomic by >1e-2).
//! * `alpha_ct == alpha_tensor - sum(alpha_atomic)` exactly (one f64 subtract
//!   of the exported arrays).
//! * a request with every switch off computes nothing and records no gaps.
//! * an unknown `c6_source` / `c6_partition` is an `Err`, and only after the
//!   properties that precede C6 were computed.
//!
//! Artifact hypothesis: if the bundle used the wrong partition or the wrong
//! density, `alpha_atomic` would differ from the direct static call by >1e-3;
//! if the dynamic path were wired to a different response than the static one,
//! the omega = 0 comparison would miss by the same order. Neither can pass at
//! 1e-10.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::{pdep_polarizability_becke, pdep_polarizability_becke_dynamic};
use ferric_rpa::property_bundle::{
    compute_pdep_properties, PdepPropertyRequest, PropertyInputs, PropertyReporter,
};
use ferric_rpa::PdepRpaConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

#[derive(Default)]
struct Collect {
    warnings: Vec<String>,
}

impl PropertyReporter for Collect {
    fn info(&mut self, _msg: &str) {}
    fn warn(&mut self, msg: &str) {
        self.warnings.push(msg.to_string());
    }
}

struct Water {
    mol: Molecule,
    obs_bs: ferric_core::basis::BasisSet,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ferric_scf::result::ScfResult,
    rhf_cfg: RhfConfig,
    cfg: PdepRpaConfig,
}

fn water() -> Water {
    let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("sto-3g").unwrap();
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux).unwrap();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf_cfg = RhfConfig::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &rhf_cfg).unwrap();
    Water {
        mol,
        obs_bs,
        obs,
        dfbs,
        rhf,
        rhf_cfg,
        cfg: PdepRpaConfig::default(),
    }
}

type Run = (
    Result<ferric_rpa::property_bundle::PdepProperties, String>,
    Vec<String>,
);

fn run(w: &Water, req: &PdepPropertyRequest) -> Run {
    let ctx = ParallelContext::default();
    let proatom = ferric_scf::properties::scf_proatom_provider(
        &ctx,
        &w.obs_bs,
        Operator::coulomb(),
        &w.rhf_cfg,
    );
    let inputs = PropertyInputs {
        ctx: &ctx,
        mol: &w.mol,
        prep: &w.obs,
        bs: &w.obs_bs,
        dfbs: &w.dfbs,
        result: &w.rhf,
        op: Operator::coulomb(),
        rpa_cfg: &w.cfg,
        rhf_config: &w.rhf_cfg,
        proatom: &proatom,
    };
    let mut rep = Collect::default();
    let out = compute_pdep_properties(&inputs, req, &mut rep);
    (out, rep.warnings)
}

fn only(f: impl FnOnce(&mut PdepPropertyRequest)) -> PdepPropertyRequest {
    let mut r = PdepPropertyRequest {
        esp: false,
        polarizability: false,
        electric_field: false,
        alpha_atomic: false,
        density_matrix: false,
        dipole: false,
        lowdin: false,
        hirshfeld: false,
        mulliken: false,
        chelpg: false,
        resp: false,
        c6: false,
        ..PdepPropertyRequest::default()
    };
    f(&mut r);
    r
}

#[test]
fn bundle_alpha_atomic_is_the_static_becke_and_dynamic_at_zero_matches_it() {
    let w = water();
    let (out, warnings) = run(
        &w,
        &only(|r| {
            r.polarizability = true;
            r.alpha_atomic = true;
        }),
    );
    let p = out.unwrap();
    assert!(
        warnings.is_empty() && p.gaps.is_empty(),
        "{warnings:?} {:?}",
        p.gaps
    );
    let op = Operator::coulomb();
    let stat =
        pdep_polarizability_becke(&w.mol, &w.obs, &w.obs_bs, &w.dfbs, &w.rhf, op, &w.cfg).unwrap();
    let bundle = p.alpha_atomic.as_ref().unwrap();
    assert_eq!(
        bundle, &stat,
        "bundle alpha_atomic must be the static Becke call, bit for bit"
    );
    let dynamic = pdep_polarizability_becke_dynamic(
        &w.mol,
        &w.obs,
        &w.obs_bs,
        &w.dfbs,
        &w.rhf,
        op,
        &w.cfg,
        &[0.0],
    )
    .unwrap();
    let mut worst = 0.0f64;
    for a in 0..stat.len() {
        for i in 0..3 {
            for j in 0..3 {
                worst = worst.max((dynamic[a][0][i][j] - stat[a][i][j]).abs());
            }
        }
    }
    assert!(
        worst < 1e-10,
        "dynamic(omega=0) vs static per-atom alpha: {worst:.3e}"
    );
    // Not a vacuous comparison: the tensors are O(1).
    assert!(stat.iter().any(|t| (0..3).any(|i| t[i][i].abs() > 1e-2)));
}

#[test]
fn alpha_ct_is_the_exact_remainder() {
    let w = water();
    let (out, _) = run(
        &w,
        &only(|r| {
            r.polarizability = true;
            r.alpha_atomic = true;
        }),
    );
    let p = out.unwrap();
    let (mol_a, per_atom, ct) = (
        p.alpha_tensor.unwrap(),
        p.alpha_atomic.unwrap(),
        p.alpha_ct.unwrap(),
    );
    for i in 0..3 {
        for j in 0..3 {
            let s: f64 = per_atom.iter().map(|t| t[i][j]).sum();
            assert!((ct[i][j] - (mol_a[i][j] - s)).abs() < 1e-13, "[{i}][{j}]");
        }
    }
}

#[test]
fn nothing_requested_computes_nothing() {
    let w = water();
    let (out, warnings) = run(&w, &only(|_| {}));
    let p = out.unwrap();
    assert!(warnings.is_empty() && p.gaps.is_empty());
    assert!(p.esp_atoms.is_none() && p.alpha_tensor.is_none() && p.hirshfeld.is_none());
    assert!(p.c6.is_none() && p.dipole.is_none() && p.density_second_moment.is_none());
    // coordinates / atomic numbers are always there
    assert_eq!(p.atomic_numbers, vec![8, 1, 1]);
}

#[test]
fn unknown_c6_strings_are_errors() {
    let w = water();
    for (src, part) in [(Some("tsx"), None), (None, Some("mulliken"))] {
        let (out, _) = run(
            &w,
            &only(|r| {
                r.c6 = true;
                r.c6_source = src.map(String::from);
                r.c6_partition = part.map(String::from);
            }),
        );
        assert!(out.is_err(), "{src:?} {part:?} must be rejected");
    }
}

#[test]
fn c6_provenance_tracks_the_parsed_source_and_partition() {
    let w = water();
    let (out, warnings) = run(
        &w,
        &only(|r| {
            r.c6 = true;
            r.c6_source = Some("pdep".into());
            r.c6_partition = Some("becke".into());
        }),
    );
    let p = out.unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let c6 = p.c6.unwrap();
    assert_eq!((c6.source, c6.partition), ("pdep", "becke"));
    assert!(
        c6.alpha_ct_dynamic.is_some(),
        "PDEP source carries alpha_ct(i omega)"
    );
    assert_eq!(c6.alpha_atomic_dynamic.len(), 3);
    // The exported nodes and weights are the configured quadrature, built
    // independently here (a dropped or swapped weight changes them).
    let (freqs, weights) = ferric_rpa::quadrature::build_quadrature(&w.cfg.quadrature);
    assert_eq!(c6.freqs, freqs);
    assert_eq!(c6.weights, weights);
    assert!(c6.weights.iter().all(|&x| x > 0.0) && c6.weights != c6.freqs);
    // partition = "becke" means the per-atom alpha(i omega) is the direct Becke
    // dynamic call at those nodes (a Hirshfeld partition differs at O(1e-2)).
    let direct = pdep_polarizability_becke_dynamic(
        &w.mol,
        &w.obs,
        &w.obs_bs,
        &w.dfbs,
        &w.rhf,
        Operator::coulomb(),
        &w.cfg,
        &c6.freqs,
    )
    .unwrap();
    let mut worst = 0.0f64;
    for (a, atom) in c6.alpha_atomic_dynamic.iter().enumerate() {
        for (k, t) in atom.iter().enumerate() {
            for i in 0..3 {
                for j in 0..3 {
                    worst = worst.max((t[i][j] - direct[a][k][i][j]).abs());
                }
            }
        }
    }
    assert!(
        worst < 1e-10,
        "bundle alpha(i omega) vs direct Becke dynamic: {worst:.3e}"
    );
}
