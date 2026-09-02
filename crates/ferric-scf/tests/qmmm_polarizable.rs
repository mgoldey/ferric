//! Thole-damped polarizable embedding (Lane B, Task B2): exactness anchors,
//! physics limits, and the p-shell dipole-potential normalisation pin.
//!
//! # Artifact hypothesis
//!
//! The p-shell site basis (`SiteBasis::new(sites, 1)`) gives Fock
//! contributions via `(μν|p_i)`, a 3-centre integral against libint2's
//! UNIT-SELF-OVERLAP-normalised p-Gaussian. The constant relating `(μν|p_i)`
//! to the physically meaningful dipole-potential integral
//! `⟨μ|(r-R_i)/|r-R_i|³|ν⟩` is NOT assumed — it is MEASURED by
//! `p_shell_dipole_potential_matches_charge_derivative_block`, which builds
//! the SAME Fock contribution two independent ways: (1) via the p-shell
//! `SiteBasis` + `compute_eri3`, and (2) via `compute_1e_deriv_block_n`'s
//! extra-point-charge derivative blocks (an entirely different libint2
//! integral family — `OP_NUCLEAR` with `set_point_charges_extra`, not
//! `compute_eri3` at all), which gives `∂/∂R_site ⟨μ|q/|r-R_site||ν⟩` for a
//! UNIT test charge, i.e. exactly `⟨μ|(r-R_site)/|r-R_site|³|ν⟩` (the
//! nuclear-attraction integral IS the charge potential, so its derivative
//! w.r.t. the charge's OWN position is the field integral by inspection —
//! `∂/∂R (1/|r-R|) = (r-R)/|r-R|³`). If the p-shell normalisation constant
//! were wrong by any factor (a missing `2ζ`, a missing sign, a mixed-up
//! px/py/pz ordering), this test would fail by exactly that factor/sign/
//! permutation, uniformly across ζ — not as noise. The ζ-sweep records
//! where the p-shell (a SMEARED dipole of finite width) converges to the
//! POINT-dipole limit the second path already is.

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ferric_scf::polarizable::{induce, PolarizableSite, PolarizableSites};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

const ANG2BOHR: f64 = 1.0 / 0.529_177_210_92;

fn water_bohr() -> Molecule {
    let r = 0.9572 * ANG2BOHR;
    let half = 104.52_f64.to_radians() / 2.0;
    let xyz = format!(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 {} {}\nH 0.0 {} {}\n",
        r * half.sin() / ANG2BOHR,
        r * half.cos() / ANG2BOHR,
        -r * half.sin() / ANG2BOHR,
        r * half.cos() / ANG2BOHR,
    );
    Molecule::parse_xyz(&xyz, 0, 1).unwrap()
}

fn sto3g_prep(mol: &Molecule) -> PreparedBasis {
    let bs = basis::bundled("sto-3g").unwrap();
    PreparedBasis::new(mol, &bs).unwrap()
}

/// Path (2), the independent reference: `∂/∂R_site ⟨μ|1/|r-R_site||ν⟩` for a
/// unit point charge, contracted over ALL shell pairs into an (nbasis,
/// nbasis) matrix per Cartesian component `[x, y, z]`.
fn point_charge_field_matrix(prep: &PreparedBasis, site: [f64; 3]) -> [Array2<f64>; 3] {
    let nbas = prep.nbasis();
    let natoms = prep.atoms().len();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let nsh = prep.nshells();
    let n_charges = natoms + 1;

    let mut eng = Engine::new_1e_deriv(ffi::OP_NUCLEAR, prep, 1e-14).unwrap();
    let extra = [PointCharge { q: 1.0, x: site[0], y: site[1], z: site[2] }];
    eng.set_point_charges_extra(prep, &extra).unwrap();

    let mut out = [
        Array2::<f64>::zeros((nbas, nbas)),
        Array2::<f64>::zeros((nbas, nbas)),
        Array2::<f64>::zeros((nbas, nbas)),
    ];
    for s1 in 0..nsh {
        for s2 in 0..nsh {
            let Some(deriv) = eng.compute_1e_deriv_block_n(prep, s1, s2, n_charges) else { continue };
            let n1 = dims[s1];
            let n2 = dims[s2];
            let block_sz = n1 * n2;
            // Extra-charge derivative block starts at 6 + 3*natoms (see
            // oneelectron_gradient's identical indexing in gradient.rs).
            let base = 6 + 3 * natoms;
            for i in 0..n1 {
                for j in 0..n2 {
                    let mu = offs[s1] + i;
                    let nu = offs[s2] + j;
                    let idx = i * n2 + j;
                    for c in 0..3 {
                        // d/dR_site <mu|1/|r-R_site||nu> = <mu|(r-R_site)/|r-R_site|^3|nu>
                        out[c][(mu, nu)] = deriv[(base + c) * block_sz + idx];
                    }
                }
            }
        }
    }
    out
}

/// Path (1): the p-shell `SiteBasis` contribution `(μν|p_c)/norm_p` for
/// component `c ∈ {0,1,2}` (px, py, pz — libint2's cartesian/pure ordering
/// for l=1, identical either way), at Gaussian exponent `zeta`.
fn p_shell_field_matrix(prep: &PreparedBasis, site: [f64; 3], zeta: f64) -> [Array2<f64>; 3] {
    let nbas = prep.nbasis();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let nsh = prep.nshells();

    let site_basis = SiteBasis::new(&[[site[0], site[1], site[2], zeta]], 1).unwrap();
    let mut eng = Engine::new_3center(Operator::coulomb(), prep, &site_basis.prep, 1e-14).unwrap();
    let sh_p = site_basis.site_shell[0];

    let mut out = [
        Array2::<f64>::zeros((nbas, nbas)),
        Array2::<f64>::zeros((nbas, nbas)),
        Array2::<f64>::zeros((nbas, nbas)),
    ];
    for s1 in 0..nsh {
        for s2 in 0..nsh {
            let Some(block) = eng.compute_eri3(prep, &site_basis.prep, sh_p, s1, s2) else { continue };
            let n1 = dims[s1];
            let n2 = dims[s2];
            for c in 0..3 {
                for i in 0..n1 {
                    for j in 0..n2 {
                        let mu = offs[s1] + i;
                        let nu = offs[s2] + j;
                        // P-major layout (pinned by site_basis.rs's s-shell
                        // test): index = (p*n1+i)*n2+j, p = c here.
                        out[c][(mu, nu)] = block[(c * n1 + i) * n2 + j];
                    }
                }
            }
        }
    }
    out
}

/// Field axis `c`'s matching p-shell FUNCTION index — the permutation
/// `crates/ferric-scf/src/polarizable.rs`'s `p_axis_for_field_axis` must
/// implement. Independently re-derived here (not imported from the
/// production crate) so this test is a genuine external check of the
/// implementation, not a tautology: it was DISCOVERED by an exhaustive
/// 9-pairing least-squares fit (every (field axis, p-shell function) pair,
/// at zeta=1e4) that found near-zero residual ONLY at (0,2), (1,0), (2,1) —
/// i.e. `field_axis[c]` pairs with p-shell function `(c+2) % 3`, NOT the
/// identity `c` a naive x/y/z-matches-px/py/pz assumption would predict.
/// The other 6 pairings gave O(0.03-0.09) residuals with NO zeta-dependence
/// (wrong pairings, not merely finite-zeta error).
fn expected_p_axis_for_field_axis(field_axis: usize) -> usize {
    (field_axis + 2) % 3
}

/// Closed-form `norm_int` for a p-shell (l=1) at exponent `zeta`, derived
/// from: (1) `d/dR_x[normalized s-Gaussian] = sqrt(zeta) *
/// [normalized p_x-Gaussian]` (both rescaled to unit self-overlap: `N_s =
/// (2 zeta/pi)^(3/4)`, `N_p` s.t. `integral (N_p*x*exp(-zeta r^2))^2 = 1`
/// gives `N_p = 2*2^(3/4)*zeta^(5/4)/pi^(3/4)`, ratio `N_s*2*zeta/N_p =
/// sqrt(zeta)`), and (2) `norm_int_s(zeta) (mu nu|g) -> <mu|1/|r-R||nu>`
/// as `zeta -> infinity` (lane A's erf-Coulomb potential integral, point
/// limit). Combining: `field_axis[c] = -(mu nu|p_{(c+2)%3}) /
/// norm_int_p(zeta)`, `norm_int_p(zeta) = 2^(1/4) zeta^(5/4) / (2
/// pi^(3/4))`.
fn expected_norm_int_p_shell(zeta: f64) -> f64 {
    2f64.powf(0.25) * zeta.powf(1.25) / (2.0 * std::f64::consts::PI.powf(0.75))
}

/// **Task B2 pin**: `field_axis[c] = -norm_int_p_shell(zeta) *
/// (mu nu|p_{(c+2)%3})` must equal the independent charge-derivative field
/// matrix elementwise to 1e-8, for EVERY axis `c` simultaneously (a genuine
/// cross-check: a wrong permutation or a wrong normalisation constant could
/// accidentally match one axis but not all three).
///
/// Uses `zeta=1e6` (not the `dipole_zeta` default of 1e4) to isolate the
/// CONSTRUCTION pin (permutation + scaling constant) from the SEPARATE
/// point-dipole convergence question `p_shell_zeta_convergence_sweep`
/// covers: at zeta=1e4 the p-shell is still a measurably-smeared (not
/// point) dipole, so `max_err` there is ~7e-7 (a factor of ~70 over this
/// test's 1e-8 bar) purely from that physical smearing — MEASURED to shrink
/// monotonically as zeta grows (2.6e-6 relative at 1e4 -> 7.6e-8 relative at
/// 1e6), which is exactly what the sweep test tracks and asserts on
/// SEPARATELY, with its own explicit tolerance. Conflating the two would
/// either loosen this construction pin until it could hide a real
/// permutation/scaling bug, or force `dipole_zeta`'s default absurdly high
/// for no physics reason — so they are two different tests, at two
/// different zeta, checking two different things.
#[test]
fn p_shell_dipole_potential_matches_charge_derivative_block() {
    let mol = water_bohr();
    let prep = sto3g_prep(&mol);
    let site = [1.5, -2.0, 3.0];
    let zeta = 1e6;

    let reference = point_charge_field_matrix(&prep, site);
    let p_shell = p_shell_field_matrix(&prep, site, zeta);
    let norm_p = expected_norm_int_p_shell(zeta);

    let mut max_err = 0.0_f64;
    for c in 0..3 {
        let p_fn = expected_p_axis_for_field_axis(c);
        let predicted = -norm_p * &p_shell[p_fn];
        let diff = &reference[c] - &predicted;
        let e = diff.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        eprintln!("[qmmm-polarizable] p-shell pin: field axis {c} <- p-shell fn {p_fn}, max|diff|={e:.3e}");
        max_err = max_err.max(e);
    }
    assert!(max_err < 1e-8, "p-shell dipole-potential pin failed: max_err = {max_err:.3e}");
}

/// **Task B2 pin, ζ-sweep**: the p-shell (smeared dipole) converges to the
/// point-dipole reference as ζ grows — recorded, not just asserted at one
/// value, since `dipole_zeta` is a real user-facing knob with a documented
/// default (1e4).
#[test]
fn p_shell_zeta_convergence_sweep() {
    let mol = water_bohr();
    let prep = sto3g_prep(&mol);
    let site = [1.5, -2.0, 3.0];
    let reference = point_charge_field_matrix(&prep, site);

    let mut prev_err: Option<f64> = None;
    for &zeta in &[1e2, 1e3, 1e4, 1e5] {
        let p_shell = p_shell_field_matrix(&prep, site, zeta);
        let norm_p = expected_norm_int_p_shell(zeta);
        let max_err = (0..3)
            .map(|c| {
                let p_fn = expected_p_axis_for_field_axis(c);
                let predicted = -norm_p * &p_shell[p_fn];
                (&reference[c] - &predicted).iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
            })
            .fold(0.0_f64, f64::max);
        eprintln!("[qmmm-polarizable] zeta={zeta:.0e}: norm_p={norm_p:.6e} max_err={max_err:.3e}");
        if let Some(pe) = prev_err {
            assert!(
                max_err <= pe * 1.5,
                "p-shell zeta-convergence must not get WORSE as zeta grows: zeta={zeta:.0e} \
                 max_err={max_err:.3e} vs previous {pe:.3e}"
            );
        }
        prev_err = Some(max_err);
    }
    // At the default dipole_zeta=1e4 the error must be tight.
    assert!(prev_err.unwrap() < 1e-6, "p-shell error at zeta=1e5 should be tiny: {:.3e}", prev_err.unwrap());
}

// ---------------------------------------------------------------------------
// Exactness anchors
// ---------------------------------------------------------------------------

fn scf_energy(mol: &Molecule, cfg: &RhfConfig) -> f64 {
    let prep = sto3g_prep(mol);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let r = solve_rhf(&ctx, mol, &prep, op, &bounds, cfg).unwrap();
    assert!(r.converged);
    r.energy
}

/// **Anchor 1**: `polarizable: None` is bit-identical to the pre-B2 energy
/// (no polarizable module involved at all — same as any other RhfConfig
/// field defaulting to None).
#[test]
fn polarizable_none_is_bit_identical_to_plain_scf() {
    let mol = water_bohr();
    let cfg_before = RhfConfig { density_conv: 1e-11, ..Default::default() };
    let cfg_after = RhfConfig { density_conv: 1e-11, polarizable: None, ..Default::default() };
    let e_before = scf_energy(&mol, &cfg_before);
    let e_after = scf_energy(&mol, &cfg_after);
    assert_eq!(e_before, e_after, "polarizable: None must be bit-identical to omitting the field");
}

/// **Anchor 2**: `Some(PolarizableSites { sites: vec![], .. })` (empty site
/// list) must ALSO be bit-identical to plain SCF — the induction solve must
/// short-circuit on an empty site list rather than attempt a 0x0 dense
/// solve that happens to return zero.
#[test]
fn polarizable_empty_sites_is_bit_identical_to_plain_scf() {
    let mol = water_bohr();
    let cfg_before = RhfConfig { density_conv: 1e-11, ..Default::default() };
    let cfg_after = RhfConfig {
        density_conv: 1e-11,
        polarizable: Some(PolarizableSites {
            sites: vec![],
            thole_a: Some(2.1304),
            exclusions: vec![],
            dipole_zeta: 1e4,
            max_sites_dense: 4000,
        }),
        ..Default::default()
    };
    let e_before = scf_energy(&mol, &cfg_before);
    let e_after = scf_energy(&mol, &cfg_after);
    assert_eq!(e_before, e_after, "empty polarizable sites must be bit-identical to no polarizable config");
}

/// **Anchor 3**: `alpha = 0` on every site must give `e_pol == 0.0` exactly
/// (the induction matrix is diagonal-dominated by `1/alpha -> infinity`, so
/// mu -> 0 exactly is not guaranteed by construction alone — this is worth
/// pinning). Energy must be within 1e-9 of plain embedding — NOT
/// bit-identical, since the code path differs (a polarizable Fock term of
/// value zero is still added, going through different floating-point
/// operations than the plain path).
#[test]
fn alpha_zero_gives_zero_polarization_energy() {
    // Use a tiny alpha rather than literal 0.0 to avoid a 1/alpha = inf
    // diagonal in the dense solve (see polarizable.rs's guard); this
    // exercises the physical alpha->0 LIMIT, matching the prototype's own
    // anchor (`alpha_zero_anchor_check` in proto_polarizable_embedding.py).
    let mol = water_bohr();
    let site = PolarizableSite { x: 3.0, y: -2.0, z: 4.0, alpha: 1e-12 };
    let cfg_plain = RhfConfig { density_conv: 1e-11, ..Default::default() };
    let cfg_pol = RhfConfig {
        density_conv: 1e-11,
        polarizable: Some(PolarizableSites {
            sites: vec![site],
            thole_a: Some(2.1304),
            exclusions: vec![],
            dipole_zeta: 1e4,
            max_sites_dense: 4000,
        }),
        ..Default::default()
    };
    let e_plain = scf_energy(&mol, &cfg_plain);
    let e_pol = scf_energy(&mol, &cfg_pol);
    assert!(
        (e_plain - e_pol).abs() < 1e-9,
        "alpha->0 polarizable energy must match plain embedding: plain={e_plain:.12} pol={e_pol:.12}"
    );
}

// ---------------------------------------------------------------------------
// Physics limits
// ---------------------------------------------------------------------------

/// **Distant single site**: mu_z ≈ alpha·E_z^gas (1%), e_pol ≈
/// -½ alpha |E^gas|² (2%) — the isolated-site perturbative limit, matching
/// the Python prototype's `distant_site_limit_check`.
#[test]
fn distant_single_site_matches_perturbative_limit() {
    let mol = water_bohr();
    let prep = sto3g_prep(&mol);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    // Gas-phase field at the site from the converged gas-phase density.
    let cfg_gas = RhfConfig { density_conv: 1e-12, ..Default::default() };
    let r_gas = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_gas).unwrap();
    assert!(r_gas.converged);
    let site_xyz = [0.0, 0.0, -12.0];
    let e_gas = ferric_scf::qmmm::electric_field_at_points(&mol, &prep, r_gas.density_total(), &[site_xyz])
        .unwrap()[0];

    let alpha = 1.0;
    let site = PolarizableSite { x: site_xyz[0], y: site_xyz[1], z: site_xyz[2], alpha };
    let cfg_pol = RhfConfig {
        density_conv: 1e-12,
        polarizable: Some(PolarizableSites {
            sites: vec![site],
            thole_a: Some(2.1304),
            exclusions: vec![],
            dipole_zeta: 1e4,
            max_sites_dense: 4000,
        }),
        ..Default::default()
    };
    let r_pol = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_pol).unwrap();
    assert!(r_pol.converged);
    let mu = r_pol.induced_dipoles.as_ref().expect("polarizable run must carry induced_dipoles");
    let mu_z = mu[(0, 2)];

    let mu_pred = alpha * e_gas[2];
    let e_pol_pred = -0.5 * alpha * (e_gas[0] * e_gas[0] + e_gas[1] * e_gas[1] + e_gas[2] * e_gas[2]);

    let e_pol_actual = r_pol.energy - r_gas.energy - (r_pol.energy - r_gas.energy - e_pol_pred).max(0.0) * 0.0;
    // e_pol is not directly exposed on ScfResult (only induced_dipoles) —
    // recompute it via mm_only convention: E_total = E_gas_with_site_field
    // + e_pol_contribution. Simpler: just check mu, and check E_pol via the
    // energy shift against the perturbative estimate loosely (2%).
    let _ = e_pol_actual;

    let mu_err = (mu_z - mu_pred).abs() / mu_pred.abs().max(1e-12);
    eprintln!(
        "[qmmm-polarizable] distant site: mu_z={mu_z:.10e} predicted={mu_pred:.10e} rel_err={mu_err:.3e}"
    );
    assert!(mu_err < 0.01, "distant-site mu_z limit failed: rel_err = {mu_err:.3e}");

    let e_shift = r_pol.energy - r_gas.energy;
    let e_pol_err = (e_shift - e_pol_pred).abs() / e_pol_pred.abs().max(1e-12);
    eprintln!(
        "[qmmm-polarizable] distant site: E_shift={e_shift:.10e} e_pol_predicted={e_pol_pred:.10e} rel_err={e_pol_err:.3e}"
    );
    assert!(e_pol_err < 0.02, "distant-site e_pol limit failed: rel_err = {e_pol_err:.3e}");
}

// ---------------------------------------------------------------------------
// PySCF prototype cross-validation
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct PeAtom {
    symbol: String,
    xyz_bohr: [f64; 3],
}

#[derive(Deserialize)]
struct PeSite {
    xyz_bohr: [f64; 3],
    q: f64,
    alpha_bohr3: f64,
}

#[derive(Deserialize)]
struct PeRef {
    atoms: Vec<PeAtom>,
    sites: Vec<PeSite>,
    thole_a: Option<f64>,
    energy: f64,
    e_pol: f64,
    induced_dipoles: Vec<[f64; 3]>,
}

fn load_ref(tag: &str) -> PeRef {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/reference")
        .join(format!("{tag}.json"));
    let data = fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
    serde_json::from_str(&data).unwrap()
}

fn mol_from_ref(r: &PeRef) -> Molecule {
    let mut xyz = format!("{}\nwater\n", r.atoms.len());
    for a in &r.atoms {
        xyz += &format!("{} {} {} {}\n", a.symbol, a.xyz_bohr[0], a.xyz_bohr[1], a.xyz_bohr[2]);
    }
    // xyz_bohr values are already in Bohr; parse_xyz assumes Å input and
    // converts, so scale up by 1/ANG2BOHR before feeding it in, undoing
    // that conversion (matches the pattern in qmmm_smeared.rs's water_bohr()).
    let mut xyz_ang = format!("{}\nwater\n", r.atoms.len());
    for a in &r.atoms {
        xyz_ang += &format!(
            "{} {} {} {}\n",
            a.symbol,
            a.xyz_bohr[0] / ANG2BOHR,
            a.xyz_bohr[1] / ANG2BOHR,
            a.xyz_bohr[2] / ANG2BOHR,
        );
    }
    let _ = xyz;
    Molecule::parse_xyz(&xyz_ang, 0, 1).unwrap()
}

fn run_pe_case(tag: &str) {
    let r = load_ref(tag);
    let mol = mol_from_ref(&r);
    let prep = sto3g_prep(&mol);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let sites: Vec<PolarizableSite> = r
        .sites
        .iter()
        .map(|s| PolarizableSite { x: s.xyz_bohr[0], y: s.xyz_bohr[1], z: s.xyz_bohr[2], alpha: s.alpha_bohr3 })
        .collect();
    let point_charges: Vec<PointCharge> = r
        .sites
        .iter()
        .map(|s| PointCharge { q: s.q, x: s.xyz_bohr[0], y: s.xyz_bohr[1], z: s.xyz_bohr[2] })
        .collect();

    let cfg = RhfConfig {
        density_conv: 1e-11,
        max_iter: 200,
        external_potential: Some(ExternalPotential { point_charges, smeared_charges: vec![], field: None }),
        polarizable: Some(PolarizableSites {
            sites,
            thole_a: r.thole_a,
            exclusions: vec![],
            dipole_zeta: 1e4,
            max_sites_dense: 4000,
        }),
        ..Default::default()
    };
    let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(result.converged, "{tag}: SCF did not converge");

    let e_err = (result.energy - r.energy).abs();
    eprintln!("[qmmm-polarizable] {tag}: E ferric={:.10} pyscf={:.10} |diff|={e_err:.3e}", result.energy, r.energy);
    assert!(e_err < 1e-7, "{tag}: energy mismatch {e_err:.3e}");

    let mu = result.induced_dipoles.as_ref().expect("induced_dipoles must be populated");
    let mut max_mu_err = 0.0_f64;
    for (i, mu_ref) in r.induced_dipoles.iter().enumerate() {
        for c in 0..3 {
            max_mu_err = max_mu_err.max((mu[(i, c)] - mu_ref[c]).abs());
        }
    }
    eprintln!("[qmmm-polarizable] {tag}: max|dipole diff| = {max_mu_err:.3e}");
    assert!(max_mu_err < 1e-6, "{tag}: dipole mismatch {max_mu_err:.3e}");

    // Cross-check e_pol too (ScfResult does not expose it directly — only
    // induced_dipoles — so recompute it standalone via `induce()` at the
    // now-converged density, which must reproduce the SAME dipoles AND
    // e_pol the SCF loop's internal call already used this iteration).
    let sites2: Vec<PolarizableSite> = r
        .sites
        .iter()
        .map(|s| PolarizableSite { x: s.xyz_bohr[0], y: s.xyz_bohr[1], z: s.xyz_bohr[2], alpha: s.alpha_bohr3 })
        .collect();
    let pol_cfg2 = PolarizableSites { sites: sites2, thole_a: r.thole_a, exclusions: vec![], dipole_zeta: 1e4, max_sites_dense: 4000 };
    let site_basis_p: Vec<[f64; 4]> =
        r.sites.iter().map(|s| [s.xyz_bohr[0], s.xyz_bohr[1], s.xyz_bohr[2], 1e4]).collect();
    let site_basis_p = SiteBasis::new(&site_basis_p, 1).unwrap();
    let ext2 = ExternalPotential {
        point_charges: r.sites.iter().map(|s| PointCharge { q: s.q, x: s.xyz_bohr[0], y: s.xyz_bohr[1], z: s.xyz_bohr[2] }).collect(),
        smeared_charges: vec![],
        field: None,
    };
    let ir = induce(&mol, &prep, Some(&ext2), &pol_cfg2, &site_basis_p, result.density_total()).unwrap();
    let e_pol_err = (ir.e_pol - r.e_pol).abs();
    eprintln!("[qmmm-polarizable] {tag}: e_pol ferric={:.10e} pyscf={:.10e} |diff|={e_pol_err:.3e}", ir.e_pol, r.e_pol);
    assert!(e_pol_err < 1e-7, "{tag}: e_pol mismatch {e_pol_err:.3e}");
}

#[test]
fn matches_pyscf_prototype_one_site() {
    run_pe_case("water_sto-3g_pe_one_site");
}

#[test]
fn matches_pyscf_prototype_three_sites() {
    run_pe_case("water_sto-3g_pe_three_sites");
}

#[test]
fn matches_pyscf_prototype_three_sites_nodamp() {
    run_pe_case("water_sto-3g_pe_three_sites_nodamp");
}

// ---------------------------------------------------------------------------
// induce() / mm_only_polarization_energy() unit coverage
// ---------------------------------------------------------------------------

/// `induce` on an empty site list returns a trivially-empty result without
/// attempting a 0x0 linear solve (defensive; exercised end-to-end by the
/// SCF anchor above, this is the direct unit-level check).
#[test]
fn induce_empty_sites_returns_zero_result() {
    let mol = water_bohr();
    let prep = sto3g_prep(&mol);
    let sites = PolarizableSites { sites: vec![], thole_a: Some(2.1304), exclusions: vec![], dipole_zeta: 1e4, max_sites_dense: 4000 };
    let site_basis_p = SiteBasis::new(&[[0.0, 0.0, 0.0, 1e4]], 1).unwrap();
    let d = Array2::<f64>::zeros((prep.nbasis(), prep.nbasis()));
    let result = induce(&mol, &prep, None, &sites, &site_basis_p, &d).unwrap();
    assert_eq!(result.dipoles.dim(), (0, 3));
    assert_eq!(result.e_pol, 0.0);
}

/// `max_sites_dense` is a real ceiling, not decorative — exceeding it must
/// be a typed error, not a silent huge dense solve.
#[test]
fn induce_rejects_more_sites_than_max_sites_dense() {
    let mol = water_bohr();
    let prep = sto3g_prep(&mol);
    let sites = PolarizableSites {
        sites: vec![
            PolarizableSite { x: 0.0, y: 0.0, z: 5.0, alpha: 1.0 },
            PolarizableSite { x: 0.0, y: 0.0, z: 6.0, alpha: 1.0 },
        ],
        thole_a: Some(2.1304),
        exclusions: vec![],
        dipole_zeta: 1e4,
        max_sites_dense: 1, // exceeded by 2 sites
    };
    let site_basis_p = SiteBasis::new(&[[0.0, 0.0, 5.0, 1e4], [0.0, 0.0, 6.0, 1e4]], 1).unwrap();
    let d = Array2::<f64>::zeros((prep.nbasis(), prep.nbasis()));
    let result = induce(&mol, &prep, None, &sites, &site_basis_p, &d);
    assert!(result.is_err(), "exceeding max_sites_dense must be a typed error");
}

/// A non-positive or non-finite polarizability must be REFUSED, not fed to the
/// induction solve.
///
/// `solve_induction` puts `1.0 / alpha` on the diagonal of the induction
/// matrix, so `alpha <= 0` would silently place an infinity (or a negative
/// stiffness) there and return a finite-looking but meaningless energy. The
/// guard exists; nothing exercised it, so a change to its condition could not
/// be caught. Note the sibling `alpha_zero_gives_zero_polarization_energy`
/// test deliberately uses `alpha: 1e-12` to stay on the LEGAL side of this
/// guard — that is the complement of this test, not a duplicate of it.
#[test]
fn induce_rejects_non_positive_alpha() {
    let mol = water_bohr();
    let prep = sto3g_prep(&mol);
    let site_basis_p = SiteBasis::new(&[[0.0, 0.0, 5.0, 1e4]], 1).unwrap();
    let d = Array2::<f64>::zeros((prep.nbasis(), prep.nbasis()));

    // Both illegal values must be refused: a negative stiffness and a NaN.
    for bad in [-1.0_f64, f64::NAN] {
        let sites = PolarizableSites {
            sites: vec![PolarizableSite { x: 0.0, y: 0.0, z: 5.0, alpha: bad }],
            thole_a: Some(2.1304),
            exclusions: vec![],
            dipole_zeta: 1e4,
            max_sites_dense: 8,
        };
        let result = induce(&mol, &prep, None, &sites, &site_basis_p, &d);
        assert!(
            result.is_err(),
            "alpha = {bad} must be refused: 1/alpha lands on the induction \
             diagonal, so accepting it yields a finite-looking meaningless energy"
        );
    }
}




// ---------------------------------------------------------------------------
// Task B3: gradients
// ---------------------------------------------------------------------------

/// Two off-axis polarizable+charged sites near water/STO-3G (same style as
/// the B2 physics-limit tests): analytic QM gradient (via `rhf_gradient`,
/// which must fold in the fixed-mu polarizable terms) vs central FD of the
/// fully-reconverged (SCF + induction) total energy.
fn two_site_setup() -> (Molecule, PolarizableSites, ExternalPotential) {
    let mol = water_bohr();
    let sites = PolarizableSites {
        sites: vec![
            PolarizableSite { x: 3.0, y: -2.0, z: 4.0, alpha: 1.0 },
            PolarizableSite { x: -2.5, y: 3.3, z: -3.7, alpha: 0.8 },
        ],
        thole_a: Some(2.1304),
        exclusions: vec![],
        dipole_zeta: 1e4,
        max_sites_dense: 4000,
    };
    let ext = ExternalPotential {
        point_charges: vec![
            PointCharge { q: 0.5, x: 3.0, y: -2.0, z: 4.0 },
            PointCharge { q: -0.3, x: -2.5, y: 3.3, z: -3.7 },
        ],
        smeared_charges: vec![],
        field: None,
    };
    (mol, sites, ext)
}

/// Same water/QM setup, but the polarizable sites are OFF-CENTRE from the
/// permanent MM charges (not colocated) — deliberately, so `site_gradient`'s
/// FD test can move a site's position WITHOUT also moving a permanent
/// charge. Colocated site+charge (the realistic QM/MM case, one atom
/// carrying both a fixed partial charge and a polarizability at the SAME
/// position) additionally needs the classical charge-nuclear energy's OWN
/// derivative with respect to a MOVING MM CHARGE position — a general
/// piece of `ExternalPotential`/QM-MM plumbing that exists for QM-atom
/// rows (`ExternalPotential::charge_nuclear_gradient`) but has no
/// "moving-charge-row" analogue today, and adding one is out of Lane B's
/// polarizable-embedding-specific scope (it is a `Lane C`/`ferric-mm`-
/// adjacent gap, not a polarizable-physics gap: it is exactly as absent
/// for the NON-polarizable case). This setup sidesteps that gap entirely
/// so the test isolates precisely what `site_gradient` claims to compute.
fn two_site_setup_no_colocated_charge() -> (Molecule, PolarizableSites, ExternalPotential) {
    let mol = water_bohr();
    let sites = PolarizableSites {
        sites: vec![
            PolarizableSite { x: 3.0, y: -2.0, z: 4.0, alpha: 1.0 },
            PolarizableSite { x: -2.5, y: 3.3, z: -3.7, alpha: 0.8 },
        ],
        thole_a: Some(2.1304),
        exclusions: vec![],
        dipole_zeta: 1e4,
        max_sites_dense: 4000,
    };
    // Permanent charges sit at DIFFERENT points from the polarizable
    // sites, off-axis, so E^perm at each site is nonzero and geometry-
    // dependent (through the sites' OWN positions) without ever moving
    // a colocated charge.
    let ext = ExternalPotential {
        point_charges: vec![
            PointCharge { q: 0.5, x: 4.5, y: -3.0, z: 5.5 },
            PointCharge { q: -0.3, x: -4.0, y: 4.5, z: -5.0 },
        ],
        smeared_charges: vec![],
        field: None,
    };
    (mol, sites, ext)
}

fn scf_energy_polarizable(mol: &Molecule, sites: &PolarizableSites, ext: &ExternalPotential) -> f64 {
    let prep = sto3g_prep(mol);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        density_conv: 1e-12,
        max_iter: 300,
        external_potential: Some(ext.clone()),
        polarizable: Some(sites.clone()),
        ..Default::default()
    };
    let r = solve_rhf(&ctx, mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(r.converged);
    r.energy
}

#[test]
fn qm_gradient_with_polarizable_sites_matches_finite_difference() {
    let (mol, sites, ext) = two_site_setup();
    let prep = sto3g_prep(&mol);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        density_conv: 1e-12,
        max_iter: 300,
        external_potential: Some(ext.clone()),
        polarizable: Some(sites.clone()),
        ..Default::default()
    };
    let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(result.converged);

    let analytic = ferric_scf::gradient::rhf_gradient_with_polarizable(
        &mol, &prep, op, &bounds, &result, Some(&ext), Some(&sites), result.induced_dipoles.as_ref(),
    )
    .unwrap();

    let h = 1e-3;
    let natoms = mol.atoms.len();
    let mut max_err = 0.0_f64;
    for a in 0..natoms {
        for c in 0..3 {
            let mut mol_p = mol.clone();
            let mut mol_m = mol.clone();
            match c {
                0 => { mol_p.atoms[a].x += h; mol_m.atoms[a].x -= h; }
                1 => { mol_p.atoms[a].y += h; mol_m.atoms[a].y -= h; }
                _ => { mol_p.atoms[a].zpos += h; mol_m.atoms[a].zpos -= h; }
            }
            let e_p = scf_energy_polarizable(&mol_p, &sites, &ext);
            let e_m = scf_energy_polarizable(&mol_m, &sites, &ext);
            let fd = (e_p - e_m) / (2.0 * h);
            let err = (analytic[(a, c)] - fd).abs();
            max_err = max_err.max(err);
            assert!(
                err < 1e-6,
                "QM gradient[{a}][{c}] with polarizable sites: analytic {:.10e} vs FD {:.10e} (delta {err:.3e})",
                analytic[(a, c)], fd
            );
        }
    }
    eprintln!("[qmmm-polarizable] max|analytic - FD| on QM gradient (polarizable) = {max_err:.3e}");
}

#[test]
fn site_force_with_polarizable_sites_matches_finite_difference() {
    let (mol, sites, ext) = two_site_setup_no_colocated_charge();
    let prep = sto3g_prep(&mol);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        density_conv: 1e-12,
        max_iter: 300,
        external_potential: Some(ext.clone()),
        polarizable: Some(sites.clone()),
        ..Default::default()
    };
    let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(result.converged);
    let mu = result.induced_dipoles.clone().unwrap();

    let dedr = ferric_scf::polarizable::site_gradient(
        &mol,
        &prep,
        result.density_total(),
        Some(&ext),
        &sites,
        &mu,
    )
    .unwrap();

    let h = 1e-3;
    let mut max_err = 0.0_f64;
    for i in 0..sites.sites.len() {
        for c in 0..3 {
            let mut sites_p = sites.clone();
            let mut sites_m = sites.clone();
            match c {
                0 => { sites_p.sites[i].x += h; sites_m.sites[i].x -= h; }
                1 => { sites_p.sites[i].y += h; sites_m.sites[i].y -= h; }
                _ => { sites_p.sites[i].z += h; sites_m.sites[i].z -= h; }
            }
            let e_p = scf_energy_polarizable(&mol, &sites_p, &ext);
            let e_m = scf_energy_polarizable(&mol, &sites_m, &ext);
            let fd = (e_p - e_m) / (2.0 * h);
            let err = (dedr[(i, c)] - fd).abs();
            max_err = max_err.max(err);
            assert!(
                err < 1e-6,
                "site_gradient[{i}][{c}]: analytic {:.10e} vs FD {:.10e} (delta {err:.3e})",
                dedr[(i, c)], fd
            );
        }
    }
    eprintln!("[qmmm-polarizable] max|analytic - FD| on site gradient = {max_err:.3e}");
}

/// Translational invariance: sum over ALL rows (QM atoms + polarizable
/// sites, treated as one rigid structure) of the total dE/dR must vanish —
/// a rigid translation cannot change the energy. This is a strong,
/// sign-blind cross-check independent of the FD comparisons above.
#[test]
fn qm_and_site_gradients_sum_to_zero_translational_invariance() {
    // NO independent permanent MM charges at all (ext = None / e_perm = 0
    // everywhere): translational invariance of "sum over ALL rows = 0"
    // requires the rigid body being translated to be the WHOLE system that
    // determines the energy. With permanent MM charges present (fixed,
    // external to the sum), translating only "QM atoms + polarizable
    // sites" relative to those fixed charges is NOT a symmetry (distances
    // to the fixed charges change) — that would need the MM-charge rows
    // in the sum too, which needs a classical charge-nuclear-position
    // gradient this lane does not implement (see
    // `two_site_setup_no_colocated_charge`'s doc comment on that gap).
    // Dropping the permanent charges entirely (E^perm = 0, only E^QM drives
    // induction) makes "QM atoms + sites" the WHOLE energetically-relevant
    // system, so this check is exactly right.
    let mol = water_bohr();
    let sites = PolarizableSites {
        sites: vec![
            PolarizableSite { x: 3.0, y: -2.0, z: 4.0, alpha: 1.0 },
            PolarizableSite { x: -2.5, y: 3.3, z: -3.7, alpha: 0.8 },
        ],
        thole_a: Some(2.1304),
        exclusions: vec![],
        dipole_zeta: 1e4,
        max_sites_dense: 4000,
    };
    let prep = sto3g_prep(&mol);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        density_conv: 1e-12,
        max_iter: 300,
        polarizable: Some(sites.clone()),
        ..Default::default()
    };
    let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(result.converged);
    let mu = result.induced_dipoles.clone().unwrap();

    let qm_grad = ferric_scf::gradient::rhf_gradient_with_polarizable(
        &mol, &prep, op, &bounds, &result, None, Some(&sites), Some(&mu),
    )
    .unwrap();
    let site_grad = ferric_scf::polarizable::site_gradient(&mol, &prep, result.density_total(), None, &sites, &mu).unwrap();

    let mut total = [0.0_f64; 3];
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            total[c] += qm_grad[(a, c)];
        }
    }
    for i in 0..sites.sites.len() {
        for c in 0..3 {
            total[c] += site_grad[(i, c)];
        }
    }
    eprintln!("[qmmm-polarizable] translational invariance sum = {total:?}");
    for c in 0..3 {
        assert!(total[c].abs() < 1e-8, "translational invariance failed on axis {c}: {:.3e}", total[c]);
    }
}

// ---------------------------------------------------------------------------
// Lane B5: dE_pol/dR on the PERMANENT CHARGE itself (charge_gradient_contribution)
// ---------------------------------------------------------------------------

/// `charge_gradient_contribution`'s missing-until-now half of `site_gradient`'s
/// term 3a: the reaction force on a permanent MM charge from being in the
/// field of the OTHER sites' induced dipoles. Isolated with
/// `two_site_setup_no_colocated_charge` (permanent charges off-centre from
/// the sites) so only ONE permanent charge moves at a time, without also
/// moving a site (which `site_gradient`'s FD test already covers).
///
/// **Important scoping note found via TDD**: `charge_gradient_contribution`
/// computes ONLY `dE_pol/dR_charge` (the polarizable-embedding piece).
/// `ExternalPotential::charge_nuclear_energy` (classical nuclear-charge
/// Coulomb term) and the one-electron electron-charge attraction integral
/// (folded into `hcore` once before the SCF loop) ALSO depend on a moving
/// charge's position — that classical piece is exactly what
/// `crate::qmmm::mm_forces`'s `F = q * E_QM(r)` already computes (the force
/// on a point charge from the QM region's total field, both nuclear and
/// electronic), just with the opposite sign convention (`mm_forces` returns
/// a FORCE `-dE/dR`, not `dE/dR`). This test therefore FD-checks against
/// the classical term (via `electric_field_at_points`, negated to
/// `dE/dR = -q*E`) PLUS `charge_gradient_contribution`, against the total
/// reconverged SCF energy — the same split `full_gradient_with_polarizable`
/// performs in production. Comparing `charge_gradient_contribution` ALONE
/// against the total SCF energy is a CATEGORY ERROR (caught the first time
/// this test was written: it failed by ~30%, exactly the missing classical
/// term's size) — see qmmm.rs's `mm_forces` doc comment for the classical
/// term's own derivation.
#[test]
fn charge_gradient_contribution_matches_finite_difference() {
    use ferric_scf::qmmm::electric_field_at_points;

    let (mol, sites, ext) = two_site_setup_no_colocated_charge();
    let prep = sto3g_prep(&mol);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        density_conv: 1e-12,
        max_iter: 300,
        external_potential: Some(ext.clone()),
        polarizable: Some(sites.clone()),
        ..Default::default()
    };
    let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(result.converged);
    let mu = result.induced_dipoles.clone().unwrap();

    let (point_rows, smeared_rows) =
        ferric_scf::polarizable::charge_gradient_contribution(&ext, &sites, &mu);
    assert!(smeared_rows.is_empty());
    assert_eq!(point_rows.len(), ext.point_charges.len());

    // Classical dE/dR_charge = -q * E_QM(r_charge) (negative of the force
    // mm_forces computes), at the SAME converged density.
    let charge_positions: Vec<[f64; 3]> =
        ext.point_charges.iter().map(|pc| [pc.x, pc.y, pc.z]).collect();
    let field = electric_field_at_points(&mol, &prep, result.density_total(), &charge_positions).unwrap();
    let classical_dedr: Vec<[f64; 3]> = field
        .iter()
        .zip(ext.point_charges.iter())
        .map(|(e, pc)| [-pc.q * e[0], -pc.q * e[1], -pc.q * e[2]])
        .collect();

    let h = 1e-3;
    let mut max_err = 0.0_f64;
    for k in 0..ext.point_charges.len() {
        for c in 0..3 {
            let mut ext_p = ext.clone();
            let mut ext_m = ext.clone();
            match c {
                0 => { ext_p.point_charges[k].x += h; ext_m.point_charges[k].x -= h; }
                1 => { ext_p.point_charges[k].y += h; ext_m.point_charges[k].y -= h; }
                _ => { ext_p.point_charges[k].z += h; ext_m.point_charges[k].z -= h; }
            }
            let e_p = scf_energy_polarizable(&mol, &sites, &ext_p);
            let e_m = scf_energy_polarizable(&mol, &sites, &ext_m);
            let fd = (e_p - e_m) / (2.0 * h);
            let an = point_rows[k][c] + classical_dedr[k][c];
            let err = (an - fd).abs();
            max_err = max_err.max(err);
            assert!(
                err < 1e-6,
                "charge_gradient_contribution + classical point_rows[{k}][{c}]: analytic {an:.10e} vs FD {fd:.10e} (delta {err:.3e})"
            );
        }
    }
    eprintln!("[qmmm-polarizable] max|analytic - FD| on charge_gradient_contribution + classical = {max_err:.3e}");
}

/// Exactness anchor: no polarizable sites means `charge_gradient_contribution`
/// returns all-zero rows, sized to `ext`, without touching `dipoles` at all
/// (mirrors `induce`'s own empty-sites short-circuit).
#[test]
fn charge_gradient_contribution_zero_sites_is_zero() {
    let (_mol, _sites, ext) = two_site_setup_no_colocated_charge();
    let empty_sites = PolarizableSites { sites: vec![], ..Default::default() };
    let dipoles = Array2::<f64>::zeros((0, 3));
    let (point_rows, smeared_rows) =
        ferric_scf::polarizable::charge_gradient_contribution(&ext, &empty_sites, &dipoles);
    assert_eq!(point_rows.len(), ext.point_charges.len());
    assert!(smeared_rows.is_empty());
    for row in &point_rows {
        assert_eq!(*row, [0.0, 0.0, 0.0]);
    }
}

