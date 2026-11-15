//! The property bundle computed on a PDEP-RPA reference: ESP, field,
//! polarizabilities (molecular, per-atom, charge-transfer), charges, dipole,
//! C6/dynamic α, orbital moments -- and its NPZ / eigenpotential-cube export.
//!
//! One implementation serves both the CLI (`[rpa] export_npz = ...`) and the
//! Python `run_pdep_rpa(properties=True / export_npz=...)`, so the two produce
//! the same arrays by construction. Progress lines and warnings go through the
//! [`crate::property_bundle::PropertyReporter`] trait: the CLI prints them,
//! the Python binding collects them.
//!
//! Failure policy: a property that fails to compute is recorded in the `gaps`
//! field of [`crate::property_bundle::PdepProperties`] (and warned about) and
//! left `None`; the rest of the bundle is still produced. An unparseable
//! `c6_source` / `c6_partition` is a hard `Err`.

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_export::ml::{
    C6Export, C6Provenance, ChargeSchemes, DispersionBundle, NpzBundle, PolarizabilityBundle,
};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::{ScfResult, Spin};
use ferric_scf::rhf::RhfConfig;
use ndarray::Array2;

use crate::config::PdepRpaConfig;
use crate::dispersion::{
    casimir_polder_c6, pdep_dynamic_polarizability, ts_dynamic_polarizability, C6Result, C6Source,
    DispersionPartition,
};
use crate::properties::{
    atomic_effective_volumes_hirshfeld, chelpg_and_resp_charges, chelpg_charges,
    electric_field_at_atoms, esp_at_atoms, hirshfeld_charges, lowdin_charges, mulliken_charges,
    pdep_polarizability_becke, pdep_polarizability_hirshfeld, pdep_polarizability_static,
    resp_charges, RadialProatom,
};
use crate::quadrature::build_quadrature;
use crate::PdepRpaResult;

type Tensor3 = [[f64; 3]; 3];

/// Sink for the progress lines and warnings of the property computation.
pub trait PropertyReporter {
    /// A progress / result line (the CLI prints it to stdout).
    fn info(&mut self, msg: &str);
    /// A warning (the CLI prints it to stderr). The text carries its own
    /// `warning:` prefix.
    fn warn(&mut self, msg: &str);
}

/// Prints [`PropertyReporter::info`] to stdout and
/// [`PropertyReporter::warn`] to stderr.
pub struct StdReporter;

impl PropertyReporter for StdReporter {
    fn info(&mut self, msg: &str) {
        println!("{msg}");
    }
    fn warn(&mut self, msg: &str) {
        eprintln!("{msg}");
    }
}

/// Which properties to compute. [`Default`] is the `[rpa]` TOML default under
/// `export_npz`: everything on except the ESP surface shell.
#[derive(Debug, Clone)]
pub struct PdepPropertyRequest {
    /// ESP at the nuclei (`compute_esp`).
    pub esp: bool,
    /// ESP on a vdW surface shell (`compute_esp_surface`).
    pub esp_surface: bool,
    /// vdW-radius multiplier of the surface shell (`esp_surface_vdw_scale`).
    pub esp_surface_vdw_scale: f64,
    /// Lebedev order per atom of the shell (`esp_surface_n_angular`).
    pub esp_surface_n_angular: usize,
    /// Molecular static polarizability (`compute_polarizability`).
    pub polarizability: bool,
    /// Electric field at the nuclei (`compute_electric_field`).
    pub electric_field: bool,
    /// Per-atom Becke polarizability and the charge-transfer remainder
    /// (`compute_alpha_atomic`).
    pub alpha_atomic: bool,
    /// AO density matrix and its second moment (`compute_density_matrix`).
    pub density_matrix: bool,
    /// Molecular dipole (`compute_dipole`).
    pub dipole: bool,
    /// Löwdin charges (`compute_lowdin_charges`).
    pub lowdin: bool,
    /// Hirshfeld charges (`compute_hirshfeld_charges`).
    pub hirshfeld: bool,
    /// Mulliken charges (`compute_mulliken_charges`).
    pub mulliken: bool,
    /// CHELPG charges (`compute_chelpg_charges`).
    pub chelpg: bool,
    /// RESP charges (`compute_resp_charges`).
    pub resp: bool,
    /// C6 / dynamic per-atom α (`compute_c6`).
    pub c6: bool,
    /// `c6_source` string (`"ts"` default, `"pdep"`, `"mbd"`), parsed strictly.
    pub c6_source: Option<String>,
    /// `c6_partition` string (`"becke"`, `"hirshfeld"`; unset = the source's
    /// default), parsed strictly.
    pub c6_partition: Option<String>,
}

impl Default for PdepPropertyRequest {
    fn default() -> Self {
        Self {
            esp: true,
            esp_surface: false,
            esp_surface_vdw_scale: 1.4,
            esp_surface_n_angular: 110,
            polarizability: true,
            electric_field: true,
            alpha_atomic: true,
            density_matrix: true,
            dipole: true,
            lowdin: true,
            hirshfeld: true,
            mulliken: true,
            chelpg: true,
            resp: true,
            c6: true,
            c6_source: None,
            c6_partition: None,
        }
    }
}

/// Everything the property computation reads.
pub struct PropertyInputs<'a> {
    /// Parallel context (free-atom SCFs).
    pub ctx: &'a ParallelContext,
    /// The molecule.
    pub mol: &'a Molecule,
    /// Prepared orbital basis.
    pub prep: &'a PreparedBasis,
    /// Orbital basis set (Hirshfeld grids, free-atom SCFs).
    pub bs: &'a BasisSet,
    /// Prepared auxiliary basis.
    pub dfbs: &'a PreparedBasis,
    /// The reference SCF.
    pub result: &'a ScfResult,
    /// Coulomb operator of the reference.
    pub op: Operator,
    /// The PDEP-RPA settings the property responses reuse.
    pub rpa_cfg: &'a PdepRpaConfig,
    /// SCF settings of the reference (free-atom SCFs).
    pub rhf_config: &'a RhfConfig,
    /// Same-basis Hirshfeld proatom provider.
    pub proatom: &'a dyn Fn(i32, i32) -> Option<RadialProatom>,
}

/// The C6 / dynamic-α block of the bundle with its provenance.
#[derive(Debug, Clone)]
pub struct C6Properties {
    /// Atomic partition that produced the per-atom arrays (`"becke"` / `"hirshfeld"`).
    pub partition: &'static str,
    /// α(iω) source (`"ts"` / `"pdep"` / `"mbd"`).
    pub source: &'static str,
    /// Imaginary-frequency nodes.
    pub freqs: Vec<f64>,
    /// Quadrature weights.
    pub weights: Vec<f64>,
    /// Per-atom α^A(iω), `[natoms][nfreq]`.
    pub alpha_atomic_dynamic: Vec<Vec<Tensor3>>,
    /// α_CT(iω), `[nfreq]` (PDEP source only).
    pub alpha_ct_dynamic: Option<Vec<Tensor3>>,
    /// Per-atom-pair isotropic C6, (N, N).
    pub c6_iso: Array2<f64>,
    /// Per-atom-pair anisotropic C6, `[N][N]` 3x3.
    pub c6_aniso: Vec<Vec<Tensor3>>,
    /// Molecular isotropic C6.
    pub c6_molecular_iso: f64,
}

/// The computed bundle. A `None` field was not requested or failed (a failure
/// is also listed in [`Self::gaps`]).
#[derive(Debug, Clone, Default)]
pub struct PdepProperties {
    /// Atomic coordinates (N, 3), Bohr.
    pub coords: Array2<f64>,
    /// Atomic numbers.
    pub atomic_numbers: Vec<usize>,
    /// ESP at the nuclei.
    pub esp_atoms: Option<Vec<f64>>,
    /// ESP on the surface shell.
    pub esp_surface: Option<Vec<f64>>,
    /// The shell's points (N, 3), Bohr.
    pub esp_points: Option<Array2<f64>>,
    /// Electric field at the nuclei, one `[x, y, z]` per atom.
    pub electric_field: Option<Vec<[f64; 3]>>,
    /// Molecular static polarizability.
    pub alpha_tensor: Option<Tensor3>,
    /// Per-atom intrinsic Becke polarizability.
    pub alpha_atomic: Option<Vec<Tensor3>>,
    /// Charge-transfer remainder `alpha_tensor - sum(alpha_atomic)`.
    pub alpha_ct: Option<Tensor3>,
    /// Whether the AO density matrix was requested.
    pub density_matrix: bool,
    /// Molecular dipole.
    pub dipole: Option<[f64; 3]>,
    /// Löwdin charges.
    pub lowdin: Option<Vec<f64>>,
    /// Hirshfeld charges.
    pub hirshfeld: Option<Vec<f64>>,
    /// Mulliken charges.
    pub mulliken: Option<Vec<f64>>,
    /// CHELPG charges.
    pub chelpg: Option<Vec<f64>>,
    /// RESP charges.
    pub resp: Option<Vec<f64>>,
    /// C6 block.
    pub c6: Option<C6Properties>,
    /// Orbital centroids (nmo, 3) (restricted references).
    pub orbital_centers: Option<Array2<f64>>,
    /// Orbital spreads (restricted references).
    pub orbital_spreads: Option<Vec<f64>>,
    /// Density second moment (3, 3) about the origin.
    pub density_second_moment: Option<Array2<f64>>,
    /// Requested properties that failed, one message each.
    pub gaps: Vec<String>,
}

fn mean_trace(t: &Tensor3) -> f64 {
    (t[0][0] + t[1][1] + t[2][2]) / 3.0
}

fn fmt_q(q: &[f64]) -> Vec<f64> {
    q.iter().map(|v| (v * 1e4).round() / 1e4).collect()
}

/// ESP surface shell, ESP at nuclei and field at nuclei.
fn compute_site_properties(
    inp: &PropertyInputs,
    req: &PdepPropertyRequest,
    rep: &mut dyn PropertyReporter,
    out: &mut PdepProperties,
) {
    let (mol, prep, d) = (inp.mol, inp.prep, inp.result.density_total());
    if req.esp {
        match esp_at_atoms(mol, prep, d) {
            Ok(v) => out.esp_atoms = Some(v),
            Err(e) => {
                rep.warn(&format!("warning: esp_at_atoms failed: {e}"));
                out.gaps.push(format!("esp_atoms: {e}"));
            }
        }
    }
    if req.esp_surface {
        match ferric_scf::properties::esp_on_surface(
            mol,
            prep,
            d,
            req.esp_surface_vdw_scale,
            req.esp_surface_n_angular,
        ) {
            Ok((pts, v)) => {
                let mut arr = Array2::<f64>::zeros((pts.len(), 3));
                for (i, p) in pts.iter().enumerate() {
                    arr[(i, 0)] = p[0];
                    arr[(i, 1)] = p[1];
                    arr[(i, 2)] = p[2];
                }
                out.esp_points = Some(arr);
                out.esp_surface = Some(v);
            }
            Err(e) => {
                rep.warn(&format!("warning: esp_on_surface failed: {e}"));
                out.gaps.push(format!("esp_surface: {e}"));
            }
        }
    }
    if req.electric_field {
        match electric_field_at_atoms(mol, prep, d) {
            Ok(v) => out.electric_field = Some(v),
            Err(e) => {
                rep.warn(&format!("warning: electric_field_at_atoms failed: {e}"));
                out.gaps.push(format!("electric_field: {e}"));
            }
        }
    }
}

/// Molecular α, per-atom Becke α and the charge-transfer remainder.
fn compute_polarizabilities(
    inp: &PropertyInputs,
    req: &PdepPropertyRequest,
    rep: &mut dyn PropertyReporter,
    out: &mut PdepProperties,
) {
    if req.polarizability {
        match pdep_polarizability_static(
            inp.mol,
            inp.prep,
            inp.dfbs,
            inp.result,
            inp.op,
            inp.rpa_cfg,
        ) {
            Ok(p) => {
                rep.info(&format!(
                    "Polarizability α (a.u.):  iso={:.4}, principal=[{:.4}, {:.4}, {:.4}]",
                    p.iso, p.principal[0], p.principal[1], p.principal[2]
                ));
                out.alpha_tensor = Some(p.tensor);
            }
            Err(e) => {
                rep.warn(&format!("warning: polarizability failed: {e}"));
                out.gaps.push(format!("alpha_tensor (polarizability): {e}"));
            }
        }
    }
    if req.alpha_atomic {
        match pdep_polarizability_becke(
            inp.mol,
            inp.prep,
            inp.bs,
            inp.dfbs,
            inp.result,
            inp.op,
            inp.rpa_cfg,
        ) {
            Ok(v) => {
                rep.info(&format!(
                    "Per-atom intrinsic Becke α (iso, a.u.): {:?}",
                    v.iter().map(mean_trace).collect::<Vec<_>>()
                ));
                out.alpha_atomic = Some(v);
            }
            Err(e) => {
                rep.warn(&format!("warning: per-atom α (Becke) failed: {e}"));
                out.gaps.push(format!("alpha_atomic (per-atom α): {e}"));
            }
        }
    }
    // Charge-transfer remainder α_CT = α_mol − Σ_A α^A: emitted exactly when
    // both inputs were computed. Both come from the same response (same RI
    // kernel, same orbitals, frozen_core = 0), so the remainder is the
    // Krishtal charge-delocalization polarizability, not a definition gap.
    if let (Some(mol_a), Some(per_atom)) = (out.alpha_tensor.as_ref(), out.alpha_atomic.as_deref())
    {
        let ct = crate::properties::charge_transfer_remainder(mol_a, per_atom);
        rep.info(&format!(
            "Charge-transfer α_CT (iso, a.u.): {:.4}",
            mean_trace(&ct)
        ));
        out.alpha_ct = Some(ct);
    }
}

/// Run one charge scheme: print it on success, warn on failure.
fn charge_scheme(
    label: &str,
    r: Result<Vec<f64>, ferric_core::FerricError>,
    rep: &mut dyn PropertyReporter,
) -> Option<Vec<f64>> {
    match r {
        Ok(q) => {
            rep.info(&format!("{label} charges (e): {:?}", fmt_q(&q)));
            Some(q)
        }
        Err(e) => {
            rep.warn(&format!("warning: {label} charges failed: {e}"));
            None
        }
    }
}

/// Dipole and the Löwdin / Hirshfeld / Mulliken / CHELPG / RESP charges.
fn compute_charges_and_dipole(
    inp: &PropertyInputs,
    req: &PdepPropertyRequest,
    rep: &mut dyn PropertyReporter,
    out: &mut PdepProperties,
) {
    let (mol, prep, d) = (inp.mol, inp.prep, inp.result.density_total());
    // Molecular dipole μ = −Tr(P·D) + Σ_A Z_A R_A of the total density
    // (QC ground truth vs partition-derived Löwdin/Hirshfeld dipoles).
    // Origin [0,0,0]; neutral molecules are origin-independent.
    if req.dipole {
        match ferric_scf::properties::dipole_moment(mol, prep, d) {
            Ok(mu) => {
                let mag = ferric_scf::properties::dipole_magnitude(&mu);
                rep.info(&format!(
                    "dipole (e·a0): [{:.4}, {:.4}, {:.4}] |μ| = {:.4} ({:.4} D)",
                    mu[0],
                    mu[1],
                    mu[2],
                    mag,
                    mag * ferric_scf::properties::DEBYE_PER_AU
                ));
                out.dipole = Some(mu);
            }
            Err(e) => rep.warn(&format!("warning: dipole failed: {e}")),
        }
    }
    if req.lowdin {
        out.lowdin = charge_scheme("Löwdin", lowdin_charges(mol, prep, d), rep);
    }
    if req.hirshfeld {
        out.hirshfeld = charge_scheme(
            "Hirshfeld",
            hirshfeld_charges(mol, inp.bs, d, Some(inp.proatom)),
            rep,
        );
    }
    if req.mulliken {
        out.mulliken = charge_scheme("Mulliken", mulliken_charges(mol, prep, d), rep);
    }
    // CHELPG and RESP differ ONLY in the least-squares solve; both evaluate
    // the same molecular ESP over the same grid, so when both are requested
    // they share one grid.
    match (req.chelpg, req.resp) {
        (true, true) => match chelpg_and_resp_charges(mol, prep, d) {
            Ok((cq, rq)) => {
                rep.info(&format!("CHELPG charges (e): {:?}", fmt_q(&cq)));
                rep.info(&format!("RESP charges (e): {:?}", fmt_q(&rq)));
                out.chelpg = Some(cq);
                out.resp = Some(rq);
            }
            Err(e) => rep.warn(&format!("warning: CHELPG/RESP charges failed: {e}")),
        },
        (true, false) => out.chelpg = charge_scheme("CHELPG", chelpg_charges(mol, prep, d), rep),
        (false, true) => out.resp = charge_scheme("RESP", resp_charges(mol, prep, d), rep),
        (false, false) => {}
    }
}

/// The TS / MBD single-pole route to a per-atom dynamic α and its C6.
///
/// Any failure warns and SKIPS C6 (`None`): there is no zero-α, unit-volume or
/// unit-ratio fallback, because those exported wrong numbers that looked like
/// results.
fn ts_or_mbd_c6(
    inp: &PropertyInputs,
    c6_source: C6Source,
    partition: DispersionPartition,
    alpha_atomic: Option<&Vec<Tensor3>>,
    rep: &mut dyn PropertyReporter,
) -> Option<C6Result> {
    let (mol, prep, bs, dfbs, result, op, rpa_cfg) = (
        inp.mol,
        inp.prep,
        inp.bs,
        inp.dfbs,
        inp.result,
        inp.op,
        inp.rpa_cfg,
    );
    let alpha_res = if partition == DispersionPartition::Hirshfeld {
        pdep_polarizability_hirshfeld(mol, prep, bs, dfbs, result, op, rpa_cfg, Some(inp.proatom))
    } else {
        match alpha_atomic {
            Some(v) => Ok(v.clone()),
            None => pdep_polarizability_becke(mol, prep, bs, dfbs, result, op, rpa_cfg),
        }
    };
    let alpha_static: Vec<Tensor3> = match alpha_res {
        Ok(v) => v,
        Err(e) => {
            rep.warn(&format!(
                "warning: TS C6 skipped — per-atom static α failed: {e}"
            ));
            return None;
        }
    };
    // TS volumes must always use Hirshfeld partition — TS was parameterized
    // with Hirshfeld volumes (TS PRL 2009). Becke volumes blow up for
    // π-system H atoms (vol_ratio >> 1) because Becke is atom-size-blind;
    // Hirshfeld proatom weights correctly compress H relative to C. The
    // c6_partition setting only governs the alpha_static shape tensor, not
    // these volumes.
    let vols = match atomic_effective_volumes_hirshfeld(
        mol,
        bs,
        result.density_total(),
        Some(inp.proatom),
    ) {
        Ok(v) => v,
        Err(e) => {
            rep.warn(&format!(
                "warning: TS C6 skipped — Hirshfeld effective volumes failed: {e}"
            ));
            return None;
        }
    };
    let z: Vec<usize> = mol.atoms.iter().map(|a| a.z as usize).collect();
    let ratio = ts_volume_ratios(inp, &z, &vols, rep)?;
    let (freqs, weights) = build_quadrature(&rpa_cfg.quadrature);
    let is_mbd = c6_source == C6Source::Mbd;
    let dp_res = if is_mbd {
        let positions: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
        crate::dispersion::mbd_dynamic_polarizability(
            &positions,
            &z,
            &ratio,
            &alpha_static,
            &freqs,
            &weights,
        )
    } else {
        ts_dynamic_polarizability(&z, &ratio, &alpha_static, &freqs, &weights)
    };
    let dp = match dp_res {
        Ok(dp) => dp,
        Err(e) => {
            rep.warn(&format!(
                "warning: {} C6 skipped: {e}",
                if is_mbd { "MBD" } else { "TS" }
            ));
            return None;
        }
    };
    let ts_res = casimir_polder_c6(&dp);
    rep.info(&format!(
        "Computed {} C6: {} atoms; molecular C6 = {:.3} a.u.",
        if is_mbd { "MBD" } else { "TS" },
        z.len(),
        ts_res.c6_molecular_iso
    ));
    Some(ts_res)
}

/// Molecular / free-atom volume ratios for the TS model, or `None` (with a
/// warning) when no scale-consistent free-atom volume is available.
///
/// vol_free comes ONLY from a live free-atom SCF in the same basis and SCF
/// settings, then Hirshfeld on the isolated atom (weight 1 wherever the Slater
/// ρ⁰ is above the 1e-12 floor) on the SAME atom-centred Becke–Lebedev
/// quadrature the molecular volumes were integrated on, so the ratio is
/// scale-consistent. There is deliberately NO table fallback: the hardcoded
/// `ts_free_atom` vol_free values were on a mismatched integration scale and
/// (for Z outside {H,He,C,N,O,F,Ne}) were never sourced. A genuine live-SCF
/// failure SKIPS TS C6 with a warning instead of substituting a
/// scale-mismatched denominator.
fn ts_volume_ratios(
    inp: &PropertyInputs,
    z: &[usize],
    vols: &[f64],
    rep: &mut dyn PropertyReporter,
) -> Option<Vec<f64>> {
    let mut vol_free_computed: std::collections::HashMap<usize, f64> =
        std::collections::HashMap::new();
    for &zi in z.iter().collect::<std::collections::HashSet<_>>() {
        if let Ok(vf) =
            crate::dispersion::live_free_atom_volume(inp.ctx, zi, inp.bs, inp.op, inp.rhf_config)
        {
            vol_free_computed.insert(zi, vf);
        }
    }
    let mut ratio = Vec::with_capacity(z.len());
    for (i, &zi) in z.iter().enumerate() {
        let sym = ferric_core::elements::z_to_symbol(zi as i32).unwrap_or("?");
        let vf = match vol_free_computed.get(&zi).copied() {
            Some(v) => v,
            None => {
                rep.warn(&format!(
                    "warning: TS C6 skipped — live free-atom SCF failed for \
                     {sym} (Z={zi}) and no scale-consistent free-atom volume \
                     is available. The TS free-atom vol_free denominator MUST \
                     come from a live SCF on the same integration scale as the \
                     molecular volume; the old hardcoded-table fallback was \
                     removed because it is on a mismatched scale and was never \
                     sourced for most elements (project wiki: vol-free-verification.md). \
                     Refusing to fabricate a C6 from a mismatched denominator \
                     (same convention as the Z>18 hard-error path — see \
                     ts_atom_params). Use \
                     c6_source=\"pdep\" for a table-free dispersion source."
                ));
                return None;
            }
        };
        if vf <= 1e-10 {
            rep.warn(&format!(
                "warning: TS C6 skipped — degenerate free-atom volume \
                 {vf:.3e} for {sym} (Z={zi})"
            ));
            return None;
        }
        ratio.push(vols[i] / vf);
    }
    Some(ratio)
}

/// C6 and per-atom dynamic α from the requested source.
fn compute_c6_block(
    inp: &PropertyInputs,
    req: &PdepPropertyRequest,
    rep: &mut dyn PropertyReporter,
    out: &mut PdepProperties,
) -> Result<(), String> {
    // Strict parse: an unknown c6_source/c6_partition used to fall through to
    // TS/Becke silently, producing different numbers than the user asked for.
    let c6_source = C6Source::parse_config_str(req.c6_source.as_deref())?;
    let partition = DispersionPartition::parse_config_str(req.c6_partition.as_deref())?
        .unwrap_or_else(|| c6_source.default_partition());
    let use_pdep = c6_source == C6Source::Pdep;
    // The PDEP dynamic path runs the Hirshfeld partition for a closed-shell
    // reference only. Label (and warn) with what will actually run.
    let partition = if use_pdep {
        let used = crate::dispersion::effective_dynamic_partition(partition, inp.result.spin);
        if used != partition {
            rep.warn(&format!(
                "warning: c6_partition = \"{}\" is closed-shell only; this open-shell \
                 reference uses the \"{}\" partition for the per-atom PDEP-RPA C6",
                partition.as_config_str(),
                used.as_config_str()
            ));
        }
        used
    } else {
        partition
    };
    let res_opt = if use_pdep {
        // PDEP-RPA dynamic α(iω). Origin-independent for the molecular total
        // AND the per-atom intrinsic α^A (atom-centred (r−R_A); bond-axis
        // anisotropy is a coupled/molecular property, not per-atom). Uses the
        // shared same-basis Hirshfeld proatom so the per-atom partition is
        // basis-consistent.
        match pdep_dynamic_polarizability(
            inp.mol,
            inp.prep,
            inp.bs,
            inp.dfbs,
            inp.result,
            inp.op,
            inp.rpa_cfg,
            partition,
            Some(inp.proatom),
        ) {
            Ok(dp) => {
                let res = casimir_polder_c6(&dp);
                rep.info(&format!(
                    "Computed PDEP-RPA C6: {} atoms, {} freqs; molecular C6 = {:.3} a.u.",
                    inp.mol.atoms.len(),
                    dp.freqs.len(),
                    res.c6_molecular_iso
                ));
                Some(res)
            }
            Err(e) => {
                rep.warn(&format!("warning: PDEP-RPA C6 failed: {e}"));
                None
            }
        }
    } else {
        ts_or_mbd_c6(inp, c6_source, partition, out.alpha_atomic.as_ref(), rep)
    };
    let Some(res) = res_opt else {
        // Reaching this point with C6 requested means every path to it warned
        // and bailed; the specific reason is already on the reporter.
        out.gaps
            .push("c6_iso/c6_aniso/alpha_atomic_dynamic (see the C6 warning above)".to_string());
        return Ok(());
    };
    let mut alpha_ct_dynamic = None;
    if use_pdep {
        // molecular_dynamic_polarizability and the per-atom intrinsic α^A(iω)
        // share the RI kernel, orbitals and frozen_core = 0, at the same
        // frequencies.
        match crate::properties::charge_transfer_remainder_dynamic(
            &res.per_atom_dynamic.molecular,
            &res.per_atom_dynamic.per_atom,
        ) {
            Ok(ct) => alpha_ct_dynamic = Some(ct),
            Err(e) => {
                rep.warn(&format!(
                    "warning: dynamic charge-transfer α_CT(iω) failed: {e}"
                ));
                out.gaps.push(format!("alpha_ct_dynamic: {e}"));
            }
        }
    }
    // Tag with the partition/source this run ACTUALLY used (the strictly
    // parsed locals above, after the source-dependent default was applied).
    out.c6 = Some(C6Properties {
        partition: partition.as_config_str(),
        source: c6_source.as_config_str(),
        freqs: res.per_atom_dynamic.freqs.clone(),
        weights: res.per_atom_dynamic.weights.clone(),
        alpha_atomic_dynamic: res.per_atom_dynamic.per_atom.clone(),
        alpha_ct_dynamic,
        c6_iso: res.c6_iso_pair.clone(),
        c6_aniso: res.c6_aniso_pair.clone(),
        c6_molecular_iso: res.c6_molecular_iso,
    });
    Ok(())
}

/// Second moments of the orbitals and of the density: one-electron cost,
/// computed whenever the pieces are already in hand.
fn compute_moments(inp: &PropertyInputs, req: &PdepPropertyRequest, out: &mut PdepProperties) {
    if inp.result.spin == Spin::Restricted {
        if let Ok((c, s)) =
            ferric_integrals::oneelectron::orbital_moments(inp.prep, inp.result.mos_r())
        {
            out.orbital_centers = Some(c);
            out.orbital_spreads = Some(s);
        }
    }
    if req.density_matrix {
        out.density_second_moment = ferric_integrals::oneelectron::density_second_moment(
            inp.prep,
            inp.result.density_total(),
            [0.0; 3],
        )
        .ok()
        .map(|m| Array2::from_shape_fn((3, 3), |(p, q)| m[p][q]));
    }
}

/// Compute the property bundle for a PDEP-RPA reference.
///
/// `Err` only for an unparseable `c6_source` / `c6_partition` (reported after
/// the properties that precede C6 have been computed, as the CLI always did);
/// per-property failures land in [`PdepProperties::gaps`].
pub fn compute_pdep_properties(
    inp: &PropertyInputs,
    req: &PdepPropertyRequest,
    rep: &mut dyn PropertyReporter,
) -> Result<PdepProperties, String> {
    let mol = inp.mol;
    let mut out = PdepProperties {
        coords: {
            let mut a = Array2::<f64>::zeros((mol.atoms.len(), 3));
            for (i, atom) in mol.atoms.iter().enumerate() {
                a[(i, 0)] = atom.x;
                a[(i, 1)] = atom.y;
                a[(i, 2)] = atom.zpos;
            }
            a
        },
        atomic_numbers: mol.atoms.iter().map(|a| a.z as usize).collect(),
        density_matrix: req.density_matrix,
        ..Default::default()
    };
    compute_site_properties(inp, req, rep, &mut out);
    compute_polarizabilities(inp, req, rep, &mut out);
    compute_charges_and_dipole(inp, req, rep, &mut out);
    if req.c6 {
        compute_c6_block(inp, req, rep, &mut out)?;
    }
    compute_moments(inp, req, &mut out);
    Ok(out)
}

/// Write the NPZ feature bundle: the property arrays plus the reference's MOs
/// and density, the PDEP eigenpotentials and the result certificate.
pub fn write_pdep_npz(
    path: &str,
    inp: &PropertyInputs,
    props: &PdepProperties,
    rpa_result: &PdepRpaResult,
    certificate_json: &str,
) -> Result<(), ferric_export::cube::ExportError> {
    let result = inp.result;
    let restricted = result.spin == Spin::Restricted;
    let dm_ref = props.density_matrix.then(|| result.density_total());
    let bundle = NpzBundle {
        mo_coeffs: restricted.then(|| result.mos_r()),
        orbital_energies: restricted.then(|| result.eps_r()),
        pdep_eigenvectors: Some(&rpa_result.eigenpotentials),
        boys_coeffs: None,
        orbital_centers: props.orbital_centers.as_ref(),
        orbital_spreads: props.orbital_spreads.as_deref(),
        density_second_moment: props.density_second_moment.as_ref(),
        coords: Some(&props.coords),
        atomic_numbers: Some(&props.atomic_numbers),
        density_matrix: dm_ref,
        dipole: props.dipole.as_ref(),
        charges: ChargeSchemes {
            hirshfeld: props.hirshfeld.as_deref(),
            lowdin: props.lowdin.as_deref(),
            mulliken: props.mulliken.as_deref(),
            chelpg: props.chelpg.as_deref(),
            resp: props.resp.as_deref(),
        },
        polarizability: PolarizabilityBundle {
            esp_atoms: props.esp_atoms.as_deref(),
            esp_surface: props.esp_surface.as_deref(),
            esp_points: props.esp_points.as_ref(),
            alpha_tensor: props.alpha_tensor.as_ref(),
            electric_field: props.electric_field.as_deref(),
            alpha_atomic: props.alpha_atomic.as_deref(),
            alpha_ct: props.alpha_ct.as_ref(),
        },
        dispersion: DispersionBundle {
            // All-or-nothing: the provenance is non-Option inside `C6Export`,
            // so the arrays are never written without their partition/source.
            c6: props.c6.as_ref().map(|c| C6Export {
                provenance: C6Provenance {
                    partition: c.partition,
                    source: c.source,
                },
                c6_freqs: c.freqs.as_slice(),
                c6_weights: c.weights.as_slice(),
                alpha_atomic_dynamic: c.alpha_atomic_dynamic.as_slice(),
                alpha_ct_dynamic: c.alpha_ct_dynamic.as_deref(),
                c6_iso: &c.c6_iso,
                c6_aniso: c.c6_aniso.as_slice(),
                c6_molecular_iso: c.c6_molecular_iso,
            }),
        },
        certificate: Some(certificate_json),
    };
    ferric_export::export_npz(path, &bundle)
}

/// Cube-export settings for the eigenpotentials.
#[derive(Debug, Clone, Copy)]
pub struct EigpotCubeSpec {
    /// Grid spacing, Bohr.
    pub spacing: f64,
    /// Margin beyond the bounding box, Bohr.
    pub margin: f64,
    /// Number of leading eigenpotentials to write (capped at the number kept).
    pub count: usize,
}

impl Default for EigpotCubeSpec {
    fn default() -> Self {
        Self {
            spacing: 0.2,
            margin: 4.0,
            count: 10,
        }
    }
}

/// Write the leading PDEP eigenpotentials' charge densities as cube files
/// `{prefix}_eigpot_{NNN}.cube`. Returns the paths written; a file that cannot
/// be written is a warning.
pub fn export_eigenpotential_cubes(
    prefix: &str,
    spec: &EigpotCubeSpec,
    mol: &Molecule,
    aux_name: &str,
    aux_bs: &BasisSet,
    rpa_result: &PdepRpaResult,
    rep: &mut dyn PropertyReporter,
) -> Vec<String> {
    use ferric_export::cube::GridSpec;
    use ferric_export::export_basis_function_cube;
    let n_export = spec.count.min(rpa_result.n_eigenpotentials);
    let grid = GridSpec::bounding_box(mol, spec.margin, spec.spacing);
    rep.info(&format!(
        "Exporting {} eigenpotential cubes (grid {}×{}×{}, spacing {} Bohr)…",
        n_export, grid.n_x, grid.n_y, grid.n_z, spec.spacing
    ));
    let mut written = Vec::new();
    for alpha in 0..n_export {
        let coeffs: Vec<f64> = rpa_result
            .eigenpotentials
            .column(alpha)
            .iter()
            .copied()
            .collect();
        let lam = rpa_result.eigenvalues_static[alpha];
        let path = format!("{prefix}_eigpot_{:03}.cube", alpha);
        let comment = format!(
            "PDEP eigenpotential α={alpha} λ(0)={lam:.6}: its charge density (basis {aux_name})"
        );
        if let Err(e) = export_basis_function_cube(&path, mol, aux_bs, &grid, &coeffs, &comment) {
            rep.warn(&format!("  warning: failed to write {}: {}", path, e));
        } else {
            rep.info(&format!("  wrote {} (λ(0)={:.6})", path, lam));
            written.push(path);
        }
    }
    written
}
