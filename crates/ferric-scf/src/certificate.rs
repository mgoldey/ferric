//! The [`Certificate`](ferric_core::certificate::Certificate) of an SCF / KS-DFT result: capability and grade from
//! `ferric_core::certificate::METHOD_KIND_GRADES`, the measured error of the
//! solver's energy anchor (`ferric_core::certificate::anchors`), the
//! approximations the solve ran with, and warnings for conditions the solver already
//! detected (non-convergence, plateau exits, stability verdicts).
//!
//! Built from the result, the config it was solved with and the molecule,
//! after the solve; nothing here changes an energy. The build stamp is left
//! `None`: only `ferric-cli` and `ferric-python` link `ferric-build-info`
//! and stamp it (see the `ferric_core::certificate` module doc).

use ferric_core::certificate::{energy_anchor, Certificate, EnergyAnchor, Grade, RunEnvelope};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;

use crate::frequencies::{FrequencyConfig, FrequencyResult, HessianSource};
use crate::optimize::OptimizeConfig;
use crate::result::{ScfExit, ScfResult, Spin};
use crate::rhf::{solve_rhf, RhfConfig};
use crate::rohf::solve_rohf;
use crate::screening::SchwarzBounds;
use crate::stability::{StabilityResult, StabilityVerdict};
use crate::uhf::solve_uhf;

/// The CLI `method.kind` a result is graded under, as `Config::dispatch_kind`
/// routes it: RHF and RKS (`xc` set) are `rhf` / `ksdft`; UHF and UKS run
/// under `uhf`; ROHF and ROKS under `rohf`.
pub fn scf_capability(result: &ScfResult, config: &RhfConfig) -> &'static str {
    match (result.spin, config.xc.is_some()) {
        (Spin::Restricted, false) => "rhf",
        (Spin::Restricted, true) => "ksdft",
        (Spin::Unrestricted, _) => "uhf",
        (Spin::RestrictedOpen, _) => "rohf",
    }
}

/// The solver label of `result`: `RHF`, `RKS`, `UHF`, `UKS`, `ROHF` or
/// `ROKS` (the [`EnergyAnchor::method`] it is anchored under).
pub fn scf_method(result: &ScfResult, config: &RhfConfig) -> &'static str {
    match (result.spin, config.xc.is_some()) {
        (Spin::Restricted, false) => "RHF",
        (Spin::Restricted, true) => "RKS",
        (Spin::Unrestricted, false) => "UHF",
        (Spin::Unrestricted, true) => "UKS",
        (Spin::RestrictedOpen, false) => "ROHF",
        (Spin::RestrictedOpen, true) => "ROKS",
    }
}

/// The headline energy anchor of `result`'s solver.
pub fn scf_energy_anchor(result: &ScfResult, config: &RhfConfig) -> Option<&'static EnergyAnchor> {
    energy_anchor(scf_method(result, config))
}

/// The certificate of `result`, solved for `mol` with `config` (for a
/// laddered solve, the config of the rung that produced `result`).
///
/// `measured_error` is the headline energy anchor of the solver
/// ([`scf_energy_anchor`]). A result with `converged = false` is capped to
/// [`Grade::Ungraded`]: the validation rows grade converged energies only.
pub fn scf_certificate(mol: &Molecule, result: &ScfResult, config: &RhfConfig) -> Certificate {
    let mut c = Certificate::for_method_kind(scf_capability(result, config))
        .with_precision(&result.precision);
    if let Some(a) = scf_energy_anchor(result, config) {
        c = c.with_energy_anchor(a);
    }
    for (name, detail) in approximations(mol, result, config) {
        c = c.with_approximation(name, detail);
    }
    if result.exit == ScfExit::Plateau {
        c.push_warning(
            "SCF exit Plateau: accepted on a near-degeneracy plateau (orbital gradient \
             stalled below the 1e-4 floor), not the standard convergence test",
        );
    }
    stability_warnings(&mut c, result);
    if !result.converged {
        let why = match result.exit {
            ScfExit::NotCertified => "SCF converged but the ROHF occupation witness found a \
                                      lower one-electron move (exit NotCertified)"
                .to_string(),
            e => format!(
                "SCF did not converge (exit {e:?}, {} iterations)",
                result.iterations
            ),
        };
        c = c.cap_grade(Grade::Ungraded, why);
    }
    c
}

/// `c` for a solve with a caller-supplied Fock hook
/// (`rhf::solve_rhf_with_hook` and its open-shell counterparts): the hook's
/// energy term and potential are part of the Hamiltonian, so they are
/// recorded as an approximation. The anchor and grade are kept.
pub fn with_fock_hook(c: Certificate) -> Certificate {
    c.with_approximation(
        "fock-hook",
        "caller-supplied energy term and Fock potential added at every SCF iteration",
    )
}

/// `c` with a warning for each way the run lies outside the solver's energy
/// anchor ([`Certificate::with_envelope`]): an element of `mol`, the orbital
/// basis `basis`, the functional. The grade is not changed.
pub fn with_scf_envelope(
    c: Certificate,
    mol: &Molecule,
    result: &ScfResult,
    config: &RhfConfig,
    basis: &ferric_core::basis::BasisSet,
) -> Certificate {
    let Some(a) = scf_energy_anchor(result, config) else {
        return c;
    };
    let zs: Vec<i32> = mol.atoms.iter().map(|at| at.z).collect();
    c.with_envelope(
        a,
        &RunEnvelope {
            atomic_numbers: &zs,
            basis: Some(basis),
            aux_basis: None,
            functional: config.xc.as_deref(),
        },
    )
}

/// Whether exact (HF-type) exchange enters the energy, and whether it is
/// range-separated. HF: `(true, false)`. Unresolvable names count as HF-like,
/// as `ferric-cli`'s `log_jk_path` does.
fn exchange_used(config: &RhfConfig) -> (bool, bool) {
    let k_mix = config
        .xc
        .as_deref()
        .and_then(|name| ferric_dft::libxc::xc_def_from_name(name).ok())
        .map(|d| ferric_dft::libxc::k_mix_from_xc_def(&d));
    match k_mix {
        None => (true, false),
        Some(m) => (m.sr > 0.0 || m.omega > 0.0, m.omega > 0.0),
    }
}

/// An integration grid as certificates write it:
/// `(n_radial, n_angular), <prune>, <weight cut>`.
pub fn grid_text(g: &ferric_dft::grid::AtomicGridConfig) -> String {
    let prune = g
        .prune
        .map_or("flat".to_string(), |p| format!("{p:?}").to_lowercase());
    let cut = g
        .weight_cut
        .map_or("no weight cut".to_string(), |w| format!("weight cut {w:e}"));
    format!("({}, {}), {prune}, {cut}", g.n_radial, g.n_angular)
}

/// `(name, detail)` of every approximation in effect, in a fixed order.
fn approximations(
    mol: &Molecule,
    result: &ScfResult,
    config: &RhfConfig,
) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    // Density fitting: what the solver recorded it ran, not what was asked.
    let route = result.df_jk.as_ref();
    if let Some(r) = route.filter(|r| r.j_aux.is_some()) {
        let a = r.j_aux.as_deref().unwrap_or_default();
        // What ran, then what was requested and why (`explicit`, or the
        // default for the config's task: f64 for geometry/frequency tasks).
        let req = config.resolved_ri_j_precision(crate::df_j::ri_j_site_for_functional(
            result.mos_alpha.nrows(),
            result.xc.as_deref(),
        ));
        out.push((
            "ri-j",
            format!(
                "aux {a}, {} tensor storage (requested {}: {})",
                r.ri_j_precision,
                req.precision,
                req.source.as_str()
            ),
        ));
    }
    // As for RI-J: the stored precision of the dressed tensor(s).
    let k_storage = route
        .map(|r| format!(", {} tensor storage", r.ri_k_precision))
        .unwrap_or_default();
    if let Some(a) = route.and_then(|r| r.k_aux.as_deref()) {
        out.push(("ri-k", format!("aux {a}{k_storage}")));
    }
    if let Some((a, w)) = route.and_then(|r| r.rsh_k.as_ref()) {
        out.push((
            "ri-k",
            format!("range-separated SR/LR pair, aux {a}, omega {w}{k_storage}"),
        ));
    }
    let (k_used, rsh) = exchange_used(config);
    let k_fitted = route.is_some_and(|r| r.k_aux.is_some() || r.rsh_k.is_some());
    if k_used && !rsh && !k_fitted {
        match config.k_builder.as_deref() {
            Some("link") => out.push(("link", "LinK screened exchange".to_string())),
            Some("cosx") => {
                let fin = result.cosx_final.map_or_else(
                    || "no final pass".to_string(),
                    |f| format!("final pass on {} points reported", f.npts_final),
                );
                out.push((
                    "cosx",
                    format!(
                        "seminumerical exchange, grid {}, {fin}",
                        grid_text(&config.cosx.grid)
                    ),
                ));
            }
            _ => {}
        }
    }
    if let Some(xc) = config.xc.as_deref() {
        let grids = crate::ks_gradient::KsGradGrids::from_config(config);
        out.push(("xc-grid", format!("{xc} on {}", grid_text(&grids.main))));
        let vv10 = ferric_dft::libxc::xc_def_from_name(xc).is_ok_and(|d| d.vv10.is_some());
        if vv10 {
            out.push(("nlc-grid", grid_text(&grids.nlc)));
        }
    }
    let shift = crate::driver::effective_level_shift(config);
    if shift > 0.0 {
        let who = if config.level_shift == 0.0 {
            "auto-applied for meta-GGA"
        } else {
            "user-set"
        };
        out.push((
            "level-shift",
            format!("{shift} Ha virtual-block shift, {who} (ramped to 0 at convergence)"),
        ));
    }
    if let Some(s) = config.smearing_sigma {
        out.push(("smearing", format!("sigma {s} Ha, fractional occupations")));
    }
    let n_ecp = mol.atoms.iter().filter(|a| a.n_core_ecp > 0).count();
    if n_ecp > 0 {
        let core: i32 = mol.atoms.iter().map(|a| a.n_core_ecp.max(0)).sum();
        out.push((
            "ecp",
            format!("{n_ecp} atom(s) with an ECP, {core} core electrons replaced"),
        ));
    }
    if let Some(ext) = config.external_potential.as_ref() {
        let field = ext
            .field
            .map_or(String::new(), |f| format!(", field {f:?} a.u."));
        out.push((
            "external-potential",
            format!(
                "{} point, {} smeared charge(s){field}",
                ext.point_charges.len(),
                ext.smeared_charges.len()
            ),
        ));
    }
    if let Some(p) = config.polarizable.as_ref() {
        out.push((
            "polarizable-embedding",
            format!("{} Thole site(s)", p.sites.len()),
        ));
    }
    if let Some(cm) = config.cosmo.as_ref() {
        out.push(("cosmo", format!("epsilon {}", cm.epsilon)));
    }
    if let Some(p) = config.pcm.as_ref() {
        out.push(("pcm", format!("IEF-PCM, epsilon {}", p.epsilon)));
    }
    out
}

/// Warnings from the stability verdicts on the result. Open-shell
/// unrestricted results that were not checked get the saddle caveat; a
/// checked result warns unless the verdict is `Stable`.
fn stability_warnings(c: &mut Certificate, result: &ScfResult) {
    let mut checked = |r: &StabilityResult| {
        if r.verdict() != StabilityVerdict::Stable {
            c.push_warning(format!("stability {}", r.summary()));
        }
    };
    if let Some(r) = result.stability.as_ref() {
        checked(r);
    }
    if let Some(r) = result.stability_external.as_ref() {
        checked(r);
    }
    if result.spin == Spin::Unrestricted && result.stability.is_none() {
        c.push_warning(
            "internal stability not checked (check_stability = false): a converged UHF/UKS \
             can be a saddle point of the orbital Hessian",
        );
    }
}

/// Solve the SCF of `spin` at `mol`, for the certificate of a geometry
/// optimization or frequency run whose drivers do not return their final
/// [`ScfResult`]. One SCF, no gradient; the config is the one the geometry
/// drivers use ([`crate::gradient::gradient_task_config`]). Returns the result
/// with the config it was solved with, the pair [`scf_certificate`] takes.
pub fn final_point_scf(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    config: &RhfConfig,
    spin: Spin,
) -> Result<(ScfResult, RhfConfig), FerricError> {
    let bs = ferric_core::basis::bundled(basis_name)?;
    let prep = PreparedBasis::new(mol, &bs)?;
    let bounds = SchwarzBounds::compute_for_screening(op, &prep, config.screening)?;
    let cfg = crate::gradient::gradient_task_config(config).into_owned();
    let r = match spin {
        Spin::Restricted => solve_rhf(ctx, mol, &prep, op, &bounds, &cfg)?,
        Spin::Unrestricted => solve_uhf(ctx, mol, &prep, &bounds, &cfg)?,
        Spin::RestrictedOpen => solve_rohf(ctx, mol, &prep, op, &bounds, &cfg)?,
    };
    Ok((r, cfg))
}

/// Certificate of a geometry optimization (`task = "optimize"`) of
/// `method.kind = kind`.
///
/// `final_point` is the certificate of the SCF at the returned geometry (its
/// grade, approximations and warnings are inherited; see
/// [`Certificate::for_task`]). `gradient` names how the gradient was formed
/// (`"analytic"`, `"analytic SCF + finite-difference correlation"`). A run
/// that did not converge is capped to [`Grade::Ungraded`].
pub fn optimize_certificate(
    kind: &str,
    final_point: &Certificate,
    converged: bool,
    steps: usize,
    energy: f64,
    config: &OptimizeConfig,
    gradient: &str,
) -> Certificate {
    let mut c = Certificate::for_task("optimize", kind, final_point)
        .with_approximation("gradient", gradient)
        .with_approximation(
            "optimizer",
            format!(
                "BFGS, {:?} coordinates, at most {} steps; thresholds g_max {:e}, \
                 g_rms {:e} Ha/Bohr, e_conv {:e} Ha",
                config.coord_system,
                config.max_steps,
                config.g_max_thresh,
                config.g_rms_thresh,
                config.e_conv
            ),
        );
    if !converged {
        c = c.cap_grade(
            Grade::Ungraded,
            format!("the optimizer did not converge ({steps} steps, final energy {energy:.10} Ha)"),
        );
    }
    c
}

/// Certificate of a harmonic frequency run (`task = "frequencies"`) of
/// `method.kind = kind`, with `reference_point` the certificate of the SCF at
/// the input geometry. Records which Hessian construction ran; an imaginary
/// frequency is a warning (the geometry is not a minimum of this surface).
pub fn frequencies_certificate(
    kind: &str,
    reference_point: &Certificate,
    result: &FrequencyResult,
    config: &FrequencyConfig,
) -> Certificate {
    let hessian = match result.hessian_source {
        HessianSource::FiniteDifference => format!(
            "finite-difference, central differences of the analytic gradient, delta {} Bohr, \
             {} gradient evaluation(s)",
            config.delta, result.n_gradient_evaluations
        ),
        s => s.label().to_string(),
    };
    let mut c = Certificate::for_task("frequencies", kind, reference_point)
        .with_approximation("hessian", hessian)
        .with_approximation(
            "hessian-asymmetry",
            format!("{:.3e} Ha/Bohr^2 before symmetrization", result.asymmetry),
        );
    let n_imag = result.frequencies.iter().filter(|w| **w < 0.0).count();
    if n_imag > 0 {
        c.push_warning(format!(
            "{n_imag} imaginary frequenc{}: the geometry is not a minimum of this surface",
            if n_imag == 1 { "y" } else { "ies" }
        ));
    }
    c
}

/// What the QM/MM wrapper adds on top of the SCF of the QM region: the
/// partition, the embedding (point, smeared and polarizable MM sites, the
/// boundary scheme and link atoms) and, when one was passed, the MM force
/// field. Recorded on a certificate of capability `capability`, which is
/// [`Grade::Ungraded`]: `site/src/reference/validation.md` lists the QM/MM
/// entry points under "Python entry points", which are not graded
/// individually, so [`METHOD_KIND_GRADES`] has no row for them. The
/// reference's approximations (including its `external-potential`),
/// warnings and build stamp are folded in by [`Certificate::with_reference`].
fn qmmm_base(
    capability: String,
    reference: &Certificate,
    sys: &crate::qmmm::QmmmSystem,
    mm_topology: Option<usize>,
) -> Certificate {
    let (n_point, n_smeared) = sys
        .to_external_potential()
        .map_or((0, 0), |e| (e.point_charges.len(), e.smeared_charges.len()));
    let n_pol = sys.to_polarizable_sites().len();
    let mut c = Certificate::ungraded(capability)
        .with_reference(reference)
        .with_approximation(
            "qmmm-partition",
            format!(
                "{} QM atom(s), {} MM atom(s), QM charge {}, multiplicity {}",
                sys.qm_indices.len(),
                sys.mm_indices.len(),
                sys.qm_charge,
                sys.qm_multiplicity
            ),
        )
        .with_approximation(
            "qmmm-embedding",
            format!(
                "electrostatic embedding: {n_point} point and {n_smeared} Gaussian-smeared MM \
                 charge(s), {n_pol} Thole polarizable site(s), boundary scheme {:?} \
                 ({} off-atom boundary charge(s)), {} link atom(s)",
                sys.boundary_scheme,
                sys.boundary_charges.len(),
                sys.link_atoms.len()
            ),
        );
    if let Some(n) = mm_topology {
        c = c.with_approximation(
            "mm-force-field",
            format!("AMBER-form bonded and Lennard-Jones terms over {n} atom(s)"),
        );
    }
    c
}

/// Certificate of a `run_qmmm` result: `reference` is the certificate of the
/// SCF it solved (`method` is `"rhf"`, `"uhf"`, `"rks"` or `"uks"`);
/// `mm_topology` is the atom count of the MM force field, when one was
/// passed. Capability `qmmm:<method>`, [`Grade::Ungraded`]; an unconverged
/// SCF is a warning.
pub fn qmmm_certificate(
    method: &str,
    reference: &Certificate,
    sys: &crate::qmmm::QmmmSystem,
    mm_topology: Option<usize>,
    converged: bool,
) -> Certificate {
    let mut c = qmmm_base(format!("qmmm:{method}"), reference, sys, mm_topology);
    if !converged {
        c.push_warning("the embedded SCF did not converge");
    }
    c
}

/// Certificate of a `run_optimize_qmmm` result: `reference` is the
/// certificate of the SCF at the returned partition `sys`. Capability
/// `qmmm-optimize:<method>`, [`Grade::Ungraded`]; records which MM atoms
/// moved and the optimizer settings; a run that did not converge is a
/// warning.
pub fn qmmm_optimize_certificate(
    method: &str,
    reference: &Certificate,
    sys: &crate::qmmm::QmmmSystem,
    cfg: &crate::qmmm::QmmmOptimizeConfig,
    converged: bool,
    steps: usize,
    energy: f64,
) -> Certificate {
    let n_top = cfg.mm_topology.as_ref().map(|t| t.n_atoms());
    let mut c = qmmm_base(format!("qmmm-optimize:{method}"), reference, sys, n_top)
        .with_approximation("qmmm-move-mm", format!("{:?}", cfg.move_mm))
        .with_approximation(
            "optimizer",
            format!(
                "BFGS, at most {} steps; thresholds g_max {:e}, g_rms {:e} Ha/Bohr, \
                 e_conv {:e} Ha",
                cfg.opt.max_steps, cfg.opt.g_max_thresh, cfg.opt.g_rms_thresh, cfg.opt.e_conv
            ),
        );
    if !converged {
        c.push_warning(format!(
            "the optimizer did not converge ({steps} steps, final energy {energy:.10} Ha)"
        ));
    }
    c
}

/// What a `run_cdft` solve reports, for [`cdft_certificate`].
pub struct CdftRun<'a> {
    /// The constraint multipliers the outer loop ended on, in constraint order.
    pub lambdas: &'a [f64],
    /// Fragment populations at the returned density.
    pub populations: &'a [f64],
    /// Outer (lambda) iterations used.
    pub outer_iterations: usize,
    /// The inner SCF at the final lambda converged.
    pub scf_converged: bool,
    /// Largest `|N_C - target_C|`.
    pub max_constraint_error: f64,
}

/// Certificate of a constrained-DFT state (`run_cdft`): `reference` is the
/// certificate of the UHF/UKS solve at the constrained density. Capability
/// `cdft:uhf` or `cdft:uks`, [`Grade::Ungraded`]
/// (`site/src/reference/validation.md` says "Not graded" for it). Records
/// each constraint with its multiplier and population, the weight-operator
/// grid and the outer-loop settings.
pub fn cdft_certificate(
    reference: &Certificate,
    config: &RhfConfig,
    run: &CdftRun<'_>,
) -> Certificate {
    let ks = config.xc.is_some();
    let family = if ks { "uks" } else { "uhf" };
    // The constrained stability descent runs on an HF reference only. When it
    // ran it is this solve's stability check, so the SCF's "not checked"
    // caveat does not apply to the returned state.
    let descent = config.cdft_stability_descent && !ks;
    let mut reference = reference.clone();
    if descent {
        reference
            .warnings
            .retain(|w| !w.starts_with("internal stability not checked"));
    }
    let mut c = Certificate::ungraded(format!("cdft:{family}")).with_reference(&reference);
    for (i, k) in config.constraints.iter().enumerate() {
        let get = |v: &[f64]| v.get(i).map_or("n/a".to_string(), |x| format!("{x:.6e}"));
        c = c.with_approximation(
            "cdft-constraint",
            format!(
                "{:?} population of atoms {:?}, target {}, lambda {}, population {}",
                k.spin,
                k.fragment,
                k.target,
                get(run.lambdas),
                get(run.populations)
            ),
        );
    }
    let w_grid = config
        .dft_grid
        .clone()
        .unwrap_or(ferric_dft::grid::AtomicGridConfig {
            n_radial: 99,
            n_angular: 302,
            ..Default::default()
        });
    c = c
        .with_approximation(
            "cdft-weight-grid",
            format!("Becke weight operator on {}", grid_text(&w_grid)),
        )
        .with_approximation(
            "cdft-outer-loop",
            format!(
                "lambda Newton, tolerance {:e} e, at most {} iterations ({} used), \
                 stability descent {}; max constraint error {:.3e} e",
                config.cdft_lambda_tol,
                config.cdft_max_outer,
                run.outer_iterations,
                if descent {
                    "on"
                } else if ks {
                    "skipped on a KS reference"
                } else {
                    "off"
                },
                run.max_constraint_error
            ),
        );
    if ks && config.cdft_stability_descent {
        c.push_warning(
            "the constrained stability descent is skipped on a KS reference: the returned \
             state is a constrained stationary point, not necessarily the lowest state",
        );
    }
    if !run.scf_converged {
        c.push_warning("the inner SCF at the final lambda did not converge");
    }
    if run.max_constraint_error >= config.cdft_lambda_tol {
        c.push_warning(format!(
            "constraint error {:.3e} e is not below lambda_tol {:e}",
            run.max_constraint_error, config.cdft_lambda_tol
        ));
    }
    c
}

/// Certificate of a cDFT coupling (`cdft_coupling`) between the diabatic
/// states certified by `a` and `b`. Capability `cdft-coupling`,
/// [`Grade::Ungraded`]. State A's certificate is folded in as the reference;
/// state B's approximations and warnings are kept with the prefix `state B`.
pub fn cdft_coupling_certificate(
    a: &Certificate,
    b: &Certificate,
    h_ab: f64,
    s_ab: f64,
) -> Certificate {
    let mut c = Certificate::ungraded("cdft-coupling")
        .with_reference(a)
        .with_approximation(
            "coupling-formula",
            format!(
                "Wu-Van Voorhis orthogonalized coupling between two charge-constrained \
                 diabats sharing one Hamiltonian; h_ab {h_ab:.6e} Ha, s_ab {s_ab:.6e}"
            ),
        );
    for ap in &b.approximations {
        c = c.with_approximation(format!("state B {}", ap.name), ap.detail.clone());
    }
    for w in &b.warnings {
        c.push_warning(format!("state B: {w}"));
    }
    c
}

/// What a `run_saddle` search reports, for [`saddle_certificate`].
pub struct SaddleRun<'a> {
    /// P-RFO steps taken.
    pub steps: usize,
    /// The gradient converged.
    pub converged: bool,
    /// Negative eigenvalues of the projected Hessian at the final geometry.
    pub n_imaginary: usize,
    /// Which construction produced the final Hessian.
    pub hessian_source: &'a str,
    /// The finite-difference displacement (Bohr), when the Hessian is finite-difference.
    pub delta: Option<f64>,
}

/// Certificate of a P-RFO transition-state search: `reference` is the
/// certificate of the SCF at the returned geometry. Capability
/// `saddle:<kind>`, [`Grade::Ungraded`] (the page lists `run_saddle` under
/// "Python entry points", not graded individually). Records the search
/// settings and the Hessian construction; a result that is not a first-order
/// saddle, or whose gradient did not converge, is a warning.
pub fn saddle_certificate(
    kind: &str,
    reference: &Certificate,
    cfg: &crate::saddle::SaddleConfig,
    run: &SaddleRun<'_>,
) -> Certificate {
    let hessian = match run.delta {
        Some(d) if run.hessian_source == "finite-difference" => {
            format!("finite-difference, delta {d} Bohr")
        }
        _ => run.hessian_source.to_string(),
    };
    let mut c = Certificate::ungraded(format!("saddle:{kind}"))
        .with_reference(reference)
        .with_approximation(
            "saddle-search",
            format!(
                "P-RFO following mode {}, trust radius {} Bohr, at most {} steps ({} used); \
                 thresholds g_max {:e}, g_rms {:e} Ha/Bohr; Hessian recomputed every {} step(s) \
                 (0 = once, then Bofill updates)",
                cfg.follow_mode,
                cfg.trust_radius,
                cfg.max_steps,
                run.steps,
                cfg.g_max_thresh,
                cfg.g_rms_thresh,
                cfg.hessian_recalc_every
            ),
        )
        .with_approximation("hessian", hessian);
    if !run.converged {
        c.push_warning(
            "the gradient did not converge: the search stopped before a stationary point",
        );
    }
    if run.n_imaginary != 1 {
        c.push_warning(format!(
            "{} imaginary mode(s) at the final geometry: not a first-order saddle point",
            run.n_imaginary
        ));
    }
    c
}

/// Solve the SCF of the QM region of `sys` for the certificate of a QM/MM
/// optimization, whose driver does not return its final [`ScfResult`]. One
/// SCF, no gradient, set up as the optimizer sets up each step: the
/// geometry-task config, the embedding potential of `sys`, and for the
/// Kohn-Sham methods the functional with the `def2-universal-jkfit` RI-J/RI-K
/// auxiliary basis. Returns the QM molecule, the basis, the result and the
/// config it was solved with.
pub fn qmmm_final_point_scf(
    ctx: &ParallelContext,
    sys: &crate::qmmm::QmmmSystem,
    basis_name: &str,
    cfg: &crate::qmmm::QmmmOptimizeConfig,
) -> Result<(Molecule, ferric_core::basis::BasisSet, ScfResult, RhfConfig), FerricError> {
    use crate::qmmm::QmmmMethod;
    let bs = ferric_core::basis::bundled(basis_name)?;
    let mol = sys.to_qm_molecule();
    let prep = PreparedBasis::new(&mol, &bs)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep)?;
    let mut scf = crate::gradient::gradient_task_config(&cfg.scf).into_owned();
    scf.external_potential = sys.to_external_potential();
    if let QmmmMethod::Rks(xc) | QmmmMethod::Uks(xc) = &cfg.method {
        scf.xc = Some(xc.clone());
        scf.df_j_aux = Some("def2-universal-jkfit".to_string());
        scf.df_k_aux = Some("def2-universal-jkfit".to_string());
    }
    let r = match cfg.method {
        QmmmMethod::Rhf | QmmmMethod::Rks(_) => solve_rhf(ctx, &mol, &prep, op, &bounds, &scf)?,
        QmmmMethod::Uhf | QmmmMethod::Uks(_) => solve_uhf(ctx, &mol, &prep, &bounds, &scf)?,
    };
    Ok((mol, bs, r, scf))
}

#[cfg(test)]
mod result_class_tests {
    //! The wrapper certificates record what the wrapper adds, and stay
    //! `Ungraded` because `site/src/reference/validation.md` has no
    //! `method.kind` row for them.
    use super::*;
    use crate::qmmm::{QmSelection, QmmmAtom, QmmmSystem};
    use ferric_core::certificate::Grade;

    fn names(c: &Certificate) -> Vec<&str> {
        c.approximations.iter().map(|a| a.name.as_str()).collect()
    }

    fn reference() -> Certificate {
        Certificate::for_method_kind("rhf").with_approximation("ri-j", "aux x")
    }

    #[test]
    fn qmmm_certificate_records_partition_and_embedding() {
        let atoms = vec![
            QmmmAtom::new("O", 8, 0.0, 0.0, 0.0, 0.0),
            QmmmAtom::new("H", 1, 0.0, 0.0, 1.8, 0.0),
            QmmmAtom::new("O", 8, 0.0, 0.0, 9.0, -0.8),
        ];
        let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1]), 0, 1).unwrap();
        let c = qmmm_certificate("rhf", &reference(), &sys, None, true);
        assert_eq!(c.capability, "qmmm:rhf");
        assert_eq!(c.grade, Grade::Ungraded);
        assert!(c.grade_source.is_none());
        let emb = c.approximations.iter().find(|a| a.name == "qmmm-embedding");
        assert!(emb.is_some(), "no embedding record: {:?}", names(&c));
        let emb = emb.unwrap();
        assert!(
            emb.detail.contains("1 point and 0 Gaussian-smeared"),
            "{}",
            emb.detail
        );
        assert!(
            emb.detail.contains("boundary scheme Keep"),
            "{}",
            emb.detail
        );
        assert!(names(&c).contains(&"scf ri-j"));
        assert!(!names(&c).contains(&"mm-force-field"));
        let c = qmmm_certificate("rhf", &reference(), &sys, Some(3), false);
        assert!(names(&c).contains(&"mm-force-field"));
        assert_eq!(c.warnings, vec!["the embedded SCF did not converge"]);
    }

    #[test]
    fn cdft_certificate_records_each_constraint() {
        use ferric_dft::cdft::{Constraint, SpinChannel};
        let cfg = RhfConfig {
            constraints: vec![Constraint {
                fragment: vec![0],
                spin: SpinChannel::Total,
                target: 2.2,
            }],
            ..Default::default()
        };
        let run = CdftRun {
            lambdas: &[0.5],
            populations: &[2.2],
            outer_iterations: 4,
            scf_converged: true,
            max_constraint_error: 1e-7,
        };
        let c = cdft_certificate(&reference(), &cfg, &run);
        assert_eq!(c.capability, "cdft:uhf");
        assert_eq!(c.grade, Grade::Ungraded);
        let k = c
            .approximations
            .iter()
            .find(|a| a.name == "cdft-constraint");
        assert!(k.is_some(), "no constraint record: {:?}", names(&c));
        let k = k.unwrap();
        assert!(k
            .detail
            .starts_with("Total population of atoms [0], target 2.2"));
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);
        let bad = CdftRun {
            scf_converged: false,
            max_constraint_error: 1.0,
            ..run
        };
        assert_eq!(cdft_certificate(&reference(), &cfg, &bad).warnings.len(), 2);
    }

    #[test]
    fn coupling_certificate_prefixes_state_b() {
        let a = Certificate::ungraded("cdft:uhf").with_approximation("cdft-constraint", "a");
        let b = Certificate::ungraded("cdft:uhf")
            .with_approximation("cdft-constraint", "b")
            .warn("w");
        let c = cdft_coupling_certificate(&a, &b, 1e-3, 0.1);
        assert_eq!(c.capability, "cdft-coupling");
        assert!(names(&c).contains(&"scf cdft-constraint"));
        assert!(names(&c).contains(&"state B cdft-constraint"));
        assert!(names(&c).contains(&"coupling-formula"));
        assert_eq!(c.warnings, vec!["state B: w".to_string()]);
    }

    #[test]
    fn saddle_certificate_warns_unless_a_first_order_saddle() {
        let cfg = crate::saddle::SaddleConfig::default();
        let run = SaddleRun {
            steps: 7,
            converged: true,
            n_imaginary: 1,
            hessian_source: "finite-difference",
            delta: Some(0.01),
        };
        let c = saddle_certificate("rhf", &reference(), &cfg, &run);
        assert_eq!(c.capability, "saddle:rhf");
        assert_eq!(c.grade, Grade::Ungraded);
        assert!(c.warnings.is_empty());
        let h = c
            .approximations
            .iter()
            .find(|a| a.name == "hessian")
            .unwrap();
        assert_eq!(h.detail, "finite-difference, delta 0.01 Bohr");
        let c = saddle_certificate(
            "rhf",
            &reference(),
            &cfg,
            &SaddleRun {
                converged: false,
                n_imaginary: 0,
                ..run
            },
        );
        assert_eq!(c.warnings.len(), 2);
    }
}
