use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub molecule: MoleculeCfg,
    pub basis: BasisCfg,
    pub method: MethodCfg,
    #[serde(default)]
    pub scf: ScfCfg,
    #[serde(default)]
    pub mp2: Mp2Cfg,
    #[serde(default)]
    pub optimize: OptimizeCfg,
    #[serde(default)]
    pub rpa: RpaCfg,
    #[serde(default)]
    pub gw: GwCfg,
    #[serde(default)]
    pub dft: DftCfg,
    #[serde(default)]
    pub memory: MemoryCfg,
    #[serde(default)]
    pub external_potential: ExternalPotentialCfg,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct MemoryCfg {
    /// Unified memory budget (in GiB) for every method's resident 3-index
    /// tensors and MO transforms (SCF DF-JK, RI-MP2, OO-MP2, RPA, GW, CC).
    /// When set, it is threaded into ALL method configs.
    ///
    /// Every memory-budget setting shares ONE precedence chain, and **TOML/config
    /// overrides env** (highest first):
    ///   1. this field / a Python kwarg  (TOML — wins over env)
    ///   2. `FERRIC_MEM_BUDGET_GB` env (GiB)
    ///   3. legacy `FERRIC_OOC_BUDGET_GB` / `FERRIC_ERI3_BUDGET_GB` env (GiB)
    ///   4. auto: 0.8 × detected available RAM (cgroup limit ∧ MemAvailable)
    ///   5. 2 GiB fallback
    /// Leave unset to auto-detect. (The only memory-related env var with no TOML
    /// field is `FERRIC_OOC_TRACE`, a debug-print toggle — env-only by the same
    /// convention as every other `FERRIC_*_TRACE` flag, not a budget setting.)
    pub budget_gb: Option<f64>,
    /// Deprecated alias for `budget_gb`, retained so existing TOML that only set
    /// `three_index_budget_gb` still parses. Prefer `budget_gb`. When both are
    /// present, `budget_gb` wins.
    pub three_index_budget_gb: Option<f64>,
}

impl MemoryCfg {
    /// The effective unified budget in GiB, preferring the new `budget_gb`
    /// field, else the deprecated `three_index_budget_gb`.
    pub fn budget_gb(&self) -> Option<f64> {
        self.budget_gb.or(self.three_index_budget_gb)
    }

    /// The unified budget in bytes for passing as an explicit `Option<usize>` to
    /// method configs / the resolver.
    pub fn budget_bytes(&self) -> Option<usize> {
        self.budget_gb().map(ferric_core::memory::gib_to_bytes)
    }
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct DftCfg {
    /// XC functional name: "LDA", "PBE", "B3LYP", "wB97X-V", or any libxc name.
    pub functional: Option<String>,
}

/// One `[[external_potential.point_charges]]` entry: a fixed point charge
/// (units: e for `q`, Bohr for coordinates) contributing to the one-electron
/// Hamiltonian and nuclear-repulsion-like energy term.
#[derive(Deserialize, Default)]
pub struct PointChargeCfg {
    pub q: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// The `[external_potential]` TOML section: an array of fixed point charges
/// plus an optional uniform external electric field (a.u.).
#[derive(Deserialize, Default)]
pub struct ExternalPotentialCfg {
    #[serde(default)]
    pub point_charges: Vec<PointChargeCfg>,
    pub field: Option<[f64; 3]>,
}

impl ExternalPotentialCfg {
    /// Convert into the solver-facing type. Returns `None` when both
    /// `point_charges` is empty and `field` is unset (a true no-op,
    /// matching `RhfConfig.external_potential`'s `None` default).
    pub fn to_external_potential(&self) -> Option<ferric_core::external_potential::ExternalPotential> {
        if self.point_charges.is_empty() && self.field.is_none() {
            return None;
        }
        Some(ferric_core::external_potential::ExternalPotential {
            point_charges: self
                .point_charges
                .iter()
                .map(|pc| ferric_core::external_potential::PointCharge {
                    q: pc.q,
                    x: pc.x,
                    y: pc.y,
                    z: pc.z,
                })
                .collect(),
            field: self.field,
        })
    }
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct OptimizeCfg {
    pub max_steps: Option<usize>,
    pub g_max_thresh: Option<f64>,
    pub g_rms_thresh: Option<f64>,
    pub e_conv: Option<f64>,
    pub trust_radius: Option<f64>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Mp2Cfg {
    pub auxbasis: Option<String>,
    #[serde(default)]
    pub frozen_core: usize,
    // NOTE: `orbital_optimize` used to live here behind `#[allow(dead_code)]`.
    // Nothing ever read it — orbital optimization is selected with
    // `method.kind = "oo-rimp2"`. Setting it did nothing, which silently gave
    // plain RI-MP2 to anyone who expected OO-RI-MP2. Removed rather than wired
    // up: `kind` is already the selector, and with `deny_unknown_fields` the
    // stale key now errors instead of lying.
    /// Range-separation parameter ω in Å⁻¹ (for att-rimp2 and rs-mp2-rpa). Default 0.420.
    pub omega: Option<f64>,
    /// SCS opposite-spin scaling coefficient.
    pub c_os: Option<f64>,
    /// SCS same-spin scaling coefficient.
    pub c_ss: Option<f64>,
    /// Number of Laplace quadrature points (for laplace-mp2).
    pub n_quad: Option<usize>,
    /// SR-MP2 + LR-RPA formulation (for rs-mp2-rpa):
    ///
    ///   "delta-lr"      (default) — Δ-form B: E_MP2[Coulomb] + (E_dRPA[erf] − 2·E_OS[erf]).
    ///                   Pure-LR rings; mixed SR×LR rings dropped. Cost: 1 dRPA[erf] call.
    ///
    ///   "coupled-rings" — formulation T: E_MP2[Coulomb] + ΔdRPA[Coulomb] − ΔdRPA[erfc].
    ///                   Screens all rings (ΔdRPA[Coulomb]), un-screens pure-SR rings
    ///                   (−ΔdRPA[erfc]). Adds all mixed SR×LR rings. Cost: 2 dRPA calls.
    ///
    /// Both formulations have the same exact limits: ω→0 ⇒ plain MP2; ω→∞ ⇒ MP2+ΔdRPA[Coulomb].
    pub formulation: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RpaCfg {
    pub auxbasis: Option<String>,
    #[serde(default)]
    pub frozen_core: usize,
    /// Number of imaginary-frequency quadrature points.
    ///
    /// NOTE: the fallback when unset is surface-dependent (historical drift,
    /// kept to avoid silently changing published numbers): 20 for a pdep-rpa
    /// energy run, 16 for `task = "optimize"` (finite-difference gradients
    /// re-run the RPA energy 6·natoms times), 40 in the Python
    /// `run_pdep_rpa` binding. Set it explicitly for reproducibility.
    pub n_quad: Option<usize>,
    /// Imaginary-frequency quadrature scheme. One of:
    ///   "gauss-legendre" | "gl"           — GL nodes mapped via ω = u₀(1+x)/(1−x) (default)
    ///   "minimax" | "mm"                  — GL nodes with literature-optimized u₀(n_quad)
    ///   "chebyshev-tan" | "chebyshev" | "ct" — Eshuis-Yarkony-Furche tan-map (bounded ω)
    ///
    /// Unknown values are a hard error. `u0` is honoured by "gauss-legendre" and
    /// "chebyshev-tan"; "minimax" derives u₀ from `n_quad` and ignores it.
    pub quadrature: Option<String>,
    pub trunc_thresh: Option<f64>,
    /// Convergence threshold for the static dielectric eigensolver (Lanczos by
    /// default, Davidson if selected). The old name `davidson_conv_thresh` is
    /// accepted as an alias — it was misleading (it never was Davidson-specific)
    /// but existing TOML files must not break.
    #[serde(alias = "davidson_conv_thresh")]
    pub eigensolver_conv_thresh: Option<f64>,
    /// χ₀ sparsity strategy. One of:
    ///   "dense"            — dense MO-basis χ₀ (default; fastest ≤~20 atoms)
    ///   "boys"             — Boys-screened, default thresh 1e-4
    ///   "boys:<thresh>"    — Boys-screened with explicit threshold, e.g. "boys:1e-3"
    ///   "auto"             — pick Dense/Boys by atom count (cutoff 30, thresh 1e-4)
    ///   "auto:<cutoff>"    — auto with explicit atom cutoff, e.g. "auto:24"
    ///   "auto:<cutoff>:<thresh>" — auto with explicit cutoff and Boys threshold
    ///
    /// Recommendations (see `boys-screening-crossover`): Boys-screening's
    /// per-orbital tile overhead makes it SLOWER than Dense below ~20 atoms and a
    /// win only above the naphthalene-scale crossover, so the conservative auto
    /// cutoff is 30. The 1e-4 default threshold keeps the auto-switch energy
    /// within ~µHa of Dense; loosen to 1e-3 only on large aromatics where ~50%
    /// pair reduction costs <1e-4 Ha. For reproducible benchmarks across sizes,
    /// pin "dense" explicitly rather than "auto".
    pub chi0_sparsity: Option<String>,
    pub u0: Option<f64>,
    #[serde(default)]
    pub run_diagnostics: bool,
    /// If set, write PDEP eigenpotentials to `<prefix>_eigpot_NNN.cube`.
    pub export_eigpot_prefix: Option<String>,
    /// Number of leading eigenpotentials to export (default: 10).
    pub export_eigpot_count: Option<usize>,
    /// Cube grid spacing in Bohr (default: 0.2).
    pub cube_spacing: Option<f64>,
    /// Cube grid margin in Bohr beyond bounding box (default: 4.0).
    pub cube_margin: Option<f64>,
    /// If set, write per-molecule features to this NPZ path (eigenpotentials,
    /// coords, atomic numbers, optional ESP-at-atoms and α tensor).
    pub export_npz: Option<String>,
    /// Compute and include ESP at each nuclear position in the NPZ bundle.
    /// Default: true when `export_npz` is set.
    pub compute_esp: Option<bool>,
    /// Compute and include the static polarizability tensor in the NPZ bundle.
    /// Default: true when `export_npz` is set.
    pub compute_polarizability: Option<bool>,
    /// Compute and include the per-atom **Becke** polarizability decomposition
    /// (`alpha_atomic`, shape (N, 3, 3), additive to `alpha_tensor`).
    ///
    /// This path always uses the Becke partition (`pdep_polarizability_becke`);
    /// it is NOT governed by `c6_partition`, which only selects the partition
    /// for the C6 lane. Per-atom magnitudes are strongly partition-dependent
    /// (~10× between schemes) — see the `per-atom-c6-status` finding — so do
    /// not compare these against Hirshfeld-partitioned per-atom α.
    /// Default: true when `export_npz` is set.
    pub compute_alpha_atomic: Option<bool>,
    /// Compute and include the electric field at each nuclear position in the
    /// NPZ bundle (stored as `electric_field`, shape (natoms, 3), a.u.).
    /// Default: true when `export_npz` is set.
    pub compute_electric_field: Option<bool>,
    /// Include the AO-basis density matrix in the NPZ bundle (stored as
    /// `density_matrix`, shape (n_bf, n_bf), float64). Needed downstream for
    /// CM5 charge derivation and density-derived properties.
    /// Default: true when `export_npz` is set.
    pub compute_density_matrix: Option<bool>,
    /// Compute and include the molecular dipole moment in the NPZ bundle
    /// (stored as `dipole`, shape (3,), float64, atomic units e·a0). This is the
    /// exact dipole of the SCF/RPA total density, μ = −Tr(P·D) + Σ_A Z_A R_A,
    /// where D is the AO dipole-integral matrix ⟨μ|r|ν⟩ about the origin. Neutral
    /// molecules → origin-independent. It is the QC ground-truth dipole against
    /// which partition-derived (Löwdin/Hirshfeld) dipoles are adjudicated.
    /// Default: true when `export_npz` is set.
    pub compute_dipole: Option<bool>,
    /// Compute and include Hirshfeld atomic charges in the NPZ bundle
    /// (stored as `hirshfeld_charges`, shape (natoms,), float64, units of e).
    /// These are the Hirshfeld baseline charges; downstream CM5 pair-correction
    /// is applied in the consumer.
    /// Default: true when `export_npz` is set.
    pub compute_hirshfeld_charges: Option<bool>,
    /// Compute and include Löwdin atomic charges (from symmetrically-
    /// orthogonalized AOs) in the NPZ bundle as `lowdin_charges`,
    /// shape (natoms,), float64, e. Recommended baseline charges for CM5
    /// (no proatom approximation, basis-set-stable).
    /// Default: true when `export_npz` is set.
    pub compute_lowdin_charges: Option<bool>,
    /// Compute per-atom anisotropic C6 dispersion coefficients and include them
    /// in the NPZ bundle (`c6_iso`, `c6_aniso`, `alpha_atomic_dynamic`,
    /// `c6_freqs`, `c6_weights`). Default: true when `export_npz` is set.
    pub compute_c6: Option<bool>,
    /// C6 polarizability source. One of:
    ///   "ts"   — Tkatchenko-Scheffler single-pole model (default)
    ///   "pdep" — true PDEP-RPA dynamic α(iω) on the RPA quadrature grid
    ///   "mbd"  — many-body dispersion (coupled-dipole) on top of the TS α.
    ///            Known-bad for soft atoms; makes silicon worse, not better.
    ///
    /// Unknown values are a hard error (they used to silently run "ts").
    pub c6_source: Option<String>,
    /// Per-atom partition for C6: "hirshfeld" or "becke". Unset defaults to
    /// Hirshfeld when `c6_source = "pdep"`, Becke for "ts"/"mbd".
    /// Hirshfeld is required for correct anisotropy in pdep C6 — Becke atom-centred
    /// dipoles lose charge-transfer contributions and invert bond-axis ordering.
    /// For TS, partition only affects alpha_static shape; Hirshfeld volumes are
    /// always used for the volume ratio regardless of this setting.
    ///
    /// Unknown values are a hard error (they used to silently use the default).
    pub c6_partition: Option<String>,
    /// XC functional for the RPA *reference* orbitals (e.g. "PBE0", "PBE").
    /// `None` (default) uses a Hartree-Fock reference (RPA@HF). Setting this
    /// runs the closed-shell KS-DFT solver first, so the RPA/PDEP response and
    /// C6 are built on KS orbitals (RPA@PBE0 etc.) — KS orbitals have smaller
    /// HOMO-LUMO gaps, raising the polarizability toward experiment.
    pub xc: Option<String>,
}

impl RpaCfg {
    /// Parse the `chi0_sparsity` TOML string into a [`Chi0Sparsity`].
    ///
    /// Accepted forms (case-insensitive, whitespace-trimmed); an optional
    /// `@<radius_bohr>` suffix on the boys/auto forms sets the G6 centroid
    /// distance pre-filter (omit → ∞ = filter off, byte-identical to pre-G6):
    ///   None / "dense"                 → Dense (default; backward compatible)
    ///   "boys"                         → BoysScreened { thresh: 1e-4, dist: ∞ }
    ///   "boys:<thresh>"                → BoysScreened with that threshold
    ///   "boys:<thresh>@<radius>"       → …and that distance-cutoff radius (Bohr)
    ///   "auto"                         → Auto { cutoff: 30, thresh: 1e-4, dist ∞ }
    ///   "auto:<cutoff>"                → Auto with that atom cutoff
    ///   "auto:<cutoff>:<thresh>"       → …and that Boys threshold
    ///   "auto:<cutoff>:<thresh>@<rad>" → …and that distance-cutoff radius (Bohr)
    pub fn parse_chi0_sparsity(&self) -> Result<ferric_rpa::config::Chi0Sparsity, String> {
        // Canonical parser lives on the type (shared with the Python bindings).
        ferric_rpa::config::Chi0Sparsity::parse_config_str(self.chi0_sparsity.as_deref())
    }

    /// Parse the `[rpa] quadrature` TOML string into a [`QuadratureScheme`],
    /// warning if `u0` was set but the chosen scheme ignores it.
    ///
    /// Unknown strings are an error (they used to silently run Gauss-Legendre).
    pub fn parse_quadrature(&self) -> Result<ferric_rpa::config::QuadratureScheme, String> {
        let scheme =
            ferric_rpa::config::QuadratureScheme::parse_config_str(self.quadrature.as_deref())
                .map_err(|e| format!("[rpa] quadrature: {e}"))?;
        if self.u0.is_some() && !scheme.honours_u0() {
            eprintln!(
                "warning: [rpa] u0 is ignored by quadrature = \"{}\" \
                 (it derives u0 from n_quad); remove u0 or pick \
                 \"gauss-legendre\"/\"chebyshev-tan\"",
                self.quadrature.as_deref().unwrap_or("minimax")
            );
        }
        Ok(scheme)
    }
}

/// The `[gw]` TOML section: `GwConfig` knobs for `method.kind = "gw"`. Reuses
/// the existing `[rpa]` section for the underlying `PdepRpaConfig` (a GW run
/// needs both — `[rpa]` for the screened-interaction PDEP basis, `[gw]` for
/// the self-energy/QP-solver knobs), exactly like `pdep-rpa` already does.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct GwCfg {
    /// GW method: "g0w0" | "cohsex" | "evgw0" | "evgw" (case-insensitive).
    /// Unknown values are a hard error — never silently defaults to G0W0.
    pub method: Option<String>,
    /// Range of MOs (absolute indices, `[lo, hi)`) for which to compute QP
    /// energies. Unset → library default `{HOMO-2..LUMO+2}`.
    pub qp_mos: Option<[usize; 2]>,
    /// Max evGW/evGW0 outer (eigenvalue self-consistency) iterations.
    pub max_ev_iter: Option<usize>,
    /// evGW/evGW0 convergence threshold on |Δε^QP|_max (Ha).
    pub ev_conv_thresh: Option<f64>,
    /// Number of Padé continued-fraction coefficients. 0/unset → use
    /// `[rpa] n_quad`.
    pub pade_npts: Option<usize>,
    /// Newton-step damping for the QP solver.
    pub qp_newton_damp: Option<f64>,
    /// Frozen core for the GW self-energy build. Must match `[rpa]
    /// frozen_core` for self-consistency between W and Σ — the CLI passes
    /// this value to both `GwConfig.frozen_core` and overrides the PDEP
    /// config's frozen_core with it.
    pub frozen_core: Option<usize>,
}

impl GwCfg {
    /// Parse the `[gw] method` TOML string into a [`ferric_gw::GwMethod`].
    /// Unset defaults to G0W0 (matches `GwConfig::default()`); unknown
    /// strings are a hard error (this repo's strict-config-parsing
    /// convention — never silently default to a method the user didn't ask
    /// for).
    pub fn parse_method(&self) -> Result<ferric_gw::GwMethod, String> {
        use ferric_gw::GwMethod;
        match self.method.as_deref().map(|s| s.trim().to_ascii_lowercase()) {
            None => Ok(GwMethod::G0W0),
            Some(ref s) if s == "g0w0" => Ok(GwMethod::G0W0),
            Some(ref s) if s == "cohsex" => Ok(GwMethod::Cohsex),
            Some(ref s) if s == "evgw0" => Ok(GwMethod::EvGw0),
            Some(ref s) if s == "evgw" => Ok(GwMethod::EvGw),
            Some(other) => Err(format!(
                "[gw] method: unknown value \"{other}\"; expected \"g0w0\", \"cohsex\", \"evgw0\", or \"evgw\""
            )),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoleculeCfg {
    pub xyz: String,
    #[serde(default)]
    pub charge: i32,
    #[serde(default = "default_multiplicity")]
    pub multiplicity: usize,
}

fn default_multiplicity() -> usize { 1 }

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BasisCfg {
    pub name: Option<String>,
    pub path: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MethodCfg {
    pub kind: String,
    #[serde(default = "default_task")]
    pub task: String,
}

fn default_task() -> String { "energy".into() }

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScfCfg {
    #[serde(default = "default_max_iter")]
    pub max_iter: usize,
    #[serde(default = "default_energy_conv")]
    pub energy_conv: f64,
    #[serde(default = "default_density_conv")]
    pub density_conv: f64,
    #[serde(default = "default_diis_size")]
    pub diis_size: usize,
    #[serde(default = "default_integral_thresh")]
    pub integral_thresh: f64,
    pub k_builder: Option<String>,
    pub df_j_aux: Option<String>,
    pub df_k_aux: Option<String>,
    /// Optional virtual-virtual block level shift (Ha) for open-shell SCF
    /// (UHF / ROHF / UKS / ROKS). The shift is rational-damped by the DIIS
    /// error so the converged Fock is the unshifted stationary point. A
    /// value of 0.2 is a useful default for OH-like doublets at LDA/PBE
    /// where DIIS otherwise plateaus.
    pub level_shift: Option<f64>,
    /// Maximum-Overlap Method: pin the occupied set by AO-overlap with the
    /// previous iteration's occupation after this many DIIS iterations
    /// (0 = aufbau throughout). Fixes occupied-set flip-flop non-convergence.
    #[serde(default)]
    pub mom_after_iter: usize,
    /// SCF convergence ladder: a sequence of `[[scf.ladder]]` rungs walked in
    /// order (density carried forward unless a rung sets `restart = true`),
    /// stopping at the first converged rung. Empty (default, no `[[scf.ladder]]`
    /// tables in the TOML) falls back to `ferric_scf::ladder::default_ladder()`
    /// at `build_ladder` time.
    #[serde(default)]
    pub ladder: Vec<LadderRungCfg>,
}

impl Default for ScfCfg {
    fn default() -> Self {
        ScfCfg {
            max_iter: 100,
            // Match the library convergence gate (rhf::scf_converged): density_conv
            // is the tight (reachable) ΔP signal, energy_conv a loose
            // "not-descending" bound. A tight energy_conv here would hang a large
            // DF molecule at MaxIter, since dE floors on the RI noise level.
            energy_conv: 1e-3,
            density_conv: 1e-6,
            diis_size: 8,
            integral_thresh: 1e-12,
            k_builder: None,
            df_j_aux: None,
            df_k_aux: None,
            level_shift: None,
            mom_after_iter: 0,
            ladder: Vec::new(),
        }
    }
}

fn default_max_iter() -> usize { 100 }
// Match the library convergence gate (rhf::scf_converged): density_conv is the
// tight (reachable) ΔP signal; energy_conv is a LOOSE "not-descending" bound. A
// tight energy_conv hangs a large DF molecule (dE floors on the RI noise level).
fn default_energy_conv() -> f64 { 1e-3 }
fn default_density_conv() -> f64 { 1e-6 }
fn default_diis_size() -> usize { 8 }
fn default_integral_thresh() -> f64 { 1e-12 }

/// One `[[scf.ladder]]` rung. Every field is optional and overrides the
/// corresponding field of the `base` `RhfConfig` passed to
/// [`ScfCfg::build_ladder`] (derived from the flat `[scf]` settings); unset
/// fields inherit from `base`.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct LadderRungCfg {
    /// Initial guess for this rung: "sad" | "sad-smallbasis" | "hcore".
    /// `None` (default) behaves like "sad".
    pub guess: Option<String>,
    pub level_shift: Option<f64>,
    pub max_iter: Option<usize>,
    pub df_j_aux: Option<String>,
    pub df_k_aux: Option<String>,
    pub stall_window: Option<usize>,
    pub divergence_tol: Option<f64>,
    /// false (default): inherit the previous rung's final density.
    /// true: discard the incoming density and use this rung's own guess.
    pub restart: bool,
}

impl Default for LadderRungCfg {
    fn default() -> Self {
        Self {
            guess: None,
            level_shift: None,
            max_iter: None,
            df_j_aux: None,
            df_k_aux: None,
            stall_window: None,
            divergence_tol: None,
            restart: false,
        }
    }
}

impl ScfCfg {
    /// Build the SCF convergence ladder. If no `[[scf.ladder]]` rungs are
    /// configured, returns the built-in `default_ladder()`. Otherwise each
    /// rung starts from `base` (the `RhfConfig` derived from the flat `[scf]`
    /// settings) and overrides the fields the rung specifies.
    pub fn build_ladder(&self, base: &ferric_scf::rhf::RhfConfig) -> Vec<ferric_scf::ladder::Rung> {
        use ferric_scf::ladder::Rung;
        if self.ladder.is_empty() {
            // Default escalation, but seeded from the user's [scf] settings
            // (base) so max_iter/energy_conv/density_conv/mom_after_iter/etc.
            // are honored -- a plain `kind = "rhf"` run with no [[scf.ladder]]
            // table must not silently discard the [scf] block. DF-JK aux and
            // stall/divergence-abort are layered on top (unless the user
            // already set df_j_aux/df_k_aux); level-shift escalates 0->0.5->1.0.
            let jk = "def2-universal-jkfit";
            let mk = |ls: f64| {
                let mut c = base.clone();
                c.level_shift = ls;
                if c.df_j_aux.is_none() {
                    c.df_j_aux = Some(jk.to_string());
                }
                if c.df_k_aux.is_none() {
                    c.df_k_aux = Some(jk.to_string());
                }
                c.stall_window = Some(15);
                c.divergence_tol = Some(0.5);
                c
            };
            return vec![
                Rung { config: mk(0.0), restart: false },
                Rung { config: mk(0.5), restart: false },
                Rung { config: mk(1.0), restart: false },
            ];
        }
        self.ladder
            .iter()
            .map(|r| {
                let mut cfg = base.clone();
                if let Some(v) = r.level_shift {
                    cfg.level_shift = v;
                }
                if let Some(v) = r.max_iter {
                    cfg.max_iter = v;
                }
                if r.df_j_aux.is_some() {
                    cfg.df_j_aux = r.df_j_aux.clone();
                }
                if r.df_k_aux.is_some() {
                    cfg.df_k_aux = r.df_k_aux.clone();
                }
                cfg.stall_window = r.stall_window;
                cfg.divergence_tol = r.divergence_tol;
                match r.guess.as_deref() {
                    Some("hcore") => {
                        cfg.use_sad_guess = false;
                    }
                    Some("sad") | None => {
                        cfg.use_sad_guess = true;
                    }
                    Some("sad-smallbasis") => {
                        eprintln!("warning: scf.ladder guess \"sad-smallbasis\" is not yet wired to the CLI rung guess; using plain SAD");
                        cfg.use_sad_guess = true;
                    }
                    Some(other) => {
                        eprintln!("warning: unknown scf.ladder guess \"{other}\", using sad");
                        cfg.use_sad_guess = true;
                    }
                }
                Rung { config: cfg, restart: r.restart }
            })
            .collect()
    }
}

pub fn load_config(path: &str) -> Result<Config, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{e}"))?;
    toml::from_str(&text).map_err(|e| format!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every shipped example must parse. With `deny_unknown_fields` on all
    /// config structs, this doubles as the guard that the strict parser never
    /// rejects a key the examples (and thus users' existing files) rely on.
    ///
    /// Also checks that each example's `[molecule].xyz` path actually resolves
    /// (relative to the workspace root, matching how `ferric` is normally
    /// invoked) — TOML syntax validity alone let `h2_opt.toml` reference a
    /// nonexistent `h2_stretched.xyz` silently for as long as the example
    /// existed (found 2026-07-18 while spot-checking geometry optimization
    /// against literature/PySCF).
    #[test]
    fn all_shipped_examples_parse() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
        let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut n = 0;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            let s = std::fs::read_to_string(&path).unwrap();
            let cfg: Config = match toml::from_str(&s) {
                Ok(c) => c,
                Err(e) => panic!("example {} no longer parses: {e}", path.display()),
            };
            let xyz_path = workspace_root.join(&cfg.molecule.xyz);
            assert!(
                xyz_path.is_file(),
                "example {} references [molecule].xyz = {:?}, which does not exist at {}",
                path.display(),
                cfg.molecule.xyz,
                xyz_path.display()
            );
            n += 1;
        }
        assert!(n > 0, "no example TOMLs found in {}", dir.display());
    }

    /// Unknown/typo'd keys must be a parse error, not silently ignored. A
    /// misspelled `trunc_thresh` used to run at the default and report success.
    #[test]
    fn unknown_keys_are_rejected() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "pdep-rpa"
[rpa]
trunc_threshold = 1e-12
"#;
        let err = match toml::from_str::<Config>(toml_str) {
            Ok(_) => panic!("typo'd key parsed successfully — deny_unknown_fields regressed"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("trunc_threshold"), "error should name the bad key: {err}");
    }

    #[test]
    fn test_parse_config() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rhf"
[scf]
max_iter = 50
energy_conv = 1e-9
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.molecule.xyz, "water.xyz");
        assert_eq!(cfg.basis.name.as_deref(), Some("sto-3g"));
        assert_eq!(cfg.method.kind, "rhf");
        assert_eq!(cfg.scf.max_iter, 50);
    }

    #[test]
    fn test_parse_attenuated_config() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "att-rimp2"
[mp2]
auxbasis = "cc-pvdz-ri"
omega = 0.420
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.method.kind, "att-rimp2");
        assert!((cfg.mp2.omega.unwrap() - 0.420).abs() < 1e-10);
    }

    #[test]
    fn test_parse_gw_config() {
        let toml_str = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "gw"
[rpa]
auxbasis = "cc-pvdz-ri"
n_quad = 16
[gw]
method = "evgw0"
qp_mos = [3, 6]
max_ev_iter = 30
ev_conv_thresh = 1e-5
qp_newton_damp = 0.8
frozen_core = 1
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.method.kind, "gw");
        assert_eq!(cfg.gw.method.as_deref(), Some("evgw0"));
        assert_eq!(cfg.gw.qp_mos, Some([3, 6]));
        assert_eq!(cfg.gw.max_ev_iter, Some(30));
        assert!((cfg.gw.ev_conv_thresh.unwrap() - 1e-5).abs() < 1e-12);
        assert!((cfg.gw.qp_newton_damp.unwrap() - 0.8).abs() < 1e-12);
        assert_eq!(cfg.gw.frozen_core, Some(1));
        assert_eq!(cfg.gw.parse_method().unwrap(), ferric_gw::GwMethod::EvGw0);
    }

    #[test]
    fn test_parse_gw_config_defaults() {
        // Empty [gw] section (or absent entirely) must parse and default to G0W0.
        let toml_str = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "gw"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.gw.parse_method().unwrap(), ferric_gw::GwMethod::G0W0);
    }

    #[test]
    fn gw_method_unknown_string_is_an_error() {
        let mut gw = GwCfg::default();
        gw.method = Some("gw-bse".to_string());
        let err = gw.parse_method().unwrap_err();
        assert!(err.contains("gw-bse"), "error should name the bad value: {err}");
    }

    #[test]
    fn gw_method_is_case_insensitive() {
        let mut gw = GwCfg::default();
        gw.method = Some("EvGW".to_string());
        assert_eq!(gw.parse_method().unwrap(), ferric_gw::GwMethod::EvGw);
    }

    #[test]
    fn test_parse_mp3_config() {
        let toml_str = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "mp3"
[mp2]
auxbasis = "cc-pvdz-ri"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.method.kind, "mp3");
        assert_eq!(cfg.mp2.auxbasis.as_deref(), Some("cc-pvdz-ri"));
        assert_eq!(cfg.mp2.frozen_core, 0);
    }

    #[test]
    fn test_parse_rs_mp2_rpa_config() {
        let toml_str = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "rs-mp2-rpa"
[mp2]
auxbasis = "cc-pvdz-ri"
omega = 0.3
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.method.kind, "rs-mp2-rpa");
        assert_eq!(cfg.mp2.omega, Some(0.3));
        // Default formulation is absent (None → "delta-lr" at runtime).
        assert_eq!(cfg.mp2.formulation, None);
    }

    #[test]
    fn test_parse_rs_mp2_rpa_formulation() {
        let toml_str = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "rs-mp2-rpa"
[mp2]
auxbasis = "cc-pvdz-ri"
omega = 0.420
formulation = "coupled-rings"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.method.kind, "rs-mp2-rpa");
        assert_eq!(cfg.mp2.formulation.as_deref(), Some("coupled-rings"));
    }

    #[test]
    fn test_parse_rs_mp2_rpa_formulation_delta_lr() {
        let toml_str = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "rs-mp2-rpa"
[mp2]
auxbasis = "cc-pvdz-ri"
omega = 0.420
formulation = "delta-lr"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.mp2.formulation.as_deref(), Some("delta-lr"));
    }

    #[test]
    fn parse_chi0_sparsity_variants() {
        use ferric_rpa::config::Chi0Sparsity;
        let mk = |s: Option<&str>| {
            let mut r = RpaCfg::default();
            r.chi0_sparsity = s.map(|x| x.to_string());
            r.parse_chi0_sparsity()
        };
        // None / "dense" → Dense (default, backward compatible).
        assert_eq!(mk(None).unwrap(), Chi0Sparsity::Dense);
        assert_eq!(mk(Some("dense")).unwrap(), Chi0Sparsity::Dense);
        // boys with default (1e-4) + explicit threshold. dist_cutoff defaults to ∞.
        const INF: f64 = f64::INFINITY;
        assert_eq!(mk(Some("boys")).unwrap(), Chi0Sparsity::BoysScreened { thresh: 1e-4, dist_cutoff: INF });
        assert_eq!(mk(Some("boys:1e-3")).unwrap(), Chi0Sparsity::BoysScreened { thresh: 1e-3, dist_cutoff: INF });
        // boys/auto with an explicit `@<radius>` distance cutoff (Bohr).
        assert_eq!(mk(Some("boys:1e-3@12")).unwrap(), Chi0Sparsity::BoysScreened { thresh: 1e-3, dist_cutoff: 12.0 });
        assert_eq!(mk(Some("auto:24:5e-4@8")).unwrap(), Chi0Sparsity::Auto { boys_thresh: 5e-4, atom_cutoff: 24, dist_cutoff: 8.0 });
        // auto with defaults (cutoff 30, thresh 1e-4), explicit cutoff, explicit cutoff+thresh.
        assert_eq!(mk(Some("auto")).unwrap(), Chi0Sparsity::Auto { boys_thresh: 1e-4, atom_cutoff: 30, dist_cutoff: INF });
        assert_eq!(mk(Some("auto:24")).unwrap(), Chi0Sparsity::Auto { boys_thresh: 1e-4, atom_cutoff: 24, dist_cutoff: INF });
        assert_eq!(mk(Some("auto:24:5e-4")).unwrap(), Chi0Sparsity::Auto { boys_thresh: 5e-4, atom_cutoff: 24, dist_cutoff: INF });
        // case-insensitive + whitespace tolerant.
        assert_eq!(mk(Some("  AUTO ")).unwrap(), Chi0Sparsity::Auto { boys_thresh: 1e-4, atom_cutoff: 30, dist_cutoff: INF });
        // garbage → error (not silently ignored).
        assert!(mk(Some("frobnicate")).is_err());
        assert!(mk(Some("boys:notanumber")).is_err());
    }

    #[test]
    fn memory_budget_parses() {
        let toml_str = r#"
[molecule]
xyz = "x.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "rhf"
[memory]
three_index_budget_gb = 6.0
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.memory.three_index_budget_gb, Some(6.0));
    }

    #[test]
    fn memory_budget_gb_parses_and_converts_to_bytes() {
        // The preferred `budget_gb` field: it must parse AND flow through the
        // budget_gb()/budget_bytes() accessors that thread it into every method.
        let toml_str = r#"
[molecule]
xyz = "x.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "rhf"
[memory]
budget_gb = 16.0
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.memory.budget_gb, Some(16.0));
        assert_eq!(cfg.memory.budget_gb(), Some(16.0));
        assert_eq!(
            cfg.memory.budget_bytes(),
            Some(16 * 1024 * 1024 * 1024),
            "budget_gb must convert to GiB bytes for the resolver"
        );
    }

    #[test]
    fn memory_budget_gb_wins_over_deprecated_alias() {
        // When both are set, the new field wins (documented precedence).
        let toml_str = r#"
[molecule]
xyz = "x.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "rhf"
[memory]
budget_gb = 20.0
three_index_budget_gb = 6.0
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.memory.budget_gb(), Some(20.0));
    }

    #[test]
    fn memory_budget_defaults_to_none() {
        let toml_str = r#"
[molecule]
xyz = "x.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "rhf"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.memory.three_index_budget_gb, None);
    }

    #[test]
    fn ladder_rungs_parse_from_toml() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rhf"

[[scf.ladder]]
guess = "sad"
max_iter = 60

[[scf.ladder]]
guess = "hcore"
level_shift = 0.5
max_iter = 80
restart = true
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.scf.ladder.len(), 2);
        assert_eq!(cfg.scf.ladder[0].guess.as_deref(), Some("sad"));
        assert_eq!(cfg.scf.ladder[1].level_shift, Some(0.5));
        assert!(cfg.scf.ladder[1].restart);
        assert!(!cfg.scf.ladder[0].restart);
    }

    #[test]
    fn no_ladder_section_is_empty_and_falls_back_to_default_ladder() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rhf"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!(cfg.scf.ladder.is_empty());
        let built = cfg.scf.build_ladder(&ferric_scf::rhf::RhfConfig::default());
        assert_eq!(built.len(), ferric_scf::ladder::default_ladder().len());
    }

    #[test]
    fn empty_ladder_default_escalation_honors_base_scf_block() {
        // Regression for I1: a plain `kind = "rhf"` run with no [[scf.ladder]]
        // table must NOT silently discard the user's [scf] settings by
        // building every rung from RhfConfig::default(). Seed `base` with
        // mom_after_iter and max_iter values that differ from RhfConfig's
        // defaults and assert every rung in the empty-ladder escalation
        // carries them through.
        let base = ferric_scf::rhf::RhfConfig {
            mom_after_iter: 5,
            max_iter: 42,
            ..Default::default()
        };
        let cfg = ScfCfg::default();
        assert!(cfg.ladder.is_empty());
        let built = cfg.build_ladder(&base);
        assert_eq!(built.len(), 3);
        for (i, rung) in built.iter().enumerate() {
            assert_eq!(rung.config.mom_after_iter, 5, "rung {i} must inherit base.mom_after_iter");
            assert_eq!(rung.config.max_iter, 42, "rung {i} must inherit base.max_iter");
            assert!(rung.config.df_j_aux.is_some(), "rung {i} must default DF-J aux");
            assert!(rung.config.df_k_aux.is_some(), "rung {i} must default DF-K aux");
        }
        assert_eq!(built[0].config.level_shift, 0.0);
        assert_eq!(built[1].config.level_shift, 0.5);
        assert_eq!(built[2].config.level_shift, 1.0);
    }

    #[test]
    fn empty_ladder_default_escalation_does_not_override_user_df_aux() {
        // If the user already set df_j_aux/df_k_aux in [scf], the default
        // escalation must not clobber it with def2-universal-jkfit.
        let base = ferric_scf::rhf::RhfConfig {
            df_j_aux: Some("cc-pvdz-jkfit".to_string()),
            df_k_aux: Some("cc-pvdz-jkfit".to_string()),
            ..Default::default()
        };
        let cfg = ScfCfg::default();
        let built = cfg.build_ladder(&base);
        for rung in &built {
            assert_eq!(rung.config.df_j_aux.as_deref(), Some("cc-pvdz-jkfit"));
            assert_eq!(rung.config.df_k_aux.as_deref(), Some("cc-pvdz-jkfit"));
        }
    }

    #[test]
    fn external_potential_section_parses() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rhf"
task = "energy"

[[external_potential.point_charges]]
q = 1.0
x = 0.0
y = 0.0
z = 5.0

[external_potential]
field = [0.0, 0.0, 0.001]
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.external_potential.point_charges.len(), 1);
        assert_eq!(cfg.external_potential.point_charges[0].q, 1.0);
        assert_eq!(cfg.external_potential.field, Some([0.0, 0.0, 0.001]));
    }

    #[test]
    fn external_potential_section_optional() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rhf"
task = "energy"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!(cfg.external_potential.point_charges.is_empty());
        assert!(cfg.external_potential.field.is_none());
    }

    #[test]
    fn configured_ladder_rung_overrides_base_fields() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rhf"

[[scf.ladder]]
guess = "hcore"
level_shift = 0.3
max_iter = 42
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        let base = ferric_scf::rhf::RhfConfig::default();
        let built = cfg.scf.build_ladder(&base);
        assert_eq!(built.len(), 1);
        assert_eq!(built[0].config.level_shift, 0.3);
        assert_eq!(built[0].config.max_iter, 42);
        assert!(!built[0].config.use_sad_guess);
        assert!(!built[0].restart);
    }
}
