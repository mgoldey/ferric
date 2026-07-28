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
    /// Optional `[cosmo]` section: COSMO implicit-solvent configuration.
    /// Absent (or explicit `None`) means no solvation — byte-identical to a
    /// build with no COSMO support, per `RhfConfig.cosmo`'s convention.
    /// Reuses `ferric_scf::cosmo::CosmoConfig` directly (already
    /// `#[serde(deny_unknown_fields)]`) so there is exactly one definition
    /// of the COSMO config surface across CLI/Python/lib. `#[serde(default)]`
    /// so the section can be omitted entirely (serde does not treat a
    /// missing `Option` field as `None` automatically without it).
    #[serde(default)]
    pub cosmo: Option<ferric_scf::cosmo::CosmoConfig>,
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
    /// Double-hybrid adiabatic-connection parameter λ scaling the WFT
    /// correlation (ωB97X-L-V, paper eqn 27). `None` → the published value
    /// carried by `DoubleHybridConfig::default()` (0.6). Only read by
    /// `method.kind = "wb97x-l-v"`.
    pub lambda: Option<f64>,
    /// Double-hybrid range-separation parameter ω in Bohr⁻¹. `None` → the
    /// published value carried by `DoubleHybridConfig::default()` (0.1).
    /// Only read by `method.kind = "wb97x-l-v"`.
    pub omega: Option<f64>,
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
    /// Number of Laplace quadrature points (for laplace-mp2 and laplace-sos-mp2).
    /// Must be one of {3, 5, 7} — `LaplaceQuadrature::new` hard-errors otherwise
    /// rather than silently capping.
    pub n_quad: Option<usize>,
    /// Laplace SOS-MP2 algebra (for `method.kind = "laplace-sos-mp2"`):
    ///
    ///   "mo" (default) — τ-weighted `(P|ia)` amplitudes, `J = B(t)B(t)ᵀ`.
    ///   "ao"           — occupied/virtual pseudo-densities; no MO transform
    ///                    inside the quadrature loop.
    ///
    /// Both compute the SAME quantity and agree to round-off (asserted in
    /// `ferric-mp2`'s tests). The AO path is the correctness reference for the
    /// pseudo-density limit — it is dense here, so selecting it is NOT a
    /// scaling win; see `docs/notebooks/11-laplace-sos-mp2.ipynb`.
    ///
    /// Distinct from `formulation`, which selects the rs-mp2-rpa Δ-form.
    pub sos_formulation: Option<String>,
    /// Domain radius in **Bohr** for `sos_formulation = "ao-sparse"`.
    ///
    /// Required by that formulation and REJECTED by the other two — there is no
    /// safe default (the right radius is system- and basis-dependent) and
    /// silently ignoring it on an exact path would run a different method than
    /// the one configured.
    ///
    /// This is the one SOS variant that is APPROXIMATE: it discards AO pairs
    /// lying outside every Boys-orbital domain. It converges to `"ao"` as the
    /// radius grows.
    pub domain_cutoff_bohr: Option<f64>,
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
    /// SR-MP2 + LR-RPA range-separation kernel (for rs-mp2-rpa):
    ///
    ///   "erf"  (default) — LR=erf(ωr)/r, SR=erfc(ωr)/r, parameterized by `omega` (Å⁻¹).
    ///   "terf"           — tempered Dutoi/Goldey split: LR=terf(r,r0)/r, SR=terfc(r,r0)/r
    ///                      with terf+terfc=Coulomb exactly; parameterized by `r0` (Å).
    ///                      When "terf", `omega` is IGNORED (ω=1/(r0·√2) is derived).
    ///
    /// Same split identity ⇒ same exact limits as erf; only the attenuator SHAPE
    /// differs. terf needs the interpolation tables (FERRIC_TERF_TABLE_DIR).
    pub attenuator: Option<String>,
    /// Range-separation length r0 in **Å**, used ONLY when `attenuator = "terf"`.
    /// The single tempered-split knob; ω is derived (ω = 1/(r0·√2), computed in
    /// Bohr internally — this field is converted Å→Bohr at the CLI boundary,
    /// same convention as `r0_bonded`/`r0_nonbonded` below). Default 1.6828 Å
    /// (= 3.18 Bohr) ⇒ ω ≈ 0.42 Å⁻¹ (the erf operating point). Ignored for erf.
    /// FIXED 2026-07-21: this field used to be Bohr, inconsistent with every
    /// other r0-shaped field in this struct (`r0_bonded`/`r0_nonbonded` below
    /// were always Å) — the mismatch directly caused a unit-conversion bug in
    /// benchmarks/grid/run_grid.py (an Å value fed through a Bohr-assuming
    /// formula, off by a factor of ~1.89).
    pub r0: Option<f64>,
    /// Sweep several `r0` values (**Å**) in ONE job, reusing a single SCF.
    ///
    /// Only meaningful with `attenuator = "terf"`. When set, `r0` is ignored
    /// and the correlation stage is evaluated once per listed r0, printing a
    /// full result block per point. Values are sorted and de-duplicated.
    ///
    /// This exists because the SCF dominates a single-r0 job at aug-cc-pVQZ:
    /// amortizing it across N points makes an N-point scan roughly N times
    /// cheaper than N separate runs (measured ~5x for a 5-point scan, which is
    /// the difference between a ~4 h and a ~20 h A24 sweep).
    ///
    /// HISTORY worth knowing: an equivalent field existed uncommitted during
    /// the 2026-07-22/23 production sweeps and was lost, which left committed
    /// output data that no committed code could regenerate. That is why this is
    /// a real config field with a regression test rather than a local patch.
    pub r0_sweep: Option<Vec<f64>>,
    /// Bonded (shorter-range) terfc cutoff **r0(1)** in **Å**, used ONLY by
    /// `method.kind = "scs-mp2-2terfc"`. Default 0.75 Å (thesis value).
    /// Requires the terfc interpolation tables (`FERRIC_TERF_TABLE_DIR`).
    pub r0_bonded: Option<f64>,
    /// Non-bonded (longer-range) terfc cutoff **r0(2)** in **Å**, used ONLY by
    /// `method.kind = "scs-mp2-2terfc"`. Must be > `r0_bonded`. Default 1.05 Å
    /// (thesis value). Requires the terfc interpolation tables
    /// (`FERRIC_TERF_TABLE_DIR`).
    pub r0_nonbonded: Option<f64>,

    // ---- MP2-V (`method.kind = "mp2-v"`) -----------------------------------
    // Attenuated MP2 + damped VV10, Goldey/Belzunces/Head-Gordon JCTC 11, 4159
    // (2015). These knobs are deliberately NOT the generic `r0`/`attenuator`
    // fields above: those two belong to `rs-mp2-rpa`, carry different defaults
    // (r0 = 1.6828 Å, attenuator = "erf") and a different meaning for
    // `attenuator` ("erf"/"terf", the *splitter*, vs MP2-V's "terfc"/"erfc",
    // the *short-range operator*). Sharing them would make one TOML key mean
    // two things depending on `method.kind`, which the config-honesty
    // convention forbids.
    /// MP2-V range-separation length r₀ in **Å**. Shared by BOTH halves of
    /// MP2-V: it sets the MP2 attenuation operator AND the VV10 damping factor
    /// `1 − terfc(R, r₀)²` (paper Eq. 11 + p. 4161, "the r0 parameter is shared
    /// with the attenuated short-range MP2 part"). Default **1.00 Å**, the
    /// published MP2-V(terfc, aTZ) value (Table 1 RMSD minimum).
    ///
    /// `b` is NOT independently tunable from this — Table 1's valley runs
    /// (0.85, 8.0) → (1.10, 14.5). Move `mp2v_b` with it or you leave the
    /// fitted valley silently.
    pub mp2v_r0: Option<f64>,
    /// VV10 damping parameter `b`. Default **11.0** (Table 1, the r₀ = 1.00 Å
    /// row). See `mp2v_r0` — these two are correlated, not independent.
    pub mp2v_b: Option<f64>,
    /// VV10 long-range correlation parameter `C`. Default **0.0089** — the
    /// paper FIXED this at the LC-VV10 value rather than fitting it (§3), so
    /// changing it leaves the published parameterization entirely.
    pub mp2v_c: Option<f64>,
    /// Short-range attenuator on the MP2 correlation operator:
    ///
    ///   "terfc" (default) — the published operator (Dutoi/Goldey tempered
    ///                       erfc). Requires the interpolation tables
    ///                       (`FERRIC_TERF_TABLE_DIR`).
    ///   "erfc"            — `erfc(ωr)/r` with ω = 1/(r₀√2). Table-free
    ///                       CONTROL only; the fitted (r₀, b, C) do NOT
    ///                       transfer to it (different tail at matched r₀).
    ///
    /// Unknown values are a hard error.
    pub mp2v_attenuator: Option<String>,
    /// VV10 short-range damping:
    ///
    ///   "terfc" (default) — `1 − terfc(R, r₀)²`, the published Eq. 11 form,
    ///                       sharing `mp2v_r0`.
    ///   "none"            — bare (ωB97X-V-style) VV10. **NOT the published
    ///                       method**: it double-counts the short-range
    ///                       correlation attenuated MP2 already carries.
    ///                       Offered only so that double-counting is
    ///                       measurable.
    ///
    /// Unknown values are a hard error.
    pub mp2v_vv10_damping: Option<String>,
    /// Radial points in the VV10 nonlocal-correlation grid. Default 50 (ferric's
    /// own NLC grid shape, the same one `ferric_scf`'s KS drivers pass for
    /// wB97X-V). The paper used SG-1, which ferric does not have — a documented
    /// convention mismatch, not a silent one.
    pub mp2v_nlc_n_radial: Option<usize>,
    /// Angular points in the VV10 nonlocal-correlation grid. Default 50.
    /// (Unpruned; `AtomicGridConfig::prune` is deliberately not exposed here
    /// because pruning hard-errors at `n_angular = 50`.)
    pub mp2v_nlc_n_angular: Option<usize>,
}

impl Mp2Cfg {
    /// Build the MP2-V (`method.kind = "mp2-v"`) library config from the
    /// `mp2v_*` keys, starting from the published MP2-V(terfc, aTZ)
    /// parameterization and overriding only what the TOML actually set.
    ///
    /// Every string knob parses strictly (unknown values are a hard error, per
    /// the config-honesty convention). `r0` is Å at the CLI boundary and
    /// converted to Bohr here, the same way `r0_bonded`/`r0_nonbonded` are —
    /// and when it is set, the VV10 damping's r₀ is moved with it via
    /// `from_r0_angstrom`, so the two halves of Eq. 11 cannot silently diverge.
    ///
    /// `frozen_core` and `memory_budget_bytes` come from the shared `[mp2]
    /// frozen_core` key and `[memory]`, matching every other MP2-family method.
    pub fn build_att_vv10_config(
        &self,
        budget_bytes: Option<usize>,
    ) -> Result<ferric_mp2::att_vv10::AttVv10Config, String> {
        use ferric_dft::grid::AtomicGridConfig;
        use ferric_dft::vv10::Vv10Damping;
        use ferric_mp2::att_vv10::{AttVv10Attenuator, AttVv10Config};

        // Start from the published parameterization: r0 = 1.00 A, b = 11.0,
        // C = 0.0089, terfc attenuator, terfc-damped VV10.
        let mut cfg = AttVv10Config::mp2_v_terfc_atz();

        cfg.attenuator = match self
            .mp2v_attenuator
            .as_deref()
            .map(|s| s.trim().to_ascii_lowercase())
        {
            None => cfg.attenuator,
            Some(ref s) if s == "terfc" => AttVv10Attenuator::Terfc,
            Some(ref s) if s == "erfc" => AttVv10Attenuator::Erfc,
            Some(other) => {
                return Err(format!(
                    "[mp2] mp2v_attenuator: unknown value \"{other}\"; expected \"terfc\" (published) or \"erfc\" (control)"
                ))
            }
        };

        // Set the damping BEFORE r0, so `from_r0_angstrom` (which only syncs a
        // Terfc damping) sees the final variant.
        cfg.vv10_damping = match self
            .mp2v_vv10_damping
            .as_deref()
            .map(|s| s.trim().to_ascii_lowercase())
        {
            None => cfg.vv10_damping,
            Some(ref s) if s == "terfc" => Vv10Damping::Terfc {
                r0_bohr: cfg.r0_bohr,
            },
            Some(ref s) if s == "none" => Vv10Damping::None,
            Some(other) => {
                return Err(format!(
                    "[mp2] mp2v_vv10_damping: unknown value \"{other}\"; expected \"terfc\" (published, Eq. 11) or \"none\" (bare VV10, double-counts short range)"
                ))
            }
        };

        if let Some(r0_ang) = self.mp2v_r0 {
            // `is_finite()` already rejects NaN/inf, so a plain `<= 0.0`
            // suffices here and reads better than the negated comparison.
            if !r0_ang.is_finite() || r0_ang <= 0.0 {
                return Err(format!(
                    "[mp2] mp2v_r0 must be finite and > 0 (got {r0_ang} A)"
                ));
            }
            // Keeps the VV10 damping r0 in lockstep with the MP2 r0.
            cfg = cfg.from_r0_angstrom(r0_ang);
        }
        if let Some(b) = self.mp2v_b {
            if !b.is_finite() {
                return Err(format!("[mp2] mp2v_b must be finite (got {b})"));
            }
            cfg.vv10.b = b;
        }
        if let Some(c) = self.mp2v_c {
            if !c.is_finite() {
                return Err(format!("[mp2] mp2v_c must be finite (got {c})"));
            }
            cfg.vv10.c = c;
        }

        let n_radial = self.mp2v_nlc_n_radial.unwrap_or(cfg.nlc_grid.n_radial);
        let n_angular = self.mp2v_nlc_n_angular.unwrap_or(cfg.nlc_grid.n_angular);
        if n_radial == 0 || n_angular == 0 {
            return Err(format!(
                "[mp2] mp2v_nlc_n_radial/mp2v_nlc_n_angular must be > 0 (got {n_radial}x{n_angular})"
            ));
        }
        cfg.nlc_grid = AtomicGridConfig {
            n_radial,
            n_angular,
            // Deliberately unpruned: `PruneScheme::NwchemLike` hard-errors at
            // n_angular = 50, which is this grid's default.
            prune: None,
        };

        cfg.frozen_core = self.frozen_core;
        cfg.memory_budget_bytes = budget_bytes;
        Ok(cfg)
    }
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
    /// Compute and include Mulliken atomic charges (the standard textbook
    /// population analysis, D@S diagonal) in the NPZ bundle as
    /// `mulliken_charges`, shape (natoms,), float64, e. More basis-set-
    /// sensitive than Löwdin — included as the standard baseline every QC
    /// package provides, not a recommended charge scheme.
    /// Default: true when `export_npz` is set.
    pub compute_mulliken_charges: Option<bool>,
    /// Compute and include CHELPG atomic charges (units of e) in the NPZ
    /// bundle as `chelpg_charges`, shape (natoms,), float64. Structurally
    /// different from Hirshfeld/Löwdin/Mulliken: an ESP-FITTED scheme (atom-
    /// centered point charges chosen to best reproduce the molecular
    /// electrostatic potential on a grid around the molecule), not a
    /// population partition. Standard scheme for force-field electrostatics.
    /// Default: true when `export_npz` is set.
    pub compute_chelpg_charges: Option<bool>,
    /// Compute and include RESP atomic charges (units of e) in the NPZ
    /// bundle as `resp_charges`, shape (natoms,), float64. Same ESP grid-fit
    /// as CHELPG plus a hyperbolic restraint damping non-hydrogen charges
    /// toward zero (single-stage restrained fit — not full multi-stage/
    /// multi-conformer RESP averaging). Default: true when `export_npz` is set.
    pub compute_resp_charges: Option<bool>,
    /// Compute per-atom anisotropic C6 dispersion coefficients and include them
    /// in the NPZ bundle (`c6_iso`, `c6_aniso`, `alpha_atomic_dynamic`,
    /// `c6_freqs`, `c6_weights`). Default: true when `export_npz` is set.
    pub compute_c6: Option<bool>,
    /// C6 polarizability source. One of:
    ///   "ts"   — Tkatchenko-Scheffler single-pole model (default)
    ///   "pdep" — true PDEP-RPA dynamic α(iω) on the RPA quadrature grid
    ///   "mbd"  — many-body dispersion (coupled-dipole) on top of the TS α.
    ///            Post-G8 (live free-atom SCF, no hardcoded vol_free table),
    ///            MBD screening modestly improves TS's worst case (SiH4 at
    ///            aug-cc-pVTZ: TS +28.7% vs DOSD, MBD +24.3%) rather than
    ///            making it worse — see docs/VALIDATION.md's "MBD@TS
    ///            screening" row for the full 10-molecule resweep. Residual
    ///            error remains real: screening a still-imperfect free-atom
    ///            reference for soft covalent atoms can't fully repair it.
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
    /// Scissor shift (Hartree) added to every virtual orbital energy before
    /// assembling the RPAx@KS diagonal. Only consumed by
    /// `method.kind = "tdhf-static-polarizability"`
    /// (`ferric_gw::bse::run_rpax_static_polarizability`'s `scissor` arg) — a
    /// cheap proxy for widening a KS gap toward a GW-level gap. Unset → 0.0
    /// (plain KS). Ignored by the other `[gw]`-consuming method kinds.
    pub scissor: Option<f64>,
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
    /// DIIS family: "pulay" (default), "adiis", or "ediis". ADIIS/EDIIS use an
    /// energy-based extrapolation in the early SCF (switching to Pulay near
    /// convergence) — a convergence aid for hard transition-metal cases.
    pub diis: Option<String>,
    /// Crossover `err_max` below which ADIIS/EDIIS revert to plain Pulay
    /// (ignored for "pulay"). Default 1e-1.
    pub diis_switch_thresh: Option<f64>,
    /// Finite-temperature Fermi-Dirac occupation smearing width σ = k_B·T in
    /// Hartree. Absent/None = integer occupation (default). A convergence aid
    /// for near-degenerate frontier manifolds (metals / TM dimers).
    pub smearing_sigma: Option<f64>,
    /// SCF initial guess: "minao" (default, no per-element free-atom SCF for
    /// heavy atoms), "sad" (legacy free-atom-SCF superposition), or "hcore".
    pub guess: Option<String>,
    /// Enable the closed-shell second-order (Newton/SOSCF) step in the SCF tail
    /// (sets newton_trigger). Default false.
    #[serde(default)]
    pub soscf: bool,
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
    /// Print one line per SCF iteration to stdout while the job runs (energy,
    /// ΔE, density/DIIS error) — live progress for a long-running job. Default
    /// `false` (unchanged, silent-until-done output). The CLI's `--verbose`/
    /// `-v` flag ORs into this, so either the TOML key or the flag turns it on;
    /// this key lets a queued/batch job opt in without changing the invocation
    /// command. See `ferric_scf::rhf::RhfConfig::verbose`.
    #[serde(default)]
    pub verbose: bool,
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
            diis: None,
            diis_switch_thresh: None,
            smearing_sigma: None,
            guess: None,
            soscf: false,
            integral_thresh: 1e-12,
            k_builder: None,
            df_j_aux: None,
            df_k_aux: None,
            level_shift: None,
            mom_after_iter: 0,
            ladder: Vec::new(),
            verbose: false,
        }
    }
}

impl ScfCfg {
    /// Parse the `diis` string into a `DiisFlavor` (strict — unknown values are a
    /// hard error, per the config-honesty convention). Absent = Pulay.
    pub fn diis_flavor(&self) -> ferric_scf::diis::DiisFlavor {
        use ferric_scf::diis::DiisFlavor;
        match self.diis.as_deref() {
            None | Some("pulay") | Some("Pulay") => DiisFlavor::Pulay,
            Some("adiis") | Some("ADIIS") => DiisFlavor::Adiis,
            Some("ediis") | Some("EDIIS") => DiisFlavor::Ediis,
            Some(other) => panic!(
                "[scf] diis = \"{other}\" is not recognized (use \"pulay\", \"adiis\", or \"ediis\")"
            ),
        }
    }
    /// Whether the guess is "sad" (legacy free-atom-SCF) vs the default MINAO.
    /// Returns `use_sad_guess`-style: true means run the density-superposition
    /// guess (MINAO or SAD via use_sad_guess), false forces hcore.
    pub fn use_density_guess(&self) -> bool {
        !matches!(self.guess.as_deref(), Some("hcore") | Some("Hcore"))
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
            // are honored -- a plain `kind = "rhf"`/`kind = "ksdft"` run with
            // no [[scf.ladder]] table must not silently discard the [scf]
            // block.
            //
            // Dispatch on whether `base` carries a functional: `default_ladder_from`
            // hard-codes DF-JK aux unconditionally and does not honor the
            // caller's own `max_iter` on rung 0 (always 60) -- correct for
            // pure-HF heavy-atom divergence, but it starves a hybrid/GGA
            // KS-DFT run of its rung-0 iteration budget (measured: benzene/
            // def2-SVP DF-B3LYP walks the whole 5-rung ladder to MaxIter
            // instead of converging on rung 0 the way `ksdft_ladder` does in
            // ~8s -- see docs/profiles-2026-07-14.md's 2026-07-19 correction
            // note). `ksdft_ladder` is the KS-DFT-specific sibling: it starts
            // rung 0 from the caller's own level_shift/max_iter, only
            // auto-defaults DF-JK aux when `xc.is_some()`, and carries the
            // DFT grid through every rung. `ferric-python`'s run_dft/run_ksdft
            // paths already call `ksdft_ladder` directly (lib.rs) -- this
            // brings the CLI's `ksdft` path in line with that, instead of
            // silently falling through to the HF-tuned ladder.
            return if base.xc.is_some() {
                ferric_scf::ladder::ksdft_ladder(base)
            } else {
                ferric_scf::ladder::default_ladder_from(base)
            };
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
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read config file {path:?}: {e}"))?;
    toml::from_str(&text).map_err(|e| format!("{path}: {e}"))
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

    /// `method.kind = "mp2-v"`: the `mp2v_*` keys parse and the builder starts
    /// from the published MP2-V(terfc, aTZ) parameterization.
    #[test]
    fn test_parse_mp2_v_config() {
        let toml_str = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "aug-cc-pvtz"
[method]
kind = "mp2-v"
[mp2]
auxbasis = "aug-cc-pvtz-rifit"
frozen_core = 1
mp2v_r0 = 1.00
mp2v_b = 11.0
mp2v_c = 0.0089
mp2v_attenuator = "terfc"
mp2v_vv10_damping = "terfc"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.method.kind, "mp2-v");
        assert_eq!(cfg.mp2.mp2v_r0, Some(1.00));
        assert_eq!(cfg.mp2.mp2v_b, Some(11.0));
        assert_eq!(cfg.mp2.mp2v_c, Some(0.0089));

        let att = cfg.mp2.build_att_vv10_config(None).unwrap();
        assert!((att.r0_angstrom() - 1.00).abs() < 1e-12);
        // 1.00 A = 1.8897259886 Bohr; ~0.529 would mean the conversion inverted.
        assert!((att.r0_bohr - 1.889_725_988_6).abs() < 1e-9, "got {}", att.r0_bohr);
        assert_eq!(att.vv10.b, 11.0);
        assert_eq!(att.vv10.c, 0.0089);
        assert_eq!(att.frozen_core, 1, "[mp2] frozen_core must thread through");
        assert_eq!(att.attenuator, ferric_mp2::att_vv10::AttVv10Attenuator::Terfc);
        // Eq. 11: the VV10 damping r0 MUST be the same r0 the MP2 half uses.
        match att.vv10_damping {
            ferric_dft::vv10::Vv10Damping::Terfc { r0_bohr } => {
                assert_eq!(r0_bohr, att.r0_bohr)
            }
            other => panic!("MP2-V must damp VV10, got {other:?}"),
        }
    }

    /// An absent `[mp2]` section must give exactly the published parameters —
    /// the CLI default IS `AttVv10Config::mp2_v_terfc_atz()`, not a re-typed
    /// copy of it that could drift from the library.
    #[test]
    fn mp2_v_defaults_are_the_published_parameters() {
        let published = ferric_mp2::att_vv10::AttVv10Config::mp2_v_terfc_atz();
        let att = Mp2Cfg::default().build_att_vv10_config(None).unwrap();
        assert_eq!(att.r0_bohr, published.r0_bohr);
        assert_eq!(att.vv10.b, published.vv10.b);
        assert_eq!(att.vv10.c, published.vv10.c);
        assert_eq!(att.attenuator, published.attenuator);
        assert_eq!(att.nlc_grid.n_radial, published.nlc_grid.n_radial);
        assert_eq!(att.nlc_grid.n_angular, published.nlc_grid.n_angular);
        assert!(matches!(
            att.vv10_damping,
            ferric_dft::vv10::Vv10Damping::Terfc { .. }
        ));
    }

    /// Setting `mp2v_r0` alone must move the VV10 damping r0 with it. If these
    /// desync, the two halves of Eq. 11 silently use different range
    /// separations — a wrong number that still looks plausible.
    #[test]
    fn mp2_v_r0_override_syncs_the_vv10_damping() {
        let mut mp2 = Mp2Cfg::default();
        mp2.mp2v_r0 = Some(1.05);
        mp2.mp2v_b = Some(12.5); // the Table 1 valley partner for r0 = 1.05
        let att = mp2.build_att_vv10_config(None).unwrap();
        assert!((att.r0_angstrom() - 1.05).abs() < 1e-12);
        assert_eq!(att.vv10.b, 12.5);
        match att.vv10_damping {
            ferric_dft::vv10::Vv10Damping::Terfc { r0_bohr } => assert_eq!(
                r0_bohr, att.r0_bohr,
                "damping r0 must follow mp2v_r0 (paper Eq. 11)"
            ),
            other => panic!("expected terfc damping, got {other:?}"),
        }
    }

    /// The two string knobs parse strictly and the `erfc`/`none` (control)
    /// variants are reachable.
    #[test]
    fn mp2_v_string_knobs_parse_strictly() {
        let mk = |att: Option<&str>, damp: Option<&str>| {
            let mut m = Mp2Cfg::default();
            m.mp2v_attenuator = att.map(|s| s.to_string());
            m.mp2v_vv10_damping = damp.map(|s| s.to_string());
            m.build_att_vv10_config(None)
        };
        use ferric_mp2::att_vv10::AttVv10Attenuator;
        assert_eq!(
            mk(Some("erfc"), None).unwrap().attenuator,
            AttVv10Attenuator::Erfc
        );
        // Case-insensitive + whitespace-tolerant, like the other string knobs.
        assert_eq!(
            mk(Some("  TERFC "), None).unwrap().attenuator,
            AttVv10Attenuator::Terfc
        );
        assert!(matches!(
            mk(None, Some("none")).unwrap().vv10_damping,
            ferric_dft::vv10::Vv10Damping::None
        ));
        // Unknown values are a hard error, never a silent default.
        let e = mk(Some("terf"), None).unwrap_err();
        assert!(e.contains("terf"), "error should name the bad value: {e}");
        let e = mk(None, Some("vv10")).unwrap_err();
        assert!(e.contains("vv10"), "error should name the bad value: {e}");
    }

    /// A nonpositive/non-finite r0 must be refused at the CLI boundary rather
    /// than propagated into a 1/(r0*sqrt(2)) division.
    #[test]
    fn mp2_v_bad_r0_and_grid_are_rejected() {
        for bad in [0.0_f64, -1.0, f64::NAN, f64::INFINITY] {
            let mut m = Mp2Cfg::default();
            m.mp2v_r0 = Some(bad);
            assert!(
                m.build_att_vv10_config(None).is_err(),
                "mp2v_r0 = {bad} must be rejected"
            );
        }
        let mut m = Mp2Cfg::default();
        m.mp2v_nlc_n_radial = Some(0);
        assert!(m.build_att_vv10_config(None).is_err());
    }

    /// Typo'd `mp2v_*` keys must hard-error (deny_unknown_fields), not silently
    /// run at the published defaults while the user thinks they changed b.
    #[test]
    fn mp2_v_typod_key_is_rejected() {
        let toml_str = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "mp2-v"
[mp2]
mp2v_bb = 11.0
"#;
        let err = match toml::from_str::<Config>(toml_str) {
            Ok(_) => panic!("typo'd mp2v key parsed successfully"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("mp2v_bb"), "error should name the bad key: {err}");
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
    fn test_parse_bse_tda_config() {
        // "bse-tda" reuses [rpa] + [gw] verbatim (no new TOML section) --
        // confirm both sections still parse and thread through with
        // method.kind = "bse-tda".
        let toml_str = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "bse-tda"
[rpa]
auxbasis = "cc-pvdz-ri"
n_quad = 16
quadrature = "gauss-legendre"
trunc_thresh = 0.0
eigensolver_conv_thresh = 1e-7
[gw]
frozen_core = 0
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.method.kind, "bse-tda");
        assert_eq!(cfg.rpa.auxbasis.as_deref(), Some("cc-pvdz-ri"));
        assert_eq!(cfg.rpa.n_quad, Some(16));
        assert_eq!(cfg.gw.frozen_core, Some(0));
    }

    #[test]
    fn test_parse_bse_tda_config_defaults() {
        // Empty [rpa]/[gw] sections (or absent entirely) must still parse.
        let toml_str = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "bse-tda"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.method.kind, "bse-tda");
        assert_eq!(cfg.gw.frozen_core, None);
    }

    #[test]
    fn test_parse_tdhf_static_polarizability_config() {
        // "tdhf-static-polarizability" reuses [rpa] + [gw] verbatim (no new
        // TOML section), same pattern as "bse-tda". Requires [rpa].xc (a KS
        // reference) -- confirm both sections + xc + scissor parse and thread
        // through with method.kind = "tdhf-static-polarizability".
        let toml_str = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "tdhf-static-polarizability"
[rpa]
auxbasis = "cc-pvdz-ri"
n_quad = 16
quadrature = "gauss-legendre"
trunc_thresh = 0.0
eigensolver_conv_thresh = 1e-7
xc = "PBE"
[gw]
frozen_core = 0
scissor = 0.1
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.method.kind, "tdhf-static-polarizability");
        assert_eq!(cfg.rpa.auxbasis.as_deref(), Some("cc-pvdz-ri"));
        assert_eq!(cfg.rpa.xc.as_deref(), Some("PBE"));
        assert_eq!(cfg.gw.frozen_core, Some(0));
        assert!((cfg.gw.scissor.unwrap() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_parse_tdhf_static_polarizability_config_defaults() {
        // Empty [rpa]/[gw] sections (or absent entirely) must still parse.
        let toml_str = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "tdhf-static-polarizability"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.method.kind, "tdhf-static-polarizability");
        assert_eq!(cfg.gw.scissor, None);
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

    /// Regression: a plain `kind = "ksdft"` run with no `[[scf.ladder]]` table
    /// must escalate via `ksdft_ladder`, NOT `default_ladder_from`. Before this
    /// fix, `build_ladder`'s empty-ladder fallback always called
    /// `default_ladder_from` regardless of whether `base.xc` was set --
    /// `default_ladder_from` hard-codes rung 0's `max_iter` to 60 (ignoring the
    /// caller's own budget) and starves a hybrid/GGA KS-DFT SCF of the
    /// iterations `ksdft_ladder`'s rung 0 gets. Measured effect: benzene/
    /// def2-SVP DF-B3LYP walked the whole 5-rung `default_ladder_from` escalation
    /// to `MaxIter` instead of converging on rung 0 in ~10 iterations the way
    /// `ksdft_ladder` does (docs/profiles-2026-07-14.md 2026-07-19 correction).
    #[test]
    fn empty_ladder_ksdft_base_dispatches_to_ksdft_ladder() {
        let base = ferric_scf::rhf::RhfConfig {
            xc: Some("B3LYP".to_string()),
            max_iter: 100,
            ..Default::default()
        };
        let cfg = ScfCfg::default();
        assert!(cfg.ladder.is_empty());
        let built = cfg.build_ladder(&base);
        let expected = ferric_scf::ladder::ksdft_ladder(&base);
        assert_eq!(built.len(), expected.len());
        // ksdft_ladder's rung 0 honors the caller's own max_iter (100 here);
        // default_ladder_from would clamp rung 0 to 60 regardless.
        assert_eq!(built[0].config.max_iter, 100,
            "ksdft rung 0 must honor the caller's own max_iter, not default_ladder_from's hardcoded 60");
        for (i, rung) in built.iter().enumerate() {
            assert_eq!(rung.config.xc.as_deref(), Some("B3LYP"), "rung {i} must carry xc");
        }
    }

    #[test]
    fn empty_ladder_default_escalation_honors_base_scf_block() {
        // Regression for I1: a plain `kind = "rhf"` run with no [[scf.ladder]]
        // table must NOT silently discard the user's [scf] settings by
        // building every rung from RhfConfig::default(). Seed `base` with a
        // mom_after_iter value that differs from RhfConfig's default and
        // assert every rung in the empty-ladder escalation carries it
        // through. max_iter is NOT checked per-rung here: the real ladder
        // (ferric_scf::ladder::default_ladder_from) deliberately escalates
        // max_iter per rung (60/60/60/80/100), so a user's flat [scf]
        // max_iter is a starting point each rung's own budget overrides, not
        // a value every rung inherits unchanged.
        let base = ferric_scf::rhf::RhfConfig {
            mom_after_iter: 5,
            ..Default::default()
        };
        let cfg = ScfCfg::default();
        assert!(cfg.ladder.is_empty());
        let built = cfg.build_ladder(&base);
        assert_eq!(built.len(), ferric_scf::ladder::default_ladder().len());
        for (i, rung) in built.iter().enumerate() {
            assert_eq!(rung.config.mom_after_iter, 5, "rung {i} must inherit base.mom_after_iter");
            // RI-JK is opt-in: the ladder escalates convergence knobs only and
            // must never silently swap exact 4-index J/K for density fitting
            // (that changes the method, ~1e-4 Ha). The base here leaves the aux
            // unset, so every rung must too.
            assert!(rung.config.df_j_aux.is_none(), "rung {i} must not inject DF-J aux");
            assert!(rung.config.df_k_aux.is_none(), "rung {i} must not inject DF-K aux");
        }
        assert_eq!(built[0].config.level_shift, 0.0);
        assert_eq!(built[1].config.level_shift, 0.0, "rung 1 adds ADIIS, not level shift yet");
        assert!(built[2].config.level_shift > 0.0, "rung 2 must add level shift");
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
    fn cosmo_section_parses() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "rhf"
task = "energy"

[cosmo]
epsilon = 78.39
radius_scale = 1.17
lebedev_order = 110
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        let cosmo = cfg.cosmo.expect("cosmo section should parse to Some");
        assert_eq!(cosmo.epsilon, 78.39);
        assert_eq!(cosmo.radius_scale, 1.17);
        assert_eq!(cosmo.lebedev_order, 110);
    }

    #[test]
    fn cosmo_section_defaults_when_partially_specified() {
        // radius_scale/lebedev_order have serde defaults; only epsilon is
        // effectively required (no #[serde(default)] on it — an omitted
        // epsilon is a real user error, not a silently-defaulted value).
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rhf"

[cosmo]
epsilon = 78.39
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        let cosmo = cfg.cosmo.unwrap();
        assert_eq!(cosmo.epsilon, 78.39);
        assert_eq!(cosmo.radius_scale, ferric_scf::cosmo::DEFAULT_RADIUS_SCALE);
        assert_eq!(cosmo.lebedev_order, ferric_scf::cosmo::DEFAULT_LEBEDEV_ORDER);
    }

    #[test]
    fn cosmo_section_optional_defaults_to_none() {
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
        assert!(cfg.cosmo.is_none());
    }

    #[test]
    fn cosmo_section_rejects_typo_key() {
        // deny_unknown_fields on CosmoConfig: a typo'd key must hard-error,
        // never silently no-op (config-honesty convention).
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rhf"

[cosmo]
epsilonn = 78.39
"#;
        let result: Result<Config, _> = toml::from_str(toml_str);
        assert!(result.is_err(), "typo'd cosmo key should fail to parse, not silently default");
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
