use ferric_core::mol::Molecule;
use serde::Deserialize;

/// Correlation (RI) auxiliary basis used when `[mp2] auxbasis` / `[rpa]
/// auxbasis` is omitted. Named here, not as a literal at each use site, so
/// `default_aux_bases_resolve` can check that the defaults a run falls back to
/// are actually bundled -- the TDDFT default below was not, for as long as it
/// existed, and no test could see it because it lived only in lib.rs.
pub const DEFAULT_CORRELATION_AUX: &str = "cc-pvdz-ri";
/// RI auxiliary basis for `method.kind = "tda" | "tddft"` when `[mp2]
/// auxbasis` is omitted. Same data as [`DEFAULT_CORRELATION_AUX`] (the BSE
/// name for it); kept as its own constant so this default is unchanged.
pub const TDDFT_DEFAULT_AUX: &str = "cc-pvdz-rifit";
/// JK-fit auxiliary basis the CLI defaults RI-J/RI-K to for KS-DFT and the
/// RPA/GW/TDDFT references.
pub const DEFAULT_SCF_JK_AUX: &str = "def2-universal-jkfit";

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
    pub frequencies: FrequenciesCfg,
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
    #[serde(default)]
    pub tddft: TddftCfg,
    /// Optional `[output]` section: where the machine-readable JSON run log
    /// goes. Absent means the default (a `.ferric.jsonl` beside the input
    /// file) -- logging is ON BY DEFAULT, see [`OutputCfg`].
    #[serde(default)]
    pub output: OutputCfg,
    /// Optional `[qmmm]` section: QM/MM embedding. Absent means no QM/MM --
    /// byte-identical to the plain single-region run, the same convention
    /// `[cosmo]` and `[external_potential]` follow.
    #[serde(default)]
    pub qmmm: Option<QmmmCfg>,
}

impl Config {
    /// Every `frozen_core` key in the file, as `(section name, spec)`.
    ///
    /// `[gw] frozen_core` is reported only when it was actually written: unset
    /// means "follow `[rpa]`", and reporting it as a second, independent 0
    /// would put a key in the audit line that the user never typed.
    fn frozen_core_keys(&self) -> Vec<(&'static str, FrozenCore)> {
        let mut keys = vec![
            ("[mp2]", self.mp2.frozen_core),
            ("[rpa]", self.rpa.frozen_core),
        ];
        if let Some(fc) = self.gw.frozen_core {
            keys.push(("[gw]", fc));
        }
        keys
    }

    /// Reject a frozen core that would leave nothing to correlate, naming the
    /// section that set it.
    ///
    /// This is the same condition `ferric_mp2::rimp2::active_occ` enforces deep
    /// in the correlation kernels; catching it here turns "SCF converged, then
    /// a bare error 40 seconds later" into an error before any integral is
    /// computed, and lets the message name the TOML key (and whether the count
    /// came from `"auto"`) instead of just a number.
    ///
    /// The bound is the **minority-spin** occupied count, which is the binding
    /// one: freezing `n` orbitals freezes them in both spin channels, so an
    /// open-shell system runs out of β occupieds first. For a closed-shell
    /// molecule the two counts coincide.
    ///
    /// `frozen_core = 0` is always accepted (there is nothing to freeze, so
    /// nothing can be over-frozen) — including for a molecule with no
    /// electrons at all, which `n_frozen > 0` below would otherwise reject
    /// with a confusing message.
    ///
    /// Every section is checked, not just the one the selected `method.kind`
    /// consumes: a `frozen_core` that cannot be satisfied for this molecule is
    /// a broken input whichever method runs, and a `[mp2]` key that is wrong
    /// but silent today is a wrong number the day someone switches
    /// `method.kind` to an MP2 variant.
    ///
    /// Unlike the checks in [`load_config`], this one needs the molecule, and
    /// specifically the molecule after [`Molecule::apply_ecp`] — so it lives
    /// here and is called from the CLI once both are in hand.
    pub fn validate_frozen_core(&self, mol: &Molecule) -> Result<(), String> {
        // n_beta = (nelec - (multiplicity - 1)) / 2. Molecule construction has
        // already validated that this is a non-negative integer.
        let two_s = mol.multiplicity as i32 - 1;
        let n_occ_minority = ((mol.nelec() - two_s) / 2).max(0) as usize;
        for (section, spec) in self.frozen_core_keys() {
            let n_frozen = spec.resolve(mol);
            if n_frozen > 0 && n_frozen >= n_occ_minority {
                let source = if spec.is_auto() {
                    " (from frozen_core = \"auto\")"
                } else {
                    ""
                };
                return Err(format!(
                    "{section} frozen_core = {n_frozen}{source} freezes all {n_occ_minority} \
                     occupied orbital(s) of the minority spin — nothing left to correlate"
                ));
            }
        }
        Ok(())
    }

    /// One `[ferric] ...` line per `frozen_core = "auto"` key, reporting the
    /// number the convention picked.
    ///
    /// An auto count is a number the user did not write down, and it moves
    /// with the molecule and the basis (an ECP changes it). Printing it keeps a
    /// run's correlation space auditable from its log alone, the same way the
    /// memory budget's resolution is. Explicit counts print nothing — they are
    /// already in the input file.
    pub fn frozen_core_audit_lines(&self, mol: &Molecule) -> Vec<String> {
        self.frozen_core_keys()
            .into_iter()
            .filter(|(_, spec)| spec.is_auto())
            .map(|(section, spec)| {
                format!(
                    "frozen core: {} orbital(s) frozen from {section} frozen_core = \"auto\"",
                    spec.resolve(mol)
                )
            })
            .collect()
    }
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
    ///
    /// Callers should run [`MemoryCfg::validate`] first — an unusable figure
    /// reaches `gib_to_bytes` as `0` here, which `resolve_budget` treats as
    /// "unset". [`MemoryCfg::validate`]'s doc explains why that is an inversion
    /// rather than a degradation.
    pub fn budget_bytes(&self) -> Option<usize> {
        self.budget_gb().map(ferric_core::memory::gib_to_bytes)
    }

    /// Reject a `budget_gb` that cannot mean what the user wrote.
    ///
    /// # The inversion this prevents
    ///
    /// `gib_to_bytes` maps NaN and any non-positive input to `0`, and truncates
    /// a positive-but-sub-byte figure to `0` as well. `resolve_budget` then
    /// documents `0` as "unset" (`if b > 0`) and falls through to
    /// `0.8 × detect_available_bytes()`.
    ///
    /// So `budget_gb = -4.0` (a typo'd sign), `budget_gb = 0.0` ("use no extra
    /// memory") and `budget_gb = 1e-12` each ask for the tightest possible
    /// ceiling and receive the LOOSEST one available — 80% of the whole box.
    /// That reaches every method, because `budget_bytes()` is the single value
    /// threaded into all of them, and the only trace was the audit line saying
    /// `[source: auto (0.8 × available RAM)]` where the user expected
    /// `explicit`.
    ///
    /// # Why erroring is this config's own convention
    ///
    /// Every config struct here carries `#[serde(deny_unknown_fields)]`, so a
    /// typo'd KEY is already fatal, and the string knobs
    /// (`QuadratureScheme`/`C6Source`/`DispersionPartition::parse_config_str`)
    /// all hard-error on an unknown VALUE rather than silently defaulting. The
    /// one knob that bounds memory should not be the exception — least of all
    /// when its silent-default direction is "ignore the limit entirely".
    ///
    /// # Scope
    ///
    /// The check is on the RESOLVED BYTE COUNT, not the sign of the input: a
    /// sign-only test would accept `1e-12`, which truncates to zero and inverts
    /// identically. It covers the deprecated `three_index_budget_gb` alias too,
    /// since that feeds the same [`MemoryCfg::budget_gb`] accessor.
    ///
    /// An ABSENT budget stays valid and still means auto-detect — the bug is
    /// only that an unusable PRESENT one was indistinguishable from absent.
    ///
    /// Inert on every currently-valid config: any positive figure that maps to
    /// a nonzero byte count resolves to the identical value it did before. Only
    /// inputs that today silently mean "auto" begin to error.
    pub fn validate(&self) -> Result<(), String> {
        let Some(gb) = self.budget_gb() else {
            return Ok(()); // absent: auto-detect, as documented.
        };
        let which = if self.budget_gb.is_some() {
            "budget_gb"
        } else {
            "three_index_budget_gb (budget_gb)"
        };
        if !gb.is_finite() || gb <= 0.0 {
            return Err(format!(
                "[memory] {which} = {gb} is not a usable budget: it must be finite and > 0. \
                 A non-positive or NaN value resolves to 0 bytes, which ferric treats as \
                 \"unset\" and replaces with 0.8 x available RAM — the opposite of what you \
                 asked for. Remove the key to request auto-detection explicitly."
            ));
        }
        if ferric_core::memory::gib_to_bytes(gb) == 0 {
            return Err(format!(
                "[memory] {which} = {gb} rounds to 0 bytes, which ferric treats as \"unset\" \
                 and replaces with 0.8 x available RAM — the opposite of what you asked for. \
                 Use a value of at least 1e-9 GiB, or remove the key to request auto-detection."
            ));
        }
        Ok(())
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
    /// Angular-grid pruning scheme for the MAIN DFT grid.
    ///
    ///   `"none"` (default) — flat grid, every radial shell at the full
    ///                        Lebedev order. Byte-identical to the historical
    ///                        behaviour.
    ///   `"nwchem"`         — NWChem-style 5-region radial pruning. Removes
    ///                        ~23% of grid points at the default 75x110 for a
    ///                        live-SCF energy shift well inside the PySCF
    ///                        reference tolerances (see
    ///                        `ferric-dft/tests/grid_prune_live_scf.rs`).
    ///
    /// Unknown values are a hard error (`PruneScheme::parse_config_str`).
    ///
    /// Applies to the main grid ONLY — the VV10/NLC grid stays unpruned,
    /// because pruning has no valid table at its 50x50 angular order.
    ///
    /// Energy runs only: `method.task = "optimize"` / `"frequencies"` go
    /// through the XC gradient's grid-response path, which is built for the
    /// unpruned grid and hard-errors on a pruned one.
    pub grid_prune: Option<String>,
    /// Empirical dispersion correction to ADD to the SCF energy.
    ///
    ///   omitted / absent  — no correction. The reported energy is the plain
    ///                       KS-DFT energy, exactly as before this key existed.
    ///   `"d3bj"`          — Grimme D3(BJ), using the damping parameters
    ///                       published for `functional`.
    ///   `"d3bj(<name>)"`  — D3(BJ) using `<name>`'s published parameters
    ///                       instead, for when ferric's XC name and the D3
    ///                       fit's name differ (e.g. a libxc spelling).
    ///
    /// Unknown values are a hard error, and so is a functional with no
    /// published D3(BJ) fit: the correction is FITTED per functional, so
    /// substituting another one's parameters would silently change the answer.
    /// There is deliberately no "off" value that reports a 0.0 correction --
    /// absent means absent.
    pub dispersion: Option<String>,
}

/// What `[dft] dispersion` asked for, after strict parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum DispersionRequest {
    /// D3(BJ) with the named functional's published damping parameters.
    D3Bj { functional: String },
}

impl DispersionRequest {
    /// Parse the `[dft] dispersion` value.
    ///
    /// `xc` is the functional being run, used when the value does not name one
    /// explicitly. Strict by this config's convention: an unknown value is an
    /// error, never a silent no-op.
    pub fn parse_config_str(s: &str, xc: Option<&str>) -> Result<Self, ferric_core::FerricError> {
        let v = s.trim();
        let lower = v.to_ascii_lowercase();
        let named = |f: &str| -> Result<Self, ferric_core::FerricError> {
            Ok(DispersionRequest::D3Bj {
                functional: f.to_string(),
            })
        };
        if lower == "d3bj" || lower == "d3(bj)" {
            let f = xc.ok_or_else(|| {
                ferric_core::FerricError::General(
                    "[dft] dispersion = \"d3bj\" needs [dft] functional to know which \
                     damping parameters to use, or name one explicitly as \
                     \"d3bj(pbe)\"."
                        .to_string(),
                )
            })?;
            return named(f);
        }
        if let Some(rest) = lower
            .strip_prefix("d3bj(")
            .and_then(|r| r.strip_suffix(')'))
        {
            if rest.trim().is_empty() {
                return Err(ferric_core::FerricError::General(
                    "[dft] dispersion = \"d3bj()\" names no functional".to_string(),
                ));
            }
            return named(rest.trim());
        }
        Err(ferric_core::FerricError::General(format!(
            "unknown [dft] dispersion value {v:?}; expected \"d3bj\" or \
             \"d3bj(<functional>)\". Omit the key entirely for no dispersion \
             correction -- there is no value that means \"compute zero\"."
        )))
    }
}

/// One `[[external_potential.point_charges]]` entry: a fixed point charge
/// (units: e for `q`, Bohr for coordinates) contributing to the one-electron
/// Hamiltonian and nuclear-repulsion-like energy term.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct PointChargeCfg {
    pub q: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// The `[external_potential]` TOML section: an array of fixed point charges
/// plus an optional uniform external electric field (a.u.).
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ExternalPotentialCfg {
    #[serde(default)]
    pub point_charges: Vec<PointChargeCfg>,
    pub field: Option<[f64; 3]>,
}

impl ExternalPotentialCfg {
    /// Convert into the solver-facing type. Returns `None` when both
    /// `point_charges` is empty and `field` is unset (a true no-op,
    /// matching `RhfConfig.external_potential`'s `None` default).
    pub fn to_external_potential(
        &self,
    ) -> Option<ferric_core::external_potential::ExternalPotential> {
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
            smeared_charges: Vec::new(),
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
    /// Coordinate system for the BFGS search: `"cartesian"` (the default) or
    /// `"internal"` / `"redundant-internal"`.
    ///
    /// Parsed by [`parse_coord_system`], which is STRICT: an unrecognized value
    /// is an error, never a silent fall back to the default. A user who typo'd
    /// `"internals"` asked for internals and must be told they did not get
    /// them.
    pub coordinates: Option<String>,
}

/// Parse the `[optimize] coordinates` key into a
/// [`CoordSystem`](ferric_scf::optimize::CoordSystem).
///
/// Strict by design — see [`OptimizeCfg::coordinates`].
///
/// # Errors
///
/// Returns a message naming the accepted values if `s` is not one of them.
pub fn parse_coord_system(s: &str) -> Result<ferric_scf::optimize::CoordSystem, String> {
    use ferric_scf::optimize::CoordSystem;
    match s.trim().to_ascii_lowercase().as_str() {
        "cartesian" | "cart" => Ok(CoordSystem::Cartesian),
        "internal" | "internals" | "redundant-internal" | "redundant_internal" => {
            Ok(CoordSystem::RedundantInternal)
        }
        other => Err(format!(
            "unknown [optimize] coordinates = \"{other}\"; expected \"cartesian\" or \"internal\""
        )),
    }
}

/// `[frequencies]` — harmonic vibrational frequencies via finite difference of
/// the ANALYTIC gradient (`method.task = "frequencies"`).
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct FrequenciesCfg {
    /// Central-difference displacement in Bohr. Default `5e-4`.
    ///
    /// This is a genuine accuracy knob and the wrong value degrades silently:
    /// too large adds truncation error, too small amplifies SCF noise. The
    /// printed `Hessian asymmetry` is the diagnostic — it is zero in exact
    /// arithmetic, so a large value means `delta` or the SCF thresholds are
    /// badly chosen for the system.
    pub delta: Option<f64>,
}

/// The `frozen_core` key of a correlation section (`[mp2]`, `[rpa]`, `[gw]`):
/// either an explicit orbital count or `"auto"`.
///
/// ```toml
/// [mp2]
/// frozen_core = "auto"   # standard small-core count for THIS molecule
/// frozen_core = 3        # exactly three orbitals, whatever the molecule is
/// frozen_core = "none"   # correlate everything (the default, = 0)
/// ```
///
/// `true`/`false` are accepted as synonyms of `"auto"`/`"none"`, for anyone
/// coming from a program that spells the key `freeze_core = true`.
///
/// **Why a type rather than a `usize` the parser fills in.** `"auto"` cannot be
/// turned into a number without the molecule, which the parser does not have —
/// and the molecule is not final until the basis is loaded, because an ECP
/// changes the answer ([`Molecule::auto_frozen_core`]). Resolving on demand at
/// the use site, where `mol` is always in scope, means there is no window in
/// which a stale `0` can be read as "the user asked for no frozen core": an
/// unresolved value is not a number and does not compile into one.
///
/// Unknown spellings are a hard error, never a silent fall-through to 0 —
/// `frozen_core = "fc"` that quietly correlated the core would change published
/// energies with no diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrozenCore {
    /// `"auto"` (or `true`): freeze the standard small-core count for the
    /// molecule at hand, ECPs and ghost centers accounted for. See
    /// [`ferric_core::mol::core_orbitals`] for the convention and its
    /// deliberate exceptions (the 3d shell of Sc–Zn stays correlated).
    Auto,
    /// An explicit orbital count, exactly as written. `0` correlates
    /// everything.
    Count(usize),
}

impl Default for FrozenCore {
    /// `Count(0)`: correlate every occupied orbital.
    ///
    /// The default is deliberately NOT `Auto` — every published ferric number
    /// predating this key was computed all-electron, and flipping the default
    /// would silently change them. Frozen core is opt-in.
    fn default() -> Self {
        FrozenCore::Count(0)
    }
}

impl FrozenCore {
    /// Number of orbitals to freeze for `mol`.
    ///
    /// `mol` must be the calculation's molecule AFTER
    /// [`Molecule::apply_ecp`] — see that method for why an ECP changes the
    /// count.
    pub fn resolve(&self, mol: &Molecule) -> usize {
        match *self {
            FrozenCore::Auto => mol.auto_frozen_core(),
            FrozenCore::Count(n) => n,
        }
    }

    /// True when the count came from `"auto"` rather than the TOML naming a
    /// number. Only used to label the audit line the run prints — an auto
    /// count is a convention the user did not write down, so the run says
    /// which number it picked.
    pub fn is_auto(&self) -> bool {
        matches!(*self, FrozenCore::Auto)
    }
}

impl<'de> Deserialize<'de> for FrozenCore {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct FrozenCoreVisitor;

        impl serde::de::Visitor<'_> for FrozenCoreVisitor {
            type Value = FrozenCore;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a non-negative orbital count, \"auto\", or \"none\"")
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<FrozenCore, E> {
                usize::try_from(v)
                    .map(FrozenCore::Count)
                    .map_err(|_| E::custom(format!("frozen_core = {v} does not fit in a usize")))
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<FrozenCore, E> {
                usize::try_from(v).map(FrozenCore::Count).map_err(|_| {
                    E::custom(format!(
                        "frozen_core must be >= 0 (got {v}); use 0 or \"none\" to correlate \
                         every occupied orbital, or \"auto\" for the standard small core"
                    ))
                })
            }

            fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<FrozenCore, E> {
                // `freeze_core = true` is how several other programs spell it;
                // accept it rather than make the user guess which word we want.
                Ok(if v {
                    FrozenCore::Auto
                } else {
                    FrozenCore::Count(0)
                })
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<FrozenCore, E> {
                match v.trim().to_ascii_lowercase().as_str() {
                    "auto" => Ok(FrozenCore::Auto),
                    "none" => Ok(FrozenCore::Count(0)),
                    other => Err(E::custom(format!(
                        "frozen_core: unknown value \"{other}\"; expected \"auto\" (standard \
                         small core for this molecule), \"none\", or a non-negative integer \
                         orbital count"
                    ))),
                }
            }
        }

        deserializer.deserialize_any(FrozenCoreVisitor)
    }
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Mp2Cfg {
    pub auxbasis: Option<String>,
    /// Core orbitals excluded from the correlation treatment: an explicit
    /// count, or `"auto"` for the standard small-core count of this molecule.
    /// Default 0 (all-electron correlation). See [`FrozenCore`].
    ///
    /// Shared by the whole MP2 family AND by the CC/double-hybrid methods,
    /// which read this key rather than defining one of their own.
    #[serde(default)]
    pub frozen_core: FrozenCore,
    // NOTE: `orbital_optimize` used to live here behind `#[allow(dead_code)]`.
    // Nothing ever read it — orbital optimization is selected with
    // `method.kind = "oo-rimp2"`. Setting it did nothing, which silently gave
    // plain RI-MP2 to anyone who expected OO-RI-MP2. Removed rather than wired
    // up: `kind` is already the selector, and with `deny_unknown_fields` the
    // stale key now errors instead of lying.
    /// Range-separation parameter ω in Å⁻¹ (for att-rimp2 and rs-mp2-rpa). Default 0.420.
    pub omega: Option<f64>,
    /// Amplitude-threshold LMP2 (`kind = "lmp2"`): the single threshold ε on
    /// localized |(ia|jb)| (WSHG23 Eq. 8). Default 1e-4. The finite-ε energy
    /// is a controlled approximation — error one-sided and ~linear in ε (see
    /// wiki/amplitude-threshold-lmp2.md for the measured map).
    pub lmp2_eps: Option<f64>,
    /// Amplitude-threshold LMP2 (`kind = "lmp2"` and `"lmp2-direct"`): also
    /// compute the canonical RI-MP2 reference and print it with the error
    /// against it. Default false (OPT-IN): the reference is a full N^5
    /// canonical RI-MP2 that forms the global (naux, nocc·nvir) tensor —
    /// the very object `lmp2-direct` exists to avoid — so with it on no run
    /// is reduced-cost. Off, the printout says the reference was not
    /// computed and the run log's `e_corr_canonical_ri` is null. A bool:
    /// any other TOML type is a parse error.
    pub lmp2_reference: Option<bool>,
    /// Integral-direct LMP2 (`kind = "lmp2-direct"`): aux fit-domain radius
    /// in Bohr (pair (i,j) fits in aux functions within this radius of
    /// either Boys centroid). Default 10.0 — the measured production value
    /// (wiki/amplitude-threshold-lmp2.md §27-30); ≥1e5 ≈ global fit.
    pub direct_aux_radius: Option<f64>,
    /// Integral-direct LMP2: virtual domain radius in Bohr on dipole
    /// centroids. Default 12.0 (production); omit-able only by setting a
    /// huge value — every default here is a CONTROLLED approximation, and
    /// `lmp2_reference = true` prints the canonical reference error alongside.
    pub direct_virt_radius: Option<f64>,
    /// Integral-direct LMP2: AO-support shell threshold on max |C|.
    /// Default 1e-3 (production); 0.0 keeps every shell.
    pub direct_ao_tail: Option<f64>,
    /// Integral-direct LMP2: Cauchy–Schwarz triple cut √(P|P)·Q(μν) on the
    /// batch integral stream. Default 1e-5 (calibrated ~1e-8 Ha at C16);
    /// MUST be 0.0 for operators without Schwarz support (terfc) — the run
    /// hard-errors otherwise, naming this knob.
    pub direct_schwarz_skip: Option<f64>,
    /// Integral-direct LMP2: nearest-atom batches merged per integral pass.
    /// Default 4 (measured ~0.4× the evaluations of per-atom batches); 1 =
    /// per-atom (the anchor limit).
    pub direct_batch_merge: Option<usize>,
    /// Integral-direct LMP2: R⁻⁶ pair-gate calibration constant (p95:
    /// ~0.7 Coulomb, ~0.02 erfc ω=1). Omitted = gate OFF (keep all pairs).
    pub direct_gate_cal: Option<f64>,
    /// Integral-direct LMP2: ε-linked Schwarz virtual-candidate screen —
    /// keep a in C_ij iff q_ia·qmax_j ≥ κ·ε (either orientation), q from
    /// strip-local fitted diagonals. Omitted = OFF (the validated distance
    /// candidates alone). κ = 1 is conservative (measured escape-free at
    /// C8, both operators); larger κ trades bounded sub-dominant error for
    /// smaller pair blocks (WIKI-APPEND-eps-linked-maps.md).
    pub direct_virt_schwarz_kappa: Option<f64>,
    /// κ-regularized MP2 (Lee/Head-Gordon JCTC 2018) for `kind = "rimp2"`:
    /// damps every amplitude by (1 − e^{−κΔ})², κ in inverse Hartree
    /// (κ→∞ recovers plain MP2; the paper's recommended value is ~1.45).
    /// Omitted = plain MP2, byte-identical code path. Validated finite/>0
    /// at the library boundary.
    pub kappa: Option<f64>,
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
    ///   "ao-sparse"    — "ao" with the pseudo-densities restricted to
    ///                    Boys-orbital AO domains; needs `domain_cutoff_bohr`.
    ///
    /// "mo" and "ao" compute the SAME quantity and agree to round-off (asserted
    /// in `ferric-mp2`'s tests). The AO path is the correctness reference for
    /// the pseudo-density limit — it is dense here, so selecting it is NOT a
    /// scaling win. "ao-sparse" is the one approximate variant; see
    /// `domain_cutoff_bohr` below. Unknown values are a hard error
    /// (`SosFormulation::parse_config_str`).
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
    ///   "delta-lr"      (default) — Δ-form B: E_MP2\[Coulomb\] + (E_dRPA\[erf\] − 2·E_OS\[erf\]).
    ///                   Pure-LR rings; mixed SR×LR rings dropped. Cost: 1 dRPA\[erf\] call.
    ///
    ///   "coupled-rings" — formulation T: E_MP2\[Coulomb\] + ΔdRPA\[Coulomb\] − ΔdRPA\[erfc\].
    ///                   Screens all rings (ΔdRPA\[Coulomb\]), un-screens pure-SR rings
    ///                   (−ΔdRPA\[erfc\]). Adds all mixed SR×LR rings. Cost: 2 dRPA calls.
    ///
    /// Both formulations have the same exact limits: ω→0 ⇒ plain MP2; ω→∞ ⇒ MP2+ΔdRPA\[Coulomb\].
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
    /// `method.kind = "scs-mp2-2terfc"`. Default 0.75 Å (the published
    /// SCS-MP2(2terfc, aTZ) value, J. Phys. Chem. B 118, 6519 (2014)).
    /// Requires the terfc interpolation tables (`FERRIC_TERF_TABLE_DIR`).
    pub r0_bonded: Option<f64>,
    /// Non-bonded (longer-range) terfc cutoff **r0(2)** in **Å**, used ONLY by
    /// `method.kind = "scs-mp2-2terfc"`. Must be > `r0_bonded`. Default 1.05 Å
    /// (published SCS-MP2(2terfc, aTZ) value). Requires the terfc interpolation tables
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
    /// Decoupled terfc seam sharpness ω in **Å⁻¹** (2026-08 modernization;
    /// `Operator::terfc_with_omega`). Omitted = the Dutoi curvature link
    /// ω = 1/(r₀√2), which is the published method and byte-identical to the
    /// pre-decoupling behavior. When set, the SAME (r₀, ω) reaches BOTH halves
    /// of Eq. 11 in lockstep — the MP2 attenuator and the VV10 damping factor
    /// (the library derives the damping's ω from this, so the halves cannot
    /// silently diverge). Terfc-attenuator only: combining it with
    /// `mp2v_attenuator = "erfc"` is a hard error. The published (r₀, b, C)
    /// were fitted at the LINKED width, so any decoupled-ω run is
    /// unparameterized extrapolation until b is refit.
    pub mp2v_omega: Option<f64>,
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
    /// Whether `lmp2`/`lmp2-direct` compute the canonical RI-MP2 reference:
    /// `[mp2] lmp2_reference`, default FALSE (opt-in — see the field doc).
    pub fn lmp2_reference(&self) -> bool {
        self.lmp2_reference.unwrap_or(false)
    }

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
    /// `mol` is needed only to resolve a `frozen_core = "auto"` against this
    /// molecule (see [`FrozenCore::resolve`]).
    pub fn build_att_vv10_config(
        &self,
        mol: &Molecule,
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
                // The seam sharpness is derived from `cfg.omega` at evaluation
                // time (`effective_vv10_damping`), so `mp2v_omega` need not —
                // and must not — be duplicated here.
                omega_bohr_inv: None,
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

        if let Some(w_ang) = self.mp2v_omega {
            if !w_ang.is_finite() || w_ang <= 0.0 {
                return Err(format!(
                    "[mp2] mp2v_omega must be finite and > 0 (got {w_ang} A^-1); omit it for \
                     the curvature-linked width 1/(r0*sqrt(2))"
                ));
            }
            if cfg.attenuator == AttVv10Attenuator::Erfc {
                return Err(
                    "[mp2] mp2v_omega applies to the terfc attenuator only; the erfc \
                     control's width is 1/(r0*sqrt(2)) by definition. Drop mp2v_omega or set \
                     mp2v_attenuator = \"terfc\"."
                        .to_string(),
                );
            }
            // Å⁻¹ at the TOML boundary (like `omega`/`terf_omega` elsewhere),
            // Bohr⁻¹ internally.
            cfg.omega = Some(w_ang * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV);
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

        cfg.frozen_core = self.frozen_core.resolve(mol);
        cfg.memory_budget_bytes = budget_bytes;
        Ok(cfg)
    }
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RpaCfg {
    pub auxbasis: Option<String>,
    /// Core orbitals excluded from the RPA correlation treatment: an explicit
    /// count, or `"auto"` for the standard small-core count of this molecule.
    /// Default 0 (all-electron). See [`FrozenCore`].
    #[serde(default)]
    pub frozen_core: FrozenCore,
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
    ///   `"boys:<thresh>"`    — Boys-screened with explicit threshold, e.g. "boys:1e-3"
    ///   "auto"             — pick Dense/Boys by atom count (cutoff 30, thresh 1e-4)
    ///   `"auto:<cutoff>"`    — auto with explicit atom cutoff, e.g. "auto:24"
    ///   `"auto:<cutoff>:<thresh>"` — auto with explicit cutoff and Boys threshold
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
    /// Sample the electrostatic potential on the van der Waals surface, not at
    /// the nuclei. `esp_atoms` is dominated by each atom's own nuclear cusp and
    /// separates elements perfectly (H, C, N, O, F occupy disjoint ranges), so
    /// it leaks identity to any model asked to predict it. The surface field is
    /// what a binding partner feels and what shape/electrostatics-conditioned
    /// generative models consume.
    pub compute_esp_surface: Option<bool>,
    /// vdW-radius multiplier for the `compute_esp_surface` shell (default 1.4).
    pub esp_surface_vdw_scale: Option<f64>,
    /// Lebedev order per atom for the `compute_esp_surface` shell (default 110).
    pub esp_surface_n_angular: Option<usize>,
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
    /// Accept an NPZ bundle that is MISSING one or more requested properties,
    /// and still exit 0. Default: false (an incomplete bundle fails the run).
    ///
    /// # Why the default is false
    ///
    /// Each property in the export path is computed in a
    /// `match { Ok => Some, Err => { warn; None } }` arm, and the bundle is
    /// written regardless with the gaps as absent arrays. With no non-zero exit
    /// anywhere in that block, a run whose polarizability step was refused by
    /// the memory gate produced a well-formed, feature-poor NPZ and reported
    /// success — indistinguishable to a caller from a complete one except by
    /// re-reading stderr.
    ///
    /// That is how a 500-molecule QM9 feature regeneration lost `alpha_atomic`
    /// on 476 of 500 molecules without a single failing job.
    ///
    /// The per-property warnings remain: a partial bundle is sometimes what a
    /// user wants, and forcing a re-run of an expensive SCF to recover the
    /// properties that DID work would be worse. What the default changes is
    /// only that the run stops claiming success. Set this to `true` to opt back
    /// in when gaps are acceptable — the honest form of what used to be
    /// implicit.
    pub allow_partial_npz: Option<bool>,
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
    /// Parse the `chi0_sparsity` TOML string into a [`ferric_rpa::config::Chi0Sparsity`].
    ///
    /// Accepted forms (case-insensitive, whitespace-trimmed); an optional
    /// `@<radius_bohr>` suffix on the boys/auto forms sets the G6 centroid
    /// distance pre-filter (omit → ∞ = filter off, byte-identical to pre-G6):
    ///   None / "dense"                 → Dense (default; backward compatible)
    ///   "boys"                         → BoysScreened { thresh: 1e-4, dist: ∞ }
    ///   `"boys:<thresh>"`                → BoysScreened with that threshold
    ///   `"boys:<thresh>@<radius>"`       → …and that distance-cutoff radius (Bohr)
    ///   "auto"                         → Auto { cutoff: 30, thresh: 1e-4, dist ∞ }
    ///   `"auto:<cutoff>"`                → Auto with that atom cutoff
    ///   `"auto:<cutoff>:<thresh>"`       → …and that Boys threshold
    ///   `"auto:<cutoff>:<thresh>@<rad>"` → …and that distance-cutoff radius (Bohr)
    pub fn parse_chi0_sparsity(&self) -> Result<ferric_rpa::config::Chi0Sparsity, String> {
        // Canonical parser lives on the type (shared with the Python bindings).
        ferric_rpa::config::Chi0Sparsity::parse_config_str(self.chi0_sparsity.as_deref())
    }

    /// Parse the `[rpa] quadrature` TOML string into a [`ferric_rpa::config::QuadratureScheme`],
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
    /// Frozen core for the GW self-energy build: an explicit count, or
    /// `"auto"` for the standard small-core count of this molecule (see
    /// [`FrozenCore`]). Must match `[rpa] frozen_core` for self-consistency
    /// between W and Σ — the CLI passes this value to both
    /// `GwConfig.frozen_core` and overrides the PDEP config's frozen_core
    /// with it. Unset (not `"none"`) falls back to `[rpa] frozen_core`, which
    /// is why this one key is an `Option` while the others are not.
    pub frozen_core: Option<FrozenCore>,
    /// Scissor shift (Hartree) added to every virtual orbital energy before
    /// assembling the RPAx@KS diagonal. Only consumed by
    /// `method.kind = "tdhf-static-polarizability"`
    /// (`ferric_gw::bse::run_rpax_static_polarizability`'s `scissor` arg) — a
    /// cheap proxy for widening a KS gap toward a GW-level gap. Unset → 0.0
    /// (plain KS). Ignored by the other `[gw]`-consuming method kinds.
    ///
    /// NOTE: the 0.0 default is the setting that triggers the known excitonic
    /// instability (a negative α diagonal) on several small closed-shell
    /// molecules. That is now REFUSED in the library rather than returned, so a
    /// `scissor = 0.0` run on an affected system aborts with an actionable
    /// error naming ~0.3–0.4 Ha as the remedy — see
    /// `ferric_gw::bse::check_alpha_diagonal_positive` and
    /// `docs/rpax-negative-diagonal-investigation.md`. The default is left at
    /// 0.0 deliberately: silently substituting a nonzero scissor would change
    /// the physics behind the user's back.
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

fn default_multiplicity() -> usize {
    1
}

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

fn default_task() -> String {
    "energy".into()
}

/// `[output]` section: the machine-readable JSON run log.
///
/// # Logging is ON BY DEFAULT
///
/// Omit this section entirely and the run still writes a JSON Lines log
/// (`<input-stem>.ferric.jsonl`, beside the input file). That is deliberate,
/// not an oversight: a result whose run left no artifact cannot be checked
/// afterwards, and this repo has already lost a load-bearing SCF measurement
/// that way -- a 27-atom PBE/6-31G run reported as "173 iterations, converged,
/// E = -390.3794234093" whose only log had been truncated to 275 bytes.
/// Opt-out is explicit: `json = false`.
///
/// ```toml
/// [output]
/// json = "runs/benzene.jsonl"   # custom path
/// # json = false                # opt out entirely
/// ```
///
/// The log never fails a calculation: an unopenable path warns on stderr and
/// the run continues without one (see `ferric_scf::runlog::init`).
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct OutputCfg {
    /// Where to write the JSON run log. A string is a path; `false` turns
    /// logging off; omitted uses the default path. `true` is accepted as an
    /// explicit "yes, the default path".
    #[serde(default)]
    pub json: Option<JsonLogSpec>,
}

/// The value of `[output] json`: a path, or a bool.
///
/// `#[serde(untagged)]` rather than two keys so the TOML reads the way a user
/// would write it (`json = "x.jsonl"` / `json = false`). Untagged means an
/// unusable value (a table, an integer) is reported as "did not match any
/// variant" rather than silently defaulting -- still an error, which is what
/// the config-honesty convention requires.
#[derive(Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum JsonLogSpec {
    /// `json = false` disables the log; `json = true` requests the default path.
    Enabled(bool),
    /// `json = "path/to/run.jsonl"`.
    Path(String),
}

impl OutputCfg {
    /// Resolve `[output] json` against the input file's path to the log path
    /// to use, or `None` when the user turned logging off.
    ///
    /// Note the asymmetry with most config knobs in this file: the ABSENT case
    /// resolves to `Some(default)`, not `None`. Logging is on unless refused.
    pub fn resolve_json_path(&self, input_toml: &std::path::Path) -> Option<std::path::PathBuf> {
        match &self.json {
            None | Some(JsonLogSpec::Enabled(true)) => {
                Some(ferric_scf::runlog::default_path_for_input(input_toml))
            }
            Some(JsonLogSpec::Enabled(false)) => None,
            Some(JsonLogSpec::Path(p)) => Some(std::path::PathBuf::from(p)),
        }
    }
}

fn default_n_roots() -> usize {
    3
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TddftCfg {
    #[serde(default = "default_n_roots")]
    pub n_roots: usize,
    pub xc: Option<String>,
    pub c_hf: Option<f64>,
}

impl Default for TddftCfg {
    fn default() -> Self {
        Self {
            n_roots: default_n_roots(),
            xc: None,
            c_hf: None,
        }
    }
}

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
    /// Exchange builder: "direct" (default), "link", or "cosx" (seminumerical
    /// COSX exchange). Honoured by RHF, UHF and ROHF. Ignored with a warning
    /// when DF-J/DF-K is active, when the functional uses no exact exchange, or
    /// for a range-separated functional.
    pub k_builder: Option<String>,
    /// Shell-quartet screening bound: `"schwarz"` (default), `"csb"`, or
    /// `"csam"`. Unknown values are a hard error (strict parse, matching every
    /// other string knob in this file — see CLAUDE.md's "config honesty"
    /// section).
    ///
    /// ```text
    ///   "schwarz"  RIGOROUS      Q_µν Q_λσ                                (default)
    ///   "csb"      RIGOROUS      min{ Q_µν Q_λσ, M_µλ M_νσ, M_µσ M_νλ }    Eq. (8)
    ///   "csam"     NON-RIGOROUS  Q_µν Q_λσ · sqrt(max(X_µλ X_νσ, X_µσ X_νλ))
    ///                                                                 Eq. (9)/(11)/(12)
    /// ```
    ///
    /// All three are from Thompson & Ochsenfeld, J. Chem. Phys. 147, 144101
    /// (2017), with `M_µλ = sqrt(|(µµ|λλ)|)` and `X_µλ` the normalised ratio
    /// `max|(µµ|λλ)| / sqrt(|(µµ|µµ)||(λλ|λλ)|)`.
    ///
    /// `"schwarz"` and `"csb"` ARE RIGOROUS UPPER BOUNDS, so choosing between
    /// THEM is NOT an accuracy tradeoff — it is speed-vs-setup-cost. Because
    /// CSB is a `min` that INCLUDES the plain Schwarz product, it can never be
    /// looser than `"schwarz"`, so selecting it cannot discard a quartet that
    /// `"schwarz"` would have kept above threshold. It costs one extra
    /// `nshells²` table (one `(PP|QQ)` quartet per shell pair, geometry-only,
    /// built once per bound construction — NOT per SCF iteration).
    ///
    /// `"csam"` IS NOT A BOUND. The authors explicitly call Eqs. (9)/(11)/(12)
    /// "non-rigorous" in their own voice, and the estimate CAN fall below the
    /// true integral magnitude — so it can discard a quartet carrying real
    /// weight. That makes it an ACCURACY-VS-THRESHOLD TRADEOFF: its error is
    /// controlled by `integral_thresh`, and it grows LINEARLY with system size
    /// at fixed threshold (the paper's Fig. 2). Measured energy errors are
    /// nonetheless small — −0.20 … +1.80 nanohartree at ϑ = 1e-12 (Table V),
    /// 0.05–9.35 µH at ϑ = 1e-10 (Table III) — which is why Psi4 ships CSAM as
    /// its own default (`SCREENING=CSAM`). ferric does NOT: `"schwarz"` remains
    /// the default here, and `"csam"` must be asked for explicitly.
    ///
    /// `"csam"` is refused for SHORT-RANGE (`erfc`) operators with a typed
    /// error naming `"csb"` — the paper's own recommendation for those kernels
    /// (Conclusion, p. 144101-8/9), where the rigorous bound is both available
    /// and excellent. A range-separated functional therefore fails loudly under
    /// `screening = "csam"` rather than silently substituting a different
    /// screen.
    ///
    /// WHERE IT PAYS: the paper states that for the long-range Coulomb
    /// operator "the CSB estimate is no more useful than the QQ estimate for
    /// currently tractable systems" (its Table I reports F_min = 1.000 for CSB
    /// under `1/r12`), and that the win appears for strongly distance-decaying
    /// kernels — `e^(-r12)`, `erfc(ω r12)/r12`. ferric's CLI runs Coulomb, so
    /// expect a small or null win on ordinary HF/hybrid jobs. That is the
    /// literature's own prediction, recorded here before any measurement;
    /// nothing in this feature has been benchmarked.
    ///
    /// SCOPE: this CLI resolves `screening` ONCE and uses the SAME resolved
    /// kind both to build its `SchwarzBounds` (via
    /// `SchwarzBounds::compute_for_screening`, which attaches the CSB `M` or
    /// CSAM `X` table to that value and thereby governs the default
    /// `DirectJ`/`DirectK`/`DirectJK`/`build_jk` path for RHF, UHF and ROHF)
    /// and to populate `RhfConfig::screening` (which governs the LinK path).
    /// `k_builder = "cosx"` consumes no Schwarz table at all, so `screening`
    /// has no effect there — a property of COSX, not a gap in this wiring.
    pub screening: Option<String>,
    /// COSX exchange grid, `cosx_grid = { radial = 50, angular = 110 }`.
    /// Omitted = (50,110), the measured operating point (coarser grids fail the
    /// 0.1 kcal/mol isodesmic reaction-energy bar in the composed-budget audit).
    /// `angular` must be a tabulated Lebedev order (6/14/26/50/110/302). Setting
    /// this with any `k_builder` other than "cosx" is a hard error.
    pub cosx_grid: Option<CosxGridCfg>,
    /// COSX overlap fit (Izsák–Neese). Omitted = `true`. At (50,110) the fit
    /// took the isodesmic reaction-energy error 0.2068 -> 0.0190 kcal/mol
    /// (water-favourable set); it is net-NEGATIVE on grids coarser than
    /// (50,110). Setting this with `k_builder != "cosx"` is a hard error.
    pub cosx_overlap_fit: Option<bool>,
    /// COSX 3c1e kernel: `"md3c1e"` (default; batched McMurchie–Davidson) or
    /// `"cosx-a"` (per-point libint2, 3.2–3.5x slower; the cross-check backend,
    /// exact vs md3c1e to ~1e-15). Unknown values and setting this with
    /// `k_builder != "cosx"` are hard errors.
    pub cosx_backend: Option<String>,
    /// COSX density-driven shell-pair screen threshold (md3c1e backend): a
    /// shell pair is evaluated for a grid sub-batch iff
    /// `bound(A^g) * max|F| >= cosx_screen_thresh`. Omitted =
    /// `ferric_scf::cosx_k::COSX_DEFAULT_SCREEN_THRESH` (1e-7: the loosest
    /// value whose K error stays < 1e-6 on water/cc-pVDZ and butane/def2-SVP,
    /// i.e. >= 100x below the (50,110) grid error). `0.0` disables the screen
    /// (bit-identical to unscreened). Negative values, and setting this with
    /// `k_builder != "cosx"` or `cosx_backend = "cosx-a"`, are hard errors.
    pub cosx_screen_thresh: Option<f64>,
    /// COSX block half transforms: `"sparse"` (default; per block only the
    /// active AOs `A`, the D-significant rows `Λ` and the kernel-touched
    /// shells `B` enter `F = D X` / `Ktilde += X G^T` / `S_num = X X^T`) or
    /// `"dense"` (the full `nbf` GEMMs — the byte-identical cross-check path).
    /// Unknown values and setting this with `k_builder != "cosx"` are hard
    /// errors.
    pub cosx_half_transform: Option<String>,
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
    /// Two-stage "DF guess" SCF (Psi4's "Andy trick 2.0"): converge a
    /// density-fitted J/K SCF first, then hand its density to a fresh
    /// exact-4-index-integral SCF. See
    /// `ferric_scf::ladder::solve_rhf_with_df_guess`.
    ///
    /// **Default `true`** (matching Psi4, whose `DF_SCF_GUESS` also defaults
    /// on). Mathematically innocuous: the DF stage only produces a starting
    /// DENSITY; the exact stage still determines the reported energy and
    /// orbitals, so the converged answer is unchanged (measured: benzene/aTZ
    /// -230.7808857506 either way, 1e-10 agreement) — only the iteration count
    /// changes. Measured 1.93x at benzene/aug-cc-pVTZ (759.69 -> 393.85 s),
    /// cutting 12 exact iterations to 7 DF + 6 exact.
    ///
    /// Set `false` to restore the pre-feature path, which is byte-identical to
    /// before this existed (no DF pre-stage is constructed at all).
    ///
    /// SCOPE — this is why the default is safe to flip: the pre-stage only
    /// engages on the closed-shell, non-laddered path (`rimp2` and friends,
    /// `mol.multiplicity == 1`). `rhf`/`ksdft` go through the convergence
    /// ladder, which does not compose with `df_guess` and warns rather than
    /// silently ignoring it; open-shell falls through untouched.
    /// `None` = not set by the user (defaults ON — see `df_guess_enabled`).
    ///
    /// Deliberately `Option<bool>` rather than `bool`: once `df_guess` defaults
    /// to TRUE, a plain `bool` cannot distinguish "the user asked for it" from
    /// "it defaulted on", and the `df_guess`/`df_increments` mutual-exclusion
    /// check then fires for anyone who sets only `df_increments = true` —
    /// making that feature unusable without also writing `df_guess = false`.
    /// CI caught exactly that (`scf_df_increments_key_parses_and_defaults_off`
    /// panicked in `df_increments_aux_resolved`). Keeping the user's intent
    /// distinguishable lets `df_increments` win over a DEFAULTED `df_guess`
    /// while still rejecting an EXPLICIT request for both.
    #[serde(default)]
    pub df_guess: Option<bool>,
    /// Auxiliary (JK-fit) basis for the `df_guess` pre-stage only. Omitted =
    /// `ferric_scf::ladder::DF_GUESS_DEFAULT_AUX` ("def2-universal-jkfit").
    /// Ignored (with a hard error) when `df_guess = false`, matching the
    /// project's cosx_*-style config-honesty convention: a knob that would
    /// silently do nothing is refused rather than accepted.
    pub df_guess_aux: Option<String>,
    /// Opt-in DF-corrected incremental Fock SCF: after a DF-guess pre-stage
    /// builds a density D0, do ONE exact 4-index Fock build F_ex(D0), then run
    /// a cheap DF-corrected inner loop on F(D) = F_ex(D0) + F_DF(D - D0)
    /// before a mandatory final exact build (+ exact cleanup iterations if
    /// needed) produces the reported answer. Default `false` — with this off,
    /// this mechanism is not constructed at all and the SCF path is
    /// byte-identical to before this feature existed. Mutually exclusive with
    /// `df_guess` (this subsumes it — the DF-guess pre-stage is always run as
    /// part of this mechanism); setting both is a hard error. See
    /// `ferric_scf::df_increments::solve_rhf_with_df_increments`.
    #[serde(default)]
    pub df_increments: bool,
    /// Auxiliary (JK-fit) basis for BOTH the `df_increments` pre-stage AND its
    /// DF-corrected inner-loop builders. Omitted =
    /// `ferric_scf::ladder::DF_GUESS_DEFAULT_AUX` ("def2-universal-jkfit").
    /// Ignored (with a hard error) when `df_increments = false`.
    pub df_increments_aux: Option<String>,
    /// Run an internal stability analysis after the SCF converges, and report
    /// whether the converged solution is a minimum or a SADDLE POINT. Default
    /// `false` — the check costs a Davidson eigensolve whose every matvec is a
    /// J/K build, and with it off the SCF path is bit-identical to a build with
    /// no stability support at all.
    ///
    /// DIAGNOSTIC ONLY: an instability prints a warning naming λ_min and the
    /// remedy, and never makes the run fail — a deliberately-unstable state (a
    /// cDFT diabat, a MOM excited state) is a legitimate thing to compute.
    ///
    /// SCOPE: honoured for RHF/RKS (singlet channel) and UHF/UKS (independent
    /// α/β rotations). ROHF/ROKS, range-separated functionals and meta-GGAs
    /// are SKIPPED with a printed reason rather than analysed with the wrong
    /// operator. See `ferric_scf::stability`.
    #[serde(default)]
    pub check_stability: bool,
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
            screening: None,
            cosx_grid: None,
            cosx_overlap_fit: None,
            cosx_backend: None,
            cosx_screen_thresh: None,
            cosx_half_transform: None,
            df_j_aux: None,
            df_k_aux: None,
            level_shift: None,
            mom_after_iter: 0,
            ladder: Vec::new(),
            verbose: false,
            df_guess: None,
            df_guess_aux: None,
            df_increments: false,
            df_increments_aux: None,
            check_stability: false,
        }
    }
}

/// `[scf] cosx_grid = { radial = .., angular = .. }` — the COSX exchange grid.
#[derive(Deserialize, Debug, Clone, Copy)]
#[serde(deny_unknown_fields)]
pub struct CosxGridCfg {
    pub radial: usize,
    pub angular: usize,
}

impl ScfCfg {
    /// Resolve the `[scf] cosx_*` knobs into a `CosxConfig` (strict).
    ///
    /// A `cosx_grid` / `cosx_overlap_fit` / `cosx_backend` key with
    /// `k_builder != "cosx"` is a hard error (a knob that silently did nothing
    /// is exactly what the config-honesty convention forbids); an untabulated
    /// Lebedev order is a hard error here rather than a panic inside the grid
    /// builder; an unknown backend name is a hard error, never a default.
    pub fn cosx_config(&self) -> Result<ferric_scf::cosx_k::CosxConfig, String> {
        use ferric_scf::cosx_k::{validate_grid, CosxBackend, CosxConfig, CosxHalfTransform};
        let is_cosx = self.k_builder.as_deref() == Some("cosx");
        let any_cosx_knob = self.cosx_grid.is_some()
            || self.cosx_overlap_fit.is_some()
            || self.cosx_backend.is_some()
            || self.cosx_screen_thresh.is_some()
            || self.cosx_half_transform.is_some();
        if !is_cosx && any_cosx_knob {
            return Err(format!(
                "[scf] cosx_grid / cosx_overlap_fit / cosx_backend / cosx_screen_thresh / cosx_half_transform are set but k_builder = {:?}; they are only read with k_builder = \"cosx\"",
                self.k_builder
            ));
        }
        let mut cfg = CosxConfig::default();
        if let Some(g) = self.cosx_grid {
            cfg.grid.n_radial = g.radial;
            cfg.grid.n_angular = g.angular;
            validate_grid(&cfg.grid).map_err(|e| format!("[scf] cosx_grid: {e}"))?;
        }
        if let Some(fit) = self.cosx_overlap_fit {
            cfg.overlap_fit = fit;
        }
        if let Some(h) = self.cosx_half_transform.as_deref() {
            cfg.half_transform = CosxHalfTransform::parse_config_str(h)
                .map_err(|e| format!("[scf] cosx_half_transform: {e}"))?;
        }
        if let Some(b) = self.cosx_backend.as_deref() {
            cfg.backend =
                CosxBackend::parse_config_str(b).map_err(|e| format!("[scf] cosx_backend: {e}"))?;
        }
        match self.cosx_screen_thresh {
            Some(t) if !(t >= 0.0) || !t.is_finite() => {
                return Err(format!("[scf] cosx_screen_thresh = {t}: must be a finite value >= 0 (0 disables the screen)"));
            }
            Some(t) if t > 0.0 && cfg.backend == CosxBackend::CosxA => {
                return Err("[scf] cosx_screen_thresh > 0 is implemented for cosx_backend = \"md3c1e\" only; the cosx-a backend runs unscreened".into());
            }
            Some(t) => cfg.screen_thresh = Some(t),
            // The cross-check backend has no batched screen: unscreened, never a refusal from the default.
            None if cfg.backend == CosxBackend::CosxA => cfg.screen_thresh = None,
            None => {}
        }
        Ok(cfg)
    }

    /// Parse the `diis` string into a `DiisFlavor` (strict — unknown values are a
    /// hard error, per the config-honesty convention). Absent = Pulay.
    ///
    /// Returns `Err` rather than panicking: this used to be a `panic!`, so a
    /// typo'd `[scf] diis` aborted the process with a Rust backtrace instead of
    /// a clean error. The parser is shared with `ferric-python`'s `diis=` kwarg.
    pub fn diis_flavor(&self) -> Result<ferric_scf::diis::DiisFlavor, String> {
        match self.diis.as_deref() {
            None => Ok(ferric_scf::diis::DiisFlavor::Pulay),
            Some(s) => ferric_scf::diis::DiisFlavor::parse_config_str(s)
                .map_err(|e| format!("[scf] diis: {e}")),
        }
    }
    /// Resolve `[scf] guess` to `RhfConfig::use_sad_guess` (strict). `true`
    /// selects the MINAO guess (the default), `false` the hcore guess; `"sad"`
    /// is an alias of `"minao"` (see `ferric_scf::guess::InitialGuess`).
    /// Unknown values are an error -- before this, any string other than
    /// "hcore" was accepted and silently ran MINAO.
    pub fn use_density_guess(&self) -> Result<bool, String> {
        match self.guess.as_deref() {
            None => Ok(true),
            Some(s) => ferric_scf::guess::InitialGuess::parse_config_str(s)
                .map(|g| g.use_sad_guess())
                .map_err(|e| format!("[scf] guess: {e}")),
        }
    }
    /// Post-parse validation of the `[scf]` string knobs whose resolution is
    /// otherwise deferred to the point of use. Called from [`load_config`] so
    /// every entry point (CLI and `ferric-batch`) fails before any integral is
    /// computed.
    pub fn validate(&self) -> Result<(), String> {
        self.diis_flavor()?;
        self.use_density_guess()?;
        for (i, rung) in self.ladder.iter().enumerate() {
            rung.use_sad_guess()
                .map_err(|e| format!("[[scf.ladder]] rung {i}: {e}"))?;
        }
        Ok(())
    }
    /// Resolve `df_guess_aux` under the config-honesty convention: setting it
    /// while `df_guess = false` would be a silent no-op, so it is a hard
    /// error instead (mirrors `cosx_config`'s treatment of `cosx_*` knobs set
    /// without `k_builder = "cosx"`).
    /// Is the DF-guess pre-stage active? Defaults ON when the key is absent.
    ///
    /// `df_increments` takes precedence over a DEFAULTED `df_guess` (it runs
    /// its own DF pre-stage internally, so the two would be redundant); an
    /// EXPLICIT `df_guess = true` alongside `df_increments` is still a hard
    /// error, because that is a user asking for two mutually exclusive things
    /// rather than a default colliding with a request.
    pub fn df_guess_enabled(&self) -> bool {
        match self.df_guess {
            Some(explicit) => explicit,
            None => !self.df_increments,
        }
    }
    pub fn df_guess_aux_resolved(&self) -> Result<Option<String>, String> {
        if !self.df_guess_enabled() && self.df_guess_aux.is_some() {
            return Err(format!(
                "[scf] df_guess_aux = {:?} is set but df_guess is off; it is only read with df_guess = true",
                self.df_guess_aux
            ));
        }
        if self.df_guess == Some(true) && self.df_increments {
            return Err(
                "[scf] df_guess = true and df_increments = true are mutually exclusive \
                 (df_increments already runs its own DF-guess pre-stage internally); set only one"
                    .to_string(),
            );
        }
        Ok(self.df_guess_aux.clone())
    }
    /// Resolve `df_increments_aux` under the same config-honesty convention as
    /// `df_guess_aux_resolved`.
    pub fn df_increments_aux_resolved(&self) -> Result<Option<String>, String> {
        if !self.df_increments && self.df_increments_aux.is_some() {
            return Err(format!(
                "[scf] df_increments_aux = {:?} is set but df_increments = false; it is only read with df_increments = true",
                self.df_increments_aux
            ));
        }
        if self.df_guess == Some(true) && self.df_increments {
            return Err(
                "[scf] df_guess = true and df_increments = true are mutually exclusive \
                 (df_increments already runs its own DF-guess pre-stage internally); set only one"
                    .to_string(),
            );
        }
        Ok(self.df_increments_aux.clone())
    }
}

// `default_df_guess()` lived here. Removed 2026-09-16: `df_guess` is now
// `Option<bool>` with a plain `#[serde(default)]` (None = user said nothing),
// and the ON default is carried by `ScfCfg::df_guess_enabled()` instead. That
// is what lets `df_increments` take precedence over a DEFAULTED df_guess while
// an EXPLICIT `df_guess = true` alongside it stays a hard error -- a
// distinction a bare `bool` cannot express. A serde default that materialised
// `true` would erase it.

fn default_max_iter() -> usize {
    100
}
// Match the library convergence gate (rhf::scf_converged): density_conv is the
// tight (reachable) ΔP signal; energy_conv is a LOOSE "not-descending" bound. A
// tight energy_conv hangs a large DF molecule (dE floors on the RI noise level).
fn default_energy_conv() -> f64 {
    1e-3
}
fn default_density_conv() -> f64 {
    1e-6
}
fn default_diis_size() -> usize {
    8
}
fn default_integral_thresh() -> f64 {
    1e-12
}

/// One `[[scf.ladder]]` rung. Every field is optional and overrides the
/// corresponding field of the `base` `RhfConfig` passed to
/// [`ScfCfg::build_ladder`] (derived from the flat `[scf]` settings); unset
/// fields inherit from `base`.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LadderRungCfg {
    /// Initial guess for this rung: "minao" | "sad" (alias of "minao") |
    /// "hcore". `None` (default) behaves like "minao". Strict: anything else
    /// -- including the former "sad-smallbasis", which was never wired to the
    /// CLI and silently ran plain MINAO after a warning -- is an error.
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

impl LadderRungCfg {
    /// This rung's `guess`, resolved strictly (see the field doc).
    pub fn use_sad_guess(&self) -> Result<bool, String> {
        match self.guess.as_deref() {
            None => Ok(true),
            Some(s) => ferric_scf::guess::InitialGuess::parse_config_str(s)
                .map(|g| g.use_sad_guess())
                .map_err(|e| format!("guess: {e}")),
        }
    }
}

impl ScfCfg {
    /// Build the SCF convergence ladder. If no `[[scf.ladder]]` rungs are
    /// configured, returns the built-in `default_ladder()`. Otherwise each
    /// rung starts from `base` (the `RhfConfig` derived from the flat `[scf]`
    /// settings) and overrides the fields the rung specifies.
    pub fn build_ladder(
        &self,
        base: &ferric_scf::rhf::RhfConfig,
    ) -> Result<Vec<ferric_scf::ladder::Rung>, String> {
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
            return Ok(if base.xc.is_some() {
                ferric_scf::ladder::ksdft_ladder(base)
            } else {
                ferric_scf::ladder::default_ladder_from(base)
            });
        }
        self.ladder
            .iter()
            .enumerate()
            .map(|(i, r)| -> Result<Rung, String> {
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
                cfg.use_sad_guess = r
                    .use_sad_guess()
                    .map_err(|e| format!("[[scf.ladder]] rung {i}: {e}"))?;
                Ok(Rung {
                    config: cfg,
                    restart: r.restart,
                })
            })
            .collect()
    }
}

pub fn load_config(path: &str) -> Result<Config, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read config file {path:?}: {e}"))?;
    let cfg: Config = toml::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    // Post-parse semantic validation. `deny_unknown_fields` already rejects a
    // typo'd KEY at parse time; this catches a typo'd VALUE whose silent
    // fall-through would be worse than an error. See `MemoryCfg::validate`.
    //
    // Validating HERE rather than at the lib.rs use site means every entry
    // point is covered by construction — the CLI, and `ferric-batch`'s
    // per-child TOML rewriting, which does not go through lib.rs's checks.
    cfg.memory.validate().map_err(|e| format!("{path}: {e}"))?;
    cfg.scf.validate().map_err(|e| format!("{path}: {e}"))?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Water — the stand-in molecule for the config tests that only need
    /// *some* molecule to resolve a `frozen_core` against. Two heavy-atom-free
    /// hydrogens and one oxygen: 5 occupied orbitals, 1 core.
    fn water() -> Molecule {
        Molecule::parse_xyz(
            "3\nwater\nO 0.000000 0.000000 0.117790\n\
             H 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
            0,
            1,
        )
        .unwrap()
    }

    // ---- [output]: the JSON run log's config surface ----

    fn parse(toml_src: &str) -> Result<Config, String> {
        toml::from_str::<Config>(toml_src).map_err(|e| e.to_string())
    }

    const MINIMAL: &str = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rhf"
"#;

    /// THE default: no `[output]` section at all still yields a log path.
    ///
    /// Logging is ON BY DEFAULT and this is the test that says so. If someone
    /// later "fixes" `resolve_json_path` to return `None` when the key is
    /// absent — the shape every other optional knob in this file has — this
    /// fails, which is the point: the asymmetry is deliberate.
    #[test]
    fn omitting_the_output_section_still_logs() {
        let cfg = parse(MINIMAL).expect("minimal config must parse");
        let p = cfg
            .output
            .resolve_json_path(std::path::Path::new("/runs/benzene.toml"));
        assert_eq!(
            p,
            Some(std::path::PathBuf::from("/runs/benzene.ferric.jsonl")),
            "a run with no [output] section must still write a log"
        );
    }

    #[test]
    fn json_false_turns_the_log_off() {
        let cfg = parse(&format!(
            "{MINIMAL}
[output]
json = false
"
        ))
        .unwrap();
        assert_eq!(
            cfg.output
                .resolve_json_path(std::path::Path::new("/runs/x.toml")),
            None
        );
    }

    #[test]
    fn json_true_means_the_default_path() {
        let cfg = parse(&format!(
            "{MINIMAL}
[output]
json = true
"
        ))
        .unwrap();
        assert_eq!(
            cfg.output
                .resolve_json_path(std::path::Path::new("/runs/x.toml")),
            Some(std::path::PathBuf::from("/runs/x.ferric.jsonl"))
        );
    }

    #[test]
    fn json_string_is_used_verbatim() {
        let cfg = parse(&format!(
            "{MINIMAL}
[output]
json = \"logs/custom.jsonl\"
"
        ))
        .unwrap();
        assert_eq!(
            cfg.output
                .resolve_json_path(std::path::Path::new("/runs/x.toml")),
            Some(std::path::PathBuf::from("logs/custom.jsonl"))
        );
    }

    /// `deny_unknown_fields` must still bite inside `[output]` — a typo'd key
    /// is a hard error, never a silent default (the config-honesty convention).
    #[test]
    fn a_typod_output_key_hard_errors() {
        let r = parse(&format!(
            "{MINIMAL}
[output]
jsonn = false
"
        ));
        assert!(
            r.is_err(),
            "typo'd [output] key parsed successfully — deny_unknown_fields regressed"
        );
    }

    /// An `[output] json` value of a type that is neither a string nor a bool
    /// must be an error, not a silent fall-through to the default. The
    /// `untagged` enum makes this the "no variant matched" path.
    #[test]
    fn a_nonsense_json_value_hard_errors() {
        assert!(parse(&format!(
            "{MINIMAL}
[output]
json = 17
"
        ))
        .is_err());
        assert!(parse(&format!(
            "{MINIMAL}
[output]
json = [1, 2]
"
        ))
        .is_err());
    }

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
    /// Workspace root, resolved at RUN time.
    ///
    /// `env!("CARGO_MANIFEST_DIR")` is baked in at COMPILE time, so under
    /// `cargo nextest archive` -- built in one job, run in another -- it names
    /// a directory that does not exist, and every example "fails to parse" for
    /// want of a file. Walk up from the cwd to the root that actually holds
    /// `examples/` + `testdata/`, falling back to the compile-time path for
    /// plain `cargo test`.
    ///
    /// Extracted rather than inlined because inlining pushed
    /// `all_shipped_examples_parse` from CC 6 to 13 and the complexity gate
    /// caught it -- correctly, since path-walking has nothing to do with what
    /// that test asserts.
    fn runtime_workspace_root() -> std::path::PathBuf {
        let looks_like_root = |p: &std::path::Path| {
            p.join("Cargo.toml").is_file()
                && p.join("examples").is_dir()
                && p.join("testdata").is_dir()
        };
        if let Ok(cwd) = std::env::current_dir() {
            let mut here: Option<&std::path::Path> = Some(cwd.as_path());
            while let Some(p) = here {
                if looks_like_root(p) {
                    return p.to_path_buf();
                }
                here = p.parent();
            }
        }
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    /// Every basis NAME a shipped example references must resolve through
    /// `ferric_core::basis::bundled`.
    ///
    /// `all_shipped_examples_parse` only checks TOML shape, so an example
    /// naming an unbundled set parses fine and dies at run time with
    /// "unknown bundled basis". Three examples (water-tda, water-tddft-pbe,
    /// water-b2plyp) did exactly that: b2plyp named `cc-pvdz-rifit`, and the two
    /// TDDFT ones fell back to the same unbundled name as the CLI default.
    /// Hence the defaults are checked too (`default_aux_bases_resolve`).
    ///
    /// Static: loads each basis, runs no calculation.
    #[test]
    fn all_shipped_examples_reference_bundled_bases() {
        let workspace_root = runtime_workspace_root();
        let dir = workspace_root.join("examples");
        let mut checked = 0usize;
        let mut failures = Vec::new();
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            let s = std::fs::read_to_string(&path).unwrap();
            let cfg: Config = toml::from_str(&s)
                .unwrap_or_else(|e| panic!("example {} no longer parses: {e}", path.display()));
            let mut names: Vec<(&str, &str)> = Vec::new();
            if let Some(n) = cfg.basis.name.as_deref() {
                names.push(("[basis] name", n));
            }
            if let Some(p) = cfg.basis.path.as_deref() {
                if !workspace_root.join(p).is_file() {
                    failures.push(format!(
                        "{}: [basis] path = {p:?} does not exist",
                        path.display()
                    ));
                }
            }
            let optional = [
                ("[mp2] auxbasis", cfg.mp2.auxbasis.as_deref()),
                ("[rpa] auxbasis", cfg.rpa.auxbasis.as_deref()),
                ("[scf] df_j_aux", cfg.scf.df_j_aux.as_deref()),
                ("[scf] df_k_aux", cfg.scf.df_k_aux.as_deref()),
                ("[scf] df_guess_aux", cfg.scf.df_guess_aux.as_deref()),
                (
                    "[scf] df_increments_aux",
                    cfg.scf.df_increments_aux.as_deref(),
                ),
            ];
            names.extend(optional.iter().filter_map(|(k, v)| v.map(|v| (*k, v))));
            for rung in &cfg.scf.ladder {
                if let Some(v) = rung.df_j_aux.as_deref() {
                    names.push(("[[scf.ladder]] df_j_aux", v));
                }
                if let Some(v) = rung.df_k_aux.as_deref() {
                    names.push(("[[scf.ladder]] df_k_aux", v));
                }
            }
            for (key, name) in names {
                checked += 1;
                if let Err(e) = ferric_core::basis::bundled(name) {
                    failures.push(format!("{}: {key} = {name:?}: {e}", path.display()));
                }
            }
        }
        assert!(checked > 0, "no basis names found in {}", dir.display());
        assert!(
            failures.is_empty(),
            "shipped examples reference basis sets that are not bundled:\n  {}",
            failures.join("\n  ")
        );
    }

    /// The aux bases a run falls back to when the TOML names none must be
    /// bundled -- an example that omits `auxbasis` exercises these, and the
    /// static scan above cannot see a default.
    #[test]
    fn default_aux_bases_resolve() {
        for name in [
            DEFAULT_CORRELATION_AUX,
            TDDFT_DEFAULT_AUX,
            DEFAULT_SCF_JK_AUX,
        ] {
            ferric_core::basis::bundled(name)
                .unwrap_or_else(|e| panic!("default aux basis {name:?} is not bundled: {e}"));
        }
    }

    // --- [qmmm]: the section that made QM/MM reachable from the CLI --------
    //
    // The load-bearing test is `the_mm_field_is_actually_applied`. Every other
    // assertion here also passes if the MM charges are parsed, stored, and
    // then ignored -- a QM/MM run that drops its field still converges and
    // still prints an energy, the VACUUM energy wearing a QM/MM label.

    fn qmmm_fixture(extra: &str) -> String {
        format!(
            "[molecule]\nxyz = \"unused.xyz\"\ncharge = 0\nmultiplicity = 1\n\n\
             [basis]\nname = \"sto-3g\"\n\n[method]\nkind = \"rhf\"\n\n\
             [qmmm]\npqr = \"{}/testdata/molecules/water_na.pqr\"\n{extra}\n",
            runtime_workspace_root().display()
        )
    }

    fn qmmm_cfg(extra: &str) -> QmmmCfg {
        toml::from_str::<Config>(&qmmm_fixture(extra))
            .expect("fixture should parse")
            .qmmm
            .expect("[qmmm] should be present")
    }

    #[test]
    fn qmmm_the_mm_field_is_actually_applied() {
        // Water (QM) + Na+ at 4 A (MM). The CLI energy for this input is
        // -74.9653197421, which matches
        // `ferric.run_rhf(point_charges=[(1.0, 0, 0, 4/0.529...)])` to all 10
        // printed digits; vacuum is -74.9629466809. A run that parsed the
        // charge and dropped it would land on the latter.
        let sys = qmmm_cfg("qm_indices = [0, 1, 2]")
            .to_system(0, 1)
            .expect("system should build");
        assert_eq!(sys.to_qm_molecule().atoms.len(), 3);

        let ext = sys
            .to_external_potential()
            .expect("one MM atom means a non-empty external potential");
        assert_eq!(ext.point_charges.len(), 1);
        let na = &ext.point_charges[0];
        assert!(
            (na.q - 1.0).abs() < 1e-12,
            "the PQR charge column must reach the field, got q = {}",
            na.q
        );
        // 4 Angstrom in Bohr. A unit slip here yields a plausible wrong
        // answer rather than an error, which is why it is asserted.
        let expect_z = 4.0 / 0.529_177_210_92;
        assert!(
            (na.z - expect_z).abs() < 1e-9,
            "Na z = {} Bohr, expected {expect_z} (4 A)",
            na.z
        );
    }

    #[test]
    fn qmmm_a_radial_selection_picks_the_same_region() {
        // 1.5 A around O catches both H (0.96 A) and not the ion (4 A). The
        // same split reached a different way, so the two selection modes
        // cannot silently disagree.
        let sys = qmmm_cfg("qm_seeds = [0]\nqm_radius_angstrom = 1.5")
            .to_system(0, 1)
            .expect("radial selection should build");
        assert_eq!(sys.to_qm_molecule().atoms.len(), 3);
        assert_eq!(sys.to_external_potential().unwrap().point_charges.len(), 1);
    }

    #[test]
    fn qmmm_two_selections_at_once_are_refused() {
        let err = qmmm_cfg("qm_indices = [0]\nqm_seeds = [0]\nqm_radius_angstrom = 3.0")
            .to_system(0, 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not both"), "{err}");
    }

    #[test]
    fn qmmm_no_selection_is_refused() {
        let err = qmmm_cfg("").to_system(0, 1).unwrap_err().to_string();
        assert!(err.contains("no QM region selected"), "{err}");
    }

    #[test]
    fn qmmm_an_unknown_boundary_scheme_is_refused() {
        let err = qmmm_cfg(
            "qm_indices = [0, 1, 2]\nlink_bonds = [[0, 3]]\nboundary_scheme = \"wishful\"",
        )
        .to_system(0, 1)
        .unwrap_err()
        .to_string();
        assert!(err.contains("unknown boundary charge scheme"), "{err}");
    }

    #[test]
    fn qmmm_a_scheme_without_link_bonds_is_refused() {
        // It would do nothing, which is worse than an error: it reads as
        // though it did something.
        let err = qmmm_cfg("qm_indices = [0, 1, 2]\nboundary_scheme = \"rc\"")
            .to_system(0, 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no link_bonds"), "{err}");
    }

    #[test]
    fn qmmm_an_out_of_range_index_is_refused_rather_than_clamped() {
        let err = qmmm_cfg("qm_indices = [0, 99]")
            .to_system(0, 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("out of range"), "{err}");
    }

    #[test]
    fn qmmm_an_unknown_key_is_a_hard_error() {
        // `deny_unknown_fields`, as every other section has: a typo'd knob
        // must not read as a default.
        // `Config` has no Debug, so match rather than unwrap_err.
        let e = match toml::from_str::<Config>(&qmmm_fixture("qm_indices = [0]\nqm_radius = 3.0")) {
            Ok(_) => panic!("an unknown [qmmm] key parsed instead of erroring"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("qm_radius"), "{e}");
    }

    #[test]
    fn qmmm_pqr_element_symbols_come_from_atom_names() {
        // PDB naming: leading digits stripped, element is the leading alpha
        // run. `CA` resolves to CARBON (alpha carbon), which is the common
        // case in a protein file and is documented as the ambiguity it is.
        assert_eq!(element_from_pqr_name("O"), "O");
        assert_eq!(element_from_pqr_name("HB2"), "H");
        assert_eq!(element_from_pqr_name("1HG1"), "H");
        assert_eq!(element_from_pqr_name("NA"), "Na");
        assert_eq!(element_from_pqr_name("CL"), "Cl");
        assert_eq!(element_from_pqr_name("CA"), "C");
    }

    #[test]
    fn qmmm_an_unresolvable_element_is_refused_in_either_region() {
        // `unwrap_or(0)` used to make this a Z = 0 atom, and what happened
        // then depended on WHERE the atom landed:
        //
        //   QM region -> the basis lookup caught it by luck ("no basis shells
        //                for Z=0"), an error about the wrong thing;
        //   MM region -> NOTHING caught it. MM atoms enter only through charge
        //                and position, so the run completed and printed an
        //                energy with no sign that a record was malformed.
        //
        // The MM case is the one that matters and the one a QM-only test
        // would miss, so both are asserted here.
        let dir = std::env::temp_dir().join("ferric_qmmm_element_test");
        std::fs::create_dir_all(&dir).unwrap();

        let good = "ATOM      1  O   WAT     1       0.000   0.000   0.000 -0.834 1.77\n\
                    ATOM      2  H   WAT     1       0.757   0.586   0.000  0.417 0.00\n\
                    ATOM      3  H   WAT     1      -0.757   0.586   0.000  0.417 0.00\n";
        let bad_mm =
            format!("{good}ATOM      4  XX  ION     2       0.000   0.000   4.000  1.000 1.87\n");
        let bad_qm = "ATOM      1  O   WAT     1       0.000   0.000   0.000 -0.834 1.77\n\
                      ATOM      2  H   WAT     1       0.757   0.586   0.000  0.417 0.00\n\
                      ATOM      3  XX  WAT     1      -0.757   0.586   0.000  0.417 0.00\n\
                      ATOM      4  NA  ION     2       0.000   0.000   4.000  1.000 1.87\n";

        for (name, text, label) in [
            ("bad_mm.pqr", bad_mm.as_str(), "MM region"),
            ("bad_qm.pqr", bad_qm, "QM region"),
        ] {
            let path = dir.join(name);
            std::fs::write(&path, text).unwrap();
            let cfg = QmmmCfg {
                pqr: path.to_str().unwrap().to_string(),
                qm_indices: vec![0, 1, 2],
                qm_seeds: vec![],
                qm_radius_angstrom: None,
                link_bonds: vec![],
                boundary_scheme: default_boundary_scheme(),
            };
            match cfg.to_system(0, 1) {
                Ok(_) => panic!("{label}: an unknown element must be refused"),
                Err(e) => {
                    let msg = e.to_string();
                    assert!(
                        msg.contains("not a known element"),
                        "{label}: wrong error: {msg}"
                    );
                    // The INDEX must be there, or the user cannot find the
                    // offending record in a 6000-atom pocket file. Asserting
                    // on the word "atom" alone is not enough -- it also
                    // appears in the prose, so that version of this check
                    // passed against a message with the index removed. The
                    // bad record is index 3 in the MM file and index 2 in the
                    // QM one, so require the specific number.
                    let want = if label == "MM region" {
                        "atom 3"
                    } else {
                        "atom 2"
                    };
                    assert!(msg.contains(want), "{label}: want {want:?} in: {msg}");
                }
            }
        }

        // And a file of resolvable elements must still build.
        let ok = dir.join("ok.pqr");
        std::fs::write(
            &ok,
            format!("{good}ATOM      4  NA  ION     2       0.000   0.000   4.000  1.000 1.87\n"),
        )
        .unwrap();
        let cfg = QmmmCfg {
            pqr: ok.to_str().unwrap().to_string(),
            qm_indices: vec![0, 1, 2],
            qm_seeds: vec![],
            qm_radius_angstrom: None,
            link_bonds: vec![],
            boundary_scheme: default_boundary_scheme(),
        };
        assert!(cfg.to_system(0, 1).is_ok(), "a valid PQR must still build");
    }

    #[test]
    fn qmmm_a_short_pqr_record_is_an_error_not_a_skipped_atom() {
        // A silently dropped atom changes the MM field without changing
        // anything a user would look at.
        let dir = std::env::temp_dir().join("ferric_qmmm_pqr_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("short.pqr");
        std::fs::write(&p, "ATOM      1  O   WAT     1       0.0   0.0   0.0\n").unwrap();
        let err = parse_pqr(p.to_str().unwrap()).unwrap_err().to_string();
        assert!(err.contains("expected 10"), "{err}");
    }

    #[test]
    fn all_shipped_examples_parse() {
        // Resolve the workspace at RUN time. `env!("CARGO_MANIFEST_DIR")` is
        // baked in at COMPILE time, so under `cargo nextest archive` -- built
        // in one job, run in another -- it names a directory that does not
        // exist and every example "fails to parse" for want of a file. Walk up
        // from the cwd to the root that actually holds examples/ + testdata/,
        // falling back to the compile-time path for plain `cargo test`. Same
        // fix as the ferric-cli integration tests' `workspace_root()`.
        let workspace_root = runtime_workspace_root();
        let dir = workspace_root.join("examples");
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

    // ---- [scf] guess / diis: strict string knobs -------------------------

    fn scf_cfg(scf: &str) -> ScfCfg {
        parse(&format!("{MINIMAL}[scf]\n{scf}"))
            .unwrap_or_else(|e| panic!("[scf] {scf:?} must parse: {e}"))
            .scf
    }

    /// `[scf] guess` used to accept ANY string: everything but "hcore" silently
    /// ran MINAO, so `guess = "hcroe"` produced a MINAO run the user did not ask
    /// for. Now: the valid spellings resolve, anything else errors and lists them.
    #[test]
    fn scf_guess_is_strict() {
        assert_eq!(scf_cfg("").use_density_guess(), Ok(true), "absent = MINAO");
        assert_eq!(scf_cfg("guess = \"minao\"").use_density_guess(), Ok(true));
        assert_eq!(scf_cfg("guess = \"sad\"").use_density_guess(), Ok(true));
        assert_eq!(scf_cfg("guess = \"hcore\"").use_density_guess(), Ok(false));
        for bad in ["hcroe", "core", "sad-smallbasis", ""] {
            let cfg = scf_cfg(&format!("guess = {bad:?}"));
            let err = cfg.use_density_guess().unwrap_err();
            assert!(err.starts_with("[scf] guess"), "{bad:?}: {err}");
            for valid in ["'minao'", "'sad'", "'hcore'"] {
                assert!(
                    err.contains(valid),
                    "{bad:?}: message must list {valid}: {err}"
                );
            }
            assert!(
                cfg.validate().is_err(),
                "{bad:?}: validate() must reject it too"
            );
        }
    }

    /// A bad `[scf] diis` used to PANIC (a `panic!` in `diis_flavor`), aborting
    /// the process with a backtrace. It must be a clean `Err` naming the valid
    /// values -- checked under `catch_unwind` so a regression to a panic fails
    /// here as a panic-was-caught assertion, not as a crashed test binary.
    #[test]
    fn scf_diis_bad_value_errors_and_does_not_panic() {
        use ferric_scf::diis::DiisFlavor;
        assert_eq!(scf_cfg("").diis_flavor(), Ok(DiisFlavor::Pulay));
        assert_eq!(
            scf_cfg("diis = \"pulay\"").diis_flavor(),
            Ok(DiisFlavor::Pulay)
        );
        assert_eq!(
            scf_cfg("diis = \"adiis\"").diis_flavor(),
            Ok(DiisFlavor::Adiis)
        );
        assert_eq!(
            scf_cfg("diis = \"EDIIS\"").diis_flavor(),
            Ok(DiisFlavor::Ediis)
        );
        let cfg = scf_cfg("diis = \"cdiis\"");
        let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cfg.diis_flavor()));
        let err = got
            .expect("a bad [scf] diis must return Err, not panic")
            .unwrap_err();
        assert!(err.starts_with("[scf] diis"), "{err}");
        for valid in ["'pulay'", "'adiis'", "'ediis'"] {
            assert!(err.contains(valid), "message must list {valid}: {err}");
        }
        assert!(cfg.validate().is_err());
    }

    /// `load_config` runs the `[scf]` validation, so a typo'd value fails at
    /// load time -- before any integral -- on every entry point (the CLI and
    /// ferric-batch), naming the file.
    #[test]
    fn load_config_rejects_bad_scf_strings() {
        let dir =
            std::env::temp_dir().join(format!("ferric-cli-scf-strict-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cases = [
            ("bad_diis.toml", "[scf]\ndiis = \"cdiis\"\n", "[scf] diis"),
            (
                "bad_guess.toml",
                "[scf]\nguess = \"huckel\"\n",
                "[scf] guess",
            ),
            (
                "bad_rung.toml",
                "[[scf.ladder]]\nguess = \"hcore\"\n[[scf.ladder]]\nguess = \"sad-smallbasis\"\n",
                "rung 1",
            ),
        ];
        for (file, extra, want) in cases {
            let p = dir.join(file);
            std::fs::write(&p, format!("{MINIMAL}{extra}")).unwrap();
            let err = match load_config(p.to_str().unwrap()) {
                Ok(_) => panic!("{file}: load_config accepted a bad value"),
                Err(e) => e,
            };
            assert!(err.contains(want), "{file}: expected {want:?} in: {err}");
        }
        // Positive control: the same file shape with valid values loads.
        let p = dir.join("good.toml");
        std::fs::write(
            &p,
            format!("{MINIMAL}[scf]\ndiis = \"adiis\"\nguess = \"hcore\"\n[[scf.ladder]]\nguess = \"minao\"\n"),
        )
        .unwrap();
        load_config(p.to_str().unwrap()).expect("valid [scf] strings must load");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- deny_unknown_fields on the nested tables that lacked it ----------

    /// `[external_potential]`, its `[[external_potential.point_charges]]`
    /// entries, and `[[scf.ladder]]` rungs silently IGNORED unknown keys: a
    /// typo'd `feild = [0, 0, 0.01]` ran the unperturbed molecule, and a rung's
    /// `levelshift = 0.5` ran unshifted. Each typo must now fail to parse, and
    /// each correctly-spelled twin must still parse (so the rejection is about
    /// the key, not the table).
    #[test]
    fn nested_tables_reject_unknown_keys() {
        let cases = [
            (
                "[external_potential]\nfield = [0.0, 0.0, 0.01]\n",
                "[external_potential]\nfeild = [0.0, 0.0, 0.01]\n",
                "feild",
            ),
            (
                "[[external_potential.point_charges]]\nq = 1.0\nx = 0.0\ny = 0.0\nz = 5.0\n",
                "[[external_potential.point_charges]]\nq = 1.0\nx = 0.0\ny = 0.0\nz = 5.0\ncharge = 1.0\n",
                "charge",
            ),
            (
                "[[scf.ladder]]\nlevel_shift = 0.5\n",
                "[[scf.ladder]]\nlevelshift = 0.5\n",
                "levelshift",
            ),
        ];
        for (good, typo, key) in cases {
            parse(&format!("{MINIMAL}{good}"))
                .unwrap_or_else(|e| panic!("valid table must parse:\n{good}\n{e}"));
            let err = match parse(&format!("{MINIMAL}{typo}")) {
                Ok(_) => panic!("typo'd key `{key}` was silently accepted:\n{typo}"),
                Err(e) => e,
            };
            assert!(err.contains(key), "error must name `{key}`: {err}");
        }
    }

    /// `[scf] cosx_grid` / `cosx_overlap_fit`: parse, resolve, and refuse
    /// when they would be dead knobs (k_builder != "cosx") or name an
    /// untabulated Lebedev order.
    #[test]
    fn cosx_knobs_resolve_strictly() {
        let parse = |scf: &str| -> Config {
            toml::from_str(&format!(
                "[molecule]\nxyz = \"w.xyz\"\n[basis]\nname = \"sto-3g\"\n[method]\nkind = \"rhf\"\n[scf]\n{scf}"
            ))
            .unwrap()
        };
        // Defaults: (50,110), fit on, density-driven screen at the library default.
        let c = parse("k_builder = \"cosx\"\n").scf.cosx_config().unwrap();
        assert_eq!((c.grid.n_radial, c.grid.n_angular), (50, 110));
        assert!(c.overlap_fit);
        assert_eq!(
            c.screen_thresh,
            Some(ferric_scf::cosx_k::COSX_DEFAULT_SCREEN_THRESH)
        );
        assert_eq!(
            c.half_transform,
            ferric_scf::cosx_k::CosxHalfTransform::SPARSE_DEFAULT
        );
        // Half transform: both spellings resolve, unknown values and dead knobs error.
        let c = parse("k_builder = \"cosx\"\ncosx_half_transform = \"dense\"\n")
            .scf
            .cosx_config()
            .unwrap();
        assert_eq!(
            c.half_transform,
            ferric_scf::cosx_k::CosxHalfTransform::Dense
        );
        let c = parse("k_builder = \"cosx\"\ncosx_half_transform = \"sparse\"\n")
            .scf
            .cosx_config()
            .unwrap();
        assert_eq!(
            c.half_transform,
            ferric_scf::cosx_k::CosxHalfTransform::SPARSE_DEFAULT
        );
        assert!(
            parse("k_builder = \"cosx\"\ncosx_half_transform = \"Dense\"\n")
                .scf
                .cosx_config()
                .is_err()
        );
        assert!(
            parse("k_builder = \"link\"\ncosx_half_transform = \"dense\"\n")
                .scf
                .cosx_config()
                .is_err()
        );
        // Screen knob: explicit value honoured, 0 disables, negative/NaN refused,
        // dead-knob refused, and > 0 refused with the unscreened cosx-a backend
        // (which resolves to None by itself, never a refusal from the default).
        let c = parse("k_builder = \"cosx\"\ncosx_screen_thresh = 1e-9\n")
            .scf
            .cosx_config()
            .unwrap();
        assert_eq!(c.screen_thresh, Some(1e-9));
        let c = parse("k_builder = \"cosx\"\ncosx_screen_thresh = 0.0\n")
            .scf
            .cosx_config()
            .unwrap();
        assert_eq!(c.screen_thresh, Some(0.0));
        assert!(parse("k_builder = \"cosx\"\ncosx_screen_thresh = -1e-7\n")
            .scf
            .cosx_config()
            .is_err());
        assert!(parse("k_builder = \"cosx\"\ncosx_screen_thresh = nan\n")
            .scf
            .cosx_config()
            .is_err());
        assert!(parse("k_builder = \"link\"\ncosx_screen_thresh = 1e-7\n")
            .scf
            .cosx_config()
            .is_err());
        assert!(parse(
            "k_builder = \"cosx\"\ncosx_backend = \"cosx-a\"\ncosx_screen_thresh = 1e-7\n"
        )
        .scf
        .cosx_config()
        .is_err());
        let c = parse("k_builder = \"cosx\"\ncosx_backend = \"cosx-a\"\n")
            .scf
            .cosx_config()
            .unwrap();
        assert!(c.screen_thresh.is_none());
        let c =
            parse("k_builder = \"cosx\"\ncosx_backend = \"cosx-a\"\ncosx_screen_thresh = 0.0\n")
                .scf
                .cosx_config()
                .unwrap();
        assert_eq!(c.screen_thresh, Some(0.0));
        // Explicit knobs are honoured.
        let c = parse("k_builder = \"cosx\"\ncosx_grid = { radial = 75, angular = 302 }\ncosx_overlap_fit = false\n")
            .scf
            .cosx_config()
            .unwrap();
        assert_eq!((c.grid.n_radial, c.grid.n_angular), (75, 302));
        assert!(!c.overlap_fit);
        // Backend: default md3c1e; both spellings resolve; anything else errors.
        use ferric_scf::cosx_k::CosxBackend;
        assert_eq!(c.backend, CosxBackend::Md3c1e);
        let c = parse("k_builder = \"cosx\"\ncosx_backend = \"cosx-a\"\n")
            .scf
            .cosx_config()
            .unwrap();
        assert_eq!(c.backend, CosxBackend::CosxA);
        let c = parse("k_builder = \"cosx\"\ncosx_backend = \"md3c1e\"\n")
            .scf
            .cosx_config()
            .unwrap();
        assert_eq!(c.backend, CosxBackend::Md3c1e);
        assert!(parse("k_builder = \"cosx\"\ncosx_backend = \"libint\"\n")
            .scf
            .cosx_config()
            .is_err());
        assert!(parse("k_builder = \"cosx\"\ncosx_backend = \"cosx_a\"\n")
            .scf
            .cosx_config()
            .is_err());
        // Dead-knob refusal.
        assert!(parse("cosx_overlap_fit = false\n")
            .scf
            .cosx_config()
            .is_err());
        assert!(parse("cosx_backend = \"md3c1e\"\n")
            .scf
            .cosx_config()
            .is_err());
        assert!(
            parse("k_builder = \"link\"\ncosx_grid = { radial = 50, angular = 110 }\n")
                .scf
                .cosx_config()
                .is_err()
        );
        // Untabulated angular order is a typed error, not a panic.
        assert!(
            parse("k_builder = \"cosx\"\ncosx_grid = { radial = 50, angular = 194 }\n")
                .scf
                .cosx_config()
                .is_err()
        );
        // Typo inside the inline table hard-errors at parse time.
        let s = "[molecule]\nxyz = \"w.xyz\"\n[basis]\nname = \"sto-3g\"\n[method]\nkind = \"rhf\"\n[scf]\nk_builder = \"cosx\"\ncosx_grid = { radial = 50, angulr = 110 }\n";
        assert!(toml::from_str::<Config>(s).is_err());
    }

    /// `[mp2] lmp2_reference` (opt-in canonical reference for lmp2 and
    /// lmp2-direct): absent means OFF; `true`/`false` parse; a non-bool value
    /// and a misspelled key are hard errors (deny_unknown_fields), never a
    /// silent default.
    ///
    /// Fails if reverted: with `lmp2_reference()` defaulting to true (the old
    /// always-on behaviour) the first assert fails; if the field were removed
    /// the `lmp2_reference = true` document would stop parsing; if the type
    /// were loosened to a string the `"yes"` case would parse.
    #[test]
    fn lmp2_reference_is_an_opt_in_strict_bool() {
        let base = "[molecule]\nxyz = \"w.xyz\"\n[basis]\nname = \"6-31g\"\n\
                    [method]\nkind = \"lmp2-direct\"\n[mp2]\nauxbasis = \"cc-pvdz-ri\"\n";
        let absent: Config = toml::from_str(base).unwrap();
        assert!(
            !absent.mp2.lmp2_reference(),
            "the canonical reference must be OFF when the key is absent"
        );
        let on: Config = toml::from_str(&format!("{base}lmp2_reference = true\n")).unwrap();
        assert!(on.mp2.lmp2_reference());
        let off: Config = toml::from_str(&format!("{base}lmp2_reference = false\n")).unwrap();
        assert!(!off.mp2.lmp2_reference());
        // a non-bool value is a parse error, not a coerced default
        for bad in ["\"yes\"", "\"true\"", "1"] {
            assert!(
                toml::from_str::<Config>(&format!("{base}lmp2_reference = {bad}\n")).is_err(),
                "lmp2_reference = {bad} must not parse"
            );
        }
        // a typo'd key errors and names itself
        let err = match toml::from_str::<Config>(&format!("{base}lmp2_referense = true\n")) {
            Ok(_) => panic!("typo'd lmp2_reference key parsed — deny_unknown_fields regressed"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("lmp2_referense"),
            "error should name the bad key: {err}"
        );
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
        assert!(
            err.contains("trunc_threshold"),
            "error should name the bad key: {err}"
        );
    }

    /// `[dft] dispersion` parses strictly: the accepted spellings resolve, and
    /// everything else is a hard error rather than a silent no-op.
    ///
    /// The last case is the load-bearing one for this repo's conventions:
    /// there must be NO value that means "compute a zero correction", because
    /// a reported 0.0 dispersion is a physics claim, not an absence.
    #[test]
    fn dispersion_config_parses_strictly() {
        use super::DispersionRequest;

        // "d3bj" takes the running functional's parameters.
        assert_eq!(
            DispersionRequest::parse_config_str("d3bj", Some("PBE")).unwrap(),
            DispersionRequest::D3Bj {
                functional: "PBE".to_string()
            }
        );
        // Case-insensitive, and the "d3(bj)" spelling is accepted too.
        assert!(DispersionRequest::parse_config_str("D3BJ", Some("PBE")).is_ok());
        assert!(DispersionRequest::parse_config_str("d3(bj)", Some("PBE")).is_ok());

        // An explicit functional overrides the running one.
        assert_eq!(
            DispersionRequest::parse_config_str("d3bj(b3lyp)", Some("PBE")).unwrap(),
            DispersionRequest::D3Bj {
                functional: "b3lyp".to_string()
            }
        );

        // "d3bj" with no functional to fall back on must error, not guess.
        assert!(DispersionRequest::parse_config_str("d3bj", None).is_err());
        // An empty parenthesised name is an error, not an empty lookup.
        assert!(DispersionRequest::parse_config_str("d3bj()", Some("PBE")).is_err());
        // Unknown schemes error.
        for bad in ["d4", "xdm", "vv10", "yes", "true", "0", "none", "off"] {
            assert!(
                DispersionRequest::parse_config_str(bad, Some("PBE")).is_err(),
                "{bad:?} must be rejected; omitting the key is the only way to \
                 ask for no dispersion"
            );
        }
    }

    /// A `[dft] dispersion` key must actually reach `DftCfg` through the TOML
    /// parser. Without this, the strict parser above could be correct and still
    /// never be called, because `deny_unknown_fields` would reject the key.
    #[test]
    fn dispersion_key_is_accepted_by_the_toml_parser() {
        let body = "[molecule]\nxyz = \"m.xyz\"\n[basis]\nname = \"sto-3g\"\n\
                    [method]\nkind = \"ksdft\"\n[dft]\nfunctional = \"PBE\"\n\
                    dispersion = \"d3bj\"\n";
        let cfg: Config = toml::from_str(body).expect("[dft] dispersion must parse");
        assert_eq!(cfg.dft.dispersion.as_deref(), Some("d3bj"));

        // ... and omitting it leaves it None, which is what "no correction" is.
        let body_off = "[molecule]\nxyz = \"m.xyz\"\n[basis]\nname = \"sto-3g\"\n\
                        [method]\nkind = \"ksdft\"\n[dft]\nfunctional = \"PBE\"\n";
        let cfg_off: Config = toml::from_str(body_off).unwrap();
        assert_eq!(cfg_off.dft.dispersion, None);
    }

    /// `[dft] grid_prune` reaches the strict parser, and unknown values are a
    /// hard error rather than a silent flat grid.
    #[test]
    fn dft_grid_prune_parses_strictly() {
        use ferric_dft::prune::PruneScheme;

        let parse = |body: &str| -> Config {
            toml::from_str(&format!(
                "[molecule]\nxyz = \"water.xyz\"\n[basis]\nname = \"sto-3g\"\n\
                 [method]\nkind = \"ksdft\"\n[dft]\nfunctional = \"PBE\"\n{body}"
            ))
            .unwrap()
        };

        // Absent → None → the flat default grid.
        assert!(parse("").dft.grid_prune.is_none());

        // Present and valid → the scheme the SCF will use.
        let cfg = parse("grid_prune = \"nwchem\"\n");
        assert_eq!(
            PruneScheme::parse_config_str(cfg.dft.grid_prune.as_deref().unwrap()).unwrap(),
            Some(PruneScheme::NwchemLike)
        );

        // Explicit "none" is accepted and means the flat grid.
        let cfg = parse("grid_prune = \"none\"\n");
        assert_eq!(
            PruneScheme::parse_config_str(cfg.dft.grid_prune.as_deref().unwrap()).unwrap(),
            None
        );

        // Unknown value: the TOML parses (it is a String), but the strict
        // parser the CLI runs it through must REJECT it. A silent default here
        // would be exactly the config-dishonesty this convention forbids.
        let cfg = parse("grid_prune = \"sg1\"\n");
        assert!(
            PruneScheme::parse_config_str(cfg.dft.grid_prune.as_deref().unwrap()).is_err(),
            "an unrecognised grid_prune must be a hard error, not a silent flat grid"
        );
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

        let att = cfg.mp2.build_att_vv10_config(&water(), None).unwrap();
        assert!((att.r0_angstrom() - 1.00).abs() < 1e-12);
        // 1.00 A = 1.8897259886 Bohr; ~0.529 would mean the conversion inverted.
        assert!(
            (att.r0_bohr - 1.889_725_988_6).abs() < 1e-9,
            "got {}",
            att.r0_bohr
        );
        assert_eq!(att.vv10.b, 11.0);
        assert_eq!(att.vv10.c, 0.0089);
        assert_eq!(att.frozen_core, 1, "[mp2] frozen_core must thread through");
        assert_eq!(
            att.attenuator,
            ferric_mp2::att_vv10::AttVv10Attenuator::Terfc
        );
        // Eq. 11: the VV10 damping r0 MUST be the same r0 the MP2 half uses.
        match att.vv10_damping {
            ferric_dft::vv10::Vv10Damping::Terfc { r0_bohr, .. } => {
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
        let att = Mp2Cfg::default()
            .build_att_vv10_config(&water(), None)
            .unwrap();
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
        let att = mp2.build_att_vv10_config(&water(), None).unwrap();
        assert!((att.r0_angstrom() - 1.05).abs() < 1e-12);
        assert_eq!(att.vv10.b, 12.5);
        match att.vv10_damping {
            ferric_dft::vv10::Vv10Damping::Terfc { r0_bohr, .. } => assert_eq!(
                r0_bohr, att.r0_bohr,
                "damping r0 must follow mp2v_r0 (paper Eq. 11)"
            ),
            other => panic!("expected terfc damping, got {other:?}"),
        }
    }

    /// `mp2v_omega` is Å⁻¹ at the TOML boundary and Bohr⁻¹ internally, and it
    /// reaches BOTH halves of Eq. 11: the config's `omega` (MP2 attenuator)
    /// and, via `effective_vv10_damping`, the VV10 damping's seam sharpness.
    #[test]
    fn mp2_v_omega_converts_and_reaches_both_halves() {
        let mp2 = Mp2Cfg {
            mp2v_omega: Some(4.0),
            ..Mp2Cfg::default()
        };
        let att = mp2.build_att_vv10_config(&water(), None).unwrap();
        let expect_bohr_inv = 4.0 * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV;
        assert_eq!(att.omega, Some(expect_bohr_inv));
        // ~2.117 Bohr⁻¹; 7.56 would mean the conversion inverted.
        assert!(
            (expect_bohr_inv - 2.116_708_3).abs() < 1e-6,
            "got {expect_bohr_inv}"
        );
        match att.effective_vv10_damping().unwrap() {
            ferric_dft::vv10::Vv10Damping::Terfc {
                r0_bohr,
                omega_bohr_inv,
            } => {
                assert_eq!(r0_bohr, att.r0_bohr);
                assert_eq!(
                    omega_bohr_inv,
                    Some(expect_bohr_inv),
                    "the VV10 damping must carry the SAME omega as the MP2 attenuator"
                );
            }
            other => panic!("expected terfc damping, got {other:?}"),
        }
        // Omitted omega = the linked width, in both halves (byte-identical
        // pre-decoupling behavior).
        let linked = Mp2Cfg::default()
            .build_att_vv10_config(&water(), None)
            .unwrap();
        assert_eq!(linked.omega, None);
        assert!(matches!(
            linked.effective_vv10_damping().unwrap(),
            ferric_dft::vv10::Vv10Damping::Terfc {
                omega_bohr_inv: None,
                ..
            }
        ));
    }

    /// A decoupled omega on the erfc control arm is a hard error, not a silent
    /// no-op (the erfc arm's width is 1/(r0*sqrt(2)) by definition), and
    /// non-positive/non-finite omegas are rejected.
    #[test]
    fn mp2_v_omega_rejects_erfc_and_nonpositive() {
        let mp2 = Mp2Cfg {
            mp2v_omega: Some(2.0),
            mp2v_attenuator: Some("erfc".to_string()),
            ..Mp2Cfg::default()
        };
        let err = mp2.build_att_vv10_config(&water(), None).unwrap_err();
        assert!(err.contains("terfc attenuator only"), "got: {err}");

        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mp2 = Mp2Cfg {
                mp2v_omega: Some(bad),
                ..Mp2Cfg::default()
            };
            let err = mp2.build_att_vv10_config(&water(), None).unwrap_err();
            assert!(err.contains("mp2v_omega"), "omega={bad}: {err}");
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
            m.build_att_vv10_config(&water(), None)
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
                m.build_att_vv10_config(&water(), None).is_err(),
                "mp2v_r0 = {bad} must be rejected"
            );
        }
        let mut m = Mp2Cfg::default();
        m.mp2v_nlc_n_radial = Some(0);
        assert!(m.build_att_vv10_config(&water(), None).is_err());
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
        assert!(
            err.contains("mp2v_bb"),
            "error should name the bad key: {err}"
        );
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
        assert_eq!(cfg.gw.frozen_core, Some(FrozenCore::Count(1)));
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
        assert_eq!(cfg.gw.frozen_core, Some(FrozenCore::Count(0)));
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
        assert_eq!(cfg.gw.frozen_core, Some(FrozenCore::Count(0)));
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
        assert!(
            err.contains("gw-bse"),
            "error should name the bad value: {err}"
        );
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
        assert_eq!(cfg.mp2.frozen_core, FrozenCore::Count(0));
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
        assert_eq!(
            mk(Some("boys")).unwrap(),
            Chi0Sparsity::BoysScreened {
                thresh: 1e-4,
                dist_cutoff: INF
            }
        );
        assert_eq!(
            mk(Some("boys:1e-3")).unwrap(),
            Chi0Sparsity::BoysScreened {
                thresh: 1e-3,
                dist_cutoff: INF
            }
        );
        // boys/auto with an explicit `@<radius>` distance cutoff (Bohr).
        assert_eq!(
            mk(Some("boys:1e-3@12")).unwrap(),
            Chi0Sparsity::BoysScreened {
                thresh: 1e-3,
                dist_cutoff: 12.0
            }
        );
        assert_eq!(
            mk(Some("auto:24:5e-4@8")).unwrap(),
            Chi0Sparsity::Auto {
                boys_thresh: 5e-4,
                atom_cutoff: 24,
                dist_cutoff: 8.0
            }
        );
        // auto with defaults (cutoff 30, thresh 1e-4), explicit cutoff, explicit cutoff+thresh.
        assert_eq!(
            mk(Some("auto")).unwrap(),
            Chi0Sparsity::Auto {
                boys_thresh: 1e-4,
                atom_cutoff: 30,
                dist_cutoff: INF
            }
        );
        assert_eq!(
            mk(Some("auto:24")).unwrap(),
            Chi0Sparsity::Auto {
                boys_thresh: 1e-4,
                atom_cutoff: 24,
                dist_cutoff: INF
            }
        );
        assert_eq!(
            mk(Some("auto:24:5e-4")).unwrap(),
            Chi0Sparsity::Auto {
                boys_thresh: 5e-4,
                atom_cutoff: 24,
                dist_cutoff: INF
            }
        );
        // case-insensitive + whitespace tolerant.
        assert_eq!(
            mk(Some("  AUTO ")).unwrap(),
            Chi0Sparsity::Auto {
                boys_thresh: 1e-4,
                atom_cutoff: 30,
                dist_cutoff: INF
            }
        );
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

    /// (d) `[scf] df_guess` parses, defaults to `false`, and an explicit
    /// `true` + `df_guess_aux` round-trips. Unknown keys in `[scf]` must
    /// still hard-error (deny_unknown_fields is unaffected by the new
    /// fields).
    #[test]
    fn scf_df_guess_key_parses_and_defaults_off() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rimp2"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        // df_guess now defaults ON (matches Psi4's DF_SCF_GUESS). The FIELD
        // stays None (user said nothing); the ACCESSOR is what carries the
        // default, so df_increments can still take precedence over it.
        assert_eq!(
            cfg.scf.df_guess, None,
            "absent key must stay None, not be materialised"
        );
        assert!(
            cfg.scf.df_guess_enabled(),
            "df_guess must default to ENABLED"
        );
        assert!(cfg.scf.df_guess_aux.is_none());

        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rimp2"
[scf]
df_guess = true
df_guess_aux = "def2-universal-jkfit"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(
            cfg.scf.df_guess,
            Some(true),
            "explicit df_guess = true must be recorded as Some(true)"
        );
        assert_eq!(
            cfg.scf.df_guess_aux.as_deref(),
            Some("def2-universal-jkfit")
        );
        assert_eq!(
            cfg.scf.df_guess_aux_resolved().unwrap().as_deref(),
            Some("def2-universal-jkfit")
        );
    }

    /// A typo'd key inside `[scf]` (e.g. `df_gess`) must still hard-error --
    /// adding `df_guess`/`df_guess_aux` must not have loosened
    /// `deny_unknown_fields` on `ScfCfg`.
    #[test]
    fn scf_section_still_rejects_typod_keys_after_df_guess_addition() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rimp2"
[scf]
df_gess = true
"#;
        let err = match toml::from_str::<Config>(toml_str) {
            Ok(_) => {
                panic!("typo'd df_guess key parsed successfully — deny_unknown_fields regressed")
            }
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("df_gess"),
            "error should name the bad key: {err}"
        );
    }

    /// `df_guess_aux` set without `df_guess = true` is a silent-no-op knob
    /// under the project's config-honesty convention -- must be a hard error
    /// from `df_guess_aux_resolved`, not silently ignored.
    #[test]
    fn df_guess_aux_without_df_guess_is_rejected() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rimp2"
[scf]
df_guess_aux = "def2-universal-jkfit"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        // With df_guess defaulting ON, `df_guess_aux` alone is no longer a
        // silent no-op — it is read. The rejection now applies only when the
        // user EXPLICITLY turned df_guess off while still setting its aux.
        assert!(cfg.scf.df_guess_enabled());
        assert!(
            cfg.scf.df_guess_aux_resolved().is_ok(),
            "df_guess_aux alone is now honoured, since df_guess defaults on"
        );

        let toml_off = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rimp2"
[scf]
df_guess = false
df_guess_aux = "def2-universal-jkfit"
"#;
        let cfg_off: Config = toml::from_str(toml_off).unwrap();
        assert!(!cfg_off.scf.df_guess_enabled());
        assert!(
            cfg_off.scf.df_guess_aux_resolved().is_err(),
            "df_guess_aux set with an EXPLICIT df_guess=false must be rejected, not silently ignored"
        );
    }

    /// `[scf] df_increments` parses, defaults to `false`, and an explicit
    /// `[scf] check_stability` must parse, default OFF, and round-trip `true`.
    ///
    /// `deny_unknown_fields` means a key that exists in `RhfConfig` but not in
    /// `ScfCfg` is unreachable from TOML while a key present in neither is a
    /// hard error, so a new config field needs BOTH halves and a test that the
    /// halves meet. The default matters as much as the parse: the check costs
    /// a Davidson eigensolve of J/K builds, so defaulting it ON would silently
    /// slow every shipped example.
    #[test]
    fn scf_check_stability_key_parses_and_defaults_off() {
        let base = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rhf"
"#;
        let cfg: Config = toml::from_str(base).unwrap();
        assert!(
            !cfg.scf.check_stability,
            "check_stability must default to false -- it costs extra J/K builds"
        );

        let with_key = format!("{base}[scf]\ncheck_stability = true\n");
        let cfg: Config = toml::from_str(&with_key).unwrap();
        assert!(
            cfg.scf.check_stability,
            "[scf] check_stability = true must round-trip"
        );

        // And the key must genuinely reach the solver config, not just parse.
        // (A field that parses into ScfCfg and is never copied into RhfConfig
        // is exactly as useless as one that does not parse.)
        let with_key = format!("{base}[scf]\ncheck_stability = false\n");
        let cfg: Config = toml::from_str(&with_key).unwrap();
        assert!(!cfg.scf.check_stability);
    }

    /// `true` + `df_increments_aux` round-trips. Mirrors
    /// `scf_df_guess_key_parses_and_defaults_off`.
    #[test]
    fn scf_df_increments_key_parses_and_defaults_off() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rimp2"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!(
            !cfg.scf.df_increments,
            "df_increments must default to false"
        );
        assert!(cfg.scf.df_increments_aux.is_none());

        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rimp2"
[scf]
df_increments = true
df_increments_aux = "def2-universal-jkfit"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!(cfg.scf.df_increments);
        assert_eq!(
            cfg.scf.df_increments_aux.as_deref(),
            Some("def2-universal-jkfit")
        );
        assert_eq!(
            cfg.scf.df_increments_aux_resolved().unwrap().as_deref(),
            Some("def2-universal-jkfit")
        );
    }

    /// REGRESSION (caught by CI, not by local runs): once `df_guess` defaults
    /// ON, a user who sets only `df_increments = true` must NOT be rejected by
    /// the df_guess/df_increments mutual-exclusion check. With a plain `bool`
    /// field the defaulted `df_guess` was indistinguishable from an explicit
    /// one, so `df_increments_aux_resolved()` errored and the feature was
    /// unusable without also writing `df_guess = false`.
    #[test]
    fn df_increments_alone_is_not_blocked_by_the_df_guess_default() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rimp2"
[scf]
df_increments = true
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!(cfg.scf.df_increments);
        assert!(
            !cfg.scf.df_guess_enabled(),
            "df_increments must take precedence over a DEFAULTED df_guess (it runs its own \
             DF pre-stage), otherwise the two are redundant"
        );
        assert!(cfg.scf.df_increments_aux_resolved().is_ok());
        assert!(cfg.scf.df_guess_aux_resolved().is_ok());

        // But asking for BOTH explicitly is still a hard error — that is a
        // user requesting two mutually exclusive things, not a default
        // colliding with a request.
        let both = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rimp2"
[scf]
df_guess = true
df_increments = true
"#;
        let cfg_both: Config = toml::from_str(both).unwrap();
        assert!(cfg_both.scf.df_increments_aux_resolved().is_err());
        assert!(cfg_both.scf.df_guess_aux_resolved().is_err());
    }

    /// `df_increments_aux` set without `df_increments = true` is a silent
    /// no-op knob otherwise -- rejected by `df_increments_aux_resolved`.
    #[test]
    fn df_increments_aux_without_df_increments_is_rejected() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rimp2"
[scf]
df_increments_aux = "def2-universal-jkfit"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!(!cfg.scf.df_increments);
        assert!(
            cfg.scf.df_increments_aux_resolved().is_err(),
            "df_increments_aux set with df_increments=false must be rejected, not silently ignored"
        );
    }

    /// `df_guess = true` and `df_increments = true` together must be a hard
    /// error, not a silent "one wins" resolution -- `df_increments` already
    /// runs its own internal DF-guess pre-stage, so composing both would be
    /// ambiguous about which mechanism actually governs the run.
    #[test]
    fn df_guess_and_df_increments_together_is_rejected() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rimp2"
[scf]
df_guess = true
df_increments = true
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!(cfg.scf.df_guess_aux_resolved().is_err());
        assert!(cfg.scf.df_increments_aux_resolved().is_err());
    }

    /// Adding `df_increments`/`df_increments_aux` must not have loosened
    /// `deny_unknown_fields` on `ScfCfg`.
    #[test]
    fn scf_section_still_rejects_typod_keys_after_df_increments_addition() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rimp2"
[scf]
df_incremnts = true
"#;
        assert!(
            toml::from_str::<Config>(toml_str).is_err(),
            "typo'd df_increments key parsed successfully — deny_unknown_fields regressed"
        );
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
        let built = cfg
            .scf
            .build_ladder(&ferric_scf::rhf::RhfConfig::default())
            .unwrap();
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
        let built = cfg.build_ladder(&base).unwrap();
        let expected = ferric_scf::ladder::ksdft_ladder(&base);
        assert_eq!(built.len(), expected.len());
        // ksdft_ladder's rung 0 honors the caller's own max_iter (100 here);
        // default_ladder_from would clamp rung 0 to 60 regardless.
        assert_eq!(built[0].config.max_iter, 100,
            "ksdft rung 0 must honor the caller's own max_iter, not default_ladder_from's hardcoded 60");
        for (i, rung) in built.iter().enumerate() {
            assert_eq!(
                rung.config.xc.as_deref(),
                Some("B3LYP"),
                "rung {i} must carry xc"
            );
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
        let built = cfg.build_ladder(&base).unwrap();
        assert_eq!(built.len(), ferric_scf::ladder::default_ladder().len());
        for (i, rung) in built.iter().enumerate() {
            assert_eq!(
                rung.config.mom_after_iter, 5,
                "rung {i} must inherit base.mom_after_iter"
            );
            // RI-JK is opt-in: the ladder escalates convergence knobs only and
            // must never silently swap exact 4-index J/K for density fitting
            // (that changes the method, ~1e-4 Ha). The base here leaves the aux
            // unset, so every rung must too.
            assert!(
                rung.config.df_j_aux.is_none(),
                "rung {i} must not inject DF-J aux"
            );
            assert!(
                rung.config.df_k_aux.is_none(),
                "rung {i} must not inject DF-K aux"
            );
        }
        assert_eq!(built[0].config.level_shift, 0.0);
        assert_eq!(
            built[1].config.level_shift, 0.0,
            "rung 1 adds ADIIS, not level shift yet"
        );
        assert!(
            built[2].config.level_shift > 0.0,
            "rung 2 must add level shift"
        );
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
        let built = cfg.build_ladder(&base).unwrap();
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
        assert_eq!(
            cosmo.lebedev_order,
            ferric_scf::cosmo::DEFAULT_LEBEDEV_ORDER
        );
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
        assert!(
            result.is_err(),
            "typo'd cosmo key should fail to parse, not silently default"
        );
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
        let built = cfg.scf.build_ladder(&base).unwrap();
        assert_eq!(built.len(), 1);
        assert_eq!(built[0].config.level_shift, 0.3);
        assert_eq!(built[0].config.max_iter, 42);
        assert!(!built[0].config.use_sad_guess);
        assert!(!built[0].restart);
    }

    // ---- frozen_core: the `"auto"` surface -------------------------------

    /// Minimal TOML with `body` appended, for the frozen-core cases.
    fn cfg_with(body: &str) -> Result<Config, toml::de::Error> {
        toml::from_str(&format!(
            "[molecule]\nxyz = \"water.xyz\"\n[basis]\nname = \"cc-pvdz\"\n\
             [method]\nkind = \"rimp2\"\n{body}"
        ))
    }

    #[test]
    fn frozen_core_accepts_auto_in_every_correlation_section() {
        let cfg = cfg_with("[mp2]\nfrozen_core = \"auto\"\n[rpa]\nfrozen_core = \"auto\"\n[gw]\nfrozen_core = \"auto\"\n")
            .unwrap();
        assert_eq!(cfg.mp2.frozen_core, FrozenCore::Auto);
        assert_eq!(cfg.rpa.frozen_core, FrozenCore::Auto);
        assert_eq!(cfg.gw.frozen_core, Some(FrozenCore::Auto));
    }

    #[test]
    fn frozen_core_still_accepts_a_plain_integer() {
        // The pre-existing surface: every input file in the wild writes a
        // number, and every one of them must keep parsing to that number.
        let cfg =
            cfg_with("[mp2]\nfrozen_core = 3\n[rpa]\nfrozen_core = 0\n[gw]\nfrozen_core = 5\n")
                .unwrap();
        assert_eq!(cfg.mp2.frozen_core, FrozenCore::Count(3));
        assert_eq!(cfg.rpa.frozen_core, FrozenCore::Count(0));
        assert_eq!(cfg.gw.frozen_core, Some(FrozenCore::Count(5)));
        assert_eq!(cfg.mp2.frozen_core.resolve(&water()), 3);
    }

    #[test]
    fn frozen_core_defaults_to_all_electron() {
        // Frozen core is OPT-IN. Flipping this default would silently change
        // every published ferric correlation energy, so it is a test, not a
        // comment.
        let cfg = cfg_with("").unwrap();
        assert_eq!(cfg.mp2.frozen_core, FrozenCore::Count(0));
        assert_eq!(cfg.rpa.frozen_core, FrozenCore::Count(0));
        assert_eq!(cfg.gw.frozen_core, None);
        assert_eq!(cfg.mp2.frozen_core.resolve(&water()), 0);
        assert!(!cfg.mp2.frozen_core.is_auto());
    }

    #[test]
    fn frozen_core_accepts_the_documented_spellings() {
        for (body, want) in [
            ("frozen_core = \"auto\"", FrozenCore::Auto),
            ("frozen_core = \"AUTO\"", FrozenCore::Auto), // case-insensitive
            ("frozen_core = \"  auto \"", FrozenCore::Auto), // trimmed
            ("frozen_core = true", FrozenCore::Auto),     // freeze_core = true elsewhere
            ("frozen_core = \"none\"", FrozenCore::Count(0)),
            ("frozen_core = false", FrozenCore::Count(0)),
        ] {
            let cfg = cfg_with(&format!("[mp2]\n{body}\n")).unwrap();
            assert_eq!(cfg.mp2.frozen_core, want, "[mp2] {body}");
        }
    }

    /// The error text from a body the parser must REJECT.
    ///
    /// Spelled out rather than `unwrap_err()` because `Config` has no `Debug`
    /// impl (and does not need one for its own sake).
    fn cfg_err(body: &str) -> String {
        match cfg_with(body) {
            Ok(_) => panic!("expected the parser to reject: {body:?}"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn frozen_core_rejects_an_unknown_spelling() {
        // The whole point of a strict parse: `"fc"` that silently meant 0
        // would correlate the core and change the energy with no diagnostic.
        let err = cfg_err("[mp2]\nfrozen_core = \"fc\"\n");
        assert!(err.contains("unknown value"), "{err}");
        assert!(
            err.contains("auto"),
            "the error must name the spelling we DO accept: {err}"
        );
    }

    #[test]
    fn frozen_core_rejects_a_negative_count() {
        let err = cfg_err("[mp2]\nfrozen_core = -1\n");
        assert!(err.contains(">= 0"), "{err}");
    }

    #[test]
    fn auto_frozen_core_resolves_against_the_molecule() {
        let cfg = cfg_with("[mp2]\nfrozen_core = \"auto\"\n").unwrap();
        // Water: O contributes its 1s, the two H nothing.
        assert_eq!(cfg.mp2.frozen_core.resolve(&water()), 1);
        // The same config on a different molecule gives a different count --
        // that is the entire point of "auto".
        let so2 = Molecule::parse_xyz(
            "3\nSO2\nS 0.0 0.0 0.0\nO 0.0 1.24 0.72\nO 0.0 -1.24 0.72\n",
            0,
            1,
        )
        .unwrap();
        assert_eq!(cfg.mp2.frozen_core.resolve(&so2), 5 + 1 + 1);
    }

    #[test]
    fn auto_frozen_core_follows_the_ecp() {
        // An ECP has already removed core orbitals from the MO space, so
        // "auto" must freeze only what is left -- freezing the full
        // small-core count would eat valence orbitals. Mirrors what
        // Molecule::apply_ecp does before the CLI resolves the key.
        let cfg = cfg_with("[mp2]\nfrozen_core = \"auto\"\n").unwrap();
        let mut hi = Molecule::parse_xyz("2\nHI\nI 0.0 0.0 0.0\nH 0.0 0.0 1.61\n", 0, 1).unwrap();
        assert_eq!(cfg.mp2.frozen_core.resolve(&hi), 23);
        hi.atoms[0].n_core_ecp = 28; // def2-ECP on iodine: 14 orbitals gone
        assert_eq!(cfg.mp2.frozen_core.resolve(&hi), 9);
    }

    #[test]
    fn validate_frozen_core_rejects_an_over_large_count() {
        // Water has 5 occupied orbitals; freezing 5 leaves nothing to
        // correlate. Caught at the config boundary, naming the section.
        let cfg = cfg_with("[mp2]\nfrozen_core = 5\n").unwrap();
        let err = cfg.validate_frozen_core(&water()).unwrap_err();
        assert!(err.contains("[mp2]"), "{err}");
        assert!(err.contains("nothing left to correlate"), "{err}");

        // ... and the same for the other two sections, so a stray key cannot
        // sail through just because the method family differs.
        let cfg = cfg_with("[rpa]\nfrozen_core = 9\n").unwrap();
        assert!(cfg
            .validate_frozen_core(&water())
            .unwrap_err()
            .contains("[rpa]"));
        let cfg = cfg_with("[gw]\nfrozen_core = 9\n").unwrap();
        assert!(cfg
            .validate_frozen_core(&water())
            .unwrap_err()
            .contains("[gw]"));
    }

    #[test]
    fn validate_frozen_core_names_auto_when_auto_is_at_fault() {
        // Li+ (2 electrons, 1 occupied orbital): the small-core convention
        // wants to freeze that one orbital. Better an error that says where
        // the number came from than a zero correlation energy.
        let cfg = cfg_with("[mp2]\nfrozen_core = \"auto\"\n").unwrap();
        let li_cation = Molecule::parse_xyz("1\nLi+\nLi 0.0 0.0 0.0\n", 1, 1).unwrap();
        let err = cfg.validate_frozen_core(&li_cation).unwrap_err();
        assert!(err.contains("auto"), "{err}");
    }

    #[test]
    fn validate_frozen_core_uses_the_minority_spin_count() {
        // CH3 radical (doublet): 9 electrons, 5 alpha / 4 beta occupied.
        // Freezing 4 leaves alpha with one correlated occupied but beta with
        // none -- the beta channel is the binding one.
        let ch3 = Molecule::parse_xyz(
            "4\nCH3\nC 0.0 0.0 0.0\nH 0.0 1.08 0.0\nH 0.94 -0.54 0.0\nH -0.94 -0.54 0.0\n",
            0,
            2,
        )
        .unwrap();
        assert!(cfg_with("[mp2]\nfrozen_core = 4\n")
            .unwrap()
            .validate_frozen_core(&ch3)
            .is_err());
        assert!(cfg_with("[mp2]\nfrozen_core = 1\n")
            .unwrap()
            .validate_frozen_core(&ch3)
            .is_ok());
    }

    #[test]
    fn validate_frozen_core_accepts_the_ordinary_cases() {
        for body in [
            "",
            "[mp2]\nfrozen_core = 0\n",
            "[mp2]\nfrozen_core = 1\n",
            "[mp2]\nfrozen_core = \"auto\"\n",
        ] {
            assert!(
                cfg_with(body)
                    .unwrap()
                    .validate_frozen_core(&water())
                    .is_ok(),
                "must accept: {body:?}"
            );
        }
        // A molecule with no electrons to freeze at all (H2) still passes with
        // the default, which is the `n_frozen > 0` guard doing its job.
        let h2 = Molecule::parse_xyz("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n", 0, 1).unwrap();
        assert!(cfg_with("[mp2]\nfrozen_core = \"auto\"\n")
            .unwrap()
            .validate_frozen_core(&h2)
            .is_ok());
        assert_eq!(
            cfg_with("[mp2]\nfrozen_core = \"auto\"\n")
                .unwrap()
                .mp2
                .frozen_core
                .resolve(&h2),
            0
        );
    }

    #[test]
    fn frozen_core_audit_line_is_printed_only_for_auto() {
        // An explicit count is already in the input file; an auto count is a
        // number nobody wrote down, so the run must report it.
        assert!(cfg_with("[mp2]\nfrozen_core = 1\n")
            .unwrap()
            .frozen_core_audit_lines(&water())
            .is_empty());
        let lines = cfg_with("[mp2]\nfrozen_core = \"auto\"\n[rpa]\nfrozen_core = \"auto\"\n")
            .unwrap()
            .frozen_core_audit_lines(&water());
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(
            lines[0].contains("[mp2]") && lines[0].contains('1'),
            "{lines:?}"
        );
        assert!(lines[1].contains("[rpa]"), "{lines:?}");
    }

    /// The `[optimize] coordinates` parser must be STRICT: an unrecognized
    /// value is an error, never a silent fall back to Cartesian. A user who
    /// typo'd the value asked for internals and must be told they did not get
    /// them — a silent default would make their iteration counts a mystery.
    #[test]
    fn optimize_coordinates_parser_is_strict() {
        use ferric_scf::optimize::CoordSystem;
        for good in ["cartesian", "CARTESIAN", " cart ", "Cart"] {
            assert_eq!(
                super::parse_coord_system(good).unwrap(),
                CoordSystem::Cartesian,
                "{good:?} should parse as Cartesian"
            );
        }
        for good in [
            "internal",
            "internals",
            "redundant-internal",
            "redundant_internal",
            "Internal",
        ] {
            assert_eq!(
                super::parse_coord_system(good).unwrap(),
                CoordSystem::RedundantInternal,
                "{good:?} should parse as RedundantInternal"
            );
        }
        // Near-misses and nonsense must ERROR, not silently default.
        for bad in [
            "",
            "internl",
            "redundant",
            "z-matrix",
            "delocalized",
            "true",
        ] {
            let r = super::parse_coord_system(bad);
            assert!(r.is_err(), "{bad:?} must be rejected, got {:?}", r.ok());
            let msg = r.unwrap_err();
            assert!(
                msg.contains("cartesian") && msg.contains("internal"),
                "the error must name the accepted values, got {msg:?}"
            );
        }
    }

    /// The `coordinates` key must actually reach the parsed [`Config`], and an
    /// absent key must leave it unset. Without this the strict parser above
    /// could be perfectly correct and simply never wired up.
    #[test]
    fn optimize_coordinates_key_is_accepted_by_the_toml_parser() {
        let cfg = cfg_with("[optimize]\ncoordinates = \"internal\"\n").unwrap();
        assert_eq!(cfg.optimize.coordinates.as_deref(), Some("internal"));

        let cfg = cfg_with("[optimize]\nmax_steps = 10\n").unwrap();
        assert_eq!(
            cfg.optimize.coordinates, None,
            "an absent key must stay None so the default applies"
        );

        // `deny_unknown_fields` must still reject a typo'd KEY name (as
        // opposed to a typo'd value, which `parse_coord_system` catches).
        assert!(
            cfg_with("[optimize]\ncoordinate = \"internal\"\n").is_err(),
            "a typo'd key must be rejected by deny_unknown_fields"
        );
    }
}

/// `[qmmm]` -- QM/MM embedding driven from TOML.
///
/// Until now QM/MM was reachable only from the Rust and Python APIs, so a CLI
/// user could not run an embedded calculation at all. This wires
/// [`ferric_scf::qmmm::QmmmSystem`] to the same `[method]` machinery every other
/// run uses: the QM region becomes the molecule that is solved, and the MM
/// region becomes the external potential it is solved in.
///
/// GEOMETRY AND CHARGES COME FROM A PQR, not from `[molecule].xyz`. An xyz has
/// no partial charges, and an MM region without charges is not an MM region --
/// it is a set of ignored coordinates. When `[qmmm]` is present, `pqr` supplies
/// BOTH the full-system geometry and the per-atom MM charges;
/// `[molecule].charge` and `.multiplicity` still apply, to the QM REGION.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QmmmCfg {
    /// PQR file: geometry (Angstrom) plus per-atom MM partial charges (e).
    /// PQR is the one common format carrying both, which is why it is the
    /// input here rather than xyz-plus-a-separate-charge-list.
    pub pqr: String,
    /// Zero-based indices of the atoms forming the QM region. Mutually
    /// exclusive with `qm_seeds`/`qm_radius_angstrom`.
    #[serde(default)]
    pub qm_indices: Vec<usize>,
    /// Seed atoms for a radial selection ("the ligand"). Requires
    /// `qm_radius_angstrom`.
    #[serde(default)]
    pub qm_seeds: Vec<usize>,
    /// Radius in ANGSTROM around `qm_seeds`, converted to Bohr internally.
    /// Stated in Angstrom because that is the unit a PQR is written in and the
    /// unit a pocket radius is quoted in.
    pub qm_radius_angstrom: Option<f64>,
    /// Bonds crossing the QM/MM boundary, as `[qm_atom, mm_atom]` index pairs.
    /// A covalent cut REQUIRES these -- without a link atom the QM region has
    /// a dangling valence.
    #[serde(default)]
    pub link_bonds: Vec<[usize; 2]>,
    /// Boundary-charge scheme for the `link_bonds` hosts: `"keep"`,
    /// `"delete-host"`, `"rc"` or `"rcd"`. Parsed by
    /// [`ferric_scf::qmmm::BoundaryChargeScheme::parse_config_str`], so an
    /// unknown value is a hard error rather than a silent default.
    ///
    /// **Defaults to `"delete-host"`, not `"keep"`.** Keeping the host charge
    /// puts a bare point charge inside the link atom's bond length (MEASURED
    /// 0.443 A on an ethane C-C cut) and a geometry optimization in that field
    /// then DIVERGES rather than failing loudly. `"keep"` stays selectable and
    /// is the right choice when the cut is not covalent.
    #[serde(default = "default_boundary_scheme")]
    pub boundary_scheme: String,
}

fn default_boundary_scheme() -> String {
    "delete-host".to_string()
}

/// One `ATOM`/`HETATM` record from a PQR file.
///
/// PQR is PDB with the occupancy and B-factor columns replaced by charge and
/// radius. It is whitespace-delimited in practice (the format has no strict
/// column spec once the charge/radius fields widen), which is how
/// `tools/active_site/pqr_parser.py` reads it and how this reader does too --
/// the two must agree, so they parse the same way.
#[derive(Debug, Clone)]
pub struct PqrAtom {
    pub name: String,
    pub q: f64,
    /// Angstrom, as written in the file.
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Element symbol from a PDB/PQR atom name (`CA` -> C, `HB2` -> H, `1HG1` -> H).
///
/// PDB naming puts a leading digit on some hydrogens, and the element is the
/// leading alphabetic run of what remains. Two-letter elements common in
/// biomolecular files are recognised explicitly; everything else takes the
/// first letter, which is right for C/N/O/S/P/H.
fn element_from_pqr_name(name: &str) -> String {
    let t = name.trim_start_matches(|c: char| c.is_ascii_digit());
    let alpha: String = t.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
    let upper = alpha.to_ascii_uppercase();
    for two in ["CL", "BR", "NA", "MG", "ZN", "FE", "CA", "MN", "CU", "SE"] {
        // `CA` is ambiguous: an alpha carbon in a protein, calcium as an ion.
        // A 2-character NAME that is exactly the symbol is the ion; `CA` as a
        // backbone atom appears in a residue with other backbone atoms and is
        // 2 characters too, so this is genuinely ambiguous in PQR and we
        // resolve it as CARBON, which is overwhelmingly the common case in a
        // protein file. An ion-heavy system needs an explicit element column,
        // which PQR does not have.
        if upper == two && two != "CA" {
            let mut c = two.chars();
            let first = c.next().unwrap();
            return format!("{first}{}", c.next().unwrap().to_ascii_lowercase());
        }
    }
    upper
        .chars()
        .next()
        .map(|c| c.to_string())
        .unwrap_or_default()
}

/// Parse the `ATOM`/`HETATM` records of a PQR file.
///
/// Deliberately strict: a record whose field count is not 10 is an ERROR, not
/// a skipped line. A silently dropped atom changes the MM field without
/// changing anything a user would look at.
pub fn parse_pqr(path: &str) -> Result<Vec<PqrAtom>, ferric_core::FerricError> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        ferric_core::FerricError::General(format!("[qmmm] pqr: cannot read {path}: {e}"))
    })?;
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if !(line.starts_with("ATOM") || line.starts_with("HETATM")) {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 10 {
            return Err(ferric_core::FerricError::General(format!(
                "[qmmm] pqr {path}:{}: expected 10 whitespace-separated fields \
                 (record serial name resName resSeq x y z charge radius), got {}: {line:?}",
                lineno + 1,
                f.len()
            )));
        }
        let num = |s: &str, what: &str| -> Result<f64, ferric_core::FerricError> {
            s.parse::<f64>().map_err(|_| {
                ferric_core::FerricError::General(format!(
                    "[qmmm] pqr {path}:{}: {what} {s:?} is not a number",
                    lineno + 1
                ))
            })
        };
        out.push(PqrAtom {
            name: f[2].to_string(),
            x: num(f[5], "x")?,
            y: num(f[6], "y")?,
            z: num(f[7], "z")?,
            q: num(f[8], "charge")?,
        });
    }
    if out.is_empty() {
        return Err(ferric_core::FerricError::General(format!(
            "[qmmm] pqr {path}: no ATOM/HETATM records"
        )));
    }
    Ok(out)
}

impl QmmmCfg {
    /// Build the `QmmmSystem` this section describes.
    ///
    /// `qm_charge`/`qm_multiplicity` come from `[molecule]` and apply to the QM
    /// REGION, not the whole structure -- the MM atoms carry their own partial
    /// charges and are not part of the SCF.
    pub fn to_system(
        &self,
        qm_charge: i32,
        qm_multiplicity: usize,
    ) -> Result<ferric_scf::qmmm::QmmmSystem, ferric_core::FerricError> {
        use ferric_core::FerricError;
        use ferric_scf::qmmm::{BoundaryChargeScheme, QmSelection, QmmmAtom, QmmmSystem};

        const ANGSTROM_TO_BOHR: f64 = 1.0 / 0.529_177_210_92;

        let have_indices = !self.qm_indices.is_empty();
        let have_radial = !self.qm_seeds.is_empty() || self.qm_radius_angstrom.is_some();
        if have_indices && have_radial {
            return Err(FerricError::General(
                "[qmmm]: give EITHER qm_indices OR qm_seeds+qm_radius_angstrom, not both -- \
                 two selections would silently disagree about which atoms are quantum"
                    .to_string(),
            ));
        }
        if !have_indices && !have_radial {
            return Err(FerricError::General(
                "[qmmm]: no QM region selected; set qm_indices, or qm_seeds with \
                 qm_radius_angstrom"
                    .to_string(),
            ));
        }

        let atoms_pqr = parse_pqr(&self.pqr)?;
        let n = atoms_pqr.len();
        // An atom name whose element cannot be resolved is a HARD ERROR, not
        // a Z = 0 atom. `unwrap_or(0)` used to be silent, and what it did
        // depended on where the atom landed: in the QM region the basis
        // lookup happened to catch it ("no basis shells for Z=0"), but in the
        // MM region nothing did -- MM atoms enter only through charge and
        // position, so a typo'd or unsupported element ran to completion and
        // printed an energy with no indication that a record was malformed.
        // Silently accepting one contradicts the strict-parse convention the
        // rest of this file follows.
        let atoms: Vec<QmmmAtom> = atoms_pqr
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let sym = element_from_pqr_name(&a.name);
                let z = ferric_core::elements::symbol_to_z(&sym).ok_or_else(|| {
                    FerricError::General(format!(
                        "[qmmm]: atom {i} in {} has name {:?}, whose element \
                         symbol {sym:?} is not a known element. PQR carries no \
                         element column, so the element is read from the atom \
                         name; rename the atom or use a format with an element \
                         column.",
                        self.pqr, a.name
                    ))
                })? as i32;
                Ok(QmmmAtom::new(
                    sym,
                    z,
                    a.x * ANGSTROM_TO_BOHR,
                    a.y * ANGSTROM_TO_BOHR,
                    a.z * ANGSTROM_TO_BOHR,
                    a.q,
                ))
            })
            .collect::<Result<Vec<_>, FerricError>>()?;

        let check = |label: &str, idx: usize| -> Result<(), FerricError> {
            if idx >= n {
                Err(FerricError::General(format!(
                    "[qmmm]: {label} index {idx} is out of range ({n} atoms in {})",
                    self.pqr
                )))
            } else {
                Ok(())
            }
        };
        for &i in &self.qm_indices {
            check("qm_indices", i)?;
        }
        for &i in &self.qm_seeds {
            check("qm_seeds", i)?;
        }

        let selection = if have_indices {
            QmSelection::Indices(self.qm_indices.clone())
        } else {
            let r = self.qm_radius_angstrom.ok_or_else(|| {
                FerricError::General("[qmmm]: qm_seeds needs qm_radius_angstrom".to_string())
            })?;
            if self.qm_seeds.is_empty() {
                return Err(FerricError::General(
                    "[qmmm]: qm_radius_angstrom needs qm_seeds to measure from".to_string(),
                ));
            }
            if !(r.is_finite() && r > 0.0) {
                return Err(FerricError::General(format!(
                    "[qmmm]: qm_radius_angstrom must be finite and > 0, got {r}"
                )));
            }
            QmSelection::WithinRadius {
                seeds: self.qm_seeds.clone(),
                radius: r * ANGSTROM_TO_BOHR,
            }
        };

        let mut sys = QmmmSystem::new(&atoms, selection, qm_charge, qm_multiplicity)?;

        if !self.link_bonds.is_empty() {
            for b in &self.link_bonds {
                check("link_bonds", b[0])?;
                check("link_bonds", b[1])?;
            }
            let bonds: Vec<(usize, usize)> = self.link_bonds.iter().map(|b| (b[0], b[1])).collect();
            sys = sys.with_link_atoms(&bonds, ferric_scf::qmmm::DEFAULT_LINK_SCALE)?;
            let scheme = BoundaryChargeScheme::parse_config_str(&self.boundary_scheme)?;
            sys = sys.with_boundary_charges(&bonds, scheme)?;
        } else if self.boundary_scheme != "delete-host" {
            return Err(FerricError::General(format!(
                "[qmmm]: boundary_scheme = {:?} but no link_bonds were given. A boundary \
                 charge scheme only acts on the hosts of a covalent cut, so this setting \
                 would do nothing -- which is worse than an error, because it reads as \
                 though it did something.",
                self.boundary_scheme
            )));
        }
        Ok(sys)
    }
}

#[cfg(test)]
#[path = "config_doc_tests.rs"]
mod doc_tests;
