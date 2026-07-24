//! Configuration types for PDEP-RPA.

/// Backend for the χ₀ kernel that powers the dielectric matrix.
///
/// `Dense` is the original O(naux² × nocc × nvir) MO-basis path used in
/// `dielectric_matrix_into`. `Laplace { n_quad }` factorizes the energy-gap
/// denominator into a sum of exponentials via minimax-Laplace quadrature; in
/// the MO-basis form it is correctness-equivalent to `Dense` (and the same
/// arithmetic complexity), but it admits an AO-basis cubic-scaling
/// reformulation as a follow-up.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum Chi0Backend {
    #[default]
    Dense,
    Laplace { n_quad: usize },
}

/// Sparsity strategy for the χ₀ build / dielectric matvec.
///
/// `Dense` (default) uses the canonical `(naux × nocc·nvir)` `b_ov` tensor.
///
/// `BoysScreened { thresh, dist_cutoff }` runs Foster-Boys on the active
/// occupied block, builds per-orbital `B^P_{i_loc, a}` tiles, and drops aux
/// rows P whose per-row L∞ norm `max_a |B^P_{i_loc, a}|` is below `thresh`. The
/// dielectric matvec then iterates over orbitals, gathering and scattering
/// through the per-orbital aux index lists.
///
/// `dist_cutoff` (Bohr) is the G6 centroid distance pre-filter length scale
/// `r_ref` (see `screen.rs`): for each localized orbital, aux shells and OBS
/// shell-pairs whose *Cauchy-Schwarz upper bound* — the same loose density-pair
/// bound the exact metric replaces — falls below `thresh` **after** a rigorous
/// `min(1, r_ref / R)` Coulomb-tail decay envelope (`R` = distance from the
/// aux/pair center to the Boys centroid) are skipped *before* their exact
/// `(P|i_loc i_loc)` integral is evaluated. Because `min(1, r_ref/R) ≤ 1`
/// always and the pre-decay factor is a genuine upper bound on `|p_ii[P]|`, a
/// shell is skipped only when its exact metric is provably ≤ `thresh` — the
/// same drop the exact-metric decision would make, just reached without
/// evaluating the integral. `dist_cutoff = f64::INFINITY` (the default) makes
/// the envelope ≡ 1 everywhere, so the retained set and every energy are
/// byte-for-byte identical to the pre-G6 code path.
///
/// Closed-shell only for now. Open-shell support is C8.
/// `Auto { boys_thresh, atom_cutoff, dist_cutoff }` picks Dense vs BoysScreened
/// by molecule size at runtime: below `atom_cutoff` atoms it resolves to `Dense`
/// (Boys localization + per-orbital tile overhead dominates the savings — see
/// the `boys-screening-crossover` finding: Boys is ~4× SLOWER than Dense at
/// benzene scale), and at/above the cutoff it resolves to `BoysScreened { thresh:
/// boys_thresh, dist_cutoff }` where locality wins kick in *on the topology the
/// crossover was measured on* (naphthalene, a dense/compact aromatic).
///
/// CAVEAT (2026-07-19): atom count alone does not predict whether
/// BoysScreened actually prunes anything — it is a proxy for locality, not
/// locality itself. A re-check on an extended alkane chain (32 atoms, just
/// above this cutoff) found the exact `|p_ii|` metric retains ~92% of aux
/// rows per orbital at `thresh` in the 1e-4..1e-3 range — i.e. essentially no
/// sparsity — so BoysScreened ran ~5x *slower* than Dense there despite being
/// past the atom-count threshold (see `docs/quickstart.md`'s chi0_sparsity
/// guidance section for the full writeup). `Auto` cannot detect this failure
/// mode; it only counts atoms. This is why `Chi0Sparsity::default()` is
/// `Dense`, not `Auto` — `Auto`/`BoysScreened` remain deliberately opt-in
/// (`chi0_sparsity = "auto"`/`"boys"` in TOML, or the equivalent Python
/// string), not something a caller falls into by omission.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum Chi0Sparsity {
    #[default]
    Dense,
    BoysScreened { thresh: f64, dist_cutoff: f64 },
    Auto { boys_thresh: f64, atom_cutoff: usize, dist_cutoff: f64 },
}

impl Chi0Sparsity {
    /// Resolve a (possibly `Auto`) sparsity choice to a concrete `Dense` or
    /// `BoysScreened` given the molecule's atom count. `Dense`/`BoysScreened`
    /// are returned unchanged; only `Auto` consults `natoms`.
    pub fn resolve(self, natoms: usize) -> Chi0Sparsity {
        match self {
            Chi0Sparsity::Auto { boys_thresh, atom_cutoff, dist_cutoff } => {
                if natoms >= atom_cutoff {
                    Chi0Sparsity::BoysScreened { thresh: boys_thresh, dist_cutoff }
                } else {
                    Chi0Sparsity::Dense
                }
            }
            other => other,
        }
    }

    /// Parse a config string into a `Chi0Sparsity`. Shared by the CLI (TOML) and
    /// the Python bindings so both accept identical syntax (case-insensitive,
    /// whitespace-trimmed). An optional `@<radius_bohr>` suffix on the `boys`
    /// and `auto` forms sets the G6 centroid distance pre-filter length scale
    /// `dist_cutoff` (Bohr); omitting it leaves `dist_cutoff = ∞` (the filter is
    /// a no-op and the retained set is byte-identical to the pre-G6 path):
    ///   None / "dense"                 → Dense (default; backward compatible)
    ///   "boys"                         → BoysScreened { thresh: 1e-4, dist: ∞ }
    ///   "boys:<thresh>"                → BoysScreened with that threshold, dist ∞
    ///   "boys:<thresh>@<radius>"       → …and that distance-cutoff radius (Bohr)
    ///   "boys@<radius>"                → default threshold, that radius
    ///   "auto"                         → Auto { cutoff: 30, thresh: 1e-4, dist ∞ }
    ///   "auto:<cutoff>"                → Auto with that atom cutoff
    ///   "auto:<cutoff>:<thresh>"       → …and that Boys threshold
    ///   "auto:<cutoff>:<thresh>@<rad>" → …and that distance-cutoff radius (Bohr)
    pub fn parse_config_str(s: Option<&str>) -> Result<Chi0Sparsity, String> {
        const DEF_THRESH: f64 = 1e-4;
        const DEF_CUTOFF: usize = 30;
        let raw = match s {
            None => return Ok(Chi0Sparsity::Dense),
            Some(s) => s.trim().to_ascii_lowercase(),
        };
        // Split off an optional `@<radius>` distance-cutoff suffix (Bohr) that
        // may follow either the `boys` or the `auto` colon-forms. Absent → ∞.
        let (head, dist_cutoff) = match raw.split_once('@') {
            None => (raw.as_str(), f64::INFINITY),
            Some((h, r)) => {
                let r = r.trim();
                let radius = r.parse::<f64>().map_err(|_| {
                    format!("chi0_sparsity: invalid distance cutoff '{r}' (expected radius in Bohr)")
                })?;
                // Explicit NaN check + `<= 0.0` instead of `!(radius > 0.0)`: the
                // negated-comparison form is equivalent (NaN fails `> 0.0`, so its
                // negation is true) but clippy::neg_cmp_op_on_partial_ord flags it
                // as unclear on a partially-ordered type; this spells out the same
                // "reject NaN or non-positive" intent without tripping the lint.
                if radius.is_nan() || radius <= 0.0 {
                    return Err(format!(
                        "chi0_sparsity: distance cutoff must be a positive radius in Bohr, got '{r}'"
                    ));
                }
                (h, radius)
            }
        };
        // `@` only makes sense on boys/auto; reject it on dense (or a bare `@`).
        if dist_cutoff.is_finite() && (head == "dense" || head.is_empty()) {
            return Err(format!(
                "chi0_sparsity: distance cutoff '@' is only valid on the boys/auto forms, not '{head}'"
            ));
        }
        let parts: Vec<&str> = head.split(':').collect();
        match parts.as_slice() {
            ["dense"] => Ok(Chi0Sparsity::Dense),
            ["boys"] => Ok(Chi0Sparsity::BoysScreened { thresh: DEF_THRESH, dist_cutoff }),
            ["boys", t] => Ok(Chi0Sparsity::BoysScreened {
                thresh: t.parse::<f64>()
                    .map_err(|_| format!("chi0_sparsity: invalid boys threshold '{t}'"))?,
                dist_cutoff,
            }),
            ["auto"] => Ok(Chi0Sparsity::Auto {
                boys_thresh: DEF_THRESH,
                atom_cutoff: DEF_CUTOFF,
                dist_cutoff,
            }),
            ["auto", c] => Ok(Chi0Sparsity::Auto {
                boys_thresh: DEF_THRESH,
                atom_cutoff: c.parse::<usize>()
                    .map_err(|_| format!("chi0_sparsity: invalid auto cutoff '{c}'"))?,
                dist_cutoff,
            }),
            ["auto", c, t] => Ok(Chi0Sparsity::Auto {
                boys_thresh: t.parse::<f64>()
                    .map_err(|_| format!("chi0_sparsity: invalid auto threshold '{t}'"))?,
                atom_cutoff: c.parse::<usize>()
                    .map_err(|_| format!("chi0_sparsity: invalid auto cutoff '{c}'"))?,
                dist_cutoff,
            }),
            _ => Err(format!(
                "chi0_sparsity: unrecognized value '{raw}' \
                 (expected dense | boys[:thresh][@radius] | auto[:cutoff[:thresh]][@radius])"
            )),
        }
    }
}

/// Choice of subspace eigensolver for the PDEP dielectric matrix.
///
/// Lanczos is the default: it agrees with Davidson to machine precision on the
/// symmetric dielectric eigenproblem, is ~17% faster on aug-cc-pVTZ, and uses
/// less memory (a stacked Krylov basis + block-tridiagonal T, rather than a
/// growing projected dielectric). Davidson is retained for comparison/fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Eigensolver {
    #[default]
    Lanczos,
    Davidson,
}

/// Top-level PDEP-RPA configuration.
#[derive(Debug, Clone)]
pub struct PdepRpaConfig {
    /// Number of frozen core orbitals.
    pub frozen_core: usize,
    /// Truncate eigenpotentials whose |λ_α(0) − 1| ≤ trunc_thresh.
    pub trunc_thresh: f64,
    /// Maximum subspace size before restart. Applies to whichever
    /// [`Eigensolver`] is selected: Davidson's subspace cap, and (via
    /// `max_vecs / block_size`) the Lanczos outer-iteration cap. `0` = auto.
    pub eigensolver_max_vecs: usize,
    /// Eigenvalue convergence threshold for the static dielectric eigenproblem.
    /// Applies to whichever [`Eigensolver`] is selected — the default is
    /// [`Eigensolver::Lanczos`], not Davidson.
    pub eigensolver_conv_thresh: f64,
    pub quadrature: QuadratureConfig,
    pub sternheimer: SternheimerConfig,
    /// If true, also compute the full-basis RI-dRPA diagnostic energy (expensive).
    pub run_diagnostics: bool,
    /// Eigensolver backend for the static dielectric eigenproblem.
    pub eigensolver: Eigensolver,
    /// χ₀ kernel backend. Default `Dense` preserves legacy behavior.
    pub chi0_backend: Chi0Backend,
    /// χ₀ sparsity strategy. Default `Dense` preserves legacy behavior.
    pub chi0_sparsity: Chi0Sparsity,
    /// Optional resident-bytes ceiling for the RI 3-index transform and the
    /// Lanczos matvec panel. `None` → resolved via
    /// [`ferric_core::memory::resolve_budget_bytes`] (env override > auto
    /// 0.8×RAM > 2 GiB). Threaded into the `RiMp2Config` this driver builds.
    pub memory_budget_bytes: Option<usize>,
    /// If true, materialize the per-frequency full inverse-dielectric matrices
    /// `PdepRpaResult.inv_dielectric_freq` (nquad × M² dense stack, up to ~1.85 GB
    /// at dimer/aTZ scale). Only the GW self-energy (ferric-gw sigma/u_sigma),
    /// BSE, and the PDEP dynamic-polarizability property paths consume them; an
    /// energy-only RPA run leaves this `false` and never allocates the stack.
    /// Default `false` — set `true` at every call site that later reads
    /// `PdepRpaResult.inv_dielectric_freq`.
    pub need_inv_dielectric_freq: bool,
    /// If true, populate `PdepRpaResult.eigenvalues_freq` (the per-frequency
    /// dielectric eigenvalues) by DIAGONALIZING at every quadrature point.
    ///
    /// The correlation energy itself does NOT need them: it needs only
    /// `Σ_α [ln λ_α + (1 − λ_α)]`, which is identically
    /// `ln det(ε) + tr(I − ε)` and is computed by LU (`dgetrf`, ~2/3 n³) for
    /// about half the FLOPs of the eigenvalues-only divide-and-conquer
    /// `dsyevd` (~4/3 n³) — measured 3.1–6.4x on the isolated kernel at
    /// n = 64…1024, since dsyevd's tridiagonal reduction is memory-bound BLAS2
    /// while dgetrf is blocked BLAS3. This is what PySCF does
    /// (`pyscf/gw/rpa.py:83`).
    ///
    /// `true` keeps the historical eigenvalue path and output. Set `false` on
    /// energy-only runs to take the log-det fast path. It must stay `true`
    /// wherever `eigenvalues_freq` is actually read: `ferric-python` exports it,
    /// `mpi_rpa_freq_banding.rs` asserts serial-vs-MPI agreement row-for-row to
    /// 1e-11, and `ferric-gw` shape-checks it.
    ///
    /// Default `true` — preserves existing behavior for every caller that does
    /// not opt out, exactly like `need_inv_dielectric_freq` above but inverted
    /// (that one defaults off because its output is huge; this one defaults on
    /// because its output is small and historically always present).
    pub need_eigenvalues_freq: bool,
    /// Print one line per Lanczos/Davidson outer iteration to stdout while
    /// the static dielectric eigensolve runs (iteration number, block size,
    /// worst Ritz residual) — live progress for a long-running job, opt-in
    /// and additive. Default `false` (unchanged, silent-until-done output).
    /// Mirrors `ferric_scf::rhf::RhfConfig::verbose`; only the Lanczos path
    /// (the default `Eigensolver`) currently consumes this — see
    /// `lanczos::run_lanczos_seeded`.
    pub verbose: bool,
}

impl Default for PdepRpaConfig {
    fn default() -> Self {
        Self {
            frozen_core: 0,
            trunc_thresh: 1e-4,
            eigensolver_max_vecs: 0,
            eigensolver_conv_thresh: 1e-6,
            quadrature: QuadratureConfig::default(),
            sternheimer: SternheimerConfig::default(),
            run_diagnostics: false,
            eigensolver: Eigensolver::Lanczos,
            chi0_backend: Chi0Backend::Dense,
            chi0_sparsity: Chi0Sparsity::Dense,
            memory_budget_bytes: None,
            need_inv_dielectric_freq: false,
            // Default ON: `eigenvalues_freq` has historically always been
            // populated, so this preserves behavior for every caller that does
            // not explicitly opt out. Energy-only paths set it false to take
            // the LU log-det fast path.
            need_eigenvalues_freq: true,
            verbose: false,
        }
    }
}

/// Imaginary-frequency quadrature configuration.
#[derive(Debug, Clone)]
pub struct QuadratureConfig {
    pub scheme: QuadratureScheme,
    /// Number of quadrature points (default 20).
    pub n_points: usize,
    /// Domain scale parameter u₀ in Eₕ (default 0.5).
    ///
    /// Honoured by [`QuadratureScheme::GaussLegendre`] and
    /// [`QuadratureScheme::ChebyshevTan`]. **Ignored** by
    /// [`QuadratureScheme::MiniMax`], which derives u₀ from `n_points` via the
    /// literature-optimized table (`optimized_u0`). Callers that set `u0`
    /// alongside `MiniMax` should expect it to have no effect — see
    /// [`QuadratureScheme::honours_u0`].
    pub u0: f64,
}

impl Default for QuadratureConfig {
    fn default() -> Self {
        Self {
            scheme: QuadratureScheme::MiniMax,
            n_points: 20,
            u0: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuadratureScheme {
    /// Eshuis-Yarkony-Furche modified Gauss-Chebyshev (JCP 2010): bounded
    /// ω-range via the tan-map. Recommended whenever `Chi0Backend::Laplace`
    /// is used so the cosine-modulated Laplace quadrature stays in its safe
    /// regime (ω·t_max < π/2).
    ChebyshevTan,
    /// GL nodes with literature-optimized u₀ scale (Furche, JCP 2005).
    MiniMax,
    /// Gauss-Legendre nodes mapped to [0,∞) via ω = u₀(1+x)/(1−x).
    GaussLegendre,
}

impl QuadratureScheme {
    /// Parse a `[rpa] quadrature` TOML string / Python kwarg into a scheme.
    ///
    /// Accepted forms (case-insensitive, whitespace-trimmed):
    ///   None                              → `GaussLegendre` (back-compat default)
    ///   "gauss-legendre" | "gl"           → `GaussLegendre`
    ///   "minimax" | "mm"                  → `MiniMax`
    ///   "chebyshev-tan" | "chebyshev" | "ct" → `ChebyshevTan`
    ///
    /// Errors on any other string rather than silently falling back to
    /// Gauss-Legendre — a typo'd scheme used to run a *different* quadrature
    /// than requested with no diagnostic.
    pub fn parse_config_str(s: Option<&str>) -> Result<Self, String> {
        match s.map(|x| x.trim().to_ascii_lowercase()).as_deref() {
            None | Some("gauss-legendre") | Some("gauss_legendre") | Some("gl") => {
                Ok(QuadratureScheme::GaussLegendre)
            }
            Some("minimax") | Some("mini-max") | Some("mm") => Ok(QuadratureScheme::MiniMax),
            Some("chebyshev-tan") | Some("chebyshev_tan") | Some("chebyshev") | Some("ct") => {
                Ok(QuadratureScheme::ChebyshevTan)
            }
            Some(other) => Err(format!(
                "unknown quadrature scheme {other:?}; expected one of \
                 \"gauss-legendre\" (\"gl\"), \"minimax\" (\"mm\"), \
                 or \"chebyshev-tan\" (\"ct\")"
            )),
        }
    }

    /// Whether this scheme uses the caller-supplied [`QuadratureConfig::u0`].
    ///
    /// `MiniMax` derives u₀ from `n_points` internally and ignores the field;
    /// callers use this to warn instead of silently dropping a user setting.
    pub fn honours_u0(&self) -> bool {
        !matches!(self, QuadratureScheme::MiniMax)
    }
}

/// Sternheimer linear solver configuration.
#[derive(Debug, Clone)]
pub struct SternheimerConfig {
    pub max_iter: usize,
    pub conv_thresh: f64,
}

impl Default for SternheimerConfig {
    fn default() -> Self {
        Self { max_iter: 50, conv_thresh: 1e-8 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_resolves_dense_below_cutoff_boys_at_or_above() {
        let auto = Chi0Sparsity::Auto { boys_thresh: 1e-3, atom_cutoff: 30, dist_cutoff: f64::INFINITY };
        // Below the cutoff → Dense (Boys overhead dominates; boys-screening-crossover).
        assert_eq!(auto.resolve(12), Chi0Sparsity::Dense);
        assert_eq!(auto.resolve(29), Chi0Sparsity::Dense);
        // At/above the cutoff → BoysScreened with the configured threshold; the
        // distance cutoff is carried through unchanged.
        assert_eq!(auto.resolve(30), Chi0Sparsity::BoysScreened { thresh: 1e-3, dist_cutoff: f64::INFINITY });
        assert_eq!(auto.resolve(120), Chi0Sparsity::BoysScreened { thresh: 1e-3, dist_cutoff: f64::INFINITY });
        // A finite distance cutoff on Auto propagates into the resolved BoysScreened.
        let auto_r = Chi0Sparsity::Auto { boys_thresh: 1e-3, atom_cutoff: 30, dist_cutoff: 12.0 };
        assert_eq!(auto_r.resolve(30), Chi0Sparsity::BoysScreened { thresh: 1e-3, dist_cutoff: 12.0 });
    }

    #[test]
    fn parse_config_str_all_forms() {
        use Chi0Sparsity::*;
        const INF: f64 = f64::INFINITY;
        assert_eq!(Chi0Sparsity::parse_config_str(None).unwrap(), Dense);
        assert_eq!(Chi0Sparsity::parse_config_str(Some("dense")).unwrap(), Dense);
        assert_eq!(Chi0Sparsity::parse_config_str(Some("boys")).unwrap(), BoysScreened { thresh: 1e-4, dist_cutoff: INF });
        assert_eq!(Chi0Sparsity::parse_config_str(Some("boys:1e-3")).unwrap(), BoysScreened { thresh: 1e-3, dist_cutoff: INF });
        assert_eq!(Chi0Sparsity::parse_config_str(Some("auto")).unwrap(), Auto { boys_thresh: 1e-4, atom_cutoff: 30, dist_cutoff: INF });
        assert_eq!(Chi0Sparsity::parse_config_str(Some("auto:24")).unwrap(), Auto { boys_thresh: 1e-4, atom_cutoff: 24, dist_cutoff: INF });
        assert_eq!(Chi0Sparsity::parse_config_str(Some("auto:24:5e-4")).unwrap(), Auto { boys_thresh: 5e-4, atom_cutoff: 24, dist_cutoff: INF });
        assert_eq!(Chi0Sparsity::parse_config_str(Some("  AUTO ")).unwrap(), Auto { boys_thresh: 1e-4, atom_cutoff: 30, dist_cutoff: INF });
        assert!(Chi0Sparsity::parse_config_str(Some("frobnicate")).is_err());
        assert!(Chi0Sparsity::parse_config_str(Some("boys:nope")).is_err());
    }

    #[test]
    fn parse_config_str_distance_cutoff_suffix() {
        use Chi0Sparsity::*;
        // `@<radius>` sets dist_cutoff (Bohr) on both boys and auto forms.
        assert_eq!(
            Chi0Sparsity::parse_config_str(Some("boys@12")).unwrap(),
            BoysScreened { thresh: 1e-4, dist_cutoff: 12.0 }
        );
        assert_eq!(
            Chi0Sparsity::parse_config_str(Some("boys:1e-3@8.5")).unwrap(),
            BoysScreened { thresh: 1e-3, dist_cutoff: 8.5 }
        );
        assert_eq!(
            Chi0Sparsity::parse_config_str(Some("auto:24:5e-4@15")).unwrap(),
            Auto { boys_thresh: 5e-4, atom_cutoff: 24, dist_cutoff: 15.0 }
        );
        assert_eq!(
            Chi0Sparsity::parse_config_str(Some("AUTO@10")).unwrap(),
            Auto { boys_thresh: 1e-4, atom_cutoff: 30, dist_cutoff: 10.0 }
        );
        // Bad radii and misplaced `@` error rather than silently defaulting.
        assert!(Chi0Sparsity::parse_config_str(Some("boys@nope")).is_err());
        assert!(Chi0Sparsity::parse_config_str(Some("boys@-3")).is_err());
        assert!(Chi0Sparsity::parse_config_str(Some("boys@0")).is_err());
        assert!(Chi0Sparsity::parse_config_str(Some("dense@10")).is_err());
        assert!(Chi0Sparsity::parse_config_str(Some("@10")).is_err());
    }

    #[test]
    fn quadrature_parse_config_str_all_forms() {
        use QuadratureScheme::*;
        let p = QuadratureScheme::parse_config_str;
        assert_eq!(p(None).unwrap(), GaussLegendre);
        assert_eq!(p(Some("gauss-legendre")).unwrap(), GaussLegendre);
        assert_eq!(p(Some("gl")).unwrap(), GaussLegendre);
        assert_eq!(p(Some("minimax")).unwrap(), MiniMax);
        assert_eq!(p(Some("mm")).unwrap(), MiniMax);
        assert_eq!(p(Some("chebyshev-tan")).unwrap(), ChebyshevTan);
        assert_eq!(p(Some("chebyshev")).unwrap(), ChebyshevTan);
        assert_eq!(p(Some("ct")).unwrap(), ChebyshevTan);
        assert_eq!(p(Some("  MiniMax ")).unwrap(), MiniMax);
    }

    /// A typo'd scheme must ERROR, not silently run Gauss-Legendre. The old
    /// `_ => GaussLegendre` catch-all ran a different quadrature than the user
    /// asked for with no diagnostic.
    #[test]
    fn quadrature_typo_errors_instead_of_silent_gauss_legendre() {
        assert!(QuadratureScheme::parse_config_str(Some("minmax")).is_err());
        assert!(QuadratureScheme::parse_config_str(Some("cheby")).is_err());
        assert!(QuadratureScheme::parse_config_str(Some("frobnicate")).is_err());
    }

    #[test]
    fn only_minimax_ignores_u0() {
        assert!(QuadratureScheme::GaussLegendre.honours_u0());
        assert!(QuadratureScheme::ChebyshevTan.honours_u0());
        assert!(!QuadratureScheme::MiniMax.honours_u0());
    }

    #[test]
    fn resolve_is_identity_for_concrete_variants() {
        // Dense/BoysScreened ignore atom count — they resolve to themselves.
        assert_eq!(Chi0Sparsity::Dense.resolve(5), Chi0Sparsity::Dense);
        assert_eq!(Chi0Sparsity::Dense.resolve(500), Chi0Sparsity::Dense);
        let b = Chi0Sparsity::BoysScreened { thresh: 1e-4, dist_cutoff: f64::INFINITY };
        assert_eq!(b.resolve(5), b);
        assert_eq!(b.resolve(500), b);
    }
}
