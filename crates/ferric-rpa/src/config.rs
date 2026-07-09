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
/// `BoysScreened { thresh }` runs Foster-Boys on the active occupied block,
/// builds per-orbital `B^P_{i_loc, a}` tiles, and drops aux rows P whose
/// per-row L∞ norm `max_a |B^P_{i_loc, a}|` is below `thresh`. The
/// dielectric matvec then iterates over orbitals, gathering and scattering
/// through the per-orbital aux index lists.
///
/// Closed-shell only for now. Open-shell support is C8.
/// `Auto { boys_thresh, atom_cutoff }` picks Dense vs BoysScreened by molecule
/// size at runtime: below `atom_cutoff` atoms it resolves to `Dense` (Boys
/// localization + per-orbital tile overhead dominates the savings — see the
/// `boys-screening-crossover` finding: Boys is ~4× SLOWER than Dense at benzene
/// scale), and at/above the cutoff it resolves to `BoysScreened { thresh:
/// boys_thresh }` where locality wins kick in. The default cutoff (30 atoms) is
/// the conservative production line above the measured naphthalene-scale crossover.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum Chi0Sparsity {
    #[default]
    Dense,
    BoysScreened { thresh: f64 },
    Auto { boys_thresh: f64, atom_cutoff: usize },
}

impl Chi0Sparsity {
    /// Resolve a (possibly `Auto`) sparsity choice to a concrete `Dense` or
    /// `BoysScreened` given the molecule's atom count. `Dense`/`BoysScreened`
    /// are returned unchanged; only `Auto` consults `natoms`.
    pub fn resolve(self, natoms: usize) -> Chi0Sparsity {
        match self {
            Chi0Sparsity::Auto { boys_thresh, atom_cutoff } => {
                if natoms >= atom_cutoff {
                    Chi0Sparsity::BoysScreened { thresh: boys_thresh }
                } else {
                    Chi0Sparsity::Dense
                }
            }
            other => other,
        }
    }

    /// Parse a config string into a `Chi0Sparsity`. Shared by the CLI (TOML) and
    /// the Python bindings so both accept identical syntax (case-insensitive,
    /// whitespace-trimmed):
    ///   None / "dense"            → Dense (default; backward compatible)
    ///   "boys"                    → BoysScreened { thresh: 1e-4 }
    ///   "boys:<thresh>"           → BoysScreened with that threshold
    ///   "auto"                    → Auto { atom_cutoff: 30, boys_thresh: 1e-4 }
    ///   "auto:<cutoff>"           → Auto with that atom cutoff
    ///   "auto:<cutoff>:<thresh>"  → Auto with that cutoff and Boys threshold
    pub fn parse_config_str(s: Option<&str>) -> Result<Chi0Sparsity, String> {
        const DEF_THRESH: f64 = 1e-4;
        const DEF_CUTOFF: usize = 30;
        let raw = match s {
            None => return Ok(Chi0Sparsity::Dense),
            Some(s) => s.trim().to_ascii_lowercase(),
        };
        let parts: Vec<&str> = raw.split(':').collect();
        match parts.as_slice() {
            ["dense"] => Ok(Chi0Sparsity::Dense),
            ["boys"] => Ok(Chi0Sparsity::BoysScreened { thresh: DEF_THRESH }),
            ["boys", t] => Ok(Chi0Sparsity::BoysScreened {
                thresh: t.parse::<f64>()
                    .map_err(|_| format!("chi0_sparsity: invalid boys threshold '{t}'"))?,
            }),
            ["auto"] => Ok(Chi0Sparsity::Auto { boys_thresh: DEF_THRESH, atom_cutoff: DEF_CUTOFF }),
            ["auto", c] => Ok(Chi0Sparsity::Auto {
                boys_thresh: DEF_THRESH,
                atom_cutoff: c.parse::<usize>()
                    .map_err(|_| format!("chi0_sparsity: invalid auto cutoff '{c}'"))?,
            }),
            ["auto", c, t] => Ok(Chi0Sparsity::Auto {
                boys_thresh: t.parse::<f64>()
                    .map_err(|_| format!("chi0_sparsity: invalid auto threshold '{t}'"))?,
                atom_cutoff: c.parse::<usize>()
                    .map_err(|_| format!("chi0_sparsity: invalid auto cutoff '{c}'"))?,
            }),
            _ => Err(format!(
                "chi0_sparsity: unrecognized value '{raw}' \
                 (expected dense | boys[:thresh] | auto[:cutoff[:thresh]])"
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
    /// Maximum Davidson subspace size before restart.
    pub davidson_max_vecs: usize,
    /// Davidson eigenvalue convergence threshold.
    pub davidson_conv_thresh: f64,
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
}

impl Default for PdepRpaConfig {
    fn default() -> Self {
        Self {
            frozen_core: 0,
            trunc_thresh: 1e-4,
            davidson_max_vecs: 0,
            davidson_conv_thresh: 1e-6,
            quadrature: QuadratureConfig::default(),
            sternheimer: SternheimerConfig::default(),
            run_diagnostics: false,
            eigensolver: Eigensolver::Lanczos,
            chi0_backend: Chi0Backend::Dense,
            chi0_sparsity: Chi0Sparsity::Dense,
            memory_budget_bytes: None,
        }
    }
}

/// Imaginary-frequency quadrature configuration.
#[derive(Debug, Clone)]
pub struct QuadratureConfig {
    pub scheme: QuadratureScheme,
    /// Number of quadrature points (default 20).
    pub n_points: usize,
    /// Gauss-Legendre domain scale parameter u₀ in Eₕ (default 0.5).
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
        let auto = Chi0Sparsity::Auto { boys_thresh: 1e-3, atom_cutoff: 30 };
        // Below the cutoff → Dense (Boys overhead dominates; boys-screening-crossover).
        assert_eq!(auto.resolve(12), Chi0Sparsity::Dense);
        assert_eq!(auto.resolve(29), Chi0Sparsity::Dense);
        // At/above the cutoff → BoysScreened with the configured threshold.
        assert_eq!(auto.resolve(30), Chi0Sparsity::BoysScreened { thresh: 1e-3 });
        assert_eq!(auto.resolve(120), Chi0Sparsity::BoysScreened { thresh: 1e-3 });
    }

    #[test]
    fn parse_config_str_all_forms() {
        use Chi0Sparsity::*;
        assert_eq!(Chi0Sparsity::parse_config_str(None).unwrap(), Dense);
        assert_eq!(Chi0Sparsity::parse_config_str(Some("dense")).unwrap(), Dense);
        assert_eq!(Chi0Sparsity::parse_config_str(Some("boys")).unwrap(), BoysScreened { thresh: 1e-4 });
        assert_eq!(Chi0Sparsity::parse_config_str(Some("boys:1e-3")).unwrap(), BoysScreened { thresh: 1e-3 });
        assert_eq!(Chi0Sparsity::parse_config_str(Some("auto")).unwrap(), Auto { boys_thresh: 1e-4, atom_cutoff: 30 });
        assert_eq!(Chi0Sparsity::parse_config_str(Some("auto:24")).unwrap(), Auto { boys_thresh: 1e-4, atom_cutoff: 24 });
        assert_eq!(Chi0Sparsity::parse_config_str(Some("auto:24:5e-4")).unwrap(), Auto { boys_thresh: 5e-4, atom_cutoff: 24 });
        assert_eq!(Chi0Sparsity::parse_config_str(Some("  AUTO ")).unwrap(), Auto { boys_thresh: 1e-4, atom_cutoff: 30 });
        assert!(Chi0Sparsity::parse_config_str(Some("frobnicate")).is_err());
        assert!(Chi0Sparsity::parse_config_str(Some("boys:nope")).is_err());
    }

    #[test]
    fn resolve_is_identity_for_concrete_variants() {
        // Dense/BoysScreened ignore atom count — they resolve to themselves.
        assert_eq!(Chi0Sparsity::Dense.resolve(5), Chi0Sparsity::Dense);
        assert_eq!(Chi0Sparsity::Dense.resolve(500), Chi0Sparsity::Dense);
        let b = Chi0Sparsity::BoysScreened { thresh: 1e-4 };
        assert_eq!(b.resolve(5), b);
        assert_eq!(b.resolve(500), b);
    }
}
