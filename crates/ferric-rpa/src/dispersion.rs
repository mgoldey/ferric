//! Many-body-dispersion C6 coefficients (data-product track).
//!
//! Two sources of the per-atom dynamic polarizability α^A(iω):
//!   * [`ts_dynamic_polarizability`] — Tkatchenko-Scheffler single-pole London
//!     model (Phase 1).
//!   * `pdep_dynamic_polarizability` — PDEP-RPA imaginary-frequency SMW
//!     (Phase 2, added later).
//!
//! Both feed [`casimir_polder_c6`], which contracts α^A(iω):α^B(iω) over the
//! imaginary-frequency quadrature to give isotropic and anisotropic C6^{AB}.

pub mod free_atom_ref;
pub mod mbd;

pub use mbd::{mbd_dynamic_polarizability, mbd_energy};

use ndarray::Array2;

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;

use crate::config::PdepRpaConfig;
use crate::PdepRpaResult;

/// Per-atom dynamic polarizability on an imaginary-frequency quadrature grid.
#[derive(Debug, Clone)]
pub struct DynamicPolarizability {
    /// Imaginary-frequency nodes ω_k (a.u.).
    pub freqs: Vec<f64>,
    /// Casimir-Polder quadrature weights w_k (a.u.).
    pub weights: Vec<f64>,
    /// `per_atom[a][k]` = 3×3 tensor α^A_{ij}(iω_k), a.u. — the ATOM-CENTRED
    /// *intrinsic* atomic polarizability (operator r − R_A, charge-transfer
    /// excluded). Origin-independent and ~isotropic per atom; this is the
    /// correct object for atom-resolved C6 (TS/MBD convention). The molecular
    /// bond-axis anisotropy is NOT here — it is a coupled property.
    pub per_atom: Vec<Vec<[[f64; 3]; 3]>>,
    /// `molecular[k]` = 3×3 molecular α_{ij}(iω_k), a.u. — the global-origin
    /// total molecular polarizability (origin-independent for the response).
    /// This drives the molecular C6 total (DOSD-comparable); it is NOT the sum
    /// of the intrinsic per-atom tensors (the difference is inter-atomic
    /// coupling / charge transfer).
    pub molecular: Vec<[[f64; 3]; 3]>,
}

/// C6 result: the fundamental per-atom α(iω) plus derived pair coefficients.
#[derive(Debug, Clone)]
pub struct C6Result {
    pub per_atom_dynamic: DynamicPolarizability,
    /// Isotropic C6^{AB}, shape (N, N), a.u.
    pub c6_iso_pair: Array2<f64>,
    /// Anisotropic C6^{AB}_{ij}: `c6_aniso_pair[a][b]` = 3×3 tensor.
    pub c6_aniso_pair: Vec<Vec<[[f64; 3]; 3]>>,
    /// Molecular isotropic C6 from the molecular α(iω) (the DOSD-comparable
    /// total), computed independently of the per-atom pair matrix.
    pub c6_molecular_iso: f64,
}

/// Partition scheme for the per-atom decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DispersionPartition {
    #[default]
    Becke,
    Hirshfeld,
}

impl DispersionPartition {
    /// Parse a `[rpa] c6_partition` string. `None` yields `Ok(None)` so the
    /// caller can apply its source-dependent default (Hirshfeld for PDEP,
    /// Becke for TS). Unknown strings are an error — they used to fall through
    /// to that same default, silently producing a *different* per-atom
    /// decomposition than requested (per-atom α/C6 are partition-dependent by
    /// ~10×, so this changed numbers, not just performance).
    pub fn parse_config_str(s: Option<&str>) -> Result<Option<Self>, String> {
        match s.map(|x| x.trim().to_ascii_lowercase()).as_deref() {
            None => Ok(None),
            Some("becke") => Ok(Some(DispersionPartition::Becke)),
            Some("hirshfeld") => Ok(Some(DispersionPartition::Hirshfeld)),
            Some(other) => Err(format!(
                "unknown c6_partition {other:?}; expected \"becke\" or \"hirshfeld\""
            )),
        }
    }
}

/// Source of the dynamic polarizability α(iω) that feeds the Casimir-Polder C6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum C6Source {
    /// Tkatchenko-Scheffler single-pole model scaled by Hirshfeld volume ratios.
    #[default]
    Ts,
    /// True PDEP-RPA dynamic α(iω) evaluated on the RPA quadrature grid.
    Pdep,
    /// Many-body dispersion on top of the TS single-pole α (coupled-dipole).
    /// Known-bad for soft atoms — see the `mbd-does-not-fix-silicon` finding.
    Mbd,
}

impl C6Source {
    /// Parse a `[rpa] c6_source` string. Unknown values are an error; they
    /// previously fell through to the TS branch silently.
    pub fn parse_config_str(s: Option<&str>) -> Result<Self, String> {
        match s.map(|x| x.trim().to_ascii_lowercase()).as_deref() {
            None | Some("ts") => Ok(C6Source::Ts),
            Some("pdep") => Ok(C6Source::Pdep),
            Some("mbd") => Ok(C6Source::Mbd),
            Some(other) => Err(format!(
                "unknown c6_source {other:?}; expected \"ts\", \"pdep\", or \"mbd\""
            )),
        }
    }

    /// The per-atom partition used when `c6_partition` is unset. PDEP needs
    /// Hirshfeld for correct anisotropy (proatom sum rule); TS/MBD default to
    /// Becke, which only shapes `alpha_static` (volumes are always Hirshfeld).
    pub fn default_partition(&self) -> DispersionPartition {
        match self {
            C6Source::Pdep => DispersionPartition::Hirshfeld,
            C6Source::Ts | C6Source::Mbd => DispersionPartition::Becke,
        }
    }
}

/// Casimir-Polder contraction. SHARED SEAM between TS and PDEP-RPA sources.
///
/// ```text
///   C6^{AB}_{ij} = (3/π) Σ_k w_k α^A_{ij}(iω_k) α^B_{ij}(iω_k)
///   C6^{AB}_iso  = (3/π) Σ_k w_k α^A_iso(iω_k) α^B_iso(iω_k)
/// ```
/// where α_iso(iω_k) = (1/3) Tr α(iω_k).
pub fn casimir_polder_c6(dyn_pol: &DynamicPolarizability) -> C6Result {
    use std::f64::consts::PI;
    let natoms = dyn_pol.per_atom.len();
    let nfreq = dyn_pol.freqs.len();
    let pref = 3.0 / PI;

    // Precompute per-atom isotropic profiles α_iso^A(iω_k).
    let mut iso: Vec<Vec<f64>> = vec![vec![0.0; nfreq]; natoms];
    for a in 0..natoms {
        for k in 0..nfreq {
            let t = dyn_pol.per_atom[a][k];
            iso[a][k] = (t[0][0] + t[1][1] + t[2][2]) / 3.0;
        }
    }

    let mut c6_iso_pair = Array2::<f64>::zeros((natoms, natoms));
    let mut c6_aniso_pair: Vec<Vec<[[f64; 3]; 3]>> =
        vec![vec![[[0.0; 3]; 3]; natoms]; natoms];

    for a in 0..natoms {
        for b in 0..natoms {
            // Isotropic.
            let mut s_iso = 0.0;
            for k in 0..nfreq {
                s_iso += dyn_pol.weights[k] * iso[a][k] * iso[b][k];
            }
            c6_iso_pair[(a, b)] = pref * s_iso;

            // Anisotropic, element-wise.
            for i in 0..3 {
                for j in 0..3 {
                    let mut s = 0.0;
                    for k in 0..nfreq {
                        s += dyn_pol.weights[k]
                            * dyn_pol.per_atom[a][k][i][j]
                            * dyn_pol.per_atom[b][k][i][j];
                    }
                    c6_aniso_pair[a][b][i][j] = pref * s;
                }
            }
        }
    }

    // Molecular isotropic C6 from the molecular α(iω) — the DOSD-comparable
    // total, computed from the global-origin molecular response, NOT the
    // intrinsic per-atom pair sum (which omits inter-atomic coupling).
    let c6_molecular_iso = if dyn_pol.molecular.len() == nfreq {
        let mut s = 0.0;
        for k in 0..nfreq {
            let t = dyn_pol.molecular[k];
            let iso_mol = (t[0][0] + t[1][1] + t[2][2]) / 3.0;
            s += dyn_pol.weights[k] * iso_mol * iso_mol;
        }
        pref * s
    } else {
        // Fallback (e.g. molecular not populated): use the intrinsic pair sum.
        c6_iso_pair.sum()
    };

    C6Result {
        per_atom_dynamic: dyn_pol.clone(),
        c6_iso_pair,
        c6_aniso_pair,
        c6_molecular_iso,
    }
}

/// Tkatchenko-Scheffler dynamic per-atom polarizability via a single London
/// pole, with directional shape inherited from the static α^A tensor.
///
/// Inputs:
///   * `z`            — atomic numbers, length N.
///   * `vol_ratio`    — effective-volume ratio v_A / v_free[Z_A], length N.
///   * `alpha_static` — static per-atom α^A_{ij} tensors (a.u.), length N.
///   * `freqs`,`weights` — imaginary-frequency quadrature (a.u.).
///
/// For each atom:
/// ```text
///   α_iso_eff = ratio · alpha_free[Z]
///   C6_eff    = ratio² · c6_free[Z]
///   ω_A       = (4/3) C6_eff / α_iso_eff²        (single-pole identity)
///   α_iso(iω) = α_iso_eff / (1 + (ω/ω_A)²)
///   α_{ij}(iω) = α_iso(iω) · (α^static_{ij} / α^static_iso)   (shape)
/// ```
///
/// Atoms with Z outside the reference table fall back to using the static
/// tensor's own isotropic average for α_iso_eff with the H London frequency;
/// the result is still finite.
pub fn ts_dynamic_polarizability(
    z: &[usize],
    vol_ratio: &[f64],
    alpha_static: &[[[f64; 3]; 3]],
    freqs: &[f64],
    weights: &[f64],
) -> DynamicPolarizability {
    let natoms = z.len();
    let nfreq = freqs.len();
    let mut per_atom: Vec<Vec<[[f64; 3]; 3]>> =
        vec![vec![[[0.0; 3]; 3]; nfreq]; natoms];

    // Per-atom (α_eff, ω_A) — shared with the MBD path (mbd::ts_atom_params).
    let params = crate::dispersion::mbd::ts_atom_params(z, vol_ratio, alpha_static);

    for a in 0..natoms {
        let st = alpha_static[a];
        let st_iso = (st[0][0] + st[1][1] + st[2][2]) / 3.0;

        let (alpha_iso_eff, omega_a) = params[a];

        // Shape factor: static tensor normalized so its iso average is 1.
        let inv_st_iso = if st_iso.abs() > 1e-12 { 1.0 / st_iso } else { 0.0 };
        let mut shape = [[0.0_f64; 3]; 3];
        if inv_st_iso != 0.0 {
            for i in 0..3 {
                for j in 0..3 {
                    shape[i][j] = st[i][j] * inv_st_iso;
                }
            }
        } else {
            // Degenerate static tensor → isotropic shape.
            shape[0][0] = 1.0;
            shape[1][1] = 1.0;
            shape[2][2] = 1.0;
        }

        for (k, &w) in freqs.iter().enumerate() {
            let a_iso = alpha_iso_eff / (1.0 + (w / omega_a).powi(2));
            for i in 0..3 {
                for j in 0..3 {
                    per_atom[a][k][i][j] = a_iso * shape[i][j];
                }
            }
        }
    }

    // TS has no inter-atomic charge transfer (atomic London oscillators), so the
    // molecular α is exactly the sum of the per-atom tensors.
    let molecular: Vec<[[f64; 3]; 3]> = (0..nfreq)
        .map(|k| {
            let mut m = [[0.0; 3]; 3];
            for at in &per_atom {
                for i in 0..3 {
                    for j in 0..3 {
                        m[i][j] += at[k][i][j];
                    }
                }
            }
            m
        })
        .collect();

    DynamicPolarizability {
        freqs: freqs.to_vec(),
        weights: weights.to_vec(),
        per_atom,
        molecular,
    }
}

/// PDEP-RPA per-atom dynamic polarizability α^A(iω) (Phase 2 source).
///
/// Evaluates the per-atom polarizability tensors at the imaginary-frequency
/// quadrature nodes drawn from `cfg.quadrature` (the same Gauss-Legendre grid
/// the RPA correlation energy uses), so the resulting `weights` are the exact
/// Casimir-Polder weights for [`casimir_polder_c6`].
///
/// Unlike [`ts_dynamic_polarizability`], this is a genuine frequency-dependent
/// response: at ω=0 it reproduces the static per-atom α exactly, and it carries
/// the true RPA frequency dependence rather than a single-pole London model.
///
/// `partition` selects the atomic decomposition. Becke is the default and the
/// only fully frequency-dependent path today; `Hirshfeld` currently falls back
/// to Becke (a dedicated Hirshfeld-dynamic grid path is future work) and emits
/// no error so callers get a usable result.
#[allow(clippy::too_many_arguments)]
pub fn pdep_dynamic_polarizability(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
    partition: DispersionPartition,
    proatom: Option<&crate::properties::ProatomProvider>,
) -> Result<DynamicPolarizability, FerricError> {
    let (freqs, weights) = crate::quadrature::build_quadrature(&cfg.quadrature);

    // The Hirshfeld dynamic per-atom path is closed-shell only. For open-shell
    // references fall back to the Becke partition, which has a complete per-spin
    // (U) branch. The molecular total below is partition-independent, so only the
    // per-atom decomposition differs — and per-atom magnitudes are already
    // partition-sensitive (see per-atom-c6-status). CLAUDE.md documents this
    // "Hirshfeld (dynamic falls back to Becke)" behavior; this wires it.
    let is_closed = matches!(rhf.spin, ferric_scf::result::Spin::Restricted);
    let per_atom = if partition == DispersionPartition::Hirshfeld && is_closed {
        crate::properties::pdep_polarizability_hirshfeld_dynamic(
            mol, obs, obs_bs, dfbs, rhf, op, cfg, &freqs, proatom,
        )?
    } else {
        crate::properties::pdep_polarizability_becke_dynamic(
            mol, obs, obs_bs, dfbs, rhf, op, cfg, &freqs,
        )?
    };

    // Molecular (whole-system) α(iω) for the DOSD-comparable molecular C6 total.
    // Partition-independent; computed from the lab-frame molecular dipole. Now
    // supports open-shell via the spin-summed dielectric (per-spin g_σ), so
    // casimir_polder_c6 gets the correct molecular total rather than falling back
    // to the per-atom pair sum.
    let molecular =
        crate::properties::molecular_dynamic_polarizability(mol, obs, dfbs, rhf, op, cfg, &freqs)?;

    Ok(DynamicPolarizability {
        freqs,
        weights,
        per_atom,
        molecular,
    })
}

/// PDEP-truncated per-atom dynamic polarizability (spike/benchmark).
///
/// Replaces the full `naux × naux` SMW solve in `pdep_dynamic_polarizability`
/// with a rank-M projection onto the PDEP eigenbasis from a prior
/// `run_pdep_rpa` call.  M ≪ naux (typically 5-30% of modes survive
/// `trunc_thresh`), so each per-frequency step costs O(M · nov) instead of
/// O(naux² · nov + naux³).
///
/// # Formula
///
/// In the full SMW path at frequency ω:
/// ```text
///   ε̃(ω) = I + B̃ diag(4g(ω)) B̃ᵀ          (naux × naux)
///   w^d   = B̃ g(ω) μ^d                      (naux,)
///   y^d   = ε̃(ω)⁻¹ w^d                      (naux,)   ← O(naux³) solve
///   α_ij  = 4 μⁱ·g·μʲ − 16 wⁱ·yʲ
/// ```
///
/// In the PDEP-truncated path the M dominant eigenvectors V_α (from
/// Davidson/Lanczos) diagonalise ε̃(0) to λ_α(0).  At frequency ω the
/// dielectric projected into that subspace gives eigenvalues λ_α(ω) already
/// stored in `rpa.eigenvalues_freq`.  The inverse is diagonal:
/// ```text
///   ε̃(ω)⁻¹ ≈ V [diag 1/λ_α(ω)] Vᵀ  +  (I − VVᵀ)   (identity on null space)
/// ```
/// Substituting into the α formula and using Vᵀ B̃ g(ω) μ = p_α^d(ω):
/// ```text
///   α^A_ij(ω) ≈ 4 bare_ij(ω)
///             − Σ_α (4/λ_α(ω) − 4) · p^{A,i}_α(ω) · p^j_α(ω)
/// ```
/// where `p^{A,i}_α(ω) = Vᵀ_α B̃ (g(ω) ⊙ μ^{A,i})` and
///       `p^i_α(ω) = Vᵀ_α B̃ (g(ω) ⊙ μ^i)` (molecular sum).
///
/// The `(4/λ − 4)` factor is the correction relative to the bare (non-interacting)
/// response.  Modes with λ_α → 1 contribute nothing (dielectric → identity on
/// those modes); only the M modes with λ_α(0) > 1+trunc_thresh matter.
///
/// # Arguments
/// * `rpa` — already-computed `PdepRpaResult` (provides `eigenpotentials` and
///   `eigenvalues_freq`).  Its quadrature grid is reused as-is.
/// * `mol`, `obs`, `obs_bs` — molecule + orbital basis (for the Becke grid).
/// * `rhf` — SCF result (for MO coefficients; must match the one used for `rpa`).
/// * `partition` — Becke (default) or Hirshfeld (falls back to Becke).
#[allow(clippy::too_many_arguments)]
pub fn pdep_dynamic_polarizability_truncated(
    rpa: &PdepRpaResult,
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
    partition: DispersionPartition,
) -> Result<DynamicPolarizability, FerricError> {
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
    use ferric_mp2::rimp2::RiMp2Config;
    use ferric_scf::result::Spin;

    let _ = partition; // Becke only for now (same as full path)

    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "pdep_dynamic_polarizability_truncated: closed-shell only".into(),
        ));
    }

    // Re-use the quadrature grid from the prior RPA run.
    let freqs = rpa.quad_freqs.clone();
    let weights = rpa.quad_weights.clone();
    let nfreq = freqs.len();
    let natoms = mol.atoms.len();

    // PDEP eigenvectors V: (naux, M).  These are the physical-basis vectors
    // from `run_pdep_rpa` (back-transformed by V^{-1/2}).
    // BUT: the α formula uses the *dressed* Davidson basis (V^{-1/2}-dressed),
    // not the physical eigenpotentials.  `rpa.eigenpotentials` holds the physical
    // coefficients c_α^P = (V^{-1/2})_PP' U_αP'.  We need the dressed eigenvectors
    // U_α = V^{1/2} c_α.  Rather than reconstructing V^{1/2}, we re-derive
    // b_ov and use the SMW identity directly in terms of:
    //   p_α^d = Vᵀ_α w^d   where V_α is the dressed eigenvector and
    //   w^d = B̃ (g ⊙ μ^d)  lives in the *dressed* aux space.
    //
    // The cleanest approach: recompute B̃ (the V^{-1/2}-dressed RI tensor)
    // from the RI intermediates — that gives us the same dressed basis the
    // Davidson used. Then Vᵀ B̃ uses the *dressed* eigenvectors from Davidson,
    // which we recover as: dressed_V = V^{1/2} eigenpotentials
    // = V^{-1/2}^{-1} · physical = inter.v_inv_sqrt^{-1} · eigenpotentials.
    //
    // Simpler: recompute the RI intermediates (cheap — it's just a Cholesky
    // + transform) and use the dressed Davidson eigenvectors directly.
    // `run_pdep_rpa` back-transforms via  eigenpotentials_aux = v_inv_sqrt · eigenvectors,
    // so: dressed_U = v_inv_sqrt^{-1} · eigenpotentials_aux = eigenvectors (the Davidson output).
    // We don't store eigenvectors from Davidson after run_pdep_rpa; we only have
    // eigenpotentials (physical). Reconstruct dressed:
    //   dressed_V = V^{1/2} · eigenpotentials
    // where V^{1/2} = inv(v_inv_sqrt) — but we don't store v_inv_sqrt either.
    //
    // PRACTICAL WORKAROUND for the spike: recompute b_ov and use b_ov directly
    // with the PHYSICAL eigenpotentials (c_α).  The projection c_α^T B̃^P_ia
    // where B̃ = V^{-1/2} (P|ia) is what we need.  Since c_α = V^{-1/2} u_α,
    // c_αᵀ B̃ = u_αᵀ V^{-1/2} V^{-1/2} (P|ia) = u_αᵀ V^{-1} (P|ia).
    // That introduces V^{-1} which we also don't have.
    //
    // CLEANEST SPIKE: use the physical B_ov (without V^{-1/2} dressing) together
    // with the physical eigenpotentials, which is equivalent to working in the
    // un-dressed (original RI) basis.  The eigenpotentials c_α satisfy:
    //   ε̃_phys V_α = λ_α V_α  (in the physical RI metric)
    // where ε̃_phys = I + B_ov diag(4/Δε) B_ovᵀ  (un-dressed, using raw B_ov).
    //
    // In the physical basis the SMW inverse is:
    //   ε̃_phys^{-1} ≈ Σ_α (1/λ_α) c_α c_αᵀ / (c_αᵀ c_α) + complement
    // But c_α are normalised: c_αᵀ c_α = 1 (columns of eigenpotentials are
    // orthonormal in the RI metric, not L2).
    //
    // For the spike we treat c_α as orthonormal (approximately true since B̃ ≈ B
    // up to V^{1/2}). This gives a computable rank-M approximation we can
    // time and compare to the full solve.

    let mp2_cfg = RiMp2Config {
        frozen_core: cfg.frozen_core,
        memory_budget_bytes: cfg.memory_budget_bytes,
    };
    let inter = ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov;   // shape (naux, nov) — un-dressed raw RI
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let nov = nocc * nvir;
    let _naux = inter.naux;

    let eps = rhf.eps_r();
    let eps_occ: Vec<f64> = eps[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + nvir].to_vec();
    let mut e_ia = ndarray::Array1::<f64>::zeros(nov);
    for i in 0..nocc { for a in 0..nvir { e_ia[i*nvir+a] = eps_vir[a] - eps_occ[i]; } }

    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();

    // Becke grid + per-atom atom-centred AO dipoles (identical to full path).
    let grid_cfg = AtomicGridConfig::default();
    let grid = build_atomic_grid(mol, &grid_cfg);
    let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let wts: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let home_atom: Vec<usize> = grid.iter().map(|g| g.home_atom).collect();
    let npts = points.len();
    let chi = eval_basis_on_points(mol, obs_bs, &points).map_err(|e| {
        FerricError::General(format!("pdep_dynamic_polarizability_truncated: chi: {e}"))
    })?;
    let nbf = chi.nrows();
    let atom_pos: Vec<[f64; 3]> = mol.atoms.iter().map(|at| [at.x, at.y, at.zpos]).collect();

    let mut d_ai_ao: Vec<[Array2<f64>; 3]> = (0..natoms)
        .map(|_| std::array::from_fn(|_| Array2::<f64>::zeros((nbf, nbf)))).collect();
    for g in 0..npts {
        let a = home_atom[g]; let wg = wts[g]; let r = points[g]; let ra = atom_pos[a];
        for d in 0..3 {
            let factor = wg * (r[d] - ra[d]);
            for mu in 0..nbf {
                let wchi = factor * chi[(mu, g)];
                if wchi.abs() < 1e-30 { continue; }
                for nu in 0..nbf { d_ai_ao[a][d][(mu, nu)] += wchi * chi[(nu, g)]; }
            }
        }
    }
    for a in 0..natoms { for d in 0..3 {
        let m = &mut d_ai_ao[a][d];
        for i in 0..nbf { for j in (i+1)..nbf {
            let avg = 0.5*(m[(i,j)]+m[(j,i)]); m[(i,j)] = avg; m[(j,i)] = avg;
        }}
    }}

    // Per-atom MO dipoles + molecular sums.
    let mu_ai_mo: Vec<[Array2<f64>; 3]> = (0..natoms).map(|a|
        std::array::from_fn(|d| c_occ.t().dot(&d_ai_ao[a][d]).dot(&c_vir))
    ).collect();
    let mut mu_mo: [Array2<f64>; 3] = std::array::from_fn(|_| Array2::<f64>::zeros((nocc, nvir)));
    for a in 0..natoms { for d in 0..3 { mu_mo[d] = &mu_mo[d] + &mu_ai_mo[a][d]; } }

    let mu_flat: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        let mut v = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc { for ax in 0..nvir { v[i*nvir+ax] = mu_mo[d][(i,ax)]; } }
        v
    });
    let mu_ai_flat: Vec<[ndarray::Array1<f64>; 3]> = (0..natoms).map(|a|
        std::array::from_fn(|d| {
            let mut v = ndarray::Array1::<f64>::zeros(nov);
            for i in 0..nocc { for ax in 0..nvir { v[i*nvir+ax] = mu_ai_mo[a][d][(i,ax)]; } }
            v
        })
    ).collect();

    // Use the dressed (V^{-1/2}-basis) eigenvectors stored in PdepRpaResult.
    // b_ov = V^{-1/2} (P|ia) is also in the dressed basis, so:
    //   u_αᵀ B̃ = dressed_Uᵀ · b_ov   ← exact projection in the dressed basis.
    // This gives the correct PDEP rank-M approximation to the SMW solve.
    let evecs = &rpa.dressed_eigenvectors; // (naux, M) dressed
    let n_modes = evecs.ncols();

    // Precompute c_αᵀ B̃_ov: (M, nov)  [B̃_ov is the dressed tensor from inter]
    let ct_b: Array2<f64> = evecs.t().dot(b_ov); // (M, nov)

    // Frequency loop.
    // Cost per frequency: O(M·nov) for g-weighted projections + O(M³) for
    // M×M solve — vs O(naux²·nov + naux³) for the full path.
    //
    // PDEP-subspace approximation:
    //   ε̃_M(ω) = Uᵀ ε̃(ω) U   (M×M projection of the full dielectric)
    //   w_M^d   = Uᵀ w^d       (M-vector, U = dressed eigenvectors)
    //   y_M^d   = ε̃_M(ω)^{-1} w_M^d   (M×M solve)
    //   α_ij   = 4 bare_ij − 16 w_M^{i,T} y_M^j
    // This is exact within the M-dimensional PDEP subspace; modes outside
    // the subspace contribute only through the bare (non-interacting) term.
    let mut out: Vec<Vec<[[f64; 3]; 3]>> = vec![vec![[[0.0; 3]; 3]; nfreq]; natoms];

    for (k, &omega) in freqs.iter().enumerate() {
        let omega2 = omega * omega;
        let mut g = ndarray::Array1::<f64>::zeros(nov);
        for ia in 0..nov { let e = e_ia[ia]; g[ia] = e / (omega2 + e*e); }

        // B̃_g = B̃ diag(g): (naux, nov) → scale columns.
        // We compute Uᵀ B̃_g = (Uᵀ B̃) diag(g) = ct_b * diag(g) efficiently
        // as a column-scaled product.
        let ct_b_g: Array2<f64> = {
            let mut m = ct_b.clone(); // (M, nov)
            for ia in 0..nov { m.column_mut(ia).mapv_inplace(|x| x * g[ia]); }
            m
        };

        // ε̃_M(ω) = I_M + 4 (Uᵀ B̃_g) (Uᵀ B̃_g)ᵀ  [M×M SPD]
        // = I + 4 ct_b_g · ct_b_gᵀ
        let mut eps_m: Array2<f64> = ct_b_g.dot(&ct_b_g.t());
        eps_m.mapv_inplace(|x| x * 4.0);
        for alpha in 0..n_modes { eps_m[(alpha, alpha)] += 1.0; }

        // Molecular projected dipole: w_M^d = Uᵀ B̃_g μ^d = ct_b_g · μ^d
        let w_mol_m: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| ct_b_g.dot(&mu_flat[d]));
        // Solve ε̃_M y_M^d = w_M^d  (M×M, small)
        let y_mol_m = crate::properties::solve_dielectric_3(&eps_m, &w_mol_m)?;

        for a in 0..natoms {
            // Per-atom projected dipole: w_M^{A,d} = ct_b_g · μ^{A,d}
            let w_ai_m: [ndarray::Array1<f64>; 3] =
                std::array::from_fn(|d| ct_b_g.dot(&mu_ai_flat[a][d]));

            for d in 0..3 {
                for j in 0..3 {
                    let bare = mu_ai_flat[a][d].dot(&(&mu_flat[j] * &g));
                    let coupled = w_ai_m[d].dot(&y_mol_m[j]);
                    out[a][k][d][j] = 4.0 * bare - 16.0 * coupled;
                }
            }
            // Symmetrize.
            for i in 0..3 { for j in (i+1)..3 {
                let avg = 0.5*(out[a][k][i][j]+out[a][k][j][i]);
                out[a][k][i][j] = avg; out[a][k][j][i] = avg;
            }}
        }
    }

    // Truncated/benchmark path: molecular total not its concern; leave empty so
    // casimir_polder_c6 falls back to the per-atom pair sum.
    Ok(DynamicPolarizability { freqs, weights, per_atom: out, molecular: Vec::new() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c6_source_parses_all_documented_values() {
        let p = C6Source::parse_config_str;
        assert_eq!(p(None).unwrap(), C6Source::Ts);
        assert_eq!(p(Some("ts")).unwrap(), C6Source::Ts);
        assert_eq!(p(Some("pdep")).unwrap(), C6Source::Pdep);
        // "mbd" was read by the CLI but undocumented; it is now first-class.
        assert_eq!(p(Some("mbd")).unwrap(), C6Source::Mbd);
        assert_eq!(p(Some("  PDEP ")).unwrap(), C6Source::Pdep);
    }

    /// A typo'd source must ERROR, not silently compute TS C6 and label it
    /// as whatever the user asked for.
    #[test]
    fn c6_source_typo_errors_instead_of_silent_ts() {
        assert!(C6Source::parse_config_str(Some("tsx")).is_err());
        assert!(C6Source::parse_config_str(Some("rpa")).is_err());
    }

    #[test]
    fn c6_partition_parse_and_defaults() {
        let p = DispersionPartition::parse_config_str;
        assert_eq!(p(None).unwrap(), None);
        assert_eq!(p(Some("becke")).unwrap(), Some(DispersionPartition::Becke));
        assert_eq!(p(Some("hirshfeld")).unwrap(), Some(DispersionPartition::Hirshfeld));
        assert!(p(Some("mulliken")).is_err());
        // Source-dependent default when unset.
        assert_eq!(C6Source::Pdep.default_partition(), DispersionPartition::Hirshfeld);
        assert_eq!(C6Source::Ts.default_partition(), DispersionPartition::Becke);
        assert_eq!(C6Source::Mbd.default_partition(), DispersionPartition::Becke);
    }

    /// Fine trapezoid grid on [0, ωmax] for analytic Casimir-Polder checks.
    fn trapezoid_grid(n: usize, wmax: f64) -> (Vec<f64>, Vec<f64>) {
        let dw = wmax / (n as f64);
        let mut freqs = Vec::with_capacity(n + 1);
        let mut weights = Vec::with_capacity(n + 1);
        for k in 0..=n {
            freqs.push(k as f64 * dw);
            weights.push(if k == 0 || k == n { 0.5 * dw } else { dw });
        }
        (freqs, weights)
    }

    /// Single-pole isotropic α(iω) = α0/(1+(ω/ω0)²): Casimir-Polder must
    /// reproduce analytic C6 = (3/4) α0² ω0.
    #[test]
    fn casimir_polder_single_pole_analytic() {
        let alpha0 = 4.5_f64;
        let omega0 = 0.5_f64;
        let (freqs, weights) = trapezoid_grid(20000, 200.0);
        let per_atom: Vec<Vec<[[f64; 3]; 3]>> = vec![freqs
            .iter()
            .map(|&w| {
                let a = alpha0 / (1.0 + (w / omega0).powi(2));
                [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
            })
            .collect()];
        let dp = DynamicPolarizability { freqs, weights, per_atom, molecular: Vec::new() };
        let res = casimir_polder_c6(&dp);
        let c6 = res.c6_iso_pair[(0, 0)];
        let analytic = 0.75 * alpha0 * alpha0 * omega0;
        assert!(
            (c6 - analytic).abs() / analytic < 2e-3,
            "C-P C6={c6} vs analytic={analytic}"
        );
        let tr = (res.c6_aniso_pair[0][0][0][0]
            + res.c6_aniso_pair[0][0][1][1]
            + res.c6_aniso_pair[0][0][2][2])
            / 3.0;
        assert!((tr - c6).abs() / c6 < 1e-12, "aniso trace {tr} != iso {c6}");
    }

    /// TS dynamic α round-trip: C6 from C-P equals the closed-form c6_eff.
    #[test]
    fn ts_dynamic_round_trip_c6() {
        let z = vec![1usize];
        let vol_ratio = vec![1.0_f64];
        let alpha_static = vec![[[4.5, 0.0, 0.0], [0.0, 4.5, 0.0], [0.0, 0.0, 4.5]]];
        let (freqs, weights) = trapezoid_grid(20000, 200.0);
        let dp = ts_dynamic_polarizability(&z, &vol_ratio, &alpha_static, &freqs, &weights);
        let res = casimir_polder_c6(&dp);
        let c6 = res.c6_iso_pair[(0, 0)];
        assert!(
            (c6 - 6.5).abs() / 6.5 < 3e-3,
            "TS round-trip C6={c6} vs c6_eff=6.5"
        );
    }

    /// Anisotropy inherited from the static tensor: prolate static → prolate C6.
    #[test]
    fn ts_dynamic_inherits_static_anisotropy() {
        let z = vec![6usize];
        let vol_ratio = vec![1.0_f64];
        let alpha_static = vec![[[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 18.0]]];
        let (freqs, weights) = trapezoid_grid(4000, 100.0);
        let dp = ts_dynamic_polarizability(&z, &vol_ratio, &alpha_static, &freqs, &weights);
        let res = casimir_polder_c6(&dp);
        let czz = res.c6_aniso_pair[0][0][2][2];
        let cxx = res.c6_aniso_pair[0][0][0][0];
        assert!(czz > cxx, "expected prolate C6: zz={czz} xx={cxx}");
    }
}
