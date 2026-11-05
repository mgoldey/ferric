//! Analytic nuclear gradient of the IEF-PCM energy at a fixed AO density.
//!
//! `E_pcm = ½ vᵀ K⁻¹R v` (`solver.rs`). With the SCF variational in the
//! density, the total-energy gradient is the solvent-free SCF gradient
//! evaluated with the solvated orbitals plus `∂E_pcm/∂R` at fixed AO density
//! `P` ([`pcm_gradient`]). That derivative has two parts
//! (`crates/ferric-pcm/theory.md` §8):
//!
//! * the potential term `qᵀ ∂v/∂R` ([`PcmGradient::potential`]): the
//!   symmetrized charges `q` held fixed as classical charges on the
//!   tesserae, each tessera moving rigidly with its sphere's atom;
//! * the response term `½ vᵀ ∂(K⁻¹R)/∂R v` ([`PcmGradient::response`]):
//!   the derivatives of `S`, `D` and the tessera areas through the tessera
//!   positions and the switching function.

use std::f64::consts::PI;

use ferric_core::external_potential::PointCharge;
use ferric_core::mol::Molecule;
use ferric_core::special::erf;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;

use crate::cavity::{build_cavity, switch_h, switch_h_deriv, SphereGeometry, Tessera};
use crate::config::{PcmConfig, ProbeKind};
use crate::matrices::{build_k_r, build_s_d_kind, SdKind};

/// `∂E_pcm/∂R` at fixed AO density, split into its two parts. Each array is
/// `(natoms, 3)` in Hartree/Bohr, rows in `Molecule::atoms` order.
#[derive(Debug, Clone)]
pub struct PcmGradient {
    /// `qᵀ ∂v/∂R`: nuclear, AO-centre and tessera-position derivatives of
    /// the solute potential, contracted with the fixed symmetrized charges.
    pub potential: Array2<f64>,
    /// `½ vᵀ ∂(K⁻¹R)/∂R v`: derivatives of `S`, `D` and the areas.
    pub response: Array2<f64>,
    /// `E_pcm = ½ q·v` at this geometry and density.
    pub e_pcm: f64,
}

impl PcmGradient {
    /// `potential + response`.
    pub fn total(&self) -> Array2<f64> {
        &self.potential + &self.response
    }
}

/// `∂E_pcm/∂R` (Hartree/Bohr, `(natoms, 3)`) of the IEF-PCM energy for the
/// AO density `density` (total, `D_α + D_β`). Add it to the SCF gradient
/// computed from the solvated orbitals; see [`pcm_gradient_parts`].
pub fn pcm_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
    cfg: &PcmConfig,
) -> Result<Array2<f64>, FerricError> {
    Ok(pcm_gradient_parts(mol, prep, density, cfg)?.total())
}

/// [`pcm_gradient`] with its potential and response parts separate.
///
/// The cavity is rebuilt from `mol` exactly as [`crate::PcmContext::new`]
/// builds it. Returns `Err` when `cfg.cavity` is `Some`: an injected tessera
/// list carries no record of how it moves with the nuclei.
pub fn pcm_gradient_parts(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
    cfg: &PcmConfig,
) -> Result<PcmGradient, FerricError> {
    if cfg.cavity.is_some() {
        return Err(FerricError::General(
            "pcm_gradient: PcmConfig::cavity is an injected tessera list with no geometry \
             dependence; the PCM gradient needs the cavity built from the molecule \
             (cavity = None)"
                .into(),
        ));
    }
    let ccfg = cfg.cavity_config();
    let tess = build_cavity(mol, &ccfg)?;
    let n_grid = ferric_quadrature::lebedev::lebedev(ccfg.lebedev_order)
        .0
        .len();
    let geom = SphereGeometry::new(mol, &ccfg, n_grid);

    let (s, d) = build_s_d_kind(&tess, cfg.sd_kind);
    let (k, r, f_eps) = build_k_r(&s, &d, &tess, cfg.epsilon)?;
    let v = Array1::from_vec(cfg.probe.potential_at_tesserae(mol, prep, density, &tess)?);

    let lapack = |what: &str, e: ndarray_linalg::error::LinalgError| {
        FerricError::Lapack(format!("pcm_gradient: {what} solve failed: {e}"))
    };
    // x = K⁻¹ R v (raw charges), y = K⁻ᵀ v, q = ½ (x + Rᵀ y) (solver.rs).
    let x = k.solve(&r.dot(&v)).map_err(|e| lapack("K x = R v", e))?;
    let y = k
        .t()
        .to_owned()
        .solve(&v)
        .map_err(|e| lapack("Kᵀ y = v", e))?;
    let q = 0.5 * (&x + &r.t().dot(&y));
    let e_pcm = 0.5 * q.dot(&v);

    let natoms = mol.atoms.len();
    let potential = match cfg.probe {
        ProbeKind::Point => potential_term_point(mol, prep, density, &tess, &q)?,
        ProbeKind::GaussianSmeared => potential_term_smeared(mol, prep, density, &tess, &q)?,
    };
    let response = response_term(
        mol,
        &geom,
        ccfg.skip_ghost_atoms,
        &tess,
        cfg.sd_kind,
        f_eps,
        (&s, &d),
        (&v, &x, &y),
    );
    debug_assert_eq!(potential.dim(), (natoms, 3));
    Ok(PcmGradient {
        potential,
        response,
        e_pcm,
    })
}

/// `S(r)` radial derivative `S'(r)`, the double-layer radial factor
/// `h(r) = −S'(r)/r` (so `D_ij = h(r_ij) (r_ij·n_j)`, `matrices.rs`), and
/// `h'(r)`, for the off-diagonal kernel of `kind` at combined width `xi`.
fn sd_kernel(kind: SdKind, xi: f64, r: f64) -> (f64, f64, f64) {
    match kind {
        SdKind::PointCharge => {
            let r2 = r * r;
            (-1.0 / r2, 1.0 / (r2 * r), -3.0 / (r2 * r2))
        }
        SdKind::GaussianSmeared => {
            let c = 2.0 / PI.sqrt();
            let er = erf(xi * r);
            let ex = (-xi * xi * r * r).exp();
            let r2 = r * r;
            let r3 = r2 * r;
            let sp = c * xi * ex / r - er / r2;
            let h = er / r3 - c * xi * ex / r2;
            let hp = 3.0 * c * xi * ex / r3 - 3.0 * er / (r2 * r2) + 2.0 * c * xi.powi(3) * ex / r;
            (sp, h, hp)
        }
    }
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn atom_pos(mol: &Molecule, a: usize) -> [f64; 3] {
    let at = &mol.atoms[a];
    [at.x, at.y, at.zpos]
}

/// Add `c · vec` to atom `on` and subtract it from atom `off`: the
/// derivative of a quantity that depends on `R_on − R_off` only.
fn push_pair(g: &mut Array2<f64>, on: usize, off: usize, c: f64, vec: [f64; 3]) {
    for ax in 0..3 {
        g[(on, ax)] += c * vec[ax];
        g[(off, ax)] -= c * vec[ax];
    }
}

/// `½ vᵀ ∂(K⁻¹R) v` with `K = (I − f/2π DA) S`, `R = −f (I − DA/2π)`:
///
/// ```text
///   ½ [ −zᵀ ∂S x + (f/2π) yᵀ ∂D u + (f/2π) Σ_j (Dᵀy)_j w_j ∂a_j ]
///   w = Sx + v,  u = A w,  z = y − (f/2π) A Dᵀ y
/// ```
///
/// Off-diagonal `S_ij`, `D_ij` move with `r_i − r_j`; diagonal `S_ii` and the
/// areas move through the switching function `swf_i` (and `D_ii` too for
/// [`SdKind::PointCharge`], where it is `−S_ii/2R_i`); the Gaussian `D_ii`,
/// the normals and the Gaussian widths are geometry independent.
#[allow(clippy::too_many_arguments)]
fn response_term(
    mol: &Molecule,
    geom: &SphereGeometry,
    skip_ghost_atoms: bool,
    tess: &[Tessera],
    kind: SdKind,
    f_eps: f64,
    (s, d): (&Array2<f64>, &Array2<f64>),
    (v, x, y): (&Array1<f64>, &Array1<f64>, &Array1<f64>),
) -> Array2<f64> {
    let n = tess.len();
    let natoms = mol.atoms.len();
    let pref = f_eps / (2.0 * PI);
    let area = Array1::from_iter(tess.iter().map(|t| t.area));
    let w = s.dot(x) + v;
    let u = &area * &w;
    let dty = d.t().dot(y);
    let z = y - &(pref * &area * &dty);

    let mut g = Array2::<f64>::zeros((natoms, 3));

    // Off-diagonal S and D: every ordered pair (i, j), i ≠ j.
    for i in 0..n {
        let ti = &tess[i];
        for j in 0..n {
            let tj = &tess[j];
            if i == j || ti.atom_index == tj.atom_index {
                continue;
            }
            let rij = sub3(ti.position, tj.position);
            let rr = norm3(rij);
            let xi_ij = ti.charge_exp * tj.charge_exp
                / (ti.charge_exp * ti.charge_exp + tj.charge_exp * tj.charge_exp).sqrt();
            let (sp, h, hp) = sd_kernel(kind, xi_ij, rr);
            let nu = rij[0] * tj.normal[0] + rij[1] * tj.normal[1] + rij[2] * tj.normal[2];
            let cs = -0.5 * z[i] * x[j];
            let cd = 0.5 * pref * y[i] * u[j];
            // ∂/∂r_i of cs·S_ij + cd·D_ij.
            let mut gi = [0.0; 3];
            for ax in 0..3 {
                let rhat = rij[ax] / rr;
                gi[ax] = cs * sp * rhat + cd * (hp * nu * rhat + h * tj.normal[ax]);
            }
            push_pair(&mut g, ti.atom_index, tj.atom_index, 1.0, gi);
        }
    }

    // Switching function: ∂S_jj and ∂a_j are both ∝ ∂ ln swf_j.
    for j in 0..n {
        let tj = &tess[j];
        let diag = match kind {
            // S_jj = ξ_j √(2/π) / swf_j.
            SdKind::GaussianSmeared => 0.5 * z[j] * x[j] * s[(j, j)],
            // S_jj = 1.0694 √(4π / a_j) and D_jj = −S_jj / (2 R_j), a_j ∝ swf_j.
            SdKind::PointCharge => {
                0.25 * z[j] * x[j] * s[(j, j)] - 0.25 * pref * y[j] * u[j] * d[(j, j)]
            }
        };
        let c = diag + 0.5 * pref * dty[j] * w[j] * tj.area;
        let a_own = tj.atom_index;
        for b in 0..natoms {
            if b == a_own || (skip_ghost_atoms && mol.atoms[b].ghost) {
                continue;
            }
            let rv = sub3(tj.position, atom_pos(mol, b));
            let dist = norm3(rv);
            let t = geom.switch_arg(b, dist);
            let hd = switch_h_deriv(t);
            if hd == 0.0 {
                continue;
            }
            // ∂ ln h(t)/∂r_j = h'(t)/h(t) · (1/R_sw,B) · r̂_jB.
            let coef = c * hd / switch_h(t) / geom.r_sw[b] / dist;
            push_pair(&mut g, a_own, b, coef, rv);
        }
    }
    g
}

/// `qᵀ ∂v/∂R` for point probes: `v_k = Σ_B Z_B/|r_k − R_B| + Σ_μν P_μν B^k_μν`.
fn potential_term_point(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
    tess: &[Tessera],
    q: &Array1<f64>,
) -> Result<Array2<f64>, FerricError> {
    let natoms = mol.atoms.len();
    let mut g = Array2::<f64>::zeros((natoms, 3));
    nuclear_tessera_term(mol, tess, q, |_, r| (-1.0 / (r * r), 1.0 / r), &mut g);

    // Electronic: Tr[P ∂V(q)] with V(q) = Σ_k q_k ⟨μ|−1/|r − r_k||ν⟩ — the
    // nuclear-attraction derivative engine with the tessera charges as the
    // ONLY charges. Blocks 0-5 are the two shell centres, block 6 + 3k + c
    // the position of charge k (moves with its sphere's atom).
    let charges: Vec<PointCharge> = tess
        .iter()
        .zip(q.iter())
        .map(|(t, &qk)| PointCharge {
            q: qk,
            x: t.position[0],
            y: t.position[1],
            z: t.position[2],
        })
        .collect();
    let mut eng = Engine::new_1e_deriv(ffi::OP_NUCLEAR, prep, 1e-14)?;
    eng.set_point_charges_only(&charges)?;
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let sh2at = prep.shell_to_atom();
    let nt = tess.len();
    for s1 in 0..prep.nshells() {
        for s2 in 0..=s1 {
            let Some(blk) = eng.compute_1e_deriv_block_n(prep, s1, s2, nt) else {
                continue;
            };
            let (n1, n2) = (dims[s1], dims[s2]);
            let bsz = n1 * n2;
            let fac = if s1 == s2 { 1.0 } else { 2.0 };
            let mut site = vec![[0.0_f64; 3]; nt];
            for a in 0..n1 {
                for b in 0..n2 {
                    let idx = a * n2 + b;
                    let pv = fac * density[(offs[s1] + a, offs[s2] + b)];
                    for ax in 0..3 {
                        g[(sh2at[s1], ax)] += pv * blk[ax * bsz + idx];
                        g[(sh2at[s2], ax)] += pv * blk[(3 + ax) * bsz + idx];
                    }
                    for (kk, sk) in site.iter_mut().enumerate() {
                        for ax in 0..3 {
                            sk[ax] += pv * blk[(6 + 3 * kk + ax) * bsz + idx];
                        }
                    }
                }
            }
            for (t, sk) in tess.iter().zip(site.iter()) {
                for ax in 0..3 {
                    g[(t.atom_index, ax)] += sk[ax];
                }
            }
        }
    }
    Ok(g)
}

/// `qᵀ ∂v/∂R` for Gaussian probes:
/// `v_k = Σ_B Z_B erf(ξ_k|r_k − R_B|)/|r_k − R_B| − Σ_μν P_μν (μν|g_k)/N_k`.
fn potential_term_smeared(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
    tess: &[Tessera],
    q: &Array1<f64>,
) -> Result<Array2<f64>, FerricError> {
    let natoms = mol.atoms.len();
    let mut g = Array2::<f64>::zeros((natoms, 3));
    let c = 2.0 / PI.sqrt();
    nuclear_tessera_term(
        mol,
        tess,
        q,
        |xi, r| {
            let er = erf(xi * r);
            let ex = (-xi * xi * r * r).exp();
            (c * xi * ex / r - er / (r * r), er / r)
        },
        &mut g,
    );

    let sites: Vec<[f64; 4]> = tess
        .iter()
        .map(|t| {
            [
                t.position[0],
                t.position[1],
                t.position[2],
                t.charge_exp * t.charge_exp,
            ]
        })
        .collect();
    let site_basis = SiteBasis::new(&sites, 0)?;
    let mut eng = Engine::new_3center_deriv(Operator::coulomb(), prep, &site_basis.prep, 1e-14)?;
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let sh2at = prep.shell_to_atom();
    for (k, t) in tess.iter().enumerate() {
        let sh_p = site_basis.site_shell[k];
        let qn = -q[k] / site_basis.norm_int[k];
        for s1 in 0..prep.nshells() {
            for s2 in 0..prep.nshells() {
                let Some(blk) = eng.compute_eri3_deriv(prep, &site_basis.prep, sh_p, s1, s2) else {
                    continue;
                };
                let (n1, n2) = (dims[s1], dims[s2]);
                let bsz = n1 * n2;
                for a in 0..n1 {
                    for b in 0..n2 {
                        let idx = a * n2 + b;
                        let cv = qn * density[(offs[s1] + a, offs[s2] + b)];
                        for ax in 0..3 {
                            g[(t.atom_index, ax)] += cv * blk[ax * bsz + idx];
                            g[(sh2at[s1], ax)] += cv * blk[(3 + ax) * bsz + idx];
                            g[(sh2at[s2], ax)] += cv * blk[(6 + ax) * bsz + idx];
                        }
                    }
                }
            }
        }
    }
    Ok(g)
}

/// Classical `Σ_k q_k Σ_B Z_B φ(ξ_k, |r_k − R_B|)`; `kernel(ξ, r)` returns
/// `(φ'(r), φ(r))`. The own-sphere atom is skipped: `|r_k − R_A(k)|` is fixed.
fn nuclear_tessera_term(
    mol: &Molecule,
    tess: &[Tessera],
    q: &Array1<f64>,
    kernel: impl Fn(f64, f64) -> (f64, f64),
    g: &mut Array2<f64>,
) {
    for (k, t) in tess.iter().enumerate() {
        for (b, atom) in mol.atoms.iter().enumerate() {
            if atom.ghost || b == t.atom_index {
                continue;
            }
            let rv = sub3(t.position, atom_pos(mol, b));
            let dist = norm3(rv);
            let (dphi, _) = kernel(t.charge_exp, dist);
            let coef = q[k] * atom.effective_z() as f64 * dphi / dist;
            push_pair(g, t.atom_index, b, coef, rv);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `sd_kernel`'s `S'`, `h`, `h'` against central differences of the
    /// `S`/`D` formulas in `matrices.rs`.
    #[test]
    fn sd_kernel_matches_finite_differences() {
        let s_of = |kind: SdKind, xi: f64, r: f64| match kind {
            SdKind::PointCharge => 1.0 / r,
            SdKind::GaussianSmeared => erf(xi * r) / r,
        };
        for kind in [SdKind::PointCharge, SdKind::GaussianSmeared] {
            for (xi, r) in [(0.7, 0.9), (2.5, 0.4), (4.0, 1.7)] {
                let hstep = 1e-5;
                let (sp, h, hp) = sd_kernel(kind, xi, r);
                let sp_fd = (s_of(kind, xi, r + hstep) - s_of(kind, xi, r - hstep)) / (2.0 * hstep);
                assert!((sp - sp_fd).abs() < 1e-8 * sp.abs().max(1.0), "{kind:?} S'");
                assert!((h + sp / r).abs() < 1e-14 * h.abs().max(1.0), "{kind:?} h");
                let h_of = |rr: f64| sd_kernel(kind, xi, rr).1;
                let hp_fd = (h_of(r + hstep) - h_of(r - hstep)) / (2.0 * hstep);
                assert!((hp - hp_fd).abs() < 1e-7 * hp.abs().max(1.0), "{kind:?} h'");
            }
        }
    }

    #[test]
    fn switch_h_deriv_matches_finite_differences() {
        for x in [0.05, 0.3, 0.5, 0.77, 0.95] {
            let hstep = 1e-6;
            let fd = (switch_h(x + hstep) - switch_h(x - hstep)) / (2.0 * hstep);
            assert!((switch_h_deriv(x) - fd).abs() < 1e-8, "x={x}");
        }
        assert_eq!(switch_h_deriv(-0.1), 0.0);
        assert_eq!(switch_h_deriv(1.2), 0.0);
    }
}
