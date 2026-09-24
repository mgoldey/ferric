//! Stage 3: per-k complex one-electron matrices `S(k)`, `T(k)`, `V(k)`,
//! `h(k)` — the SAME shifted-shell lattice sums as
//! [`periodic_hcore`](super::periodic_hcore), each image weighted by the
//! Bloch phase `e^{ik·L}` (PySCF's convention, `KPointMesh` module doc):
//!
//! ```text
//! S(k)   = Σ_L e^{ik·L} ⟨μ_0|ν_L⟩          T(k) likewise
//! V_SR(k)= −Σ_L e^{ik·L} Σ_{C,M} Z_C ⟨μ_0| erfc(ω|r−R_C−M|)/|r−R_C−M| |ν_L⟩   (nuclei periodic: no phase on M)
//! V_LR(k)= −(1/Ω) Σ_{G≠0} (4π/G²) e^{−G²/4ω²} P^k_μν(G) conj(S(G)),   S(G) = Σ_C Z_C e^{−iG·R_C}
//! V_G0(k)= + π Z_tot S(k) / (ω² Ω)
//! ```
//!
//! `P^k(G) = Σ_L e^{ik·L} ∫ φ_μ φ_ν(·−L) e^{−iG·r}` comes from the
//! residue-resolved pair FT ([`crate::pair_ft::residues`]). The LR sum runs
//! over the FULL G sphere: `P^k(−G) = conj(P^{−k}(G))`, not `conj(P^k(G))`,
//! so the Gamma half-sphere trick does not apply. Every matrix is
//! Hermitian by construction and is Hermitised (`½(X + X^H)`) to remove
//! roundoff, as the Gamma path symmetrises; `X(−k) = X(k)*` holds EXACTLY
//! because the phases are exact conjugates ([`crate::kpts`]).
//!
//! Truncation, screening, Gaussian nuclei and memory gating are exactly
//! those of `periodic_hcore` (same `PeriodicHcoreConfig`, same image and
//! nucleus-candidate sets, same `SrBound::Derived` screen), so at Gamma
//! (1×1×1 mesh) this reproduces `periodic_hcore` up to the order of the LR
//! sum (full vs half sphere).

use super::{
    gvector_list_bytes, max_pair_exponent, nonzero_nuclei, nucleus_radius_m, pair_bound,
    pair_images, pair_radius, prim_shells, segment_distance, sr_candidates, PeriodicHcoreConfig,
    SrBound, ERI3_ENGINE_PRECISION, G_CHUNK_BYTES, ONE_E_ENGINE_PRECISION,
};
use crate::budget::{bytes_of, Ledger};
use crate::ewald::{default_ewald_omega, ewald_nuclear_repulsion};
use crate::kpts::{lattice_coords, KPointMesh};
use crate::lattice::Cell;
use crate::pair_ft::residues::{pair_ft_residues_chunked, residue_coords};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Output of [`periodic_hcore_kpts`]: one `(nbasis, nbasis)` complex
/// Hermitian matrix per mesh k-point (mesh order).
#[derive(Debug, Clone)]
pub struct PeriodicHcoreK {
    /// `S(k)`.
    pub s: Vec<Array2<Complex64>>,
    /// `T(k)`.
    pub t: Vec<Array2<Complex64>>,
    /// `V(k) = V_SR + V_LR + V_G0`.
    pub v: Vec<Array2<Complex64>>,
    /// `h(k) = T(k) + V(k)`.
    pub h: Vec<Array2<Complex64>>,
    /// Ewald nuclear repulsion per cell.
    pub enn: f64,
    /// The ω used.
    pub omega: f64,
    /// Pair images summed.
    pub n_images: usize,
    /// Shifted 3-centre calls in the SR attraction.
    pub n_sr_triplets: usize,
    /// Full-sphere G vectors in the LR attraction.
    pub n_g_lr: usize,
    /// Resolved memory budget (bytes).
    pub budget_bytes: usize,
}

fn czero(n: usize) -> Array2<Complex64> {
    Array2::<Complex64>::zeros((n, n))
}

/// `m_k[o1+i, o2+j] += ph_k · (f · blk[i, j])` for every k.
#[allow(clippy::too_many_arguments)]
fn add_block_k(
    ms: &mut [Array2<Complex64>],
    ph: &[Complex64],
    blk: &[f64],
    o1: usize,
    n1: usize,
    o2: usize,
    n2: usize,
    f: f64,
) {
    for (m, p) in ms.iter_mut().zip(ph) {
        for i in 0..n1 {
            for j in 0..n2 {
                m[(o1 + i, o2 + j)] += *p * (f * blk[i * n2 + j]);
            }
        }
    }
}

/// `½(X + X^H)`.
pub(crate) fn hermitize(m: &Array2<Complex64>) -> Array2<Complex64> {
    let n = m.nrows();
    Array2::from_shape_fn((n, n), |(i, j)| 0.5 * (m[(i, j)] + m[(j, i)].conj()))
}

/// Build `S(k)`, `T(k)`, `V(k)`, `h(k)` on every point of `mesh`, and
/// `E_nn`. `prep` must be built from `cell.mol()`; `mesh` from `cell`.
pub fn periodic_hcore_kpts(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    cfg: &PeriodicHcoreConfig,
) -> Result<PeriodicHcoreK, FerricError> {
    cfg.validate()?;
    let shells = prim_shells(cell, prep)?;
    let n = prep.nbasis();
    let nk = mesh.nk();
    let omega = cfg.omega;
    let thresh = cfg.precision;
    let pair_thresh = 0.1 * thresh;
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    ledger.reserve(
        &format!(
            "periodic_hcore_kpts complex n×n matrices (n = {n}, N_k = {nk}: S, T, V_SR, V_LR, V, h + 2 temporaries)"
        ),
        bytes_of((nk * n * n) as u64, 16 * 8),
    )?;

    let images = pair_images(cell, &shells, pair_thresh, &mut ledger)?;
    let rpair = pair_radius(&shells, pair_thresh);
    ledger.reserve(
        &format!(
            "periodic_hcore_kpts image phases ({} images × {nk} k)",
            images.len()
        ),
        bytes_of((images.len() * nk) as u64, 16),
    )?;
    let b = cell.reciprocal();
    let ph: Vec<Vec<Complex64>> = images
        .iter()
        .map(|l| {
            let nl = lattice_coords(&b, l);
            (0..nk).map(|k| mesh.phase(k, nl)).collect()
        })
        .collect();

    // --- S(k), T(k)
    let mut eng_s = Engine::new_1e(ffi::OP_OVERLAP, prep, ONE_E_ENGINE_PRECISION)?;
    let mut eng_t = Engine::new_1e(ffi::OP_KINETIC, prep, ONE_E_ENGINE_PRECISION)?;
    let mut s: Vec<Array2<Complex64>> = (0..nk).map(|_| czero(n)).collect();
    let mut t: Vec<Array2<Complex64>> = (0..nk).map(|_| czero(n)).collect();
    for (il, l) in images.iter().enumerate() {
        for (i1, a) in shells.iter().enumerate() {
            for (i2, bsh) in shells.iter().enumerate() {
                let blk = eng_s.compute_1e_block_shifted(prep, i1, i2, *l)?;
                add_block_k(&mut s, &ph[il], blk, a.off, a.dim, bsh.off, bsh.dim, 1.0);
                let blk = eng_t.compute_1e_block_shifted(prep, i1, i2, *l)?;
                add_block_k(&mut t, &ph[il], blk, a.off, a.dim, bsh.off, bsh.dim, 1.0);
            }
        }
    }
    let s: Vec<Array2<Complex64>> = s.iter().map(hermitize).collect();
    let t: Vec<Array2<Complex64>> = t.iter().map(hermitize).collect();

    // --- V_SR(k): the production SR loop of `sr_attraction` (Derived bound,
    // no tracking), phase-weighted by the ν image L.
    let zs = cell.nuclear_charges();
    let (nuc, zmax) = nonzero_nuclei(cell);
    let mut v_sr: Vec<Array2<Complex64>> = (0..nk).map(|_| czero(n)).collect();
    let mut n_sr_triplets = 0usize;
    if !nuc.is_empty() {
        let cands = sr_candidates(cell, &shells, &nuc, omega, zmax, thresh, rpair, &mut ledger)?;
        let sites: Vec<[f64; 4]> = nuc
            .iter()
            .map(|(_, r)| [r[0], r[1], r[2], cfg.nucleus_exponent])
            .collect();
        let site = SiteBasis::new(&sites, 0)?;
        let mut eng = Engine::new_3center(
            Operator::erfc(omega),
            prep,
            &site.prep,
            ERI3_ENGINE_PRECISION,
        )?;
        let bound = SrBound::Derived;
        for (il, l) in images.iter().enumerate() {
            for (i1, a) in shells.iter().enumerate() {
                for (i2, bsh) in shells.iter().enumerate() {
                    let bc = [
                        bsh.center[0] + l[0],
                        bsh.center[1] + l[1],
                        bsh.center[2] + l[2],
                    ];
                    let r2 = (a.center[0] - bc[0]).powi(2)
                        + (a.center[1] - bc[1]).powi(2)
                        + (a.center[2] - bc[2]).powi(2);
                    let (q, pmin, pmax) = pair_bound(a, bsh, r2);
                    let wp = bound.omega_p(omega, pmin);
                    let Some(rad) = nucleus_radius_m(q, zmax, pmax, wp, thresh, bound.margin())
                    else {
                        continue;
                    };
                    for (kc, m, x) in &cands {
                        if segment_distance(*x, a.center, bc) > rad {
                            continue;
                        }
                        n_sr_triplets += 1;
                        let f = -nuc[*kc].0 / site.norm_int[*kc];
                        if let Some(blk) = eng.compute_eri3_shifted(
                            prep,
                            &site.prep,
                            site.site_shell[*kc],
                            i1,
                            i2,
                            [*m, [0.0; 3], *l],
                        )? {
                            add_block_k(&mut v_sr, &ph[il], blk, a.off, a.dim, bsh.off, bsh.dim, f);
                        }
                    }
                }
            }
        }
    }
    let v_sr: Vec<Array2<Complex64>> = v_sr.iter().map(hermitize).collect();

    // --- V_LR(k): full G sphere, residue-resolved pair FT at q = 0.
    let smooth = (1.0 / thresh).ln().sqrt();
    let gcut = (2.0 * omega).min(2.0 * max_pair_exponent(prep).sqrt()) * smooth;
    ledger.reserve(
        &format!("periodic_hcore_kpts LR G list (|G| <= {gcut:.3})"),
        gvector_list_bytes(cell, gcut)?,
    )?;
    let gv: Vec<[f64; 3]> = cell
        .gvectors(gcut)?
        .into_iter()
        .filter(|g| g[0] * g[0] + g[1] * g[1] + g[2] * g[2] > 0.0)
        .collect();
    let moduli = mesh.residue_moduli();
    let nr = moduli[0] * moduli[1] * moduli[2];
    let phr: Vec<Vec<Complex64>> = (0..nk)
        .map(|k| {
            (0..nr)
                .map(|r| mesh.phase(k, residue_coords(r, moduli)))
                .collect()
        })
        .collect();
    let pos = cell.positions();
    let vol = cell.volume();
    let mut v_lr: Vec<Array2<Complex64>> = (0..nk).map(|_| czero(n)).collect();
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    let accumulate =
        |_g0: usize, gs: &[[f64; 3]], q: &[Array3<Complex64>]| -> Result<(), FerricError> {
            for (g, gvec) in gs.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let kern = 4.0 * PI / g2 * (-g2 / (4.0 * omega * omega)).exp();
                // S(G) = Σ_C Z_C e^{−iG·R_C}
                let mut sg = Complex64::new(0.0, 0.0);
                for (zc, r) in zs.iter().zip(&pos) {
                    let a = gvec[0] * r[0] + gvec[1] * r[1] + gvec[2] * r[2];
                    sg += Complex64::new(zc * a.cos(), -zc * a.sin());
                }
                let w = sg.conj() * (-kern / vol);
                for (vk, pk) in v_lr.iter_mut().zip(&phr) {
                    for m in 0..n {
                        for nu in 0..n {
                            let mut p = Complex64::new(0.0, 0.0);
                            for (qr, ph) in q.iter().zip(pk) {
                                p += *ph * qr[[m, nu, g]];
                            }
                            vk[(m, nu)] += p * w;
                        }
                    }
                }
            }
            Ok(())
        };
    pair_ft_residues_chunked(
        cell,
        prep,
        &gv,
        moduli,
        pair_thresh,
        chunk_budget,
        0,
        accumulate,
    )?;
    let ztot: f64 = zs.iter().sum();
    let c0 = PI / (omega * omega * vol);
    let mut v = Vec::with_capacity(nk);
    let mut h = Vec::with_capacity(nk);
    for k in 0..nk {
        let vk = &(&v_sr[k] + &hermitize(&v_lr[k])) + &s[k].mapv(|z| z * (c0 * ztot));
        h.push(&t[k] + &vk);
        v.push(vk);
    }
    let enn = ewald_nuclear_repulsion(cell, default_ewald_omega(cell))?;
    ferric_core::memory::warn_if_rss_over("ferric-pbc periodic_hcore_kpts", ledger.budget(), 1.1);
    Ok(PeriodicHcoreK {
        s,
        t,
        v,
        h,
        enn,
        omega,
        n_images: images.len(),
        n_sr_triplets,
        n_g_lr: gv.len(),
        budget_bytes: ledger.budget(),
    })
}
