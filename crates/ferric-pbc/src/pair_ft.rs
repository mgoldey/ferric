//! Analytic Fourier transform of lattice-summed Gaussian pair densities.
//!
//! ```text
//! P[m, n, g] = Σ_L ∫ φ_m(r) φ_n(r − L) e^{−i G_g · r} dr
//! ```
//!
//! (the sign convention of PySCF `pbc.df.ft_ao.ft_aopair` at Gamma,
//! `ft_ao.py:52-53`). `P[:, :, g]` at `G = 0` is the lattice-summed overlap
//! `S_latt = Σ_L ⟨φ_m | φ_n(· − L)⟩` — the exactness anchor in
//! `tests/pbc_stage0.rs`.
//!
//! # Method (McMurchie–Davidson)
//!
//! A primitive pair is `φ_a φ_b = Σ_{tuv} E_t^{ij} E_u^{kl} E_v^{mn} Λ_{tuv}` with
//! Hermite Gaussians `Λ_{tuv} = ∂_{P_x}^t ∂_{P_y}^u ∂_{P_z}^v e^{−p|r−P|²}`, and
//! `FT[e^{−p|r−P|²}](G) = (π/p)^{3/2} e^{−G²/4p} e^{−iG·P}`, so each `∂_{P_d}`
//! becomes a factor `(−iG_d)`. Per Cartesian direction
//! `F_d^{ij}(G) = Σ_t E_t^{ij} (−iG_d)^t`, and the Cartesian pair block is
//! `c_a c_b (π/p)^{3/2} e^{−G²/4p − iG·P} F_x F_y F_z`. The E-coefficients are
//! [`ferric_integrals::md3c1e::e_table`] — the same recursion the COSX kernel
//! uses — with `q = A_x − B_x`.
//!
//! # AO basis: ordering and normalisation (ferric's, not PySCF's calibration)
//!
//! * Shells and AO offsets come from the [`PreparedBasis`] itself
//!   (`located_shells()`, `shell_offsets()`), i.e. atom-major and basis order
//!   within an atom, the list handed to libint2 (`basis_bridge.rs`,
//!   `PreparedBasis::new`). Every shell's function count is cross-checked
//!   against `shell_dims()` (libint2's own count).
//! * Cartesian components in CCA order (`md3c1e::cart_components`).
//! * Normalisation: stored `BasisSet` coefficients are contraction-renormalised
//!   at load for unit-normalised primitives (`basis.rs::renormalize_contraction`);
//!   the primitive factor `md3c1e::prim_norm(a, l)` is folded in here, which
//!   makes the `(l,0,0)` Cartesian component unit-normalised — libint2's
//!   convention (md3c1e.rs module doc, "Conventions reproduced").
//! * Pure shells (`l >= 2` only; ferric never builds pure s/p) are transformed
//!   with `md3c1e::ferric_cart2sph(l)` (ncart × 2l+1, row-major, columns
//!   `m = −l..=l`), exactly as `Md3c1e::cart_to_out` does. That transform is
//!   validated against libint2 by `ferric-integrals/tests/md3c1e_matches_cosx_a.rs`.
//!
//! No calibration against the overlap is done: `P(G=0) ≡ S_latt` is a TEST,
//! not an input.
//!
//! # Per-primitive-pair G window (Stage 1)
//!
//! G vectors are processed in `|G|`-ascending order and a surviving
//! primitive pair of magnitude `m` only visits
//! `G² <= 4p (ln(m/thresh) + 10 + (l_a+l_b) ln max(1, |G|_max))`: beyond it
//! the term is below `thresh·e^{−10}` even with the `(−iG)^t` factors at
//! their largest (`pbc_gamma.py` uses the same `+10` window). Diffuse pairs,
//! the ones that survive at many lattice images, need only the first few
//! per cent of a dense-AFT G sphere, which is what makes the triclinic
//! s+p oracle tractable. Output is scattered back to the caller's G order.
//!
//! # Screening and lattice range
//!
//! A primitive pair at separation `R` is skipped when its s-type overlap
//! magnitude `|c_a c_b| (π/p)^{3/2} e^{−ab R²/p}` is below `thresh`. Images
//! are taken from [`Cell::translations`] at
//! `r_pair = √(2 ln(10³/thresh) / α_min) + 2` Bohr (`ab/p >= α_min/2`), which
//! contains every pair that can survive the screen up to the polynomial
//! prefactor. MEASURED (numpy mirror of this exact algorithm vs PySCF
//! `ft_aopair`): screening `|c_a c_b| e^{−ab R²/p} < 1e-14` WITHOUT the
//! `(π/p)^{3/2}` factor left 1.4e-10 errors for LiH/STO-3G (diffuse Li 2sp);
//! with it at 1e-15 the worst error over H₂/STO-3G, LiH/STO-3G and
//! H₂O/cc-pVDZ (s, p, pure d) triclinic cells was 6.7e-13.

use crate::lattice::Cell;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::md3c1e::{cart_components, e_table, ferric_cart2sph, prim_norm, MAX_L};
use ndarray::Array3;
use num_complex::Complex64;

/// Default primitive-pair screening threshold for [`pair_ft`].
pub const DEFAULT_PAIR_FT_THRESH: f64 = 1e-15;

/// Extra e-folds in the per-primitive-pair G window (the prototype's `+10`).
const WINDOW_MARGIN: f64 = 10.0;

struct FtShell {
    l: usize,
    pure: bool,
    center: [f64; 3],
    exps: Vec<f64>,
    /// Contraction coefficients WITH `prim_norm(a, l)` folded in.
    coefs: Vec<f64>,
    ncart: usize,
    nfun: usize,
    off: usize,
}

fn build_shells(cell: &Cell, prep: &PreparedBasis) -> Result<Vec<FtShell>, FerricError> {
    // The lattice images are generated from the cell's atoms, so the basis
    // must sit on exactly those atoms.
    let pos = cell.positions();
    let atoms = prep.atoms();
    if atoms.len() != pos.len() {
        return Err(FerricError::General(format!(
            "pair_ft: PreparedBasis has {} atoms but the cell has {}",
            atoms.len(),
            pos.len()
        )));
    }
    for (k, (a, p)) in atoms.iter().zip(&pos).enumerate() {
        let d = ((a.x - p[0]).powi(2) + (a.y - p[1]).powi(2) + (a.z - p[2]).powi(2)).sqrt();
        if d > 1e-10 {
            return Err(FerricError::General(format!(
                "pair_ft: PreparedBasis atom {k} is {d:.3e} Bohr from the cell's atom {k}; \
                 build the PreparedBasis from cell.mol()"
            )));
        }
    }
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut shells = Vec::with_capacity(prep.nshells());
    for (s, sh) in prep.located_shells().iter().enumerate() {
        if sh.l < 0 || sh.l as usize > MAX_L {
            return Err(FerricError::Basis(format!(
                "pair_ft: shell {s} has l={} (supported 0..={MAX_L})",
                sh.l
            )));
        }
        let l = sh.l as usize;
        if sh.pure && l == 1 {
            return Err(FerricError::Basis(format!(
                "pair_ft: shell {s} is a pure p shell (libint2 orders it y,z,x); unsupported"
            )));
        }
        if sh.exponents.len() != sh.coefficients.len() || sh.exponents.is_empty() {
            return Err(FerricError::Basis(format!(
                "pair_ft: shell {s} has {} exponents and {} coefficients",
                sh.exponents.len(),
                sh.coefficients.len()
            )));
        }
        let pure = sh.pure && l >= 2;
        let ncart = (l + 1) * (l + 2) / 2;
        let nfun = if pure { 2 * l + 1 } else { ncart };
        if nfun != dims[s] {
            return Err(FerricError::Basis(format!(
                "pair_ft: shell {s} (l={l}, pure={pure}) has {nfun} functions but libint2 reports {}",
                dims[s]
            )));
        }
        let coefs = sh
            .exponents
            .iter()
            .zip(sh.coefficients)
            .map(|(&a, &c)| c * prim_norm(a, l))
            .collect();
        shells.push(FtShell {
            l,
            pure,
            center: sh.center,
            exps: sh.exponents.to_vec(),
            coefs,
            ncart,
            nfun,
            off: offs[s],
        });
    }
    Ok(shells)
}

/// [`pair_ft_with_thresh`] at [`DEFAULT_PAIR_FT_THRESH`].
pub fn pair_ft(
    cell: &Cell,
    prep: &PreparedBasis,
    gvecs: &[[f64; 3]],
) -> Result<Array3<Complex64>, FerricError> {
    pair_ft_with_thresh(cell, prep, gvecs, DEFAULT_PAIR_FT_THRESH)
}

/// Lattice-summed pair-density Fourier transforms `P[m, n, g]`, shape
/// `(nbasis, nbasis, gvecs.len())`, in the AO basis of `prep` (which must be
/// built from `cell.mol()`). `gvecs` may be any vectors (Bohr⁻¹), not only
/// reciprocal-lattice ones, and may include `G = 0`. See the module doc for
/// conventions.
pub fn pair_ft_with_thresh(
    cell: &Cell,
    prep: &PreparedBasis,
    gvecs: &[[f64; 3]],
    thresh: f64,
) -> Result<Array3<Complex64>, FerricError> {
    if !(f64::MIN_POSITIVE..1.0).contains(&thresh) {
        return Err(FerricError::General(format!(
            "pair_ft: thresh must lie in (0, 1), got {thresh}"
        )));
    }
    if gvecs.iter().flatten().any(|v| !v.is_finite()) {
        return Err(FerricError::General("pair_ft: non-finite G vector".into()));
    }
    let shells = build_shells(cell, prep)?;
    let nbf = prep.nbasis();
    let ng = gvecs.len();
    let mut out = Array3::<Complex64>::zeros((nbf, nbf, ng));
    if ng == 0 || shells.is_empty() {
        return Ok(out);
    }

    let amin = shells
        .iter()
        .flat_map(|s| s.exps.iter().copied())
        .fold(f64::INFINITY, f64::min);
    if !(amin > 0.0) {
        return Err(FerricError::Basis(format!(
            "pair_ft: smallest exponent is {amin}; must be > 0"
        )));
    }
    let rpair = (2.0 * (1e3 / thresh).ln() / amin).sqrt() + 2.0;
    let images = cell.translations(rpair)?;

    // Work in |G|-ascending order so each primitive pair touches only the
    // prefix of G it can reach (window below); results are scattered back
    // to the caller's order at the end.
    let norm2 = |g: &[f64; 3]| g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    let mut order: Vec<usize> = (0..ng).collect();
    order.sort_by(|&x, &y| norm2(&gvecs[x]).total_cmp(&norm2(&gvecs[y])));
    let gsorted: Vec<[f64; 3]> = order.iter().map(|&i| gvecs[i]).collect();
    let gvecs: &[[f64; 3]] = &gsorted;
    let gmax = norm2(&gvecs[ng - 1]).sqrt();

    let lmax = shells.iter().map(|s| s.l).max().unwrap_or(0);
    let nt = 2 * lmax + 1;
    let g2: Vec<f64> = gvecs
        .iter()
        .map(|g| g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
        .collect();
    // pw[d][t * ng + g] = (−i G_d)^t.
    let pw: Vec<Vec<Complex64>> = (0..3)
        .map(|d| {
            let mut v = vec![Complex64::new(0.0, 0.0); nt * ng];
            for (g, gv) in gvecs.iter().enumerate() {
                let step = Complex64::new(0.0, -gv[d]);
                let mut acc = Complex64::new(1.0, 0.0);
                for t in 0..nt {
                    v[t * ng + g] = acc;
                    acc *= step;
                }
            }
            v
        })
        .collect();
    let comps: Vec<Vec<[u8; 3]>> = (0..=lmax).map(cart_components).collect();
    let c2s: Vec<Vec<f64>> = (0..=lmax).map(ferric_cart2sph).collect();

    let zero = Complex64::new(0.0, 0.0);
    let pi = std::f64::consts::PI;
    let mut common = vec![zero; ng];
    let mut ebuf: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut fbuf: [Vec<Complex64>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    for sa in &shells {
        for sb in &shells {
            let (la, lb) = (sa.l, sb.l);
            let (nca, ncb) = (sa.ncart, sb.ncart);
            let st = la + lb + 1; // t-extent of one (i,j) E row
            let nij = (la + 1) * (lb + 1);
            for d in 0..3 {
                ebuf[d].resize(nij * st, 0.0);
                fbuf[d].resize(nij * ng, zero);
            }
            let mut cart = vec![zero; nca * ncb * ng];
            let ca_comps = &comps[la];
            let cb_comps = &comps[lb];

            for l in &images {
                let bc = [
                    sb.center[0] + l[0],
                    sb.center[1] + l[1],
                    sb.center[2] + l[2],
                ];
                let ab = [
                    sa.center[0] - bc[0],
                    sa.center[1] - bc[1],
                    sa.center[2] - bc[2],
                ];
                let r2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
                for (&a, &ca) in sa.exps.iter().zip(&sa.coefs) {
                    for (&b, &cb) in sb.exps.iter().zip(&sb.coefs) {
                        let p = a + b;
                        let cc = ca * cb * (pi / p).powf(1.5);
                        let mag = (cc * (-a * b / p * r2).exp()).abs();
                        if mag < thresh {
                            continue;
                        }
                        // G window: drop G with mag·e^{−G²/4p}·gmax^{la+lb}
                        // < thresh·e^{−WINDOW_MARGIN} (the (−iG)^t factors
                        // are bounded by gmax^{la+lb}).
                        let g2max = 4.0
                            * p
                            * ((mag / thresh).ln().max(0.0)
                                + WINDOW_MARGIN
                                + (la + lb) as f64 * gmax.max(1.0).ln());
                        let ngp = g2.partition_point(|&x| x <= g2max);
                        if ngp == 0 {
                            continue;
                        }
                        let pc = [
                            (a * sa.center[0] + b * bc[0]) / p,
                            (a * sa.center[1] + b * bc[1]) / p,
                            (a * sa.center[2] + b * bc[2]) / p,
                        ];
                        for (g, gv) in gvecs.iter().enumerate().take(ngp) {
                            let mag = cc * (-g2[g] / (4.0 * p)).exp();
                            let ph = gv[0] * pc[0] + gv[1] * pc[1] + gv[2] * pc[2];
                            // e^{−iG·P}
                            common[g] = Complex64::new(mag * ph.cos(), -mag * ph.sin());
                        }
                        for d in 0..3 {
                            e_table(la, lb, a, b, ab[d], &mut ebuf[d]);
                            let (e, f, w) = (&ebuf[d], &mut fbuf[d], &pw[d]);
                            for ij in 0..nij {
                                let frow = &mut f[ij * ng..ij * ng + ngp];
                                frow.fill(zero);
                                for t in 0..st {
                                    let et = e[ij * st + t];
                                    if et == 0.0 {
                                        continue;
                                    }
                                    let wrow = &w[t * ng..t * ng + ngp];
                                    for (x, wv) in frow.iter_mut().zip(wrow) {
                                        *x += *wv * et;
                                    }
                                }
                            }
                        }
                        let (fx, fy, fz) = (&fbuf[0], &fbuf[1], &fbuf[2]);
                        for (u, ac) in ca_comps.iter().enumerate() {
                            for (v, bcmp) in cb_comps.iter().enumerate() {
                                let ix = (ac[0] as usize * (lb + 1) + bcmp[0] as usize) * ng;
                                let iy = (ac[1] as usize * (lb + 1) + bcmp[1] as usize) * ng;
                                let iz = (ac[2] as usize * (lb + 1) + bcmp[2] as usize) * ng;
                                let dst = &mut cart[(u * ncb + v) * ng..(u * ncb + v) * ng + ngp];
                                for g in 0..ngp {
                                    dst[g] += common[g] * fx[ix + g] * fy[iy + g] * fz[iz + g];
                                }
                            }
                        }
                    }
                }
            }

            // Cartesian [nca][ncb][ng] -> AO [nfa][nfb][ng].
            let (nfa, nfb) = (sa.nfun, sb.nfun);
            let right: Vec<Complex64> = if sb.pure {
                let cbm = &c2s[lb];
                let mut tmp = vec![zero; nca * nfb * ng];
                for m in 0..nca {
                    for j in 0..nfb {
                        for n in 0..ncb {
                            let c = cbm[n * nfb + j];
                            if c == 0.0 {
                                continue;
                            }
                            for g in 0..ng {
                                tmp[(m * nfb + j) * ng + g] += cart[(m * ncb + n) * ng + g] * c;
                            }
                        }
                    }
                }
                tmp
            } else {
                cart
            };
            for i in 0..nfa {
                for j in 0..nfb {
                    for g in 0..ng {
                        let val = if sa.pure {
                            let cam = &c2s[la];
                            let mut acc = zero;
                            for m in 0..nca {
                                let c = cam[m * nfa + i];
                                if c != 0.0 {
                                    acc += right[(m * nfb + j) * ng + g] * c;
                                }
                            }
                            acc
                        } else {
                            right[(i * nfb + j) * ng + g]
                        };
                        out[[sa.off + i, sb.off + j, order[g]]] = val;
                    }
                }
            }
        }
    }
    Ok(out)
}
