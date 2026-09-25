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

use crate::budget::{bytes_of, Ledger};
use crate::lattice::Cell;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::md3c1e::{cart_components, e_table, ferric_cart2sph, prim_norm, MAX_L};
use ndarray::Array3;
use num_complex::Complex64;

/// Stage 3: residue-resolved (k-mesh) variant of the kernel below.
pub mod residues;

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

/// Bytes [`pair_ft_with_thresh`] holds per G vector: the output column
/// (`16 nao²`) plus its per-G scratch (sort order, sorted copy, `|G|²`, the
/// `(−iG)^t` powers, the common factor, the three `F` rows, the Cartesian and
/// half-transformed shell blocks). `lmax` is the largest shell `l`.
pub fn pair_ft_bytes_per_g(nao: usize, lmax: usize) -> usize {
    let ncart = (lmax + 1) * (lmax + 2) / 2;
    let c = 16usize; // Complex64
    c.saturating_mul(nao.saturating_mul(nao))
        + 8 + 24 + 8 // order, sorted G, |G|²
        + c * 3 * (2 * lmax + 1) // pw
        + c // common
        + c * 3 * (lmax + 1) * (lmax + 1) // fbuf
        + c * 2 * ncart * ncart // cart + pure half-transform
}

fn validate_inputs(gvecs: &[[f64; 3]], thresh: f64) -> Result<(), FerricError> {
    if !(f64::MIN_POSITIVE..1.0).contains(&thresh) {
        return Err(FerricError::General(format!(
            "pair_ft: thresh must lie in (0, 1), got {thresh}"
        )));
    }
    if gvecs.iter().flatten().any(|v| !v.is_finite()) {
        return Err(FerricError::General("pair_ft: non-finite G vector".into()));
    }
    Ok(())
}

fn max_gnorm(gvecs: &[[f64; 3]]) -> f64 {
    gvecs
        .iter()
        .map(|g| g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
        .fold(0.0_f64, f64::max)
        .sqrt()
}

fn basis_lmax(prep: &PreparedBasis) -> usize {
    prep.located_shells()
        .iter()
        .map(|s| s.l.max(0) as usize)
        .max()
        .unwrap_or(0)
}

/// Lattice-summed pair-density Fourier transforms `P[m, n, g]`, shape
/// `(nbasis, nbasis, gvecs.len())`, in the AO basis of `prep` (which must be
/// built from `cell.mol()`). `gvecs` may be any vectors (Bohr⁻¹), not only
/// reciprocal-lattice ones, and may include `G = 0`. See the module doc for
/// conventions.
///
/// Materialises every G at once; the output plus scratch
/// (`gvecs.len() ×` [`pair_ft_bytes_per_g`]) is gated against ferric's
/// unified memory budget ([`crate::budget::resolve`]`(None)`) before any
/// allocation. Large G sets should use [`pair_ft_chunked`].
pub fn pair_ft_with_thresh(
    cell: &Cell,
    prep: &PreparedBasis,
    gvecs: &[[f64; 3]],
    thresh: f64,
) -> Result<Array3<Complex64>, FerricError> {
    validate_inputs(gvecs, thresh)?;
    let nao = prep.nbasis();
    let lmax = basis_lmax(prep);
    let bytes = bytes_of(gvecs.len() as u64, pair_ft_bytes_per_g(nao, lmax));
    Ledger::new(crate::budget::resolve(None)).check(
        &format!(
            "pair_ft output P[m,n,g] + scratch (nao = {nao}, n_G = {}, lmax = {lmax})",
            gvecs.len()
        ),
        bytes,
    )?;
    pair_ft_block(cell, prep, gvecs, thresh, max_gnorm(gvecs))
}

/// [`pair_ft_with_thresh`] in G chunks: `sink(g0, gs, P)` receives
/// `P[:, :, j]` for `gs[j] = gvecs[g0 + j]`, in order. Each chunk (its
/// [`pair_ft_bytes_per_g`] plus the caller's `extra_bytes_per_g` for its own
/// per-G scratch) fits in `chunk_budget_bytes`; if a single G does not, the
/// call fails before allocating, naming the per-G byte count. Returns the
/// number of chunks.
///
/// Chunking does not change a single bit of `P`: the per-primitive-pair G
/// window uses `max |G|` over ALL of `gvecs` (not the chunk's), and every G
/// column is computed independently, so the chunks concatenate to exactly
/// `pair_ft_with_thresh(cell, prep, gvecs, thresh)`.
pub fn pair_ft_chunked<F>(
    cell: &Cell,
    prep: &PreparedBasis,
    gvecs: &[[f64; 3]],
    thresh: f64,
    chunk_budget_bytes: usize,
    extra_bytes_per_g: usize,
    mut sink: F,
) -> Result<usize, FerricError>
where
    F: FnMut(usize, &[[f64; 3]], &Array3<Complex64>) -> Result<(), FerricError>,
{
    validate_inputs(gvecs, thresh)?;
    if gvecs.is_empty() {
        return Ok(0);
    }
    let nao = prep.nbasis();
    let lmax = basis_lmax(prep);
    let per_g = pair_ft_bytes_per_g(nao, lmax).saturating_add(extra_bytes_per_g);
    Ledger::new(chunk_budget_bytes).check(
        &format!("pair_ft chunk, one G vector (nao = {nao}, lmax = {lmax})"),
        per_g,
    )?;
    let chunk = (chunk_budget_bytes / per_g).max(1);
    let gmax = max_gnorm(gvecs);
    let mut n_chunks = 0usize;
    for (c, gs) in gvecs.chunks(chunk).enumerate() {
        let p = pair_ft_block(cell, prep, gs, thresh, gmax)?;
        sink(c * chunk, gs, &p)?;
        n_chunks += 1;
    }
    Ok(n_chunks)
}

/// The kernel: `P` for `gvecs` with the G window evaluated at `gmax_window`
/// (`>= max |G|` of `gvecs`). Inputs already validated; no memory gate.
fn pair_ft_block(
    cell: &Cell,
    prep: &PreparedBasis,
    gvecs: &[[f64; 3]],
    thresh: f64,
    gmax_window: f64,
) -> Result<Array3<Complex64>, FerricError> {
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
    // The window bound must not depend on how the caller chunked G (see
    // `pair_ft_chunked`): the caller passes max |G| of the whole set.
    let gmax = gmax_window;

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

// ---------------------------------------------------------------------------
// Bra-centre derivative of the pair FT (Gamma-point forces, `crate::grad`).
// ---------------------------------------------------------------------------

/// Bytes [`pair_ft_deriv_chunked`] holds per G vector: `P` plus the three
/// `Q_x, Q_y, Q_z` columns (`4 × 16 nao²`) and the kernel's per-G scratch at
/// `l + 1` on the bra (the raised E table), times four for the Cartesian
/// staging of `P` and the three `Q`.
pub fn pair_ft_deriv_bytes_per_g(nao: usize, lmax: usize) -> usize {
    pair_ft_bytes_per_g(nao, lmax + 1).saturating_mul(4)
}

/// `(P, Q)` with `P[m,n,g]` the lattice-summed pair FT (as [`pair_ft`]) and
/// `Q[x][m,n,g] = ∂P[m,n,g]/∂A_{m,x}`: the derivative with respect to the
/// BRA function's centre only (the home-cell function `m`; the ket images
/// stay put). Materialises every G (memory-gated like
/// [`pair_ft_with_thresh`]); use [`pair_ft_deriv_chunked`] for large sets.
///
/// Identities (tested in `tests/pbc_grad.rs`): at a reciprocal-lattice G,
/// `P_mn = P_nm`, so the KET-centre derivative is `Q_nm`, and moving both
/// centres translates the pair: `Q_mn + Q_nm = −iG P_mn`. When atom `A` (all
/// images of its functions) moves, `∂P_mn/∂R_A = δ_{m∈A} Q_mn + δ_{n∈A} Q_nm`.
///
/// Method: `∂/∂A_x [x_A^i e^{−a x_A²}] = 2a x_A^{i+1} e^{−a x_A²} − i x_A^{i−1}
/// e^{−a x_A²}` (`x_A = x − A_x`), i.e. the McMurchie–Davidson E table with
/// the bra raised by one, combined as `2a F[i+1] − i F[i−1]`.
pub fn pair_ft_deriv(
    cell: &Cell,
    prep: &PreparedBasis,
    gvecs: &[[f64; 3]],
    thresh: f64,
) -> Result<(Array3<Complex64>, [Array3<Complex64>; 3]), FerricError> {
    validate_inputs(gvecs, thresh)?;
    let nao = prep.nbasis();
    let lmax = basis_lmax(prep);
    Ledger::new(crate::budget::resolve(None)).check(
        &format!(
            "pair_ft_deriv output P + 3 Q + scratch (nao = {nao}, n_G = {}, lmax = {lmax})",
            gvecs.len()
        ),
        bytes_of(gvecs.len() as u64, pair_ft_deriv_bytes_per_g(nao, lmax)),
    )?;
    pair_ft_deriv_block(cell, prep, gvecs, thresh, max_gnorm(gvecs))
}

/// [`pair_ft_deriv`] in G chunks: `sink(g0, gs, P, Q)` receives the chunk's
/// `P[:, :, j]` and `Q[x][:, :, j]` for `gs[j] = gvecs[g0 + j]`, in order.
/// Each chunk ([`pair_ft_deriv_bytes_per_g`] plus `extra_bytes_per_g`) fits
/// in `chunk_budget_bytes`; a single G that does not fit fails before
/// allocating. `Q` is never held for more than one chunk. Returns the number
/// of chunks. As for [`pair_ft_chunked`], the G window uses `max |G|` over
/// ALL of `gvecs`, so chunking does not change a bit.
pub fn pair_ft_deriv_chunked<F>(
    cell: &Cell,
    prep: &PreparedBasis,
    gvecs: &[[f64; 3]],
    thresh: f64,
    chunk_budget_bytes: usize,
    extra_bytes_per_g: usize,
    mut sink: F,
) -> Result<usize, FerricError>
where
    F: FnMut(
        usize,
        &[[f64; 3]],
        &Array3<Complex64>,
        &[Array3<Complex64>; 3],
    ) -> Result<(), FerricError>,
{
    validate_inputs(gvecs, thresh)?;
    if gvecs.is_empty() {
        return Ok(0);
    }
    let nao = prep.nbasis();
    let lmax = basis_lmax(prep);
    let per_g = pair_ft_deriv_bytes_per_g(nao, lmax).saturating_add(extra_bytes_per_g);
    Ledger::new(chunk_budget_bytes).check(
        &format!("pair_ft_deriv chunk, one G vector (nao = {nao}, lmax = {lmax})"),
        per_g,
    )?;
    let chunk = (chunk_budget_bytes / per_g).max(1);
    let gmax = max_gnorm(gvecs);
    let mut n_chunks = 0usize;
    for (c, gs) in gvecs.chunks(chunk).enumerate() {
        let (p, q) = pair_ft_deriv_block(cell, prep, gs, thresh, gmax)?;
        sink(c * chunk, gs, &p, &q)?;
        n_chunks += 1;
    }
    Ok(n_chunks)
}

/// Cartesian `[nca][ncb][ng]` shell block -> AO `[nfa][nfb]`, scattered into
/// `out[.., .., order[g]]` (pure shells through `ferric_cart2sph`).
fn scatter_shell_block(
    out: &mut Array3<Complex64>,
    cart: &[Complex64],
    sa: &FtShell,
    sb: &FtShell,
    c2s: &[Vec<f64>],
    order: &[usize],
) {
    let zero = Complex64::new(0.0, 0.0);
    let ng = order.len();
    let (nca, ncb) = (sa.ncart, sb.ncart);
    let (nfa, nfb) = (sa.nfun, sb.nfun);
    let right: Vec<Complex64> = if sb.pure {
        let cbm = &c2s[sb.l];
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
        cart.to_vec()
    };
    for i in 0..nfa {
        for j in 0..nfb {
            for g in 0..ng {
                let val = if sa.pure {
                    let cam = &c2s[sa.l];
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

/// The derivative kernel: `pair_ft_block` with the bra E table raised by one.
/// Screening: a primitive pair is skipped when `mag (1 + 2a) < thresh`
/// (`2a` bounds the raised term's prefactor), and the G window gains one
/// power of `|G|_max` and `ln(1 + 2a)` over `pair_ft_block`'s. The image set
/// is `pair_ft_block`'s radius at `thresh / (1 + 2 a_max)`.
fn pair_ft_deriv_block(
    cell: &Cell,
    prep: &PreparedBasis,
    gvecs: &[[f64; 3]],
    thresh: f64,
    gmax_window: f64,
) -> Result<(Array3<Complex64>, [Array3<Complex64>; 3]), FerricError> {
    let shells = build_shells(cell, prep)?;
    let nbf = prep.nbasis();
    let ng = gvecs.len();
    let mut out = Array3::<Complex64>::zeros((nbf, nbf, ng));
    let mut outq = [
        Array3::<Complex64>::zeros((nbf, nbf, ng)),
        Array3::<Complex64>::zeros((nbf, nbf, ng)),
        Array3::<Complex64>::zeros((nbf, nbf, ng)),
    ];
    if ng == 0 || shells.is_empty() {
        return Ok((out, outq));
    }
    let amin = shells
        .iter()
        .flat_map(|s| s.exps.iter().copied())
        .fold(f64::INFINITY, f64::min);
    let amax = shells
        .iter()
        .flat_map(|s| s.exps.iter().copied())
        .fold(0.0_f64, f64::max);
    if !(amin > 0.0) {
        return Err(FerricError::Basis(format!(
            "pair_ft_deriv: smallest exponent is {amin}; must be > 0"
        )));
    }
    let rpair = (2.0 * (1e3 * (1.0 + 2.0 * amax) / thresh).ln() / amin).sqrt() + 2.0;
    let images = cell.translations(rpair)?;

    let norm2 = |g: &[f64; 3]| g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    let mut order: Vec<usize> = (0..ng).collect();
    order.sort_by(|&x, &y| norm2(&gvecs[x]).total_cmp(&norm2(&gvecs[y])));
    let gsorted: Vec<[f64; 3]> = order.iter().map(|&i| gvecs[i]).collect();
    let gvecs: &[[f64; 3]] = &gsorted;
    let gmax = gmax_window;

    let lmax = shells.iter().map(|s| s.l).max().unwrap_or(0);
    // t runs to (la + 1) + lb.
    let nt = 2 * lmax + 2;
    let g2: Vec<f64> = gvecs.iter().map(norm2).collect();
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
            let la1 = la + 1;
            let (nca, ncb) = (sa.ncart, sb.ncart);
            let st = la1 + lb + 1;
            let nij = (la1 + 1) * (lb + 1);
            for d in 0..3 {
                ebuf[d].resize(nij * st, 0.0);
                fbuf[d].resize(nij * ng, zero);
            }
            let mut cart = vec![zero; nca * ncb * ng];
            let mut cartq = [
                vec![zero; nca * ncb * ng],
                vec![zero; nca * ncb * ng],
                vec![zero; nca * ncb * ng],
            ];
            let ca_comps = &comps[la];
            let cb_comps = &comps[lb];
            // Row offset of F_d[i][j] in fbuf[d].
            let row = |i: usize, j: usize| (i * (lb + 1) + j) * ng;

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
                        let mag = (cc * (-a * b / p * r2).exp()).abs() * (1.0 + 2.0 * a);
                        if mag < thresh {
                            continue;
                        }
                        let g2max = 4.0
                            * p
                            * ((mag / thresh).ln().max(0.0)
                                + WINDOW_MARGIN
                                + (la + lb + 1) as f64 * gmax.max(1.0).ln());
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
                            let m = cc * (-g2[g] / (4.0 * p)).exp();
                            let ph = gv[0] * pc[0] + gv[1] * pc[1] + gv[2] * pc[2];
                            common[g] = Complex64::new(m * ph.cos(), -m * ph.sin());
                        }
                        for d in 0..3 {
                            e_table(la1, lb, a, b, ab[d], &mut ebuf[d]);
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
                            let (ax, ay, az) = (ac[0] as usize, ac[1] as usize, ac[2] as usize);
                            for (v, bcmp) in cb_comps.iter().enumerate() {
                                let (bx, by, bz) =
                                    (bcmp[0] as usize, bcmp[1] as usize, bcmp[2] as usize);
                                let (ix, iy, iz) = (row(ax, bx), row(ay, by), row(az, bz));
                                let (ixp, iyp, izp) =
                                    (row(ax + 1, bx), row(ay + 1, by), row(az + 1, bz));
                                let base = (u * ncb + v) * ng;
                                for g in 0..ngp {
                                    let c = common[g];
                                    let (x, y, z) = (fx[ix + g], fy[iy + g], fz[iz + g]);
                                    // 2a F[i+1] − i F[i−1] per direction.
                                    let mut dx = fx[ixp + g] * (2.0 * a);
                                    if ax > 0 {
                                        dx -= fx[row(ax - 1, bx) + g] * ax as f64;
                                    }
                                    let mut dy = fy[iyp + g] * (2.0 * a);
                                    if ay > 0 {
                                        dy -= fy[row(ay - 1, by) + g] * ay as f64;
                                    }
                                    let mut dz = fz[izp + g] * (2.0 * a);
                                    if az > 0 {
                                        dz -= fz[row(az - 1, bz) + g] * az as f64;
                                    }
                                    cart[base + g] += c * x * y * z;
                                    cartq[0][base + g] += c * dx * y * z;
                                    cartq[1][base + g] += c * x * dy * z;
                                    cartq[2][base + g] += c * x * y * dz;
                                }
                            }
                        }
                    }
                }
            }
            scatter_shell_block(&mut out, &cart, sa, sb, &c2s, &order);
            for d in 0..3 {
                scatter_shell_block(&mut outq[d], &cartq[d], sa, sb, &c2s, &order);
            }
        }
    }
    Ok((out, outq))
}

// ---------------------------------------------------------------------------
// Strain derivative of the pair FT at fixed Miller indices (Gamma stress,
// `crate::stress`).
// ---------------------------------------------------------------------------

/// Which parts of `dP/dε` [`pair_ft_strain_chunked`] includes. Production is
/// [`PairFtStrainTerms::ALL`]; the others are the stress mutation seams
/// (`crate::stress::StressMutation::{FtNoCentres, GUnstrained}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairFtStrainTerms {
    /// The basis-centre (pair-vector) part `[Q_a + iG_a (α/p) P] (A − B′)_b`.
    pub centres: bool,
    /// The G-shape part `G_a G_b/(2p) P − G_a P^g_b`.
    pub g_shape: bool,
}

impl PairFtStrainTerms {
    /// Both parts (production).
    pub const ALL: Self = Self {
        centres: true,
        g_shape: true,
    };
}

/// Bytes [`pair_ft_strain_chunked`] holds per G vector: `P` plus the nine
/// `dP/dε_ab` columns and their Cartesian staging, with the bra-raised E
/// table and the G-derivative rows as scratch (a conservative
/// `11 ×` [`pair_ft_bytes_per_g`] at `l + 1`).
pub fn pair_ft_strain_bytes_per_g(nao: usize, lmax: usize) -> usize {
    pair_ft_bytes_per_g(nao, lmax + 1).saturating_mul(11)
}

/// `(P, dP)` with `P` the lattice-summed pair FT ([`pair_ft`]) and
/// `dP[3a + b][m, n, g] = dP[m,n,g]/dε_ab` under the homogeneous strain
/// `r → (1 + ε) r` of the atoms AND the lattice, at FIXED Miller index
/// (`G → (1 + ε)^{−T} G`). Per primitive pair (bra `α` at `A`, ket `β` at
/// `B′ = B + L`, `p = α + β`, `P_c` the product centre):
///
/// ```text
/// P_prim(G) = e^{−iG·P_c} R(G, A − B′),   G·P_c strain-invariant, so
/// dP/dε_ab = [Q_a + i G_a (α/p) P] (A − B′)_b  +  G_a G_b/(2p) P  −  G_a P^g_b
/// ```
///
/// with `Q_a = ∂P/∂A_a` (the bra-centre derivative of [`pair_ft_deriv`]:
/// `2α F[i+1] − i F[i−1]`) and `P^g_b` the `G_b`-derivative of the Hermite
/// polynomial factor, `Σ_t E_t t (−i) (−iG_b)^{t−1}` in direction `b` (the
/// Gaussian factor's `G` dependence is the `G_a G_b/(2p)` term). FINDINGS
/// "Iteration 19"; prototype `pbc_stress.pair_ft_strain`.
///
/// Chunked like [`pair_ft_deriv_chunked`] (same screening, image radius and
/// window rules, one extra power of `|G|_max` in the window): `sink(g0, gs,
/// P, dP)` for each chunk, `dP` never held for more than one chunk. `terms`
/// selects the parts (mutation seams). Returns the number of chunks.
#[allow(clippy::too_many_arguments)]
pub fn pair_ft_strain_chunked<F>(
    cell: &Cell,
    prep: &PreparedBasis,
    gvecs: &[[f64; 3]],
    thresh: f64,
    chunk_budget_bytes: usize,
    extra_bytes_per_g: usize,
    terms: PairFtStrainTerms,
    mut sink: F,
) -> Result<usize, FerricError>
where
    F: FnMut(
        usize,
        &[[f64; 3]],
        &Array3<Complex64>,
        &[Array3<Complex64>; 9],
    ) -> Result<(), FerricError>,
{
    validate_inputs(gvecs, thresh)?;
    if gvecs.is_empty() {
        return Ok(0);
    }
    let nao = prep.nbasis();
    let lmax = basis_lmax(prep);
    let per_g = pair_ft_strain_bytes_per_g(nao, lmax).saturating_add(extra_bytes_per_g);
    Ledger::new(chunk_budget_bytes).check(
        &format!("pair_ft_strain chunk, one G vector (nao = {nao}, lmax = {lmax})"),
        per_g,
    )?;
    let chunk = (chunk_budget_bytes / per_g).max(1);
    let gmax = max_gnorm(gvecs);
    let mut n_chunks = 0usize;
    for (c, gs) in gvecs.chunks(chunk).enumerate() {
        let (p, dp) = pair_ft_strain_block(cell, prep, gs, thresh, gmax, terms)?;
        sink(c * chunk, gs, &p, &dp)?;
        n_chunks += 1;
    }
    Ok(n_chunks)
}

/// The strain kernel: [`pair_ft_deriv_block`]'s loop with the G-derivative
/// rows added and the nine `dP/dε_ab` accumulated per primitive pair.
fn pair_ft_strain_block(
    cell: &Cell,
    prep: &PreparedBasis,
    gvecs: &[[f64; 3]],
    thresh: f64,
    gmax_window: f64,
    terms: PairFtStrainTerms,
) -> Result<(Array3<Complex64>, [Array3<Complex64>; 9]), FerricError> {
    let shells = build_shells(cell, prep)?;
    let nbf = prep.nbasis();
    let ng = gvecs.len();
    let mut out = Array3::<Complex64>::zeros((nbf, nbf, ng));
    let mut outd: [Array3<Complex64>; 9] =
        std::array::from_fn(|_| Array3::<Complex64>::zeros((nbf, nbf, ng)));
    if ng == 0 || shells.is_empty() {
        return Ok((out, outd));
    }
    let amin = shells
        .iter()
        .flat_map(|s| s.exps.iter().copied())
        .fold(f64::INFINITY, f64::min);
    let amax = shells
        .iter()
        .flat_map(|s| s.exps.iter().copied())
        .fold(0.0_f64, f64::max);
    if !(amin > 0.0) {
        return Err(FerricError::Basis(format!(
            "pair_ft_strain: smallest exponent is {amin}; must be > 0"
        )));
    }
    let rpair = (2.0 * (1e3 * (1.0 + 2.0 * amax) / thresh).ln() / amin).sqrt() + 2.0;
    let images = cell.translations(rpair)?;

    let norm2 = |g: &[f64; 3]| g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    let mut order: Vec<usize> = (0..ng).collect();
    order.sort_by(|&x, &y| norm2(&gvecs[x]).total_cmp(&norm2(&gvecs[y])));
    let gsorted: Vec<[f64; 3]> = order.iter().map(|&i| gvecs[i]).collect();
    let gvecs: &[[f64; 3]] = &gsorted;
    let gmax = gmax_window;

    let lmax = shells.iter().map(|s| s.l).max().unwrap_or(0);
    let nt = 2 * lmax + 2;
    let g2: Vec<f64> = gvecs.iter().map(norm2).collect();
    let zero = Complex64::new(0.0, 0.0);
    // pw[d][t·ng + g] = (−iG_d)^t;  dpw[d][t·ng + g] = t (−i) (−iG_d)^{t−1}.
    let pw: Vec<Vec<Complex64>> = (0..3)
        .map(|d| {
            let mut v = vec![zero; nt * ng];
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
    let dpw: Vec<Vec<Complex64>> = (0..3)
        .map(|d| {
            let mut v = vec![zero; nt * ng];
            for t in 1..nt {
                for g in 0..ng {
                    v[t * ng + g] = pw[d][(t - 1) * ng + g] * Complex64::new(0.0, -(t as f64));
                }
            }
            v
        })
        .collect();
    let comps: Vec<Vec<[u8; 3]>> = (0..=lmax).map(cart_components).collect();
    let c2s: Vec<Vec<f64>> = (0..=lmax).map(ferric_cart2sph).collect();

    let pi = std::f64::consts::PI;
    let mut common = vec![zero; ng];
    let mut ebuf: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut fbuf: [Vec<Complex64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut gbuf: [Vec<Complex64>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    for sa in &shells {
        for sb in &shells {
            let (la, lb) = (sa.l, sb.l);
            let la1 = la + 1;
            let (nca, ncb) = (sa.ncart, sb.ncart);
            let st = la1 + lb + 1;
            let nij = (la1 + 1) * (lb + 1);
            for d in 0..3 {
                ebuf[d].resize(nij * st, 0.0);
                fbuf[d].resize(nij * ng, zero);
                gbuf[d].resize(nij * ng, zero);
            }
            let mut cart = vec![zero; nca * ncb * ng];
            let mut cartd: Vec<Vec<Complex64>> =
                (0..9).map(|_| vec![zero; nca * ncb * ng]).collect();
            let ca_comps = &comps[la];
            let cb_comps = &comps[lb];
            let row = |i: usize, j: usize| (i * (lb + 1) + j) * ng;

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
                        let mag = (cc * (-a * b / p * r2).exp()).abs() * (1.0 + 2.0 * a);
                        if mag < thresh {
                            continue;
                        }
                        let g2max = 4.0
                            * p
                            * ((mag / thresh).ln().max(0.0)
                                + WINDOW_MARGIN
                                + (la + lb + 2) as f64 * gmax.max(1.0).ln());
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
                            let m = cc * (-g2[g] / (4.0 * p)).exp();
                            let ph = gv[0] * pc[0] + gv[1] * pc[1] + gv[2] * pc[2];
                            common[g] = Complex64::new(m * ph.cos(), -m * ph.sin());
                        }
                        for d in 0..3 {
                            e_table(la1, lb, a, b, ab[d], &mut ebuf[d]);
                            let e = &ebuf[d];
                            let (f, fg) = (&mut fbuf[d], &mut gbuf[d]);
                            let (w, dw) = (&pw[d], &dpw[d]);
                            for ij in 0..nij {
                                let frow = &mut f[ij * ng..ij * ng + ngp];
                                let grow = &mut fg[ij * ng..ij * ng + ngp];
                                frow.fill(zero);
                                grow.fill(zero);
                                for t in 0..st {
                                    let et = e[ij * st + t];
                                    if et == 0.0 {
                                        continue;
                                    }
                                    let wrow = &w[t * ng..t * ng + ngp];
                                    let dwrow = &dw[t * ng..t * ng + ngp];
                                    for ((x, y), (wv, dv)) in frow
                                        .iter_mut()
                                        .zip(grow.iter_mut())
                                        .zip(wrow.iter().zip(dwrow))
                                    {
                                        *x += *wv * et;
                                        *y += *dv * et;
                                    }
                                }
                            }
                        }
                        let (fx, fy, fz) = (&fbuf[0], &fbuf[1], &fbuf[2]);
                        let (hx, hy, hz) = (&gbuf[0], &gbuf[1], &gbuf[2]);
                        let a_over_p = a / p;
                        let inv2p = 0.5 / p;
                        for (u, ac) in ca_comps.iter().enumerate() {
                            let (ax, ay, az) = (ac[0] as usize, ac[1] as usize, ac[2] as usize);
                            for (v, bcmp) in cb_comps.iter().enumerate() {
                                let (bx, by, bz) =
                                    (bcmp[0] as usize, bcmp[1] as usize, bcmp[2] as usize);
                                let (ix, iy, iz) = (row(ax, bx), row(ay, by), row(az, bz));
                                let (ixp, iyp, izp) =
                                    (row(ax + 1, bx), row(ay + 1, by), row(az + 1, bz));
                                let base = (u * ncb + v) * ng;
                                for g in 0..ngp {
                                    let c = common[g];
                                    let gv = &gvecs[g];
                                    let (x, y, z) = (fx[ix + g], fy[iy + g], fz[iz + g]);
                                    let p0 = c * x * y * z;
                                    cart[base + g] += p0;
                                    let mut dmat = [[zero; 3]; 3];
                                    if terms.centres {
                                        let mut dx = fx[ixp + g] * (2.0 * a);
                                        if ax > 0 {
                                            dx -= fx[row(ax - 1, bx) + g] * ax as f64;
                                        }
                                        let mut dy = fy[iyp + g] * (2.0 * a);
                                        if ay > 0 {
                                            dy -= fy[row(ay - 1, by) + g] * ay as f64;
                                        }
                                        let mut dz = fz[izp + g] * (2.0 * a);
                                        if az > 0 {
                                            dz -= fz[row(az - 1, bz) + g] * az as f64;
                                        }
                                        let q = [c * dx * y * z, c * x * dy * z, c * x * y * dz];
                                        for i in 0..3 {
                                            // e^{−iG·Pc} ∂R/∂(A−B′)_i = Q_i + i G_i (α/p) P
                                            let r =
                                                q[i] + Complex64::new(0.0, gv[i] * a_over_p) * p0;
                                            for j in 0..3 {
                                                dmat[i][j] += r * ab[j];
                                            }
                                        }
                                    }
                                    if terms.g_shape {
                                        let pg = [
                                            c * hx[ix + g] * y * z,
                                            c * x * hy[iy + g] * z,
                                            c * x * y * hz[iz + g],
                                        ];
                                        for i in 0..3 {
                                            for j in 0..3 {
                                                dmat[i][j] +=
                                                    p0 * (gv[i] * gv[j] * inv2p) - pg[j] * gv[i];
                                            }
                                        }
                                    }
                                    for i in 0..3 {
                                        for j in 0..3 {
                                            cartd[3 * i + j][base + g] += dmat[i][j];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            scatter_shell_block(&mut out, &cart, sa, sb, &c2s, &order);
            for k in 0..9 {
                scatter_shell_block(&mut outd[k], &cartd[k], sa, sb, &c2s, &order);
            }
        }
    }
    Ok((out, outd))
}
