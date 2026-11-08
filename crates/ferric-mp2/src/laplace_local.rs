//! Laplace SOS-MP2, `local` formulation: Boys LMOs on the occupied side,
//! VV-HV orthonormal localized virtuals (LVOs, [`crate::lmp2_amplitude::build_vvhv`])
//! on the virtual side, per-LMO orbital domains, and the global Coulomb metric.
//!
//! # Why this exists
//!
//! The `ao-sparse` path ([`crate::laplace::LaplaceMp2::compute_sos_ao`] with a
//! cutoff) applies its domain cutoff to the VALUES of the occupied
//! pseudo-density only. Every loop and tensor stays `naux·nao²`, so the cutoff
//! changes the energy but never the cost. Here the cutoff sets the INNER index
//! of the dense Gram GEMM.
//!
//! # Why the virtual side is NOT in AOs
//!
//! The first version of this module kept the virtual index in AOs, with the
//! AO virtual pseudo-density `Q(t) = C_v e^{-tε} C_vᵀ`, and truncated that AO
//! index to a sphere around each LMO. Measured on C12H26/6-31G (2026-11-07,
//! wiki/notes/laplace-aosparse-2026-11-07/): E_OS = -8.01 / -0.940 / -0.924 /
//! -0.871 Ha at r = 4 / 6 / 8 / 10 Bohr, against the exact -0.8654 (STO-3G
//! hid this). The cause is the AO cut itself (see TA below): it breaks the
//! orthogonality of the virtual side to LMO i. With localized virtuals the
//! cost-setting truncation drops whole `(i,a)` pairs instead, and each of
//! those is a chargeless density. lmp2_direct makes the same choice.
//!
//! # Algebra
//!
//! `L = C_o U` (Boys, `U = C_oᵀ S L`) and `Cvl = C_v R` (VV-HV,
//! `R = C_vᵀ S Cvl`), both orthogonal rotations. Exactly,
//! ```text
//! E(t) = Uᵀ diag(e^{+t ε_o}) U,   Q(t) = Rᵀ diag(e^{-t ε_v}) R
//! Z̃_PQ(t) = Σ_ij Σ_ab E_ij(t) (P|ia) Q_ab(t) (Q|jb)
//! E_OS    = -Σ_k w_k ‖L_V⁻¹ Z̃(t_k) L_V⁻ᵀ‖²_F
//! ```
//! which is the canonical MO expression rotated, so nothing is approximated
//! before the truncations below.
//!
//! # Truncations (Bohr). Each is EXACT as its radius → ∞
//!
//! * T1 `occ`: LMO coefficient support, `L_{μi}` for AO shells within `occ` of
//!   centroid i.
//! * TA `pair_tau`: the virtual-side AO index of the half transform
//!   `X^P_{iλ} = Σ_μ L_{μi}(P|μλ)` is kept for shells λ that are Schwarz-
//!   significant (`Q_{μλ} ≥ pair_tau`) with some μ shell of the LMO support.
//!   It is an integral-significance rule and deliberately not a radius. A
//!   radius cut on λ was measured on C12H26/6-31G at 6/8/10/12/14 Bohr to give
//!   +8.6/+6.8/+0.63/+0.10/+0.016% overestimates of |E_OS|. Cutting an
//!   orbital's AO expansion breaks `⟨i|a⟩ = 0`, and the charged `(ia)` density
//!   couples through the 1/R tail. The same numbers reproduce the AO-virtual
//!   failure above, which was this effect, not `Q(t)`.
//! * T2 `vir`: virtual domain `V_i` = LVOs whose centroid lies within `vir` of
//!   centroid i; `(P|ia) = Σ_λ X^P_{iλ} Cvl_{λa}` is kept for `a ∈ V_i` only.
//!   This one sets the cost.
//! * T3 `pair`: `E_ij` is kept when `V_i` and `V_j` share an LVO.
//!
//! # Cost
//!
//! With `T = {(i,a): a ∈ V_i}`, the per-point Gram is one `naux × |T| × naux`
//! GEMM and the whitening is one `naux × naux × |T|` GEMM. `|T| = Σ_i |V_i|`
//! grows linearly with system size at fixed radius, against `no·nv` (MO path)
//! or `nao²` (AO paths). The `naux²` factor stays: with the Coulomb metric the
//! aux Gram is dense (1/R tails), and removing it needs a local metric or
//! local robust fits, which this path does not attempt. Memory is
//! `naux·|T|`, with no `naux·nao²` and no `naux·no·nv` tensor. The VV-HV build
//! and Boys localization are dense `nao³`-class steps.

use std::time::Instant;

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;
use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array2};
use rayon::prelude::*;

use crate::boys::boys_localize;
use crate::lmp2_amplitude::{checked_vvhv, column_expectations};

/// Thresholds of the truncations; see the module doc.
#[derive(Debug, Clone, Copy)]
pub struct LocalRadii {
    /// T2: virtual domain radius, Bohr (LVO centroid to LMO centroid). Sets `|T|`.
    pub vir: f64,
    /// TA: shell `λ` joins `N_i` when `max_{μ-shell ∈ D1_i} Q_{μλ} ≥ pair_tau`
    /// (Schwarz). NOT a radius: a distance cut on an orbital's AO expansion
    /// breaks `⟨i|a⟩ = 0`, gives the `(ia)` density a spurious charge, and the
    /// 1/R tail turns that into a large overestimate (measured, module doc).
    /// `0.0` is vacuous. Costs integrals only.
    pub pair_tau: f64,
    /// T1: LMO coefficient support radius, Bohr. Costs integrals only.
    pub occ: f64,
    /// TE (optional energy cut, applied AFTER T2 and the whitening): keep
    /// `(i,a)` only when `s_ia = (ia|ia)/(f_aa - f_ii) ≥ energy_tau` (Hartree),
    /// with `(ia|ia) = Σ_P (B̂^P_ia)²` and the diagonal localized Fock
    /// elements. Selects per LMO by a pair-energy-like score instead of
    /// distance. `0.0` is vacuous.
    pub energy_tau: f64,
}

impl LocalRadii {
    /// The single user radius `r` drives T2. T1 and TA cost only `O(N²)`
    /// integral work, so they are held at `max(r, OCC_FLOOR)` and
    /// [`Self::PAIR_TAU`] and never become the leading error at small `r`.
    pub fn from_cutoff(r: f64) -> Self {
        Self {
            vir: r,
            pair_tau: Self::PAIR_TAU,
            occ: r.max(Self::OCC_FLOOR),
            energy_tau: 0.0,
        }
    }

    /// Floor (Bohr) of the T1 radius in [`Self::from_cutoff`].
    pub const OCC_FLOOR: f64 = 12.0;
    /// Schwarz threshold of TA in [`Self::from_cutoff`].
    pub const PAIR_TAU: f64 = 1e-10;
}

/// Structural counters and stage timings of one `local` evaluation.
#[derive(Debug, Clone, Default)]
pub struct LocalStats {
    /// Active (localized) occupied orbitals and localized virtuals.
    pub n_lmo: usize,
    pub n_lvo: usize,
    /// `|T| = Σ_i |V_i|`, the inner GEMM dimension (MO path: `n_lmo·n_lvo`),
    /// after the energy cut, and the geometric `|T|` before it.
    pub t_size: usize,
    pub t_size_geom: usize,
    /// Mean and max `|V_i|`.
    pub vdom_mean: f64,
    pub vdom_max: usize,
    /// Mean AO neighbourhood `|N_i|` (TA).
    pub ao_mean: f64,
    /// Kept ordered `(i, j)` occupied pairs (T3), including `i = j`.
    pub n_pairs: usize,
    /// Distinct shell pairs whose 3-index integrals are evaluated, and the total.
    pub n_shell_pairs: usize,
    pub n_shell_pairs_total: usize,
    pub boys_converged: bool,
    /// Wall seconds: Boys + VV-HV + domains, integrals + half transform,
    /// whitening, quadrature points.
    pub t_spaces: f64,
    pub t_eri3: f64,
    pub t_whiten: f64,
    pub t_points: f64,
}

impl std::fmt::Display for LocalStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sos-local: n_lmo={} n_lvo={} t_size={} t_geom={} t_full={} vdom_mean={:.1} vdom_max={} \
             ao_mean={:.1} pairs={} shell_pairs={}/{} boys_conv={} t_spaces={:.3} \
             t_eri3={:.3} t_whiten={:.3} t_points={:.3}",
            self.n_lmo,
            self.n_lvo,
            self.t_size,
            self.t_size_geom,
            self.n_lmo * self.n_lvo,
            self.vdom_mean,
            self.vdom_max,
            self.ao_mean,
            self.n_pairs,
            self.n_shell_pairs,
            self.n_shell_pairs_total,
            self.boys_converged,
            self.t_spaces,
            self.t_eri3,
            self.t_whiten,
            self.t_points
        )
    }
}

/// Per-LMO domains and the derived index sets.
struct LocalDomains {
    /// Per LMO: AO indices of the neighbourhood `N_i`, in shell order.
    ao: Vec<Vec<usize>>,
    /// Column offsets of each LMO's block of `X` (over `N_i`); length `n_lmo + 1`.
    x_off: Vec<usize>,
    /// Per LMO, per shell: local AO offset of that shell in `N_i`, or `u32::MAX`.
    shell_pos: Vec<Vec<u32>>,
    /// Per LMO: sorted LVO indices of `V_i`.
    vir: Vec<Vec<usize>>,
    /// Column offsets of each LMO's block of `T`; length `n_lmo + 1`.
    t_off: Vec<usize>,
    /// Per LMO: kept partners `j` (T3), sorted, including `i`.
    nbrs: Vec<Vec<usize>>,
    /// Shell pairs `(s1, s2)`, `s1 ≥ s2`, with the LMOs they feed and how
    /// (bit 1: μ ∈ s1 ⊂ D1_i, λ ∈ s2 ⊂ N_i; bit 2: μ ∈ s2 ⊂ D1_i, λ ∈ s1 ⊂ N_i).
    pair_owners: Vec<((usize, usize), Vec<(u32, u8)>)>,
}

fn dist2(a: &Array2<f64>, i: usize, b: [f64; 3]) -> f64 {
    (0..3).map(|k| (a[(i, k)] - b[k]).powi(2)).sum()
}

fn build_local_domains(
    occ_c: &Array2<f64>,
    vir_c: &Array2<f64>,
    obs: &PreparedBasis,
    q_sh: &Array2<f64>,
    radii: LocalRadii,
) -> LocalDomains {
    let n_lmo = occ_c.nrows();
    let n_lvo = vir_c.nrows();
    let nsh = obs.nshells();
    let sc = obs.shell_centers();
    let offs = obs.shell_offsets();
    let shells_within = |r: f64| -> Vec<Vec<usize>> {
        (0..n_lmo)
            .map(|i| {
                (0..nsh)
                    .filter(|&s| dist2(occ_c, i, sc[s]) <= r * r)
                    .collect()
            })
            .collect()
    };
    let occ_shells = shells_within(radii.occ);
    // TA: Schwarz-significant partners of the LMO support.
    let nshells: Vec<Vec<usize>> = occ_shells
        .iter()
        .map(|d1| {
            (0..nsh)
                .filter(|&y| d1.iter().any(|&x| q_sh[(x, y)] >= radii.pair_tau))
                .collect()
        })
        .collect();

    let mut ao = Vec::with_capacity(n_lmo);
    let mut shell_pos = Vec::with_capacity(n_lmo);
    let mut x_off = vec![0usize; n_lmo + 1];
    for (i, dom) in nshells.iter().enumerate() {
        let mut pos = vec![u32::MAX; nsh];
        let mut list = Vec::new();
        for &s in dom {
            pos[s] = list.len() as u32;
            list.extend(offs[s]..offs[s + 1]);
        }
        x_off[i + 1] = x_off[i] + list.len();
        ao.push(list);
        shell_pos.push(pos);
    }

    let vir: Vec<Vec<usize>> = (0..n_lmo)
        .map(|i| {
            (0..n_lvo)
                .filter(|&a| {
                    let c = [vir_c[(a, 0)], vir_c[(a, 1)], vir_c[(a, 2)]];
                    dist2(occ_c, i, c) <= radii.vir * radii.vir
                })
                .collect()
        })
        .collect();
    let mut t_off = vec![0usize; n_lmo + 1];
    for (i, v) in vir.iter().enumerate() {
        t_off[i + 1] = t_off[i] + v.len();
    }

    let nbrs = partners(&vir, n_lvo);

    let mut owners: std::collections::BTreeMap<(usize, usize), Vec<(u32, u8)>> = Default::default();
    for i in 0..n_lmo {
        let mut bits: std::collections::BTreeMap<(usize, usize), u8> = Default::default();
        for &x in &occ_shells[i] {
            for &y in &nshells[i] {
                if x >= y {
                    *bits.entry((x, y)).or_default() |= 1;
                } else {
                    *bits.entry((y, x)).or_default() |= 2;
                }
            }
        }
        for (k, b) in bits {
            owners.entry(k).or_default().push((i as u32, b));
        }
    }

    LocalDomains {
        ao,
        x_off,
        shell_pos,
        vir,
        t_off,
        nbrs,
        pair_owners: owners.into_iter().collect(),
    }
}

/// T3: `j` is a partner of `i` when `V_i` and `V_j` share an LVO.
fn partners(vir: &[Vec<usize>], n_lvo: usize) -> Vec<Vec<usize>> {
    let n_lmo = vir.len();
    let mut lmos_of_lvo: Vec<Vec<usize>> = vec![Vec::new(); n_lvo];
    for (i, v) in vir.iter().enumerate() {
        for &a in v {
            lmos_of_lvo[a].push(i);
        }
    }
    vir.iter()
        .map(|v| {
            let mut seen = vec![false; n_lmo];
            for &a in v {
                for &j in &lmos_of_lvo[a] {
                    seen[j] = true;
                }
            }
            (0..n_lmo).filter(|&j| seen[j]).collect()
        })
        .collect()
}

/// TE: keep the `(i,a)` columns of `Ĥ` whose score `(ia|ia)/(f_aa - f_ii)`
/// reaches `tau`; rebuild `V_i`, the `T` offsets and the T3 partners.
fn energy_cut(
    h_hat: Array2<f64>,
    dom: &mut LocalDomains,
    f_occ: &[f64],
    f_vir: &[f64],
    tau: f64,
    n_lvo: usize,
) -> Array2<f64> {
    let mut keep_cols = Vec::new();
    let mut new_vir = Vec::with_capacity(dom.vir.len());
    for (i, v) in dom.vir.iter().enumerate() {
        let mut kept = Vec::new();
        for (q, &a) in v.iter().enumerate() {
            let col = dom.t_off[i] + q;
            let norm2: f64 = h_hat.column(col).iter().map(|x| x * x).sum();
            if norm2 / (f_vir[a] - f_occ[i]) >= tau {
                kept.push(a);
                keep_cols.push(col);
            }
        }
        new_vir.push(kept);
    }
    let mut t_off = vec![0usize; new_vir.len() + 1];
    for (i, v) in new_vir.iter().enumerate() {
        t_off[i + 1] = t_off[i] + v.len();
    }
    dom.nbrs = partners(&new_vir, n_lvo);
    dom.vir = new_vir;
    dom.t_off = t_off;
    h_hat.select(ndarray::Axis(1), &keep_cols)
}

/// Occupied half transform `X^P_{iλ} = Σ_{μ∈D1_i} L_{μi}(P|μλ)`, `λ ∈ N_i`,
/// shape `(naux, Σ_i |N_i|)`.
///
/// Parallel over aux shells: each aux shell owns a disjoint band of rows, so the
/// raw-pointer accumulation below never races (same argument as
/// `threeindex::eri3_tensor`).
fn occ_half_transform(
    op: Operator,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    l: &Array2<f64>,
    dom: &LocalDomains,
) -> Result<Array2<f64>, FerricError> {
    let naux = dfbs.nbasis();
    let ntot = *dom.x_off.last().unwrap();
    let dims_obs = obs.shell_dims();
    let offs_obs = obs.shell_offsets();
    let dims_df = dfbs.shell_dims();
    let offs_df = dfbs.shell_offsets();
    Engine::new_3center(op, obs, dfbs, 1e-14)?;

    let mut x = Array2::<f64>::zeros((naux, ntot));
    let x_ptr = x.as_mut_ptr() as usize;
    (0..dfbs.nshells()).into_par_iter().for_each_init(
        || Engine::new_3center(op, obs, dfbs, 1e-14).expect("3-center engine (pre-validated)"),
        |eng, sp| {
            let np = dims_df[sp];
            let p0 = offs_df[sp];
            for ((s1, s2), owners) in &dom.pair_owners {
                let (s1, s2) = (*s1, *s2);
                let Some(block) = eng.compute_eri3(obs, dfbs, sp, s1, s2) else {
                    continue;
                };
                let (n1, n2) = (dims_obs[s1], dims_obs[s2]);
                let (m0, n0) = (offs_obs[s1], offs_obs[s2]);
                for &(i, bits) in owners {
                    let i = i as usize;
                    let ci = dom.x_off[i];
                    let fwd = bits & 1 != 0;
                    let rev = bits & 2 != 0 && s1 != s2;
                    let pos1 = dom.shell_pos[i][s1] as usize;
                    let pos2 = dom.shell_pos[i][s2] as usize;
                    for p in 0..np {
                        // SAFETY: row p0+p belongs to aux shell sp, which only this
                        // worker touches; columns stay inside [0, ntot).
                        let row = unsafe {
                            std::slice::from_raw_parts_mut(
                                (x_ptr as *mut f64).add((p0 + p) * ntot),
                                ntot,
                            )
                        };
                        for a in 0..n1 {
                            let l_mu = l[(m0 + a, i)];
                            for b in 0..n2 {
                                let v = block[(p * n1 + a) * n2 + b];
                                if fwd {
                                    // μ in s1 (D1_i), λ in s2 (N_i).
                                    row[ci + pos2 + b] += l_mu * v;
                                }
                                if rev {
                                    // μ in s2 (D1_i), λ in s1 (N_i): (P|λμ) = (P|μλ).
                                    row[ci + pos1 + a] += l[(n0 + b, i)] * v;
                                }
                            }
                        }
                    }
                }
            }
        },
    );
    Ok(x)
}

/// Virtual transform `(P|ia) = Σ_{λ∈N_i} X^P_{iλ} Cvl_{λa}`, `a ∈ V_i`, laid out
/// over `T`, shape `(naux, |T|)`.
fn vir_transform(x: &Array2<f64>, c_vl: &Array2<f64>, dom: &LocalDomains) -> Array2<f64> {
    let naux = x.nrows();
    let blocks: Vec<Array2<f64>> = (0..dom.vir.len())
        .into_par_iter()
        .map(|i| {
            let n_i = &dom.ao[i];
            let v_i = &dom.vir[i];
            let c = Array2::from_shape_fn((n_i.len(), v_i.len()), |(p, q)| c_vl[(n_i[p], v_i[q])]);
            x.slice(s![.., dom.x_off[i]..dom.x_off[i + 1]]).dot(&c)
        })
        .collect();
    let mut h = Array2::<f64>::zeros((naux, *dom.t_off.last().unwrap()));
    for (i, b) in blocks.into_iter().enumerate() {
        h.slice_mut(s![.., dom.t_off[i]..dom.t_off[i + 1]])
            .assign(&b);
    }
    h
}

/// One quadrature point: `‖Ĥ Wᵀ‖²_F` with `W_i = Σ_{j∈nbrs(i)} Ĥ_j (E_ij Q[V_j, V_i])`.
fn point_energy(h_hat: &Array2<f64>, dom: &LocalDomains, e: &Array2<f64>, q: &Array2<f64>) -> f64 {
    let mut w = Array2::<f64>::zeros(h_hat.dim());
    for (i, vi) in dom.vir.iter().enumerate() {
        let (ci0, ci1) = (dom.t_off[i], dom.t_off[i + 1]);
        for &j in &dom.nbrs[i] {
            let vj = &dom.vir[j];
            let eij = e[(i, j)];
            let g = Array2::from_shape_fn((vj.len(), vi.len()), |(x, y)| eij * q[(vj[x], vi[y])]);
            let hj = h_hat.slice(s![.., dom.t_off[j]..dom.t_off[j + 1]]);
            let mut wi = w.slice_mut(s![.., ci0..ci1]);
            general_mat_mul(1.0, &hj, &g, 1.0, &mut wi);
        }
    }
    let z = h_hat.dot(&w.t());
    z.iter().map(|&x| x * x).sum()
}

/// `exp(scale·ε)`-weighted rotation `Mᵀ diag(e^{scale·ε}) M`.
fn rotated_exp(m: &Array2<f64>, eps: &[f64], scale: f64) -> Array2<f64> {
    let wm = Array2::from_shape_fn(m.dim(), |(k, i)| (scale * eps[k]).exp() * m[(k, i)]);
    m.t().dot(&wm)
}

/// `E_OS` (unscaled) of the `local` formulation at the given radii, on the
/// given Laplace grid. See the module doc for the algebra and the truncations.
#[allow(clippy::too_many_arguments)]
pub fn sos_local_energy(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    frozen_core: usize,
    radii: LocalRadii,
    points: &[f64],
    weights: &[f64],
    memory_budget_bytes: Option<usize>,
) -> Result<(f64, LocalStats), FerricError> {
    let nocc_total = mol.nelec() as usize / 2;
    let naux = dfbs.nbasis();
    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(s![.., frozen_core..nocc_total]).to_owned();
    let c_vir = c.slice(s![.., nocc_total..]).to_owned();
    let eps_o: Vec<f64> = eps[frozen_core..nocc_total].to_vec();
    let eps_v: Vec<f64> = eps[nocc_total..].to_vec();
    let mut st = LocalStats::default();

    let t0 = Instant::now();
    let dip = ferric_integrals::oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let boys = boys_localize(&c_occ, &dip, 200);
    let (vvhv, _) = checked_vvhv(mol, obs, obs_bs, rhf, "laplace-sos-mp2 local")?;
    let c_vl = vvhv.c_vloc;
    let s_ao = ferric_integrals::oneelectron::overlap(obs);
    // L = C_o U and Cvl = C_v R exactly (both spaces are rotated, not
    // truncated), so E(t), Q(t) below are the canonical exponentials rotated,
    // with no dependence on how well the SCF diagonalised F.
    let u = c_occ.t().dot(&s_ao.dot(&boys.c_loc));
    let r = c_vir.t().dot(&s_ao.dot(&c_vl));
    drop(s_ao);
    let mut vir_c = Array2::<f64>::zeros((c_vl.ncols(), 3));
    for (k, dm) in dip.iter().enumerate() {
        for (a, v) in column_expectations(dm, &c_vl).into_iter().enumerate() {
            vir_c[(a, k)] = v;
        }
    }
    let q_sh = ferric_integrals::schwarz::schwarz(op, obs)?;
    let mut dom = build_local_domains(&boys.centers, &vir_c, obs, &q_sh, radii);
    st.n_lmo = c_occ.ncols();
    st.n_lvo = c_vl.ncols();
    st.t_size = *dom.t_off.last().unwrap();
    st.vdom_max = dom.vir.iter().map(Vec::len).max().unwrap_or(0);
    st.vdom_mean = st.t_size as f64 / st.n_lmo.max(1) as f64;
    st.ao_mean = *dom.x_off.last().unwrap() as f64 / st.n_lmo.max(1) as f64;
    st.n_pairs = dom.nbrs.iter().map(Vec::len).sum();
    st.n_shell_pairs = dom.pair_owners.len();
    st.n_shell_pairs_total = obs.nshells() * (obs.nshells() + 1) / 2;
    st.boys_converged = boys.converged;
    st.t_spaces = t0.elapsed().as_secs_f64();

    // Resident estimate: X, then H and Ĥ during whitening, then one W and one
    // Z per concurrently running point, plus the metric and its factor.
    let concurrent = points.len().min(rayon::current_num_threads()).max(1);
    let nx = *dom.x_off.last().unwrap();
    let est = 8
        * (naux * nx.max(2 * st.t_size)
            + concurrent * (naux * st.t_size + naux * naux)
            + 2 * naux * naux);
    let budget = ferric_core::memory::resolve_budget_bytes(memory_budget_bytes);
    if est > budget {
        return Err(FerricError::General(format!(
            "laplace-SOS-MP2 local needs ~{:.2} GB (naux={naux}, |T|={}, {concurrent} \
             concurrent points) but the budget is {:.2} GB. Raise [memory] budget_gb or \
             reduce domain_cutoff_bohr.",
            est as f64 / 1e9,
            st.t_size,
            budget as f64 / 1e9
        )));
    }

    let t1 = Instant::now();
    let x = occ_half_transform(op, obs, dfbs, &boys.c_loc, &dom)?;
    let h = vir_transform(&x, &c_vl, &dom);
    drop(x);
    st.t_eri3 = t1.elapsed().as_secs_f64();

    let t2 = Instant::now();
    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = crate::rimp2::cholesky_inverse_sqrt(&v2c)?;
    drop(v2c);
    let h_hat = v_inv_sqrt.dot(&h);
    drop(h);
    drop(v_inv_sqrt);
    st.t_whiten = t2.elapsed().as_secs_f64();

    st.t_size_geom = st.t_size;
    let h_hat = if radii.energy_tau > 0.0 {
        // Diagonal localized Fock elements from the exact rotations:
        // f_ii = Σ_k U_ki² ε_k, f_aa = Σ_b R_ba² ε_b.
        let f_occ: Vec<f64> = (0..u.ncols())
            .map(|i| (0..u.nrows()).map(|k| u[(k, i)].powi(2) * eps_o[k]).sum())
            .collect();
        let f_vir: Vec<f64> = (0..r.ncols())
            .map(|a| (0..r.nrows()).map(|b| r[(b, a)].powi(2) * eps_v[b]).sum())
            .collect();
        let cut = energy_cut(h_hat, &mut dom, &f_occ, &f_vir, radii.energy_tau, st.n_lvo);
        st.t_size = *dom.t_off.last().unwrap();
        st.vdom_max = dom.vir.iter().map(Vec::len).max().unwrap_or(0);
        st.vdom_mean = st.t_size as f64 / st.n_lmo.max(1) as f64;
        st.n_pairs = dom.nbrs.iter().map(Vec::len).sum();
        cut
    } else {
        h_hat
    };

    let t3 = Instant::now();
    let e_os: f64 = points
        .par_iter()
        .zip(weights.par_iter())
        .map(|(&t, &w)| {
            let e = rotated_exp(&u, &eps_o, t);
            let q = rotated_exp(&r, &eps_v, -t);
            -w * point_energy(&h_hat, &dom, &e, &q)
        })
        .sum();
    st.t_points = t3.elapsed().as_secs_f64();
    Ok((e_os, st))
}
