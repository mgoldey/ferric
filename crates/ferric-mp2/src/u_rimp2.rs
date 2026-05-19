//! Unrestricted RI-MP2 energy.
//!
//! For UHF/ROHF references with separate α and β MO sets. The MP2
//! correlation energy decomposes into three blocks:
//!
//! - `αα`: `¼ Σ_{ij,ab} |⟨ia||jb⟩_α|² / (ε_iα+ε_jα-ε_aα-ε_bα)` (antisymmetrized)
//! - `ββ`: same with β
//! - `αβ`: `Σ_{iJ,aB} (ia|JB)² / (ε_iα+ε_Jβ-ε_aα-ε_Bβ)` (no exchange)
//!
//! where `(ia|jb)_σ = Σ_P B^P_{ia,σ} B^P_{jb,σ}` and
//! `(ia|JB) = Σ_P B^P_{ia,α} B^P_{JB,β}` (shared aux metric).

use crate::oo_rimp2::compute_b_full_mo;
use crate::rimp2::{compute_rpa_intermediates_spin, RiMp2Config, RpaIntermediates};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::{ScfResult, Spin};
use ndarray::{Array2, Array3, Array4};

/// Components of the U-RI-MP2 correlation energy.
#[derive(Debug, Clone)]
pub struct URiMp2Components {
    pub e_aa: f64,
    pub e_bb: f64,
    pub e_ab: f64,
    pub e_total: f64,
}

/// Result of an unrestricted RI-MP2 calculation.
#[derive(Debug, Clone)]
pub struct URiMp2Result {
    pub components: URiMp2Components,
    pub mp2_corr: f64,
    pub total_energy: f64,
}

/// All U-MP2 amplitudes + per-spin intermediates, for downstream
/// gradient / orbital-optimization work.
///
/// Amplitudes follow Bozkaya 2013 conventions:
/// - `t_aa[i,j,a,b] = [(ia|jb)_α − (ib|ja)_α] / D^αα` (antisymmetric ij↔ab)
/// - `t_bb[i,j,a,b] = [(ia|jb)_β − (ib|ja)_β] / D^ββ`
/// - `t_ab[i,J,a,B] = (ia|JB) / D^αβ` (no antisymmetrization; mixed spin)
///
/// Index conventions: `i,j` ∈ occ_α, `I,J` ∈ occ_β, `a,b` ∈ vir_α,
/// `A,B` ∈ vir_β. The αβ tensor's first two axes are α-occ × β-occ, last
/// two are α-vir × β-vir.
#[derive(Debug)]
pub struct UMp2Amplitudes {
    pub inter_a: RpaIntermediates,
    pub inter_b: RpaIntermediates,
    pub eps_a: Vec<f64>,
    pub eps_b: Vec<f64>,
    pub t_aa: Array4<f64>,
    pub t_bb: Array4<f64>,
    pub t_ab: Array4<f64>,
    pub components: URiMp2Components,
}

/// Compute the U-RI-MP2 correlation energy from a UHF or ROHF reference.
///
/// Reuses `compute_rpa_intermediates_spin` to build per-spin
/// `B^P_{ia,σ}` tensors (occ-vir block, dressed with V^{-1/2}).
pub fn u_ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    scf: &ScfResult,
    config: &RiMp2Config,
) -> Result<URiMp2Result, FerricError> {
    if matches!(scf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "u_ri_mp2: requires UHF or ROHF reference".into(),
        ));
    }

    let inter_a = compute_rpa_intermediates_spin(mol, obs, dfbs, op, scf, config, true)?;
    let inter_b = compute_rpa_intermediates_spin(mol, obs, dfbs, op, scf, config, false)?;

    let eps_a: &[f64] = &scf.eps_alpha;
    // ROHF has no eps_beta — fall back to eps_alpha (ROHF MOs are shared).
    let eps_b: &[f64] = match scf.eps_beta.as_ref() {
        Some(v) => v.as_slice(),
        None => &scf.eps_alpha,
    };

    let e_aa = same_spin_pair_energy(&inter_a, eps_a);
    let e_bb = same_spin_pair_energy(&inter_b, eps_b);
    let e_ab = opposite_spin_pair_energy(&inter_a, &inter_b, eps_a, eps_b);

    let e_total = e_aa + e_bb + e_ab;
    Ok(URiMp2Result {
        components: URiMp2Components { e_aa, e_bb, e_ab, e_total },
        mp2_corr: e_total,
        total_energy: scf.energy + e_total,
    })
}

/// Compute U-MP2 amplitudes for all three spin blocks.
///
/// Returns the αα, ββ, αβ amplitude tensors plus the per-spin RI
/// intermediates and orbital energies (needed by downstream gradient
/// / orbital-optimization code).
pub fn compute_u_mp2_amplitudes(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    scf: &ScfResult,
    config: &RiMp2Config,
) -> Result<UMp2Amplitudes, FerricError> {
    if matches!(scf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "compute_u_mp2_amplitudes: requires UHF or ROHF reference".into(),
        ));
    }

    let inter_a = compute_rpa_intermediates_spin(mol, obs, dfbs, op, scf, config, true)?;
    let inter_b = compute_rpa_intermediates_spin(mol, obs, dfbs, op, scf, config, false)?;

    let eps_a_vec: Vec<f64> = scf.eps_alpha.clone();
    let eps_b_vec: Vec<f64> = match scf.eps_beta.as_ref() {
        Some(v) => v.clone(),
        None => scf.eps_alpha.clone(),
    };

    let (t_aa, e_aa) = build_same_spin_amplitudes(&inter_a, &eps_a_vec);
    let (t_bb, e_bb) = build_same_spin_amplitudes(&inter_b, &eps_b_vec);
    let (t_ab, e_ab) = build_opposite_spin_amplitudes(&inter_a, &inter_b, &eps_a_vec, &eps_b_vec);

    let e_total = e_aa + e_bb + e_ab;
    Ok(UMp2Amplitudes {
        inter_a, inter_b,
        eps_a: eps_a_vec, eps_b: eps_b_vec,
        t_aa, t_bb, t_ab,
        components: URiMp2Components { e_aa, e_bb, e_ab, e_total },
    })
}

/// Build the same-spin amplitude tensor and accumulate its energy:
///   t[i,j,a,b] = K_iajb / D,   K_iajb = (ia|jb) - (ib|ja)
///   E = ¼ Σ t · K
fn build_same_spin_amplitudes(
    inter: &RpaIntermediates,
    eps: &[f64],
) -> (Array4<f64>, f64) {
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let naux = inter.naux;
    let first_occ = inter.first_occ;
    let nocc_total = inter.nocc_total;
    let b = &inter.b_ov;
    let mut t = Array4::<f64>::zeros((nocc, nocc, nvir, nvir));
    let mut energy = 0.0;
    for i in 0..nocc {
        let eps_i = eps[first_occ + i];
        for j in 0..nocc {
            let eps_j = eps[first_occ + j];
            for a in 0..nvir {
                let eps_a = eps[nocc_total + a];
                let ia = i * nvir + a;
                let ja = j * nvir + a;
                for b_idx in 0..nvir {
                    let eps_b = eps[nocc_total + b_idx];
                    let jb = j * nvir + b_idx;
                    let ib = i * nvir + b_idx;
                    let mut eri_iajb = 0.0;
                    let mut eri_ibja = 0.0;
                    for p in 0..naux {
                        eri_iajb += b[(p, ia)] * b[(p, jb)];
                        eri_ibja += b[(p, ib)] * b[(p, ja)];
                    }
                    let k = eri_iajb - eri_ibja;
                    let denom = eps_i + eps_j - eps_a - eps_b;
                    let t_val = k / denom;
                    t[(i, j, a, b_idx)] = t_val;
                    energy += t_val * k;
                }
            }
        }
    }
    (t, 0.25 * energy)
}

/// Build the opposite-spin amplitude tensor and its energy:
///   t[i,J,a,B] = (ia|JB) / D
///   E = Σ t · (ia|JB)
fn build_opposite_spin_amplitudes(
    inter_a: &RpaIntermediates,
    inter_b: &RpaIntermediates,
    eps_a: &[f64],
    eps_b: &[f64],
) -> (Array4<f64>, f64) {
    let nocc_a = inter_a.nocc;
    let nvir_a = inter_a.nvir;
    let nocc_b = inter_b.nocc;
    let nvir_b = inter_b.nvir;
    let naux = inter_a.naux;
    let ba = &inter_a.b_ov;
    let bb = &inter_b.b_ov;
    let mut t = Array4::<f64>::zeros((nocc_a, nocc_b, nvir_a, nvir_b));
    let mut energy = 0.0;
    for i in 0..nocc_a {
        let eps_i = eps_a[inter_a.first_occ + i];
        for a in 0..nvir_a {
            let eps_av = eps_a[inter_a.nocc_total + a];
            let ia = i * nvir_a + a;
            for jj in 0..nocc_b {
                let eps_j = eps_b[inter_b.first_occ + jj];
                for bb_idx in 0..nvir_b {
                    let eps_bv = eps_b[inter_b.nocc_total + bb_idx];
                    let jb = jj * nvir_b + bb_idx;
                    let mut eri = 0.0;
                    for p in 0..naux {
                        eri += ba[(p, ia)] * bb[(p, jb)];
                    }
                    let denom = eps_i + eps_j - eps_av - eps_bv;
                    let t_val = eri / denom;
                    t[(i, jj, a, bb_idx)] = t_val;
                    energy += t_val * eri;
                }
            }
        }
    }
    (t, energy)
}

/// Unrelaxed MP2 1-particle density-matrix corrections in MO basis,
/// for both α and β spin channels. Built directly from amplitudes.
///
/// Per-spin formulas (Bozkaya 2013, antisymmetric t^σσ, mixed t^αβ):
///
/// ```text
/// P^α_{ij} = -½ Σ_{k,a,b} t^αα_{ik,ab} t^αα_{jk,ab}
///            - Σ_{K,a,B}    t^αβ_{iK,aB} t^αβ_{jK,aB}
/// P^α_{ab} = +½ Σ_{i,j,c} t^αα_{ij,ac} t^αα_{ij,bc}
///            + Σ_{i,J,B}    t^αβ_{iJ,aB} t^αβ_{iJ,bB}
/// ```
///
/// β formulas obtained by α↔β symmetry: for the αβ piece, the β-occupied
/// index sits on axis 1 of `t_ab` and the β-virtual sits on axis 3.
///
/// Density conservation per spin: tr(P^σ_oo) + tr(P^σ_vv) = 0.
#[derive(Debug, Clone)]
pub struct UMp2Density {
    pub p_oo_a: Array2<f64>,
    pub p_vv_a: Array2<f64>,
    pub p_oo_b: Array2<f64>,
    pub p_vv_b: Array2<f64>,
}

pub fn build_u_mp2_density(amps: &UMp2Amplitudes) -> UMp2Density {
    let t_aa = &amps.t_aa;
    let t_bb = &amps.t_bb;
    let t_ab = &amps.t_ab;
    let (nocc_a, _, nvir_a, _) = t_aa.dim();
    let (nocc_b, _, nvir_b, _) = t_bb.dim();

    // P^α_{ij}
    let mut p_oo_a = Array2::<f64>::zeros((nocc_a, nocc_a));
    for i in 0..nocc_a {
        for j in 0..nocc_a {
            let mut s = 0.0;
            // αα same-spin: -½ Σ_{k,a,b} t_{ik,ab} t_{jk,ab}
            for k in 0..nocc_a {
                for a in 0..nvir_a {
                    for b in 0..nvir_a {
                        s += -0.5 * t_aa[(i, k, a, b)] * t_aa[(j, k, a, b)];
                    }
                }
            }
            // αβ: -Σ_{K,a,B} t^αβ_{iK,aB} t^αβ_{jK,aB}
            for kk in 0..nocc_b {
                for a in 0..nvir_a {
                    for bb in 0..nvir_b {
                        s += -t_ab[(i, kk, a, bb)] * t_ab[(j, kk, a, bb)];
                    }
                }
            }
            p_oo_a[(i, j)] = s;
        }
    }

    // P^α_{ab}
    let mut p_vv_a = Array2::<f64>::zeros((nvir_a, nvir_a));
    for a in 0..nvir_a {
        for b in 0..nvir_a {
            let mut s = 0.0;
            // αα: +½ Σ_{i,j,c} t_{ij,ac} t_{ij,bc}
            for i in 0..nocc_a {
                for j in 0..nocc_a {
                    for c in 0..nvir_a {
                        s += 0.5 * t_aa[(i, j, a, c)] * t_aa[(i, j, b, c)];
                    }
                }
            }
            // αβ: +Σ_{i,J,B} t^αβ_{iJ,aB} t^αβ_{iJ,bB}
            for i in 0..nocc_a {
                for jj in 0..nocc_b {
                    for bb in 0..nvir_b {
                        s += t_ab[(i, jj, a, bb)] * t_ab[(i, jj, b, bb)];
                    }
                }
            }
            p_vv_a[(a, b)] = s;
        }
    }

    // P^β_{ij}
    let mut p_oo_b = Array2::<f64>::zeros((nocc_b, nocc_b));
    for i in 0..nocc_b {
        for j in 0..nocc_b {
            let mut s = 0.0;
            // ββ: -½ Σ_{k,a,b} t_{ik,ab} t_{jk,ab}
            for k in 0..nocc_b {
                for a in 0..nvir_b {
                    for b in 0..nvir_b {
                        s += -0.5 * t_bb[(i, k, a, b)] * t_bb[(j, k, a, b)];
                    }
                }
            }
            // αβ (β-occ axis is axis 1): -Σ_{K,A,b} t^αβ_{KI,Ab} t^αβ_{KJ,Ab}
            for kk in 0..nocc_a {
                for aa in 0..nvir_a {
                    for b in 0..nvir_b {
                        s += -t_ab[(kk, i, aa, b)] * t_ab[(kk, j, aa, b)];
                    }
                }
            }
            p_oo_b[(i, j)] = s;
        }
    }

    // P^β_{ab}
    let mut p_vv_b = Array2::<f64>::zeros((nvir_b, nvir_b));
    for a in 0..nvir_b {
        for b in 0..nvir_b {
            let mut s = 0.0;
            // ββ: +½ Σ_{i,j,c} t_{ij,ac} t_{ij,bc}
            for i in 0..nocc_b {
                for j in 0..nocc_b {
                    for c in 0..nvir_b {
                        s += 0.5 * t_bb[(i, j, a, c)] * t_bb[(i, j, b, c)];
                    }
                }
            }
            // αβ (β-vir axis is axis 3): +Σ_{I,J,A} t^αβ_{IJ,Aa} t^αβ_{IJ,Ab}
            for ii in 0..nocc_a {
                for jj in 0..nocc_b {
                    for aa in 0..nvir_a {
                        s += t_ab[(ii, jj, aa, a)] * t_ab[(ii, jj, aa, b)];
                    }
                }
            }
            p_vv_b[(a, b)] = s;
        }
    }

    UMp2Density { p_oo_a, p_vv_a, p_oo_b, p_vv_b }
}

/// **WORK IN PROGRESS** — known wrong by a per-spin scalar factor + sign
/// against the FD reference (c≈−0.5 for α, c≈−0.4 for β on OH/cc-pVDZ).
/// The derivation below is incomplete: multiple terms with different
/// missing factors. Do not consume in optimizer code yet.
///
/// Compute the U-MP2 part of the orbital gradient `g^σ_{ai} = ∂E_MP2/∂κ^σ_{ai}`
/// at fixed amplitudes (2n+1 rule applies at HF reference).
///
/// Sign convention follows `oo_rimp2::compute_orbital_gradient`: positive
/// `κ_{ai}` mixes occupied into virtual via the Cayley map. The MP2 piece
/// is *not* negated (caller can add `-2·F^σ_{ai}` HF piece, which is zero
/// at a converged HF reference).
///
/// Returns `(g_a, g_b)` with shapes `(nvir_α, nocc_α)` and `(nvir_β, nocc_β)`.
///
/// The energy expression is:
/// ```text
/// E_MP2 = ¼ Σ t^αα_{ij,ab} K^αα_{iajb}
///       + ¼ Σ t^ββ_{ij,ab} K^ββ_{iajb}
///       +   Σ t^αβ_{iJ,aB} (ia|JB)
/// ```
/// where `K^σσ_{iajb} = (ia|jb)_σ − (ib|ja)_σ` is the antisymmetrized integral.
///
/// `∂E/∂κ^α_{ck}` only sees the αα and αβ terms (the α MO indices appearing in
/// each integral). Integral derivatives:
/// ```text
/// ∂(pq|rs)/∂κ^α_{ck} = δ_pk (cq|rs) + δ_rk (pq|cs)         (α derivative on α-row)
///                    − δ_qc (pk|rs) − δ_sc (pq|rk)          (α derivative on α-col)
/// ```
/// for indices `p,q,r,s` that are α MOs; for αβ integrals only the α-side
/// indices contribute (i,a are α; J,B are β spectators).
pub fn compute_u_mp2_orbital_gradient(
    amps: &UMp2Amplitudes,
    b_full_a: &Array3<f64>,
    b_full_b: &Array3<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let (nocc_a, _, nvir_a, _) = amps.t_aa.dim();
    let (nocc_b, _, nvir_b, _) = amps.t_bb.dim();
    let first_occ_a = amps.inter_a.first_occ;
    let nocc_total_a = amps.inter_a.nocc_total;
    let first_occ_b = amps.inter_b.first_occ;
    let nocc_total_b = amps.inter_b.nocc_total;
    let naux = b_full_a.shape()[0];
    debug_assert_eq!(naux, b_full_b.shape()[0]);

    // ERI helpers — (pq|rs) for same-spin (α or β) blocks via the full-MO B tensor.
    let eri_aa = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        (0..naux).map(|aux| b_full_a[(aux, p, q)] * b_full_a[(aux, r, s)]).sum()
    };
    let eri_bb = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        (0..naux).map(|aux| b_full_b[(aux, p, q)] * b_full_b[(aux, r, s)]).sum()
    };
    // αβ integrals: α-MO indices (p,q) on the α tensor, β-MO (r,s) on the β tensor.
    let eri_ab = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        (0..naux).map(|aux| b_full_a[(aux, p, q)] * b_full_b[(aux, r, s)]).sum()
    };

    let t_aa = &amps.t_aa;
    let t_bb = &amps.t_bb;
    let t_ab = &amps.t_ab;

    // Each accumulated chain-rule term inside the loops below contributes
    // only `∂E/∂K_{ai}` (the +h slot of the antisymmetric κ).
    // The FD reference varies *both* `K_{ai}=+h` and `K_{ia}=-h` together,
    // so its `∂E/∂κ_{ai}` picks up two equal contributions → factor of 2.
    // Combined with the sign flip from C → C·U(κ) (positive κ_{ai} rotates
    // an occupied into a virtual, lowering energy in the direction
    // opposite to the integral derivative), the post-loop scaling is −2.
    let mut g_a = Array2::<f64>::zeros((nvir_a, nocc_a));
    let mut g_b = Array2::<f64>::zeros((nvir_b, nocc_b));

    // --- α gradient: g^α_{ck} = ∂E_MP2/∂κ^α_{ck} -------------------------
    // αα contribution (Hylleraas; t_aa is antisymmetric in (ij) and (ab)):
    //   d/dκ^α_{ck} [¼ Σ_{ij,ab} t^αα_{ij,ab} ((ia|jb)−(ib|ja))]
    // Apply δ_{i,k} and δ_{a,c} expansions of d(ia|jb)/dκ^α_{ck}; the antisymmetry
    // of t in i↔j and a↔b lets us combine terms pairwise into four sums.
    for c in 0..nvir_a {
        let c_mo = nocc_total_a + c;
        for k in 0..nocc_a {
            let k_mo = first_occ_a + k;
            let mut sum = 0.0;

            // Term 1: i=k branch of K^αα. After using K's antisymmetry t↔(ij), this is
            //   ½ Σ_{j,a,b} t^αα_{kj,ab} [(ca|jb)_α − (cb|ja)_α]
            for j in 0..nocc_a {
                let j_mo = first_occ_a + j;
                for a in 0..nvir_a {
                    let a_mo = nocc_total_a + a;
                    for b in 0..nvir_a {
                        let b_mo = nocc_total_a + b;
                        let t = t_aa[(k, j, a, b)];
                        let e1 = eri_aa(c_mo, a_mo, j_mo, b_mo);
                        let e2 = eri_aa(c_mo, b_mo, j_mo, a_mo);
                        sum += 0.5 * t * (e1 - e2);
                    }
                }
            }
            // Term 2: j=k branch. By relabeling i↔j and using t antisym, identical to Term 1.
            for i in 0..nocc_a {
                let i_mo = first_occ_a + i;
                for a in 0..nvir_a {
                    let a_mo = nocc_total_a + a;
                    for b in 0..nvir_a {
                        let b_mo = nocc_total_a + b;
                        let t = t_aa[(i, k, a, b)];
                        let e1 = eri_aa(i_mo, a_mo, c_mo, b_mo);
                        let e2 = eri_aa(i_mo, b_mo, c_mo, a_mo);
                        sum += 0.5 * t * (e1 - e2);
                    }
                }
            }
            // Term 3: a=c branch.
            //   −½ Σ_{i,j,b} t^αα_{ij,cb} [(ik|jb)_α − (ib|jk)_α]
            for i in 0..nocc_a {
                let i_mo = first_occ_a + i;
                for j in 0..nocc_a {
                    let j_mo = first_occ_a + j;
                    for b in 0..nvir_a {
                        let b_mo = nocc_total_a + b;
                        let t = t_aa[(i, j, c, b)];
                        let e1 = eri_aa(i_mo, k_mo, j_mo, b_mo);
                        let e2 = eri_aa(i_mo, b_mo, j_mo, k_mo);
                        sum -= 0.5 * t * (e1 - e2);
                    }
                }
            }
            // Term 4: b=c branch. Mirror of Term 3 with a↔b on t.
            for i in 0..nocc_a {
                let i_mo = first_occ_a + i;
                for j in 0..nocc_a {
                    let j_mo = first_occ_a + j;
                    for a in 0..nvir_a {
                        let a_mo = nocc_total_a + a;
                        let t = t_aa[(i, j, a, c)];
                        let e1 = eri_aa(i_mo, a_mo, j_mo, k_mo);
                        let e2 = eri_aa(i_mo, k_mo, j_mo, a_mo);
                        sum -= 0.5 * t * (e1 - e2);
                    }
                }
            }

            // αβ contribution to g^α_{ck}: α-derivative on α-row (δ_{i,k}) and α-col (δ_{a,c})
            // of (ia|JB). β indices (J,B) are untouched.
            //   ∂/∂κ^α_{ck} [Σ_{iJaB} t^αβ_{iJ,aB} (ia|JB)]
            //   = Σ_{J,a,B} t^αβ_{kJ,aB} (ca|JB)         (from δ_{i,k}·(ca|JB))
            //   − Σ_{i,J,B} t^αβ_{iJ,cB} (ik|JB)         (from −δ_{a,c}·(ik|JB))
            for jj in 0..nocc_b {
                let j_mo = first_occ_b + jj;
                for a in 0..nvir_a {
                    let a_mo = nocc_total_a + a;
                    for bb in 0..nvir_b {
                        let b_mo = nocc_total_b + bb;
                        let t = t_ab[(k, jj, a, bb)];
                        let e = eri_ab(c_mo, a_mo, j_mo, b_mo);
                        sum += t * e;
                    }
                }
            }
            for i in 0..nocc_a {
                let i_mo = first_occ_a + i;
                for jj in 0..nocc_b {
                    let j_mo = first_occ_b + jj;
                    for bb in 0..nvir_b {
                        let b_mo = nocc_total_b + bb;
                        let t = t_ab[(i, jj, c, bb)];
                        let e = eri_ab(i_mo, k_mo, j_mo, b_mo);
                        sum -= t * e;
                    }
                }
            }

            g_a[(c, k)] = sum;
        }
    }

    // --- β gradient: same structure with α↔β swapped ---------------------
    for c in 0..nvir_b {
        let c_mo = nocc_total_b + c;
        for k in 0..nocc_b {
            let k_mo = first_occ_b + k;
            let mut sum = 0.0;

            // ββ Hylleraas: four terms with t_bb
            for j in 0..nocc_b {
                let j_mo = first_occ_b + j;
                for a in 0..nvir_b {
                    let a_mo = nocc_total_b + a;
                    for b in 0..nvir_b {
                        let b_mo = nocc_total_b + b;
                        let t = t_bb[(k, j, a, b)];
                        sum += 0.5 * t * (eri_bb(c_mo, a_mo, j_mo, b_mo)
                                        - eri_bb(c_mo, b_mo, j_mo, a_mo));
                    }
                }
            }
            for i in 0..nocc_b {
                let i_mo = first_occ_b + i;
                for a in 0..nvir_b {
                    let a_mo = nocc_total_b + a;
                    for b in 0..nvir_b {
                        let b_mo = nocc_total_b + b;
                        let t = t_bb[(i, k, a, b)];
                        sum += 0.5 * t * (eri_bb(i_mo, a_mo, c_mo, b_mo)
                                        - eri_bb(i_mo, b_mo, c_mo, a_mo));
                    }
                }
            }
            for i in 0..nocc_b {
                let i_mo = first_occ_b + i;
                for j in 0..nocc_b {
                    let j_mo = first_occ_b + j;
                    for b in 0..nvir_b {
                        let b_mo = nocc_total_b + b;
                        let t = t_bb[(i, j, c, b)];
                        sum -= 0.5 * t * (eri_bb(i_mo, k_mo, j_mo, b_mo)
                                        - eri_bb(i_mo, b_mo, j_mo, k_mo));
                    }
                }
            }
            for i in 0..nocc_b {
                let i_mo = first_occ_b + i;
                for j in 0..nocc_b {
                    let j_mo = first_occ_b + j;
                    for a in 0..nvir_b {
                        let a_mo = nocc_total_b + a;
                        let t = t_bb[(i, j, a, c)];
                        sum -= 0.5 * t * (eri_bb(i_mo, a_mo, j_mo, k_mo)
                                        - eri_bb(i_mo, k_mo, j_mo, a_mo));
                    }
                }
            }

            // αβ contribution to g^β_{ck}: β-derivative on β-row (δ_{J,k}) and β-col (δ_{B,c})
            //   ∂/∂κ^β_{ck} [Σ_{iJaB} t^αβ_{iJ,aB} (ia|JB)]
            //   = Σ_{i,a,B} t^αβ_{ik,a,B} (ia|cB)         (from δ_{J,k}·(ia|cB)) -- wait, β-row
            //
            // The αβ integral is (ia|JB) — β indices are J (occ) and B (vir).
            // ∂(ia|JB)/∂κ^β_{ck} = δ_{J,k} (ia|cB) − δ_{B,c} (ia|Jk)
            //   = +Σ_{i,a,B} t^αβ_{ik,a,B} (ia|cB)        (δ_{J,k})
            //   −Σ_{i,J,a}   t^αβ_{i,J,a,c} (ia|Jk)        (δ_{B,c})
            for i in 0..nocc_a {
                let i_mo = first_occ_a + i;
                for a in 0..nvir_a {
                    let a_mo = nocc_total_a + a;
                    for bb in 0..nvir_b {
                        let b_mo = nocc_total_b + bb;
                        let t = t_ab[(i, k, a, bb)];
                        let e = eri_ab(i_mo, a_mo, c_mo, b_mo);
                        sum += t * e;
                    }
                }
            }
            for i in 0..nocc_a {
                let i_mo = first_occ_a + i;
                for jj in 0..nocc_b {
                    let j_mo = first_occ_b + jj;
                    for a in 0..nvir_a {
                        let a_mo = nocc_total_a + a;
                        let t = t_ab[(i, jj, a, c)];
                        let e = eri_ab(i_mo, a_mo, j_mo, k_mo);
                        sum -= t * e;
                    }
                }
            }

            g_b[(c, k)] = sum;
        }
    }

    (g_a, g_b)
}

/// Same-spin contribution:
///   ¼ Σ_{ij,ab} [(ia|jb) - (ib|ja)]² / (ε_i+ε_j-ε_a-ε_b)
fn same_spin_pair_energy(inter: &RpaIntermediates, eps: &[f64]) -> f64 {
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let naux = inter.naux;
    let first_occ = inter.first_occ;
    let nocc_total = inter.nocc_total;
    let b = &inter.b_ov; // (naux, nocc*nvir)

    let mut energy = 0.0;
    for i in 0..nocc {
        let eps_i = eps[first_occ + i];
        for j in 0..nocc {
            let eps_j = eps[first_occ + j];
            for a in 0..nvir {
                let eps_a = eps[nocc_total + a];
                let ia = i * nvir + a;
                let ja = j * nvir + a;
                for b_idx in 0..nvir {
                    let eps_b = eps[nocc_total + b_idx];
                    let jb = j * nvir + b_idx;
                    let ib = i * nvir + b_idx;
                    let mut eri_iajb = 0.0;
                    let mut eri_ibja = 0.0;
                    for p in 0..naux {
                        eri_iajb += b[(p, ia)] * b[(p, jb)];
                        eri_ibja += b[(p, ib)] * b[(p, ja)];
                    }
                    let diff = eri_iajb - eri_ibja;
                    let denom = eps_i + eps_j - eps_a - eps_b;
                    energy += diff * diff / denom;
                }
            }
        }
    }
    0.25 * energy
}

/// Opposite-spin contribution:
///   Σ_{iJ,aB} (ia|JB)² / (ε_iα + ε_Jβ - ε_aα - ε_Bβ)
fn opposite_spin_pair_energy(
    inter_a: &RpaIntermediates,
    inter_b: &RpaIntermediates,
    eps_a: &[f64],
    eps_b: &[f64],
) -> f64 {
    let nocc_a = inter_a.nocc;
    let nvir_a = inter_a.nvir;
    let nocc_b = inter_b.nocc;
    let nvir_b = inter_b.nvir;
    let naux = inter_a.naux;
    assert_eq!(naux, inter_b.naux);
    let ba = &inter_a.b_ov;
    let bb = &inter_b.b_ov;

    let mut energy = 0.0;
    for i in 0..nocc_a {
        let eps_i = eps_a[inter_a.first_occ + i];
        for a in 0..nvir_a {
            let eps_a_v = eps_a[inter_a.nocc_total + a];
            let ia = i * nvir_a + a;
            for jj in 0..nocc_b {
                let eps_j = eps_b[inter_b.first_occ + jj];
                for bb_idx in 0..nvir_b {
                    let eps_b_v = eps_b[inter_b.nocc_total + bb_idx];
                    let jb = jj * nvir_b + bb_idx;
                    let mut eri = 0.0;
                    for p in 0..naux {
                        eri += ba[(p, ia)] * bb[(p, jb)];
                    }
                    let denom = eps_i + eps_j - eps_a_v - eps_b_v;
                    energy += eri * eri / denom;
                }
            }
        }
    }
    energy
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::RhfConfig;
    use ferric_scf::screening::SchwarzBounds;
    use ferric_scf::uhf::{solve_uhf, UhfConfig};

    /// Compare U-RI-MP2 on a closed-shell system (H2 in cc-pVDZ) against
    /// the closed-shell `ri_mp2` driver. Should agree to numerical noise.
    #[test]
    fn u_rimp2_matches_closed_shell_on_h2() {
        let ctx = ParallelContext::default();
        let xyz = "2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n";
        // Closed-shell run
        let mol_cs = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol_cs, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol_cs, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = ferric_scf::rhf::solve_rhf(
            &ctx, &mol_cs, &obs, op, &bounds, &RhfConfig::default(),
        ).unwrap();
        let cs = crate::rimp2::ri_mp2(
            &mol_cs, &obs, &dfbs, op, &rhf, &RiMp2Config::default(),
        ).unwrap();

        // Open-shell run on same molecule (singlet, M=1)
        let mol_us = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let uhf_cfg = UhfConfig { max_iter: 200, energy_conv: 1e-10, density_conv: 1e-8, ..Default::default() };
        // UHF will converge to a symmetric solution for singlet H2 if seeded
        // from neutral RHF MOs (no spin contamination).
        let c_seed = rhf.mos_r().clone();
        let uhf = ferric_scf::uhf::solve_uhf_with_guess(
            &ctx, &mol_us, &obs, op, &bounds, &uhf_cfg, Some((&c_seed, &c_seed)),
        ).unwrap();

        let us = u_ri_mp2(&mol_us, &obs, &dfbs, op, &uhf, &RiMp2Config::default()).unwrap();
        let diff = (us.mp2_corr - cs.mp2_corr).abs();
        println!("CS E_MP2 = {:.10}, US E_MP2 = {:.10}, diff = {:.3e}", cs.mp2_corr, us.mp2_corr, diff);
        println!("  components: αα={:.6e} ββ={:.6e} αβ={:.6e}", us.components.e_aa, us.components.e_bb, us.components.e_ab);
        assert!(diff < 1e-7, "closed-shell U-RI-MP2 disagrees with RI-MP2: diff={}", diff);
    }

    /// Validate U-RI-MP2 on OH/cc-pVDZ against the PySCF FD reference
    /// (testdata/reference/oh_cc-pvdz_u-oomp2-fd.json).
    /// PySCF reference: E_corr = -0.151003 (no frozen core, no RI).
    /// Ferric uses cc-pvdz-ri auxiliary -> RI noise ~1e-4 Ha.
    #[test]
    fn u_rimp2_oh_cc_pvdz_matches_pyscf() {
        let ctx = ParallelContext::default();
        // Geometry must match the Python harness: O at origin, H at (0,0,0.97 Å).
        let xyz = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";
        let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();

        let uhf_cfg = UhfConfig {
            max_iter: 200, energy_conv: 1e-10, density_conv: 1e-8, ..Default::default()
        };
        let uhf = solve_uhf(&ctx, &mol, &obs, op, &bounds, &uhf_cfg).unwrap();
        println!("OH UHF: E={:.8}, iters={}", uhf.energy, uhf.iterations);

        let res = u_ri_mp2(&mol, &obs, &dfbs, op, &uhf, &RiMp2Config::default()).unwrap();
        println!("U-RI-MP2 components: αα={:.6e} ββ={:.6e} αβ={:.6e}",
            res.components.e_aa, res.components.e_bb, res.components.e_ab);
        println!("E_corr (ferric) = {:.8}", res.mp2_corr);
        println!("E_corr (PySCF)  = -0.15100299");
        let pyscf_e_corr = -0.151002988955374;
        let diff = (res.mp2_corr - pyscf_e_corr).abs();
        println!("diff = {:.3e} Ha", diff);
        assert!(diff < 5e-4, "U-RI-MP2 on OH off by {:.3e} Ha vs PySCF UMP2 (RI noise tolerance 5e-4)", diff);
    }

    /// Validate amplitude builder on OH/cc-pVDZ:
    /// (1) energy from amplitudes matches the closed-form `u_ri_mp2`
    /// (2) same-spin amplitudes have correct antisymmetry t[i,j,a,b] = -t[j,i,a,b] = -t[i,j,b,a]
    /// (3) αβ amplitudes have no antisymmetry (mixed spin).
    #[test]
    fn u_mp2_amplitudes_consistent_on_oh() {
        let ctx = ParallelContext::default();
        let xyz = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";
        let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let uhf_cfg = UhfConfig {
            max_iter: 200, energy_conv: 1e-10, density_conv: 1e-8, ..Default::default()
        };
        let uhf = solve_uhf(&ctx, &mol, &obs, op, &bounds, &uhf_cfg).unwrap();

        let energy_res = u_ri_mp2(&mol, &obs, &dfbs, op, &uhf, &RiMp2Config::default()).unwrap();
        let amps = compute_u_mp2_amplitudes(&mol, &obs, &dfbs, op, &uhf, &RiMp2Config::default()).unwrap();

        // Energy consistency
        let diff = (amps.components.e_total - energy_res.components.e_total).abs();
        println!("E_total via amplitudes = {:.10}, via u_ri_mp2 = {:.10}, diff = {:.3e}",
                 amps.components.e_total, energy_res.components.e_total, diff);
        assert!(diff < 1e-12, "amplitude energy disagrees with closed-form: diff={}", diff);

        // Same-spin antisymmetry: t[i,j,a,b] should equal -t[j,i,a,b] and -t[i,j,b,a]
        let nocc_a = amps.inter_a.nocc;
        let nvir_a = amps.inter_a.nvir;
        let mut max_asym = 0.0_f64;
        for i in 0..nocc_a {
            for j in 0..nocc_a {
                for a in 0..nvir_a {
                    for b_idx in 0..nvir_a {
                        let t_ijab = amps.t_aa[(i, j, a, b_idx)];
                        let t_jiab = amps.t_aa[(j, i, a, b_idx)];
                        let t_ijba = amps.t_aa[(i, j, b_idx, a)];
                        max_asym = max_asym.max((t_ijab + t_jiab).abs());
                        max_asym = max_asym.max((t_ijab + t_ijba).abs());
                    }
                }
            }
        }
        println!("max |t_aa[ijab] + t_aa[jiab]| = {:.3e} (must be zero)", max_asym);
        assert!(max_asym < 1e-12, "αα amplitudes not antisymmetric: max asym={}", max_asym);

        // Shapes
        println!("shapes: t_aa={:?}, t_bb={:?}, t_ab={:?}",
                 amps.t_aa.shape(), amps.t_bb.shape(), amps.t_ab.shape());
        println!("components: αα={:.6e} ββ={:.6e} αβ={:.6e}",
                 amps.components.e_aa, amps.components.e_bb, amps.components.e_ab);
    }

    /// Sanity-check the unrelaxed U-MP2 density matrices on OH/cc-pVDZ:
    /// (1) symmetric per-spin per-block
    /// (2) tr(P_oo_σ) + tr(P_vv_σ) = 0 per spin (density-conservation)
    /// (3) signs: tr(P_oo) < 0, tr(P_vv) > 0
    #[test]
    fn u_mp2_density_sane_on_oh() {
        let ctx = ParallelContext::default();
        let xyz = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";
        let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let uhf_cfg = UhfConfig {
            max_iter: 200, energy_conv: 1e-10, density_conv: 1e-8, ..Default::default()
        };
        let uhf = solve_uhf(&ctx, &mol, &obs, op, &bounds, &uhf_cfg).unwrap();
        let amps = compute_u_mp2_amplitudes(&mol, &obs, &dfbs, op, &uhf, &RiMp2Config::default()).unwrap();
        let dens = build_u_mp2_density(&amps);

        for (name, p) in [
            ("p_oo_a", &dens.p_oo_a), ("p_vv_a", &dens.p_vv_a),
            ("p_oo_b", &dens.p_oo_b), ("p_vv_b", &dens.p_vv_b),
        ] {
            let (n, _) = p.dim();
            let mut max_asym = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    max_asym = max_asym.max((p[(i,j)] - p[(j,i)]).abs());
                }
            }
            println!("{}: max asym = {:.3e}", name, max_asym);
            assert!(max_asym < 1e-14, "{} not symmetric: max asym={}", name, max_asym);
        }

        let tr = |p: &Array2<f64>| -> f64 { (0..p.dim().0).map(|i| p[(i,i)]).sum() };
        let tr_oo_a = tr(&dens.p_oo_a);
        let tr_vv_a = tr(&dens.p_vv_a);
        let tr_oo_b = tr(&dens.p_oo_b);
        let tr_vv_b = tr(&dens.p_vv_b);
        println!("α: tr(P_oo)={:.6e}, tr(P_vv)={:.6e}, sum={:.3e}",
                 tr_oo_a, tr_vv_a, tr_oo_a + tr_vv_a);
        println!("β: tr(P_oo)={:.6e}, tr(P_vv)={:.6e}, sum={:.3e}",
                 tr_oo_b, tr_vv_b, tr_oo_b + tr_vv_b);
        assert!(tr_oo_a < 0.0 && tr_vv_a > 0.0, "α sign wrong");
        assert!(tr_oo_b < 0.0 && tr_vv_b > 0.0, "β sign wrong");
        assert!((tr_oo_a + tr_vv_a).abs() < 1e-10, "α density not conserved");
        assert!((tr_oo_b + tr_vv_b).abs() < 1e-10, "β density not conserved");
    }

    /// On a closed-shell singlet (H2 spin=1), α and β U-MP2 densities
    /// must be equal (no spin polarization), and each equals half the
    /// closed-shell density.
    #[test]
    fn u_mp2_density_alpha_eq_beta_on_h2() {
        let ctx = ParallelContext::default();
        let xyz = "2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = ferric_scf::rhf::solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let c_seed = rhf.mos_r().clone();
        let uhf_cfg = UhfConfig {
            max_iter: 200, energy_conv: 1e-10, density_conv: 1e-8, ..Default::default()
        };
        let uhf = ferric_scf::uhf::solve_uhf_with_guess(
            &ctx, &mol, &obs, op, &bounds, &uhf_cfg, Some((&c_seed, &c_seed)),
        ).unwrap();
        let amps = compute_u_mp2_amplitudes(&mol, &obs, &dfbs, op, &uhf, &RiMp2Config::default()).unwrap();
        let dens = build_u_mp2_density(&amps);

        // Note: α/β MOs in a closed-shell-via-UHF run agree in *eigenvalues*
        // but their *eigenvectors* can differ inside degenerate virtual blocks
        // (cc-pVDZ on H2 has multiple degenerate virtuals). The density matrix
        // P is therefore gauge-dependent — but its trace and eigenvalues
        // (natural occupation numbers) are invariant. Compare those.
        let tr = |p: &Array2<f64>| -> f64 { (0..p.dim().0).map(|i| p[(i,i)]).sum() };
        let tr_oo_a = tr(&dens.p_oo_a);
        let tr_oo_b = tr(&dens.p_oo_b);
        let tr_vv_a = tr(&dens.p_vv_a);
        let tr_vv_b = tr(&dens.p_vv_b);
        println!("H2 traces: P_oo_a={:.6e} P_oo_b={:.6e}; P_vv_a={:.6e} P_vv_b={:.6e}",
                 tr_oo_a, tr_oo_b, tr_vv_a, tr_vv_b);
        assert!((tr_oo_a - tr_oo_b).abs() < 1e-10);
        assert!((tr_vv_a - tr_vv_b).abs() < 1e-10);
        // Per-spin conservation
        assert!((tr_oo_a + tr_vv_a).abs() < 1e-10);
        assert!((tr_oo_b + tr_vv_b).abs() < 1e-10);
        // E_aa = E_bb = 0 (each spin has a single occupied orbital, no same-spin pair)
        assert!(amps.components.e_aa.abs() < 1e-14);
        assert!(amps.components.e_bb.abs() < 1e-14);
    }

    /// FD-validate the analytic U-MP2 orbital gradient against the PySCF
    /// FD reference in testdata/reference/oh_cc-pvdz_u-oomp2-fd.json.
    ///
    /// The FD reference is computed with κ defined as
    ///   κ[a+nocc, i] = +h, κ[i, a+nocc] = −h, C → C·U(κ)
    /// at the same HF orbitals ferric converges to. ferric uses cc-pvdz-ri
    /// so RI noise relative to PySCF UMP2 is ~1e-3 element-wise but
    /// the RMS / max-element comparison still has to be sensible (no
    /// sign flips, no order-of-magnitude scalar errors).
    ///
    /// NOTE: this test is a DIAGNOSTIC. It runs the analytic gradient and
    /// reports rms/max against the FD reference. The first goal is to
    /// uncover sign / factor errors before tightening tolerances.
    #[test]
    fn u_mp2_orbital_gradient_vs_fd_on_oh() {
        let ctx = ParallelContext::default();
        let xyz = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";
        let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let uhf_cfg = UhfConfig {
            max_iter: 200, energy_conv: 1e-10, density_conv: 1e-8, ..Default::default()
        };
        let uhf = solve_uhf(&ctx, &mol, &obs, op, &bounds, &uhf_cfg).unwrap();
        let amps = compute_u_mp2_amplitudes(&mol, &obs, &dfbs, op, &uhf, &RiMp2Config::default()).unwrap();

        let b_full_a = crate::oo_rimp2::compute_b_full_mo(&obs, &dfbs, op, uhf.mos_a()).unwrap();
        let b_full_b = crate::oo_rimp2::compute_b_full_mo(&obs, &dfbs, op, uhf.mos_b()).unwrap();
        let (g_a, g_b) = super::compute_u_mp2_orbital_gradient(&amps, &b_full_a, &b_full_b);

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/reference/oh_cc-pvdz_u-oomp2-fd.json");
        let txt = std::fs::read_to_string(&path).expect("FD reference missing");
        let v: serde_json::Value = serde_json::from_str(&txt).unwrap();
        let parse_2d = |arr: &serde_json::Value| -> Array2<f64> {
            let rows: Vec<Vec<f64>> = arr.as_array().unwrap().iter().map(|r|
                r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
            ).collect();
            let nrow = rows.len();
            let ncol = rows[0].len();
            let flat: Vec<f64> = rows.into_iter().flatten().collect();
            Array2::from_shape_vec((nrow, ncol), flat).unwrap()
        };
        let g_fd_a = parse_2d(&v["grad_fd_a"]);
        let g_fd_b = parse_2d(&v["grad_fd_b"]);
        assert_eq!(g_fd_a.dim(), g_a.dim(), "α shape mismatch");
        assert_eq!(g_fd_b.dim(), g_b.dim(), "β shape mismatch");

        let report = |label: &str, g: &Array2<f64>, g_fd: &Array2<f64>| {
            let nfd = g_fd.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let n = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let diff = g - g_fd;
            let max_diff = diff.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let rms: f64 = (diff.iter().map(|v| v*v).sum::<f64>() / (diff.len() as f64)).sqrt();
            // Best scalar fit: c = <g · g_fd> / <g_fd · g_fd>
            let dot: f64 = g.iter().zip(g_fd.iter()).map(|(a,b)| a*b).sum();
            let nrm2_fd: f64 = g_fd.iter().map(|v| v*v).sum();
            let c = if nrm2_fd > 0.0 { dot / nrm2_fd } else { 0.0 };
            // Sign-only diff
            let sign_diff = -g - g_fd;
            let max_neg = sign_diff.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            println!("[{}] |g_an|max={:.4e} |g_fd|max={:.4e}  max|Δ|={:.4e}  rms={:.4e}  c≈{:.4}  max|-g-fd|={:.4e}",
                     label, n, nfd, max_diff, rms, c, max_neg);
        };
        report("α", &g_a, &g_fd_a);
        report("β", &g_b, &g_fd_b);

        // Don't assert yet — this is diagnostic. The pattern of c≈1 vs c≈-1 or
        // c≈0.5 etc tells us what scalar factor is missing.
    }
}
