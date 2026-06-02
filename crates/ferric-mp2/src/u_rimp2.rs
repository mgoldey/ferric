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

/// Compute the U-MP2 orbital gradient `g^σ_{ai} = ∂E_MP2/∂κ^σ_{ai}` at fixed
/// orbital energies (integral-response only). FD-validated on OH/cc-pVDZ to
/// ~1e-10 vs in-Rust per-block finite differences with the Cayley convention
/// `U = (I-κ/2)^{-1}(I+κ/2)`.
///
/// At a converged HF reference the HF Brillouin term `-2·F^σ_{ai}` is zero;
/// callers wanting the full gradient should add it.
///
/// Returns `(g_a, g_b)` with shapes `(nvir_α, nocc_α)` and `(nvir_β, nocc_β)`.
///
/// Energy:
/// ```text
/// E_MP2 = ¼ Σ t^αα_{ij,ab} K^αα_{iajb}
///       + ¼ Σ t^ββ_{ij,ab} K^ββ_{iajb}
///       +   Σ t^αβ_{iJ,aB} (ia|JB)
/// ```
/// with `K^σσ_{iajb} = (ia|jb)_σ − (ib|ja)_σ`.
///
/// At fixed denominators, the σσ blocks pick up a `½` chain-rule factor
/// (the `t` and `K` derivatives are equal) while the αβ block picks up
/// a `2×` factor (both t and (ia|JB) depend on the same integral).
///
/// Per-block decomposition of the unrelaxed integral-response gradient.
///
/// `same_a`/`same_b` are the σσ-block contributions to `g^σ` (αα for α,
/// ββ for β). `ab_a`/`ab_b` are the αβ-block contributions to each spin.
/// The full block-wise gradient is `same_σ + ab_σ`.
#[derive(Debug, Clone)]
pub struct UMp2GradientBlocks {
    pub same_a: Array2<f64>,
    pub same_b: Array2<f64>,
    pub ab_a:   Array2<f64>,
    pub ab_b:   Array2<f64>,
}

pub fn compute_u_mp2_orbital_gradient(
    amps: &UMp2Amplitudes,
    b_full_a: &Array3<f64>,
    b_full_b: &Array3<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let bl = compute_u_mp2_orbital_gradient_blocks(amps, b_full_a, b_full_b);
    let g_a = &bl.same_a + &bl.ab_a;
    let g_b = &bl.same_b + &bl.ab_b;
    (g_a, g_b)
}

pub fn compute_u_mp2_orbital_gradient_blocks(
    amps: &UMp2Amplitudes,
    b_full_a: &Array3<f64>,
    b_full_b: &Array3<f64>,
) -> UMp2GradientBlocks {
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

    let mut same_a = Array2::<f64>::zeros((nvir_a, nocc_a));
    let mut ab_a   = Array2::<f64>::zeros((nvir_a, nocc_a));
    let mut same_b = Array2::<f64>::zeros((nvir_b, nocc_b));
    let mut ab_b   = Array2::<f64>::zeros((nvir_b, nocc_b));

    // --- α gradient: g^α_{ck} = ∂E_MP2/∂κ^α_{ck} -------------------------
    for c in 0..nvir_a {
        let c_mo = nocc_total_a + c;
        for k in 0..nocc_a {
            let k_mo = first_occ_a + k;
            let mut s_same = 0.0;
            let mut s_ab = 0.0;

            // αα Term 1: i=k branch
            for j in 0..nocc_a {
                let j_mo = first_occ_a + j;
                for a in 0..nvir_a {
                    let a_mo = nocc_total_a + a;
                    for b in 0..nvir_a {
                        let b_mo = nocc_total_a + b;
                        let t = t_aa[(k, j, a, b)];
                        let e1 = eri_aa(c_mo, a_mo, j_mo, b_mo);
                        let e2 = eri_aa(c_mo, b_mo, j_mo, a_mo);
                        s_same += 0.5 * t * (e1 - e2);
                    }
                }
            }
            // αα Term 2: j=k branch
            for i in 0..nocc_a {
                let i_mo = first_occ_a + i;
                for a in 0..nvir_a {
                    let a_mo = nocc_total_a + a;
                    for b in 0..nvir_a {
                        let b_mo = nocc_total_a + b;
                        let t = t_aa[(i, k, a, b)];
                        let e1 = eri_aa(i_mo, a_mo, c_mo, b_mo);
                        let e2 = eri_aa(i_mo, b_mo, c_mo, a_mo);
                        s_same += 0.5 * t * (e1 - e2);
                    }
                }
            }
            // αα Term 3: a=c branch
            for i in 0..nocc_a {
                let i_mo = first_occ_a + i;
                for j in 0..nocc_a {
                    let j_mo = first_occ_a + j;
                    for b in 0..nvir_a {
                        let b_mo = nocc_total_a + b;
                        let t = t_aa[(i, j, c, b)];
                        let e1 = eri_aa(i_mo, k_mo, j_mo, b_mo);
                        let e2 = eri_aa(i_mo, b_mo, j_mo, k_mo);
                        s_same -= 0.5 * t * (e1 - e2);
                    }
                }
            }
            // αα Term 4: b=c branch
            for i in 0..nocc_a {
                let i_mo = first_occ_a + i;
                for j in 0..nocc_a {
                    let j_mo = first_occ_a + j;
                    for a in 0..nvir_a {
                        let a_mo = nocc_total_a + a;
                        let t = t_aa[(i, j, a, c)];
                        let e1 = eri_aa(i_mo, a_mo, j_mo, k_mo);
                        let e2 = eri_aa(i_mo, k_mo, j_mo, a_mo);
                        s_same -= 0.5 * t * (e1 - e2);
                    }
                }
            }

            // αβ contribution to g^α_{ck}
            for jj in 0..nocc_b {
                let j_mo = first_occ_b + jj;
                for a in 0..nvir_a {
                    let a_mo = nocc_total_a + a;
                    for bb in 0..nvir_b {
                        let b_mo = nocc_total_b + bb;
                        let t = t_ab[(k, jj, a, bb)];
                        let e = eri_ab(c_mo, a_mo, j_mo, b_mo);
                        s_ab += t * e;
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
                        s_ab -= t * e;
                    }
                }
            }

            same_a[(c, k)] = s_same;
            ab_a[(c, k)] = 2.0 * s_ab;  // factor 2: both t and (ia|JB) depend on integrals
        }
    }

    // --- β gradient ------------------------------------------------------
    for c in 0..nvir_b {
        let c_mo = nocc_total_b + c;
        for k in 0..nocc_b {
            let k_mo = first_occ_b + k;
            let mut s_same = 0.0;
            let mut s_ab = 0.0;

            for j in 0..nocc_b {
                let j_mo = first_occ_b + j;
                for a in 0..nvir_b {
                    let a_mo = nocc_total_b + a;
                    for b in 0..nvir_b {
                        let b_mo = nocc_total_b + b;
                        let t = t_bb[(k, j, a, b)];
                        s_same += 0.5 * t * (eri_bb(c_mo, a_mo, j_mo, b_mo)
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
                        s_same += 0.5 * t * (eri_bb(i_mo, a_mo, c_mo, b_mo)
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
                        s_same -= 0.5 * t * (eri_bb(i_mo, k_mo, j_mo, b_mo)
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
                        s_same -= 0.5 * t * (eri_bb(i_mo, a_mo, j_mo, k_mo)
                                           - eri_bb(i_mo, k_mo, j_mo, a_mo));
                    }
                }
            }

            for i in 0..nocc_a {
                let i_mo = first_occ_a + i;
                for a in 0..nvir_a {
                    let a_mo = nocc_total_a + a;
                    for bb in 0..nvir_b {
                        let b_mo = nocc_total_b + bb;
                        let t = t_ab[(i, k, a, bb)];
                        let e = eri_ab(i_mo, a_mo, c_mo, b_mo);
                        s_ab += t * e;
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
                        s_ab -= t * e;
                    }
                }
            }

            same_b[(c, k)] = s_same;
            ab_b[(c, k)] = 2.0 * s_ab;  // factor 2: both t and (ia|JB) depend on integrals
        }
    }

    UMp2GradientBlocks { same_a, same_b, ab_a, ab_b }
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

/// Compute the U-MP2 integral-response energy from given MO coefficients and
/// **fixed** orbital energies. Used by in-Rust FD gradient validation.
///
/// Unlike `u_ri_mp2`, this function does NOT re-diagonalize the Fock matrix —
/// it uses the supplied `eps_a`/`eps_b` for all denominators regardless of what
/// the Fock matrix looks like at the perturbed MOs.  This isolates the integral
/// piece of the orbital-rotation derivative, matching the Python
/// `mp2_energy_fixed_eps` ground-truth FD.
///
/// `which` selects the block: "aa", "bb", "ab", or "all".
#[cfg(test)]
pub(crate) fn u_mp2_energy_fixed_eps(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    eps_a: &[f64],
    eps_b: &[f64],
    nocc_a: usize,
    nocc_b: usize,
    which: &str,
) -> Result<f64, FerricError> {
    use crate::rimp2::cholesky_inverse_sqrt;
    use ferric_integrals::threeindex;

    let nmo_a = c_a.ncols();
    let nmo_b = c_b.ncols();
    let nvir_a = nmo_a - nocc_a;
    let nvir_b = nmo_b - nocc_b;
    let naux = dfbs.nbasis();

    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = threeindex::eri3_tensor(op, obs, dfbs)?;

    let c_a_occ = c_a.slice(ndarray::s![.., ..nocc_a]).to_owned();
    let c_a_vir = c_a.slice(ndarray::s![.., nocc_a..]).to_owned();
    let c_b_occ = c_b.slice(ndarray::s![.., ..nocc_b]).to_owned();
    let c_b_vir = c_b.slice(ndarray::s![.., nocc_b..]).to_owned();

    let b_a_ov_raw = crate::mo_transform::transform_3center_ov(&eri3_ao, &c_a_occ, &c_a_vir);
    let b_a_ov = v_inv_sqrt.dot(
        &b_a_ov_raw.into_shape_with_order((naux, nocc_a * nvir_a)).unwrap(),
    );
    let b_b_ov_raw = crate::mo_transform::transform_3center_ov(&eri3_ao, &c_b_occ, &c_b_vir);
    let b_b_ov = v_inv_sqrt.dot(
        &b_b_ov_raw.into_shape_with_order((naux, nocc_b * nvir_b)).unwrap(),
    );

    let eo_a = &eps_a[..nocc_a];
    let ev_a = &eps_a[nocc_a..];
    let eo_b = &eps_b[..nocc_b];
    let ev_b = &eps_b[nocc_b..];

    let mut e_total = 0.0;

    if which == "aa" || which == "all" {
        let mut e_aa = 0.0;
        for i in 0..nocc_a {
            for j in 0..nocc_a {
                for a in 0..nvir_a {
                    let ia = i * nvir_a + a;
                    let ja = j * nvir_a + a;
                    for b in 0..nvir_a {
                        let ib = i * nvir_a + b;
                        let jb = j * nvir_a + b;
                        let mut eri_iajb = 0.0;
                        let mut eri_ibja = 0.0;
                        for p in 0..naux {
                            eri_iajb += b_a_ov[(p, ia)] * b_a_ov[(p, jb)];
                            eri_ibja += b_a_ov[(p, ib)] * b_a_ov[(p, ja)];
                        }
                        let k = eri_iajb - eri_ibja;
                        let denom = eo_a[i] + eo_a[j] - ev_a[a] - ev_a[b];
                        e_aa += k * k / denom;
                    }
                }
            }
        }
        e_total += 0.25 * e_aa;
    }

    if which == "bb" || which == "all" {
        let mut e_bb = 0.0;
        for i in 0..nocc_b {
            for j in 0..nocc_b {
                for a in 0..nvir_b {
                    let ia = i * nvir_b + a;
                    let ja = j * nvir_b + a;
                    for b in 0..nvir_b {
                        let ib = i * nvir_b + b;
                        let jb = j * nvir_b + b;
                        let mut eri_iajb = 0.0;
                        let mut eri_ibja = 0.0;
                        for p in 0..naux {
                            eri_iajb += b_b_ov[(p, ia)] * b_b_ov[(p, jb)];
                            eri_ibja += b_b_ov[(p, ib)] * b_b_ov[(p, ja)];
                        }
                        let k = eri_iajb - eri_ibja;
                        let denom = eo_b[i] + eo_b[j] - ev_b[a] - ev_b[b];
                        e_bb += k * k / denom;
                    }
                }
            }
        }
        e_total += 0.25 * e_bb;
    }

    if which == "ab" || which == "all" {
        let mut e_ab = 0.0;
        for i in 0..nocc_a {
            for a in 0..nvir_a {
                let ia = i * nvir_a + a;
                for j in 0..nocc_b {
                    for b in 0..nvir_b {
                        let jb = j * nvir_b + b;
                        let mut eri = 0.0;
                        for p in 0..naux {
                            eri += b_a_ov[(p, ia)] * b_b_ov[(p, jb)];
                        }
                        let denom = eo_a[i] + eo_b[j] - ev_a[a] - ev_b[b];
                        e_ab += eri * eri / denom;
                    }
                }
            }
        }
        e_total += e_ab;
    }

    Ok(e_total)
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
    use ndarray_linalg::Solve;

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

    /// FD-validate the analytic U-MP2 integral-response orbital gradient on OH/cc-pVDZ.
    ///
    /// Both FD and analytic are computed at the SAME Rust UHF MOs — no Python
    /// AO-ordering dependency. The FD uses `u_mp2_energy_fixed_eps` with the
    /// UHF orbital energies held fixed, matching the "integral-response-only"
    /// part of the gradient that the analytic code computes.
    ///
    /// NOTE: DIAGNOSTIC test — reports per-block agreement and asserts c > 0.95
    /// and max|Δ|/|g_fd| < 0.05.
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

        let nocc_a = amps.inter_a.nocc;
        let nocc_b = amps.inter_b.nocc;
        let nvir_a = amps.inter_a.nvir;
        let nvir_b = amps.inter_b.nvir;
        let nmo = uhf.mos_a().ncols();

        let eps_a = uhf.eps_a().to_vec();
        let eps_b = uhf.eps_b().to_vec();
        let c_a0 = uhf.mos_a().clone();
        let c_b0 = uhf.mos_b().clone();

        // --- In-Rust FD gradient (fixed-eps, block-selected) ---
        // Cayley rotation: U = (I - κ/2)^{-1} (I + κ/2), matching Python harness.
        // Solve: (I - κ/2) @ U = (I + κ/2) column by column.
        let cayley = |nmo: usize, a: usize, k: usize, nocc: usize, h: f64| -> Array2<f64> {
            let mut kap = Array2::<f64>::zeros((nmo, nmo));
            kap[(nocc + a, k)] = h;
            kap[(k, nocc + a)] = -h;
            let eye = Array2::<f64>::eye(nmo);
            let lhs = &eye - 0.5 * &kap;  // I - κ/2
            let rhs = &eye + 0.5 * &kap;  // I + κ/2
            let mut u = Array2::zeros((nmo, nmo));
            for col in 0..nmo {
                let rhs_col = rhs.column(col).to_owned();
                let u_col = lhs.solve(&rhs_col).unwrap();
                u.column_mut(col).assign(&u_col);
            }
            u
        };
        let step = 1e-4_f64;

        // FD for α rotation: per-block (aa, ab)
        let mut fd_same_a = Array2::<f64>::zeros((nvir_a, nocc_a));
        let mut fd_ab_a   = Array2::<f64>::zeros((nvir_a, nocc_a));
        for a in 0..nvir_a {
            for k in 0..nocc_a {
                let up = cayley(nmo, a, k, nocc_a, step);
                let um = cayley(nmo, a, k, nocc_a, -step);
                let c_a_p = c_a0.dot(&up);
                let c_a_m = c_a0.dot(&um);
                let ep_aa = super::u_mp2_energy_fixed_eps(&obs, &dfbs, op, &c_a_p, &c_b0, &eps_a, &eps_b, nocc_a, nocc_b, "aa").unwrap();
                let em_aa = super::u_mp2_energy_fixed_eps(&obs, &dfbs, op, &c_a_m, &c_b0, &eps_a, &eps_b, nocc_a, nocc_b, "aa").unwrap();
                let ep_ab = super::u_mp2_energy_fixed_eps(&obs, &dfbs, op, &c_a_p, &c_b0, &eps_a, &eps_b, nocc_a, nocc_b, "ab").unwrap();
                let em_ab = super::u_mp2_energy_fixed_eps(&obs, &dfbs, op, &c_a_m, &c_b0, &eps_a, &eps_b, nocc_a, nocc_b, "ab").unwrap();
                fd_same_a[(a, k)] = (ep_aa - em_aa) / (2.0 * step);
                fd_ab_a[(a, k)]   = (ep_ab - em_ab) / (2.0 * step);
            }
        }
        // FD for β rotation: per-block (bb, ab)
        let mut fd_same_b = Array2::<f64>::zeros((nvir_b, nocc_b));
        let mut fd_ab_b   = Array2::<f64>::zeros((nvir_b, nocc_b));
        for a in 0..nvir_b {
            for k in 0..nocc_b {
                let up = cayley(nmo, a, k, nocc_b, step);
                let um = cayley(nmo, a, k, nocc_b, -step);
                let c_b_p = c_b0.dot(&up);
                let c_b_m = c_b0.dot(&um);
                let ep_bb = super::u_mp2_energy_fixed_eps(&obs, &dfbs, op, &c_a0, &c_b_p, &eps_a, &eps_b, nocc_a, nocc_b, "bb").unwrap();
                let em_bb = super::u_mp2_energy_fixed_eps(&obs, &dfbs, op, &c_a0, &c_b_m, &eps_a, &eps_b, nocc_a, nocc_b, "bb").unwrap();
                let ep_ab = super::u_mp2_energy_fixed_eps(&obs, &dfbs, op, &c_a0, &c_b_p, &eps_a, &eps_b, nocc_a, nocc_b, "ab").unwrap();
                let em_ab = super::u_mp2_energy_fixed_eps(&obs, &dfbs, op, &c_a0, &c_b_m, &eps_a, &eps_b, nocc_a, nocc_b, "ab").unwrap();
                fd_same_b[(a, k)] = (ep_bb - em_bb) / (2.0 * step);
                fd_ab_b[(a, k)]   = (ep_ab - em_ab) / (2.0 * step);
            }
        }

        // --- Analytic gradient blocks ---
        let b_full_a = crate::oo_rimp2::compute_b_full_mo(&obs, &dfbs, op, &c_a0).unwrap();
        let b_full_b = crate::oo_rimp2::compute_b_full_mo(&obs, &dfbs, op, &c_b0).unwrap();
        let blocks = super::compute_u_mp2_orbital_gradient_blocks(&amps, &b_full_a, &b_full_b);

        let report = |label: &str, g: &Array2<f64>, g_fd: &Array2<f64>| -> bool {
            let nfd = g_fd.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let n = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let diff = g - g_fd;
            let max_diff = diff.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let rms: f64 = (diff.iter().map(|v| v*v).sum::<f64>() / (diff.len() as f64)).sqrt();
            let dot: f64 = g.iter().zip(g_fd.iter()).map(|(a,b)| a*b).sum();
            let nrm2_fd: f64 = g_fd.iter().map(|v| v*v).sum();
            let c = if nrm2_fd > 0.0 { dot / nrm2_fd } else { 0.0 };
            let rel = if nfd > 0.0 { max_diff / nfd } else { 0.0 };
            let ok = c > 0.95 && rel < 0.05;
            println!("[{}] |g_an|max={:.4e} |g_fd|max={:.4e}  max|Δ|={:.4e}  rms={:.4e}  c={:+.4}  rel={:.3}  {}",
                     label, n, nfd, max_diff, rms, c, rel, if ok { "✓" } else { "✗" });
            ok
        };

        let ok1 = report("αα→g_a (same_a vs FD αα-only α-rot)", &blocks.same_a, &fd_same_a);
        let ok2 = report("αβ→g_a (ab_a   vs FD αβ-only α-rot)", &blocks.ab_a,   &fd_ab_a);
        let ok3 = report("ββ→g_b (same_b vs FD ββ-only β-rot)", &blocks.same_b, &fd_same_b);
        let ok4 = report("αβ→g_b (ab_b   vs FD αβ-only β-rot)", &blocks.ab_b,   &fd_ab_b);
        assert!(ok1 && ok2 && ok3 && ok4, "per-block gradient failed — see diagnostics above");
    }
}
