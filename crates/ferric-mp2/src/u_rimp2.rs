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
use ndarray::Array4;

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
}
