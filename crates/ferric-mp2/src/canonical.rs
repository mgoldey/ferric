//! Canonical MP2 using full 4-center ERIs transformed to MO basis.
//! For cross-validation only -- O(N^5) or worse.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfResult;

pub fn canonical_mp2(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    rhf: &RhfResult,
    frozen_core: usize,
) -> Result<f64, FerricError> {
    let nbas = prep.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let nocc = nocc_total - frozen_core;
    let first_occ = frozen_core;
    let nvir = nbas - nocc_total;
    let eps = &rhf.orbital_energies;
    let c = &rhf.mos;
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();

    // Build (ia|jb) MO integrals directly from AO shell-quartet loop
    let nov = nocc * nvir;
    let mut mo_eri = vec![0.0f64; nov * nov];
    let mut eng = Engine::new_2e(op, prep, 1e-14)?;

    for s1 in 0..nsh {
        for s2 in 0..nsh {
            for s3 in 0..nsh {
                for s4 in 0..nsh {
                    let quartet = eng.compute_quartet(prep, s1, s2, s3, s4);
                    if let Some(q) = quartet {
                        let (n1, n2, n3, n4) = (dims[s1], dims[s2], dims[s3], dims[s4]);
                        let (o1, o2, o3, o4) = (offs[s1], offs[s2], offs[s3], offs[s4]);
                        for a in 0..n1 {
                            let mu = o1 + a;
                            for b in 0..n2 {
                                let nu = o2 + b;
                                for cc in 0..n3 {
                                    let la = o3 + cc;
                                    for dd in 0..n4 {
                                        let sg = o4 + dd;
                                        let val = q[((a * n2 + b) * n3 + cc) * n4 + dd];
                                        if val.abs() < 1e-15 {
                                            continue;
                                        }
                                        // Transform (mu nu | la sg) to MO basis
                                        for i in 0..nocc {
                                            let ci = c[(mu, first_occ + i)];
                                            if ci.abs() < 1e-15 {
                                                continue;
                                            }
                                            for aa in 0..nvir {
                                                let ca = c[(nu, nocc_total + aa)];
                                                if ca.abs() < 1e-15 {
                                                    continue;
                                                }
                                                let left = ci * ca * val;
                                                for j in 0..nocc {
                                                    let cj = c[(la, first_occ + j)];
                                                    if cj.abs() < 1e-15 {
                                                        continue;
                                                    }
                                                    for bb in 0..nvir {
                                                        let cb = c[(sg, nocc_total + bb)];
                                                        mo_eri[(i * nvir + aa) * nov + j * nvir + bb] +=
                                                            left * cj * cb;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // MP2 energy
    let mut e_mp2 = 0.0;
    for i in 0..nocc {
        for j in 0..nocc {
            for a in 0..nvir {
                for b in 0..nvir {
                    let ia = i * nvir + a;
                    let jb = j * nvir + b;
                    let ib = i * nvir + b;
                    let ja = j * nvir + a;
                    let denom = eps[first_occ + i] + eps[first_occ + j]
                        - eps[nocc_total + a]
                        - eps[nocc_total + b];
                    e_mp2 += mo_eri[ia * nov + jb]
                        * (2.0 * mo_eri[ia * nov + jb] - mo_eri[ib * nov + ja])
                        / denom;
                }
            }
        }
    }
    Ok(e_mp2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    #[test]
    fn test_canonical_mp2_h2_sto3g() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let rhf = solve_rhf(
            &mol,
            &prep,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        let e_corr = canonical_mp2(&mol, &prep, op, &rhf, 0).unwrap();
        eprintln!(
            "H2/STO-3G: RHF={:.10}, canonical MP2 corr={:.10}",
            rhf.energy, e_corr
        );
        // PySCF: -0.0131380736
        assert!(
            (e_corr - (-0.0131380736)).abs() < 1e-7,
            "H2/STO-3G MP2 corr: {e_corr:.10}"
        );
    }

    #[test]
    fn test_canonical_vs_ri_mp2_h2_ccpvdz() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let rhf = solve_rhf(
            &mol,
            &prep,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        let e_canonical = canonical_mp2(&mol, &prep, op, &rhf, 0).unwrap();

        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let ri_result = crate::rimp2::ri_mp2(
            &mol,
            &prep,
            &dfbs,
            op,
            &rhf,
            &crate::rimp2::RiMp2Config::default(),
        )
        .unwrap();

        eprintln!(
            "H2/cc-pVDZ: canonical={:.10}, RI={:.10}, diff={:.2e}",
            e_canonical,
            ri_result.mp2_corr,
            (e_canonical - ri_result.mp2_corr).abs()
        );

        let diff = (e_canonical - ri_result.mp2_corr).abs();
        assert!(
            diff < 1e-4,
            "canonical={e_canonical:.10} ri={:.10} diff={diff:.2e}",
            ri_result.mp2_corr
        );
    }
}
