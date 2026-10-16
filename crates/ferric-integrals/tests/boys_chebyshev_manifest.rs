//! Bit-identity manifest for the Boys table-edge fix (issue #433).
//!
//! Prints bit-pattern hashes of integral families that must NOT change when
//! `FmEval_Chebyshev7::eval` is changed from `x > T_crit` to `x >= T_crit`
//! (none of them has a Boys argument of exactly 117.0): molecular erfc, erf
//! and Coulomb 3-centre tensors (the attenuated RI-MP2 path), 2-centre
//! metrics, and hcore-style lattice-shifted erfc blocks of a point-site basis
//! at the hcore engine precision. The hashes are compared across builds by
//! hand (`cargo test ... -- --nocapture | grep MANIFEST`); they are not pinned
//! here, because bit patterns are not portable across machines. The test only
//! asserts that every block is finite.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ferric_integrals::threeindex::eri3_tensor;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

fn fnv(h: &mut u64, x: f64) {
    assert!(x.is_finite(), "non-finite integral {x}");
    for b in x.to_bits().to_le_bytes() {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

#[test]
fn manifest_prints_hashes() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).expect("water");
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let report = |name: &str, h: u64| println!("MANIFEST {name}={h:016x}");

    for (name, op) in [
        ("eri3_coulomb", Operator::coulomb()),
        ("eri3_erfc042", Operator::erfc(0.42)),
        ("eri3_erfc08", Operator::erfc(0.8)),
        ("eri3_erf05", Operator::erf(0.5)),
    ] {
        let t = eri3_tensor(op, &obs, &aux).unwrap();
        let mut h = FNV_OFFSET;
        t.iter().for_each(|&x| fnv(&mut h, x));
        report(name, h);
    }

    for (name, op) in [
        ("eri2_coulomb", Operator::coulomb()),
        ("eri2_erfc042", Operator::erfc(0.42)),
    ] {
        let mut eng = Engine::new_2center(op, &aux, 1e-14).unwrap();
        let mut h = FNV_OFFSET;
        for p in 0..aux.nshells() {
            for q in 0..aux.nshells() {
                eng.compute_eri2(&aux, p, q)
                    .iter()
                    .for_each(|&x| fnv(&mut h, x));
            }
        }
        report(name, h);
    }

    // hcore-style SR blocks: water orbitals against point sites on the atoms,
    // lattice-shifted, erfc(0.8).
    let sites: Vec<[f64; 4]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos, 1e8]).collect();
    let site = SiteBasis::new(&sites, 0).unwrap();
    for (name, prec) in [
        ("hcore_sr_minpos", f64::MIN_POSITIVE),
        ("hcore_sr_1e14", 1e-14),
    ] {
        let mut eng = Engine::new_3center(Operator::erfc(0.8), &obs, &site.prep, prec).unwrap();
        let mut h = FNV_OFFSET;
        let mut n = 0usize;
        for &sh in &site.site_shell {
            for l in [-1.0, 0.0, 1.0] {
                for m in [-1.0, 0.0, 2.0] {
                    for s1 in 0..obs.nshells() {
                        for s2 in 0..obs.nshells() {
                            let shifts = [[5.5 * m, 0.0, 0.0], [0.0; 3], [0.0, 0.0, 6.5 * l]];
                            match eng
                                .compute_eri3_shifted(&obs, &site.prep, sh, s1, s2, shifts)
                                .unwrap()
                            {
                                Some(b) => {
                                    b.iter().for_each(|&x| fnv(&mut h, x));
                                    n += 1;
                                }
                                None => fnv(&mut h, 0.0),
                            }
                        }
                    }
                }
            }
        }
        report(name, h);
        println!("MANIFEST {name}_kept_blocks={n}");
    }
}
