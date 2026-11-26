//! Write libint one-electron references (S, T, V) for the cuEST comparison.
//!
//!     cargo run --release -p ferric-integrals --example gen_cuest_oe_reference
//!
//! Output: testdata/reference/cuest/<system>_<basis>.json, read by
//! crates/ferric-cuest/tests/oe_vs_libint.rs on a GPU host (which therefore
//! needs no libint). The molecule is NH3 in a generic orientation: four
//! atoms, non-planar, no coordinate axis along a bond, so every AO has a
//! non-zero fingerprint for ferric_cuest::layout::infer_map.

use ferric_core::basis::bundled;
use ferric_core::mol::{Atom, Molecule};
use ferric_cuest::reference::{reference_dir, OneElectronReference, RefAtom};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::{kinetic, nuclear, overlap};

const ANGSTROM_TO_BOHR: f64 = 1.0 / 0.529_177_210_903;
const BASES: [&str; 4] = ["sto-3g", "6-31g*", "cc-pvdz", "cc-pvtz"];

fn rotate(p: [f64; 3], (a, b, c): (f64, f64, f64)) -> [f64; 3] {
    // R = Rz(c) Ry(b) Rx(a)
    let [x, y, z] = p;
    let (y, z) = (y * a.cos() - z * a.sin(), y * a.sin() + z * a.cos());
    let (x, z) = (x * b.cos() + z * b.sin(), -x * b.sin() + z * b.cos());
    let (x, y) = (x * c.cos() - y * c.sin(), x * c.sin() + y * c.cos());
    [x, y, z]
}

fn nh3_generic() -> Molecule {
    let raw = [
        ("N", 7, [0.0, 0.0, 0.1173]),
        ("H", 1, [0.0, 0.9377, -0.2738]),
        ("H", 1, [0.8121, -0.4689, -0.2738]),
        ("H", 1, [-0.8121, -0.4689, -0.2738]),
    ];
    let shift = [0.1, -0.2, 0.3];
    let atoms = raw
        .iter()
        .map(|&(sym, z, p)| {
            let r = rotate(p, (0.3, 0.7, 1.1));
            Atom {
                symbol: sym.into(),
                z,
                x: (r[0] + shift[0]) * ANGSTROM_TO_BOHR,
                y: (r[1] + shift[1]) * ANGSTROM_TO_BOHR,
                zpos: (r[2] + shift[2]) * ANGSTROM_TO_BOHR,
                ghost: false,
                n_core_ecp: 0,
            }
        })
        .collect();
    Molecule {
        atoms,
        charge: 0,
        multiplicity: 1,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mol = nh3_generic();
    let dir = reference_dir();
    std::fs::create_dir_all(&dir)?;
    for name in BASES {
        let bs = bundled(name)?;
        let prep = PreparedBasis::new(&mol, &bs)?;
        let shell_l = mol
            .atoms
            .iter()
            .flat_map(|a| bs.for_element(a.z).unwrap_or_default().iter().map(|s| s.l))
            .collect();
        let (s, t, v) = (overlap(&prep), kinetic(&prep), nuclear(&prep));
        let nao = s.nrows();
        let r = OneElectronReference {
            system: "nh3_generic".into(),
            basis: name.into(),
            generator: format!(
                "ferric-integrals libint2 overlap/kinetic/nuclear, {}",
                ferric_build_info_string()
            ),
            atoms: mol
                .atoms
                .iter()
                .map(|a| RefAtom {
                    symbol: a.symbol.clone(),
                    z: a.z,
                    xyz: [a.x, a.y, a.zpos],
                })
                .collect(),
            shell_l,
            nao,
            s: s.iter().copied().collect(),
            t: t.iter().copied().collect(),
            v: v.iter().copied().collect(),
        };
        let file = dir.join(format!("nh3_generic_{}.json", name.replace('*', "star")));
        std::fs::write(&file, serde_json::to_string(&r)?)?;
        println!("{} nao={nao}", file.display());
    }
    Ok(())
}

fn ferric_build_info_string() -> String {
    format!("ferric-integrals {}", env!("CARGO_PKG_VERSION"))
}
