// Audit ferric's bundled basis sets against cuEST's constraints:
//   1. Angular momentum span contiguous from L=0 (no holes).
//   2. Uniform spherical/cartesian representation within a basis set
//      (all shells share the same `pure` value).
// Reports violations; does not fail. Used for Phase 0.4.

use ferric_core::basis;
use std::collections::BTreeSet;

fn main() {
    let mut bad_contig = Vec::new();
    let mut bad_mixed = Vec::new();
    let mut total = 0;
    for name in basis::bundled_names() {
        total += 1;
        let bs = match basis::bundled(name) {
            Ok(b) => b,
            Err(_) => continue,
        };
        // Collect L values and pure flags across all elements/shells.
        let mut ls: BTreeSet<i32> = BTreeSet::new();
        let mut purities: BTreeSet<bool> = BTreeSet::new();
        for shells in bs.shells.values() {
            for sh in shells {
                ls.insert(sh.l);
                // cuEST's spherical/cartesian rule only matters for l>=2;
                // s and p have the same function count either way.
                if sh.l >= 2 {
                    purities.insert(sh.pure);
                }
            }
        }
        // Contiguity from 0: max L must equal count-1 of distinct L starting at 0.
        if !ls.is_empty() {
            let maxl = *ls.iter().max().unwrap();
            let contiguous = (0..=maxl).all(|l| ls.contains(&l));
            if !contiguous {
                bad_contig.push((name.to_string(), ls.iter().copied().collect::<Vec<_>>()));
            }
        }
        if purities.len() > 1 {
            bad_mixed.push((
                name.to_string(),
                purities.iter().copied().collect::<Vec<_>>(),
            ));
        }
    }
    println!("audited {} bundled basis sets", total);
    println!("non-contiguous L: {}", bad_contig.len());
    for (n, ls) in &bad_contig {
        println!("  {} L={:?}", n, ls);
    }
    println!("mixed pure/cartesian: {}", bad_mixed.len());
    for (n, p) in &bad_mixed {
        println!("  {} pure={:?}", n, p);
    }
}
