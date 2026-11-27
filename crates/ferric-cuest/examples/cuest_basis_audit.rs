// Audit ferric's bundled basis sets against cuEST's structural rules
// (ferric_cuest::audit): per-element angular momenta contiguous from l = 0,
// and one spherical/Cartesian representation for l >= 2. Reports; never fails.

use ferric_core::basis;
use ferric_cuest::audit::audit_basis;

fn main() {
    let names = basis::bundled_names();
    let mut n_bad = 0;
    for name in &names {
        let Ok(bs) = basis::bundled(name) else {
            continue;
        };
        let audit = audit_basis(&bs);
        if !audit.is_ok() {
            n_bad += 1;
            for v in &audit.violations {
                println!("  {name}: {:?}: {}", v.kind, v.detail);
            }
        }
    }
    println!(
        "audited {} bundled basis sets, {n_bad} with violations",
        names.len()
    );
}
