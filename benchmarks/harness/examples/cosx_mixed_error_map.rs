//! COSX mixed-precision (`cosx-kern`) error map: the table of Task D2's
//! pre-registered sweep. Counts and errors only; no timing.
//!
//! ```text
//! cargo run --release -p ferric-benchmarks --example cosx_mixed_error_map -- \
//!     [--alkanes 4,8,12] [--tzvp-butane] [--scf-all]
//! ```
//!
//! Per row: routed flop share, `max|dK|`, its derived bound, `dE_x`, kappa,
//! and (for the multipliers with an SCF) `|dE_SCF|` in Eh and kcal/mol and per
//! atom with the iteration counts. The ship gate (decided by D6, not here) is
//! `|dE_total| <= 1 mEh` on every row and a `ln|dE|` vs `ln N` slope not above
//! about 1 within its uncertainty; Ochsenfeld's 1.8 microEh is reported.
//! The core is shared with `crates/ferric-scf/tests/cosx_mixed_error_law.rs`.

#[path = "../../../crates/ferric-scf/tests/common/cosx_error_map.rs"]
mod cosx_error_map;

use cosx_error_map as map;

const MULTS: [f64; 3] = [1e4, 1e5, 1e6];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut sizes = vec![4usize, 8, 12];
    let mut tzvp = false;
    let mut scf_all = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--alkanes" => {
                sizes = it
                    .next()
                    .expect("--alkanes needs a list")
                    .split(',')
                    .map(|s| s.parse().expect("alkane size"))
                    .collect();
            }
            "--tzvp-butane" => tzvp = true,
            "--scf-all" => scf_all = true,
            other => panic!("unknown argument {other}"),
        }
    }
    let mut svp = Vec::new();
    for &n in &sizes {
        let sys = map::load(&format!("C{n}"), &format!("alkane_{n}.xyz"), "def2-svp");
        let scf: &[f64] = if scf_all {
            &MULTS
        } else if n <= 8 {
            &[1e5]
        } else {
            &[]
        };
        svp.extend(map::run_system(&sys, &MULTS, scf, &map::Opts::production()));
    }
    map::print_rows(&svp);
    let sl = map::slopes(&svp);
    map::print_slopes(&sl);
    map::print_device_rule(&sl);
    if tzvp {
        let sys = map::load("C4", "alkane_4.xyz", "def2-tzvp");
        let rows = map::run_system(&sys, &MULTS, &[1e5], &map::Opts::production());
        map::print_rows(&rows);
    }
}
