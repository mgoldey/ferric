//! Dump one CAS-CI run with RDMs for `scripts/gen_casci_rdm_refs.py`.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-ci --example casci_rdm_dump -- \
//!     <n2_r1.10|n2_r2.20|h2o> <out.json>
//! ```
//!
//! Writes ferric's active orbitals (`c_act`, AO x active, row-major), the
//! active-space integrals (`h`, `eri` flat `((p*n+q)*n+r)*n+s`, `e_core`), the
//! α/β strings (bitmasks), the CI vector (`ia * n_beta + ib`), ferric's
//! `dm1`/`dm2` (flat, row-major) and the Davidson eigenvalue. The generator
//! runs PySCF on these integrals; PySCF never supplies orbitals to ferric.
//!
//! The systems and settings are those of `tests/casci_rdms.rs`
//! (`casci_rdm_common::SYSTEMS`). The same dump feeds
//! `scripts/gen_casci_rdm3_refs.py` (`tests/casci_rdm3.rs`).

#[path = "../tests/casci_rdm_common/mod.rs"]
mod casci_rdm_common;

use casci_rdm_common::{run_system, system};
use serde_json::json;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: casci_rdm_dump <system> <out.json>");
        std::process::exit(2);
    }
    let sys = system(&args[1]);
    let run = run_system(&sys);
    let res = &run.res;
    let rdms = res.rdms.as_ref().expect("compute_rdms was set");
    let n = sys.n_active;
    let c = run.rhf.mos_r();
    let nbas = c.nrows();
    let mut c_act = Vec::with_capacity(nbas * n);
    for mu in 0..nbas {
        for p in 0..n {
            c_act.push(c[(mu, sys.active_start + p)]);
        }
    }
    let ints = &res.integrals;
    let out = json!({
        "system": sys.name,
        "basis": sys.basis,
        "active_start": sys.active_start,
        "n_active": n,
        "n_elec_active": [sys.n_elec.0, sys.n_elec.1],
        "nbas": nbas,
        "c_act": c_act,
        "h": ints.h.iter().copied().collect::<Vec<f64>>(),
        "eri": ints.eri,
        "e_core": ints.e_core,
        "alpha_strings": run.space.alpha_strings,
        "beta_strings": run.space.beta_strings,
        "ci_vector": res.ci_vector,
        "e_total": res.e_total,
        "dm1": rdms.dm1.iter().copied().collect::<Vec<f64>>(),
        "dm2": rdms.dm2.iter().copied().collect::<Vec<f64>>(),
    });
    std::fs::write(&args[2], serde_json::to_string(&out).expect("json")).expect("write");
    eprintln!(
        "{}: E_CASCI {:.12} N_det {} written to {}",
        sys.name, res.e_total, res.n_determinants, args[2]
    );
}
