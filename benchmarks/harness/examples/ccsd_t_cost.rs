//! Where does CCSD(T) time actually go — the CCSD iterations, or the (T)
//! triples?
//!
//! `689b2e3` routed closed-shell CCSD to the spin-adapted solver (22x), but
//! `run_ccsd_t` deliberately still uses the SPIN-ORBITAL `ccsd` because
//! `ccsd_t` consumes its amplitudes and the two conventions differ in shape.
//! Before writing a spin-adapted (T) — real derivation work — measure what that
//! would actually buy: if (T) itself dominates, the CCSD half being slow
//! matters little; if the CCSD half dominates, an amplitude conversion alone
//! would capture most of the win for far less risk.
use std::time::Instant;

use ferric_cc::ccsd::ccsd;
use ferric_cc::ccsd_closed_shell::ccsd_closed_shell;
use ferric_cc::ccsd_t::ccsd_t;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn main() {
    let mol_name = std::env::var("FERRIC_MOL").unwrap_or_else(|_| "water".into());
    let obs_name = std::env::var("FERRIC_OBS").unwrap_or_else(|_| "cc-pvdz".into());
    let aux_name = std::env::var("FERRIC_AUX").unwrap_or_else(|_| "cc-pvdz-ri".into());

    let mol = Molecule::load_xyz(&format!("testdata/molecules/{mol_name}.xyz")).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(&obs_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(&aux_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig { max_iter: 100, ..Default::default() },
    )
    .unwrap();
    assert!(rhf.converged);

    let nocc = rhf.eps_r().iter().filter(|&&e| e < 0.0).count();
    let nvir = obs.nbasis() - nocc;
    println!(
        "{mol_name}/{obs_name}: nbasis={} no={nocc} nv={nvir}  (spin-orbital: 2no={} 2nv={})",
        obs.nbasis(),
        2 * nocc,
        2 * nvir
    );

    let cfg = ferric_cc::CcConfig::default();

    // The path `run_ccsd_t` takes today.
    let t = Instant::now();
    let r_so = ccsd(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let t_ccsd_so = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let e_t = ccsd_t(&mol, &obs, &dfbs, op, &rhf, &r_so, &cfg).unwrap();
    let t_triples = t.elapsed().as_secs_f64();

    // The spin-adapted CCSD, for reference — what the CCSD half COULD cost.
    let t = Instant::now();
    let r_cs = ccsd_closed_shell(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let t_ccsd_cs = t.elapsed().as_secs_f64();

    let total = t_ccsd_so + t_triples;
    println!("\nCCSD(T) as run today (spin-orbital CCSD + spin-orbital (T)):");
    println!("  spin-orbital CCSD   {t_ccsd_so:8.2} s  ({:.0}%)", 100.0 * t_ccsd_so / total);
    println!("  (T) triples         {t_triples:8.2} s  ({:.0}%)", 100.0 * t_triples / total);
    println!("  TOTAL               {total:8.2} s");
    println!("\nFor comparison:");
    println!("  spin-adapted CCSD   {t_ccsd_cs:8.2} s  ({:.1}x faster than spin-orbital)", t_ccsd_so / t_ccsd_cs);
    println!(
        "\n  E_corr(CCSD) so={:.10}  cs={:.10}  diff={:.2e}",
        r_so.correlation_energy,
        r_cs.correlation_energy,
        (r_so.correlation_energy - r_cs.correlation_energy).abs()
    );
    println!("  E_(T) = {e_t:.10}");

    // What an amplitude-conversion-only fix would buy (CCSD half fast, (T)
    // unchanged) vs a full spin-adapted (T) (both fast).
    let convert_only = t_ccsd_cs + t_triples;
    println!("\nProjected:");
    println!(
        "  convert amplitudes only (fast CCSD + current (T)): {convert_only:.2} s  => {:.2}x",
        total / convert_only
    );
    println!(
        "  ceiling if (T) were ALSO free:                     {t_ccsd_cs:.2} s  => {:.2}x",
        total / t_ccsd_cs
    );
}
