//! **DIIS vs AURORA vs TRAH on identical systems.**
//!
//! The two accelerators were developed independently and measured on
//! DIFFERENT systems with DIFFERENT metrics, which makes any ranking taken
//! from their own reports an invented comparison. This harness runs all three
//! arms on the same molecules, same basis, same thresholds, same process.
//!
//! # What to read, and what to ignore
//!
//! ITERATIONS and TARGET J/K BUILDS are integer counts, immune to machine
//! load — they are the comparison. WALL TIME is reported but is only
//! trustworthy when the box is quiet; a contended box has produced 138%
//! run-to-run spread in this workspace. Check `uptime` before believing the
//! seconds column.
//!
//! Both accelerators must reach the SAME stationary point as DIIS. That is
//! asserted, not eyeballed: an accelerator that changes the answer is broken,
//! however fast it is.
//!
//! `#[ignore]`d: this is a measurement harness, not a CI gate.
//!
//! # MEASURED 2026-09-18 (full log: `scripts/queue/accelerator_head_to_head.txt`)
//!
//! ```text
//!   system                 it_D  it_A  it_T      t_D     t_A      t_T
//!   water/cc-pVDZ   RHF      12    10     8     0.63    0.26     1.29
//!   water/cc-pVDZ   PBE      87    87     7     3.30    1.86     7.96
//!   benzene/STO-3G  RHF      10     8     6     0.24    0.18     4.16
//!   benzene/cc-pVDZ RHF      12    10     7    11.08   10.76   497.09
//! ```
//!
//! All three arms reach the same stationary point on every row (asserted).
//!
//! **TRAH wins ITERATIONS everywhere and loses WALL TIME everywhere**, and the
//! wall-time gap GROWS with system size: 2.0x DIIS on water, 17x on
//! benzene/STO-3G, 45x on benzene/cc-pVDZ. Each TRAH step pays a Davidson of
//! Fock builds, so it trades many cheap iterations for few expensive ones.
//! That is why TRAH is opt-in and off by default, and why an
//! iteration-count-only comparison would have been actively misleading here.
//!
//! **AURORA wins or ties on both axes on every row.** It is the better default
//! accelerator; TRAH's value is convergence RESCUE, not speed -- most visibly
//! on the RKS/PBE row where DIIS and AURORA both need 87 iterations and TRAH
//! needs 7.
//!
//! The honest summary: they are complementary, not competing. AURORA for
//! throughput, TRAH for cases that converge badly or not at all.

use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::aurora::{target_jk_builds, AuroraConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Diis,
    Aurora,
    Trah,
}

struct Row {
    iters: usize,
    jk: usize,
    secs: f64,
    energy: f64,
    converged: bool,
}

fn run(xyz: &str, bas: &str, xc: Option<&str>, arm: Arm) -> Row {
    let mol = Molecule::load_xyz(xyz).expect("xyz");
    let bs = basis::bundled(bas).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let ctx = ParallelContext::default();

    let cfg = RhfConfig {
        xc: xc.map(|s| s.to_string()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 300,
        aurora: AuroraConfig {
            enabled: arm == Arm::Aurora,
            ..Default::default()
        },
        trah_trigger: if arm == Arm::Trah { Some(1e-2) } else { None },
        ..Default::default()
    };

    let jk0 = target_jk_builds();
    let t0 = Instant::now();
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).expect("scf");
    let secs = t0.elapsed().as_secs_f64();
    Row {
        iters: r.iterations,
        jk: target_jk_builds() - jk0,
        secs,
        energy: r.energy,
        converged: r.converged,
    }
}

#[test]
#[ignore = "measurement harness: run deliberately, ideally on a quiet box"]
fn diis_vs_aurora_vs_trah() {
    // Closed-shell RHF and RKS, spanning a size range. The RKS/PBE water row
    // is the case TRAH used to FAIL on (200 iters, stale dp_rms) and now
    // solves in 7 — it is in the list precisely because it is where the two
    // accelerators are expected to differ most.
    let systems: &[(&str, &str, &str, Option<&str>)] = &[
        (
            "water/cc-pVDZ   RHF ",
            "../../testdata/molecules/water.xyz",
            "cc-pvdz",
            None,
        ),
        (
            "water/cc-pVDZ   PBE ",
            "../../testdata/molecules/water.xyz",
            "cc-pvdz",
            Some("PBE"),
        ),
        (
            "benzene/STO-3G  RHF ",
            "../../testdata/molecules/benzene.xyz",
            "sto-3g",
            None,
        ),
        (
            "benzene/cc-pVDZ RHF ",
            "../../testdata/molecules/benzene.xyz",
            "cc-pvdz",
            None,
        ),
    ];

    println!(
        "\n{:<22}{:>8}{:>8}{:>8}{:>10}{:>8}{:>8}{:>10}{:>8}{:>8}{:>10}",
        "system", "it_D", "JK_D", "t_D", "it_A", "JK_A", "t_A", "it_T", "JK_T", "t_T", "same E?"
    );
    println!("{}", "-".repeat(118));

    for (name, xyz, bas, xc) in systems {
        let d = run(xyz, bas, *xc, Arm::Diis);
        let a = run(xyz, bas, *xc, Arm::Aurora);
        let t = run(xyz, bas, *xc, Arm::Trah);

        // An accelerator that changes the answer is broken, however fast.
        let da = (a.energy - d.energy).abs();
        let dt = (t.energy - d.energy).abs();
        let same = da < 1e-7 && dt < 1e-7;

        println!(
            "{name:<22}{:>8}{:>8}{:>8.2}{:>10}{:>8}{:>8.2}{:>10}{:>8}{:>8.2}{:>10}",
            d.iters,
            d.jk,
            d.secs,
            a.iters,
            a.jk,
            a.secs,
            t.iters,
            t.jk,
            t.secs,
            if same { "yes" } else { "NO" }
        );
        println!(
            "{:22}conv {}/{}/{}   dE(A)={da:.2e}  dE(T)={dt:.2e}",
            "", d.converged, a.converged, t.converged
        );

        assert!(
            d.converged,
            "{name}: the DIIS baseline must converge or the row means nothing"
        );
        assert!(
            da < 1e-7,
            "{name}: AURORA reached a DIFFERENT state ({:.3e} Ha from DIIS) -- \
             an accelerator changes the path, never the fixed point",
            da
        );
        assert!(
            dt < 1e-7,
            "{name}: TRAH reached a DIFFERENT state ({:.3e} Ha from DIIS) -- \
             an accelerator changes the path, never the fixed point",
            dt
        );
    }
    println!(
        "\nITERATIONS and J/K are load-immune and are the comparison; wall time \
         is only meaningful on a quiet box.\n"
    );
}
