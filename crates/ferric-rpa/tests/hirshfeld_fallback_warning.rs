//! The Hirshfeld proatom-fallback warning must describe the case it is in.
//!
//! One message covered two different situations:
//!
//!   SOME atoms fall back  -> the partitioning MIXES SCF and crude proatoms,
//!                            so it is internally inconsistent;
//!   ALL atoms fall back   -> there is no mixture. It is uniformly crude,
//!                            which is what you get with no proatom provider
//!                            (e.g. `tools/active_site/binding_energy.py`).
//!
//! The single message said "The other atoms used the SCF density, so these
//! charges mix two different proatom sources". In the all-fallback case there
//! ARE no other atoms and nothing is mixed, so it sent a reader looking for an
//! inconsistency that was not there while under-stating the real problem --
//! every charge came from the crude model.
//!
//! OBSERVED 2026-09-20 running `compute_binding_energy` on danuglipron in the
//! 7LCJ pocket: "71 of 71 atoms ... The other atoms used the SCF density".
//!
//! The warning goes to stderr, so this drives it through a SUBPROCESS and
//! reads what was actually printed. Asserting on a string the test itself
//! formats would pin the format and never touch the branch.

/// The real check, in-process: call the library both ways and assert the
/// CHARGES differ, which is what proves the two branches are distinct code
/// paths rather than one message with a reworded string.
#[test]
fn all_fallback_and_mixed_are_different_code_paths() {
    use ferric_core::{basis, mol::Molecule};
    use ferric_rpa::properties::{hirshfeld_charges, spherically_averaged_proatom};

    let xyz = "3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.755 -0.471\nH 0.000 -0.755 -0.471\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = ferric_integrals::basis_bridge::PreparedBasis::new(&mol, &bs).unwrap();
    let nbf = prep.nbasis();
    let dens = ndarray::Array2::<f64>::eye(nbf) * 0.2;

    // (a) no provider at all -> EVERY atom falls back.
    let q_none = hirshfeld_charges(&mol, &bs, &dens, None).unwrap();

    // (b) a provider that answers for oxygen only -> hydrogens fall back.
    let radii = [0.0f64, 0.25, 0.5, 1.0, 2.0, 4.0];
    let o_mol = Molecule::parse_xyz("1\nO\nO 0 0 0\n", 0, 3).unwrap();
    let o_prep = ferric_integrals::basis_bridge::PreparedBasis::new(&o_mol, &bs).unwrap();
    let o_dens = ndarray::Array2::<f64>::eye(o_prep.nbasis()) * 0.5;
    let partial = |z: i32, _q: i32| -> Option<ferric_rpa::properties::RadialProatom> {
        if z == 1 {
            None
        } else {
            spherically_averaged_proatom(z, &bs, &o_dens, &radii).ok()
        }
    };
    let pp: &ferric_rpa::properties::ProatomProvider = &partial;
    let q_mixed = hirshfeld_charges(&mol, &bs, &dens, Some(pp)).unwrap();

    assert_eq!(q_none.len(), 3);
    assert_eq!(q_mixed.len(), 3);

    // Both must conserve charge -- the partitioning is renormalised, so this
    // holds regardless of which proatom source was used. If it ever does not,
    // the branch split broke something real.
    let s_none: f64 = q_none.iter().sum();
    let s_mixed: f64 = q_mixed.iter().sum();
    assert!(
        s_none.abs() < 1e-6,
        "all-fallback charges do not sum to zero: {s_none:.3e} ({q_none:?})"
    );
    assert!(
        s_mixed.abs() < 1e-6,
        "mixed charges do not sum to zero: {s_mixed:.3e} ({q_mixed:?})"
    );

    // And they must actually DIFFER, or the "mixed" case never happened and
    // this test is asserting the same path twice.
    let diff = q_none
        .iter()
        .zip(&q_mixed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        diff > 1e-9,
        "the all-fallback and mixed proatom paths gave identical charges \
         (max diff {diff:.3e}), so the partial provider was never consulted \
         and this test cannot distinguish the two branches"
    );
}

/// The WORDING, read from a real subprocess.
///
/// The charge-value test above proves the two paths are distinct CODE. It does
/// not read a single character of the warning -- mutation-verified, merging
/// the two messages back into one passed it unchanged, which is the entire
/// defect this file exists for.
///
/// `eprintln!` cannot be captured in-process, so this runs
/// `examples/hirshfeld_warning_probe` and asserts each message separately.
/// NOTE `cargo test --examples` does NOT link an example binary -- it compiles
/// examples as test targets hunting for `#[test]` and leaves .rmeta behind --
/// so the probe must be built with `cargo build --example`. CI's `build-tests`
/// job does that for `runlog_probe` already; here the test SKIPS with an
/// actionable message rather than failing when the binary is absent, because a
/// missing build artefact is not a defect in the code under test.
#[test]
fn the_two_warnings_say_different_things() {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    // Resolved at RUN TIME: a sibling of this test binary, so it survives
    // `nextest archive` extracting to a different directory.
    let probe: Option<PathBuf> = std::env::current_exe().ok().and_then(|exe| {
        let dir = exe.parent().and_then(Path::parent)?;
        let p = dir.join("examples").join("hirshfeld_warning_probe");
        p.is_file().then_some(p)
    });
    let Some(probe) = probe else {
        eprintln!(
            "SKIP: examples/hirshfeld_warning_probe not built. \
             Run `cargo build -p ferric-rpa --example hirshfeld_warning_probe` \
             (`cargo test --examples` does not link it)."
        );
        return;
    };

    let stderr_of = |arg: &str| -> String {
        let out = Command::new(&probe)
            .arg(arg)
            .env("OPENBLAS_NUM_THREADS", "1")
            .output()
            .expect("run probe");
        assert!(out.status.success(), "probe {arg} failed: {out:?}");
        String::from_utf8_lossy(&out.stderr).into_owned()
    };

    let all = stderr_of("all");
    let mixed = stderr_of("mixed");

    // The all-fallback case must NOT claim a mixture. This is the exact
    // sentence that was wrong: with every atom on the crude model there are
    // no "other atoms" and nothing is mixed.
    assert!(
        all.contains("ALL 3 atoms"),
        "all-fallback message does not say ALL: {all}"
    );
    assert!(
        all.contains("CONSISTENT") && all.contains("qualitative"),
        "all-fallback message must say the partitioning is consistent but \
         qualitative: {all}"
    );
    assert!(
        !all.contains("The other atoms used the SCF density"),
        "the all-fallback message claims a mixture that cannot exist: {all}"
    );
    assert!(
        !all.contains("NOT uniformly converged"),
        "all-fallback is uniformly crude, not unevenly converged: {all}"
    );

    // The mixed case must still carry the mixture warning AND the indices.
    assert!(
        mixed.contains("2 of 3 atoms") && mixed.contains("[1, 2]"),
        "mixed message must name the count and the offending indices: {mixed}"
    );
    assert!(
        mixed.contains("mix two different proatom sources"),
        "mixed message must say the sources are mixed: {mixed}"
    );

    // And they must be DIFFERENT text, which is what re-merging breaks.
    assert_ne!(
        all.trim(),
        mixed.trim(),
        "the two cases emit identical warnings -- they were merged back"
    );
}
