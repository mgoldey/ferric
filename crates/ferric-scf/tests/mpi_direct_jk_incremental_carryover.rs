//! Cross-rank correctness of the ACCUMULATE-onto exact-ERI Fock builders.
//!
//! # The defect this pins
//!
//! `DirectJK::build`/`build_uhf` ACCUMULATE their result onto the caller's
//! `j`/`k` buffers (`*j += &total_j`) — they do not zero them. The incremental
//! Fock path exploits that: `DirectJK::build_incremental` passes `ΔD` into a
//! buffer that already holds the previous iteration's `J(D_last)`, and linearity
//! lands it on `J(D_new)`. `solve_rhf`/`solve_uhf` take that path by default
//! (`FERRIC_SCF_INCREMENTAL` unset) on every iteration except a periodic full
//! rebuild.
//!
//! The MPI reduction was applied to the OUTPUT buffer, after the accumulate:
//!
//! ```text
//! *j += &total_j;                    // rank-local contribution
//! Allreduce(j)                       // <-- sums the carry-over too
//! ```
//!
//! `J(D_last)` is the previous iteration's already-reduced GLOBAL matrix, so it
//! is identical on every rank. Allreducing the output therefore returns
//! `N·J(D_last) + Σ_r ΔJ_r` — the rank-local part reduces correctly, the
//! carry-over gets multiplied by the world size, every iteration. Measured
//! before the fix on water/STO-3G RHF: `-74.9631468000` Ha at `-np 1` vs
//! `+156.3238081949` Ha at `-np 2`.
//!
//! The fix reduces the rank-local partial BEFORE it touches the caller's buffer
//! (`reduce::reduce_partial_across_ranks`).
//!
//! # Why this file exists alongside `mpi_direct_jk_correctness.rs`
//!
//! That file DID go red at `-np 2` with the bug live, so it was not inert. But
//! it caught the defect through `assert!(res.converged)` — a convergence FLAG,
//! not an energy. That is the weak form of the guard: the same carry-over defect
//! elsewhere in this repo has been measured converging to a self-consistent but
//! badly wrong energy, which a `converged` assert passes. It also covered only
//! closed-shell `solve_rhf`, and only implicitly: nothing in it named the
//! incremental path, so a future change defaulting `FERRIC_SCF_INCREMENTAL=0`
//! would have silently removed all coverage of this bug while the file kept
//! passing.
//!
//! This file asserts ENERGIES against a serial anchor, on BOTH spin paths, and
//! pins the incremental and from-scratch routes SEPARATELY and by name.
//!
//! # Running it
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --features mpi \
//!     --test mpi_direct_jk_incremental_carryover --no-run
//! mpirun -np 2 --oversubscribe <bin> --nocapture --test-threads=1
//! ```
//!
//! ferric never calls `MPI_Finalize`, so `mpirun` exits non-zero even when every
//! test passes — assert on one `test result: ok.` per rank, never on `$?`.
//!
//! Without `--features mpi` (or at `-np 1`) `ParallelContext::world()` is `None`
//! and the reduction is a no-op, so this runs as an ordinary serial test and
//! guards the non-MPI path. It cannot observe a multi-rank defect there: the
//! cross-rank assertions below are only meaningful under `mpirun -np >= 2`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

/// Water, STO-3G — the geometry the bug was measured on.
fn water() -> Molecule {
    Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap()
}

/// Water cation doublet: the same cheap system on the open-shell
/// `DirectJK::build_uhf` / `build_uhf_incremental` code path, which carries its
/// own copy of the accumulate-then-reduce sequence (three buffers: J, K_α, K_β).
fn water_cation() -> Molecule {
    Molecule::parse_xyz("3\nH2O+\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 1, 2).unwrap()
}

/// `df_j_aux`/`df_k_aux` deliberately unset so `solve_rhf`/`solve_uhf` route the
/// per-iteration Fock build through `DirectJK` (the exact 4-index builder), not
/// the DF/RI path.
fn direct_config() -> RhfConfig {
    RhfConfig {
        df_j_aux: None,
        df_k_aux: None,
        energy_conv: 1e-10,
        density_conv: 1e-9,
        ..Default::default()
    }
}

/// Reduce `value` over the world and return `(min, max)`. `(v, v)` at one rank.
fn cross_rank_min_max(ctx: &ParallelContext, value: f64) -> (f64, f64) {
    #[cfg(feature = "mpi")]
    if let Some(world) = ctx.world() {
        use mpi::collective::SystemOperation;
        use mpi::traits::CommunicatorCollectives;
        let mut lo = 0.0f64;
        let mut hi = 0.0f64;
        world.all_reduce_into(
            std::slice::from_ref(&value),
            std::slice::from_mut(&mut lo),
            SystemOperation::min(),
        );
        world.all_reduce_into(
            std::slice::from_ref(&value),
            std::slice::from_mut(&mut hi),
            SystemOperation::max(),
        );
        return (lo, hi);
    }
    let _ = ctx;
    (value, value)
}

/// The assertion every case below shares.
///
/// Two independent checks, because they fail to different bugs:
///
/// 1. **Energy vs a serial anchor.** This is the load-bearing one. The
///    carry-over defect multiplies a rank-identical matrix by the world size, so
///    it corrupts every rank in exactly the SAME way — a cross-rank agreement
///    check alone cannot see it. Only comparison against the `-np 1` value can.
/// 2. **Cross-rank spread.** Catches a genuine partition defect (ranks holding
///    different matrices), which check 1 would only catch by luck.
///
/// Deliberately NOT asserted on: `ScfResult::converged`. The pre-fix `-np 2` run
/// happened to report `converged = false`, but that is luck, not a guard — the
/// same class of defect has been measured converging to a self-consistent wrong
/// energy elsewhere in this repo. The energy is the observable.
fn assert_matches_serial_anchor(ctx: &ParallelContext, label: &str, energy: f64, anchor: f64) {
    let (lo, hi) = cross_rank_min_max(ctx, energy);
    let spread = hi - lo;
    eprintln!(
        "[{label}] rank {}/{}: E = {energy:.15} Ha | anchor {anchor:.15} \
         | diff {:.3e} | cross-rank spread {spread:.3e}",
        ctx.rank,
        ctx.size,
        (energy - anchor).abs(),
    );

    // 1e-9 Ha: far below the ~1e2 Ha the carry-over defect produces at 2 ranks
    // and the O(N) scaling it has in the world size, and far ABOVE the incremental
    // path's f64 reassociation floor vs a from-scratch build (measured ~2e-13 on
    // this system). It is a correctness bar, not a precision claim.
    assert!(
        (energy - anchor).abs() < 1e-9,
        "[{label}] rank {}/{}: E = {energy:.15} Ha vs serial anchor {anchor:.15} Ha \
         (diff {:.3e}). A diff that grows with the rank count is the accumulate-then-\
         Allreduce carry-over defect: the reduction must act on the rank-local partial, \
         not on the caller's output buffer.",
        ctx.rank,
        ctx.size,
        (energy - anchor).abs(),
    );

    assert!(
        spread == 0.0,
        "[{label}] ranks disagree by {spread:.3e} Ha (min {lo:.15}, max {hi:.15}). \
         The DF path is bit-identical across ranks; the exact path must be too.",
    );
}

/// Serial RHF/STO-3G water anchor, measured at `-np 1` on this build.
const RHF_WATER_STO3G: f64 = -74.963_227_299_664_183;
/// Serial UHF/STO-3G water-cation doublet anchor, measured at `-np 1`.
const UHF_WATER_CATION_STO3G: f64 = -74.658_102_589_567_676;

fn run_rhf(ctx: &ParallelContext, incremental: bool) -> f64 {
    let mol = water();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    with_env(
        RHF_INCREMENTAL_KEY,
        if incremental { None } else { Some("0") },
        || {
            solve_rhf(ctx, &mol, &prep, op, &bounds, &direct_config())
                .unwrap()
                .energy
        },
    )
}

fn run_uhf(ctx: &ParallelContext, incremental: bool) -> f64 {
    let mol = water_cation();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    with_env(UHF_INCREMENTAL_KEY, incremental.then_some("1"), || {
        // NOT `.unwrap()`. A corrupted Fock drives the energy so far off that
        // UHF exhausts `max_iter` and `solve_uhf` returns `Err(ScfConvergence)`
        // carrying `last_energy` — measured `+4648.83` Ha under a deliberate
        // break of the reduction. Unwrapping there panics with a bare `Err`
        // message and throws the number away, so the failure reads as "SCF did
        // not converge" rather than naming the defect. Surface `last_energy` and
        // let the shared energy assertion below report it against the anchor.
        match solve_uhf(ctx, &mol, &prep, &bounds, &direct_config()) {
            Ok(r) => r.energy,
            Err(ferric_core::FerricError::ScfConvergence {
                iterations,
                last_energy,
            }) => {
                eprintln!(
                    "[uhf] solve_uhf did not converge in {iterations} iterations; \
                     reporting last_energy = {last_energy:.15} Ha for the anchor check"
                );
                last_energy
            }
            Err(e) => panic!("solve_uhf failed for an unrelated reason: {e}"),
        }
    })
}

/// Closed-shell incremental-Fock escape hatch. DEFAULT ON; `0` forces the
/// historical full-rebuild-every-iteration behaviour.
const RHF_INCREMENTAL_KEY: &str = "FERRIC_SCF_INCREMENTAL";

/// Open-shell incremental-Fock opt-in. DEFAULT **OFF** — the polarity is the
/// opposite of the closed-shell knob, and that asymmetry is a live trap: an
/// earlier draft of this file toggled `FERRIC_SCF_INCREMENTAL` for the UHF cases
/// too, which left `build_uhf_incremental` switched off in BOTH of them. A
/// deliberate break of the UHF reduction was then not caught by either test.
/// Mutation-tested: with this key set correctly, breaking the UHF site fails
/// `uhf_incremental_...` at -np 2.
const UHF_INCREMENTAL_KEY: &str = "FERRIC_SCF_UHF_INCREMENTAL";

/// Run `f` with `key` set to `value`, or removed when `value` is `None`,
/// restoring the previous setting afterwards.
///
/// Toggling these knobs is what separates "the incremental route is fixed" from
/// "the from-scratch route was never broken" — a single default-config run
/// cannot distinguish the two, because the from-scratch path zeroes the buffers
/// and is therefore accidentally immune to the carry-over defect (`N·0 == 0`).
///
/// `--test-threads=1` (required under mpirun anyway, for parseable output) makes
/// the process-global env mutation safe here.
fn with_env<T>(key: &str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
    let prev = std::env::var(key).ok();
    match value {
        Some(v) => std::env::set_var(key, v),
        None => std::env::remove_var(key),
    }
    let out = f();
    match prev {
        Some(v) => std::env::set_var(key, v),
        None => std::env::remove_var(key),
    }
    out
}

/// THE regression test: closed-shell, incremental path (`build_incremental` →
/// `DirectJK::build` with a non-zero `j`/`k`). This is the exact configuration
/// that produced `+156.3238081949` Ha at `-np 2` before the fix.
#[test]
fn rhf_incremental_direct_jk_matches_serial_across_ranks() {
    let ctx = ParallelContext::default();
    let e = run_rhf(&ctx, true);
    assert_matches_serial_anchor(&ctx, "rhf-incremental", e, RHF_WATER_STO3G);
}

/// The from-scratch route (`FERRIC_SCF_INCREMENTAL=0`), where the caller zeroes
/// the buffers. Expected to have been correct even before the fix — it is here so
/// that a regression can be ATTRIBUTED: if this goes red too, the defect is in
/// the rank partition or the reduction itself, not in the carry-over.
#[test]
fn rhf_from_scratch_direct_jk_matches_serial_across_ranks() {
    let ctx = ParallelContext::default();
    let e = run_rhf(&ctx, false);
    assert_matches_serial_anchor(&ctx, "rhf-from-scratch", e, RHF_WATER_STO3G);
}

/// Open-shell incremental path: `DirectJK::build_uhf_incremental` →
/// `build_uhf`, which carries its OWN accumulate-then-reduce sequence over three
/// buffers (J, K_α, K_β). It had the identical defect and no MPI test at all.
#[test]
fn uhf_incremental_direct_jk_matches_serial_across_ranks() {
    let ctx = ParallelContext::default();
    let e = run_uhf(&ctx, true);
    assert_matches_serial_anchor(&ctx, "uhf-incremental", e, UHF_WATER_CATION_STO3G);
}

/// Open-shell from-scratch route — the attribution control for the UHF path,
/// exactly as `rhf_from_scratch_...` is for the closed-shell one.
#[test]
fn uhf_from_scratch_direct_jk_matches_serial_across_ranks() {
    let ctx = ParallelContext::default();
    let e = run_uhf(&ctx, false);
    assert_matches_serial_anchor(&ctx, "uhf-from-scratch", e, UHF_WATER_CATION_STO3G);
}

/// The unit-level anchor, independent of any SCF loop.
///
/// Builds J/K from a FIXED density into a buffer that is deliberately PRE-LOADED
/// with a known non-zero matrix, exactly as the incremental path does, and checks
/// the result equals `preload + J(D)` — the accumulate contract — with the same
/// value at every rank count.
///
/// This is the trivial-limit anchor for the reduction: it does not depend on SCF
/// convergence, on DIIS, or on the periodic-full-rebuild schedule, so it cannot
/// be accidentally satisfied by a run that happened to reconverge. With the
/// pre-fix code it fails at `-np >= 2` by exactly `(N-1)·preload`.
#[test]
fn direct_jk_accumulate_contract_holds_across_ranks() {
    use ferric_scf::direct_jk::DirectJK;
    use ndarray::Array2;

    let ctx = ParallelContext::default();
    let mol = water();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let n = prep.nbasis();

    // Fixed deterministic symmetric density — same on every rank, no SCF needed.
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            d[(i, j)] = 0.05 * ((i * 7 + j * 3) % 11) as f64;
        }
    }
    let d = 0.5 * (&d + &d.t());

    // A distinctive, rank-identical preload standing in for `J(D_last)`/`K(D_last)`.
    // Non-zero everywhere so the defect cannot hide in a sparsity pattern.
    let mut preload_j = Array2::<f64>::zeros((n, n));
    let mut preload_k = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            preload_j[(i, j)] = 1.0 + (i as f64) - 0.5 * (j as f64);
            preload_k[(i, j)] = 2.0 - 0.25 * (i as f64) + (j as f64);
        }
    }

    // Pass 1: from a ZEROED buffer → the pure contribution J(D), K(D).
    let mut j_zero = Array2::<f64>::zeros((n, n));
    let mut k_zero = Array2::<f64>::zeros((n, n));
    let mut djk = DirectJK::new(&ctx, &prep, &bounds, 1e-14, 0);
    djk.build(&d, &mut j_zero, &mut k_zero).unwrap();

    // Pass 2: from the PRELOADED buffer → must equal preload + J(D), K(D).
    let mut j_pre = preload_j.clone();
    let mut k_pre = preload_k.clone();
    let mut djk2 = DirectJK::new(&ctx, &prep, &bounds, 1e-14, 0);
    djk2.build(&d, &mut j_pre, &mut k_pre).unwrap();

    let max_dev = |a: &Array2<f64>, b: &Array2<f64>, p: &Array2<f64>| -> f64 {
        a.iter()
            .zip(b.iter())
            .zip(p.iter())
            .fold(0.0f64, |m, ((x, y), z)| m.max((x - y - z).abs()))
    };
    let dev_j = max_dev(&j_pre, &j_zero, &preload_j);
    let dev_k = max_dev(&k_pre, &k_zero, &preload_k);

    eprintln!(
        "[accumulate-contract] rank {}/{}: max|J_pre - J(D) - preload_J| = {dev_j:.3e}, \
         K: {dev_k:.3e}",
        ctx.rank, ctx.size
    );

    // NOT asserted exact, and the reason matters. `build` does `*j += &total_j`,
    // so pass 2 computes `preload + total` while pass 1 computes `0 + total`;
    // subtracting `preload` back off is a different f64 operation sequence, and
    // adding `total` onto a preload of order 1e1 discards bits below its ulp.
    // Measured floor on this system: 4.44e-16 for J, 1.11e-16 for K — i.e. a few
    // ulp of max|preload|, which is exactly the rounding one should expect and
    // NOT evidence of a defect. Asserting 0.0 here was wrong and failed even at
    // -np 1.
    //
    // The bar is that floor, scaled generously. It is ~14 orders of magnitude
    // below what the defect produces: the pre-fix code deviates by exactly
    // `(world_size - 1) * preload`, i.e. O(1e1) at -np 2 and O(1e2) at -np 3.
    // The discriminating property is that this deviation must NOT grow with the
    // rank count, which is why the same constant is asserted at every -np.
    let floor = 1e-12;
    assert!(
        dev_j < floor,
        "[accumulate-contract] rank {}/{}: J violated `build`'s accumulate contract by \
         {dev_j:.3e} (bar {floor:.0e}). At -np N the pre-fix code deviates by exactly \
         (N-1)*preload — O(1e1) here — because the Allreduce summed the caller's \
         carry-over along with the rank-local contribution.",
        ctx.rank,
        ctx.size,
    );
    assert!(
        dev_k < floor,
        "[accumulate-contract] rank {}/{}: K violated `build`'s accumulate contract by \
         {dev_k:.3e} (bar {floor:.0e}) — see the J message.",
        ctx.rank,
        ctx.size,
    );

    // And the pure contribution itself must be rank-count independent.
    let trace_j: f64 = (0..n).map(|i| j_zero[(i, i)]).sum();
    let (lo, hi) = cross_rank_min_max(&ctx, trace_j);
    assert_eq!(
        hi - lo,
        0.0,
        "[accumulate-contract] ranks disagree on trace(J(D)): min {lo:.15}, max {hi:.15}",
    );
}

/// The same accumulate contract for the STANDALONE [`ferric_scf::rhf::build_jk`],
/// which `uhf_newton.rs` / `rohf_newton.rs` and external callers use directly.
///
/// # Why this needs its own test rather than riding on the existing one
///
/// `build_jk` has the identical accumulate-then-reduce shape the `DirectJK`
/// builders had, but it was LATENT rather than live: every in-tree caller zeroes
/// `j`/`k` first, and `N·0 == 0`, so the carry-over term vanishes. That was
/// confirmed by mutation — reverting `build_jk`'s reduction order leaves
/// `mpi_direct_jk_correctness.rs`'s `mpi_build_jk_standalone_...` case GREEN at
/// `-np 2`, because that case passes zeroed buffers.
///
/// "Correct only because today's callers happen to zero" is not a property of
/// the function, and `build_jk` is public. This pins the contract at the
/// function boundary where it belongs, by passing a pre-loaded buffer.
#[test]
fn build_jk_accumulate_contract_holds_across_ranks() {
    use ferric_scf::rhf::build_jk;
    use ndarray::Array2;

    let ctx = ParallelContext::default();
    let mol = water();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let n = prep.nbasis();

    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            d[(i, j)] = 0.05 * ((i * 7 + j * 3) % 11) as f64;
        }
    }
    let d = 0.5 * (&d + &d.t());

    let mut preload = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            preload[(i, j)] = 3.0 + (i as f64) - 0.5 * (j as f64);
        }
    }

    let mut j_zero = Array2::<f64>::zeros((n, n));
    let mut k_zero = Array2::<f64>::zeros((n, n));
    build_jk(&ctx, &prep, &bounds, 1e-14, &d, &mut j_zero, &mut k_zero).unwrap();

    let mut j_pre = preload.clone();
    let mut k_pre = preload.clone();
    build_jk(&ctx, &prep, &bounds, 1e-14, &d, &mut j_pre, &mut k_pre).unwrap();

    let max_dev = |a: &Array2<f64>, b: &Array2<f64>| -> f64 {
        a.iter()
            .zip(b.iter())
            .zip(preload.iter())
            .fold(0.0f64, |m, ((x, y), z)| m.max((x - y - z).abs()))
    };
    let dev_j = max_dev(&j_pre, &j_zero);
    let dev_k = max_dev(&k_pre, &k_zero);

    eprintln!(
        "[build_jk-contract] rank {}/{}: J dev {dev_j:.3e}, K dev {dev_k:.3e}",
        ctx.rank, ctx.size
    );

    // Same f64 reasoning as `direct_jk_accumulate_contract_holds_across_ranks`:
    // a few ulp of max|preload| is the expected floor, and the defect signature
    // is O((N-1)*preload) — here O(1e1) at -np 2 — which is ~13 orders above it.
    let floor = 1e-12;
    assert!(
        dev_j < floor,
        "[build_jk-contract] rank {}/{}: J violated build_jk's accumulate contract by \
         {dev_j:.3e} (bar {floor:.0e}) — the Allreduce summed the caller's carry-over.",
        ctx.rank,
        ctx.size,
    );
    assert!(
        dev_k < floor,
        "[build_jk-contract] rank {}/{}: K violated build_jk's accumulate contract by \
         {dev_k:.3e} (bar {floor:.0e}).",
        ctx.rank,
        ctx.size,
    );
}
