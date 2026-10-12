//! Schwarz upper-bound screening matrix for shell-pair integrals.

use crate::basis_bridge::PreparedBasis;
use crate::engine::Engine;
use crate::operator::{Operator, OperatorKind};
use ferric_core::FerricError;
use ndarray::Array2;

/// Below this many shells, run the single-engine serial FFI loop — avoids
/// rayon/engine-construction overhead for free-atom/tiny-basis jobs (same
/// rationale as [`PAR_AUX_SHELL_THRESHOLD`] below and `oneelectron.rs`).
const PAR_SCHWARZ_SHELL_THRESHOLD: usize = 64;

/// Engine precision used to build the SCREENING TABLE, as distinct from the
/// threshold the table is later used to enforce.
///
/// A Schwarz bound's whole contract is that it never UNDERestimates: `Q(ij) *
/// Q(kl) >= |(ij|kl)|` is what makes discarding a quartet safe. libint2 applies
/// its own internal prescreening at the precision it is constructed with, and
/// declines to compute a shell quartet whose result would fall below it —
/// correct, documented behaviour for an integral engine, but fatal for a table
/// builder, because [`Engine::compute_quartet`] then returns `None` and the
/// pair is recorded as `Q = 0.0`. A stored zero on a pair with a nonzero true
/// `(ij|ij)` underestimates by construction, and since `SignificantPairs::build`
/// compares with a strict `estimate(..) > threshold`, such a pair becomes
/// unreachable at EVERY threshold — including 0, which destroys the trivial
/// limit in which screening must do nothing at all.
///
/// MEASURED before this constant existed (alkane_8/cc-pVDZ, 102 shells):
/// 1049 of 5253 unique pairs stored `Q == 0.0` while only 121 are genuinely
/// zero, and the smallest nonzero stored Q was 4.632e-7 — i.e. exactly
/// `sqrt(~1e-13)`, the fingerprint of the engine's precision cliff rather than
/// of any property of the molecule. alkane_16 zeroed 9977 of 19701.
///
/// The remedy is what every production code does: build the table TIGHTER than
/// the threshold it enforces. PySCF builds `q_cond` at `direct_scf_tol**2` with
/// a `1e-100` floor; Molpro, NWChem, Q-Chem, ORCA and Psi4 (via its own
/// `eps * thresh`) do the equivalent. ferric previously used one hardcoded
/// `1e-14` for BOTH roles, which is the outlier.
///
/// `0.0` disables libint2's internal prescreening outright, which is the
/// tightest available table. MEASURED one-time cost, both arms run back to back
/// on an otherwise-idle box (release build, best of five,
/// OPENBLAS_NUM_THREADS=1) so the comparison is like for like:
///
/// ```text
///                    1e-14      0.0      ratio
/// alkane_8/cc-pVDZ   0.0262 s   0.0412 s  1.6x
/// alkane_16/cc-pVDZ  0.0452 s   0.1328 s  2.9x
/// ```
///
/// That is a setup cost paid once per Fock-builder construction, not per SCF
/// iteration, and it buys an alkane_8 RHF energy that moves from 1.46e-4 Ha off
/// PySCF to 6.96e-9 Ha off — tens of milliseconds against five orders of
/// magnitude of accuracy. The ratio grows with system size (the zeroed fraction
/// did too: 20% of pairs on alkane_8, 51% on alkane_16), so it is worth
/// re-measuring if it ever shows up in a profile at much larger N.
///
/// A cheaper `thresh * thresh` table (PySCF's actual policy) would also restore
/// the invariant and would scale better; `0.0` is chosen because the measured
/// cost is small enough that the extra knob is not worth its failure modes —
/// notably that it re-couples the table to a threshold the caller can change.
///
/// `pub(crate)` (not private) so [`crate::csb`] builds its `M` table at the
/// IDENTICAL precision rather than carrying a second copy of the literal that
/// could silently drift. The underestimation hazard this constant exists to
/// defeat applies verbatim to the `(PP|QQ)` table, and the remedy must not be
/// allowed to diverge between the two.
pub(crate) const SCHWARZ_TABLE_PRECISION: f64 = 0.0;

/// Floor applied to every stored Q so that no table entry is ever exactly zero.
///
/// Even with prescreening disabled a genuinely vanishing `(ij|ij)` (or one that
/// underflows to zero in double precision) would still be stored as `0.0` and
/// hit the same strict-`>` unreachability. Flooring costs nothing — `1e-100`
/// squared is `1e-200`, still finite and ~180 orders of magnitude below any
/// threshold anyone would enforce, so it never makes a negligible quartet
/// survive a real screen — but it makes `estimate(..) > 0.0` true everywhere,
/// which is precisely the trivial-limit guarantee. Same value and same
/// rationale as PySCF's `q_cond` floor.
///
/// `pub(crate)` (not private) so [`crate::csb`] floors its `M` table with the
/// IDENTICAL value rather than a second, silently-drifting copy. A stored
/// `M = 0.0` is exactly as fatal there as a stored `Q = 0.0` is here — worse,
/// in fact, because CSB takes a `min` and a zero `M` would win that `min` at
/// every quartet touching the affected shell.
pub(crate) const SCHWARZ_Q_FLOOR: f64 = 1e-100;

/// Relative inflation applied to every terfc Schwarz factor `Q` (and to the
/// aux `sqrt(P|P)` factor in the integral-direct triple cut).
///
/// EMPIRICAL, not derived. Schwarz is rigorous for the exact terfc kernel
/// (positive definite), but the engine is table-interpolated and a diagonal
/// element computed low by a relative delta lets a bound undershoot by about
/// delta/2 per factor. The table error does not translate into a clean bound on
/// that delta: the engine forms Coulomb - terf, so a diagonal's relative error
/// is the table error times Coulomb/terfc (large for diffuse functions at small
/// r0), and the per-element table error itself reaches 2.2e-5 relative at
/// m = 10 (`tests/terfc_interp_sweep.rs`, both sweeps; between-node worst
/// ABSOLUTE error in G 5.4e-10 at m = 12, 4.4e-12 for m <= 7). What supports
/// 1e-5 is the end-to-end check, not that arithmetic: over the validated
/// parameter box and bases (`TERFC_SCHWARZ_*`, `tests/terfc_schwarz_screening.rs`)
/// the smallest bound/true over all significant triples is 1.00003 WITH the
/// inflation, i.e. the bound would still hold without it. A bound that
/// undershoots by a relative delta is equivalent to screening at
/// threshold*(1 + delta), so a residual miss at this level changes which
/// triples sit exactly at the threshold and nothing else.
pub const TERFC_TABLE_Q_INFLATION: f64 = 1e-5;

/// Total angular momentum at which the terfc 4-center table engine refuses a
/// quartet (`TERFC_DIMM` in the shim; pinned by a unit test).
const TERFC_QUARTET_L_LIMIT: i32 = 24;

/// Largest orbital angular momentum for which the terfc bound is validated
/// (g, `tests/terfc_schwarz_screening.rs`, water/def2-QZVP). The engine itself
/// would accept up to 4*l < 24, i.e. h shells.
pub const TERFC_SCHWARZ_MAX_L: i32 = 4;

/// Validated terfc parameter box: r0 in [0.3, 2.5] Angstrom and
/// r0*omega in (0, 8.9] (the shipped tables end at s = 80). Outside it the
/// bound has not been checked against true integrals and is refused.
pub const TERFC_SCHWARZ_R0_ANG: (f64, f64) = (0.3, 2.5);
pub const TERFC_SCHWARZ_MAX_R0_OMEGA: f64 = 8.9;

/// The operator-parameter half of the terfc validated box (no basis needed),
/// so front ends can refuse at load time.
pub fn check_terfc_parameters_validated(op: Operator) -> Result<(), FerricError> {
    use ferric_core::units::ANGSTROM_TO_BOHR;
    let (lo, hi) = TERFC_SCHWARZ_R0_ANG;
    let r0_ang = op.distance / ANGSTROM_TO_BOHR;
    let s = op.distance * op.omega;
    // The 1e-9 slack absorbs the unit round trip of the edge values.
    if !(r0_ang >= lo * (1.0 - 1e-9) && r0_ang <= hi * (1.0 + 1e-9))
        || !(s > 0.0 && s <= TERFC_SCHWARZ_MAX_R0_OMEGA * (1.0 + 1e-9))
    {
        return Err(FerricError::Libint(format!(
            "terfc Schwarz bound: r0 = {r0_ang:.4} A, r0*omega = {s:.4} is outside the validated \
             range (r0 in [{lo}, {hi}] A, 0 < r0*omega <= {TERFC_SCHWARZ_MAX_R0_OMEGA}). \
             Set schwarz_skip = 0.0"
        )));
    }
    Ok(())
}

fn check_terfc_validated_range(op: Operator, prep: &PreparedBasis) -> Result<(), FerricError> {
    if prep.max_l() > TERFC_SCHWARZ_MAX_L || 4 * prep.max_l() >= TERFC_QUARTET_L_LIMIT {
        return Err(FerricError::Libint(format!(
            "terfc Schwarz bound: orbital basis has max angular momentum {}; validated up to {} \
             (the engine refuses quartets with total L >= {}). Set schwarz_skip = 0.0",
            prep.max_l(),
            TERFC_SCHWARZ_MAX_L,
            TERFC_QUARTET_L_LIMIT
        )));
    }
    check_terfc_parameters_validated(op)
}

/// Multiplier applied to Schwarz factors for `op`: exactly `1.0` for every
/// operator without table error, `1 + TERFC_TABLE_Q_INFLATION` for terfc.
pub fn bound_inflation(op: Operator) -> f64 {
    match op.kind {
        OperatorKind::Terfc => 1.0 + TERFC_TABLE_Q_INFLATION,
        _ => 1.0,
    }
}

/// Q(i,j) = sqrt(max_{a,b} |(ab|ab)|) over the functions of shell pair (i,j),
/// from one computed (ij|ij) quartet block, floored at [`SCHWARZ_Q_FLOOR`] so
/// that the returned bound is never exactly zero.
///
/// Two different "no block" outcomes, handled differently:
///  * `Ok(None)`: the engine screened the block to zero (libint2 prescreening;
///    for terfc, an exactly-zero block, since the table 4-center path has no
///    numerical primitive cutoff). The stored value is the floor.
///  * `Err(status)`: a table engine (terf/terfc) REFUSED the quartet (total L
///    beyond the table depth, null tables, a caught C++ exception). Nothing is
///    known about the block, so storing the floor would silently skip every
///    triple on this pair; it is an error instead.
fn schwarz_pair(
    eng: &mut Engine,
    prep: &PreparedBasis,
    i: usize,
    j: usize,
    scale: f64,
) -> Result<f64, FerricError> {
    let dims = prep.shell_dims();
    let (n1, n2) = (dims[i], dims[j]);
    let maxv = match eng.compute_quartet_checked(prep, i, j, i, j) {
        Ok(Some(block)) => {
            // The (ab|ab) magnitudes live on the generalized diagonal of the
            // (n1·n2)×(n1·n2) block: index ((a·n2+b)·n1+a)·n2+b.
            let mut m = 0.0f64;
            for a in 0..n1 {
                for b in 0..n2 {
                    let v = block[((a * n2 + b) * n1 + a) * n2 + b].abs();
                    if v > m {
                        m = v;
                    }
                }
            }
            m
        }
        Ok(None) => 0.0,
        Err(status) => {
            return Err(FerricError::Libint(format!(
                "Schwarz bound: the table engine refused shell quartet ({i}{j}|{i}{j}) \
                 with status {status}; the pair cannot be bounded"
            )))
        }
    };
    // `scale` is exactly 1.0 for every table-free operator (x * 1.0 is
    // bit-identical); see `bound_inflation` for the terfc value.
    Ok((maxv.sqrt() * scale).max(SCHWARZ_Q_FLOOR))
}

/// Compute the Schwarz screening matrix Q(i,j) = sqrt(|(ij|ij)|) for all shell pairs.
///
/// Q(i,j) * Q(k,l) provides an upper bound on |(ij|kl)|, enabling integral screening.
///
/// Parallelized over upper-triangle shell pairs once `nsh` clears
/// `PAR_SCHWARZ_SHELL_THRESHOLD` — the historical `scf_compute_schwarz` FFI
/// call ran the whole O(nsh²) diagonal-quartet loop serially on one engine,
/// which dominated setup on large direct jobs. Each rayon worker builds its own
/// [`Engine`] via `map_init` (construction is serialized behind a global ctor
/// mutex — see the ROW_BLOCK note in the body for why granularity matters).
/// Every pair is computed by the same
/// `scf_compute_eri_quartet` kernel with unit coefficient as the serial loop,
/// and each (i, j) is written exactly once, so the result is bit-identical to
/// the serial loop regardless of thread count or block size.
///
/// # Why the operator set is `{Coulomb, ErfCoulomb, ErfcCoulomb, Terfc}`
///
/// Two independent gates, and it is worth knowing which one a given operator
/// fails, because they have different remedies:
///
/// 1. **Positive-definiteness.** Schwarz requires `G` to be a positive-definite
///    function (Thompson & Ochsenfeld, JCP 147, 144101 (2017), Appendix B;
///    Bochner's theorem on `F_G(k) > 0`). Their Table VI discharges this for
///    all three accepted kinds, and also for `Yukawa`, `SlaterGeminal` and the
///    Gaussian geminal. See [`crate::csb`]'s header, which extends the same
///    recipe to ferric's own kernels and finds `Terfc` positive-definite (with
///    a closed form, `F(k) = (1 - cos(k r0) e^{-k²/4ω²})/(2π²k²)`) but `Terf`
///    NOT — `F_terf(k) = cos(k r0) e^{-k²/4ω²}/(2π²k²)` changes sign, so a
///    "Schwarz bound" for `Terf` could be violated outright. No engine work
///    would ever make `Terf` admissible here.
/// 2. **An available 4-centre engine.** `Engine::new_2e` must be able to build
///    the operator at all, and this pass needs `(ij|ij)` QUARTETS. `Terfc` is
///    served by the table engine's `scf_compute_terfc_eri4`; because that
///    engine is interpolated, its factors carry [`bound_inflation`] and a
///    basis whose quartets exceed the table depth is refused up front.
pub fn schwarz(op: Operator, prep: &PreparedBasis) -> Result<Array2<f64>, FerricError> {
    match op.kind {
        OperatorKind::Coulomb | OperatorKind::ErfCoulomb | OperatorKind::ErfcCoulomb => {}
        // INVALID BY CONSTRUCTION, not merely unimplemented. Schwarz screening
        // IS the Cauchy-Schwarz inequality, which needs the two-electron form
        // to be an inner product — i.e. the kernel positive-definite. terf is
        // not: its Fourier transform
        //     F_terf(k) = cos(k*r0) * exp(-k^2/4w^2) / (2 pi^2 k^2)
        // changes sign at k*r0 = pi (Bochner), so the form is indefinite,
        // `(ij|ij)` can be NEGATIVE, and `Q = sqrt((ij|ij))` is not even real.
        //
        // MEASURED, not merely derived — water/cc-pVDZ 2-center diagonals:
        //   r0=2: min (P|P)_terf = -7.543e-3    r0=4: -4.439e-4
        // with plain erf (same omegas) and terfc both staying >= 0 on the same
        // basis and engine. See tests/terf_positivity_probe.rs. The cause is
        // the FINITE SHELL RADIUS r0 (the cos factor), NOT long-rangedness:
        // plain erf is long-range and positive-definite.
        //
        // No tightening will fix this and no future implementation should
        // "add terf support" here. Long-range terf work relies on its own lack
        // of sparsity instead (the integrals approach 1/R and do not decay, so
        // there is little for a distance screen to discard) plus RI/DF, which
        // is what production RSH codes do for the long-range piece.
        OperatorKind::Terf => {
            return Err(FerricError::Libint(
                "Schwarz screening is INVALID for the Terf operator, not merely unimplemented: \
                 terf's kernel is not positive-definite (its Fourier transform cos(k*r0) \
                 exp(-k^2/4w^2)/(2 pi^2 k^2) changes sign at k*r0 = pi), so the two-electron \
                 form is indefinite, (ij|ij) can be negative, and Q = sqrt((ij|ij)) is not real. \
                 Measured: min (P|P)_terf = -7.543e-3 on water/cc-pVDZ at r0=2. Use RI/DF for \
                 the long-range piece; terf integrals do not decay with separation, so there is \
                 little for a distance screen to discard in any case."
                    .to_string(),
            ))
        }
        // Positive-definite (proven: F_terfc(k) = (1 - cos(k*r0) exp(-k^2/4w^2))
        // / (2 pi^2 k^2) >= (1 - exp(-k^2/4w^2)) / (2 pi^2 k^2) > 0 for all
        // (r0, omega); checked against radial quadrature, and r0 -> 0
        // reproduces the published erfc transform). The 4-center table engine
        // (`scf_compute_terfc_eri4`) supplies the (ij|ij) diagonals, so plain
        // Schwarz is rigorous for the exact kernel. The engine is INTERPOLATED:
        // a diagonal computed slightly low would break the bound by that much,
        // so every terfc Q is inflated by [`bound_inflation`] (an empirical
        // margin, see [`TERFC_TABLE_Q_INFLATION`]). CSB stays refused (`csb.rs`): its M table is
        // not built for terfc.
        //
        // VALIDATED RANGE (`check_terfc_validated_range`, tests/
        // terfc_schwarz_screening.rs): the bound never undershot a true (P|ab)
        // over r0 in {0.3, 0.4, 0.5, 1.0, 1.5, 2.5} A x r0*omega in {0.35,
        // 0.7071, 1.354, 2, 4, 8.9} (8.9 = the table edge), on
        // methane/cc-pVDZ, water/aug-cc-pVDZ, water/aug-cc-pVTZ (f),
        // water/def2-TZVP and water/def2-QZVP (g) with the def2-universal-jkfit
        // aux; no quartet bound undershot on water/cc-pVDZ. Smallest
        // bound/true over the grids: 1.00003. Outside that box (h shells,
        // r0 outside [0.3, 2.5] A, r0*omega > 8.9) the call is refused.
        // The Coulomb diagonal is NOT a substitute: terfc <= Coulomb in r but
        // not in Fourier space (1 - cos(k r0) e^{-k^2/4w^2} > 1 where cos < 0),
        // and it undershoots by up to 26%.
        //
        // Separately, the 4-center path refuses a quartet whose total angular
        // momentum reaches the table depth (TERFC_DIMM = 24); schwarz_pair
        // reports any such refusal as an error, never as a stored floor.
        OperatorKind::Terfc => check_terfc_validated_range(op, prep)?,
        _ => {
            return Err(FerricError::Libint(format!(
                "operator {:?} not implemented",
                op.kind
            )))
        }
    }
    let nsh = prep.nshells();
    let scale = bound_inflation(op);
    let mut qmat = Array2::zeros((nsh, nsh));

    if nsh < PAR_SCHWARZ_SHELL_THRESHOLD {
        let mut eng = Engine::new_2e(op, prep, SCHWARZ_TABLE_PRECISION)?;
        for i in 0..nsh {
            for j in 0..=i {
                let q = schwarz_pair(&mut eng, prep, i, j, scale)?;
                qmat[(i, j)] = q;
                qmat[(j, i)] = q;
            }
        }
        return Ok(qmat);
    }

    use rayon::prelude::*;

    // Validate engine construction once up front so worker-side construction
    // can't fail (mirrors schwarz3_aux below).
    Engine::new_2e(op, prep, SCHWARZ_TABLE_PRECISION)?;

    // Parallelize over ROW BLOCKS, not individual shell pairs. libint2 engine
    // construction is expensive AND serialized behind a global ctor mutex in the
    // shim, and rayon invokes a `map_init` closure once per work-CHUNK, not once
    // per thread. The previous version made each of the nsh(nsh+1)/2 shell pairs
    // (31,878 at benzene/aug-cc-pVTZ) its own work item, so dozens of engine
    // constructions queued on that mutex: MEASURED 123 ms at RAYON=1 vs 3211 ms
    // at RAYON=12 — parallelism made this function 26x SLOWER. Row blocks keep
    // the item count at ceil(nsh/ROW_BLOCK) (8 here) so the init closure fires
    // about once per thread, while each item still carries real integral work.
    // Sweep at 252 shells / RAYON=12: block 8 -> 211 ms, 16 -> 144, 32 -> 75,
    // 64 -> 86; serial baseline 124 ms.
    //
    // NOT an `EnginePool` (engine_pool.rs), despite that module existing for
    // exactly this ctor-mutex pathology: it eagerly builds `nthreads + 1`
    // engines, which costs MORE than this whole loop when there are fewer
    // blocks than threads (measured: 125.7 ms of pool construction, 190 ms
    // total, vs 75 ms here). The pool wins where engines are reused across many
    // chunks and many SCF iterations (the direct Fock builders); a one-shot
    // setup loop with ~nthreads items is the opposite regime.
    const ROW_BLOCK: usize = 32;
    let row_blocks: Vec<(usize, usize)> = (0..nsh)
        .step_by(ROW_BLOCK)
        .map(|i0| (i0, (i0 + ROW_BLOCK).min(nsh)))
        .collect();
    // Each block owns its rows exclusively, so the (i, j) writes never overlap;
    // collect per-block triangles and scatter serially to keep `qmat` unshared.
    let blocks: Vec<Result<Vec<(usize, usize, f64)>, FerricError>> = row_blocks
        .par_iter()
        .map_init(
            || {
                Engine::new_2e(op, prep, SCHWARZ_TABLE_PRECISION)
                    .expect("2e engine (pre-validated)")
            },
            |eng, &(i0, i1)| {
                let mut out = Vec::with_capacity((i1 - i0) * (i1 + 1));
                for i in i0..i1 {
                    for j in 0..=i {
                        out.push((i, j, schwarz_pair(eng, prep, i, j, scale)?));
                    }
                }
                Ok(out)
            },
        )
        .collect();
    for block in blocks {
        for (i, j, q) in block? {
            qmat[(i, j)] = q;
            qmat[(j, i)] = q;
        }
    }
    Ok(qmat)
}

/// Below this many aux shells, run the loop serially — avoids
/// rayon/engine-construction overhead for free-atom/tiny-basis jobs.
const PAR_AUX_SHELL_THRESHOLD: usize = 64;

/// Per-shell Schwarz bound for an auxiliary (density-fitting) basis:
/// `Q3[P] = sqrt(max_a |(P_a | P_a)|)` over the functions a in aux shell P.
///
/// Combined with the orbital-pair matrix Q(μ,ν) = sqrt(|(μν|μν)|), this gives
/// the rigorous 3-index Cauchy–Schwarz bound
///   `|(P | μν)|  ≤  Q3[P] · Q(μ,ν)`
/// which lets `eri3_tensor_screened` skip shell triples whose contribution
/// is below threshold without computing them.
///
/// Built at `SCHWARZ_TABLE_PRECISION` and floored at `SCHWARZ_Q_FLOOR` (both
/// crate-private, hence code spans rather than doc links) for
/// the same reason as the orbital-pair table above: this is a screening bound,
/// so it must never underestimate. (Note `compute_eri2` returns an EMPTY slice
/// rather than an `Option` when libint2 prescreens a block away, so that case
/// would index out of bounds and PANIC rather than silently store a zero —
/// disabling prescreening removes that path too, though the floor is what
/// guarantees the invariant for a genuinely vanishing diagonal.)
///
/// In practice this arm was never the one biting — MEASURED on
/// alkane_8 and alkane_16 with cc-pVDZ-RI, the pre-fix aux table stored ZERO
/// zeros and its smallest nonzero entry was 4.475e-1, because a 2-centre
/// `(P|P)` diagonal over a normalized aux shell is O(1) and nowhere near the
/// engine's precision cliff. It is changed for invariant consistency rather
/// than to repair an observed failure, and the measured cost is 0.004 s.
///
/// Parallelized over `p` once `nsh` clears `PAR_AUX_SHELL_THRESHOLD`: each
/// rayon worker builds its own `Engine` via `for_each_init` (never per-item —
/// construction runs under a global ctor mutex). Each iteration writes only
/// `q3[p]` — a single distinct index per task, so the write set is trivially
/// disjoint across workers; `into_par_iter().map().collect()` is
/// order-preserving, giving a `Vec` bit-identical to the serial loop.
pub fn schwarz3_aux(op: Operator, dfbs: &PreparedBasis) -> Result<Vec<f64>, FerricError> {
    let nsh = dfbs.nshells();
    let dims = dfbs.shell_dims();
    let scale = bound_inflation(op);

    if nsh < PAR_AUX_SHELL_THRESHOLD {
        let mut eng = Engine::new_2center(op, dfbs, SCHWARZ_TABLE_PRECISION)?;
        let mut q3 = vec![0.0f64; nsh];
        for p in 0..nsh {
            let block = eng.compute_eri2(dfbs, p, p);
            let np = dims[p];
            // (P_a | P_a) lives on the diagonal of the np×np block.
            let mut maxv = 0.0f64;
            for a in 0..np {
                let v = block[a * np + a].abs();
                if v > maxv {
                    maxv = v;
                }
            }
            q3[p] = (maxv.sqrt() * scale).max(SCHWARZ_Q_FLOOR);
        }
        return Ok(q3);
    }

    use rayon::prelude::*;
    Engine::new_2center(op, dfbs, SCHWARZ_TABLE_PRECISION)?;
    let q3: Vec<f64> = (0..nsh)
        .into_par_iter()
        .map_init(
            || {
                Engine::new_2center(op, dfbs, SCHWARZ_TABLE_PRECISION)
                    .expect("2-center engine (pre-validated)")
            },
            |eng, p| {
                let block = eng.compute_eri2(dfbs, p, p);
                let np = dims[p];
                let mut maxv = 0.0f64;
                for a in 0..np {
                    let v = block[a * np + a].abs();
                    if v > maxv {
                        maxv = v;
                    }
                }
                (maxv.sqrt() * scale).max(SCHWARZ_Q_FLOOR)
            },
        )
        .collect();
    Ok(q3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_bridge::PreparedBasis;
    use crate::operator::Operator;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    #[test]
    fn test_schwarz_erfc_bounded_by_coulomb() {
        // erfc(ωr)/r ≤ 1/r pointwise, so |(ij|ij)_erfc| ≤ |(ij|ij)_Coulomb|
        // and therefore Q_erfc(i,j) ≤ Q_Coulomb(i,j) for every shell pair.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let q_c = schwarz(Operator::coulomb(), &prep).unwrap();
        let q_e = schwarz(Operator::erfc(0.222), &prep).unwrap();
        let nsh = prep.nshells();
        for i in 0..nsh {
            for j in 0..nsh {
                assert!(q_e[(i, j)] >= 0.0, "Q_erfc[{i},{j}] < 0");
                assert!(
                    q_e[(i, j)] <= q_c[(i, j)] + 1e-12,
                    "Q_erfc[{i},{j}]={} exceeds Q_Coulomb={}",
                    q_e[(i, j)],
                    q_c[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_schwarz_symmetric() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let q = schwarz(Operator::coulomb(), &prep).unwrap();
        let nsh = prep.nshells();
        for i in 0..nsh {
            for j in 0..nsh {
                assert!(
                    (q[(i, j)] - q[(j, i)]).abs() < 1e-12,
                    "Q not symmetric at ({i},{j})"
                );
                assert!(q[(i, j)] >= 0.0, "Q[{i},{j}] < 0");
            }
        }
    }

    /// Serial reference for `schwarz3_aux` (pre-parallelization implementation,
    /// kept verbatim).
    fn schwarz3_aux_serial(op: Operator, dfbs: &PreparedBasis) -> Vec<f64> {
        let nsh = dfbs.nshells();
        let dims = dfbs.shell_dims();
        let mut eng = Engine::new_2center(op, dfbs, SCHWARZ_TABLE_PRECISION).unwrap();
        let mut q3 = vec![0.0f64; nsh];
        for p in 0..nsh {
            let block = eng.compute_eri2(dfbs, p, p);
            let np = dims[p];
            let mut maxv = 0.0f64;
            for a in 0..np {
                let v = block[a * np + a].abs();
                if v > maxv {
                    maxv = v;
                }
            }
            q3[p] = maxv.sqrt().max(SCHWARZ_Q_FLOOR);
        }
        q3
    }

    /// Serial reference for `schwarz` (single engine, pair loop — the exact
    /// small-system path), used to prove the parallel path is bit-identical.
    fn schwarz_serial(op: Operator, prep: &PreparedBasis) -> Array2<f64> {
        let nsh = prep.nshells();
        let mut eng = Engine::new_2e(op, prep, SCHWARZ_TABLE_PRECISION).unwrap();
        let mut qmat = Array2::zeros((nsh, nsh));
        for i in 0..nsh {
            for j in 0..=i {
                let q = super::schwarz_pair(&mut eng, prep, i, j, bound_inflation(op)).unwrap();
                qmat[(i, j)] = q;
                qmat[(j, i)] = q;
            }
        }
        qmat
    }

    /// INVARIANT: the Schwarz table never stores an exact zero.
    ///
    /// A Schwarz bound's contract is that it never underestimates
    /// (arXiv:2302.11307). `SignificantPairs::build` compares with a strict
    /// `estimate(..) > threshold`, so a stored `Q = 0.0` makes that shell pair
    /// unreachable at every threshold INCLUDING 0 — which is exactly what
    /// destroyed the trivial limit before [`SCHWARZ_TABLE_PRECISION`] and
    /// [`SCHWARZ_Q_FLOOR`] existed.
    ///
    /// alkane_8/cc-pVDZ is the system the defect was characterised on: it
    /// stored 1049 zeros out of 5253 unique pairs (of which only 121 are
    /// genuinely zero), and its smallest nonzero entry was 4.632e-7 — the
    /// sqrt of the engine's 1e-14 precision cliff rather than any property of
    /// the molecule. Water/CH4 show NOTHING here, which is why the older tests
    /// passed while proving nothing; the molecule has to be big enough for
    /// distant shell pairs to fall under the cliff.
    ///
    /// This asserts the invariant at its source rather than through a
    /// downstream symptom, and it is the standing replacement for the
    /// deleted `schwarz_table_stores_zero_for_nonzero_pairs_alkane_8` anchor.
    #[test]
    fn schwarz_table_never_stores_a_zero_alkane_8() {
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_8.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let nsh = prep.nshells();
        assert_eq!(
            nsh, 102,
            "alkane_8/cc-pVDZ should give 102 shells; the counts quoted here were measured on \
             that decomposition"
        );

        for op in [Operator::coulomb(), Operator::erfc(0.222)] {
            let q = schwarz(op, &prep).unwrap();
            let zeros = (0..nsh)
                .flat_map(|i| (0..=i).map(move |j| (i, j)))
                .filter(|&(i, j)| q[(i, j)] == 0.0)
                .count();
            assert_eq!(
                zeros,
                0,
                "op={op:?}: {zeros} of {} unique shell pairs store Q == 0.0. Such an entry \
                 UNDERestimates a nonzero (ij|ij) and, because SignificantPairs uses a strict \
                 `> threshold`, makes the pair unreachable even at threshold 0 — the bound is \
                 no longer a bound. Check SCHWARZ_TABLE_PRECISION and SCHWARZ_Q_FLOOR.",
                nsh * (nsh + 1) / 2
            );
            // Every entry must also be finite and non-negative: a NaN would
            // compare false against every threshold and silently screen
            // everything away, which is the same failure wearing a disguise.
            for (idx, &v) in q.indexed_iter() {
                assert!(
                    v.is_finite() && v >= SCHWARZ_Q_FLOOR,
                    "op={op:?}: Q{idx:?} = {v} is not a valid bound (finite and >= the floor)"
                );
            }
        }
    }

    /// Same invariant for the auxiliary-basis bound used by the 3-index
    /// screened path. Measured pre-fix, this table stored NO zeros on
    /// alkane_8/cc-pVDZ-RI (smallest entry 4.475e-1) because a 2-centre
    /// `(P|P)` diagonal is O(1); the assertion guards the contract rather than
    /// repairing an observed failure, and would catch a future aux basis or
    /// operator whose diagonals do approach the cliff.
    #[test]
    fn schwarz3_aux_never_stores_a_zero_alkane_8() {
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_8.xyz").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        for op in [Operator::coulomb(), Operator::erfc(0.222)] {
            let q3 = schwarz3_aux(op, &dfbs).unwrap();
            for (p, &v) in q3.iter().enumerate() {
                assert!(
                    v.is_finite() && v >= SCHWARZ_Q_FLOOR,
                    "op={op:?}: Q3[{p}] = {v} is not a valid bound (finite and >= the floor)"
                );
            }
        }
    }

    #[test]
    fn test_schwarz_parallel_bitidentical_to_serial() {
        // alkane_6/cc-pVDZ clears PAR_SCHWARZ_SHELL_THRESHOLD (64 shells) so
        // schwarz() takes the parallel path; compare bitwise vs the serial loop.
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_6.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        assert!(
            prep.nshells() >= PAR_SCHWARZ_SHELL_THRESHOLD,
            "test basis too small to exercise the parallel path: {} shells",
            prep.nshells()
        );
        for op in [Operator::coulomb(), Operator::erfc(0.222)] {
            let par = schwarz(op, &prep).unwrap();
            let ser = schwarz_serial(op, &prep);
            let n_diff = par
                .iter()
                .zip(ser.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                n_diff, 0,
                "schwarz: {n_diff} elements differ bitwise (op={op:?})"
            );
        }
    }

    #[test]
    fn test_schwarz3_aux_bitidentical_to_serial() {
        // alkane_6/cc-pVDZ-RI clears PAR_AUX_SHELL_THRESHOLD (64 aux shells).
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_6.xyz").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        assert!(
            dfbs.nshells() >= 64,
            "test aux basis too small to exercise the parallel path: {} shells",
            dfbs.nshells()
        );
        for op in [Operator::coulomb(), Operator::erfc(0.222)] {
            let par = schwarz3_aux(op, &dfbs).unwrap();
            let ser = schwarz3_aux_serial(op, &dfbs);
            assert_eq!(par.len(), ser.len());
            let n_diff = par
                .iter()
                .zip(ser.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                n_diff, 0,
                "schwarz3_aux: {n_diff} elements differ bitwise (op={op:?})"
            );
        }
    }

    fn ensure_tables() -> bool {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
                let d = concat!(env!("CARGO_MANIFEST_DIR"), "/../../terf-tables");
                if std::path::Path::new(d).is_dir() {
                    std::env::set_var("FERRIC_TERF_TABLE_DIR", d);
                }
            }
        });
        std::env::var("FERRIC_TERF_TABLE_DIR").is_ok()
    }

    /// Hydrogen STO-3G plus one extra shell of angular momentum `l`.
    fn h2_with_shell(l: i32) -> PreparedBasis {
        let mol = Molecule::load_xyz("../../testdata/molecules/h2.xyz").unwrap();
        let mut bs = basis::bundled("sto-3g").unwrap();
        bs.shells.get_mut(&1).unwrap().push(basis::Shell {
            l,
            pure: true,
            exponents: vec![1.0],
            coefficients: vec![1.0],
        });
        PreparedBasis::new(&mol, &bs).unwrap()
    }

    #[test]
    fn terfc_quartet_limit_equals_the_shim_table_depth() {
        let shim = include_str!("../shim/shim.cc");
        let line = shim
            .lines()
            .find(|l| l.contains("constexpr int TERFC_DIMM"))
            .expect("TERFC_DIMM definition");
        let val: i32 = line
            .split('=')
            .nth(1)
            .unwrap()
            .split(';')
            .next()
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_eq!(val, TERFC_QUARTET_L_LIMIT);
        assert!(4 * TERFC_SCHWARZ_MAX_L < TERFC_QUARTET_L_LIMIT);
    }

    #[test]
    fn terfc_schwarz_refuses_outside_the_validated_range() {
        // The range check precedes engine construction: no tables needed.
        let prep = h2_with_shell(0);
        let ang = ferric_core::units::ANGSTROM_TO_BOHR;
        let w = |r0_ang: f64, s: f64| Operator::terfc_with_omega(r0_ang * ang, s / (r0_ang * ang));
        // inside, including both r0 edges and the s edge
        for (r0, s) in [(1.0, 0.7071), (0.3, 1.0), (2.5, 8.9), (1.0, 8.9)] {
            check_terfc_validated_range(w(r0, s), &prep)
                .unwrap_or_else(|e| panic!("r0={r0} s={s} refused: {e:?}"));
        }
        // outside: small r0, large r0, too sharp
        for (r0, s) in [(0.25, 1.0), (2.6, 1.0), (1.0, 9.5)] {
            let e = check_terfc_validated_range(w(r0, s), &prep)
                .expect_err(&format!("r0={r0} s={s} must be refused"));
            assert!(
                format!("{e:?}").contains("outside the validated range"),
                "{e:?}"
            );
        }
        // and through the public entry point
        assert!(schwarz(w(2.6, 1.0), &prep).is_err());
    }

    #[test]
    fn terfc_schwarz_refuses_unvalidated_angular_momentum() {
        let op = Operator::terfc(2.0);
        // g (l=4) is validated; h (l=5) is below the engine limit but not validated.
        check_terfc_validated_range(op, &h2_with_shell(4)).unwrap();
        let e = check_terfc_validated_range(op, &h2_with_shell(5)).expect_err("h refused");
        assert!(format!("{e:?}").contains("validated up to 4"), "{e:?}");
    }

    /// MUTATION-style: a table-engine refusal must surface as an error, not
    /// as the floor of a screened block (which would skip every triple on the
    /// pair). l = 6 reaches the engine's table depth on the (ii|ii) diagonal.
    #[test]
    fn terfc_engine_refusal_is_an_error_not_a_floor() {
        if !ensure_tables() {
            eprintln!("SKIP: terf tables absent");
            return;
        }
        let prep = h2_with_shell(6);
        let op = Operator::terfc(2.0);
        let shell = (0..prep.nshells())
            .find(|&i| prep.shell_dims()[i] == 13)
            .expect("the l=6 shell");
        let mut eng = Engine::new_2e(op, &prep, SCHWARZ_TABLE_PRECISION).unwrap();
        // the plain wrapper cannot tell refusal from screening ...
        assert!(eng
            .compute_quartet(&prep, shell, shell, shell, shell)
            .is_none());
        // ... the checked one and the Schwarz pair builder can
        assert!(eng
            .compute_quartet_checked(&prep, shell, shell, shell, shell)
            .is_err());
        let e = schwarz_pair(&mut eng, &prep, shell, shell, 1.0).expect_err("must not floor");
        assert!(format!("{e:?}").contains("refused"), "{e:?}");
        // a table-free Coulomb block that is genuinely fine is unaffected
        let mut c = Engine::new_2e(Operator::coulomb(), &prep, SCHWARZ_TABLE_PRECISION).unwrap();
        assert!(schwarz_pair(&mut c, &prep, 0, 0, 1.0).unwrap() > 0.0);
    }
}
