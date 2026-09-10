//! MWE: chunking the SCF property grid must not move a single bit.
//!
//! # The allocation being fixed
//!
//! `ferric_scf::properties::becke_charges` and
//! `atomic_effective_volumes_becke` each called
//! `eval_basis_on_points(mol, bs, &points)` on the ENTIRE grid, materialising a
//! contiguous `(nbf, npts)` matrix up front — even though every consumer reads
//! it one grid point at a time. `becke_charges` then forms `D · chi`, a second
//! matrix of the same shape, so its peak is 2x that.
//!
//! With the default 75x110 grid (8250 pts/atom):
//!
//! ```text
//!   system                 npts      chi (nbf,npts)
//!   benzene/cc-pVDZ       99,000       0.09 GB
//!   danuglipron/def2-SVP 602,250       3.37 GB   (becke_charges: 6.7 GB)
//!   danuglipron/def2-TZVP 602,250      7.71 GB   (becke_charges: 15.4 GB)
//! ```
//!
//! None of it was bounded by `[memory] budget_gb`: neither function consults a
//! budget, and neither goes through `check_ao_grid_budget` (which guards two of
//! the six large sites in `ao_grid.rs`, but not `eval_basis_on_points`). This
//! is the same shape as the 2026-07-13 incidents where ferric-cli reached
//! 16-17 GB anon-RSS on the Becke-grid property path.
//!
//! `ferric_rpa::properties::accumulate_atom_centred_dipoles` already solved
//! exactly this, and the fix here follows its idiom: chunk the grid at
//! `TARGET_CHUNKS = 1024`, a pure function of `npts` and never of
//! `rayon::current_num_threads()`.
//!
//! # Why this test, and why bit-identity is the right bar here
//!
//! Chunking a grid *looks* obviously safe and is not automatically so — this
//! repo has repeatedly been bitten by block-boundary choices that changed
//! floating-point results (`DRESS_ROW_BLOCK`'s doc measures ~7e-15 from every
//! odd output-row split of a GEMM). So the claim has to be checked, not
//! asserted.
//!
//! The reason it IS inert here, stated so a reviewer can check the reasoning
//! against the loop rather than trust this comment:
//!
//! * `chi` is a materialised lookup table, not a reduction. `chi[mu, g]` is a
//!   pure function of point `g`'s coordinates, so evaluating points in groups
//!   reproduces identical values.
//! * `becke_charges`'s `d_chi = D.dot(&chi)` reduces over `nbf` (the shared
//!   index); `g` is a FREE index of that product. Splitting `g` therefore
//!   splits independent output COLUMNS and leaves the k-axis accumulation
//!   untouched — unlike `DRESS_ROW_BLOCK`, where the split moved a GEMM's
//!   output rows and shifted its accumulation lanes.
//! * The one genuine reduction over `g` (`n_e[home] += w*rho`, `vol[a] += ...`)
//!   must stay in ascending point order. The fix keeps a serial ascending fold
//!   over chunks, so the addition sequence is unchanged.
//!
//! Bit-identity is therefore the correct bar, and a weaker tolerance would let
//! a real reordering through. Contrast the spilled 3-index tensor, where a
//! budget legitimately regroups a sum and the honest bar is ulp-bounded.
//!
//! # What this test must NOT do — a mistake this file already made
//!
//! Its first version asserted bit-identity against hardcoded reference values
//! captured on the dev box. **CI failed them by 3.3e-7** (relative 1.9e-6),
//! nine orders larger than any chunking reassociation (~1e-15). The magnitude
//! is the whole argument: at SCF scale rather than rounding scale, the CHARGES
//! differed because the DENSITY differed, before the grid loop was ever
//! reached. The test was pinning a machine-dependent SCF result, not the
//! property it names.
//!
//! WHICH machine difference is NOT established. `ci.yml:67` sets
//! `OPENBLAS_CORETYPE=Haswell` and that was the first suspicion, but it does
//! NOT reproduce locally: native and Haswell give a bit-identical water/cc-pVDZ
//! RHF energy on this box, so the cause is something else in the CI toolchain
//! (CPU microarchitecture, a different OpenBLAS build, libint2, LAPACK). That
//! does not weaken the conclusion — a cross-machine bit-identity assertion on
//! an SCF-derived quantity is invalid whichever of those it is — but do not
//! repeat "it's CORETYPE" as if it were measured.
//!
//! The fix is to compare TWO CHUNK WIDTHS ON THE SAME MACHINE, via
//! `becke_charges_chunked` / `atomic_effective_volumes_becke_chunked`. Both
//! runs then share one density, so any difference is attributable to the
//! chunking alone — which is the actual claim — and the assertion is
//! machine-independent by construction.
//!
//! GENERAL RULE this cost: a cross-machine bit-identity assertion is only valid
//! for a quantity with NO upstream floating-point dependence. Anything
//! downstream of an SCF is not such a quantity.
//!
//! Run with `OPENBLAS_NUM_THREADS=1` per the project's rayon/BLAS convention.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::properties::becke_charges;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Water/cc-pVDZ: 3 atoms x 8250 = 24,750 grid points, comfortably more than
/// the 1024 target chunks, so the chunking is genuinely exercised (many chunks,
/// not one). A single-atom fixture would leave `chunk_size >= npts` and the
/// mechanism inert — this test would then pass whatever the code did.
fn fixture() -> (Molecule, PreparedBasis, basis::BasisSet, ndarray::Array2<f64>) {
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ferric_core::parallel::ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    let d = rhf.density_total().to_owned();
    (mol, obs, obs_bs, d)
}

/// Grid chunk widths to compare. `npts` for water/cc-pVDZ is 3 x 8250 =
/// 24,750, so these span "one chunk" (fully unchunked) down to very narrow.
///
/// Comparing widths to EACH OTHER, rather than to a stored constant, is what
/// makes this machine-independent — see the module doc.
const WIDTHS: [usize; 5] = [24_750, 8_192, 1_024, 97, 13];

/// CONTRACT 1: `becke_charges` is bit-identical across every chunk width.
///
/// The core claim. One SCF density, several chunk widths, same machine: any
/// difference is the chunking and nothing else. The unchunked width (one chunk
/// spanning all points) is included, so this also pins that chunking did not
/// change the answer relative to the pre-fix behaviour.
#[test]
fn becke_charges_are_bit_identical_across_chunk_widths() {
    use ferric_scf::properties::becke_charges_chunked;
    let (mol, obs, obs_bs, d) = fixture();
    let reference = becke_charges_chunked(&mol, &obs, &obs_bs, &d, Some(WIDTHS[0])).unwrap();
    assert_eq!(reference.len(), 3, "water has three atoms");
    for w in WIDTHS {
        let got = becke_charges_chunked(&mol, &obs, &obs_bs, &d, Some(w)).unwrap();
        for (i, (&g, &r)) in got.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                r.to_bits(),
                "becke_charges[{i}] moved between chunk widths {} and {w}: {r:.17e} vs \
                 {g:.17e} (difference {:.3e}). Grid chunking must be numerically INERT — chi \
                 is a lookup table and the g axis is a free index of D.dot(chi), so a \
                 difference here means the ascending fold over grid points was disturbed.",
                WIDTHS[0],
                (g - r).abs()
            );
        }
    }
}

/// CONTRACT 2: the charges still sum to zero for a neutral molecule.
///
/// A physics-level invariant that survives any legitimate reference
/// regeneration, so CONTRACT 1 cannot be "fixed" by pasting in whatever the
/// broken code produced without this also holding.
#[test]
fn becke_charges_sum_to_the_molecular_charge() {
    let (mol, obs, obs_bs, d) = fixture();
    let q = becke_charges(&mol, &obs, &obs_bs, &d).unwrap();
    let sum: f64 = q.iter().sum();
    assert!(
        sum.abs() < 1e-10,
        "neutral water's Becke charges must sum to 0, got {sum:.3e}"
    );
}

/// CONTRACT 3: `atomic_effective_volumes_becke` likewise, plus a physical check.
///
/// The sibling call site, chunked by the same change. Bit-identity across
/// widths is the chunking claim; the ordering assertion is a physics invariant
/// that stays meaningful on any machine and would catch a chunking error that
/// dropped or double-counted a band of grid points.
#[test]
fn atomic_effective_volumes_are_bit_identical_across_chunk_widths() {
    use ferric_scf::properties::atomic_effective_volumes_becke_chunked;
    let (mol, obs, obs_bs, d) = fixture();
    let reference =
        atomic_effective_volumes_becke_chunked(&mol, &obs, &obs_bs, &d, Some(WIDTHS[0])).unwrap();
    assert_eq!(reference.len(), 3);
    for w in WIDTHS {
        let got =
            atomic_effective_volumes_becke_chunked(&mol, &obs, &obs_bs, &d, Some(w)).unwrap();
        for (i, (&g, &r)) in got.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                r.to_bits(),
                "atomic_effective_volumes_becke[{i}] moved between chunk widths {} and {w}: \
                 {r:.17e} vs {g:.17e} (difference {:.3e})",
                WIDTHS[0],
                (g - r).abs()
            );
        }
    }
    for (i, &vi) in reference.iter().enumerate() {
        assert!(vi > 0.0, "atom {i} effective volume must be positive, got {vi:.6e}");
    }
    assert!(
        reference[0] > reference[1] && reference[0] > reference[2],
        "oxygen's effective volume ({:.4}) must exceed both hydrogens' ({:.4}, {:.4}) — a \
         chunking error that dropped a band of points would show up here",
        reference[0], reference[1], reference[2]
    );
}

/// CONTRACT 4: results do not depend on the ambient rayon worker count.
///
/// # Honest scope — this contract is WEAKER than it looks
///
/// Mutation-checked, and it does NOT catch a thread-derived chunk size:
/// replacing `deterministic_group_size(npts)` with
/// `npts / rayon::current_num_threads()` leaves every contract in this file
/// GREEN.
///
/// The reason is the same property that makes the fix safe. Because chunking
/// here is genuinely inert — chi is a lookup table and the g-fold stays
/// ascending — *any* chunk width produces identical numbers. So a
/// thread-derived width would be a latent MEMORY defect (peak scaling with core
/// count instead of with npts, the mechanism behind the 16-17 GB anon-RSS
/// incidents) without being a numerics defect, and this file only observes
/// numbers.
///
/// Keep the contract anyway: it costs nothing and it pins the property against
/// a future change that makes the chunking non-inert (a parallel fold, a
/// per-chunk GEMM over the g axis), at which point it would start catching
/// exactly what it names. But do not read a green run here as proof that the
/// chunk width is thread-independent — read
/// `deterministic_group_size(npts)` at the call site for that, and see
/// `ferric_scf::reduce::TARGET_GROUPS` for why it is a pure function of the
/// item count.
#[test]
fn grid_chunking_does_not_depend_on_the_worker_count() {
    let (mol, obs, obs_bs, d) = fixture();
    let at = |n: usize| -> Vec<f64> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
            .install(|| becke_charges(&mol, &obs, &obs_bs, &d).unwrap())
    };
    let reference = at(1);
    for n in [2usize, 3, 8] {
        let got = at(n);
        for (i, (&g, &r)) in got.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                r.to_bits(),
                "becke_charges[{i}] changed between 1 and {n} workers ({r:.17e} vs \
                 {g:.17e}). The chunk size must be a pure function of npts, never of \
                 the ambient pool."
            );
        }
    }
}
