//! GATE (GO/NO-GO): does localize-then-TRUNCATE of the virtual space survive
//! the per-domain UNION?
//!
//! # Why this test exists
//!
//! `pv_sparsity_diagnostic.rs` proved that ROTATING the virtuals and rebuilding
//! the SAME full-rank pseudo-density is a mathematical no-op:
//! `P_v(τ) = C_vir exp(−F_vir τ) C_virᵀ` sums over the COMPLETE virtual space,
//! so it is a weighted projector onto the whole subspace and any internal
//! rotation `U` cancels (measured 1.6e-15). That is airtight and NOT re-litigated
//! here.
//!
//! What that result does NOT refute is localize-then-**TRUNCATE**: summing only
//! over the virtuals ASSIGNED TO a domain gives an incomplete virtual sum, so the
//! invariance argument says nothing about it. Localization is a *selection
//! criterion* — you localize so you can DROP orbitals.
//!
//! # The known failure mode being gated against
//!
//! Per `rpa-cannot-consume-pair-indexed-bases`: per-domain subspaces individually
//! compress well, but their UNION re-inflates to full rank because different
//! domains select DIFFERENT subsets. The bound is
//!
//!     rank(Σ_d P^d)  ≤  Σ_d rank(P^d)
//!
//! with EQUALITY when the subspaces are mutually orthogonal. Compression survives
//! a union ONLY if different domains select OVERLAPPING subspaces.
//!
//! # Decision rule (stated BEFORE measuring, applied honestly)
//!
//!   * `rank(⋃) / n_vir` near 1.0  ⇒ union re-inflates ⇒ NO-GO, same wall as RPA.
//!   * `rank(⋃)` meaningfully below `n_vir` AND not growing with the number of
//!     domains unioned ⇒ GO signal.
//!   * `rank(⋃)` growing with system size / domain count ⇒ no asymptotic win
//!     regardless of small-system numbers.
//!
//! # Assignment rule (explicit)
//!
//! Domains are ATOM-CENTERED: one domain per atom. A Boys-localized virtual `a`
//! is assigned to domain `d` iff `|centroid(a) − R_d| <= r`. This is the standard
//! local-correlation selection criterion (orbital-to-domain by centroid
//! proximity), and it is the natural one for localized virtuals since a
//! localized virtual has a well-defined center.
//!
//! `V_d` = the localized virtuals assigned to domain `d`. Because the localized
//! virtuals are one ORTHONORMAL set, `rank(V_d) = |V_d|` exactly, and
//! `rank(⋃_d V_d) = |⋃_d V_d|` (the count of distinct virtuals assigned
//! anywhere). Both are asserted numerically via the Gram matrix / SVD rather than
//! assumed — see `orthonormality_makes_rank_equal_count`.
//!
//! No timings are reported anywhere in this file, deliberately. Ranks,
//! dimensions and retention fractions only.
//!
//! Run with:
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --test localized_virtual_union_rank_gate -- --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::{dipole, overlap};
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{s, Array2};
use ndarray_linalg::SVD;

// ---------------------------------------------------------------------------
// System setup
// ---------------------------------------------------------------------------

struct SystemData {
    label: String,
    basis: String,
    prep: PreparedBasis,
    /// Atom positions in Bohr (the domain centers).
    atom_pos: Vec<[f64; 3]>,
    c_vir: Array2<f64>,
    /// Boys-localized virtuals, (nbas, nvir).
    c_vir_loc: Array2<f64>,
    /// Boys centroids of the localized virtuals, (nvir, 3).
    vir_centers: Array2<f64>,
    nvir: usize,
    diameter: f64,
    /// Boys functional before/after localization — the honesty evidence.
    boys_before: f64,
    boys_after: f64,
}

fn run_scf(path: &str, basis_name: &str, label: &str) -> SystemData {
    let mol = Molecule::load_xyz(path).unwrap_or_else(|e| panic!("load {path}: {e}"));
    let bs = basis::bundled(basis_name).unwrap_or_else(|e| panic!("basis {basis_name}: {e}"));
    let prep = PreparedBasis::new(&mol, &bs).unwrap_or_else(|e| panic!("prep {label}: {e}"));
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap_or_else(|e| panic!("schwarz: {e}"));
    let ctx = ParallelContext::default();
    let config = RhfConfig::default();
    let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config)
        .unwrap_or_else(|e| panic!("scf {label}: {e}"));

    let nocc = mol.nelec() as usize / 2;
    let nbas = prep.nbasis();
    let c_vir = rhf.mos_r().slice(s![.., nocc..]).to_owned();
    let nvir = c_vir.ncols();

    let dip = dipole(&prep, [0.0, 0.0, 0.0]).unwrap_or_else(|e| panic!("dipole: {e}"));
    let boys = ferric_mp2::boys::boys_localize(&c_vir, &dip, 400);

    let boys_functional = |c: &Array2<f64>| -> f64 {
        let mut acc = 0.0;
        for a in 0..3 {
            let dc = dip[a].dot(c);
            for i in 0..c.ncols() {
                let v = c.column(i).dot(&dc.column(i));
                acc += v * v;
            }
        }
        acc
    };
    let boys_before = boys_functional(&c_vir);
    let boys_after = boys_functional(&boys.c_loc);

    let atom_pos: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let mut diameter = 0.0_f64;
    for a in &atom_pos {
        for b in &atom_pos {
            let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            if d > diameter {
                diameter = d;
            }
        }
    }

    println!(
        "\n=== {label} [{basis_name}] : natoms={} nbas={nbas} nocc={nocc} nvir={nvir} \
         diameter={diameter:.2} Bohr ===",
        atom_pos.len()
    );
    println!(
        "    virtual Boys functional {boys_before:.4} -> {boys_after:.4}  \
         (converged={}, {} sweeps)",
        boys.converged, boys.iterations
    );

    SystemData {
        label: label.to_string(),
        basis: basis_name.to_string(),
        prep,
        atom_pos,
        c_vir,
        c_vir_loc: boys.c_loc,
        vir_centers: boys.centers,
        nvir,
        diameter,
        boys_before,
        boys_after,
    }
}

// ---------------------------------------------------------------------------
// Rank utilities — numerical, not assumed
// ---------------------------------------------------------------------------

/// Numerical rank of a coefficient block (nbas × k) via SVD of the
/// METRIC-orthonormalized columns. Since the MOs satisfy CᵀSC = I, the singular
/// values of S^{1/2}C are all 1; we instead work directly with the Gram matrix
/// CᵀSC and count eigenvalue-equivalent singular values above tol.
fn rank_of_block(block: &Array2<f64>, s_ao: &Array2<f64>, tol: f64) -> usize {
    if block.ncols() == 0 {
        return 0;
    }
    let gram = block.t().dot(&s_ao.dot(block));
    let (_, sv, _) = gram.svd(false, false).expect("svd of gram failed");
    let max = sv.iter().cloned().fold(0.0_f64, f64::max);
    if max == 0.0 {
        return 0;
    }
    sv.iter().filter(|v| **v > tol * max).count()
}

/// Columns of `c` selected by index list.
fn select_cols(c: &Array2<f64>, idx: &[usize]) -> Array2<f64> {
    let mut out = Array2::zeros((c.nrows(), idx.len()));
    for (k, &a) in idx.iter().enumerate() {
        out.column_mut(k).assign(&c.column(a));
    }
    out
}

// ---------------------------------------------------------------------------
// The assignment rule
// ---------------------------------------------------------------------------

/// Assign each localized virtual to every atom-centered domain whose center is
/// within `radius` Bohr of the virtual's Boys centroid.
///
/// Returns `(per_domain, assigned_anywhere)`.
fn assign_virtuals_to_domains(
    sys: &SystemData,
    radius: f64,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    let natoms = sys.atom_pos.len();
    let r2 = radius * radius;
    let mut per_domain: Vec<Vec<usize>> = vec![Vec::new(); natoms];
    let mut seen = vec![false; sys.nvir];

    for d in 0..natoms {
        let rd = sys.atom_pos[d];
        for a in 0..sys.nvir {
            let dx = sys.vir_centers[(a, 0)] - rd[0];
            let dy = sys.vir_centers[(a, 1)] - rd[1];
            let dz = sys.vir_centers[(a, 2)] - rd[2];
            if dx * dx + dy * dy + dz * dz <= r2 {
                per_domain[d].push(a);
                seen[a] = true;
            }
        }
    }
    let assigned: Vec<usize> = (0..sys.nvir).filter(|&a| seen[a]).collect();
    (per_domain, assigned)
}

// ---------------------------------------------------------------------------
// THE GATE
// ---------------------------------------------------------------------------

struct GateRow {
    system: String,
    basis: String,
    radius: f64,
    natoms: usize,
    nvir: usize,
    n_nonempty: usize,
    max_domain_rank: usize,
    union_rank: usize,
    sum_domain_ranks: usize,
    /// Virtuals assigned to NO domain (silently dropped by the union).
    n_uncovered: usize,
    /// True when rank(U) == Sum_d rank(V_d): the mutual-ORTHOGONALITY equality
    /// case of the rank bound, i.e. domains select DISJOINT subspaces and the
    /// union buys nothing.
    orthogonal_equality: bool,
}

fn measure(sys: &SystemData, radii: &[f64], s_ao: &Array2<f64>) -> Vec<GateRow> {
    let mut rows = Vec::new();
    println!("\n  radius  n_dom  max|V_d|  rk(max V_d)  rank(U)  Sum rk  rk(U)/nvir  uncov  orth_eq");
    for &r in radii {
        let (per_domain, assigned) = assign_virtuals_to_domains(sys, r);

        let n_nonempty = per_domain.iter().filter(|d| !d.is_empty()).count();

        // Per-domain ranks, computed NUMERICALLY (not assumed = count).
        let mut domain_ranks = Vec::new();
        for dom in &per_domain {
            if dom.is_empty() {
                domain_ranks.push(0);
                continue;
            }
            let blk = select_cols(&sys.c_vir_loc, dom);
            domain_ranks.push(rank_of_block(&blk, s_ao, 1e-9));
        }
        let max_domain_rank = domain_ranks.iter().cloned().max().unwrap_or(0);
        let sum_domain_ranks: usize = domain_ranks.iter().sum();

        // Union rank, computed NUMERICALLY over the stacked assigned virtuals.
        let union_rank = if assigned.is_empty() {
            0
        } else {
            let blk = select_cols(&sys.c_vir_loc, &assigned);
            rank_of_block(&blk, s_ao, 1e-9)
        };

        let max_count = per_domain.iter().map(|d| d.len()).max().unwrap_or(0);
        // CRITICAL DISAMBIGUATION. rank(U) < nvir has TWO possible causes, and
        // only one of them is a real win:
        //
        //   (a) domains select OVERLAPPING subspaces  -> rank(U) < Sum_d rank(V_d)
        //       This is genuine compression: the union is cheaper than the parts.
        //   (b) some virtuals are assigned to NO domain and are silently dropped
        //       -> rank(U) == Sum_d rank(V_d) (the mutual-ORTHOGONALITY equality
        //       case of the rank bound). This is NOT compression — it is an
        //       uncovered virtual space, i.e. a truncation that discards orbitals
        //       nobody selected, which is exactly the RPA failure mode.
        //
        // `orthogonal_equality` flags case (b). `n_uncovered` quantifies it.
        let n_uncovered = sys.nvir - assigned.len();
        let orthogonal_equality = union_rank == sum_domain_ranks;
        println!(
            "  {:6.1}  {:>5}  {:>8}  {:>11}  {:>7}  {:>6}  {:>10.3}  {:>5}  {:>7}",
            r,
            n_nonempty,
            max_count,
            max_domain_rank,
            union_rank,
            sum_domain_ranks,
            union_rank as f64 / sys.nvir as f64,
            n_uncovered,
            if orthogonal_equality { "YES" } else { "no" }
        );

        rows.push(GateRow {
            system: sys.label.clone(),
            basis: sys.basis.clone(),
            radius: r,
            natoms: sys.atom_pos.len(),
            nvir: sys.nvir,
            n_nonempty,
            max_domain_rank,
            union_rank,
            sum_domain_ranks,
            n_uncovered,
            orthogonal_equality,
        });
    }
    rows
}

#[test]
fn union_rank_gate_localized_virtual_truncation() {
    // Radii spanning "domain smaller than a bond" up to "domain spans the
    // molecule". The interesting regime is small r — that is where truncation
    // would actually discard something.
    let radii = [1.0_f64, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0];

    // (path, label, bases) — basis quality BEATS system size under load, so the
    // largest system carries the smaller basis list. STO-3G is a MINIMAL basis
    // (nbas ≈ nocc, tiny atypical virtual space): a union-rank verdict there
    // alone would be a small-basis artifact, so every system is also run at a
    // real basis with a genuine virtual space to truncate.
    let owned = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).collect() };
    let default_bases = owned(&["sto-3g", "6-31g", "cc-pvdz"]);

    let mut systems: Vec<(&str, &str, Vec<String>)> = vec![
        ("../../testdata/molecules/water.xyz", "water", default_bases.clone()),
        ("../../testdata/molecules/alkane_4.xyz", "alkane_4", default_bases.clone()),
    ];
    // alkane_8 is opt-in via GATE_ALKANE8_BASES (comma-separated; empty string
    // skips it entirely). Under load, drop the largest SYSTEM before dropping
    // the real BASIS — basis quality beats system size for this question.
    let heavy: Vec<String> = match std::env::var("GATE_ALKANE8_BASES") {
        Ok(v) if v.trim().is_empty() => Vec::new(),
        Ok(v) => v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
        Err(_) => owned(&["sto-3g", "6-31g"]),
    };
    if !heavy.is_empty() {
        systems.push(("../../testdata/molecules/alkane_8.xyz", "alkane_8", heavy));
    }

    let mut all_rows: Vec<GateRow> = Vec::new();
    let mut boys_evidence: Vec<(String, String, f64, f64)> = Vec::new();
    let mut dims: Vec<(String, String, usize, usize, usize)> = Vec::new();

    let pairs: Vec<(String, String, String)> = systems
        .iter()
        .flat_map(|(p, l, bs)| {
            bs.iter()
                .map(move |b| (p.to_string(), l.to_string(), b.clone()))
        })
        .collect();

    for (path, label, basis_name) in pairs {
        let sys = run_scf(&path, &basis_name, &label);
        let tag = format!("{}/{}", sys.label, sys.basis);

        // TEETH #1: the localizer must ACTUALLY have localized. Per
        // `boys-localization-sign-bug`, a wrong-branch localizer converges
        // happily to the MINIMUM (maximally delocalized, all centroids
        // collapsed) while reporting converged=true. If the virtuals are not
        // genuinely localized, every number below is meaningless, and a
        // centroid-based assignment rule cannot discriminate BY CONSTRUCTION.
        assert!(
            sys.boys_after > sys.boys_before,
            "{}: virtual Boys localization must MAXIMIZE the functional, but it went \
             {:.6} -> {:.6}. Centroid-based domain assignment is meaningless against \
             delocalized orbitals.",
            tag,
            sys.boys_before,
            sys.boys_after
        );
        // TEETH #2: the centroids must be mutually SEPARATED. Collapsed centroids
        // (the sign-bug signature) would put every virtual in every domain and
        // fake a union re-inflation, or put them all in one domain and fake
        // compression. Either way the gate would be measuring nothing.
        let mut max_sep = 0.0_f64;
        for i in 0..sys.nvir {
            for j in 0..sys.nvir {
                let d: f64 = (0..3)
                    .map(|a| (sys.vir_centers[(i, a)] - sys.vir_centers[(j, a)]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                max_sep = max_sep.max(d);
            }
        }
        println!("    max virtual-centroid separation = {max_sep:.3} Bohr");
        assert!(
            max_sep > 1.0,
            "{}: all localized-virtual centroids lie within {max_sep:.4} Bohr of each \
             other — the virtuals are NOT localized and the domain assignment below \
             cannot discriminate",
            tag
        );
        // TEETH #3: the localization must have genuinely rotated the coefficients,
        // else `c_vir_loc == c_vir` and this whole harness is the canonical case.
        let coef_change = (&sys.c_vir_loc - &sys.c_vir)
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            coef_change > 0.1,
            "{}: Boys localization barely moved the virtual coefficients \
             (max Δ = {coef_change:.3e})",
            tag
        );

        boys_evidence.push((
            sys.label.clone(),
            sys.basis.clone(),
            sys.boys_before,
            sys.boys_after,
        ));
        dims.push((
            sys.label.clone(),
            sys.basis.clone(),
            sys.prep.nbasis(),
            sys.nvir,
            sys.atom_pos.len(),
        ));

        let s_ao = overlap(&sys.prep);

        // TEETH #4: the rank measurement itself must be correct. The FULL
        // localized virtual set must measure rank == nvir. If `rank_of_block`
        // under-reports, every union number below is spuriously small and the
        // gate would fabricate a GO.
        let full_rank = rank_of_block(&sys.c_vir_loc, &s_ao, 1e-9);
        assert_eq!(
            full_rank, sys.nvir,
            "{}: rank of the full localized virtual block must equal nvir={} but \
             measured {}. The rank metric is broken; no union number is trustworthy.",
            tag, sys.nvir, full_rank
        );

        let rows = measure(&sys, &radii, &s_ao);

        // TEETH #5: the sweep must contain a NON-TRIVIAL truncation regime — at
        // least one radius where some domain discards virtuals (max|V_d| < nvir)
        // AND at least one radius where domains are non-empty. Without both, the
        // gate passes vacuously by never truncating anything.
        let has_truncating = rows
            .iter()
            .any(|r| r.max_domain_rank > 0 && r.max_domain_rank < r.nvir);
        assert!(
            has_truncating,
            "{}: no radius in the sweep produced a domain that both is non-empty AND \
             discards virtuals — the gate has nothing to measure",
            tag
        );

        // TEETH #6: the disambiguation must be LIVE, not decorative. At least one
        // radius in the sweep must exhibit the orthogonal-equality case
        // (rank(U) == Sum_d rank(V_d)) and at least one must NOT. If every row
        // were on one side, the `orthogonal_equality` discriminator would be
        // constant and the decision rule's clause (3) would be untested — which
        // is exactly how a gate silently degrades into a rubber stamp.
        let any_orth = rows.iter().any(|r| r.orthogonal_equality && r.n_nonempty > 1);
        let any_shared = rows.iter().any(|r| !r.orthogonal_equality && r.n_nonempty > 1);
        assert!(
            any_orth && any_shared,
            "{}: the orthogonal-equality discriminator is constant across the whole \
             radius sweep (any_orth={any_orth}, any_shared={any_shared}); clause (3) of \
             the decision rule is untested and the gate cannot distinguish genuine \
             subspace sharing from a disjoint partition",
            tag
        );

        all_rows.extend(rows);
    }

    // -----------------------------------------------------------------------
    // Summary + verdict
    // -----------------------------------------------------------------------
    println!("\n\n================ VIRTUAL-SPACE DIMENSIONS ================");
    println!(
        "(nvir is what makes an STO-3G number interpretable or not — a minimal \n\
         basis has nbas ~ nocc and an atypically tiny virtual space.)"
    );
    println!(
        "{:<12} {:<10} {:>7} {:>7} {:>7}",
        "system", "basis", "natoms", "nbas", "nvir"
    );
    for (l, b, nbas, nvir, na) in &dims {
        println!("{l:<12} {b:<10} {na:>7} {nbas:>7} {nvir:>7}");
    }

    println!("\n\n================ UNION-RANK GATE SUMMARY ================");
    println!(
        "{:<12} {:<10} {:>7} {:>6} {:>8} {:>11} {:>9} {:>8} {:>11}",
        "system", "basis", "radius", "nvir", "n_dom", "max rk(V_d)", "rank(U)", "Sum rk", "rk(U)/nvir"
    );
    for r in &all_rows {
        println!(
            "{:<12} {:<10} {:>7.1} {:>6} {:>8} {:>11} {:>9} {:>8} {:>11.3}",
            r.system,
            r.basis,
            r.radius,
            r.nvir,
            r.n_nonempty,
            r.max_domain_rank,
            r.union_rank,
            r.sum_domain_ranks,
            r.union_rank as f64 / r.nvir as f64
        );
    }

    println!("\nBoys evidence (virtual functional before -> after):");
    for (l, b, f0, f1) in &boys_evidence {
        println!("  {l:<12} {b:<10} {f0:.4} -> {f1:.4}");
    }

    // The decision rule, applied mechanically. A radius "pays" only if the
    // union genuinely stays below the full virtual space while domains are
    // actually populated (n_nonempty > 1 — a single domain is not a union).
    println!("\n--- DECISION RULE APPLIED (per basis) ---");
    println!("A radius PAYS iff ALL of:");
    println!("  (1) n_dom_nonempty > 1                  (a single domain is not a union)");
    println!("  (2) rank(U)/nvir <= 0.9                 (the union stays below full rank)");
    println!("  (3) NOT orthogonal_equality             (rank(U) < Sum_d rank(V_d), i.e.");
    println!("      domains share subspaces rather than partitioning disjoint ones)");
    println!("  (4) n_uncovered == 0                    (no virtual is dropped merely by");
    println!("      being assigned to NO domain -- that is an uncovered space, not");
    println!("      compression, and it discards orbitals no domain ever selected)");

    let mut bases: Vec<String> = all_rows.iter().map(|r| r.basis.clone()).collect();
    bases.sort();
    bases.dedup();

    let mut verdicts: Vec<(String, bool, f64)> = Vec::new();
    for b in &bases {
        let rows: Vec<&GateRow> = all_rows.iter().filter(|r| &r.basis == b).collect();
        let mut pays = false;
        let mut best_ratio = f64::INFINITY;
        for r in &rows {
            let ratio = r.union_rank as f64 / r.nvir as f64;
            if r.n_nonempty > 1 && r.union_rank > 0 {
                if ratio < best_ratio {
                    best_ratio = ratio;
                }
                let real = ratio <= 0.9 && !r.orthogonal_equality && r.n_uncovered == 0;
                if real {
                    println!(
                        "  [{b}] PAYS: {} r={:.1}  rank(U)={}/{} = {:.3} over {} domains \
                         (Sum rk={}, uncovered={})",
                        r.system, r.radius, r.union_rank, r.nvir, ratio,
                        r.n_nonempty, r.sum_domain_ranks, r.n_uncovered
                    );
                    pays = true;
                } else if ratio <= 0.9 {
                    // Sub-unity ratio that FAILS the disambiguation. Name why.
                    let why = if r.orthogonal_equality && r.n_uncovered > 0 {
                        "DISJOINT subspaces (rank(U)==Sum rk) AND uncovered virtuals"
                    } else if r.orthogonal_equality {
                        "DISJOINT subspaces (rank(U)==Sum rk): union buys nothing"
                    } else {
                        "uncovered virtuals dropped by no-domain, not compression"
                    };
                    println!(
                        "  [{b}] rejected: {} r={:.1}  rank(U)={}/{} = {:.3} -- {} \
                         (Sum rk={}, uncovered={})",
                        r.system, r.radius, r.union_rank, r.nvir, ratio, why,
                        r.sum_domain_ranks, r.n_uncovered
                    );
                }
            }
        }
        if !pays {
            println!(
                "  [{b}] NO-GO: no radius yields a union that is BOTH below 0.9*nvir AND \
                 genuinely compressive; best rank(U)/nvir = {:.3}",
                best_ratio
            );
        }
        verdicts.push((b.clone(), pays, best_ratio));
    }

    println!("\n--- PER-BASIS VERDICT ---");
    for (b, pays, best) in &verdicts {
        println!(
            "  {:<10} {}   best rank(U)/nvir = {:.3}",
            b,
            if *pays { "GO   " } else { "NO-GO" },
            best
        );
    }
    let all_same = verdicts.iter().all(|(_, p, _)| *p == verdicts[0].1);
    if all_same {
        println!(
            "  => verdict is CONSISTENT across minimal and real bases; the result is \n\
             not a small-basis artifact."
        );
    } else {
        println!(
            "  => *** HEADLINE: THE VERDICT DIFFERS BETWEEN BASES. *** The minimal-basis \n\
             and real-basis answers disagree, which is a more important finding than \n\
             either alone. Do NOT average them."
        );
    }

    // Size growth: does rank(U) grow with system size at matched radius AND
    // matched basis?
    println!("\n--- SIZE GROWTH of rank(U) at matched radius, WITHIN each basis ---");
    for b in &bases {
        println!("  [{b}]");
        for &r in &radii {
            let mut line = format!("    r={r:5.1}:");
            let mut any = false;
            for sysname in ["water", "alkane_4", "alkane_8"] {
                if let Some(row) = all_rows.iter().find(|x| {
                    x.system == sysname && &x.basis == b && (x.radius - r).abs() < 1e-12
                }) {
                    any = true;
                    line.push_str(&format!(
                        "  {}: {}/{} ({:.2})",
                        sysname,
                        row.union_rank,
                        row.nvir,
                        row.union_rank as f64 / row.nvir as f64
                    ));
                }
            }
            if any {
                println!("{line}");
            }
        }
    }
    println!(
        "\nIf rank(U) GROWS with system size at fixed radius and basis, there is no\n\
         asymptotic win regardless of the small-system numbers."
    );
}

/// TEETH for the rank metric itself, independent of the gate: the localized
/// virtuals are one ORTHONORMAL set (CᵀSC = I), so any subset has
/// rank == count, and the union of subsets has rank == count of distinct
/// members. This is what makes the union-rank question equivalent to a
/// set-cover question — and it is why the union CANNOT compress unless some
/// virtuals are assigned to NO domain.
///
/// A negative control is included: a deliberately RANK-DEFICIENT block (a
/// duplicated column) must measure a rank BELOW its column count, proving
/// `rank_of_block` is not just returning `ncols()`.
#[test]
fn rank_metric_is_not_just_column_count() {
    let sys = run_scf(
        "../../testdata/molecules/alkane_4.xyz",
        "sto-3g",
        "alkane_4/STO-3G [rank metric]",
    );
    let s_ao = overlap(&sys.prep);

    // Orthonormality of the localized virtuals: CᵀSC = I.
    let gram = sys.c_vir_loc.t().dot(&s_ao.dot(&sys.c_vir_loc));
    let mut max_dev = 0.0_f64;
    for i in 0..sys.nvir {
        for j in 0..sys.nvir {
            let expect = if i == j { 1.0 } else { 0.0 };
            max_dev = max_dev.max((gram[(i, j)] - expect).abs());
        }
    }
    println!("localized-virtual |CᵀSC − I|max = {max_dev:.3e}");
    assert!(
        max_dev < 1e-8,
        "localized virtuals are not orthonormal (dev {max_dev:.3e}); the \
         rank==count identity underpinning the gate does not hold"
    );

    // POSITIVE: a subset of orthonormal columns has rank == count.
    let subset: Vec<usize> = (0..sys.nvir.min(5)).collect();
    let blk = select_cols(&sys.c_vir_loc, &subset);
    assert_eq!(
        rank_of_block(&blk, &s_ao, 1e-9),
        subset.len(),
        "orthonormal subset must have full rank"
    );

    // NEGATIVE CONTROL: duplicate a column. rank MUST drop below ncols, else
    // `rank_of_block` is a fancy way of writing `ncols()` and every union number
    // in the gate is fabricated.
    let mut dup: Vec<usize> = subset.clone();
    dup.push(subset[0]);
    let blk_dup = select_cols(&sys.c_vir_loc, &dup);
    let r_dup = rank_of_block(&blk_dup, &s_ao, 1e-9);
    println!(
        "negative control: {} columns with one duplicate -> rank {}",
        dup.len(),
        r_dup
    );
    assert_eq!(
        r_dup,
        dup.len() - 1,
        "a duplicated column must reduce the measured rank by exactly 1; \
         rank_of_block is not measuring rank"
    );
}

// ===========================================================================
// RETRACTION (2026-07-27, same day) — THIS GATE IS A TAUTOLOGY
// ===========================================================================
//
// The NO-GO verdict this file reports is NOT a measurement. The gate could not
// have returned GO for any molecule, basis, radius, or assignment rule.
//
// Every domain subspace V_d here is a SUBSET OF ONE ORTHONORMAL SET (teeth #4
// asserts CᵀSC = I at full rank). Subsets of an orthonormal set are mutually
// orthogonal unless they share identical members, so
//
//     rank(U V_d)  ==  n_vir - n_uncovered      (identically, always)
//
// Verified against every row this file printed: 89-39=50, 89-4=85, 89-0=89,
// 75-57=18, 75-0=75. The SVD machinery rigorously re-measures a counting fact.
//
// The decision rule (see `real` at the GO/NO-GO classification below) requires
// BOTH `ratio <= 0.9` AND `n_uncovered == 0`. By the identity, n_uncovered == 0
// forces ratio == 1.000, which fails ratio <= 0.9. THE TWO CONDITIONS ARE
// MUTUALLY EXCLUSIVE BY CONSTRUCTION.
//
// The "exact orthogonal equality" (rank(U) == sum_d rank(V_d)) reported as
// reproducing the RPA obstruction "on the nose" is likewise forced: at r = 1.0
// Bohr the min interatomic distance in alkane_4/alkane_8 is 2.067 Bohr > 2r, so
// no centroid can lie within the radius of two atoms. Bond geometry, not
// physics. At r = 2.0 the equality already breaks (85 vs 130).
//
// DEEPER SCOPE ERROR: subsets of one orthonormal basis can only "overlap" by
// sharing identical members, so this construction removed the very degree of
// freedom the PNO-union question is about -- pair-specific NON-ORTHOGONAL bases
// whose ranges can genuinely nest. Requiring the union to compress at all is
// the RPA-DIELECTRIC requirement; DLPNO-MP2/CC never need it, since they pay
// per-pair costs in per-pair bases. And "union rank grows linearly with system
// size" is the linear-scaling SUCCESS signature (N domains of O(1) size), not a
// failure.
//
// KEPT, NOT DELETED, because the harness (orthonormality checks, SVD rank, the
// duplicate-column negative control) is reusable and because a future reader
// will otherwise re-derive the same tautology. To make this a real gate it must
// use NON-ORTHOGONAL, PAIR-SPECIFIC bases, where rank(U) is genuinely not a
// counting identity.
