//! Hirshfeld volumes, volume gradients and charges on grids.
//!
//! These need an SCF density and a proatom provider, not PDEP eigenpairs, so
//! they live beside the other population-partition properties of
//! [`super`]. `ferric_rpa::properties` re-exports every public item here.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ndarray::Array2;

use super::{
    debug_toggle, hirshfeld_margin, hirshfeld_spacing, slater_xi_for_z, ProatomProvider,
    RadialProatom,
};

/// Basis-function count for `bs` on `mol`, without materializing an AO-on-grid
/// tensor to read it off `chi.nrows()`.
///
/// `hirshfeld_i_charges`, `atomic_effective_volumes_hirshfeld` and
/// `atomic_effective_volumes_hirshfeld_on_grid` need `nbf` BEFORE evaluating
/// the AOs so their pre-flight gate can run before `chi` allocates (the point
/// at which refusing is still cheap — the same convention
/// `preflight_grid_path`'s doc states for the Becke path).
///
/// Delegates to `ferric_integrals::ao_grid::nbasis`, the canonical shell-sum
/// (already used by `PreparedBasis::nbasis`), rather than re-deriving it —
/// re-deriving it with a `filter_map` over `bs.for_element` would silently
/// DROP any atom whose element is missing from the basis instead of erroring,
/// which would under-count `nbf` and defeat the whole point of gating before
/// `chi` allocates (a silently wrong, too-small charge is worse than no
/// charge, since it looks like coverage that is not there).
pub fn nbf_for_basis(
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
) -> Result<usize, FerricError> {
    Ok(ferric_integrals::ao_grid::nbasis(mol, bs)?)
}

/// Pure byte estimate for [`preflight_hirshfeld_grid_scan`], split out so it
/// can be unit-tested against a hand-derived figure at an arbitrary shape
/// without needing to control the ambient resolved memory budget (mirrors the
/// `estimate_peak_bytes` / `check_alloc` split used throughout `budget.rs`).
///
/// Co-resident terms: `chi` and `d_chi` (both `(nbf, npts)`, genuinely read in
/// the same loop), `rho_free` (`(natoms, npts)`), and the `O(npts)` side
/// vectors (`rho`, `rho_sum`, and for `hirshfeld_i_charges` also `gx`/`gy`/
/// `gz` — 5 vectors covers every caller without under-charging any).
///
/// `atomic_effective_volumes_hirshfeld` charges this shape at `npts =
/// HIRSHFELD_VOLUME_CHUNK_POINTS` (its real peak — it holds no whole-grid
/// plane) and keeps only a `natoms` proatom scratch rather than a
/// `(natoms, npts)` block, so for that caller the `rho_free` term is slack,
/// deliberately: over-charging refuses a run that would have fitted,
/// under-charging admits one that will not.
pub fn estimate_hirshfeld_grid_scan_bytes(nbf: usize, npts: usize, natoms: usize) -> usize {
    const F64_BYTES: usize = 8;
    let plane = nbf.saturating_mul(npts).saturating_mul(F64_BYTES);
    let rho_free = natoms.saturating_mul(npts).saturating_mul(F64_BYTES);
    let side_vectors = npts.saturating_mul(5).saturating_mul(F64_BYTES);
    // chi + d_chi (two planes) + rho_free + side vectors.
    plane
        .saturating_add(plane)
        .saturating_add(rho_free)
        .saturating_add(side_vectors)
}

/// Pre-flight gate for [`atomic_effective_volumes_hirshfeld`],
/// [`atomic_effective_volumes_hirshfeld_on_grid`] and
/// [`hirshfeld_i_charges`], called BEFORE the AO values allocate `chi`.
///
/// The only gate these functions had before this was the one INSIDE
/// `eval_basis_on_grid` (`ferric_integrals::ao_grid::eval_basis_on_grid` ->
/// `check_alloc` on `nbf*npts*8`, i.e. `chi` alone) — a callee-gate-cannot-
/// see-caller defect: the callee has no way to know its caller immediately
/// builds a second full `(nbf, npts)` block (`d_chi = density.dot(&chi)`,
/// read in the SAME loop as `chi` so genuinely co-resident) plus a
/// `(natoms, npts)` `rho_free` and several `O(npts)` side vectors on top. That
/// callee gate also always resolves `resolve_budget_bytes(None)`, so it never
/// sees whatever budget ferric-cli's `[memory] budget_gb` resolved for this
/// call either way — a pre-existing, separate gap this preflight does not
/// change (no `memory_budget_bytes` parameter is threaded to these two public
/// functions today; adding one would change their signatures for every
/// caller, out of scope here).
///
/// Measured at nbf=200, npts≈5.6e6 (a ~24 Bohr molecule under the default
/// 6-Bohr-margin/0.20-spacing box), natoms=20, `[memory] budget_gb=10`: the
/// callee's own gate approves `chi` alone (9.0 GB) against the 10 GB budget;
/// the actual peak once `d_chi` (+9.0 GB) and `rho_free` (+0.9 GB) are added
/// is ~18.9 GB — essentially 2x what was charged, on the export_npz default
/// chain implicated in the 2026-07-13 16-17 GB anon-RSS incidents.
pub fn preflight_hirshfeld_grid_scan(
    label: &str,
    nbf: usize,
    npts: usize,
    natoms: usize,
) -> Result<(), FerricError> {
    preflight_hirshfeld_grid_scan_reserved(label, nbf, npts, natoms).map(|_| ())
}

/// [`preflight_hirshfeld_grid_scan`], but hands back the [`Reservation`](ferric_core::memory::pool::Reservation) that
/// admitted the buffers so the OWNER can hold the charge for their whole
/// lifetime.
///
/// Mirrors `ferric_integrals::ao_grid::check_ao_grid_budget_reserved`, and for
/// the same reason: `check_*` on its own debits the pool, returns, and the
/// guard drops at the end of the statement -- so the bytes are credited back
/// while `chi` and `d_chi` are still resident, and the next plane to ask sees
/// a pool that looks empty. That is decoration, not accounting.
///
/// # Hard, not soft
///
/// `chi` is a single contiguous `(nbf, npts)` block built by one
/// `eval_basis_on_grid` call, and `d_chi = density.dot(&chi)` is a full GEMM
/// against it. Neither has a chunked variant on this path, so a `try_reserve`
/// whose `None` branch could only shrug would be a lie. This refuses.
///
/// The measured shape from this function's own doc (nbf=200, npts≈5.6e6,
/// natoms=20): ~18.9 GB, against a callee gate that approved 9.0 GB.
pub fn preflight_hirshfeld_grid_scan_reserved(
    label: &str,
    nbf: usize,
    npts: usize,
    natoms: usize,
) -> Result<ferric_core::memory::pool::Reservation, FerricError> {
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let est = estimate_hirshfeld_grid_scan_bytes(nbf, npts, natoms);
    ferric_core::memory::check_alloc(label, est, budget)?;
    ferric_core::memory::pool::reserve_global(label, est)
}

/// The Becke–Lebedev grid [`atomic_effective_volumes_hirshfeld`] integrates on.
///
/// NOT `AtomicGridConfig::default()` (75 radial × 110 angular, the XC grid).
/// The volume integrand `w_A ρ |r − R_A|³` is NOT the XC integrand: the `r³`
/// factor moves its mass outward, into the region where the Hirshfeld weight
/// `ρ⁰_A/Σ_B ρ⁰_B` and the Becke home-atom factor both vary fastest with
/// ANGLE, so the volumes need a higher Lebedev order than the energy does.
///
/// MEASURED (`tests/measure_volume_grid_convergence.rs`, worst per-atom
/// relative deviation from the dense (200, 590) reference of
/// `testdata/reference/validation/hirshfeld/`, over H2O, CO and CH3OH ×
/// cc-pVDZ, def2-SVP, at 75 radial points):
///
/// | Lebedev order | H2O | CO | CH3OH | CH3OH points |
/// |---|---|---|---|---|
/// | 110 (the XC default) | 1.5e-5 | 1.7e-5 | 6.9e-4 | 49,500 |
/// | 194 | 1.0e-6 | 9.7e-7 | 2.0e-4 | 87,300 |
/// | 302 | 1.0e-7 | 6.6e-8 | 4.4e-5 | 135,900 |
/// | 434 | 1.3e-7 | 1.1e-7 | 2.6e-6 | 195,300 |
/// | **590** | **1.1e-7** | **1.1e-7** | **1.1e-7** | **265,500** |
///
/// 590 is where all three systems reach one plateau, ~1.1e-7, and that
/// plateau is the RADIAL floor, not the angular one: 99 radial × 590 drops
/// CH3OH to 1.8e-8 and 200 × 590 (the reference's own grid) to 3.0e-13, so
/// ferric's quadrature is an exact reimplementation of the reference
/// construction and 590 is simply where the cheap axis stops paying.
///
/// Raising only the RADIAL count does nothing at all: 75 → 200 radial at
/// order 110 leaves CH3OH at 6.927e-4 to four digits. The volume error is
/// ANGULAR because `w_A` and the Becke home-atom factor vary fastest with
/// angle in the region `r³` weights up — which is exactly why the XC grid's
/// 110 is not a safe default here even though it is for the energy.
///
/// Cost: 5.4x the XC grid's point count on CH3OH, once per volume call (a
/// post-SCF property, not an SCF iteration), against a 6300x accuracy gain.
pub fn hirshfeld_volume_grid_config() -> ferric_dft::grid::AtomicGridConfig {
    ferric_dft::grid::AtomicGridConfig {
        n_radial: 75,
        n_angular: 590,
        // Pruning reduces the angular order exactly where this integrand needs
        // it most (the far tail, which `r³` weights up).
        prune: None,
        weight_cut: None,
    }
}

/// Grid points per chunk of [`atomic_effective_volumes_hirshfeld`]'s
/// Becke-grid loop. A constant (not derived from the thread count or the
/// budget), so the chunk partition — and with it the order of the final fold —
/// is a pure function of the grid, exactly as
/// [`HIRSHFELD_GRAD_CHUNK_POINTS`] is for the gradient.
///
/// The whole point of chunking here is that `chi` and `D·chi` are both
/// `(nbf, npts)`: on the 75×110 default grid a 73-atom, 700-function molecule
/// puts 602,250 points × 700 functions × 8 B = 3.4 GB in EACH, and neither is
/// bounded by `[memory] budget_gb` (see
/// `ferric_scf::properties::atomic_effective_volumes_becke_chunked`, which
/// chunks the same shape for the same reason, and the CLAUDE.md
/// "memory ceiling" note behind the 2026-07-13 16-17 GB incidents).
pub const HIRSHFELD_VOLUME_CHUNK_POINTS: usize = 8192;

/// Per-atom effective volume via Hirshfeld partitioning:
/// ```text
///   v_A = ∫ w^A_Hirsh(r) ρ(r) |r − R_A|³ dV,   w^A = ρ⁰_A / (Σ_B ρ⁰_B + 1e-12)
/// ```
/// This is the partition the TS dispersion model was calibrated for.
///
/// `proatom` supplies each atom's neutral free-atom density ρ⁰_A (ferric-cli
/// passes [`scf_proatom_provider`](super::scf_proatom_provider)); with `None`, or for
/// an atom the provider returns `None` for, ρ⁰_A is the single-Slater proatom of
/// [`slater_xi_for_z`].
///
/// Integrated on the **atom-centred Becke–Lebedev grid** — the same
/// [`ferric_dft::grid::build_atomic_grid`] quadrature [`hirshfeld_charges`]
/// uses, with the home-atom Becke factor already folded into each point's
/// weight. The grid is radially dense at every nucleus and carries the
/// molecular point group, so symmetry-equivalent atoms integrate identically.
///
/// MEASURED against the dense (200, 590) Becke reference in
/// `testdata/reference/validation/hirshfeld/`, over H2O, CO and CH3OH ×
/// cc-pVDZ, def2-SVP, same density and proatoms: worst per-atom deviation
/// 1.1e-7 relative, ∫ρ within 2.0e-7 e of N_e, and CH3OH's two mirror-image
/// methyl hydrogens equal to 2.9e-14 relative (not exactly zero: the Lebedev
/// rules are O_h-invariant and that molecule's mirror plane is not a grid
/// plane). On a uniform
/// Cartesian lattice — whose nodes land
/// exactly on nuclei, where ρ has its cusp — the same three figures are
/// 2.4e-4 relative, 0.70 e (CO/cc-pVDZ) and 2.6e-5. The grid is
/// [`hirshfeld_volume_grid_config`], which is NOT the XC grid; read its doc
/// before changing it.
///
/// The lattice is still reachable as
/// [`atomic_effective_volumes_hirshfeld_on_grid`] on
/// [`hirshfeld_volume_grid`], which is what MBD@rsSCS integrates on (its
/// gradient's lattice-response term is derived for a lattice; see
/// [`mbd_volume_grid`]).
///
/// Evaluated in [`HIRSHFELD_VOLUME_CHUNK_POINTS`]-point chunks so neither
/// `chi` nor `D·chi` is ever materialized for the whole grid.
pub fn atomic_effective_volumes_hirshfeld(
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    density: &Array2<f64>,
    proatom: Option<&ProatomProvider>,
) -> Result<Vec<f64>, FerricError> {
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ferric_dft::grid::build_atomic_grid;

    let natoms = mol.atoms.len();
    let grid = build_atomic_grid(mol, &hirshfeld_volume_grid_config());
    let npts = grid.len();

    let nbf = density.nrows();
    if density.ncols() != nbf {
        return Err(FerricError::General(format!(
            "atomic_effective_volumes_hirshfeld: density shape {:?} is not square",
            density.dim()
        )));
    }
    let nbf_expected = nbf_for_basis(mol, obs_bs)?;
    if nbf != nbf_expected {
        return Err(FerricError::General(format!(
            "atomic_effective_volumes_hirshfeld: density {:?} != nbf {nbf_expected}",
            density.dim()
        )));
    }

    // Pre-flight the CHUNK, not the whole grid: that is the real peak now
    // (`chi` and `d_chi` are both (nbf, chunk)), and charging the full-grid
    // shape would refuse molecules this path can comfortably integrate.
    let chunk = HIRSHFELD_VOLUME_CHUNK_POINTS.min(npts.max(1));
    let _charge = preflight_hirshfeld_grid_scan_reserved(
        &format!(
            "atomic_effective_volumes_hirshfeld (nbf={nbf}, npts={npts}, chunk={chunk}, \
             natoms={natoms})"
        ),
        nbf,
        chunk,
        natoms,
    )?;

    // Proatom tables are fetched ONCE per atom, outside the point loop: the
    // provider runs a free-atom SCF behind the scenes on a miss, so calling it
    // per chunk would re-solve every atom O(npts/chunk) times.
    let mut fallback_atoms: Vec<usize> = Vec::new();
    let proatoms: Vec<Option<RadialProatom>> = (0..natoms)
        .map(|a| {
            let pa = proatom.and_then(|p| p(mol.atoms[a].z, 0));
            if pa.is_none() {
                fallback_atoms.push(a);
            }
            pa
        })
        .collect();
    if !fallback_atoms.is_empty() && fallback_atoms.len() < natoms {
        // Mixing SCF proatoms with the crude Slater model across atoms of one
        // molecule changes the partitioning, and the returned volumes carry no
        // sign of it. The all-fallback case is NOT a mixture (it is what
        // `proatom = None` means — the free-atom TS denominator path uses it on
        // purpose), so it is not warned about here.
        eprintln!(
            "[hirshfeld] WARNING: {} of {natoms} atoms (indices {fallback_atoms:?}) had no \
             free-atom SCF proatom density and fell back to a crude single-Slater model, so \
             these volumes mix two different proatom sources.",
            fallback_atoms.len()
        );
    }
    let xi: Vec<f64> = (0..natoms)
        .map(|a| slater_xi_for_z(mol.atoms[a].z))
        .collect();
    let prefac: Vec<f64> = (0..natoms)
        .map(|a| mol.atoms[a].z as f64 * xi[a].powi(3) / std::f64::consts::PI)
        .collect();
    let pos: Vec<[f64; 3]> = mol.atoms.iter().map(|at| [at.x, at.y, at.zpos]).collect();

    const EPS_FLOOR: f64 = 1e-12;
    let mut vol = vec![0.0_f64; natoms];
    let mut g0 = 0usize;
    // Scratch reused across chunks so the per-atom proatom values are not
    // reallocated 70+ times per chunk.
    let mut rho0 = vec![0.0_f64; natoms];
    while g0 < npts {
        let g1 = (g0 + chunk).min(npts);
        let points: Vec<[f64; 3]> = grid[g0..g1].iter().map(|g| g.xyz).collect();
        let chi = eval_basis_on_points(mol, obs_bs, &points).map_err(|e| {
            FerricError::General(format!(
                "atomic_effective_volumes_hirshfeld: chi eval failed: {e}"
            ))
        })?;
        // ρ(r_g) = Σ_μ χ_μ(g) · (D·χ)(μ,g). g is a FREE index of this GEMM
        // (the reduction is over the shared AO index), so splitting the grid
        // splits independent output COLUMNS and leaves the accumulation lanes
        // untouched — the same argument `becke_charges_chunked` spells out.
        let d_chi = density.dot(&chi);
        for (gc, pt) in points.iter().enumerate() {
            let g = g0 + gc;
            let mut rho = 0.0_f64;
            for mu in 0..nbf {
                rho += chi[(mu, gc)] * d_chi[(mu, gc)];
            }
            // Σ_B ρ⁰_B(r) first: the Hirshfeld weight needs the whole
            // promolecule before any atom's share can be taken.
            let mut rho_sum = 0.0_f64;
            for a in 0..natoms {
                let dx = pt[0] - pos[a][0];
                let dy = pt[1] - pos[a][1];
                let dz = pt[2] - pos[a][2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                let r0 = match &proatoms[a] {
                    Some(p) => p.at(r),
                    None => prefac[a] * (-2.0 * xi[a] * r).exp(),
                };
                rho0[a] = r0;
                rho_sum += r0;
            }
            let inv = 1.0 / (rho_sum + EPS_FLOOR);
            let w_rho = grid[g].weight * rho;
            for a in 0..natoms {
                let dx = pt[0] - pos[a][0];
                let dy = pt[1] - pos[a][1];
                let dz = pt[2] - pos[a][2];
                let r3 = (dx * dx + dy * dy + dz * dz).powf(1.5);
                vol[a] += rho0[a] * inv * w_rho * r3;
            }
        }
        g0 = g1;
    }
    Ok(vol)
}

/// A uniform Cartesian lattice around the molecule:
/// `GridSpec::bounding_box(mol, hirshfeld_margin(), hirshfeld_spacing())`
/// (`FERRIC_HIRSHFELD_MARGIN`, default 6 Bohr; `FERRIC_HIRSHFELD_SPACING`,
/// default 0.20 Bohr).
///
/// NOT what [`atomic_effective_volumes_hirshfeld`] integrates on — that is the
/// atom-centred Becke–Lebedev grid. This lattice's origin is `min − margin`
/// with `margin/h = 30` an integer, so the extreme atom on each axis sits
/// exactly on a node, where ρ has its cusp: ∫ρ on it misses N_e by 0.64 e for
/// CO/def2-SVP and the volumes miss a dense Becke reference by 2.4e-4
/// relative, differently per atom (CH3OH's mirror-image hydrogens by 2.6e-5).
/// It remains the quadrature `pdep_polarizability_hirshfeld` and
/// `hirshfeld_i_charges` use, and the lattice
/// [`atomic_effective_volumes_hirshfeld_on_grid`] accepts.
pub fn hirshfeld_volume_grid(mol: &Molecule) -> ferric_integrals::ao_grid::GridSpec {
    ferric_integrals::ao_grid::GridSpec::bounding_box(mol, hirshfeld_margin(), hirshfeld_spacing())
}

/// The lattice MBD@rsSCS integrates its Hirshfeld volumes on: the same
/// spacing and margin as [`hirshfeld_volume_grid`], but anchored to the
/// CENTROID c of the nuclei instead of the bounding box. Points are
/// `c + i·h` along each axis, `i = −m..=m`, `m = ceil((max_A |R_A − c| + margin)/h)`.
///
/// Why: on the bounding-box lattice the points follow whichever atom is
/// extreme on each axis, so the volumes are not differentiable where two atoms
/// tie for the extreme (water's two H share z). Here every point moves by
/// exactly `1/N` of an atom's displacement, so the lattice response of the
/// volumes is the smooth term `−(1/N) Σ_B ∂v/∂R_B|_lattice fixed` (translation
/// invariance at fixed D). A change of `m` re-labels the points and adds or
/// drops an edge row at least `margin` from every nucleus, so the volumes stay
/// continuous.
pub fn mbd_volume_grid(mol: &Molecule) -> ferric_integrals::ao_grid::GridSpec {
    let h = hirshfeld_spacing();
    let margin = hirshfeld_margin();
    let n = mol.atoms.len().max(1) as f64;
    let mut c = [0.0_f64; 3];
    for a in &mol.atoms {
        c[0] += a.x / n;
        c[1] += a.y / n;
        c[2] += a.zpos / n;
    }
    let mut origin = [0.0_f64; 3];
    let mut counts = [1usize; 3];
    for k in 0..3 {
        let half = mol
            .atoms
            .iter()
            .map(|a| ([a.x, a.y, a.zpos][k] - c[k]).abs())
            .fold(0.0_f64, f64::max);
        let m = ((half + margin) / h).ceil() as usize;
        origin[k] = c[k] - m as f64 * h;
        counts[k] = 2 * m + 1;
    }
    ferric_integrals::ao_grid::GridSpec {
        origin,
        n_x: counts[0],
        n_y: counts[1],
        n_z: counts[2],
        step_x: [h, 0.0, 0.0],
        step_y: [0.0, h, 0.0],
        step_z: [0.0, 0.0, h],
    }
}

/// Reject a lattice the Hirshfeld volume loops cannot integrate on: they read
/// only the diagonal step components (`step_x[0]`, `step_y[1]`, `step_z[2]`)
/// and use their product as the cell volume, so the lattice must be
/// axis-aligned with positive steps.
fn check_axis_aligned_grid(
    label: &str,
    grid: &ferric_integrals::ao_grid::GridSpec,
) -> Result<(), FerricError> {
    let off_diag = [
        grid.step_x[1],
        grid.step_x[2],
        grid.step_y[0],
        grid.step_y[2],
        grid.step_z[0],
        grid.step_z[1],
    ];
    let diag = [grid.step_x[0], grid.step_y[1], grid.step_z[2]];
    if off_diag.iter().any(|&v| v != 0.0) || diag.iter().any(|&h| !(h.is_finite() && h > 0.0)) {
        return Err(FerricError::General(format!(
            "{label}: the integration lattice must be axis-aligned with positive steps \
             (got step_x={:?}, step_y={:?}, step_z={:?})",
            grid.step_x, grid.step_y, grid.step_z
        )));
    }
    Ok(())
}

/// The same Hirshfeld volume integral as [`atomic_effective_volumes_hirshfeld`],
/// but on a caller-supplied uniform **lattice** `grid` (axis-aligned; cell
/// volume `step_x[0]·step_y[1]·step_z[2]`) instead of the atom-centred
/// Becke–Lebedev grid. Holding one `grid` fixed while atoms move is the
/// "lattice fixed" convention of [`hirshfeld_volume_gradient`], which is why
/// MBD@rsSCS integrates here, on [`mbd_volume_grid`].
///
/// A lattice cannot resolve the nuclear cusp and does not carry the molecular
/// point group, so this quadrature is the less accurate of the two: see
/// [`hirshfeld_volume_grid`] for the measured errors. Use it only when the
/// caller needs a geometry-independent point set (i.e. a gradient whose
/// lattice-response term is derived for one).
pub fn atomic_effective_volumes_hirshfeld_on_grid(
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    density: &Array2<f64>,
    proatom: Option<&ProatomProvider>,
    grid: &ferric_integrals::ao_grid::GridSpec,
) -> Result<Vec<f64>, FerricError> {
    use ferric_integrals::ao_grid::eval_basis_on_grid;

    check_axis_aligned_grid("atomic_effective_volumes_hirshfeld", grid)?;
    let natoms = mol.atoms.len();
    let dv = grid.step_x[0] * grid.step_y[1] * grid.step_z[2];
    let npts = grid.n_x * grid.n_y * grid.n_z;
    let hx = grid.step_x[0];
    let hy = grid.step_y[1];
    let hz = grid.step_z[2];

    // Pre-flight BEFORE `chi` allocates: `nbf` is available from the basis
    // without materializing the grid tensor (see `nbf_for_basis`'s doc for
    // why the callee's own gate is not enough here).
    let nbf_pre = nbf_for_basis(mol, obs_bs)?;
    // `_charge` bound at FUNCTION scope: `chi`, `d_chi` and `rho_free` below
    // are all live to the end of this call.
    let _charge = preflight_hirshfeld_grid_scan_reserved(
        &format!(
            "atomic_effective_volumes_hirshfeld (nbf={nbf_pre}, npts={npts}, natoms={natoms})"
        ),
        nbf_pre,
        npts,
        natoms,
    )?;

    let chi = eval_basis_on_grid(mol, obs_bs, grid).map_err(|e| {
        FerricError::General(format!(
            "atomic_effective_volumes_hirshfeld: chi failed: {e}"
        ))
    })?;
    let nbf = chi.nrows();
    debug_assert_eq!(
        nbf, nbf_pre,
        "nbf_for_basis must match eval_basis_on_grid's chi.nrows()"
    );

    // ρ(r_g) = Σ_{μν} D_{μν} χ_μ χ_ν via matrix product.
    let d_chi = density.dot(&chi);
    let mut rho = vec![0.0_f64; npts];
    for mu in 0..nbf {
        for g in 0..npts {
            rho[g] += chi[(mu, g)] * d_chi[(mu, g)];
        }
    }

    // Proatom weights: validated same-basis density when a provider is given,
    // else legacy single-Slater.
    let mut rho_free: Vec<Vec<f64>> = vec![vec![0.0; npts]; natoms];
    let mut rho_sum = vec![0.0_f64; npts];
    for a in 0..natoms {
        let z_a = mol.atoms[a].z;
        let xi = slater_xi_for_z(z_a);
        let prefac = z_a as f64 * xi.powi(3) / std::f64::consts::PI;
        let pa = proatom.and_then(|p| p(z_a, 0));
        let (rax, ray, raz) = (mol.atoms[a].x, mol.atoms[a].y, mol.atoms[a].zpos);
        let row = &mut rho_free[a];
        for ix in 0..grid.n_x {
            let x = grid.origin[0] + ix as f64 * hx;
            for iy in 0..grid.n_y {
                let y = grid.origin[1] + iy as f64 * hy;
                for iz in 0..grid.n_z {
                    let z = grid.origin[2] + iz as f64 * hz;
                    let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                    let r = ((x - rax) * (x - rax) + (y - ray) * (y - ray) + (z - raz) * (z - raz))
                        .sqrt();
                    let r0 = match &pa {
                        Some(p) => p.at(r),
                        None => prefac * (-2.0 * xi * r).exp(),
                    };
                    row[g] = r0;
                    rho_sum[g] += r0;
                }
            }
        }
    }

    let eps_floor = 1e-12;
    let mut vol = vec![0.0_f64; natoms];
    for a in 0..natoms {
        let (rax, ray, raz) = (mol.atoms[a].x, mol.atoms[a].y, mol.atoms[a].zpos);
        let mut acc = 0.0;
        for ix in 0..grid.n_x {
            let x = grid.origin[0] + ix as f64 * hx;
            for iy in 0..grid.n_y {
                let y = grid.origin[1] + iy as f64 * hy;
                for iz in 0..grid.n_z {
                    let z = grid.origin[2] + iz as f64 * hz;
                    let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                    let w = rho_free[a][g] / (rho_sum[g] + eps_floor);
                    let dx = x - rax;
                    let dy = y - ray;
                    let dz = z - raz;
                    let r3 = (dx * dx + dy * dy + dz * dz).powf(1.5);
                    acc += w * rho[g] * r3 * dv;
                }
            }
        }
        vol[a] = acc;
    }
    Ok(vol)
}

/// Lattice points per chunk of [`hirshfeld_volume_gradient`]. A constant (not
/// derived from the thread count or the budget), so the chunk partition — and
/// with it the order of the final fold — is a pure function of the lattice.
pub const HIRSHFELD_GRAD_CHUNK_POINTS: usize = 1024;

/// Distance (Bohr) below which [`hirshfeld_volume_gradient`] treats a lattice
/// point as sitting on a nucleus, where the proatom gradient direction
/// `d_B / r_B` is undefined (the Slater proatom's cusp).
const NUCLEUS_TOL: f64 = 1e-9;

/// Bytes [`hirshfeld_volume_gradient`] holds at its peak, split out so the
/// formula can be checked by hand. Per chunk in flight: `chi`, the three
/// `∇chi` planes and `D·chi` (5 × `(nbf, chunk)`), plus the point list and
/// the per-point scalars (`ρ`, `q`, 3 coordinates — 5 × `chunk`). Shared:
/// the symmetrized density `(nbf, nbf)` and one `(natoms, 3)` partial per
/// chunk.
pub fn estimate_hirshfeld_volume_gradient_bytes(
    nbf: usize,
    natoms: usize,
    chunk_points: usize,
    n_chunks: usize,
    workers: usize,
) -> usize {
    const F64_BYTES: usize = 8;
    let per_chunk = nbf
        .saturating_mul(5)
        .saturating_add(5)
        .saturating_mul(chunk_points)
        .saturating_mul(F64_BYTES);
    let d_sym = nbf.saturating_mul(nbf).saturating_mul(F64_BYTES);
    let partials = n_chunks
        .saturating_mul(natoms)
        .saturating_mul(3)
        .saturating_mul(F64_BYTES);
    per_chunk
        .saturating_mul(workers)
        .saturating_add(d_sym)
        .saturating_add(partials)
}

/// Nuclear derivative of the contracted Hirshfeld volumes of
/// [`atomic_effective_volumes_hirshfeld_on_grid`]:
/// ```text
///   G[B, :] = Σ_A de_dv[A] · ∂v_A/∂R_B
/// ```
/// at FIXED AO density matrix `D` and FIXED lattice `grid`: basis functions
/// and proatoms move with their atoms, the lattice points do not. This is the
/// exact derivative of the quadrature `atomic_effective_volumes_hirshfeld_on_grid`
/// evaluates on the same `grid`, not of the continuous
/// integral: there is no lattice-response term, so Σ_B G[B, :] need not vanish.
///
/// With `v_A = Σ_g dV w_A ρ |r−R_A|³`, `w_A = ρ⁰_A / (S + ε)`,
/// `S = Σ_C ρ⁰_C`, `ε = 1e-12` (the energy's floor), `d_C = r − R_C`,
/// `r_C = |d_C|` and `q(g) = Σ_A de_dv[A] w_A r_A³`, each lattice point adds
/// ```text
///   (i)   −2 dV q Σ_{μ on B} ∇χ_μ (D_s χ)_μ           ∂ρ/∂R_B, D_s = (D + Dᵀ)/2
///   (ii)  dV ρ (de_dv[B] r_B³ − q) g_B / (S + ε)       ∂w_A/∂R_B = (δ_AB − w_A) g_B/(S+ε)
///   (iii) −3 dV ρ de_dv[B] w_B r_B d_B                 ∂r_B³/∂R_B
/// ```
/// where `g_B = ∂ρ⁰_B/∂R_B = −ρ⁰_B'(r_B) d_B / r_B` (0 at `r_B = 0`, where the
/// Slater profile has a cusp and the tabulated proatom, an even function of
/// r, is flat).
/// `ρ⁰_B'` is [`RadialProatom::deriv`] for a provider proatom and
/// `−2ξ ρ⁰_B` for the Slater fallback; each atom uses the same proatom the
/// energy uses (`proatom(z, 0)`, else the [`slater_xi_for_z`] Slater). The
/// provider is called once per distinct element.
///
/// Evaluated in [`HIRSHFELD_GRAD_CHUNK_POINTS`]-point lattice chunks, in
/// parallel; per-chunk `(natoms, 3)` partials are folded in ascending chunk
/// order, so the result does not depend on the thread count.
///
/// # Errors
///
/// `de_dv` or `density` of the wrong size, a non-axis-aligned `grid`, an AO
/// evaluation failure, or a peak (see
/// [`estimate_hirshfeld_volume_gradient_bytes`]) above the memory budget.
pub fn hirshfeld_volume_gradient(
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    density: &Array2<f64>,
    proatom: Option<&ProatomProvider>,
    grid: &ferric_integrals::ao_grid::GridSpec,
    de_dv: &[f64],
) -> Result<Array2<f64>, FerricError> {
    use ferric_integrals::ao_grid::{collect_shells, eval_shell_and_grad};
    use rayon::prelude::*;

    const LABEL: &str = "hirshfeld_volume_gradient";
    check_axis_aligned_grid(LABEL, grid)?;
    let natoms = mol.atoms.len();
    if de_dv.len() != natoms {
        return Err(FerricError::General(format!(
            "{LABEL}: de_dv has {} entries for {natoms} atoms",
            de_dv.len()
        )));
    }

    // AO → atom, in collect_shells' order (atom-major, shell, function).
    let shells =
        collect_shells(mol, obs_bs).map_err(|e| FerricError::General(format!("{LABEL}: {e}")))?;
    let mut ao_atom: Vec<usize> = Vec::new();
    for (a, atom) in mol.atoms.iter().enumerate() {
        let atom_shells = obs_bs
            .for_element(atom.z)
            .ok_or_else(|| FerricError::General(format!("{LABEL}: no basis for Z={}", atom.z)))?;
        for sh in atom_shells {
            let n = ferric_core::basis::num_functions(sh.l, sh.pure);
            ao_atom.extend(std::iter::repeat_n(a, n));
        }
    }
    let nbf = ao_atom.len();
    if density.nrows() != nbf || density.ncols() != nbf {
        return Err(FerricError::General(format!(
            "{LABEL}: density {:?} does not match nbf {nbf}",
            density.dim()
        )));
    }
    let shell_nf: Vec<usize> = shells
        .iter()
        .map(|s| ferric_core::basis::num_functions(s.l, s.pure))
        .collect();

    let hx = grid.step_x[0];
    let hy = grid.step_y[1];
    let hz = grid.step_z[2];
    let dv = hx * hy * hz;
    let npts = grid.n_x * grid.n_y * grid.n_z;
    let chunk = HIRSHFELD_GRAD_CHUNK_POINTS;
    let chunk_starts: Vec<usize> = (0..npts).step_by(chunk).collect();
    let n_chunks = chunk_starts.len();

    // Thread count enters only the memory estimate (how many chunks can be
    // in flight), never the partition or the fold.
    let workers = rayon::current_num_threads().max(1).min(n_chunks.max(1));
    let est = estimate_hirshfeld_volume_gradient_bytes(nbf, natoms, chunk, n_chunks, workers);
    let label = format!(
        "{LABEL} (nbf={nbf}, npts={npts}, natoms={natoms}, chunk={chunk}, workers={workers})"
    );
    ferric_core::memory::check_alloc(&label, est, ferric_core::memory::resolve_budget_bytes(None))?;
    // Held for the whole call: every buffer it charges is live until return.
    let _charge = ferric_core::memory::pool::reserve_global(&label, est)?;

    let d_sym = {
        let mut d = density.to_owned();
        d += &density.t();
        d *= 0.5;
        d
    };

    // Per-atom proatom (same choice as the energy), one provider call per Z.
    let mut by_z: std::collections::BTreeMap<i32, Option<RadialProatom>> =
        std::collections::BTreeMap::new();
    let mut pro: Vec<Option<RadialProatom>> = Vec::with_capacity(natoms);
    let mut slater: Vec<(f64, f64)> = Vec::with_capacity(natoms); // (xi, prefac)
    let mut pos: Vec<[f64; 3]> = Vec::with_capacity(natoms);
    for atom in &mol.atoms {
        let z_a = atom.z;
        let pa = by_z
            .entry(z_a)
            .or_insert_with(|| proatom.and_then(|p| p(z_a, 0)))
            .clone();
        pro.push(pa);
        let xi = slater_xi_for_z(z_a);
        slater.push((xi, z_a as f64 * xi.powi(3) / std::f64::consts::PI));
        pos.push([atom.x, atom.y, atom.zpos]);
    }

    let eps_floor = 1e-12;
    let (n_y, n_z) = (grid.n_y, grid.n_z);
    let origin = grid.origin;

    let partials: Vec<Vec<[f64; 3]>> = chunk_starts
        .par_iter()
        .map(|&g0| -> Result<Vec<[f64; 3]>, FerricError> {
            let g1 = (g0 + chunk).min(npts);
            let c = g1 - g0;
            // Lattice coordinates, same arithmetic as the energy loops.
            let pts: Vec<[f64; 3]> = (g0..g1)
                .map(|g| {
                    let ix = g / (n_y * n_z);
                    let rem = g % (n_y * n_z);
                    let iy = rem / n_z;
                    let iz = rem % n_z;
                    [
                        origin[0] + ix as f64 * hx,
                        origin[1] + iy as f64 * hy,
                        origin[2] + iz as f64 * hz,
                    ]
                })
                .collect();

            // χ and ∇χ for this chunk, serially (no nested parallelism, so at
            // most `workers` chunks are ever resident).
            let mut chi = Array2::<f64>::zeros((nbf, c));
            let mut dchi: [Array2<f64>; 3] = std::array::from_fn(|_| Array2::zeros((nbf, c)));
            let mut buf = [0.0f64; 15];
            let mut gbuf = [[0.0f64; 15]; 3];
            for (gc, p) in pts.iter().enumerate() {
                let mut row0 = 0usize;
                for (sh, &n) in shells.iter().zip(&shell_nf) {
                    buf.fill(0.0);
                    for row in gbuf.iter_mut() {
                        row.fill(0.0);
                    }
                    eval_shell_and_grad(
                        sh,
                        p[0] - sh.center[0],
                        p[1] - sh.center[1],
                        p[2] - sh.center[2],
                        &mut buf[..n],
                        &mut gbuf,
                    )
                    .map_err(|e| FerricError::General(format!("{LABEL}: {e}")))?;
                    for i in 0..n {
                        chi[(row0 + i, gc)] = buf[i];
                        for k in 0..3 {
                            dchi[k][(row0 + i, gc)] = gbuf[k][i];
                        }
                    }
                    row0 += n;
                }
            }
            let d_chi = d_sym.dot(&chi);

            let mut rho = vec![0.0_f64; c];
            for mu in 0..nbf {
                for gc in 0..c {
                    rho[gc] += chi[(mu, gc)] * d_chi[(mu, gc)];
                }
            }

            let mut local = vec![[0.0_f64; 3]; natoms];
            let mut q = vec![0.0_f64; c];
            let mut rho0 = vec![0.0_f64; natoms];
            let mut drho0 = vec![0.0_f64; natoms];
            let mut dvec = vec![[0.0_f64; 3]; natoms];
            let mut rr = vec![0.0_f64; natoms];
            for gc in 0..c {
                let p = pts[gc];
                let mut s = 0.0;
                for a in 0..natoms {
                    let d = [p[0] - pos[a][0], p[1] - pos[a][1], p[2] - pos[a][2]];
                    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    let (v, dvdr) = match &pro[a] {
                        Some(pa) => pa.value_and_deriv(r),
                        None => {
                            let (xi, prefac) = slater[a];
                            let v = prefac * (-2.0 * xi * r).exp();
                            (v, -2.0 * xi * v)
                        }
                    };
                    dvec[a] = d;
                    rr[a] = r;
                    rho0[a] = v;
                    drho0[a] = dvdr;
                    s += v;
                }
                let den = s + eps_floor;
                let mut qg = 0.0;
                for a in 0..natoms {
                    qg += de_dv[a] * (rho0[a] / den) * rr[a] * rr[a] * rr[a];
                }
                q[gc] = qg;
                let rho_g = rho[gc];
                for b in 0..natoms {
                    let r = rr[b];
                    let d = dvec[b];
                    let r3 = r * r * r;
                    // (ii): g_B = −ρ⁰_B'(r_B) d_B / r_B. At the nucleus the
                    // Slater proatom has a cusp and d_B/r_B no direction; a
                    // lattice point there is common (the bounding-box lattice
                    // puts an atom that is extreme in x, y and z on a point)
                    // and sits at r ~ 1e-16, not 0, so the guard is a distance:
                    // zero is the symmetric subgradient a central difference
                    // sees. The tabulated proatom is even in r, so its term
                    // vanishes there anyway.
                    if r > NUCLEUS_TOL {
                        let f = dv * rho_g * (de_dv[b] * r3 - qg) / den * (-drho0[b] / r);
                        for k in 0..3 {
                            local[b][k] += f * d[k];
                        }
                    }
                    // (iii)
                    let f3 = -3.0 * dv * rho_g * de_dv[b] * (rho0[b] / den) * r;
                    for k in 0..3 {
                        local[b][k] += f3 * d[k];
                    }
                }
            }

            // (i): −2 dV q Σ_{μ on B} ∇χ_μ (D_s χ)_μ.
            for mu in 0..nbf {
                let b = ao_atom[mu];
                let mut acc = [0.0_f64; 3];
                for gc in 0..c {
                    let t = q[gc] * d_chi[(mu, gc)];
                    for k in 0..3 {
                        acc[k] += t * dchi[k][(mu, gc)];
                    }
                }
                for k in 0..3 {
                    local[b][k] += -2.0 * dv * acc[k];
                }
            }
            Ok(local)
        })
        .collect::<Result<Vec<_>, FerricError>>()?;

    // Serial fold in ascending chunk order — the determinism anchor.
    let mut out = Array2::<f64>::zeros((natoms, 3));
    for part in &partials {
        for (b, row) in part.iter().enumerate() {
            for k in 0..3 {
                out[(b, k)] += row[k];
            }
        }
    }
    Ok(out)
}

/// Density-matrix derivative of the contracted Hirshfeld volumes of
/// [`atomic_effective_volumes_hirshfeld_on_grid`]:
/// ```text
///   V_μν = ∂(Σ_A c_A v_A)/∂D_μν = Σ_g dV q(r_g) χ_μ(r_g) χ_ν(r_g),
///   q(r) = Σ_A c_A w_A(r) |r − R_A|³
/// ```
/// on the same lattice, proatoms and floor as the volumes (`c` = `de_dv`).
/// Symmetric. Contracted with a density-matrix change δD it gives the
/// first-order change of Σ_A c_A v_A at fixed geometry; the MBD nuclear
/// gradient uses it for the orbital-orthonormality term −½ Tr[V D S^x D].
pub fn hirshfeld_volume_density_derivative(
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    proatom: Option<&ProatomProvider>,
    grid: &ferric_integrals::ao_grid::GridSpec,
    de_dv: &[f64],
) -> Result<Array2<f64>, FerricError> {
    use ferric_integrals::ao_grid::{collect_shells, eval_shell};
    use rayon::prelude::*;

    const LABEL: &str = "hirshfeld_volume_density_derivative";
    check_axis_aligned_grid(LABEL, grid)?;
    let natoms = mol.atoms.len();
    if de_dv.len() != natoms {
        return Err(FerricError::General(format!(
            "{LABEL}: de_dv has {} entries for {natoms} atoms",
            de_dv.len()
        )));
    }
    let shells =
        collect_shells(mol, obs_bs).map_err(|e| FerricError::General(format!("{LABEL}: {e}")))?;
    let shell_nf: Vec<usize> = shells
        .iter()
        .map(|s| ferric_core::basis::num_functions(s.l, s.pure))
        .collect();
    let nbf: usize = shell_nf.iter().sum();

    let hx = grid.step_x[0];
    let hy = grid.step_y[1];
    let hz = grid.step_z[2];
    let dv = hx * hy * hz;
    let npts = grid.n_x * grid.n_y * grid.n_z;
    let chunk = HIRSHFELD_GRAD_CHUNK_POINTS;
    let chunk_starts: Vec<usize> = (0..npts).step_by(chunk).collect();
    let n_chunks = chunk_starts.len();
    let workers = rayon::current_num_threads().max(1).min(n_chunks.max(1));
    // Chunks are summed in HIRSHFELD_V_GROUPS fixed contiguous groups (a
    // constant, not the thread count, so the fold order and the result do not
    // depend on it). Resident: per in-flight group χ, q·χ (2·nbf·chunk) and a
    // GEMM temporary (nbf²); one nbf² accumulator per group; the output.
    const HIRSHFELD_V_GROUPS: usize = 16;
    let est = {
        const F64_BYTES: usize = 8;
        let nbf2 = nbf.saturating_mul(nbf);
        let per = nbf
            .saturating_mul(2)
            .saturating_mul(chunk)
            .saturating_add(nbf2)
            .saturating_mul(F64_BYTES);
        per.saturating_mul(workers.min(HIRSHFELD_V_GROUPS))
            .saturating_add(
                nbf2.saturating_mul(HIRSHFELD_V_GROUPS + 1)
                    .saturating_mul(F64_BYTES),
            )
    };
    let label = format!("{LABEL} (nbf={nbf}, npts={npts}, chunk={chunk}, workers={workers})");
    ferric_core::memory::check_alloc(&label, est, ferric_core::memory::resolve_budget_bytes(None))?;
    let _charge = ferric_core::memory::pool::reserve_global(&label, est)?;

    let mut by_z: std::collections::BTreeMap<i32, Option<RadialProatom>> =
        std::collections::BTreeMap::new();
    let mut pro: Vec<Option<RadialProatom>> = Vec::with_capacity(natoms);
    let mut slater: Vec<(f64, f64)> = Vec::with_capacity(natoms);
    let mut pos: Vec<[f64; 3]> = Vec::with_capacity(natoms);
    for atom in &mol.atoms {
        let z_a = atom.z;
        let pa = by_z
            .entry(z_a)
            .or_insert_with(|| proatom.and_then(|p| p(z_a, 0)))
            .clone();
        pro.push(pa);
        let xi = slater_xi_for_z(z_a);
        slater.push((xi, z_a as f64 * xi.powi(3) / std::f64::consts::PI));
        pos.push([atom.x, atom.y, atom.zpos]);
    }
    let eps_floor = 1e-12;
    let (n_y, n_z) = (grid.n_y, grid.n_z);
    let origin = grid.origin;

    let per_group = n_chunks.div_ceil(HIRSHFELD_V_GROUPS).max(1);
    let chunk_contrib = |g0: usize| -> Result<Array2<f64>, FerricError> {
        let g1 = (g0 + chunk).min(npts);
        let c = g1 - g0;
        let mut chi = Array2::<f64>::zeros((nbf, c));
        let mut qchi = Array2::<f64>::zeros((nbf, c));
        let mut buf = [0.0f64; 15];
        for (gc, g) in (g0..g1).enumerate() {
            let ix = g / (n_y * n_z);
            let rem = g % (n_y * n_z);
            let iy = rem / n_z;
            let iz = rem % n_z;
            let p = [
                origin[0] + ix as f64 * hx,
                origin[1] + iy as f64 * hy,
                origin[2] + iz as f64 * hz,
            ];
            let mut s = 0.0;
            let mut num = 0.0;
            for a in 0..natoms {
                let d = [p[0] - pos[a][0], p[1] - pos[a][1], p[2] - pos[a][2]];
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let r0 = match &pro[a] {
                    Some(pa) => pa.at(r),
                    None => {
                        let (xi, prefac) = slater[a];
                        prefac * (-2.0 * xi * r).exp()
                    }
                };
                s += r0;
                num += de_dv[a] * r0 * r * r * r;
            }
            let q = dv * num / (s + eps_floor);
            let mut row0 = 0usize;
            for (sh, &n) in shells.iter().zip(&shell_nf) {
                buf.fill(0.0);
                eval_shell(
                    sh,
                    p[0] - sh.center[0],
                    p[1] - sh.center[1],
                    p[2] - sh.center[2],
                    &mut buf[..n],
                )
                .map_err(|e| FerricError::General(format!("{LABEL}: {e}")))?;
                for i in 0..n {
                    chi[(row0 + i, gc)] = buf[i];
                    qchi[(row0 + i, gc)] = q * buf[i];
                }
                row0 += n;
            }
        }
        Ok(qchi.dot(&chi.t()))
    };
    let partials: Vec<Array2<f64>> = chunk_starts
        .par_chunks(per_group)
        .map(|group| -> Result<Array2<f64>, FerricError> {
            let mut acc = Array2::<f64>::zeros((nbf, nbf));
            for &g0 in group {
                acc += &chunk_contrib(g0)?;
            }
            Ok(acc)
        })
        .collect::<Result<Vec<_>, FerricError>>()?;

    let mut out = Array2::<f64>::zeros((nbf, nbf));
    for part in &partials {
        out += part;
    }
    Ok(out)
}

/// Iterative Hirshfeld (Hirshfeld-I) charges using ad-hoc same-basis free-atom
/// proatom densities.
///
/// `proatom` supplies the spherically-averaged free-atom density for element
/// `z` at integer charge state `q_int` (0, +1, −1, …), built by the caller from
/// an atomic SCF in the molecule's basis (so the Hirshfeld weight ratio is
/// basis-consistent with the molecular density). The loop iterates each atom's
/// fractional charge to self-consistency, interpolating the proatom between
/// bracketing integer charge states; charges outside the available bracket fall
/// back to the nearest state.
///
/// Returns per-atom charges q_A = Z_A − N_A, charge-conserving and equal for
/// symmetry-equivalent atoms.
pub fn hirshfeld_i_charges(
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    density: &Array2<f64>,
    proatom: &dyn Fn(i32, i32) -> Option<RadialProatom>,
) -> Result<Vec<f64>, FerricError> {
    use ferric_integrals::ao_grid::eval_basis_on_grid;
    use ferric_integrals::ao_grid::GridSpec;

    let natoms = mol.atoms.len();
    let spacing = hirshfeld_spacing();
    let margin = hirshfeld_margin();
    let grid = GridSpec::bounding_box(mol, margin, spacing);
    let dv = spacing * spacing * spacing;
    let npts = grid.n_x * grid.n_y * grid.n_z;
    let hx = grid.step_x[0];
    let hy = grid.step_y[1];
    let hz = grid.step_z[2];

    // Pre-flight BEFORE `chi` allocates — see `preflight_hirshfeld_grid_scan`'s
    // doc: this path (and its sibling `atomic_effective_volumes_hirshfeld`)
    // had no gate but the one inside `eval_basis_on_grid`, which charges only
    // `chi` and misses the co-resident `d_chi`/`rho_free`/side vectors built
    // right after it returns.
    let nbf_pre = nbf_for_basis(mol, obs_bs)?;
    let _charge = preflight_hirshfeld_grid_scan_reserved(
        &format!("hirshfeld_i_charges (nbf={nbf_pre}, npts={npts}, natoms={natoms})"),
        nbf_pre,
        npts,
        natoms,
    )?;

    let chi = eval_basis_on_grid(mol, obs_bs, &grid)
        .map_err(|e| FerricError::General(format!("hirshfeld_i: chi eval failed: {e}")))?;
    let nbf = chi.nrows();
    debug_assert_eq!(
        nbf, nbf_pre,
        "nbf_for_basis must match eval_basis_on_grid's chi.nrows()"
    );
    if density.nrows() != nbf {
        return Err(FerricError::General(
            "hirshfeld_i: density/nbf mismatch".into(),
        ));
    }
    let d_chi = density.dot(&chi);
    let mut rho = vec![0.0_f64; npts];
    for mu in 0..nbf {
        for g in 0..npts {
            rho[g] += chi[(mu, g)] * d_chi[(mu, g)];
        }
    }

    // Precompute grid coordinates.
    let mut gx = vec![0.0; npts];
    let mut gy = vec![0.0; npts];
    let mut gz = vec![0.0; npts];
    for ix in 0..grid.n_x {
        let x = grid.origin[0] + ix as f64 * hx;
        for iy in 0..grid.n_y {
            let y = grid.origin[1] + iy as f64 * hy;
            for iz in 0..grid.n_z {
                let z = grid.origin[2] + iz as f64 * hz;
                let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                gx[g] = x;
                gy[g] = y;
                gz[g] = z;
            }
        }
    }

    // Cache integer-charge proatoms per element (z, q_int).
    let mut cache: std::collections::HashMap<(i32, i32), Option<RadialProatom>> =
        std::collections::HashMap::new();
    let get = |z: i32,
               qi: i32,
               cache: &mut std::collections::HashMap<(i32, i32), Option<RadialProatom>>|
     -> Option<RadialProatom> {
        cache
            .entry((z, qi))
            .or_insert_with(|| proatom(z, qi))
            .clone()
    };

    // Interpolated proatom density at distance r for fractional charge q.
    let proatom_rho = |z: i32,
                       q: f64,
                       r: f64,
                       cache: &mut std::collections::HashMap<(i32, i32), Option<RadialProatom>>|
     -> f64 {
        let q_lo = q.floor() as i32;
        let q_hi = q_lo + 1;
        let f = q - q_lo as f64;
        let p_lo = get(z, q_lo, cache);
        let p_hi = get(z, q_hi, cache);
        match (p_lo, p_hi) {
            (Some(a), Some(b)) => ((1.0 - f) * a.at(r) + f * b.at(r)).max(0.0),
            (Some(a), None) => a.at(r),
            (None, Some(b)) => b.at(r),
            (None, None) => {
                // Fall back to the neutral if available.
                get(z, 0, cache).map(|p| p.at(r)).unwrap_or(0.0)
            }
        }
    };

    let eps_floor = 1e-12;
    let n_elec_target = mol.nelec() as f64;
    let mut q = vec![0.0_f64; natoms];
    let max_iter = 50;
    let tol = 1e-4;

    for _it in 0..max_iter {
        let mut rho_free: Vec<Vec<f64>> = vec![vec![0.0; npts]; natoms];
        let mut rho_sum = vec![0.0_f64; npts];
        for a in 0..natoms {
            let z_a = mol.atoms[a].z;
            let (rax, ray, raz) = (mol.atoms[a].x, mol.atoms[a].y, mol.atoms[a].zpos);
            let row = &mut rho_free[a];
            for g in 0..npts {
                let dx = gx[g] - rax;
                let dy = gy[g] - ray;
                let dz = gz[g] - raz;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                let pr = proatom_rho(z_a, q[a], r, &mut cache);
                row[g] = pr;
                rho_sum[g] += pr;
            }
        }
        let mut n_e = vec![0.0_f64; natoms];
        for a in 0..natoms {
            let mut acc = 0.0;
            for g in 0..npts {
                let w = rho_free[a][g] / (rho_sum[g] + eps_floor);
                acc += rho[g] * w * dv;
            }
            n_e[a] = acc;
        }
        let n_sum: f64 = n_e.iter().sum();
        let scale = if n_sum.abs() > 1e-12 {
            n_elec_target / n_sum
        } else {
            1.0
        };
        let mut max_dq = 0.0_f64;
        for a in 0..natoms {
            let q_new = mol.atoms[a].effective_z() as f64 - scale * n_e[a];
            max_dq = max_dq.max((q_new - q[a]).abs());
            q[a] = 0.5 * q[a] + 0.5 * q_new; // damped
        }
        if debug_toggle("FERRIC_HI_DEBUG") {
            let r: Vec<f64> = q.iter().map(|x| (x * 1000.0).round() / 1000.0).collect();
            eprintln!("HI iter {_it}: q = {r:?} (max_dq={max_dq:.2e})");
        }
        if max_dq < tol {
            break;
        }
    }
    Ok(q)
}

/// Hirshfeld atomic charges q_A = Z_A − ∫ w^A(r) ρ(r) dr, with
/// w^A = ρ⁰_A / (Σ_B ρ⁰_B + 1e-12), on the atom-centred Becke–Lebedev grid.
///
/// `proatom` supplies each atom's neutral free-atom density ρ⁰_A; ferric-cli
/// and the Python `ferric.hirshfeld_charges` (default `proatom="scf"`) pass
/// [`scf_proatom_provider`](super::scf_proatom_provider). With `None`, or for an atom
/// the provider returns `None` for, ρ⁰_A is the single-Slater proatom of
/// [`slater_xi_for_z`], and a warning is printed.
///
/// The total electronic charge is renormalized so Σ_A (Z_A − q_A) = N_e
/// exactly (compensates for grid quadrature error in the density integral).
///
/// Returns `Vec<f64>` of length `mol.atoms.len()`, in units of e.
pub fn hirshfeld_charges(
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    density: &Array2<f64>,
    proatom: Option<&ProatomProvider>,
) -> Result<Vec<f64>, FerricError> {
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};

    let natoms = mol.atoms.len();
    // Atom-centred Becke-Lebedev quadrature. The previous implementation used a
    // uniform Cartesian lattice (FERRIC_HIRSHFELD_SPACING, default 0.20 Bohr),
    // which is not invariant under the molecular point group and cannot resolve
    // the nuclear cusp: symmetry-equivalent atoms sampled the core region
    // differently, so their charges came out unequal. Measured on 500 QM9
    // molecules, 52 had symmetry-equivalent atoms differing by >0.05 e (worst
    // 2.37 e, including a fluorine at -2.13 e). Renormalising to N_e hid the
    // total error while redistributing it across atoms. An atom-centred grid is
    // radially dense at each nucleus and carries the point-group symmetry, which
    // is why `becke_charges` in ferric-scf has never shown this failure.
    // Atom-centred Becke-Lebedev quadrature instead of a uniform Cartesian
    // lattice: 8.5x fewer points on a 9-atom QM9 molecule (74,250 vs 629,800 on a
    // 0.20 Bohr spacing / 6 Bohr margin lattice), with the point density placed
    // where the integrand is largest rather than spread over a bounding box.
    // `becke_charges` in ferric-scf already integrates this way.
    //
    // On weights: `build_atomic_grid` folds the Becke partition into each
    // point's weight (w_radial * w_lebedev * becke[home_atom]), so summing
    // weight * rho over the whole grid integrates the density to N_e exactly
    // once. That is the property this function needs -- the Hirshfeld weight
    // below then redistributes each point's already-correct share between
    // atoms, and because sum_A w^A(r) = 1 everywhere the total is conserved.
    // Dividing the Becke factor back out is NOT the fix for anything: it
    // diverges where that factor approaches zero, far from every nucleus.
    let grid = build_atomic_grid(mol, &AtomicGridConfig::default());
    let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let gw: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let npts = points.len();

    let chi = eval_basis_on_points(mol, obs_bs, &points)
        .map_err(|e| FerricError::General(format!("hirshfeld_charges: chi eval failed: {e}")))?;
    let nbf = chi.nrows();
    if density.nrows() != nbf || density.ncols() != nbf {
        return Err(FerricError::General(format!(
            "hirshfeld_charges: density {:?} != nbf {}",
            density.dim(),
            nbf
        )));
    }

    // ρ(r_g) = Σ_{μν} D_{μν} χ_μ(r_g) χ_ν(r_g) = sum_μ χ_μ(g) · (D · χ)(μ,g)
    let d_chi = density.dot(&chi); // (nbf, npts)
    let mut rho: Vec<f64> = vec![0.0; npts];
    for mu in 0..nbf {
        for g in 0..npts {
            rho[g] += chi[(mu, g)] * d_chi[(mu, g)];
        }
    }

    // Proatom densities and Hirshfeld weights. With a `proatom` provider, use
    // the validated same-basis free-atom density (fixes H-starvation); else the
    // legacy single-Slater proatom.
    let mut rho_free: Vec<Vec<f64>> = vec![vec![0.0; npts]; natoms];
    let mut rho_sum: Vec<f64> = vec![0.0; npts];
    // Atoms that fell back to the crude single-Slater proatom model. Collected
    // and reported ONCE after the loop: mixing two different proatom sources
    // across atoms of one molecule changes the partitioning, and the caller
    // cannot tell from the returned charges that it happened.
    let mut fallback_atoms: Vec<usize> = Vec::new();
    for a in 0..natoms {
        let z_a = mol.atoms[a].z;
        let xi = slater_xi_for_z(z_a);
        let prefac = z_a as f64 * xi.powi(3) / std::f64::consts::PI;
        let pa = proatom.and_then(|p| p(z_a, 0));
        if pa.is_none() {
            fallback_atoms.push(a);
        }
        let rax = mol.atoms[a].x;
        let ray = mol.atoms[a].y;
        let raz = mol.atoms[a].zpos;
        let row = &mut rho_free[a];
        for (g, pt) in points.iter().enumerate() {
            let dx = pt[0] - rax;
            let dy = pt[1] - ray;
            let dz = pt[2] - raz;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let r0 = match &pa {
                Some(p) => p.at(r),
                None => prefac * (-2.0 * xi * r).exp(),
            };
            row[g] = r0;
            rho_sum[g] += r0;
        }
    }

    if !fallback_atoms.is_empty() {
        // TWO different situations, and they need different warnings. The
        // mixed case is the subtle one -- some atoms on SCF proatoms and some
        // on the crude model means the partitioning is internally
        // inconsistent. The all-fallback case is not a mixture at all (it is
        // what you get with no proatom provider, e.g. `binding_energy`), and
        // saying "the other atoms used the SCF density" there is simply false:
        // there are no other atoms. A single mixed message would send a reader
        // looking for an inconsistency that is not present while
        // under-stating the one that is -- every charge came from the crude
        // model.
        if fallback_atoms.len() == natoms {
            eprintln!(
                "[hirshfeld] WARNING: ALL {natoms} atoms fell back to the crude \
                 single-Slater proatom model -- no free-atom SCF proatom density was \
                 supplied. The partitioning is internally CONSISTENT but uniformly \
                 crude; treat these charges as qualitative. Pass a proatom provider \
                 for SCF-quality proatoms."
            );
        } else {
            eprintln!(
                "[hirshfeld] WARNING: {} of {natoms} atoms (indices {fallback_atoms:?}) had no \
                 free-atom SCF proatom density and fell back to a crude single-Slater model. \
                 The other atoms used the SCF density, so these charges mix two different \
                 proatom sources and the partitioning is NOT uniformly converged.",
                fallback_atoms.len()
            );
        }
    }

    let eps_floor = 1e-12;
    // n_elec_A_grid = ∫ ρ · w^A
    let mut n_e_grid = vec![0.0_f64; natoms];
    for a in 0..natoms {
        let mut acc = 0.0;
        for g in 0..npts {
            let w = rho_free[a][g] / (rho_sum[g] + eps_floor);
            acc += rho[g] * w * gw[g];
        }
        n_e_grid[a] = acc;
    }

    // Renormalize: Σ_A n_e_grid_A should equal N_e (= trace(D·S)). Use the
    // electron count from the molecule (nelec) so the partition is exact.
    let n_elec_target = mol.nelec() as f64;
    let n_elec_sum: f64 = n_e_grid.iter().sum();
    let scale = if n_elec_sum.abs() > 1e-12 {
        n_elec_target / n_elec_sum
    } else {
        1.0
    };

    Ok((0..natoms)
        .map(|a| mol.atoms[a].effective_z() as f64 - scale * n_e_grid[a])
        .collect())
}
