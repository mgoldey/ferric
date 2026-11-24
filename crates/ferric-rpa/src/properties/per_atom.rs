//! Per-atom (Krishtal intrinsic) PDEP-RPA polarizabilities, static and dynamic,
//! on the Becke and Hirshfeld partitions, and the charge-transfer remainder
//! that closes them to the molecular tensor.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::RpaIntermediates;
use ferric_scf::properties::{hirshfeld_margin, hirshfeld_spacing, ProatomProvider, SlaterProatom};
use ferric_scf::result::{ScfResult, Spin};
use ndarray::{Array1, Array2};

use super::{
    open_shell_dynamic_extra_bytes, preflight_grid_path, preflight_hirshfeld_path,
    solve_dielectric_3,
};
use crate::config::PdepRpaConfig;

#[doc(hidden)]
/// Fraction of the resolved memory budget that in-flight dipole partials may
/// occupy: `BAND_BUDGET_NUMER / BAND_BUDGET_DENOM`.
///
/// Module-scoped so the band-width tests assert against the SAME constants the
/// implementation uses. They were function-local, which let a test hardcode the
/// pre-fraction answer (10) and stay green until the fraction landed.
pub(super) const BAND_BUDGET_NUMER: usize = 1;
pub(super) const BAND_BUDGET_DENOM: usize = 4;

/// Test-only re-export of the private `dipole_band_width_with_threads` so the
/// budget-contract MWEs in `tests/mwe_budget_respected.rs` can exercise the
/// band-width policy at an EXPLICIT worker count, without standing up an SCF
/// and without inheriting the ambient rayon pool size.
///
/// Taking `nthreads` as a parameter is load-bearing, not a convenience: the
/// first version of this MWE called the ambient-pool variant under
/// `RAYON_NUM_THREADS=2`, where the thread floor and the byte cap happen to
/// coincide — so all three contracts passed against the *unfixed* tree and the
/// bug hid. The overage is proportional to worker count (measured on the
/// 3-atom/nbf=24 shape: 1.0x at 2 threads, 6.0x at 12, 32.0x at 64), so a test
/// that cannot vary the worker count cannot see the defect at all.
///
/// Returns the number of per-atom dipole partials (each
/// `natoms·3·nbf²` f64s) that fit in a quarter of `budget_bytes`, at least 1.
/// The current band-width policy does not read `nthreads`.
pub fn dipole_band_width_for_test(
    natoms: usize,
    nbf: usize,
    budget_bytes: usize,
    nthreads: usize,
) -> usize {
    dipole_band_width_with_threads(natoms, nbf, budget_bytes, nthreads)
}

pub(super) fn dipole_band_width(natoms: usize, nbf: usize, budget_bytes: usize) -> usize {
    dipole_band_width_with_threads(natoms, nbf, budget_bytes, rayon::current_num_threads())
}

/// Worker-count-explicit core of [`dipole_band_width`].
///
/// The byte cap WINS over parallelism. A band of `w` chunks holds
/// `w * natoms * 3 * nbf^2 * 8` bytes of partials live at once, so `w` is
/// `(budget_bytes * BAND_BUDGET_NUMER / BAND_BUDGET_DENOM) / per_partial_bytes`,
/// floored at 1 so a starvation budget still makes progress (slow, never
/// stuck). The band takes a FRACTION of the budget, not all of it -- see the
/// comment on the fraction below for why charging the whole budget made the
/// pre-flight estimate track its own limit.
///
/// This deliberately does NOT floor at the rayon worker count. That floor used
/// to be here, on the reasoning that banding should never starve parallelism
/// below one chunk per worker — but it silently overrode the byte cap, letting
/// the resident set reach `nthreads * per_partial_bytes` no matter how small
/// the budget was. Memory then scaled with core count rather than with the
/// budget, which is the mechanism behind the 16-17 GB anon-RSS incidents of
/// 2026-07-13 (measured on the MWE shape: 1.0x over at 2 workers, 6.0x at 12,
/// 32.0x at 64). A budget that can be exceeded by adding cores is not a budget,
/// so on a tight budget the correct trade is fewer chunks in flight, not more
/// memory. `tests/mwe_budget_respected.rs` pins all three contracts.
///
/// Narrowing the band is numerically INERT: `chunk_size` in
/// `accumulate_atom_centred_dipoles` is a pure function of `npts`, and each
/// band's partials are folded in ascending chunk order via `collect()`, so the
/// band width changes only how many chunks are in flight — never the fold
/// order, and hence never the result.
pub(super) fn dipole_band_width_with_threads(
    natoms: usize,
    nbf: usize,
    budget_bytes: usize,
    _nthreads: usize,
) -> usize {
    let per_partial_bytes =
        natoms.max(1) * 3 * nbf.max(1) * nbf.max(1) * std::mem::size_of::<f64>();
    // Spend a FRACTION of the budget on in-flight partials, not all of it.
    //
    // A width of `budget_bytes / per_partial_bytes` ("make the band
    // as wide as the whole budget allows") would make the pre-flight estimate a
    // function of the budget itself, because `estimate_peak_bytes` charges for
    // the band this function sizes: raising the budget widened the
    // band, which raised the estimate, which failed the check again one rung
    // higher. Measured on a 9-atom aug-cc-pVDZ molecule (naux=648, nbf=207):
    // budget 4 GB -> "requires 4.45 GB"; 8 GB -> "requires 8.74 GB"; 12 GB ->
    // passes only because the 1024-chunk clamp finally binds. A gate whose
    // requirement tracks its own limit cannot be satisfied by raising the
    // limit, and it told users to do exactly that.
    //
    // The band is a throughput knob, never a correctness one (fold order is a
    // pure function of `npts`), so capping it at a fraction costs at most some
    // parallelism on tight budgets and keeps the estimate a property of the
    // problem rather than of the budget.
    let band_bytes = budget_bytes / BAND_BUDGET_DENOM * BAND_BUDGET_NUMER;
    (band_bytes / per_partial_bytes.max(1)).max(1)
}

/// Rayon-parallel, thread-count-independent, MEMORY-BOUNDED accumulation of
/// the atom-centred Becke-weighted AO dipole matrices
/// `D^{A,d}_{μν} = Σ_g w^A(r_g) (r_g − R_A)_d χ_μ(r_g) χ_ν(r_g)` over all grid
/// points `g`.
///
/// A serial `for g in 0..npts` loop would be the dominant
/// wall-clock cost of `pdep_polarizability_becke`/`_dynamic` for large systems
/// (O(npts·nbf²): ~586k grid points × nbf² for a 71-atom/def2-svp system is
/// ~4e11 scalar ops). Follows the exact two-level banded pattern documented in
/// `ferric_scf::reduce::grouped_deterministic_sum`: partition into fixed-size,
/// thread-count-independent chunks (a pure function of `npts`, never of the
/// worker count, so the fold order — and hence the exact floating-point
/// result — cannot depend on `RAYON_NUM_THREADS`), process one BAND of chunks
/// in parallel at a time (bounded by the caller's resolved memory budget),
/// and fold each band
/// into the running accumulator in ascending order before moving to the next
/// band. At most one band of partials is ever live, not all chunks at once.
///
/// Crate-visible alias so `dispersion.rs` can share this exact accumulation
/// instead of maintaining a second, serial copy over a monolithic chi.
#[allow(clippy::too_many_arguments)]
pub(crate) fn accumulate_atom_centred_dipoles_pub(
    npts: usize,
    natoms: usize,
    nbf: usize,
    home_atom: &[usize],
    weights: &[f64],
    points: &[[f64; 3]],
    atom_pos: &[[f64; 3]],
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    budget_bytes: usize,
) -> Result<Vec<[Array2<f64>; 3]>, FerricError> {
    accumulate_atom_centred_dipoles(
        npts,
        natoms,
        nbf,
        home_atom,
        weights,
        points,
        atom_pos,
        mol,
        obs_bs,
        budget_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
fn accumulate_atom_centred_dipoles(
    npts: usize,
    natoms: usize,
    nbf: usize,
    home_atom: &[usize],
    weights: &[f64],
    points: &[[f64; 3]],
    atom_pos: &[[f64; 3]],
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    budget_bytes: usize,
) -> Result<Vec<[Array2<f64>; 3]>, FerricError> {
    use ferric_integrals::ao_grid::eval_basis_on_points;
    use rayon::prelude::*;

    // Chunk size: same "≥1024 groups, floored at 1" convention as
    // ferric_scf::reduce::TARGET_GROUPS — a pure function of npts, never of
    // rayon::current_num_threads().
    const TARGET_CHUNKS: usize = 1024;
    let chunk_size = npts.div_ceil(TARGET_CHUNKS).max(1);
    let chunk_starts: Vec<usize> = (0..npts).step_by(chunk_size).collect();
    let n_chunks = chunk_starts.len();

    // The RESOLVED budget, not a hardcoded constant: the pre-flight gate
    // estimated this path using the caller's budget, so the accumulation must
    // band against the same number or the two disagree (a fixed 512 MB const
    // would exceed what the gate approved below it, and band more tightly than
    // necessary above it, costing wall time).
    let band_width = dipole_band_width(natoms, nbf, budget_bytes);

    let mut d_ai_ao: Vec<[Array2<f64>; 3]> = (0..natoms)
        .map(|_| std::array::from_fn(|_| Array2::<f64>::zeros((nbf, nbf))))
        .collect();

    let mut band0 = 0usize;
    while band0 < n_chunks {
        let band1 = (band0 + band_width).min(n_chunks);
        // Parallel over this band's chunks only; collect preserves ascending
        // chunk-index order regardless of worker count.
        let band_partials: Vec<Vec<[Array2<f64>; 3]>> = chunk_starts[band0..band1]
            .to_vec()
            .into_par_iter()
            .map(|g0| -> Result<Vec<[Array2<f64>; 3]>, FerricError> {
                let g1 = (g0 + chunk_size).min(npts);
                // Evaluate chi for THIS CHUNK's points only. The full
                // (nbf, npts) matrix is never built up front: one contiguous
                // allocation would be ~3.75 GB at nbf=800/npts=586k, though
                // every consumer reads it one grid point at a time. A
                // chunk is `chunk_size` points wide (npts/1024, so ~573 at
                // incident scale), reducing this to a few MB per worker.
                //
                // Numerically inert: chi is a materialized lookup table, not a
                // reduction. Each point's value is a pure function of its
                // coordinates, so evaluating it in chunks reproduces the same
                // numbers; the fold order that DOES matter is the ascending
                // chunk fold below, which is untouched.
                let chunk_pts = &points[g0..g1];
                let chi_c = eval_basis_on_points(mol, obs_bs, chunk_pts).map_err(|e| {
                    FerricError::General(format!(
                        "accumulate_atom_centred_dipoles: chi eval failed: {e}"
                    ))
                })?;
                let mut local: Vec<[Array2<f64>; 3]> = (0..natoms)
                    .map(|_| std::array::from_fn(|_| Array2::<f64>::zeros((nbf, nbf))))
                    .collect();
                for g in g0..g1 {
                    let gc = g - g0; // chunk-local column index
                    let a = home_atom[g];
                    let w = weights[g];
                    let r = points[g];
                    let ra = atom_pos[a];
                    for d in 0..3 {
                        let factor = w * (r[d] - ra[d]);
                        for mu in 0..nbf {
                            let chi_mu = chi_c[(mu, gc)];
                            let weighted_chi_mu = factor * chi_mu;
                            if weighted_chi_mu.abs() < 1e-30 {
                                continue;
                            }
                            for nu in 0..nbf {
                                local[a][d][(mu, nu)] += weighted_chi_mu * chi_c[(nu, gc)];
                            }
                        }
                    }
                }
                Ok(local)
            })
            .collect::<Result<Vec<_>, FerricError>>()?;
        // Serial fold in ascending chunk order — the determinism anchor.
        for chunk in &band_partials {
            for a in 0..natoms {
                for d in 0..3 {
                    d_ai_ao[a][d] += &chunk[a][d];
                }
            }
        }
        band0 = band1;
    }
    Ok(d_ai_ao)
}

/// Charge-transfer (charge-delocalization) polarizability remainder
/// α_CT = α_mol − Σ_A α^A for the Krishtal–Senet–Van Alsenoy intrinsic
/// per-atom tensors (JCP 125, 034312 (2006)).
///
/// With α^A = ⟨m^A|R|μ⟩ and α_mol = ⟨μ|R|μ⟩ (same response R: same RI
/// kernel, same orbitals, frozen_core = 0), linearity gives
/// α_CT = ⟨Σ_A R_A w_A|R|μ⟩ — the dipole carried by field-induced charge
/// flow between atoms. `α_CT + Σ_A α^A = α_mol` holds by construction; the
/// quantity is only meaningful when `molecular` and `per_atom` come from
/// matching paths (`pdep_polarizability_static` with
/// `pdep_polarizability_becke`, or `molecular_dynamic_polarizability` with
/// `pdep_polarizability_becke_dynamic` / `_hirshfeld_dynamic` at the same
/// frequencies). The atom sum is accumulated in atom order from 0.0.
pub fn charge_transfer_remainder(
    molecular: &[[f64; 3]; 3],
    per_atom: &[[[f64; 3]; 3]],
) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let sum: f64 = per_atom.iter().map(|t| t[i][j]).sum();
            molecular[i][j] - sum
        })
    })
}

/// Frequency-resolved [`charge_transfer_remainder`]: `molecular[k]` is
/// α_mol(iω_k), `per_atom[A][k]` is α^A(iω_k); returns α_CT(iω_k) for every
/// k. Errors if the frequency counts disagree (an empty `molecular`, as
/// returned by the truncated benchmark path, is a mismatch, not a zero).
pub fn charge_transfer_remainder_dynamic(
    molecular: &[[[f64; 3]; 3]],
    per_atom: &[Vec<[[f64; 3]; 3]>],
) -> Result<Vec<[[f64; 3]; 3]>, FerricError> {
    let nfreq = molecular.len();
    if let Some((a, row)) = per_atom.iter().enumerate().find(|(_, r)| r.len() != nfreq) {
        return Err(FerricError::General(format!(
            "charge_transfer_remainder_dynamic: atom {a} has {} frequencies, molecular has {nfreq}",
            row.len()
        )));
    }
    Ok((0..nfreq)
        .map(|k| {
            let at_k: Vec<[[f64; 3]; 3]> = per_atom.iter().map(|r| r[k]).collect();
            charge_transfer_remainder(&molecular[k], &at_k)
        })
        .collect())
}

// ---------------------------------------------------------------------------
// One kernel for the four per-atom drivers.
//
// Every driver evaluates the same Krishtal intrinsic tensor
//
//   α^A_dj(ω) = 4 m^{A,d}·g·μ^j − 16 (B·(m^{A,d}⊙g))ᵀ ε̃⁻¹ (B·(μ^j⊙g)),
//   ε̃ = I + 4 B̃ diag(g) B̃ᵀ,   g_ia = 1/Δε_ia (static) or Δε_ia/(ω²+Δε_ia²),
//
// and differs only in where the atom-centred dipole `m^A` (the bra) comes from:
// the Becke home-atom weights of the atomic grid, or Hirshfeld weights on a
// bounding-box lattice. The pipeline is therefore three stages: the closed-shell
// response ([`ClosedResponse`]), one bra per partition ([`becke_bra`],
// [`hirshfeld_bra`]), and the per-frequency kernel ([`closed_per_atom_alpha`]).
// The drivers own what is specific to them: the memory pre-flight and its
// label, how long the pool reservation is held, and static versus parallel
// frequency evaluation.
// ---------------------------------------------------------------------------

/// Imaginary-frequency point of the response.
///
/// `Static` evaluates the ω = 0 limit as g = 1/Δε, the form the static drivers
/// use; `Imaginary(0.0)` evaluates Δε/(0 + Δε²), which differs from it in the
/// last bit, so the two stay distinct.
#[derive(Clone, Copy)]
enum Frequency {
    Static,
    Imaginary(f64),
}

impl Frequency {
    /// g_ia over the flattened occupied-virtual pairs.
    fn weights(self, e_ia: &Array1<f64>) -> Array1<f64> {
        match self {
            Frequency::Static => e_ia.mapv(|e| 1.0 / e),
            Frequency::Imaginary(omega) => {
                let omega2 = omega * omega;
                e_ia.mapv(|e| e / (omega2 + e * e))
            }
        }
    }
}

/// Row-major `(nocc·nvir,)` flattening of an occupied-virtual block.
fn flatten_ov(mo: &Array2<f64>) -> Array1<f64> {
    let (nocc, nvir) = mo.dim();
    let mut v = Array1::<f64>::zeros(nocc * nvir);
    for i in 0..nocc {
        for a in 0..nvir {
            v[i * nvir + a] = mo[(i, a)];
        }
    }
    v
}

/// The closed-shell RI intermediates, the occupied/virtual coefficients and
/// the excitation energies shared by every closed-shell per-atom driver. All
/// occupied orbitals respond (`frozen_core = 0`), as in every polarizability
/// path.
struct ClosedResponse {
    inter: RpaIntermediates,
    c_occ: Array2<f64>,
    c_vir: Array2<f64>,
    e_ia: Array1<f64>,
}

impl ClosedResponse {
    fn new(
        mol: &Molecule,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        rhf: &ScfResult,
        op: Operator,
        cfg: &PdepRpaConfig,
    ) -> Result<Self, FerricError> {
        let inter = ferric_mp2::rimp2::compute_rpa_intermediates(
            mol,
            obs,
            dfbs,
            op,
            rhf,
            &polarizability_mp2_config(cfg),
        )?;
        let (nocc, nvir) = (inter.nocc, inter.nvir);
        let (first_occ, nocc_total) = (inter.first_occ, inter.nocc_total);
        let eps = rhf.eps_r();
        let c = rhf.mos_r();
        let c_occ = c
            .slice(ndarray::s![.., first_occ..first_occ + nocc])
            .to_owned();
        let c_vir = c
            .slice(ndarray::s![.., nocc_total..nocc_total + nvir])
            .to_owned();
        let mut e_ia = Array1::<f64>::zeros(nocc * nvir);
        for i in 0..nocc {
            for a in 0..nvir {
                e_ia[i * nvir + a] = eps[nocc_total + a] - eps[first_occ + i];
            }
        }
        Ok(Self {
            inter,
            c_occ,
            c_vir,
            e_ia,
        })
    }

    /// `C_occᵀ X C_virt`, flattened.
    fn ov_flat(&self, ao: &Array2<f64>) -> Array1<f64> {
        flatten_ov(&self.c_occ.t().dot(ao).dot(&self.c_vir))
    }

    /// The analytic lab-frame molecular dipole in the occupied-virtual basis:
    /// the field-side ket every driver pairs with its atom-centred bra.
    fn molecular_dipole_flat(&self, obs: &PreparedBasis) -> Result<[Array1<f64>; 3], FerricError> {
        let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
        Ok(std::array::from_fn(|d| self.ov_flat(&dip_ao[d])))
    }
}

/// RI configuration of the polarizability paths: no frozen core, and the
/// caller's explicit memory budget (not an env/auto-detected one).
fn polarizability_mp2_config(cfg: &PdepRpaConfig) -> ferric_mp2::rimp2::RiMp2Config {
    ferric_mp2::rimp2::RiMp2Config {
        frozen_core: 0,
        memory_budget_bytes: cfg.memory_budget_bytes,
        ..Default::default()
    }
}

/// α^A(ω) for every atom: the per-frequency kernel. `bra[a][d]` is atom `a`'s
/// atom-centred dipole in the occupied-virtual basis, `mu` the molecular one.
/// `b_scaled` is `(naux, nov)` scratch, overwritten.
fn closed_per_atom_alpha(
    resp: &ClosedResponse,
    bra: &[[Array1<f64>; 3]],
    mu: &[Array1<f64>; 3],
    freq: Frequency,
    b_scaled: &mut Array2<f64>,
) -> Result<Vec<[[f64; 3]; 3]>, FerricError> {
    let b_ov = &resp.inter.b_ov;
    let naux = resp.inter.naux;
    let nov = resp.e_ia.len();
    let g = freq.weights(&resp.e_ia);

    // ε̃ = I + 4 B̃ diag(g) B̃ᵀ
    b_scaled.assign(b_ov);
    for ia in 0..nov {
        b_scaled
            .column_mut(ia)
            .mapv_inplace(|x| x * (4.0 * g[ia]).sqrt());
    }
    let mut eps_mat: Array2<f64> = b_scaled.dot(&b_scaled.t());
    for p in 0..naux {
        eps_mat[(p, p)] += 1.0;
    }

    // Solve ε̃ y^j = B·(g⊙μ^j) once per direction.
    let mu_g: [Array1<f64>; 3] = std::array::from_fn(|d| &mu[d] * &g);
    let w_mol: [Array1<f64>; 3] = std::array::from_fn(|d| b_ov.dot(&mu_g[d]));
    let y_mol = solve_dielectric_3(&eps_mat, &w_mol)?;

    Ok(bra
        .iter()
        .map(|atom| {
            let mut tensor = [[0.0_f64; 3]; 3];
            for d in 0..3 {
                let w_ai = b_ov.dot(&(&atom[d] * &g));
                for j in 0..3 {
                    let bare = atom[d].dot(&mu_g[j]);
                    let coupled = w_ai.dot(&y_mol[j]);
                    tensor[d][j] = 4.0 * bare - 16.0 * coupled;
                }
            }
            symmetrize_tensor(&mut tensor);
            tensor
        })
        .collect())
}

fn symmetrize_tensor(t: &mut [[f64; 3]; 3]) {
    for i in 0..3 {
        for j in (i + 1)..3 {
            let avg = 0.5 * (t[i][j] + t[j][i]);
            t[i][j] = avg;
            t[j][i] = avg;
        }
    }
}

/// `out[atom][freq]` from the per-frequency rows `rows[freq][atom]`.
fn rows_to_atom_major(
    rows: Vec<Result<Vec<[[f64; 3]; 3]>, FerricError>>,
    natoms: usize,
) -> Result<Vec<Vec<[[f64; 3]; 3]>>, FerricError> {
    let nfreq = rows.len();
    let mut out: Vec<Vec<[[f64; 3]; 3]>> = vec![vec![[[0.0; 3]; 3]; nfreq]; natoms];
    for (k, row) in rows.into_iter().enumerate() {
        let row = row?;
        for a in 0..natoms {
            out[a][k] = row[a];
        }
    }
    Ok(out)
}

/// The closed-shell kernel over a frequency list, in parallel over frequencies.
/// One `(naux, nov)` scratch per rayon worker (never cloned per frequency),
/// BLAS pinned to 1 inside the rayon region.
fn closed_per_atom_dynamic(
    resp: &ClosedResponse,
    bra: &[[Array1<f64>; 3]],
    mu: &[Array1<f64>; 3],
    freqs: &[f64],
    natoms: usize,
) -> Result<Vec<Vec<[[f64; 3]; 3]>>, FerricError> {
    use ferric_integrals::blas_threads::with_blas_threads;
    use rayon::prelude::*;

    let rows: Vec<Result<Vec<[[f64; 3]; 3]>, FerricError>> = with_blas_threads(1, || {
        freqs
            .par_iter()
            .map_init(
                // Zeros, not `.clone()`: overwritten before it is read.
                || Array2::<f64>::zeros(resp.inter.b_ov.raw_dim()),
                |b_scaled, &omega| {
                    closed_per_atom_alpha(resp, bra, mu, Frequency::Imaginary(omega), b_scaled)
                },
            )
            .collect()
    });
    rows_to_atom_major(rows, natoms)
}

/// The static kernel: the ω = 0 limit, evaluated serially.
fn closed_per_atom_static(
    resp: &ClosedResponse,
    bra: &[[Array1<f64>; 3]],
    mu: &[Array1<f64>; 3],
) -> Result<Vec<[[f64; 3]; 3]>, FerricError> {
    let mut b_scaled = Array2::<f64>::zeros(resp.inter.b_ov.raw_dim());
    closed_per_atom_alpha(resp, bra, mu, Frequency::Static, &mut b_scaled)
}

// --- Becke bra ---------------------------------------------------------------

/// The Becke–Lebedev atomic grid in the flat form the dipole accumulation takes.
struct BeckeGrid {
    points: Vec<[f64; 3]>,
    weights: Vec<f64>,
    home_atom: Vec<usize>,
}

fn becke_grid(mol: &Molecule) -> BeckeGrid {
    let grid =
        ferric_dft::grid::build_atomic_grid(mol, &ferric_dft::grid::AtomicGridConfig::default());
    BeckeGrid {
        points: grid.iter().map(|g| g.xyz).collect(),
        weights: grid.iter().map(|g| g.weight).collect(),
        home_atom: grid.iter().map(|g| g.home_atom).collect(),
    }
}

/// Symmetrized per-atom atom-centred AO dipoles `∫ w^A (r − R_A)_d χ_μ χ_ν`
/// on the Becke grid, accumulated in memory-bounded chunks. The bra operator
/// is (r − R_A), so each α^A is origin-independent; the lab-frame
/// renormalization that would break that gauge is deliberately absent.
fn becke_ao_dipoles(
    grid: &BeckeGrid,
    mol: &Molecule,
    nbf: usize,
    obs_bs: &ferric_core::basis::BasisSet,
    budget_bytes: usize,
) -> Result<Vec<[Array2<f64>; 3]>, FerricError> {
    let natoms = mol.atoms.len();
    let atom_pos: Vec<[f64; 3]> = mol.atoms.iter().map(|at| [at.x, at.y, at.zpos]).collect();
    let mut d_ai_ao = accumulate_atom_centred_dipoles(
        grid.points.len(),
        natoms,
        nbf,
        &grid.home_atom,
        &grid.weights,
        &grid.points,
        &atom_pos,
        mol,
        obs_bs,
        budget_bytes,
    )?;
    for atom in d_ai_ao.iter_mut() {
        for m in atom.iter_mut() {
            for i in 0..nbf {
                for j in (i + 1)..nbf {
                    let avg = 0.5 * (m[(i, j)] + m[(j, i)]);
                    m[(i, j)] = avg;
                    m[(j, i)] = avg;
                }
            }
        }
    }
    Ok(d_ai_ao)
}

/// Per-atom Becke dipoles in the closed-shell occupied-virtual basis.
fn becke_bra(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    resp: &ClosedResponse,
    grid: &BeckeGrid,
    budget_bytes: usize,
) -> Result<Vec<[Array1<f64>; 3]>, FerricError> {
    // nbf from the prepared basis, never from a materialized chi.
    let d_ai_ao = becke_ao_dipoles(grid, mol, obs.nbasis(), obs_bs, budget_bytes)?;
    Ok(d_ai_ao
        .iter()
        .map(|atom| std::array::from_fn(|d| resp.ov_flat(&atom[d])))
        .collect())
}

// --- Hirshfeld bra -----------------------------------------------------------

/// The bounding-box lattice the Hirshfeld bra integrates on.
struct HirshfeldLattice {
    grid: ferric_integrals::ao_grid::GridSpec,
    spacing: f64,
    margin: f64,
    dv: f64,
    npts: usize,
}

fn hirshfeld_lattice(mol: &Molecule) -> HirshfeldLattice {
    let spacing = hirshfeld_spacing();
    let margin = hirshfeld_margin();
    let grid = ferric_integrals::ao_grid::GridSpec::bounding_box(mol, margin, spacing);
    let npts = grid.n_x * grid.n_y * grid.n_z;
    HirshfeldLattice {
        grid,
        spacing,
        margin,
        dv: spacing * spacing * spacing,
        npts,
    }
}

/// Per-atom Hirshfeld dipoles `∫ χ_μ χ_ν w^A (r − R_A)_d dV`,
/// `w^A = ρ⁰_A/(Σ_B ρ⁰_B + 1e-12)`, by quadrature on the lattice, in the
/// closed-shell occupied-virtual basis.
///
/// The proatom is the validated same-basis free-atom density when `proatom`
/// supplies one for the element, else the single-exponential Slater density.
/// The (r − R_A) operator keeps α^A origin-independent; no renormalization to
/// the lab-frame dipole is applied (that is the gauge-breaking step).
/// `driver` names the caller in the error text.
fn hirshfeld_bra(
    driver: &str,
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    resp: &ClosedResponse,
    lat: &HirshfeldLattice,
    nbf: usize,
    proatom: Option<&ProatomProvider>,
) -> Result<Vec<[Array1<f64>; 3]>, FerricError> {
    let grid = &lat.grid;
    let (npts, dv) = (lat.npts, lat.dv);
    let natoms = mol.atoms.len();

    // χ_μ(r_g): shape (nbf, npts).
    let chi = ferric_integrals::ao_grid::eval_basis_on_grid(mol, obs_bs, grid)
        .map_err(|e| FerricError::General(format!("{driver}: chi eval failed: {e}")))?;
    debug_assert_eq!(chi.nrows(), nbf);

    // Proatom densities ρ⁰_A and their sum on the lattice.
    let mut rho_free: Vec<Vec<f64>> = vec![vec![0.0; npts]; natoms];
    let mut rho_sum: Vec<f64> = vec![0.0; npts];
    let hx = grid.step_x[0];
    let hy = grid.step_y[1];
    let hz = grid.step_z[2];
    for a in 0..natoms {
        let z_a = mol.atoms[a].z;
        let slater = SlaterProatom::for_atom(&mol.atoms[a]);
        let pa = proatom.and_then(|p| p(z_a, 0)); // neutral same-basis proatom
        let rax = mol.atoms[a].x;
        let ray = mol.atoms[a].y;
        let raz = mol.atoms[a].zpos;
        let row = &mut rho_free[a];
        for ix in 0..grid.n_x {
            let x = grid.origin[0] + ix as f64 * hx;
            for iy in 0..grid.n_y {
                let y = grid.origin[1] + iy as f64 * hy;
                for iz in 0..grid.n_z {
                    let z = grid.origin[2] + iz as f64 * hz;
                    let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                    let dx = x - rax;
                    let dy = y - ray;
                    let dz = z - raz;
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    let rho = match &pa {
                        Some(p) => p.at(r),
                        None => slater.at(r),
                    };
                    row[g] = rho;
                    rho_sum[g] += rho;
                }
            }
        }
    }

    // Lab-frame lattice coordinates r_i(g).
    let mut ri_grid: [Vec<f64>; 3] = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
    for ix in 0..grid.n_x {
        let x = grid.origin[0] + ix as f64 * hx;
        for iy in 0..grid.n_y {
            let y = grid.origin[1] + iy as f64 * hy;
            for iz in 0..grid.n_z {
                let z = grid.origin[2] + iz as f64 * hz;
                let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                ri_grid[0][g] = x;
                ri_grid[1][g] = y;
                ri_grid[2][g] = z;
            }
        }
    }

    // One atom's three AO dipole matrices are live at a time: each is reduced
    // to the occupied-virtual vector and dropped before the next is built.
    let eps_floor = 1e-12;
    Ok((0..natoms)
        .map(|a| {
            let ra = [mol.atoms[a].x, mol.atoms[a].y, mol.atoms[a].zpos];
            let mut wa = vec![0.0_f64; npts];
            for g in 0..npts {
                wa[g] = rho_free[a][g] / (rho_sum[g] + eps_floor);
            }
            std::array::from_fn(|i_cart| {
                let ra_d = ra[i_cart];
                let mut combined = Array2::<f64>::zeros((nbf, npts));
                for mu in 0..nbf {
                    let chi_mu = chi.row(mu);
                    let mut row = combined.row_mut(mu);
                    for g in 0..npts {
                        row[g] = chi_mu[g] * wa[g] * (ri_grid[i_cart][g] - ra_d) * dv;
                    }
                }
                let d: Array2<f64> = chi.dot(&combined.t());
                let mut d_sym = Array2::<f64>::zeros((nbf, nbf));
                for mu in 0..nbf {
                    for nu in 0..nbf {
                        d_sym[(mu, nu)] = 0.5 * (d[(mu, nu)] + d[(nu, mu)]);
                    }
                }
                resp.ov_flat(&d_sym)
            })
        })
        .collect())
}

// --- Drivers -------------------------------------------------------------------

// --- Open-shell Becke ----------------------------------------------------------

/// One spin channel of the open-shell response: its RI factor, excitation
/// energies and the per-atom and molecular dipoles in its occupied-virtual
/// basis. An empty channel (no occupied orbitals of that spin) contributes
/// nothing: its vectors are length-1 placeholders that are never read.
struct SpinChannel<'a> {
    b_ov: &'a Array2<f64>,
    e_ia: Array1<f64>,
    nocc: usize,
    bra: Vec<[Array1<f64>; 3]>,
    mu: [Array1<f64>; 3],
}

impl<'a> SpinChannel<'a> {
    fn new(
        inter: &'a RpaIntermediates,
        eps: &[f64],
        c: &Array2<f64>,
        d_ai_ao: &[[Array2<f64>; 3]],
        dip_ao: &[Array2<f64>; 3],
    ) -> Self {
        let (nocc, nvir) = (inter.nocc, inter.nvir);
        let mut e_ia = Array1::<f64>::zeros(nocc * nvir);
        for i in 0..nocc {
            for a in 0..nvir {
                e_ia[i * nvir + a] = eps[inter.nocc_total + a] - eps[inter.first_occ + i];
            }
        }
        if nocc == 0 {
            return Self {
                b_ov: &inter.b_ov,
                e_ia,
                nocc,
                bra: d_ai_ao
                    .iter()
                    .map(|_| std::array::from_fn(|_| Array1::zeros(1)))
                    .collect(),
                mu: std::array::from_fn(|_| Array1::zeros(1)),
            };
        }
        let c_occ = c
            .slice(ndarray::s![.., inter.first_occ..inter.first_occ + nocc])
            .to_owned();
        let c_vir = c
            .slice(ndarray::s![.., inter.nocc_total..inter.nocc_total + nvir])
            .to_owned();
        let ov = |ao: &Array2<f64>| flatten_ov(&c_occ.t().dot(ao).dot(&c_vir));
        Self {
            b_ov: &inter.b_ov,
            e_ia,
            nocc,
            bra: d_ai_ao
                .iter()
                .map(|atom| std::array::from_fn(|d| ov(&atom[d])))
                .collect(),
            mu: std::array::from_fn(|d| ov(&dip_ao[d])),
        }
    }

    /// g_ia(ω) = Δε/(ω² + Δε²); a placeholder for an empty channel.
    fn g(&self, omega2: f64) -> Array1<f64> {
        if self.nocc > 0 {
            self.e_ia.mapv(|e| e / (omega2 + e * e))
        } else {
            Array1::zeros(1)
        }
    }

    /// B̃ (μ ⊙ g) for a vector in this channel's occupied-virtual basis.
    fn rhs(&self, v: &Array1<f64>, g: &Array1<f64>) -> Array1<f64> {
        self.b_ov.dot(&(v * g))
    }
}

/// α^A(iω) for every atom of an open-shell reference: both spin channels
/// enter ε̃(ω) = I + 2 Σ_σ B̃_σ diag(g_σ) B̃_σᵀ and the coupled term.
fn open_shell_alpha(
    chs: [&SpinChannel; 2],
    naux: usize,
    omega: f64,
    scratch: (&mut Array2<f64>, &mut Array2<f64>),
) -> Result<Vec<[[f64; 3]; 3]>, FerricError> {
    let omega2 = omega * omega;
    let [a, b] = chs;
    let (g_a, g_b) = (a.g(omega2), b.g(omega2));

    // ε̃(ω) = I + 2 B̃_α diag(g_α) B̃_αᵀ + 2 B̃_β diag(g_β) B̃_βᵀ.
    let mut eps_mat = Array2::<f64>::zeros((naux, naux));
    for p in 0..naux {
        eps_mat[(p, p)] = 1.0;
    }
    for (ch, g, b_scaled) in [(a, &g_a, scratch.0), (b, &g_b, scratch.1)] {
        let nov = ch.b_ov.shape()[1];
        if nov == 0 {
            continue;
        }
        b_scaled.assign(ch.b_ov);
        for ia in 0..nov {
            let s = (2.0 * g[ia]).sqrt();
            b_scaled.column_mut(ia).mapv_inplace(|x| x * s);
        }
        let chi_s = b_scaled.dot(&b_scaled.t());
        eps_mat += &chi_s;
    }

    // w^d = B̃_α (v_α ⊙ g_α) + B̃_β (v_β ⊙ g_β).
    let total = |va: &Array1<f64>, vb: &Array1<f64>| {
        let wa = a.rhs(va, &g_a);
        if b.nocc == 0 {
            return wa;
        }
        wa + b.rhs(vb, &g_b)
    };
    let w_total: [Array1<f64>; 3] = std::array::from_fn(|d| total(&a.mu[d], &b.mu[d]));
    let y_total = solve_dielectric_3(&eps_mat, &w_total)?;

    Ok((0..a.bra.len())
        .map(|at| {
            let w_ai: [Array1<f64>; 3] =
                std::array::from_fn(|d| total(&a.bra[at][d], &b.bra[at][d]));
            let mut row = [[0.0_f64; 3]; 3];
            for d in 0..3 {
                for j in 0..3 {
                    let bare_a = 2.0 * a.bra[at][d].dot(&(&a.mu[j] * &g_a));
                    let bare_b = if b.nocc > 0 {
                        2.0 * b.bra[at][d].dot(&(&b.mu[j] * &g_b))
                    } else {
                        0.0
                    };
                    let coupled = w_ai[d].dot(&y_total[j]);
                    row[d][j] = bare_a + bare_b - 4.0 * coupled;
                }
            }
            symmetrize_tensor(&mut row);
            row
        })
        .collect())
}

/// [`pdep_polarizability_becke_dynamic`] for UHF/UKS and ROHF/ROKS references.
#[allow(clippy::too_many_arguments)]
fn becke_dynamic_open_shell(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
    freqs: &[f64],
) -> Result<Vec<Vec<[[f64; 3]; 3]>>, FerricError> {
    use ferric_integrals::blas_threads::with_blas_threads;
    use ferric_mp2::rimp2::compute_rpa_intermediates_spin;
    use rayon::prelude::*;

    let natoms = mol.atoms.len();
    let mp2_cfg = polarizability_mp2_config(cfg);

    // ROHF/ROKS -> per-spin semi-canonical orbitals and energies (the same view
    // run_u_pdep_rpa uses); UHF borrowed.
    let rhf_view = ferric_scf::semicanonical::unrestricted_reference(mol, rhf)?;
    let rhf: &ScfResult = &rhf_view;

    let inter_a = compute_rpa_intermediates_spin(mol, obs, dfbs, op, rhf, &mp2_cfg, true)?;
    let inter_b = compute_rpa_intermediates_spin(mol, obs, dfbs, op, rhf, &mp2_cfg, false)?;
    let naux = inter_a.naux;

    // Becke grid (spin-agnostic).
    let grid = becke_grid(mol);
    let npts = grid.points.len();

    // Pre-flight before the grid work, charged at the LARGER spin channel's
    // (nocc, nvir): both `RpaIntermediates` are resident. The per-worker
    // frequency scratch holds BOTH channels at once (a `(b_scaled_a,
    // b_scaled_b)` pair per worker), which the base estimate cannot express
    // (it takes a single (naux, nocc, nvir) shape), so a second explicit check
    // charges the difference.
    let nov_a = inter_a.nocc.saturating_mul(inter_a.nvir);
    let nov_b = inter_b.nocc.saturating_mul(inter_b.nvir);
    let nov_min = nov_a.min(nov_b);
    let n_workers = rayon::current_num_threads().max(1);
    let (grid_budget, _charge) = preflight_grid_path(
        &format!(
            "pdep_polarizability_becke_dynamic (U) (natoms={natoms}, nbf={}, npts={npts}, naux={naux})",
            obs.nbasis()
        ),
        cfg.memory_budget_bytes,
        naux,
        inter_a.nocc.max(inter_b.nocc),
        inter_a.nvir.max(inter_b.nvir),
        npts,
        obs.nbasis(),
        natoms,
    )?;
    ferric_core::memory::check_alloc(
        &format!(
            "pdep_polarizability_becke_dynamic (U) second-spin frequency scratch \
             (naux={naux}, nov_a={nov_a}, nov_b={nov_b}, n_workers={n_workers})"
        ),
        open_shell_dynamic_extra_bytes(naux, nov_min, n_workers),
        grid_budget,
    )?;

    let d_ai_ao = becke_ao_dipoles(
        &grid,
        mol,
        obs.nbasis(),
        obs_bs,
        ferric_core::memory::resolve_budget_bytes(cfg.memory_budget_bytes),
    )?;
    // Field-side (ket) dipole per spin: the analytic lab-frame molecular
    // dipole, NOT Σ_A of the atom-centred bra pieces.
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let ch_a = SpinChannel::new(&inter_a, rhf.eps_a(), rhf.mos_a(), &d_ai_ao, &dip_ao);
    let ch_b = SpinChannel::new(&inter_b, rhf.eps_b(), rhf.mos_b(), &d_ai_ao, &dip_ao);

    // Each ω is independent: parallelize over frequencies with one
    // (b_scaled_a, b_scaled_b) pair per rayon worker (zeros, never cloned),
    // BLAS pinned to 1 inside the rayon region.
    let rows: Vec<Result<Vec<[[f64; 3]; 3]>, FerricError>> = with_blas_threads(1, || {
        freqs
            .par_iter()
            .map_init(
                || {
                    (
                        Array2::<f64>::zeros(inter_a.b_ov.raw_dim()),
                        Array2::<f64>::zeros(inter_b.b_ov.raw_dim()),
                    )
                },
                |(sa, sb), &omega| open_shell_alpha([&ch_a, &ch_b], naux, omega, (sa, sb)),
            )
            .collect()
    });
    rows_to_atom_major(rows, natoms)
}

// --- The four public drivers -----------------------------------------------------

/// Per-atom static PDEP-RPA polarizability tensors α^A (Bohr³), one 3×3 per
/// atom, on ferric's Becke-Lebedev grid.
///
/// Definition: the Krishtal–Senet–Van Alsenoy "intrinsic" atomic
/// polarizability (A. Krishtal, P. Senet, C. Van Alsenoy, J. Chem. Phys. 125,
/// 034312 (2006)),
///
/// ```text
///   α^A_dj = ⟨ m^A_d | R | μ_j ⟩,   m^A = w_A (r − R_A)  (Becke weight w_A),
/// ```
///
/// i.e. the ATOM-CENTRED dipole of atom A on the bra and the true lab-frame
/// molecular dipole μ (analytic AO integrals) on the field-side ket: the
/// first-order change of atom A's own dipole in a uniform field.
/// Charge transfer between atoms is EXCLUDED, so Σ_A α^A ≠ α_mol; the
/// remainder α_CT = α_mol − Σ_A α^A = ⟨Σ_A R_A w_A | R | μ⟩ is the
/// charge-transfer (charge-delocalization) polarizability — see
/// [`charge_transfer_remainder`]. Each α^A is origin-independent.
///
/// Closed and open shell (open shell delegates to
/// [`pdep_polarizability_becke_dynamic`] at ω = 0, the same definition).
pub fn pdep_polarizability_becke(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
) -> Result<Vec<[[f64; 3]; 3]>, FerricError> {
    // Open-shell: the dynamic Becke path has a complete per-spin (U) branch, and
    // ω=0 reproduces the static per-atom α exactly. Delegate to it rather than
    // duplicate the per-spin static math.
    if !matches!(rhf.spin, Spin::Restricted) {
        let dyn0 = pdep_polarizability_becke_dynamic(mol, obs, obs_bs, dfbs, rhf, op, cfg, &[0.0])?;
        // dyn0[atom][freq=0] → per-atom static tensor.
        return Ok(dyn0.into_iter().map(|per_freq| per_freq[0]).collect());
    }

    let natoms = mol.atoms.len();
    let resp = ClosedResponse::new(mol, obs, dfbs, rhf, op, cfg)?;
    let grid = becke_grid(mol);
    let npts = grid.points.len();
    // Pre-flight BEFORE chi allocates; npts comes from the grid just built.
    let (budget, _charge) = preflight_grid_path(
        &format!(
            "pdep_polarizability_becke (natoms={natoms}, nbf={}, npts={npts}, naux={})",
            obs.nbasis(),
            resp.inter.naux
        ),
        cfg.memory_budget_bytes,
        resp.inter.naux,
        resp.inter.nocc,
        resp.inter.nvir,
        npts,
        obs.nbasis(),
        natoms,
    )?;
    let bra = becke_bra(mol, obs, obs_bs, &resp, &grid, budget)?;
    let mu = resp.molecular_dipole_flat(obs)?;
    closed_per_atom_static(&resp, &bra, &mu)
}

/// Per-atom Becke polarizability tensors α^A_{ij}(iω) at a list of imaginary
/// frequencies. Returns `out[a][k]` = 3×3 tensor for atom `a`, frequency `k`.
///
/// This is the frequency generalization of [`pdep_polarizability_becke`],
/// with the same Krishtal intrinsic definition α^A_dj(iω) = ⟨m^A_d|R(iω)|μ_j⟩
/// (atom-centred Becke dipole bra, analytic molecular-dipole ket; per spin
/// for an open-shell reference). The grid, Becke partition, and per-atom MO
/// dipoles are built once (ω-independent); only the χ₀ "denominator"
/// g_ia(ω) = e_ia/(ω²+e_ia²) and the SMW dielectric
/// ε̃(ω) = I + 4 B̃ diag(g(ω)) B̃^T change per frequency. At ω=0,
/// g_ia = 1/Δε_ia, so a frequency list containing 0.0 reproduces
/// `pdep_polarizability_becke`. All occupied orbitals respond
/// (`cfg.frozen_core` is ignored, as in every polarizability path), so
/// `molecular_dynamic_polarizability(..) − Σ_A out[A]` is the dynamic
/// charge-transfer remainder.
///
/// The Casimir-Polder C6 follow-up consumes these via
/// `dispersion::pdep_dynamic_polarizability`.
#[allow(clippy::too_many_arguments)]
pub fn pdep_polarizability_becke_dynamic(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
    freqs: &[f64],
) -> Result<Vec<Vec<[[f64; 3]; 3]>>, FerricError> {
    if !matches!(rhf.spin, Spin::Restricted) {
        return becke_dynamic_open_shell(mol, obs, obs_bs, dfbs, rhf, op, cfg, freqs);
    }

    let natoms = mol.atoms.len();
    let resp = ClosedResponse::new(mol, obs, dfbs, rhf, op, cfg)?;
    let grid = becke_grid(mol);
    let npts = grid.points.len();
    let (_budget, _charge) = preflight_grid_path(
        &format!(
            "pdep_polarizability_becke_dynamic (natoms={natoms}, nbf={}, npts={npts}, naux={})",
            obs.nbasis(),
            resp.inter.naux
        ),
        cfg.memory_budget_bytes,
        resp.inter.naux,
        resp.inter.nocc,
        resp.inter.nvir,
        npts,
        obs.nbasis(),
        natoms,
    )?;
    let budget = ferric_core::memory::resolve_budget_bytes(cfg.memory_budget_bytes);
    let bra = becke_bra(mol, obs, obs_bs, &resp, &grid, budget)?;
    let mu = resp.molecular_dipole_flat(obs)?;
    closed_per_atom_dynamic(&resp, &bra, &mu, freqs, natoms)
}

/// Per-atom static polarizability tensors via PDEP-RPA with Hirshfeld partitioning.
///
/// The same Krishtal intrinsic definition as [`pdep_polarizability_becke`]
/// (atom-centred bra, analytic molecular-dipole ket) with the Hirshfeld weight
/// `w^A = ρ⁰_A/Σ_B ρ⁰_B` on a bounding-box lattice. Σ_A α^A is not α_mol: the
/// remainder is [`charge_transfer_remainder`]. Closed shell only.
///
/// `proatom`: optional provider of validated same-basis free-atom densities
/// (fixes the H-starvation of the legacy single-Slater proatom); `None` uses
/// the single-exponential Slater proatom.
#[allow(clippy::too_many_arguments)]
pub fn pdep_polarizability_hirshfeld(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
    proatom: Option<&ProatomProvider>,
) -> Result<Vec<[[f64; 3]; 3]>, FerricError> {
    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "pdep_polarizability_hirshfeld: only closed-shell (Restricted) supported".into(),
        ));
    }

    let natoms = mol.atoms.len();
    let resp = ClosedResponse::new(mol, obs, dfbs, rhf, op, cfg)?;
    let lat = hirshfeld_lattice(mol);
    let nbf = obs.nbasis();

    // Pre-flight BEFORE chi allocates. `npts` is the live bounding-box point
    // count and grows with molecular VOLUME (not atom count as on the Becke
    // path); the serial static path has no per-frequency fan-out.
    let (_budget, _charge) = preflight_hirshfeld_path(
        &format!(
            "pdep_polarizability_hirshfeld (natoms={natoms}, nbf={nbf}, npts={}, naux={})",
            lat.npts, resp.inter.naux
        ),
        cfg.memory_budget_bytes,
        resp.inter.naux,
        resp.e_ia.len(),
        lat.npts,
        nbf,
        natoms,
        1,
    )?;
    let bra = hirshfeld_bra(
        "pdep_polarizability_hirshfeld",
        mol,
        obs_bs,
        &resp,
        &lat,
        nbf,
        proatom,
    )?;
    let mu = resp.molecular_dipole_flat(obs)?;
    let alpha = closed_per_atom_static(&resp, &bra, &mu)?;

    if std::env::var("FERRIC_DEBUG_HIRSHFELD").is_ok() {
        eprintln!(
            "[hirshfeld] grid {}×{}×{} (spacing={}, margin={})",
            lat.grid.n_x, lat.grid.n_y, lat.grid.n_z, lat.spacing, lat.margin
        );
        for (a, t) in alpha.iter().enumerate() {
            eprintln!(
                "[hirshfeld] atom {} (Z={}) α_iso = {:.4}",
                a,
                mol.atoms[a].z,
                (t[0][0] + t[1][1] + t[2][2]) / 3.0
            );
        }
    }
    Ok(alpha)
}

/// Dynamic Hirshfeld per-atom polarizability α^A(iω) on an imaginary-frequency grid.
///
/// The Hirshfeld dipoles (grid work, ω-independent) are built once; the
/// full-rank SMW formula of [`pdep_polarizability_becke_dynamic`] is then
/// evaluated at each frequency. The tensors are the Krishtal intrinsic ones
/// (atom-centred bra, analytic molecular-dipole ket), so Σ_A α^A(iω) is not
/// α_mol(iω); see [`charge_transfer_remainder_dynamic`]. Origin-independent.
///
/// Returns `per_atom[A][k][i][j]` = α^A_{ij}(iω_k), shape `(natoms, nfreq, 3, 3)`.
///
/// `proatom`: optional provider of validated same-basis free-atom densities for
/// the Hirshfeld partition (fixes the H-starvation of the legacy single-Slater
/// proatom). `None` uses the legacy single-exponential Slater proatom.
#[allow(clippy::too_many_arguments)]
pub fn pdep_polarizability_hirshfeld_dynamic(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
    freqs: &[f64],
    proatom: Option<&ProatomProvider>,
) -> Result<Vec<Vec<[[f64; 3]; 3]>>, FerricError> {
    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "pdep_polarizability_hirshfeld_dynamic: only closed-shell (Restricted) supported"
                .into(),
        ));
    }

    let natoms = mol.atoms.len();
    let resp = ClosedResponse::new(mol, obs, dfbs, rhf, op, cfg)?;
    let lat = hirshfeld_lattice(mol);
    let nbf = obs.nbasis();

    // Pre-flight BEFORE chi allocates. This is the heaviest of the property
    // paths: on top of the static footprint it adds the per-frequency scratch,
    // a `(naux, nov)` buffer plus an `(naux, naux)` `eps_mat` PER RAYON WORKER.
    preflight_hirshfeld_path(
        &format!(
            "pdep_polarizability_hirshfeld_dynamic (natoms={natoms}, nbf={nbf}, npts={}, \
             naux={}, nfreq={})",
            lat.npts,
            resp.inter.naux,
            freqs.len()
        ),
        cfg.memory_budget_bytes,
        resp.inter.naux,
        resp.e_ia.len(),
        lat.npts,
        nbf,
        natoms,
        rayon::current_num_threads().max(1),
    )?;
    let bra = hirshfeld_bra(
        "pdep_polarizability_hirshfeld_dynamic",
        mol,
        obs_bs,
        &resp,
        &lat,
        nbf,
        proatom,
    )?;
    let mu = resp.molecular_dipole_flat(obs)?;
    closed_per_atom_dynamic(&resp, &bra, &mu, freqs, natoms)
}
