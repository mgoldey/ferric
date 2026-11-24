//! Spin-blocked unrestricted MP3 that keeps the first- and second-order
//! doubles amplitudes `c^(1)` and `c^(2)`.
//!
//! The amplitudes are held in three spin blocks, as in [`crate::u_rimp2`]:
//! αα `[i,j,a,b]`, ββ `[I,J,A,B]` and αβ `[i,J,a,B]` (α first). With
//! `D = ε_i + ε_j − ε_a − ε_b`,
//!
//! ```text
//! c^(1) = K / D,   K^σσ_ijab = (ia|jb) − (ib|ja),   K^αβ_iJaB = (ia|JB)
//! c^(2) = R / D,   R the MP3 doubles residual of c^(1)  (crate::u_mp3_residual)
//! E^(n) = ¼ Σ c^(n−1)_aa K_aa + ¼ Σ c^(n−1)_bb K_bb + Σ c^(n−1)_ab K_ab
//! ```
//!
//! so `E^(2)` and `E^(3)` come out per channel: the αα + ββ sum is the
//! same-spin part and αβ the opposite-spin part (`u_rimp2`'s split).
//!
//! # References
//!
//! RHF (α = β), UHF, and ROHF/ROKS. An ROHF reference is semicanonicalized on
//! its spin Fock matrices first ([`ferric_scf::semicanonical::unrestricted_reference`]),
//! and the energy is doubles-only: the non-Brillouin singles `Σ |f_ia|²/(ε_i−ε_a)`
//! are not included. `frozen_core` drops the lowest spatial orbitals of each spin.
//!
//! # Integrals
//!
//! [`UMp3Eri::Ri`](crate::u_mp3::UMp3Eri::Ri) (Coulomb-metric RI, the same B build as [`crate::mp3`]) or
//! [`UMp3Eri::Exact`](crate::u_mp3::UMp3Eri::Exact) (dense four-centre AO tensor, O(nbas⁴) memory).
//!
//! # Memory
//!
//! The whole call is pre-flighted against the resolved budget before the first
//! large allocation, with ONE peak ([`u_mp3_peak_bytes`](crate::u_mp3::u_mp3_peak_bytes)) that is both checked
//! and charged to the shared pool. The largest spin-blocked VVVV piece is at
//! most `max(nv_α, nv_β)⁴` (exact; the RI path holds one `nv³` slab at a time),
//! against `(2nv)⁴ = 16 nv⁴` for the spin-orbital [`crate::mp3`]. The returned
//! [`UMp3Result`](crate::u_mp3::UMp3Result) keeps `K`, `c^(1)` and `c^(2)` and holds a pool charge for
//! exactly those nine blocks.

use crate::rimp2::active_occ;
use crate::u_mp3_eri::{
    build_ri_spins, exact_block_transient, EriSource, Space, SpinOrbitals, ALPHA, BETA,
};
use crate::u_mp3_residual::{mp3_residual, perm_std, RingInputs};
use ferric_core::memory::pool::Reservation;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ferric_scf::{ScfResult, Spin};
use ndarray::{Array4, Zip};

/// Which integrals the driver uses.
#[derive(Debug, Clone, Copy)]
pub enum UMp3Eri<'a> {
    /// Coulomb-metric RI with this auxiliary basis.
    Ri {
        /// Prepared auxiliary basis.
        aux: &'a PreparedBasis,
    },
    /// Exact four-centre integrals (dense AO tensor).
    Exact,
}

/// Integral kind recorded on the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UMp3EriKind {
    /// Coulomb-metric RI.
    Ri,
    /// Exact four-centre integrals.
    Exact,
}

/// Options of [`u_mp3`].
#[derive(Debug, Clone, Default)]
pub struct UMp3Config {
    /// Lowest spatial orbitals of EACH spin left uncorrelated.
    pub frozen_core: usize,
    /// Memory budget in bytes (`None` = the resolved process budget).
    pub memory_budget_bytes: Option<usize>,
}

/// Doubles tensors in the three stored spin blocks.
#[derive(Debug, Clone)]
pub struct SpinBlockedDoubles {
    /// αα block `[i, j, a, b]`, antisymmetric in `ij` and in `ab`.
    pub aa: Array4<f64>,
    /// ββ block `[I, J, A, B]`, antisymmetric in `IJ` and in `AB`.
    pub bb: Array4<f64>,
    /// αβ block `[i, J, a, B]` (α occupied, β occupied, α virtual, β virtual).
    pub ab: Array4<f64>,
}

impl SpinBlockedDoubles {
    /// Elements held by the three blocks.
    pub fn len(&self) -> usize {
        self.aa.len() + self.bb.len() + self.ab.len()
    }

    /// Whether every block is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An energy split by spin channel (Hartree).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UMp3Channels {
    /// αα same-spin part.
    pub aa: f64,
    /// ββ same-spin part.
    pub bb: f64,
    /// αβ opposite-spin part.
    pub ab: f64,
}

impl UMp3Channels {
    /// Same-spin part, `aa + bb`.
    pub fn same_spin(&self) -> f64 {
        self.aa + self.bb
    }

    /// Opposite-spin part, `ab`.
    pub fn opposite_spin(&self) -> f64 {
        self.ab
    }

    /// `aa + bb + ab`.
    pub fn total(&self) -> f64 {
        self.aa + self.bb + self.ab
    }
}

/// Active orbital counts of a [`u_mp3`] run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UMp3Dims {
    /// Active α occupied.
    pub no_a: usize,
    /// Active β occupied.
    pub no_b: usize,
    /// α virtual.
    pub nv_a: usize,
    /// β virtual.
    pub nv_b: usize,
    /// Basis functions.
    pub nbas: usize,
    /// Auxiliary functions (0 for exact integrals).
    pub naux: usize,
    /// Spatial orbitals frozen in each spin.
    pub frozen_core: usize,
}

/// Result of [`u_mp3`]: channel energies plus the amplitudes and integrals a
/// W-matrix build (`W_ij = ⅛ Σ_kab c_ikab ⟨jk‖ab⟩ + i↔j`) needs.
///
/// Not `Clone`: it holds a pool reservation for the nine tensors (see
/// [`crate::u_rimp2::UMp2Amplitudes`] for why a clone cannot carry one).
#[derive(Debug)]
#[must_use]
pub struct UMp3Result {
    /// Reference SCF energy (Hartree).
    pub e_ref: f64,
    /// Second-order correlation energy by channel.
    pub e2: UMp3Channels,
    /// Third-order correlation increment by channel.
    pub e3: UMp3Channels,
    /// `e_ref + e2 + e3`.
    pub e_total: f64,
    /// `⟨ij‖ab⟩` (αα, ββ) and `⟨iJ|aB⟩ = (ia|JB)` (αβ).
    pub ints: SpinBlockedDoubles,
    /// First-order amplitudes `c^(1) = K / D`.
    pub c1: SpinBlockedDoubles,
    /// Second-order amplitudes `c^(2) = R / D`.
    pub c2: SpinBlockedDoubles,
    /// Active occupied orbital energies `[α, β]` used in `D`.
    pub eps_occ: [Vec<f64>; 2],
    /// Virtual orbital energies `[α, β]` used in `D`.
    pub eps_vir: [Vec<f64>; 2],
    /// Orbital counts.
    pub dims: UMp3Dims,
    /// Integrals used.
    pub eri: UMp3EriKind,
    /// Bytes the call pre-flighted and charged at its peak.
    pub peak_bytes: usize,
    _charge: Reservation,
}

impl UMp3Result {
    /// `e2 + e3` (Hartree).
    pub fn e_corr(&self) -> f64 {
        self.e2.total() + self.e3.total()
    }
}

impl std::fmt::Display for UMp3Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "U-MP3 total: {:.10} Ha (MP2: {:.10}, MP3: {:.10})",
            self.e_total,
            self.e2.total(),
            self.e3.total()
        )
    }
}

/// `(nocc_α, nocc_β)` from the electron count and multiplicity.
fn occupations(mol: &Molecule) -> Result<(usize, usize), FerricError> {
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    if two_s < 0 || nelec < two_s || (nelec + two_s) % 2 != 0 {
        return Err(FerricError::General(format!(
            "u_mp3: {nelec} electrons cannot have multiplicity {}",
            mol.multiplicity
        )));
    }
    Ok((
        ((nelec + two_s) / 2) as usize,
        ((nelec - two_s) / 2) as usize,
    ))
}

/// One spin's active orbitals from full MO coefficients and energies.
fn spin_orbitals(
    c: &ndarray::Array2<f64>,
    eps: &[f64],
    nocc: usize,
    frozen_core: usize,
) -> Result<SpinOrbitals, FerricError> {
    let first = nocc - active_occ(nocc, frozen_core)?;
    let nmo = c.ncols();
    if eps.len() < nmo || nocc > nmo {
        return Err(FerricError::General(format!(
            "u_mp3: {} orbital energies and {nmo} MOs for {nocc} occupied",
            eps.len()
        )));
    }
    Ok(SpinOrbitals {
        c_occ: c.slice(ndarray::s![.., first..nocc]).to_owned(),
        c_vir: c.slice(ndarray::s![.., nocc..]).to_owned(),
        e_occ: eps[first..nocc].to_vec(),
        e_vir: eps[nocc..nmo].to_vec(),
    })
}

/// α and β active orbitals of the reference (ROHF semicanonicalized).
fn reference_orbitals(
    mol: &Molecule,
    scf: &ScfResult,
    frozen_core: usize,
) -> Result<[SpinOrbitals; 2], FerricError> {
    let (na, nb) = occupations(mol)?;
    if matches!(scf.spin, Spin::Restricted) {
        if na != nb {
            return Err(FerricError::General(format!(
                "u_mp3: a restricted (RHF) reference needs a singlet, got multiplicity {}",
                mol.multiplicity
            )));
        }
        let o = spin_orbitals(scf.mos_r(), scf.eps_r(), na, frozen_core)?;
        return Ok([o.clone(), o]);
    }
    let view = ferric_scf::semicanonical::unrestricted_reference(mol, scf)?;
    Ok([
        spin_orbitals(view.mos_a(), view.eps_a(), na, frozen_core)?,
        spin_orbitals(view.mos_b(), view.eps_b(), nb, frozen_core)?,
    ])
}

fn prod(d: [usize; 4]) -> usize {
    d.iter().fold(1usize, |p, &x| p.saturating_mul(x))
}

/// Elements co-resident when the source builds the block of dims `d` (the
/// output itself for RI, the quarter-transform intermediates for exact).
fn block_cost(exact: bool, nbas: usize, d: [usize; 4]) -> usize {
    if exact {
        exact_block_transient(nbas, d).max(prod(d))
    } else {
        prod(d)
    }
}

/// The RI source build: `3A²` (metric, its Cholesky copy and `L⁻¹`), then
/// `V^{-1/2}` + the AO 3-index tensor + the blocks so far + one block's
/// transform output and dressed copy, in the order α (ov, oo, vv), β (...).
fn ri_build_elems(dm: &UMp3Dims) -> usize {
    let (a, n) = (dm.naux, dm.nbas);
    let blocks = [
        dm.no_a * dm.nv_a,
        dm.no_a * dm.no_a,
        dm.nv_a * dm.nv_a,
        dm.no_b * dm.nv_b,
        dm.no_b * dm.no_b,
        dm.nv_b * dm.nv_b,
    ];
    let base = a
        .saturating_mul(a)
        .saturating_add(a.saturating_mul(n).saturating_mul(n));
    let mut held = 0usize;
    let mut peak = a.saturating_mul(a).saturating_mul(3);
    for b in blocks {
        let ab = a.saturating_mul(b);
        peak = peak.max(
            base.saturating_add(held)
                .saturating_add(ab.saturating_mul(2)),
        );
        held = held.saturating_add(ab);
    }
    peak
}

/// Peak elements of the amplitude stage on top of the held source, stage by
/// stage in the order [`u_mp3`] and [`crate::u_mp3_residual::mp3_residual`]
/// allocate (`s_xy` is one spin block of a doubles tensor, `S` all three):
///
/// ```text
/// ovov αα, ββ, αβ builds   (g, K) pairs accumulate to 2S        + one block build
/// c^(1)                     3S (g, K, c1)
/// residual R                4S; ring kernels M^σ in place of g   + one oovv build
/// ring GEMMs                5S (+ the three T^σσ' copies)        + one product
/// exchange rings (αβ)       3S (K, c1, R)  + block + its copy, then copy + T + product
/// hole ladders              3S + oooo block + its copy
/// particle ladders          3S + (exact: full vvvv block) + slab, its copy, one output
/// ```
fn amplitude_stage_elems(dm: &UMp3Dims, exact: bool) -> usize {
    let (oa, ob, va, vb, n) = (dm.no_a, dm.no_b, dm.nv_a, dm.nv_b, dm.nbas);
    let bc = |d: [usize; 4]| block_cost(exact, n, d);
    let (saa, sbb, sab) = (
        prod([oa, oa, va, va]),
        prod([ob, ob, vb, vb]),
        prod([oa, ob, va, vb]),
    );
    let s = saa + sbb + sab;
    let builds = bc([oa, va, oa, va])
        .max(2 * saa + bc([ob, vb, ob, vb]))
        .max(2 * (saa + sbb) + bc([oa, va, ob, vb]))
        .max(3 * s);
    let ring =
        (4 * s + bc([oa, oa, va, va]).max(bc([ob, ob, vb, vb]))).max(5 * s + saa.max(sbb).max(sab));
    let exchange = |d: [usize; 4]| {
        let z = prod(d);
        bc(d).max(2 * z).max(z + 2 * sab)
    };
    let exch = exchange([oa, oa, vb, vb]).max(exchange([va, va, ob, ob]));
    let hole = |d: [usize; 4]| bc(d).max(2 * prod(d));
    let holes = hole([oa, oa, oa, oa])
        .max(hole([ob, ob, ob, ob]))
        .max(hole([oa, oa, ob, ob]));
    // RI: the slab, its [c,d,b] copy and one output; exact: the whole block
    // with the slab borrowed from it, plus the copy and the output.
    let ladder = |v1: usize, v2: usize, o12: usize| {
        let copy_and_out = v1 * v2 * v2 + o12 * v2;
        if exact {
            bc([v1, v1, v2, v2]).max(prod([v1, v1, v2, v2]) + copy_and_out)
        } else {
            v1 * v2 * v2 + copy_and_out
        }
    };
    let ladders = ladder(va, va, oa * oa)
        .max(ladder(va, vb, oa * ob))
        .max(ladder(vb, vb, ob * ob));
    builds.max(ring).max(3 * s + exch.max(holes).max(ladders))
}

/// Bytes [`u_mp3`] holds at its co-resident peak: the orbital copies, then the
/// max of the source build and (held source + amplitude stage).
///
/// Excluded: O(n) vectors (orbital energies), per-rayon-worker scratch inside
/// the AO integral and 3-index transforms (it reads the thread count, which
/// must not decide admission), and an ROHF reference's semicanonical copy,
/// which is dropped before the gate runs.
pub fn u_mp3_peak_bytes(dm: &UMp3Dims, exact: bool) -> usize {
    let n = dm.nbas;
    let coeffs = n.saturating_mul(dm.no_a + dm.nv_a + dm.no_b + dm.nv_b);
    let (build, held) = if exact {
        let ao = n.saturating_pow(4);
        (ao, ao)
    } else {
        let a = dm.naux;
        let held = a.saturating_mul(
            dm.no_a * dm.no_a
                + dm.no_a * dm.nv_a
                + dm.nv_a * dm.nv_a
                + dm.no_b * dm.no_b
                + dm.no_b * dm.nv_b
                + dm.nv_b * dm.nv_b,
        );
        (ri_build_elems(dm), held)
    };
    let stage = build.max(held.saturating_add(amplitude_stage_elems(dm, exact)));
    coeffs.saturating_add(stage).saturating_mul(8)
}

/// Bytes the returned [`UMp3Result`] holds: `K`, `c^(1)`, `c^(2)`.
fn held_result_bytes(dm: &UMp3Dims) -> usize {
    let s = prod([dm.no_a, dm.no_a, dm.nv_a, dm.nv_a])
        + prod([dm.no_b, dm.no_b, dm.nv_b, dm.nv_b])
        + prod([dm.no_a, dm.no_b, dm.nv_a, dm.nv_b]);
    s.saturating_mul(3).saturating_mul(8)
}

/// `D[i,j,a,b] = e_i + e_j − e_a − e_b`, applied as `x ↦ x / D` in place.
fn divide_by_denominator(x: &mut Array4<f64>, eo1: &[f64], eo2: &[f64], ev1: &[f64], ev2: &[f64]) {
    Zip::indexed(x).for_each(|(i, j, a, b), v| {
        *v /= eo1[i] + eo2[j] - ev1[a] - ev2[b];
    });
}

/// `⟨ij‖ab⟩[i,j,a,b] = g[i,a,j,b] − g[i,b,j,a]` from a same-spin ovov block.
fn antisymmetrize(g: &Array4<f64>) -> Array4<f64> {
    let mut k = perm_std(g, [0, 2, 1, 3]);
    k -= &g.view().permuted_axes([0, 2, 3, 1]);
    k
}

fn contract(a: &Array4<f64>, b: &Array4<f64>) -> f64 {
    Zip::from(a).and(b).fold(0.0, |s, &x, &y| s + x * y)
}

/// `¼ c^αα·k^αα`, `¼ c^ββ·k^ββ` and `c^αβ·k^αβ`: the per-channel energy of
/// amplitudes `c` against the integrals (or residual) `k`.
pub(crate) fn channel_energy(c: &SpinBlockedDoubles, k: &SpinBlockedDoubles) -> UMp3Channels {
    UMp3Channels {
        aa: 0.25 * contract(&c.aa, &k.aa),
        bb: 0.25 * contract(&c.bb, &k.bb),
        ab: contract(&c.ab, &k.ab),
    }
}

/// Apply `x ↦ x / D` to all three blocks.
fn divide_blocks(x: &mut SpinBlockedDoubles, orbs: &[SpinOrbitals; 2]) {
    let (a, b) = (&orbs[ALPHA], &orbs[BETA]);
    divide_by_denominator(&mut x.aa, &a.e_occ, &a.e_occ, &a.e_vir, &a.e_vir);
    divide_by_denominator(&mut x.bb, &b.e_occ, &b.e_occ, &b.e_vir, &b.e_vir);
    divide_by_denominator(&mut x.ab, &a.e_occ, &b.e_occ, &a.e_vir, &b.e_vir);
}

/// Build the integral source (after the gate).
pub(crate) fn build_source<'a>(
    obs: &PreparedBasis,
    op: Operator,
    eri: UMp3Eri<'_>,
    orbs: &'a [SpinOrbitals; 2],
) -> Result<EriSource<'a>, FerricError> {
    match eri {
        UMp3Eri::Exact => Ok(EriSource::Exact {
            ao: crate::canonical::dense_ao_eri(obs, op)?,
            nbas: obs.nbasis(),
            orbs,
        }),
        UMp3Eri::Ri { aux } => {
            let v_inv_sqrt = {
                let v2c = threeindex::coulomb_metric_2c(op, aux)?;
                crate::rimp2::cholesky_inverse_sqrt(&v2c)?
            };
            let eri3_ao = threeindex::eri3_tensor(op, obs, aux)?;
            Ok(EriSource::Ri(Box::new(build_ri_spins(
                &eri3_ao,
                &v_inv_sqrt,
                orbs,
            ))))
        }
    }
}

/// The doubles integrals `K` (`⟨ij‖ab⟩` αα / ββ, `(ia|JB)` αβ) of a built
/// source, and the ovov blocks the residual's ring terms consume. Valid in
/// any orthonormal orbitals the source was built over (nothing here reads
/// an orbital energy).
pub(crate) fn doubles_integrals(
    src: &EriSource<'_>,
) -> Result<(SpinBlockedDoubles, RingInputs), FerricError> {
    let (o, v) = (Space::Occ, Space::Vir);
    let g_aa = src.block((ALPHA, o, v), (ALPHA, o, v))?;
    let k_aa = antisymmetrize(&g_aa);
    let g_bb = src.block((BETA, o, v), (BETA, o, v))?;
    let k_bb = antisymmetrize(&g_bb);
    let g_ab = src.block((ALPHA, o, v), (BETA, o, v))?;
    let k_ab = perm_std(&g_ab, [0, 2, 1, 3]);
    let ints = SpinBlockedDoubles {
        aa: k_aa,
        bb: k_bb,
        ab: k_ab,
    };
    Ok((ints, RingInputs { g_aa, g_bb, g_ab }))
}

/// The amplitudes and energies from a built source.
fn amplitudes(
    src: &EriSource<'_>,
    orbs: &[SpinOrbitals; 2],
) -> Result<(SpinBlockedDoubles, SpinBlockedDoubles, SpinBlockedDoubles), FerricError> {
    let (ints, ring) = doubles_integrals(src)?;
    let mut c1 = ints.clone();
    divide_blocks(&mut c1, orbs);
    let mut c2 = mp3_residual(src, ring, &c1)?;
    divide_blocks(&mut c2, orbs);
    Ok((ints, c1, c2))
}

/// One spin of a reference determinant handed to [`u_mp3_from_orbitals`].
#[derive(Debug, Clone, Copy)]
pub struct UMp3SpinReference<'a> {
    /// MO coefficients `(nbas, nmo)`, occupied first.
    pub mos: &'a ndarray::Array2<f64>,
    /// Orbital energies used in the denominators, at least `nmo` of them.
    pub eps: &'a [f64],
    /// Occupied orbitals of this spin (frozen core included).
    pub nocc: usize,
}

/// Spin-blocked MP2 + MP3 from an RHF, UHF or ROHF/ROKS reference, keeping
/// `c^(1)` and `c^(2)` per spin channel.
///
/// `op` is the two-electron operator of the correlation integrals (Coulomb
/// for MP3). Errors before any large allocation when [`u_mp3_peak_bytes`]
/// exceeds the budget.
pub fn u_mp3(
    mol: &Molecule,
    obs: &PreparedBasis,
    op: Operator,
    scf: &ScfResult,
    eri: UMp3Eri<'_>,
    config: &UMp3Config,
) -> Result<UMp3Result, FerricError> {
    let orbs = reference_orbitals(mol, scf, config.frozen_core)?;
    run_on_orbitals(obs, op, orbs, scf.energy, eri, config)
}

/// [`u_mp3`] on explicitly given per-spin orbitals `[α, β]` and reference
/// energy `e_ref`, for a caller that built its own (semi)canonical orbitals.
///
/// The orbitals are used as given: the denominators take `eps`, and nothing
/// is semicanonicalized. `config.frozen_core` drops the first occupied
/// orbitals of each spin.
pub fn u_mp3_from_orbitals(
    obs: &PreparedBasis,
    op: Operator,
    spins: [UMp3SpinReference<'_>; 2],
    e_ref: f64,
    eri: UMp3Eri<'_>,
    config: &UMp3Config,
) -> Result<UMp3Result, FerricError> {
    let [a, b] = spins;
    let orbs = [
        spin_orbitals(a.mos, a.eps, a.nocc, config.frozen_core)?,
        spin_orbitals(b.mos, b.eps, b.nocc, config.frozen_core)?,
    ];
    run_on_orbitals(obs, op, orbs, e_ref, eri, config)
}

fn run_on_orbitals(
    obs: &PreparedBasis,
    op: Operator,
    orbs: [SpinOrbitals; 2],
    e_ref: f64,
    eri: UMp3Eri<'_>,
    config: &UMp3Config,
) -> Result<UMp3Result, FerricError> {
    let (exact, naux, kind) = match eri {
        UMp3Eri::Exact => (true, 0, UMp3EriKind::Exact),
        UMp3Eri::Ri { aux } => (false, aux.nbasis(), UMp3EriKind::Ri),
    };
    let dims = UMp3Dims {
        no_a: orbs[ALPHA].n(Space::Occ),
        no_b: orbs[BETA].n(Space::Occ),
        nv_a: orbs[ALPHA].n(Space::Vir),
        nv_b: orbs[BETA].n(Space::Vir),
        nbas: obs.nbasis(),
        naux,
        frozen_core: config.frozen_core,
    };
    let label = format!(
        "U-MP3 ({kind:?} integrals; occ {}a/{}b, vir {}a/{}b, nbas {}, naux {naux})",
        dims.no_a, dims.no_b, dims.nv_a, dims.nv_b, dims.nbas
    );
    let peak = u_mp3_peak_bytes(&dims, exact);
    ferric_core::memory::check_alloc(
        &label,
        peak,
        ferric_core::memory::resolve_budget_bytes(config.memory_budget_bytes),
    )?;
    let peak_charge = crate::rimp2::charge_mo_side(&label, peak)?;

    let (ints, c1, c2) = {
        let src = build_source(obs, op, eri, &orbs)?;
        amplitudes(&src, &orbs)?
    };
    let e2 = channel_energy(&c1, &ints);
    let e3 = channel_energy(&c2, &ints);
    drop(peak_charge);
    let held =
        crate::rimp2::charge_mo_side(&format!("{label} amplitudes"), held_result_bytes(&dims))?;
    let [a, b] = orbs;
    Ok(UMp3Result {
        e_ref,
        e2,
        e3,
        e_total: e_ref + e2.total() + e3.total(),
        ints,
        c1,
        c2,
        eps_occ: [a.e_occ, b.e_occ],
        eps_vir: [a.e_vir, b.e_vir],
        dims,
        eri: kind,
        peak_bytes: peak,
        _charge: held,
    })
}
