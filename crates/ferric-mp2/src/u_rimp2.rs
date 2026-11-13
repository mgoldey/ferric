//! Unrestricted RI-MP2 energy.
//!
//! For UHF/ROHF references with separate α and β MO sets. The MP2
//! correlation energy decomposes into three blocks:
//!
//! - `αα`: `¼ Σ_{ij,ab} |⟨ia||jb⟩_α|² / (ε_iα+ε_jα-ε_aα-ε_bα)` (antisymmetrized)
//! - `ββ`: same with β
//! - `αβ`: `Σ_{iJ,aB} (ia|JB)² / (ε_iα+ε_Jβ-ε_aα-ε_Bβ)` (no exchange)
//!
//! where `(ia|jb)_σ = Σ_P B^P_{ia,σ} B^P_{jb,σ}` and
//! `(ia|JB) = Σ_P B^P_{ia,α} B^P_{JB,β}` (shared aux metric).
//!
//! A ROHF/ROKS reference is semi-canonicalized on entry
//! (`ferric_scf::semicanonical::unrestricted_reference`): each spin uses the
//! occ–occ / virt–virt eigenvectors and eigenvalues of its own Fock `F_σ`. The
//! energy is the doubles-only UMP2 expression on those orbitals; the ROHF-MBPT
//! single-excitation term `Σ_ia |f_ia^σ|² / (ε_i − ε_a)` (non-zero because ROHF is
//! not a UHF stationary point) is NOT included.

use crate::rimp2::{compute_rpa_intermediates_spin, RiMp2Config, RpaIntermediates};
use ferric_core::mol::Molecule;
use ferric_core::timing::PhaseAccumulator;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::{ScfResult, Spin};
use ndarray::{Array2, Array3, Array4, ArrayView2};

/// Components of the U-RI-MP2 correlation energy.
#[derive(Debug, Clone, PartialEq)]
pub struct URiMp2Components {
    /// αα same-spin correlation energy, `¼ Σ t^αα K^αα` (Hartree).
    pub e_aa: f64,
    /// ββ same-spin correlation energy, `¼ Σ t^ββ K^ββ` (Hartree).
    pub e_bb: f64,
    /// αβ opposite-spin correlation energy, `Σ t^αβ (ia|JB)` (Hartree).
    pub e_ab: f64,
    /// `e_aa + e_bb + e_ab` (Hartree).
    pub e_total: f64,
}

/// Result of an unrestricted RI-MP2 calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct URiMp2Result {
    /// Spin-resolved correlation energy.
    pub components: URiMp2Components,
    /// UMP2 correlation energy, equal to `components.e_total` (Hartree).
    pub mp2_corr: f64,
    /// Reference SCF energy plus `mp2_corr` (Hartree).
    pub total_energy: f64,
    /// Precision provenance of the correlation energy (see
    /// `RiMp2Result::precision`).
    pub precision: ferric_core::gpu::provenance::PrecisionRecord,
}

impl std::fmt::Display for URiMp2Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "U-RI-MP2 total: {:.10} Ha (corr: {:.10})",
            self.total_energy, self.mp2_corr
        )
    }
}

/// All U-MP2 amplitudes + per-spin intermediates, for downstream
/// gradient / orbital-optimization work.
///
/// Amplitudes follow Bozkaya 2013 conventions:
/// - `t_aa[i,j,a,b] = [(ia|jb)_α − (ib|ja)_α] / D^αα` (antisymmetric ij↔ab)
/// - `t_bb[i,j,a,b] = [(ia|jb)_β − (ib|ja)_β] / D^ββ`
/// - `t_ab[i,J,a,B] = (ia|JB) / D^αβ` (no antisymmetrization; mixed spin)
///
/// Index conventions: `i,j` ∈ occ_α, `I,J` ∈ occ_β, `a,b` ∈ vir_α,
/// `A,B` ∈ vir_β. The αβ tensor's first two axes are α-occ × β-occ, last
/// two are α-vir × β-vir.
///
/// # Not `Clone` since the pool migration
///
/// This derived `Clone`. It now holds an RAII `Reservation` covering all three
/// amplitude tensors, and there is no honest way to clone that: duplicating
/// the guard credits the pool twice for bytes released once, and handing the
/// copy an inert guard leaves several GB resident with nothing outstanding.
/// Nothing in the workspace clones one (verified: `build_u_mp2_density`,
/// `compute_u_mp2_orbital_gradient` and `_blocks` all take
/// `&UMp2Amplitudes`, and it is constructed at exactly one site).
#[derive(Debug)]
pub struct UMp2Amplitudes {
    /// α-spin RI intermediates (`b_ov` over active α occupieds × α virtuals).
    pub inter_a: RpaIntermediates,
    /// β-spin RI intermediates.
    pub inter_b: RpaIntermediates,
    /// α orbital energies (Hartree), all MOs, used in the denominators.
    pub eps_a: Vec<f64>,
    /// β orbital energies (Hartree), all MOs, used in the denominators.
    pub eps_b: Vec<f64>,
    /// αα amplitudes `t[i,j,a,b] = [(ia|jb) − (ib|ja)] / D` with
    /// `D = ε_i + ε_j − ε_a − ε_b`, shape `(nocc_α, nocc_α, nvir_α, nvir_α)`.
    pub t_aa: Array4<f64>,
    /// ββ amplitudes, same form and layout as `t_aa`.
    pub t_bb: Array4<f64>,
    /// αβ amplitudes `t[i,J,a,B] = (ia|JB) / D`, shape
    /// `(nocc_α, nocc_β, nvir_α, nvir_β)`.
    pub t_ab: Array4<f64>,
    /// Energies accumulated while building the amplitudes.
    pub components: URiMp2Components,
    /// RAII charges against the shared memory pool for everything this struct
    /// owns, held for as long as the struct is.
    ///
    /// A PAIR, because the two groups have different hardness: `.0` is the
    /// SOFT charge for both spins' `b_ov` (the AO 3-index tensor spilling is
    /// its real fallback -- see the measured window at the reservation site),
    /// and `.1` is the HARD charge for the three amplitude tensors, which have
    /// no fallback at all.
    ///
    /// The three amplitude tensors are the dominant planes on this path and
    /// they are ALL live simultaneously (`check_u_mo_side_alloc`'s own doc
    /// says so and charges `n_amplitude_sets = 3`). They escape into this
    /// struct, so a guard dropped when `compute_u_mp2_amplitudes` returns
    /// would credit the pool back while the tensors are resident for the whole
    /// downstream gradient / orbital-optimization pipeline.
    ///
    /// The two `b_ov` tensors in `inter_a`/`inter_b` are charged separately by
    /// the caller's own guard, which outlives this one, because
    /// `RpaIntermediates` is constructed by struct literal in ferric-rpa and
    /// so cannot carry a field of its own.
    ///
    /// Inert when no pool is installed.
    pub(crate) _charge: (
        ferric_core::memory::pool::Reservation,
        ferric_core::memory::pool::Reservation,
    ),
}

/// The `(naux, nocc_α, nvir_α, nocc_β, nvir_β)` shapes
/// [`compute_rpa_intermediates_spin`] will produce, derived WITHOUT building
/// them.
///
/// This exists so the pre-flight gate can run *before* the two spin
/// intermediates are allocated rather than after. Previously both entry points
/// built `inter_a` and `inter_b` first and only then called
/// `check_u_mo_side_alloc` — a guard that charges those very buffers. On a job
/// too large for the budget the process died inside the allocation the guard
/// was meant to refuse, and the error message was an OOM kill rather than the
/// actionable "raise budget_gb" diagnostic.
///
/// The occupancy arithmetic MIRRORS `compute_rpa_intermediates_spin` exactly
/// (`nocc_σ = (N ± 2S)/2`, `nvir_σ = nbas − nocc_total_σ`, active occupied via
/// [`active_occ`]). That is a duplication, but a small, closed-form one over
/// integer counts — not a second implementation of the ALLOCATION shapes, which
/// is the drift class this pass is removing. Returning an error for a bad
/// frozen-core count matches what the builder would have done.
fn u_mo_side_dims(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    scf: &ScfResult,
    mol: &Molecule,
    config: &RiMp2Config,
) -> Result<(usize, usize, usize, usize, usize), FerricError> {
    let nbas = obs.nbasis();
    let naux = dfbs.nbasis();
    let nelec_total = mol.nelec() as usize;
    let two_s = mol.multiplicity as i32 - 1;
    let nocc_total_a = ((nelec_total as i32 + two_s) / 2) as usize;
    let nocc_total_b = ((nelec_total as i32 - two_s) / 2) as usize;
    let no_a = crate::rimp2::active_occ(nocc_total_a, config.frozen_core)?;
    let no_b = crate::rimp2::active_occ(nocc_total_b, config.frozen_core)?;
    let _ = scf; // shapes depend only on the molecule/basis, not the MO values
    Ok((naux, no_a, nbas - nocc_total_a, no_b, nbas - nocc_total_b))
}

/// Compute the U-RI-MP2 correlation energy from a UHF or ROHF reference.
///
/// Reuses `compute_rpa_intermediates_spin` to build per-spin
/// `B^P_{ia,σ}` tensors (occ-vir block, dressed with V^{-1/2}).
pub fn u_ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    scf: &ScfResult,
    config: &RiMp2Config,
) -> Result<URiMp2Result, FerricError> {
    let probe = ferric_core::gpu::provenance::PrecisionProbe::start();
    if matches!(scf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "u_ri_mp2: requires UHF or ROHF reference".into(),
        ));
    }
    // ROHF/ROKS -> per-spin semi-canonical orbitals and energies; UHF borrowed.
    let scf_view = ferric_scf::semicanonical::unrestricted_reference(mol, scf)?;
    let scf: &ScfResult = &scf_view;

    // Gate BEFORE the spin intermediates, not after.
    //
    // `check_u_mo_side_alloc`'s own doc says it charges the "TWO spin
    // intermediates ... retained simultaneously", and `check_alloc`'s doc says
    // a pre-flight gate belongs "before the first large allocation". Both were
    // violated here: `inter_a` and `inter_b` — the very `(naux, nocc_σ·nvir_σ)`
    // buffers the guard charges — were allocated FIRST, so on a job too large
    // for the budget the process OOM-killed inside
    // `compute_rpa_intermediates_spin` and the guard never ran. A gate that
    // runs after the allocation it bounds is not a gate.
    //
    // The shapes are derived from the SCF result rather than from the built
    // intermediates, which is what makes the reordering possible at all.
    let (naux, no_a, nv_a, no_b, nv_b) = u_mo_side_dims(obs, dfbs, scf, mol, config)?;
    crate::rimp2::check_u_mo_side_alloc(
        "U-RI-MP2",
        naux,
        no_a,
        nv_a,
        no_b,
        nv_b,
        0,
        crate::rimp2::eri3_budget_bytes(config.memory_budget_bytes),
    )?;

    let inter_a = compute_rpa_intermediates_spin(mol, obs, dfbs, op, scf, config, true)?;
    let inter_b = compute_rpa_intermediates_spin(mol, obs, dfbs, op, scf, config, false)?;

    // DEBIT the shared pool for BOTH spins' b_ov, held for the rest of this
    // function -- which is exactly how long they are resident.
    //
    // Taken AFTER the two `compute_rpa_intermediates_spin` calls return, not
    // before, and that ordering is the fix for a MEASURED regression rather
    // than a style choice.
    //
    // Each of those calls builds an AO 3-index tensor through
    // `ferric_integrals::ThreeIndexSource`, which takes its OWN hard pool
    // charge for it and releases that charge when the source drops at the end
    // of the call. A b_ov reservation held ACROSS the call is therefore
    // outstanding at the moment the integrals layer asks for the AO tensor,
    // and the pool refuses THAT allocation -- naming the AO tensor as the
    // plane that failed and this charge as the incumbent. The gate refuses the
    // allocation it reserved for. Benzene cation / cc-pVDZ (nao=114, naux=420,
    // alpha 21o/93v, beta 20o/94v), same binary, no pool (= pre-migration) vs
    // a pool of the same size, with the charge taken BEFORE:
    //
    //   budget   pre-migration   charge-taken-before
    //    32 MB   OK              OK       (AO tensor spills)
    //    43 MB   OK              REFUSED  "DF 3-index (P|mn) in-core needs
    //                                      0.044 GB but only 0.031 GB free"
    //    50 MB   OK              REFUSED  "... only 0.037 GB free"
    //    54 MB   OK              REFUSED  "... only 0.042 GB free"
    //    56 MB   OK              OK
    //
    // The window is exactly [ao, ao + b_ov_a + b_ov_b) = [43.67, 56.55] MB.
    // Making the charge SOFT does NOT fix this: the soft charge SUCCEEDS
    // (12.88 MB fits), so it is held, and the subsequent hard AO charge is the
    // one that fails. Only moving the reservation past the AO build removes
    // the false overlap -- which is correct on the facts, because the AO
    // tensor and b_ov are NOT co-resident here: the source is dropped inside
    // `compute_rpa_intermediates_spin` before it returns b_ov.
    //
    // It stays SOFT nonetheless. The two spins' b_ov ARE co-resident with each
    // other, and this reservation covers both; if a future caller reintroduces
    // an overlap the soft branch degrades to a warning rather than a refusal,
    // per the standing decision that the migration must not refuse a job the
    // current tree completes.
    let b_ov_bytes =
        |no: usize, nv: usize| naux.saturating_mul(no).saturating_mul(nv).saturating_mul(8);
    let _both_spins_charge = crate::rimp2::charge_mo_side_soft(
        "U-RI-MP2 b_ov (alpha + beta)",
        b_ov_bytes(no_a, nv_a).saturating_add(b_ov_bytes(no_b, nv_b)),
    );

    // The per-worker energy transient of the pair-energy loops below, charged
    // for the LARGER spin only: `same_spin_pair_energy` runs sequentially for
    // alpha then beta, so the two fan-outs do not coexist. SOFT, because the
    // size reads `rayon::current_num_threads()` -- see `charge_mo_side_soft`.
    let g_i = |no: usize, nv: usize| {
        no.saturating_mul(nv)
            .saturating_mul(nv)
            .saturating_mul(8)
            .saturating_mul(rayon::current_num_threads().max(1))
    };
    let _g_i_charge = crate::rimp2::charge_mo_side_soft(
        "U-RI-MP2 g_i per-worker energy transient",
        g_i(no_a, nv_a).max(g_i(no_b, nv_b)),
    );

    // Stage-seam RSS safety net: both spin intermediates are now resident, so
    // this is the first point where the pre-flight estimate above can be
    // compared against reality. Observational only, never an error.
    ferric_core::memory::warn_if_rss_over(
        "U-RI-MP2 spin intermediates built",
        crate::rimp2::eri3_budget_bytes(config.memory_budget_bytes),
        1.1,
    );

    // `scf` is UHF-shaped here (a ROHF input was semi-canonicalized above).
    let eps_a: &[f64] = scf.eps_a();
    let eps_b: &[f64] = scf.eps_b();

    // The mixed-precision error map (`[gpu] precision = "mixed"`, kernel
    // `rimp2-energy`) covers the Coulomb, erfc and terfc energies
    // (`rimp2::mixed_energy_operator`); any other operator runs the device in f64.
    let mixed_ok = config.kappa.is_none() && crate::rimp2::mixed_energy_operator(op);
    let e_aa = same_spin_pair_energy(&inter_a, eps_a, mixed_ok);
    let e_bb = same_spin_pair_energy(&inter_b, eps_b, mixed_ok);
    let e_ab = opposite_spin_pair_energy(&inter_a, &inter_b, eps_a, eps_b, mixed_ok);

    let e_total = e_aa + e_bb + e_ab;
    Ok(URiMp2Result {
        components: URiMp2Components {
            e_aa,
            e_bb,
            e_ab,
            e_total,
        },
        mp2_corr: e_total,
        total_energy: scf.energy + e_total,
        precision: probe.finish(ferric_core::gpu::provenance::RIMP2_KERNELS),
    })
}

/// U-RI-MP2 correlation components from caller-supplied parts: the two
/// per-spin RI intermediates (dressed `B^P_{ia,σ}` in ONE shared aux metric)
/// and each spin's FULL orbital-energy array (`eps_σ[first_occ + i]` occupied,
/// `eps_σ[nocc_total + a]` virtual — the indexing the pair kernels use).
///
/// Added for the periodic (Gamma-point) UHF driver in `ferric-pbc`, whose
/// `b_ov` per spin is transformed from one lattice-summed RS-GDF tensor (no
/// molecular aux basis exists) and whose occupied energies carry a per-spin
/// Madelung shift the caller applies. It calls the SAME
/// `same_spin_pair_energy` (α, β) and `opposite_spin_pair_energy` kernels as
/// [`u_ri_mp2`]; `u_ri_mp2` itself is unchanged and does not call this
/// function, so the molecular path is byte-identical by construction.
///
/// Errors (typed, never a panic): `inter_a.naux != inter_b.naux` (the αβ
/// block contracts over one shared aux index), a `b_ov` whose shape is not
/// `(naux, nocc·nvir)`, `first_occ + nocc != nocc_total`, or an energy array
/// shorter than `nocc_total + nvir`. An EMPTY spin channel (`nocc = 0` or
/// `nvir = 0`) contributes exactly 0 to its same-spin and the αβ block
/// without forming a zero-width product. Non-finite energies (zero
/// denominators) are returned as-is for the caller to reject. No pool charge
/// is taken: the caller owns (and has budgeted) both `b_ov`.
pub fn u_ri_mp2_from_parts(
    inter_a: &RpaIntermediates,
    inter_b: &RpaIntermediates,
    eps_a: &[f64],
    eps_b: &[f64],
) -> Result<URiMp2Components, FerricError> {
    if inter_a.naux != inter_b.naux {
        return Err(FerricError::General(format!(
            "u_ri_mp2_from_parts: alpha and beta intermediates have different aux dimensions \
             ({} vs {}); the opposite-spin block needs ONE shared aux metric",
            inter_a.naux, inter_b.naux
        )));
    }
    for (label, inter, eps) in [("alpha", inter_a, eps_a), ("beta", inter_b, eps_b)] {
        if inter.b_ov.dim() != (inter.naux, inter.nocc * inter.nvir) {
            return Err(FerricError::General(format!(
                "u_ri_mp2_from_parts: {label} b_ov is {:?}, expected (naux, nocc·nvir) = ({}, {})",
                inter.b_ov.dim(),
                inter.naux,
                inter.nocc * inter.nvir
            )));
        }
        if inter.first_occ + inter.nocc != inter.nocc_total {
            return Err(FerricError::General(format!(
                "u_ri_mp2_from_parts: {label} first_occ {} + nocc {} != nocc_total {}",
                inter.first_occ, inter.nocc, inter.nocc_total
            )));
        }
        if eps.len() < inter.nocc_total + inter.nvir {
            return Err(FerricError::General(format!(
                "u_ri_mp2_from_parts: {label} has {} orbital energies, needs nocc_total {} + \
                 nvir {}",
                eps.len(),
                inter.nocc_total,
                inter.nvir
            )));
        }
    }
    let live = |i: &RpaIntermediates| i.nocc > 0 && i.nvir > 0;
    let e_aa = if live(inter_a) {
        same_spin_pair_energy(inter_a, eps_a, false)
    } else {
        0.0
    };
    let e_bb = if live(inter_b) {
        same_spin_pair_energy(inter_b, eps_b, false)
    } else {
        0.0
    };
    let e_ab = if live(inter_a) && live(inter_b) {
        opposite_spin_pair_energy(inter_a, inter_b, eps_a, eps_b, false)
    } else {
        0.0
    };
    Ok(URiMp2Components {
        e_aa,
        e_bb,
        e_ab,
        e_total: e_aa + e_bb + e_ab,
    })
}

/// Compute U-MP2 amplitudes for all three spin blocks.
///
/// Returns the αα, ββ, αβ amplitude tensors plus the per-spin RI
/// intermediates and orbital energies (needed by downstream gradient
/// / orbital-optimization code).
pub fn compute_u_mp2_amplitudes(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    scf: &ScfResult,
    config: &RiMp2Config,
) -> Result<UMp2Amplitudes, FerricError> {
    if matches!(scf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "compute_u_mp2_amplitudes: requires UHF or ROHF reference".into(),
        ));
    }
    // ROHF/ROKS -> per-spin semi-canonical orbitals and energies; UHF borrowed.
    let scf_view = ferric_scf::semicanonical::unrestricted_reference(mol, scf)?;
    let scf: &ScfResult = &scf_view;

    // Gate BEFORE the spin intermediates — same defect and same reasoning as in
    // `u_ri_mp2` above: this guard charges `inter_a`/`inter_b` themselves, so
    // running it after they are built means an over-budget job OOMs inside the
    // allocation the guard exists to refuse. See `u_mo_side_dims`.
    let (naux, no_a, nv_a, no_b, nv_b) = u_mo_side_dims(obs, dfbs, scf, mol, config)?;
    crate::rimp2::check_u_mo_side_alloc(
        "U-MP2 amplitudes",
        naux,
        no_a,
        nv_a,
        no_b,
        nv_b,
        3,
        crate::rimp2::eri3_budget_bytes(config.memory_budget_bytes),
    )?;

    // DEBIT the shared pool for everything this function's RESULT will hold:
    // both spins' `b_ov` and all three amplitude tensors.
    //
    // Taken HERE -- before `compute_rpa_intermediates_spin` allocates the
    // first byte -- because the point of a gate is to answer "does this fit?"
    // BEFORE the allocation, not to observe an OOM afterwards. And held in ONE
    // guard that moves into the returned `UMp2Amplitudes` below, because every
    // tensor it covers escapes into that struct: a guard scoped to this
    // function body would credit the bytes back at `return` while the caller
    // still holds all five tensors, which is precisely the lifetime bug the
    // `_charge` field pattern exists to prevent.
    //
    // `t_ab` is charged at its REAL shape (nocc_a*nocc_b*nvir_a*nvir_b) rather
    // than `check_u_mo_side_alloc`'s `max(no)^2 * max(nv)^2` upper bound: an
    // over-estimating guard refuses jobs that would have fit, and on a
    // spin-polarized system that bound overshoots by (max/min)^2 per axis.
    //
    // SPLIT into two charges with different hardness, because the two groups
    // have different fallbacks:
    //
    //   b_ov x2   SOFT. Taken before `compute_rpa_intermediates_spin`, it is
    //             outstanding when that function asks `ThreeIndexSource` for
    //             the AO 3-index tensor, and a HARD charge then refuses THAT
    //             allocation with this one as the incumbent -- the gate
    //             refusing the allocation it reserved for. Measured on the
    //             sibling `u_ri_mp2` path (benzene cation / cc-pVDZ): the
    //             refusal window is [ao, ao + b_ov_a + b_ov_b) =
    //             [43.67, 56.55] MB, a band the pre-migration tree completes.
    //             The AO tensor spilling IS the fallback, and it is already
    //             exercised below that window. See `u_ri_mp2`'s note.
    //
    //   t_aa/t_bb/t_ab  HARD. Dense by construction, no blocked or streaming
    //             alternative anywhere on this path, and they escape into the
    //             returned struct. Charged AFTER the intermediates are built,
    //             so the AO tensor's own charge has already been released and
    //             this cannot refuse it.
    let b_ov_bytes =
        |no: usize, nv: usize| naux.saturating_mul(no).saturating_mul(nv).saturating_mul(8);
    let t_bytes = |no1: usize, no2: usize, nv1: usize, nv2: usize| {
        no1.saturating_mul(no2)
            .saturating_mul(nv1)
            .saturating_mul(nv2)
            .saturating_mul(8)
    };
    let inter_a = compute_rpa_intermediates_spin(mol, obs, dfbs, op, scf, config, true)?;
    let inter_b = compute_rpa_intermediates_spin(mol, obs, dfbs, op, scf, config, false)?;

    // Taken AFTER the two builds, for the reason spelled out in `u_ri_mp2`:
    // a b_ov reservation held ACROSS `compute_rpa_intermediates_spin` is
    // outstanding when `ThreeIndexSource` asks for its AO tensor, and the pool
    // then refuses THAT allocation with this charge as the incumbent. The two
    // are not actually co-resident -- the source drops before b_ov is
    // returned -- so charging after is both correct and free of the false
    // overlap. SOFT, per `u_ri_mp2`'s note.
    let b_ov_charge = crate::rimp2::charge_mo_side_soft(
        "U-MP2 amplitudes b_ov (alpha + beta)",
        b_ov_bytes(no_a, nv_a).saturating_add(b_ov_bytes(no_b, nv_b)),
    );

    // Stage-seam RSS safety net: both spin intermediates resident, amplitudes
    // not yet built. Observational only, never an error.
    ferric_core::memory::warn_if_rss_over(
        "U-MP2 spin intermediates built",
        crate::rimp2::eri3_budget_bytes(config.memory_budget_bytes),
        1.1,
    );

    // `scf` is UHF-shaped here (a ROHF input was semi-canonicalized above).
    let eps_a_vec: Vec<f64> = scf.eps_a().to_vec();
    let eps_b_vec: Vec<f64> = scf.eps_b().to_vec();

    // HARD charge for the three amplitude tensors, taken here -- after the AO
    // 3-index tensor's own charge has been released by
    // `compute_rpa_intermediates_spin` returning, and before the first
    // amplitude byte is allocated. Both halves of that placement matter: any
    // earlier and it would refuse the AO tensor it does not overlap with; any
    // later and it would be observing an OOM rather than preventing one.
    let amplitudes_charge = crate::rimp2::charge_mo_side(
        "U-MP2 amplitudes t_aa + t_bb + t_ab",
        t_bytes(no_a, no_a, nv_a, nv_a)
            .saturating_add(t_bytes(no_b, no_b, nv_b, nv_b))
            .saturating_add(t_bytes(no_a, no_b, nv_a, nv_b)),
    )?;

    let (t_aa, e_aa) = build_same_spin_amplitudes(&inter_a, &eps_a_vec);
    let (t_bb, e_bb) = build_same_spin_amplitudes(&inter_b, &eps_b_vec);
    let (t_ab, e_ab) = build_opposite_spin_amplitudes(&inter_a, &inter_b, &eps_a_vec, &eps_b_vec);

    let e_total = e_aa + e_bb + e_ab;
    Ok(UMp2Amplitudes {
        inter_a,
        inter_b,
        eps_a: eps_a_vec,
        eps_b: eps_b_vec,
        t_aa,
        t_bb,
        t_ab,
        components: URiMp2Components {
            e_aa,
            e_bb,
            e_ab,
            e_total,
        },
        _charge: (b_ov_charge, amplitudes_charge),
    })
}

// ---------------------------------------------------------------------------
// Shared GEMM+rayon spin-pair kernel.
//
// Ports `rimp2::spin_components_from_b_ov`'s i-blocked wide-GEMM restructure
// to the unrestricted same-spin and opposite-spin pair terms. Each per-element
// `eri_pqrs = Σ_P B^P_pq B^P_rs` closure used to cost an O(naux) strided dot,
// called O(nocc²·nvir²) times. Instead we slice the dressed occ-vir B block
// per occupied index `i` out of the (naux, nocc*nvir) tensor once and form
//   G_i[a, j*nvir+b] = (ia|jb)   (or the two-B analogue for opposite-spin)
// via a single wide GEMM `B_i^T · B` per `i` — same FLOPs at BLAS3 throughput.
// The outer `i` loop is near-embarrassingly parallel (each i owns an
// independent G_i transient and a private partial), so it is fanned across
// rayon via `into_par_iter` and reduced with an ORDER-PRESERVING
// collect-then-serial-sum (never a rayon tree `reduce`, which would make the
// float accumulation order — and thus the last-bit energy — depend on
// RAYON_NUM_THREADS; see spin_components_from_b_ov's doc comment for the same
// idiom). Denominators are applied element-wise after the GEMM, exactly as in
// the scalar loops this replaces.
//
// Two calling shapes are needed:
//  - same-spin (αα or ββ): one B tensor, antisymmetrized kernel
//      K_iajb = (ia|jb) − (ib|ja),  E = ¼ Σ t·K,  t = K/D
//  - opposite-spin (αβ): two independent B tensors (different nocc/nvir per
//    side), no antisymmetrization, single eri²/D.

/// One spin channel's dressed occ-vir intermediates + orbital-space shape,
/// bundled so [`same_spin_pair_kernel`]/[`opposite_spin_pair_kernel`] take one
/// named argument per spin instead of 4-8 loose `usize`/slice parameters.
/// Mirrors `ferric-rpa::budget::PeakEstimateShape`'s named-fields-over-tuple
/// convention. `Copy` (all fields are references or plain `usize`), so callers
/// may pass by value freely.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct SpinChannel<'a> {
    /// Dressed occ-vir tensor, shape `(naux, nocc*nvir)`.
    pub b: &'a Array2<f64>,
    /// Full per-spin orbital-energy slice (denominators index
    /// `eps[first_occ+i]` / `eps[nocc_total+a]`).
    pub eps: &'a [f64],
    pub nocc: usize,
    pub nvir: usize,
    pub first_occ: usize,
    pub nocc_total: usize,
}
//
// Both optionally materialize the amplitude tensor `t[i,j,a,b]` (same layout
// as the pre-GEMM scalar loops) for the U-OO gradient path; when the caller
// only wants the energy (same_spin_pair_energy /
// opposite_spin_pair_energy), the write is skipped entirely.

/// The energy-only pair arithmetic of one occupied block `i` of the same-spin
/// kernel: `0.25 * sum_{j >= i} fac * sum_ab K^2 / D` with `K = g_ab - g_ba`,
/// from `g_i[a, (j-i)*nvir + b] = (ia|jb)`, `j >= i` (shape
/// `(nvir, (nocc-i)*nvir)`). Shared by the CPU loop and the device path, which
/// differ ONLY in how `g_i` was formed; the floating-point operation order is
/// the one the inline loop had (j ascending, a, b ascending).
#[inline]
pub(crate) fn same_spin_block_energy(g_i: &ArrayView2<f64>, i: usize, ch: SpinChannel) -> f64 {
    let SpinChannel {
        eps,
        nocc,
        nvir,
        first_occ,
        nocc_total,
        ..
    } = ch;
    let eps_i = eps[first_occ + i];
    let mut energy_i = 0.0;
    for j in i..nocc {
        let fac = if i == j { 1.0 } else { 2.0 };
        let jcol = (j - i) * nvir; // column offset within the tail
        let eps_j = eps[first_occ + j];
        let mut energy_ij = 0.0;
        for a in 0..nvir {
            let eps_a = eps[nocc_total + a];
            for b_idx in 0..nvir {
                let eps_b = eps[nocc_total + b_idx];
                let g_ab = g_i[(a, jcol + b_idx)]; // (ia|jb)
                let g_ba = g_i[(b_idx, jcol + a)]; // (ib|ja)
                let k = g_ab - g_ba;
                let denom = eps_i + eps_j - eps_a - eps_b;
                let t_val = k / denom;
                energy_ij += t_val * k;
            }
        }
        energy_i += fac * energy_ij;
    }
    0.25 * energy_i
}

/// The energy-only pair arithmetic of one alpha-occupied block `i` of the
/// opposite-spin kernel, from `g_i[a, J*nvir_b + B] = (ia|JB)` (shape
/// `(nvir_a, nocc_b*nvir_b)`). Shared by the CPU loop and the device path;
/// operation order is the inline loop's (a, J, B ascending).
#[inline]
pub(crate) fn opposite_spin_block_energy(
    g_i: &ArrayView2<f64>,
    i: usize,
    ch_a: SpinChannel,
    ch_b: SpinChannel,
) -> f64 {
    let eps_i = ch_a.eps[ch_a.first_occ + i];
    let mut energy_i = 0.0;
    for a in 0..ch_a.nvir {
        let eps_av = ch_a.eps[ch_a.nocc_total + a];
        for jj in 0..ch_b.nocc {
            let eps_j = ch_b.eps[ch_b.first_occ + jj];
            for bb_idx in 0..ch_b.nvir {
                let eps_bv = ch_b.eps[ch_b.nocc_total + bb_idx];
                let eri = g_i[(a, jj * ch_b.nvir + bb_idx)];
                let denom = eps_i + eps_j - eps_av - eps_bv;
                let t_val = eri / denom;
                energy_i += t_val * eri;
            }
        }
    }
    energy_i
}

/// The energy-only same-spin kernel: the device first (mixed only when
/// `mixed_ok`, see [`crate::rimp2_gpu::try_u_same_spin_on_device`]), else the
/// CPU wide-GEMM loop.
fn same_spin_energy_only(ch: SpinChannel, mixed_ok: bool) -> f64 {
    use rayon::prelude::*;
    let SpinChannel { b, nocc, nvir, .. } = ch;
    #[cfg(feature = "gpu")]
    if let Some(e) = crate::rimp2_gpu::try_u_same_spin_on_device(ch, mixed_ok) {
        return e;
    }
    crate::rimp2::note_mixed_ran_on_cpu(mixed_ok);
    // Energy-only: no t tensor to write, so the per-i partial is just a
    // scalar — no allocation/collection of a Vec<Option<..>> needed at all.
    //
    // PAIR SYMMETRY (energy-only path only). The summand
    //   K_iajb^2 / D_ijab,   K_iajb = (ia|jb) - (ib|ja)
    // is invariant under the JOINT swap (i,j)<->(j,i), (a,b)<->(b,a): K is
    // antisymmetric under it (K_jbia = (jb|ia) - (ja|ib) = -K_iajb, using
    // the real-orbital 2-electron symmetry (ia|jb) = (jb|ia)), so K^2 is
    // symmetric, and D_ijab = e_i + e_j - e_a - e_b is manifestly symmetric.
    // The (j,i) block therefore contributes exactly what the (i,j) block
    // does, so we visit only unique pairs j >= i and weight the strictly
    // off-diagonal ones by 2 — the same argument and the same fac=2/fac=1
    // convention as `rimp2::spin_components_from_b_ov`.
    //
    // That also lets `g_i` be formed over just the j >= i tail rather than
    // full width, so the discarded lower-triangle GEMM flops
    // ((nocc-1)/(2*nocc) of the total) are never computed. Both halve the
    // work; together the stage does ~half the flops it used to.
    //
    // The `want_amplitudes` branch of `same_spin_pair_kernel` CANNOT use this: it fills
    // t[i,j,a,b] for every (i,j) for the U-OO gradient, so it needs the
    // full j range and the full-width g_i.
    let partials: Vec<f64> = (0..nocc)
        .into_par_iter()
        .map(|i| {
            let b_i = b.slice(ndarray::s![.., i * nvir..(i + 1) * nvir]);
            // (nvir, (nocc-i)*nvir); g_i[a, (j-i)*nvir+b] = (ia|jb), j >= i
            let b_tail = b.slice(ndarray::s![.., i * nvir..]);
            let g_i = b_i.t().dot(&b_tail);
            same_spin_block_energy(&g_i.view(), i, ch)
        })
        .collect();
    partials.into_iter().sum()
}

/// Same-spin (αα or ββ) pair kernel: builds the energy and, optionally, the
/// antisymmetrized amplitude tensor `t[i,j,a,b] = [(ia|jb) − (ib|ja)] / D`.
///
/// `ch`: this spin channel's dressed occ-vir tensor + orbital-space shape
/// (see [`SpinChannel`]; `ch.eps` indexes as `eps[first_occ+i]` /
/// `eps[nocc_total+a]`, matching the original scalar loops' frozen-core
/// convention).
///
/// When `want_amplitudes`, `t` is allocated ONCE up front and each rayon
/// worker writes its disjoint `t[i, .., .., ..]` slab directly in place via
/// `axis_iter_mut(Axis(0)).into_par_iter()` — no intermediate per-`i` `Vec`
/// slab is collected first and copied in afterward, so the peak transient is
/// 1× the `t` tensor, not 2×. Only the per-`i` energies are collected and
/// summed SEQUENTIALLY in ascending `i` (never a rayon tree `reduce`, which
/// would make the float accumulation order — and thus the last-bit energy —
/// depend on `RAYON_NUM_THREADS`); the per-element arithmetic that fills each
/// slab is bit-for-bit identical to the previous two-pass version.
///
/// Returns `(energy, Some(t))` when `want_amplitudes`, else `(energy, None)`.
/// The energy-only call may run on the device, always in f64 (only `u_ri_mp2`
/// with a Coulomb, erfc or terfc operator may use the mixed kernel).
#[doc(hidden)]
pub fn same_spin_pair_kernel(ch: SpinChannel, want_amplitudes: bool) -> (f64, Option<Array4<f64>>) {
    use ndarray::Axis;
    use rayon::prelude::*;

    let SpinChannel {
        b,
        eps,
        nocc,
        nvir,
        first_occ,
        nocc_total,
    } = ch;

    if !want_amplitudes {
        return (same_spin_energy_only(ch, false), None);
    }

    // want_amplitudes: allocate t ONCE, write each i-slab in place (1x peak,
    // not 2x). Energies still collected in i-order and summed serially.
    let mut t = Array4::<f64>::zeros((nocc, nocc, nvir, nvir));
    let mut energies = vec![0.0f64; nocc];
    t.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(energies.par_iter_mut())
        .enumerate()
        .for_each(|(i, (mut t_i, energy_i_slot))| {
            let b_i = b.slice(ndarray::s![.., i * nvir..(i + 1) * nvir]);
            let g_i = b_i.t().dot(b); // (nvir, nocc*nvir); g_i[a, j*nvir+b] = (ia|jb)
            let eps_i = eps[first_occ + i];
            let mut energy_i = 0.0;
            for j in 0..nocc {
                let eps_j = eps[first_occ + j];
                for a in 0..nvir {
                    let eps_a = eps[nocc_total + a];
                    for b_idx in 0..nvir {
                        let eps_b = eps[nocc_total + b_idx];
                        let g_ab = g_i[(a, j * nvir + b_idx)]; // (ia|jb)
                        let g_ba = g_i[(b_idx, j * nvir + a)]; // (ib|ja)
                        let k = g_ab - g_ba;
                        let denom = eps_i + eps_j - eps_a - eps_b;
                        let t_val = k / denom;
                        energy_i += t_val * k;
                        t_i[(j, a, b_idx)] = t_val;
                    }
                }
            }
            *energy_i_slot = 0.25 * energy_i;
        });

    // Ascending-i serial sum — same accumulation order as the original
    // collect-then-serial-sum (energies is already i-ordered by construction).
    let energy: f64 = energies.into_iter().sum();
    (energy, Some(t))
}

/// The energy-only opposite-spin kernel: the device first (mixed only when
/// `mixed_ok`, see [`crate::rimp2_gpu::try_u_opposite_spin_on_device`]), else
/// the CPU wide-GEMM loop.
fn opposite_spin_energy_only(ch_a: SpinChannel, ch_b: SpinChannel, mixed_ok: bool) -> f64 {
    use rayon::prelude::*;
    #[cfg(feature = "gpu")]
    if let Some(e) = crate::rimp2_gpu::try_u_opposite_spin_on_device(ch_a, ch_b, mixed_ok) {
        return e;
    }
    crate::rimp2::note_mixed_ran_on_cpu(mixed_ok);
    let (b_a, b_b, nvir_a) = (ch_a.b, ch_b.b, ch_a.nvir);
    let partials: Vec<f64> = (0..ch_a.nocc)
        .into_par_iter()
        .map(|i| {
            let bi = b_a.slice(ndarray::s![.., i * nvir_a..(i + 1) * nvir_a]);
            let g_i = bi.t().dot(b_b); // (nvir_a, nocc_b*nvir_b); g_i[a, J*nvir_b+B] = (ia|JB)
            opposite_spin_block_energy(&g_i.view(), i, ch_a, ch_b)
        })
        .collect();
    partials.into_iter().sum()
}

/// Opposite-spin (αβ) pair kernel: builds the energy and, optionally, the
/// (non-antisymmetrized) amplitude tensor `t[i,J,a,B] = (ia|JB) / D`.
///
/// `ch_a`/`ch_b`: each spin's dressed occ-vir tensor + orbital-space shape
/// (see [`SpinChannel`]), independent `(nocc, nvir)` per side. No
/// antisymmetrization — single `eri²/D` per caller.
///
/// Same write-in-place amplitude discipline as [`same_spin_pair_kernel`]: `t`
/// is allocated ONCE and each rayon worker writes its disjoint `t[i, .., ..,
/// ..]` slab directly (peak transient 1× the `t` tensor, not 2×); per-`i`
/// energies are collected in ascending `i` and summed serially. The
/// energy-only call may run on the device, always in f64.
#[doc(hidden)]
pub fn opposite_spin_pair_kernel(
    ch_a: SpinChannel,
    ch_b: SpinChannel,
    want_amplitudes: bool,
) -> (f64, Option<Array4<f64>>) {
    use ndarray::Axis;
    use rayon::prelude::*;

    let SpinChannel {
        b: b_a,
        eps: eps_a,
        nocc: nocc_a,
        nvir: nvir_a,
        first_occ: first_occ_a,
        nocc_total: nocc_total_a,
    } = ch_a;
    let SpinChannel {
        b: b_b,
        eps: eps_b,
        nocc: nocc_b,
        nvir: nvir_b,
        first_occ: first_occ_b,
        nocc_total: nocc_total_b,
    } = ch_b;

    if !want_amplitudes {
        return (opposite_spin_energy_only(ch_a, ch_b, false), None);
    }

    let mut t = Array4::<f64>::zeros((nocc_a, nocc_b, nvir_a, nvir_b));
    let mut energies = vec![0.0f64; nocc_a];
    t.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(energies.par_iter_mut())
        .enumerate()
        .for_each(|(i, (mut t_i, energy_i_slot))| {
            let bi = b_a.slice(ndarray::s![.., i * nvir_a..(i + 1) * nvir_a]);
            let g_i = bi.t().dot(b_b); // (nvir_a, nocc_b*nvir_b); g_i[a, J*nvir_b+B] = (ia|JB)
            let eps_i = eps_a[first_occ_a + i];
            let mut energy_i = 0.0;
            for a in 0..nvir_a {
                let eps_av = eps_a[nocc_total_a + a];
                for jj in 0..nocc_b {
                    let eps_j = eps_b[first_occ_b + jj];
                    for bb_idx in 0..nvir_b {
                        let eps_bv = eps_b[nocc_total_b + bb_idx];
                        let eri = g_i[(a, jj * nvir_b + bb_idx)];
                        let denom = eps_i + eps_j - eps_av - eps_bv;
                        let t_val = eri / denom;
                        energy_i += t_val * eri;
                        t_i[(jj, a, bb_idx)] = t_val;
                    }
                }
            }
            *energy_i_slot = energy_i;
        });

    let energy: f64 = energies.into_iter().sum();
    (energy, Some(t))
}

/// Build the same-spin amplitude tensor and accumulate its energy:
///   t[i,j,a,b] = K_iajb / D,   K_iajb = (ia|jb) - (ib|ja)
///   E = ¼ Σ t · K
fn build_same_spin_amplitudes(inter: &RpaIntermediates, eps: &[f64]) -> (Array4<f64>, f64) {
    let ch = SpinChannel {
        b: &inter.b_ov,
        eps,
        nocc: inter.nocc,
        nvir: inter.nvir,
        first_occ: inter.first_occ,
        nocc_total: inter.nocc_total,
    };
    let (energy, t) = same_spin_pair_kernel(ch, true);
    (t.unwrap(), energy)
}

/// Build the opposite-spin amplitude tensor and its energy:
///   t[i,J,a,B] = (ia|JB) / D
///   E = Σ t · (ia|JB)
fn build_opposite_spin_amplitudes(
    inter_a: &RpaIntermediates,
    inter_b: &RpaIntermediates,
    eps_a: &[f64],
    eps_b: &[f64],
) -> (Array4<f64>, f64) {
    let ch_a = SpinChannel {
        b: &inter_a.b_ov,
        eps: eps_a,
        nocc: inter_a.nocc,
        nvir: inter_a.nvir,
        first_occ: inter_a.first_occ,
        nocc_total: inter_a.nocc_total,
    };
    let ch_b = SpinChannel {
        b: &inter_b.b_ov,
        eps: eps_b,
        nocc: inter_b.nocc,
        nvir: inter_b.nvir,
        first_occ: inter_b.first_occ,
        nocc_total: inter_b.nocc_total,
    };
    let (energy, t) = opposite_spin_pair_kernel(ch_a, ch_b, true);
    (t.unwrap(), energy)
}

/// Unrelaxed MP2 1-particle density-matrix corrections in MO basis,
/// for both α and β spin channels. Built directly from amplitudes.
///
/// Per-spin formulas (Bozkaya 2013, antisymmetric t^σσ, mixed t^αβ):
///
/// ```text
/// P^α_{ij} = -½ Σ_{k,a,b} t^αα_{ik,ab} t^αα_{jk,ab}
///            - Σ_{K,a,B}    t^αβ_{iK,aB} t^αβ_{jK,aB}
/// P^α_{ab} = +½ Σ_{i,j,c} t^αα_{ij,ac} t^αα_{ij,bc}
///            + Σ_{i,J,B}    t^αβ_{iJ,aB} t^αβ_{iJ,bB}
/// ```
///
/// β formulas obtained by α↔β symmetry: for the αβ piece, the β-occupied
/// index sits on axis 1 of `t_ab` and the β-virtual sits on axis 3.
///
/// Density conservation per spin: tr(P^σ_oo) + tr(P^σ_vv) = 0.
#[derive(Debug, Clone)]
pub struct UMp2Density {
    /// α occupied-occupied correction, `(nocc_α, nocc_α)`.
    pub p_oo_a: Array2<f64>,
    /// α virtual-virtual correction, `(nvir_α, nvir_α)`.
    pub p_vv_a: Array2<f64>,
    /// β occupied-occupied correction, `(nocc_β, nocc_β)`.
    pub p_oo_b: Array2<f64>,
    /// β virtual-virtual correction, `(nvir_β, nvir_β)`.
    pub p_vv_b: Array2<f64>,
}

/// Contraction-axis slab width, in elements, of one task of the `P_oo` Gram
/// products in [`build_u_mp2_density`]. A constant, so the slab boundaries and
/// the summation order do not depend on the thread count or the budget.
const U_DENSITY_K_SLAB: usize = 1 << 15;

/// Build the unrelaxed MP2 one-particle density matrices from UHF amplitudes.
///
/// Every block is a sum of Gram products of amplitude slices (`o`/`v` the
/// active occupied/virtual counts of each spin):
///
/// ```text
///   P^α_oo = −½ X_αα X_ααᵀ − X_αβ X_αβᵀ     X_αα = t_αα as (o_α, o_α·v_α²), X_αβ = t_αβ as (o_α, o_β·v_α·v_β)
///   P^β_oo = −½ X_ββ X_ββᵀ − Σ_K Y_K Y_Kᵀ   Y_K = t_αβ[K] as (o_β, v_α·v_β)
///   P^α_vv = ½ Σ_ij A_ij A_ijᵀ + Σ_iJ C_iJ C_iJᵀ   A_ij = t_αα[i,j], C_iJ = t_αβ[i,J]
///   P^β_vv = ½ Σ_ij B_ij B_ijᵀ + Σ_I Z_Iᵀ Z_I      B_ij = t_ββ[i,j], Z_I = t_αβ[I] as (o_β·v_α, v_β)
/// ```
///
/// The `P_oo` products run in contraction slabs of `U_DENSITY_K_SLAB`
/// columns, one pool task per slab; the `P_vv` products one task per leading
/// occupied index. Each task is single-threaded BLAS on its own accumulator,
/// and the accumulators are added in task order, so the densities do not
/// depend on the thread count. The amplitudes need no symmetry.
pub fn build_u_mp2_density(amps: &UMp2Amplitudes) -> UMp2Density {
    u_mp2_density_from_amplitudes(
        amps.t_aa.view(),
        amps.t_bb.view(),
        amps.t_ab.view(),
        U_DENSITY_K_SLAB,
    )
}

/// [`build_u_mp2_density`] on bare amplitude tensors, with the `P_oo` slab
/// width as an argument so a test can force several slabs on a small system.
fn u_mp2_density_from_amplitudes(
    t_aa: ndarray::ArrayView4<'_, f64>,
    t_bb: ndarray::ArrayView4<'_, f64>,
    t_ab: ndarray::ArrayView4<'_, f64>,
    k_slab: usize,
) -> UMp2Density {
    use ndarray::s;
    let (t_aa, t_bb, t_ab) = (
        t_aa.as_standard_layout(),
        t_bb.as_standard_layout(),
        t_ab.as_standard_layout(),
    );
    let (oa, _, va, _) = t_aa.dim();
    let (ob, _, vb, _) = t_bb.dim();
    assert_eq!(
        t_ab.dim(),
        (oa, ob, va, vb),
        "build_u_mp2_density: t_ab shape"
    );
    let y = |k: usize| {
        t_ab.slice(s![k, .., .., ..])
            .into_shape_with_order((ob, va * vb))
            .expect("standard-layout t_ab")
    };

    let x_aa = t_aa.view().into_shape_with_order((oa, oa * va * va));
    let x_ab = t_ab.view().into_shape_with_order((oa, ob * va * vb));
    let x_bb = t_bb.view().into_shape_with_order((ob, ob * vb * vb));
    let (x_aa, x_ab, x_bb) = (
        x_aa.expect("standard-layout t_aa"),
        x_ab.expect("standard-layout t_ab"),
        x_bb.expect("standard-layout t_bb"),
    );
    let oo_a_terms = [(x_aa, -0.5), (x_ab, -1.0)];
    let p_oo_a = gram_sum_slabbed("U-MP2 density P_oo^α partials", &oo_a_terms, oa, k_slab);

    let mut oo_b_terms = vec![(x_bb, -0.5)];
    oo_b_terms.extend((0..oa).map(|k| (y(k), -1.0)));
    let p_oo_b = gram_sum_slabbed("U-MP2 density P_oo^β partials", &oo_b_terms, ob, k_slab);

    use crate::orbital_gradient_gemm::ordered_task_sum;
    use ndarray::linalg::general_mat_mul;
    let p_vv_a = ordered_task_sum("U-MP2 density P_vv^α partials", oa, (va, va), |i, acc| {
        for j in 0..oa {
            let a = t_aa.slice(s![i, j, .., ..]);
            general_mat_mul(0.5, &a, &a.t(), 1.0, acc);
        }
        for jj in 0..ob {
            let c = t_ab.slice(s![i, jj, .., ..]);
            general_mat_mul(1.0, &c, &c.t(), 1.0, acc);
        }
    });
    let p_vv_b = ordered_task_sum(
        "U-MP2 density P_vv^β partials",
        ob + oa,
        (vb, vb),
        |n, acc| {
            if n < ob {
                for j in 0..ob {
                    let b = t_bb.slice(s![n, j, .., ..]);
                    general_mat_mul(0.5, &b, &b.t(), 1.0, acc);
                }
            } else {
                let z = t_ab
                    .slice(s![n - ob, .., .., ..])
                    .into_shape_with_order((ob * va, vb))
                    .expect("standard-layout t_ab");
                general_mat_mul(1.0, &z.t(), &z, 1.0, acc);
            }
        },
    );

    UMp2Density {
        p_oo_a,
        p_vv_a,
        p_oo_b,
        p_vv_b,
    }
}

/// `Σ_terms alpha · X Xᵀ` (`rows × rows`), each `X` cut into contraction
/// slabs of `k_slab` columns; one pool task per slab, partials added in task
/// order.
fn gram_sum_slabbed(
    label: &str,
    terms: &[(ArrayView2<'_, f64>, f64)],
    rows: usize,
    k_slab: usize,
) -> Array2<f64> {
    use ndarray::s;
    let k_slab = k_slab.max(1);
    let tasks: Vec<(usize, usize, usize)> = terms
        .iter()
        .enumerate()
        .flat_map(|(t, (x, _))| {
            let k = x.ncols();
            (0..k)
                .step_by(k_slab)
                .map(move |k0| (t, k0, (k0 + k_slab).min(k)))
        })
        .collect();
    crate::orbital_gradient_gemm::ordered_task_sum(label, tasks.len(), (rows, rows), |n, acc| {
        let (t, k0, k1) = tasks[n];
        let (x, alpha) = &terms[t];
        let xs = x.slice(s![.., k0..k1]);
        ndarray::linalg::general_mat_mul(*alpha, &xs, &xs.t(), 1.0, acc);
    })
}

/// Per-block decomposition of the unrelaxed integral-response gradient.
///
/// `same_a`/`same_b` are the σσ-block contributions to `g^σ` (αα for α,
/// ββ for β). `ab_a`/`ab_b` are the αβ-block contributions to each spin.
/// The full block-wise gradient is `same_σ + ab_σ`.
#[derive(Debug, Clone)]
pub struct UMp2GradientBlocks {
    /// αα-block contribution to `g^α`, shape `(nvir_α, nocc_α)`.
    pub same_a: Array2<f64>,
    /// ββ-block contribution to `g^β`, shape `(nvir_β, nocc_β)`.
    pub same_b: Array2<f64>,
    /// αβ-block contribution to `g^α`, shape `(nvir_α, nocc_α)`.
    pub ab_a: Array2<f64>,
    /// αβ-block contribution to `g^β`, shape `(nvir_β, nocc_β)`.
    pub ab_b: Array2<f64>,
}

/// Compute the U-MP2 orbital gradient `g^σ_{ai} = ∂E_MP2/∂κ^σ_{ai}` at fixed
/// orbital energies (integral-response only). FD-validated on OH/cc-pVDZ to
/// ~1e-10 vs in-Rust per-block finite differences with the Cayley convention
/// `U = (I-κ/2)^{-1}(I+κ/2)`.
///
/// At a converged HF reference the HF Brillouin term `-2·F^σ_{ai}` is zero;
/// callers wanting the full gradient should add it.
///
/// Returns `(g_a, g_b)` with shapes `(nvir_α, nocc_α)` and `(nvir_β, nocc_β)`.
///
/// Energy:
/// ```text
/// E_MP2 = ¼ Σ t^αα_{ij,ab} K^αα_{iajb}
///       + ¼ Σ t^ββ_{ij,ab} K^ββ_{iajb}
///       +   Σ t^αβ_{iJ,aB} (ia|JB)
/// ```
/// with `K^σσ_{iajb} = (ia|jb)_σ − (ib|ja)_σ`.
///
/// At fixed denominators, the σσ blocks pick up a `½` chain-rule factor
/// (the `t` and `K` derivatives are equal) while the αβ block picks up
/// a `2×` factor (both t and (ia|JB) depend on the same integral).
///
/// # Construction
///
/// Per spin σ, with `A_kj = t^σσ[k, j, :, :]` and the RI factors
/// `(ca|jb) = Σ_P B^P_{ca} B^P_{jb}`, `(ik|jb) = ooov[(i,k),(j,b)]`:
///
/// ```text
///   same_σ[c,k] = Σ_{P,x} B^P_{cx} Γ_k[P,x] − ½ Σ_{i,j,y} N_ij[c,y] (ik|jy)
///   Γ_k         = ½ Σ_j B_j (A_kj − A_kjᵀ + A_jkᵀ − A_jk)ᵀ,   N_ij = A_ij − A_ji + A_jiᵀ − A_ijᵀ
/// ```
///
/// and the αβ blocks the same way with `t^αβ` slices (`Γ^α_k = Σ_J B^β_J
/// t^αβ[k,J]ᵀ`, `Γ^β_k = Σ_i B^α_i t^αβ[i,k]`). No VVOV block is formed; the
/// GEMM building blocks and their determinism contract are in
/// `orbital_gradient_gemm`. The scalar definition this reproduces is the
/// `u_mp2_orbital_gradient_scalar_reference` test oracle.
pub fn compute_u_mp2_orbital_gradient(
    amps: &UMp2Amplitudes,
    b_full_a: &Array3<f64>,
    b_full_b: &Array3<f64>,
) -> (Array2<f64>, Array2<f64>) {
    compute_u_mp2_orbital_gradient_timed(amps, b_full_a, b_full_b, &mut PhaseAccumulator::new())
}

/// The per-block parts of [`compute_u_mp2_orbital_gradient`].
pub fn compute_u_mp2_orbital_gradient_blocks(
    amps: &UMp2Amplitudes,
    b_full_a: &Array3<f64>,
    b_full_b: &Array3<f64>,
) -> UMp2GradientBlocks {
    compute_u_mp2_orbital_gradient_blocks_timed(
        amps,
        b_full_a,
        b_full_b,
        &mut PhaseAccumulator::new(),
    )
}

/// [`compute_u_mp2_orbital_gradient`] with its sub-phases recorded in `acc` as
/// nested rows (inert unless `FERRIC_TIMING` is set; the arithmetic is the same
/// either way).
pub(crate) fn compute_u_mp2_orbital_gradient_timed(
    amps: &UMp2Amplitudes,
    b_full_a: &Array3<f64>,
    b_full_b: &Array3<f64>,
    acc: &mut PhaseAccumulator,
) -> (Array2<f64>, Array2<f64>) {
    let bl = compute_u_mp2_orbital_gradient_blocks_timed(amps, b_full_a, b_full_b, acc);
    (&bl.same_a + &bl.ab_a, &bl.same_b + &bl.ab_b)
}

fn compute_u_mp2_orbital_gradient_blocks_timed(
    amps: &UMp2Amplitudes,
    b_full_a: &Array3<f64>,
    b_full_b: &Array3<f64>,
    acc: &mut PhaseAccumulator,
) -> UMp2GradientBlocks {
    use crate::orbital_gradient_gemm::{GradientB, VV_CONTRACT_AUX_SLAB};
    let (nocc_a, _, nvir_a, _) = amps.t_aa.dim();
    let (nocc_b, _, nvir_b, _) = amps.t_bb.dim();
    debug_assert_eq!(b_full_a.shape()[0], b_full_b.shape()[0]);
    let tok = acc.begin("  U-MP2 grad: B block extraction");
    let (ia, ib) = (&amps.inter_a, &amps.inter_b);
    let ba = GradientB::extract(b_full_a, nocc_a, nvir_a, ia.first_occ, ia.nocc_total);
    let bb = GradientB::extract(b_full_b, nocc_b, nvir_b, ib.first_occ, ib.nocc_total);
    acc.stop_nested(tok);
    u_mp2_orbital_gradient_kernel(
        amps.t_aa.view(),
        amps.t_bb.view(),
        amps.t_ab.view(),
        &ba,
        &bb,
        VV_CONTRACT_AUX_SLAB,
        acc,
    )
}

/// Element count of the dense intermediates [`compute_u_mp2_orbital_gradient`]
/// holds at once: both spins' B blocks, the four OOOV-type integral blocks and
/// the two `Γ` row stacks (`2·nocc_σ × naux·nvir_σ` each).
pub(crate) fn u_gradient_intermediate_elems(
    naux: usize,
    nocc_a: usize,
    nvir_a: usize,
    nocc_b: usize,
    nvir_b: usize,
) -> usize {
    use crate::orbital_gradient_gemm::GradientB;
    let (ova, ovb) = (nocc_a * nvir_a, nocc_b * nvir_b);
    let ooov = (nocc_a * nocc_a) * (ova + ovb) + nocc_b * nocc_b * (ovb + ova);
    let gamma = 2 * naux * (ova + ovb);
    GradientB::elems(naux, nocc_a, nvir_a)
        .saturating_add(GradientB::elems(naux, nocc_b, nvir_b))
        .saturating_add(ooov)
        .saturating_add(gamma)
}

/// The U-MP2 orbital-gradient blocks from bare amplitudes and B blocks (see
/// [`compute_u_mp2_orbital_gradient`] for the construction). `aux_slab` is the
/// `P` slab of the `(ca|jb)` contraction, an argument so a test can force
/// several slabs.
#[allow(clippy::too_many_arguments)]
fn u_mp2_orbital_gradient_kernel(
    t_aa: ndarray::ArrayView4<'_, f64>,
    t_bb: ndarray::ArrayView4<'_, f64>,
    t_ab: ndarray::ArrayView4<'_, f64>,
    ba: &crate::orbital_gradient_gemm::GradientB,
    bb: &crate::orbital_gradient_gemm::GradientB,
    aux_slab: usize,
    acc: &mut PhaseAccumulator,
) -> UMp2GradientBlocks {
    use crate::orbital_gradient_gemm::{
        charge_pair_scratch, contract_vv_gamma, cross_gamma_rows, ordered_task_sum,
        pair_gamma_rows, pair_ooov_term, par_gemm_tn,
    };
    use ndarray::linalg::general_mat_mul;
    use ndarray::s;
    let (oa, va, ob, vb, naux) = (ba.nocc, ba.nvir, bb.nocc, bb.nvir, ba.naux);
    assert_eq!(t_aa.dim(), (oa, oa, va, va), "U-MP2 gradient: t_aa shape");
    assert_eq!(t_bb.dim(), (ob, ob, vb, vb), "U-MP2 gradient: t_bb shape");
    assert_eq!(t_ab.dim(), (oa, ob, va, vb), "U-MP2 gradient: t_ab shape");
    let blk_aa = |i: usize, j: usize| t_aa.slice_move(s![i, j, .., ..]);
    let blk_bb = |i: usize, j: usize| t_bb.slice_move(s![i, j, .., ..]);
    let ab = |i: usize, jj: usize| t_ab.slice_move(s![i, jj, .., ..]);
    let _scratch = charge_pair_scratch("U-MP2 gradient pair scratch", va.max(vb));

    // (ik|jb)_α, (ik|jb)_β, (ik_α|JB_β), (ia_α|JK_β).
    let tok = acc.begin("  U-MP2 grad: ooov GEMMs");
    let ooov_a = par_gemm_tn(&ba.boo, &ba.bov);
    let ooov_b = par_gemm_tn(&bb.boo, &bb.bov);
    let ooov_ab = par_gemm_tn(&ba.boo, &bb.bov);
    let ovoo_ab = par_gemm_tn(&ba.bov, &bb.boo);
    acc.stop_nested(tok);

    // Γ rows: same-spin `k` first, then the αβ `k`.
    let tok = acc.begin("  U-MP2 grad: Γ pair GEMMs");
    let mut gamma_a = Array2::<f64>::zeros((2 * oa, naux * va));
    pair_gamma_rows(&blk_aa, 1.0, 1.0, 0.5, ba, gamma_a.slice_mut(s![..oa, ..]));
    cross_gamma_rows(&ab, va, bb, gamma_a.slice_mut(s![oa.., ..]));
    let mut gamma_b = Array2::<f64>::zeros((2 * ob, naux * vb));
    pair_gamma_rows(&blk_bb, 1.0, 1.0, 0.5, bb, gamma_b.slice_mut(s![..ob, ..]));
    let ab_t = |k: usize, i: usize| ab(i, k).reversed_axes();
    cross_gamma_rows(&ab_t, vb, ba, gamma_b.slice_mut(s![ob.., ..]));
    acc.stop_nested(tok);

    let tok = acc.begin("  U-MP2 grad: (ca|jb) contraction");
    let mut g_a = contract_vv_gamma(
        "U-MP2 gradient α partials",
        &ba.bvv,
        &gamma_a,
        naux,
        va,
        aux_slab,
    );
    let mut g_b = contract_vv_gamma(
        "U-MP2 gradient β partials",
        &bb.bvv,
        &gamma_b,
        naux,
        vb,
        aux_slab,
    );
    acc.stop_nested(tok);

    let tok = acc.begin("  U-MP2 grad: (ik|jb) terms");
    g_a += &ordered_task_sum(
        "U-MP2 gradient α oo partials",
        oa,
        (va, 2 * oa),
        |i, acc| {
            let (mut same, mut cross) = acc.multi_slice_mut((s![.., ..oa], s![.., oa..]));
            pair_ooov_term(&blk_aa, 1.0, 1.0, -0.5, &ooov_a, i, oa, va, &mut same);
            for jj in 0..ob {
                let o = ooov_ab.slice(s![i * oa..(i + 1) * oa, jj * vb..(jj + 1) * vb]);
                general_mat_mul(-1.0, &ab(i, jj), &o.t(), 1.0, &mut cross);
            }
        },
    );
    g_b += &ordered_task_sum(
        "U-MP2 gradient β oo partials",
        ob + oa,
        (vb, 2 * ob),
        |n, acc| {
            let (mut same, mut cross) = acc.multi_slice_mut((s![.., ..ob], s![.., ob..]));
            if n < ob {
                pair_ooov_term(&blk_bb, 1.0, 1.0, -0.5, &ooov_b, n, ob, vb, &mut same);
            } else {
                let i = n - ob;
                for jj in 0..ob {
                    let o = ovoo_ab.slice(s![i * va..(i + 1) * va, jj * ob..(jj + 1) * ob]);
                    general_mat_mul(-1.0, &ab(i, jj).t(), &o, 1.0, &mut cross);
                }
            }
        },
    );
    acc.stop_nested(tok);

    // The αβ blocks carry the factor 2 of the chain rule (both t and (ia|JB)
    // depend on the integrals).
    UMp2GradientBlocks {
        same_a: g_a.slice(s![.., ..oa]).to_owned(),
        ab_a: 2.0 * &g_a.slice(s![.., oa..]),
        same_b: g_b.slice(s![.., ..ob]).to_owned(),
        ab_b: 2.0 * &g_b.slice(s![.., ob..]),
    }
}

/// Same-spin contribution:
///   ¼ Σ_{ij,ab} [(ia|jb) - (ib|ja)]² / (ε_i+ε_j-ε_a-ε_b)
fn same_spin_pair_energy(inter: &RpaIntermediates, eps: &[f64], mixed_ok: bool) -> f64 {
    let ch = SpinChannel {
        b: &inter.b_ov,
        eps,
        nocc: inter.nocc,
        nvir: inter.nvir,
        first_occ: inter.first_occ,
        nocc_total: inter.nocc_total,
    };
    same_spin_energy_only(ch, mixed_ok)
}

/// Opposite-spin contribution:
///   Σ_{iJ,aB} (ia|JB)² / (ε_iα + ε_Jβ - ε_aα - ε_Bβ)
fn opposite_spin_pair_energy(
    inter_a: &RpaIntermediates,
    inter_b: &RpaIntermediates,
    eps_a: &[f64],
    eps_b: &[f64],
    mixed_ok: bool,
) -> f64 {
    assert_eq!(inter_a.naux, inter_b.naux);
    let ch_a = SpinChannel {
        b: &inter_a.b_ov,
        eps: eps_a,
        nocc: inter_a.nocc,
        nvir: inter_a.nvir,
        first_occ: inter_a.first_occ,
        nocc_total: inter_a.nocc_total,
    };
    let ch_b = SpinChannel {
        b: &inter_b.b_ov,
        eps: eps_b,
        nocc: inter_b.nocc,
        nvir: inter_b.nvir,
        first_occ: inter_b.first_occ,
        nocc_total: inter_b.nocc_total,
    };
    opposite_spin_energy_only(ch_a, ch_b, mixed_ok)
}

/// Compute the U-MP2 integral-response energy from given MO coefficients and
/// **fixed** orbital energies. Used by in-Rust FD gradient validation.
///
/// Unlike `u_ri_mp2`, this function does NOT re-diagonalize the Fock matrix —
/// it uses the supplied `eps_a`/`eps_b` for all denominators regardless of what
/// the Fock matrix looks like at the perturbed MOs.  This isolates the integral
/// piece of the orbital-rotation derivative, matching the Python
/// `mp2_energy_fixed_eps` ground-truth FD.
///
/// `which` selects the block: "aa", "bb", "ab", or "all".
#[cfg(test)]
#[allow(clippy::too_many_arguments)] // FD ground-truth helper: passes both spin MO sets + eps explicitly
pub(crate) fn u_mp2_energy_fixed_eps(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    eps_a: &[f64],
    eps_b: &[f64],
    nocc_a: usize,
    nocc_b: usize,
    which: &str,
) -> Result<f64, FerricError> {
    use crate::rimp2::cholesky_inverse_sqrt;
    use ferric_integrals::threeindex;

    let nmo_a = c_a.ncols();
    let nmo_b = c_b.ncols();
    let nvir_a = nmo_a - nocc_a;
    let nvir_b = nmo_b - nocc_b;
    let naux = dfbs.nbasis();

    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = threeindex::eri3_tensor(op, obs, dfbs)?;

    let c_a_occ = c_a.slice(ndarray::s![.., ..nocc_a]).to_owned();
    let c_a_vir = c_a.slice(ndarray::s![.., nocc_a..]).to_owned();
    let c_b_occ = c_b.slice(ndarray::s![.., ..nocc_b]).to_owned();
    let c_b_vir = c_b.slice(ndarray::s![.., nocc_b..]).to_owned();

    let b_a_ov_raw = crate::mo_transform::transform_3center_ov(&eri3_ao, &c_a_occ, &c_a_vir);
    let b_a_ov = v_inv_sqrt.dot(
        &b_a_ov_raw
            .into_shape_with_order((naux, nocc_a * nvir_a))
            .unwrap(),
    );
    let b_b_ov_raw = crate::mo_transform::transform_3center_ov(&eri3_ao, &c_b_occ, &c_b_vir);
    let b_b_ov = v_inv_sqrt.dot(
        &b_b_ov_raw
            .into_shape_with_order((naux, nocc_b * nvir_b))
            .unwrap(),
    );

    // This FD helper has no frozen core: first_occ=0, nocc_total=nocc for
    // each spin, so eps[first_occ+i] == eps[i] and eps[nocc_total+a] ==
    // eps[nocc+a], matching the `eo_*`/`ev_*` slices the old scalar loops
    // read from directly.
    let mut e_total = 0.0;

    if which == "aa" || which == "all" {
        let ch = SpinChannel {
            b: &b_a_ov,
            eps: eps_a,
            nocc: nocc_a,
            nvir: nvir_a,
            first_occ: 0,
            nocc_total: nocc_a,
        };
        let (e_aa, _) = same_spin_pair_kernel(ch, false);
        e_total += e_aa;
    }

    if which == "bb" || which == "all" {
        let ch = SpinChannel {
            b: &b_b_ov,
            eps: eps_b,
            nocc: nocc_b,
            nvir: nvir_b,
            first_occ: 0,
            nocc_total: nocc_b,
        };
        let (e_bb, _) = same_spin_pair_kernel(ch, false);
        e_total += e_bb;
    }

    if which == "ab" || which == "all" {
        let ch_a = SpinChannel {
            b: &b_a_ov,
            eps: eps_a,
            nocc: nocc_a,
            nvir: nvir_a,
            first_occ: 0,
            nocc_total: nocc_a,
        };
        let ch_b = SpinChannel {
            b: &b_b_ov,
            eps: eps_b,
            nocc: nocc_b,
            nvir: nvir_b,
            first_occ: 0,
            nocc_total: nocc_b,
        };
        let (e_ab, _) = opposite_spin_pair_kernel(ch_a, ch_b, false);
        e_total += e_ab;
    }

    Ok(e_total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::RhfConfig;
    use ferric_scf::screening::SchwarzBounds;
    use ferric_scf::uhf::{solve_uhf, UhfConfig};
    use ndarray_linalg::Solve;

    /// Compare U-RI-MP2 on a closed-shell system (H2 in cc-pVDZ) against
    /// the closed-shell `ri_mp2` driver. Should agree to numerical noise.
    #[test]
    fn u_rimp2_matches_closed_shell_on_h2() {
        let ctx = ParallelContext::default();
        let xyz = "2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n";
        // Closed-shell run
        let mol_cs = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol_cs, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol_cs, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf =
            ferric_scf::rhf::solve_rhf(&ctx, &mol_cs, &obs, op, &bounds, &RhfConfig::default())
                .unwrap();
        let cs =
            crate::rimp2::ri_mp2(&mol_cs, &obs, &dfbs, op, &rhf, &RiMp2Config::default()).unwrap();

        // Open-shell run on same molecule (singlet, M=1)
        let mol_us = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let uhf_cfg = UhfConfig {
            max_iter: 200,
            energy_conv: 1e-10,
            density_conv: 1e-8,
            ..Default::default()
        };
        // UHF will converge to a symmetric solution for singlet H2 if seeded
        // from neutral RHF MOs (no spin contamination).
        let c_seed = rhf.mos_r().clone();
        let uhf = ferric_scf::uhf::solve_uhf_with_guess(
            &ctx,
            &mol_us,
            &obs,
            &bounds,
            &uhf_cfg,
            Some((&c_seed, &c_seed)),
        )
        .unwrap();

        let us = u_ri_mp2(&mol_us, &obs, &dfbs, op, &uhf, &RiMp2Config::default()).unwrap();
        let diff = (us.mp2_corr - cs.mp2_corr).abs();
        println!(
            "CS E_MP2 = {:.10}, US E_MP2 = {:.10}, diff = {:.3e}",
            cs.mp2_corr, us.mp2_corr, diff
        );
        println!(
            "  components: αα={:.6e} ββ={:.6e} αβ={:.6e}",
            us.components.e_aa, us.components.e_bb, us.components.e_ab
        );
        assert!(
            diff < 1e-7,
            "closed-shell U-RI-MP2 disagrees with RI-MP2: diff={}",
            diff
        );
    }

    /// Validate U-RI-MP2 on OH/cc-pVDZ against the PySCF FD reference
    /// (testdata/reference/oh_cc-pvdz_u-oomp2-fd.json).
    /// PySCF reference: E_corr = -0.151003 (no frozen core, no RI).
    /// Ferric uses cc-pvdz-ri auxiliary -> RI noise ~1e-4 Ha.
    #[test]
    fn u_rimp2_oh_cc_pvdz_matches_pyscf() {
        let ctx = ParallelContext::default();
        // Geometry must match the Python harness: O at origin, H at (0,0,0.97 Å).
        let xyz = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";
        let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();

        let uhf_cfg = UhfConfig {
            max_iter: 200,
            energy_conv: 1e-10,
            density_conv: 1e-8,
            ..Default::default()
        };
        let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &uhf_cfg).unwrap();
        println!("OH UHF: E={:.8}, iters={}", uhf.energy, uhf.iterations);

        let res = u_ri_mp2(&mol, &obs, &dfbs, op, &uhf, &RiMp2Config::default()).unwrap();
        println!(
            "U-RI-MP2 components: αα={:.6e} ββ={:.6e} αβ={:.6e}",
            res.components.e_aa, res.components.e_bb, res.components.e_ab
        );
        println!("E_corr (ferric) = {:.8}", res.mp2_corr);
        println!("E_corr (PySCF)  = -0.15100299");
        let pyscf_e_corr = -0.151002988955374;
        let diff = (res.mp2_corr - pyscf_e_corr).abs();
        println!("diff = {:.3e} Ha", diff);
        assert!(
            diff < 5e-4,
            "U-RI-MP2 on OH off by {:.3e} Ha vs PySCF UMP2 (RI noise tolerance 5e-4)",
            diff
        );
    }

    /// Validate amplitude builder on OH/cc-pVDZ:
    /// (1) energy from amplitudes matches the closed-form `u_ri_mp2`
    /// (2) same-spin amplitudes have correct antisymmetry t[i,j,a,b] = -t[j,i,a,b] = -t[i,j,b,a]
    /// (3) αβ amplitudes have no antisymmetry (mixed spin).
    #[test]
    fn u_mp2_amplitudes_consistent_on_oh() {
        let ctx = ParallelContext::default();
        let xyz = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";
        let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let uhf_cfg = UhfConfig {
            max_iter: 200,
            energy_conv: 1e-10,
            density_conv: 1e-8,
            ..Default::default()
        };
        let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &uhf_cfg).unwrap();

        let energy_res = u_ri_mp2(&mol, &obs, &dfbs, op, &uhf, &RiMp2Config::default()).unwrap();
        let amps =
            compute_u_mp2_amplitudes(&mol, &obs, &dfbs, op, &uhf, &RiMp2Config::default()).unwrap();

        // Energy consistency
        let diff = (amps.components.e_total - energy_res.components.e_total).abs();
        println!(
            "E_total via amplitudes = {:.10}, via u_ri_mp2 = {:.10}, diff = {:.3e}",
            amps.components.e_total, energy_res.components.e_total, diff
        );
        assert!(
            diff < 1e-12,
            "amplitude energy disagrees with closed-form: diff={}",
            diff
        );

        // Same-spin antisymmetry: t[i,j,a,b] should equal -t[j,i,a,b] and -t[i,j,b,a]
        let nocc_a = amps.inter_a.nocc;
        let nvir_a = amps.inter_a.nvir;
        let mut max_asym = 0.0_f64;
        for i in 0..nocc_a {
            for j in 0..nocc_a {
                for a in 0..nvir_a {
                    for b_idx in 0..nvir_a {
                        let t_ijab = amps.t_aa[(i, j, a, b_idx)];
                        let t_jiab = amps.t_aa[(j, i, a, b_idx)];
                        let t_ijba = amps.t_aa[(i, j, b_idx, a)];
                        max_asym = max_asym.max((t_ijab + t_jiab).abs());
                        max_asym = max_asym.max((t_ijab + t_ijba).abs());
                    }
                }
            }
        }
        println!(
            "max |t_aa[ijab] + t_aa[jiab]| = {:.3e} (must be zero)",
            max_asym
        );
        assert!(
            max_asym < 1e-12,
            "αα amplitudes not antisymmetric: max asym={}",
            max_asym
        );

        // Shapes
        println!(
            "shapes: t_aa={:?}, t_bb={:?}, t_ab={:?}",
            amps.t_aa.shape(),
            amps.t_bb.shape(),
            amps.t_ab.shape()
        );
        println!(
            "components: αα={:.6e} ββ={:.6e} αβ={:.6e}",
            amps.components.e_aa, amps.components.e_bb, amps.components.e_ab
        );
    }

    /// Sanity-check the unrelaxed U-MP2 density matrices on OH/cc-pVDZ:
    /// (1) symmetric per-spin per-block
    /// (2) tr(P_oo_σ) + tr(P_vv_σ) = 0 per spin (density-conservation)
    /// (3) signs: tr(P_oo) < 0, tr(P_vv) > 0
    #[test]
    fn u_mp2_density_sane_on_oh() {
        let ctx = ParallelContext::default();
        let xyz = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";
        let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let uhf_cfg = UhfConfig {
            max_iter: 200,
            energy_conv: 1e-10,
            density_conv: 1e-8,
            ..Default::default()
        };
        let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &uhf_cfg).unwrap();
        let amps =
            compute_u_mp2_amplitudes(&mol, &obs, &dfbs, op, &uhf, &RiMp2Config::default()).unwrap();
        let dens = build_u_mp2_density(&amps);

        for (name, p) in [
            ("p_oo_a", &dens.p_oo_a),
            ("p_vv_a", &dens.p_vv_a),
            ("p_oo_b", &dens.p_oo_b),
            ("p_vv_b", &dens.p_vv_b),
        ] {
            let (n, _) = p.dim();
            let mut max_asym = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    max_asym = max_asym.max((p[(i, j)] - p[(j, i)]).abs());
                }
            }
            println!("{}: max asym = {:.3e}", name, max_asym);
            assert!(
                max_asym < 1e-14,
                "{} not symmetric: max asym={}",
                name,
                max_asym
            );
        }

        let tr = |p: &Array2<f64>| -> f64 { (0..p.dim().0).map(|i| p[(i, i)]).sum() };
        let tr_oo_a = tr(&dens.p_oo_a);
        let tr_vv_a = tr(&dens.p_vv_a);
        let tr_oo_b = tr(&dens.p_oo_b);
        let tr_vv_b = tr(&dens.p_vv_b);
        println!(
            "α: tr(P_oo)={:.6e}, tr(P_vv)={:.6e}, sum={:.3e}",
            tr_oo_a,
            tr_vv_a,
            tr_oo_a + tr_vv_a
        );
        println!(
            "β: tr(P_oo)={:.6e}, tr(P_vv)={:.6e}, sum={:.3e}",
            tr_oo_b,
            tr_vv_b,
            tr_oo_b + tr_vv_b
        );
        assert!(tr_oo_a < 0.0 && tr_vv_a > 0.0, "α sign wrong");
        assert!(tr_oo_b < 0.0 && tr_vv_b > 0.0, "β sign wrong");
        assert!((tr_oo_a + tr_vv_a).abs() < 1e-10, "α density not conserved");
        assert!((tr_oo_b + tr_vv_b).abs() < 1e-10, "β density not conserved");
    }

    /// On a closed-shell singlet (H2 spin=1), α and β U-MP2 densities
    /// must be equal (no spin polarization), and each equals half the
    /// closed-shell density.
    #[test]
    fn u_mp2_density_alpha_eq_beta_on_h2() {
        let ctx = ParallelContext::default();
        let xyz = "2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = ferric_scf::rhf::solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default())
            .unwrap();
        let c_seed = rhf.mos_r().clone();
        let uhf_cfg = UhfConfig {
            max_iter: 200,
            energy_conv: 1e-10,
            density_conv: 1e-8,
            ..Default::default()
        };
        let uhf = ferric_scf::uhf::solve_uhf_with_guess(
            &ctx,
            &mol,
            &obs,
            &bounds,
            &uhf_cfg,
            Some((&c_seed, &c_seed)),
        )
        .unwrap();
        let amps =
            compute_u_mp2_amplitudes(&mol, &obs, &dfbs, op, &uhf, &RiMp2Config::default()).unwrap();
        let dens = build_u_mp2_density(&amps);

        // Note: α/β MOs in a closed-shell-via-UHF run agree in *eigenvalues*
        // but their *eigenvectors* can differ inside degenerate virtual blocks
        // (cc-pVDZ on H2 has multiple degenerate virtuals). The density matrix
        // P is therefore gauge-dependent — but its trace and eigenvalues
        // (natural occupation numbers) are invariant. Compare those.
        let tr = |p: &Array2<f64>| -> f64 { (0..p.dim().0).map(|i| p[(i, i)]).sum() };
        let tr_oo_a = tr(&dens.p_oo_a);
        let tr_oo_b = tr(&dens.p_oo_b);
        let tr_vv_a = tr(&dens.p_vv_a);
        let tr_vv_b = tr(&dens.p_vv_b);
        println!(
            "H2 traces: P_oo_a={:.6e} P_oo_b={:.6e}; P_vv_a={:.6e} P_vv_b={:.6e}",
            tr_oo_a, tr_oo_b, tr_vv_a, tr_vv_b
        );
        assert!((tr_oo_a - tr_oo_b).abs() < 1e-10);
        assert!((tr_vv_a - tr_vv_b).abs() < 1e-10);
        // Per-spin conservation
        assert!((tr_oo_a + tr_vv_a).abs() < 1e-10);
        assert!((tr_oo_b + tr_vv_b).abs() < 1e-10);
        // E_aa = E_bb = 0 (each spin has a single occupied orbital, no same-spin pair)
        assert!(amps.components.e_aa.abs() < 1e-14);
        assert!(amps.components.e_bb.abs() < 1e-14);
    }

    /// FD-validate the analytic U-MP2 integral-response orbital gradient on OH/cc-pVDZ.
    ///
    /// Both FD and analytic are computed at the SAME Rust UHF MOs — no Python
    /// AO-ordering dependency. The FD uses `u_mp2_energy_fixed_eps` with the
    /// UHF orbital energies held fixed, matching the "integral-response-only"
    /// part of the gradient that the analytic code computes.
    ///
    /// Gate: per-block analytic-vs-central-difference agreement max|Δ| ≤ 1e-6
    /// (h = 1e-4 central difference; observed ~1e-10 across all four blocks, so
    /// the bound also catches any reindexing sign/stride regression). Direction
    /// (c ≈ +1) is reported alongside for diagnostics.
    #[test]
    fn u_mp2_orbital_gradient_vs_fd_on_oh() {
        let ctx = ParallelContext::default();
        let xyz = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";
        let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let uhf_cfg = UhfConfig {
            max_iter: 200,
            energy_conv: 1e-10,
            density_conv: 1e-8,
            ..Default::default()
        };
        let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &uhf_cfg).unwrap();
        let amps =
            compute_u_mp2_amplitudes(&mol, &obs, &dfbs, op, &uhf, &RiMp2Config::default()).unwrap();

        let nocc_a = amps.inter_a.nocc;
        let nocc_b = amps.inter_b.nocc;
        let nvir_a = amps.inter_a.nvir;
        let nvir_b = amps.inter_b.nvir;
        let nmo = uhf.mos_a().ncols();

        let eps_a = uhf.eps_a().to_vec();
        let eps_b = uhf.eps_b().to_vec();
        let c_a0 = uhf.mos_a().clone();
        let c_b0 = uhf.mos_b().clone();

        // --- In-Rust FD gradient (fixed-eps, block-selected) ---
        // Cayley rotation: U = (I - κ/2)^{-1} (I + κ/2), matching Python harness.
        // Solve: (I - κ/2) @ U = (I + κ/2) column by column.
        let cayley = |nmo: usize, a: usize, k: usize, nocc: usize, h: f64| -> Array2<f64> {
            let mut kap = Array2::<f64>::zeros((nmo, nmo));
            kap[(nocc + a, k)] = h;
            kap[(k, nocc + a)] = -h;
            let eye = Array2::<f64>::eye(nmo);
            let lhs = &eye - 0.5 * &kap; // I - κ/2
            let rhs = &eye + 0.5 * &kap; // I + κ/2
            let mut u = Array2::zeros((nmo, nmo));
            for col in 0..nmo {
                let rhs_col = rhs.column(col).to_owned();
                let u_col = lhs.solve(&rhs_col).unwrap();
                u.column_mut(col).assign(&u_col);
            }
            u
        };
        let step = 1e-4_f64;

        // FD for α rotation: per-block (aa, ab)
        let mut fd_same_a = Array2::<f64>::zeros((nvir_a, nocc_a));
        let mut fd_ab_a = Array2::<f64>::zeros((nvir_a, nocc_a));
        for a in 0..nvir_a {
            for k in 0..nocc_a {
                let up = cayley(nmo, a, k, nocc_a, step);
                let um = cayley(nmo, a, k, nocc_a, -step);
                let c_a_p = c_a0.dot(&up);
                let c_a_m = c_a0.dot(&um);
                let ep_aa = super::u_mp2_energy_fixed_eps(
                    &obs, &dfbs, op, &c_a_p, &c_b0, &eps_a, &eps_b, nocc_a, nocc_b, "aa",
                )
                .unwrap();
                let em_aa = super::u_mp2_energy_fixed_eps(
                    &obs, &dfbs, op, &c_a_m, &c_b0, &eps_a, &eps_b, nocc_a, nocc_b, "aa",
                )
                .unwrap();
                let ep_ab = super::u_mp2_energy_fixed_eps(
                    &obs, &dfbs, op, &c_a_p, &c_b0, &eps_a, &eps_b, nocc_a, nocc_b, "ab",
                )
                .unwrap();
                let em_ab = super::u_mp2_energy_fixed_eps(
                    &obs, &dfbs, op, &c_a_m, &c_b0, &eps_a, &eps_b, nocc_a, nocc_b, "ab",
                )
                .unwrap();
                fd_same_a[(a, k)] = (ep_aa - em_aa) / (2.0 * step);
                fd_ab_a[(a, k)] = (ep_ab - em_ab) / (2.0 * step);
            }
        }
        // FD for β rotation: per-block (bb, ab)
        let mut fd_same_b = Array2::<f64>::zeros((nvir_b, nocc_b));
        let mut fd_ab_b = Array2::<f64>::zeros((nvir_b, nocc_b));
        for a in 0..nvir_b {
            for k in 0..nocc_b {
                let up = cayley(nmo, a, k, nocc_b, step);
                let um = cayley(nmo, a, k, nocc_b, -step);
                let c_b_p = c_b0.dot(&up);
                let c_b_m = c_b0.dot(&um);
                let ep_bb = super::u_mp2_energy_fixed_eps(
                    &obs, &dfbs, op, &c_a0, &c_b_p, &eps_a, &eps_b, nocc_a, nocc_b, "bb",
                )
                .unwrap();
                let em_bb = super::u_mp2_energy_fixed_eps(
                    &obs, &dfbs, op, &c_a0, &c_b_m, &eps_a, &eps_b, nocc_a, nocc_b, "bb",
                )
                .unwrap();
                let ep_ab = super::u_mp2_energy_fixed_eps(
                    &obs, &dfbs, op, &c_a0, &c_b_p, &eps_a, &eps_b, nocc_a, nocc_b, "ab",
                )
                .unwrap();
                let em_ab = super::u_mp2_energy_fixed_eps(
                    &obs, &dfbs, op, &c_a0, &c_b_m, &eps_a, &eps_b, nocc_a, nocc_b, "ab",
                )
                .unwrap();
                fd_same_b[(a, k)] = (ep_bb - em_bb) / (2.0 * step);
                fd_ab_b[(a, k)] = (ep_ab - em_ab) / (2.0 * step);
            }
        }

        // --- Analytic gradient blocks ---
        let b_full_a = crate::oo_rimp2::compute_b_full_mo(&obs, &dfbs, op, &c_a0).unwrap();
        let b_full_b = crate::oo_rimp2::compute_b_full_mo(&obs, &dfbs, op, &c_b0).unwrap();
        let blocks = super::compute_u_mp2_orbital_gradient_blocks(&amps, &b_full_a, &b_full_b);

        let report = |label: &str, g: &Array2<f64>, g_fd: &Array2<f64>| -> bool {
            let nfd = g_fd.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let n = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let diff = g - g_fd;
            let max_diff = diff.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let rms: f64 = (diff.iter().map(|v| v * v).sum::<f64>() / (diff.len() as f64)).sqrt();
            let dot: f64 = g.iter().zip(g_fd.iter()).map(|(a, b)| a * b).sum();
            let nrm2_fd: f64 = g_fd.iter().map(|v| v * v).sum();
            let c = if nrm2_fd > 0.0 { dot / nrm2_fd } else { 0.0 };
            let rel = if nfd > 0.0 { max_diff / nfd } else { 0.0 };
            // Absolute gate: analytic and FD share the same RI B-tensor path, so
            // the only residual is the O(h²) central-difference truncation
            // (~1e-10 here). 1e-6 is a strict correctness bound for the GEMM
            // reindexings, far tighter than the direction check c>0.95.
            let ok = max_diff < 1e-6 && c > 0.95;
            println!("[{}] |g_an|max={:.4e} |g_fd|max={:.4e}  max|Δ|={:.4e}  rms={:.4e}  c={:+.4}  rel={:.3}  {}",
                     label, n, nfd, max_diff, rms, c, rel, if ok { "✓" } else { "✗" });
            ok
        };

        let ok1 = report(
            "αα→g_a (same_a vs FD αα-only α-rot)",
            &blocks.same_a,
            &fd_same_a,
        );
        let ok2 = report(
            "αβ→g_a (ab_a   vs FD αβ-only α-rot)",
            &blocks.ab_a,
            &fd_ab_a,
        );
        let ok3 = report(
            "ββ→g_b (same_b vs FD ββ-only β-rot)",
            &blocks.same_b,
            &fd_same_b,
        );
        let ok4 = report(
            "αβ→g_b (ab_b   vs FD αβ-only β-rot)",
            &blocks.ab_b,
            &fd_ab_b,
        );
        assert!(
            ok1 && ok2 && ok3 && ok4,
            "per-block gradient failed — see diagnostics above"
        );
    }

    // -----------------------------------------------------------------------
    // Regression: GEMM+rayon kernel vs the OLD serial scalar quintuple loops.
    //
    // Kept as a #[cfg(test)]-only reference implementation (never called by
    // production code) so a future edit to `same_spin_pair_kernel` /
    // `opposite_spin_pair_kernel` can be checked against the original,
    // unoptimized per-element O(naux) strided-dot loops this task replaced.

    /// Old scalar same-spin energy: ¼ Σ [(ia|jb)-(ib|ja)]² / D, unblocked.
    fn old_scalar_same_spin_energy(
        b: &Array2<f64>,
        eps: &[f64],
        nocc: usize,
        nvir: usize,
        first_occ: usize,
        nocc_total: usize,
    ) -> f64 {
        let naux = b.nrows();
        let mut energy = 0.0;
        for i in 0..nocc {
            let eps_i = eps[first_occ + i];
            for j in 0..nocc {
                let eps_j = eps[first_occ + j];
                for a in 0..nvir {
                    let eps_a = eps[nocc_total + a];
                    let ia = i * nvir + a;
                    let ja = j * nvir + a;
                    for b_idx in 0..nvir {
                        let eps_b = eps[nocc_total + b_idx];
                        let jb = j * nvir + b_idx;
                        let ib = i * nvir + b_idx;
                        let mut eri_iajb = 0.0;
                        let mut eri_ibja = 0.0;
                        for p in 0..naux {
                            eri_iajb += b[(p, ia)] * b[(p, jb)];
                            eri_ibja += b[(p, ib)] * b[(p, ja)];
                        }
                        let diff = eri_iajb - eri_ibja;
                        let denom = eps_i + eps_j - eps_a - eps_b;
                        energy += diff * diff / denom;
                    }
                }
            }
        }
        0.25 * energy
    }

    /// Old scalar opposite-spin energy: Σ (ia|JB)² / D, unblocked.
    #[allow(clippy::too_many_arguments)]
    fn old_scalar_opposite_spin_energy(
        b_a: &Array2<f64>,
        b_b: &Array2<f64>,
        eps_a: &[f64],
        eps_b: &[f64],
        nocc_a: usize,
        nvir_a: usize,
        first_occ_a: usize,
        nocc_total_a: usize,
        nocc_b: usize,
        nvir_b: usize,
        first_occ_b: usize,
        nocc_total_b: usize,
    ) -> f64 {
        let naux = b_a.nrows();
        assert_eq!(naux, b_b.nrows());
        let mut energy = 0.0;
        for i in 0..nocc_a {
            let eps_i = eps_a[first_occ_a + i];
            for a in 0..nvir_a {
                let eps_a_v = eps_a[nocc_total_a + a];
                let ia = i * nvir_a + a;
                for jj in 0..nocc_b {
                    let eps_j = eps_b[first_occ_b + jj];
                    for bb_idx in 0..nvir_b {
                        let eps_b_v = eps_b[nocc_total_b + bb_idx];
                        let jb = jj * nvir_b + bb_idx;
                        let mut eri = 0.0;
                        for p in 0..naux {
                            eri += b_a[(p, ia)] * b_b[(p, jb)];
                        }
                        let denom = eps_i + eps_j - eps_a_v - eps_b_v;
                        energy += eri * eri / denom;
                    }
                }
            }
        }
        energy
    }

    /// Regression: on OH/cc-pVDZ, the new GEMM+rayon kernel's total U-MP2
    /// energy (αα+ββ+αβ, as built by `compute_u_mp2_amplitudes` /
    /// `u_ri_mp2`) must agree with the OLD serial scalar quintuple loops
    /// (kept above as `old_scalar_*`) to ≤1e-12 Ha — the GEMM restructure only
    /// changes the floating-point *reduction order*, not the algorithm, so
    /// any residual beyond a few ULPs would indicate an indexing/formula bug
    /// introduced by the port.
    #[test]
    fn u_mp2_kernel_matches_old_scalar_loops_on_oh() {
        let ctx = ParallelContext::default();
        let xyz = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";
        let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let uhf_cfg = UhfConfig {
            max_iter: 200,
            energy_conv: 1e-10,
            density_conv: 1e-8,
            ..Default::default()
        };
        let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &uhf_cfg).unwrap();

        let inter_a = compute_rpa_intermediates_spin(
            &mol,
            &obs,
            &dfbs,
            op,
            &uhf,
            &RiMp2Config::default(),
            true,
        )
        .unwrap();
        let inter_b = compute_rpa_intermediates_spin(
            &mol,
            &obs,
            &dfbs,
            op,
            &uhf,
            &RiMp2Config::default(),
            false,
        )
        .unwrap();
        let eps_a: &[f64] = &uhf.eps_alpha;
        let eps_b: &[f64] = uhf.eps_beta.as_deref().unwrap_or(&uhf.eps_alpha);

        // New kernel path (same call as u_ri_mp2 / same_spin_pair_energy /
        // opposite_spin_pair_energy).
        let e_aa_new = same_spin_pair_energy(&inter_a, eps_a, false);
        let e_bb_new = same_spin_pair_energy(&inter_b, eps_b, false);
        let e_ab_new = opposite_spin_pair_energy(&inter_a, &inter_b, eps_a, eps_b, false);
        let e_total_new = e_aa_new + e_bb_new + e_ab_new;

        // Old scalar quintuple-loop path, same intermediates.
        let e_aa_old = old_scalar_same_spin_energy(
            &inter_a.b_ov,
            eps_a,
            inter_a.nocc,
            inter_a.nvir,
            inter_a.first_occ,
            inter_a.nocc_total,
        );
        let e_bb_old = old_scalar_same_spin_energy(
            &inter_b.b_ov,
            eps_b,
            inter_b.nocc,
            inter_b.nvir,
            inter_b.first_occ,
            inter_b.nocc_total,
        );
        let e_ab_old = old_scalar_opposite_spin_energy(
            &inter_a.b_ov,
            &inter_b.b_ov,
            eps_a,
            eps_b,
            inter_a.nocc,
            inter_a.nvir,
            inter_a.first_occ,
            inter_a.nocc_total,
            inter_b.nocc,
            inter_b.nvir,
            inter_b.first_occ,
            inter_b.nocc_total,
        );
        let e_total_old = e_aa_old + e_bb_old + e_ab_old;

        let diff_aa = (e_aa_new - e_aa_old).abs();
        let diff_bb = (e_bb_new - e_bb_old).abs();
        let diff_ab = (e_ab_new - e_ab_old).abs();
        let diff_total = (e_total_new - e_total_old).abs();
        println!(
            "OH kernel-vs-scalar: αα new={:.15e} old={:.15e} diff={:.3e}",
            e_aa_new, e_aa_old, diff_aa
        );
        println!(
            "OH kernel-vs-scalar: ββ new={:.15e} old={:.15e} diff={:.3e}",
            e_bb_new, e_bb_old, diff_bb
        );
        println!(
            "OH kernel-vs-scalar: αβ new={:.15e} old={:.15e} diff={:.3e}",
            e_ab_new, e_ab_old, diff_ab
        );
        println!(
            "OH kernel-vs-scalar: total new={:.15e} old={:.15e} diff={:.3e}",
            e_total_new, e_total_old, diff_total
        );
        assert!(
            diff_aa < 1e-12,
            "αα kernel disagrees with old scalar loop: diff={diff_aa:e}"
        );
        assert!(
            diff_bb < 1e-12,
            "ββ kernel disagrees with old scalar loop: diff={diff_bb:e}"
        );
        assert!(
            diff_ab < 1e-12,
            "αβ kernel disagrees with old scalar loop: diff={diff_ab:e}"
        );
        assert!(
            diff_total < 1e-12,
            "total kernel disagrees with old scalar loop: diff={diff_total:e}"
        );

        // Also check the amplitude builders agree bit-for-bit in energy with
        // the closed-form path (already covered by
        // u_mp2_amplitudes_consistent_on_oh), and additionally cross-check
        // against the old-scalar total here so both amplitude AND
        // energy-only call sites are pinned by one test.
        let amps =
            compute_u_mp2_amplitudes(&mol, &obs, &dfbs, op, &uhf, &RiMp2Config::default()).unwrap();
        let diff_amp_total = (amps.components.e_total - e_total_old).abs();
        println!(
            "OH amplitude-path total = {:.15e} vs old scalar total = {:.15e}, diff={:.3e}",
            amps.components.e_total, e_total_old, diff_amp_total
        );
        assert!(
            diff_amp_total < 1e-12,
            "amplitude-path total disagrees with old scalar loop: diff={diff_amp_total:e}"
        );
    }

    /// Deterministic values in `[-0.5, 0.5)` with no index symmetry.
    fn lcg_values(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
            })
            .collect()
    }

    fn lcg_array4(shape: (usize, usize, usize, usize), seed: u64) -> Array4<f64> {
        let n = shape.0 * shape.1 * shape.2 * shape.3;
        Array4::from_shape_vec(shape, lcg_values(n, seed)).unwrap()
    }

    /// Per-element value, absolute term sum `S` and term count `N` of a block.
    struct Scalar {
        val: Array2<f64>,
        abs: Array2<f64>,
        n: usize,
    }

    impl Scalar {
        fn new(dim: (usize, usize), n: usize) -> Self {
            Self {
                val: Array2::zeros(dim),
                abs: Array2::zeros(dim),
                n,
            }
        }

        /// Assert `got` is within `(N + 2)·ε·S` of the scalar value, per
        /// element, and that the bar is tight (see the calling test's doc).
        fn check(&self, label: &str, got: &Array2<f64>) {
            assert_eq!(got.dim(), self.val.dim(), "{label}: shape");
            let n = (self.n + 2) as f64;
            for ((idx, g), (w, s)) in got.indexed_iter().zip(self.val.iter().zip(self.abs.iter())) {
                let bar = n * f64::EPSILON * s;
                assert!(bar < 1e-10, "{label}{idx:?}: bar {bar:e} is not tight");
                assert!(
                    (g - w).abs() <= bar,
                    "{label}{idx:?}: {g:e} vs scalar {w:e}, |diff| {:e} > bar {bar:e}",
                    (g - w).abs()
                );
            }
        }
    }

    /// The U-MP2 density blocks element by element, as the Bozkaya formulas
    /// in [`UMp2Density`]'s doc read, with each element's absolute term sum.
    /// Independent of the Gram-product construction.
    fn u_mp2_density_scalar_reference(
        t_aa: &Array4<f64>,
        t_bb: &Array4<f64>,
        t_ab: &Array4<f64>,
    ) -> [Scalar; 4] {
        let (oa, _, va, _) = t_aa.dim();
        let (ob, _, vb, _) = t_bb.dim();
        let mut oo_a = Scalar::new((oa, oa), oa * va * va + ob * va * vb);
        let mut vv_a = Scalar::new((va, va), oa * oa * va + oa * ob * vb);
        let mut oo_b = Scalar::new((ob, ob), ob * vb * vb + oa * va * vb);
        let mut vv_b = Scalar::new((vb, vb), ob * ob * vb + oa * ob * va);
        let add = |blk: &mut Scalar, p: usize, q: usize, c: f64, x: f64, y: f64| {
            blk.val[(p, q)] += c * x * y;
            blk.abs[(p, q)] += (c * x * y).abs();
        };
        for i in 0..oa {
            for j in 0..oa {
                for k in 0..oa {
                    for a in 0..va {
                        for b in 0..va {
                            add(
                                &mut oo_a,
                                i,
                                j,
                                -0.5,
                                t_aa[(i, k, a, b)],
                                t_aa[(j, k, a, b)],
                            );
                        }
                    }
                }
                for kk in 0..ob {
                    for a in 0..va {
                        for bb in 0..vb {
                            add(
                                &mut oo_a,
                                i,
                                j,
                                -1.0,
                                t_ab[(i, kk, a, bb)],
                                t_ab[(j, kk, a, bb)],
                            );
                        }
                    }
                }
            }
        }
        for a in 0..va {
            for b in 0..va {
                for i in 0..oa {
                    for j in 0..oa {
                        for c in 0..va {
                            add(&mut vv_a, a, b, 0.5, t_aa[(i, j, a, c)], t_aa[(i, j, b, c)]);
                        }
                    }
                    for jj in 0..ob {
                        for bb in 0..vb {
                            add(
                                &mut vv_a,
                                a,
                                b,
                                1.0,
                                t_ab[(i, jj, a, bb)],
                                t_ab[(i, jj, b, bb)],
                            );
                        }
                    }
                }
            }
        }
        for i in 0..ob {
            for j in 0..ob {
                for k in 0..ob {
                    for a in 0..vb {
                        for b in 0..vb {
                            add(
                                &mut oo_b,
                                i,
                                j,
                                -0.5,
                                t_bb[(i, k, a, b)],
                                t_bb[(j, k, a, b)],
                            );
                        }
                    }
                }
                for kk in 0..oa {
                    for aa in 0..va {
                        for b in 0..vb {
                            add(
                                &mut oo_b,
                                i,
                                j,
                                -1.0,
                                t_ab[(kk, i, aa, b)],
                                t_ab[(kk, j, aa, b)],
                            );
                        }
                    }
                }
            }
        }
        for a in 0..vb {
            for b in 0..vb {
                for i in 0..ob {
                    for j in 0..ob {
                        for c in 0..vb {
                            add(&mut vv_b, a, b, 0.5, t_bb[(i, j, a, c)], t_bb[(i, j, b, c)]);
                        }
                    }
                }
                for ii in 0..oa {
                    for jj in 0..ob {
                        for aa in 0..va {
                            add(
                                &mut vv_b,
                                a,
                                b,
                                1.0,
                                t_ab[(ii, jj, aa, a)],
                                t_ab[(ii, jj, aa, b)],
                            );
                        }
                    }
                }
            }
        }
        [oo_a, vv_a, oo_b, vv_b]
    }

    /// `build_u_mp2_density` (Gram products of amplitude slices on the pool)
    /// against the scalar definition, on amplitudes with no symmetry and
    /// unequal spin dimensions, so a transposed block, a swapped spin axis
    /// or a mislabelled `t_αβ` axis changes the result at order one.
    ///
    /// Bar, per element: both routes evaluate the same `N` products `c·x·y`
    /// (`c` = ±½ or ±1, exact) and differ only in the order of the additions
    /// and in fused multiply-adds. Each evaluation is within `(N + 1)·u·S` of
    /// the exact sum (`u` the unit roundoff, `S = Σ|c·x·y|`), so the two
    /// differ by at most `2(N + 1)·u·S ≤ 2(N + 2)·u·S = (N + 2)·ε·S`.
    ///
    /// Run at the production slab and at a 7-element slab, which cuts every
    /// `P_oo` Gram product into several contraction slabs. Pools of 1, 2 and
    /// 5 threads must give the same bits.
    ///
    /// FAILS if a block is transposed, a slab is dropped or overwritten, or
    /// the per-task results are reduced in a thread-dependent order.
    #[test]
    fn u_mp2_density_gemm_matches_the_scalar_definition() {
        let (oa, ob, va, vb) = (3usize, 2usize, 5usize, 6usize);
        let t_aa = lcg_array4((oa, oa, va, va), 0x9E37_79B9_7F4A_7C15);
        let t_bb = lcg_array4((ob, ob, vb, vb), 0xD1B5_4A32_D192_ED03);
        let t_ab = lcg_array4((oa, ob, va, vb), 0x8CB9_2BA7_2F3D_8DD7);
        let [oo_a, vv_a, oo_b, vv_b] = u_mp2_density_scalar_reference(&t_aa, &t_bb, &t_ab);
        let scale = vv_a.val.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(scale > 0.1, "reference P_vv^α is degenerate: max {scale:e}");
        let run =
            |k_slab| u_mp2_density_from_amplitudes(t_aa.view(), t_bb.view(), t_ab.view(), k_slab);
        for k_slab in [U_DENSITY_K_SLAB, 7] {
            let d = run(k_slab);
            oo_a.check("P_oo^α", &d.p_oo_a);
            vv_a.check("P_vv^α", &d.p_vv_a);
            oo_b.check("P_oo^β", &d.p_oo_b);
            vv_b.check("P_vv^β", &d.p_vv_b);
            for threads in [1usize, 2, 5] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let e = pool.install(|| run(k_slab));
                assert_eq!(
                    e.p_oo_a, d.p_oo_a,
                    "P_oo^α bits differ on {threads} threads"
                );
                assert_eq!(
                    e.p_vv_a, d.p_vv_a,
                    "P_vv^α bits differ on {threads} threads"
                );
                assert_eq!(
                    e.p_oo_b, d.p_oo_b,
                    "P_oo^β bits differ on {threads} threads"
                );
                assert_eq!(
                    e.p_vv_b, d.p_vv_b,
                    "P_vv^β bits differ on {threads} threads"
                );
            }
        }
    }

    /// Random `b_full` (no `p <-> q` symmetry), shape `(naux, nmo, nmo)`.
    fn lcg_b_full(naux: usize, nmo: usize, seed: u64) -> Array3<f64> {
        Array3::from_shape_vec((naux, nmo, nmo), lcg_values(naux * nmo * nmo, seed)).unwrap()
    }

    /// The U-MP2 orbital-gradient blocks by the scalar loops of the
    /// per-element formula (the four same-spin delta terms and the two αβ
    /// parts per spin, as listed in [`compute_u_mp2_orbital_gradient`]'s
    /// doc), with every integral `(pq|rs) = Σ_P B^P_{pq} B^P_{rs}` read
    /// straight from `b_full`. Each element also carries `S`, the sum of the
    /// absolute values of its `t·B·B` monomials, and `N`, their count.
    // One spin's dimensions per argument pair; all distinct.
    #[allow(clippy::too_many_arguments)]
    fn u_mp2_orbital_gradient_scalar_reference(
        t_aa: &Array4<f64>,
        t_bb: &Array4<f64>,
        t_ab: &Array4<f64>,
        b_a: &Array3<f64>,
        b_b: &Array3<f64>,
        first_occ: usize,
        nocc_total_a: usize,
        nocc_total_b: usize,
    ) -> [Scalar; 4] {
        let (oa, _, va, _) = t_aa.dim();
        let (ob, _, vb, _) = t_bb.dim();
        let naux = b_a.shape()[0];
        let (fo, ta, tb) = (first_occ, nocc_total_a, nocc_total_b);
        // (pq|rs) and Σ_P |B^P_pq|·|B^P_rs| across the two spin tensors.
        let eri = |x: &Array3<f64>, p: usize, q: usize, y: &Array3<f64>, r: usize, s: usize| {
            (0..naux).fold((0.0, 0.0), |(v, a), pp| {
                let m = x[(pp, p, q)] * y[(pp, r, s)];
                (v + m, a + m.abs())
            })
        };
        let n_same = |o: usize, v: usize| 4 * naux * (o * v * v + o * o * v);
        let mut same_a = Scalar::new((va, oa), n_same(oa, va));
        let mut ab_a = Scalar::new((va, oa), naux * (ob * va * vb + oa * ob * vb));
        let mut same_b = Scalar::new((vb, ob), n_same(ob, vb));
        let mut ab_b = Scalar::new((vb, ob), naux * (oa * va * vb + oa * ob * va));
        // coef·t·[(e1) − (e2)] into one element.
        let term = |blk: &mut Scalar,
                    c: usize,
                    k: usize,
                    coef: f64,
                    t: f64,
                    e1: (f64, f64),
                    e2: (f64, f64)| {
            blk.val[(c, k)] += coef * t * (e1.0 - e2.0);
            blk.abs[(c, k)] += (coef * t).abs() * (e1.1 + e2.1);
        };
        let zero = (0.0, 0.0);
        // Same-spin σ: `t`, its B tensor, active-occupied offset and virtual offset.
        let same_spin =
            |blk: &mut Scalar, t: &Array4<f64>, b: &Array3<f64>, o: usize, v: usize, tot: usize| {
                let (occ, vir) = (|i: usize| fo + i, |a: usize| tot + a);
                for c in 0..v {
                    for k in 0..o {
                        for j in 0..o {
                            for a in 0..v {
                                for bb in 0..v {
                                    // i = k: ½ t_kjab [(ca|jb) − (cb|ja)]
                                    let e1 = eri(b, vir(c), vir(a), b, occ(j), vir(bb));
                                    let e2 = eri(b, vir(c), vir(bb), b, occ(j), vir(a));
                                    term(blk, c, k, 0.5, t[(k, j, a, bb)], e1, e2);
                                    // j = k: ½ t_jkab [(cb|ja) − (ca|jb)]
                                    term(blk, c, k, 0.5, t[(j, k, a, bb)], e2, e1);
                                }
                            }
                        }
                        for i in 0..o {
                            for j in 0..o {
                                for x in 0..v {
                                    // a = c: −½ t_ijcx [(ik|jx) − (jk|ix)]
                                    let e1 = eri(b, occ(i), occ(k), b, occ(j), vir(x));
                                    let e2 = eri(b, occ(j), occ(k), b, occ(i), vir(x));
                                    term(blk, c, k, -0.5, t[(i, j, c, x)], e1, e2);
                                    // b = c: −½ t_ijxc [(jk|ix) − (ik|jx)]
                                    term(blk, c, k, -0.5, t[(i, j, x, c)], e2, e1);
                                }
                            }
                        }
                    }
                }
            };
        same_spin(&mut same_a, t_aa, b_a, oa, va, ta);
        same_spin(&mut same_b, t_bb, b_b, ob, vb, tb);
        for c in 0..va {
            for k in 0..oa {
                for jj in 0..ob {
                    for a in 0..va {
                        for bb in 0..vb {
                            // 2 t_kJaB (ca_α|JB_β)
                            let e = eri(b_a, ta + c, ta + a, b_b, fo + jj, tb + bb);
                            term(&mut ab_a, c, k, 2.0, t_ab[(k, jj, a, bb)], e, zero);
                        }
                    }
                    for i in 0..oa {
                        for bb in 0..vb {
                            // −2 t_iJcB (ik_α|JB_β)
                            let e = eri(b_a, fo + i, fo + k, b_b, fo + jj, tb + bb);
                            term(&mut ab_a, c, k, -2.0, t_ab[(i, jj, c, bb)], e, zero);
                        }
                    }
                }
            }
        }
        for c in 0..vb {
            for k in 0..ob {
                for i in 0..oa {
                    for a in 0..va {
                        for bb in 0..vb {
                            // 2 t_ikaB (ia_α|cB_β)
                            let e = eri(b_a, fo + i, ta + a, b_b, tb + c, tb + bb);
                            term(&mut ab_b, c, k, 2.0, t_ab[(i, k, a, bb)], e, zero);
                        }
                    }
                    for jj in 0..ob {
                        for a in 0..va {
                            // −2 t_iJac (ia_α|Jk_β)
                            let e = eri(b_a, fo + i, ta + a, b_b, fo + jj, fo + k);
                            term(&mut ab_b, c, k, -2.0, t_ab[(i, jj, a, c)], e, zero);
                        }
                    }
                }
            }
        }
        [same_a, ab_a, same_b, ab_b]
    }

    /// The U-MP2 orbital gradient (RI `Γ` route on the pool, no VVOV block)
    /// against the scalar definition, on amplitudes and B tensors with no
    /// symmetry, unequal spin dimensions and a frozen core, so a transposed
    /// block, a swapped spin or a misplaced orbital offset changes the result
    /// at order one.
    ///
    /// Bar, per element: the element is `Σ c·t·B·B` over `N` monomials (`c` a
    /// power of two). Each route forms every monomial with two rounded
    /// multiplications and sums the monomials (with the amplitude or integral
    /// combinations as internal nodes) in a tree of `N − 1` additions, so each
    /// is within `(N + 1)·u·S` of the exact sum, `S = Σ|c·t·B·B|`, and the two
    /// differ by at most `(N + 2)·ε·S`.
    ///
    /// Run at the production `P` slab and at a 2-function slab (four slabs of
    /// the seven auxiliary functions). Pools of 1, 2 and 5 threads must give
    /// the same bits.
    ///
    /// FAILS if a block is transposed, a slab is dropped, or a reduction runs
    /// in a thread-dependent order.
    #[test]
    fn u_mp2_orbital_gradient_gemm_matches_the_scalar_definition() {
        use crate::orbital_gradient_gemm::{GradientB, VV_CONTRACT_AUX_SLAB};
        let (fo, oa, ob, va, vb, naux) = (1usize, 3usize, 2usize, 5usize, 6usize, 7usize);
        let (ta, tb) = (fo + oa, fo + ob);
        let nmo = ta + va;
        assert_eq!(nmo, tb + vb, "one MO count for both spins");
        let t_aa = lcg_array4((oa, oa, va, va), 0x9E37_79B9_7F4A_7C15);
        let t_bb = lcg_array4((ob, ob, vb, vb), 0xD1B5_4A32_D192_ED03);
        let t_ab = lcg_array4((oa, ob, va, vb), 0x8CB9_2BA7_2F3D_8DD7);
        let b_a = lcg_b_full(naux, nmo, 0x2545_F491_4F6C_DD1D);
        let b_b = lcg_b_full(naux, nmo, 0x5851_F42D_4C95_7F2D);
        let reference =
            u_mp2_orbital_gradient_scalar_reference(&t_aa, &t_bb, &t_ab, &b_a, &b_b, fo, ta, tb);
        let scale = reference[0].val.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(scale > 0.1, "reference same_a is degenerate: max {scale:e}");
        let ba = GradientB::extract(&b_a, oa, va, fo, ta);
        let bb = GradientB::extract(&b_b, ob, vb, fo, tb);
        let run = |aux_slab| {
            u_mp2_orbital_gradient_kernel(
                t_aa.view(),
                t_bb.view(),
                t_ab.view(),
                &ba,
                &bb,
                aux_slab,
                &mut PhaseAccumulator::new(),
            )
        };
        for aux_slab in [VV_CONTRACT_AUX_SLAB, 2] {
            let g = run(aux_slab);
            let [same_a, ab_a, same_b, ab_b] = &reference;
            same_a.check("same_a", &g.same_a);
            ab_a.check("ab_a", &g.ab_a);
            same_b.check("same_b", &g.same_b);
            ab_b.check("ab_b", &g.ab_b);
            for threads in [1usize, 2, 5] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let h = pool.install(|| run(aux_slab));
                assert_eq!(
                    h.same_a, g.same_a,
                    "same_a bits differ on {threads} threads"
                );
                assert_eq!(h.ab_a, g.ab_a, "ab_a bits differ on {threads} threads");
                assert_eq!(
                    h.same_b, g.same_b,
                    "same_b bits differ on {threads} threads"
                );
                assert_eq!(h.ab_b, g.ab_b, "ab_b bits differ on {threads} threads");
            }
        }
    }
}
