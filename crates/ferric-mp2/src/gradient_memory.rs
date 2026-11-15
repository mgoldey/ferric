//! Memory pre-flight of the MP2-family gradients on a fitted (RI-JK)
//! reference.
//!
//! On a four-centre reference the analytic gradient holds the MP2
//! intermediates and little else. On an RI-JK reference it also keeps the
//! reference's RI-J and RI-K tensors for the Z-vector
//! ([`ferric_scf::response_jk::ResponseJk`]) and differentiates the fitted
//! two-electron energy of the full-rank relaxed density
//! ([`ferric_scf::response_jk::fitted_bilinear_gradient`]), whose exchange
//! intermediates are `(naux_JK, m²)` per channel with `m` up to the number of
//! basis functions. That term alone can exceed the RI-J tensor several times
//! over, and the Python entry points install no memory pool to refuse it.
//!
//! [`fitted_gradient_plan`](crate::gradient_memory::fitted_gradient_plan) declares the peak from the shapes before anything
//! is allocated: the terms resident from the J/K route set-up to the end of
//! the gradient, plus the largest stage transient. [`preflight_fitted_gradient`](crate::gradient_memory::preflight_fitted_gradient)
//! checks it against the budget and errors with the breakdown when it does
//! not fit. The model is in `crates/ferric-scf/theory.md`, "Peak memory of the
//! fitted MP2 gradient".

use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::three_index_source::jk_storage::{memory_tier, Tier};

/// Upper bound on `Σ_s m_s²` over the exchange channels of the fitted
/// bilinear two-electron derivative, in units of `nao²`. The derivative
/// splits `B(d1, d2) = Q(u) − Q(v)`; a positive-semidefinite density is one
/// channel of rank `m ≤ nao`, an indefinite one the three channels `P`, `N`,
/// `P + N` with `rank P + rank N ≤ nao`, so `m_P² + m_N² + m_{P+N}² ≤ 2 nao²`.
/// Both `u` and `v` indefinite is the worst case, `4 nao²`.
pub const EXCHANGE_CHANNEL_RANK2_BOUND: usize = 4;

/// 3c-derivative pass block cap of `ferric_scf::df_gradient`
/// (`GRADIENT_3C_BLOCK_CAP_BYTES`, 512 MiB).
const GRADIENT_BLOCK_CAP_BYTES: usize = 512 * 1024 * 1024;

/// The shapes the peak depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FittedGradientShape {
    /// Orbital basis functions.
    pub nao: usize,
    /// Correlation (RI-MP2) auxiliary functions.
    pub naux_ri: usize,
    /// JK-fitting auxiliary functions of the reference.
    pub naux_jk: usize,
    /// Widest shell of the JK-fitting basis (functions).
    pub max_jk_shell: usize,
    /// Correlated (active) occupied orbitals.
    pub nocc: usize,
    /// Virtual orbitals.
    pub nvir: usize,
    /// Rayon workers.
    pub workers: usize,
}

impl FittedGradientShape {
    /// The shapes of `mol` in `obs` with correlation aux `dfbs` and JK aux
    /// `jk_aux` (a bundled name), `frozen_core` frozen orbitals.
    pub fn new(
        mol: &Molecule,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        jk_aux: &str,
        frozen_core: usize,
    ) -> Result<Self, FerricError> {
        let jk = PreparedBasis::new(mol, &ferric_core::basis::bundled(jk_aux)?)?;
        let nocc_total = (mol.nelec() as usize) / 2;
        Ok(FittedGradientShape {
            nao: obs.nbasis(),
            naux_ri: dfbs.nbasis(),
            naux_jk: jk.nbasis(),
            max_jk_shell: jk.shell_dims().iter().copied().max().unwrap_or(0),
            nocc: crate::rimp2::active_occ(nocc_total, frozen_core)?,
            nvir: obs.nbasis().saturating_sub(nocc_total),
            workers: rayon::current_num_threads().max(1),
        })
    }
}

/// Elements (f64) a `(naux, nao, nao)` raw tensor keeps in core under
/// `budget_bytes`: the tier `ferric_integrals` would choose, its packed
/// lower triangle, or one streaming block when it spills.
fn raw_tensor_elems(nao: usize, naux: usize, budget_bytes: usize) -> usize {
    let square = naux * nao * nao;
    let packed = naux * nao * (nao + 1) / 2;
    match memory_tier(nao, naux, budget_bytes) {
        Some(Tier::InCore) => square,
        Some(Tier::InCorePacked) => packed + gradient_block_elems(nao, naux, budget_bytes),
        _ => gradient_block_elems(nao, naux, budget_bytes),
    }
}

/// Elements of one `(b, nao, nao)` 3c block of a `naux`-row tensor: a
/// quarter of the budget, at most 512 MiB, plus one aux shell's slack, and
/// never more rows than the tensor has.
fn gradient_block_elems(nao: usize, naux: usize, budget_bytes: usize) -> usize {
    let row = (nao * nao).max(1);
    let rows = ((budget_bytes / 4).min(GRADIENT_BLOCK_CAP_BYTES) / (row * 8)).max(1);
    (rows + 32).min(naux.max(1)) * row
}

/// The declared peak of one fitted MP2-family gradient evaluation (reference
/// SCF, MP2 energy, Z-vector, Lagrangian and RI response) under
/// `budget_bytes`, with `baseline_bytes` already resident in the process.
pub fn fitted_gradient_plan(
    s: FittedGradientShape,
    budget_bytes: usize,
    baseline_bytes: usize,
) -> MemoryPlan {
    let FittedGradientShape {
        nao,
        naux_ri,
        naux_jk,
        max_jk_shell,
        nocc,
        nvir,
        workers,
    } = s;
    let ov = nocc * nvir;
    let n2 = nao * nao;
    let mut plan = MemoryPlan::with_budget_bytes(budget_bytes, "fitted MP2 gradient");
    plan.reserve_sized(
        "process resident at entry",
        baseline_bytes,
        1,
        Lifetime::Resident,
        1,
    );
    // From the J/K route set-up to the end of the gradient.
    plan.reserve("MP2 amplitudes t2 (nov²)", ov * ov, Lifetime::Resident)
        .reserve("B(P|ia) (naux_RI·nov)", naux_ri * ov, Lifetime::Resident)
        .reserve(
            "RI metric V^-1/2 (naux_RI²)",
            naux_ri * naux_ri,
            Lifetime::Resident,
        )
        .reserve(
            "RI-J raw (P|μν) (JK aux)",
            raw_tensor_elems(nao, naux_jk, budget_bytes),
            Lifetime::Resident,
        )
        .reserve(
            "RI-K dressed band, packed (JK aux)",
            naux_jk * nao * (nao + 1) / 2,
            Lifetime::Resident,
        )
        .reserve(
            "JK metric + V^-1/2 (2 naux_JK²)",
            2 * naux_jk * naux_jk,
            Lifetime::Resident,
        );
    // Stage transients on top of that; only the largest counts.
    plan.reserve(
        "stage: MP2 intermediates (RI raw tensor, (ia|jb), g_i per worker)",
        raw_tensor_elems(nao, naux_ri, budget_bytes) + ov * ov + workers * nocc * nvir * nvir,
        Lifetime::Transient,
    )
    .reserve(
        "stage: RI-K set-up (second raw JK tensor)",
        raw_tensor_elems(nao, naux_jk, budget_bytes),
        Lifetime::Transient,
    )
    .reserve(
        "stage: fitted 2e derivative (exchange channels 2·naux_JK·Σm², metrics, 3c block, slabs)",
        2 * naux_jk * EXCHANGE_CHANNEL_RANK2_BOUND * n2
            + 5 * naux_jk * naux_jk
            + 2 * gradient_block_elems(nao, naux_jk, budget_bytes)
            + workers * max_jk_shell * n2,
        Lifetime::Transient,
    )
    .reserve(
        "stage: RI response (x_ov, y_ov, c_fit, Γ_2c, per-worker scratch)",
        3 * naux_ri * ov + naux_ri * naux_ri + workers * nvir * ov.max(n2),
        Lifetime::Transient,
    );
    plan
}

/// Refuse a fitted MP2-family gradient whose declared peak
/// ([`fitted_gradient_plan`], with the process's resident set at entry) does
/// not fit `budget_bytes`, before any integral is computed. The error names
/// every term and the two ways out.
pub fn preflight_fitted_gradient(
    s: FittedGradientShape,
    budget_bytes: usize,
) -> Result<(), FerricError> {
    let baseline = ferric_core::memory::read_own_rss_bytes().unwrap_or(0);
    fitted_gradient_plan(s, budget_bytes, baseline)
        .check()
        .map_err(|e| {
            FerricError::MemoryExhausted(format!(
                "{e}\nthe RI-JK reference's gradient does not fit the memory budget: raise \
                 [memory] budget_gb (Python memory_budget_gb), or use the four-centre \
                 reference ([scf] jk = \"exact\", Python scf_jk=\"exact\")"
            ))
        })
}

/// [`EXCHANGE_CHANNEL_RANK2_BOUND`] of the open-shell bilinear derivative
/// ([`ferric_scf::response_jk::fitted_bilinear_gradient_open`]): four exchange
/// densities `u_α`, `u_β`, `v_α`, `v_β`, each at most `2 nao²` when
/// indefinite, so `8 nao²`.
pub const OPEN_EXCHANGE_CHANNEL_RANK2_BOUND: usize = 8;

/// The shapes the peak of the unrestricted gradient depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnrestrictedFittedGradientShape {
    /// Orbital basis functions.
    pub nao: usize,
    /// Correlation (RI-MP2) auxiliary functions.
    pub naux_ri: usize,
    /// JK-fitting auxiliary functions of the reference.
    pub naux_jk: usize,
    /// Widest shell of the JK-fitting basis (functions).
    pub max_jk_shell: usize,
    /// Correlated (active) occupied orbitals per spin, `[α, β]`.
    pub nocc: [usize; 2],
    /// Virtual orbitals per spin, `[α, β]`.
    pub nvir: [usize; 2],
    /// Rayon workers.
    pub workers: usize,
}

impl UnrestrictedFittedGradientShape {
    /// The shapes of `mol` (its charge and multiplicity fix `n_α`, `n_β`) in
    /// `obs` with correlation aux `dfbs` and JK aux `jk_aux`, `frozen_core`
    /// orbitals of each spin frozen.
    pub fn new(
        mol: &Molecule,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        jk_aux: &str,
        frozen_core: usize,
    ) -> Result<Self, FerricError> {
        let jk = PreparedBasis::new(mol, &ferric_core::basis::bundled(jk_aux)?)?;
        let nelec = mol.nelec() as usize;
        let two_s = mol.multiplicity.saturating_sub(1);
        if two_s > nelec || !(nelec - two_s).is_multiple_of(2) {
            return Err(FerricError::General(format!(
                "{nelec} electrons with multiplicity {} is not a spin state",
                mol.multiplicity
            )));
        }
        let n = [(nelec + two_s) / 2, (nelec - two_s) / 2];
        let nao = obs.nbasis();
        Ok(Self {
            nao,
            naux_ri: dfbs.nbasis(),
            naux_jk: jk.nbasis(),
            max_jk_shell: jk.shell_dims().iter().copied().max().unwrap_or(0),
            nocc: [
                crate::rimp2::active_occ(n[0], frozen_core)?,
                crate::rimp2::active_occ(n[1], frozen_core)?,
            ],
            nvir: [nao.saturating_sub(n[0]), nao.saturating_sub(n[1])],
            workers: rayon::current_num_threads().max(1),
        })
    }
}

/// The declared peak of one fitted unrestricted RI-MP2 gradient evaluation
/// ([`crate::u_rimp2_gradient::u_rimp2_gradient_analytical`]: UHF, U-MP2
/// energy, coupled Z-vector, Lagrangian and RI response) under
/// `budget_bytes`, with `baseline_bytes` already resident. The terms are the
/// closed-shell ones of [`fitted_gradient_plan`] resolved by spin: three
/// amplitude blocks `t_αα`, `t_ββ`, `t_αβ`, two `B^σ` and two energy weights
/// `x^σ`, one full-MO `b^σ(P|pq)` at a time for `Imat^σ`, and four exchange
/// channel densities in the two-electron derivative.
pub fn fitted_gradient_plan_unrestricted(
    s: UnrestrictedFittedGradientShape,
    budget_bytes: usize,
    baseline_bytes: usize,
) -> MemoryPlan {
    let UnrestrictedFittedGradientShape {
        nao,
        naux_ri,
        naux_jk,
        max_jk_shell,
        nocc: [oa, ob],
        nvir: [va, vb],
        workers,
    } = s;
    let (ova, ovb) = (oa * va, ob * vb);
    let n2 = nao * nao;
    let mut plan = MemoryPlan::with_budget_bytes(budget_bytes, "fitted U-MP2 gradient");
    plan.reserve_sized(
        "process resident at entry",
        baseline_bytes,
        1,
        Lifetime::Resident,
        1,
    );
    plan.reserve(
        "U-MP2 amplitudes t_aa, t_bb, t_ab (nov_a² + nov_b² + nov_a·nov_b)",
        ova * ova + ovb * ovb + ova * ovb,
        Lifetime::Resident,
    )
    .reserve(
        "B^a, B^b and x^a, x^b (2 naux_RI·(nov_a + nov_b))",
        2 * naux_ri * (ova + ovb),
        Lifetime::Resident,
    )
    .reserve(
        "RI metric V^-1/2 (naux_RI²)",
        naux_ri * naux_ri,
        Lifetime::Resident,
    )
    .reserve(
        "RI-J raw (P|μν) (JK aux)",
        raw_tensor_elems(nao, naux_jk, budget_bytes),
        Lifetime::Resident,
    )
    .reserve(
        "RI-K dressed band, packed (JK aux)",
        naux_jk * nao * (nao + 1) / 2,
        Lifetime::Resident,
    )
    .reserve(
        "JK metric + V^-1/2 (2 naux_JK²)",
        2 * naux_jk * naux_jk,
        Lifetime::Resident,
    );
    let (ov_max, v_max) = (ova.max(ovb), va.max(vb));
    plan.reserve(
        "stage: U-MP2 intermediates (RI raw tensor, (ia|jb) blocks, per-worker slabs)",
        raw_tensor_elems(nao, naux_ri, budget_bytes)
            + ov_max * ov_max
            + 2 * workers * ov_max * v_max,
        Lifetime::Transient,
    )
    .reserve(
        "stage: Imat^s (RI raw tensor + one b^s(P|pq), naux_RI·nao²)",
        raw_tensor_elems(nao, naux_ri, budget_bytes) + 2 * naux_ri * n2,
        Lifetime::Transient,
    )
    .reserve(
        "stage: RI-K set-up (second raw JK tensor)",
        raw_tensor_elems(nao, naux_jk, budget_bytes),
        Lifetime::Transient,
    )
    .reserve(
        "stage: fitted open 2e derivative (exchange channels 2·naux_JK·Σm², metrics, 3c block, slabs)",
        2 * naux_jk * OPEN_EXCHANGE_CHANNEL_RANK2_BOUND * n2
            + 5 * naux_jk * naux_jk
            + 2 * gradient_block_elems(nao, naux_jk, budget_bytes)
            + workers * max_jk_shell * n2,
        Lifetime::Transient,
    )
    .reserve(
        "stage: RI response (y^s, c_fit^s, Γ_2c, per-worker scratch)",
        2 * naux_ri * (ova + ovb) + naux_ri * naux_ri + workers * v_max * ov_max.max(n2),
        Lifetime::Transient,
    );
    plan
}

/// [`preflight_fitted_gradient`] for the unrestricted gradient
/// ([`fitted_gradient_plan_unrestricted`]).
pub fn preflight_fitted_gradient_unrestricted(
    s: UnrestrictedFittedGradientShape,
    budget_bytes: usize,
) -> Result<(), FerricError> {
    let baseline = ferric_core::memory::read_own_rss_bytes().unwrap_or(0);
    fitted_gradient_plan_unrestricted(s, budget_bytes, baseline)
        .check()
        .map_err(|e| {
            FerricError::MemoryExhausted(format!(
                "{e}\nthe RI-JK reference's U-MP2 gradient does not fit the memory budget: \
                 raise [memory] budget_gb (Python memory_budget_gb), or use the four-centre \
                 reference ([scf] jk = \"exact\", Python scf_jk=\"exact\")"
            ))
        })
}
