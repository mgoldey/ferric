# Shared intermediates across the two kernels {#shared-intermediates}

<!-- status: edit; registry: rs-mp2-rpa -->

The MP2 components and the dRPA energy of one operator need the same
three-index tensor. The driver builds it once per operator and hands it
to both kernels. This section describes that sharing, the memory it costs,
and the tests that pin it.

| Symbol | Meaning | Code name |
|---|---|---|
| $\tilde B^P_{ia}[v]$ | metric-dressed three-index tensor of operator $v$, $\tilde{\mathbf B} = \mathbf F\,(Q|ia)_v$ with $\mathbf F\mathbf V_v\mathbf F^{\mathsf T} = \mathbf 1$ | `RpaIntermediates::{b_ov,v_inv_sqrt}` |
| $N_{\mathrm{aux}}, n_o, n_v$ | auxiliary functions, active occupied, virtual orbitals | `naux`, `nocc`, `nvir` |

| Function | Role |
|---|---|
| `ferric-mp2/src/rimp2.rs::compute_rpa_intermediates` | builds $\tilde{\mathbf B}[v]$: the aux-blocked $(Q|ia)_v$ transform and the metric factor of section @sec:two-formulations-code |
| `ferric-mp2/src/rimp2.rs::spin_components_from_b_ov` | $E_{\mathrm{OS}}[v]$, $E_{\mathrm{SS}}[v]$ from $(ia|jb)_v = \sum_P\tilde B^P_{ia}\tilde B^P_{jb}$ |
| `ferric-rpa/src/lib.rs::run_pdep_rpa_from_intermediates` | $E_{\mathrm{dRPA}}[v]$ from the dielectric matrix (@eq:drpa-three-ways-3) built on the same $\tilde{\mathbf B}$ |

## One transform per operator, and the ω-independent stage {#shared-intermediates-transform-per-operator-omega}

The transform to $\tilde{\mathbf B}$ is the expensive step of both kernels.
The driver (`rs_mp2_lr_rpa_scan`; `rs_mp2_lr_rpa` is its one-point case)
builds each operator's tensor once and hands it to every kernel that needs
that operator. Every term of (B) and (T) is a functional of one kernel
(section @sec:shared-intermediates-transform-per-operator-omega), so the Coulomb tensor does not depend on
$\omega$ and is built once per run, not once per point.

| Stage | Tensor | Used for | Built |
|---|---|---|---|
| ω-independent | $\tilde{\mathbf B}[\mathrm C]$ | $E_{\mathrm{MP2}}[v]$; $E_{\mathrm{dRPA}}[v]$ only when `coupled-rings` is requested | once per run |
| per ω, long range | $\tilde{\mathbf B}[\mathrm{lr}]$ | $E_{\mathrm{OS}}[\mathrm{lr}]$ always; $E_{\mathrm{dRPA}}[\mathrm{lr}]$ only when `delta-lr` is requested | once per ω |
| per ω, short range | $\tilde{\mathbf B}[\mathrm{sr}]$ | spin components always; $E_{\mathrm{dRPA}}[\mathrm{sr}]$ only when `coupled-rings` is requested | once per ω |

A run over $n_\omega$ points therefore performs $1 + 2n_\omega$ transforms,
whether it asks for one formulation or both. With the terf attenuator it
performs $1 + n_\omega$: the long range is $\tilde{\mathbf B}[\mathrm C]$ and
$\tilde{\mathbf B}[\mathrm{sr}]$ with signature $(+,-)$
(section @sec:factorization-and-metric-signed), so the Coulomb tensor stays
resident for the whole run and no terf tensor is built. Both formulations at one
$\omega$ share the per-ω tensors.

Sharing changes no number: the MP2 components and the dRPA energy of an
operator see the same fitted integrals whether or not the tensor is shared.
That is also why the second-order cancellation of section @sec:cross-term-rule holds for
any fit: $2E_{\mathrm{OS}}[v]$ and the second-order ring of $E_{\mathrm{dRPA}}[v]$
are built from one $\tilde{\mathbf B}[v]$.

| Test | What it pins |
|---|---|
| `coupled_rings_fusion_is_bit_identical` (`crates/ferric-rpa/src/rs_mp2_rpa.rs`) | the `coupled-rings` energy for H2 at $\omega = 0.42$ Bohr⁻¹ from independent `ri_mp2_spin_components` and `run_pdep_rpa` calls agrees with the shared path to $10^{-10}$ Ha (the name says bit-identical; the assertion is the $10^{-10}$ bar) |
| `scan_is_bitwise_the_single_omega_calls_{water,water_dimer,terf}` (`crates/ferric-rpa/tests/rs_mp2_rpa_scan.rs`) | every field of every scan point equals a separate single-ω call, bit for bit, for both formulations, erf and terf |
| `t_carries_the_independently_computed_coulomb_drpa` (same file) | the once-built Coulomb dRPA in (T) equals an independent `run_pdep_rpa` / `ri_mp2_spin_components` evaluation |

## Memory {#shared-intermediates-memory}

Each tensor lives in its own block and is dropped before the next is
built, so at most one $\tilde{\mathbf B}$ is resident at a time, for any
number of $\omega$ points (two with terf: the held Coulomb tensor and the
current terfc one). Each operator's set runs three phases on top
of the planes it keeps throughout, the metric buffers and one
$\tilde{\mathbf B}$:

| Phase | Holds besides the persistent planes | Code |
|---|---|---|
| build | the AO tensor $(Q|\mu\nu)_v$ at its storage tier, and one stream chunk | `compute_rpa_intermediates` |
| spin | the per-worker $n_o n_v^2$ pair blocks | `spin_components_from_b_ov` |
| dRPA | the static eigensolve planes and the per-worker frequency scratch | `run_pdep_rpa_from_intermediates` |

The AO tensor is dropped when $\tilde{\mathbf B}$ is built, before the
other two phases. The pre-flight charge is the persistent planes plus the
largest phase:

$$ \texttt{scan\_peak\_bytes}(\text{shape}, n_\omega, b) = P + \max(\text{build}, \text{spin}, \text{dRPA}) \quad \text{for all } n_\omega. \label{eq:shared-intermediates-1} $$

The AO tensor takes the tier `ThreeIndexSource::build` picks at the budget
$b - P - \text{chunk}$, and the driver hands the build that budget. The
tier is in core when the tensor fits, the packed triangle when that fits,
and a disk spill (two blocks resident) otherwise. The charge is the tier's
resident bytes, so a run whose in-core tensor does not fit is admitted and
spills. The tier is the same for every operator, every $\omega$ and both
formulations, since they share one shape.

With terf, `ferric-rpa/src/budget.rs::with_signed_lr` adds the held Coulomb tensor to $P$
(and re-plans the AO tier at the smaller budget left), and charges the
signed log-determinant's per-worker scratch, a $2N_{\mathrm{aux}}$-row panel
and a $(2N_{\mathrm{aux}})^2$ matrix, in place of the frequency scratch
when it is larger. `scan_peak_bytes` does not know the attenuator and
leaves both out; `scan_charge_for` and the driver include them.

| Test | What it pins |
|---|---|
| `preflight_estimate_is_independent_of_the_number_of_points` (`crates/ferric-rpa/tests/rs_mp2_rpa_scan.rs`) | (@eq:shared-intermediates-1) at 1, 5 and 50 points; the estimate covers the build and dRPA phases and is tensor-sized |
| `the_charge_follows_the_tier_the_build_takes` (`crates/ferric-rpa/tests/rs_mp2_rpa_spill_tier.rs`) | in core, packed and spill are each reachable on the water dimer, each charge fits its budget, the build takes the planned tier, and the run is admitted |
| `ample_budget_is_bitwise_the_in_core_path` (same file) | at an ample budget both formulations equal the public pieces called at the full budget, bit for bit |
| `packed_and_spilled_match_in_core_within_the_reduction_order_bar` (same file) | both formulations at two $\omega$ agree with the in-core run within a bar derived from the summation order of the $\tilde{\mathbf B}$ stream |

Only the summation order of $\tilde{\mathbf B} = \mathbf F\,(Q|ia)$ depends
on the tier. Two orders of a floating-point sum of $n$ terms $x_k$ differ by
at most $2\gamma_n\sum_k|x_k|$, with $\gamma_n = n\mathbf u/(1 - n\mathbf u)$
and $\mathbf u$ the unit round-off, so elementwise

$$ |\delta\tilde{\mathbf B}| \le 2\gamma_{N_{\mathrm{aux}}}\,|\mathbf F|\,|(Q|ia)| . $$

The energy bar of `packed_and_spilled_match_in_core_within_the_reduction_order_bar`,
for both formulations, is derived from this bound.

The check runs before any large allocation, so a job whose budget does not
fit even a spilled tensor is refused at the start rather than killed part
way.

After each tensor is built, `warn_if_rss_over` compares the resident memory
with the budget and prints one line if it is more than 10% over. It never
stops the run.

## Known limits (2026-11-15)

- The fusion test covers `coupled-rings` on H2 only; `delta-lr` has no
  separate-call comparison.
- `delta-lr` builds the short-range tensor only for the reported
  `e_sr_mp2` and `e_corr_naive`; `coupled-rings` builds the long-range one
  only for `e_lr_mp2` and `e_dmp2_lr`. Neither enters the returned energy.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-rpa/src/rs_mp2_rpa.rs::{scan_impl,scan_peak_bytes}`, the comments at the pre-flight gate and at `inter_of` | @sec:shared-intermediates-transform-per-operator-omega, @sec:shared-intermediates-memory |
| `crates/ferric-rpa/src/budget.rs::scan_charge`, `crates/ferric-integrals/src/three_index_source.rs::planned_residency` | @sec:shared-intermediates-memory |
| `crates/ferric-mp2/src/rimp2.rs::RpaIntermediates` | the tensor and its metric factor |
| tests named above | the anchors |
