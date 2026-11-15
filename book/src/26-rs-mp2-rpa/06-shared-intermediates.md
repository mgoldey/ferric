# 26.6 Shared intermediates across the two kernels

<!-- section: 26.6; status: edit; registry: rs-mp2-rpa;
     theory: crates/ferric-rpa/theory.md#"Range-separated MP2 + RPA" -->

The MP2 components and the dRPA energy of one operator need the same
three-index tensor. `rs_mp2_lr_rpa` builds it once per operator and hands it
to both kernels. This section describes that sharing, the memory it costs,
and the test that pins it.

| Symbol | Meaning | Code name |
|---|---|---|
| $\tilde B^P_{ia}[v]$ | metric-dressed three-index tensor of operator $v$, $\tilde{\mathbf B} = \mathbf F\,(Q|ia)_v$ with $\mathbf F\mathbf V_v\mathbf F^{\mathsf T} = \mathbf 1$ | `RpaIntermediates::{b_ov,v_inv_sqrt}` |
| $N_{\mathrm{aux}}, n_o, n_v$ | auxiliary functions, active occupied, virtual orbitals | `naux`, `nocc`, `nvir` |

| Function | Role |
|---|---|
| `ferric-mp2/src/rimp2.rs::compute_rpa_intermediates` | builds $\tilde{\mathbf B}[v]$: the aux-blocked $(Q|ia)_v$ transform and the metric factor of section 26.4.3 |
| `ferric-mp2/src/rimp2.rs::spin_components_from_b_ov` | $E_{\mathrm{OS}}[v]$, $E_{\mathrm{SS}}[v]$ from $(ia|jb)_v = \sum_P\tilde B^P_{ia}\tilde B^P_{jb}$ |
| `ferric-rpa/src/lib.rs::run_pdep_rpa_from_intermediates` | $E_{\mathrm{dRPA}}[v]$ from the dielectric matrix (18.3) built on the same $\tilde{\mathbf B}$ |

## 26.6.1 One transform per operator

The transform to $\tilde{\mathbf B}$ is the expensive step of both kernels.
Without sharing, a `coupled-rings` call would run five transforms: three for
the MP2 components (C, sr, lr) and two for the dRPA evaluations (C, sr). With
sharing it runs three, one per operator. A `delta-lr` call also runs three
and reuses $\tilde{\mathbf B}[\mathrm{lr}]$ for its single dRPA evaluation.

| Formulation | Transforms | dRPA evaluations | Shared tensor |
|---|---|---|---|
| `delta-lr` | 3 (lr, C, sr) | 1 (lr) | $\tilde{\mathbf B}[\mathrm{lr}]$ |
| `coupled-rings` | 3 (lr, C, sr) | 2 (C, sr) | $\tilde{\mathbf B}[\mathrm C]$, $\tilde{\mathbf B}[\mathrm{sr}]$ |

Sharing changes no number: the MP2 components and the dRPA energy of an
operator see the same fitted integrals whether or not the tensor is shared.
That is also why the second-order cancellation of section 26.1.3 holds for
any fit: $2E_{\mathrm{OS}}[v]$ and the second-order ring of $E_{\mathrm{dRPA}}[v]$
are built from one $\tilde{\mathbf B}[v]$.

`coupled_rings_fusion_is_bit_identical` in `crates/ferric-rpa/src/rs_mp2_rpa.rs`
recomputes the `coupled-rings` energy for H2 at $\omega = 0.42$ Bohr⁻¹ from
independent `ri_mp2_spin_components` and `run_pdep_rpa` calls, one transform
each, and asserts agreement with the shared path to $10^{-10}$ Ha. The name
says bit-identical; the assertion is the $10^{-10}$ bar.

## 26.6.2 Memory

Each `match` arm holds two $\tilde{\mathbf B}$ tensors at once:

- `delta-lr` keeps $\tilde{\mathbf B}[\mathrm{lr}]$ bound for its dRPA
  evaluation while it builds, uses and drops $\tilde{\mathbf B}[\mathrm C]$
  and then $\tilde{\mathbf B}[\mathrm{sr}]$ for their MP2 components.
- `coupled-rings` keeps $\tilde{\mathbf B}[\mathrm C]$ alive to the end of the
  arm, so it is resident through the whole short-range dRPA evaluation on
  $\tilde{\mathbf B}[\mathrm{sr}]$.

The shared peak estimate (`ferric-rpa/src/budget.rs::estimate_peak_bytes`)
charges one such tensor, as for a single `pdep-rpa` call. The driver's
pre-flight adds the second,

$$ \texttt{second\_intermediate\_bov\_bytes} = 8\,N_{\mathrm{aux}}\,n_o\,n_v \ \text{bytes}, \tag{26.11} $$

to the estimate and to the hard (in-core) side of the memory-pool
reservation, before any large allocation. A job whose budget fits one tensor
but not two is refused at the start rather than killed part way.

| Test (`crates/ferric-rpa/tests/mwe_rs_mp2_rpa_second_intermediate.rs`) | What it pins |
|---|---|
| `second_intermediate_bov_bytes_matches_one_bov_buffer` | (26.11) is one `b_ov` buffer |
| `second_bov_term_is_additive_and_material` | the term adds to the estimate and is not negligible |
| `ample_budget_still_passes_with_second_bov_charged` | an ample budget still passes |
| `budget_sized_to_old_estimate_is_now_refused` | a budget sized to the one-tensor estimate is refused |

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
| `crates/ferric-rpa/src/rs_mp2_rpa.rs`, the comments at the pre-flight gate and at `inter_of`, `second_intermediate_bov_bytes`, the `match` arms | 26.6.1, 26.6.2 |
| `crates/ferric-mp2/src/rimp2.rs::RpaIntermediates` | the tensor and its metric factor |
| tests named above | the anchors |
