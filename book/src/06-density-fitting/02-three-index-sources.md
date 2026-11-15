# 6.2 Three-index sources: in core, packed, spilled, recomputed

<!-- section: 6.2; status: edit; registry: none;
     theory: crates/ferric-integrals/theory.md#"Three-index storage" -->

The raw tensor $(P|\mu\nu)$ is the largest object most density-fitted methods
hold, $N_{\mathrm{aux}}N^2$ doubles. Every consumer reads it through one type
that decides at build time where the tensor lives, and serves it in blocks of
auxiliary rows. Consumers apply their own metric (section 6.1), so the same
source serves RI-J, RI-K, RI-MP2 and RPA.

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/three_index_source.rs::ThreeIndexSource` (`build_band`, `build_band_screened`, `build_dressed`, `for_each_block`) |
| CLI | `[scf] jk_storage`, `[memory] budget_gb` (`site/src/reference/input.md`) |
| Environment | `FERRIC_JK_STORAGE` |

<!-- include: crates/ferric-integrals/theory.md#"Three-index storage" -->

## 6.2.1 The options

| `jk_storage` | Where the RI-J/K tensor lives |
|---|---|
| `auto` (default) | in memory if it fits, else the packed half, else (one rank, Coulomb) Split-RI-J for J; a tensor still needed goes to `disk` or `direct`, whichever two timings on this machine predict cheaper over 30 passes |
| `memory` | in memory, packed if only that fits; an error if neither fits |
| `disk` | spilled to `$TMPDIR` |
| `direct` | recomputed per pass; nothing written |
| `split` | no RI-J tensor; J by Split-RI-J on one rank |

`site/src/reference/input.md` is the authority for these rules, including the
shared-memory pool and the multi-rank cases.

## 6.2.2 Reproducibility

| Consumer | Across tiers |
|---|---|
| RI-J from the packed, spilled and recomputed tensors | bit-identical: one packed block stream |
| RI-J from the unpacked in-core tensor | last bits differ (summation order) |
| RI-K and other consumers | can differ at summation order between tiers |
| the dressed tensor across thread counts | bit-identical: row split on a 64-row grid, contraction in 128 blocks |

The tier `auto` picks depends on the budget and the free memory in the shared
pool, so the same input can choose differently on a busier machine; pinning
`jk_storage` removes that dependence.

## 6.2.3 Numerical anchors

| What | Test | Bar |
|---|---|---|
| spilled DF tensor bitwise equal to in core | `crates/ferric-scf/tests/mwe_spilled_df_is_bitwise_in_core.rs` | bitwise |
| recompute backend writes nothing | `crates/ferric-integrals/tests/mwe_recompute_backend_avoids_disk.rs` | — |
| spill packs the $\mu\nu$ triangle | `crates/ferric-integrals/tests/mwe_spill_packs_the_mu_nu_triangle.rs` | — |
| the packed tier respects the shared pool | `crates/ferric-integrals/tests/mwe_packed_tier_respects_pool.rs` | — |
| blocked dressing that overshoots the budget is reported | `crates/ferric-integrals/tests/mwe_blocked_dressing_overshoot_always_reported.rs` | — |

No proof script covers this section: it changes where numbers live and the
order they are summed in, not what they are. The bitwise tests above are its
evidence.
