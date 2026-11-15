<!-- section: 11.4; status: verify; registry: ri-mp2;
     theory: crates/ferric-mp2/theory.md#"2. Closed-shell RI-MP2" -->

# 11.4 The memory budget, the admission checks and the shared pool

| Entry point | Location |
|---|---|
| Budget | `ferric-mp2/src/rimp2.rs::eri3_budget_bytes`, `ferric-core/src/memory.rs::resolve_budget_bytes` |
| Admission | `ferric-mp2/src/rimp2.rs::check_mo_side_alloc`, `ferric-mp2/src/rimp2.rs::mo_side_alloc_bytes`, `ferric-core/src/memory.rs::check_alloc` |
| Pool charges | `ferric-mp2/src/rimp2.rs::charge_mo_side` (hard), `ferric-mp2/src/rimp2.rs::charge_mo_side_soft` (soft), `ferric-core/src/memory/pool.rs` |
| Raw-tensor tier | `ferric-integrals/src/three_index_source/jk_storage.rs::memory_tier` |
| CLI | `[memory] budget_gb` (`site/src/reference/input.md`) |
| Trace | `FERRIC_OOC_TRACE=1` logs the raw-tensor tier |

The budget bounds the ledgered allocations of a run, not the total process
memory. RI-MP2 uses it three ways: to choose where the raw AO tensor lives,
to refuse a run whose MO-side tensors cannot fit, and to size the transform
chunk (section 11.2.3).

## 11.4.1 The budget

The budget is resolved once at the start of the energy and passed down. The
first source that is set wins:

| Order | Source |
|---|---|
| 1 | an explicit value (`[memory] budget_gb`; in Rust `RiMp2Config::memory_budget_bytes`), if positive |
| 2 | `FERRIC_MEM_BUDGET_GB` |
| 3 | the legacy `FERRIC_OOC_BUDGET_GB` and `FERRIC_ERI3_BUDGET_GB`; the smaller wins if both are set |
| 4 | 0.8 times the detected available memory |
| 5 | `DEFAULT_BUDGET_BYTES` = 2 GiB, with a one-time warning |

Source 4 reads the available memory at the time of the call, so the same
input can resolve to a different budget on a busier machine. A run whose
last digits matter should set `budget_gb` (section 11.2.4).

## 11.4.2 What the run holds

| Quantity | Bytes | Held during |
|---|---|---|
| $V$, $M$ | $2 \cdot 8 N_{\mathrm{aux}}^2$ | the whole energy |
| raw $(P\vert\mu\nu)$, in core | $8 N_{\mathrm{aux}} N^2$ | the transform; dropped before the pair loop |
| raw $(P\vert\mu\nu)$, packed | $8 N_{\mathrm{aux}} N(N+1)/2$ plus a scratch of up to 64 unpacked rows, capped at 512 MiB | the transform |
| raw $(P\vert\mu\nu)$, spilled | one block, sized to half the budget (the pipeline holds two) | the transform |
| $B_{ov}$ | $8 N_{\mathrm{aux}} n_o n_v$ | the transform and the pair loop |
| $G^{(i)}$ | at most $8 n_o n_v^2$ per active worker | one task |

The raw tensor takes the first tier that fits
(`ferric-integrals/src/three_index_source/jk_storage.rs::memory_tier`):

| Tier | Condition |
|---|---|
| In core | $8 N_{\mathrm{aux}} N^2 \le$ budget |
| In core, packed | the packed tensor plus its scratch fit the budget **and** the shared pool's free bytes |
| Spilled to disk | otherwise |

The recompute tier exists for the SCF but is an error on this path.

## 11.4.3 The admission check

Before any three-index work, the driver calls `check_mo_side_alloc`. It adds

$$ \underbrace{8 N_{\mathrm{aux}} n_o n_v}_{B_{ov}} \;+\; \underbrace{8\, n_o n_v^2 \cdot n_w}_{G^{(i)},\ n_w \text{ workers}} \;\big[\, +\ 8 N_{\mathrm{aux}} n_v^2 \ (B_{vv}) \,\big] \tag{11.9} $$

and passes the sum to `check_alloc`, which returns an error naming the size,
the budget and the `[memory] budget_gb` / `FERRIC_MEM_BUDGET_GB` knobs when
the sum exceeds the budget. The $B_{vv}$ term is included only by the paths
that build it (`compute_mp2_intermediates_impl`, used by the gradient and polarizability
paths);
the energy passes `false`.
$n_w$ is `rayon::current_num_threads()`.

`mo_side_alloc_bytes` is the same sum without the $G^{(i)}$ term. It is pure,
so callers can report a size without running the check.

| Test (`crates/ferric-mp2/tests/mwe_rimp2_has_no_guard.rs`) | What it pins |
|---|---|
| `b_vv_reaches_the_documented_13gb` | the $B_{vv}$ term is counted |
| `g_i_transient_scales_with_worker_count` | the $G^{(i)}$ term scales with $n_w$ |
| `an_honest_floor_exceeds_every_single_buffer` | the sum exceeds each part |
| `ao_side_can_fit_while_the_mo_side_blows_the_budget` | the AO tensor fitting does not admit the MO side |

## 11.4.4 The shared pool

The pool (`ferric-core/src/memory/pool.rs`) is one process-wide ledger of
bytes. An allocation takes a reservation, which is returned when the guard is
dropped. With no pool installed every reservation is inert and the run is
bit-identical to a run that never touches the pool
(`unbudgeted_rimp2_energy_is_bit_identical_across_repeated_runs` and
`installing_an_ample_pool_does_not_move_rimp2_one_bit` in
`crates/ferric-mp2/tests/mwe_mp2_pool_is_inert_without_a_pool.rs`).

| Charge | Kind | Why |
|---|---|---|
| raw tensor | taken by `ThreeIndexSource` | held while the value lives; released at `drop(src)` |
| $B_{ov}$ in the energy | soft (`charge_mo_side_soft`) | a hard charge refuses a job that completes without the pool, in a narrow band of budgets (below) |
| $G^{(i)}$ per worker | soft | its size reads the worker count; a hard charge would make admission depend on `RAYON_NUM_THREADS` |
| $B_{ov} + B_{oo} + B_{vv}$ for the gradient intermediates | hard (`charge_mo_side`) | no fallback: the tensors are dense and must be allocated |
| canonical MP2, MP3, Laplace dressing, dRPA, local MP2 dense blocks | hard | as above |

A soft charge takes the bytes if the pool has them and otherwise returns an
inert guard. A hard charge refuses with an occupancy breakdown.

**Why $B_{ov}$ is soft.** The comment in the driver records a budget sweep on
benzene/cc-pVDZ ($N = 114$, $N_{\mathrm{aux}} = 420$, $n_o = 21$,
$n_v = 93$). A hard charge refused the run for budgets in
$[\,8N_{\mathrm{aux}}N^2,\ 8N_{\mathrm{aux}}N^2 + 8N_{\mathrm{aux}}n_on_v\,)$,
where the raw tensor is just small enough to be held in core and leaves the
pool nothing for $B_{ov}$. Below that band the raw tensor spills and the run
fits; above it both fit. The proper fix is to let the tier decision see the
MO side, which is a change in `ferric-integrals`.
`crates/ferric-mp2/tests/mwe_mp2_soft_planes_do_not_refuse.rs` pins that a
budget inside the band completes
(`a_budget_inside_the_refusal_window_still_completes`).

## 11.4.5 Known limits (2026-11-15)

- `check_mo_side_alloc` is a hard check that includes $n_w \cdot G^{(i)}$
  (equation 11.9). Admission of the closed-shell energy therefore depends on
  the rayon worker count, which the doc comment of `charge_mo_side_soft`
  rules out for the pool charge of the same quantity.
- The packed-in-core test reads the shared pool's free bytes when a pool is
  installed. The tier sets the block structure of the transform, so with a
  pool installed and a budget between the packed and unpacked sizes, the
  energy's last digits can depend on what else the pool holds. The driver's
  comment says the tier reads the budget and not the pool; that holds for
  the unpacked test only.
- The budget bounds the ledgered tensors. Process memory also holds BLAS
  workspace, the per-row half transforms and allocator overhead.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-core/src/memory.rs`, `resolve_budget_with` and module doc | the precedence table |
| `crates/ferric-mp2/src/rimp2.rs`, `check_mo_side_alloc`, `mo_side_alloc_bytes`, `charge_mo_side`, `charge_mo_side_soft` and the driver comments | equation 11.9, the charge table, the benzene band |
| `crates/ferric-integrals/src/three_index_source.rs`, `crates/ferric-integrals/src/three_index_source/jk_storage.rs` | the tier table |
| `site/src/reference/input.md`, `[memory]` | the user-facing key |
| the tests named above | the pinned behaviour |

Changes from the sources: sizes are written in bytes with the factor 8
explicit; $n_w$ names the worker count.
