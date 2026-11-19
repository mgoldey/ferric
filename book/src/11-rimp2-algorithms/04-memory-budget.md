<!-- status: verify; registry: ri-mp2 -->

# The memory budget, the admission checks and the shared pool {#memory-budget}

| Entry point | Location |
|---|---|
| Budget | `ferric-mp2/src/rimp2.rs::eri3_budget_bytes`, `ferric-core/src/memory.rs::resolve_budget_bytes` |
| Admission | `ferric-mp2/src/rimp2.rs::check_mo_side_alloc`, `ferric-mp2/src/rimp2.rs::mo_side_alloc_bytes`, `ferric-core/src/memory.rs::check_alloc` |
| Pool charges | `ferric-mp2/src/rimp2.rs::charge_mo_side` (hard), `ferric-mp2/src/rimp2.rs::charge_mo_side_soft` (soft), `ferric-core/src/memory/pool.rs` |
| Raw-tensor tier | `ferric-integrals/src/three_index_source/jk_storage.rs::memory_tier` |
| CLI | `[memory] budget_gb` (`site/src/reference/input.md`) |
| Trace | `FERRIC_OOC_TRACE=1` logs the raw-tensor tier |

A job that allocates past the memory of the node does not fail cleanly: it
swaps, or the kernel kills it, sometimes with an unrelated job beside it.
RI-MP2 knows its large tensors before it allocates them, so it can decide
up front where each one lives and refuse a run that cannot fit. The budget
bounds the ledgered allocations of a run, not the total process memory.
RI-MP2 uses it three ways: to choose where the raw AO tensor lives, to
refuse a run whose MO-side tensors cannot fit, and to size the transform
chunk (section @sec:streaming-transform-chunk-width).

## The budget {#memory-budget-budget}

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
last digits matter should set `budget_gb` (section @sec:streaming-transform-reproducibility).

## What the run holds {#memory-budget-run-holds}

| Quantity | Bytes | Held during |
|---|---|---|
| $V$, $M$ | $2 \cdot 8 N_{\mathrm{aux}}^2$ | the whole energy |
| raw $(P\vert\mu\nu)$, in core | $8 N_{\mathrm{aux}} N^2$ | the transform; dropped before the pair loop |
| raw $(P\vert\mu\nu)$, packed | $8 N_{\mathrm{aux}} N(N+1)/2$ plus a scratch of up to 64 unpacked rows, capped at 512 MiB | the transform |
| raw $(P\vert\mu\nu)$, spilled | one block, sized to half the budget (the pipeline holds two) | the transform |
| $B_{ov}$ | $8 N_{\mathrm{aux}} n_o n_v$ | the transform and the pair loop |
| $G^{(i)}$ | at most $8 n_o n_v^2$ per active worker | one task |

The packed tier stores only $\mu \le \nu$, since $(P|\mu\nu) = (P|\nu\mu)$.
That is $N(N+1)/2$ of the $N^2$ elements per row, a little over half.

The raw tensor takes the first tier that fits
(`ferric-integrals/src/three_index_source/jk_storage.rs::memory_tier`):

| Tier | Condition |
|---|---|
| In core | $8 N_{\mathrm{aux}} N^2 \le$ budget |
| In core, packed | the packed tensor plus its scratch fit the budget **and** the shared pool's free bytes |
| Spilled to disk | otherwise |

The recompute tier exists for the SCF but is an error on this path.

A concrete case, benzene in cc-pVDZ ($N = 114$, $N_{\mathrm{aux}} = 420$,
$n_o = 21$, $n_v = 93$):

| Quantity | Bytes |
|---|---|
| raw tensor, in core | $8 \cdot 420 \cdot 114^2 = 43.67$ MB |
| raw tensor, packed | 22.02 MB, plus scratch |
| $B_{ov}$ | $8 \cdot 420 \cdot 21 \cdot 93 = 6.56$ MB |
| $G^{(0)}$ per worker | $8 \cdot 21 \cdot 93^2 = 1.45$ MB |

The table lists the pieces. A job has two peaks, with $k$ the chunk width
of section @sec:streaming-transform-chunk-width and $2 \cdot 8N_{\mathrm{aux}}^2$ for $V$ and $M$ held
throughout:

- during the transform, the raw tier $+\ 8N_{\mathrm{aux}}n_on_v + 8kn_on_v$;
- during the pair loop, after `drop(src)`, $8N_{\mathrm{aux}}n_on_v + n_w \cdot 8n_on_v^2$.

For benzene in cc-pVDZ with the raw tensor in core and an ample budget
($k = 256$), the transform peak is $2.82 + 43.67 + 6.56 + 4.00 = 57.05$ MB.
The pair-loop peak is $2.82 + 6.56 + n_w \cdot 1.45$ MB: 10.84 MB on one
worker and 26.82 MB on 12. The per-row half transforms of
section @sec:streaming-transform-memory-pass come on top of the first.

## The admission check {#memory-budget-admission-check}

Before any three-index work, the driver calls `check_mo_side_alloc`. It adds

$$ \underbrace{8 N_{\mathrm{aux}} n_o n_v}_{B_{ov}} \;+\; \underbrace{8\, n_o n_v^2 \cdot n_w}_{G^{(i)},\ n_w \text{ workers}} \;\big[\, +\ 8 N_{\mathrm{aux}} n_v^2 \ (B_{vv}) \,\big] \label{eq:memory-budget-1} $$

and passes the sum to `check_alloc`, which returns an error naming the size,
the budget and the `[memory] budget_gb` / `FERRIC_MEM_BUDGET_GB` knobs when
the sum exceeds the budget. Here is the sum in the code, with the
saturating arithmetic written as plain products and the error label
shortened:

```rust
let n_workers = rayon::current_num_threads().max(1); // n_w
let b_flat = naux * nocc * nvir * 8;                 // B_ov
let g_i = nocc * nvir * nvir * 8 * n_workers;        // G^(0) per worker, n_w of them
let b_vv = if include_b_vv { naux * nvir * nvir * 8 } else { 0 }; // B_vv
ferric_core::memory::check_alloc(&format!(/* ... */), b_flat + g_i + b_vv, budget_bytes)
```

$G^{(i)}$ is allocated inside the parallel loop, so each active worker holds
one. The widest is $G^{(0)}$, so the check charges $n_o n_v^2$ doubles per
worker. For benzene in cc-pVDZ the sum is $6.56 + 12 \cdot 1.45 = 24.0$ MB
on 12 workers and $8.0$ MB on one.

The $B_{vv}$ term is included only by the paths that build it
(`compute_mp2_intermediates_impl`, used by the gradient and polarizability
paths); the energy passes `false`. $B_{vv}$ is $n_v/n_o$ times $B_{ov}$,
so on any basis with more virtuals than occupied orbitals it is the largest
MO-side term. $n_w$ is `rayon::current_num_threads()`.

`mo_side_alloc_bytes` is the same sum without the $G^{(i)}$ term. It is pure,
so callers can report a size without running the check.

For water in cc-pVDZ with cc-pVDZ-RI, $B_{ov}$ is 63 840 bytes and
$G^{(i)}$ is 14 440 bytes per worker, so on a 12-thread pool the sum is
237 120 bytes. A budget of $10^{-5}$ GiB (10 737 bytes) is refused before any
three-index work, and 1 GiB admits (the `589b50832` build). The message
prints both sizes in GB to two decimals, which at this scale reads
"0.00 GB" for each:

```python
# Water/cc-pVDZ, cc-pVDZ-RI: B_ov = 8*84*5*19 = 63 840 bytes, G^(i) = 8*5*19*19 = 14 440 bytes per worker.
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

for gb in (1e-5, 1.0):  # 10 kB is below B_ov alone; 1 GB admits
    try:
        r = ferric.run_rimp2(mol, bs, aux, scf_jk="exact", memory_budget_gb=gb)
        print(f"budget {gb:g} GB: E_corr = {r.mp2_corr:.10f}")
    except Exception as e:  # the check_alloc error names the size, the budget and the knobs
        print(f"budget {gb:g} GB: refused: {e}")
```

```text
budget 1e-05 GB: refused: General error: RI-MP2 MO-side blocks (naux=84, nocc=5, nvir=19, 12 workers) requires 0.00 GB; budget is 0.00 GB — raise [memory] budget_gb / FERRIC_MEM_BUDGET_GB or shrink the system
budget 1 GB: E_corr = -0.2040334567
```

The worker count in the message is `rayon::current_num_threads()`; the run
above used a 12-thread pool, and under `RAYON_NUM_THREADS=1` the same
snippet prints "1 workers".

| Test (`crates/ferric-mp2/tests/mwe_rimp2_has_no_guard.rs`) | What it pins |
|---|---|
| `b_vv_reaches_the_documented_13gb` | the $B_{vv}$ term is counted |
| `g_i_transient_scales_with_worker_count` | the $G^{(i)}$ term scales with $n_w$ |
| `an_honest_floor_exceeds_every_single_buffer` | the sum exceeds each part |
| `ao_side_can_fit_while_the_mo_side_blows_the_budget` | the AO tensor fitting does not admit the MO side |

## The shared pool {#memory-budget-shared-pool}

The pool (`ferric-core/src/memory/pool.rs`) is one process-wide ledger of
bytes. An allocation takes a reservation, which is returned when the guard is
dropped. With no pool installed every reservation is inert and the run is
bit-identical to a run that never touches the pool
(`unbudgeted_rimp2_energy_is_bit_identical_across_repeated_runs` and
`installing_an_ample_pool_does_not_move_rimp2_one_bit` in
`crates/ferric-mp2/tests/mwe_mp2_pool_is_inert_without_a_pool.rs`).

The admission check compares the MO side with the whole budget. The pool
sees what is resident at the same time: the raw tensor, charged by the
source while it lives, and $B_{ov}$ charged on top of it.

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

$$ \big[\,8N_{\mathrm{aux}}N^2,\ 8N_{\mathrm{aux}}N^2 + 8N_{\mathrm{aux}}n_on_v\,\big) = [\,43.67,\ 50.23\,)\ \text{MB} . $$

The left end is the unpacked raw tensor, the width of the band is
$B_{ov}$. At or above 43.67 MB the raw tensor is held in core unpacked and
the source charges all 43.67 MB. Below 50.23 MB the pool then has less than
the 6.56 MB of $B_{ov}$ left, and a hard charge refuses. The recorded sweep,
with the same binary, no pool against a pool of the same size:

| Budget | No pool | Hard charge |
|---|---|---|
| 30 MB | completes | completes (raw tensor spills; the sum fits) |
| 40 MB | completes | completes (raw tensor spills; the sum fits) |
| 44 MB | completes | refused: $B_{ov}$ needs 0.007 GB, 0.000 GB free |
| 48 MB | completes | refused: $B_{ov}$ needs 0.007 GB, 0.002 GB free |
| 51 MB | completes | completes |
| 120 MB | completes | completes |

The band exists because the tier decision does not see the MO side.
Letting it see the MO side is a change in `ferric-integrals` that has not
been made. Until then
`crates/ferric-mp2/tests/mwe_mp2_soft_planes_do_not_refuse.rs` pins that a
budget inside the band completes
(`a_budget_inside_the_refusal_window_still_completes`).

## Known limits (2026-11-15) {#memory-budget-known-limits-2026-11}

- `check_mo_side_alloc` is a hard check that includes $n_w \cdot G^{(i)}$
  (equation @eq:memory-budget-1). Admission of the closed-shell energy therefore depends on
  the rayon worker count, which the doc comment of `charge_mo_side_soft`
  rules out for the pool charge of the same quantity. For benzene in
  cc-pVDZ, a budget of 20 MB is admitted on one worker (8.0 MB) and refused
  on 12 (24.0 MB).
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
| `crates/ferric-mp2/src/rimp2.rs`, `check_mo_side_alloc`, `mo_side_alloc_bytes`, `charge_mo_side`, `charge_mo_side_soft` and the driver comments | equation @eq:memory-budget-1, the excerpt, the charge table, the benzene band |
| `crates/ferric-integrals/src/three_index_source.rs`, `crates/ferric-integrals/src/three_index_source/jk_storage.rs` | the tier table |
| `site/src/reference/input.md`, `[memory]` | the user-facing key |
| the tests named above | the pinned behaviour |

Changes from the sources: sizes are written in bytes with the factor 8
explicit; $n_w$ names the worker count. The benzene sizes, the two peaks,
the packed size and the 20 MB example are computed for this section.
