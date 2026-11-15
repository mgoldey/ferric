<!-- section: 11.7; status: verify; registry: ri-mp2;
     theory: none -->

# 11.7 Distributed RI-MP2 by auxiliary bands

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/mpi_rimp2.rs::run_mpi_ri_mp2` (cargo feature `mpi`) |
| Bands | `ferric-core/src/parallel.rs::aux_band`, `ferric-core/src/parallel.rs::aux_band_for` |
| Threads per rank | `ferric-core/src/parallel.rs::threads_per_rank` |
| CLI, Python | none; library only |

The dissertation chose shared memory over MPI because the MPI algorithms of
the time distributed the $B$ matrices "either through duplicated
computational effort or significant internode communication costs"
(chapter 4, section 4.1). The code's distributed path is a different design:
it splits the auxiliary index, which is the contracted index of
equation 11.3, so each rank holds a band of $B$ and the ranks sum partial
integrals.

## 11.7.1 The decomposition

`aux_band` splits $[0, N_{\mathrm{aux}})$ into contiguous bands
$\mathcal{P}_r$, one per rank. The bands are disjoint, cover the range and
differ in length by at most one; ranks below the remainder get the longer
band, and a band can be empty when there are more ranks than auxiliary
functions (tests `aux_bands_partition_disjointly_and_cover`,
`aux_bands_balanced_within_one`, `aux_bands_more_ranks_than_items` in
`crates/ferric-core/src/parallel.rs`).

Each rank $r$ builds its band of $B$ with the streamed pass of section 11.2,
restricted to the output rows $\mathcal{P}_r$. Then for every $i$:

$$ G^{(i,r)}_{a,\,jn_v+b} = \sum_{P \in \mathcal{P}_r} B^P_{ia}\,B^P_{jb}, \qquad G^{(i)} = \sum_r G^{(i,r)}, \tag{11.12} $$

$$ E_{\mathrm{OS}} = \sum_r \sum_{i \,:\, i \bmod n_r = r}\; \sum_{j} e^{\mathrm{OS}}_{ij}\big[G^{(i)}\big], \tag{11.13} $$

and the same for $E_{\mathrm{SS}}$. $n_r$ is the number of ranks. The sum over
$r$ in (11.12) is an `MPI_Allreduce` of the whole block, so every rank holds
the complete $G^{(i)}$. Only the rank with $i \bmod n_r = r$ then evaluates
the pair energies for that $i$; a final Allreduce adds the per-rank
$(E_{\mathrm{OS}}, E_{\mathrm{SS}})$.

The order matters: the pair energy is quadratic in $G$, so the band partials
must be summed before the square. $\sum_r (G^{(i,r)})^2 \ne (G^{(i)})^2$.

> **Unverified equation 11.12** · the band decomposition
>
> For a partition of the auxiliary index into bands, the band partials of
> (11.12) sum to the full-width form of $G^{(i)}$ (equation 11.7 with all $j$), and (11.13) with the complete
> $G^{(i)}$ equals the serial energy; squaring the partials before the sum
> does not.
>
> | | |
> |---|---|
> | Proof | none committed. To add: a claim in `scripts/mp2_variant_identities.py` that splits a random $B$ into bands, checks (11.12) and (11.13) against claim 1's energy, with the square-before-sum negative control. |
> | Code tests | `mpi_rimp2_np1_matches_serial_water` in `crates/ferric-mp2/tests/mpi_rimp2_banding.rs`: one rank against `ri_mp2_spin_components`, $E_{\mathrm{corr}}$, $E_{\mathrm{OS}}$ and $E_{\mathrm{SS}}$ each within $10^{-11}$ Ha; `mpi_rimp2_cross_rank_agreement_water` (within $10^{-12}$ Ha across ranks) and its benzene sibling ($10^{-9}$ Ha) under `mpirun` |
>
> <!-- unverified: id=rimp2alg.aux-band-decomposition; todo=add band-split claim to scripts/mp2_variant_identities.py -->

## 11.7.2 Cost and memory per rank

| Quantity | Per rank |
|---|---|
| metric $V$, $M$ | full, $N_{\mathrm{aux}}^2$ each, on every rank |
| raw $(P\vert\mu\nu)$ | full tensor, streamed (no distance screen) |
| $B$ band | $\lvert\mathcal{P}_r\rvert\, n_o n_v$ |
| GEMM for $G^{(i,r)}$ | $2 n_v \cdot n_o n_v \cdot \lvert\mathcal{P}_r\rvert$ per $i$, for every $i$ |
| Allreduce | $n_o$ messages of $n_o n_v^2$ doubles |
| pair loop | about $1/n_r$ of the $i$ values |

The $B$ band and the $O(N^5)$ GEMM divide by the rank count. The raw
integrals do not: every rank computes all of them, because every $Q$
contributes to every band through $M_{PQ}$. The Allreduce volume,
$n_o^2 n_v^2$ doubles in total, grows as the fourth power of the size, one
power below the GEMM.

Each rank runs a scoped rayon pool of `threads_per_rank(local_ranks,
physical_cores)` threads, the physical core count divided by the ranks on the
node, at least one.

## 11.7.3 How it differs from the serial kernel

| | Serial (section 11.3) | Distributed |
|---|---|---|
| $G^{(i)}$ | $j \ge i$ tail | full width, all $j$ |
| Pair weights | $j \ge i$, off-diagonal weight 2 | all $(i, j)$, weight 1 |
| Loop over $i$ | rayon tasks | serial on each rank |
| Three-index screen, `kappa`, `metric_op` | honoured | not read |

With one rank the two paths compute the same sum in a different order, so
the energies agree to rounding, not bit for bit. The test bar of $10^{-11}$
Ha is far above that rounding.

## 11.7.4 Known limits (2026-11-15)

- No CLI kind or Python function calls `run_mpi_ri_mp2`; it is reachable from
  Rust with the `mpi` feature only.
- `RiMp2Config::kappa`, `metric_op` and `eri3_screen_thresh` are not read;
  a run that sets them gets unregularized, unscreened RI-MP2 with the metric
  of its operator, without a warning.
- The cross-rank tests need `mpirun` and are not part of the default
  `cargo test` run.
- The raw three-index integrals are computed in full on every rank.

## Sources

| Source | Used for |
|---|---|
| Dissertation, chapter 4, section 4.1 | the 2014 reason for shared memory |
| `crates/ferric-mp2/src/mpi_rimp2.rs`, module doc and `run_mpi_ri_mp2_inner` | equations 11.12 and 11.13, the cost table, the comparison table |
| `crates/ferric-core/src/parallel.rs` | the band rule and the threads per rank |
| `crates/ferric-mp2/tests/mpi_rimp2_banding.rs` | the agreement bars |

Changes from the sources: the code's `g_i_partial` is written
$G^{(i,r)}$; the round-robin owner rule is written $i \bmod n_r = r$.
