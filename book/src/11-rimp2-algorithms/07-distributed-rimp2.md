<!-- section: 11.7; status: verify; registry: ri-mp2;
     theory: none -->

# 11.7 Distributed RI-MP2 by auxiliary bands

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/mpi_rimp2.rs::run_mpi_ri_mp2` (cargo feature `mpi`) |
| Bands | `ferric-core/src/parallel.rs::aux_band`, `ferric-core/src/parallel.rs::aux_band_for` |
| Threads per rank | `ferric-core/src/parallel.rs::threads_per_rank` |
| CLI, Python | none; library only |

One node bounds both the memory for $B$ and the cores for the $O(N^5)$
GEMMs. Splitting the work over several nodes needs an index to split. The
dissertation chose shared memory over MPI because the MPI algorithms of
the time distributed the $B$ matrices "either through duplicated
computational effort or significant internode communication costs"
(chapter 4, section 4.1). The code splits the auxiliary index, which is the
contracted index of equation 11.3: each rank holds a band of $B$, and the
ranks sum partial integrals.

## 11.7.1 The decomposition

`aux_band` splits $[0, N_{\mathrm{aux}})$ into contiguous bands
$\mathcal{P}_r$, one per rank. With $N_{\mathrm{aux}} = q\,n_r + s$,
$0 \le s < n_r$, ranks $r < s$ get $q + 1$ rows and the others $q$. The
bands are disjoint, cover the range and differ in length by at most one; a
band is empty when there are more ranks than auxiliary functions (tests
`aux_bands_partition_disjointly_and_cover`, `aux_bands_balanced_within_one`,
`aux_bands_more_ranks_than_items` in `crates/ferric-core/src/parallel.rs`).

Each rank $r$ builds its band of $B$ with the streamed pass of section 11.2,
restricted to the output rows $\mathcal{P}_r$. Then for every $i$:

$$ G^{(i,r)}_{a,\,jn_v+b} = \sum_{P \in \mathcal{P}_r} B^P_{ia}\,B^P_{jb}, \qquad G^{(i)} = \sum_r G^{(i,r)}, \tag{11.12} $$

$$ E_{\mathrm{OS}} = \sum_r \sum_{i \,:\, i \bmod n_r = r}\; \sum_{j} e^{\mathrm{OS}}_{ij}\big[G^{(i)}\big], \tag{11.13} $$

and the same for $E_{\mathrm{SS}}$. $n_r$ is the number of ranks. The sum over
$r$ in (11.12) is an `MPI_Allreduce` of the whole block, so every rank holds
the complete $G^{(i)}$. Only the rank with $i \bmod n_r = r$ then evaluates
the pair energies for that $i$; a final Allreduce adds the per-rank
$(E_{\mathrm{OS}}, E_{\mathrm{SS}})$.

Why reduce $G^{(i)}$ and not the energies? The pair energy is quadratic in
$G$. Write $G = \sum_r G_r$ for one element; then

$$ G^2 = \Big(\sum_r G_r\Big)^2 = \sum_r G_r^2 + \sum_{r \ne s} G_r G_s , $$

and the cross terms are missing from $\sum_r G_r^2$. The band partials must
be summed before the square. A linear quantity, such as the exchange matrix
of DF-K, can be reduced after each rank's contraction; this one cannot.

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

Here is the loop of `run_mpi_ri_mp2_inner` with the source comments replaced
and the pair loop over $j, a, b$ elided (it is the loop of `pair_energy`
with `jcol` $= j\,n_v$ and every $j$ from 0):

```rust
for i in 0..nocc {
    let b_i_band = b_band.slice(ndarray::s![.., i * nvir..(i + 1) * nvir]); // B_{P,ia}, P in band
    let mut g_i = b_i_band.t().dot(&b_band);       // G^(i,r), full width, (11.12)
    if let Some(world) = &world {
        use mpi::traits::CommunicatorCollectives;
        let mut g_i_global = Array2::<f64>::zeros(g_i.dim());
        world.all_reduce_into(
            g_i.as_slice().unwrap(),
            g_i_global.as_slice_mut().unwrap(),
            mpi::collective::SystemOperation::sum(), // G^(i) = sum_r G^(i,r)
        );
        g_i = g_i_global;
    }
    if i % size == rank {                          // owner of i, (11.13)
        // ... for j in 0..nocc, a, b: e_os, e_ss with weight 1 ...
    }
}
```

## 11.7.2 Cost and memory per rank

| Quantity | Per rank |
|---|---|
| metric $V$, $M$ | full, $N_{\mathrm{aux}}^2$ each, on every rank |
| raw $(P\vert\mu\nu)$ | full tensor, streamed (no distance screen) |
| $B$ band | $\lvert\mathcal{P}_r\rvert\, n_o n_v$ |
| $G^{(i,r)}$ and the reduced $G^{(i)}$ | $2\, n_o n_v^2$ doubles, both held during the Allreduce |
| GEMM for $G^{(i,r)}$ | $2 n_v \cdot n_o n_v \cdot \lvert\mathcal{P}_r\rvert$ flops per $i$, for every $i$ |
| Allreduce | $n_o$ messages of $n_o n_v^2$ doubles |
| pair loop | about $1/n_r$ of the $i$ values |

The $B$ band and the $O(N^5)$ GEMM divide by the rank count. The raw
integrals do not: every rank computes all of them, because every $Q$
contributes to every band through $M_{PQ}$.

**GEMM work.** Sum the per-rank GEMM over $i$ and over ranks:
$\sum_r \sum_i 2 n_o n_v^2 |\mathcal{P}_r| = 2 n_o^2 n_v^2 N_{\mathrm{aux}}$
flops. The serial kernel does $n_o(n_o+1)\,n_v^2 N_{\mathrm{aux}}$
(section 11.3.1). The distributed path therefore does $2n_o/(n_o+1)$ times
the serial GEMM work in total, close to twice, because it forms every $j$,
not the tail. In exchange every owned $i$ costs the same in the pair loop
($n_o n_v^2$ terms), so round-robin ownership is balanced to within one
$i$ per rank.

**Communication against computation.** Each rank takes part in $n_o$
Allreduces of $8 n_o n_v^2$ bytes, $8 n_o^2 n_v^2$ bytes reduced in all
(the bytes on the wire depend on the MPI library's algorithm), and does
$2 n_o^2 n_v^2 N_{\mathrm{aux}}/n_r$ GEMM flops when the bands are equal.
The ratio is

$$ \frac{2 n_o^2 n_v^2 N_{\mathrm{aux}}/n_r}{8 n_o^2 n_v^2} = \frac{N_{\mathrm{aux}}}{4 n_r} \ \text{flops per byte reduced} . $$

It falls as ranks are added. The Allreduce volume, $n_o^2 n_v^2$ doubles,
grows as the fourth power of the size, one power below the GEMM, so a
larger molecule raises the ratio through $N_{\mathrm{aux}}$.

Each rank runs a scoped rayon pool of `threads_per_rank(local_ranks,
physical_cores)` threads, $\max(1, \lfloor \text{physical cores} /
\text{local ranks}\rfloor)$. The floor keeps ranks times threads at or below
the physical cores of the node. The band of $B$ is built with
`stream_dressed_mo_band`, without a budget, so its chunk width is the fixed
256 of section 11.2.3.

## 11.7.3 How it differs from the serial kernel

| | Serial (section 11.3) | Distributed |
|---|---|---|
| $G^{(i)}$ | $j \ge i$ tail | full width, all $j$ |
| Pair weights | $j \ge i$, off-diagonal weight 2 | all $(i, j)$, weight 1 |
| Loop over $i$ | rayon tasks | serial on each rank |
| Three-index screen, `kappa`, `metric_op` | honoured | refused with an error |

With one rank the two paths compute the same sum in a different order, so
the energies agree to rounding, not bit for bit. The module doc of
`crates/ferric-mp2/src/mpi_rimp2.rs` records the measured deviation over
five shapes ($N_{\mathrm{aux}}$ 40 to 912, $n_o$ 3 to 15, $n_v$ 1 to 393)
as at most $4.3 \times 10^{-16}$ relative. The test bar of $10^{-11}$ Ha is
far above that rounding.

## 11.7.4 Known limits (2026-11-15)

- No CLI kind or Python function calls `run_mpi_ri_mp2`; it is reachable from
  Rust with the `mpi` feature only.
- `RiMp2Config::kappa`, `metric_op` and `eri3_screen_thresh` are not
  implemented. `ferric-mp2/src/mpi_rimp2.rs::refuse_unsupported_mpi_config`
  returns an error naming each one that is set
  (`mpi_ri_mp2_refuses_knobs_it_would_ignore` in the same file).
- The cross-rank tests need `mpirun` and are not part of the default
  `cargo test` run.
- The raw three-index integrals are computed in full on every rank.

## Sources

| Source | Used for |
|---|---|
| Dissertation, chapter 4, section 4.1 | the 2014 reason for shared memory |
| `crates/ferric-mp2/src/mpi_rimp2.rs`, module doc, `refuse_unsupported_mpi_config` and `run_mpi_ri_mp2_inner` | equations 11.12 and 11.13, the excerpt, the cost table, the comparison table |
| `crates/ferric-core/src/parallel.rs` | the band rule and the threads per rank |
| `crates/ferric-mp2/tests/mpi_rimp2_banding.rs` | the agreement bars |

Changes from the sources: the code's `g_i_partial` is written
$G^{(i,r)}$; the round-robin owner rule is written $i \bmod n_r = r$. The
total GEMM work, the flops per byte reduced and the per-rank $G$ memory are
derived for this section.
