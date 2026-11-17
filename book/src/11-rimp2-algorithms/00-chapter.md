<!-- section: 11.0; status: ready; registry: ri-mp2;
     theory: crates/ferric-mp2/theory.md#"1. Density fitting", "2. Closed-shell RI-MP2" -->

# 11. RI-MP2 algorithms and memory

*Notation follows `book/notation.md`. Code references are to commit
`4eca3fd47`.*

Chapter 10 gives the RI-MP2 energy and its variants. This chapter is about
how the code computes it. The equations change little. What changes is the
order of the steps, where each intermediate lives, how many bytes are
resident at the peak, and how the work is split across threads, a GPU and
MPI ranks.

Two numbers drive every choice in the chapter. The energy needs
$\tfrac12 n_o(n_o+1)\,n_v^2 N_{\mathrm{aux}}$ multiply-adds in its last
step, the one step that grows as $N^5$. The largest tensor, the raw AO
integrals $(P|\mu\nu)$, holds $N_{\mathrm{aux}}N^2$ doubles. The algorithm
puts the $N^5$ step into large matrix products, keeps the smaller MO tensor
$B$ in memory, and lets only the raw tensor go to disk. It also fixes the
order of every floating-point sum, so that a run repeated at the same memory
budget gives the same bits.

The algorithm started as the shared-memory Q-Chem implementation in the
dissertation (chapter 4). Section 11.1 sets the two side by side.

| Symbol | Meaning | Code name | Dissertation |
|---|---|---|---|
| $n_o$ | active occupied orbitals | `nocc` | $O$ |
| $n_v$ | virtual orbitals | `nvir` | $V$ |
| $N$ | AO basis functions | `nao`, `nbas` | $N$ |
| $N_{\mathrm{aux}}$ | auxiliary functions | `naux` | $X$ |
| $i, j$ | occupied indices | `0..nocc`, offset by `first_occ` in `eps` | $i, j$ |
| $a, b$ | virtual indices | `0..nvir`, offset by `nocc_total` in `eps` | $a, b$ |
| $P, Q$ | auxiliary indices | `0..naux` | $P, Q$ |

The dissertation uses $n$ for the number of threads; this chapter writes
$n_t$. Costs are counted in multiply-adds unless a table says flops; one
multiply-add is two flops. Sizes are in doubles unless a table says bytes;
one double is 8 bytes.

| Section | Topic |
|---|---|
| 11.1 | The four-step algorithm, 2014 and now |
| 11.2 | Streaming the MO transform and the dressing |
| 11.3 | The pair-energy kernel |
| 11.4 | The memory budget, the admission checks and the shared pool |
| 11.5 | Parallel performance, then and now |
| 11.6 | Device and mixed-precision kernels |
| 11.7 | Distributed RI-MP2 by auxiliary bands |

## Sources

| Source | Used for |
|---|---|
| Dissertation, chapter 4, sections 4.1 to 4.4 | the index names and the original algorithm |
| `book/notation.md` | the symbols |
