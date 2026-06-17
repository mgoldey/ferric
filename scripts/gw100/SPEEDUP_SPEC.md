# GW100 speed — MEASURED breakdown (supersedes the redundancy hypothesis)

## Verdict: the "4× setup redundancy" fix is NOT worth doing (<1% saving).

Measured with FERRIC_TIMING (commit 45dca5f) on CO and H2O at def2-TZVP.
The prior diagnosis (share PDEP+ERI3 across the 4 GW columns, est. "~60%")
was **falsified by measurement** — setup is a rounding error.

## Per-molecule wall time (def2-TZVP)

| stage                | CO        | H2O       | note |
|----------------------|-----------|-----------|------|
| solve_rhf            | 2212 ms   |  581 ms   | already 1×; SCF itself |
| run_gw[G0W0]         | 2646 ms   |  912 ms   | ~150/70 ms setup → REST is the Σc QP solve |
| run_gw[COHSEX]       |  158 ms   |   60 ms   | trivial (static, no freq Σc) |
| run_gw[evGW0]        | 7764 ms   | 2614 ms   | **dominant** — self-consistency iterations |
| run_gw[evGW]         | 7758 ms   | 3646 ms   | **dominant** — self-consistency + W rebuild |

Internal PDEP setup sub-stages (per run_pdep_rpa call), CO:
- rpa_intermediates (ERI3 + 2c metric + MO transform): **16–46 ms**
- eigensolve (Davidson): **7–15 ms**
- freq_quad (λ(iω) + inverse-dielectric matrices): **~95 ms** ← largest setup piece

## Why the hoist doesn't pay

Sharing the setup across G0W0/COHSEX/evGW0 saves ~3 × (setup ≈ 30 ms) ≈ 90 ms
out of a ~18,000 ms per-molecule GW total (CO). That is **0.5%**. The AO ERI3
rebuild I flagged as "9× redundant" is real but each rebuild is ~20 ms — the
redundancy is in the cheapest stage.

## Where the time ACTUALLY is (real optimization targets, by payoff)

1. **evGW0 / evGW self-consistency (~85% of GW time).** Each is ~8 iterations,
   each re-evaluating Σc over the QP range. Levers:
   - Tighter/earlier convergence: ev_conv_thresh=1e-4 with max_ev_iter=8 — does
     it converge in fewer? Profile iteration-count vs Δ. A molecule converging in
     3 iters but running 8 wastes >half its evGW time.
   - evGW0 and evGW share most of their machinery — is anything recomputed across
     the two that could be shared? (They ARE two separate run_gw calls here.)
2. **G0W0 Σc QP solve (~2500 ms CO).** The per-orbital frequency integration +
   Padé. Default qp_range is HOMO±3 (6 orbitals). Is the freq grid (16 pts)
   bigger than needed for the HOMO IP? freq_quad inverse-dielectric (~95 ms × ...)
   recurs — is it rebuilt per orbital or per call? (Earlier static check said per
   call — confirm it isn't per orbital in the Σc loop.)
3. **freq_quad inverse-dielectric (~95 ms/call).** Largest single setup cost.
   eval_inv_dielectric_matrices builds K full M×M inverses. If only the HOMO IP
   is wanted, are all K needed at full M?

## What changed in the driver: NOTHING yet

No code change to gw100_full.rs is justified by this. The instrumentation
(timing.rs, gw_profile.rs) is the deliverable; it redirects future speed work
from a <1% target to the evGW self-consistency loop (the real ~85%).
