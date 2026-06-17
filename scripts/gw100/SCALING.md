# ferric vs PySCF G0W0 — scaling analysis

Answers "what has better scaling?" Derived from the algorithms (code-grounded),
cross-checked against measured timings. NOT a measured scaling curve yet — see
"open job" below.

## Both are asymptotically O(N⁴), dominated by the same step

Both ferric and PySCF gw_ac are RI-density-fitted G0W0; both are dominated by
building the RI dielectric / screened interaction W — an N_aux·n_ov contraction
≈ **O(N⁴)** (N_aux ~ N, n_ov ~ N²). Neither has an asymptotic edge in the
standard (untruncated) regime; that's expected for two RI-G0W0 codes.

### ferric cost structure (from the code)

| stage | operation | scaling |
|-------|-----------|---------|
| RI intermediates | ERI3 + MO transform → b_ov (N_aux × n_ov) | O(N⁴) |
| Davidson eigensolve | dielectric_apply = 2 GEMMs (M×N_aux)(N_aux×n_ov), ×~fixed iters | ~O(N⁴) |
| eval_inv_dielectric | K full M×M inverses | O(K·M³)~O(N³) |
| Σc loop (sigma_c_at_z) | n_qp·K·M² contraction with W̃ | O(N²) (n_qp,K const) |

### Measured-timing check (def2-TZVP, from gw_profile)

run_gw[G0W0]: H2O 912 ms, CO 2646 ms → ratio 2.90×. Basis dim ~43→56.
- N³ predicts 2.21×, **N⁴ predicts 2.88×**, N⁵ predicts 3.75×.
- Measured 2.90× ≈ N⁴ → ferric is N⁴-dominated, NOT N⁵ (Davidson converges in
  ~fixed iters; it does not blow the eigensolve to N⁵).
- Caveat: 2-point fit — rules out N⁵, does not pin the exponent precisely.

## Where the algorithms differ — ferric's structural advantage

| | ferric (PDEP-as-W) | PySCF gw_ac |
|-|--------------------|-------------|
| W representation | M PDEP eigenmodes, M ≤ N_aux, **truncatable** | full N_aux dielectric on a freq grid |
| Σc cost | n_qp·K·M² — **shrinks as M truncates** | full-N_aux grid contraction |
| tunable knob | **trunc_thresh** drops low-weight modes | none equivalent |

The PDEP eigenpotential representation is ferric's scaling lever: the trust-map
size series (this session) showed ~75% of modes are droppable at the 1e-4 default
with zero observable cost, and that fraction GROWS with system size. So the
W-dependent terms move from O(N_aux²) toward O(M²) with M sub-linear — the regime
where eigenpotential GW (Govoni–Galli WEST lineage) is designed to win.

## Verdict

- **Small molecules (this benchmark, ≤18 atoms):** same O(N⁴) class, no
  advantage — and we run UNTRUNCATED (trunc_thresh=0) for apples-to-apples, so
  ferric carries full M and is at parity with / slightly behind PySCF's mature
  vectorized AC.
- **Larger systems:** ferric's PDEP truncation is the better-scaling knob, but
  it is OFF in this benchmark by design and only proven to pay off beyond GW100
  small-molecule scale.

## Open job (gated on a quiet box)

Turn this analysis into a PROVEN crossover: clean single-thread timing curve
over a size series (H2O → C2H6 → benzene), ferric trunc on/off vs PySCF, median
of 3 runs each. Cannot run under the current SR-MP2 grid load — queue gated.
