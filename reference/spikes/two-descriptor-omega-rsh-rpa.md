# Spike: two-descriptor ω model (λ_max + a3/V) for RSH-RPA C6

Research record, dated 2026-06-04. Provisional. Ported 2026-10-23 from source
commit `e60d17148` (worktree `.claude/worktrees/agent-ac0af625af17b4f10`,
history from before the repository rewrite). That commit changed no files:
the fit was a Python leave-one-out (LOO) fit on dumped descriptors, and
neither the script nor its input table was committed. Everything below is
quoted from the commit messages of `e60d17148` and the two spikes it builds
on (`4aad07d4f`, `37e931e37`, same day). No number here was re-measured.

## Context

The RSH-RPA dispersion lane (`crates/ferric-rpa/tests/rsh_rpa_dispersion.rs`)
predicts a per-molecule range-separation parameter ω from a cheap descriptor
and scores the resulting C6 against DOSD reference values, out of sample
(LOO). `ω_opt` is the ω that reproduces a molecule's DOSD C6. The descriptors:

* `λ_max`: the largest static dielectric eigenvalue (scalar).
* `a1 ≥ a2 ≥ a3`: principal values of the static polarizability tensor α;
  `V` is the molecular volume. `a3/V` is the WEAKEST-direction screening.

The descriptor dump and the one-descriptor LOO-C6 runs are on main as
ignored tests in `rsh_rpa_dispersion.rs`: `directional_descriptor_dump`
(prints `label,a1,a2,a3,V,lmax,omega_opt`), `lambda_max_loo_c6` and
`a3_directional_loo_c6`.

## What came before (same day, quoted)

`4aad07d4f`, correlation of descriptors with ω_opt on the full set:

| descriptor | r_all | r_fit (drop hf, h2) |
|---|---|---|
| lmax (baseline) | +0.72 | +0.72 |
| a_iso/V | -0.55 | -0.73 |
| a1/V (strongest) | -0.37 | -0.31 |
| ‖a‖_F/V | -0.53 | -0.70 |
| (a1 a2 a3)^1/3/V | -0.57 | -0.76 |
| a3/V (weakest direction) | -0.62 | -0.83 |
| aniso (a1-a3)/aiso | +0.42 | +0.68 |

The commit's own caveat: r_fit = -0.83 on 11 points after scanning about 8
descriptors and picking the best is the same overfit setup that produced an
earlier λ_max r = 0.97 mirage.

`37e931e37`, LOO-C6 with ω = a·(a3/V) + b on the fittable 12-molecule set
(hf, h2 dropped as floored): **a3/V LOO-C6 MAE 4.94%**, against fixed-ω
12.4% and the parameter-free (1/3)(λ-2) model at 12% (new molecules only).
a3/V fixed the systems λ_max broke on (hcl +17.6% → -12.4%, h2s +23% →
-3.3%, c6h6 → -1.3%) but made N2 the worst case at +16.8%. The failures were
complementary: λ_max was good on anisotropic molecules (n2, co) and broke on
isotropic ones (hcl, h2s); a3/V the reverse. That suggested a model needing
both directions, which is what this spike tested.

## The spike (`e60d17148`)

Hypothesis: ω = a·λ_max + b·(a3/V) + c fixes the complementary failures.

Method: Python LOO fit of ω on the 11 fittable molecules with a clean a3
value (c6h6 excluded, no clean a3).

| model | LOO-MAE(ω) |
|---|---|
| λ_max | 0.081 |
| a3/V | 0.058 |
| λ_max + a3/V | 0.051 |

Result, as recorded:

* **Hypothesis not confirmed.** Adding λ_max to a3/V did not fix N2, which
  stays the worst case (ω error -0.110). The two-descriptor model is mostly
  a3/V with a small correction; its 0.051 versus 0.058 is marginal and costs
  a free parameter.
* **N2 reinterpreted.** Its ω_opt of 0.464 is the maximum of the set (next:
  co2 0.371, co 0.365). Leaving N2 out removes the only high-ω anchor, so any
  linear model under-predicts it by extrapolation. Its error is largely an
  edge-of-range LOO artifact (too few high-ω triple-bonded molecules), not
  necessarily a descriptor failure.
* **Decision.** The Rust LOO-C6 run for the two-descriptor model was NOT
  done: the ω fit indicated it would match a3/V's 4.94%, and adding
  parameters on 11-12 points is not justified. a3/V alone (4.94% LOO-C6 MAE)
  remains the best single-descriptor result. The indicated next step is a
  larger set with more high-ω multiply-bonded molecules to anchor the top of
  the range, not more descriptors.

## Status and limits

* Verdict: negative for the two-descriptor model, provisional. n = 11, one
  basis, and the comparison rests on LOO-MAE(ω), not on LOO-C6.
* Not reproducible from the repository as it stands: the Python fit and its
  input table were never committed. Reproducing it means running
  `directional_descriptor_dump` (ignored, heavy) and refitting.
* The N2 edge-of-range explanation is an interpretation, not a measurement.
  Testing it needs the larger high-ω set described above.
