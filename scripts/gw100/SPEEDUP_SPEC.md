# Speed fix: share PDEP+B across the 4 GW columns in gw100_full.rs

## Problem (verified, commit 66aff44 diagnosis)

`crates/ferric-gw/examples/gw100_full.rs:190-197` loops over the 4 GW methods
and calls `run_gw(...)` once each. Every `run_gw` (lib.rs:195) internally:
- `run_pdep_rpa` (lib.rs:210) — Coulomb metric V (2c), Cholesky inv-sqrt, ERI3
  tensor (P|μν), full PDEP Davidson eigensolve.
- `mo_b::build_full_b` (lib.rs:213) — rebuilds V, inv-sqrt, ERI3 again, full MO
  transform.

So the expensive setup runs **4× per molecule**. But G0W0, COHSEX, evGW₀ all use
the SAME W₀ (screened interaction from the same neutral RHF). Only evGW rebuilds
W self-consistently (and it rebuilds internally per iteration regardless).

## Constraint discovered (lib.rs:228-240)

The per-method solvers take `pdep: PdepRpaResult` **by value (move)**:
- `run_g0w0(mol, rhf, &mo_b, &v_dressed, pdep, qp_range, gw_cfg)` (sigma.rs:183)
- `run_cohsex(...)`, `run_evgw0(...)` likewise.
- `run_evgw(mol, obs, dfbs, op, rhf, pdep_cfg, &mo_b, pdep, ...)` — takes pdep0
  by value AND re-runs run_pdep_rpa internally per outer iteration.

So sharing one `pdep`/`mo_b` across 4 dispatches needs `PdepRpaResult: Clone`
(+ `MoB` clone) OR a borrow-based refactor of the solver signatures.

## Recommended approach (lowest risk)

Add a thin public entry point that takes pre-built intermediates, leaving
`run_gw` untouched (back-comp):

```rust
// in ferric-gw/src/lib.rs
pub fn run_gw_with_intermediates(
    mol, obs, dfbs, op, rhf, pdep_cfg, gw_cfg,
    pdep: PdepRpaResult, mo_b: &MoB, v_dressed: &Array2<f64>,
) -> Result<GwResult, FerricError>
```

Then in gw100_full.rs, build PDEP+B ONCE before the method loop and pass clones:

```rust
let pdep0 = run_pdep_rpa(&neutral, &obs_n, &dfbs_n, op, &rhf_n, &pdep_cfg_gw)?;
let mo_b  = mo_b::build_full_b(&neutral, &obs_n, &dfbs_n, op, &rhf_n, 0)?;
let (v_dressed, _) = w_pdep::redress_with_check(&mo_b.v_inv_sqrt, &pdep0.eigenpotentials)?;
for (method, slot) in [...] {
    let gcfg = GwConfig { method, ... };
    let res = run_gw_with_intermediates(..., pdep0.clone(), &mo_b, &v_dressed)?;
    ...
}
```

evGW still rebuilds W internally — that is correct (W self-consistency), leave it.

## Acceptance criteria (MUST verify, don't assume)

1. `cargo build --release --example gw100_full -p ferric-gw` — exit 0.
2. **Numerical identity**: G0W0/COHSEX/evGW0/evGW HOMO IPs for H2O at def2-TZVP
   MUST match the pre-refactor values to <1e-4 eV (the refactor is a pure hoist;
   any change in output = bug). Check against results.json aTZ H2O or a fresh
   pre/post run on one molecule.
3. **Timing**: profile ONE molecule (e.g. H2O or CO at aTZ) pre vs post. The
   estimate is ~halved GW-column time; CONFIRM the actual saving and that setup
   (not the Davidson solve / evGW iterations) was the dominant cost — the 60%
   figure was unmeasured. Report real numbers.
4. `PdepRpaResult` / `MoB` deriving `Clone` must not be prohibitively large; if
   the clone itself is expensive, prefer the borrow refactor instead.

## Scope guard

This touches a benchmark driver + one new public fn. Do NOT change the GW
physics, the per-method solver internals, or evGW's W self-consistency. If the
clone is costly or the borrow refactor balloons, STOP and report — a 2× driver
speedup is not worth destabilizing the validated GW path.
