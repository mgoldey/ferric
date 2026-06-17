# GW100 benchmark — ferric trustworthiness package

Entry point to ferric's GW validation on a 18-molecule first/second-row subset
of the GW100 test set. Three independent layers of evidence, weakest→strongest.

## TL;DR

ferric's GW is **proven correct** on this subset: it reproduces PySCF's G0W0@HF
to **4.9 meV** on identical geometries+basis, and its residual vs experiment is
the textbook HF-starting-point overshoot, not an implementation defect.

| layer | what it proves | result |
|-------|----------------|--------|
| 1. vs experiment | end-to-end accuracy (conflated) | evGW MAE 0.48 eV (aTZ, one-signed +0.47) |
| 2. vs published refs | error is physics, not bug | ferric≡pub GW@HF 0.09 eV; vs CCSD(T) 0.37 (lit 0.33) |
| 3. vs PySCF same-setup | bit-level implementation | **G0W0@HF MAD 4.9 meV, 18/18** |

## The three layers

**Layer 1 — experiment** (`run_sweep.py`, `results.json`). evGW vs experimental
vertical IP, two bases. aDZ 0.37 / aTZ 0.48 eV MAE. The aDZ number is *flattered*
by basis-incompleteness ↔ HF-overshoot cancellation; aTZ is the honest figure.
dRPA (0.37) is the most basis-stable method. See `REVIEW_2026-06-17.md`.

**Layer 2 — published GW100 references** (`compare_literature.py`, `lit_ref/`).
Pulls CCSD(T) + GW@HF from the official GW100 DB (github.com/setten/GW100).
Theory-vs-theory separates ferric's error from GW's physical error:
- ferric evGW vs published GW@HF: MAE 0.090 eV (0.032 excl N2/CO).
- ferric vs CCSD(T): 0.366; published GW@HF vs CCSD(T): 0.331 — agree to 35 meV,
  both one-signed → ferric's residual IS the HF starting point.
- ferric's hand-entered exp refs validated vs the DB: mean 0.006 eV.
- Caveat (closed by layer 3): used canonical geoms but ferric's own basis.

**Layer 3 — same-geometry same-basis PySCF** (`xcheck_runner.py`,
`gw_xcheck.rs`, `pyscf_g0w0.py`, `geom/`, `xcheck_results.json`). ferric G0W0@HF
(PDEP-as-W) vs PySCF gw_ac (analytic continuation), IDENTICAL canonical geometry
+ def2-TZVP, all 18:
- **MAD 4.9 meV, max 18.5 meV (H2/He, few-electron Padé-sensitive).**
- Koopmans ≤0.5 meV everywhere → RHF+basis bit-matched; residual is pure G0W0
  discretization (PDEP vs AC).
- N2/CO lit-anchor "outliers" VANISH (−0.6 / +3.0 meV) → they were the
  evGW-vs-scGW self-consistency flavor, not a ferric bug.

## What is NOT yet proven (open trustworthiness jobs)

- **Coverage**: 18/100, all 1st/2nd-row. Heavy/3rd-row atoms (Br/Ge/I) untested
  and will stress ferric's SCF. Next: extend to Li2/Na2/P2/SiH4/PH3/SO2/Cl2.
- **Single starting point**: every number is @HF. `feat/gw-pbe-driver`
  (validated GW@PBE, unmerged) would bracket the overshoot with @PBE undershoot.
- **aTZ same-setup xcheck**: layer 3 is at def2-TZVP; aTZ would confirm the
  basis-trend conclusion bit-for-bit (heavier compute, gated).

## Reproduce

```
# layer 1 (heavy — user launches memory-scoped/gated)
python3 run_sweep.py aug-cc-pvdz          # or aug-cc-pvtz
python3 run_sweep.py --show
# layer 2 (cheap, reads committed lit_ref/)
python3 compare_literature.py
# layer 3 (medium — builds gw_xcheck, runs ferric+PySCF per mol)
cargo build --release --example gw_xcheck -p ferric-gw
python3 xcheck_runner.py def2-tzvp        # idempotent, caches xcheck_results.json
python3 xcheck_runner.py --show
```

## Speed — MEASURED (not estimated)

The earlier "4× setup redundancy" hypothesis was **falsified by measurement**
(FERRIC_TIMING, commit 45dca5f). Sharing PDEP/ERI3 setup across the 4 GW columns
saves <1% — setup is ~20–95 ms vs an ~18 s/molecule GW total. The real cost is
the **evGW0/evGW self-consistency loops (~85% of GW time)**, then the G0W0 Σc QP
solve. Full breakdown + the real optimization targets: `SPEEDUP_SPEC.md`.
Profile any molecule with: `FERRIC_TIMING=1 ./target/release/examples/gw_profile
geom/CO.xyz def2-tzvp def2-tzvp-rifit`.
