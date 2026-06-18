# GW100 benchmark — ferric trustworthiness package

Entry point to ferric's GW validation on the GW100 test set. The deep validation
(3 evidence layers below) is on an 18-molecule first/second-row core; coverage is
being expanded to the 93 all-electron-tractable molecules (see Coverage).

## TL;DR

ferric's GW is **proven correct**: it reproduces PySCF's G0W0@HF to **4.9 meV**
on identical geometries+basis, and its residual vs experiment is the textbook
HF-starting-point overshoot, not an implementation defect. The @PBE starting
point is also validated (PBE-KS ε_HOMO ≡ PySCF <0.1 meV), giving an @HF/@PBE
bracket on the true IP.

## Coverage (what's runnable)

| set | count | status |
|-----|------:|--------|
| GW100 total | 100 | — |
| ECP-free (Z≤36) | 93 | in `cases()`; @HF/@PBE sweep running (~1/3 done, resumes past slow mols) |
| runnable at aDZ (have RI-aux) | ~80 | bundled aug-cc-pvdz-rifit lacks Li/Na/K/d-block |
| deeply validated (3 layers) | 18 | the core proof |
| ECP molecules (Z≥37) | 7 | I2/Xe/Ag2/C2H3I/AlI3/CI4 RUNNABLE (aug-cc-pVDZ-PP + inline ECP, bundled); G0W0@HF ≡ PySCF gw_ac to 0.1–11 meV (`gw_xcheck_ecp.rs`, `results_ecp.json`). Rb2 BLOCKED (no aug-cc-pVDZ-PP for Rb upstream) → **99/100** |

NOTE: the 7 ECP molecules are I2, **C2H3I** (vinyl iodide, 593-66-8), AlI3
(7784-23-8), **CI4** (carbon tetraiodide, 507-25-5), Xe, Ag2, Rb2 — verified from
setten/GW100 `structures/`. Earlier "CH3I/AgCl" labels in this README were
wrong (those CAS map to C2H3I/Ag2). The GW-through-ECP path is now validated
(spec `docs/superpowers/specs/2026-06-17-gw100-ecp-molecules.md`).

Limits: the aug-cc-pV*Z-RIFIT aux for alkalis was never fit upstream (use
def2-rifit aux; `basis_gaps/CONVERSION_NOTES.md`). ECP support is now built &
verified (`ECP_ASSESSMENT.md` is the original cost analysis; RHF@def2-ECP now
matches PySCF — see crates/ferric-scf/tests/ecp_rhf.rs).

**Sweep status (live):** ~1/3 of 93 attempted per basis. The runner (`run_sweep.py`)
has a per-molecule stall-watchdog that FAILs+skips molecules exceeding a budget
(Na/K clusters, heavy 3rd-row like As2 — these honestly map ferric's
slow-molecule boundary) and RESUMES past them. NOTE: an earlier commit message
("aTZ sweep COMPLETE, 24 conv") was wrong — that run STOPPED at 28/93 due to a
watchdog-resume bug (commit 6a674bb fixed it). Neither basis is complete yet;
`run_sweep.py --show` gives live MAEs. Slow molecules: see `SODIUM_FIX.md`.

| layer | what it proves | result |
|-------|----------------|--------|
| 1. vs experiment | end-to-end accuracy (conflated) | evGW MAE 0.48 eV (aTZ, one-signed +0.47) |
| 2. vs published refs | error is physics, not bug | ferric≡pub GW@HF 0.09 eV; vs CCSD(T) 0.37 (lit 0.33) |
| 3. vs PySCF same-setup | bit-level implementation | **G0W0@HF MAD 4.9 meV (dTZ) / 3.8 meV (aTZ)** |

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

**Layer 4 — @PBE starting point** (`../../crates/ferric-scf/tests/pbe_ks_orbital_energies.rs`).
ferric does TRUE self-consistent PBE-KS (inside `solve_rhf`, xc="pbe"; no separate
solve_rks). Validated vs PySCF dft.RKS on H2O/cc-pVDZ: total E Δ 29 µHa, ε_HOMO
Δ <0.1 meV — GW-grade orbital energies. This is the foundation for an @HF/@PBE
bracket: @HF overshoots, @PBE undershoots, true IP between. The merged
feat/gw-pbe-driver G0W0@PBE matches PySCF gw_ac@PBE <0.1 eV (H2O). Wiring a full
@PBE sweep COLUMN with real PBE-KS orbitals is the open follow-on.

## What is NOT yet proven (open trustworthiness jobs)

- **Coverage**: deep validation is 18 mols; @HF sweep expanding to ~80 (of 93
  ECP-free). The harder of the 80 (Br/Se/As/Kr, radicals) will stress ferric SCF;
  the runner logs failures to map the boundary.
- **@PBE sweep column**: PBE-KS is validated (layer 4), but gw100_full still runs
  @HF only. Wiring the @PBE column (real PBE-KS orbitals) is Task #5 — needs the
  alkali/d-block aux-pairing decision.
- ~~aTZ same-setup xcheck~~ **DONE (2026-06-17): ferric≡PySCF G0W0@HF MAD
  3.84 meV, max 7.90 meV, 17/17** (Koopmans 0.08 meV bit-matched, none >0.3).
  TIGHTER than def2-TZVP (4.93 meV) — the bit-level proof now holds at BOTH the
  def2-TZVP and the production aTZ basis. He excluded (PySCF-side basis-library
  gap at aug-cc-pvtz, not a ferric issue).

Remaining open jobs: coverage extension (heavier/3rd-row atoms), @PBE starting
point (feat/gw-pbe-driver merge), and the measured scaling crossover curve
(SCALING.md, gated on a quiet box).

See also `SCALING.md` — ferric vs PySCF scaling (both ~O(N⁴); ferric's PDEP
truncation is the better-scaling knob, off in this benchmark by design).

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
