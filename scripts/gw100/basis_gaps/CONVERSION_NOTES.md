# GW100 RI-fit aux gaps — fetch results & conversion notes

Date: 2026-06-17. Source: Basis Set Exchange (BSE) v0.1 `complete` JSON API.

## TL;DR

- **Format conversion is the IDENTITY map.** BSE's `format/json` "complete" schema is
  byte-for-byte the schema ferric's `parse_bse_json` already consumes. The bundled files
  in `crates/ferric-core/src/basis/bundled/*.json` ARE raw BSE downloads. No transform
  needed: drop the element block in, ferric renormalizes contractions at load
  (`renormalize_contraction`, see `crates/ferric-core/src/basis.rs`).
- **The blocking problem is NOT format — it is that the requested aux does not exist.**
  `aug-cc-pVDZ-RIFIT` and `aug-cc-pVTZ-RIFIT` were never fit for the alkali metals
  Li(3), Na(11), K(19); and `aug-cc-pVDZ-RIFIT` was never fit for the d-block Sc–Zn(21–30).
  These are upstream library gaps, not ferric bugs.

## Verification that conversion = identity

Carbon (Z=6) pulled from BSE `aug-cc-pVDZ-RIFIT` is numerically identical to ferric's
existing `aug-cc-pvdz-rifit.json` carbon block (22 shells, exact float match). Both have
the same top-level keys and the same per-shell keys
(`function_type, angular_momentum, exponents, coefficients, region`).
Proof file: `bse_adz_rifit_C.json` (diff against the bundled file).

## GAP-by-gap status

### GAP 1 — `aug-cc-pvdz-rifit` missing Li(3), Na(11), K(19), d-block Sc–Zn(21–30)

| element(s) | available on BSE as `aug-cc-pVDZ-RIFIT`? | resolution |
|---|---|---|
| Li, Na, K | **NO** — HTTP 500, element absent from the set entirely | no aug-cc-pVDZ-RIFIT exists; see below |
| Sc–Zn (21–30) | **NO** — absent from aug-cc-pVDZ-RIFIT (and from aug-cc-pVDZ-PP-RIFIT, which only covers Y–Pd/Hf–Pt) | borrow `aug-cc-pVTZ-RIFIT` d-block (provided here) — POLICY DECISION |

### GAP 2 — `aug-cc-pvtz-rifit` missing Li(3), Na(11), K(19)

Same as GAP 1 alkalis: `aug-cc-pVTZ-RIFIT` was never fit for Li/Na/K. The d-block IS
present in aug-cc-pVTZ-RIFIT (ferric already bundles it).

## What the alkali metals actually have

The ONLY correlation/MP2-fitting RI aux on BSE that covers Li/Na/K are the **def2** family
(`def2-{sv(p),svp,tzvp,tzvpp,qzvp,...}-rifit`) and the Coulomb-fit
`def2-universal-jkfit` / `def2-universal-jfit`. ferric ALREADY bundles
`def2-universal-jkfit` (Z 1–86) and `def2-tzvp-rifit` (Z 1–57,72–86, includes Li/Na/K).

There is no aug-cc-pVnZ-named RI aux for the alkalis. You cannot bundle one faithfully.

## Files in this directory

| file | contents | status |
|---|---|---|
| `aug-cc-pvtz-rifit_dblock_21-30.json` | aTZ-RIFIT aux for Sc–Zn, ferric-ready schema | valid, parses, NOT auto-spliced (policy) |
| `def2-tzvp-rifit_alkali_3-11-19.json` | def2-TZVP-RIFIT aux for Li/Na/K, ferric-ready schema | valid, parses, reference only |
| `bse_adz_rifit_C.json` | carbon proof of identity conversion | reference |
| `bse_metadata.json` | full BSE basis-set metadata (coverage scans) | reference |

## Why nothing was auto-spliced into bundled/

Both pending choices are SCIENTIFIC, not mechanical, so they were left for human review per
the "file changes only where clearly safe" instruction:

1. **d-block aDZ molecules (Cu2, CCuN, F4Ti).** Splicing the aTZ-RIFIT d-block into a file
   named `aug-cc-pvdz-rifit.json` silently creates a mixed-zeta aux (aTZ aux under an aDZ
   label). Pairing an aDZ orbital basis with aTZ-level RI aux is common and usually fine
   (RI aux is routinely one cardinal number higher), but it is a methodological call and it
   would make the bundled name misleading. Recommended instead: add a per-element aux
   override in the GW100 runner, OR accept and document the aTZ-aux pairing explicitly.

2. **alkali molecules (Li2, Na2/4/6, K2, FLi, HLi, ClNa, BrK, HK).** These need a
   *different basis family* (def2-rifit) as the aux — there is no aug-cc RI aux for them at
   all. The cleanest fix is to run these molecules with a def2 orbital+aux pair, or use the
   already-bundled `def2-tzvp-rifit` as the aux while keeping the aug-cc-pVnZ orbital basis
   (another cross-family pairing decision). This is NOT a "drop a JSON in" fix.

## If a human decides to splice (mechanical recipe — conversion is identity)

```python
import json
bundled = json.load(open("crates/ferric-core/src/basis/bundled/aug-cc-pvdz-rifit.json"))
addition = json.load(open("scripts/gw100/basis_gaps/aug-cc-pvtz-rifit_dblock_21-30.json"))
bundled["elements"].update(addition["elements"])   # straight merge, no transform
json.dump(bundled, open(".../aug-cc-pvdz-rifit.json","w"), indent=2)
```
ferric's loader renormalizes contractions at load, so raw BSE coefficients are correct as-is.
(But note this would mislabel the basis — see the warning above.)
