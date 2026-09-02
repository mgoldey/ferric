# What is validated

> **Implemented ≠ validated.** Working code is not a checked number.

This is the most important page in this documentation, and the one to read
before trusting any result.

## The authority is the wiki

The project wiki's **`VALIDATION.md`** is the authority on what each capability's
*numbers* are checked against, and where they are known to fail. It grades each
capability — **proven / smoke / stub** — rather than presenting them as a flat
list of equals.

This documentation describes what *exists*. It is not a claim that everything
here is equally trustworthy, and the feature list on these pages should not be
read as a validation claim.

## Why the distinction is drawn so sharply

A quantum chemistry code can produce a plausible number in many ways that are
wrong:

- a **non-converged SCF** returned as an ordinary result, because convergence is
  a flag rather than an error
- a method missing a **physical term** it does not mention — TDDFT's XC-kernel
  response is exactly this case, which is why it now warns
- a **fallback model** silently substituted for one atom in a molecule, changing
  a partitioning without changing the shape of the output
- a **screening or truncation threshold** that happens to be safe for the test
  system and not for yours

None of these look like failures. All of them have occurred and been fixed in
this codebase. The remedy is not optimism — it is grading each capability
separately and saying which ones are checked against ground truth.

## Some anchors

Where numbers are checked, they are checked against external references rather
than self-consistency:

| Capability | Anchor |
|---|---|
| CCSD(T) | H2O/cc-pVDZ matches PySCF to ~1e-6 Ha |
| G0W0@HF | matches MOLGW to ~5 meV |
| Gradients | validated against finite differences |
| RI-MP2 extensivity | 2e-12 Ha on a separated dimer |

These are the figures stated in the repository itself. The wiki's
`VALIDATION.md` carries the full per-capability grading, including benchmark
sweeps (GW100 and others) whose numbers are not reproduced here — quoting a
benchmark MAE from memory rather than from the record is exactly the kind of
unchecked claim this page exists to discourage.

## Known negatives

Reported rather than omitted:

- **AO-sparse Laplace SOS-MP2** — truncation radius tracks molecular diameter
  instead of saturating; not a reduced-scaling path
- **Local MP2** — J build still dense-from-RI; no scaling claim made
- **TDHF/RPAx \\( C_6 \\)** — ~60% low regardless of gap
- **TDDFT with a DFT reference** — omits the \\( (ia|f_{xc}|jb) \\) kernel
  response; warns at run time
- **Hessians / frequencies** — not implemented; the mpqc4 libint2 export lacks
  second-derivative integrals

## Testing discipline

Some properties are pinned by tests rather than asserted in prose:

- **Bit-identity across thread counts** for reductions and permutations —
  results do not depend on `RAYON_NUM_THREADS`
- **ERI 8-fold permutational symmetry** — verified against the engine, not
  assumed, since the MP2 code exploits it to compute only ~1/8 of quartets
- **Size-extensivity** and **rotational invariance** of total energies
- **Memory guards in both directions** — a starved budget must be refused *and*
  an ample budget must still run, because an over-estimating guard is also a bug

New guards are **mutation-tested**: a deliberate defect is injected and the test
confirmed to fail before the guard is trusted. This has caught guards that
passed while proving nothing.
