# Methods overview

What exists, grouped by family. **Maturity varies a great deal** between these —
see [What is validated](../reference/validation.md) before trusting any
particular number.

| Family | Methods | Page |
|---|---|---|
| SCF | RHF, UHF, ROHF, KS-DFT, gradients | [SCF and DFT](./scf.md) |
| MP2 | RI, attenuated, SCS, OO, Laplace, MP3, LMP2 | [The MP2 family](./mp2.md) |
| Coupled cluster | CCD, CCSD, (T), LinLCCD | [Coupled cluster](./cc.md) |
| Response | PDEP-RPA, G0W0, COHSEX, evGW, TDDFT | [RPA and GW](./rpa-gw.md) |
| Electron transfer | cDFT, \\( H_{ab} \\) couplings | [Constrained DFT](./cdft.md) |

## Infrastructure

Shared machinery underneath all of the above:

- **QQR screening** (Maurer, Lambrecht & Ochsenfeld 2012) and **LinK exchange**
  (Ochsenfeld, White & Head-Gordon 1998) for the Fock build
- **Spherical and Cartesian** basis support, with BSE-JSON and Gaussian-94
  parsers
- **`einsum!`** — a tensor-contraction macro routing contractions through BLAS3
  GEMMs, used throughout the CC and MP3 code
- **Memory budgets** — enforced allocation ceilings that refuse an oversized job
  with a named breakdown rather than letting it be OOM-killed
- **Python bindings** (pyo3) and a **TOML-driven CLI**

## Properties

ESP at nuclei, electric field, static and atom-partitioned polarizabilities,
Hirshfeld and Löwdin charges, density matrices, and NPZ export of ML-ready
features for downstream generative-model conditioning.

## A note on negative results

Several reduced-scaling approaches in this codebase are implemented and
**measured negative** — they are documented as such rather than omitted:

- **AO-sparse Laplace SOS-MP2** — truncation radius tracks the molecular
  diameter instead of saturating
- **Local MP2** — the J build is still dense-from-RI, so no scaling claim is made
- **TDHF/RPAx \\( C_6 \\)** — stays ~60% low regardless of gap

Knowing which ideas did *not* work is part of the documentation.
