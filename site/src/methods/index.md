# Methods overview

What exists, grouped by family. **Maturity varies a great deal** between these.
Check the grade on [Capabilities](../reference/capabilities.md) and the
anchors on [What is validated](../reference/validation.md) before trusting a
number. To go from a task to a method, start with
[Choosing a method](../using/choosing-a-method.md).

| Family | Methods | Page |
|---|---|---|
| SCF and DFT | RHF, UHF, ROHF; KS-DFT (LDA, GGA, hybrid, range-separated, VV10, SCAN/r2SCAN); D3(BJ); IEF-PCM and COSMO solvation; gradients, optimization, finite-difference frequencies, TS search and IRC | [SCF and DFT](./scf.md) |
| MP2 | RI-MP2, attenuated (erfc, terfc), SCS, SCS-MP2(2terfc), MP2-V, RS-MP2 + LR-RPA, OO-RI-MP2, MP3, Laplace MP2 and SOS-MP2, local MP2 | [The MP2 family](./mp2.md) |
| Coupled cluster | CCD, CCSD, CCSD(T), LinLCCD(hh); double hybrids B2PLYP, DSD-PBEP86, ωB97X-L-V | [Coupled cluster](./cc.md) |
| Response | PDEP-RPA, G0W0, COHSEX, evGW0, evGW, BSE-TDA, TDA/TDDFT, polarizabilities and \\( C_6 \\) | [RPA, GW and excited states](./rpa-gw.md) |
| Electron transfer | cDFT, \\( H_{ab} \\) couplings (Rust library only) | [Constrained DFT](./cdft.md) |

QM/MM embedding is on its own page: [QM/MM](../using/qmmm.md).

Formal scaling with system size N, for orientation (these are the textbook
exponents, not measured timings): SCF and DFT N⁴ for exact four-centre
integrals, lower with density fitting and screening; RI-MP2 and its variants
N⁵; CCSD N⁶; (T) N⁷.

## Infrastructure

Shared machinery underneath all of the above:

- **Screening**: Schwarz bounds on every four-centre path, CSB/CSAM
  (Thompson & Ochsenfeld 2017) as options, LinK exchange (Ochsenfeld, White &
  Head-Gordon 1998). QQR (Maurer, Lambrecht & Ochsenfeld 2012) is implemented
  and validated as a bound but not used in production.
- **Spherical and Cartesian** basis support, with BSE-JSON and Gaussian-94
  parsers and a set of bundled orbital, RI, JK and ECP bases.
- **`einsum!`**: a tensor-contraction macro that routes contractions through
  BLAS3 GEMMs, used throughout the CC and MP3 code.
- **Memory budgets**: allocation ceilings that refuse an oversized job with a
  named breakdown instead of letting it be OOM-killed.
- **Python bindings** (pyo3) and a **TOML-driven CLI**. Not every method has
  both; [Capabilities](../reference/capabilities.md) says which.

## Properties

ESP at nuclei and on the molecular surface, electric field, static and
atom-partitioned polarizabilities, Mulliken, Löwdin, Hirshfeld, CHELPG and RESP
charges, density matrices, and NPZ export of ML-ready features.

## Negative results and retractions

Known limits and measured negatives are kept in one place,
[What is validated: known limits](../reference/validation.md#known-limits-and-negatives),
so a correction has one row to change. Two are worth knowing before you pick a
method:

- **TDHF/RPAx \\( C_6 \\)** stays about 60% low regardless of gap. Use that
  kernel for static polarizabilities only.
- **Local MP2** has no scaling claim; the J build is still dense.

One earlier negative has been **retracted**: the AO-sparse Laplace SOS-MP2
truncation radius does *not* track the molecular diameter. That result was an
indexing bug. See [the MP2 page](./mp2.md#laplace-formulations).
