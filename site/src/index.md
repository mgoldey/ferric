# ferric

A Rust-native quantum chemistry engine, wrapping libint2 for electron integrals,
with pyo3 Python bindings.

`ferric` is organized around one object: **electronic response** — how the
density reacts to a perturbation. That object shows up as the polarizability
\\( \alpha \\), the dielectric function \\( \varepsilon \\), and the
susceptibility \\( \chi \\), and the methods here are three faces of getting it
right where standard methods get it wrong.

- **[Attenuated MP2](./methods/mp2.md)** — MP2 builds dispersion from an
  *uncoupled* polarizability that over-polarizes, giving too-large \\( C_6 \\)
  and overestimated π-stacking. Attenuating the correlation operator tames that
  response error with a single tunable parameter.
- **[PDEP-RPA / GW](./methods/rpa-gw.md)** — the dielectric matrix *is* the
  density–density response. PDEP keeps only its dominant low-rank eigenmodes, so
  RPA correlation and the GW screened interaction need no explicit sum over
  empty states.
- **[Constrained DFT](./methods/cdft.md)** — a constraint couples to the density
  and reads its response (\\( \partial N / \partial \lambda \\) is a
  susceptibility), building charge-localized diabatic states and their
  electron-transfer couplings.

The motivating claim is that response is **local in real space and low-rank in
its eigenspectrum**, so organizing around it should make the computation
cheaper.

## Implemented ≠ validated

Working code is not a checked number. This documentation describes what exists;
it is not a claim that every number is trustworthy.

For how strongly each capability's numbers are checked against ground truth —
and where they are known to fail — see **[What is validated](./reference/validation.md)**.
Capability maturity varies a great deal between methods, and they are graded
individually rather than presented as a flat list of equals.

## Where to start

| If you want to | Go to |
|---|---|
| Understand the design | [Electronic response](./idea/response.md) |
| Run a calculation | [Quick start](./using/quickstart.md) |
| Build it | [Installation](./using/installation.md) |
| Call it from Python | [Python bindings](./using/python.md) |
| Read the crate docs | [API documentation](./reference/api.md) |
| Know what to trust | [What is validated](./reference/validation.md) |

## Source

[github.com/mgoldey/ferric](https://github.com/mgoldey/ferric) — dual-licensed
MIT / Apache-2.0.
