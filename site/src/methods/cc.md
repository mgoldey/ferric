# Coupled cluster

RI-based coupled cluster, validated against exact-integral and PySCF references.

## What exists

- **RI-CCD** — doubles only
- **RI-CCSD** — singles and doubles
- **(T)** — the perturbative triples correction

H2O/cc-pVDZ CCSD(T) matches PySCF to roughly **1e-6 Ha**.

## CLI coverage is partial

Only `method.kind = "ccsd"` is wired into the CLI — see
`examples/water-ccsd.toml`. **CCD and CCSD(T) are library/Python-only**:

```python
ccd    = ferric.run_ccd(mol, bs, aux)
ccsd_t = ferric.run_ccsd_t(mol, bs, aux)
```

## LinLCCD

**Linearized hole-hole ladder CCD**, closed-shell only. Its main use here is as
the correlation component of **wB97X-L-V** — a double hybrid that converges its
own KS reference and then adds a short-range LinLCCD(hh) correction, rather than
the MP2 correction a conventional double hybrid uses.

The `[dft]` `lambda` and `omega` keys override the published 0.6 / 0.1
Bohr⁻¹ values; omitting them gives the published parameters.

## Implementation

All contractions route through **`einsum!`**, a macro that maps tensor
contractions onto BLAS3 GEMMs. That matters for performance: the alternative —
many small explicit loops — leaves most of the machine's throughput unused.

The permutation copies that feed those GEMMs are parallelized. This is less
trivial than it sounds: for a strided permutation the *copy* can dominate the
contraction it feeds — measured at 47% at nv=40 and **70% at nv=80**, since it
is memory-bandwidth-bound and gets relatively worse with size.

Those copies are **bit-identical regardless of thread count**. A permutation is
pure data movement — every output element written exactly once — so unlike a
reduction there is no summation order to perturb. That property is pinned by a
test that has been verified to fail when deliberately broken.

## Memory

The amplitude tensors dominate, and they grow as \\( n_o^2 n_v^2 \\) — or
\\( (2n_o)^2 (2n_v)^2 \\) in the spin-orbital drivers. Memory budgets are
enforced: an oversized job is refused with a breakdown naming the dominant term
rather than being OOM-killed midway.

The DLPNO family additionally reads its own budget, which it previously ignored.
