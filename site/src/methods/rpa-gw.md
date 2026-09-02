# RPA and GW

The part of the [response](../idea/response.md) claim that is **actually
demonstrated**.

## PDEP

The dielectric matrix *is* the density–density response function. Conventional
RPA and GW evaluate it through an explicit sum over empty orbital states —
expensive, and slow to converge with basis size.

**PDEP** — projective dielectric eigenpotentials — instead builds a low-rank
basis from the dominant eigenmodes of the dielectric matrix. The spectrum decays
quickly, so a modest number of modes captures the physics and the empty-state
sum disappears.

This low-rank compression works and is used in production paths. It is the
demonstrated half of the design premise; the real-space locality half is not
(see [Electronic response](../idea/response.md)).

## RPA

- **PDEP-RPA** — RPA correlation via a low-rank W basis in Gaussians
- **U-PDEP-RPA** — open-shell, over a spin-summed dielectric
- **Attenuated RPA** — short-range correlation via erfc

The solver defaults to **Lanczos**; a dense path is used for small problems.
Note that the eigensolve is **serial by design** — that is a deliberate choice,
not an oversight, and RPA here is already faster than the PySCF reference.

## GW

- **G0W0**, **COHSEX**, **evGW0**, **evGW**
- **U-GW** — unrestricted

G0W0@HF matches MOLGW to roughly **5 meV**.

The quasiparticle solve runs a Newton root-find on the self-energy, which is
fragile near \\( \Sigma_c \\) poles. That fragility has a consequence worth
recording: the frequency-quadrature loop inside the self-energy is a
**sequential floating-point accumulation**, so it cannot be parallelized without
changing summation order — and reordering would perturb quasiparticle energies
in a thread-count-dependent way. The loop is deliberately left serial.

## TDDFT

Linear response in both the **Tamm–Dancoff approximation** (TDA/CIS) and the
full **Casida** equations, closed-shell references.

**Important limitation.** The \\( (ia|f_{xc}|jb) \\) XC-kernel response is
**not implemented**. With a pure Hartree–Fock reference (\\( c_{HF} = 1 \\))
that term is identically zero and the result is exactly CIS/TDHF. With any DFT
reference it is not zero, and the excitation energies omit it — they are
approximate.

The code warns on stderr when \\( c_{HF} \neq 1 \\) rather than returning
silently incomplete numbers.

## Double hybrids

**B2PLYP** and **DSD-PBEP86**, plus **wB97X-L-V** — see
[Coupled cluster](./cc.md).

## Dispersion and polarizability

Static and atom-partitioned polarizabilities, Casimir–Polder \\( C_6 \\)
coefficients, and many-body dispersion.

**TDHF/RPAx is a measured negative** for dispersion: the static α it produces is
reasonable, but the \\( C_6 \\) stays roughly **60% low regardless of gap**. It
is a polarizability tool, not a dispersion one.

Dynamic dRPA@PBE α, by contrast, gives \\( C_6 \\) roughly 3× better than the
static Tkatchenko–Scheffler (TS) model.
