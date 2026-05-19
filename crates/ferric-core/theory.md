# ferric-core Scientific Conventions & Data Structures

This module implements the foundational structures of the `ferric` engine, handling molecular geometry, atomic elements, and basis set parsing.

---

## 1. Molecular Representation & Coordinate Conventions

### Coordinates (Bohr)
All chemical calculations in `ferric` (integrals, solvers, gradients) are executed using **atomic units (a.u.)**.
* Molecular coordinates are stored internally in **Bohr**.
* The external XYZ file parser expects coordinates in **Angstroms** and performs a conversion to Bohr using the conversion factor:
  $$ 1\,\text{Bohr} \approx 0.5291772109\,\text{Å} $$
* Nuclear repulsion is calculated directly as:
  $$ E_{\text{nuc}} = \sum_{A < B} \frac{Z_A Z_B}{|R_A - R_B|} $$

### Open-Shell vs. Closed-Shell Spin States
Molecular electronic states are defined by:
* **`charge`**: An integer indicating total charge.
* **`multiplicity`**: An integer $2S + 1$, where $S$ is the total spin angular momentum.
From these, the number of electrons is derived, and spin restriction constraints are set:
* **Restricted (RHF)**: Spin-restricted states (even electron count, multiplicity = 1).
* **Unrestricted (UHF)**: Spin-unrestricted states (different spatial orbitals for $\alpha$ and $\beta$ spins).
* **Restricted Open-Shell (ROHF)**: Single set of spatial orbitals with open-shell constraints.

---

## 2. Elements Lookup Table
The `elements` module provides a compile-time lookup table mapping:
* **Atomic Symbol** (e.g., `"H"`, `"He"`, `"Li"`) $\leftrightarrow$ **Atomic Number $Z$** (e.g., `1`, `2`, `3`).
* Atomic mass reference tables.
* Element valency details.

---

## 3. Basis Set Serialization & Parsing

`ferric` parses basis sets into native `Shell` structures. Two parser formats are supported:

### BSE-JSON format
The Basis Set Exchange JSON schema. Bundled basis sets (embedded via `include_str!` at build-time) are compiled in this format under `src/basis/bundled/`.

### Gaussian-94 (G94) format
A standard plain-text format listing contractions for each element. This format is supported by the custom G94 parser for custom external basis sets.

### Data Layout: `Shell` and `BasisSet`
* **`Shell`**: Represents a shell of contracted Gaussian functions.
  * `l`: Angular momentum quantum number ($0$ for $s$, $1$ for $p$, $2$ for $d$, etc.).
  * `pure`: Boolean indicating whether functions are spherical harmonics (5 $d$, 7 $f$) or Cartesian (6 $d$, 10 $f$).
  * `exponents`: Array of primitive exponents $\alpha_i$.
  * `coefficients`: Array of contraction coefficients $d_i$.
* **`BasisSet`**: Maps each atom's element number to a list of matching `Shell` objects.
