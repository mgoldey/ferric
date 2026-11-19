# Notation and conventions {#notation}

This chapter fixes one notation for the whole book. You can read it once
straight through, or come back to a single section when a symbol in a
chapter looks unfamiliar. Each section states the convention, lists where
the sources disagree, and names the choice the book makes. The sources are
the code, the site pages and the dissertation.

Code is cited by item, as `ferric-mp2/src/rimp2.rs::pair_energy`, as in every other chapter;
the build checks that the item exists and resolves its line (`book/BUILD.md`).
A line inside a function is quoted in an excerpt that the build compares with
the source. Pages of the user guide are cited by file and heading. Paths
starting with `crates/` or `site/` are in the repository. `dissertation.txt`
is a local file outside the repository that does not change; it is cited by
line so a claim can be checked, and is never copied whole.

When a source disagrees with the book, the code is the authority for what is
computed and the book is the authority for how it is written down.

## Index letters {#notation-index-letters}

| Range | Letters | Count symbol |
|---|---|---|
| Active occupied MOs | $i, j, k, l$ | $n_o$ |
| Virtual MOs | $a, b, c, d$ | $n_v$ |
| General MOs | $p, q, r, s$ | $n$ |
| CAS active MOs | $t, u, v, w$ | $n_a$ |
| AO basis functions | $\mu, \nu, \lambda, \sigma$ | $N$ |
| Primitive Gaussians of a shell | $p, q$ with exponent $\alpha_p$ and contraction coefficient $c_p$ | $K$ |
| Auxiliary (fitting) functions | $P, Q, R, S$ | $N_{\mathrm{aux}}$ |
| Atoms | $A, B, C$ | $N_{\mathrm{at}}$ |
| Cartesian components | $x, y, z$ or $\alpha_c, \beta_c$ written out | 3 |
| Grid points | $g$ | $N_g$ |
| Spin labels | $\sigma, \tau \in \{\alpha, \beta\}$ | |
| PDEP eigenpotentials | $\xi, \eta$ | $N_{\mathrm{pdep}}$ |
| Quadrature nodes | $k$ with weight $w_k$ | $n_q$ |
| Electrons | | $N_{\mathrm{el}} = N_\alpha + N_\beta$ |

Atom $A$ carries the atomic number $Z_A$, the number of core electrons
$n^{\mathrm{ECP}}_A$ that an effective core potential replaces, and the
effective charge $Z^{\mathrm{eff}}_A$ that the electrons and the other nuclei
see (0 for a ghost atom, $Z_A - n^{\mathrm{ECP}}_A$ otherwise; section
@sec:molecule-ghost-atoms-effective-charge). A contracted Gaussian is
defined once, in section @sec:basis-sets-contracted-shell. The primitive
labels $p, q$ are local to a shell and never meet an orbital index in one
equation.

"Active occupied" excludes frozen core orbitals. In code the active occupied
block starts at `first_occ = frozen_core`, and virtual orbitals start at
`nocc_total`. From `ferric-mp2/src/rimp2.rs::ri_mp2_spin_components`:

<!-- excerpt: ferric-mp2/src/rimp2.rs::ri_mp2_spin_components -->
```rust
let nocc_total = nelec / 2;                              // all occupied orbitals
let nocc = active_occ(nocc_total, config.frozen_core)?;  // n_o, the active ones
let first_occ = config.frozen_core;                      // first active occupied
let nvir = nbas - nocc_total;                            // n_v, starting at nocc_total
```

The count of active occupied orbitals comes from
`ferric-mp2/src/rimp2.rs::active_occ`, which returns an error when the freeze
covers every occupied orbital.

In a CAS chapter, $i, j$ index the doubly occupied core orbitals and
$t, u, v, w$ the active orbitals (chapter @sec:multireference).

A $\beta$-spin spatial orbital carries an overbar: $\bar\imath, \bar a$.

**Disagreements.**

- Chapter @sec:molecules-and-numerics writes $N_{\mathrm{el}}$ for the
  electron count, never $N$, which is the AO count here.
- The PCM chapter uses $i, j, k$ for tesserae and declares them in its
  header, because no occupied-orbital index appears in its equations.
- The dissertation uses $O, V, N, X$ for the numbers of occupied, virtual,
  AO and auxiliary functions (`dissertation.txt:2867-2868`, Table 4.1 at
  `dissertation.txt:2871-2895`). The book uses $n_o, n_v, N, N_{\mathrm{aux}}$,
  which match the code's `nocc`, `nvir`, `nbas`, `naux`.

Each section that writes indexed equations opens with a table of the index
sets it uses; a chapter overview may give one table for all its sections. A
chapter may add local letters; it may not reuse a letter from the table above
for a different range.

## Spin orbitals and spatial orbitals {#notation-spin-orbitals-spatial-orbitals}

Formal derivations are written in spin orbitals. Working equations are written
in spatial orbitals, with the spin sum carried out and the spin factor shown.
The same letter $i$ means a different range in the two: in a spin-orbital sum
it runs over $2n_o$ spin orbitals, in a spatial sum over $n_o$ orbitals, and the
prefactors change with it (the MP2 energy has $\tfrac14$ in spin orbitals and
no prefactor in spatial orbitals, section @sec:closed-shell-ri-mp2).
Every spin-adapted working equation has a verified-equation box that proves it
equal to the spin-orbital form.

Two spin-orbital orderings exist in the code.

| Ordering | Rule | Used by |
|---|---|---|
| Interleaved | spin orbital $2p + s$, $s = 0$ for $\alpha$ and $1$ for $\beta$ | `ferric-mp2/src/spinorbital.rs::{spin,spat}`, `ferric-mp2/src/mp3.rs::mp3_energy`, the spin-orbital coupled-cluster codes (section @sec:ccd-ccsd-spin-orbital) |
| Blocked | mode $p$ is $p\alpha$, mode $n + p$ is $p\beta$ | CAS-CI determinants (section @sec:determinants-convention) |

The ordering does not change any energy. It changes determinant phases, so
the CAS-CI phase convention of section @sec:determinants-convention holds only
for the blocked ordering. The book states the ordering in the header of each
chapter that writes spin-orbital tensors or determinant phases.

**Disagreements.**

- The dissertation labels its Eq. 4.3 "spin-orbital notation"
  (`dissertation.txt:2847-2857`) and writes it with Mulliken integrals and a
  prefactor of $\tfrac12$. Its Eq. 1.23 (`dissertation.txt:780-792`) writes the
  same energy with antisymmetrized integrals and a prefactor of $\tfrac14$.
  Both are spin-orbital sums. The book uses the Eq. 1.23 form for spin-orbital
  expressions and the spatial form of section @sec:notation-orbital-energy-denominators for working equations.

## Two-electron integrals {#notation-electron-integrals}

**Chemist (Mulliken) notation is the default.**

$$ (pq|rs) = \iint \phi_p(\mathbf r_1)\phi_q(\mathbf r_1)\,\hat w(r_{12})\,\phi_r(\mathbf r_2)\phi_s(\mathbf r_2)\,d\mathbf r_1 d\mathbf r_2 . $$

Orbitals are real, so $(pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)$. The operator
$\hat w$ is the Coulomb operator unless a subscript or an explicit operator is
shown: $(ia|\hat w|jb)$.

**Physicist notation appears only antisymmetrized, in spin orbitals.**

$$ \langle pq\|rs\rangle = \langle pq|rs\rangle - \langle pq|sr\rangle = (pr|qs) - (ps|qr) . $$

The two notations group the orbitals differently. Chemists' notation puts
the two orbitals of electron 1 together on the left, $(pq|\,\cdot\,)$.
Physicists' notation puts the orbitals of electrons 1 and 2 side by side in
the bra, $\langle pq|$, so $\langle pq|rs\rangle = (pr|qs)$. For example,
$(ia|jb) = \langle ij|ab\rangle$, and its exchange partner $(ib|ja)$ is
$\langle ij|ba\rangle$.

The spin-orbital coupled-cluster code uses this convention
(section @sec:ccd-ccsd-spin-orbital), as does the antisymmetrized integral of
section @sec:hamiltonian. The spin integration in
$(pr|qs)$ is implied: the integral vanishes unless $p, r$ share a spin and
$q, s$ share a spin.

The exchange partner of an $ov$-$ov$ integral is written out, $(ib|ja)$.

**Disagreements.**

- The dissertation writes the three-center integral as $(ia|P)$
  (`dissertation.txt:2871-2876`). The code
  stores the auxiliary index first, shape `(naux, nocc*nvir)` (the `b_ov`
  field of `ferric-mp2/src/rimp2.rs::Mp2Intermediates`). The book writes
  $(P|\mu\nu)$ and $(P|ia)$, auxiliary index first, without conjugates.

## Orbital-energy denominators {#notation-orbital-energy-denominators}

The book uses one symbol with one sign:

$$ \Delta_{ij}^{ab} = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j > 0 , \qquad \Delta_i^a = \varepsilon_a - \varepsilon_i > 0 . $$

Every correlation energy is written with an explicit minus sign over a
positive $\Delta$. The closed-shell MP2 energy is

$$ E^{(2)} = -\sum_{ijab} \frac{(ia|jb)\,\big[\,2(ia|jb) - (ib|ja)\,\big]}{\Delta_{ij}^{ab}} . $$

**Disagreements.** The sources differ in the sign of the denominator.

| Source | Definition | Sign |
|---|---|---|
| `dissertation.txt:2852-2857` (Eq. 4.3) | $\varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j$, unnamed | positive |
| Code, `ferric-mp2/src/rimp2.rs::pair_energy` | `denom = e_ij - eps[a] - eps[b]` | negative |
| Code doc, the `kappa` field of `ferric-mp2/src/rimp2.rs::RiMp2Config` | $\Delta = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j$ | positive |

The pair kernel divides by the negative denominator, with
`e_ij` $= \varepsilon_i + \varepsilon_j$:

<!-- excerpt: ferric-mp2/src/rimp2.rs::pair_energy -->
```rust
let denom = e_ij - eps[nocc_total + a] - eps[nocc_total + b];   // -Delta_ij^ab
e_os_ij += g_ab * g_ab / denom;                                 // carries the minus sign
```

The book adopts the positive $\Delta$ and never writes $D_{ij}^{ab}$. $D$ is
reserved for the density matrix. Where the code divides by the negative
`denom`, the chapter says so next to the code reference, as in the sample
section.

## Density matrices and the Fock matrix {#notation-density-matrices-fock-matrix}

The closed-shell AO density is the total density,

$$ D_{\mu\nu} = 2\sum_i C_{\mu i} C_{\nu i}, \qquad \operatorname{tr}(DS) = N_{\mathrm{el}}, \qquad F = h + J[D] - \tfrac12 K[D] . $$

Spin densities are $D^\sigma = \sum_{i \in \sigma} C^\sigma_{\mu i} C^\sigma_{\nu i}$
with $D = D^\alpha + D^\beta$.

This is what the code builds, what the Python surface documents (`AoIntegrals`
in `crates/ferric-python/ferric.pyi`: "`jk()` takes the TOTAL density"), and
what the DFT pages assume (`site/src/methods/dft-derivations.md`, "Closed
shell, total density matrix"). From `ferric-scf/src/rhf.rs::take_orbitals`:

<!-- excerpt: ferric-scf/src/rhf.rs::take_orbitals -->
```rust
let d_new = with_blas_threads(opt_in_blas_threads(), || 2.0 * c_occ.dot(&c_occ.t()));  // D = 2 C_o C_o^T
```

The factor $\tfrac12$ on $K$ comes from the factor 2 in $D$. Exchange acts only
between electrons of the same spin, and each spin carries half the closed-shell
density, $D^\alpha = D^\beta = D/2$. So the exchange term is
$K[D^\alpha] = \tfrac12 K[D]$, while the Coulomb term sees the whole density.

Reduced density matrices of a CAS wavefunction follow section @sec:rdms-1-2-rdm:
$\gamma_{pq} = \langle E_{pq}\rangle$ and
$\Gamma_{pqrs} = \sum_{\sigma\tau}\langle a^\dagger_{p\sigma} a^\dagger_{r\tau} a_{s\tau} a_{q\sigma}\rangle$.

## Orbital rotations {#notation-orbital-rotations}

One convention holds for every orbital-rotation method in the code.

- The generator $\kappa$ is real and antisymmetric, $\kappa^{\mathsf T} = -\kappa$.
- The stored parameters are the lower-triangle elements with the higher
  orbital index first: $\kappa_{ai}$ for occupied-virtual rotations, and
  $\kappa_{pq}$ with $p > q$ in general. $\kappa_{ia} = -\kappa_{ai}$.
- Coefficients rotate on the right: $C \leftarrow C\,U(\kappa)$.
- To first order $U = 1 + \kappa$.
- The gradient is $g_{ai} = +\partial E/\partial\kappa_{ai}$, and a descent
  step has $\kappa_{ai} = -g_{ai}/(\Delta_i^a + \mu)$ with level shift
  $\mu \ge 0$.

The code uses three forms of $U$, which agree through second order in $\kappa$.

| Form | $U(\kappa)$ | Function |
|---|---|---|
| Cayley | $(1 - \tfrac12\kappa)^{-1}(1 + \tfrac12\kappa)$ | `ferric-mp2/src/orbital_rotation.rs::cayley_rotation`; `ferric-scf/src/cayley.rs::cayley_unitary` |
| Exponential | $e^{\kappa}$ by scaling and squaring | `ferric-ci/src/casscf.rs::expm_antisymmetric` |
| Geodesic | $e^{\kappa}$ in closed form from the SVD of the $vo$ block | `ferric-scf/src/aurora.rs::geodesic_rotation` |

The Cayley form is orthogonal to rounding for any antisymmetric $\kappa$ and
differs from $e^{\kappa}$ at third order. Claim 6 of
`scripts/mp2_variant_identities.py` checks both statements, and checks that the
reversed form $(1 + \tfrac12\kappa)^{-1}(1 - \tfrac12\kappa)$ tracks
$e^{-\kappa}$.

The index placement is the same in every caller: `kappa[(a, i)] = v`,
`kappa[(i, a)] = -v` (`ferric-scf/src/cayley.rs::apply_cayley_ov`,
`ferric-scf/src/rohf_newton.rs::rohf_newton_step`,
`ferric-scf/src/cdft_driver.rs::rotate_mos`). The first of them:

<!-- excerpt: ferric-scf/src/cayley.rs::apply_cayley_ov -->
```rust
for (ir, a) in (nocc..n).enumerate() {
    for i in 0..nocc {
        let v = k_ov[(ir, i)];
        kappa[(a, i)] = v;     // kappa_ai, the stored parameter
        kappa[(i, a)] = -v;    // kappa_ia = -kappa_ai
    }
}
```

CASSCF uses $C' = C\,e^{K}$ with $K_{pq} = \kappa_{pq} = -K_{qp}$ for
$p > q$ (module documentation of `crates/ferric-ci/src/casscf.rs`).

**Disagreements.**

- The module documentation of
  `crates/ferric-mp2/src/orbital_rotation.rs` records that the closed-shell
  driver once used the reversed Cayley form. Any derivation older than that
  module is suspect for the sign of $\kappa$ and of the gradient.
- `wiki/sympy_derivations.md:210-211` writes the generator as the operator
  $\hat\kappa = \sum_{ai}\kappa_{ai}(a^\dagger_a a_i - a^\dagger_i a_a)$. That
  matches the matrix convention above.
- The letter $\kappa$ has two other meanings in the sources. It is the
  regularization strength of $\kappa$-MP2, in Ha$^{-1}$
  (the `kappa` field of `ferric-mp2/src/rimp2.rs::RiMp2Config`),
  and it is a common name for the particle-hole kernel of response theory.
  The book keeps $\kappa$ for
  orbital rotations, writes the regularization strength as
  $\kappa_{\mathrm{reg}}$, and writes the particle-hole kernel as
  $\mathbf K$ with $K_{ia,jb} = 2(ia|jb)$.

## Density fitting {#notation-density-fitting}

$$ V_{PQ} = (P|\hat w|Q), \qquad B^P_{pq} = \sum_Q M_{PQ}\,(Q|\hat w|pq), \qquad M^{\mathsf T} M = V^{-1}, \qquad (pq|\hat w|rs) \approx \sum_P B^P_{pq} B^P_{rs} . $$

- $V$ is the **metric**. $(Q|pq)$ is the **raw** three-index integral.
  $B^P_{pq}$ is the **dressed** factor. The word "B tensor" always means the
  dressed factor.
- $M$ is any matrix with $M^{\mathsf T} M = V^{-1}$. Only $B^{\mathsf T}B$
  enters an energy, so the choice of $M$ does not change it. The code uses
  $M = L^{-1}$ from the Cholesky factor $V = LL^{\mathsf T}$
  (`ferric-mp2/src/rimp2.rs::cholesky_inverse_sqrt`) and the
  symmetric $M = U\lambda^{-1/2}U^{\mathsf T}$ with small eigenvalues dropped
  for the erf and terf operators (`ferric-mp2/src/rimp2.rs::eigh_inverse_sqrt`;
  dispatch in `ferric-mp2/src/rimp2.rs::metric_inverse_sqrt`).
- The auxiliary index is the upper index and the first array axis. Blocks are
  named by their orbital spaces: $B_{ov}$, $B_{oo}$, $B_{vv}$. The flattened
  $ov$ block has column index $i\,n_v + a$, so the columns of occupied
  orbital $i$ are one contiguous slice. From
  `ferric-mp2/src/rimp2.rs::spin_components_from_b_ov_kappa_cpu`:

  <!-- excerpt: ferric-mp2/src/rimp2.rs::spin_components_from_b_ov_kappa_cpu -->
  ```rust
  let b_i = b_ov.slice(ndarray::s![.., i * nvir..(i + 1) * nvir]);   // B^P_{ia}, a = 0..n_v
  ```

- The metric and the three-index integrals use the same operator $\hat w$
  unless a section says otherwise
  (`met_op = config.metric_op.unwrap_or(op)` in
  `ferric-mp2/src/rimp2.rs::ri_mp2_spin_components`).

**Disagreements.**

- The dissertation dresses with the symmetric inverse square root and puts the
  auxiliary index last: $B^Q_{ia} = \sum_P (ia|P)(P|Q)^{-1/2}$
  (`dissertation.txt:2871-2876`, Eq. 4.5 at `dissertation.txt:2909-2925`). The
  code's default is the Cholesky factor
  (section @sec:factorization-and-metric). The energies are identical. The book
  writes $M$ and names the factor used where it matters (conditioning, rank
  loss).
- The module documentation of `crates/ferric-scf/src/df_j.rs` calls the
  **raw** tensor $B_{P,\mu\nu} = (P|\mu\nu)$, while that of
  `crates/ferric-scf/src/df_k.rs` calls the **dressed** tensor
  $B[P,\mu,\nu]$. The book writes the raw tensor as
  $(P|\mu\nu)$ and the RI-J fit coefficients as
  $c_P = \sum_Q V^{-1}_{PQ}\,(Q|\mu\nu)D_{\mu\nu}$.

## Attenuated Coulomb operators {#notation-attenuated-coulomb-operators}

| Name | Kernel $\hat w(r)$ | Constructor |
|---|---|---|
| Coulomb | $1/r$ | `ferric-integrals/src/operator.rs::Operator::coulomb` |
| erf (long range) | $\operatorname{erf}(\omega r)/r$ | `ferric-integrals/src/operator.rs::Operator::erf` |
| erfc (short range) | $\operatorname{erfc}(\omega r)/r$ | `ferric-integrals/src/operator.rs::Operator::erfc` |
| terf (long range) | $\operatorname{terf}(r; r_0)/r$ | `ferric-integrals/src/operator.rs::Operator::terf` |
| terfc (short range) | $\operatorname{terfc}(r; r_0)/r$ | `ferric-integrals/src/operator.rs::Operator::terfc` |

$$ \operatorname{terf}(r; r_0, \omega) = \tfrac12\big[\operatorname{erf}\big(\omega(r - r_0)\big) + \operatorname{erf}\big(\omega(r + r_0)\big)\big], \qquad \operatorname{terfc} = 1 - \operatorname{terf} . $$

Each pair sums to the Coulomb operator: $\operatorname{erf} + \operatorname{erfc} = 1$
and $\operatorname{terf} + \operatorname{terfc} = 1$.

**The curvature constraint.** By default the terf sharpness is tied to the
cutoff, $r_0\,\omega = 1/\sqrt2$, so $r_0$ is the only parameter
(`ferric-integrals/src/operator.rs::Operator::terfc`,
`dissertation.txt:1404`).
`ferric-integrals/src/operator.rs::{Operator::terfc_with_omega,Operator::terf_with_omega}`
take $r_0$ and $\omega$ independently. The book writes
$\operatorname{terfc}(r; r_0)$ for the constrained form and
$\operatorname{terfc}(r; r_0, \omega)$ for the free form.

**Short and long range.** $s$ and $l$ denote the short- and long-range parts of
an integral, $(ia|jb) = s + l$, as in `dissertation.txt:2826-2839`. Energies of a method evaluated with one part
carry the operator in brackets: $E_{\mathrm{OS}}[\operatorname{erfc}]$.

**Units of $\omega$ and $r_0$.** Inside the Rust API both are atomic units:
$\omega$ in Bohr$^{-1}$ and $r_0$ in Bohr (the `omega` and `distance` fields
of `ferric-integrals/src/operator.rs::Operator`). The book's equations use
atomic units. The user surfaces do not all agree:

| Surface | Parameter | Unit | Reference |
|---|---|---|---|
| TOML `[mp2] omega`, `terf_omega`, `att_omega`, `mp2v_omega` | $\omega$ | Å$^{-1}$ | `site/src/reference/input.md`, `[mp2]` |
| TOML `[mp2] att_r0` and the other $r_0$ keys | $r_0$ | Å | `ferric-cli/src/config.rs::att_rimp2_operator` |
| TOML `[dft] omega` (`wb97x-l-v`) | $\omega$ | Bohr$^{-1}$ | `site/src/reference/input.md`, `[dft]` |
| TOML `[cell] omega`, `gdf_omega` (Ewald splits) | $\omega$ | inverse of `[cell] unit`, Å$^{-1}$ by default | `site/src/reference/input.md`, `[cell]` |
| Python `run_attenuated_rimp2(omega=)` and the other MP2 entry points | $\omega$ | Å$^{-1}$ | `run_attenuated_rimp2` in `crates/ferric-python/ferric.pyi` |
| Python `AoIntegrals` `operator='erf'/'erfc'` | $\omega$ | Bohr$^{-1}$ | `AoIntegrals` in `crates/ferric-python/ferric.pyi` |
| Python range-separated functional property `omega` | $\omega$ | Bohr$^{-1}$ | `TddftResult.omega` in `crates/ferric-python/ferric.pyi` |
| Dissertation | $\omega$, $r_0$ | Å$^{-1}$, Å | `dissertation.txt:1538`, `:3739` (the text extraction drops the exponent of Å$^{-1}$) |

The conversion is one constant. An inverse length in Å$^{-1}$ times
`BOHR_TO_ANGSTROM` is Bohr$^{-1}$ (`ferric-core/src/units.rs::BOHR_TO_ANGSTROM`;
the alias `ferric-mp2/src/attenuated.rs::BOHR_INV_PER_ANG_INV`).
The default $\omega = 0.420$ Å$^{-1}$ is $0.222254$ Bohr$^{-1}$ (the
`Default` of `ferric-mp2/src/attenuated.rs::AttenuatedMp2Config`).

The book quotes every parameter value in the unit of the surface that takes
it, followed by the atomic-unit value in parentheses: "$\omega = 0.420$
Å$^{-1}$ ($0.2223$ Bohr$^{-1}$)". Equations never contain a unit conversion.

**The letter $\omega$** has three meanings in the sources: the range-separation
parameter, the frequency of a response function, and the Ewald splitting
parameter (the `[cell] omega` key). The range-separated RPA methods need
the first two in the same equations. The book assigns:

| Quantity | Symbol |
|---|---|
| Range-separation parameter | $\omega$ |
| Imaginary frequency | $u$, as in $\chi_0(iu)$ |
| Real excitation energy | $\Omega$ |
| Ewald splitting parameter | $\eta$ |

The range-separation parameter keeps $\omega$ because the CLI key, the Python
keyword and the dissertation all use it.

## Units {#notation-units}

Equations are in Hartree atomic units: energies in Hartree, lengths in Bohr,
gradients in Hartree/Bohr. Real orbitals throughout. One Hartree is about
27.211 eV and 627.509 kcal/mol (`crates/ferric-cli/src/thermo_ir.rs` and
`crates/ferric-mm/src/units.rs` carry the kcal/mol factor).

Lengths change unit at the API boundary. You give a geometry in Å, in an XYZ
file or a Python constructor, and the code stores and computes in Bohr.
`Molecule.coords()` returns Å and `Molecule.coords_bohr()` returns the
internal Bohr values (`crates/ferric-python/ferric.pyi`). A length or inverse
length passed through a TOML key or Python keyword is in the unit the table
below gives for that surface.

| Quantity | Internal | At the TOML and Python surface | Reference |
|---|---|---|---|
| Coordinates | Bohr | Å in XYZ files and Python constructors | `ferric-core/src/mol.rs::Molecule::parse_xyz`, `Molecule` in `crates/ferric-python/ferric.pyi` |
| Bohr radius | 0.52917721092 Å (CODATA 2010) | | `ferric-core/src/units.rs::BOHR_RADIUS_ANGSTROM` |
| Attenuation $\omega$, $r_0$ | Bohr$^{-1}$, Bohr | see section @sec:notation-attenuated-coulomb-operators | |
| $\kappa_{\mathrm{reg}}$ | Ha$^{-1}$ | Ha$^{-1}$ | `ferric-mp2/src/rimp2.rs::RiMp2Config` |
| Lattice vectors, k-points | Bohr, Bohr$^{-1}$ | Å, Å$^{-1}$ (or `unit = "bohr"`) | `site/src/reference/input.md`, `[cell]` |
| Force-field parameters | atomic units | kcal/mol, Å, degrees | `ferric-mm/src/topology.rs::from_amber_units` |
| Excitation and quasiparticle energies | Hartree | eV in printed output | `ferric-gw/src/bse.rs::lowest_ev` |
| Memory budget | bytes | GB (`[memory] budget_gb`) | `site/src/reference/input.md`, `[memory]` |

Benchmark statistics taken from the dissertation are in kcal/mol, as printed
there.

**Disagreements.**

- The Hartree-to-eV factor is written as the literal `27.211_386_245_988`
  (CODATA 2018) in `ferric-gw/src/bse.rs::lowest_ev`, while the length and
  energy constants `ferric-core/src/units.rs::{BOHR_RADIUS_ANGSTROM,HARTREE_JOULE}`
  are CODATA 2010. The difference is below $10^{-7}$ relative. The book states
  which constant a printed eV value uses.
- `[dft] omega` is in Bohr$^{-1}$ and `[mp2] omega` is in Å$^{-1}$
  (`site/src/reference/input.md`, `[dft]` and `[mp2]`). The input reference documents
  the difference. The book repeats the warning wherever both keys appear in
  one example.

## Typography {#notation-typography}

- Matrices and tensors over compound indices are bold when the indices are
  suppressed: $\mathbf A$, $\mathbf B$, $\mathbf K$, $\boldsymbol\chi_0$.
- Operators carry a hat: $\hat w$, $\hat T_2$.
- Amplitudes are $t_{ij}^{ab}$ with occupied indices below. The closed-shell
  antisymmetrized combination is $\tilde t_{ij}^{ab} = 2t_{ij}^{ab} - t_{ij}^{ba}$.
- Spin components: $E_{\mathrm{OS}}$ and $E_{\mathrm{SS}}$, with scaling
  coefficients $c_{\mathrm{OS}}$ and $c_{\mathrm{SS}}$.
- Code names are in monospace and are never translated: `b_ov`, `nocc`,
  `frozen_core`.
- Math uses `$...$` and `$$...$$`. Inside a Markdown table the bar of an
  integral is written `\vert`, because a bare bar ends the table cell.
