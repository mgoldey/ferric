# Notation and conventions {#notation}

This chapter fixes one notation for the whole book. You can read it once
straight through, or come back to a single section when a symbol in a
chapter looks unfamiliar. Each section states the convention, lists where
the sources disagree, and names the choice the book makes. The sources are
the code, the per-crate `theory.md` files, the site pages, the
dissertation, and the SR-MP2 + LR-RPA derivation notes.

File and line references are to commit `81486b984`. Paths starting with
`crates/` or `site/` are in the repository. `dissertation.txt` and
`wiki/papers/…` are local files outside the repository; they are cited by line
so a claim can be checked, and are never copied whole.

When a source disagrees with the book, the code is the authority for what is
computed and the book is the authority for how it is written down. A
disagreement between a `theory.md` file and the code is a defect in the
`theory.md` file; section @sec:notation-disagreements-between-files-code lists them.

## Index letters {#notation-index-letters}

| Range | Letters | Count symbol |
|---|---|---|
| Active occupied MOs | $i, j, k, l$ | $n_o$ |
| Virtual MOs | $a, b, c, d$ | $n_v$ |
| General MOs | $p, q, r, s$ | $n$ |
| CAS active MOs | $t, u, v, w$ | $n_a$ |
| AO basis functions | $\mu, \nu, \lambda, \sigma$ | $N$ |
| Auxiliary (fitting) functions | $P, Q, R, S$ | $N_{\mathrm{aux}}$ |
| Atoms | $A, B, C$ | $N_{\mathrm{at}}$ |
| Cartesian components | $x, y, z$ or $\alpha_c, \beta_c$ written out | 3 |
| Grid points | $g$ | $N_g$ |
| Spin labels | $\sigma, \tau \in \{\alpha, \beta\}$ | |
| PDEP eigenpotentials | $\xi, \eta$ | $N_{\mathrm{pdep}}$ |
| Quadrature nodes | $k$ with weight $w_k$ | $n_q$ |

"Active occupied" excludes frozen core orbitals. In code the active occupied
block starts at `first_occ = frozen_core`, and virtual orbitals start at
`nocc_total` (`crates/ferric-mp2/src/rimp2.rs:1017-1020`). The count of active
occupied orbitals comes from `active_occ` (`crates/ferric-mp2/src/rimp2.rs:98`),
which returns an error when the freeze covers every occupied orbital.

In a CAS chapter, $i, j$ index the doubly occupied core orbitals and
$t, u, v, w$ the active orbitals (`crates/ferric-ci/theory.md:315-316`).

A $\beta$-spin spatial orbital carries an overbar: $\bar\imath, \bar a$.

**Disagreements.**

- `crates/ferric-mp2/theory.md` sections 6.1, 14 and 15 write $\beta$
  orbitals as capitals ($J, B$). Capitals collide with atoms ($A, B$) and auxiliary functions
  ($P, Q$). The book uses overbars.
- `crates/ferric-gw/theory.md:16` uses $\alpha, \beta$ for PDEP modes. Both
  letters also label spin, and $\alpha$ is the
  polarizability. The book uses $\xi, \eta$ for PDEP modes.
- `crates/ferric-gw/theory.md:16` uses $m, n, p, q$ for any active MO. The book
  uses $p, q, r, s$ only.
- `crates/ferric-pcm/theory.md:13-14` uses $i, j, k$ for tesserae. The PCM
  chapter keeps those letters and declares them in its header, because no
  occupied-orbital index appears in its equations.
- The dissertation uses $O, V, N, X$ for the numbers of occupied, virtual,
  AO and auxiliary functions (`dissertation.txt:2867-2868`, Table 4.1 at
  `dissertation.txt:2871-2895`). The book uses $n_o, n_v, N, N_{\mathrm{aux}}$,
  which match the code's `nocc`, `nvir`, `nbas`, `naux`.

Each chapter opens with a table of the index sets it uses. A chapter may add
local letters; it may not reuse a letter from the table above for a different
range.

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
| Interleaved | spin orbital $2p + s$, $s = 0$ for $\alpha$ and $1$ for $\beta$ | `crates/ferric-mp2/src/spinorbital.rs:22`, `crates/ferric-mp2/src/mp3.rs:246-253`, the spin-orbital coupled-cluster codes (`crates/ferric-cc/theory.md:6-8`) |
| Blocked | mode $p$ is $p\alpha$, mode $n + p$ is $p\beta$ | CAS-CI determinants (`crates/ferric-ci/theory.md:11-14`) |

The ordering does not change any energy. It changes determinant phases, so
the CAS-CI phase convention in `crates/ferric-ci/theory.md:16-21` holds only
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
(`crates/ferric-cc/theory.md:9`), and so do the derivation notes
(`wiki/papers/sr-mp2-lr-rpa-derivation.tex:125-131`). The spin integration in
$(pr|qs)$ is implied: the integral vanishes unless $p, r$ share a spin and
$q, s$ share a spin.

The exchange partner of an $ov$-$ov$ integral is written out, $(ib|ja)$.

**Disagreements.**

- `crates/ferric-tddft/theory.md:22-24` writes the same integral as $(ia|jb)$
  in $A$ and $(ia|bj)$ in $B$. The two are equal for real orbitals. The book
  writes $(ia|jb)$ in both and states the reality assumption once.
- `crates/ferric-integrals/theory.md:16` and `:20` write the integrals with
  complex conjugates and the three-center integral as $(P|\mu\nu)$. The
  dissertation writes it as $(ia|P)$ (`dissertation.txt:2871-2876`). The code
  stores the auxiliary index first, shape `(naux, nocc*nvir)`
  (`crates/ferric-mp2/src/rimp2.rs:682`). The book writes $(P|\mu\nu)$ and
  $(P|ia)$, auxiliary index first, without conjugates.

## Orbital-energy denominators {#notation-orbital-energy-denominators}

The book uses one symbol with one sign:

$$ \Delta_{ij}^{ab} = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j > 0 , \qquad \Delta_i^a = \varepsilon_a - \varepsilon_i > 0 . $$

Every correlation energy is written with an explicit minus sign over a
positive $\Delta$. The closed-shell MP2 energy is

$$ E^{(2)} = -\sum_{ijab} \frac{(ia|jb)\,\big[\,2(ia|jb) - (ib|ja)\,\big]}{\Delta_{ij}^{ab}} . $$

**Disagreements.** The same letter $D$ carries opposite signs in two sources.

| Source | Definition | Sign |
|---|---|---|
| `crates/ferric-mp2/theory.md`, Conventions | $\Delta_{ij}^{ab} = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j$ | positive |
| `crates/ferric-cc/theory.md:16-17` | same | negative |
| `wiki/papers/sr-mp2-lr-rpa-derivation.tex:207` | $D_{ij}^{ab} = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j$ | positive |
| `dissertation.txt:2852-2857` (Eq. 4.3) | $\varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j$, unnamed | positive |
| `crates/ferric-quadrature/theory.md:9` | $x = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j$ | positive |
| Code, `crates/ferric-mp2/src/rimp2.rs:1332` | `denom = e_ij - eps[a] - eps[b]` | negative |
| Code doc, `crates/ferric-mp2/src/rimp2.rs:52-53` | $\Delta = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j$ | positive |

The book adopts the positive $\Delta$ and never writes $D_{ij}^{ab}$. $D$ is
reserved for the density matrix. Where the code divides by the negative
`denom`, the chapter says so next to the code reference, as in the sample
section.

## Density matrices and the Fock matrix {#notation-density-matrices-fock-matrix}

The closed-shell AO density is the total density,

$$ D_{\mu\nu} = 2\sum_i C_{\mu i} C_{\nu i}, \qquad \operatorname{tr}(DS) = N_{\mathrm{el}}, \qquad F = h + J[D] - \tfrac12 K[D] . $$

Spin densities are $D^\sigma = \sum_{i \in \sigma} C^\sigma_{\mu i} C^\sigma_{\nu i}$
with $D = D^\alpha + D^\beta$.

This is what the code builds (`crates/ferric-scf/src/rhf.rs:3738`,
`D = 2·C_occ·C_occᵀ`), what the Python surface documents
(`crates/ferric-python/ferric.pyi:567`), and what the DFT pages assume
(`site/src/methods/dft-derivations.md:13-15`).

The factor $\tfrac12$ on $K$ comes from the factor 2 in $D$. Exchange acts only
between electrons of the same spin, and each spin carries half the closed-shell
density, $D^\alpha = D^\beta = D/2$. So the exchange term is
$K[D^\alpha] = \tfrac12 K[D]$, while the Coulomb term sees the whole density.

**Disagreement.** `crates/ferric-scf/theory.md:11-12` defines
$D_{\lambda\sigma} = \sum_i C_{\lambda i} C_{\sigma i}$ without the factor 2
and writes $F = H + D[2J - K]$. The two forms give the same Fock matrix. The
book uses the total density because the code does.

Reduced density matrices of a CAS wavefunction follow
`crates/ferric-ci/theory.md` ("Reduced density matrices"):
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
| Cayley | $(1 - \tfrac12\kappa)^{-1}(1 + \tfrac12\kappa)$ | `cayley_rotation`, `crates/ferric-mp2/src/orbital_rotation.rs:53`; `cayley_unitary`, `crates/ferric-scf/src/cayley.rs:24` |
| Exponential | $e^{\kappa}$ by scaling and squaring | `expm_antisymmetric`, `crates/ferric-ci/src/casscf.rs:855` |
| Geodesic | $e^{\kappa}$ in closed form from the SVD of the $vo$ block | `geodesic_rotation`, `crates/ferric-scf/src/aurora.rs:270` |

The Cayley form is orthogonal to rounding for any antisymmetric $\kappa$ and
differs from $e^{\kappa}$ at third order. Claim 6 of
`scripts/mp2_variant_identities.py` checks both statements, and checks that the
reversed form $(1 + \tfrac12\kappa)^{-1}(1 - \tfrac12\kappa)$ tracks
$e^{-\kappa}$.

The index placement is the same in every caller: `kappa[(a, i)] = v`,
`kappa[(i, a)] = -v` (`crates/ferric-scf/src/cayley.rs:55-61`,
`crates/ferric-scf/src/rohf_newton.rs:219-236`,
`crates/ferric-scf/src/cdft_driver.rs:1371-1380`). CASSCF uses
$C' = C\,e^{K}$ with $K_{pq} = \kappa_{pq} = -K_{qp}$ for $p > q$
(`crates/ferric-ci/src/casscf.rs:21`, `crates/ferric-ci/theory.md:365`).

**Disagreements.**

- The code and the `theory.md` files agree today
  (`crates/ferric-mp2/theory.md` section 7). The module documentation of
  `crates/ferric-mp2/src/orbital_rotation.rs:5-13` records that the closed-shell
  driver once used the reversed Cayley form. Any derivation older than that
  module is suspect for the sign of $\kappa$ and of the gradient.
- `wiki/sympy_derivations.md:210-211` writes the generator as the operator
  $\hat\kappa = \sum_{ai}\kappa_{ai}(a^\dagger_a a_i - a^\dagger_i a_a)$. That
  matches the matrix convention above.
- The letter $\kappa$ has two other meanings in the sources. It is the
  regularization strength of $\kappa$-MP2, in Ha$^{-1}$
  (`crates/ferric-mp2/src/rimp2.rs:50-56`, `crates/ferric-mp2/theory.md` section 2),
  and it is the particle-hole kernel in the derivation notes
  (`wiki/papers/sr-mp2-lr-rpa-derivation.tex:417`). The book keeps $\kappa$ for
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
  (`cholesky_inverse_sqrt`, `crates/ferric-mp2/src/rimp2.rs:2219`) and the
  symmetric $M = U\lambda^{-1/2}U^{\mathsf T}$ with small eigenvalues dropped
  for the erf and terf operators (`eigh_inverse_sqrt`,
  `crates/ferric-mp2/src/rimp2.rs:2264`; dispatch in `metric_inverse_sqrt`,
  `crates/ferric-mp2/src/rimp2.rs:2356`).
- The auxiliary index is the upper index and the first array axis. Blocks are
  named by their orbital spaces: $B_{ov}$, $B_{oo}$, $B_{vv}$. The flattened
  $ov$ block has column index $i\,n_v + a$
  (`crates/ferric-mp2/src/rimp2.rs:1431-1433`).
- The metric and the three-index integrals use the same operator $\hat w$
  unless a section says otherwise
  (`crates/ferric-mp2/src/rimp2.rs:1042`, `met_op = config.metric_op.unwrap_or(op)`).

**Disagreements.**

- The dissertation dresses with the symmetric inverse square root and puts the
  auxiliary index last: $B^Q_{ia} = \sum_P (ia|P)(P|Q)^{-1/2}$
  (`dissertation.txt:2871-2876`, Eq. 4.5 at `dissertation.txt:2909-2925`). The
  code's default is the Cholesky factor
  (`crates/ferric-mp2/theory.md` section 1). The energies are identical. The book
  writes $M$ and names the factor used where it matters (conditioning, rank
  loss).
- `crates/ferric-gw/theory.md:21-25` writes the dressed factor as $\tilde B$
  and calls the dressing matrix $F$. $F$ is the Fock matrix in the book; the
  dressing matrix is $M$ and the dressed factor is $B$.
- `crates/ferric-scf/src/df_j.rs:9` calls the **raw** tensor
  $B_{P,\mu\nu} = (P|\mu\nu)$, while `crates/ferric-scf/src/df_k.rs:27` calls
  the **dressed** tensor $B[P,\mu,\nu]$. The book writes the raw tensor as
  $(P|\mu\nu)$ and the RI-J fit coefficients as
  $c_P = \sum_Q V^{-1}_{PQ}\,(Q|\mu\nu)D_{\mu\nu}$.
- `crates/ferric-tddft/theory.md:22-24` uses $A$ and $B$ for the response
  matrices and `:36`, `:44` use $B_{ov}$ for the RI factor. The book writes the
  response matrices in bold, $\mathbf A$ and $\mathbf B$, and keeps the italic
  $B^P_{ia}$ for the RI factor.

## Attenuated Coulomb operators {#notation-attenuated-coulomb-operators}

| Name | Kernel $\hat w(r)$ | Constructor (`crates/ferric-integrals/src/operator.rs`) |
|---|---|---|
| Coulomb | $1/r$ | `Operator::coulomb`, `:119` |
| erf (long range) | $\operatorname{erf}(\omega r)/r$ | `Operator::erf`, `:124` |
| erfc (short range) | $\operatorname{erfc}(\omega r)/r$ | `Operator::erfc`, `:129` |
| terf (long range) | $\operatorname{terf}(r; r_0)/r$ | `Operator::terf`, `:239` |
| terfc (short range) | $\operatorname{terfc}(r; r_0)/r$ | `Operator::terfc`, `:227` |

$$ \operatorname{terf}(r; r_0, \omega) = \tfrac12\big[\operatorname{erf}\big(\omega(r - r_0)\big) + \operatorname{erf}\big(\omega(r + r_0)\big)\big], \qquad \operatorname{terfc} = 1 - \operatorname{terf} . $$

Each pair sums to the Coulomb operator: $\operatorname{erf} + \operatorname{erfc} = 1$
and $\operatorname{terf} + \operatorname{terfc} = 1$.

**The curvature constraint.** By default the terf sharpness is tied to the
cutoff, $r_0\,\omega = 1/\sqrt2$, so $r_0$ is the only parameter
(`crates/ferric-integrals/src/operator.rs:224-229`,
`dissertation.txt:1404`). `Operator::terfc_with_omega` and
`Operator::terf_with_omega` (`crates/ferric-integrals/src/operator.rs:252`,
`:258`) take $r_0$ and $\omega$ independently. The book writes
$\operatorname{terfc}(r; r_0)$ for the constrained form and
$\operatorname{terfc}(r; r_0, \omega)$ for the free form.

**Short and long range.** $s$ and $l$ denote the short- and long-range parts of
an integral, $(ia|jb) = s + l$, as in `crates/ferric-mp2/theory.md` section 5 and
`dissertation.txt:2826-2839`. Energies of a method evaluated with one part
carry the operator in brackets: $E_{\mathrm{OS}}[\operatorname{erfc}]$.

**Units of $\omega$ and $r_0$.** Inside the Rust API both are atomic units:
$\omega$ in Bohr$^{-1}$ and $r_0$ in Bohr
(`crates/ferric-integrals/src/operator.rs:57-61`). The book's equations use
atomic units. The user surfaces do not all agree:

| Surface | Parameter | Unit | Reference |
|---|---|---|---|
| TOML `[mp2] omega`, `terf_omega`, `att_omega`, `mp2v_omega` | $\omega$ | Å$^{-1}$ | `site/src/reference/input.md:166`, `:176`, `:179`, `:188` |
| TOML `[mp2] att_r0` and the other $r_0$ keys | $r_0$ | Å | `crates/ferric-cli/src/config.rs:1981-1983` |
| TOML `[dft] omega` (`wb97x-l-v`) | $\omega$ | Bohr$^{-1}$ | `site/src/reference/input.md:152` |
| TOML `[cell] omega`, `gdf_omega` (Ewald splits) | $\omega$ | inverse of `[cell] unit`, Å$^{-1}$ by default | `site/src/reference/input.md:632`, `:639-640` |
| Python `run_attenuated_rimp2(omega=)` and the other MP2 entry points | $\omega$ | Å$^{-1}$ | `crates/ferric-python/ferric.pyi:5039` |
| Python `AoIntegrals` `operator='erf'/'erfc'` | $\omega$ | Bohr$^{-1}$ | `crates/ferric-python/ferric.pyi:568` |
| Python range-separated functional property `omega` | $\omega$ | Bohr$^{-1}$ | `crates/ferric-python/ferric.pyi:2685` |
| Dissertation | $\omega$, $r_0$ | Å$^{-1}$, Å | `dissertation.txt:1538`, `:3739` (the text extraction drops the exponent of Å$^{-1}$) |

The conversion is one constant. An inverse length in Å$^{-1}$ times
`BOHR_TO_ANGSTROM` is Bohr$^{-1}$ (`crates/ferric-core/src/units.rs:22-24`;
the alias `BOHR_INV_PER_ANG_INV` at `crates/ferric-mp2/src/attenuated.rs:160`).
The default $\omega = 0.420$ Å$^{-1}$ is $0.222254$ Bohr$^{-1}$
(`crates/ferric-mp2/src/attenuated.rs:165`).

The book quotes every parameter value in the unit of the surface that takes
it, followed by the atomic-unit value in parentheses: "$\omega = 0.420$
Å$^{-1}$ ($0.2223$ Bohr$^{-1}$)". Equations never contain a unit conversion.

**The letter $\omega$** has three meanings in the sources: the range-separation
parameter, the frequency of a response function
(`crates/ferric-rpa/theory.md:9`), and the Ewald splitting parameter
(`crates/ferric-pbc/theory.md:22-23`). The derivation notes use it for the
first two in the same chapter
(`wiki/papers/sr-mp2-lr-rpa-derivation.tex:359`, `:410`). The book assigns:

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
| Coordinates | Bohr | Å in XYZ files and Python constructors | `crates/ferric-core/src/mol.rs:196-198`, `crates/ferric-python/ferric.pyi:180` |
| Bohr radius | 0.52917721092 Å (CODATA 2010) | | `crates/ferric-core/src/units.rs:17` |
| Attenuation $\omega$, $r_0$ | Bohr$^{-1}$, Bohr | see section @sec:notation-attenuated-coulomb-operators | |
| $\kappa_{\mathrm{reg}}$ | Ha$^{-1}$ | Ha$^{-1}$ | `crates/ferric-mp2/src/rimp2.rs:56` |
| Lattice vectors, k-points | Bohr, Bohr$^{-1}$ | Å, Å$^{-1}$ (or `unit = "bohr"`) | `site/src/reference/input.md:632` |
| Force-field parameters | atomic units | kcal/mol, Å, degrees | `crates/ferric-mm/src/topology.rs:302-318` |
| Excitation and quasiparticle energies | Hartree | eV in printed output | `crates/ferric-gw/src/bse.rs:102` |
| Memory budget | bytes | GB (`[memory] budget_gb`) | `site/src/reference/input.md:396` |

Benchmark statistics taken from the dissertation are in kcal/mol, as printed
there.

**Disagreements.**

- The Hartree-to-eV factor is written as the literal `27.211_386_245_988`
  (CODATA 2018) in `crates/ferric-gw/src/bse.rs:102`, while the length and
  energy constants in `crates/ferric-core/src/units.rs:17` and `:27` are
  CODATA 2010. The difference is below $10^{-7}$ relative. The book states
  which constant a printed eV value uses.
- `[dft] omega` is in Bohr$^{-1}$ and `[mp2] omega` is in Å$^{-1}$
  (`site/src/reference/input.md:152`, `:166`). The input reference documents
  the difference. The book repeats the warning wherever both keys appear in
  one example.

## Disagreements between `theory.md` files and the code {#notation-disagreements-between-files-code}

These are defects in a `theory.md` file or a doc comment, found while writing
this chapter. The book does not repeat them. They are listed for the
derivation backfill (branch `docs/derivations-backfill`), which owns those
files.

| File | Statement | Code |
|---|---|---|
| `crates/ferric-scf/theory.md:11-12` | $D = \sum_i C C^{\mathsf T}$, no factor 2 | `crates/ferric-scf/src/rhf.rs:3738` builds $2 C C^{\mathsf T}$ |

Line references drift with every edit to a source file. The book cites
functions by name and resolves the line at build time; see `book/BUILD.md`.

## Typography {#notation-typography}

- Matrices and tensors over compound indices are bold when the indices are
  suppressed: $\mathbf A$, $\mathbf B$, $\mathbf K$, $\boldsymbol\chi_0$.
- Operators carry a hat: $\hat w$, $\hat T_2$.
- Amplitudes are $t_{ij}^{ab}$ with occupied indices below. The closed-shell
  antisymmetrized combination is $\tilde t_{ij}^{ab} = 2t_{ij}^{ab} - t_{ij}^{ba}$
  (written $\tilde T = 2t - t^{\mathrm{swap}}$ in `crates/ferric-mp2/theory.md` section 7).
- Spin components: $E_{\mathrm{OS}}$ and $E_{\mathrm{SS}}$, with scaling
  coefficients $c_{\mathrm{OS}}$ and $c_{\mathrm{SS}}$.
- Code names are in monospace and are never translated: `b_ov`, `nocc`,
  `frozen_core`.
- Math uses `$...$` and `$$...$$`. Inside a Markdown table the bar of an
  integral is written `\vert`, because a bare bar ends the table cell.
