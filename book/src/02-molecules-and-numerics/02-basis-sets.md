# 2.2 Contracted Gaussians, normalization and parsers

<!-- section: 2.2; status: edit; registry: ;
     theory: crates/ferric-integrals/theory.md#"Basis Sets" -->

Molecular orbitals are expanded in atom-centered contracted Gaussian shells.
This section defines a shell, states the normalization the code guarantees,
and follows a basis set from a file or the bundled library to the handle the
integral engine uses.

| Quantity | Symbol | Code name |
|---|---|---|
| Shell angular momentum | $l$ | `Shell::l` |
| Primitive exponents | $\alpha_p$ | `Shell::exponents` (Bohr$^{-2}$) |
| Contraction coefficients | $c_p$ | `Shell::coefficients` |
| Pure or Cartesian | | `Shell::pure` |
| AO functions | $\mu,\nu$ | `0..nbas` |

| Entry point | Location |
|---|---|
| Rust | `ferric-core/src/basis.rs::{BasisSet,Shell,bundled,load_bse_json,load_g94,num_functions}` |
| Bridge to libint2 | `ferric-integrals/src/basis_bridge.rs::PreparedBasis` |
| CLI | `[basis]` section, key `name` (a bundled name), `examples/water-rhf.toml` |
| Python | `ferric.BasisSet.bundled`, `ferric.BasisSet.from_bse_json` |

## 2.2.1 A contracted shell

A shell on atom $A$ with angular momentum $l$ and $K$ primitives is a set of
functions that share exponents and coefficients and differ in the angular part:

$$ \chi_{\mu}(\mathbf r) = Y_{lm}(\mathbf r - \mathbf A)\sum_{p=1}^{K} c_p\,N_p\,|\mathbf r - \mathbf A|^{l} e^{-\alpha_p|\mathbf r - \mathbf A|^{2}} . \tag{2.4} $$

$N_p$ normalizes the primitive. The coefficients $c_p$ in `Shell::coefficients`
are over **unit-normalized primitives**. A pure shell has $2l+1$ functions
(spherical harmonics) and a Cartesian shell $(l+1)(l+2)/2$
(`ferric-core/src/basis.rs::num_functions`). Whether a shell is pure follows
the source: BSE type `gto_spherical` makes $l \ge 2$ pure, and $l < 2$ is
never pure. A Gaussian-94 file is pure for $l \ge 2$.

The figure shows the radial part of the STO-3G hydrogen 1s contraction. It has
no cusp at the nucleus and falls faster than the exact hydrogen function.

![STO-3G hydrogen 1s: the three primitives, their contraction and the exact hydrogen 1s function. Produced by `book/src/figures/02-molecules-and-numerics/contracted_s_shell.py` from the bundled `sto-3g.json`.](../figures/02-molecules-and-numerics/contracted_s_shell.svg)

## 2.2.2 Normalization

Both parsers rescale every contraction so that the contracted function has
unit self-overlap. For unit-normalized primitives of angular momentum $l$ the
overlap of two primitives is $S_{pq} = \big(2\sqrt{\alpha_p\alpha_q}/(\alpha_p+\alpha_q)\big)^{l+3/2}$, and the code divides every $c_p$ by
$\sqrt{S}$ with

$$ S = \sum_{p,q} c_p c_q\,S_{pq} . \tag{2.5} $$

The reason is that two conventions exist in basis-set files. Gaussian-style
files already ship coefficients that give unit overlap. Turbomole-style
files (def2 and others) ship raw coefficients. libint2 renormalizes
contracted shells internally, so its integrals do not care. The grid code
applies only the per-primitive factor $N_p$, so without the rescaling a def2
shell would be wrong on the DFT grid and right in every analytic integral.
After the rescaling libint2's own renormalization changes nothing.

> **Verified equation 2.5** · unit self-overlap
>
> The overlap of two normalized radial Gaussians $r^l e^{-ar^2}$ is
> $\big(2\sqrt{ab}/(a+b)\big)^{l+3/2}$, and a contraction renormalized with
> that overlap matrix has unit self-overlap when integrated numerically,
> independently of the parser's renormalizer and of libint2.
>
> | | |
> |---|---|
> | Proof | `scripts/basis_normalization_sympy.py`, claim 3. Claim 1 is the symbolic integral for $l = 0$ to $6$; claim 2 is quadrature of the closed form, to $10^{-12}$ |
> | Negative control | an unrenormalized contraction is not unit-norm; the runner applies two edits to the script (the exponent $l + \tfrac32$ and the renormalization) and requires a red run |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Rust test | `every_bundled_contraction_has_unit_self_overlap` in `crates/ferric-core/src/basis.rs` checks every bundled shell |
> | Code | `ferric-core/src/basis.rs::parse_bse_json` |
>
> <!-- verified: id=basis.unit-self-overlap; script=scripts/basis_normalization_sympy.py; claim=3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-core/src/basis.rs::parse_bse_json; rust_test=ferric-core/src/basis.rs::every_bundled_contraction_has_unit_self_overlap -->

## 2.2.3 Parsers and the bundled library

| Source | Function | Notes |
|---|---|---|
| Basis Set Exchange JSON | `ferric-core/src/basis.rs::load_bse_json` | a shell with several angular momenta (Pople `SP`) becomes one shell per angular momentum, and a general contraction becomes one shell per coefficient column; ECP blocks are read when present |
| Gaussian-94 text | `ferric-core/src/basis.rs::load_g94` | `SP` is split into an s and a p shell |
| Bundled | `ferric-core/src/basis.rs::bundled` | names and aliases are case-insensitive; an unknown name lists the known ones |

The bundled sets are compiled into the binary as one blob that is inflated
once per process. Which sets exist, with their provenance, is in the user
guide and the validation page. Three conventions matter to a result:

- The Pople 6-31G family is Cartesian (6D), as the Basis Set Exchange types it.
  The 6-311G family is spherical (5D) on every element. This is the one
  deliberate departure from the Basis Set Exchange type, so that the numbers
  match Gaussian's convention. PySCF applies one `cart` flag to a whole
  molecule and defaults to spherical, so a comparison with ferric's
  6-31G-family numbers needs `cart=True` on the PySCF side.
- Element coverage is that of the source set. An element the set lacks is not
  patched: the run stops at basis setup with an error that names the element,
  the basis and the range it covers
  (`ferric-core/src/basis.rs::BasisSet::missing_element_error`).
- A basis set that carries an ECP changes the molecule through
  `Molecule::apply_ecp` (section 2.1) and the integral code (chapter 3).

## 2.2.4 From a basis set to the engine

`ferric-integrals/src/basis_bridge.rs::PreparedBasis::new` takes a molecule
and a basis set and builds the libint2 handle. It

1. refuses a molecule whose ECP the integral code cannot build: an ECP
   couples every shell, and the ECP integrals stop at $l = 4$ and need a
   spherical basis;
2. computes each atom's charge for the nuclear-attraction operator: $0$ for a
   ghost, $Z - n^{\mathrm{core}}$ from the basis set's ECP table, otherwise $Z$;
3. walks the atoms in input order and appends that element's shells, so the
   shell order and the AO order are atom by atom, shell by shell, as in the
   basis set;
4. records the shell dimensions and offsets: AO function $\mu$ of shell $s$ is
   `shell_offsets[s] + m`, and `nbasis` is their sum.

Every later chapter's $\mu,\nu$ index follows this order. The same handle is
shared by all threads, since nothing in it changes after construction.

## Known limits

- Normalization of the angular part of Cartesian $l \ge 2$ functions follows
  libint2's convention. The section states the radial normalization, which
  the code controls.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-core/theory.md`, section 3, and `crates/ferric-integrals/theory.md`, "Basis Sets" | the shell data layout and the contraction form |
| `crates/ferric-core/src/basis.rs` (the renormalizer and its comment) | equation (2.5) and the file conventions |
| `crates/ferric-integrals/src/basis_bridge.rs` | section 2.2.4 |
| Dissertation, chapter 1, section 1.2.1 | the motivation for Gaussian over Slater functions |

Changes from the sources: `theory.md` writes a contracted function without the
spherical-harmonic factor and with conjugates. The book uses real functions
and states the normalization the code applies.
