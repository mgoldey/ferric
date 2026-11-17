# Contracted Gaussians, normalization and parsers {#basis-sets}

<!-- status: edit; registry: ;
     theory: crates/ferric-integrals/theory.md#"Conventions" -->

Molecular orbitals are expanded in functions centered on the atoms. The
physically natural choice would be Slater functions, $e^{-\zeta r}$, which
have the cusp at the nucleus and the exponential tail of a real atomic
orbital. Their multi-center two-electron integrals have no closed form. Gaussians,
$e^{-\alpha r^2}$, have neither the cusp nor the right tail, but the product
of two Gaussians on different centers is a Gaussian on a third center, and
the integral algorithms of Chapter @sec:integrals build on that. Ferric uses Gaussians and
fixes their shape by summing several of them with fixed coefficients: a
contracted shell.

This section defines a shell, derives the normalization the code
guarantees, and follows a basis set from a file or the bundled library to
the handle the integral engine uses.

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

## A contracted shell {#basis-sets-contracted-shell}

A shell on atom $A$ with angular momentum $l$ and $K$ primitives is a set of
functions that share exponents and coefficients and differ in the angular part:

$$ \chi_{\mu}(\mathbf r) = Y_{lm}(\mathbf r - \mathbf A)\sum_{p=1}^{K} c_p\,N_p\,|\mathbf r - \mathbf A|^{l} e^{-\alpha_p|\mathbf r - \mathbf A|^{2}} . \label{eq:basis-sets-1} $$

Read it from the right. Each primitive is a Gaussian of width set by
$\alpha_p$: a large $\alpha_p$ is tight around the nucleus, a small one is
diffuse. The factor $|\mathbf r - \mathbf A|^l$ gives the function the right
behavior near the nucleus for angular momentum $l$. $N_p$ normalizes the
primitive by itself. The coefficients $c_p$ mix the primitives into one
radial shape, and $Y_{lm}$ supplies the angular part, one function per $m$.

The coefficients $c_p$ in `Shell::coefficients` are over **unit-normalized
primitives**: the $N_p$ is not folded into them. A pure shell has $2l+1$
functions (real spherical harmonics) and a Cartesian shell $(l+1)(l+2)/2$
(all monomials $x^{l_x}y^{l_y}z^{l_z}$ with $l_x+l_y+l_z = l$). The two counts
agree for $l = 0$ and $1$ and first differ at $l = 2$: 5 against 6. The six
Cartesian d functions span the five pure ones plus the combination
$x^2+y^2+z^2$, which has s symmetry. The count
comes from `ferric-core/src/basis.rs::num_functions`:

```rust
pub fn num_functions(l: i32, pure: bool) -> usize {
    if pure {
        (2 * l + 1) as usize                // m = -l..l
    } else {
        ((l + 1) * (l + 2) / 2) as usize    // l_x + l_y + l_z = l
    }
}
```

Whether a shell is pure follows the source: BSE type `gto_spherical` makes
$l \ge 2$ pure, and $l < 2$ is never pure. A Gaussian-94 file is pure for
$l \ge 2$.

The figure shows the radial part of the STO-3G hydrogen 1s contraction. It has
no cusp at the nucleus and falls faster than the exact hydrogen function.
Three Gaussians are a compromise: more primitives would bring the contraction
closer to the exponential, at the cost of more primitive integrals per
contracted one.

![STO-3G hydrogen 1s: the three primitives, their contraction and the exact hydrogen 1s function. Produced by `book/src/figures/02-molecules-and-numerics/contracted_s_shell.py` from the bundled `sto-3g.json`.](../figures/02-molecules-and-numerics/contracted_s_shell.svg)

## Normalization {#basis-sets-normalization}

Basis-set files do not agree on what their coefficients mean. Gaussian-style
files ship coefficients that already give the contracted function unit
self-overlap. Turbomole-style files (def2 and others) ship raw coefficients
and expect the reader to normalize. If ferric used both as given, a def2
function would not have unit norm, and the error would show up in only some
of the code (see "Why the code rescales" below). So both parsers rescale
every contraction to unit self-overlap when they read it.

To rescale you need the self-overlap of a contraction. Derive it in three
steps.

**Step 1: the radial integral.** The angular part is a normalized spherical
harmonic, so it integrates to 1 and leaves the radial integral with the
$r^2\,dr$ volume element. For two primitives with exponents $a = \alpha_p$
and $b = \alpha_q$, substitute $u = (a+b)r^2$:

$$ \int_0^\infty r^{2l+2}\, e^{-(a+b)r^2}\,dr = \frac{\Gamma(l+\tfrac32)}{2\,(a+b)^{l+3/2}} . $$

**Step 2: normalize.** The primitive normalization is
$N_p = \big[\Gamma(l+\tfrac32)/(2(2\alpha_p)^{l+3/2})\big]^{-1/2}$, the integral
above with $b = a$. Multiply the integral by $N_pN_q$. The gamma functions
cancel, and

$$ S_{pq} = \frac{(2\alpha_p)^{(l+3/2)/2}\,(2\alpha_q)^{(l+3/2)/2}}{(\alpha_p+\alpha_q)^{l+3/2}} = \left(\frac{2\sqrt{\alpha_p\alpha_q}}{\alpha_p+\alpha_q}\right)^{l+3/2} . $$

Check two limits. For $\alpha_p = \alpha_q$ the ratio is 1, as it must be for
a normalized function with itself. For $\alpha_p \ne \alpha_q$,
$2\sqrt{\alpha_p\alpha_q} < \alpha_p+\alpha_q$ because
$(\sqrt{\alpha_p}-\sqrt{\alpha_q})^2 > 0$, so $S_{pq} < 1$: two different
normalized Gaussians overlap by less than one.

**Step 3: sum over the contraction.** The contracted radial function is
$\sum_p c_p N_p r^l e^{-\alpha_p r^2}$. Its self-overlap is the double sum over
primitive pairs, and the code divides every $c_p$ by $\sqrt{S}$ with

$$ S = \sum_{p,q} c_p c_q\,S_{pq} . \label{eq:basis-unit-self-overlap} $$

After the division the self-overlap is
$\sum_{pq} (c_p/\sqrt S)(c_q/\sqrt S)\,S_{pq} = S/S = 1$. For a single
primitive $S = c_1^2$, and the rescaled coefficient is $c_1/|c_1| = \pm 1$.

The code is (@eq:basis-unit-self-overlap) as a double loop, in
`ferric-core/src/basis.rs::renormalize_contraction`:

```rust
let lf = l as f64;
let mut s = 0.0_f64;
for (a, ca) in exps.iter().zip(coefs.iter()) {
    for (b, cb) in exps.iter().zip(coefs.iter()) {
        let prim_ov = (2.0 * (a * b).sqrt() / (a + b)).powf(lf + 1.5);  // S_pq
        s += ca * cb * prim_ov;                                          // c_p c_q S_pq
    }
}
if s > 0.0 {
    let scale = 1.0 / s.sqrt();                                          // 1 / sqrt(S)
    for c in coefs.iter_mut() {
        *c *= scale;
    }
}
```

Both parsers call it once per contraction, which for a general contraction
means once per coefficient column. The `s > 0.0` guard leaves a contraction
of zero norm alone instead of dividing by zero.

### Why the code rescales

libint2 renormalizes contracted shells internally, so its integrals give the
same answer with or without the rescaling. The grid code that evaluates
basis functions for DFT does not. It applies only the per-primitive factor
$N_p$ (`ferric-integrals/theory.md`, "Basis functions on a grid"). Without
the rescaling a def2 shell would be right in every analytic integral and
wrong on the DFT grid. After the rescaling libint2's own renormalization changes
nothing, since $S$ is already 1.

> **Verified equation @eq:basis-unit-self-overlap** · unit self-overlap
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

The proof covers the radial normalization. The angular factor of a Cartesian
$l \ge 2$ function is libint2's convention and is not part of it (see Known
limits).

## Parsers and the bundled library {#basis-sets-parsers-bundled-library}

| Source | Function | Notes |
|---|---|---|
| Basis Set Exchange JSON | `ferric-core/src/basis.rs::load_bse_json` | a shell with several angular momenta (Pople `SP`) becomes one shell per angular momentum, and a general contraction becomes one shell per coefficient column; ECP blocks are read when present |
| Gaussian-94 text | `ferric-core/src/basis.rs::load_g94` | `SP` is split into an s and a p shell |
| Bundled | `ferric-core/src/basis.rs::bundled` | names and aliases are case-insensitive; an unknown name lists the known ones |

Why split an `SP` shell? Pople sets share one set of exponents between the s
and p functions of a valence shell, with a separate coefficient column for
each. Ferric's `Shell` has one $l$, so the parser emits two shells with the
same exponents, and each column is renormalized with its own $l$ in (@eq:basis-unit-self-overlap).

The bundled sets are compiled into the binary as one blob that is inflated
once per process. Which sets exist, with their provenance, is in the user
guide and the validation page. Three conventions matter to a result:

- The Pople 6-31G family is Cartesian (6D), as the Basis Set Exchange types it.
  The 6-311G family is spherical (5D) on every element. It is the one
  deliberate departure from the Basis Set Exchange type, so that the numbers
  match Gaussian's convention. PySCF applies one `cart` flag to a whole
  molecule and defaults to spherical, so a comparison with ferric's
  6-31G-family numbers needs `cart=True` on the PySCF side. The two choices
  span different spaces, since 6D carries the extra s-type function
  $x^2+y^2+z^2$ on each d shell. For water in 6-31G* the two energies are
  about 1.4 mHa apart (`ferric-core/src/basis.rs::bundled`, documentation).
- Element coverage is that of the source set. An element the set lacks is not
  patched: the run stops at basis setup with an error that names the element,
  the basis and the range it covers
  (`ferric-core/src/basis.rs::BasisSet::missing_element_error`).
- A basis set that carries an ECP changes the molecule through
  `Molecule::apply_ecp` (section @sec:molecule) and the integral code (chapter @sec:integrals).

## From a basis set to the engine {#basis-sets-basis-set-engine}

A `BasisSet` is per element: it lists the shells an oxygen gets and holds no
positions. The integral engine needs a flat list of shells with centers and
a numbering of the AO functions.
`ferric-integrals/src/basis_bridge.rs::PreparedBasis::new` takes a molecule
and a basis set and builds that list as a libint2 handle. It

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

Steps 3 and 4 are these lines, with the FFI shell record and the copies kept
for it removed:

```rust
for (ai, atom) in mol.atoms.iter().enumerate() {          // atoms in input order
    let tmpls = bs
        .for_element(atom.z)                              // real Z, also for ghosts
        .ok_or_else(|| bs.missing_element_error(atom.z))?;
    for sh in tmpls {                                     // shells as in the basis set
        // ...
        shell_to_atom.push(ai);
        shell_l.push(sh.l);
        shell_pure.push(sh.pure);
        // ...
    }
}
// ...
let mut shell_offsets = vec![0usize; nshells + 1];
for i in 0..nshells {
    shell_offsets[i + 1] = shell_offsets[i] + shell_dims[i];   // first AO of shell i+1
}
```

Two details:

- **The charge in step 2 does not trust `apply_ecp`.** It reads the ECP
  table of the basis set directly, so the nuclear attraction is right even
  if the caller skipped `Molecule::apply_ecp`. The shell lookup uses the
  physical `atom.z`, so ghost and ECP atoms keep their full basis.
- **`shell_dims` comes from libint2.** The code passes the shells and reads
  back each shell's function count, and errors if libint2 reports a
  different number of shells than it was given. The offsets are the running
  sum of those counts, so $\mu$ runs from 0 to `nbasis` with no gaps.

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
| `crates/ferric-core/src/basis.rs` (the renormalizer and its comment) | equation (@eq:basis-unit-self-overlap) and the file conventions |
| `crates/ferric-integrals/src/basis_bridge.rs` | section @sec:basis-sets-basis-set-engine |
| Dissertation, chapter 1, section 1.2.1 | the motivation for Gaussian over Slater functions |

Changes from the sources: `theory.md` writes a contracted function without the
spherical-harmonic factor and with conjugates. The book uses real functions
and states the normalization the code applies.
