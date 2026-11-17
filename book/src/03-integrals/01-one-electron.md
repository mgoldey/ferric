# 3.1 One-electron integrals and multipoles

<!-- section: 3.1; status: edit; registry: none;
     theory: crates/ferric-integrals/theory.md#"Conventions", "One-electron integrals" -->

An electron in a molecule has kinetic energy, and it is attracted to every
nucleus. In a finite basis those two facts become two matrices, $T$ and $V$.
A third matrix, the overlap $S$, records that the basis functions are not
orthogonal: a $1s$ function on oxygen and a $1s$ function on a neighbouring
hydrogen share some of the same space. Their sum $h = T + V$ is the core
Hamiltonian, the energy of one electron with no other electrons present. Every
SCF calculation builds $S$, $T$ and $V$ first.

The multipole matrices answer a different question: where the charge is. The
dipole matrices give the dipole moment of a density, and the second moments
give quadrupoles and the spatial spread of an orbital.

The plan:

1. Fix the basis functions and their normalization, since every later
   number depends on it.
2. Show why integrals over Gaussians are cheap (the Gaussian product
   theorem).
3. Define the one-electron matrices and the multipoles, and derive the
   origin-shift rule the tests rely on.
4. Read the loop that assembles the matrices.

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/oneelectron.rs::{overlap,kinetic,nuclear,hcore,hcore_ecp,dipole,second_moment,r2_moment}` |
| Python | `AoIntegrals` (`crates/ferric-python/ferric.pyi`) |

## 3.1.1 Basis functions and the integrals

A contracted Cartesian Gaussian on centre $\mathbf A$ is a polynomial times a
fixed linear combination of Gaussians of different widths:

$$ \chi_\mu(\mathbf r) = x_A^{l_x} y_A^{l_y} z_A^{l_z} \sum_k d_k\, N(\alpha_k, l)\, e^{-\alpha_k r_A^2}, \qquad \mathbf r_A = \mathbf r - \mathbf A, \quad l = l_x + l_y + l_z, $$

$$ N(\alpha, l) = \Big(\frac{2\alpha}{\pi}\Big)^{3/4} \frac{(4\alpha)^{l/2}}{\sqrt{(2l-1)!!}} . $$

The polynomial carries the angular momentum: $l = 0$ is an $s$ function,
$l = 1$ a $p$ function, and so on. The exponents $\alpha_k$ set the widths,
and the contraction coefficients $d_k$ are fixed by the basis set.

**One normalization serves the whole shell.** Square a primitive and
integrate. The integral factorizes into three one-dimensional Gaussian
moments,

$$ \int x^{2l_x} e^{-2\alpha x^2}\,dx = \frac{(2l_x - 1)!!}{(4\alpha)^{l_x}}\sqrt{\frac{\pi}{2\alpha}}, $$

so the squared norm of $x^{l_x}y^{l_y}z^{l_z}e^{-\alpha r^2}$ is proportional
to $(2l_x-1)!!\,(2l_y-1)!!\,(2l_z-1)!!$. $N(\alpha,l)$ uses $(2l-1)!!$, which
is the product for the $(l,0,0)$ component. So $(l,0,0)$ is unit normalized
and the others are not. For a $d$ shell, $xx$ has norm 1 and $xy$ has squared
norm $1/3$; for an $f$ shell, $xyz$ has squared norm $1/15$. One factor per
shell keeps every component a fixed multiple of the same radial function,
which is the convention libint2 uses, so ferric's blocks and libint2's agree
without a per-component rescale.

The stored contraction coefficients $d_k$ do not contain $N$. The basis loader
renormalizes each contraction to unit self-overlap
(`ferric-core/src/basis.rs::parse_bse_json`).

**Ordering.** Shells with $l \ge 2$ may be pure (solid harmonics, ordered
$m = -l, \dots, l$). $s$ and $p$ shells are always Cartesian, with $p$ ordered
$x, y, z$. Cartesian components are in CCA order ($l_x$ descending, then $l_y$
descending): a Cartesian $d$ shell is $xx, xy, xz, yy, yz, zz$.

**The integrals.** The one-electron matrices are

$$ S_{\mu\nu} = \int \chi_\mu\chi_\nu, \qquad T_{\mu\nu} = -\tfrac12\int \chi_\mu \nabla^2 \chi_\nu, \qquad V_{\mu\nu} = -\sum_C Z_C \int \frac{\chi_\mu(\mathbf r)\chi_\nu(\mathbf r)}{|\mathbf r - \mathbf C|}\,d\mathbf r, $$

and the two-electron integrals over a kernel $\hat w$ (section 3.2) are

$$ (\mu\nu|\hat w|\lambda\sigma) = \iint \chi_\mu(1)\chi_\nu(1)\, w(r_{12})\, \chi_\lambda(2)\chi_\sigma(2), \qquad (P|\hat w|\mu\nu), \qquad V_{PQ} = (P|\hat w|Q) . $$

$Z_C$ is the effective nuclear charge: $Z - n_{\mathrm{core}}$ on an ECP atom
and $0$ on a ghost atom. A ghost atom keeps its basis functions and loses its
nucleus, which is what a counterpoise calculation needs.

### Why Gaussians: the product theorem

Every integral above contains the product $\chi_\mu\chi_\nu$ of two functions
on different centres. For Gaussians that product is one Gaussian. In one
Cartesian direction, with $p = a + b$, complete the square in the exponent:

$$ e^{-a(x - A_x)^2}\, e^{-b(x - B_x)^2} = e^{-\frac{ab}{p}X_{AB}^2}\; e^{-p(x - P_x)^2}, \qquad P_x = \frac{a A_x + b B_x}{p}, \quad X_{AB} = A_x - B_x . $$

The product sits at $\mathbf P$, on the line between the two centres and
closer to the tighter function. Its height is the prefactor
$e^{-ab X_{AB}^2/p}$, which falls off as a Gaussian in the distance between
the centres.

Check the limits. With $A = B$ the prefactor is 1 and $\mathbf P = \mathbf A$:
two Gaussians on one centre multiply to one Gaussian of exponent $a + b$. With
the centres far apart the prefactor goes to zero, so the overlap of distant
functions vanishes. That prefactor is the source of the sparsity that the
screening of chapter 5 exploits. A three- or four-center integral therefore
reduces to integrals over one or two Gaussians, and those have closed forms.
Section 3.4 carries this through for the Coulomb kernel.

## 3.1.2 The one-electron matrices

`ferric-integrals/src/oneelectron.rs::overlap`,
`ferric-integrals/src/oneelectron.rs::kinetic` and
`ferric-integrals/src/oneelectron.rs::nuclear` build $S$, $T$ and $V$ from
libint2 shell-pair blocks. `ferric-integrals/src/oneelectron.rs::hcore` adds
$h = T + V$. `hcore` does not contain the ECP operator;
`ferric-integrals/src/oneelectron.rs::hcore_ecp` adds it (section 3.5).

All three matrices are symmetric, so each unordered shell pair is computed
once and the block is mirrored. Here is the serial branch of
`ferric-integrals/src/oneelectron.rs::build_symmetric`, unchanged:

```rust
let mut eng = make_eng();                       // one engine for the whole loop
for s1 in 0..nsh {
    for s2 in 0..=s1 {                          // each unordered pair once
        let block = eng.compute_1e_block(prep, s1, s2);   // n1 x n2, row-major
        let n1 = dims[s1];
        let n2 = dims[s2];
        for i in 0..n1 {
            for j in 0..n2 {
                let v = block[i * n2 + j];
                out[(offs[s1] + i, offs[s2] + j)] = v;    // (mu, nu)
                out[(offs[s2] + j, offs[s1] + i)] = v;    // (nu, mu) = (mu, nu)
            }
        }
    }
}
```

Three points about it:

- **The operator lives in the engine.** The same loop builds $S$, $T$ and $V$.
  The caller passes `make_eng`, which builds an overlap, kinetic or
  nuclear-attraction engine with precision $10^{-14}$. For $V$ the engine also
  receives the point charges $Z_C$ (`Engine::set_point_charges`).
- **The parallel branch writes the same numbers.** At
  `PAR_SHELL_PAIR_THRESHOLD` = 64 shells or more, the outer index `s1` is
  split across rayon workers. A fixed `s1` writes only the rows of its own
  shell, directly or through the mirror, and every element is written once
  with no accumulation. So the parallel result is bit-identical to the serial
  one for any thread count.
- **Small jobs stay serial.** Below 64 shells the loop above runs on one
  thread. A free-atom SCF then does not pay for engine construction on every
  worker.

### Multipoles

The multipole integrals are taken about an origin $\mathbf O$, with no
electron charge applied (the sign of the electron is the caller's to add):

$$ d^{p}_{\mu\nu} = \langle\mu|(r - O)_p|\nu\rangle, \qquad Q^{pq}_{\mu\nu} = \langle\mu|(r - O)_p (r - O)_q|\nu\rangle . $$

`ferric-integrals/src/oneelectron.rs::dipole` returns three matrices,
`ferric-integrals/src/oneelectron.rs::second_moment` returns six in the order
$[xx, xy, xz, yy, yz, zz]$, and `ferric-integrals/src/oneelectron.rs::r2_moment`
is $Q^{xx} + Q^{yy} + Q^{zz}$, the operator from which orbital spreads
$\sigma^2 = \langle r^2\rangle - \langle r\rangle^2$ are built.

How do the moments change when the origin moves? Put
$\mathbf O' = \mathbf O + \mathbf d$. Then $(r - O')_p = (r - O)_p - d_p$, and
expanding the product of two such factors inside the integral gives

$$ Q^{pq}_{O'} = Q^{pq}_O - d_p\, d^q_O - d_q\, d^p_O + d_p d_q\, S . $$

The same expansion with one factor gives $d^p_{O'} = d^p_O - d_p S$. Check the
limit $\mathbf d = 0$: both rules return the unshifted matrices. The shift rule
connects three engines (overlap, dipole, second moment), so a test that
shifts the origin and compares pins every second-moment component against
the other two. The crate's tests do this.

## 3.1.3 Options that change the matrices

| Option | Effect |
|---|---|
| ECP atoms (`mol.apply_ecp`) | $Z_C \to Z_C - n_{\mathrm{core}}$ in $V$; $\hat U_{\mathrm{ECP}}$ added by `hcore_ecp` (section 3.5) |
| ghost atoms | $Z_C = 0$; the basis functions stay |
| external point charges and field | folded into $h$ once before the SCF (`ferric-integrals/src/oneelectron.rs::hcore_with_external`); `None` leaves $h$ bit-identical |

## 3.1.4 Numerical anchors

| What | Test | Bar |
|---|---|---|
| every bundled basis set loads to the published contractions (RHF and RI-MP2 against PySCF) | `crates/ferric-mp2/tests/basis_coverage_vs_pyscf.rs`, `crates/ferric-mp2/tests/validation_basis_coverage.rs`; the basis row of `site/src/reference/validation.md` | per row |
| translation identity of the second moments | `ferric-integrals/src/oneelectron.rs::second_moment_translational_identity` | rounding |
| shifted-origin $S$, $T$, $V$ blocks: zero shift is bitwise the plain block | `crates/ferric-integrals/tests/one_electron_shifted.rs` | bitwise |

No proof script covers this section: the integrals are libint2's, and the
section adds conventions, not derivations. The translation identity of the
second moments is the one identity the code relies on, and the unit tests
check it directly.

## Sources

- libint2: E. F. Valeev, Libint, a library for the evaluation of molecular
  integrals over Gaussian functions, <https://github.com/evaleev/libint>.
- T. Helgaker, P. Jørgensen and J. Olsen, *Molecular Electronic-Structure
  Theory* (Wiley, 2000), chapter 9.
