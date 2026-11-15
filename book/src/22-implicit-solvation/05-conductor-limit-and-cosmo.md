# 22.5 The conductor limit and COSMO

<!-- section: 22.5; status: gap; registry: cosmo;
     theory: crates/ferric-pcm/theory.md#"6. Conductor limit and COSMO" -->

A perfect conductor screens a charge completely: the potential on its surface
is constant, and with the right reference, zero. Replacing the dielectric by a
conductor removes the double-layer operator from the problem and leaves a
single symmetric linear system for the surface charges. COSMO, the
conductor-like screening model, solves that system and rescales the answer by
a function of $\varepsilon$ to account for a real, finite dielectric.

This section has two halves. The limit of the IEF-PCM operators is derived and
proved in `ferric-pcm`. The COSMO driver in `ferric-scf` has no `theory.md`
text and no proof script yet, so its equations here are read from the code and
marked unverified.

| Symbol | Meaning | Code name |
|---|---|---|
| $S$ | segment interaction matrix, symmetric | `build_s_matrix`, `s_mat` |
| $q_k$ | apparent charge on segment $k$ | `charges` |
| $v_k$ | solute potential at segment $k$ | `v` |
| $f_x(\varepsilon)$ | $(\varepsilon-1)/(\varepsilon+x)$, with $x = \tfrac12$ here | `CosmoConfig::f_epsilon` |
| $M_k$ | $\langle\mu\vert -1/\vert\mathbf r-\mathbf s_k\vert\vert\nu\rangle$ | the engine block with a $+1$ charge at the segment |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/cosmo.rs::{CosmoCavity,cosmo_reaction_field,CosmoConfig}`, `ferric-scf/src/driver.rs::solvent_terms` |
| CLI | `[cosmo]` with `epsilon`, `radius_scale`, `lebedev_order`, `s_matrix_kind`, for `task = "energy"` only |
| Python | none |

## 22.5.1 The limit of IEF-PCM

<!-- include: crates/ferric-pcm/theory.md#"6. Conductor limit and COSMO" -->

> **Verified equation 22.8** · IEF-PCM at $f = 1$ is the conductor equation
>
> At $f = 1$, $K = (I - DA/2\pi)S$ and $R = -(I - DA/2\pi)$ for any $S$, $D$
> and $A$, so $K^{-1}R = -S^{-1}$ exactly and $Kq = Rv$ becomes $Sq = -v$.
> The check is exact rational arithmetic on a random symmetric positive
> definite $S$ and a non-symmetric $D$.
>
> | | |
> |---|---|
> | Proof | `scripts/pcm_ief_sympy.py`, claim 4 |
> | Negative control | $f < 1$ is not the conductor equation |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pcm/src/matrices.rs::build_k_r` |
> | Code test | `ief_reduces_to_conductor_equation_as_eps_goes_to_infinity` in `crates/ferric-pcm/tests/ief_limits.rs` |
>
> <!-- verified: id=pcm.conductor.limit; script=scripts/pcm_ief_sympy.py; claim=4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pcm/src/matrices.rs::build_k_r; rust_test=ferric-pcm/tests/ief_limits.rs::ief_reduces_to_conductor_equation_as_eps_goes_to_infinity; registry=pcm -->

The finite-$\varepsilon$ scale is where the conductor-like models differ from
IEF-PCM. C-PCM scales $Sq = -v$ by $(\varepsilon-1)/\varepsilon$, which is
Kirkwood's $k_0$; COSMO scales it by $(\varepsilon-1)/(\varepsilon+\tfrac12)$,
which is Kirkwood's $k_1 = 2(\varepsilon-1)/(2\varepsilon+1)$ written with the
$2$ cancelled. Each is exact in one channel and approximate in the others,
and IEF-PCM is exact in all of them (the control rows of equation 22.4).

## 22.5.2 The COSMO driver

`ferric-scf/src/cosmo.rs` implements the Klamt–Schüürmann model with the
IEF-PCM machinery's conventions where it can. Per SCF iteration it computes
the segment potential, solves the symmetric system, and returns the operator
and energy.

**Equation 22.9, the COSMO charges, energy and operator.** (Transcribed from the code loop; see Known limits.)

$$ S\,q = -f_{1/2}(\varepsilon)\,v, \qquad f_{1/2}(\varepsilon) = \frac{\varepsilon-1}{\varepsilon+\tfrac12}, \qquad E_{\mathrm{cosmo}} = \tfrac12\,q\cdot v, \qquad V^{\mathrm{cosmo}}_{\mu\nu} = \sum_k q_k M_{k,\mu\nu} . $$

$S$ is symmetric, so $q$ is already the symmetric response and no
symmetrization is needed. The operator is the density derivative of
$E_{\mathrm{cosmo}}$ by the argument of equation 22.6 with
$Q = -f_{1/2}S^{-1}$ in place of $Q_{\mathrm{sym}}$.

The solve and the energy are three lines of
`ferric-scf/src/cosmo.rs::cosmo_reaction_field`:

<!-- excerpt: ferric-scf/src/cosmo.rs::cosmo_reaction_field -->
```rust
let s_mat = cavity.build_s_matrix(config.s_matrix_kind);
let rhs = -config.f_epsilon() * &v;
```

<!-- excerpt: ferric-scf/src/cosmo.rs::cosmo_reaction_field -->
```rust
let e_cosmo = 0.5 * charges.dot(&v);
```

| Symbol | Code |
|---|---|
| $S$ | `s_mat` |
| $-f_{1/2}v$ | `rhs` |
| $q$ | `charges`, from `s_mat.solve(&rhs)` |
| $E_{\mathrm{cosmo}}$ | `e_cosmo` |

The potential is the point-probe one of section 22.3: the electronic part is
the nuclear-attraction engine with a $+1$ charge at the segment, the nuclear
part is $\sum_BZ_B/|\mathbf s_k-\mathbf R_B|$. The Gaussian probe of PySCF is
not used.

### The cavity

`CosmoCavity::build` is a second implementation of the cavity of section
22.1. It shares `ferric_pcm::cavity::switch_h` and
`ferric_pcm::matrices::gaussian_xi_table` and differs in what it feeds them.

| | IEF-PCM (`ferric-pcm`) | COSMO (`ferric-scf`) |
|---|---|---|
| radii | modified Bondi (H 1.10 Å, B 1.92 Å), untabulated elements fall back to 2.0 Å | Bondi, H 1.20 Å; an element with no entry is an error |
| scale | 1.2 | 1.17 (`DEFAULT_RADIUS_SCALE`) |
| Lebedev order | 110 | 110 |
| switching | SWIG, quintic, $\sqrt{14/N_{\mathrm{Leb}}}$ band | the same |
| $S$ | `GaussianSmeared` or `PointCharge` | `GaussianSmeared` or `PointCharge` (`SMatrixKind`) |

The `PointCharge` $S$ matrix keeps the original Klamt–Schüürmann diagonal
$S_{kk} = 3.8\sqrt{4\pi/a_k}$ (`COSMO_XI`); the default is the Gaussian form of
section 22.1.

## 22.5.3 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | segments: position, area, $\xi_k$, $\mathrm{swf}_k$; once per SCF | $N_{\mathrm{at}}^2N_{\mathrm{Leb}}$ | `CosmoCavity::build` |
| 2 | $v_k$: one nuclear-attraction engine pass per segment, contracted with $P$ | $N\,N^2_{\mathrm{bf}}$ | `cosmo_reaction_field`, pass 1 |
| 3 | $S$, then $q$ from one LU solve | $N^3$ | `build_s_matrix`, `s_mat.solve` |
| 4 | $V^{\mathrm{cosmo}}$: the engine again, per segment, weighted by $q_k$ | $N\,N^2_{\mathrm{bf}}$ | `cosmo_reaction_field`, pass 2 |
| 5 | $V^{\mathrm{cosmo}}$ added to every spin Fock matrix, $E_{\mathrm{cosmo}}$ to the energy | | `solvent_terms` |

The two passes recompute the integrals rather than cache one $N_{\mathrm{bf}}^2$
matrix per segment, which would cost $N\,N^2_{\mathrm{bf}}$ memory. The loops in
`cosmo_reaction_field` are serial, with none of the thread parallelism of the
PCM operator in section 22.3, and $S$ is rebuilt every iteration.

## 22.5.4 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[cosmo] epsilon` | float $> 1$ | 78.39 | $\varepsilon$; $\le 1$ or non-finite is an error |
| `[cosmo] radius_scale` | float $> 0$ | 1.17 | scale on the Bondi radii |
| `[cosmo] lebedev_order` | a positive-weight Lebedev order | 110 | segments per sphere |
| `[cosmo] s_matrix_kind` | `GaussianSmeared`, `PointCharge` | `GaussianSmeared` | the $S$ formulation |
| `[cosmo]` with `[pcm]` | | | refused: two models of one solvent would be added |
| `[cosmo]` with a task other than `energy` | | | refused: there is no gradient |

## 22.5.5 Numerical anchors

| Quantity | Systems | Result | Source |
|---|---|---|---|
| solvated energy and $E_{\mathrm{cosmo}}$ against PySCF's COSMO driver with ferric's radii, cavity and point-charge potential | H2O, NH3, CH3OH, acetate($-$) / STO-3G, cc-pVDZ; $\varepsilon = 4.7, 78.4$; HO2 (UHF) / cc-pVDZ | $\le 1.9\times10^{-11}$ Ha and $\le 9.7\times10^{-11}$ Ha | `site/src/reference/validation.md`, COSMO row |
| against stock PySCF COSMO (point against Gaussian surface charges) | same | 0.05 to 0.51 % | same |
| solvation energy | water / cc-pVDZ, $\varepsilon = 78.39$ | $-5.955$ kcal/mol against PySCF COSMO $-5.94$ (the Gaussian-smeared $S$; the bare point-charge $S$ gives $-3.39$) | `crates/ferric-scf/tests/cosmo_water.rs`, `site/src/methods/scf.md` |

The first row compares the same model on both sides. The second measures the
formulation difference to stock PySCF, whose potential is Gaussian.

## 22.5.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-pcm/theory.md`, section 6 | the conductor limit and the finite-$\varepsilon$ factors |
| `scripts/pcm_ief_sympy.py`, claims 3 and 4 | the limit and the Kirkwood comparison |
| `crates/ferric-scf/src/cosmo.rs` | the driver (no `theory.md` text) |
| `crates/ferric-scf/tests/validation_cosmo.rs`, `crates/ferric-scf/tests/cosmo_gradient_refusal.rs` | the PySCF comparison and the gradient refusal |

**Known limits (2026-11-15).**

- The displayed equation 22.9 are transcribed from the code loop and await a
  `theory.md` derivation and a sympy proof; they are not machine-checked.
- Energy only. `ferric-scf/src/gradient.rs::refuse_cosmo_gradient` stops every
  gradient dispatcher, geometry optimization, saddle search, IRC and
  finite-difference frequency run on a COSMO SCF, because each would follow
  the vacuum-form gradient of a solvated energy.
- No outlying-charge correction, no cavitation, dispersion or repulsion
  terms, and no re-tessellation of partly buried spheres beyond the smooth
  switch.
- COSMO is reachable from the CLI and the Rust library only; the Python
  drivers take `solvent=` for IEF-PCM.
