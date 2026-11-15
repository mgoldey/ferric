# 23.3 Polarizable embedding

<!-- section: 23.3; status: gap; registry: none; theory: none -->

Fixed charges cannot respond to the QM region. A real environment does: the
electron density of a ligand polarizes the protein around it, and the induced
polarization acts back on the ligand. Polarizable embedding puts an
isotropic polarizability on selected MM sites and lets an induced point dipole
at each site respond to the total field, including the field of the other
dipoles. The dipoles are re-solved from the current QM density in every SCF
iteration, the same pattern as the reaction field of section 22.3.

This section has no `theory.md` text and no proof script. The equations are
read from the code loops. The gradient of the site positions has a subtlety
that the code documents as a measured bug, and the derivation below is the
reason the code looks the way it does.

| Symbol | Meaning | Code name |
|---|---|---|
| $i, j$ | polarizable sites | indices into `PolarizableSites::sites` |
| $\alpha_i$ | isotropic polarizability, Bohr$^3$ | `PolarizableSite::alpha` |
| $\boldsymbol\mu_i$ | induced dipole | `dipoles` |
| $\mathbf E_i^{\mathrm{QM}}$ | field of QM nuclei and electrons at site $i$ | `e_qm`, from `electric_field_at_points` |
| $\mathbf E_i^{\mathrm{perm}}$ | field of the permanent MM charges at site $i$ | `permanent_field_at_sites` |
| $\mathbf E_i^{\mathrm{ext}}$ | $\mathbf E_i^{\mathrm{QM}} + \mathbf E_i^{\mathrm{perm}}$ | `e_total` |
| $\mathsf T_{ij}$ | Thole-damped dipole field tensor | `thole_tensor` |
| $a$ | Thole damping parameter | `PolarizableSites::thole_a`, `DEFAULT_THOLE_A` |
| $\zeta_p$ | exponent of the point-dipole p shell | `dipole_zeta`, `DEFAULT_DIPOLE_ZETA` |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/polarizable.rs::{induce,solve_induction,thole_tensor,site_gradient,qm_gradient_contribution,charge_gradient_contribution,mm_only_polarization_energy}`, `ferric-scf/src/qmmm.rs::{polarizable_site_gradient,full_gradient_with_polarizable}`, `ferric-scf/src/driver.rs::solvent_terms` |
| CLI | `[qmmm] polarizabilities_angstrom3` and `thole_a`, for `task = "energy"` on `rhf`, `uhf`, `ksdft` only (`examples/water-qmmm-thole.toml`) |
| Python | `QmmmSystem(polarizabilities_angstrom3=...)` with `run_qmmm(thole_a=...)`; `QmmmResult.e_pol`, `induced_dipoles()` |

## 23.3.1 The induction model

Each site's dipole responds to the field there. The field has three parts: the
QM region, the permanent MM charges, and the other induced dipoles, through the
tensor $\mathsf T_{ij}$.

$$ \boldsymbol\mu_i = \alpha_i\Big[\mathbf E_i^{\mathrm{QM}}(D) + \mathbf E_i^{\mathrm{perm}} + \sum_{j\neq i}\mathsf T_{ij}\boldsymbol\mu_j\Big] . $$

A bare point dipole has $\mathsf T_{ij} = (3\hat{\mathbf r}\hat{\mathbf r}^{\mathsf T} - I)/r^3$,
which diverges at short range and lets two close sites polarize each other
without bound (the polarization catastrophe). Thole's remedy smears each
dipole as an exponential density $\rho(u)\sim e^{-au}$, with
$u = r/(\alpha_i\alpha_j)^{1/6}$, which damps the tensor at short range:

$$ \mathsf T_{ij} = \frac{3\lambda_5\,\hat{\mathbf r}\hat{\mathbf r}^{\mathsf T} - \lambda_3 I}{r^3}, \qquad \lambda_3 = 1 - \big(1 + v + \tfrac{v^2}{2}\big)e^{-v}, \qquad \lambda_5 = \lambda_3 - \tfrac{v^3}{6}e^{-v}, \qquad v = a u . $$

$\mathsf T_{ij}\boldsymbol\mu_j$ is the field at $i$ of a dipole at $j$. Any spherically
smeared dipole has $\lambda_5 = \lambda_3 - \tfrac{u}{3}\,d\lambda_3/du$, and the
pair above satisfies it, because $d\lambda_3/dv = \tfrac{v^2}{2}e^{-v}$. The
default is $a = 2.1304$ (Thole 1981; van Duijnen and Swart 1998), which belongs
to this exponential form and not to the cubic $e^{-au^3}$ form. `thole_a = None`
(Python `thole_a=0.0`) switches the damping off.

<!-- excerpt: ferric-scf/src/polarizable.rs::thole_screening -->
```rust
let lam3 = 1.0 - (1.0 + v + 0.5 * v * v) * e;
(lam3, lam3 - v * v * v / 6.0 * e)
```

<!-- excerpt: ferric-scf/src/polarizable.rs::thole_tensor -->
```rust
t[a][b] = (3.0 * lam5 * rhat[a] * rhat[b] - lam3 * eye) * inv_r3;
```

| Symbol | Code |
|---|---|
| $v = au$ | the argument `v` of `thole_screening` |
| $e^{-v}$ | `e` |
| $\lambda_3$, $\lambda_5$ | `lam3`, `lam5` |
| $\hat r_a\hat r_b$ | `rhat[a] * rhat[b]` |
| $\delta_{ab}$ | `eye` |
| $1/r^3$ | `inv_r3` |

Collecting the unknown dipoles on the left gives one dense linear system:

**Equation 23.4, the induction system, the energy and the Fock term.** (Transcribed from the code loop; see Known limits.)

$$ \big[\operatorname{diag}(\alpha^{-1}) - \mathsf T\big]\,\boldsymbol\mu = \mathbf E^{\mathrm{ext}}, \qquad E_{\mathrm{pol}} = -\tfrac12\sum_i\boldsymbol\mu_i\cdot\mathbf E_i^{\mathrm{ext}}, \qquad V^{\mathrm{ind}}_{\mu\nu} = -\sum_i\boldsymbol\mu_i\cdot\mathbf F_i^{\mu\nu}, $$

where $\mathsf T$ is the $3N\times3N$ block matrix of the $\mathsf T_{ij}$ with
zero diagonal blocks, and $\mathbf F_i^{\mu\nu}$ is the field integral at site
$i$, defined by $\mathbf E_i^{\mathrm{QM,el}} = \sum_{\mu\nu}D_{\mu\nu}\mathbf F_i^{\mu\nu}$.

The Fock term has no factor of $\tfrac12$. Define the functional
$$ W[\boldsymbol\mu] = -\boldsymbol\mu\cdot\mathbf E^{\mathrm{ext}} + \tfrac12\boldsymbol\mu\cdot\operatorname{diag}(\alpha^{-1})\boldsymbol\mu - \tfrac12\boldsymbol\mu\cdot\mathsf T\boldsymbol\mu . $$
Its stationary point in $\boldsymbol\mu$ is the induction system, and there
$W = -\tfrac12\boldsymbol\mu\cdot\mathbf E^{\mathrm{ext}} = E_{\mathrm{pol}}$. Because
$\partial W/\partial\boldsymbol\mu = 0$ at the solution, the derivative of
$E_{\mathrm{pol}}$ with respect to the density is the explicit one at fixed
$\boldsymbol\mu$: $-\boldsymbol\mu\cdot\partial\mathbf E^{\mathrm{ext}}/\partial D$,
with a factor of 1, not $\tfrac12$. That is the same variational structure as
the COSMO functional of section 22.5.

The system is assembled in `solve_induction`, with the diagonal blocks
$1/\alpha_i$ and the off-diagonal blocks $-\mathsf T_{ij}$:

<!-- excerpt: ferric-scf/src/polarizable.rs::solve_induction -->
```rust
b[(3 * i + a, 3 * i + a)] = 1.0 / site.alpha;
```

<!-- excerpt: ferric-scf/src/polarizable.rs::solve_induction -->
```rust
b[(3 * i + a, 3 * j + c)] = -t[a][c];
```

and `induce` forms the energy from the solved dipoles:

<!-- excerpt: ferric-scf/src/polarizable.rs::induce -->
```rust
e_pol -= 0.5
    * (dipoles[(i, 0)] * e_total[i][0]
        + dipoles[(i, 1)] * e_total[i][1]
        + dipoles[(i, 2)] * e_total[i][2]);
```

| Symbol | Code |
|---|---|
| $\operatorname{diag}(\alpha^{-1})$ | `b[(3 * i + a, 3 * i + a)]`, from `1.0 / site.alpha` |
| $-\mathsf T_{ij}$ | `b[(3 * i + a, 3 * j + c)]`, from `-t[a][c]` |
| $\mathbf E^{\mathrm{ext}}$ | `e_total`, the right-hand side |
| $E_{\mathrm{pol}}$ | `e_pol` |

`E_pol` includes the self-polarization of the MM region by its own permanent
charges. `mm_only_polarization_energy` returns that constant (the dipoles
induced by the permanent charges alone, with the QM region ignored), to
subtract when an interaction energy is wanted.

## 23.3.2 The integrals

$\mathbf F_i^{\mu\nu}$ is the integral of the field of a point dipole at the
site, which a p-shell Gaussian at the site delivers through the same
three-index machinery as the smeared charges of section 23.1
(`SiteBasis::new(sites, 1)`). The exponent `dipole_zeta` is $10^4$ Bohr$^{-2}$ by
default, a width of 0.01 Bohr, so the shell is numerically a point dipole and
not an exact one.

Two facts about this construction were measured, not derived, and are pinned by
`p_shell_dipole_potential_matches_charge_derivative_block` in
`crates/ferric-scf/tests/qmmm_polarizable.rs`. libint2's Cartesian p-shell
order here is not $(p_x,p_y,p_z)$ matched to the field axes: the function that
carries field axis $c$ is $(c+2) \bmod 3$ (`p_axis_for_field_axis`). And the raw
three-index integral must be multiplied, not divided, by
`norm_int_p_shell` $= 2^{1/4}\zeta_p^{5/4}/(2\pi^{3/4})$, with a minus sign. A symbolic
derivation from first principles predicted the identity ordering, a positive sign
and a division, and was falsified by the measurement.

A second sign convention bit a first implementation. The raw derivative
convention and the physical field convention of `electric_field_at_points`
differ by one sign, and the Fock term must use the physical one:

<!-- excerpt: ferric-scf/src/polarizable.rs::build_v_induced -->
```rust
let coeff = -mu[c] * norm_p;
```

| Symbol | Code |
|---|---|
| $-\mu_{i,c}$ | `-mu[c]` |
| $2^{1/4}\zeta_p^{5/4}/(2\pi^{3/4})$ | `norm_p`, from `norm_int_p_shell(dipole_zeta)` |
| $(\mu\nu\vert p_{(c+2)\bmod3})$ | `block[(p_fn * n1 + a) * n2 + bb]` with `p_fn = p_axis_for_field_axis(c)` |

A wrong-sign Fock term still converges, to a self-consistent but wrong fixed
point; it was caught only by comparing against an independent PySCF prototype,
not by the p-shell pin test, which does not exercise that sign.

## 23.3.3 The gradient

Two different formulas are needed. Getting them mixed up is the classic error
in this model, and the code records that a naive version was wrong.

**QM-atom rows** (`qm_gradient_contribution`). Hold $\boldsymbol\mu$ fixed. There are two pieces. The
Hellmann–Feynman contraction $\sum_{\mu\nu}D_{\mu\nu}\,\partial V^{\mathrm{ind}}_{\mu\nu}/\partial\mathbf R_A$,
and the nuclear part of the field, since a QM nucleus is a point charge from a
site's point of view: $-\boldsymbol\mu_i\cdot\partial\mathbf E_i^{\mathrm{QM,nuc}}/\partial\mathbf R_A$.
Omitting the second left a residual of $2.6\times10^{-5}$ in the finite-difference test
while the first piece alone matched an isolated fixed-density check.

**Site rows** (`site_gradient`). Here the naive route fails. Differentiating
$E_{\mathrm{pol}} = -\tfrac12\boldsymbol\mu\cdot\mathbf E^{\mathrm{ext}}$ at fixed $\boldsymbol\mu$
is wrong: it misses the dependence of $\mathsf T$ on the site, and it has the
wrong coefficient on the field term. The correct derivative is that of $W$, which is
stationary in $\boldsymbol\mu$, at fixed $\boldsymbol\mu$:

**Equation 23.5, the site gradient of the polarization energy.** (Transcribed from the code loop; see Known limits.)

$$ \frac{\partial E_{\mathrm{pol}}}{\partial\mathbf R_i} = -\boldsymbol\mu_i\cdot\frac{\partial\mathbf E_i^{\mathrm{ext}}}{\partial\mathbf R_i} - \sum_{j\neq i}\boldsymbol\mu_i\cdot\frac{\partial\mathsf T_{ij}}{\partial\mathbf R_i}\cdot\boldsymbol\mu_j , $$

where $\partial\mathbf E_i^{\mathrm{ext}}/\partial\mathbf R_i$ is the explicit dependence
of the field at site $i$ on its own position, at fixed $\boldsymbol\mu$. The first term has
coefficient 1 (not the $\tfrac12$ of $E_{\mathrm{pol}}$ itself), and the second is the
whole $\mathsf T$ term. Its coefficient is 1 and not
$\tfrac12$: $W$ contains $-\tfrac12\boldsymbol\mu_i\cdot\mathsf T_{ij}\boldsymbol\mu_j$ and
$-\tfrac12\boldsymbol\mu_j\cdot\mathsf T_{ji}\boldsymbol\mu_i$, and
$\mathsf T_{ji}(\mathbf R_j,\mathbf R_i) = \mathsf T_{ij}(\mathbf R_i,\mathbf R_j)^{\mathsf T}$ makes the two
equal, so differentiating both with respect to $\mathbf R_i$ gives $-\boldsymbol\mu_i\cdot\partial\mathsf T_{ij}/\partial\mathbf R_i\cdot\boldsymbol\mu_j$. The loop in
`site_gradient` visits each $j$ once per $i$, so its coefficient is 1.

On a classical two-site Thole system with no QM region, the formula of equation
23.5 reproduced central-difference gradients on every component, magnitude and
sign, while the naive $-\tfrac12\boldsymbol\mu\cdot\partial\mathbf E/\partial\mathbf R$ was off by
a factor of roughly 2 to 20 and the wrong sign on some components (the code's
own comment on the measurement). A first implementation with coefficient
$\tfrac12$ in the $\mathsf T$ loop was off by exactly 2.

The code takes $\partial\mathsf T_{ij}/\partial\mathbf R_i$ by central difference
($h = 10^{-6}$ Bohr) of the closed-form `thole_tensor`, not by hand-derived analytic
derivative, because the rank-3 tensor derivative carries real sign risk and the
tensor costs nanoseconds to evaluate. It is a finite difference of the tensor
function, not of any SCF energy.

<!-- excerpt: ferric-scf/src/polarizable.rs::site_gradient -->
```rust
g[k] += -1.0 * d_dot_dr;
```

| Symbol | Code |
|---|---|
| $\boldsymbol\mu_i\cdot\mathsf T_{ij}\cdot\boldsymbol\mu_j$ displaced by $\pm h$ along $k$ | `dot_p`, `dot_m` |
| $\partial(\boldsymbol\mu_i\mathsf T\boldsymbol\mu_j)/\partial R_{i,k}$ | `d_dot_dr`, $(\mathtt{dot\_p}-\mathtt{dot\_m})/2h$ |
| coefficient $-1$ | `-1.0` |

**Rows of the permanent charges** (`charge_gradient_contribution`). $\mathbf E_i^{\mathrm{perm}}$ depends
on the charges' positions, so by the same envelope argument
$\partial E_{\mathrm{pol}}/\partial\mathbf R_c = -\sum_i\boldsymbol\mu_i\cdot\partial\mathbf E_i^{\mathrm{perm}}/\partial\mathbf R_c$
for every site $i$ that sees charge $c$ (not colocated with it). This is
the other half of the site term's point-charge field derivative, which
`site_gradient` computes and discards, and it is separate from the classical
charge–nucleus energy. `mm_forces` (section 23.2) does not include it. Use
`full_gradient_with_polarizable`, which adds the QM rows, the site rows and
the charge rows.

## 23.3.4 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | site basis (p shells), once per geometry | $N_s$ | `SiteBasis::new(sites, 1)` |
| 2 | $\mathbf E^{\mathrm{QM}}$ at the sites from $D$, per iteration | $N_s\,N^2_{\mathrm{bf}}$ | `electric_field_at_points` |
| 3 | $\mathbf E^{\mathrm{perm}}$ at the sites | $N_sN_c$ | `permanent_field_at_sites` |
| 4 | dense $3N_s\times3N_s$ solve for $\boldsymbol\mu$ | $(3N_s)^3$ | `solve_induction` |
| 5 | $E_{\mathrm{pol}}$ and $V^{\mathrm{ind}}$ | $N_s\,N^2_{\mathrm{bf}}$ | `induce`, `build_v_induced` |
| 6 | $V^{\mathrm{ind}}$ added to every spin Fock matrix; $E_{\mathrm{pol}}$ to the energy | | `solvent_terms` |
| 7 | gradient: QM rows, site rows, charge rows | | `qm_gradient_contribution`, `site_gradient`, `charge_gradient_contribution` |

Steps 2 to 6 run every SCF iteration, in the same place as the COSMO and PCM
terms (`solvent_terms`). An empty site list returns before any solve and the
energy is bit-identical to a run with `polarizable = None`
(`polarizable_none_is_bit_identical_to_plain_scf` and
`polarizable_empty_sites_is_bit_identical_to_plain_scf` in
`crates/ferric-scf/tests/qmmm_polarizable.rs`).

## 23.3.5 Exclusions

`PolarizableSites::exclusions` is a set of site-index pairs with two effects:
$\mathsf T_{ij}$ is zero for the pair (neither polarizes the other), and the
permanent charge of an excluded partner's colocated MM atom is dropped from
that site's permanent field. The second is by position: a permanent charge
within $10^{-8}$ Bohr of an excluded partner site is dropped, which matches the
usual case where a polarizable site is also a permanently charged MM atom (one
`QmmmAtom` with both `charge` and `alpha`). With no exclusions, every site
sees every other site's permanent charge and induced dipole.

## 23.3.6 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[qmmm] polarizabilities_angstrom3` | one entry per PQR atom, Å$^3$; 0.0 = not polarizable | empty | the sites; entries on QM atoms are ignored; a nonempty list with no positive MM entry is an error |
| `[qmmm] thole_a`, Python `thole_a=` | float; `0.0` disables damping | 2.1304 | the Thole parameter; needs polarizabilities |
| `PolarizableSites::dipole_zeta` | Bohr$^{-2}$ | $10^4$ | the point-dipole p shell; library only |
| `PolarizableSites::max_sites_dense` | integer | 4000 | the dense-solve ceiling; more sites is a typed error |
| a site with non-positive or non-finite $\alpha$ | | | an error |

## 23.3.7 Numerical anchors

| Quantity | System | Result | Source |
|---|---|---|---|
| total energy, $E_{\mathrm{pol}}$, induced dipoles, QM and MM gradients, against a numpy induction model on PySCF point-dipole field integrals, self-consistent in the SCF | H2O + 4 TIP3P waters, with and without intramolecular exclusions / cc-pVDZ, RHF; NH2 + 3 waters / cc-pVDZ, UHF | total energy $\le 3.7\times10^{-12}$ Ha; $E_{\mathrm{pol}} \le 1.5\times10^{-9}$; dipoles $\le 1.5\times10^{-9}$ a.u.; gradients $\le 3.0\times10^{-9}$ Ha/Bohr | `site/src/reference/validation.md`, Thole row |
| one site 30 Å away | same | $E_{\mathrm{pol}}$ within $7.7\times10^{-8}$ relative of the classical limit $-\tfrac12\alpha\lvert E_0\rvert^2$ | same |
| $\alpha\to0$ | same | PySCF `qmmm.mm_charge` | same |

The gradients in this row are 5-point finite differences of the reference
model, not an analytic reference.

## 23.3.8 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-scf/src/polarizable.rs` | the model, the integrals, the gradients (no `theory.md` text) |
| `crates/ferric-scf/tests/validation_thole.rs`, `crates/ferric-scf/tests/qmmm_polarizable.rs` | the PySCF-prototype comparison and the FD checks |
| `crates/ferric-scf/tests/qmmm_polarizable_forces.rs`, `crates/ferric-scf/tests/qmmm_polarizable_multivariant.rs` | the force rows and the SCF variants |

**Known limits (2026-11-15).**

- The displayed equations 23.4 and 23.5 are transcribed from the code loop and await a
  `theory.md` derivation and a sympy proof; they are not machine-checked.
- The permanent-field of a Gaussian-smeared charge at a site uses the
  point-charge field, and so does its field derivative. The code documents
  this as accurate at typical separations and not as exact inside a smeared
  cloud.
- The dense solve limits a run to `max_sites_dense` sites; an iterative solve
  is a documented follow-up.
- `optimize_qmmm` refuses a polarizable embedding. The CLI carries the
  polarizable terms for the energy task on `rhf`, `uhf` and `ksdft` only, and
  refuses a geometry or frequency task, a correlated kind, `rohf`, `[cosmo]`
  and `[pcm]` with sites. `run_qmmm` carries the gradient for all four methods.
- A boundary scheme (RC or RCD) combined with a polarizable site has no
  finite-difference cross-check.
- $\partial\mathsf T/\partial\mathbf R$ is a central difference of the closed-form tensor.

**Theory text still needed in `crates/ferric-scf/theory.md`, which another pass is editing.**
The induction system, $E_{\mathrm{pol}}$ and the Fock term of equation 23.4 with
the stationarity argument; the Thole screening pair and its consistency relation;
the $W$-based site and charge gradients of equation 23.5, including the
coefficient on the $\mathsf T$ term; and the p-shell field-integral map (axis
permutation and scale), stated as a measured relation with its pin test.
