# Polarizable embedding {#polarizable-embedding}

<!-- status: edit; registry: qmmm -->

Fixed charges cannot respond to the QM region, and a real environment does.
The electron density of a ligand polarizes the protein around it, and that
induced polarization acts back on the ligand. Polarizable embedding puts an
isotropic polarizability $\alpha_i$ on selected MM sites and lets an induced
point dipole at each site respond to the total field there, including the
field of the other dipoles (the Applequist model, with Thole damping). The
dipoles depend on the density, so they are re-solved from the current density
every SCF iteration, the same pattern as the reaction field of section @sec:reaction-field.

The plan: the induction equations and the damping that keeps them finite
(@sec:polarizable-embedding-induction-model), the integrals that put the dipoles into the Fock matrix (@sec:polarizable-embedding-integrals),
and the gradient (@sec:polarizable-embedding-gradient), then the algorithm, exclusions, options and
anchors. One idea runs through all of it. The polarization energy is the
value of a functional that is stationary in the dipoles, and every
derivative, the Fock term and the gradient alike, must come from that
functional. The proofs are in `scripts/thole_polarizable_sympy.py`, claims
T1 to T6.

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
| $s$ | $(\alpha_i\alpha_j)^{1/6}$, the length that scales the damping | `s` in `thole_tensor` |
| $\zeta_p$ | exponent of the point-dipole p shell | `dipole_zeta`, `DEFAULT_DIPOLE_ZETA` |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/polarizable.rs::{induce,solve_induction,thole_tensor,site_gradient,qm_gradient_contribution,charge_gradient_contribution,mm_only_polarization_energy}`, `ferric-scf/src/qmmm.rs::{polarizable_site_gradient,full_gradient_with_polarizable}`, `ferric-scf/src/driver.rs::solvent_terms` |
| CLI | `[qmmm] polarizabilities_angstrom3` and `thole_a`, for `task = "energy"` on `rhf`, `uhf`, `ksdft` only (`examples/water-qmmm-thole.toml`) |
| Python | `QmmmSystem(polarizabilities_angstrom3=...)` with `run_qmmm(thole_a=...)`; `QmmmResult.e_pol`, `induced_dipoles()` |

## The induction model {#polarizable-embedding-induction-model}

Each site's dipole is its polarizability times the field it feels. That
field has three parts: the QM nuclei and total density
(`ferric-scf/src/qmmm.rs::electric_field_at_points`, the point-charge
nuclear-attraction derivative blocks of section @sec:qmmm-boundaries-energies-forces), the permanent MM
charges, and the other induced dipoles, through the tensor $\mathsf T_{ij}$:

$$ \boldsymbol\mu_i = \alpha_i\Big[\mathbf E_i^{\mathrm{QM}}(P) + \mathbf E_i^{\mathrm{perm}} + \sum_{j\neq i}\mathsf T_{ij}\boldsymbol\mu_j\Big] . $$

$\mathsf T_{ij}\boldsymbol\mu_j$ is the field at $i$ of a dipole at $j$, with
$\mathbf r = \mathbf R_i - \mathbf R_j$. For a bare point dipole,
$\mathsf T_{ij} = (3\hat{\mathbf r}\hat{\mathbf r}^{\mathsf T} - \mathbf 1)/r^3$.
Head-to-tail dipoles reinforce each other, and at short range they do so
without bound.

### Why the bare tensor fails at short range

Take two sites on the $z$ axis. Along the axis the bare tensor is $2/r^3$, so
the $z$ components of the two dipoles obey a $2\times2$ system with matrix
$\begin{pmatrix}1/\alpha_i & -2/r^3\\ -2/r^3 & 1/\alpha_j\end{pmatrix}$. Its
determinant $1/(\alpha_i\alpha_j) - 4/r^6$ vanishes at
$r = (4\alpha_i\alpha_j)^{1/6} = 2^{1/3}s$. Closer than that the matrix is
no longer positive definite, and a finite field induces an unbounded, or
sign-flipped, response: the polarization catastrophe. It is an
artifact of treating the dipoles as points: two real charge clouds at that
distance overlap.

Thole's remedy smears each dipole as an exponential density
$\rho(u)\sim e^{-au}$, with $u = r/s$. This multiplies the two parts of the
tensor by screening functions:

$$ \mathsf T_{ij} = \frac{3\lambda_5\,\hat{\mathbf r}\hat{\mathbf r}^{\mathsf T} - \lambda_3 \mathbf 1}{r^3}, \qquad \lambda_3 = 1 - \big(1 + v + \tfrac{v^2}{2}\big)e^{-v}, \qquad \lambda_5 = \lambda_3 - \tfrac{v^3}{6}e^{-v}, \qquad v = a u . $$

Two limits check it. At long range, $v\to\infty$, $\lambda_3$ and $\lambda_5$
tend to 1 and the bare tensor returns; so does $a\to\infty$. At short range
expand in $v$: $\lambda_3 = v^3/6 - v^4/8 + \ldots$ and
$\lambda_5 = v^4/24 + \ldots$, so the $1/r^3$ cancels and $\mathsf T_{ij}$
tends to the finite matrix $-(a^3/6s^3)\,\mathbf 1$. For a pair, the matrix
stays positive definite as long as each eigenvalue of $s^3\mathsf T_{ij}$
has magnitude below 1, since $\sqrt{\alpha_i\alpha_j} = s^3$. With the
default $a$ this holds for $u > 0.304$ (the mode across the axis fails first;
the mode along it fails at $u = 0.142$), against $u < 1.26$ for the bare
tensor. These pair thresholds come from evaluating the formulas above
numerically; the proof script does not assert them. Pairs closer than that
can be removed from the coupling with exclusions (section @sec:polarizable-embedding-exclusions).

Any spherically smeared dipole satisfies $\lambda_5 = \lambda_3 - \tfrac{u}{3}\,d\lambda_3/du$.
The pair above does, because $d\lambda_3/dv = \tfrac{v^2}{2}e^{-v}$. The
identity has a consequence the gradient needs: it makes
$\mathsf T_{ab} = \partial_a\partial_b\phi(r)$ for the radial function with
$\phi'(r) = -\lambda_3/r^2$ (claim T1). The damped tensor is the Hessian of a
radial function, so it is symmetric in $a, b$ and even in $\mathbf r$, and
$\mathsf T_{ij} = \mathsf T_{ji}$.

The default is $a = 2.1304$ (Thole 1981; van Duijnen and Swart, J. Phys.
Chem. A 102, 2399, 1998). That value belongs to this exponential form and
not to the cubic $e^{-au^3}$ form. `thole_a = None` (Python `thole_a=0.0`)
switches the damping off and gives $\lambda_3 = \lambda_5 = 1$.

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

### The linear system and the energy

Move the unknown dipoles to the left. The induction equations become one
dense linear system:

**Equation @eq:thole-fock-term, the induction system, the energy and the Fock term.**

$$ \big[\operatorname{diag}(\alpha^{-1}) - \mathsf T\big]\,\boldsymbol\mu = \mathbf E^{\mathrm{ext}}, \qquad E_{\mathrm{pol}} = -\tfrac12\sum_i\boldsymbol\mu_i\cdot\mathbf E_i^{\mathrm{ext}}, \qquad V^{\mathrm{ind}}_{\mu\nu} = -\sum_i\boldsymbol\mu_i\cdot\mathbf F_i^{\mu\nu}, $$

where $\mathsf T$ is the $3N\times3N$ block matrix of the $\mathsf T_{ij}$,
with $\mathsf T_{ii} = 0$ and $\mathsf T_{ij} = 0$ for excluded pairs, and
$\mathbf F_i^{\mu\nu}$ is the field integral at site $i$, defined by
$\mathbf E_i^{\mathrm{QM,el}} = \sum_{\mu\nu}P_{\mu\nu}\mathbf F_i^{\mu\nu}$.
The matrix is symmetric because $\mathsf T$ is.

Where does $E_{\mathrm{pol}}$ come from, and why is there no $\tfrac12$ in the
Fock term? Both follow from one functional of the dipoles:

$$ W[\boldsymbol\mu] = -\boldsymbol\mu\cdot\mathbf E^{\mathrm{ext}} + \tfrac12\boldsymbol\mu\cdot\operatorname{diag}(\alpha^{-1})\boldsymbol\mu - \tfrac12\boldsymbol\mu\cdot\mathsf T\boldsymbol\mu . $$

The first term is the energy of the dipoles in the external field, the second
is the cost of polarizing each site, and the third is the dipole–dipole
interaction. Set $\partial W/\partial\boldsymbol\mu = 0$, using the symmetry
of $\mathsf T$: that is the induction system. At its solution
$\boldsymbol\mu^*$, substitute $\big[\operatorname{diag}(\alpha^{-1}) - \mathsf T\big]\boldsymbol\mu^* = \mathbf E^{\mathrm{ext}}$
into the two quadratic terms:

$$ W[\boldsymbol\mu^*] = -\boldsymbol\mu^*\cdot\mathbf E^{\mathrm{ext}} + \tfrac12\boldsymbol\mu^*\cdot\mathbf E^{\mathrm{ext}} = -\tfrac12\boldsymbol\mu^*\cdot\mathbf E^{\mathrm{ext}} = E_{\mathrm{pol}} $$

(claim T2). Check one site: $\mu = \alpha E$ and
$E_{\mathrm{pol}} = -\tfrac12\alpha|E|^2$, the classical energy of a
polarizable particle in a field.

Now differentiate with respect to the density. $W$ depends on $P$ directly,
through $\mathbf E^{\mathrm{QM}}$, and through $\boldsymbol\mu^*$. The second
route contributes $(\partial W/\partial\boldsymbol\mu)(\partial\boldsymbol\mu^*/\partial P)$,
and that is zero, because $W$ is stationary. Only the explicit dependence is
left. $\mathbf E^{\mathrm{QM}}_i$ is linear in $P$,
$\mathbf E^{\mathrm{QM}}_i = \mathbf E^{\mathrm{nuc}}_i + \sum_{\mu\nu}P_{\mu\nu}\mathbf F_i^{\mu\nu}$, so

$$ V^{\mathrm{ind}}_{\mu\nu} = \frac{\partial W}{\partial P_{\mu\nu}}\Big|_{\boldsymbol\mu^*} = -\sum_i\boldsymbol\mu_i\cdot\mathbf F_i^{\mu\nu} , $$

with a factor of 1, not $\tfrac12$. The same variational structure gives the
COSMO Fock term of section @sec:conductor-limit-and-cosmo. $V^{\mathrm{ind}}$ is added to every spin Fock
matrix, and $E_{\mathrm{pol}}$ is added outside the
$\tfrac12\mathrm{Tr}[P(h+F)]$ trace, the PCM convention of section @sec:reaction-field.

> **Verified equation {#eq:thole-fock-term}** · stationarity and the Fock term
>
> $\boldsymbol\alpha^{-1} - \mathsf T$ is symmetric because $\mathsf T_{ab} = \partial_a\partial_b\phi$
> with $\phi' = -\lambda_3/r^2$, which holds because $\lambda_5 = \lambda_3 - (u/3)\,d\lambda_3/du$
> (claim T1). $W$ is stationary at the induction solution with
> $W[\boldsymbol\mu^*] = E_{\mathrm{pol}}$ (claim T2). With $\mathbf E^{\mathrm{QM}}$
> linear in $D$, the derivative of $E_{\mathrm{pol}}$ with $\boldsymbol\mu$ re-solved
> equals $-\sum_i\boldsymbol\mu_i\cdot\mathbf F_i$, with no $\tfrac12$ (claim T4), and
> $\alpha\to0$ gives no induction (claim T5).
>
> | | |
> |---|---|
> | Proof | `scripts/thole_polarizable_sympy.py`, claims T1, T2, T4 and T5 |
> | Negative control | the runner puts $\tfrac12$ on the Fock term (claim T4 fails) and $v^3/2$ in $\lambda_5$ (claim T1 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/polarizable.rs::build_v_induced` |
> | Code test | `matches_pyscf_prototype_three_sites` in `crates/ferric-scf/tests/qmmm_polarizable.rs` |
>
> <!-- verified: id=thole.fock.term; script=scripts/thole_polarizable_sympy.py; claim=T4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/polarizable.rs::build_v_induced; rust_test=ferric-scf/tests/qmmm_polarizable.rs::matches_pyscf_prototype_three_sites; registry=qmmm -->

Claim T4 checks the Fock term by central difference. $E_{\mathrm{pol}}$ is
quadratic in the density, so that difference is exact up to rounding.

`solve_induction` assembles the matrix with the diagonal blocks $1/\alpha_i$
and the off-diagonal blocks $-\mathsf T_{ij}$:

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

The system is solved densely (`solve_into`). A site with non-positive or
non-finite $\alpha$ is an error, raised while the diagonal is filled.

$E_{\mathrm{pol}}$ includes the self-polarization of the MM region by its own
permanent charges. `mm_only_polarization_energy` returns that constant (the
dipoles induced by the permanent charges alone, with the QM region ignored),
to subtract when an interaction energy is wanted.

## The integrals {#polarizable-embedding-integrals}

$\mathbf F_i^{\mu\nu}$ is the integral of the field of a point dipole at the
site. A p-shell Gaussian at the site delivers it through the same
three-index machinery as the smeared charges of section @sec:external-potentials
(`SiteBasis::new(sites, 1)`). The exponent `dipole_zeta` is $10^4$
Bohr$^{-2}$ by default, a width $\zeta_p^{-1/2}$ of 0.01 Bohr, so the shell is
numerically a point dipole but not an exact one.

Two facts fix the map from the shell to the field axes, and both follow from
the shell's definition (claim T6):

- **Axis order.** The site shell is a pure $l = 1$ shell (`SiteBasis` is
  built with `pure: true`). Its real solid harmonics are ordered
  $m = -1, 0, +1$, that is $(y, z, x)$, so the function that carries field
  axis $c$ is $(c+2) \bmod 3$ (`p_axis_for_field_axis`).
- **Normalization.** `norm_int_p_shell` $= 2^{1/4}\zeta_p^{5/4}/(2\pi^{3/4})$
  is the reciprocal of the dipole moment $\int x\,p_x$ of the unit-normalized
  $p_x = \mathcal N x\,e^{-\zeta_p r^2}$. Multiplying the raw three-index
  integral by it gives the potential integral of a unit point dipole as
  $\zeta_p\to\infty$.

So $F_{i,c}^{\mu\nu} = \mathcal N_p(\zeta_p)\,(\mu\nu|p_{(c+2)\bmod3})$, with
the overall sign carried by `build_v_induced`. Both facts are pinned against
nuclear-attraction derivative blocks by
`p_shell_dipole_potential_matches_charge_derivative_block` in
`crates/ferric-scf/tests/qmmm_polarizable.rs`.

A sign convention sits between the integral and the Fock term. The raw
derivative convention of that pin test,
$\partial\langle\mu|1/|\mathbf r-\mathbf R|\,|\nu\rangle/\partial R_c = -\mathcal N_p(\mu\nu|p)$,
and the physical field convention of `electric_field_at_points`, which
accumulates minus the contraction, differ by one sign. The Fock term must use
the physical one:

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
point: about $10^{-5}$ absolute error in the dipoles and $1.5\times10^{-4}$ Ha
in the energy on the multi-site cases. The p-shell pin test does not exercise
that sign. The comparison with an independent PySCF prototype
(`matches_pyscf_prototype_three_sites` and its siblings) does. The scaling
uses `norm_int_p_shell` and not `SiteBasis::norm_int`, which is the $l = 0$
charge normalization of section @sec:external-potentials.

At finite $\zeta_p$ the Fock and gradient integrals are those of a Gaussian
dipole of width $\zeta_p^{-1/2} \approx 0.01$ Bohr, while
$\mathbf E^{\mathrm{QM}}$ in the induction uses the point field. So
$V^{\mathrm{ind}}$ equals $\partial W/\partial P$ up to that finite-width
error.

## The gradient {#polarizable-embedding-gradient}

Two different formulas are needed, one for the QM atoms and one for the
sites, and the second is where a naive derivation goes wrong.

**QM-atom rows** (`qm_gradient_contribution`). Hold $\boldsymbol\mu$ fixed;
the stationarity of $W$ allows it. There are two pieces. The first is the
Hellmann–Feynman contraction
$\sum_{\mu\nu}P_{\mu\nu}\,\partial V^{\mathrm{ind}}_{\mu\nu}/\partial\mathbf R_A$,
from the three-index derivative blocks 1 and 2 (the two basis-function
centres). The second is the nuclear part of the field. A QM nucleus is a
point charge from a site's point of view, so moving it changes
$\mathbf E_i^{\mathrm{QM,nuc}} = \sum_AZ_A(\mathbf R_i-\mathbf R_A)/|\mathbf R_i-\mathbf R_A|^3$,
and the term is $-\boldsymbol\mu_i\cdot\partial\mathbf E_i^{\mathrm{QM,nuc}}/\partial\mathbf R_A$
in closed form. Without the nuclear piece,
`qm_gradient_with_polarizable_sites_matches_finite_difference` shows a
residual of $2.6\times10^{-5}$. The first piece alone matches an isolated
fixed-density, fixed-$\boldsymbol\mu$ check to 6 significant figures; that
check holds the nuclei still, so it cannot see the nuclear term.

**Site rows** (`site_gradient`). Why not
differentiate $E_{\mathrm{pol}} = -\tfrac12\boldsymbol\mu\cdot\mathbf E^{\mathrm{ext}}$
at fixed $\boldsymbol\mu$? Because that expression is not stationary in
$\boldsymbol\mu$, so holding $\boldsymbol\mu$ fixed drops a real term. Its
full derivative is
$-\tfrac12\boldsymbol\mu\cdot\partial\mathbf E/\partial\mathbf R - \tfrac12(d\boldsymbol\mu/d\mathbf R)\cdot\mathbf E$,
and the second half is not zero. The one-site case shows the size of the
error. With $\mu = \alpha E(R)$ and $E_{\mathrm{pol}} = -\tfrac12\alpha E^2$,
the true derivative is $-\alpha E\,\partial E/\partial R = -\mu\,\partial E/\partial R$,
and the naive one is half of that. With two or more sites the naive form also
misses the dependence of $\mathsf T$ on the site positions entirely.

$W$ is stationary, so its derivative at fixed $\boldsymbol\mu$ is the total
derivative of $E_{\mathrm{pol}}$. Differentiate each term of $W$ with respect
to $\mathbf R_i$. The field term gives
$-\boldsymbol\mu_i\cdot\partial\mathbf E_i^{\mathrm{ext}}/\partial\mathbf R_i$. The
$\alpha$ term does not depend on positions. The $\mathsf T$ term contains, for
each $j$, both $-\tfrac12\boldsymbol\mu_i\cdot\mathsf T_{ij}\boldsymbol\mu_j$ and
$-\tfrac12\boldsymbol\mu_j\cdot\mathsf T_{ji}\boldsymbol\mu_i$. The symmetry
$\mathsf T_{ji}(\mathbf R_j,\mathbf R_i) = \mathsf T_{ij}(\mathbf R_i,\mathbf R_j)^{\mathsf T}$
makes the two equal, so together they give coefficient 1:

**Equation @eq:thole-site-gradient, the site gradient of the polarization energy.**

$$ \frac{\partial E_{\mathrm{pol}}}{\partial\mathbf R_i} = -\boldsymbol\mu_i\cdot\frac{\partial\mathbf E_i^{\mathrm{ext}}}{\partial\mathbf R_i} - \sum_{j\neq i}\boldsymbol\mu_i\cdot\frac{\partial\mathsf T_{ij}}{\partial\mathbf R_i}\cdot\boldsymbol\mu_j , $$

where $\partial\mathbf E_i^{\mathrm{ext}}/\partial\mathbf R_i$ is the explicit
dependence of the field at site $i$ on its own position. Both terms have
coefficient 1, not the $\tfrac12$ of $E_{\mathrm{pol}}$ itself. The loop in
`site_gradient` visits each $j$ once per $i$, so its coefficient is 1 too.
The field term has three parts in the code: the electronic part by
translational invariance of the p-shell integral, $-(\text{block 1} + \text{block 2})$,
and the QM-nuclear and permanent-charge parts in closed form.

> **Verified equation {#eq:thole-site-gradient}** · the site and charge gradients
>
> On a classical three-site model with fixed point charges standing in for the
> QM field, the transcribed site and charge gradients, with $\boldsymbol\mu$ held
> at the solution, equal central differences of $E_{\mathrm{pol}}$ with
> $\boldsymbol\mu$ re-solved at every displacement. Coefficient $\tfrac12$ on the
> $\mathsf T$ term and the naive fixed-$\boldsymbol\mu$ derivative of
> $-\tfrac12\boldsymbol\mu\cdot\mathbf E$ both fail. The $h = 10^{-6}$ central
> difference of `thole_tensor` agrees with the analytic third derivative of
> $\phi$ within its rounding bound (claim T3).
>
> | | |
> |---|---|
> | Proof | `scripts/thole_polarizable_sympy.py`, claim T3 |
> | Negative control | the runner puts $-0.5$ on the $\mathsf T$ term in the site loop (claim T3 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/polarizable.rs::site_gradient` |
> | Code test | `site_force_with_polarizable_sites_matches_finite_difference` in `crates/ferric-scf/tests/qmmm_polarizable.rs` |
>
> <!-- verified: id=thole.site.gradient; script=scripts/thole_polarizable_sympy.py; claim=T3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/polarizable.rs::site_gradient; rust_test=ferric-scf/tests/qmmm_polarizable.rs::site_force_with_polarizable_sites_matches_finite_difference; registry=qmmm -->

The code records a measurement on a classical two-site Thole system with no
QM region. Equation @eq:thole-site-gradient reproduced central-difference gradients to
$10^{-8}$ absolute on every component, magnitude and sign. The naive
$-\tfrac12\boldsymbol\mu\cdot\partial\mathbf E/\partial\mathbf R$ was off by a
factor of roughly 2 to 20 and had the wrong sign on 2 of the 6 components.
Coefficient $\tfrac12$ in the $\mathsf T$ loop is off by exactly a factor of 2,
which `site_force_with_polarizable_sites_matches_finite_difference` detects.

The code takes $\partial\mathsf T_{ij}/\partial\mathbf R_i$ by central difference
of the closed-form `thole_tensor`, step $h = 10^{-6}$ Bohr, in a loop over
the three Cartesian directions:

<!-- excerpt: ferric-scf/src/polarizable.rs::site_gradient -->
```rust
let h = 1e-6;
for k in 0..3 {
    let mut ri_p = ri;
    let mut ri_m = ri;
    ri_p[k] += h;
    ri_m[k] -= h;
    let t_p = thole_tensor(ri_p, rj, ai, aj, sites.thole_a);
    let t_m = thole_tensor(ri_m, rj, ai, aj, sites.thole_a);
    // mu_i . dT_ij/dR_i[k] . mu_j
    let mut dot_p = 0.0_f64;
    let mut dot_m = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            dot_p += mu_i[a] * t_p[a][b] * mu_j[b];
            dot_m += mu_i[a] * t_m[a][b] * mu_j[b];
        }
    }
    let d_dot_dr = (dot_p - dot_m) / (2.0 * h);
    g[k] += -1.0 * d_dot_dr;
}
```

| Symbol | Code |
|---|---|
| $\mathsf T_{ij}$ at $\mathbf R_i \pm h\hat{\mathbf e}_k$ | `t_p`, `t_m` |
| $\boldsymbol\mu_i\cdot\mathsf T_{ij}\cdot\boldsymbol\mu_j$ at the two points | `dot_p`, `dot_m` |
| $\partial(\boldsymbol\mu_i\cdot\mathsf T_{ij}\cdot\boldsymbol\mu_j)/\partial R_{i,k}$ | `d_dot_dr`, $(\mathtt{dot\_p}-\mathtt{dot\_m})/2h$ |
| coefficient $-1$ of equation @eq:thole-site-gradient | `-1.0` |

Why a finite difference? The analytic form is
$\partial_k\mathsf T_{ab} = \partial_a\partial_b\partial_k\phi$, with the $\phi$
of claim T1, but the rank-3 derivative written out by hand carries sign risk,
and the tensor costs nanoseconds to evaluate. It is a difference of the tensor
function, not of any SCF energy. Its rounding error is about
$\epsilon|\mathsf T|/h$, roughly $10^{-10}$ relative, and its $O(h^2)$
truncation error is far below that.

**Rows of the permanent charges** (`charge_gradient_contribution`).
$\mathbf E_i^{\mathrm{perm}}$ depends on the charges' positions as well. By the
same stationarity argument,
$\partial E_{\mathrm{pol}}/\partial\mathbf R_c = -\sum_i\boldsymbol\mu_i\cdot\partial\mathbf E_i^{\mathrm{perm}}/\partial\mathbf R_c$,
summed over every site $i$ that sees charge $c$ (not colocated with it or
excluded from it). It is the other half of the point-charge field
derivative that `site_gradient` computes and discards. It is separate from
the classical charge–nucleus energy of section @sec:external-potentials, and `mm_forces`
(section @sec:qmmm-boundaries) does not include it. Use `full_gradient_with_polarizable`,
which adds the QM rows, the site rows and the charge rows to
`full_gradient`.

Check the trivial limit. With every $\alpha_i\to0$ the dipoles vanish and
every term of this section is zero (claim T5).

## The algorithm {#polarizable-embedding-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | site basis (p shells), once per geometry | $N_s$ | `SiteBasis::new(sites, 1)` |
| 2 | $\mathbf E^{\mathrm{QM}}$ at the sites from $P$, per iteration | $N_s\,N^2_{\mathrm{bf}}$ | `electric_field_at_points` |
| 3 | $\mathbf E^{\mathrm{perm}}$ at the sites | $N_sN_c$ | `permanent_field_at_sites` |
| 4 | dense $3N_s\times3N_s$ solve for $\boldsymbol\mu$ | $(3N_s)^3$ | `solve_induction` |
| 5 | $E_{\mathrm{pol}}$ and $V^{\mathrm{ind}}$ | $N_s\,N^2_{\mathrm{bf}}$ | `induce`, `build_v_induced` |
| 6 | $V^{\mathrm{ind}}$ added to every spin Fock matrix; $E_{\mathrm{pol}}$ to the energy | | `solvent_terms` |
| 7 | gradient: QM rows, site rows, charge rows | | `qm_gradient_contribution`, `site_gradient`, `charge_gradient_contribution` |

Steps 2 to 6 run every SCF iteration, in the same place as the COSMO and PCM
terms (`solvent_terms`). The site basis depends only on the geometry, so it is
built once per SCF, outside the loop, like the COSMO cavity and the PCM
context. An empty site list returns before any solve, without building a
$0\times0$ system or touching the density, and the energy is bit-identical to a
run with `polarizable = None`
(`polarizable_none_is_bit_identical_to_plain_scf` and
`polarizable_empty_sites_is_bit_identical_to_plain_scf` in
`crates/ferric-scf/tests/qmmm_polarizable.rs`).

## Exclusions {#polarizable-embedding-exclusions}

`PolarizableSites::exclusions` is a set of site-index pairs with two
effects. $\mathsf T_{ij}$ is zero for the pair, so neither polarizes the
other. And the permanent charge colocated with an excluded partner is dropped
from the site's permanent field. The second is by position: a permanent
charge within $10^{-8}$ Bohr of site $i$ itself, or of a site excluded from
$i$, is skipped. That matches the usual case, where a polarizable site is also
a permanently charged MM atom (one `QmmmAtom` with both `charge` and
`alpha`): a site never feels its own charge, and an excluded neighbour's
charge goes with the neighbour. With no exclusions, every site sees every
other site's permanent charge and induced dipole.

A Gaussian-smeared permanent charge enters $\mathbf E^{\mathrm{perm}}$ with
the point-charge field $q\,\mathbf r/r^3$, not the field of the Gaussian
(`permanent_field_at_sites`). The gradients use the same point form, so the
gradient is that of the energy computed.

## Options {#polarizable-embedding-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[qmmm] polarizabilities_angstrom3` | one entry per PQR atom, Å$^3$; 0.0 = not polarizable | empty | the sites; entries on QM atoms are ignored; a nonempty list with no positive MM entry is an error |
| `[qmmm] thole_a`, Python `thole_a=` | float; `0.0` disables damping | 2.1304 | the Thole parameter; needs polarizabilities |
| `PolarizableSites::dipole_zeta` | Bohr$^{-2}$ | $10^4$ | the point-dipole p shell; library only |
| `PolarizableSites::max_sites_dense` | integer | 4000 | the dense-solve ceiling; more sites is a typed error |
| a site with non-positive or non-finite $\alpha$ | | | an error |

## Numerical anchors {#polarizable-embedding-numerical-anchors}

| Quantity | System | Result | Source |
|---|---|---|---|
| total energy, $E_{\mathrm{pol}}$, induced dipoles, QM and MM gradients, against a numpy induction model on PySCF point-dipole field integrals, self-consistent in the SCF | H2O + 4 TIP3P waters, with and without intramolecular exclusions / cc-pVDZ, RHF; NH2 + 3 waters / cc-pVDZ, UHF | total energy $\le 3.7\times10^{-12}$ Ha; $E_{\mathrm{pol}} \le 1.5\times10^{-9}$; dipoles $\le 1.5\times10^{-9}$ a.u.; gradients $\le 3.0\times10^{-9}$ Ha/Bohr | `site/src/reference/validation.md`, Thole row |
| one site 30 Å away | same | $E_{\mathrm{pol}}$ within $7.7\times10^{-8}$ relative of the classical limit $-\tfrac12\alpha\lvert E_0\rvert^2$ | same |
| $\alpha\to0$ | same | PySCF `qmmm.mm_charge` | same |

The second row is the one-site check of section @sec:polarizable-embedding-induction-model measured at the far
limit: one distant site sees an almost uniform field $\mathbf E_0$ and no
other dipole. The gradients in the first row are 5-point finite differences
of the reference model, not an analytic reference.

## Sources {#polarizable-embedding-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-scf/theory.md`, "Thole polarizable embedding" | the derivation |
| `scripts/thole_polarizable_sympy.py` | the proofs, claims T1 to T6 |
| `crates/ferric-scf/src/polarizable.rs` | the model, the integrals, the gradients |
| `crates/ferric-scf/tests/validation_thole.rs`, `crates/ferric-scf/tests/qmmm_polarizable.rs` | the PySCF-prototype comparison and the FD checks |
| `crates/ferric-scf/tests/qmmm_polarizable_forces.rs`, `crates/ferric-scf/tests/qmmm_polarizable_multivariant.rs` | the force rows and the SCF variants |

**Known limits (2026-11-15).**

- The permanent field of a Gaussian-smeared charge at a site uses the
  point-charge field $q\,\mathbf r/r^3$, and so do its derivatives in
  `site_gradient` and `charge_gradient_contribution`. The gradient is
  therefore that of the energy computed, but the energy is not that of a
  smeared charge inside its cloud.
- The induction uses the point field of the QM density, while $V^{\mathrm{ind}}$
  and the gradients use the $p$ shell of width $\zeta_p^{-1/2}$, so the Fock term
  equals the density derivative of $E_{\mathrm{pol}}$ only up to that finite-width
  error.
- The dense solve limits a run to `max_sites_dense` sites; an iterative solve
  is a documented follow-up.
- `optimize_qmmm` refuses a polarizable embedding. The CLI carries the
  polarizable terms for the energy task on `rhf`, `uhf` and `ksdft` only, and
  refuses a geometry or frequency task, a correlated kind, `rohf`, `[cosmo]`
  and `[pcm]` with sites. `run_qmmm` carries the gradient for all four methods.
- A boundary scheme (RC or RCD) combined with a polarizable site has no
  finite-difference cross-check.
- $\partial\mathsf T/\partial\mathbf R$ is a central difference of the closed-form tensor
  with $h = 10^{-6}$ Bohr, whose rounding error is about $\epsilon\lvert\mathsf T\rvert/h$;
  the analytic form $\partial_a\partial_b\partial_k\phi$ is in `theory.md` and not
  in the code.
