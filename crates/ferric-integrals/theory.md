# Molecular integrals

This crate evaluates Gaussian integrals: one-electron matrices, two-, three- and
four-center two-electron integrals over several kernels, their derivatives, basis
functions on grids, the screening bounds that decide which integrals are computed,
and the storage of the three-index tensor that density fitting consumes. Most
integral classes go through libint2 behind the C shim (`shim/shim.cc`); the terf
and terfc kernels, the three-center one-electron batch kernel and the ECPs have
their own implementations.

Notation follows the book (`book/notation.md`): real functions, chemist
notation, $(P|\mu\nu)$ with the auxiliary index first, $\hat w$ for a general
kernel, atomic units. Every heading below is a stable anchor; book sections
include these sections by heading text.

## Conventions

A contracted Cartesian Gaussian on centre $\mathbf A$ is

$$ \chi_\mu(\mathbf r) = x_A^{l_x} y_A^{l_y} z_A^{l_z} \sum_k d_k\, N(\alpha_k, l)\, e^{-\alpha_k r_A^2}, \qquad \mathbf r_A = \mathbf r - \mathbf A, \quad l = l_x + l_y + l_z, $$

$$ N(\alpha, l) = \Big(\frac{2\alpha}{\pi}\Big)^{3/4} \frac{(4\alpha)^{l/2}}{\sqrt{(2l-1)!!}} . $$

One normalization serves the whole shell: the $(l,0,0)$ component is unit
normalized and the others are not. The stored contraction coefficients $d_k$ do
not contain $N$; the basis loader renormalizes each contraction to unit
self-overlap (`ferric-core/src/basis.rs::parse_bse_json`). Shells with $l \ge 2$
may be pure (solid harmonics, ordered $m = -l, \dots, l$); $s$ and $p$ shells are
always Cartesian, with $p$ ordered $x, y, z$. Cartesian components are in CCA
order ($l_x$ descending, then $l_y$ descending).

The integrals are

$$ S_{\mu\nu} = \int \chi_\mu\chi_\nu, \qquad T_{\mu\nu} = -\tfrac12\int \chi_\mu \nabla^2 \chi_\nu, \qquad V_{\mu\nu} = -\sum_C Z_C \int \frac{\chi_\mu(\mathbf r)\chi_\nu(\mathbf r)}{|\mathbf r - \mathbf C|}\,d\mathbf r, $$

$$ (\mu\nu|\hat w|\lambda\sigma) = \iint \chi_\mu(1)\chi_\nu(1)\, w(r_{12})\, \chi_\lambda(2)\chi_\sigma(2), \qquad (P|\hat w|\mu\nu), \qquad V_{PQ} = (P|\hat w|Q) . $$

$Z_C$ is the effective nuclear charge: $Z - n_{\mathrm{core}}$ on an ECP atom and
$0$ on a ghost atom.

## One-electron integrals

`oneelectron.rs` builds $S$, $T$, $V$ and $h = T + V$ from libint2 shell-pair
blocks over the upper triangle of shell pairs and mirrors each block
(`overlap`, `kinetic`, `nuclear`, `hcore`). `hcore` does not contain the ECP
operator; `hcore_ecp` adds it.

The multipole integrals are taken about an origin $\mathbf O$ with no electron
charge applied:

$$ d^{p}_{\mu\nu} = \langle\mu|(r - O)_p|\nu\rangle, \qquad Q^{pq}_{\mu\nu} = \langle\mu|(r - O)_p (r - O)_q|\nu\rangle , $$

returned as three matrices (`dipole`) and as $[xx, xy, xz, yy, yz, zz]$
(`second_moment`). A shift of origin $\mathbf O' = \mathbf O + \mathbf d$ maps

$$ Q^{pq}_{O'} = Q^{pq}_O - d_p\, d^q_O - d_q\, d^p_O + d_p d_q\, S , $$

which the crate's tests use to pin every second-moment component against the
dipole and overlap engines. `r2_moment` is $Q^{xx} + Q^{yy} + Q^{zz}$.

## Two-, three- and four-center integrals

`engine.rs` wraps one libint2 engine per kernel and integral class:
`Engine::new_2e` (four-center), `Engine::new_3center` with
`Engine::compute_eri3` for $(P|\mu\nu)$, and `Engine::new_2center` with
`Engine::compute_eri2` for $(P|Q)$. Libint2 serves the Coulomb, erf, erfc,
Yukawa and Slater-geminal kernels. The terf and terfc kernels go to the table
engine of this crate (two-, three- and four-center, `scf_compute_terfc_eri2`,
`_eri3`, `_eri4`; a libint2-native terf route exists behind the
`libint2_terf` feature and a patched libint2) ("Two-center integrals over the terf kernel"); a linear
combination of kernels (`Operator::composite`) is a sum of engines, each with
its weight.

An engine is constructed with a precision. Libint2 skips a shell quartet whose
estimated magnitude is below it and the engine returns no block (`None`). Most
callers pass $10^{-14}$; the Schwarz table builder passes $0$, which disables
that skip ("The Schwarz bound").

A block for shells $(s_1 s_2|s_3 s_4)$ is row-major over the shells' functions,
$n_1 n_2 n_3 n_4$ doubles. A table engine can refuse a quartet (total angular
momentum past the table depth, tables not loaded); `compute_quartet_checked`,
`compute_eri3_checked` and `compute_eri2_checked` return that refusal as an
error, distinct from a screened block.

## Derivative integrals and buffer sizing

A basis function on centre $\mathbf A$ depends on $\mathbf A$ only through
$\mathbf r - \mathbf A$, so

$$ \frac{\partial \chi_\mu}{\partial A_x} = -\frac{\partial \chi_\mu}{\partial x}, \qquad \frac{\partial}{\partial A_x}\big(x_A^{l} e^{-\alpha r_A^2}\big) = 2\alpha\, x_A^{l+1} e^{-\alpha r_A^2} - l\, x_A^{l-1} e^{-\alpha r_A^2} . $$

Translational invariance makes the derivatives with respect to all centres of an
integral sum to zero. Libint2 nevertheless returns every centre's derivative,
and the number of blocks a derivative engine writes is:

| Integral | Blocks | Function |
|---|---|---|
| overlap, kinetic | $6$: $\partial_{A}$, $\partial_{B}$ | `Engine::compute_1e_deriv_block` |
| nuclear attraction with $n_c$ point charges | $3(2 + n_c)$: bra, ket, then each charge | `Engine::compute_1e_deriv_block_n` |
| four-center | $12$ | `Engine::compute_eri_deriv_quartet` |
| three-center | $9$ | `Engine::compute_eri3_deriv` |
| two-center | $6$ | `Engine::compute_eri2_deriv` |

The nuclear count grows with the number of charges, so a caller that adds
external charges (`Engine::set_point_charges_extra`) must size the buffer for
the actual charge count with `compute_1e_deriv_block_n`. Second derivatives are
returned as the upper triangle of coordinate pairs
(`deriv2_pair_index`, `deriv2_pairs`).

## McMurchie-Davidson three-center one-electron integrals

Seminumerical exchange needs, for a batch of grid points $\mathbf r_g$,

$$ A^{g}_{\mu\nu} = \int \frac{\chi_\mu(\mathbf r)\chi_\nu(\mathbf r)}{|\mathbf r - \mathbf r_g|}\, d\mathbf r , $$

with the repulsive sign (raw libint2 nuclear attraction has the opposite sign).
`md3c1e.rs` evaluates it by the McMurchie-Davidson expansion. For primitives of
exponents $a$, $b$ on centres $A$, $B$, with $p = a + b$ and
$\mathbf P = (a\mathbf A + b\mathbf B)/p$, the product expands in Hermite
Gaussians $\Lambda_{tuv}$ on $\mathbf P$,

$$ x_A^{i} x_B^{j}\, e^{-a x_A^2 - b x_B^2} = \sum_{t=0}^{i+j} E^{ij}_t\, \Lambda_t(x), $$

with the recursion (one Cartesian direction, $X_{AB} = A_x - B_x$)

$$ E^{00}_0 = e^{-ab X_{AB}^2/p}, \qquad E^{i+1,j}_t = \frac{1}{2p}E^{ij}_{t-1} + X_{PA}E^{ij}_t + (t+1)E^{ij}_{t+1}, \qquad X_{PA} = -\frac{b}{p}X_{AB}, $$

and the same with $X_{PB} = (a/p)X_{AB}$ for $j+1$ (`e_table`). The Coulomb
kernel acts on each Hermite Gaussian analytically:

$$ A^g = \frac{2\pi}{p} \sum_{tuv} E^{ij}_t E^{kl}_u E^{mn}_v\, R^0_{tuv}(p, \mathbf P - \mathbf r_g), $$

$$ R^n_{000} = (-2p)^n F_n(T), \quad T = p|\mathbf P - \mathbf r_g|^2, \qquad R^{n}_{t+1,u,v} = t R^{n+1}_{t-1,u,v} + X_{PC} R^{n+1}_{tuv} , $$

and likewise in $u$, $v$. The $E$ coefficients do not depend on the grid point;
only $R$ does, and its recursion is a chain of multiply-adds that the code
vectorizes over a tile of 8 points (`Md3c1e::pair_block`; the COSX caller is
`cosx_a.rs::a_matrix_at_point`).
The Boys function $F_n(T) = \int_0^1 t^{2n} e^{-T t^2}\,dt$ comes from a
Taylor table on nodes $T_k = k/8$ with downward recursion below $T = 35$ and
from the asymptotic form with upward recursion above (`boys`, `BOYS_T_SWITCH`).

## Effective core potentials

`ecp.rs` evaluates the semi-local ECP operator on centre $C$,

$$\hat U = U_L(r) + \sum_{l=0}^{L-1}\sum_{m=-l}^{l} |lm\rangle\,\bigl[U_l(r) - U_L(r)\bigr]\,\langle lm|,\qquad U_l(r) = \sum_t d_t\, r^{n_t-2} e^{-\zeta_t r^2},$$

with $r = |\mathbf r - \mathbf C|$, through one of two backends (`FERRIC_ECP_BACKEND`):
`quadrature`, the default (`ecp_quad/`, pure Rust), or `libecpint` (C++ via
`shim/ecp_shim.cc`, kept as a cross-check). Both return Cartesian blocks that
`ecp.rs` transforms to spherical harmonics.

**Angular projection.** A Gaussian on centre $A$, with $\mathbf a = \mathbf A - \mathbf C$,
expands about $C$ through the plane-wave (Bessel) identity

$$e^{-\alpha|\mathbf r-\mathbf a|^2} = 4\pi\, e^{-\alpha(r^2+a^2)} \sum_{\lambda}\, i_\lambda(2\alpha a r)\sum_{\mu} Y_{\lambda\mu}(\hat{\mathbf r})\,Y_{\lambda\mu}(\hat{\mathbf a}),$$

where $i_\lambda$ is the modified spherical Bessel function of the first kind
and $Y_{\lambda\mu}$ are real spherical harmonics. The semi-local (type-2) term
needs $F_{lm}(r) = \int Y_{lm}(\Omega)\,\chi(\mathbf C + r\Omega)\,d\Omega$; its angular
factor does not depend on $r$ and is built once per shell pair, centre and
channel (`ecp_quad/type2.rs`). The local (type-1) term expands the bra-ket
product Gaussian the same way (`ecp_quad/type1.rs`).

**Radial integrals.** The code uses the exponentially scaled
$\tilde k_\lambda(z) = e^{-z} i_\lambda(z)$ (`ecp_quad/bessel.rs`) and combines the
Gaussian exponents into $e^{-p(r-r_0)^2 - K}$ with $p = \alpha+\beta+\zeta$,
$r_0 = (\alpha a + \beta b)/p$ and $K = \alpha a^2 + \beta b^2 - p r_0^2 \ge 0$, so no
factor overflows at long range. Each (primitive pair, ECP term) radial
integral runs on its own window: Gauss-Hermite away from $r = 0$, adaptive
Gauss-Legendre otherwise (`ecp_quad/radial.rs`).

**Derivatives.** Shell derivatives use raised and lowered shells,
$\partial\chi/\partial A_x = 2\alpha\,\chi(i+1) - i\,\chi(i-1)$, and the ECP-centre
derivative follows from translation invariance as $-(\partial_A + \partial_B)$.

The formulation follows McMurchie and Davidson, *J. Comput. Phys.* **44**, 289
(1981); libecpint is Shaw and Hill, *J. Chem. Phys.* **147**, 074108 (2017).
Tests: `tests/ecp_quadrature.rs` (quadrature against libecpint, and negative controls: broken
constructions that must fail), `tests/ecp_matrix.rs`
(spherical blocks against PySCF `ECPscalar`), `tests/ecp_matrix_deriv.rs`,
`tests/ecp_deriv_block.rs`.

## The pVp integral

The spin-free X2C-1e and DKH Hamiltonians need the block

$$ W_{\mu\nu} = \sum_{a \in \{x,y,z\}} \langle \partial_a \chi_\mu | V | \partial_a \chi_\nu \rangle = \langle \chi_\mu | \hat{\mathbf p}\cdot V \hat{\mathbf p} | \chi_\nu\rangle, \qquad V(\mathbf r) = -\sum_C \frac{q_C}{|\mathbf r - \mathbf C|} , $$

(PySCF's `int1e_pnucp`), where $\hat{\mathbf p} = -i\nabla$ and the two factors
of $-i$ and $+i$ from the bra and ket cancel. The code takes $W$ from libint2's
second derivatives of the nuclear-attraction integral. With $\chi_\mu$ on
centre $\mathbf A$ and $\chi_\nu$ on centre $\mathbf B$, the rule of "Derivative
integrals and buffer sizing" gives $\partial\chi_\mu/\partial A_a = -\partial_a\chi_\mu$, so
the mixed second derivative is

$$ \frac{\partial^2 V_{\mu\nu}}{\partial A_a\, \partial B_a} = \langle \partial_a\chi_\mu | V | \partial_a\chi_\nu \rangle , \qquad W_{\mu\nu} = \sum_a \frac{\partial^2 V_{\mu\nu}}{\partial A_a\, \partial B_a} . $$

The derivative with respect to the charge positions does not enter. Libint2
reports the bra and ket centres separately even when both functions sit on the
same atom, so the identity holds for same-centre pairs as well. In the second
derivative engine's coordinate list (bra $0..2$, ket $3..5$, then the charges),
$W$ is the sum of the three blocks $(a, 3 + a)$ (`pvp.rs::pvp`). The result is
linear in the charges, so they are passed to the engine in chunks
(`CHARGE_CHUNK`), which bounds the second-derivative buffer.
`pvp.rs::pvp_nuclear` uses the same effective charges as `nuclear`.

The bra-bra form $\langle \nabla^2\chi_\mu|V|\chi_\nu\rangle$ is not $W$: it
differs by $\langle\chi_\mu|\nabla V\cdot\nabla\chi_\nu\rangle$, which is
nonzero whenever a charge is away from the centre. For primitive $s$
functions on one centre, $W$ has the closed form

$$ W = -4ab\,Z\,N_aN_b\, I_2, \qquad I_2 = \frac{3\pi^{3/2}\operatorname{erf}(\sqrt p D)}{2Dp^{5/2}} - \frac{\pi e^{-pD^2}}{p^2}, \qquad p = a + b , $$

for a charge $Z$ at distance $D$ (limit $I_2 \to 2\pi/p^2$ at $D = 0$), because
$\nabla\chi_a\cdot\nabla\chi_b = 4ab\,r^2 e^{-pr^2}N_aN_b$ and $r^2e^{-pr^2} = -\partial_p e^{-pr^2}$.
`scripts/pvp_ss_sympy.py` derives it, checks it against a radial quadrature of
the definition and checks the $p$-shell trace; `tests/pvp.rs` checks the code
against these closed forms.

## Basis functions on a grid

`ao_grid.rs` evaluates basis functions, gradients and Hessians at points. With
coordinates relative to the shell centre, a contracted function factors into an
angular polynomial and a radial function of $r^2$,

$$ \chi(\mathbf r) = A(x, y, z)\, \mathcal R(r^2), \qquad \mathcal R(r^2) = \sum_k d_k N(\alpha_k, l)\, e^{-\alpha_k r^2}, $$

with $A$ a Cartesian monomial or a solid harmonic. The code accumulates
$\mathcal R$, $\mathcal R' = d\mathcal R/d(r^2) = -\sum_k d_k N \alpha_k e^{-\alpha_k r^2}$
and $\mathcal R'' = \sum_k d_k N \alpha_k^2 e^{-\alpha_k r^2}$ in one pass over the
primitives (`radial_and_d`, `radial_and_d_d2`), and the chain rule gives

$$ \partial_a \chi = (\partial_a A)\,\mathcal R + 2 x_a\, A\, \mathcal R', $$

$$ \partial_a\partial_b \chi = (\partial_a\partial_b A)\,\mathcal R + 2\big[x_a\, \partial_b A + x_b\, \partial_a A + \delta_{ab} A\big]\mathcal R' + 4 x_a x_b\, A\, \mathcal R'' . $$

Components come in libint2 order: $p$ as $x, y, z$, pure shells $m = -l..l$,
Cartesian shells in CCA order. The evaluators support $l \le 4$ and return
`UnsupportedL` above (`eval_basis_on_points`, `eval_basis_and_grad_on_points`,
`eval_basis_grad_hess_on_points`). The Hessian is returned as the full
$3 \times 3$ tensor, so a value-gradient-Hessian evaluation holds 13 planes of
$N \times N_g$ doubles at peak; `check_ao_grid_budget` refuses an evaluation
whose planes exceed the memory budget before it allocates them.

## The terf and terfc kernels

$$ \operatorname{terf}(r; r_0, \omega) = \tfrac12\big[\operatorname{erf}\big(\omega(r - r_0)\big) + \operatorname{erf}\big(\omega(r + r_0)\big)\big], \qquad \operatorname{terfc} = 1 - \operatorname{terf} . $$

The kernels are $\operatorname{terf}(r)/r$ (long range) and
$\operatorname{terfc}(r)/r$ (short range); their sum is $1/r$ exactly.
$\operatorname{terf}$ is odd in $r$ and lies in $[0, 1)$ for $r \ge 0$. At fixed
$\omega$, $r_0 \to 0$ gives $\operatorname{erf}(\omega r)$, and $r_0 \to \infty$
gives $0$.

**The curvature constraint.** Because $\operatorname{terf}$ is odd, the long-range
kernel is regular at the origin:

$$ \frac{\operatorname{terf}(r)}{r} = c_0 + c_2 r^2 + O(r^4), \qquad c_2 = \frac{2\omega^3\,(2\omega^2r_0^2 - 1)}{3\sqrt\pi}\, e^{-\omega^2 r_0^2} . $$

$c_2$ vanishes exactly when $r_0\omega = 1/\sqrt2$. Under that constraint
$\operatorname{terfc}(r)/r = 1/r - c_0 + O(r^4)$: the short-range kernel keeps
the derivative of the Coulomb kernel at short range. `Operator::terf` and
`Operator::terfc` take $r_0$ and set $\omega = 1/(r_0\sqrt2)$;
`Operator::terfc_with_omega` and `Operator::terf_with_omega` take both.
Proof: `scripts/terf_identities_sympy.py` [complement], [curvature].

## Kernels in Fourier space

For a radial kernel $K(r)$ the three-dimensional transform is a sine transform
of $rK(r)$:

$$ \hat K(k) = \int K(r)\, e^{-i\mathbf k\cdot\mathbf r}\, d\mathbf r = \frac{4\pi}{k}\int_0^\infty r K(r)\sin(kr)\, dr . $$

$\operatorname{terf}(r)$ is the odd function $\operatorname{erf}(\omega r)$ convolved
in one dimension with $\tfrac12[\delta(r - r_0) + \delta(r + r_0)]$, and that
convolution multiplies the sine transform by $\cos(kr_0)$. So

| Kernel | $\hat K(k)$ |
|---|---|
| $1/r$ | $4\pi/k^2$ |
| $\operatorname{erf}(\omega r)/r$ | $(4\pi/k^2)\, e^{-k^2/4\omega^2}$ |
| $\operatorname{erfc}(\omega r)/r$ | $(4\pi/k^2)\,\big(1 - e^{-k^2/4\omega^2}\big)$ |
| $\operatorname{terf}(r)/r$ | $(4\pi/k^2)\cos(kr_0)\, e^{-k^2/4\omega^2}$ |
| $\operatorname{terfc}(r)/r$ | $(4\pi/k^2)\,\big(1 - \cos(kr_0)\, e^{-k^2/4\omega^2}\big)$ |

Two consequences carry the screening chapter.

- $\hat K_{\mathrm{terf}}$ is negative for $kr_0 \in (\pi/2, 3\pi/2)$ and
  periodically beyond; its first zero is at $kr_0 = \pi/2$. By Bochner's
  theorem the terf kernel is not positive definite.
- $\hat K_{\mathrm{terfc}} \ge (4\pi/k^2)(1 - e^{-k^2/4\omega^2}) > 0$: terfc is
  positive definite for every $(r_0, \omega)$. Where $\cos(kr_0) < 0$ it
  exceeds $4\pi/k^2$, so $\operatorname{terfc}(r) \le 1$ pointwise does not
  make $(\rho|\operatorname{terfc}|\rho) \le (\rho|1/r|\rho)$.

Proof: `scripts/screening_bounds_sympy.py` [fourier], [sign], [indefinite].
The transforms carry the factor $1/(2\pi)^3$ of the inverse transform in the
code comments of `schwarz.rs`, which write $\hat K/(2\pi)^3 = \cos(kr_0)e^{-k^2/4\omega^2}/(2\pi^2k^2)$.

## Two-center integrals over the terf kernel

Take $s$ Gaussians $e^{-p|\mathbf r - \mathbf A|^2}$ and
$e^{-q|\mathbf r - \mathbf B|^2}$ at distance $R$. Their interaction through a
radial kernel depends only on the convolution of the two Gaussians, a Gaussian
of exponent $\theta^2 = pq/(p+q)$:

$$ (a|K|b) = \Big(\frac{\pi^2}{pq}\Big)^{3/2} \int \Big(\frac{\theta^2}{\pi}\Big)^{3/2} e^{-\theta^2|\mathbf x - \mathbf R|^2}\, K(x)\, d\mathbf x = \Big(\frac{\pi^2}{pq}\Big)^{3/2} \int \frac{d\mathbf k}{(2\pi)^3}\, e^{-k^2/4\theta^2}\, \hat K(k)\, e^{i\mathbf k\cdot\mathbf R} . $$

For $\hat K = (4\pi/k^2)\cos(kr_0)e^{-k^2/4\omega^2}$ the two Gaussian factors
combine into $e^{-k^2/4\phi^2}$ with

$$ \frac{1}{\phi^2} = \frac{1}{\theta^2} + \frac{1}{\omega^2} = \frac1p + \frac1q + \frac{1}{\omega^2}, $$

the angular integral gives $4\pi\sin(kR)/(kR)$, and
$\sin(kR)\cos(kr_0) = \tfrac12[\sin k(R + r_0) + \sin k(R - r_0)]$ with
$\tfrac{2}{\pi}\int_0^\infty e^{-k^2/4\phi^2}\sin(kx)/k\,dk = \operatorname{erf}(\phi x)$.
The result is the kernel itself at sharpness $\phi$:

$$ (a|\operatorname{terf}|b) = \Big(\frac{\pi^2}{pq}\Big)^{3/2} \frac{\operatorname{terf}(R; r_0, \phi)}{R} = \frac{2\pi^{5/2}}{pq\sqrt{p+q}}\, \frac{\phi}{\theta}\, G_0(S, s), \qquad S = \phi^2R^2, \quad s = \phi^2 r_0^2, $$

$$ G_0(S, s) = \frac{\sqrt\pi}{4\sqrt S}\Big[\operatorname{erf}\big(\sqrt S + \sqrt s\big) + \operatorname{erf}\big(\sqrt S - \sqrt s\big)\Big], \qquad G_0(0, s) = e^{-s} . $$

At $s = 0$ this is the erf-kernel integral, and with $\phi \to \theta$ the
Coulomb integral $2\pi^{5/2}F_0(\theta^2R^2)/(pq\sqrt{p+q})$. The Coulomb
prefactor is common to every kernel; the kernel enters through $\phi/\theta$
and through $G_0$ in place of $F_0$.

**Higher angular momentum.** McMurchie-Davidson builds every other integral
from derivatives of the $s$ integral with respect to $\mathbf R$ through the
$R^n_{tuv}$ recursion of "McMurchie-Davidson three-center one-electron
integrals", seeded with $R^n_{000} = (-2\alpha)^n F_n(\alpha R^2)$. The
dependence on $R$ is through $S$ alone at fixed $s$, so the terf integrals use
the same recursion with reduced exponent $\phi^2$ and the seed

$$ R^n_{000} = (-2\phi^2)^n\, \frac{\phi}{\theta}\, G_n(S, s), \qquad G_n(S, s) = \Big(-\frac{\partial}{\partial S}\Big)^n G_0(S, s) . $$

The shim calls $A_n = (\phi/\theta)G_n(S, s)$ the Boys-replacement vector
(`terf_aux`). terfc integrals are Coulomb minus terf, built by the same
McMurchie-Davidson pass so the two share ordering and normalization. The
$R^n_{tuv}$ recursion is linear in its seeds and does not contain the exponent,
so the 3-centre values and derivatives form the difference on the seeds,

$$ R^n_{000} = (-2\theta^2)^n F_n(\theta^2 R^2) - (-2\phi^2)^n \frac{\phi}{\theta} G_n(S, s) , $$

and run one recursion and one contraction (`md_terms` in the shim). The
2-centre values subtract two passes.

**The defining series.** The code evaluates $G_n$ from a Poisson series. With
$\pi_S(i) = e^{-S}S^i/i!$, $c_s(i) = \sum_{j \le i} e^{-s}s^j/j!$,
$\nabla f(i) = f(i) - f(i-1)$ and $(2i)!!/(2i+1)!!$,

$$ G_n(S, s) = \sum_{i \ge 0} \frac{(2i)!!}{(2i+1)!!}\, [\nabla^{n}\pi_S]{}(i)\, c_s(i) . $$

$-\partial_S \pi_S(i) = \pi_S(i) - \pi_S(i-1)$, so each $\nabla$ is one
$S$-derivative, and at $s = 0$ the series is the Boys series
$F_n(S) = e^{-S}\sum_i (2S)^i/(2i+1)!!$ differentiated $n$ times. Writing
$c_s = 1 - \tau_s$ with the Poisson tail $\tau_s(i) = \sum_{j > i} e^{-s}s^j/j!$
splits it exactly:

$$ G_n(S, s) = F_n(S) - \Delta_n(S, s), \qquad \Delta_n(S, s) = \sum_{i \ge 0} \frac{(2i)!!}{(2i+1)!!}\, [\nabla^{n}\pi_S]{}(i)\, \tau_s(i) . $$

$\Delta_n$ is the terfc auxiliary itself, and its sum length is set by $s$, not
$S$, because $\tau_s(i)$ decays super-exponentially once $i > s$
(`terf_delta`, `terf_G_tail`; the truncation index is `terf_tail_index`).

**Far field.** For $\sqrt S - \sqrt s = \phi(R - r_0) \gg 1$ both error
functions in $G_0$ are $1$ to within $\operatorname{erfc}(\sqrt S - \sqrt s)$
and $G_n \to F_n(S)$. The shim takes $G_n = F_n(S)$ when $S > 50$ and
$\sqrt S - \sqrt s > 6$ (`terf_far_field`). The second test matters when $\omega$
is free: $S > 50$ alone admits pairs inside the $r_0$ shell.

**The range of $s$.** $\phi \le \omega$, so $s \le (\omega r_0)^2$; under the
curvature constraint $s < 1/2$ for every pair.

**Tables.** $G_{m,n}(S, s)$ is tabulated on four grids, finest first,
covering $S \le 20$ and $s \le 80$ (files `16_4_2.bin`, `8_10_5.bin`,
`4_20_20.bin`, `2_20_80.bin`, named by points per unit, $S_{\max}$ and
$s_{\max}$), with $m < 24$. A lookup takes the finest covering table and
interpolates with a $10 \times 10$ Lagrange stencil on the integer nodes.
The second index $n$ is an $s$-derivative index; energy integrals use $n = 0$.
Outside every table the tail form above is summed directly.
Proof: `scripts/terf_identities_sympy.py` [two-center], [series], [boys-shape],
[s-zero], [tail-form], [far-field], [s-bound], [terfc-split].

## Derivative integrals of terf and terfc

`Engine::new_3center_deriv` and `new_2center_deriv` with `Operator::terf` or
`Operator::terfc` return the first nuclear derivatives of $(P|\mu\nu)$ and
$(P|Q)$ in libint2's block order (9 and 6 blocks, "Derivative integrals and
buffer sizing"). libint2 has no terf kernel, so the shim computes them with
its own McMurchie-Davidson pass (`md_terms::eri3_derivs` for 3 centres,
`compute_cart_eri2_deriv` for 2).

**Term lists.** The 3-centre kernels (values and derivatives) write each
Cartesian component as a list of Hermite terms $(c, t, u, v)$, and store
$(t, u, v)$ as its flat index $(tS + u)S + v$ in an $R$ cube of side
$S = L + 1$. The index is linear, so $R_{t'+t,\,u'+u,\,v'+v}$ is the cube at
the sum of the two indices, and an integral is a double sum over the aux list
and the orbital-pair list. Pair lists are built once per primitive pair and
aux lists once per shell triple. The kernels they replaced
(`compute_cart_eri3`, `compute_cart_eri3_deriv`) stay as the reference the
tests compare against.

**The Hermite derivative.** A Cartesian Gaussian on $\mathbf A$ differentiates as

$$ \frac{\partial}{\partial A_x}\big(x_A^{i} e^{-a x_A^2}\big) = 2a\, x_A^{i+1} e^{-a x_A^2} - i\, x_A^{i-1} e^{-a x_A^2} . $$

The product $x_A^i x_B^j e^{-a x_A^2 - b x_B^2}$ expands in Hermite Gaussians
$\Lambda_t(x; P)$ with coefficients $E^{ij}_t$. The derivative is the same
expansion with

$$ \frac{\partial E^{ij}_t}{\partial A_x} \to 2a\,E^{i+1,j}_t - i\,E^{i-1,j}_t, \qquad t \le i + j + 1 , $$

over the same $\Lambda_t$ (`hermite_deriv_first`). The ket side is unchanged,
so the contraction with $R_{tuv}$ needs $R$ one order higher: total order
$L = l_P + l_A + l_B + 1$. An auxiliary function on $\mathbf P$ alone has no
partner, and its derivative uses $E^{i\pm1,0}$ the same way.

**The third centre.** The derivatives with respect to all centres sum to
zero, so the shim computes $\partial_P$ and $\partial_A$ and sets
$\partial_B = -(\partial_P + \partial_A)$. In the 2-centre case
$\partial_Q = -\partial_P$.

**The kernel.** The kernel enters only through $R^n_{000}$. For terf it is
$(-2\phi^2)^n(\phi/\theta)G_n(S, s)$ ("Two-center integrals over the terf
kernel"), read from the tables at $n \le L$. For terfc the shim forms
$R_{\mathrm{terfc}} = R_{\mathrm{Coulomb}} - R_{\mathrm{terf}}$ before the
contraction, from the combined seeds in 3 centres and from two $R$ tables in
2 (`md_kernel_R`), so both share the $E$ coefficients and the prefactor. The derivative reads $G$ one order higher than the value. The
tables hold $m < 24$, so the engine refuses $L + 1 \ge 24$ (with s..g
auxiliaries and s..f orbitals $L \le 11$).

**Screening.** The derivative loop of the RI-MP2 gradient screens nothing:
every $(P|\mu\nu)$ shell triple is evaluated. The terfc energy is also
unscreened (no `eri3_screen_thresh`), so the gradient differentiates the same
sum. Schwarz bounds are not used for terfc derivatives.

**Checks.** `crates/ferric-integrals/tests/terfc_deriv_integrals.rs`:

| Check | Systems | Measured | Bar |
|---|---|---|---|
| Coulomb kernel through the same MD derivative vs libint2 | water, cc-pVTZ + cc-pVTZ-RI, every class to $(g\vert ff)$ and $(g\vert g)$ | 3.8e-14 | 1e-10 |
| terf and terfc derivatives vs five-point FD of the engine's values ($\omega r_0 = 1/\sqrt2$ and 4) | water, cc-pVDZ + cc-pVDZ-RI, every class | 2.6e-11 | 10× the Coulomb FD floor (3.9e-11) |
| terfc derivative = libint2 Coulomb − table terf | same | 2.1e-14 | 1e-10 |

`crates/ferric-integrals/tests/terfc_md_terms_vs_reference.rs` compares the
term-list kernels with the reference kernels on every class of water /
cc-pVTZ + cc-pVTZ-RI: terf values are bit-identical, terfc values and all
derivatives agree to 5.1e-15 (bar $10^{-13}$ times the largest element).
Flipping the sign of the terf seed, swapping $u$ and $v$ in the pair index,
or dropping $-i\,E^{i-1,j}$ each fails it.

Dropping the $-i\,E^{i-1,j}$ term fails all three. Writing
$R_{\mathrm{Coulomb}} + R_{\mathrm{terf}}$ fails the second and third.
Proof of the $s$-type closed forms, their derivatives against a $k$-space
quadrature and the Hermite identity for $i, j \le 3$:
`scripts/terfc_rimp2_gradient_sympy.py` [K], [H].

## Yukawa and Slater-geminal kernels

| Kernel | `Operator` | libint2 class | Parameter |
|---|---|---|---|
| $e^{-\zeta r}/r$ | `Operator::yukawa` | `stg_x_coulomb` | $\zeta > 0$ |
| $e^{-\zeta r}$ | `Operator::slater_geminal` | `stg` | $\zeta > 0$ |
| $-\gamma^{-1}e^{-\gamma r} \approx \sum_i c_i e^{-\alpha_i r^2}$ | `Operator::stg` | `cgtg`, `cgtg_x_coulomb`, `delcgtg2` | $\gamma$ |

The first two are exact kernels; libint2 evaluates them through its
`TennoGmEval` core, which replaces the Boys function by a two-variable function
of $T = \rho R^2$ and $U = \zeta^2/4\rho$, with $\rho$ the reduced exponent of
the bra and ket. Its interpolation covers $U \in [10^{-7}, 10^{3}]$; outside
that range the result depends on the libint2 build, which `Operator::yukawa`
documents case by case. $\zeta \to 0$ turns the Yukawa kernel into Coulomb.
The third is a fit: the six-term Tew-Klopper least-squares fit of $e^{-r}$,
scaled $r \to \gamma r$, so $\alpha_i \to \gamma^2\alpha_i$, with the
$-1/\gamma$ of the F12 geminal folded into the coefficients. One fitted
geminal is one libint2 engine (`Engine::new_2e_geminal`). None of the three is
admitted by the Schwarz table builder.

## The Schwarz bound

For a positive-definite kernel, $(f|\hat w|g)$ is an inner product on charge
distributions, and the Cauchy-Schwarz inequality gives

$$ |(\mu\nu|\hat w|\lambda\sigma)| \le Q_{\mu\nu} Q_{\lambda\sigma}, \qquad Q_{\mu\nu} = (\mu\nu|\hat w|\mu\nu)^{1/2} . $$

A kernel is positive definite exactly when $\hat K(k) > 0$ for $k > 0$
(Bochner). From the table in "Kernels in Fourier space", Coulomb, erf, erfc and
terfc qualify and terf does not: on terf, $(\rho|\rho)$ can be negative and
$Q$ is not real. Ferric's table builder accepts Coulomb, erf, erfc and terfc
and refuses terf as invalid (`schwarz.rs::check_operator_supported`).

The shell-pair factor is $Q(s_1, s_2) = \max_{\mu \in s_1, \nu \in s_2} |(\mu\nu|\mu\nu)|^{1/2}$,
and $|(s_1 s_2|s_3 s_4)| \le Q(s_1,s_2)\,Q(s_3,s_4)$ element by element
(`schwarz.rs::schwarz`). For the three-index integrals,
$|(P|\mu\nu)| \le Q_3(P)\, Q(\mu,\nu)$ with $Q_3(P) = \max_a |(P_a|P_a)|^{1/2}$
(`schwarz3_aux`).

A bound must never be below the integral it bounds, so the table is built with
engine precision $0$ (libint2's own skip disabled, `SCHWARZ_TABLE_PRECISION`)
and every factor is floored at $10^{-100}$ (`SCHWARZ_Q_FLOOR`). A screen that
compares `estimate > threshold` then keeps every pair at threshold $0$, which is
the trivial limit. terfc factors come from an interpolated engine, so each is
multiplied by $1 + 10^{-5}$ (`bound_inflation`, `TERFC_TABLE_Q_INFLATION`), an
empirical margin, and the terfc bound is refused outside the parameter box it
was checked in (`check_terfc_parameters_validated`: $r_0 \in [0.3, 2.5]$ Å,
$0 < r_0\omega \le 8.9$, $l \le 4$).

**Schwarz does not see attenuation.** $Q_{\mu\nu}$ is the self-interaction of
one pair density, so it does not depend on where the other pair is, and the
Schwarz product of two pairs is the same at every separation. For an $s$ pair
density of exponent $P$ the self-interactions are
$(\rho|1/r|\rho) \propto \theta$ and $(\rho|\operatorname{erf}|\rho) \propto \phi$
with $\theta^2 = P/2$ and $\phi^2 = 1/(2/P + 1/\omega^2)$, so

$$ \frac{Q^2_{\mathrm{erfc}}}{Q^2_{\mathrm{Coulomb}}} = 1 - \frac{\phi}{\theta} \;\longrightarrow\; 1 \quad (P/\omega^2 \to \infty) . $$

A compact pair is not attenuated at all in its own factor; a diffuse pair is.
The true erfc integral between two separated pairs decays like a Gaussian in
their distance, but no factor of the bound does. The attenuation reaches the
bound only through the per-pair factors, so erfc and terfc screen almost the
same set of quartets as Coulomb.
Proof: `scripts/screening_bounds_sympy.py` [schwarz], [blind].

## The combined Schwarz bound (CSB)

If the kernel is positive pointwise, $w(r) > 0$ for $r > 0$, the four-center
integral is also an inner product of the two-electron products
$\chi_\mu(1)\chi_\lambda(2)$ and $\chi_\nu(1)\chi_\sigma(2)$ under the weight
$w(r_{12})$. That gives two more bounds, and their minimum with Schwarz
(Thompson and Ochsenfeld, *J. Chem. Phys.* **147**, 144101 (2017), Eqs. 6 to 8):

$$ |(\mu\nu|\lambda\sigma)| \le \min\big\{ Q_{\mu\nu}Q_{\lambda\sigma},\; M_{\mu\lambda}M_{\nu\sigma},\; M_{\mu\sigma}M_{\nu\lambda} \big\}, \qquad M_{\mu\lambda} = (\mu\mu|\lambda\lambda)^{1/2} . $$

$M$ depends on the distance between $\mu$ and $\lambda$, which $Q$ does not
(`csb.rs`). The $M$ terms need pointwise positivity, not positive
definiteness: with $F = \chi_\mu(1)\chi_\lambda(2)$ and
$G = \chi_\nu(1)\chi_\sigma(2)$, $\iint F G\, w$ is an inner product on
$L^2(\mathbb R^6, w)$ whenever $w \ge 0$. So $M_{\mu\lambda}M_{\nu\sigma}$ bounds
terf integrals too, but the full CSB minimum contains $Q_{\mu\nu}Q_{\lambda\sigma}$,
which needs positive definiteness, and is invalid for terf.

| Kernel | $QQ$ term | $MM$ terms | `csb_m_table` |
|---|---|---|---|
| Coulomb, erf, erfc | valid | valid | built |
| terfc | valid | valid | refused: no interpolation-error budget yet |
| terf | invalid | valid | refused |
Proof: `scripts/screening_bounds_sympy.py` [csb].

## Distance-dependent bounds (QQR)

For well-separated pair densities the leading term of the Coulomb interaction is
the product of their charges over the distance $R$ between their centres. The
QQR bound multiplies Schwarz by an envelope in $R$:

$$ |(ij|kl)| \le Q_{ij}Q_{kl}\,\min\Big(1, \frac{1.10\,(\epsilon_{ij} + \epsilon_{kl})}{R_{\mathrm{eff}}}\Big), \qquad R_{\mathrm{eff}} = \max(0, R - \epsilon_{ij} - \epsilon_{kl}), $$

with the pair centre $(\alpha_i\mathbf R_i + \alpha_j\mathbf R_j)/(\alpha_i + \alpha_j)$
and extent $\epsilon_{ij} = (\alpha_i + \alpha_j)^{-1/2}$ from the most diffuse
exponent of each shell (`ferric-scf/src/qqr.rs::QqrBounds`). The three-index
version replaces one pair by an auxiliary shell with its own centre and extent
(`qqr3.rs::QqrBounds3::estimate3`). At $R_{\mathrm{eff}} = 0$ (overlapping
clouds) the envelope is $1$ and the bound is Schwarz.

**Primitive $s$ pairs.** For pair densities of exponents $P_1$, $P_2$ and
charges $q_1$, $q_2$, $(12|12) = q_1q_2\operatorname{erf}(\theta R)/R \le q_1q_2/R$
and $Q_1Q_2 = q_1q_2\sqrt{2/\pi}\,(P_1P_2)^{1/4}$, so, with
$\epsilon = P^{-1/2}$ and $\sqrt{\epsilon_1\epsilon_2} \le (\epsilon_1 + \epsilon_2)/2$,

$$ \frac{|(12|12)|}{Q_1Q_2} \le \sqrt{\frac{\pi}{2}}\, \frac{\sqrt{\epsilon_1\epsilon_2}}{R} \le \sqrt{\frac{\pi}{8}}\, \frac{\epsilon_1 + \epsilon_2}{R} \le \frac{1.10\,(\epsilon_1 + \epsilon_2)}{R_{\mathrm{eff}}} , $$

since $\sqrt{\pi/8} = 0.627$ and $R_{\mathrm{eff}} < R$. For these pairs the
envelope is a proven bound. For contracted and higher-$l$ pairs the
neglected dipole and quadrupole terms are not covered by this argument, and
the factor $1.10$ is a measured margin.

**No operator factor.** The envelope is the Coulomb one for every kernel, and
attenuation enters only through the erfc Schwarz factors. QQR therefore shares
the blindness of "The Schwarz bound": on the pair lists, which are built from
diagonal estimates $(ij|ij)$ with $R = 0$, QQR is identical to Schwarz.
Proof: `scripts/screening_bounds_sympy.py` [qqr-s], [qqr-erfc].

## Significant pair lists

LinK walks shell pairs from two lists (`ferric-scf/src/pairs.rs`).

**Geometry pairs.** A quartet $(ij|kl)$ is computed only if
$Q_{ij}Q_{kl}|D| \ge \tau$ for the density element it multiplies. With
$Q_{kl} \le Q_{\max}$ and $|D| \le D_{\max}$, a pair can contribute only if
$Q_{ij}Q_{\max}D_{\max} \ge \tau$. `SignificantPairs::build` keeps the pairs with

$$ Q_{ij}\, Q_{\max} > \tau, \qquad Q_{\max} = \max_{kl} Q_{kl} , $$

the density-free test of PySCF's `q_cond`. It is the necessary condition when
$D_{\max} \le 1$. Where density elements exceed 1 in magnitude, which a
non-orthogonal basis allows, the list is stricter than necessary by the factor
$D_{\max}$: a pair with $\tau/D_{\max} < Q_{ij}Q_{\max} \le \tau$ is dropped
although a quartet through it can pass the per-quartet screen. How often that
happens has not been measured.

**Density pairs.** A density block $D_{j\sigma}$ enters exchange only through
quartets $(ij|\sigma l)$, each bounded by $Q_{ij}Q_{\sigma l}|D_{j\sigma}|$.
With $q_{\max}(x) = \max_y Q_{xy}$,

$$ \max|D_{j\sigma}|\; q_{\max}(j)\; q_{\max}(\sigma) > \tau $$

is the tightest necessary condition at pair granularity
(`DensityPairs::build`). It prunes little: $q_{\max}(x)$ is nearly always
realized by a tight core pair, so the product $q_{\max}(j)q_{\max}(\sigma)$ is
close to a molecule-wide constant. The locality is recovered per quartet,
where `LinkK::build` screens on the density blocks of the four shells in hand
(`DensityScreen::FourPairK`). The list bounds the ket loop; the quartet screen
bounds the work. Both lists are sorted, and LinK intersects them with
`intersect_sorted`.

## Density fitting: the fit and its error

Fitting a product density $\rho_x$ ($x$ an orbital or AO pair) in auxiliary
functions $\chi_P$ under a kernel $\hat w$ minimizes
$(\rho_x - \tilde\rho_x|\hat w|\rho_x - \tilde\rho_x)$ over
$\tilde\rho_x = \sum_P c_{Px}\chi_P$, which gives

$$ c_x = V_w^{-1} A_x, \qquad A_{Px} = (P|\hat w|x), \qquad (V_w)_{PQ} = (P|\hat w|Q) . $$

The residual $\delta_x = \rho_x - \tilde\rho_x$ is $\hat w$-orthogonal to every
$\chi_P$. When the physical kernel is the fitting kernel, the fitted
interaction $(\tilde\rho_x|\hat w|\tilde\rho_y) = A_x^{\mathsf T}V_w^{-1}A_y$ has
the error

$$ (x|\hat w|y) - A_x^{\mathsf T}V_w^{-1}A_y = (\delta_x|\hat w|\delta_y) , $$

second order in the residuals, and the diagonal error is stationary in $c$
(Dunlap). Standard RI writes this as $B^{\mathsf T}B$ with $B = MA$ and any
$M^{\mathsf T}M = V^{-1}$; ferric's $M = L^{-1}$ from $V = LL^{\mathsf T}$ is
not symmetric, so $MM \ne V^{-1}$, and a formula that is not a Gram product
must solve with $V$ instead of applying $M$ twice.

## Robust fitting with a mismatched metric

When the fit uses a kernel $\hat w$ different from the physical kernel
$\hat C$, the naive substitution $J_x^{\mathsf T}V_w^{-1}J_y$, with
$J_{Px} = (P|\hat C|x)$, has no cancelling term: for an attenuated $\hat w$,
$V_w$ is small and $V_w^{-1}$ large. The robust (Dunlap) form fits with
$\hat w$ and evaluates the physical interaction symmetrically:

$$ (x|\hat C|y) \approx G_{xy} = (\tilde\rho_x|\hat C|\rho_y) + (\rho_x|\hat C|\tilde\rho_y) - (\tilde\rho_x|\hat C|\tilde\rho_y) = c_x^{\mathsf T}J_y + J_x^{\mathsf T}c_y - c_x^{\mathsf T}V_C\,c_y , $$

with $c_x = V_w^{-1}A_x$ and $(V_C)_{PQ} = (P|\hat C|Q)$. Expanding
$(\delta_x|\hat C|\delta_y)$ shows, for any coefficients $c$,

$$ (x|\hat C|y) - G_{xy} = (\delta_x|\hat C|\delta_y) , $$

so the error is bilinear in the residuals whatever kernel produced them, and
the diagonal error is $\ge 0$ because $\hat C$ is positive definite. With
$\hat w = \hat C$, $G = J^{\mathsf T}V_C^{-1}J$, which is standard RI. With
$\hat w \ne \hat C$ the coefficients are no longer the $\hat C$-optimal ones,
the error is not stationary in $c$, and it grows with the mismatch while
staying bounded.

`ferric-mp2/src/rimp2.rs::ri_mp2_robust_attenuated_metric` builds $J$, $V_C$
over the physical operator and $A$, $V_w$ over `metric_op`, solves
$V_w c = A$ by Cholesky (`solve_metric_system`), forms the $(n_on_v)^2$ matrix
$G$ and passes it to `spin_components_from_g`. A Cholesky failure of $V_w$
(an attenuated metric that has lost rank in a Coulomb-fitted auxiliary basis)
is an error.
Proof: `scripts/robust_fit_sympy.py` [residual], [collapse], [cholesky],
[stationary], [model], [diverges], [psd-diag].

## Three-index storage

`ThreeIndexSource` holds the raw tensor $(P|\mu\nu)$ for one operator and
serves it in blocks of auxiliary rows; consumers apply their own metric. The
backend is chosen at build time against the memory budget:

| Backend | Resident | Per pass |
|---|---|---|
| in core | $N_{\mathrm{aux}}N^2$ doubles | none |
| in core, packed | $N_{\mathrm{aux}}N(N+1)/2$ doubles (the $\mu\nu$ triangle) | unpack a block |
| in core, pair screened (RI-J only) | the Schwarz-kept $\mu\nu$ pairs | none |
| in core, single precision (RI-J only, `ri_j_precision = "f32"`) | the packed triangle or the kept pairs, as f32 | none; f64 accumulation |
| disk spill | one block, the tensor on disk | read the file |
| recompute | one block | recompute the integrals |

`[scf] jk_storage` selects the RI-J/K policy (`auto`, `memory`, `disk`,
`direct`, `split`); under `auto` the first memory tier that fits wins, and
between spill and recompute the choice comes from two timings taken on the
machine at build time (`three_index_source/jk_storage.rs`). A source may hold
only a band of auxiliary rows (`build_band`, for distributed runs) and may
carry a QQR-3 screen that must stay fixed for its lifetime, so a recomputed
block skips exactly the triples the first build skipped
(`build_band_screened`).

`ThreeIndexSource::build_dressed` forms $\sum_Q M_{PQ}(Q|\mu\nu)$ by streaming
the raw source. Two constants keep that sum independent of the thread count
and of the backend: output rows are split on a grid of 64 rows
(`DRESS_ROW_BLOCK`), an even height, because an odd split changes the vector
lanes of the matrix product's accumulation, and the contracted index is summed
in blocks of 128 (`DRESS_K_BLOCK`), a coarse pairwise summation.
