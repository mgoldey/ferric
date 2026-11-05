# IEF-PCM Theory

This crate implements the isotropic integral-equation-formalism polarizable
continuum model (IEF-PCM) on a discretized molecular cavity. Every equation
below is the one the code executes; references are `path:line` relative to
`crates/ferric-pcm/src/` unless another crate is named.

Identities marked **[sympy n]** are proven (exact arithmetic) or checked by
claim *n* of [`scripts/pcm_ief_sympy.py`](../../scripts/pcm_ief_sympy.py),
which exits nonzero on any mismatch and carries a negative control per claim.
The discrete sphere limits are also Rust tests in `tests/ief_limits.rs`.

Units are Hartree atomic units (Bohr, Hartree, charge in $e$). $i,j,k$ index
tesserae, $\mu,\nu$ AO basis functions, $A,B$ atoms. $N$ is the tessera count.

## 1. Cavity

One sphere per non-ghost atom, radius $R_A = s_{\mathrm{vdW}}\, r_{\mathrm{mB}}(Z_A)$
with $s_{\mathrm{vdW}} = 1.2$ (`cavity.rs:131`) and $r_{\mathrm{mB}}$ the
modified Bondi radius: Bondi 1964 except H = 1.10 Å and B = 1.92 Å
(`radii.rs:97` `modified_bondi_radius_bohr`; PySCF's `modified_Bondi` values).
Each sphere carries a Lebedev grid $\{\hat n_g, w_g\}$ with
$\sum_g w_g = 1$ (`ferric_quadrature::lebedev`); tessera $k$ on sphere $A$ is
(`cavity.rs:147` `build_cavity`)

$$ \mathbf r_k = \mathbf R_A + R_A \hat n_g, \qquad \hat{\mathbf n}_k = \hat n_g \ \text{(outward)}, \qquad a_k = 4\pi R_A^2\, w_g\, \mathrm{swf}_k , $$

$$ \xi_k = \frac{\Xi(N_{\mathrm{Leb}})}{R_A\sqrt{4\pi w_g}}, \qquad \mathrm{swf}_k = \prod_{B\ne A} h\!\left(\frac{|\mathbf r_k-\mathbf R_B| - R^{\mathrm{in}}_B}{R^{\mathrm{sw}}_B}\right), $$

with $h$ the quintic smoothstep of Lange & Herbert (JCP 133, 244111 (2010),
eq. 3.19; `cavity.rs` `switch_h`), $R^{\mathrm{sw}}_B = R_B\sqrt{14/N_{\mathrm{Leb}}}$
(`cavity.rs:174`) and $\Xi$ the per-Lebedev-order Gaussian-width table
`matrices.rs:88` `gaussian_xi_table` (the values of PySCF's `pcm.py` `XI`). Tesserae with $w_g\,\mathrm{swf}_k \le 10^{-16}$ are
dropped (`cavity.rs:242`). A lone sphere has $\mathrm{swf}_k = 1$ and total
area exactly $4\pi R_A^2$. Ferric's cavity has no interstitial spheres
(GEPOL "added spheres").

`PcmConfig::cavity` (`config.rs`) replaces this construction with an
explicit tessera list; `crates/ferric-scf/tests/validation_pcm.rs` uses it
to feed PySCF's cavity through ferric's solver.

## 2. Boundary-element matrices $S$ and $D$

Notation: $\mathbf r_{ij} = \mathbf r_i - \mathbf r_j$, $r_{ij} = |\mathbf r_{ij}|$,
and the normal is taken at the **source** tessera $j$:

$$ \nu_{ij} = \mathbf r_{ij}\cdot\hat{\mathbf n}_j . $$

$D_{ij} = \hat{\mathbf n}_j\cdot\nabla_{\mathbf r_j} (1/|\mathbf r_i-\mathbf r_j|)$ is
the potential at $i$ of a unit dipole layer at $j$; it is not symmetric. With
this sign a closed surface gives $\sum_j D_{ij} a_j \to -2\pi$ for $i$ on the
surface (Gauss), which is what the $-2\pi$ eigenvalue below expresses.

### 2.1 `SdKind::GaussianSmeared` (default)

Each tessera charge is a normalized Gaussian of exponent $\xi_k^2$
(`matrices.rs:126` `build_s_d_kind`, branch at `:161`), with
$\xi_{ij} = \xi_i\xi_j/\sqrt{\xi_i^2+\xi_j^2}$:

$$ S_{ij} = \frac{\operatorname{erf}(\xi_{ij} r_{ij})}{r_{ij}}, \qquad
   D_{ij} = \frac{S_{ij}\,\nu_{ij}}{r_{ij}^2} - \frac{2}{\sqrt\pi}\,\frac{\xi_{ij} r_{ij}\, e^{-\xi_{ij}^2 r_{ij}^2}\,\nu_{ij}}{r_{ij}^3}
   \quad (i\ne j), $$

$$ S_{ii} = \frac{\xi_i\sqrt{2/\pi}}{\mathrm{swf}_i}, \qquad
   D_{ii} = -\frac{\xi_i\sqrt{2/\pi}}{2R_i} \quad (`matrices.rs:168`,\ `:169`). $$

### 2.2 `SdKind::PointCharge`

Bare point charges (`matrices.rs:132`): $S_{ij} = 1/r_{ij}$,
$D_{ij} = \nu_{ij}/r_{ij}^3$, $S_{ii} = 1.0694\sqrt{4\pi/a_i}$ (`matrices.rs:80`),
$D_{ii} = -S_{ii}/(2R_i)$ (`matrices.rs:137`).

### 2.3 Why $D_{ii} = -S_{ii}/(2R_i)$

On a sphere of radius $R$, with $\cos\gamma = \hat n\cdot\hat n'$,
$|\mathbf r-\mathbf r'|^2 = 2R^2(1-\cos\gamma)$ and
$(\mathbf r-\mathbf r')\cdot\hat n' = -R(1-\cos\gamma)$, so the two kernels are
proportional **[sympy 1]**:

$$ \frac{(\mathbf r-\mathbf r')\cdot\hat n'}{|\mathbf r-\mathbf r'|^3} = -\frac{1}{2R}\,\frac{1}{|\mathbf r-\mathbf r'|}. $$

Both diagonal formulas apply this to the self-term of the tessera's own
sphere. By Funk–Hecke the continuum operators are diagonal in $Y_{lm}$
**[sympy 2]**:

$$ \hat S\, Y_{lm} = \frac{4\pi R}{2l+1} Y_{lm}, \qquad \hat D\, Y_{lm} = -\frac{2\pi}{2l+1} Y_{lm}, $$

where $\hat S\sigma = \int \sigma(\mathbf r')/|\mathbf r-\mathbf r'|\,dS'$ and
$\hat D$ likewise with the double-layer kernel; discretely
$\hat S \leftrightarrow S A$, $\hat D \leftrightarrow D A$ with
$A = \operatorname{diag}(a_k)$.

## 3. The IEF-PCM operators $K$ and $R$

The isotropic IEF equation (Cancès, Mennucci & Tomasi, JCP 107, 3032 (1997))

$$ \left[ 2\pi\frac{\varepsilon+1}{\varepsilon-1} I - DA \right] S\,q = -\left( 2\pi I - DA\right) v $$

is multiplied through by $f/2\pi$ with

$$ f(\varepsilon) = \frac{\varepsilon-1}{\varepsilon+1} \quad (`matrices.rs:227`) , $$

giving (`matrices.rs:215` `build_k_r`, `:242`, `:246`)

$$ K = S - \frac{f}{2\pi}\, DA\, S = \left(I - \tfrac{f}{2\pi}DA\right) S, \qquad
   R = -f\left(I - \frac{DA}{2\pi}\right), \qquad K q = R v . $$

$DA$ is $D$ with column $j$ scaled by $a_j$. `build_k_r` rejects
$\varepsilon \le 1$ and non-finite $\varepsilon$ (vacuum has no reaction field).
$K$, $R$ depend on geometry and $\varepsilon$ only and are built once per SCF
(`lib.rs:110` `PcmContext::from_tesserae`).

## 4. Symmetrized charge solve and energy

$Q = K^{-1}R$ is not symmetric (because $D$ is not). The code applies its
symmetric part (`solver.rs:58` `solve_pcm_charges`):

$$ q = \tfrac12\left[ K^{-1} R v + R^{\mathsf T} (K^{\mathsf T})^{-1} v \right] = \tfrac12 (Q + Q^{\mathsf T})\, v \qquad (`solver.rs:73`,\ `:85`,\ `:87`), $$

two LU solves ($Kx = Rv$ and $K^{\mathsf T}y = v$). The energy is

$$ E_{\mathrm{pcm}} = \tfrac12\, q\cdot v = \tfrac12\, v^{\mathsf T} Q v \qquad (`solver.rs:88`), $$

identical to the unsymmetrized quadratic form **[sympy 5]**. The order of
the second term matters: $(K^{\mathsf T})^{-1}R^{\mathsf T}v = (RK^{-1})^{\mathsf T}v$
has quadratic form $v^{\mathsf T}RK^{-1}v \ne v^{\mathsf T}K^{-1}Rv$ (control in
**[sympy 5]**; 5% of $E_{\mathrm{pcm}}$ on water/STO-3G, pinned by
`solver.rs` `symmetrized_energy_equals_unsymmetrized_quadratic_form`). The
symmetrized and raw charges coincide exactly when $DAS$ is symmetric, the
discrete form of the continuum Calderón identity $\hat S\hat D^{\dagger} = \hat D\hat S$
**[sympy 5]**; a Lebedev sphere satisfies it only approximately and a
molecular cavity not at all, so $q \ne Qv$ in practice while
$E_{\mathrm{pcm}}$ is unaffected.

On PySCF's own cavity these charges match PySCF's `pcm.py` to 1e-17 and the
SCF energy to 5e-12 Ha (`ferric-scf/tests/validation_pcm.rs`).

## 5. Sphere limits: Born, Kirkwood, Gauss

In channel $l$ the operators of §3 are scalars,
$K_l = S_l(1 + f/(2l+1))$, $R_l = -f(1 + 1/(2l+1))$, so $q_l = -k_l v_l/S_l$ with
**[sympy 3]**

$$ k_l = \frac{2(l+1) f}{2l+1+f} = \frac{(\varepsilon-1)(l+1)}{(l+1)\varepsilon + l}, $$

Kirkwood's reaction-field factor for a dielectric sphere, for every $l$. A
unit charge at distance $d$ from the centre of a sphere of radius $R$ then has

$$ E = -\frac{1}{2R}\sum_{l\ge0} \frac{(l+1)(\varepsilon-1)}{(l+1)\varepsilon+l}\left(\frac dR\right)^{2l}, $$

whose $d=0$ term is the Born energy $-(\varepsilon-1)/(2\varepsilon R)$; the
total induced charge is $-(\varepsilon-1)/\varepsilon$ (Gauss) for any interior
position. Replacing $f$ by $(\varepsilon-1)/\varepsilon$ breaks every channel
(control).

Discrete check (`tests/ief_limits.rs`
`point_charge_in_sphere_matches_born_kirkwood_and_gauss_law`, Ne sphere;
**[sympy 7]** the same in numpy on a scipy Lebedev sphere). Relative energy
error of the default `GaussianSmeared` S/D at Lebedev 50 / 110 / 302:

| ε | d/R | 50 | 110 | 302 | $\lvert E_{302}-E_{110}\rvert/\lvert E\rvert$ |
|---|---|---:|---:|---:|---:|
| 78.4 | 0 | 3.1e-8 | 2.7e-8 | 1.1e-8 | 1.6e-8 |
| 78.4 | 0.25 | 2.8e-4 | 8.7e-5 | 1.9e-5 | 6.8e-5 |
| 78.4 | 0.5 | 2.8e-3 | 8.2e-4 | 1.8e-4 | 6.4e-4 |
| 4 | 0 | 5.4e-7 | 4.1e-7 | 1.7e-7 | 2.4e-7 |
| 4 | 0.25 | 2.4e-4 | 7.3e-5 | 1.6e-5 | 5.7e-5 |
| 4 | 0.5 | 2.4e-3 | 6.9e-4 | 1.5e-4 | 5.4e-4 |

The test asserts monotone convergence and that the 302-point error is below
the a-posteriori estimate in the last column; the total charge is within
$|Q_{302}-Q_{110}|$ of $-(\varepsilon-1)/\varepsilon$. The `PointCharge` S/D on
the same spheres (numpy) converges far more slowly: central-charge relative
error 4.5e-3 / 3.2e-3 / 2.0e-3 at 50 / 110 / 302, from the
$1.0694\sqrt{4\pi/a_i}$ self-term on a grid with unequal weights. On a sphere $D$ is symmetric
($\nu_{ij} = \nu_{ji} = -R(1-\hat n_i\cdot\hat n_j)$), so the sphere tests do
not see a source/field normal swap; the molecular-cavity PySCF comparison does.

## 6. Conductor limit and COSMO

At $f = 1$ ($\varepsilon\to\infty$), $K = (I - DA/2\pi)S$ and $R = -(I - DA/2\pi)$,
so on any discretization **[sympy 4]**

$$ K^{-1}R = -S^{-1}, \qquad S q = -v , $$

the conductor-like (C-PCM / COSMO) equation; $S$ is symmetric, so the
symmetrization is exact there. The approach is first order in
$1 - f = 2/(\varepsilon+1)$ (`tests/ief_limits.rs`
`ief_reduces_to_conductor_equation_as_eps_goes_to_infinity`: on water's
cavity the relative gap is 1.649e-6 at ε = 1e6 and 1.649e-8 at ε = 1e8, ratio
100.000; 2.1e-2 at ε = 78.4).

Conductor-like models at finite ε scale $Sq = -v$ by a single factor:
C-PCM $(\varepsilon-1)/\varepsilon$ equals Kirkwood's $k_0$ (exact for a
central charge), and `ferric_scf::cosmo`'s $(\varepsilon-1)/(\varepsilon+\tfrac12)$
(PySCF COSMO) equals $k_1$ (exact for a central dipole); neither matches
every $l$ **[sympy 3]**. IEF-PCM reproduces all channels. The cavity and the
S matrix of `ferric_scf::cosmo` are a separate implementation.

## 7. Solute potential, Fock contribution, total energy

The potential at tessera $k$ from the current total density $P$
(`potential.rs:37` `solute_potential_at_tesserae`, default `ProbeKind::Point`):

$$ v_k = \sum_A \frac{Z_A}{|\mathbf r_k-\mathbf R_A|} + \sum_{\mu\nu} P_{\mu\nu} B^k_{\mu\nu}, \qquad
   B^k_{\mu\nu} = \left\langle \mu \middle| -\frac{1}{|\mathbf r-\mathbf r_k|} \middle| \nu \right\rangle , $$

$B^k$ being libint's nuclear-attraction integral for a $+1$ charge at
$\mathbf r_k$. `ProbeKind::GaussianSmeared` (`potential.rs:256`, `:347`) replaces
every $1/r$ by $\operatorname{erf}(\xi_k r)/r$ (PySCF's probe convention).

Per SCF iteration `pcm_step` (`lib.rs:166`) evaluates $v$, solves §4, and
returns the reaction-field operator (`potential.rs:150`
`build_reaction_field_operator`)

$$ V^{\mathrm{pcm}}_{\mu\nu} = \sum_k q_k\, B^k_{\mu\nu} $$

with $E_{\mathrm{pcm}}$. Because $q$ is the symmetrized response,
$E_{\mathrm{pcm}}(P) = \tfrac12 v(P)^{\mathsf T} Q_{\mathrm{sym}}\, v(P)$ has
$\partial E_{\mathrm{pcm}}/\partial P_{\mu\nu} = V^{\mathrm{pcm}}_{\mu\nu}$
**[sympy 6]**; the raw $Qv$ would not be this gradient (control). The SCF adds
$V^{\mathrm{pcm}}$ to every spin Fock matrix (`ferric-scf/src/driver.rs:588`
in `solvent_terms`) and $E_{\mathrm{pcm}}$ once to the total energy, outside
the $\tfrac12\operatorname{Tr}[P(h+F)]$ trace
(`ferric-scf/src/rhf.rs:2805`; PySCF's `e_solvent` convention). The
self-consistency between $q$ and $P$ is carried by the outer SCF/DIIS loop;
the charge solve itself is direct.

## 8. Nuclear gradient

The SCF energy is variational in the density, so its nuclear gradient is the
solvent-free SCF gradient evaluated with the solvated orbitals and
energy-weighted density (which already carry $V^{\mathrm{pcm}}$ through the
Fock matrix) plus the derivative of $E_{\mathrm{pcm}}$ at fixed AO density $P$
(`gradient.rs:71` `pcm_gradient_parts`; added to the SCF gradient in
`ferric-scf/src/gradient.rs` `with_pcm_gradient` by
`restricted_scf_gradient`, `unrestricted_scf_gradient` and
`restricted_open_scf_gradient`). With

$$ x = K^{-1}Rv, \qquad y = K^{-\mathsf T}v, \qquad q = \tfrac12\left(x + R^{\mathsf T}y\right) \quad (`gradient.rs:100`,\ `:106`), $$

$$ \frac{\partial E_{\mathrm{pcm}}}{\partial R_{A\alpha}} = q^{\mathsf T}\frac{\partial v}{\partial R_{A\alpha}} + \tfrac12\, v^{\mathsf T}\frac{\partial (K^{-1}R)}{\partial R_{A\alpha}}\, v . $$

Each tessera moves rigidly with its sphere's atom $A(k)$:
$\partial \mathbf r_k/\partial \mathbf R_B = \delta_{A(k)B}\,I$. The normals
$\hat{\mathbf n}_k$, the widths $\xi_k$ and the sphere radii are geometry
independent; the areas and $S_{kk}$ depend on the geometry only through
$\mathrm{swf}_k$.

### 8.1 Potential term

$q^{\mathsf T}\partial v$ is the gradient of the interaction of fixed charges
$q_k$ at $\mathbf r_k$ with the solute's nuclei and electrons. For point
probes (`gradient.rs:269` `potential_term_point`) the electronic part is
$\sum_{\mu\nu}P_{\mu\nu}\,\partial V^{\mathrm{pcm}}_{\mu\nu}(q)$ from libint's
nuclear-attraction derivative engine with the tesserae as the only charges:
the two shell-centre blocks go to the AO atoms, the block of charge $k$
(`gradient.rs:319`) to $A(k)$. The classical part
$\sum_k q_k \sum_{B\ne A(k)} Z_B\,\phi(|\mathbf r_k-\mathbf R_B|)$,
$\phi(r) = 1/r$, moves $A(k)$ against $B$ (`gradient.rs:403`
`nuclear_tessera_term`). For Gaussian probes (`gradient.rs:336`
`potential_term_smeared`) $\phi(r) = \operatorname{erf}(\xi_k r)/r$ and the
electronic part uses the derivative 3-centre integrals
$\partial(\mu\nu|g_k)/N_k$ (site block to $A(k)$).

### 8.2 Response term

With $f/2\pi$ written $p$, differentiating $K = (I - pDA)S$ and
$R = -f(I - DA/2\pi)$ gives **[sympy 8]** (`gradient.rs:189` `response_term`)

$$ \tfrac12 v^{\mathsf T}\partial(K^{-1}R)v = \tfrac12\Big[ -z^{\mathsf T}(\partial S)\,x + p\, y^{\mathsf T}(\partial D)\,u + p\sum_j (D^{\mathsf T}y)_j\, w_j\, \partial a_j \Big], $$

$$ w = Sx + v, \qquad u = Aw, \qquad z = y - p\,A D^{\mathsf T} y . $$

Off-diagonal elements depend on $\mathbf r_{ij}$. With $S_{ij} = S(r_{ij})$
and $D_{ij} = h(r_{ij})\,\nu_{ij}$, $h(r) = -S'(r)/r$ (`gradient.rs:135`
`sd_kernel`),

$$ \nabla_{\mathbf r_i} S_{ij} = S'(r)\,\hat{\mathbf r}_{ij}, \qquad
   \nabla_{\mathbf r_i} D_{ij} = h'(r)\,\nu_{ij}\,\hat{\mathbf r}_{ij} + h(r)\,\hat{\mathbf n}_j , $$

and $\nabla_{\mathbf r_j} = -\nabla_{\mathbf r_i}$ (`gradient.rs:230`; pairs on
one sphere cancel and are skipped). For `GaussianSmeared`
($c = 2/\sqrt\pi$, $\xi = \xi_{ij}$):

$$ S' = \frac{c\,\xi e^{-\xi^2r^2}}{r} - \frac{\operatorname{erf}(\xi r)}{r^2}, \qquad
   h' = \frac{3c\,\xi e^{-\xi^2r^2}}{r^3} - \frac{3\operatorname{erf}(\xi r)}{r^4} + \frac{2c\,\xi^3 e^{-\xi^2r^2}}{r}, $$

and for `PointCharge` $S' = -1/r^2$, $h = 1/r^3$, $h' = -3/r^4$.

Diagonal and area terms are all proportional to
$\partial\ln\mathrm{swf}_j$: $\partial a_j = a_j\,\partial\ln\mathrm{swf}_j$;
`GaussianSmeared` $\partial S_{jj} = -S_{jj}\,\partial\ln\mathrm{swf}_j$ and
$D_{jj}$ constant; `PointCharge` $\partial S_{jj} = -\tfrac12 S_{jj}\,\partial\ln\mathrm{swf}_j$
and $\partial D_{jj} = -\tfrac12 D_{jj}\,\partial\ln\mathrm{swf}_j$ (`gradient.rs:243`,
`:247`). The switching derivative is

$$ \nabla_{\mathbf r_j}\ln\mathrm{swf}_j = \sum_{B\ne A(j)} \frac{h_{\mathrm{sw}}'(t_{jB})}{h_{\mathrm{sw}}(t_{jB})}\,\frac{1}{R^{\mathrm{sw}}_B}\,\hat{\mathbf r}_{jB},
   \qquad t_{jB} = \frac{|\mathbf r_j-\mathbf R_B| - R^{\mathrm{in}}_B}{R^{\mathrm{sw}}_B}, $$

with $h_{\mathrm{sw}}'(t) = 30t^2(1-t)^2$ (`cavity.rs:79` `switch_h_deriv`),
$A(j)$ receiving $+$ and $B$ receiving $-$ (`gradient.rs:261`). $R^{\mathrm{sw}}$,
$R^{\mathrm{in}}$ come from the shared `cavity.rs:100` `SphereGeometry::new`
that `build_cavity` uses.

### 8.3 Validation

`tests/pcm_gradient.rs` (fixed $P$, FD of `pcm_step`'s energy; bar = the
change between steps $h$ and $2h$ plus a round-off floor) and
`ferric-scf/tests/pcm_gradient.rs` (SCF level, same bar plus $4\,\delta E/h$):

| check | system | max \|analytic − reference\| (Ha/Bohr) |
|---|---|---:|
| FD, fixed $P$, default kinds, Lebedev 110 | water/STO-3G | 5.7e-8 |
| FD, fixed $P$, point / Gaussian probes, Lebedev 50 | C1 methanol/STO-3G | 2.0e-7 / 2.0e-7 |
| FD, fixed $P$, `PointCharge` S/D, $h = 2.5\times10^{-4}$ | water/STO-3G | 8.6e-7 |
| FD of the solvated RHF energy | water/STO-3G | 1.4e-7 |
| FD of the solvated RHF energy, Lebedev 50 | C1 methanol/STO-3G | 2.9e-7 |
| FD of the solvated RKS (SVWN) energy, Gaussian probes | water/STO-3G | 1.4e-7 |
| FD of the solvated UHF / ROHF energy | NH2/STO-3G | 1.3e-7 / 1.3e-7 |
| torque vs FD rigid rotation (torque 1.36e-4), Lebedev 50 | C1 methanol/STO-3G | 8.0e-9 |
| PySCF 2.13 `nuc_grad_method()`, IEF-PCM, 302 points, Gaussian probes | water / methanol, STO-3G | 4.2e-11 / 7.2e-11 |

The PySCF comparison runs on ferric's own cavity at `lebedev_order = 302`,
which reproduces PySCF's `gen_surface` (tessera count, total area and area
centroid asserted to 1e-10). Each part of the gradient sums to zero over the
atoms (translation, cavity included); the torque equals the finite-difference
derivative under a rigid rotation (the Lebedev frame is fixed in the lab, so
the torque is not zero in general). At $\varepsilon = 1 + 2\times10^{-9}$ the
solvated gradient equals the vacuum gradient to 3.3e-12.

`PointCharge` S/D energies are far rougher in the switching zone than
`GaussianSmeared` ones: on C1 methanol at Lebedev 50 (fixed model density)
components reach ~10 Ha/Bohr and central differences are not yet in their
$O(h^2)$ regime at $h = 2.5\times10^{-4}$.

`PcmConfig::cavity` (an injected tessera list) has no gradient: it carries no
geometry dependence, and `pcm_gradient` refuses it.

## 9. Scope

No cavitation, dispersion or repulsion terms; no analytic nuclear Hessian
(frequencies with PCM are refused by the CLI); no anisotropic or
ionic-strength IEF variants. The cavity differs from
PySCF's `gen_surface` in the default Lebedev order (110 vs 302), point vs
Gaussian probes, and untabulated elements (ferric's 2.0 Å fallback);
`ferric-scf/tests/validation_pcm.rs` measures ferric's own-cavity solvation
energy within 0.07-0.61% of PySCF's.
