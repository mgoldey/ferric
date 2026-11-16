# Periodic Boundary Conditions Theory

Gaussian-basis periodic Hartree-Fock and Kohn-Sham in the PySCF conventions:
Ewald nuclear repulsion with a neutralising background, an analytic Fourier
transform of lattice-summed Gaussian pair densities, an Ewald-split nuclear
attraction, the Madelung treatment of the exchange divergence, and the
strain derivative (stress) of each term; range-separated density fitting,
the k-point mesh, and periodic MP2 and dRPA.

Units are atomic (Bohr, Hartree); energies are per cell. Lattice rows
$\mathbf a_i$, cell volume $\Omega = \det(\mathbf a)$, reciprocal rows
$\mathbf b_i$ with $\mathbf a_i\cdot\mathbf b_j = 2\pi\delta_{ij}$, translations
$\mathbf L = \sum_i n_i \mathbf a_i$, reciprocal vectors
$\mathbf G = \sum_i m_i \mathbf b_i$ ($m_i$ the Miller index).

Sections 1-5 are asserted by
[`scripts/pbc_ewald_pairft_stress_sympy.py`](https://github.com/mgoldey/ferric/blob/main/scripts/pbc_ewald_pairft_stress_sympy.py)
(section tags S1–S5), section 7 by
[`scripts/pbc_rsgdf_sympy.py`](https://github.com/mgoldey/ferric/blob/main/scripts/pbc_rsgdf_sympy.py), and
sections 8 and 9 by
[`scripts/pbc_kpoint_corr_sympy.py`](https://github.com/mgoldey/ferric/blob/main/scripts/pbc_kpoint_corr_sympy.py),
except where a section says it has no proof; the Rust tests listed check
the implementation numerically. Paths are relative to `crates/ferric-pbc/`.

## 1. Ewald nuclear repulsion and the Madelung constant

For point charges $Z_j$ at $\mathbf R_j$ and any Ewald splitting parameter
$\eta > 0$ (the code's `omega`, which the heading of the next subsection keeps;
`ewald.rs:109` `ewald_point_charges`):

$$
\begin{aligned}
E ={}& \tfrac12 \sum_{\mathbf L}{\sum_{i,j}}' Z_i Z_j
      \frac{\operatorname{erfc}(\eta d_{ij\mathbf L})}{d_{ij\mathbf L}},
      \qquad d_{ij\mathbf L} = |\mathbf R_i - \mathbf R_j - \mathbf L| \\
&+ \frac{2\pi}{\Omega}\sum_{\mathbf G\neq 0}
      \frac{e^{-G^2/4\eta^2}}{G^2}\,|S(\mathbf G)|^2,
      \qquad S(\mathbf G) = \sum_j Z_j e^{-i\mathbf G\cdot\mathbf R_j} \\
&- \frac{\eta}{\sqrt\pi}\sum_j Z_j^2
 \;-\; \frac{\pi Q^2}{2\Omega\eta^2}, \qquad Q = \sum_j Z_j .
\end{aligned}
$$

The prime drops $i=j$ at $\mathbf L = 0$. The last term is the uniform
neutralising background (PySCF `Cell.energy_nuc()`); it vanishes for a
neutral set. Code: SR `ewald.rs:164`, LR `ewald.rs:185`, self `ewald.rs:189`,
background `ewald.rs:190`. Cutoffs: $\operatorname{erfc}(x) < e^{-x^2}$ and
$e^{-G^2/4\eta^2} = e^{-x^2}$ at $x = G/2\eta$, so
$r_\text{cut} = s/\eta$, $G_\text{cut} = 2\eta s$ with
$s = \sqrt{\ln(1/\epsilon)} + 1$; all sums are Neumaier-compensated.

### Independence of $\omega$ (S1)

$$
\frac{\partial}{\partial\eta}\frac{\operatorname{erfc}(\eta r)}{r}
 = -\frac{2}{\sqrt\pi} e^{-\eta^2 r^2},
\qquad
\int e^{-\eta^2 r^2} e^{-i\mathbf G\cdot\mathbf r}\,d^3r
 = \Big(\frac{\pi}{\eta^2}\Big)^{3/2} e^{-G^2/4\eta^2},
$$

$$
\frac{\partial}{\partial\eta}\Big[\frac{4\pi}{G^2}e^{-G^2/4\eta^2}\Big]
 = \frac{2}{\sqrt\pi}\Big(\frac{\pi}{\eta^2}\Big)^{3/2} e^{-G^2/4\eta^2}.
$$

The Poisson sum
$\sum_{\mathbf L} e^{-\eta^2|\mathbf d - \mathbf L|^2}
 = \Omega^{-1}\sum_{\mathbf G}(\pi/\eta^2)^{3/2} e^{-G^2/4\eta^2}
 e^{i\mathbf G\cdot\mathbf d}$
turns $\partial_\eta E_\text{SR}$ into a reciprocal sum over
$|S(\mathbf G)|^2$ plus the excluded self pair. Per $\mathbf G \neq 0$ it
cancels $\partial_\eta E_\text{LR}$; its $\mathbf G = 0$ term
$-\pi Q^2/(\Omega\eta^3)$ cancels $\partial_\eta$ of the background, and
the excluded self pair $+\pi^{-1/2}\sum Z_j^2$ cancels $\partial_\eta$ of the
self term. Hence $dE/d\eta = 0$ exactly; truncation is the only source of
$\eta$-dependence.

### Madelung constant

$v_M = -2\,E[\text{one unit charge per cell, with background}]$, a function
of the lattice only (`ewald.rs:94` `madelung_constant`, at
$\eta = \sqrt\pi/\Omega^{1/3}$, `ewald.rs:61`). Simple cubic:
$v_M a = 2.8372974794806$. A $k$-point mesh $N_1\times N_2\times N_3$ uses the
Gamma constant of the $\operatorname{diag}(N)$ supercell (`kpts.rs:342`).

### Gradient

$\partial E/\partial\mathbf R_i$ from the SR radial derivative
$f'(d) = -[\operatorname{erfc}(\eta d)/d^2 + (2\eta/\sqrt\pi)e^{-\eta^2 d^2}/d]$
and the LR phase derivative; the self and background terms are
position-independent (`ewald.rs:213` `ewald_nuclear_gradient_parts`).

**Tests.** `tests/pbc_stage0.rs`
`ewald_matches_pyscf_and_is_omega_independent` (PySCF `energy_nuc` to 1e-10,
$\eta$-spread < 1e-12 at $\eta$ = 0.8, 1.5, 3 on three charged nuclear
cells), `madelung_matches_pyscf` (PySCF to 1e-11, simple-cubic textbook value),
`ewald_rock_salt_madelung_is_textbook_and_omega_independent` (mixed-sign
neutral set: rock-salt $M = 1.747564594633182$, $E = -8M/a$, $\eta$-spread
< 1e-12); `tests/pbc_grad.rs` `ewald_gradient_matches_pyscf_grad_nuc_and_fd`.

## 2. Fourier transform of lattice-summed Gaussian pairs

$$
P_{\mu\nu}(\mathbf G) = \sum_{\mathbf L}\int \phi_\mu(\mathbf r)\,\phi_\nu(\mathbf r - \mathbf L)\,
 e^{-i\mathbf G\cdot\mathbf r}\,d^3r
$$

(PySCF `ft_aopair` sign; `pair_ft.rs:183` `pair_ft`). For primitives
$\alpha$ at $\mathbf A$ and $\beta$ at $\mathbf B' = \mathbf B + \mathbf L$,
$p = \alpha + \beta$, $\mathbf P_c = (\alpha\mathbf A + \beta\mathbf B')/p$, the
Gaussian product theorem (S2)

$$
-\alpha|\mathbf r - \mathbf A|^2 - \beta|\mathbf r - \mathbf B'|^2
 = -p|\mathbf r - \mathbf P_c|^2 - \frac{\alpha\beta}{p}|\mathbf A - \mathbf B'|^2
$$

and the McMurchie–Davidson expansion
$x_A^{i}x_B^{j}e^{\cdots} = \sum_t E_t^{ij}\,\partial_{P_x}^t e^{-p(x-P_x)^2}$
give, with $\partial_{P_d}\to(-iG_d)$ under the transform,

$$
P^\text{prim}(\mathbf G) = c_a c_b \Big(\frac{\pi}{p}\Big)^{3/2}
 e^{-\alpha\beta|\mathbf A - \mathbf B'|^2/p}\;
 e^{-G^2/4p}\; e^{-i\mathbf G\cdot\mathbf P_c}
 \prod_{d=x,y,z}\sum_t E_t^{i_d j_d}(-iG_d)^t .
$$

Low orders: $E^{00}_0 = 1$; $E^{10} = (X_{PA},\,1/2p)$;
$E^{11} = (X_{PA}X_{PB} + 1/2p,\,(X_{PA}+X_{PB})/2p,\,1/4p^2)$;
$E^{20} = (X_{PA}^2 + 1/2p,\,X_{PA}/p,\,1/4p^2)$, asserted against
quadrature of the complex 1-D integral in S2. Code: prefactor and overlap
Gaussian `pair_ft/plan.rs:883-884`, $E$-tables and the
$\sum_t E_t(-iG)^t$ rows `pair_ft/plan.rs:1336` `f_rows` (E from
`ferric_integrals::md3c1e::e_table`), phase $e^{-i\mathbf G\cdot\mathbf P_c}$
in `site_terms`. At $\mathbf G = 0$ only $E_0$ survives and
$P_{\mu\nu}(0) = S^\text{latt}_{\mu\nu}$.

**Tests.** `tests/pbc_stage0.rs`
`pair_ft_at_g0_equals_independent_libint_lattice_overlap` (every element vs an
independent libint2 lattice overlap, 1e-10) and
`pair_ft_at_nonzero_g_matches_pyscf_ft_aopair` (s, p, pure d; reciprocal-lattice
and off-lattice $\mathbf G$; 1e-10; asserts the imaginary part is exercised).

## 3. Ewald-split nuclear attraction (`hcore`)

$$
V = V_\text{SR}(\eta) + V_\text{LR}(\eta) + V_\text{G0}(\eta),
$$

$$
\begin{aligned}
V_\text{SR} &= -\sum_{\mathbf L}\sum_{C,\mathbf M} Z_C
  \Big\langle \mu_0 \Big| \frac{\operatorname{erfc}(\eta|\mathbf r - \mathbf R_C - \mathbf M|)}
  {|\mathbf r - \mathbf R_C - \mathbf M|}\Big| \nu_{\mathbf L}\Big\rangle, \\
V_\text{LR} &= -\frac{1}{\Omega}\sum_{\mathbf G\neq 0}\frac{4\pi}{G^2}e^{-G^2/4\eta^2}
  \operatorname{Re}\big[P^*_{\mu\nu}(\mathbf G)\,S(\mathbf G)\big], \\
V_\text{G0} &= +\frac{\pi Z_\text{tot}}{\eta^2\Omega}\,S^\text{latt}_{\mu\nu}.
\end{aligned}
$$

S3 asserts

$$
\int\frac{\operatorname{erfc}(\eta r)}{r}e^{-i\mathbf G\cdot\mathbf r}d^3r
 = \frac{4\pi}{G^2}\big(1 - e^{-G^2/4\eta^2}\big),
\qquad
\lim_{G\to 0} = \int\frac{\operatorname{erfc}(\eta r)}{r}d^3r = \frac{\pi}{\eta^2}.
$$

So per $\mathbf G\neq 0$, $V_\text{SR} + V_\text{LR}$ is the bare
$-4\pi Z/(\Omega G^2)$ kernel, and the real-space erfc sum carries the
$\mathbf G = 0$ average $-\pi Z_\text{tot}/(\eta^2\Omega)$, which $V_\text{G0}$
removes. $V$ is therefore $\eta$-independent and equals the pure
reciprocal-space attraction with $\mathbf G = 0$ dropped (the convention that
cancels against the same term of $J$ in a neutral cell).

Nuclei in $V_\text{SR}$ are unit-normalised s Gaussians of exponent
$\zeta = 10^{16}$ through the 3-centre engine (libint2's erf/erfc nuclear
kernels are wrong); the finite-nucleus error is
$Z\int\rho\,\operatorname{erfc}(\sqrt\zeta r)/r \approx \pi Z\rho(\mathbf R_C)/\zeta$.

Code: `hcore.rs:688` `reciprocal_nuclear` (kernel `hcore.rs:719`, half-sphere
factor $-2/\Omega$ `hcore.rs:730`), $V_\text{G0}$ `hcore.rs:1485-1486`,
oracle `hcore.rs:766` `pure_aft_nuclear` ($\eta\to\infty$, no SR, no
$\mathbf G = 0$ bookkeeping), `hcore.rs:796` `gaussian_nucleus_attraction`.

**Tests.** `tests/pbc_hcore.rs`
`periodic_hcore_is_omega_independent_triclinic_sp` ($V(\eta)$ at three
$\eta$ vs `pure_aft_nuclear`, with the negative control that $V_\text{G0}$ is
visible), `single_s_primitive_matches_closed_form`
($V_\text{erf} = -(2/\sqrt\pi)(1/\eta^2 + 1/p + 1/\zeta)^{-1/2}$),
`gaussian_nucleus_erf_plus_erfc_is_point_nuclear`,
`periodic_hcore_matches_prototype_h2_sto3g_a4`.

## 4. Exchange divergence (`exxdiv`)

Dense pure-AFT ERIs (`dense_aft.rs`, test/oracle scale):

$$
(\mu\nu|\lambda\sigma) = \frac{1}{\Omega}\sum_{\mathbf G\neq 0}\frac{4\pi}{G^2}
 \operatorname{Re}\big[P^*_{\mu\nu}(\mathbf G)P_{\lambda\sigma}(\mathbf G)\big].
$$

Dropping $\mathbf G = 0$ is exact for $J$ of a neutral cell but leaves
exchange with a finite-size error converging as $1/L$ (`ExxDiv::None`).
`ExxDiv::Ewald` adds the Madelung probe-charge correction (PySCF
`exxdiv='ewald'`, error $\sim L^{-3}$):

$$
K \mathrel{+}= v_M\, S D S,
$$

with $K_{\mu\nu} = \sum (\mu\lambda|\nu\sigma)D_{\lambda\sigma}$ on the total
density and $F = h + J - \tfrac12 K$ (`dense_aft.rs:471`). For a closed shell
$DSD = 2D$, so (S4)

$$
\Delta E = -\tfrac14 v_M \operatorname{tr}(DSDS) = -\tfrac12 N v_M,
\qquad
\frac{\partial\Delta E}{\partial D} = -\tfrac12 v_M SDS .
$$

Open shells carry the per-spin form $-\tfrac{\alpha v_M}{2}\sum_\sigma
\operatorname{tr}(D_\sigma S D_\sigma S)$ ($\alpha$ the exact-exchange
fraction). RS-GDF J/K use the same constant (`rsgdf.rs`); range-separated
hybrids at $k$ use the mixed supercell constant (`rsh.rs:201`
`kpoint_madelung`, `kdense_aft.rs:221`).

**Tests.** `tests/pbc_stress.rs` `h2_rhf_ewald_minus_none_is_the_madelung_strain`;
the PySCF energy pins of `tests/pbc_dense_aft_scf.rs` run both `exxdiv`
values.

## 5. Stress tensor (strain derivative)

Homogeneous strain $\mathbf r\to(1+\varepsilon)\mathbf r$ of lattice rows and
atoms (fixed fractional coordinates), Miller indices fixed:

$$
\sigma_{ab} = \frac{1}{\Omega}\frac{dE}{d\varepsilon_{ab}},\qquad
\mathbf G \to (1+\varepsilon)^{-T}\mathbf G,\qquad
\Omega\to\Omega\det(1+\varepsilon).
$$

Kinematics (S5):

$$
\frac{d\Omega}{d\varepsilon_{ab}} = \Omega\delta_{ab},\qquad
\frac{dG^2}{d\varepsilon_{ab}} = -2G_aG_b,\qquad
\frac{d|\mathbf d|}{d\varepsilon_{ab}} = \frac{d_a d_b}{|\mathbf d|},
$$

and $S(\mathbf G)$, every $\mathbf G\cdot\mathbf X$ phase, are strain-invariant.

### Ewald part (`ewald.rs:362` `ewald_point_charges_strain`)

With $k(G^2) = e^{-G^2/4\eta^2}/G^2$,
$k' = -k\,(1/4\eta^2 + 1/G^2)$:

$$
\begin{aligned}
\frac{dE_\text{SR}}{d\varepsilon_{ab}} &= \tfrac12\sum' Z_iZ_j f'(d)\frac{d_a d_b}{d}, \\
\frac{dE_\text{LR}}{d\varepsilon_{ab}} &= -\delta_{ab}E_\text{LR}
 + \frac{2\pi}{\Omega}\sum_{\mathbf G\neq 0}|S(\mathbf G)|^2\,k'(G^2)\,(-2G_aG_b), \\
\frac{dE_\text{bg}}{d\varepsilon_{ab}} &= -\delta_{ab}E_\text{bg},\qquad
\frac{dE_\text{self}}{d\varepsilon_{ab}} = 0 .
\end{aligned}
$$

Code: `ewald.rs:455` ($k'$), `ewald.rs:469` ($-\delta E_\text{LR}$),
`ewald.rs:477` (background). The total is $\eta$-independent; the parts are
not. $v_M$ is homogeneous of degree $-1$ in lengths, so
$\operatorname{tr}\,dv_M/d\varepsilon = -v_M$ (`ewald.rs:507`
`madelung_strain`). S5 checks the formula against central finite
differences of a pure-Python Ewald at frozen index sets (1.5e-10) and the
Euler identity (1e-12).

### Pair-FT part (`pair_ft.rs:1129` `pair_ft_strain_chunked`)

Per primitive pair, $\mathbf G\cdot\mathbf P_c$ is invariant and

$$
\frac{dP}{d\varepsilon_{ab}} = \Big[Q_a + iG_a\frac{\alpha}{p}P\Big](\mathbf A - \mathbf B')_b
 + \frac{G_aG_b}{2p}P - G_a P^g_b,
$$

$Q_a = \partial P/\partial A_a$ (bra-centre derivative), $P^g_b$ the
$G_b$-derivative of the Hermite polynomial factor
$\prod_d\sum_t E_t(-iG_d)^t$ (the Gaussian factor's $G$-dependence gives the
$G_aG_b/2p$ term). S5 asserts it against direct differentiation of the
strained closed form for s|s and p$_x$|s at exact rational points.

### Assembly (`stress.rs:449` `gamma_rhf_stress` and siblings)

Each reciprocal energy $E = \Omega^{-1}\sum_{\mathbf G} v(G^2)\Phi(\mathbf G)$
contributes $-\delta_{ab}E + \Omega^{-1}\sum[v'(-2G_aG_b)\Phi + v\,d\Phi]$;
overlap-coupled terms (Pulay, $c_0 = \pi/(\eta^2\Omega)$ and Madelung
$S$-terms) enter through the forces' energy-weighted matrix contracted with
per-image pair vectors; the Madelung energy adds
$-\tfrac{\alpha}{2}\sum_\sigma\operatorname{tr}(D_\sigma SD_\sigma S)\,dv_M/d\varepsilon$.
Truncated sets are frozen as integer indices at the reference cell
(`Cell::strained`), so the stress is the exact derivative of that energy.
The module doc of `stress.rs` lists the XC, ECP and RS-GDF terms.

**Tests.** `tests/pbc_stress.rs`
`ewald_strain_matches_fd_is_omega_independent_and_madelung_is_homogeneous`
(FD 1e-8, $\eta$ 1 vs 2.3 to 1e-10, Euler to 1e-12),
`pair_ft_strain_matches_fd_at_fixed_miller_index`,
`h2_rhf_stress_matches_fd_of_own_energy_all_nine_both_exxdiv`,
`h2_rhf_pressure_matches_isotropic_scaling`,
`h2_fd_anchor_catches_each_mutant`; $k$-point stress in
`tests/pbc_kstress.rs` and the `validation_*stress*.rs` files.

## 6. Refusal gates and tolerances of the open-shell force and stress paths

A force or stress is first order in the departure of the SCF state from
stationarity, so the open-shell entry points (Gamma and $k$-point ROHF/ROKS/
UHF/UKS) refuse a state instead of returning a number. Provenance is taken
from the code and `git log -S`: **measured** = calibrated against a force
error, **roundoff** = a decade or more above the observed matching-input
residual, **arbitrary** = no derivation or measurement recorded, **definitional**
= no tolerance. No threshold is a function of system size.

| Gate | Value | Where | Provenance |
|---|---|---|---|
| ROHF/ROKS orbital gradient $g = \max(\lvert f^\beta_{co}\rvert,\lvert f^\alpha_{ov}\rvert,\lvert (f^\alpha{+}f^\beta)_{cv}\rvert)$ | $g \le 10^{-5}$ Ha (NaN refused) | `grad.rs` `RO_ORBITAL_GRADIENT_TOL`, `ro_gate`; Gamma force and stress | measured: force error $\approx 0.1$–$0.74\,g$ (first order), so $10^{-5}$ caps it near $7\times10^{-6}$ Ha/Bohr, below optimizer thresholds ($\sim 4.5\times10^{-4}$); not a precision guarantee |
| Same, $k$-point | same number on the largest spectral norm per block, maximised over $k$ | `kuxcgrad.rs` `ro_orbital_gradient_k`, `kuxcstress.rs` | measured on H3/STO-3G 1x1x1 only (rank-1 blocks); the spectral norm is $\ge$ the Gamma max-element, so the $k$ gate can refuse a state the Gamma gate accepts. Constant shared |
| UKS spin commutator $\max_{\sigma,k}\lvert F_\sigma D_\sigma S - S D_\sigma F_\sigma\rvert$ | $\le 10^{-5}$ Ha | `kuxcgrad.rs` `UKS_COMMUTATOR_TOL`; $k$-point UKS force and stress | measured: commutator $1.6\times10^{-3}\to$ force error $7.2\times10^{-4}$, $2.7\times10^{-5}\to1.4\times10^{-5}$, $2.6\times10^{-6}\to5\times10^{-7}$. Gamma UKS has no commutator gate (only `converged`) |
| MO occupation classification (closed 1,1 / open 1,0 / virtual 0,0) | $\lvert n_i - \mathrm{round}(n_i)\rvert < 10^{-6}$ | `grad.rs` `RO_OCC_TOL`, `rohf_orbital_gradient` | arbitrary |
| Class counts vs $\mathrm{tr}(D_\sigma S)$ | $\lvert\Delta\rvert \le 10^{-6}$ electrons | `grad.rs` `rohf_orbital_gradient` | arbitrary |
| $\beta$-occupied, $\alpha$-empty MO | none | same | definitional |
| $k$ densities nested orthogonal projectors in the MO space ($D_\alpha^2{=}D_\alpha$, $D_\beta^2{=}D_\beta$, $D_\beta D_\alpha{=}D_\beta$) | max element deviation $\le 10^{-6}$ | `kuxcgrad.rs` `PROJECTOR_TOL` | arbitrary |
| Stored Fock $=$ Fock rebuilt from $D$ and the J/K source (all force/stress entries, per spin) | $\le 10^{-8}$ Ha (also for the Madelung-shift form) | `kgrad.rs` `FOCK_CONSISTENCY_BAR` | measured separation: matching inputs $2\times10^{-16}$; wrong `exxdiv` $0.79$; KS result to an HF entry $0.34$; wrong functional $5.9\times10^{-3}$ |
| $k$ ECP: $\lvert h - T - V - V_{\mathrm{ECP}}\rvert$ (or $T{+}V$) | $\le 10^{-10}$ Ha | `kgrad.rs` `HK_ECP_CONSISTENCY_BAR` | roundoff ($h$ is assembled from exactly these pieces) |
| ECP strain/force screen keeps the same triple count as hcore | exact integer equality | `stress.rs` (Gamma), `kgrad.rs` | definitional (same ECP precision/config and geometry) |
| SCF $k$-points equal the mesh | $\max\lvert\Delta k\rvert \le 10^{-12}$ | `kgrad.rs` `check_inputs_rsh` | roundoff |
| RS-GDF overlap equals `hc.s` (J3 $G{=}0$ term) | $\max\lvert\Delta S\rvert \le 10^{-12}\max(1,\max\lvert S\rvert)$ | `grad.rs` `check_rsgdf_inputs` | roundoff |
| `hcore.omega == hcore_cfg.omega` | exact `f64` equality | `grad.rs`, `kgrad.rs`, `kstress.rs` | definitional (the strain differentiates the build's own partition) |
| Range split on the RS-GDF stress | $\lambda \le 1$ | `kstress.rs` `refuse_split_mismatch` | measured: $\lambda\le1$ matches FD to $\sim10^{-11}$, $\lambda=2$ to $\sim10^{-8}$ (the Fock-guard bar); also refused: build/config disagreeing on whether a split ran |
| Exact-exchange fraction | $\in[0,1]$ | every `*stress.rs`, `*grad.rs` entry | definitional |
| SCF `converged` | boolean | every gradient entry (stress follows the force checks) | definitional |
| Non-finite $E_{xc}$, $V_{xc}$, force, strain derivative | any NaN/Inf | each XC and total assembly | definitional |

`gamma_rsh.rs` `IMAG_TOL = 10^{-9}` (imaginary part of the $k=0$ exchange of a
real density) belongs to the Gamma range-separated energy, not to a force or
stress gate. The memory pre-flights (`Ledger::check`) are budget refusals, not
tolerances.

Scope gates with no tolerance (functional and route allow-lists; each has a
named-refusal test): VV10 and double hybrids on every derivative path;
meta-GGA derivatives only for pure SCAN/r2SCAN (TPSS closed-shell dense only;
ROKS meta-GGA on RS-GDF refused); range-separated hybrids only for the
validated HSE06/CAM-B3LYP set; Gamma ROKS range-separated stress.

**Tests.** `tests/pbc_gate_tolerances.rs` (Gamma occupation and count gates,
bracketing each tolerance), `kuxcgrad.rs` `gate_tests::projector_tolerance_is_one_micro`,
`tests/pbc_kuxcgrad_ro.rs` `scf_kpoints_displaced_from_the_mesh_are_refused`;
the orbital-gradient, commutator, Fock-consistency, lambda and exchange-fraction
gates are pinned by the existing `pbc_kuxcgrad*.rs`, `pbc_grad_ro.rs`,
`pbc_kstress*.rs` and `validation_*stress*.rs` files. Unit tests with crafted
inputs reach three more refusals: `grad.rs` `gate_tests` (RS-GDF overlap,
$\max\lvert\Delta S\rvert$ at $5\times10^{-13}$ accepted, $2\times10^{-12}$ and
NaN refused), `kgrad.rs` `gate_tests` (`ecp_presence`: the $10^{-10}$ bar with
and without an ECP at $5\times10^{-11}$ / $2\times10^{-10}$ / NaN, and the
kept-triple count mismatch), and `tests/pbc_grad_ecp.rs` (Gamma force and
stress triple-count mismatch). Every max-reduction feeding a refusal
propagates NaN (`grad::max_nan`) and the gate is written `!(x <= bar)`, so a
NaN is refused.

## 7. Range-separated Gaussian density fitting (RS-GDF)

`jk = "rsgdf"` replaces the dense AFT tensor of §4 by a fitted one,
$I_{\mu\nu,\lambda\sigma} \approx \sum_k B_{k,\mu\nu}B_{k,\lambda\sigma}$,
in the PySCF RSGDF conventions (`rsgdf.rs`). Every fitted quantity uses the
same kernel as the energy, $v(\mathbf G) = 4\pi/G^2$ with $\mathbf G = 0$
dropped. With $X_P(\mathbf G)$ the Fourier transform of the auxiliary function
$\chi_P$ and $P_{\mu\nu}(\mathbf G)$ the lattice-summed pair transform of §2,

$$
J2_{PQ} = \frac{1}{\Omega}\sum_{\mathbf G\neq0}v(\mathbf G)\,X_P^*(\mathbf G)X_Q(\mathbf G),\qquad
J3_{\mu\nu,P} = \frac{1}{\Omega}\sum_{\mathbf G\neq0}v(\mathbf G)\,P_{\mu\nu}^*(\mathbf G)X_P(\mathbf G).
$$

### Evaluation by an Ewald split

Neither sum is evaluated as written. With $1/r = \operatorname{erfc}(\eta r)/r
+ \operatorname{erf}(\eta r)/r$ (the code's `omega`, PySCF `omega` of the
fit):

$$
\begin{aligned}
J2_{PQ} &= \sum_{\mathbf T}(P_0|Q_{\mathbf T})_{\operatorname{erfc}}
 + \frac{1}{\Omega}\sum_{\mathbf G\neq0}\frac{4\pi}{G^2}e^{-G^2/4\eta^2}X_P^*X_Q
 - c_0\,q_Pq_Q, \\
J3_{\mu\nu,P} &= \sum_{\mathbf L,\mathbf T}(\mu_0\nu_{\mathbf L}|P_{\mathbf T})_{\operatorname{erfc}}
 + \frac{1}{\Omega}\sum_{\mathbf G\neq0}\frac{4\pi}{G^2}e^{-G^2/4\eta^2}P_{\mu\nu}^*X_P
 - c_0\,S_{\mu\nu}\,q_P,
\end{aligned}
$$

with $c_0 = \pi/(\eta^2\Omega)$, $q_P = \int\chi_P = X_P(0)$ and
$S_{\mu\nu} = P_{\mu\nu}(0)$ the lattice overlap. The real-space erfc sums
(libint2 `erfc` integrals on translated shells, `Engine::compute_eri2_shifted`
and `compute_eri3_shifted`) contain the $\mathbf G = 0$ term of the erfc
kernel, $\int\operatorname{erfc}(\eta r)/r\,d^3r = \pi/\eta^2$ (§3), times the
two charges; `subtract_g0` removes it from both $J2$ and $J3$, the one place
the convention lives. Per $\mathbf G\neq0$ the erfc and erf pieces add to the
bare kernel, as in §3, so $J2$ and $J3$ are independent of $\eta$ up to
truncation. The erfc interaction of two unit Gaussian charges of exponents
$a, b$ at distance $d$ is

$$
\frac{\operatorname{erf}(\gamma d) - \operatorname{erf}(\gamma_\eta d)}{d},\qquad
\frac{1}{\gamma^2} = \frac1a + \frac1b,\quad
\frac{1}{\gamma_\eta^2} = \frac1a + \frac1b + \frac{1}{\eta^2},
$$

with limit $2(\gamma - \gamma_\eta)/\sqrt\pi$ at $d = 0$; the per-triplet
screen of `rsgdf.rs` bounds this function.

### The fit

$J2 = U s U^{\mathsf T}$ by a symmetric eigendecomposition; the eigenvalues
$s_k \le$ `lindep` (default $10^{-10}$, PySCF `LINEAR_DEP_THR`) are dropped and
counted, never inverted (`fit_with_metric`):

$$
W = U_\text{kept}\,s_\text{kept}^{-1/2},\qquad B = (J3\,W)^{\mathsf T},\qquad
B^{\mathsf T}B = J3\,J2^{+}J3^{\mathsf T}.
$$

Because $I$, $J2$ and $J3$ are Gram matrices of one positive kernel, the block
matrix $\begin{pmatrix}I & J3\\ J3^{\mathsf T} & J2\end{pmatrix}$ is positive
semidefinite and the fitting error $I - B^{\mathsf T}B$ is its Schur
complement: positive semidefinite. The fit is therefore variational in the
kernel the energy uses (the fitted Coulomb energy never exceeds the exact
one), and a charged auxiliary function needs no compensating charge, because
the only divergent term ($\mathbf G = 0$) is absent from $I$, $J2$ and $J3$
alike. When the auxiliary functions span the pair densities the fit is exact,
$B^{\mathsf T}B = I$.

### J and K

One $B$ serves both builders (`RsGdfJ`, `RsGdfK`), so the fitted ERI stays a
single positive semidefinite $B^{\mathsf T}B$:

$$
J_{\mu\nu} = \sum_k B_{k,\mu\nu}\sum_{\lambda\sigma}B_{k,\lambda\sigma}D_{\lambda\sigma},\qquad
K = \sum_k B_k\,D\,B_k^{\mathsf T} + v_M\,SDS ,
$$

with $B_k$ the $(n_\text{ao}, n_\text{ao})$ slice of row $k$ and the Madelung
term of §4 for `exxdiv = ewald`. The exchange sum runs over 24 fixed
auxiliary groups (`EXCHANGE_AUX_GROUPS`), so every bit of $K$ depends on
$n_\text{aux}$ alone, not on the thread count.

### At a k-point mesh

`rsgdf/kpoint.rs` fits each momentum-transfer class $q = k' - k$ separately:
the Bloch pair density $P^{k'}_{\mu\nu}(\mathbf K)$ at $\mathbf K = \mathbf G
+ q$ (§8) in the q-Bloch auxiliary set
$X^q_P = \sum_{\mathbf T}e^{iq\cdot\mathbf T}\chi_P(\mathbf r - \mathbf T)$,
with a Hermitian metric $J2(q) = U s U^{\mathsf H}$ cut per $q$. The
real-space erfc integrals do not depend on $q$ or $k$: the Gamma walk is
computed once, binned by $(\mathbf L \bmod N, \mathbf T \bmod N)$, and each
$(q, k')$ is a phase contraction of the bins. Only $q = 0$ contains
$\mathbf K = 0$, so only $q = 0$ subtracts $c_0$. Time reversal gives
$J2(-q) = J2(q)^*$, and only one of each $(q, -q)$ is built. The mesh fit is
the Gamma fit of the $\operatorname{diag}(N)$ supercell, block-diagonal in $q$.

**Proof.** `scripts/pbc_rsgdf_sympy.py`: claim `erfc-pair` (the pair formula,
its limits, and a radial quadrature), `split-j2` and `split-j3` (the split
sums against the pure reciprocal sums at two $\eta$ on s Gaussians in a
triclinic cell), `fit-psd`, `fit-trivial` and `fit-jk`.

**Tests.** `tests/pbc_rsgdf.rs` `rsgdf_matches_dense_aft_in_the_trivial_limit`
(fitted ERI vs the dense AFT tensor below $10^{-10}$ when the auxiliary set
spans the pairs), `g0_mutants_and_incomplete_aux_fail_the_anchor`,
`fitted_eri_is_omega_independent`,
`rhf_energy_is_rsgdf_omega_independent_and_the_build_follows_omega`;
`tests/pbc_krsgdf.rs` for the k-point fit.

## 8. k-point Hartree-Fock and Kohn-Sham

### Mesh and Bloch functions

A mesh $N_1\times N_2\times N_3$ (`kpts.rs`, PySCF `cell.make_kpts`) has
fractional points $f_i = \mathrm{num}_i/(2N_i)$ with integer numerators:
$\mathrm{num}_i = 2m_i$ for the Gamma-centred mesh and
$\mathrm{num}_i = 2m_i + 1 - N_i$ for the Monkhorst-Pack mesh, $m_i = 0, \dots,
N_i - 1$. For odd $N_i$ the two are the same set modulo the reciprocal
lattice; for even $N_i$ the Monkhorst-Pack set is shifted by half a step. The
Bloch AOs and the one-electron matrices are

$$
\chi_{\mu k}(\mathbf r) = \sum_{\mathbf L}e^{ik\cdot\mathbf L}\phi_\mu(\mathbf r - \mathbf L),\qquad
S_{\mu\nu}(k) = \sum_{\mathbf L}e^{ik\cdot\mathbf L}\langle\mu_0|\nu_{\mathbf L}\rangle ,
$$

and $h(k)$ likewise. Every phase is computed from the integer
$\sum_i\mathrm{num}_in_iD/(2N_i) \bmod D$, $D = \prod_i 2N_i$, so
$X(-k) = X(k)^*$ holds exactly for every one-particle matrix.

### Energy and Fock matrix

With orbitals normalized per cell and $D(k) = 2C_o(k)C_o(k)^{\mathsf H}$
(`kscf.rs`):

$$
\begin{aligned}
F(k) &= h(k) + J(k) - \tfrac12K(k), \\
J_{\mu\nu}(k) &= \frac{1}{N_k}\sum_{k'}\frac{1}{\Omega}\sum_{\mathbf G\neq0}v(\mathbf G)\,
 P^{k}_{\mu\nu}(\mathbf G)\sum_{\lambda\sigma}P^{k'}_{\lambda\sigma}(\mathbf G)^*D_{\lambda\sigma}(k'), \\
K_{\mu\nu}(k) &= \frac{1}{N_k}\sum_{k'}\frac{1}{\Omega}\sum_{\mathbf K\in\mathbf G+q,\ \mathbf K\neq0}
 v(\mathbf K)\sum_{\lambda\sigma}P^{k'}_{\mu\lambda}(\mathbf K)P^{k'}_{\nu\sigma}(\mathbf K)^*D_{\lambda\sigma}(k')
 + v_M\,S(k)D(k)S(k), \\
E/\text{cell} &= \frac{1}{N_k}\sum_k\tfrac12\operatorname{Re}\operatorname{tr}\big[(h(k) + F(k))D(k)\big] + E_{nn},
\end{aligned}
$$

with $q = k' - k$ and
$P^{k'}_{\mu\nu}(\mathbf K) = \sum_{\mathbf L}e^{ik'\cdot\mathbf L}\int\phi_\mu\,\phi_\nu(\cdot - \mathbf L)\,e^{-i\mathbf K\cdot\mathbf r}$,
the transform of the Bloch pair $\chi_{\mu k}^*\chi_{\nu k'}$, which is
nonzero only on $\mathbf G + q$ (`kdense_aft.rs`). Exchange takes the kernel
at $\mathbf G + q$, Coulomb at $\mathbf G$; the only dropped term is
$\mathbf K = 0$. $v_M$ is the Madelung constant of the
$\operatorname{diag}(N)$ supercell (`KPointMesh::madelung`), added at every
$k$ with no $1/N_k$ (PySCF `_ewald_exxdiv_for_G0`).

For a Gamma-centred mesh $\{\mathbf G + q\}$ is the reciprocal lattice of the
$\operatorname{diag}(N)$ supercell. With the supercell density
$D^\text{sc}_{(\mu\mathbf L_1),(\nu\mathbf L_2)} = N_k^{-1}\sum_k
e^{ik\cdot(\mathbf L_1 - \mathbf L_2)}D_{\mu\nu}(k)$, the $J$, $K$ and energy
above are, term by term, the Gamma ones of the supercell per cell, and
$X(k) = \sum_{\mathbf L}e^{ik\cdot\mathbf L}X^\text{sc}_{(\mu0),(\nu\mathbf L)}$
maps the supercell Fock matrix to $F(k)$. A shifted mesh is a
twisted-boundary supercell and has no such anchor.

### The SCF

| Step | What | Code |
|---|---|---|
| orthogonalizer | eigenvectors of $S(k)$ with eigenvalue $\ge$ `lindep` ($10^{-6}$), scaled by $\lambda^{-1/2}$ | `complex_canonical_orthogonalizer` |
| diagonalize | one of each $\{k, -k\}$; $C(-k) = C(k)^*$, $\varepsilon(-k) = \varepsilon(k)$ | `diagonalize_all` |
| occupy | the $N_kN_e/2$ lowest levels over all $k$ (PySCF `KRHF.get_occ`); a converged gap below `min_gap` is an error (no metals) | `aufbau` |
| extrapolate | DIIS on $X^{\mathsf H}(FDS - SDF)X$ stacked over $k$, Gram $\operatorname{Re}\sum_k\langle e_i, e_j\rangle$, real weights | `KDiis::step` |

### Open shells

Per spin, $D_\sigma(k) = C_{\sigma,o}C_{\sigma,o}^{\mathsf H}$ and (`kuscf.rs`)

$$
F_\sigma(k) = h(k) + J[D_\alpha + D_\beta]{}(k) - K[D_\sigma]{}(k),\qquad
E/\text{cell} = \frac{1}{N_k}\sum_{k,\sigma}\tfrac12\operatorname{Re}\operatorname{tr}\big[(h + F_\sigma)D_\sigma\big] + E_{nn},
$$

where $K[D_\sigma]$ carries $v_M S D_\sigma S$ with coefficient 1: the builder
is linear in the density, and there is no per-spin $\tfrac12$. Occupations
are a global aufbau per spin over the whole mesh. ROHF and ROKS combine the
same per-spin Fock matrices into the Guest-Saunders (Roothaan) effective Fock
on one orbital set: at Gamma through `ferric_scf::rohf`, on a mesh per $k$ in
`kroks.rs` (PySCF `get_roothaan_fock`).

### The Madelung shift

For S-orthonormal orbitals and a closed shell,
$C^{\mathsf H}\big(-\tfrac12 v_MSDS\big)C = -v_M\,\mathrm{diag}(1_\text{occ}, 0_\text{virt})$
at every $k$: `exxdiv = ewald` lowers every occupied level by $v_M$, leaves the
virtual levels and the orbitals unchanged, and changes the energy by
$-v_MN_e/2$ per cell. The same holds per spin, $-v_M(N_\alpha + N_\beta)/2$.
`none` and `ewald` therefore share their stationary densities; a state with a
hole shallower than $v_M$ can be aufbau-consistent under `ewald` only, which
is why the open-shell drivers converge with `none` first and restart with
`ewald` (`EwaldStart::Staged`).

In the force (`grad.rs`), the overlap-coupled matrix carries
$-\alpha v_M\sum_\sigma D_\sigma SD_\sigma$ per spin. Because
$D_\sigma SD_\sigma = D_\sigma$, it cancels the $-\alpha v_M$ shift inside
$W = \sum_\sigma D_\sigma F_\sigma D_\sigma$, and the forces for `ewald` and
`none` agree. The total-density form $-\tfrac{\alpha v_M}{2}DSD$ does not
cancel on an open shell.

### Kohn-Sham

With an exact-exchange fraction $a$ (`kdft.rs`, `solve_krks_injected`):

$$
F(k) = h(k) + J(k) - \tfrac a2K(k) + V_{xc}(k),\qquad
E/\text{cell} = \frac{1}{N_k}\sum_k\operatorname{Re}\operatorname{tr}\big[D(k)\big(h + \tfrac12J - \tfrac a4K\big)(k)\big] + E_{xc} + E_{nn},
$$

$$
\rho(\mathbf r) = \frac{1}{N_k}\sum_k\sum_{\mu\nu}D_{\mu\nu}(k)\,\chi_{\mu k}(\mathbf r)\chi_{\nu k}(\mathbf r)^*,\qquad
V_{xc}(k)_{\mu\nu} = \int_\text{cell}\chi_{\mu k}^*\,v_\rho\,\chi_{\nu k} + \text{(GGA terms)},
$$

the $D(k)$ derivative of $E_{xc}$ times $N_k$, the same convention as $h$,
$J$ and $K$. The injected $K$ carries the Madelung term, so it is scaled by
$a$ with the rest of the exchange. The cell integral runs on atom-centred
grids weighted by an SSF partition of the infinite crystal (`dft.rs`,
`PeriodicGrid`).

**Proof.** `scripts/pbc_kpoint_corr_sympy.py`: claims `mesh`,
`krhf-supercell` (the $J(k)$, $K(k)$ and energy above against the supercell's
Gamma $J$, $K$ and energy), `kuhf-supercell`, `madelung-shift` and
`madelung-force`. The model is one-dimensional with s Gaussians; the
k-point bookkeeping it checks does not depend on the dimension. The
periodic XC grid and the XC force and stress terms have no proof script.

**Tests.** `tests/pbc_krhf.rs` `h2_1x1x3_mesh_is_the_gamma_supercell` and
`h2_2x2x2_mesh_is_the_gamma_supercell` (k-mesh energy against the explicit
supercell, $10^{-11}$ Ha, both `exxdiv`),
`bloch_convention_time_reversal_and_hermiticity`,
`monkhorst_pack_points_and_mesh_madelung_match_pyscf`, the PySCF pins
`h2_1x1x2_matches_pinned_pyscf_krhf_aftdf` and siblings ($10^{-9}$ Ha), and
the mutants `mutant_kernel_without_q_shift_is_caught`,
`mutant_phase_sign_in_pair_ft_is_caught`,
`mutant_primitive_madelung_is_caught_and_leaves_exxdiv_none_exact`;
`tests/pbc_kuhf.rs` for the open shell.

## 9. Periodic MP2 and dRPA

### Gamma point

At $\Gamma$ the orbitals are real and the molecular closed-shell formulas
apply unchanged (`mp2.rs`, `drpa.rs`):

$$
E_\text{MP2} = \sum_{ijab}\frac{(ia|jb)\big[2(ia|jb) - (ib|ja)\big]}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b},\qquad
E_c^\text{dRPA} = \frac{1}{2\pi}\int_0^\infty\Big[\ln\det\big(1 + \Pi(iu)\big) - \operatorname{tr}\Pi(iu)\Big]du,
$$

$$
\Pi_{PQ}(iu) = 4\sum_{ia}B^P_{ia}\frac{\Delta_{ia}}{u^2 + \Delta_{ia}^2}B^Q_{ia},\qquad
\Delta_{ia} = \varepsilon_a - \varepsilon_i .
$$

$(ia|jb)$ is in the $\mathbf G = 0$-dropped kernel of the SCF, and needs no
correction: $C_o^{\mathsf T}SC_v = 0$, so an occupied-virtual pair density
has no $\mathbf G = 0$ component. The production integrals are the SCF's own
RS-GDF $B$ of §7 transformed to $B_{k,ia}$ (`b_ov_from_ao_b`), handed to the
molecular kernels: `ferric_mp2::rimp2::spin_components_from_b_ov` for MP2 and
`ferric_rpa::run_pdep_rpa_from_parts` (full rank, the log-det summand) for
dRPA. $B$ is already metric-dressed, so no further $V^{-1/2}$ is applied.
The dRPA quadrature is Gauss-Legendre mapped by
$u = u_0(1 + x)/(1 - x)$, $u_0 = 0.5$, with 40 points by default
(`DEFAULT_GAMMA_DRPA_QUAD_POINTS`).

### Denominators

The exchange divergence enters correlation only through the occupied
orbital energies (§8, the Madelung shift). Two conventions exist
(`Mp2Denominators`): `MadelungShifted`, occupied $\varepsilon$ of the `ewald`
Fock, and `Unshifted`, those of the `none` Fock. The reference's `exxdiv`
cannot be read off the SCF result, so the caller states it, and
`occupied_shift` adds the shift that converts it:

| reference `exxdiv` | denominators | added to $\varepsilon_\text{occ}$ |
|---|---|---|
| ewald | `MadelungShifted` | 0 |
| none | `MadelungShifted` | $-v_M$ |
| none | `Unshifted` | 0 |
| ewald | `Unshifted` | $+v_M$ |

The conversion is exact because the orbitals are the same under both.

### k-point mesh

With per-cell normalized $C(k)$ and the pair tensors of the class
$q = k' - k$ (`kcorr.rs`):

$$
\begin{aligned}
V^{k_ik_jk_a}_{iajb} &= (i k_i\,a k_a|j k_j\,b k_b) = \frac{1}{\Omega}\sum_{\mathbf K\in\mathbf G+q}v(\mathbf K)\,
 P^{ov}_{ia}(\mathbf K)\,P^{vo}_{bj}(\mathbf K)^*,\qquad k_b = k_i + k_j - k_a,\ q = k_a - k_i, \\
E_\text{MP2}/\text{cell} &= \frac{1}{N_k^3}\sum_{k_ik_jk_a}\sum_{ijab}
 \frac{V_{iajb}^*\big[2V_{iajb} - V_{ibja}\big]}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}, \\
\Pi(q, iu) &= \frac{4}{N_k}\sum_{k'}\sum_{ia}B_{ia}(k'-q, k')\,B_{ia}(k'-q, k')^{\mathsf H}
 \frac{\Delta}{u^2 + \Delta^2},\qquad \Delta = \varepsilon_a(k') - \varepsilon_i(k' - q), \\
E_c^\text{dRPA}/\text{cell} &= \frac{1}{N_k}\sum_q\frac{1}{2\pi}\int_0^\infty
 \Big[\ln\det\big(1 + \Pi(q, iu)\big) - \operatorname{tr}\Pi(q, iu)\Big]du ,
\end{aligned}
$$

where $V_{ibja}$ is the element of $V^{k_ik_jk_b}$. Both legs of $V$ lie in one
class ($k_a - k_i = k_j - k_b$) and share its auxiliary set. With these
normalizations a Gamma-centred mesh is, term by term, the Gamma MP2 and dRPA
of the $\operatorname{diag}(N)$ supercell per cell. The second leg $B^{vo}$ is
transformed from $B(k_b, k_j)$ itself, not obtained by conjugating the $-q$
class, which would require a time-reversal gauge that a caller-built SCF
result need not have.

$\Pi(q)$ is complex Hermitian. The production energy (`drpa_q_realified`)
writes $B_q = X + iY$ and uses the real
$B_R = \begin{pmatrix}X & -Y\\ Y & X\end{pmatrix}$ with every excitation
energy duplicated: $B_R F B_R^{\mathsf T} =
\begin{pmatrix}\operatorname{Re}\Pi & -\operatorname{Im}\Pi\\ \operatorname{Im}\Pi & \operatorname{Re}\Pi\end{pmatrix}$
has the spectrum of $\Pi$ with every eigenvalue twice, so
$E_q = \tfrac12E[\Pi_R]$ through the unchanged real Gamma pipeline. The
$O(\Pi^2)$ term of the dRPA energy is the direct part
$\sum 2|V|^2/(\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b)$ of
MP2, from $\int_0^\infty e_1e_2/((u^2 + e_1^2)(u^2 + e_2^2))\,du = \pi/(2(e_1 + e_2))$.

Scope: closed shell; every $k$ must hold the same number of occupied levels
(the global aufbau of §8 can violate this; refused). The $q = 0$ head
correction is not implemented.

**Proof.** `scripts/pbc_kpoint_corr_sympy.py`: claims `ov-neutral`,
`kmp2-supercell`, `kdrpa-supercell` (the k-point formulas above against the
supercell's real closed-shell MP2 and plasmon dRPA), `realify` and
`drpa-second-order`.

**Tests.** `tests/pbc_mp2.rs`
`mp2_from_rsgdf_matches_dense_aft_in_the_trivial_aux_limit`,
`h2_sto3g_a4_matches_pinned_pyscf_rmp2`,
`box_limit_shifted_is_a3_with_the_predicted_coefficient_unshifted_is_1_over_a`;
`tests/pbc_drpa.rs` `drpa_from_rsgdf_matches_dense_plasmon_in_the_trivial_aux_limit`,
`drpa_second_order_term_is_direct_mp2_from_gamma_mp2`;
`tests/pbc_kcorr.rs` `h2_1x1x3_mesh_is_the_gamma_supercell` (k-point MP2 and
plasmon dRPA against the supercell, $10^{-11}$ Ha),
`supercell_anchor_catches_momentum_conj_madelung_occupied_k_and_normalisation_mutants`,
`kmp2_h2_1x1x2_matches_pinned_pyscf_kmp2_both_exxdiv`.
