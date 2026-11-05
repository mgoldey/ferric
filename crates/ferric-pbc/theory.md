# Periodic Boundary Conditions Theory

Gaussian-basis periodic Hartree-Fock and Kohn-Sham in the PySCF conventions:
Ewald nuclear repulsion with a neutralising background, an analytic Fourier
transform of lattice-summed Gaussian pair densities, an Ewald-split nuclear
attraction, the Madelung treatment of the exchange divergence, and the
strain derivative (stress) of each term.

Units are atomic (Bohr, Hartree); energies are per cell. Lattice rows
$\mathbf a_i$, cell volume $\Omega = \det(\mathbf a)$, reciprocal rows
$\mathbf b_i$ with $\mathbf a_i\cdot\mathbf b_j = 2\pi\delta_{ij}$, translations
$\mathbf L = \sum_i n_i \mathbf a_i$, reciprocal vectors
$\mathbf G = \sum_i m_i \mathbf b_i$ ($m_i$ the Miller index).

Every identity below is asserted by
[`scripts/pbc_ewald_pairft_stress_sympy.py`](../../scripts/pbc_ewald_pairft_stress_sympy.py)
(section tags S1–S5); the Rust tests listed check the implementation
numerically. Paths are relative to `crates/ferric-pbc/`.

## 1. Ewald nuclear repulsion and the Madelung constant

For point charges $Z_j$ at $\mathbf R_j$ and any splitting parameter
$\omega > 0$ (`ewald.rs:109` `ewald_point_charges`):

$$
\begin{aligned}
E ={}& \tfrac12 \sum_{\mathbf L}{\sum_{i,j}}' Z_i Z_j
      \frac{\operatorname{erfc}(\omega d_{ij\mathbf L})}{d_{ij\mathbf L}},
      \qquad d_{ij\mathbf L} = |\mathbf R_i - \mathbf R_j - \mathbf L| \\
&+ \frac{2\pi}{\Omega}\sum_{\mathbf G\neq 0}
      \frac{e^{-G^2/4\omega^2}}{G^2}\,|S(\mathbf G)|^2,
      \qquad S(\mathbf G) = \sum_j Z_j e^{-i\mathbf G\cdot\mathbf R_j} \\
&- \frac{\omega}{\sqrt\pi}\sum_j Z_j^2
 \;-\; \frac{\pi Q^2}{2\Omega\omega^2}, \qquad Q = \sum_j Z_j .
\end{aligned}
$$

The prime drops $i=j$ at $\mathbf L = 0$. The last term is the uniform
neutralising background (PySCF `Cell.energy_nuc()`); it vanishes for a
neutral set. Code: SR `ewald.rs:164`, LR `ewald.rs:185`, self `ewald.rs:189`,
background `ewald.rs:190`. Cutoffs: $\operatorname{erfc}(x) < e^{-x^2}$ and
$e^{-G^2/4\omega^2} = e^{-x^2}$ at $x = G/2\omega$, so
$r_\text{cut} = s/\omega$, $G_\text{cut} = 2\omega s$ with
$s = \sqrt{\ln(1/\epsilon)} + 1$; all sums are Neumaier-compensated.

### Independence of $\omega$ (S1)

$$
\frac{\partial}{\partial\omega}\frac{\operatorname{erfc}(\omega r)}{r}
 = -\frac{2}{\sqrt\pi} e^{-\omega^2 r^2},
\qquad
\int e^{-\omega^2 r^2} e^{-i\mathbf G\cdot\mathbf r}\,d^3r
 = \Big(\frac{\pi}{\omega^2}\Big)^{3/2} e^{-G^2/4\omega^2},
$$

$$
\frac{\partial}{\partial\omega}\Big[\frac{4\pi}{G^2}e^{-G^2/4\omega^2}\Big]
 = \frac{2}{\sqrt\pi}\Big(\frac{\pi}{\omega^2}\Big)^{3/2} e^{-G^2/4\omega^2}.
$$

The Poisson sum
$\sum_{\mathbf L} e^{-\omega^2|\mathbf d - \mathbf L|^2}
 = \Omega^{-1}\sum_{\mathbf G}(\pi/\omega^2)^{3/2} e^{-G^2/4\omega^2}
 e^{i\mathbf G\cdot\mathbf d}$
turns $\partial_\omega E_\text{SR}$ into a reciprocal sum over
$|S(\mathbf G)|^2$ plus the excluded self pair. Per $\mathbf G \neq 0$ it
cancels $\partial_\omega E_\text{LR}$; its $\mathbf G = 0$ term
$-\pi Q^2/(\Omega\omega^3)$ cancels $\partial_\omega$ of the background, and
the excluded self pair $+\pi^{-1/2}\sum Z_j^2$ cancels $\partial_\omega$ of the
self term. Hence $dE/d\omega = 0$ exactly; truncation is the only source of
$\omega$-dependence.

### Madelung constant

$v_M = -2\,E[\text{one unit charge per cell, with background}]$, a function
of the lattice only (`ewald.rs:94` `madelung_constant`, at
$\omega = \sqrt\pi/\Omega^{1/3}$, `ewald.rs:61`). Simple cubic:
$v_M a = 2.8372974794806$. A $k$-point mesh $N_1\times N_2\times N_3$ uses the
Gamma constant of the $\operatorname{diag}(N)$ supercell (`kpts.rs:342`).

### Gradient

$\partial E/\partial\mathbf R_i$ from the SR radial derivative
$f'(d) = -[\operatorname{erfc}(\omega d)/d^2 + (2\omega/\sqrt\pi)e^{-\omega^2 d^2}/d]$
and the LR phase derivative; the self and background terms are
position-independent (`ewald.rs:213` `ewald_nuclear_gradient_parts`).

**Tests.** `tests/pbc_stage0.rs`
`ewald_matches_pyscf_and_is_omega_independent` (PySCF `energy_nuc` to 1e-10,
$\omega$-spread < 1e-12 at $\omega$ = 0.8, 1.5, 3 on three charged nuclear
cells), `madelung_matches_pyscf` (PySCF to 1e-11, simple-cubic textbook value),
`ewald_rock_salt_madelung_is_textbook_and_omega_independent` (mixed-sign
neutral set: rock-salt $M = 1.747564594633182$, $E = -8M/a$, $\omega$-spread
< 1e-12); `tests/pbc_grad.rs` `ewald_gradient_matches_pyscf_grad_nuc_and_fd`.

## 2. Fourier transform of lattice-summed Gaussian pairs

$$
P_{mn}(\mathbf G) = \sum_{\mathbf L}\int \phi_m(\mathbf r)\,\phi_n(\mathbf r - \mathbf L)\,
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
$E^{20} = (X_{PA}^2 + 1/2p,\,X_{PA}/p,\,1/4p^2)$ — asserted against
quadrature of the complex 1-D integral in S2. Code: prefactor and overlap
Gaussian `pair_ft/plan.rs:883-884`, $E$-tables and the
$\sum_t E_t(-iG)^t$ rows `pair_ft/plan.rs:1336` `f_rows` (E from
`ferric_integrals::md3c1e::e_table`), phase $e^{-i\mathbf G\cdot\mathbf P_c}$
in `site_terms`. At $\mathbf G = 0$ only $E_0$ survives and
$P_{mn}(0) = S^\text{latt}_{mn}$.

**Tests.** `tests/pbc_stage0.rs`
`pair_ft_at_g0_equals_independent_libint_lattice_overlap` (every element vs an
independent libint2 lattice overlap, 1e-10) and
`pair_ft_at_nonzero_g_matches_pyscf_ft_aopair` (s, p, pure d; reciprocal-lattice
and off-lattice $\mathbf G$; 1e-10; asserts the imaginary part is exercised).

## 3. Ewald-split nuclear attraction (`hcore`)

$$
V = V_\text{SR}(\omega) + V_\text{LR}(\omega) + V_\text{G0}(\omega),
$$

$$
\begin{aligned}
V_\text{SR} &= -\sum_{\mathbf L}\sum_{C,\mathbf M} Z_C
  \Big\langle \mu_0 \Big| \frac{\operatorname{erfc}(\omega|\mathbf r - \mathbf R_C - \mathbf M|)}
  {|\mathbf r - \mathbf R_C - \mathbf M|}\Big| \nu_{\mathbf L}\Big\rangle, \\
V_\text{LR} &= -\frac{1}{\Omega}\sum_{\mathbf G\neq 0}\frac{4\pi}{G^2}e^{-G^2/4\omega^2}
  \operatorname{Re}\big[P^*_{\mu\nu}(\mathbf G)\,S(\mathbf G)\big], \\
V_\text{G0} &= +\frac{\pi Z_\text{tot}}{\omega^2\Omega}\,S^\text{latt}_{\mu\nu}.
\end{aligned}
$$

S3 asserts

$$
\int\frac{\operatorname{erfc}(\omega r)}{r}e^{-i\mathbf G\cdot\mathbf r}d^3r
 = \frac{4\pi}{G^2}\big(1 - e^{-G^2/4\omega^2}\big),
\qquad
\lim_{G\to 0} = \int\frac{\operatorname{erfc}(\omega r)}{r}d^3r = \frac{\pi}{\omega^2}.
$$

So per $\mathbf G\neq 0$, $V_\text{SR} + V_\text{LR}$ is the bare
$-4\pi Z/(\Omega G^2)$ kernel, and the real-space erfc sum carries the
$\mathbf G = 0$ average $-\pi Z_\text{tot}/(\omega^2\Omega)$, which $V_\text{G0}$
removes. $V$ is therefore $\omega$-independent and equals the pure
reciprocal-space attraction with $\mathbf G = 0$ dropped (the convention that
cancels against the same term of $J$ in a neutral cell).

Nuclei in $V_\text{SR}$ are unit-normalised s Gaussians of exponent
$\zeta = 10^{16}$ through the 3-centre engine (libint2's erf/erfc nuclear
kernels are wrong); the finite-nucleus error is
$Z\int\rho\,\operatorname{erfc}(\sqrt\zeta r)/r \approx \pi Z\rho(\mathbf R_C)/\zeta$.

Code: `hcore.rs:688` `reciprocal_nuclear` (kernel `hcore.rs:719`, half-sphere
factor $-2/\Omega$ `hcore.rs:730`), $V_\text{G0}$ `hcore.rs:1485-1486`,
oracle `hcore.rs:766` `pure_aft_nuclear` ($\omega\to\infty$, no SR, no
$\mathbf G = 0$ bookkeeping), `hcore.rs:796` `gaussian_nucleus_attraction`.

**Tests.** `tests/pbc_hcore.rs`
`periodic_hcore_is_omega_independent_triclinic_sp` ($V(\omega)$ at three
$\omega$ vs `pure_aft_nuclear`, with the negative control that $V_\text{G0}$ is
visible), `single_s_primitive_matches_closed_form`
($V_\text{erf} = -(2/\sqrt\pi)(1/\omega^2 + 1/p + 1/\zeta)^{-1/2}$),
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

With $k(G^2) = e^{-G^2/4\omega^2}/G^2$,
$k' = -k\,(1/4\omega^2 + 1/G^2)$:

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
`ewald.rs:477` (background). The total is $\omega$-independent; the parts are
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
overlap-coupled terms (Pulay, $c_0 = \pi/(\omega^2\Omega)$ and Madelung
$S$-terms) enter through the forces' energy-weighted matrix contracted with
per-image pair vectors; the Madelung energy adds
$-\tfrac{\alpha}{2}\sum_\sigma\operatorname{tr}(D_\sigma SD_\sigma S)\,dv_M/d\varepsilon$.
Truncated sets are frozen as integer indices at the reference cell
(`Cell::strained`), so the stress is the exact derivative of that energy.
The module doc of `stress.rs` lists the XC, ECP and RS-GDF terms.

**Tests.** `tests/pbc_stress.rs`
`ewald_strain_matches_fd_is_omega_independent_and_madelung_is_homogeneous`
(FD 1e-8, $\omega$ 1 vs 2.3 to 1e-10, Euler to 1e-12),
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
| Same, $k$-point | same number on the largest spectral norm per block, maximised over $k$ | `kuxcgrad.rs` `ro_orbital_gradient_k`, `kuxcstress.rs` | measured on H3/STO-3G 1x1x1 only (rank-1 blocks); the spectral norm is $\ge$ the Gamma max-element, so the $k$ gate can refuse a state the Gamma gate accepts. Constant deliberately shared |
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
`pbc_kstress*.rs` and `validation_*stress*.rs` files.
