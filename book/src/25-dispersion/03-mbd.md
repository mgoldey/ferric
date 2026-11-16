# 25.3 Many-body dispersion

<!-- section: 25.3; status: edit; registry: mbd;
     theory: crates/ferric-rpa/theory.md#"Many-body dispersion (MBD@rsSCS)", "Dispersion: Casimir-Polder C6" -->

A pairwise $C_6$ sum treats each atom's polarizability as fixed. In a
molecule the atoms' induced dipoles screen one another, and the dispersion
energy of the whole system is not a sum over pairs. Many-body dispersion
(MBD) models each atom as an isotropic quantum harmonic oscillator and couples
the oscillators through the dipole tensor. ferric implements MBD@rsSCS
(Ambrosetti, Reilly, DiStasio and Tkatchenko, JCP 140, 18A508 (2014)) as the
`[dft] dispersion = "mbd"` correction, term for term as libMBD and pymbd
0.15.0 build it, and the full-range self-consistent screening of Tkatchenko,
DiStasio, Car and Scheffler (PRL 108, 236402 (2012)) as the `mbd` source of
section 25.2.

The derivation, transcribed from the loops, is in
`crates/ferric-rpa/theory.md`, "Many-body dispersion (MBD@rsSCS)", and
`scripts/mbd_coupled_dipole_sympy.py` proves its identities and the gradient.

| Symbol | Meaning | Code name |
|---|---|---|
| $A, B$ | atoms; $3N$-dimensional dipole space with blocks $AB$ | `a`, `b`; `3 * a + i` |
| $r_A$ | Hirshfeld volume ratio $V_A/V_A^{\mathrm{free}}$ | `volume_ratios` |
| $\alpha_A, C_{6,A}, R_A$ | TS static polarizability, $C_6$, vdW radius | `MbdAtomParams::{alpha_0,c6,r_vdw}` |
| $\omega_A$ | oscillator frequency $\tfrac43C_{6,A}/\alpha_A^2$ | `omega` |
| $u_k, w_k$ | libMBD frequency grid | `mbd_freq_grid` |
| $\beta, a$ | range-separation scale, Fermi steepness | `MbdRsscsConfig::{beta,a}` |

| Entry point | Location |
|---|---|
| Rust, energy | `ferric-rpa/src/dispersion/mbd_rsscs.rs::{mbd_rsscs_energy,mbd_rsscs_energy_from_params}` |
| Rust, gradient | `ferric-rpa/src/dispersion/mbd_rsscs.rs::{mbd_rsscs_gradient,mbd_rsscs_gradient_from_params}`, `ferric-rpa/src/dispersion/mbd_scf.rs::mbd_rsscs_for_scf` |
| Rust, $C_6$ screening | `ferric-rpa/src/dispersion/mbd.rs::{mbd_screen,mbd_dynamic_polarizability,dipole_coupling_tensor}` |
| CLI | `[dft] dispersion = "mbd"` or `"mbd(<name>)"` on `ksdft`, `uhf`, `rohf` with a functional; `[rpa] c6_source = "mbd"` |
| Python | `run_dft(dispersion="mbd")`, `run_frequencies(xc=..., dispersion="mbd")`, `mbd_rsscs_energy(mol, volume_ratios, beta=, functional=)` |

## 25.3.1 Atomic oscillators

From the Hirshfeld volume ratio of the converged density and the free-atom
table (section 25.2.2),

$$ \alpha_A = r_A\alpha_A^{\mathrm{free}}, \qquad C_{6,A} = r_A^2C_{6,A}^{\mathrm{free}}, \qquad R_A = r_A^{1/3}R_A^{\mathrm{free}}, \qquad \alpha_A(iu) = \frac{\alpha_A}{1 + (u/\omega_A)^2} . \tag{25.12} $$

A dipole of polarizability $\alpha$ is given a Gaussian charge density of
width

$$ \sigma_A(u) = \Big(\sqrt{2/\pi}\;\alpha_A(iu)/3\Big)^{1/3}, \qquad \sigma_{AB} = \sqrt{\sigma_A^2 + \sigma_B^2} . \tag{25.13} $$

## 25.3.2 Dipole tensors

$$ \mathbf T^{\mathrm{bare}}_{AB} = -\nabla\nabla\frac1R = \frac{R^2\mathbf 1 - 3\mathbf R\mathbf R^{\mathsf T}}{R^5}, \qquad \mathbf T^{\mathrm{GG}}_{AB} = -\nabla\nabla\frac{\operatorname{erf}(R/\sigma_{AB})}{R} = (\operatorname{erf}z - \theta)\,\mathbf T^{\mathrm{bare}}_{AB} + 2z^2\theta\,\frac{\mathbf R\mathbf R^{\mathsf T}}{R^5}, \tag{25.14} $$

with $\mathbf R = \mathbf R_A - \mathbf R_B$, $z = R/\sigma_{AB}$ and
$\theta = 2ze^{-z^2}/\sqrt\pi$. $\mathbf T^{\mathrm{GG}}$ is the interaction
of two Gaussian dipole densities; it tends to $\mathbf T^{\mathrm{bare}}$ when
$R \gg \sigma_{AB}$ and stays finite as $R \to 0$. In this sign convention
$\mathbf A^{-1} + \mathbf T$ is the screening operator: on the bond axis
$T_{zz} = -2/R^3$, so parallel dipoles along the bond enhance each other.

The width (25.13) is not arbitrary. With $\sigma_{AA} = \sqrt2\,\sigma_A$, the
$R \to 0$ limit of $\mathbf T^{\mathrm{GG}}$ is $\mathbf 1/\alpha_A$: an
atom's own diagonal block $1/\alpha_A$ is the self-interaction of its
Gaussian dipole density.

> **Verified equation 25.14** · the Gaussian dipole tensor and its width
>
> $\mathbf T^{\mathrm{bare}} = -\nabla\nabla(1/R)$ symbolically. The coded
> $\mathbf T^{\mathrm{GG}}$ equals $-\nabla\nabla[\operatorname{erf}(R/s)/R]$
> (exact Hessian at 30 digits against the f64 transcription, $10^{-13}$
> relative), and `mbd.rs`'s $(\zeta, \eta)$ form of the same tensor is
> identical symbolically. With (25.13), $\lim_{R\to0}T^{\mathrm{GG}}_{zz} = 1/\alpha$.
>
> | | |
> |---|---|
> | Proof | `scripts/mbd_coupled_dipole_sympy.py`, checks `t-bare-is-hessian`, `t-gg-is-hessian`, `sigma-self-energy` |
> | Negative control | the runner drops the factor 2 of the $2z^2\theta$ term, and sets the width prefactor to $1/3$; each turns its check red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/dispersion/mbd_rsscs.rs::{t_bare,t_gg}`, `ferric-rpa/src/dispersion/mbd.rs::dipole_coupling_tensor` |
> | Code test | `mbd_rsscs_matches_pymbd_and_libmbd` in `crates/ferric-rpa/tests/validation_mbd_rsscs.rs` |
>
> <!-- verified: id=disp.mbd.dipole-tensor; script=scripts/mbd_coupled_dipole_sympy.py; claim=t-gg-is-hessian; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/dispersion/mbd_rsscs.rs::{t_bare,t_gg}, ferric-rpa/src/dispersion/mbd.rs::dipole_coupling_tensor; rust_test=ferric-rpa/tests/validation_mbd_rsscs.rs::mbd_rsscs_matches_pymbd_and_libmbd -->

## 25.3.3 Range-separated self-consistent screening

The short-range part of the coupling screens each atom's polarizability. With
the Fermi function

$$ f_{AB}(R) = \Big[1 + e^{-a\,(R/S_{AB} - 1)}\Big]^{-1}, \qquad S_{AB} = \beta\,(R_A + R_B), \qquad a = 6, \tag{25.15} $$

at each node $u_k$

$$ \mathbf C(u_k) = \operatorname{diag}\big(1/\alpha_A(iu_k)\big) + (1 - f)\,\mathbf T^{\mathrm{GG}}, \qquad \alpha^{\mathrm{rsSCS}}_A(iu_k) = \tfrac13\operatorname{tr}\sum_B\big[\mathbf C(u_k)^{-1}\big]_{AB} . \tag{25.16} $$

The row-block sum over $B$ assigns to atom $A$ the dipole it carries when a
uniform field polarizes the whole coupled system. From the screened values,

$$ \alpha_A^{\mathrm{rsSCS}} = \alpha^{\mathrm{rsSCS}}_A(0), \quad C_{6,A}^{\mathrm{rsSCS}} = \frac3\pi\sum_k w_k\,\alpha^{\mathrm{rsSCS}}_A(iu_k)^2, \quad \omega_A^{\mathrm{rsSCS}} = \frac43\frac{C_{6,A}^{\mathrm{rsSCS}}}{(\alpha_A^{\mathrm{rsSCS}})^2}, \quad R_A^{\mathrm{rsSCS}} = R_A\Big(\frac{\alpha_A^{\mathrm{rsSCS}}}{\alpha_A}\Big)^{1/3}. \tag{25.17} $$

The grid is libMBD's: 15 Gauss–Legendre nodes mapped by
$u = 0.6\,(1+x)/(1-x)$ plus a $u = 0$ node of weight 0 that supplies the
static value. $f(0) = 1/(1 + e^a) \approx 2.5\times10^{-3}$, so a large
$\beta$ does not switch the long-range coupling off completely.

> **Verified equation 25.16** · screening, in its trivial and dimer limits
>
> With no coupling the coded screening returns $\alpha_A(iu_k)$ exactly at
> every node, and (25.17) on the 15-node grid returns the TS $C_6$ to
> about $10^{-8}$ relative (bar $10^{-6}$), so $\omega$ follows to the same precision and $R$ is unchanged. For a homonuclear
> dimer on the $z$ axis the row-block contraction equals the closed form
> $\tfrac13\sum_d\alpha/(1 + \alpha t_d)$, $t = (1-f)\operatorname{diag}\mathbf T^{\mathrm{GG}}$.
>
> | | |
> |---|---|
> | Proof | `scripts/mbd_coupled_dipole_sympy.py`, checks `scs-trivial-limit`, `scs-dimer-closed-form` |
> | Negative control | the contraction over the diagonal block $B = A$ only turns `scs-dimer-closed-form` red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/dispersion/mbd_rsscs.rs::{rsscs_screen,mbd_freq_grid}` |
> | Code test | `mbd_dimer_matches_closed_form` in `crates/ferric-rpa/tests/validation_ts_mbd.rs` (the full-range screening of section 25.3.6) |
>
> <!-- verified: id=disp.mbd.rsscs-screening; script=scripts/mbd_coupled_dipole_sympy.py; claim=scs-dimer-closed-form; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/dispersion/mbd_rsscs.rs::{rsscs_screen,mbd_freq_grid}; rust_test=ferric-rpa/tests/validation_ts_mbd.rs::mbd_dimer_matches_closed_form -->

## 25.3.4 The coupled-oscillator energy

The long-range coupling acts between the screened oscillators. With all
quantities from (25.17) and $f$ evaluated with the screened radii,

$$ \mathbf H_{AB} = \omega_A^2\,\delta_{AB}\mathbf 1 + \omega_A\omega_B\sqrt{\alpha_A\alpha_B}\,f_{AB}\,\mathbf T^{\mathrm{bare}}_{AB}, \qquad E_{\mathrm{MBD}} = \frac12\sum_p\sqrt{\lambda_p} - \frac32\sum_A\omega_A, \tag{25.18} $$

where $\lambda_p$ are the $3N$ eigenvalues of $\mathbf H$: the zero-point energy
of the coupled oscillators minus that of the free ones. The same number is a
frequency integral, the dipole form of the dRPA energy (18.4):

$$ E_{\mathrm{MBD}} = \frac{1}{2\pi}\int_0^\infty \ln\det\big(\mathbf 1 + \mathbf A(iu)\,\mathbf T\big)\,du, \qquad \mathbf A(iu) = \operatorname{diag}\big(\alpha_A(iu)\big), \quad \mathbf T_{AB} = f_{AB}\mathbf T^{\mathrm{bare}}_{AB} . \tag{25.19} $$

(25.19) follows from $\int_0^\infty\ln\frac{u^2 + a^2}{u^2 + b^2}\,du = \pi(a - b)$
and $\det(u^2 + \mathbf H) = \det(u^2 + \boldsymbol\Omega^2)\det(\mathbf 1 + \mathbf A(iu)\mathbf T)$.
Expanding the logarithm to second order in $\mathbf T$ (the first-order term
is zero: $\mathbf T$ has no diagonal blocks) gives
$-\sum_{A<B}f_{AB}^2C_6^{AB}/R_{AB}^6$ with the Casimir–Polder $C_6$ of
(25.8): pairwise dispersion is the leading term of MBD.

> **Verified equation 25.19** · oscillators, frequency integral, London limit
>
> On a random four-atom geometry with Fermi-damped coupling, (25.18) equals a
> 2000-node quadrature of (25.19) to $10^{-10}$ relative. For two atoms the
> coupling-squared term of (25.18) is $-f^2\cdot\tfrac32\alpha_A\alpha_B\omega_A\omega_B/(\omega_A + \omega_B)/R^6$
> symbolically, the London $C_6$.
>
> | | |
> |---|---|
> | Proof | `scripts/mbd_coupled_dipole_sympy.py`, checks `qho-vs-frequency`, `qho-second-order-london` |
> | Negative control | the reference energy $-\tfrac32\sum\omega_A$ written as $-\sum\omega_A$ turns `qho-vs-frequency` red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/dispersion/mbd_rsscs.rs::rsscs_forward`, `ferric-rpa/src/dispersion/mbd.rs::coupled_qho_energy_plain_gg` |
> | Code test | `mbd_energy_tends_to_ts_pair_c6` in `crates/ferric-rpa/tests/validation_ts_mbd.rs` |
>
> <!-- verified: id=disp.mbd.qho-energy; script=scripts/mbd_coupled_dipole_sympy.py; claim=qho-vs-frequency; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/dispersion/mbd_rsscs.rs::rsscs_forward, ferric-rpa/src/dispersion/mbd.rs::coupled_qho_energy_plain_gg; rust_test=ferric-rpa/tests/validation_ts_mbd.rs::mbd_energy_tends_to_ts_pair_c6 -->

If $\mathbf H$ has a non-positive eigenvalue the oscillators are unstable
(a polarization catastrophe: atoms too close for this $\beta$) and
`rsscs_forward` returns an error. A singular $\mathbf C(u_k)$ or a
non-positive screened polarizability is also an error. None of these returns
a substitute number.

| Symbol | Code (`rsscs_screen`, `rsscs_forward`) |
|---|---|
| $\mathbf C(u_k)$, $\mathbf C^{-1}$ | `c`, `cinv` |
| $1 - f$ | `sr` |
| $\alpha^{\mathrm{rsSCS}}_A(iu_k)$ | `a_dyn[k][a]` |
| $\alpha^{\mathrm{rsSCS}}_A, C^{\mathrm{rsSCS}}_{6,A}, \omega^{\mathrm{rsSCS}}_A, R^{\mathrm{rsSCS}}_A$ | `alpha_0_rsscs`, `c6_rsscs`, `omega_rsscs`, `r_vdw_rsscs` |
| $\mathbf H$, $\lambda_p$ | `h`, `evals` |

## 25.3.5 The gradient

`mbd_rsscs_gradient_from_params` differentiates the forward pass in reverse
(adjoint) mode at the cost of the energy, $O(n_u(3N)^3)$. Its first step is

$$ \frac{\partial E_{\mathrm{MBD}}}{\partial\mathbf H} = \tfrac14\,\mathbf H^{-1/2} = \tfrac14\,\mathbf V\operatorname{diag}(\lambda_p^{-1/2})\mathbf V^{\mathsf T}, \tag{25.20} $$

from which the adjoint runs back through $\mathbf H$, (25.17), the inverses
$\mathbf C(u_k)^{-1}$ kept on a tape, the Fermi functions and the dipole
tensors, to $\partial E/\partial\mathbf R_A$ at fixed TS inputs and to
$\partial E/\partial\alpha_A$, $\partial E/\partial C_{6,A}$,
$\partial E/\partial R_A$.

> **Verified equation 25.20** · the adjoint seed
>
> $\langle\tfrac14\mathbf H^{-1/2}, \delta\mathbf H\rangle$ equals a central
> finite difference of $\tfrac12\sum_p\sqrt{\lambda_p}$ along a random
> symmetric $\delta\mathbf H$.
>
> | | |
> |---|---|
> | Proof | `scripts/mbd_coupled_dipole_sympy.py`, check `energy-adjoint` |
> | Negative control | the seed $\tfrac12\lambda^{-1/2}$ turns the check red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/dispersion/mbd_rsscs.rs::rsscs_backward` |
> | Code test | `mbd_rsscs_gradient_matches_libmbd` in `crates/ferric-rpa/tests/validation_mbd_rsscs.rs` |
>
> <!-- verified: id=disp.mbd.adjoint-seed; script=scripts/mbd_coupled_dipole_sympy.py; claim=energy-adjoint; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/dispersion/mbd_rsscs.rs::rsscs_backward; rust_test=ferric-rpa/tests/validation_mbd_rsscs.rs::mbd_rsscs_gradient_matches_libmbd -->

> **Verified equation 25.20** · the whole backward pass
>
> The script transcribes `rsscs_backward` (the five adjoint steps, the Fermi
> partials and the vector-Jacobian products of $\mathbf T^{\mathrm{bare}}$ and
> $\mathbf T^{\mathrm{GG}}$). The vector-Jacobian products equal sympy's
> gradient of $\sum M_{ij}T_{ij}$ to $7\times10^{-15}$ relative (bar
> $10^{-12}$). The transcribed gradient equals a Richardson difference of the
> transcribed energy on a four-atom model: positions $1.2\times10^{-8}$,
> $\alpha_A$ $4.9\times10^{-9}$, $C_{6,A}$ $2.0\times10^{-9}$, $R_A$
> $1.6\times10^{-10}$ relative, each under 50 times the round-off floor
> $\epsilon S/(h\max|g|)$ with $S = \tfrac12\sum\sqrt{\lambda}$. The position
> gradient sums to zero ($10^{-20}$), and the chain rule to volume ratios
> agrees with a difference in $r_A$ to $2.5\times10^{-9}$ (bar $10^{-7}$).
>
> | | |
> |---|---|
> | Proof | `scripts/mbd_coupled_dipole_sympy.py`, checks `vjp-partials`, `gradient-vs-fd`, `gradient-translation`, `ratio-chain-rule` |
> | Negative control | the runner flips a sign in $\partial g_2/\partial\sigma$, uses $3/\pi$ for the $6/\pi$ of step 4, uses $\tfrac12$ for the $\tfrac13$ of the $\tilde R$ chain, drops the static adjoint at the $u = 0$ node, uses $r^{-1/3}$ for $r^{-2/3}$ in the ratio chain; each turns its check red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/dispersion/mbd_rsscs.rs::mbd_rsscs_gradient_from_params`, `ferric-rpa/src/dispersion/mbd_rsscs.rs::mbd_rsscs_gradient` |
> | Code test | `mbd_rsscs_gradient_matches_libmbd` in `crates/ferric-rpa/tests/validation_mbd_rsscs.rs` |
>
> <!-- verified: id=disp.mbd.backward-pass; script=scripts/mbd_coupled_dipole_sympy.py; claim=gradient-vs-fd; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/dispersion/mbd_rsscs.rs::{mbd_rsscs_gradient_from_params,mbd_rsscs_gradient}; rust_test=ferric-rpa/tests/validation_mbd_rsscs.rs::mbd_rsscs_gradient_matches_libmbd; registry=mbd -->

On an SCF density the volume ratios move with the nuclei too.
`mbd_rsscs_for_scf` assembles

| Term | What it carries | Function |
|---|---|---|
| explicit | $\partial E/\partial\mathbf R_B$ at fixed volume ratios | `mbd_rsscs_gradient` |
| Hirshfeld volumes | $\sum_Ac_A\,\partial v_A/\partial\mathbf R_B$ at fixed density and lattice, $c_A = (\partial E/\partial r_A)/v_A^{\mathrm{free}}$ | `ferric-scf/src/properties/hirshfeld_grid.rs::hirshfeld_volume_gradient` |
| orthonormality | $-\tfrac12\operatorname{tr}[\mathbf V\mathbf D\mathbf S^x\mathbf D]$ (per spin for UKS; a closed–open term for ROKS), $\mathbf V = \partial(\sum_Ac_Av_A)/\partial\mathbf D$ | `ferric-rpa/src/dispersion/mbd_scf.rs::orthonormality_term` |
| lattice | the volume lattice (`ferric-scf/src/properties/hirshfeld_grid.rs::mbd_volume_grid`, centred on the atom centroid) moves by $1/N$ of each displacement; translation invariance turns this into minus $1/N$ of the summed fixed-lattice term | `mbd_rsscs_for_scf` |
| relaxation | $\sum_{ai}Z_{ai}\,\partial F_{ai}/\partial\mathbf R_B$ with the Kohn–Sham Z-vector driven by $\mathbf V$ | `ferric_scf::zvector_ks` |

The total is checked against central finite differences of the full
SCF + MBD energy (25.3.8). The orthonormality and lattice terms are also
proved on their own:

> **Verified** · the orthonormality and lattice terms
>
> With $\delta C_o = -\tfrac12C_oS^x_{oo}$ (Löwdin), the change of
> $\operatorname{tr}[\mathbf V\mathbf D]$ along an overlap derivative equals
> the code's closed-shell, UKS and ROKS traces to $1.0\times10^{-12}$,
> $2.2\times10^{-12}$ and $1.2\times10^{-12}$ (bar $10^{-9}$); the ROKS form
> $-\operatorname{tr}[\mathbf S^x\mathbf W_Q]$,
> $\mathbf W_Q = \mathbf D_\alpha\mathbf V\mathbf D_\alpha + \mathbf P_c\mathbf V\mathbf P_c + \tfrac12(\mathbf P_c\mathbf V\mathbf P_o + \mathbf P_o\mathbf V\mathbf P_c)$,
> orthonormalizes the closed and open orbitals as one set. On a toy volume
> sum with a centroid-anchored lattice, the fixed-lattice term minus $1/N$ of
> its atom sum equals the total derivative to $3.2\times10^{-11}$ relative
> (bar $10^{-8}$) while the lattice term is 7.5% of the gradient.
>
> | | |
> |---|---|
> | Proof | `scripts/mbd_coupled_dipole_sympy.py`, checks `orthonormality-term`, `lattice-term` |
> | Negative control | the runner uses $-1$ for the closed-shell $-\tfrac12$, drops the closed–open term of $\mathbf W_Q$, drops the $-1/N$ lattice shift; each turns its check red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/dispersion/mbd_scf.rs::orthonormality_term`, `ferric-rpa/src/dispersion/mbd_scf.rs::mbd_rsscs_for_scf` |
> | Code test | `mbd_exact_gradient_matches_fd_of_full_scf_pipeline` in `crates/ferric-rpa/tests/mbd_scf_gradient.rs` |
>
> <!-- verified: id=disp.mbd.scf-terms; script=scripts/mbd_coupled_dipole_sympy.py; claim=orthonormality-term; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/dispersion/mbd_scf.rs::{orthonormality_term,mbd_rsscs_for_scf}; rust_test=ferric-rpa/tests/mbd_scf_gradient.rs::mbd_exact_gradient_matches_fd_of_full_scf_pipeline; registry=mbd -->

The Hirshfeld volume term and the Z-vector are not transcribed by the
script; they rest on `hirshfeld_volume_gradient_matches_fd_slater_proatom`
and the full-pipeline difference test.

## 25.3.6 Full-range screening for $C_6$

`[rpa] c6_source = "mbd"` screens the TS poles of section 25.2 with the
full-range tensor and no Fermi function, at each node of the RPA grid:

$$ \mathbf C(u) = \operatorname{diag}\big(1/\bar\alpha_A(iu)\big) + \mathbf T^{\mathrm{GG}}, \qquad \alpha^{\mathrm{SCS}}_A(iu) = \sum_B\big[\mathbf C(u)^{-1}\big]_{AB}, \tag{25.21} $$

with $\bar\alpha_A(iu)$ the isotropic TS value (the static per-atom tensor's
shape is not inverted) and the full $3\times3$ block sum kept as the
per-atom tensor. The molecular tensor is the sum over atoms, and
`casimir_polder_c6` takes it from there. `coupled_qho_energy_plain_gg`
evaluates (25.18) with this full-range Gaussian coupling as a check of the
oscillator algebra against libMBD; it is not a dispersion energy
(`site/src/reference/validation.md`, row "TS C6 and MBD@TS").

## 25.3.7 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | Hirshfeld volumes of the SCF density and of each free atom on one lattice (free-atom SCFs cached per element) | grid × basis | `ferric-rpa/src/dispersion/mbd_scf.rs::MbdFreeAtomCache`, `ferric-scf/src/properties/hirshfeld_grid.rs::atomic_effective_volumes_hirshfeld_on_grid` |
| 2 | TS inputs (25.12) | $N$ | `ferric-rpa/src/dispersion/mbd_rsscs.rs::ts_params_from_volume_ratios` |
| 3 | per node: $\mathbf C(u_k)$, inverse, (25.16) | $16\,(3N)^3$ | `rsscs_screen` |
| 4 | (25.17), $\mathbf H$, eigensolve, (25.18) | $(3N)^3$ | `rsscs_forward` |
| 5 | gradient: adjoint (25.20) back through steps 4 and 3; then the SCF terms | $O(16\,(3N)^3)$ plus one Z-vector | `rsscs_backward`, `mbd_rsscs_for_scf` |

## 25.3.8 Options and anchors

| Option | Values | Effect |
|---|---|---|
| `[dft] dispersion` | `"mbd"` (the running functional's $\beta$), `"mbd(<name>)"` | $\beta$ = 0.83 (PBE), 0.85 (PBE0, HSE06); any other functional is refused |
| `mbd_rsscs_energy(beta=)` | positive float | an explicit $\beta$ for the standalone energy |
| $a$, $n_u$ | 6, 15 | libMBD defaults (`MBD_FERMI_A_DEFAULT`, `MBD_N_FREQ_DEFAULT`); not exposed in the CLI |

From `site/src/reference/validation.md` (rows "MBD@rsSCS energy", "MBD@rsSCS
nuclear gradient" and "TS C6 and MBD@TS"):

| Quantity | Reference | Error |
|---|---|---|
| $E_{\mathrm{MBD}}$, H2O, CO, CH3OH, CH4, benzene, two $\beta$ | pymbd 0.15.0 and libMBD 0.15.0, same inputs | $\le 9.5\times10^{-12}$ rel. |
| screened $\alpha_0$, $C_6$, $\omega$, $R$ | same | $\le 1.5\times10^{-14}$ rel. |
| explicit gradient | libMBD `force=True` | $6.2\times10^{-16}$ abs. |
| exact SCF gradient, H2O / 6-31G (PBE, PBE0, HSE06, PBE + RI-J) | central FD of the full SCF + MBD pipeline | $\le 3.5\times10^{-9}$ Ha/Bohr |
| full-range SCS $\alpha(iu)$ and $C_6$ | libMBD `scs`, mpmath dimer closed forms | see the validation row |

## Known limits (2026-11-15)

- Finite systems only; $Z \le 54$; $\beta$ published for PBE, PBE0 and HSE06
  only.
- The exact SCF gradient needs a Kohn–Sham Z-vector for the functional;
  `mbd_rsscs_for_scf` names the reason when there is none.
- The Hirshfeld volume gradient and the Z-vector relaxation of 25.3.5 rest
  on finite-difference tests, not on a proof script.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-rpa/src/dispersion/mbd_rsscs.rs` (module doc and loops) | (25.12)–(25.20) |
| `crates/ferric-rpa/src/dispersion/mbd_scf.rs` (module doc) | the SCF gradient terms |
| `crates/ferric-rpa/src/dispersion/mbd.rs` | (25.21) |
| `scripts/mbd_coupled_dipole_sympy.py` | verified boxes |
| `site/src/reference/validation.md` | anchors |

Changes from the sources: the code and its docs write the imaginary frequency
as $u$ (MBD) and $\omega$ (`mbd.rs`); this section writes $u$ throughout.
The doc comment of `dipole_coupling_tensor` states the tensor with the
opposite overall sign to the loop; (25.14) follows the loop, which agrees
with `t_gg`.
