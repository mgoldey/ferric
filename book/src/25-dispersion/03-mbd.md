# Many-body dispersion {#mbd}

<!-- status: edit; registry: mbd -->

A pairwise $C_6$ sum gives each atom a fixed polarizability and lets each
pair interact as if the other atoms were absent. In a molecule they are not
absent. A fluctuating dipole on atom $A$ polarizes $B$, and $B$'s induced
dipole acts back on $A$ and on every other atom: the neighbours screen each
fluctuation, and the dispersion energy of the whole system is not a sum over
pairs. Many-body dispersion (MBD) models each atom as an isotropic quantum
harmonic oscillator, couples the oscillators through the dipole tensor, and
takes the energy from the coupled modes.

ferric implements MBD@rsSCS (Ambrosetti, Reilly, DiStasio and Tkatchenko,
JCP 140, 18A508 (2014)) as the `[dft] dispersion = "mbd"` correction, term for
term as libMBD and pymbd 0.15.0 build it, and the full-range self-consistent
screening of Tkatchenko, DiStasio, Car and Scheffler (PRL 108, 236402 (2012))
as the `mbd` source of section @sec:casimir-polder-c6. Where the papers and libMBD could
differ, the code follows libMBD.

The plan has two stages, as the method does. First, short-range coupling
screens each atom's polarizability (@sec:mbd-atomic-oscillators–@sec:mbd-range-separated-self-consistent). Then the screened atoms
couple at long range, and the energy is the zero-point energy of the coupled
oscillators (@sec:mbd-coupled-oscillator-energy). The gradient (@sec:mbd-gradient), the $C_6$ screening
(@sec:mbd-full-range-screening), the algorithm and the anchors follow.

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

Atomic units throughout. `[dft] dispersion` takes one value, so a run
carries either D3(BJ) or MBD, never both: MBD already contains the pairwise
term (@sec:mbd-coupled-oscillator-energy), and adding D3 on top would count it twice.

## Atomic oscillators {#mbd-atomic-oscillators}

Each atom starts as the Tkatchenko–Scheffler oscillator of section @sec:casimir-polder-c6-three-sources,
built from the Hirshfeld volume ratio of the converged density and the
free-atom table (`ts_params_from_volume_ratios`):

$$ \alpha_A = r_A\alpha_A^{\mathrm{free}}, \qquad C_{6,A} = r_A^2C_{6,A}^{\mathrm{free}}, \qquad R_A = r_A^{1/3}R_A^{\mathrm{free}}, \qquad \alpha_A(iu) = \frac{\alpha_A}{1 + (u/\omega_A)^2} . \label{eq:mbd-1} $$

The vdW radius scales as the cube root of the volume, as a length should.
$\omega_A = \tfrac43C_{6,A}/\alpha_A^2$ is the homonuclear London identity of
@sec:casimir-polder-c6-casimir-polder-integral. A volume ratio that is not finite and positive, and an element
outside $Z = 1$–$54$, are errors (`ts_free_atom`, `ts_free_atom_r_vdw`).

A point dipole interacting with another point dipole diverges as the two
approach. MBD smears each dipole into a Gaussian charge density of width

$$ \sigma_A(u) = \Big(\sqrt{2/\pi}\;\alpha_A(iu)/3\Big)^{1/3}, \qquad \sigma_{AB} = \sqrt{\sigma_A^2 + \sigma_B^2} . \label{eq:mbd-2} $$

A more polarizable atom is a larger cloud: $\sigma_A^3 \propto \alpha_A(iu)$.
$\sigma_{AB}$ is the width of the convolution of two Gaussians, which is
what the interaction of two Gaussian densities depends on.

## Dipole tensors {#mbd-dipole-tensors}

The energy of dipole $\boldsymbol\mu_A$ in the field of dipole
$\boldsymbol\mu_B$ is $\boldsymbol\mu_A^{\mathsf T}\mathbf T_{AB}\boldsymbol\mu_B$,
with $\mathbf T$ the second derivative of the Coulomb interaction. For
point dipoles and for Gaussian ones,

$$ \mathbf T^{\mathrm{bare}}_{AB} = -\nabla\nabla\frac1R = \frac{R^2\mathbf 1 - 3\mathbf R\mathbf R^{\mathsf T}}{R^5}, \qquad \mathbf T^{\mathrm{GG}}_{AB} = -\nabla\nabla\frac{\operatorname{erf}(R/\sigma_{AB})}{R} = (\operatorname{erf}z - \theta)\,\mathbf T^{\mathrm{bare}}_{AB} + 2z^2\theta\,\frac{\mathbf R\mathbf R^{\mathsf T}}{R^5}, \label{eq:disp-mbd-dipole-tensor} $$

with $\mathbf R = \mathbf R_A - \mathbf R_B$, $z = R/\sigma_{AB}$ and
$\theta = 2ze^{-z^2}/\sqrt\pi$. $\operatorname{erf}(R/\sigma)/R$ is the
Coulomb potential between two Gaussian charges; differentiate it twice. For
$R \gg \sigma_{AB}$, $\operatorname{erf}z \to 1$ and $\theta \to 0$, so
$\mathbf T^{\mathrm{GG}} \to \mathbf T^{\mathrm{bare}}$. As $R \to 0$ it
stays finite. In this sign convention $\mathbf A^{-1} + \mathbf T$ is the
screening operator: on the bond axis $T_{zz} = -2/R^3$, so parallel dipoles
along the bond enhance each other.

The width (@eq:mbd-2) is not arbitrary. With $\sigma_{AA} = \sqrt2\,\sigma_A$, the
$R \to 0$ limit of $\mathbf T^{\mathrm{GG}}$ is $\mathbf 1/\alpha_A$: an
atom's own diagonal block $1/\alpha_A$ is the self-interaction of its
Gaussian dipole density. That is why the next section can put
$1/\alpha_A$ on the diagonal and $\mathbf T^{\mathrm{GG}}$ off it in one
matrix.

> **Verified equation @eq:disp-mbd-dipole-tensor** · the Gaussian dipole tensor and its width
>
> $\mathbf T^{\mathrm{bare}} = -\nabla\nabla(1/R)$ symbolically. The coded
> $\mathbf T^{\mathrm{GG}}$ equals $-\nabla\nabla[\operatorname{erf}(R/s)/R]$
> (exact Hessian at 30 digits against the f64 transcription, $10^{-13}$
> relative), and `mbd.rs`'s $(\zeta, \eta)$ form of the same tensor is
> identical symbolically. With (@eq:mbd-2), $\lim_{R\to0}T^{\mathrm{GG}}_{zz} = 1/\alpha$.
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

`ferric-rpa/src/dispersion/mbd.rs::dipole_coupling_tensor`, used by the
$C_6$ path of @sec:mbd-full-range-screening, writes the same tensor as
$\zeta'(\mathbf 1 - 3\hat n\hat n^{\mathsf T})/R^3 + \eta\,3\hat n\hat n^{\mathsf T}/R^3$
with $\zeta' = \operatorname{erf}z - 2ze^{-z^2}/\sqrt\pi$ and
$\eta = 4z^3e^{-z^2}/(3\sqrt\pi)$; the box shows it equals $\mathbf T^{\mathrm{GG}}$.
That path uses an Abramowitz–Stegun `erf`; `mbd_rsscs.rs` uses the
double-precision `erf` of `ferric_core::special`.

## Range-separated self-consistent screening {#mbd-range-separated-self-consistent}

Which part of the coupling should screen the polarizabilities, and which
part should give the energy? MBD@rsSCS splits $\mathbf T$ by distance with a
Fermi function. The short-range part screens each atom's polarizability.
The long-range part gives the dispersion energy of @sec:mbd-coupled-oscillator-energy. The switch is

$$ f_{AB}(R) = \Big[1 + e^{-a\,(R/S_{AB} - 1)}\Big]^{-1}, \qquad S_{AB} = \beta\,(R_A + R_B), \qquad a = 6, \label{eq:mbd-4} $$

which is $\tfrac12$ at $R = S_{AB}$ and tends to 1 at long range. $\beta$ is
the one parameter fitted per functional. At each node $u_k$ the short-range
part $(1 - f)\mathbf T^{\mathrm{GG}}$ couples the atoms:

$$ \mathbf C(u_k) = \operatorname{diag}\big(1/\alpha_A(iu_k)\big) + (1 - f)\,\mathbf T^{\mathrm{GG}}, \qquad \alpha^{\mathrm{rsSCS}}_A(iu_k) = \tfrac13\operatorname{tr}\sum_B\big[\mathbf C(u_k)^{-1}\big]_{AB} . \label{eq:disp-mbd-rsscs-screening} $$

Where does this come from? Put a uniform field $\mathbf E$ on the system.
Each induced dipole satisfies
$\boldsymbol\mu_A = \alpha_A(\mathbf E - \sum_B\mathbf T_{AB}\boldsymbol\mu_B)$.
Divide by $\alpha_A$ and collect: $\mathbf C\boldsymbol\mu = \mathbf E$ for
every atom, so $\boldsymbol\mu = \mathbf C^{-1}\mathbf E$. Atom $A$'s dipole
in a uniform field is the row-block sum $\sum_B[\mathbf C^{-1}]_{AB}\mathbf E$,
and its isotropic part is the screened polarizability of (@eq:disp-mbd-rsscs-screening).

Check the two limits. With no coupling $\mathbf C$ is diagonal and
$\alpha^{\mathrm{rsSCS}}_A(iu_k) = \alpha_A(iu_k)$. For a homonuclear dimer
on the $z$ axis $\mathbf C^{-1}$ is block-diagonal by direction, and the
contraction is $\tfrac13\sum_d\alpha/(1 + \alpha t_d)$ with
$t = (1-f)\operatorname{diag}\mathbf T^{\mathrm{GG}}$. Where
$\mathbf T^{\mathrm{GG}} \approx \mathbf T^{\mathrm{bare}}$, $t_z < 0$ along
the bond and the polarizability grows in that direction, while
$t_x = t_y > 0$ across it and it shrinks there.

From the screened values, define new oscillators. The static value is the
$u = 0$ node, $C_6$ comes from the Casimir–Polder integral (@eq:casimir-polder-c6-1) of each
atom with itself, the frequency from the London identity, and the radius
from the cube-root scaling of @sec:mbd-atomic-oscillators:

$$ \alpha_A^{\mathrm{rsSCS}} = \alpha^{\mathrm{rsSCS}}_A(0), \quad C_{6,A}^{\mathrm{rsSCS}} = \frac3\pi\sum_k w_k\,\alpha^{\mathrm{rsSCS}}_A(iu_k)^2, \quad \omega_A^{\mathrm{rsSCS}} = \frac43\frac{C_{6,A}^{\mathrm{rsSCS}}}{(\alpha_A^{\mathrm{rsSCS}})^2}, \quad R_A^{\mathrm{rsSCS}} = R_A\Big(\frac{\alpha_A^{\mathrm{rsSCS}}}{\alpha_A}\Big)^{1/3}. \label{eq:mbd-6} $$

The grid is libMBD's: 15 Gauss–Legendre nodes mapped by
$u = 0.6\,(1+x)/(1-x)$ (`MBD_FREQ_GRID_L`), listed in reverse order after a
$u = 0$ node of weight 0 that supplies the static value. The Fermi function
is not zero at $R = 0$: $f(0) = 1/(1 + e^a) \approx 2.5\times10^{-3}$, so a
large $\beta$ scales the long-range coupling by $1/(1 + e^a)$ instead of
switching it off completely.

> **Verified equation @eq:disp-mbd-rsscs-screening** · screening, in its trivial and dimer limits
>
> With no coupling the coded screening returns $\alpha_A(iu_k)$ exactly at
> every node, and (@eq:mbd-6) on the 15-node grid returns the TS $C_6$ to
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
> | Code test | `mbd_dimer_matches_closed_form` in `crates/ferric-rpa/tests/validation_ts_mbd.rs` (the full-range screening of section @sec:mbd-full-range-screening) |
>
> <!-- verified: id=disp.mbd.rsscs-screening; script=scripts/mbd_coupled_dipole_sympy.py; claim=scs-dimer-closed-form; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/dispersion/mbd_rsscs.rs::{rsscs_screen,mbd_freq_grid}; rust_test=ferric-rpa/tests/validation_ts_mbd.rs::mbd_dimer_matches_closed_form -->

The contraction in `ferric-rpa/src/dispersion/mbd_rsscs.rs::rsscs_screen`,
after the inverse, unchanged:

<!-- excerpt: ferric-rpa/src/dispersion/mbd_rsscs.rs::rsscs_screen -->
```rust
let mut tr = 0.0;
for b in 0..n {
    for i in 0..3 {
        tr += cinv[(3 * a + i, 3 * b + i)];
    }
}
*slot = tr / 3.0;
```

`cinv` is $\mathbf C(u_k)^{-1}$. The inner loop over `i` takes the trace of
block $AB$, and the loop over `b` sums the row of blocks. A singular
$\mathbf C(u_k)$, or a screened polarizability that is not finite and
positive, is an error, never a substituted number.

## The coupled-oscillator energy {#mbd-coupled-oscillator-energy}

Now the long-range part. Give each screened atom a three-dimensional
oscillator with charge $q_A$, mass $m_A$ and frequency $\omega_A$; its
polarizability is $\alpha_A = q_A^2/(m_A\omega_A^2)$. The potential energy
is $\tfrac12\sum_A m_A\omega_A^2\mathbf x_A^2$ plus the dipole coupling
$\sum_{A<B}q_Aq_B\,\mathbf x_A^{\mathsf T}f_{AB}\mathbf T^{\mathrm{bare}}_{AB}\mathbf x_B$.
Change to mass-weighted coordinates $\boldsymbol\xi_A = \sqrt{m_A}\,\mathbf x_A$,
so the kinetic energy is $\tfrac12\sum\dot{\boldsymbol\xi}^2$, and use
$q_A/\sqrt{m_A} = \omega_A\sqrt{\alpha_A}$. The Hessian of the potential is
then

$$ \mathbf H_{AB} = \omega_A^2\,\delta_{AB}\mathbf 1 + \omega_A\omega_B\sqrt{\alpha_A\alpha_B}\,f_{AB}\,\mathbf T^{\mathrm{bare}}_{AB}, \qquad E_{\mathrm{MBD}} = \frac12\sum_p\sqrt{\lambda_p} - \frac32\sum_A\omega_A, \label{eq:mbd-7} $$

with all quantities from (@eq:mbd-6) and $f$ evaluated with the screened radii.
The normal-mode frequencies are $\sqrt{\lambda_p}$, the square roots of the
$3N$ eigenvalues of $\mathbf H$. Each mode carries the zero-point energy
$\tfrac12\sqrt{\lambda_p}$, each free three-dimensional oscillator
$\tfrac32\omega_A$. $E_{\mathrm{MBD}}$ is the zero-point energy of the
coupled oscillators minus that of the free ones. The screening is inside:
diagonalizing $\mathbf H$ sums the coupling to every order.

The same number is a frequency integral, the dipole form of the dRPA energy
(@eq:drpa-three-ways-4):

$$ E_{\mathrm{MBD}} = \frac{1}{2\pi}\int_0^\infty \ln\det\big(\mathbf 1 + \mathbf A(iu)\,\mathbf T\big)\,du, \qquad \mathbf A(iu) = \operatorname{diag}\big(\alpha_A(iu)\big), \quad \mathbf T_{AB} = f_{AB}\mathbf T^{\mathrm{bare}}_{AB} . \label{eq:disp-mbd-qho-energy} $$

Derive it in three steps. Write $\boldsymbol\Omega = \operatorname{diag}(\omega_A)$
and $\mathbf A_0 = \operatorname{diag}(\alpha_A)$, so
$\mathbf H = \boldsymbol\Omega^2 + \boldsymbol\Omega\mathbf A_0^{1/2}\mathbf T\mathbf A_0^{1/2}\boldsymbol\Omega$.
First, factor $u^2 + \boldsymbol\Omega^2$ out of $u^2 + \mathbf H$ and use
$\det(\mathbf 1 + \mathbf X\mathbf Y) = \det(\mathbf 1 + \mathbf Y\mathbf X)$
to move the diagonal factors next to each other. They combine into
$\alpha_A\omega_A^2/(\omega_A^2 + u^2) = \alpha_A(iu)$, so

$$ \det(u^2 + \mathbf H) = \det(u^2 + \boldsymbol\Omega^2)\,\det\big(\mathbf 1 + \mathbf A(iu)\mathbf T\big) . $$

Second, take the logarithm and write both determinants through their
eigenvalues: $\ln\det(\mathbf 1 + \mathbf A\mathbf T) = \sum_p\ln(u^2 + \lambda_p) - 3\sum_A\ln(u^2 + \omega_A^2)$.
Third, integrate term by term with
$\int_0^\infty\ln\frac{u^2 + a^2}{u^2 + b^2}\,du = \pi(a - b)$ and divide by
$2\pi$. That gives $\tfrac12\sum_p\sqrt{\lambda_p} - \tfrac32\sum_A\omega_A$,
(@eq:mbd-7).

Now expand the logarithm in powers of $\mathbf T$:
$\ln\det(\mathbf 1 + \mathbf A\mathbf T) = \operatorname{tr}(\mathbf A\mathbf T) - \tfrac12\operatorname{tr}(\mathbf A\mathbf T\mathbf A\mathbf T) + \dots$
The first-order term is zero: $\mathbf T$ has no diagonal blocks. Take two
atoms for the second-order term. The trace visits the $AB$ and $BA$ blocks,
$\mathbf T^{\mathrm{bare}}_{BA} = \mathbf T^{\mathrm{bare}}_{AB}$, and
$\mathbf T^{\mathrm{bare}}$ has eigenvalues $(1, 1, -2)/R^3$, so
$\operatorname{tr}(\mathbf T^{\mathrm{bare}})^2 = 6/R^6$. The term is
$-\tfrac12\cdot2\,\alpha_A(iu)\alpha_B(iu)\cdot6f^2/R^6$. Integrate and
divide by $2\pi$:

$$ E^{(2)} = -\frac{f^2}{R^6}\cdot\frac3\pi\int_0^\infty\alpha_A(iu)\,\alpha_B(iu)\,du = -\frac{f^2C_6^{AB}}{R^6}, $$

with the Casimir–Polder $C_6$ of (@eq:casimir-polder-c6-1). Pairwise dispersion is the leading
term of MBD. The higher orders are the many-body screening that a pair sum
leaves out.

> **Verified equation @eq:disp-mbd-qho-energy** · oscillators, frequency integral, London limit
>
> On a random four-atom geometry with Fermi-damped coupling, (@eq:mbd-7) equals a
> 2000-node quadrature of (@eq:disp-mbd-qho-energy) to $10^{-10}$ relative. For two atoms the
> coupling-squared term of (@eq:mbd-7) is $-f^2\cdot\tfrac32\alpha_A\alpha_B\omega_A\omega_B/(\omega_A + \omega_B)/R^6$
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

For two atoms at fixed volume ratios, the ratio
$E_{\mathrm{MBD}}/(-C_6/R^6)$ tends to 1 as $R$ grows, because $f \to 1$,
the screening of @sec:mbd-range-separated-self-consistent dies out, and the higher orders fall off faster
than $R^{-6}$. In a numpy model of two bare oscillators ($\alpha = 9$ and
4.5, $\omega = 0.6$ and 0.43, $f = 1$) the ratio is 1.0008 at 6 Bohr and
$1 + 6\times10^{-7}$ at 20 Bohr.

The same limit in ferric, from the `589b50832` build: Ar$_2$ with free-atom
volume ratios and $\beta = 0.83$, then MBD@rsSCS for water at PBE/cc-pVDZ,
once inside the SCF and once from the standalone energy with the SCF's
volume ratios.

```python
# Standalone; run from the ferric repo root with OPENBLAS_NUM_THREADS=1.
import ferric

BOHR = 0.529177210903
# Ar2 with free-atom volume ratios: screening and damping die out with R, and the
# energy approaches -C6/R^6 with the TS free-atom C6 of argon (64.3 a.u.).
for r_bohr in (10.0, 15.0, 20.0, 30.0):
    mol = ferric.Molecule.from_xyz_string(f"2\n\nAr 0 0 0\nAr 0 0 {r_bohr * BOHR}\n")
    res = ferric.mbd_rsscs_energy(mol, [1.0, 1.0], beta=0.83)
    pair = -64.3 / r_bohr**6
    print(f"R = {r_bohr:5.1f} Bohr  E_MBD = {res.energy:.6e}  -C6/R^6 = {pair:.6e}  "
          f"ratio = {res.energy / pair:.6f}  C6_rsSCS = {res.c6_rsscs}")

# Water at PBE/cc-pVDZ with MBD@rsSCS, then the standalone energy from the same ratios.
mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
r = ferric.run_dft(mol, bs, functional="PBE", dispersion="mbd")
print("water PBE+MBD: E_disp =", r.e_dispersion, "volume ratios =", r.volume_ratios)
m = ferric.mbd_rsscs_energy(mol, r.volume_ratios, functional="PBE")
print("standalone E_MBD =", m.energy, " alpha_TS =", m.alpha_0_ts, " alpha_rsSCS =", m.alpha_0_rsscs)
```

```text
R =  10.0 Bohr  E_MBD = -6.238680e-05  -C6/R^6 = -6.430000e-05  ratio = 0.970246  C6_rsSCS = [64.30000448298603, 64.30000448298604]
R =  15.0 Bohr  E_MBD = -5.643989e-06  -C6/R^6 = -5.644993e-06  ratio = 0.999822  C6_rsSCS = [64.29999999941695, 64.29999999941695]
R =  20.0 Bohr  E_MBD = -1.004688e-06  -C6/R^6 = -1.004687e-06  ratio = 1.000001  C6_rsSCS = [64.29999999940162, 64.2999999994016]
R =  30.0 Bohr  E_MBD = -8.820303e-08  -C6/R^6 = -8.820302e-08  ratio = 1.000000  C6_rsSCS = [64.2999999994016, 64.2999999994016]
water PBE+MBD: E_disp = -0.00024228853342078693 volume ratios = [0.9631543356796384, 0.5818934328900807, 0.5818934328900766]
standalone E_MBD = -0.00024228853342078693  alpha_TS = [5.201033412670047, 2.6185204480053628, 2.6185204480053446]  alpha_rsSCS = [5.018088320383405, 2.1613451373827854, 2.1613451373827703]
```

The ratio is 0.970 at 10 Bohr and 1.000001 at 20 Bohr. Screening moves the
$C_6$ of each argon atom off the free-atom 64.3 a.u. by only
$4.5\times10^{-6}$ a.u. even at 10 Bohr, so the 3% gap there is the damping
and the higher orders. The standalone energy for water reproduces the SCF's `e_dispersion`
bit for bit when it is given the same volume ratios.

`ferric-rpa/src/dispersion/mbd_rsscs.rs::rsscs_forward` builds
$\mathbf H$ pair by pair. The off-diagonal block and the energy, unchanged:

<!-- excerpt: ferric-rpa/src/dispersion/mbd_rsscs.rs::rsscs_forward -->
```rust
let s = config.beta * (r_vdw_rsscs[a] + r_vdw_rsscs[b]);
let pref = omega_rsscs[a]
    * omega_rsscs[b]
    * (alpha_0_rsscs[a] * alpha_0_rsscs[b]).sqrt()
    * fermi(r, s, config.a);
let t = t_bare(&d, r);
```

<!-- excerpt: ferric-rpa/src/dispersion/mbd_rsscs.rs::rsscs_forward -->
```rust
let energy =
    0.5 * evals.iter().map(|&l| l.sqrt()).sum::<f64>() - 1.5 * omega_rsscs.iter().sum::<f64>();
```

| Symbol | Code (`rsscs_screen`, `rsscs_forward`) |
|---|---|
| $\mathbf C(u_k)$, $\mathbf C^{-1}$ | `c`, `cinv` |
| $1 - f$ | `sr` |
| $\alpha^{\mathrm{rsSCS}}_A(iu_k)$ | `a_dyn[k][a]` |
| $\alpha^{\mathrm{rsSCS}}_A, C^{\mathrm{rsSCS}}_{6,A}, \omega^{\mathrm{rsSCS}}_A, R^{\mathrm{rsSCS}}_A$ | `alpha_0_rsscs`, `c6_rsscs`, `omega_rsscs`, `r_vdw_rsscs` |
| $S_{AB}$, $\omega_A\omega_B\sqrt{\alpha_A\alpha_B}f_{AB}$ | `s`, `pref` |
| $\mathbf H$, $\lambda_p$ | `h`, `evals` |

What if $\mathbf H$ has a non-positive eigenvalue? Then a mode has an
imaginary frequency, the oscillators are unstable (a polarization
catastrophe: atoms too close for this $\beta$), and the energy is undefined.
`rsscs_forward` returns an error. Together with the screening errors of
@sec:mbd-range-separated-self-consistent, none of these returns a substitute number.

## The gradient {#mbd-gradient}

`mbd_rsscs_gradient_from_params` differentiates the forward pass in reverse
(adjoint) mode at the cost of the energy, $O(n_u(3N)^3)$. The forward pass
keeps $\mathbf C(u_k)^{-1}$ for every node and the eigenpairs of
$\mathbf H$ on a tape. The reverse pass starts at the energy. Perturb
$\mathbf H$: $\delta\operatorname{tr}\mathbf H^{1/2} = \tfrac12\operatorname{tr}(\mathbf H^{-1/2}\delta\mathbf H)$,
which holds with degenerate eigenvalues too. With the factor $\tfrac12$ of
(@eq:mbd-7), the first step is

$$ \frac{\partial E_{\mathrm{MBD}}}{\partial\mathbf H} = \tfrac14\,\mathbf H^{-1/2} = \tfrac14\,\mathbf V\operatorname{diag}(\lambda_p^{-1/2})\mathbf V^{\mathsf T}, \label{eq:disp-mbd-adjoint-seed} $$

and the term $-\tfrac32\sum_A\omega_A$ seeds each frequency adjoint with
$-\tfrac32$. The opening of `ferric-rpa/src/dispersion/mbd_rsscs.rs::rsscs_backward`,
unchanged:

<!-- excerpt: ferric-rpa/src/dispersion/mbd_rsscs.rs::rsscs_backward -->
```rust
let mut vs = tape.h_evecs.clone();
for (pidx, mut col) in vs.axis_iter_mut(Axis(1)).enumerate() {
    let s = 0.25 / tape.h_evals[pidx].sqrt();
    col.mapv_inplace(|x| x * s);
}
let g = vs.dot(&tape.h_evecs.t());
```

Each column of $\mathbf V$ is scaled by $\tfrac14\lambda_p^{-1/2}$ and
multiplied back by $\mathbf V^{\mathsf T}$. From there the adjoint runs
back through $\mathbf H$, (@eq:mbd-6), the inverses $\mathbf C(u_k)^{-1}$ on the
tape, the Fermi functions and the dipole tensors, to
$\partial E/\partial\mathbf R_A$ at fixed TS inputs and to
$\partial E/\partial\alpha_A$, $\partial E/\partial C_{6,A}$,
$\partial E/\partial R_A$. Each pair term depends on
$\mathbf R_A - \mathbf R_B$ only, so its derivative goes to $A$ with $+$
and to $B$ with $-$, and the explicit force sums to zero.

Through the screening, the adjoint of $\mathbf C^{-1}$ needs no new
inverse. The contraction (@eq:disp-mbd-rsscs-screening) has the adjoint
$\bar B_{(A,i)(B,j)} = \delta_{ij}\bar\alpha_A/3$, and
$\bar{\mathbf C} = -\mathbf B^{\mathsf T}\bar{\mathbf B}\mathbf B^{\mathsf T}$
with $\mathbf B = \mathbf C^{-1}$ factors through rank 3, one rank per
Cartesian direction. `mbd_rsscs_gradient` then chains to the volume
ratios through (@eq:mbd-1):
$\partial E/\partial r_A = \alpha^{\mathrm{free}}_A\bar\alpha_A + 2r_AC^{\mathrm{free}}_{6,A}\bar C_{6,A} + \tfrac13r_A^{-2/3}R^{\mathrm{free}}_A\bar R_A$,
with $\bar x = \partial E/\partial x$.

### The adjoint steps {#mbd-gradient-adjoint-steps}

`mbd_rsscs_gradient_from_params` runs the forward pass once, keeping
$\mathbf B(u_k) = \mathbf C(u_k)^{-1}$ for every node and the eigenpairs
$(\lambda_p, \mathbf v_p)$ of $\mathbf H$, then one reverse pass,
`rsscs_backward`. In this subsection $\tilde x$ is the screened
$x^{\mathrm{rsSCS}}$ of (@eq:mbd-6), $\tilde f_{AB}$ the Fermi function on
the screened radii, $\tilde S_{AB} = \beta(\tilde R_A + \tilde R_B)$, and
$\mathbf G = \tfrac14\mathbf V\operatorname{diag}(\lambda^{-1/2})\mathbf V^{\mathsf T}$
the seed (@eq:disp-mbd-adjoint-seed). Each step applies the chain rule to
one formula of the forward pass, in reverse order.

1. **Seed.** $\bar{\mathbf H} = \mathbf G$ and $\bar{\tilde\omega}_A = -\tfrac32$.
2. **Through $\mathbf H$** (@eq:mbd-7). The diagonal block
   $\tilde\omega_A^2\mathbf 1$ gives
   $\bar{\tilde\omega}_A \mathrel{+}= 2\tilde\omega_A\sum_i G_{(A,i)(A,i)}$.
   A pair $A > B$ puts $P\,T^{\mathrm{bare}}_{ij}$ at $(A,i),(B,j)$ and its
   transpose, with $P = \tilde\omega_A\tilde\omega_B\sqrt{\tilde\alpha_A\tilde\alpha_B}\tilde f_{AB}$.
   Its adjoints are $\bar v_{ij} = G_{(A,i)(B,j)} + G_{(B,j)(A,i)}$ and
   $\bar P = \sum_{ij}\bar v_{ij}T^{\mathrm{bare}}_{ij}$. Differentiate the
   product $P$:
   $\bar{\tilde\omega}_A \mathrel{+}= \bar PP/\tilde\omega_A$,
   $\bar{\tilde\alpha}_A \mathrel{+}= \tfrac12\bar PP/\tilde\alpha_A$,
   $\bar{\tilde f} = \bar PP/\tilde f$. The Fermi function passes this on to
   both screened radii, $\bar{\tilde R}_A, \bar{\tilde R}_B \mathrel{+}= \beta\bar{\tilde f}\,\partial f/\partial S$,
   and to the positions through
   $\nabla_{\mathbf R}\sum_{ij}P\bar v_{ij}T^{\mathrm{bare}}_{ij} + \bar{\tilde f}\,(\partial f/\partial R)\,\mathbf R/R$.
3. **Through $\tilde\omega$ and $\tilde R$** (@eq:mbd-6). From
   $\tilde\omega_A = 4\tilde C_{6,A}/(3\tilde\alpha_A^2)$:
   $\bar{\tilde C}_{6,A} = \bar{\tilde\omega}_A\cdot4/(3\tilde\alpha_A^2)$ and
   $\bar{\tilde\alpha}_A \mathrel{-}= 2\bar{\tilde\omega}_A\tilde\omega_A/\tilde\alpha_A$.
   From $\tilde R_A = R_A(\tilde\alpha_A/\alpha_A)^{1/3}$:
   $\bar{\tilde\alpha}_A \mathrel{+}= \bar{\tilde R}_A\tilde R_A/(3\tilde\alpha_A)$,
   $\bar R_A \mathrel{+}= \bar{\tilde R}_A\tilde R_A/R_A$,
   $\bar\alpha_A \mathrel{-}= \bar{\tilde R}_A\tilde R_A/(3\alpha_A)$.
4. **Per node $k$** (@eq:disp-mbd-rsscs-screening). The $C_6$ sum of
   (@eq:mbd-6) gives $\bar{\tilde\alpha}_A(iu_k) = \bar{\tilde C}_{6,A}\,(6/\pi)\,w_k\,\tilde\alpha_A(iu_k)$,
   plus $\bar{\tilde\alpha}_A$ at the $u = 0$ node. The contraction
   $\tilde\alpha_A = \tfrac13\sum_{B,i}B_{(A,i)(B,i)}$ has the adjoint
   $\bar B_{(A,i)(B,j)} = \delta_{ij}\bar{\tilde\alpha}_A/3$, and
   $\bar{\mathbf C} = -\mathbf B^{\mathsf T}\bar{\mathbf B}\mathbf B^{\mathsf T}$
   factors through rank 3, one rank per Cartesian direction $i$:
   $$ \bar C_{mn} = -\sum_i X_{mi}Y_{ni}, \qquad X_{mi} = \sum_AB_{(A,i)m}\,\bar{\tilde\alpha}_A/3, \qquad Y_{ni} = \sum_BB_{n(B,i)} . $$
   The diagonal $1/\alpha_A(iu_k)$ gives
   $\bar\alpha_A(iu_k) \mathrel{-}= \sum_i\bar C_{(A,i)(A,i)}/\alpha_A(iu_k)^2$.
   A pair block $v = (1 - f)\mathbf T^{\mathrm{GG}}(\mathbf R, \sigma_{AB})$
   gives $\bar f = -\sum\bar v\cdot\mathbf T^{\mathrm{GG}}$ and, through
   `t_gg_vjp` with $M = (1 - f)\bar v$, the position term and
   $\bar\sigma_{AB}$. The widths (@eq:mbd-2) and the TS form (@eq:mbd-1) then
   give $\bar\sigma_A \mathrel{+}= \bar\sigma_{AB}\sigma_A/\sigma_{AB}$,
   $\bar\alpha_A(iu_k) \mathrel{+}= \bar\sigma_A\sigma_A/(3\alpha_A(iu_k))$,
   $\bar\alpha_A \mathrel{+}= \bar\alpha_A(iu_k)\,\alpha_A(iu_k)/\alpha_A$ and
   $\bar\omega_A \mathrel{+}= \bar\alpha_A(iu_k)\,2\alpha_A(iu_k)^2u_k^2/(\alpha_A\omega_A^3)$.
   The radius partial of $f$ goes to $\bar R_A, \bar R_B \mathrel{+}= \beta\bar f\,\partial f/\partial S$.
5. **Through $\omega$.** $\bar C_{6,A} \mathrel{+}= \bar\omega_A\cdot4/(3\alpha_A^2)$,
   $\bar\alpha_A \mathrel{-}= 2\bar\omega_A\omega_A/\alpha_A$.

The Fermi partials (`ferric-rpa/src/dispersion/mbd_rsscs.rs::fermi_partials`)
follow from $f = 1/(1 + e^{-s})$, $s = a(R/S - 1)$:

$$ \frac{\partial f}{\partial R} = \frac{f'a}{S}, \qquad \frac{\partial f}{\partial S} = -\frac{f'aR}{S^2}, \qquad f' = \frac{e^{-s}}{(1 + e^{-s})^2} , $$

and $f' = 0$ when $e^{-s}$ overflows ($R \ll S$, where $f = 0$ in double
precision). The vector-Jacobian products of the dipole tensors
(`t_bare_vjp`, `t_gg_vjp`) take an adjoint block $M$, symmetrized first,
$q = \mathbf R^{\mathsf T}M\mathbf R$ and $L_b = \sum_{ij}M_{ij}T^{\mathrm{bare}}_{ij}$:

$$ \frac{\partial L_b}{\partial\mathbf R} = -\frac{3\operatorname{tr}M\,\mathbf R}{R^5} - \frac{6M\mathbf R}{R^5} + \frac{15q\,\mathbf R}{R^7} . $$

For the Gaussian tensor write (@eq:disp-mbd-dipole-tensor) as
$\mathbf T^{\mathrm{GG}} = g_1\mathbf T^{\mathrm{bare}} + g_2\mathbf R\mathbf R^{\mathsf T}$,
$g_1 = \operatorname{erf}z - \theta$, $g_2 = 2z^2\theta/R^5$, $z = R/\sigma$.
Differentiating $g_1$ and $g_2$ with $\partial\theta/\partial z = 2(1 - 2z^2)e^{-z^2}/\sqrt\pi$:

$$ \frac{\partial g_1}{\partial R} = \frac{2z^2\theta}{R}, \quad \frac{\partial g_1}{\partial\sigma} = -\frac{2z^2\theta}{\sigma}, \quad \frac{\partial g_2}{\partial R} = -\frac{4z^2\theta(1 + z^2)}{R^6}, \quad \frac{\partial g_2}{\partial\sigma} = -\frac{2z^2\theta(3 - 2z^2)}{\sigma R^5}, $$

and with $L = \sum_{ij}M_{ij}T^{\mathrm{GG}}_{ij} = g_1L_b + g_2q$,

$$ \frac{\partial L}{\partial\mathbf R} = g_1\frac{\partial L_b}{\partial\mathbf R} + 2g_2M\mathbf R + \Big(L_b\frac{\partial g_1}{\partial R} + q\frac{\partial g_2}{\partial R}\Big)\frac{\mathbf R}{R}, \qquad \frac{\partial L}{\partial\sigma} = L_b\frac{\partial g_1}{\partial\sigma} + q\frac{\partial g_2}{\partial\sigma} . $$

The second box below checks these against sympy and the whole pass against
finite differences.

> **Verified equation @eq:disp-mbd-adjoint-seed** · the adjoint seed
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

> **Verified equation @eq:disp-mbd-adjoint-seed** · the whole backward pass
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

### On an SCF density {#mbd-gradient-scf-density}

On an SCF density the volume ratios move with the nuclei too. The volumes
$v_A = \sum_g\Delta V\,w_A(\mathbf r_g)\rho(\mathbf r_g)|\mathbf r_g - \mathbf R_A|^3$,
with Hirshfeld weights $w_A = \rho^0_A/(\sum_C\rho^0_C + 10^{-12})$, are
integrated on a cubic lattice anchored to the centroid of the nuclei, and
the free-atom volumes on the same lattice, cached per element. Why a lattice
and not the Becke grid of the TS $C_6$ path? A lattice that moves rigidly
makes the lattice term below a single translation identity. With
$c_A = (\partial E/\partial r_A)/v_A^{\mathrm{free}}$,
`mbd_rsscs_for_scf` assembles

$$ \frac{dE}{d\mathbf R_B} = \frac{\partial E}{\partial\mathbf R_B}\Big|_r + \sum_Ac_A\frac{\partial v_A}{\partial\mathbf R_B}\Big|_{D,\ \text{lattice}} + \mathcal O_B - \frac1N\sum_C\sum_Ac_A\frac{\partial v_A}{\partial\mathbf R_C}\Big|_{D,\ \text{lattice}} + \sum_{ai}Z_{ai}\frac{\partial F_{ai}}{\partial\mathbf R_B} , $$

term by term:

| Term | What it carries | Function |
|---|---|---|
| explicit | $\partial E/\partial\mathbf R_B$ at fixed volume ratios | `mbd_rsscs_gradient` |
| Hirshfeld volumes | $\sum_Ac_A\,\partial v_A/\partial\mathbf R_B$ at fixed density and lattice, $c_A = (\partial E/\partial r_A)/v_A^{\mathrm{free}}$ | `ferric-scf/src/properties/hirshfeld_grid.rs::hirshfeld_volume_gradient` |
| orthonormality | $-\tfrac12\operatorname{tr}[\mathbf V\mathbf D\mathbf S^x\mathbf D]$ (per spin for UKS; a closed–open term for ROKS), $\mathbf V = \partial(\sum_Ac_Av_A)/\partial\mathbf D$ | `ferric-rpa/src/dispersion/mbd_scf.rs::orthonormality_term` |
| lattice | the volume lattice (`ferric-scf/src/properties/hirshfeld_grid.rs::mbd_volume_grid`, centred on the atom centroid) moves by $1/N$ of each displacement; translation invariance turns this into minus $1/N$ of the summed fixed-lattice term | `mbd_rsscs_for_scf` |
| relaxation | $\sum_{ai}Z_{ai}\,\partial F_{ai}/\partial\mathbf R_B$ with the Kohn–Sham Z-vector driven by $\mathbf V$ | `ferric_scf::zvector_ks` |

The free-atom volumes use the same lattice
(`FreeAtomVolumeQuadrature::Lattice`), so a lattice error in the free-atom
denominator cancels against the same error in the molecular volumes. The
density derivative is
`ferric-scf/src/properties/hirshfeld_grid.rs::hirshfeld_volume_density_derivative`:

$$ \mathbf V = \frac{\partial\big(\sum_Ac_Av_A\big)}{\partial\mathbf D} = \sum_g\Delta V\,q(\mathbf r_g)\,\chi_\mu(\mathbf r_g)\chi_\nu(\mathbf r_g), \qquad q(\mathbf r) = \sum_Ac_Aw_A(\mathbf r)\,|\mathbf r - \mathbf R_A|^3 . $$

Each term answers one way the volumes can move. The AOs and proatoms move
with their atoms (volumes). Differentiating $v_A$ at a fixed lattice point
gives three terms per point, with $\mathbf d_B = \mathbf r - \mathbf R_B$,
$r_B = |\mathbf d_B|$ and $\mathbf g_B = \partial\rho^0_B/\partial\mathbf R_B$:
the density term $-2\Delta V\,q\sum_{\mu\in B}\nabla\chi_\mu(\mathbf D\chi)_\mu$,
the weight term $\Delta V\rho\,(c_Br_B^3 - q)\,\mathbf g_B/(\sum_C\rho^0_C + 10^{-12})$,
and the radial term $-3\Delta V\rho\,c_Bw_Br_B\,\mathbf d_B$. The occupied orbitals follow their basis
functions and are Löwdin re-orthonormalized, $\delta C_o = -\tfrac12C_oS^x_{oo}$
(orthonormality). For a closed shell, $\mathbf D = 2C_oC_o^{\mathsf T}$,
$\delta\mathbf D = -\tfrac12\mathbf D\mathbf S^x\mathbf D$ and
$\mathcal O = -\tfrac12\operatorname{tr}[\mathbf V\mathbf D\mathbf S^x\mathbf D]$.
For UKS each spin separately,
$\mathcal O = -\sum_\sigma\operatorname{tr}[\mathbf V\mathbf D_\sigma\mathbf S^x\mathbf D_\sigma]$.
For ROKS, with $\mathbf D_\alpha = \mathbf P_c + \mathbf P_o$ and
$\mathbf D_\beta = \mathbf P_c$, the closed and open orbitals are
orthonormalized as one set, $\mathcal O = -\operatorname{tr}[\mathbf S^x\mathbf W_Q]$
with $\mathbf W_Q$ of the box below. The code first checks $\mathbf D\mathbf S\mathbf D = 2\mathbf D$
for a closed shell, $\mathbf D_\sigma\mathbf S\mathbf D_\sigma = \mathbf D_\sigma$
per spin, and $\mathbf D_\alpha\mathbf S\mathbf D_\beta = \mathbf D_\beta$ for
ROKS. Moving all atoms and the lattice together leaves $v_A$ unchanged at
fixed density, so the lattice shift contributes minus $1/N$ of the summed
fixed-lattice term to every atom (lattice). The density itself relaxes
(relaxation): `ferric-scf/src/zvector_ks.rs::{relaxation_gradient_closed,relaxation_gradient_unrestricted,relaxation_gradient_roks}`
solve the Kohn–Sham Z-vector with $\mathbf V$ as the right-hand side. The
first four terms, without the relaxation, are returned as the result field
`gradient_unrelaxed`. MBD depends on the total density, so both spins see the same
$\mathbf V$. The Z-vector derivation is section @sec:ks-zvector. A reference the Z-vector does
not support is an error; no unrelaxed gradient is returned as the exact
one.

The total is checked against central finite differences of the full
SCF + MBD energy (@sec:mbd-options-anchors). The orthonormality and lattice terms are also
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
and the full-pipeline difference tests: central differences of the full
SCF + MBD energy in `crates/ferric-rpa/tests/mbd_scf_gradient.rs` (RKS,
$\le 3.5\times10^{-9}$ Ha/Bohr at 6-31G),
`crates/ferric-rpa/tests/mbd_scf_gradient_uks.rs` (UKS) and
`crates/ferric-rpa/tests/mbd_scf_gradient_roks.rs` (ROKS).

## Full-range screening for $C_6$ {#mbd-full-range-screening}

`[rpa] c6_source = "mbd"` screens the TS poles of section @sec:casimir-polder-c6 with the
full-range tensor and no Fermi function, at each node of the RPA grid:

$$ \mathbf C(u) = \operatorname{diag}\big(1/\bar\alpha_A(iu)\big) + \mathbf T^{\mathrm{GG}}, \qquad \alpha^{\mathrm{SCS}}_A(iu) = \sum_B\big[\mathbf C(u)^{-1}\big]_{AB}, \label{eq:mbd-10} $$

This is (@eq:disp-mbd-rsscs-screening) with $f = 0$, the uniform-field argument of @sec:mbd-range-separated-self-consistent applied
to the whole coupling. $\bar\alpha_A(iu)$ is the isotropic TS value, floored
at $10^{-10}$, and the widths come from it. Each atom is one isotropic
oscillator, as Ambrosetti et al. prescribe: the static per-atom tensor's
shape is not inverted, because inverting a near-singular anisotropic
tensor distorts the screening. The full $3\times3$ block sum is kept as the
per-atom tensor. The molecular tensor is the sum over atoms, and
`casimir_polder_c6` takes it from there. A singular $\mathbf C(u)$ is an
error, never the identity in its place.

`coupled_qho_energy_plain_gg` evaluates (@eq:mbd-7) with this full-range Gaussian
coupling, the unscreened $\alpha$ and $\omega$, and no Fermi function, as a
check of the oscillator algebra against libMBD `variant = "plain"`. It is
not a dispersion energy (`site/src/reference/validation.md`, row "TS C6 and
MBD@TS"). Without the range separation the coupling is far too strong: the
function's documentation records 39 to 226 times MBD@rsSCS on molecules
(water: about $-34$ against $-0.15$ kcal/mol).

## The algorithm {#mbd-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | Hirshfeld volumes of the SCF density and of each free atom on one lattice (free-atom SCFs cached per element) | grid × basis | `ferric-rpa/src/dispersion/mbd_scf.rs::MbdFreeAtomCache`, `ferric-scf/src/properties/hirshfeld_grid.rs::atomic_effective_volumes_hirshfeld_on_grid` |
| 2 | TS inputs (@eq:mbd-1) | $N$ | `ferric-rpa/src/dispersion/mbd_rsscs.rs::ts_params_from_volume_ratios` |
| 3 | per node: $\mathbf C(u_k)$, inverse, (@eq:disp-mbd-rsscs-screening) | $16\,(3N)^3$ | `rsscs_screen` |
| 4 | (@eq:mbd-6), $\mathbf H$, eigensolve, (@eq:mbd-7) | $(3N)^3$ | `rsscs_forward` |
| 5 | gradient: adjoint (@eq:disp-mbd-adjoint-seed) back through steps 4 and 3; then the SCF terms | $O(16\,(3N)^3)$ plus one Z-vector | `rsscs_backward`, `mbd_rsscs_for_scf` |

Why diagonalize $\mathbf H$ instead of integrating (@eq:disp-mbd-qho-energy)? The
eigenvalues give the energy exactly in one $(3N)^3$ step, with no
frequency grid in the energy; the 16 nodes are needed only for the
screening.

## Options and anchors {#mbd-options-anchors}

| Option | Values | Effect |
|---|---|---|
| `[dft] dispersion` | `"mbd"` (the running functional's $\beta$), `"mbd(<name>)"` | $\beta$ = 0.83 (PBE), 0.85 (PBE0, HSE06), from `ferric-rpa/src/dispersion/mbd_rsscs.rs::mbd_rsscs_beta_for_functional`; any other functional is refused and needs an explicit $\beta$ |
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
- The Hirshfeld volume gradient and the Z-vector relaxation of @sec:mbd-gradient rest
  on finite-difference tests, not on a proof script.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-rpa/src/dispersion/mbd_rsscs.rs` (module doc and loops) | (@eq:mbd-1)–(@eq:disp-mbd-adjoint-seed) |
| `crates/ferric-rpa/src/dispersion/mbd_scf.rs` (module doc) | the SCF gradient terms |
| `crates/ferric-rpa/src/dispersion/mbd.rs` | (@eq:mbd-10) |
| `scripts/mbd_coupled_dipole_sympy.py` | verified boxes |
| `site/src/reference/validation.md` | anchors |

Changes from the sources: the code and its docs write the imaginary frequency
as $u$ (MBD) and $\omega$ (`mbd.rs`); this section writes $u$ throughout,
and marks the screened quantities with the superscript rsSCS. The doc comment of `dipole_coupling_tensor` states
the tensor with the opposite overall sign to the loop; (@eq:disp-mbd-dipole-tensor) follows the
loop, which agrees with `t_gg`.
