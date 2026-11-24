# Harmonic frequencies {#frequencies}

<!-- status: ready; registry: harmonic-frequencies, rrho-thermo, ir-intensities -->

Near a minimum the energy surface is a bowl, and the nuclei sit at its
bottom. Push them a little and they oscillate. The curvature of the bowl is
the Hessian, and it sets how stiff each motion is. Stiffness alone does not
fix the frequency: a hydrogen on the same spring as a carbon vibrates
faster, because it is lighter. So the vibrational frequencies are the
curvature measured per unit mass.

The plan: the harmonic eigenproblem and its units, the code, then the two
quantities built on the modes, thermochemistry and IR intensities.

Given the Hessian and the masses, nothing else enters. Whether the Hessian
came from section @sec:hf-hessian, from section @sec:ks-hessian or from differences of analytic
gradients does not matter to what follows: one linear-algebra routine turns
it into wavenumbers. The one delicate step is
removing the six motions that are not vibrations, three translations and
three rotations. That has to happen in mass-weighted coordinates, and it
must not mistake a small imaginary mode at a saddle point for one of them.

| Symbol | Meaning | Code name |
|---|---|---|
| $H$ | Cartesian Hessian, Hartree/Bohr², index $3A+\alpha$ | `cartesian_hessian` |
| $M$ | diagonal mass matrix, masses in electron masses | `mass_au` |
| $\tilde H = M^{-1/2}HM^{-1/2}$ | mass-weighted Hessian | `mw` |
| $t_k$ | orthonormal translation and rotation vectors, mass weighted | `tr_basis` |
| $P = 1-\sum_kt_kt_k^{\mathsf T}$ | projector off the rigid motions | `proj` |
| $\omega_k$, $\tilde\nu_k$ | angular frequency in atomic units, wavenumber in cm⁻¹ | `evals`, `to_cm` |

| Entry point | Location |
|---|---|
| Rust, Hessian to frequencies | `ferric-scf/src/frequencies.rs::frequencies_from_cartesian_hessian` |
| Rust, drivers | `ferric-scf/src/frequencies.rs::{harmonic_frequencies,harmonic_frequencies_analytic,harmonic_frequencies_from_gradient}` |
| Rust, analytic Hessian at a converged SCF | `ferric-scf/src/hessian.rs::{analytic_frequencies,analytic_frequencies_uhf,analytic_frequencies_rohf,analytic_frequencies_ks_moving_grid}`, `ferric-scf/src/hessian/uks.rs::analytic_frequencies_uks_moving_grid`; each builds the Hessian parts and hands them to the same projection and diagonalization as the finite-difference route |
| CLI | `task = "frequencies"`, section `[frequencies]` |
| Python | `run_frequencies` |

## From Hessian to wavenumbers {#frequencies-hessian-wavenumbers}

### The harmonic eigenproblem

$H$ is in Hartree/Bohr². It is the analytic Hessian of section @sec:hf-hessian or @sec:ks-hessian,
or the central difference of the analytic gradient (default step
$\delta = 5\times10^{-3}$ Bohr), symmetrized as $(H + H^{\mathsf T})/2$.

Expand the energy to second order about the minimum. The gradient is zero
there, so with $\Delta x$ the displacement,

$$ V \approx \tfrac12\,\Delta x^{\mathsf T} H\,\Delta x, \qquad T = \tfrac12\,\dot x^{\mathsf T} M\,\dot x . $$

The masses make $T$ awkward: each coordinate has its own. Substitute the
mass-weighted coordinate $q = M^{1/2}\Delta x$. Then $T = \tfrac12\dot
q^{\mathsf T}\dot q$, every coordinate has unit mass, and the potential
becomes $V = \tfrac12 q^{\mathsf T}\tilde Hq$ with

$$ \tilde H = M^{-1/2} H M^{-1/2} . $$

The equations of motion are $\ddot q = -\tilde Hq$. Expand $q$ in the
orthonormal eigenvectors of the symmetric matrix $\tilde H$. The equations
decouple, one per eigenvector, each a harmonic oscillator:

$$ \tilde H u_k = \omega_k^2 u_k . $$

The eigenvalue is the squared angular frequency of mode $k$.
`ferric-scf/src/frequencies.rs::frequencies_from_cartesian_hessian` converts
the masses from u to electron masses before the weighting, with
`ferric-md/src/units.rs::AMU_TO_ELECTRON_MASS`
$= 1/5.48579909065\times10^{-4}$ (CODATA 2018). Then everything is in atomic
units and $\omega_k$ comes out in $E_h/\hbar$.

**Units.** A wavenumber is $\tilde\nu = \omega/(2\pi c)$ for $\omega$ in
rad/s. Convert $\omega$ from $E_h/\hbar$ to rad/s and divide:
$\tilde\nu = \omega E_h/(2\pi\hbar c) = \omega\,E_h/(hc)$. The constant
$E_h/(hc)$ in cm⁻¹ is `AU_FREQ_TO_CM` $= 219474.6313632$ cm⁻¹.

**Check the diatomic.** Take two atoms on a spring of force constant $k$,
along one axis. The Cartesian Hessian is
$k\left[\begin{smallmatrix}1&-1\\-1&1\end{smallmatrix}\right]$. Mass weighting
gives
$\left[\begin{smallmatrix}k/m_1&-k/\sqrt{m_1m_2}\\-k/\sqrt{m_1m_2}&k/m_2\end{smallmatrix}\right]$.
Its determinant is zero, so one eigenvalue is 0: the two atoms moving
together. The other equals the trace, $k(1/m_1 + 1/m_2) = k/\mu$ with $\mu$
the reduced mass. So $\omega = \sqrt{k/\mu}$, the textbook result.

> **Verified equation {#eq:freq-units-diatomic}** · units and the diatomic limit
>
> The two conversion constants equal their CODATA 2018 forms
> ($\texttt{AU\_FREQ\_TO\_CM} = E_h/(hc)$ and the u-to-electron-mass ratio) to
> $10^{-10}$ relative; the mass-weighted diatomic spring has eigenvalues
> $\{0, k/\mu\}$, in symbols; and the pipeline's diatomic wavenumber equals
> the SI route $(2\pi c)^{-1}\sqrt{k/\mu}$ to $10^{-9}$ relative.
>
> | | |
> |---|---|
> | Proof | `scripts/hf_frequencies_sympy.py`, claims C1, C2, S1 and D1 |
> | Negative control | the runner drops the factor $2\pi$ from the wavenumber constant and replaces $M^{-1/2}$ by $M^{-1}$ (claim D1 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/frequencies.rs::frequencies_from_cartesian_hessian` |
> | Code test | `anchor_postprocessing_matches_pyscf` in `crates/ferric-scf/tests/validation_frequencies.rs` |
>
> <!-- verified: id=freq.units-diatomic; script=scripts/hf_frequencies_sympy.py; claim=D1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/frequencies.rs::frequencies_from_cartesian_hessian; rust_test=ferric-scf/tests/validation_frequencies.rs::anchor_postprocessing_matches_pyscf; registry=harmonic-frequencies -->

### Removing translations and rotations

In three dimensions a molecule of $N$ atoms has $3N$ coordinates. Moving it
as a whole, or rotating it as a rigid body, does not change the energy, so
those directions have zero curvature at a minimum. A nonlinear molecule has
three translations and three rotations, six zero frequencies. A linear
molecule has five: a rotation about the molecular axis moves no atom.

Numerically these zeros are not clean. A finite-difference Hessian carries
noise, and away from a stationary point the rotations carry real curvature
through the gradient term. So the code removes them explicitly.

`ferric-scf/src/frequencies.rs::translation_rotation_basis` builds them in
mass-weighted coordinates. A translation along axis $k$ displaces every atom
by the same unit vector $e_k$, which in $q = \sqrt m\,\Delta x$ is

$$ t_k[3A + i] = \sqrt{m_A}\,\delta_{ik} . $$

An infinitesimal rotation about axis $k$ through the centre of mass moves
atom $A$ by $e_k \times (R_A - R_{\rm com})$, so

$$ r_k[3A + \cdot\,] = \sqrt{m_A}\,\big(e_k \times (R_A - R_{\rm com})\big) . $$

The six vectors are orthonormalized by modified Gram–Schmidt. A vector that
vanishes is dropped; that is the axial rotation of a linear molecule. The
rank is 6, or 5 if linear, and it is cross-checked against
`ferric-scf/src/frequencies.rs::detect_linear`.

Does the centre of rotation matter? No. Replace $R_{\rm com}$ by any point
$c$. Then $e_k\times(R_A - c) = e_k\times(R_A - R_{\rm com}) + e_k\times(R_{\rm com} - c)$,
and the second term is the same vector for every atom, a translation. The
span of the six vectors does not change.

The projector is $P = 1 - \sum_k t_kt_k^{\mathsf T}$ over the orthonormal set,
and the projected Hessian is $P\tilde HP$. The code applies it as one rank-one
sweep per vector $t$:

$$ \tilde H \leftarrow \tilde H - t(t^{\mathsf T}\tilde H) - (\tilde Ht)t^{\mathsf T} + t\,t^{\mathsf T}(t^{\mathsf T}\tilde Ht) . $$

One sweep is $(1 - tt^{\mathsf T})\tilde H(1 - tt^{\mathsf T})$ multiplied
out. The $t_k$ are orthonormal, so the product of the single-vector
projectors is $P$, and the sweeps together give $P\tilde HP$.

What does the projection do to the vibrations? At a minimum it leaves them
alone: the vibrations are already orthogonal to the rigid motions. Away
from a stationary point the rotations carry curvature (the gradient term),
and the projection sets those eigenvalues to zero. The remaining eigenvalues
are then those of the projected Hessian, not of the full one, because the
rotations it removes were coupled to the vibrations.

> **Verified equation {#eq:freq-projector}** · the projector
>
> For a nonlinear molecule the translation and rotation basis has rank 6, is
> orthonormal, and $P$ is idempotent and annihilates every rigid mode, which
> preserve all interatomic distances (claim P1). A linear molecule has rank 5
> (claim P2). At a minimum of a spring model the projection leaves the
> vibrational eigenvalues unchanged (claim P3). Away from the minimum the
> rotations carry curvature, and the projection zeroes exactly six modes
> (claim P4).
>
> | | |
> |---|---|
> | Proof | `scripts/hf_frequencies_sympy.py`, claims P1 to P4 |
> | Negative control | the runner flips the sign of a rotation-vector component (claim P1 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/frequencies.rs::{translation_rotation_basis,detect_linear}` |
> | Code test | `anchor_postprocessing_matches_pyscf` in `crates/ferric-scf/tests/validation_frequencies.rs` |
>
> <!-- verified: id=freq.projector; script=scripts/hf_frequencies_sympy.py; claim=P1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/frequencies.rs::{translation_rotation_basis,detect_linear}; rust_test=ferric-scf/tests/validation_frequencies.rs::anchor_postprocessing_matches_pyscf; registry=harmonic-frequencies -->

**A negative eigenvalue is an imaginary frequency.** Its $\omega$ is
$i\sqrt{\lvert\lambda\rvert}$, and the code reports it as a negative
wavenumber, $-\sqrt{\lvert\lambda\rvert}\,\mathtt{AU\_FREQ\_TO\_CM}$.

Which eigenvectors are the rigid motions after projection? The code decides
by overlap with the translation and rotation subspace: above $0.5$ means
rigid. It does not take the six eigenvalues closest to zero. At a
transition state a genuine imaginary mode can be smaller in magnitude than
the numerical zeros, and a magnitude rule would discard it.

Normal modes are returned as Cartesian displacements $x = M^{-1/2}u$, not
renormalized.

The script `scripts/hf_frequencies_sympy.py` checks both constants against
CODATA 2018, the diatomic wavenumber against the SI route
$(2\pi c)^{-1}\sqrt{k/\mu}$ (relative $<10^{-9}$), the projector (rank,
orthonormality, idempotence, annihilation of the rigid modes,
distance-preserving rotations, rank 5 for a linear triatomic), and that the
projection leaves the vibrations of a pair-spring model unchanged at its
minimum and zeroes six modes away from it.

The rank rule on two molecules at RHF/STO-3G, from the `3f612e0b3` build.
Water is first optimized at the same level, because the thermochemistry
needs a stationary point (the optimizer's step table is left out of the
output). Water keeps six projected rigid modes, all within $10^{-4}$ cm⁻¹ of
zero; linear CO2 keeps five and has $3N - 5 = 4$ vibrations. The last three lines
use the thermochemistry and IR intensities of section @sec:frequencies-thermochemistry-ir-intensities.

```python
# Water (nonlinear, six rigid modes) and CO2 (linear, five) at RHF/STO-3G,
# then RRHO thermochemistry and IR intensities of water at its RHF/STO-3G minimum.
import ferric

start = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
water = ferric.run_optimize(start, "sto-3g", e_conv=1e-11).mol()  # RRHO needs a stationary point
r = ferric.run_frequencies(water, "sto-3g", hessian="analytic")
print("water linear:", r.is_linear, " hessian:", r.hessian_source)
print("water rigid-mode cm-1 (expect 6 near zero):", [f"{f:.2e}" for f in r.trans_rot_frequencies])
print("water vibrations cm-1:", [f"{f:.2f}" for f in r.frequencies])

co2 = ferric.Molecule.from_arrays(["O", "C", "O"], [(0, 0, -1.16), (0, 0, 0), (0, 0, 1.16)])
c = ferric.run_frequencies(co2, "sto-3g", hessian="analytic")
print("co2 linear:", c.is_linear, " rigid modes:", len(c.trans_rot_frequencies),
      " vibrations:", len(c.frequencies))

t = r.thermo(temperature=298.15, symmetry_number=2)
print("zpe (Ha):", t["zpe"], " S_vib (Ha/K):", t["vibrational"]["entropy"])
print("G total (Ha):", t["total"]["gibbs"])
print("IR intensities (km/mol):", [f"{a:.2f}" for a in r.ir_intensities()])
```

```text
water linear: False  hessian: analytic
water rigid-mode cm-1 (expect 6 near zero): ['-5.20e-05', '-1.12e-05', '-1.21e-08', '4.80e-12', '1.84e-05', '5.18e-05']
water vibrations cm-1: ['2169.85', '4139.63', '4390.67']
co2 linear: True  rigid modes: 5  vibrations: 4
zpe (Ha): 0.024376749299640968  S_vib (Ha/K): 1.0299306321016321e-09
G total (Ha): -74.95926274869072
IR intensities (km/mol): ['7.24', '44.28', '29.97']
```

## The algorithm {#frequencies-algorithm}

| Step | Quantity | Function |
|---|---|---|
| 1 | $H$: analytic (sections @sec:hf-hessian and @sec:ks-hessian, boxes @eq:scf-hess-assembled and @eq:ks-hess-moving-grid) or central differences of the analytic gradient, $H_{ab} = [g_a(x_b+\delta)-g_a(x_b-\delta)]/2\delta$, $\delta = 5\times10^{-3}$ Bohr, $6N$ gradients (the undisplaced gradient is not needed; the saddle search of section @sec:geometry-search evaluates it as well, hence its $6N+1$) | `ferric-scf/src/frequencies.rs::fd_hessian_frequencies` |
| 2 | $H\leftarrow\tfrac12(H+H^{\mathsf T})$; the largest asymmetry is reported as a noise probe, `FrequencyResult::asymmetry` | same |
| 3 | $\tilde H_{ij} = H_{ij}/\sqrt{m_im_j}$ | `frequencies_from_cartesian_hessian` |
| 4 | $P\tilde HP$ by rank-one sweeps, one per rigid-body vector | same |
| 5 | `eigh_dc`, classify rigid modes by overlap, convert $\omega$ to cm⁻¹, back-transform modes $x = M^{-1/2}u$ | same |

Why difference the gradient and not the energy? Differencing a gradient once
leaves $O(\delta^2)$ truncation plus $O(\epsilon_g/\delta)$ noise, where
$\epsilon_g$ is the gradient's error from integral screening, the grid and
SCF convergence. Differencing energies twice divides the energy's error
$\epsilon_E$ by $\delta^2$: $O(\epsilon_E/\delta^2)$. For integral and grid
noise $\epsilon_E$ and $\epsilon_g$ are of the same order, so the gradient
route wins by a factor $1/\delta$. The SCF-convergence part of $\epsilon_E$
is second order in the density error and would favour the energy route, but
integral and grid noise set the floor. So the finite-difference route needs
only an analytic gradient, and there is no fallback that differences
energies twice: a method without an analytic gradient is an error naming
the method.

Steps 3 to 5 are the kernel. The mass weighting is one line:

<!-- excerpt: ferric-scf/src/frequencies.rs::frequencies_from_cartesian_hessian -->
```rust
mw[(i, j)] = cartesian_hessian[(i, j)] * inv_sqrt_m[i] * inv_sqrt_m[j];
```

The projection is the loop of rank-one sweeps:

<!-- excerpt: ferric-scf/src/frequencies.rs::frequencies_from_cartesian_hessian -->
```rust
let mut proj = mw.clone();
for t in &tr_basis {
    // H <- H - t (t^T H) - (H t) t^T + t (t^T H t) t^T
    let ht: Array1<f64> = proj.dot(t);
    let tht: f64 = t.dot(&ht);
    for i in 0..n_coord {
        for j in 0..n_coord {
            proj[(i, j)] -= t[i] * ht[j] + ht[i] * t[j] - t[i] * t[j] * tht;
        }
    }
}
```

| Code | Equation |
|---|---|
| `inv_sqrt_m[i]` | $m_i^{-1/2}$ for Cartesian coordinate $i$, atom $i/3$ |
| `mw` | $\tilde H$ |
| `t` | one orthonormal rigid-motion vector $t_k$ |
| `ht` | $\tilde H\,t_k$, with $\tilde H$ already swept by the earlier vectors |
| `tht` | $t_k^{\mathsf T}\tilde Ht_k$ |
| the update | $\tilde H\leftarrow\tilde H - t(t^{\mathsf T}\tilde H) - (\tilde Ht)t^{\mathsf T} + t\,(t^{\mathsf T}\tilde Ht)\,t^{\mathsf T}$ |

`ht` is computed from `proj`, the matrix already projected by the earlier
vectors. That is the product of single-vector projectors from @sec:frequencies-hessian-wavenumbers, so the
loop needs no stored $P$. Each sweep costs $O(n^2)$ for $n = 3N$, against
$O(n^3)$ for forming $P$ and multiplying. The matrix is re-symmetrized
afterwards to remove accumulated round-off.

The rigid-mode test after `eigh_dc` sums the squared overlaps of each
eigenvector with the basis:

<!-- excerpt: ferric-scf/src/frequencies.rs::frequencies_from_cartesian_hessian -->
```rust
for k in 0..n_coord {
    let col = evecs.column(k);
    let overlap: f64 = tr_basis
        .iter()
        .map(|t| {
            let d: f64 = t.iter().zip(col.iter()).map(|(a, b)| a * b).sum();
            d * d
        })
        .sum();
```

A vector inside the rigid subspace has overlap 1, a vibration has overlap
near 0, and the code sets `is_tr[k] = overlap > 0.5`. If the count of rigid
vectors found differs from the rank of the basis, the projection is called
inconsistent and the run is an error.

## Thermochemistry and IR intensities {#frequencies-thermochemistry-ir-intensities}

A frequency run ends with two quantities built on the modes: the ideal-gas
thermochemistry of the rigid rotor and harmonic oscillator, and the IR
intensity of each mode.

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/thermo.rs::rrho_thermochemistry`, `ferric-scf/src/infrared.rs::{ir_spectrum,dipole_derivatives_field,ir_intensities}` |
| CLI | `task = "frequencies"`, keys `temperature`, `pressure`, `symmetry_number`, `ir_intensities` |
| Python | `FrequencyResult.thermo`, `FrequencyResult.ir_intensities` |

### RRHO thermochemistry

A frequency calculation gives one molecule at 0 K, frozen at the bottom of
its well. A measured enthalpy or free energy is for a mole of gas at a
temperature, where molecules fly, tumble and vibrate in excited levels.
The rigid-rotor, harmonic-oscillator (RRHO) model adds those motions
independently, each from its partition function.
`ferric-scf/src/thermo.rs::rrho_thermochemistry` (also
`FrequencyResult::thermochemistry`, CLI `task = "frequencies"`, Python
`FrequencyResult.thermo`) evaluates it at temperature $T$ and pressure $P$.
The proof is `scripts/rrho_thermo_sympy.py`, and the conventions are those
of PySCF `pyscf.hessian.thermo`.

Everything is per molecule, with $k_{\rm B}$ the Boltzmann constant. Each energy follows from its
partition function $q$ as $E = k_{\rm B}T^2\,\partial\ln q/\partial T$, and each
entropy as $S = k_{\rm B}\ln q + E/T$. The indistinguishability term $k_{\rm B}(1 - \ln N)$
(Stirling) is carried by the translations.

**Translation.** With $q_V = (2\pi M k_{\rm B}T/h^2)^{3/2} V$ and the ideal-gas
volume per molecule $V/N = k_{\rm B}T/P$,

$$ q_{\rm tr} = \left(\frac{2\pi M k_{\rm B} T}{h^2}\right)^{3/2}\frac{k_{\rm B}T}{P},\qquad
E_{\rm tr} = \tfrac32 k_{\rm B}T,\quad H_{\rm tr} = \tfrac52 k_{\rm B}T,\quad
S_{\rm tr} = k_{\rm B}\left(\tfrac52 + \ln q_{\rm tr}\right) . $$

That entropy is the Sackur–Tetrode entropy. The heat capacities are
$C_V = \tfrac32 k_{\rm B}$ and $C_P = \tfrac52 k_{\rm B}$.

**Rotation.** The principal moments $I_a$ of the inertia tensor about the
centre of mass give rotational constants $B_a = h/(8\pi^2 I_a)$ in Hz. With
the rotational symmetry number $\sigma$,

$$ q_{\rm rot}^{\rm lin} = \frac{k_{\rm B}T}{\sigma h B},\quad
E = H = k_{\rm B}T,\quad S = k_{\rm B}(1 + \ln q);\qquad
q_{\rm rot}^{\rm nonlin} = \frac{\sqrt\pi}{\sigma}\left(\frac{k_{\rm B}T}{h}\right)^{3/2}
\frac{1}{\sqrt{B_A B_B B_C}},\quad E = H = \tfrac32 k_{\rm B}T,\quad S = k_{\rm B}(\tfrac32 + \ln q) . $$

These are the high-temperature (classical) limits of the rigid-rotor level
sums. For a linear rotor the level sum is
$k_{\rm B}T/(\sigma hB)\,[1 + y/3 + y^2/15 + \dots]$ with $y = hB/k_{\rm B}T$, so the
classical form is the first term. A single atom has no rotational part.
Linearity is `detect_linear`, the same test the frequency projection uses.

**Vibration.** Each real mode of wavenumber $\tilde\nu$ has a vibrational
temperature $\theta = hc\tilde\nu/k_{\rm B}$ (the local `c2` $= 100\,hc/k_{\rm B}$ of
`ferric-scf/src/thermo.rs::vibrational`, in K per cm⁻¹) and $x = \theta/T$. Measure the energy from the bottom of the well, so
the levels are $(n+\tfrac12)k_{\rm B}\theta$ and the sum is geometric:
$q = e^{-x/2}/(1 - e^{-x})$. Then
$\ln q = -\theta/(2T) - \ln(1 - e^{-\theta/T})$. Differentiate with respect
to $T$ and multiply by $k_{\rm B}T^2$:

$$ E_{\rm vib} = \tfrac12 k_{\rm B}\theta + \frac{k_{\rm B}\theta\,e^{-x}}{1 - e^{-x}} = \tfrac12 k_{\rm B}\theta + k_{\rm B}T\,\frac{x}{e^x - 1} . $$

Add $E/T$ to $k_{\rm B}\ln q$; the $\pm k_{\rm B}x/2$ cancel. Summed over modes:

$$ E_{\rm vib} = H_{\rm vib} = \sum\left[\tfrac12 k_{\rm B}\theta + k_{\rm B}T\,\frac{x}{e^x - 1}\right],\quad
S_{\rm vib} = k_{\rm B}\sum\left[\frac{x}{e^x - 1} - \ln(1 - e^{-x})\right],\quad
C_V = k_{\rm B}\sum\frac{x^2 e^x}{(e^x - 1)^2} . $$

The first sum is the zero-point energy. Check both limits. For a soft mode,
$x \to 0$: $E - E_{\rm ZPE} \to k_{\rm B}T$, $C_V \to k_{\rm B}$ and $S \to k_{\rm B}(1 - \ln x)$,
the classical oscillator. For a stiff mode, $x \to \infty$: $E \to E_{\rm ZPE}$
and $S \to 0$. Imaginary modes (negative wavenumbers) are left out, with a
warning.

Both limits are where floating point can fail, so the loop in
`ferric-scf/src/thermo.rs::vibrational` writes each expression in a form
that stays accurate there:

<!-- excerpt: ferric-scf/src/thermo.rs::vibrational -->
```rust
let theta = c2 * nu;
let x = theta / t;
// exp_m1 keeps x/(e^x - 1) accurate for soft modes (x -> 0).
let bose = 1.0 / x.exp_m1();
zpe += 0.5 * R_EH * theta;
s += x * bose - (-(-x).exp_m1()).ln();
e_th += x * bose;
// x^2 e^-x / (1 - e^-x)^2: no overflow for stiff modes at low T.
let em1 = (-x).exp_m1();
cv += x * x * (-x).exp() / (em1 * em1);
```

| Code | Equation |
|---|---|
| `theta`, `x` | $\theta = hc\tilde\nu/k_{\rm B}$, $x = \theta/T$ |
| `bose` | $1/(e^x - 1)$ |
| `zpe` | $\sum\tfrac12 k_{\rm B}\theta$ (`R_EH` is $k_{\rm B}$ in Hartree/K) |
| `s` | $\sum[x/(e^x-1) - \ln(1-e^{-x})]$, since `-(-x).exp_m1()` is $1 - e^{-x}$ |
| `e_th` | $\sum x/(e^x - 1)$, multiplied by $k_{\rm B}T$ after the loop |
| `cv` | $\sum x^2e^{-x}/(1-e^{-x})^2$, the same as $x^2e^x/(e^x-1)^2$ |

`exp_m1` evaluates $e^x - 1$ without the cancellation of `exp(x) - 1` at
small $x$. Multiplying the $C_V$ fraction by $e^{-2x}/e^{-2x}$ keeps $e^x$
from overflowing for a stiff mode at low $T$.

**Electronic.** $E = H = E_0$, the SCF energy with any dispersion
correction, and $S_{\rm el} = k_{\rm B}\ln g$ with $g = 2S_{\rm spin} + 1$ the spin
multiplicity.

**Totals.** The totals are sums of the four parts, and $G = H - TS$ for
each. Energies are in Hartree and entropies in Hartree/K, so $k_{\rm B}$ is used as
$k_{\rm B}/E_h = 3.166811563\times10^{-6}$ Hartree/K. Constants are CODATA 2018.
Lengths go to metres with ferric's own Bohr radius
(`units::BOHR_RADIUS_ANGSTROM`, the value that read the Ångström input), and
masses are the isotope-averaged standard atomic weights of the frequency
calculation.

The script derives each closed form from its partition function in sympy,
takes the high- and low-temperature limits, checks the transcribed formulas
for water against $\ln Q(T)$ summed over the vibrational levels and
differentiated numerically (relative $10^{-10}$), the rigid-rotor level sum
against its Euler–Maclaurin series, and the argon entropy against
NIST-JANAF (154.845 J/(mol K) at 298.15 K and 1 bar). The Python test
`test_thermo_ir.py` compares the full result with PySCF's `thermo` on the
same frequencies, geometry, masses and constants.

> **Verified equation {#eq:freq-rrho-thermo}** · RRHO partition functions
>
> Each closed form follows from its partition function in symbols, with the
> classical and low-temperature limits of the oscillator; for water the
> transcribed formulas equal $\ln Q(T)$ summed over the vibrational levels and
> differentiated numerically to $10^{-10}$ relative.
>
> | | |
> |---|---|
> | Proof | `scripts/rrho_thermo_sympy.py`, claims S1 to S3, L1 and N1 |
> | Negative control | the runner drops the symmetry number from the nonlinear rotor (claim N1 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/thermo.rs::rrho_thermochemistry_with_masses` |
> | Code test | `thermo_matches_exact_in_the_trivial_limit` in `crates/ferric-scf/src/thermo.rs` |
>
> <!-- verified: id=freq.rrho-thermo; script=scripts/rrho_thermo_sympy.py; claim=N1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/thermo.rs::rrho_thermochemistry_with_masses; rust_test=ferric-scf/src/thermo.rs::thermo_matches_exact_in_the_trivial_limit; registry=rrho-thermo -->

**The symmetry number is an input.** The default is $\sigma = 1$. ferric
detects Abelian point groups only, so it cannot count the proper rotations
of NH3 or CH4. With $\sigma = 1$ for a molecule whose true value is
$\sigma$, $S_{\rm rot}$ is too large by $k_{\rm B}\ln\sigma$ and $G$ too low by
$k_{\rm B}T\ln\sigma$.

### Infrared intensities

A mode absorbs infrared light only if the dipole moment changes as the atoms
move along it. The symmetric stretch of CO2 leaves the dipole at zero and is
dark; the asymmetric stretch creates one and absorbs strongly. The
intensity is the squared derivative of the dipole along the mode.
`ferric-scf/src/infrared.rs::ir_spectrum` and `ir_intensities` (CLI
`task = "frequencies"`; Python `FrequencyResult.ir_intensities`) give it for
every normal mode. The proof is `scripts/ir_intensity_sympy.py`.

**Intensity.** Take the $0 \to 1$ band of a harmonic mode with normal
coordinate $Q$, mass weighted: $Q = \sum_i \sqrt{m_i}\,\Delta x_i\,u_i$. The
integrated cross section is
$\int\sigma\,d\omega = \pi\omega|\langle0|\mu|1\rangle|^2/(3\varepsilon_0\hbar c)$,
with the isotropic average included. Expand $\mu$ to first order in $Q$, so
$\langle0|\mu|1\rangle = (\partial\mu/\partial Q)\langle0|Q|1\rangle$, and use
the harmonic matrix element $|\langle 0|Q|1\rangle|^2 = \hbar/(2\omega)$.
Integrate over wavenumber ($d\omega = 2\pi c\,d\tilde\nu$) and take it per
mole:

$$ A_k = \frac{N_A}{12\,\varepsilon_0 c^2}\left|\frac{\partial\boldsymbol\mu}{\partial Q_k}\right|^2 . $$

$A_k$ is the Napierian integrated absorption coefficient in m/mol. $\hbar$ and
$\omega$ cancel, so the result is the classical one. It is also independent
of $T$: the $n \to n+1$ band is $n+1$ times stronger, and the
Boltzmann-weighted net absorption is $\sum_n (p_n - p_{n+1})(n+1) = 1$.

**Units.** With $\partial\mu/\partial Q$ in atomic units, $e/\sqrt{m_e}$,

$$ A_k\,[\text{km/mol}] = C\,\left|\frac{\partial\boldsymbol\mu}{\partial Q_k}\right|^2_{\rm a.u.},\qquad
C = \frac{N_A e^2}{12\,\varepsilon_0 c^2 m_e}\times10^{-3} = 1\,777\,097.7313 $$

(`AU_TO_KM_PER_MOL`, CODATA 2018). In other units that is 974.880 km/mol
per $e^2/u$ and 42.2561 km/mol per (D/Å)$^2$/u.

**Normal modes.** `FrequencyResult::normal_modes` stores $x_k = M^{-1/2}u_k$,
with $u_k$ the unit eigenvectors of the mass-weighted Hessian in electron
masses. So $\sum_i m_i x_{k,i}^2 = 1$ and $x_k = \partial R/\partial Q_k$. The
chain rule then gives the dipole derivative along the mode,

$$ \frac{\partial\mu_a}{\partial Q_k} = \sum_i \Pi_{ai}\,x_{k,i},\qquad \Pi_{ai} = \frac{\partial\mu_a}{\partial R_i} , $$

with no further mass factor. $\Pi$ is the atomic polar tensor
($3\times3N$, in $e$). In `ir_intensities` this is one matrix product,
`apt.dot(&normal_modes.t())`, followed by $C$ times the sum of squares over
the three components.

**The dipole derivative.** In a uniform field $\mathbf F$ ferric's energy is
$E(\mathbf F) = E(0) - \boldsymbol\mu\cdot\mathbf F - \tfrac12\mathbf F\alpha\mathbf F + \dots$
(a charge $q$ at $\mathbf R$ gains $-q\mathbf F\cdot\mathbf R$, see section
@sec:external-potentials-operator-classical-energy), so
$\boldsymbol\mu = -\partial E/\partial\mathbf F$. Differentiate once more,
by $R_i$, and swap the order:

$$ \Pi_{ai} = -\frac{\partial^2 E}{\partial F_a\,\partial R_i}
\approx -\frac{g_i(+h\mathbf e_a) - g_i(-h\mathbf e_a)}{2h} . $$

That is the production route, `dipole_derivatives_field`: six SCF plus
analytic gradient evaluations in fields $\pm h$ ($h = 10^{-3}$ a.u.),
independent of the molecule size. The gradient of an SCF converged in the
field carries the orbital relaxation, so the result is the relaxed (response)
dipole derivative of the SCF energy. The truncation is
$h^2/6\,|\partial^4E/\partial F^3\partial R|$. SCF noise is at most
$K n\,\delta_D/h$, with $K$ the gradient's density sensitivity, $n$ the basis
size and $\delta_D$ the density threshold, so the field SCFs run at
$\delta_D \le 10^{-9}$ (`IR_DENSITY_CONV`).

The validation route, `dipole_derivatives_dipole_fd`, differences the SCF
dipole $\mu_a = \sum_A Z_A R_{A,a} - \mathrm{tr}(D\,r_a)$ along each Cartesian
displacement ($6N$ SCFs). Both routes satisfy the translational sum rule
$\sum_A \Pi_{a,(A,b)} = Q\,\delta_{ab}$ for molecular charge $Q$ without
external point charges: moving every atom by the same amount moves the
charge $Q$. `IrSpectrum::sum_rule_residual` records the deviation. The dipole
origin is fixed at 0. A different fixed origin changes $\boldsymbol\mu$ by a
constant, so $\Pi$ and the intensities do not depend on it.

**Scope.** The SCF references: RHF, UHF, ROHF and their Kohn–Sham variants.
Correlated kinds are refused: their dipole derivative would be the field
derivative of the correlated gradient, which is not validated. With
`[dft] dispersion` the modes come from the corrected Hessian and the dipole
derivative from the SCF alone. D3 does not depend on the field, and MBD's
density dependence is not included in $\Pi$.

The script derives $C$ from CODATA, the $0\to1$ band strength from the
matrix element (classical limit), the temperature independence, and the
field-route identity on a model $E(R,F)$. It checks the transcribed
contraction against differencing $\mu$ along each mode of a bent triatomic
model, the closed form $A = C q^2/\mu_{\rm red}$ for point charges on a
diatomic, and the field route's $O(h^2)$ truncation. The Rust test
`crates/ferric-scf/tests/ir_intensity_routes.rs` compares the two routes for
RHF, RKS, UHF, UKS and ROHF against a bar built from Richardson truncation
estimates and measured SCF-noise bounds.

> **Verified equation {#eq:freq-ir-intensity}** · the intensity and its units
>
> The constant equals $N_Ae^2/(12\varepsilon_0c^2m_e)\times10^{-3}$ (km/mol) from CODATA 2018;
> the $0\to1$ band strength from the harmonic matrix element is the classical
> one; contracting the dipole derivative with the stored modes equals
> differencing the dipole along each mode of a model triatomic; and the field
> route recovers the dipole derivative to its $O(h^2)$ truncation.
>
> | | |
> |---|---|
> | Proof | `scripts/ir_intensity_sympy.py`, claims C1, S1, N1 to N3 |
> | Negative control | the runner drops the mass weighting of the modes (claim N2 fails) and flips the sign of the field route (claim N3 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/infrared.rs::ir_intensities` |
> | Code test | `ir_intensity_matches_exact_in_the_trivial_limit` in `crates/ferric-scf/src/infrared.rs` |
>
> <!-- verified: id=freq.ir-intensity; script=scripts/ir_intensity_sympy.py; claim=N1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/infrared.rs::ir_intensities; rust_test=ferric-scf/src/infrared.rs::ir_intensity_matches_exact_in_the_trivial_limit; registry=ir-intensities -->

## Reproducibility and refusals {#frequencies-reproducibility-refusals}

- The rank check is a refusal: a translation and rotation subspace of rank
  other than 6, or 5 for a linear molecule, means a degenerate geometry
  (coincident atoms) and is an error.
- Masses are isotope averages (`ferric-scf/src/frequencies.rs::atom_masses`)
  for $Z = 1$ to $54$; a heavier element is an error naming it.
- A molecule is linear when the smallest principal moment of inertia is
  below `LINEARITY_THRESH` ($10^{-6}$) times the largest. The test is a
  ratio, so it has no unit. Any two-atom system is linear.
- With `scf_warm_start` off (the default), each displaced SCF starts cold. A
  warm start from the undisplaced orbitals is measurably noisier at equal
  density convergence, so it is opt-in.

## Options that change the equation {#frequencies-options-change-equation}

| Option | Effect |
|---|---|
| `hessian = "auto"` (default) | analytic where the availability predicate of section @sec:hf-hessian or @sec:ks-hessian accepts the system, finite differences otherwise; the choice is made before the SCF |
| `hessian = "analytic"` | the Hartree–Fock analytic Hessian only; a Kohn–Sham reference is refused before the SCF |
| `hessian = "analytic-moving-grid"` | the analytic RKS or UKS Hessian of section @sec:ks-hessian |
| `hessian = "fd"` | central differences of the analytic gradient |
| `delta` | FD displacement; unused by analytic Hessians |
| dispersion D3(BJ) | the route the run without dispersion takes, plus the analytic D3(BJ) Hessian (`harmonic_frequencies_with_geometric_correction`); on the finite-difference route, differences of the corrected gradient |
| dispersion MBD@rsSCS | finite differences of the gradient on the corrected surface (`harmonic_frequencies_with_scf_correction`); the MBD energy follows the density, so `analytic` and `analytic-moving-grid` are refused |

## Numerical anchors {#frequencies-numerical-anchors}

| Check | Reference | Agreement | Test |
|---|---|---|---|
| Harmonic frequencies from FD of analytic gradients: H2O, NH3 × RHF, PBE, B3LYP / 6-31G, cc-pVDZ; UHF OH, CH3, HO2, UKS-PBE, ROHF CH3 / 6-31G | PySCF same-step FD of analytic gradients; PySCF analytic Hessians; `thermo.harmonic_analysis` with ferric's masses | same-step FD 2.4e-7 Ha/Bohr² and 7.5e-4 cm⁻¹, bars 2e-6 and 5e-3; HF against the analytic Hessian 2.75e-5 Ha/Bohr² and 0.10 cm⁻¹ (the 5e-3 Bohr step's truncation), bars 1e-4 and 0.5; KS against PySCF's analytic Hessian differs by 5 to 35 cm⁻¹ because that Hessian has no grid response | `crates/ferric-scf/tests/validation_frequencies.rs` |
| Analytic RHF Hessian frequencies: H2O, NH3, CH2O / cc-pVDZ | PySCF `hessian.rhf` | at most 6.9e-4 cm⁻¹, bar 1e-2 | `crates/ferric-scf/tests/validation_rhf_hessian.rs` |
| RRHO thermochemistry: water, CO2 (RHF) and OH (UHF) / STO-3G at their minima, 298.15 K and 1 atm (water also 500 K, 2 bar) | PySCF `hessian.thermo` on ferric's Hessian, geometry and masses, CODATA 2018 constants | every E, H, TS and G term within 8.0e-13 Ha (measured on the `3f612e0b3` build), bar 1e-8 Ha; with PySCF's stock CODATA 2014 constants 7.8e-9 to 1.5e-8 Ha, reported only | `crates/ferric-python/tests/test_thermo_ir.py` |
| IR intensities: water / STO-3G at its minimum | PySCF RHF dipoles at displaced geometries contracted with PySCF's normal modes | at most 1.1e-3 km/mol on intensities of 7 to 44 km/mol (measured, `3f612e0b3`), within bars of 1.9e-2 to 4.6e-2 km/mol built from the dipole route's Richardson estimate | same |
| IR polar tensor, field route against the dipole route: RHF, RKS, UHF, UKS, ROHF | `dipole_derivatives_dipole_fd` | below 1e-5 $e$ on $\Pi$ for RHF water / 6-31G; bar from Richardson truncation estimates and measured SCF-noise bounds | `crates/ferric-scf/tests/ir_intensity_routes.rs` |

<!-- figures: site/src/reference/validation.md "Harmonic frequencies (FD of analytic gradients; Cartesian Hessian and cm⁻¹)"; site/src/reference/validation.md "Analytic RHF Hessian (skeleton one- and two-electron, overlap/W and CPHF response terms) and its harmonic frequencies" -->

The two HF rows measure different things. Against PySCF with the same step,
the finite-difference route agrees to 7.5e-4 cm⁻¹: the two codes make the
same truncation error. Against the analytic Hessian the same route is off
by 0.10 cm⁻¹, and that is the $O(\delta^2)$ truncation of the 5e-3 Bohr
step.

**Known limits.** The analytic Hessians cover the references and options
listed in sections @sec:hf-hessian and @sec:ks-hessian. UHF states are not checked for internal
stability, so an unstable UHF solution gets the frequencies of that
solution.

## Sources

| Source | Used for |
|---|---|
| `scripts/hf_frequencies_sympy.py`, claims C1, C2, S1, D1, P1 to P4 | verified equations @eq:freq-units-diatomic and @eq:freq-projector |
| `crates/ferric-scf/src/frequencies.rs` module documentation and `detect_linear` | the pipeline, the Hessian options and the linearity test |
| `site/src/reference/validation.md` | the anchor rows |
| `scripts/rrho_thermo_sympy.py`, `scripts/ir_intensity_sympy.py` | verified equations @eq:freq-rrho-thermo and @eq:freq-ir-intensity |
