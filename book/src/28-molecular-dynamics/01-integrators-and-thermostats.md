# Integrators and thermostats: velocity Verlet, Berendsen, Langevin BAOAB {#molecular-dynamics}

<!-- status: edit; registry: molecular-dynamics -->

Given the forces on the nuclei, how do you advance them in time without the
energy running away? A textbook Runge–Kutta step is accurate for a short run
but does not respect the structure of Hamilton's equations. Over the
$10^4$ to $10^6$ steps of a trajectory its energy drifts. The velocity-Verlet
step has the same cost per step, one gradient, and its energy error stays
bounded over very long runs: the step follows a nearby Hamiltonian, exactly
for a harmonic potential and to all orders of an asymptotic series
otherwise. Constant-energy
dynamics is rarely the experiment, though. A simulation at fixed temperature
needs a thermostat, and thermostats differ in what ensemble they sample.

ferric implements three integrators in `ferric-md`: velocity Verlet (NVE),
velocity Verlet with Berendsen velocity rescaling, and Langevin dynamics with
the BAOAB splitting. The force comes from a `ForceProvider`, so the same
driver runs ab-initio MD on an SCF gradient and force-field MD on the
molecular-mechanics gradient. A trajectory is also a source of geometries
for machine-learning labels (section @sec:training-labels-trajectories).

The plan: units (@sec:molecular-dynamics-units), degrees of freedom and
initial velocities (@sec:molecular-dynamics-initialisation), velocity Verlet
(@sec:molecular-dynamics-velocity-verlet), the Berendsen thermostat
(@sec:molecular-dynamics-berendsen), Langevin BAOAB
(@sec:molecular-dynamics-baoab), the conserved quantity
(@sec:molecular-dynamics-conserved), the code
(@sec:molecular-dynamics-code), running it (@sec:molecular-dynamics-running)
and what is proven (@sec:molecular-dynamics-proven).

| Symbol | Meaning | Code name |
|---|---|---|
| $\mathbf R_a$, $\mathbf v_a$ | position (Bohr) and velocity (Bohr per atomic unit of time) of atom $a$ | `coords`, `velocities` |
| $m_a$ | mass in electron masses | `masses_me` |
| $\mathbf g_a = \partial E/\partial\mathbf R_a$ | gradient, Ha/Bohr; the force is $-\mathbf g_a$ | `gradient` |
| $\Delta t$ | time step | `dt_fs` (fs) |
| $K$, $E_{\mathrm{pot}}$ | kinetic and potential energy, Ha | `kinetic`, `potential` |
| $N_{\mathrm{dof}}$ | degrees of freedom | `n_dof` |
| $T$, $T_0$ | instantaneous and target temperature, K | `temperature_k` |
| $k_B$ | Boltzmann constant, Ha/K | `BOLTZMANN_HARTREE_PER_K` |
| $\tau$ | Berendsen coupling time, fs | `Thermostat::Berendsen { tau_fs }` |
| $\gamma$ | Langevin friction, 1/ps | `Thermostat::Langevin { friction_per_ps }` |
| $Q$ | kinetic energy the thermostat has injected | conserved-quantity bookkeeping |

| Entry point | Location |
|---|---|
| Rust driver | `ferric-md/src/driver.rs::run_md` |
| Configuration | `ferric-md/src/config.rs::{MdConfig,Thermostat}` |
| Velocities | `ferric-md/src/velocities.rs::{maxwell_boltzmann,remove_com_momentum,kinetic_energy,temperature}` |
| Random numbers | `ferric-md/src/rng.rs::Xoshiro256pp` |
| CLI | `task = "md"` with an `[md]` section, methods `rhf`, `uhf`, `rohf`, `ksdft` |
| Python | `run_aimd(mol, basis_name, n_steps, ...)`, `run_ffmd(topology, symbols, coords_angstrom, n_steps, ...)` |

## Units {#molecular-dynamics-units}

The integrators work in atomic units: Bohr, Hartree, electron masses and the
atomic unit of time. Users give the time step in femtoseconds, the
temperature in kelvin, the Berendsen time in femtoseconds, the friction in
inverse picoseconds and masses in atomic mass units; the driver converts at
the boundary with three constants from `ferric-md/src/units.rs`:

| Constant | Value | Use |
|---|---|---|
| `AU_TIME_FS` | $2.4188843265857\times10^{-2}$ fs | $\Delta t = \texttt{dt\_fs}/\texttt{AU\_TIME\_FS}$ |
| `AMU_TO_ELECTRON_MASS` | $1/5.48579909065\times10^{-4} = 1822.888\ldots$ | $m = m_{\mathrm{amu}} \times 1822.888$ |
| `BOLTZMANN_HARTREE_PER_K` | $3.166811563455608\times10^{-6}$ Ha/K | $k_B T$ in Hartree |

A step of 0.5 fs is 20.67 atomic units of time. A friction of $\gamma$ per
picosecond is $\gamma \times \texttt{AU\_TIME\_FS}/1000$ per atomic unit
of time. The equations below use velocities in Bohr per atomic unit of
time, so they carry no conversion factor.

## Degrees of freedom and initial velocities {#molecular-dynamics-initialisation}

The temperature of a set of velocities is defined by equipartition. Each
quadratic degree of freedom carries $\tfrac12 k_B T$ on average, so

$$ K = \sum_a \tfrac12 m_a \lvert\mathbf v_a\rvert^2, \qquad T = \frac{2K}{N_{\mathrm{dof}}\,k_B}. \label{eq:molecular-dynamics-1} $$

`ferric-md/src/velocities.rs::temperature` returns this, and zero when
$N_{\mathrm{dof}} = 0$. How many degrees of freedom does a molecule of $N$
atoms have? If the centre-of-mass momentum is removed and stays zero, three
of the $3N$ velocity components are not free: $N_{\mathrm{dof}} = 3N - 3$ for
$N > 1$. Otherwise $N_{\mathrm{dof}} = 3N$. Velocity Verlet and Berendsen
scaling keep a zero total momentum at zero, so with `remove_com_motion` they use
$3N-3$. Langevin noise kicks the centre of mass, so BAOAB removes the
centre-of-mass motion only at the start and counts $3N$ degrees of freedom. A
single atom has $N_{\mathrm{dof}} = 3$.

Initial velocities come from the Maxwell–Boltzmann distribution. Each
Cartesian component of atom $a$ is Gaussian with variance $k_B T/m_a$,

$$ v_{a,\alpha} = \sqrt{\frac{k_B T}{m_a}}\;\xi_{a,\alpha}, \qquad \xi \sim \mathcal N(0,1), \label{eq:molecular-dynamics-2} $$

which is `ferric-md/src/velocities.rs::maxwell_boltzmann`, drawn atom by atom in
$x,y,z$ order. A finite sample has a nonzero total momentum and a kinetic
temperature that scatters around $T$, so the driver then subtracts the
centre-of-mass velocity (`remove_com_momentum`) and multiplies all velocities
by $\sqrt{T_{\mathrm{init}}/T_{\mathrm{inst}}}$, where $T_{\mathrm{inst}}$ is
(@eq:molecular-dynamics-1) with the run's $N_{\mathrm{dof}}$. The run
therefore starts at exactly the requested temperature, not at a random draw
around it. The random numbers come from `ferric-md/src/rng.rs::Xoshiro256pp`,
xoshiro256++ seeded through SplitMix64, with Box–Muller Gaussians. It is
self-contained, so a given `seed` reproduces a trajectory on every platform.
If the user supplies `initial_velocities`, nothing is drawn.

Check the limit $T = 0$: every standard deviation $\sqrt{k_BT/m_a}$ vanishes, the velocities are zero and
the rescaling factor is not needed. With `init_temperature_k` unset and no
`temperature_k`, the run starts from rest.

## Velocity Verlet {#molecular-dynamics-velocity-verlet}

Newton's equations in Hamiltonian form for $H = \sum_a \mathbf p_a^2/2m_a +
E(\mathbf R)$ are $\dot{\mathbf R}_a = \mathbf p_a/m_a$ and
$\dot{\mathbf p}_a = -\mathbf g_a$. Splitting the flow into a kick (momentum
changes, positions fixed) and a drift (positions change, momenta fixed),
and composing a half kick, a full drift and a half kick, gives the
velocity-Verlet step (`ferric-md/src/driver.rs::run_md`):

1. $\mathbf v_a \leftarrow \mathbf v_a - \tfrac{\Delta t}{2}\,\mathbf g_a/m_a$
2. $\mathbf R_a \leftarrow \mathbf R_a + \Delta t\,\mathbf v_a$
3. evaluate $E$ and $\mathbf g$ at the new positions
4. $\mathbf v_a \leftarrow \mathbf v_a - \tfrac{\Delta t}{2}\,\mathbf g_a/m_a$

The gradient from step 3 is reused as the first half kick of the next step,
so a run of $n$ steps costs $n+1$ gradient evaluations (`n_force_evaluations`).
With the momentum $p = m v$ and a one-dimensional abstract potential $V$ with
gradient $g = V'$, the map $(x,p) \mapsto (x',p')$ is

$$ p_h = p - \tfrac{\Delta t}{2}\,g(x), \qquad x' = x + \Delta t\,\frac{p_h}{m}, \qquad p' = p_h - \tfrac{\Delta t}{2}\,g(x'). \label{eq:molecular-dynamics-vv} $$

Three properties follow without choosing $V$.

**It preserves phase-space area.** Each of the three substeps is a shear:
$p_h$ depends on $x$ and $p$ with unit coefficient on $p$, $x'$ on $x$ with
unit coefficient, and so on. A shear has Jacobian determinant 1, and so does
the composition,

$$ \det\frac{\partial(x',p')}{\partial(x,p)} = 1 \quad\text{for every } g. \label{eq:molecular-dynamics-symplectic} $$

That is the symplectic property in one degree of freedom, and it is why the
energy does not drift (the shadow Hamiltonian below). The $3N$-dimensional step
is a product of the same shears, each symplectic.

> **Verified equation @eq:molecular-dynamics-symplectic** · the velocity-Verlet map is area preserving
>
> For an abstract smooth potential, the Jacobian determinant of
> (@eq:molecular-dynamics-vv) simplifies to exactly 1. A finite-difference
> Jacobian of a Morse step in 40-digit arithmetic gives $\lvert\det J - 1\rvert
> < 10^{-25}$ (measured $8.5\times10^{-27}$).
>
> | | |
> |---|---|
> | Proof | `scripts/md_integrators_sympy.py`, claim verlet-symplectic |
> | Negative control | the second half kick evaluated at the old position |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-md/src/driver.rs::run_md` |
>
> <!-- verified: id=md.verlet.symplectic; script=scripts/md_integrators_sympy.py; claim=verlet-symplectic; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-md/src/driver.rs::run_md; rust_test=ferric-md/tests/integrator_harmonic.rs::velocity_verlet_matches_closed_form_discrete_solution; registry=molecular-dynamics -->

**It is time reversible.** Flip the momentum after a step, take the same
step, flip again: the second step lands on the starting point. In
(@eq:molecular-dynamics-vv) the second half kick of the reversed step undoes
the first half kick of the forward step, because both are evaluated at
the same position $x'$ (and then $x$), and the drift undoes the drift. With
$\Phi$ the step (@eq:molecular-dynamics-vv) and $R(x,p) = (x,-p)$ the momentum flip,

$$ R\,\Phi\,R\,\Phi = \mathrm{id}. \label{eq:molecular-dynamics-time-reversal} $$

The symbolic composition equals the identity for abstract $g$, and 100 Morse steps
forward, a momentum flip, and 100 steps back return to the start to
$6\times10^{-41}$. Time reversibility is what makes the integrator
symmetric, which removes every odd power of $\Delta t$ from its error.

> **Verified equation @eq:molecular-dynamics-time-reversal** · velocity Verlet is time reversible
>
> Symbolically, step, momentum flip, step, momentum flip equals the identity
> for an abstract force. Numerically, 100 Morse steps and 100 reversed steps
> return to the start inside $10^{-25}$.
>
> | | |
> |---|---|
> | Proof | `scripts/md_integrators_sympy.py`, claim verlet-time-reversal |
> | Negative control | the first half kick doubled in the numeric step |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-md/src/driver.rs::run_md` |
>
> <!-- verified: id=md.verlet.time-reversal; script=scripts/md_integrators_sympy.py; claim=verlet-time-reversal; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-md/src/driver.rs::run_md; rust_test=ferric-md/tests/integrator_harmonic.rs::reversing_velocities_retraces_the_trajectory; registry=molecular-dynamics -->

**Its energy error is $O(\Delta t^2)$ and bounded.** Take the harmonic
oscillator $V = \tfrac12 k x^2$ with $\omega^2 = k/m$, where the step is linear
and everything can be written out. The step matrix has trace
$2 - \omega^2\Delta t^2$ and determinant 1, so it is a rotation with
$\cos\theta = 1 - \omega^2\Delta t^2/2$. The trajectory is stable when
$\lvert\cos\theta\rvert < 1$, which is $\omega\Delta t < 2$. At equality the
step matrix is $-\mathbf 1$ plus a nilpotent part, and $x$ grows linearly
with the number of steps. For a vibration of period
$T_{\mathrm{vib}} = 2\pi/\omega$ the bound is
$\Delta t < T_{\mathrm{vib}}/\pi$. A practical step is a tenth of the
shortest period or less: the O–H stretch of water (about 10 fs) wants
$\Delta t \lesssim 1$ fs.

The energy $H = p^2/2m + \tfrac12 k x^2$ is not conserved by the step, but this
quantity is:

$$ \tilde H = \frac{p^2}{2m} + \frac{k x^2}{2}\Big(1 - \frac{\omega^2\Delta t^2}{4}\Big). \label{eq:molecular-dynamics-shadow} $$

Substituting (@eq:molecular-dynamics-vv) with $g = kx$ and expanding gives
$\tilde H(x',p') = \tilde H(x,p)$ identically. The true energy differs from
the shadow by

$$ H - \tilde H = \frac{k x^2}{2}\,\frac{\omega^2\Delta t^2}{4} = \frac{k x^2\,\omega^2\Delta t^2}{8}, $$

so along the trajectory $H$ oscillates with the position inside a band of
height $\propto\Delta t^2$ and does not drift. Check the limit
$\Delta t \to 0$: $\tilde H \to H$. Check $k \to 0$ (a free particle): $\tilde H = p^2/2m$ and
the step is exact. For general $V$ the same statement holds as a series: the
step is the exact flow of a modified Hamiltonian
$\tilde H = H + \Delta t^2 H_2 + \Delta t^4 H_4 + \dots$ (backward error
analysis for symplectic integrators; the series is asymptotic and the
proof here covers the harmonic case, where it truncates). So the
energy fluctuation of a velocity-Verlet run falls by a factor 4 when you halve
the time step. Over 2000 steps at $\omega\Delta t \approx 0.16$ the script
finds $\tilde H$ constant to $2\times10^{-40}$ while $H$ spans $4\times10^{-3}$.

> **Verified equation @eq:molecular-dynamics-shadow** · the harmonic shadow Hamiltonian of velocity Verlet
>
> For $V = \tfrac12 kx^2$, one velocity-Verlet step has
> $\cos\theta = 1 - \omega^2\Delta t^2/2$ and conserves
> $\tilde H = p^2/2m + (kx^2/2)(1 - \omega^2\Delta t^2/4)$ exactly. The true energy
> exceeds it by $kx^2\omega^2\Delta t^2/8$.
>
> | | |
> |---|---|
> | Proof | `scripts/md_integrators_sympy.py`, claim verlet-shadow-energy |
> | Negative control | the factor $1/4$ replaced by $1/2$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-md/src/driver.rs::run_md` |
>
> <!-- verified: id=md.verlet.shadow-energy; script=scripts/md_integrators_sympy.py; claim=verlet-shadow-energy; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-md/src/driver.rs::run_md; rust_test=ferric-md/tests/integrator_harmonic.rs::shadow_energy_is_conserved_while_true_energy_oscillates; registry=molecular-dynamics -->

What does this mean for an ab-initio run? The bound on the energy
fluctuation holds only if the gradient is the exact gradient of the energy the
integrator sees. If the SCF is converged loosely the force is not
$-\nabla E$ of any smooth surface, and the energy drifts regardless of
$\Delta t$. The SCF convergence threshold, not the time step, then sets the
drift. The AIMD tests (`crates/ferric-scf/tests/md_aimd.rs`) therefore set
`density_conv = 1e-10` and `energy_conv = 1e-12` in place of the defaults
$10^{-6}$ and $10^{-3}$; from Python pass `scf_density_conv` and
`scf_energy_conv` to `run_aimd`.

## The Berendsen thermostat {#molecular-dynamics-berendsen}

To hold a system near a target temperature $T_0$, Berendsen and co-workers
couple it to a heat bath by requiring the temperature to relax at a rate set by
a time constant $\tau$:

$$ \frac{dT}{dt} = \frac{T_0 - T}{\tau}. \label{eq:molecular-dynamics-berendsen-rate} $$

How do you realise that rate with the velocities? Scaling every velocity by
the same factor $\lambda$ scales $K$ by $\lambda^2$ and $T$ by $\lambda^2$.
Take one velocity-Verlet step, then multiply the velocities by $\lambda$ with

$$ \lambda^2 = 1 + \frac{\Delta t}{\tau}\Big(\frac{T_0}{T} - 1\Big). \label{eq:molecular-dynamics-berendsen-lambda} $$

The new temperature is $T' = \lambda^2 T = T + (\Delta t/\tau)(T_0 - T)$, so
$(T' - T)/\Delta t = (T_0 - T)/\tau$ exactly: the scaling is one explicit Euler
step of (@eq:molecular-dynamics-berendsen-rate). Iterating from $T(0)$ converges
to $T_0 + (T(0) - T_0)e^{-t/\tau}$ with an error that halves when $\Delta t$
halves (first order). Check the limits. At $T = T_0$, $\lambda = 1$. As
$\tau \to \infty$, $\lambda \to 1$ and the step is velocity Verlet. For
$\tau = \Delta t$, $\lambda^2 = T_0/T$ and the temperature is set to $T_0$ in
one step. A value $\tau \approx 100$ fs, much longer than $\Delta t$, equilibrates
smoothly.

> **Verified equation @eq:molecular-dynamics-berendsen-lambda** · the Berendsen scaling factor realises the first-order rate
>
> $\lambda^2 = 1 + (\Delta t/\tau)(T_0/T - 1)$ gives
> $(T'-T)/\Delta t = (T_0-T)/\tau$ identically. The iterated map relaxes to
> $T_0$ with error ratio 2.0 between $\Delta t$ and $\Delta t/2$.
>
> | | |
> |---|---|
> | Proof | `scripts/md_integrators_sympy.py`, claim berendsen-rate |
> | Negative control | $T_0/T - 1$ replaced by $T/T_0 - 1$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-md/src/config.rs::Thermostat` |
>
> <!-- verified: id=md.berendsen.rate; script=scripts/md_integrators_sympy.py; claim=berendsen-rate; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-md/src/config.rs::Thermostat; rust_test=ferric-md/tests/thermostats.rs::berendsen_relaxes_to_target_and_conserves_e_minus_q; registry=molecular-dynamics -->

The Berendsen thermostat gives the right mean temperature but not the
canonical ensemble. The rate equation drives $T$ to $T_0$ deterministically
and suppresses the fluctuations of the kinetic energy below the canonical
value (a literature result; the proof here covers the rate, not the
distribution). Use it to bring a system to a temperature, then switch to
NVE or Langevin to sample. ferric documents it as equilibration only.

## Langevin dynamics with BAOAB {#molecular-dynamics-baoab}

Langevin dynamics adds friction and random kicks to Newton's equations,

$$ d\mathbf p = -\mathbf g\,dt - \gamma\,\mathbf p\,dt + \sqrt{2\gamma m k_B T_0}\;d\mathbf W, $$

where $\mathbf W$ is a Wiener process. The friction removes energy and the noise
returns it, balanced so that the stationary distribution of the continuous
dynamics is canonical at $T_0$. The velocity part alone is the
Ornstein–Uhlenbeck (OU) process $dv = -\gamma v\,dt + \sqrt{2\gamma k_B T_0/m}\,dW$.

The BAOAB splitting integrates the three parts, B (kick), A (drift) and
O (the OU step), in the order B, A, O, A, B with halves on B and A:

1. B: $\mathbf v \leftarrow \mathbf v - \tfrac{\Delta t}{2}\,\mathbf g/m$
2. A: $\mathbf R \leftarrow \mathbf R + \tfrac{\Delta t}{2}\,\mathbf v$
3. O: $\mathbf v \leftarrow c\,\mathbf v + \sqrt{(1 - c^2)\,k_B T_0/m}\;\boldsymbol\xi$, with $c = e^{-\gamma\Delta t}$ and $\xi\sim\mathcal N(0,1)$
4. A: $\mathbf R \leftarrow \mathbf R + \tfrac{\Delta t}{2}\,\mathbf v$
5. B: evaluate the gradient at the new positions, then $\mathbf v \leftarrow \mathbf v - \tfrac{\Delta t}{2}\,\mathbf g/m$

The closing B of one step and the opening B of the next use the same
gradient, so a step costs one gradient evaluation. The O step is the exact solution of the OU equation
over $\Delta t$ and it preserves the Maxwell–Boltzmann variance. If
$\operatorname{Var}\,v = k_B T_0/m$, then

$$ \operatorname{Var}\,v' = c^2\,\frac{k_B T_0}{m} + (1 - c^2)\,\frac{k_B T_0}{m} = \frac{k_B T_0}{m}. \label{eq:molecular-dynamics-ou} $$

More generally the variance obeys $\dot s = -2\gamma s + 2\gamma k_B T_0/m$,
whose solution after $\Delta t$ is $k_B T_0/m + (s_0 - k_B T_0/m)e^{-2\gamma\Delta t}$,
and the O step reproduces it for every initial variance $s_0$. Check
the limits. For $\gamma = 0$, $c = 1$, the noise amplitude is zero and the two A
half steps combine: BAOAB equals velocity Verlet to rounding. For
$\gamma\Delta t \to \infty$, $c \to 0$ and the velocities are redrawn from the
Maxwell–Boltzmann distribution at every step.

> **Verified equation @eq:molecular-dynamics-ou** · the O step preserves the Maxwell–Boltzmann variance
>
> With $c = e^{-\gamma\Delta t}$ and noise variance $(1-c^2)k_BT_0/m$, the
> variance $k_BT_0/m$ is fixed, and the step equals the exact solution of the OU variance
> equation for every initial variance. A Gauss–Hermite quadrature of the update
> confirms the fixed point to $10^{-40}$.
>
> | | |
> |---|---|
> | Proof | `scripts/md_integrators_sympy.py`, claim baoab-ou-variance |
> | Negative control | the noise variance $(1-c^2)$ replaced by $(1-c)$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-md/src/driver.rs::run_md` |
>
> <!-- verified: id=md.baoab.ou-variance; script=scripts/md_integrators_sympy.py; claim=baoab-ou-variance; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-md/src/driver.rs::run_md; rust_test=ferric-md/tests/thermostats.rs::langevin_samples_the_canonical_harmonic_ensemble; registry=molecular-dynamics -->

Why BAOAB rather than another splitting? For the harmonic oscillator the
whole step is a linear map $z' = M z + \eta$ on $z = (x, v)$, with $M$
the product of the five matrices above and the noise $\eta$ entering at O and
carried through A and B. The stationary covariance $S$ solves
$S = M S M^{\mathsf T} + N$. Solving the three linear equations for
$(S_{xx}, S_{xv}, S_{vv})$ gives

$$ \langle x^2\rangle = \frac{k_B T_0}{k}, \qquad \langle xv\rangle = 0, \qquad \langle v^2\rangle = \frac{k_B T_0}{m}\Big(1 - \frac{\omega^2\Delta t^2}{4}\Big). \label{eq:molecular-dynamics-baoab-config} $$

The configurational variance is exact at every time step and every
friction: BAOAB samples the correct position distribution of a harmonic well
with no $\Delta t$ error. The velocity variance carries the factor
$1 - \omega^2\Delta t^2/4$, which is the same factor as in the shadow
Hamiltonian (@eq:molecular-dynamics-shadow), so the kinetic temperature of a
BAOAB run is low by $O(\Delta t^2)$ while the configurational temperature is
not. Use configurational quantities (potential energy, bond lengths) for sampling.
For the O–H stretch of water (3657 cm$^{-1}$, period 9.1 fs) the factor
$\omega^2\Delta t^2/4$ puts the kinetic energy of that mode 12% low at a
1 fs step and 3% low at 0.5 fs; softer modes are affected less. For an anharmonic surface the exactness is lost and the position
error is $O(\Delta t^2)$ (literature result, not proved here).

> **Verified equation @eq:molecular-dynamics-baoab-config** · BAOAB is exact in the configurational variance of a harmonic well
>
> The stationary covariance of the BAOAB step on $V = \tfrac12kx^2$ is
> (@eq:molecular-dynamics-baoab-config) for every $\Delta t$ and $\gamma$.
> An independent Lyapunov series $\sum_n M^n N M^{n\mathsf T}$ in 40-digit
> arithmetic reproduces it to $10^{-25}$.
>
> | | |
> |---|---|
> | Proof | `scripts/md_integrators_sympy.py`, claim baoab-harmonic-config |
> | Negative control | the friction factor $c$ replaced by $c^2$ in the matrix |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-md/src/driver.rs::run_md` |
>
> <!-- verified: id=md.baoab.harmonic-config; script=scripts/md_integrators_sympy.py; claim=baoab-harmonic-config; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-md/src/driver.rs::run_md; rust_test=ferric-md/tests/thermostats.rs::langevin_samples_the_canonical_harmonic_ensemble; registry=molecular-dynamics -->

## The conserved quantity {#molecular-dynamics-conserved}

How do you know a run is correct? In NVE, $E_{\mathrm{pot}} + E_{\mathrm{kin}}$
is the check: it fluctuates inside the $O(\Delta t^2)$ band of
(@eq:molecular-dynamics-shadow) and must not drift. With a thermostat the total
energy changes by design. The driver bookkeeps the kinetic energy each
thermostat step injects, $Q$: the change of $K$ across the Berendsen rescaling
and across the O step of BAOAB (the half kicks and drifts are conservative and are
not counted). Then

$$ E_{\mathrm{cons}} = E_{\mathrm{pot}} + E_{\mathrm{kin}} - Q \label{eq:molecular-dynamics-conserved} $$

is constant up to the integrator's error, because each thermostat step moves
$E_{\mathrm{kin}}$ by exactly the amount added to $Q$. It is the
quantity that exposes a too-large time step or an unconverged SCF under a
thermostat. Each recorded frame holds `potential`, `kinetic`, `total`
(without $Q$), `conserved` and `temperature_k`; `MdResult::max_conserved_drift`
is the largest departure of `conserved` from its first value. In NVE
$Q = 0$ and the two coincide. This identity is bookkeeping, not
a derived result, so no sympy claim covers it; the Rust conservation tests
do.

## The code {#molecular-dynamics-code}

The Langevin step of `ferric-md/src/driver.rs::run_md` is the method
`baoab`, unchanged:

<!-- excerpt: ferric-md/src/driver.rs::baoab -->
```rust
        self.kick(0.5 * self.dt);
        self.drift(0.5 * self.dt);
        self.ornstein_uhlenbeck(gamma, t0);
        self.drift(0.5 * self.dt);
        self.force(provider, step)?;
        self.kick(0.5 * self.dt);
        Ok(())
```

and its O step is `ornstein_uhlenbeck`, unchanged:

<!-- excerpt: ferric-md/src/driver.rs::ornstein_uhlenbeck -->
```rust
        let ke = kinetic_energy(&self.masses, &self.v);
        let c = (-gamma * self.dt).exp();
        let kt = BOLTZMANN_HARTREE_PER_K * t0;
        for (mut v, &m) in self.v.rows_mut().into_iter().zip(&self.masses) {
            let sigma = ((1.0 - c * c) * kt / m).sqrt();
            for vk in v.iter_mut() {
                *vk = c * *vk + sigma * self.rng.next_gaussian();
            }
        }
        self.q += kinetic_energy(&self.masses, &self.v) - ke;
```

- The six calls of `baoab` are steps 1 to 5 of section
  @sec:molecular-dynamics-baoab: `kick(h)` is $\mathbf v \leftarrow \mathbf v - h\,\mathbf g/m$
  and `drift(h)` is $\mathbf R \leftarrow \mathbf R + h\,\mathbf v$. The
  opening kick uses the gradient the previous step's `force` stored, which
  is why a step costs one gradient. The velocity-Verlet method `verlet` is
  the same sequence without the O step and with one full drift.
- `c` is $e^{-\gamma\Delta t}$ and `sigma` is $\sqrt{(1 - c^2)k_BT_0/m}$ of
  (@eq:molecular-dynamics-ou), per atom; `next_gaussian` is the Box–Muller
  draw of `Xoshiro256pp`. `gamma` arrives in inverse atomic units of time and
  `self.dt` in atomic units (section @sec:molecular-dynamics-units).
- The last line is the bookkeeping of (@eq:molecular-dynamics-conserved):
  `self.q` accumulates the change of $K$ across the O step. The Berendsen
  rescale (`berendsen`) ends with the same line, and the kicks and drifts
  never touch `q`.

## Running it {#molecular-dynamics-running}

The CLI reads `[md]` with `method.task = "md"`; `run_aimd` and `run_ffmd`
take the same settings as keyword arguments (`crates/ferric-cli/src/config.rs::MdCfg`,
`ferric-md/src/config.rs::MdConfig`).

| Key | Default | Meaning |
|---|---|---|
| `n_steps` | required | number of time steps |
| `dt_fs` | 0.5 (`run_ffmd`: 1.0) | time step, fs |
| `thermostat` | `"nve"` | `"nve"`, `"berendsen"` (needs `tau_fs` and `temperature_k`) or `"langevin"` (needs `friction_per_ps` and `temperature_k`) |
| `temperature_k` | unset | target temperature $T_0$, K |
| `init_temperature_k` | `temperature_k` | Maxwell–Boltzmann temperature of the initial velocities; with neither set the run starts from rest |
| `tau_fs`, `friction_per_ps` | unset | $\tau$ and $\gamma$ of the chosen thermostat; setting one for the other thermostat is an error |
| `seed` | 0 | seed of the initial velocities and the Langevin noise |
| `remove_com_motion` | true | remove the centre-of-mass velocity (section @sec:molecular-dynamics-initialisation) |
| `stride` | 1 | record every `stride`-th step |
| `scf_warm_start` | true | start each SCF from the previous step's orbitals (AIMD only) |
| `trajectory`, `energies` | `<input stem>.md.xyz`, `.md.csv` | output files (Python: `trajectory_path`, `energies_path`) |

`MdConfig` also has `initial_velocities` (Rust only) and `keep_trajectory`
(default true), which the CLI and Python do not expose.

The factor-4 band of section @sec:molecular-dynamics-velocity-verlet is
visible on water. NVE in cc-pVDZ from 300 K, the same 10 fs at two time
steps, with the SCF converged as the AIMD tests converge it, from the
`3f612e0b3` build (about 10 s):

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
for dt in (0.5, 0.25):
    r = ferric.run_aimd(mol, "cc-pvdz", n_steps=int(10 / dt), dt_fs=dt,
                        init_temperature_k=300.0, seed=1,
                        scf_density_conv=1e-10, scf_energy_conv=1e-12)
    e = r.total_energies
    print(f"dt = {dt:4.2f} fs  steps = {len(e) - 1:3d}  band of E_pot + E_kin = {e.max() - e.min():.3e} Ha")
```

```text
dt = 0.50 fs  steps =  20  band of E_pot + E_kin = 7.543e-06 Ha
dt = 0.25 fs  steps =  40  band of E_pot + E_kin = 1.886e-06 Ha
```

Halving the step shrinks the band by a factor of 4.00, the $\Delta t^2$ of
(@eq:molecular-dynamics-shadow).

## What is proven and what is not {#molecular-dynamics-proven}

The sympy script proves the integrator algebra named in the boxes above: the
area preservation, time reversal and harmonic shadow Hamiltonian of velocity
Verlet, the variance fixed point of the OU step, the exact configurational
variance of BAOAB on a harmonic well, and the first-order rate of the
Berendsen factor. Each has a numerical cross-check in 40-digit arithmetic and a
recorded mutation that makes the script fail.

It does not prove that `ferric-md/src/driver.rs::run_md` implements the
steps as written. That is the job of the Rust tests of the `ferric-md` crate, which
compare the driver with the discrete harmonic solution, check the energy drift
of anharmonic systems, and measure the mean temperature and $\langle x^2\rangle$ of Langevin
runs. The general-potential shadow series, the non-canonical ensemble of the
Berendsen thermostat and the $O(\Delta t^2)$ position error of BAOAB beyond the
harmonic well are literature statements, not checked here.

No grade covers a trajectory. The validation page lists molecular dynamics
in a table outside the `method.kind` matrix, and appendix @sec:appendix-c
grades no `task = "md"`. The certificate of a run, from the CLI or from
`MdResult.certificate`, has capability `md:<kind>` or `md:<label>`, the
grade `ungraded`, the integrator, thermostat, time step and seed as
approximations, and `MdResult::max_conserved_drift` as a warning. For an
AIMD run the SCF at the final geometry is folded in as the reference.

## Known limits (2026-11-19) {#molecular-dynamics-known-limits}

- Velocity Verlet, Berendsen and BAOAB only. No Nosé–Hoover chains, no
  barostat, no constraints (a rigid or SHAKE-constrained water needs a different
  step), no periodic boundary conditions.
- Masses are tabulated for $Z \le 54$ (`ferric_core::elements::atomic_mass`);
  heavier elements and isotopes need explicit `masses_amu`.
- The time step is the user's responsibility: the stability bound
  $\omega\Delta t < 2$ and the energy-band scaling above say what to expect,
  but the code does not choose it.
- Hybrid QM/MM dynamics and xTB dynamics are not implemented.

## Sources {#molecular-dynamics-sources}

| Source | Used for |
|---|---|
| `crates/ferric-md/src/driver.rs`, `crates/ferric-md/src/config.rs`, `crates/ferric-md/src/velocities.rs`, `crates/ferric-md/src/rng.rs` | the integrators, thermostats, velocities, random numbers |
| `scripts/md_integrators_sympy.py` | verified boxes |
| `crates/ferric-cli/src/config.rs` (`MdCfg`), `crates/ferric-cli/src/md.rs`, `crates/ferric-python/python/ferric/ferric.pyi` (`run_aimd`, `run_ffmd`, `MdResult`) | the `[md]` keys, defaults and the certificate |
| `crates/ferric-scf/tests/md_aimd.rs` | the SCF thresholds of the AIMD tests |
