# Level shift, MOM and smearing {#level-shift-mom-smearing}

<!-- status: ready; registry: rhf, uhf, rohf -->

Some molecules defeat DIIS in one of three ways. With a small HOMO–LUMO gap
each diagonalization rotates the orbitals too far, and the iteration
oscillates or diverges. Two orbitals of similar energy can trade places from
one iteration to the next, so aufbau occupies a different orbital each time.
And when the frontier orbitals are degenerate, integer occupations make the
density jump by a whole orbital whenever their order flips.

Three devices address these. A level shift damps the step. The
maximum-overlap method (MOM) keeps the occupied set continuous from one
iteration to the next. Fermi smearing gives fractional occupations near a
degenerate frontier. The plan: derive the shift as a damped step, read the
code that applies it, then define MOM and smearing.

| Device | Option | Changes the converged answer? | Solvers |
|---|---|---|---|
| level shift | `[scf] level_shift` (Ha) | no: ramped off, the convergence test uses the unshifted $F$ | RHF, UHF, ROHF |
| MOM | `[scf] mom_after_iter`, `mom_reference` | yes: it selects the state | RHF, UHF, ROHF |
| Fermi smearing | `[scf] smearing_sigma` (Ha) | yes: fractional occupations | RHF only |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/driver.rs::{effective_level_shift,auto_level_shift_applies}`, `ferric-scf/src/mom.rs::mom_reorder`, `ferric-scf/src/smearing.rs::{fermi_dirac,solve_fermi_level}` |
| CLI | the options above |
| Python | `level_shift=`, `mom_after_iter=`, `smearing_sigma=` on the SCF entry points |

## The level shift damps the step {#level-shift-mom-smearing-level-shift-damps-step}

### What one Roothaan step does

Work in the MO basis of the previous orbitals, and suppose their
occupied–occupied and virtual–virtual Fock blocks are nearly diagonal, with
diagonal elements $\varepsilon_p$. Diagonalizing $f$ mixes virtual $a$ into
occupied $i$. To first order in the off-diagonal element $f_{ai}$ the new
occupied orbital is

$$ \phi_i' = \phi_i + \kappa_{ai}\phi_a, \qquad \kappa_{ai} = -\frac{f_{ai}}{\varepsilon_a - \varepsilon_i} . $$

That is a step $\kappa = -g/\Delta$ with gradient $g = f_{ai}$ and
denominator $\Delta = \varepsilon_a - \varepsilon_i$, the gap. Section @sec:second-order
writes the orbital Hessian in the same scale: its diagonal is $\Delta$ plus
two-electron terms. A Roothaan step is therefore a Newton step with a
diagonal Hessian that keeps only the orbital-energy gaps and leaves out the
two-electron terms.

Follow one rotation through the iteration. Near the solution the gradient
is linear in the rotation, $g = H\kappa$, with $H$ the curvature along it.
One Roothaan step maps $\kappa_n$ to

$$ \kappa_{n+1} = \kappa_n - \frac{H\kappa_n}{\Delta} = \Big(1 - \frac{H}{\Delta}\Big)\kappa_n . $$

For $0 < H < \Delta$ the error shrinks without changing sign. For
$\Delta < H < 2\Delta$ it shrinks while flipping sign every step: the
density oscillates. For $H > 2\Delta$ it grows: the iteration diverges. The
Coulomb part of $H$, $4(ai|bj)$, is positive semidefinite, because
$\sum_{ai,bj}\kappa_{ai}\,4(ai|bj)\,\kappa_{bj} = 4(\rho_\kappa|\rho_\kappa) \ge 0$
for the density change $\rho_\kappa = \sum_{ai}\kappa_{ai}\phi_a\phi_i$. It
raises $H$ above $\Delta$. A small gap with the same Coulomb term makes the
ratio $H/\Delta$ large.

### Adding the shift

After extrapolation the solvers add a multiple of a projector onto the
previous virtual orbitals $C_v$:

$$ \bar F \leftarrow \bar F + \mu_{\rm eff}\,SC_vC_v^{\mathsf T}S . $$

Transform to the MO basis of the previous orbitals. Because
$C^{\mathsf T}SC = 1$, the product $C^{\mathsf T}SC_v$ is the identity on the
virtual columns and zero on the occupied ones:

$$ C^{\mathsf T}SC_vC_v^{\mathsf T}SC = (C^{\mathsf T}SC_v)(C_v^{\mathsf T}SC) = \mathrm{diag}(0_o, 1_v) . $$

The shift adds $\mu_{\rm eff}$ to the virtual–virtual block and nothing
else. The occupied–virtual coupling $f_{ai}$ is unchanged, and the gap in
the first-order step grows:

$$ \kappa_{ai} = -\frac{f_{ai}}{\varepsilon_a - \varepsilon_i + \mu_{\rm eff}}, \qquad \kappa_{n+1} = \Big(1 - \frac{H}{\Delta + \mu_{\rm eff}}\Big)\kappa_n . $$

For $H > 0$ any shift with $\Delta + \mu_{\rm eff} \ge H$ makes the factor lie
in $[0, 1)$, so the error shrinks without oscillating. The shifted step is
a Newton step with the diagonal Hessian $\Delta + \mu_{\rm eff}$, the same quantity
$\kappa^0 = -g/(\Delta + \mu)$ that the Newton solvers of section @sec:second-order start
from.

Check the limits. At $\mu_{\rm eff} = 0$ the step is Roothaan's. As
$\mu_{\rm eff} \to \infty$ the step goes to zero and the orbitals freeze. A
large shift therefore converges slowly, with a factor close to 1, which is
why the code ramps the shift down as the error falls (section @sec:level-shift-mom-smearing-shift-applied).

Does the shift change the answer? At self-consistency every $f_{ai}$ is
zero, so the shifted matrix is block diagonal in the same orbitals. Its
eigenvectors and its density are those of the unshifted $F$, with the
virtual energies raised by $\mu_{\rm eff}$. The energy and the convergence
test use the unshifted $F$.

> **Verified equation {#eq:scf-level-shift}** · the level shift acts on the virtual block only
>
> $\bar F + \mu SC_vC_v^{\mathsf T}S$ in the MO basis of $C$ is
> $\bar f + \mu\,\mathrm{diag}(0_o, 1_v)$: the occupied–virtual block is
> untouched, and a self-consistent solution of the shifted problem has the
> density of the unshifted one.
>
> | | |
> |---|---|
> | Proof | `scripts/scf_diis_sympy.py`, claim L1 |
> | Negative control | the projector $SC_vC_v^{\mathsf T}S \to SC_vC_v^{\mathsf T}$ must fail L1 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/driver.rs::effective_level_shift` |
>
> <!-- verified: id=scf.level-shift; script=scripts/scf_diis_sympy.py; claim=L1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/driver.rs::effective_level_shift; registry=rhf -->

The box proves the block structure and the unchanged density. The
first-order step and the one-rotation model above are a perturbation
argument, checked numerically for this section and not part of the proof.

## The shift that is applied {#level-shift-mom-smearing-shift-applied}

The value used is not always the option's value:

<!-- excerpt: ferric-scf/src/driver.rs::auto_level_shift_applies -->
```rust
config.meta_gga_auto_level_shift
        && config.level_shift == 0.0
        && crate::rohf::xc_is_metagga(config.xc.as_deref())
```

`effective_level_shift` returns 0.5 when this holds and `config.level_shift`
otherwise.

| Code | Meaning |
|---|---|
| `config.level_shift` | `[scf] level_shift`, default 0 |
| `meta_gga_auto_level_shift` | `true` unless a level shift was written; the CLI and Python set it to `false` whenever `level_shift` is given, so an explicit 0 turns the automatic shift off |
| `xc_is_metagga` | a meta-GGA functional (SCAN, r2SCAN, TPSS, ...) |
| `0.5` | the automatic shift for meta-GGAs, whose $\tau$ term makes plain DIIS limit-cycle |

That shift $\mu$ is then ramped by the current error, differently per solver:

| Solver | $\mu_{\rm eff}$ |
|---|---|
| RHF | $\mu\min(\max\lvert e\rvert/0.1,\ 1)$ |
| UHF, ROHF | $\mu\max\lvert e\rvert/(\max\lvert e\rvert + 10^{-3})$ |

Both ramps go to zero with the error, so the last iterations run unshifted
and take full steps. In RHF the shift is added inside `diis_step` in
`crates/ferric-scf/src/rhf.rs`; here it is with only a test probe removed:

```rust
if inp.effective_level_shift > 0.0 {
    if let Some(c_p) = self.c_prev.as_ref() {                   // previous orbitals C
        let shift_ramp = inp.effective_level_shift * (err_max / 0.1).min(1.0); // mu_eff
        if shift_ramp > 0.0 {
            let shift_term = with_blas_threads(opt_in_blas_threads(), || {
                let c_vir = c_p.slice(ndarray::s![.., nocc..]); // C_v
                let scv = s.dot(&c_vir);                        // S C_v
                shift_ramp * scv.dot(&scv.t())                  // mu_eff S C_v C_v^T S
            });
            f_new = &f_new + &shift_term;                       // added to the DIIS F
        }
    }
}
```

Three points about it:

- **The projector needs a previous $C$.** The shift starts at the second
  iteration. $C$ is kept across iterations only when a shift is in force,
  so an unshifted run does not pay for the copy.
- **The copy is taken after MOM.** `c_prev` is cloned after `mom_reorder`,
  so the next shift projects onto the virtuals of the occupation MOM kept,
  not those of the aufbau order MOM overrode. A test pins this on CO / 6-31G
  with MOM from iteration 1 and a shift of 0.5.
- **$SC_v$ is formed once.** The projector is $(SC_v)(SC_v)^{\mathsf T}$, one
  $n_{\rm bas}\times n_v$ product and one rank-$n_v$ update.

## MOM and Fermi smearing {#level-shift-mom-smearing-mom-fermi-smearing}

### The maximum-overlap method

Aufbau occupies the $n_o$ orbitals of lowest energy. When two orbitals have
nearly the same energy, which one is lower can change from one iteration to
the next, and the occupied set flips. MOM occupies instead the orbitals that
look most like the previous occupied set.

Take the previous occupied orbitals $C^{\rm old}_o$ and the new eigenvectors
$C$, and form their overlap $O = C_o^{{\rm old}\,\mathsf T}SC$. Score each new
orbital $j$ by

$$ p_j = \sum_iO_{ij}^2 , $$

the squared norm of its projection onto the old occupied space. Each new
orbital is normalized, so $0 \le p_j \le 1$. Summed over all $j$ the scores
give $n_o$: the new orbitals are a complete orthonormal set, so each old
occupied orbital contributes $\sum_jO_{ij}^2 = 1$. The $n_o$ highest scores
are occupied, ties going to the lower index. For ROHF the open shell is then
chosen the same way from the remaining orbitals. The closed-set scores, from `mom_order`, which
`ferric-scf/src/mom.rs::mom_reorder` calls:

```rust
let o_closed = c_prev_closed.t().dot(s).dot(c_new);      // O = C_o^old^T S C
(0..n)
    .map(|p| {
        let col = o_closed.column(p);
        col.iter().map(|x| x * x).sum::<f64>()           // p_j = sum_i O_ij^2
    })
    .collect::<Vec<f64>>()
```

The scores are then sorted in descending order with a stable sort, so the
virtuals keep their energy order. The reference is the previous iteration's
orbitals or, for UHF, optionally the initial guess
(`mom_reference = "initial"`). MOM starts after iteration `mom_after_iter`.
MOM pins whatever basin the SCF holds when it is armed: it stops the
occupied set flipping, and it does not steer toward the ground state. That
is why it changes the converged answer, and why it is armed only after DIIS
has settled.

### Fermi smearing

Near a degenerate frontier the density is a discontinuous function of the
orbital energies under aufbau. Smearing (`[scf] smearing_sigma`, RHF only,
`crates/ferric-scf/src/smearing.rs`) replaces the step function by a
Fermi–Dirac occupation:

$$f_p = \frac{1}{1 + e^{(\varepsilon_p - \mu_F)/\sigma}}, \qquad \sum_pf_p = N_{\rm el}/2, \qquad D = C\,\mathrm{diag}(2f)\,C^{\mathsf T},$$

with $\sigma$ in Hartree. Each $f_p$ increases with $\mu_F$, so $\sum_pf_p$
does too, and the Fermi level that holds $N_{\rm el}/2$ pairs is unique.
Bisection finds it:

```rust
let pad = 20.0 * sigma + 1.0;
let mut lo = e_min - pad;                                   // sum f ~ 0 below every eps_p
let mut hi = e_max + pad;                                   // sum f ~ n_orb above every eps_p
for _ in 0..200 {
    let mid = 0.5 * (lo + hi);
    let residual = count_electrons(eps, mid, sigma) - target; // sum_p f_p - N/2
    if residual > 0.0 {
        hi = mid;                                           // too many electrons: lower mu_F
    } else {
        lo = mid;
    }
    if (hi - lo) < 1e-14 {
        break;
    }
}
```

This is `ferric-scf/src/smearing.rs::solve_fermi_level` with the argument
checks and comments removed. The bracket is
$[\varepsilon_{\min} - (20\sigma+1), \varepsilon_{\max} + (20\sigma+1)]$ and the
loop stops at a width of $10^{-14}$. Halving a bracket 100 Ha wide down to
$10^{-14}$ takes 54 steps, well inside the cap of 200. `fermi_dirac` clamps
the exponent at $\pm40$: beyond it the occupation is set to exactly 0 or 1,
and $1/(1 + e^{40}) = 4.2\times10^{-18}$ is smaller than the spacing of
doubles near 1.

Check the limit: as $\sigma \to 0$ every orbital below $\mu_F$ gets $f_p \to 1$
and every orbital above gets $f_p \to 0$, which is aufbau.

> **Verified equation {#eq:scf-fermi-level}** · the Fermi level
>
> The bisection finds $\mu_F$ with $\sum_pf_p = N_{\rm el}/2$, and as
> $\sigma \to 0$ the occupations become aufbau.
>
> | | |
> |---|---|
> | Proof | `scripts/scf_diis_sympy.py`, claim F1 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/smearing.rs::{fermi_dirac,solve_fermi_level}` |
>
> <!-- verified: id=scf.fermi-level; script=scripts/scf_diis_sympy.py; claim=F1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/smearing.rs::{fermi_dirac,solve_fermi_level}; registry=rhf -->

### Smearing and the energy

The code computes the entropy
$S = -2\sum_p[f_p\ln f_p + (1-f_p)\ln(1-f_p)]$, but the returned energy of a
smeared run is $E[D]$ at the fractional density, not the Mermin free energy
$E - \sigma S$; the entropy is reported only in the trace output. A smeared
energy is therefore not variational in the occupations. In Python, a
gradient run with `smearing_sigma` is a `ValueError` before the SCF
(`crates/ferric-python/ferric.pyi`). The stability descent of section @sec:stability
skips a smeared run, because its integer-occupation Hessian does not
describe fractional occupations.

Smearing is RHF only, and so is `[scf] diis`. The CLI refuses both keys on
an open-shell run (`Config::validate_open_shell_scf_keys` in
`crates/ferric-cli/src/config.rs`), and the Python open-shell entry points
do not take them.

## Doc fixes needed {#level-shift-mom-smearing-doc-fixes-needed}

| Text | Code |
|---|---|
| the `mom_after_iter` doc counts DIIS iterations | the code counts SCF iterations: the `iter > config.mom_after_iter` test in `solve_rhf` |
| the ramp docs state one formula for all solvers | the two formulas of the ramp table above |

## Sources

| Source | Used for |
|---|---|
| `scripts/scf_diis_sympy.py` | verified equations @eq:scf-level-shift and @eq:scf-fermi-level |
| `crates/ferric-scf/src/mom.rs`, `crates/ferric-scf/src/smearing.rs` | MOM and the Fermi level |
