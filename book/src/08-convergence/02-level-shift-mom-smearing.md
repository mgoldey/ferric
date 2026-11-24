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
argument that no claim covers.

The same water / cc-pVDZ SCF with three shifts, from the `589b50832` build,
lands on the same energy to every printed digit; only the iteration count
moves:

```python
# Run from the repository root. The converged energy does not depend on the
# level shift; the iteration count does.
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")

for shift in (0.0, 0.5, 2.0):
    r = ferric.run_rhf(mol, bs, level_shift=shift, max_iter=200)
    print(f"level_shift={shift}: E = {r.energy:.10f}  converged={r.converged}  iterations={r.iterations}")
```

```text
level_shift=0.0: E = -76.0267679974  converged=True  iterations=10
level_shift=0.5: E = -76.0267679974  converged=True  iterations=10
level_shift=2.0: E = -76.0267679974  converged=True  iterations=11
```

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

That shift $\mu$ is then ramped by the current error. There are two ramps,
each one function in `ferric-scf/src/driver.rs`:

| Solvers | Function | $\mu_{\rm eff}$ |
|---|---|---|
| RHF, RKS | `closed_shell_level_shift_ramp` | $\mu\min(\max\lvert e\rvert/0.1,\ 1)$ |
| UHF, UKS, ROHF, ROKS, periodic Gamma and k-point ROKS | `open_shell_level_shift_ramp` | $\mu\max\lvert e\rvert/(\max\lvert e\rvert + 10^{-3})$ |

<!-- excerpt: ferric-scf/src/driver.rs::open_shell_level_shift_ramp -->
```rust
shift * err_max / (err_max + OPEN_SHELL_RAMP_ERR)
```

Both go to zero with the error, so the last iterations run unshifted and
take full steps, and the converged energy does not depend on either. They
differ in the tail: the linear ramp is down to 10% of the shift at
$\max\lvert e\rvert = 10^{-2}$, the rational one is still at 91%. Each was
measured in the other's place, with the energies equal to $10^{-10}$ Ha in
every case:

| System | Linear | Rational |
|---|---|---|
| water RHF/cc-pVDZ, $\mu = 2.0$ (the example above) | 11 | 16 |
| water RKS/SCAN, CH4 RKS/r2SCAN (automatic shift); COSe RHF, $\mu = 0.5$ | 10, 9; 16 | 11, 11; 17 |
| 17 molecular open-shell SCFs (NH2, CH3, OH, O2; UHF, UKS, ROHF, ROKS) | 0 to 7 fewer | the reference |
| Gamma ROKS PBE0, tri 4H triplet, six perturbed starts (`ferric-pbc/tests/pbc_rohf.rs`) | 1005 in total; 314 and 229 on two starts | 1145 in total; 177 and 172 on those |
| k-point ROKS PBE0, the same cell (`ferric-pbc/tests/pbc_kroks.rs`) | not converged in 200, $\max\lvert e\rvert$ parked at $7\times10^{-3}$ | 34 |

The closed-shell ramp is never slower on the closed shells measured. The
open-shell family includes the ROKS cell whose minimum repels the unshifted
Roothaan map, and there the shift has to survive into the tail: at
$\max\lvert e\rvert = 7\times10^{-3}$ the linear ramp keeps 7% of it and the
rational one 87%.

In RHF the shift is added inside `diis_step` in
`crates/ferric-scf/src/rhf.rs`; here it is with only a test probe removed:

```rust
if inp.effective_level_shift > 0.0 {
    if let Some(c_p) = self.c_prev.as_ref() {                   // previous orbitals C
        let shift_ramp =
            crate::driver::closed_shell_level_shift_ramp(inp.effective_level_shift, err_max); // mu_eff
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
chosen the same way from the remaining orbitals. The closed-set scores come
from `mom_order`, which `ferric-scf/src/mom.rs::mom_reorder` calls:

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
virtuals keep their energy order. MOM pins whatever basin the SCF holds when
it is armed: it stops the occupied set flipping, and it does not steer
toward the ground state. That is why it changes the converged answer. With
the previous iteration's orbitals as reference (the default) it is armed
after `mom_after_iter` SCF iterations, once DIIS has settled. With
`mom_reference = "initial"` (UHF and UKS; requires `mom_after_iter = 0`) it
holds the guess occupation from the first iteration, which is how an excited
configuration is targeted.

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
> | Negative control | the bisection branch reversed (raise the lower bound when the count is too high) must fail F1 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/smearing.rs::{fermi_dirac,solve_fermi_level}` |
>
> <!-- verified: id=scf.fermi-level; script=scripts/scf_diis_sympy.py; claim=F1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/smearing.rs::{fermi_dirac,solve_fermi_level}; registry=rhf -->

### Smearing and the energy

The code computes the entropy
$S = -2\sum_p[f_p\ln f_p + (1-f_p)\ln(1-f_p)]$, but the returned energy of a
smeared run is $E[D]$ at the fractional density, not the Mermin free energy
$E - \sigma S$; the entropy is reported only in the trace output. A smeared
energy is therefore not variational in the occupations. The Fermi–Dirac
occupations make the free energy $E - \sigma S$ stationary with respect to the
occupations (Mermin), not $E$ itself, so two smeared energies at different
$\sigma$ cannot be compared without the $-\sigma S$ term. In Python, a
gradient run with `smearing_sigma` is a `ValueError` before the SCF
(`crates/ferric-python/ferric.pyi`). The stability descent of section @sec:stability
skips a smeared run, because its integer-occupation Hessian does not
describe fractional occupations.

Smearing is RHF only, and so is `[scf] diis`. The CLI refuses both keys on
an open-shell run (`Config::validate_open_shell_scf_keys` in
`crates/ferric-cli/src/config.rs`), and the Python open-shell entry points
do not take them.

## Sources

| Source | Used for |
|---|---|
| `scripts/scf_diis_sympy.py` | verified equations @eq:scf-level-shift and @eq:scf-fermi-level |
| `crates/ferric-scf/src/mom.rs`, `crates/ferric-scf/src/smearing.rs` | MOM and the Fermi level |
| V. R. Saunders and I. H. Hillier, Int. J. Quantum Chem. 7, 699 (1973) | the level shift |
| A. T. B. Gilbert, N. A. Besley and P. M. W. Gill, J. Phys. Chem. A 112, 13164 (2008) | MOM |
| N. D. Mermin, Phys. Rev. 137, A1441 (1965); A. D. Rabuck and G. E. Scuseria, J. Chem. Phys. 110, 695 (1999) | the free energy and Fermi smearing in SCF |
