# 8.2 Level shift, MOM and smearing

<!-- section: 8.2; status: ready; registry: rhf, uhf, rohf;
     theory: crates/ferric-scf/theory.md#"Level shift, MOM and Fermi smearing" -->

Three devices change which orbitals the SCF step occupies or how far it
moves, for the molecules DIIS alone does not converge. A level shift damps
the step. The maximum-overlap method keeps the occupied set continuous from
one iteration to the next. Fermi smearing gives fractional occupations near
a degenerate frontier.

| Device | Option | Changes the converged answer? | Solvers |
|---|---|---|---|
| level shift | `[scf] level_shift` (Ha) | no: ramped off, the convergence test uses the unshifted $F$ | RHF, UHF, ROHF |
| MOM | `[scf] mom_after_iter`, `mom_reference` | yes: it selects the state | RHF, UHF, ROHF |
| Fermi smearing | `[scf] smearing_sigma` (Ha) | yes: fractional occupations | RHF only |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/driver.rs::effective_level_shift`, `ferric-scf/src/mom.rs::mom_reorder`, `ferric-scf/src/smearing.rs::{fermi_dirac,solve_fermi_level}` |
| CLI | the options above |
| Python | `level_shift=`, `mom_after_iter=`, `smearing_sigma=` on the SCF entry points |

## 8.2.1 The equations

<!-- include: crates/ferric-scf/theory.md#"Level shift, MOM and Fermi smearing" -->

> **Verified equation 8.3** · the level shift acts on the virtual block only
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

> **Verified equation 8.4** · the Fermi level
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

## 8.2.2 The shift that is applied

The value used is not always the option's value:

<!-- excerpt: ferric-scf/src/driver.rs::effective_level_shift -->
```rust
if config.level_shift == 0.0 && crate::rohf::xc_is_metagga(config.xc.as_deref()) {
        0.5
    } else {
        config.level_shift
    }
```

| Code | Meaning |
|---|---|
| `config.level_shift` | `[scf] level_shift`, default 0 |
| `xc_is_metagga` | a meta-GGA functional (SCAN, r2SCAN, TPSS, ...) |
| `0.5` | the automatic shift for meta-GGAs, whose $\tau$ term makes plain DIIS limit-cycle |

That shift $\mu$ is then ramped by the current error, differently per solver:

| Solver | $\mu_{\rm eff}$ |
|---|---|
| RHF | $\mu\min(\max\lvert e\rvert/0.1,\ 1)$ |
| UHF, ROHF | $\mu\max\lvert e\rvert/(\max\lvert e\rvert + 10^{-3})$ |

## 8.2.3 Smearing and the energy

The returned energy of a smeared run is $E[D]$ at the fractional density, not
the Mermin free energy $E - \sigma S$; the entropy is computed but reported
only in the trace output. A smeared energy is therefore not variational in
the occupations. In Python, a gradient run with `smearing_sigma` is a
`ValueError` before the SCF (`crates/ferric-python/ferric.pyi`).

## 8.2.4 Code issues found

These are reported, not changed here.

| Finding | Evidence |
|---|---|
| Under MOM, the RHF level shift projects onto the pre-MOM virtual block. `c_prev` is cloned from the eigenvectors before `mom_reorder` permutes them, so if MOM keeps an orbital above the aufbau frontier occupied, that orbital is shifted as a virtual. | `solve_rhf` in `crates/ferric-scf/src/rhf.rs`: the shift reads `c_p.slice(s![.., nocc..])`, `c_prev = Some(c.clone())` precedes the `mom_reorder` call |
| An explicit `level_shift = 0` cannot turn off the meta-GGA shift: 0 is the default and the sentinel | the excerpt above |
| UHF and ROHF ignore `[scf] diis` (`diis_flavor`) and `smearing_sigma` without a warning | the open-shell solvers in `crates/ferric-scf/src/uhf.rs`, `crates/ferric-scf/src/rohf.rs` |
| the `mom_after_iter` doc counts DIIS iterations; the code counts SCF iterations | the `iter > config.mom_after_iter` test in `solve_rhf` |
| the ramp docs state one formula for all solvers | the table above |

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "Level shift, MOM and Fermi smearing" | the equations |
| `scripts/scf_diis_sympy.py` | verified equations 8.3 and 8.4 |
| `crates/ferric-scf/src/mom.rs`, `crates/ferric-scf/src/smearing.rs` | MOM and the Fermi level |
