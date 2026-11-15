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
| Rust | `ferric-scf/src/driver.rs::{effective_level_shift,auto_level_shift_applies}`, `ferric-scf/src/mom.rs::mom_reorder`, `ferric-scf/src/smearing.rs::{fermi_dirac,solve_fermi_level}` |
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

## 8.2.3 Smearing and the energy

The returned energy of a smeared run is $E[D]$ at the fractional density, not
the Mermin free energy $E - \sigma S$; the entropy is computed but reported
only in the trace output. A smeared energy is therefore not variational in
the occupations. In Python, a gradient run with `smearing_sigma` is a
`ValueError` before the SCF (`crates/ferric-python/ferric.pyi`).

## 8.2.4 Code issues found

Three findings of the first draft are fixed (2026-11-15):

| Finding | Fix |
|---|---|
| Under MOM, the RHF level shift projected onto the pre-MOM virtual block | `c_prev` is now cloned after `mom_reorder`, so the shift follows the occupation MOM kept; pinned on CO / 6-31G (MOM from iteration 1, shift 0.5) |
| An explicit `level_shift = 0` could not turn off the meta-GGA shift | the `meta_gga_auto_level_shift` flag above |
| UHF and ROHF ignored `[scf] diis` and `smearing_sigma` without a warning | the CLI refuses both keys on an open-shell run (`Config::validate_open_shell_scf_keys`); the Python open-shell entry points do not take them |

Still open, reported, not changed here:

| Finding | Evidence |
|---|---|
| the `mom_after_iter` doc counts DIIS iterations; the code counts SCF iterations | the `iter > config.mom_after_iter` test in `solve_rhf` |
| the ramp docs state one formula for all solvers | the table above |

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "Level shift, MOM and Fermi smearing" | the equations |
| `scripts/scf_diis_sympy.py` | verified equations 8.3 and 8.4 |
| `crates/ferric-scf/src/mom.rs`, `crates/ferric-scf/src/smearing.rs` | MOM and the Fermi level |
