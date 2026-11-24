# Molecular dynamics

ferric propagates nuclei classically on two surfaces: the SCF surface
(Born-Oppenheimer AIMD on RHF, UHF, ROHF or their Kohn-Sham variants) and the
AMBER-form force field of an `MmTopology` (FFMD). Both use the integrators of
the `ferric-md` crate: velocity Verlet (NVE), Berendsen rescaling and Langevin
BAOAB. The derivations are in
[`book/src/28-molecular-dynamics/`](https://github.com/mgoldey/ferric/tree/main/book/src/28-molecular-dynamics).

| Surface | Python | CLI |
|---|---|---|
| SCF / Kohn-Sham | `ferric.run_aimd` | `task = "md"` with `[md]` ([input reference](../reference/input.md#md)) |
| MM force field | `ferric.run_ffmd` | none |

Molecular dynamics has no validation grade. Every result carries an ungraded
`md:<surface>` certificate that records the integrator, thermostat, time step,
seed and the largest conserved-quantity drift.

## Units

| Quantity | Input | Output |
|---|---|---|
| Time step, time | `dt_fs`, fs | `times_fs`, fs |
| Temperature | `temperature_k`, `init_temperature_k`, K | `temperatures`, K |
| Thermostat | `tau_fs` (fs), `friction_per_ps` (ps⁻¹) | |
| Positions | XYZ or `coords_angstrom`, Å | `positions_angstrom`, Å |
| Velocities | | `velocities`, Å/fs |
| Energies | | Hartree |
| Masses | `masses_amu`, u | |

Internally everything is in atomic units. Masses default to the standard
atomic weights, which are tabulated up to xenon (Z = 54); a heavier element
needs `masses_amu`. Ghost atoms are refused.

## AIMD from Python

<!-- doctest -->
```python
import ferric

h2 = ferric.Molecule.from_xyz_string("2\n\nH 0 0 0\nH 0 0 0.7414\n")
r = ferric.run_aimd(
    h2, "sto-3g", n_steps=20, dt_fs=0.2, init_temperature_k=300.0,
    scf_density_conv=1e-10, scf_energy_conv=1e-10,
)
print(r.label, r.n_dof, len(r.times_fs))
print(f"{r.max_conserved_drift:.1e}")
```

```text
rhf/sto-3g 3 21
8.4e-06
```

`run_aimd` takes the SCF kwargs of `run_optimize` (`reference`, `xc`,
`dispersion`, `df_j_aux`, `df_k_aux`, `scf_max_iter`, `scf_energy_conv`,
`scf_density_conv`) and the MD kwargs below. `MdResult` holds one entry per
recorded frame: `times_fs`, `potential_energies`, `kinetic_energies`,
`total_energies`, `conserved`, `temperatures` (1-D arrays),
`positions_angstrom` and `velocities` (`(n_frames, n_atoms, 3)`), `n_dof`,
`max_conserved_drift`, `scf_iterations`, `certificate` and `mol()` (the final
geometry). `trajectory_path=` and `energies_path=` write the XYZ and CSV
files the CLI writes.

| Kwarg | Default | Meaning |
|---|---|---|
| `n_steps` | required | Number of steps; `n_steps + 1` energy and gradient evaluations |
| `dt_fs` | `0.5` (`run_ffmd`: `1.0`) | Time step |
| `thermostat` | `"nve"` | `"nve"`, `"berendsen"` or `"langevin"` |
| `temperature_k` | `None` | Target temperature; required by `berendsen` and `langevin` |
| `init_temperature_k` | `temperature_k` | Maxwell-Boltzmann initial velocities, rescaled to this temperature exactly; with neither set the atoms start at rest |
| `tau_fs` | `None` | Berendsen coupling time; only with `berendsen`, which requires it |
| `friction_per_ps` | `None` | Langevin friction; only with `langevin`, which requires it |
| `seed` | `0` | Seed of the initial velocities and the Langevin noise |
| `remove_com_motion` | `True` | Remove the centre-of-mass velocity of the initial velocities |
| `stride` | `1` | Record every `stride`-th step (step 0 always) |
| `scf_warm_start` | `True` | `run_aimd` only: start each SCF from the previous step's orbitals |
| `masses_amu` | standard masses | Per-atom masses |

A thermostat parameter that does not belong to the chosen thermostat raises
`ValueError`, as do a missing one, a non-positive `dt_fs`, a `stride` of 0 and
an unknown thermostat name. The checks run before the first SCF.

## AIMD from the CLI

```toml
[molecule]
xyz = "testdata/molecules/water.xyz"

[basis]
name = "sto-3g"

[method]
kind = "rhf"
task = "md"

[scf]
density_conv = 1e-9

[md]
n_steps = 20
dt_fs = 0.25
init_temperature_k = 300.0
```

This is `examples/water-rhf-md.toml`. The keys of `[md]` are the Python
kwargs, with `trajectory` and `energies` for the output paths (default
`<input stem>.md.xyz` and `<input stem>.md.csv` beside the input file). Each
recorded frame is printed as a row of step, time, potential and kinetic
energy, conserved quantity and temperature. The run ends with a summary, the
warm-start line, the certificate and an `md` record in the
[run log](./run-logs.md).

`task = "md"` runs for `kind = "rhf"`, `"ksdft"`, `"uhf"` and `"rohf"`; a
functional in `[dft]` makes the surface Kohn-Sham and `[dft] dispersion` adds
D3(BJ) or MBD@rsSCS to every step. `[cell]`, `[qmmm]`, `[cosmo]`, `[pcm]`,
`[scf] relativistic` and `[scf] symmetry` are refused with this task.

## FFMD from Python

`run_ffmd` takes an `MmTopology` (see [QM/MM](./qmmm.md)), the element
symbols and the coordinates in topology order. The topology has no masses, so
they come from the symbols unless `masses_amu` is given. `cutoff_angstrom`
switches the nonbonded sum to OpenMM's `CutoffNonPeriodic` (reaction-field
Coulomb, truncated Lennard-Jones); unset, every pair is summed.

<!-- doctest -->
```python
import numpy as np

top = ferric.MmTopology.from_amber_units(
    charges=[-0.834, 0.417, 0.417],
    sigmas_angstrom=[3.15061, 1.0, 1.0],
    epsilons_kcal=[0.1521, 0.0, 0.0],
    bonds=[(0, 1, 450.0, 0.9572), (0, 2, 450.0, 0.9572)],
    angles=[(1, 0, 2, 55.0, 104.52)],
    torsions=[],
)
xyz = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.24, 0.9266, 0.0]])
r = ferric.run_ffmd(
    top, ["O", "H", "H"], xyz, n_steps=1000, dt_fs=0.25,
    thermostat="langevin", friction_per_ps=10.0, temperature_k=300.0, seed=1,
)
print(r.label, r.n_dof, r.positions_angstrom.shape)
```

```text
mm 9 (1001, 3, 3)
```

FFMD is Python only.

## Choosing the thermostat

| Thermostat | Integrator | Ensemble | Use |
|---|---|---|---|
| `nve` | velocity Verlet | microcanonical | Production NVE; checking the time step |
| `berendsen` | velocity Verlet, then `v *= sqrt(1 + (dt/tau)(T0/T - 1))` | none | Bringing a structure to temperature; discard these frames |
| `langevin` | BAOAB | canonical | Sampling at a temperature |

Berendsen suppresses kinetic-energy fluctuations and does not sample the
canonical ensemble; use it for equilibration only. `tau_fs` of 100 fs is a
common choice; a `tau_fs` equal to `dt_fs` rescales to the target every step.
For Langevin, the friction sets how strongly the noise perturbs the dynamics:
1 to 10 ps⁻¹ samples the canonical ensemble with little damping of the
vibrations, larger values thermalize faster and slow diffusion. With
`friction_per_ps = 0` Langevin reduces to velocity Verlet.

The `conserved` column is the total energy minus the kinetic energy the
thermostat has added, so it is constant up to integration error for every
thermostat, and under NVE it equals the total energy.

Temperatures count `3N - 3` degrees of freedom for NVE and Berendsen when the
centre-of-mass motion is removed and N > 1, and `3N` otherwise. Langevin
always counts `3N`: its noise does not conserve momentum, so the
centre-of-mass motion is removed only from the initial velocities.

## Convergence

The velocity Verlet error in the conserved quantity grows as `dt²`. Measured
on water / RHF / STO-3G from 300 K over 10 fs (2026-11-19):

| `dt_fs` | Max \|conserved − conserved₀\| (Hartree) |
|---|---|
| 1.0 | 2.9e-4 |
| 0.5 | 8.4e-5 |
| 0.25 | 2.1e-5 |
| 0.125 | 5.3e-6 |

Over the same run the potential energy swings by 3.8e-3 Hartree. Choose
`dt_fs` so the conserved quantity fluctuates well below the energy changes of
interest; 0.5 fs or less resolves O-H and C-H stretches, and hydrogen is the
atom that limits the step. A conserved quantity that trends steadily, rather
than fluctuating, points at the gradient: an SCF converged too loosely, or a
surface whose gradient does not match its energy. Tighten `scf_density_conv`
(1e-9 in the example) before shrinking the step further.

`scf_warm_start` changes the SCF iteration count, not the trajectory beyond
the SCF tolerance. On H2 it lowers the count; on water / STO-3G (20 steps of
0.5 fs) it took 233 iterations against 213 cold, and on an open-shell OH UHF
run warm steps took 14 to 15 iterations against 10 cold. Compare
`scf_iterations` with `scf_warm_start=False` on a short run and keep the
cheaper setting.

## Reproducibility

The random number generator is a self-contained xoshiro256++ seeded from
`seed`, so the same input, seed and build give the same trajectory: the
initial velocities and every Langevin noise draw are fixed by `seed`. An NVE
or Berendsen run with no initial temperature does not draw from it.

## Other routes

The [ASE calculator](./ase.md) also runs molecular dynamics through ASE's own
integrators (`ase.md`), one ferric energy and gradient per step. The native
drivers above warm-start each SCF from the previous step, write the conserved
quantity and attach a certificate.
