# Avogadro plugin

[Avogadro 2](https://avogadro.cc) can run ferric from its menus through the
`avogadro-ferric` package plugin, which lives in this repository at
`plugins/avogadro-ferric/`. You need Avogadro 2.1.0 or later on Linux x86_64,
because ferric publishes wheels for that platform only. The plugin installs
ferric 0.1.0rc11 or newer from PyPI by itself.

## What it adds

| Avogadro feature | What it does |
|---|---|
| Energy model **ferric (QM)** (Extensions > Calculate, and the Auto Optimize tool) | SCF energy and analytic gradient (RHF, UHF, RKS, UKS), so Avogadro's own optimizer and force display run on the QM surface |
| Charge models **ferric Lowdin / Hirshfeld / CHELPG** | Partial charges and the full QM electrostatic potential, for charge colouring and ESP surfaces |
| Input generator **ferric** (Input > ferric) | A ferric CLI input, `job.toml` plus `job.xyz` |
| Extensions > ferric > Single Point | Total energy; partial charges stored on the molecule |
| Extensions > ferric > Optimize Geometry | Geometry relaxed by ferric's optimizer |
| Extensions > ferric > Frequencies | Harmonic modes, animated in the Vibrations panel |
| Extensions > ferric > Molecular Orbitals | Basis set and MO coefficients, so Avogadro draws orbitals |
| Extensions > ferric > Configure… | Default method, basis, charge model, threads and memory |

Each menu command opens a dialog for the method (HF, LDA, PBE, BLYP, BP86,
PBE0, B3LYP or r2SCAN) and the basis set. Charge and multiplicity come from
the molecule; multiplicity above 1 runs unrestricted.

## Install

In Avogadro, open Extensions > Plugins > Manage Plugins… and install
`ferric` from the online catalog. Avogadro runs `pixi install` in the plugin
directory, and that installs ferric from PyPI. If your Avogadro's catalog does
not list it yet, install it by hand.

To install by hand, copy `plugins/avogadro-ferric` into Avogadro's plugin
directory and run `pixi install` there:

- native Avogadro: `~/.local/share/OpenChemistry/Avogadro/plugins/`
- Flathub Avogadro:
  `~/.var/app/org.openchemistry.Avogadro2/data/OpenChemistry/Avogadro/plugins/`
  (run pixi as `flatpak run --command=pixi org.openchemistry.Avogadro2 install --manifest-path <plugin>/pyproject.toml`)

The plugin README covers using a local ferric build instead of PyPI.

## Limits

- **Time limits set by Avogadro.** Avogadro waits 30 s for each energy and
  gradient during interactive optimization, and kills charge and potential
  calculations after 10 s. Use a small basis for these. The ferric menu
  commands have no time limit.
- **Closed shell only** for charges, the ESP and orbitals. Open-shell molecules
  get an energy from Single Point and an error for the rest.
- Elements H to Kr. No unit cells.

## How it was checked

The plugin's tests drive it the way Avogadro does: one process per call, JSON
on stdin. They check the gradient against finite differences of the reported
energy, the energy and ESP units, the frequencies and modes against ferric's
own, and the exported orbitals, evaluated by a port of Avogadro's evaluator,
against `ferric.eval_mo_on_points` to 1e-10. They pass against the ferric
0.1.0rc11 wheel from PyPI.

Inside Avogadro 2.1.0 (Flathub), the energy model's energies and forces
matched ferric's to 1e-10 kJ/mol, and Avogadro's optimizer on the ferric
surface reached ferric's own minimum to 3e-5 Å. The menu commands and orbital
rendering were checked against the avogadrolibs source rather than in the
GUI. The plugin README has the details, the units and conventions, and a
five-minute manual check.
