# Gaussian-basis periodic calculations {#periodic}

A crystal is one cell repeated without end, and every energy of a molecular
code becomes a sum over that repetition. Three of those sums need care.
The Coulomb energy of an infinite neutral lattice converges only
conditionally, so its value depends on the order of summation until a
convention fixes it. The exchange kernel $4\pi/G^2$ is infinite at
$\mathbf G = 0$, and dropping that point leaves an error that falls only as
the inverse edge of the cell, or of the supercell a $k$ mesh represents. And Bloch's theorem turns one infinite
problem into one finite problem per crystal momentum $k$. This chapter
builds three-dimensional periodic Hartree–Fock, Kohn–Sham, MP2 and dRPA on
Gaussian basis functions around those three facts, in the conventions of
PySCF `pyscf.pbc`. Energies are per cell, in Hartree; lengths are in Bohr.

The chapter follows the data. Section @sec:ewald sums the nuclear repulsion and
defines the Madelung constant. Section @sec:pair-ft gives the Fourier transform of a
lattice-summed Gaussian pair, which every reciprocal-space term uses.
Section @sec:nuclear-attraction builds the nuclear attraction from it, and section @sec:exchange-divergence treats the
divergence that the exchange energy inherits from the dropped $\mathbf G = 0$
term. Section @sec:rsgdf replaces the dense Coulomb tensor by a range-separated
density fit, and section @sec:kpoint-scf runs the SCF on a $k$-point mesh. Sections @sec:forces and
@sec:stress differentiate the energy with respect to atoms and to strain, and
section @sec:mp2-drpa adds MP2 and dRPA on the periodic reference.

Notation. Lattice rows $\mathbf a_i$, cell volume $\Omega$ (in this chapter
$\Omega$ is always the cell volume, never an excitation energy), translations
$\mathbf L$ and $\mathbf T$, reciprocal vectors $\mathbf G$, mesh points
$k$, momentum transfer $q = k' - k$; $k$ is always a mesh point. Auxiliary rows
and kept metric eigenvectors are indexed by $P$, $Q$ and $\kappa$. The Ewald splitting parameter is
$\eta$; the code and the input call it `omega` (`[cell] omega`,
`gdf_omega`), and it is not the range-separation parameter $\omega$ of
chapter @sec:operators. The imaginary frequency is $u$. The densities follow chapter @sec:notation:
$D = 2C_oC_o^{\mathsf H}$ and $F = h + J - \tfrac12K$.

Each section derives its equations, checks them in a limit (independence of
the Ewald split, the Gamma point, the large cell that approaches the
molecule) and then shows the code and the tests. The user guide is
`site/src/methods/pbc.md`.

**Known limits (2026-11-15).** Three-dimensional periodicity only: no slabs,
wires or two-dimensional Coulomb truncation. A charged cell is an error. The
lattice is never relaxed: `task = "optimize"` moves atoms at a fixed lattice,
and `task = "frequencies"` is an error. No metals or fractional occupation:
a closed HOMO–LUMO gap on the mesh is an error.

What each route computes (2026-11-19; the per-functional limits are in the
input reference, `site/src/reference/input.md`, `[cell]` section):

| Route | Gamma | mesh | `jk` | Force | Stress |
|---|---|---|---|---|---|
| RHF, UHF, ROHF | yes | yes | `dense`, `rsgdf` | yes | yes |
| RKS, UKS, ROKS, LDA/GGA/global hybrid | yes | yes | `dense`, `rsgdf` | yes | yes |
| RKS, UKS, ROKS, range-separated hybrid | yes | per route | `dense`, `rsgdf` | HSE06, CAM-B3LYP | per route |
| RKS, UKS, ROKS, SCAN/r2SCAN/TPSS | per route | per route | per route | per route | per route |
| MP2, dRPA (RHF reference) | yes | yes | per route | no | no |

"Per route" means the input reference lists which spin route and builder
runs; anything it does not list is an error, not a fallback. The Gamma-point
stress is Python-only (`with_stress=True`); the CLI `[cell] stress` needs a mesh.
