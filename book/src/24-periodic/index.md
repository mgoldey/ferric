# 24. Gaussian-basis periodic calculations

Three-dimensional periodic Hartree–Fock, Kohn–Sham, MP2 and dRPA on Gaussian
basis functions, in the conventions of PySCF `pyscf.pbc`. Energies are per
cell, in Hartree; lengths are in Bohr.

The chapter follows the data. Section 24.1 sums the nuclear repulsion and
defines the Madelung constant. Section 24.2 gives the Fourier transform of a
lattice-summed Gaussian pair, which every reciprocal-space term uses.
Section 24.3 builds the nuclear attraction from it, and 24.4 treats the
divergence that the exchange energy inherits from the dropped $\mathbf G = 0$
term. Section 24.5 replaces the dense Coulomb tensor by a range-separated
density fit, and 24.6 runs the SCF on a $k$-point mesh. Sections 24.7 and
24.8 differentiate the energy with respect to atoms and to strain, and 24.9
adds MP2 and dRPA on the periodic reference.

Notation. Lattice rows $\mathbf a_i$, cell volume $\Omega$, translations
$\mathbf L$ and $\mathbf T$, reciprocal vectors $\mathbf G$, mesh points
$k$, momentum transfer $q = k' - k$. The Ewald splitting parameter is
$\eta$; the code and the input call it `omega` (`[cell] omega`,
`gdf_omega`), and it is not the range-separation parameter $\omega$ of
chapter 4. The imaginary frequency is $u$. The densities follow chapter 0:
$D = 2C_oC_o^{\mathsf H}$ and $F = h + J - \tfrac12K$.

Every derivation lives in `crates/ferric-pbc/theory.md`; each section
includes its part and adds the algorithm, the code and the anchors. The user
guide is `site/src/methods/pbc.md`.

**Known limits (2026-11-15).** Three-dimensional periodicity only: no slabs,
wires or two-dimensional Coulomb truncation. A charged cell is an error. The
lattice is never relaxed: `task = "optimize"` moves atoms at a fixed lattice,
and `task = "frequencies"` is an error. No metals or fractional occupation:
a closed HOMO–LUMO gap on the mesh is an error.
