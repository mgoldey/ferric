# CAS-CI, CASSCF and NEVPT2 {#multireference}

Hartree–Fock and every method built on it in the earlier chapters start
from one determinant. That fails when two or more determinants are equally
important, as for a bond pulled apart, a diradical or a transition-metal
centre with near-degenerate $d$ orbitals. This chapter covers the methods
in `crates/ferric-ci` that handle it. CAS-CI diagonalizes $H$ over every
determinant of a small active space. CASSCF also optimizes the orbitals for
that wavefunction. Strongly contracted NEVPT2 adds the correlation of the
electrons outside the active space by perturbation theory.

The sections follow the code in order:

1. bit-string determinants and their phase convention (@sec:determinants);
2. the Slater–Condon rules that the direct sigma is tested against (@sec:slater-condon);
3. the direct sigma product and its Davidson solver (@sec:direct-sigma-davidson);
4. the spin-summed RDMs and the higher RDMs of NEVPT2 (@sec:rdms);
5. the CASSCF orbital gradient and step (@sec:casscf);
6. the SC-NEVPT2 class energies (@sec:nevpt2).

Every method here starts from a closed-shell RHF with exact $J$ and $K$, and
uses one restricted orbital set: $n_c$ doubly occupied core orbitals
$i, j$, $n_a$ active orbitals $t, u, v, w$ and virtual orbitals $a$, with
$p, q, r, s$ over all of them. Determinants put every α spin orbital before
every β one, and that order fixes every phase below. The terse reference
for the equations is `crates/ferric-ci/theory.md`. The second-quantized
operators are those of chapter @sec:second-quantization, and the orbital-rotation machinery of
CASSCF is the one chapter @sec:convergence builds for SCF.
