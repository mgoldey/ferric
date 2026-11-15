# 17. CAS-CI, CASSCF and NEVPT2

Full configuration interaction in an active space, the two-step CASSCF that
optimizes the orbitals for it, and strongly contracted NEVPT2 on top. The
chapter follows the code in `crates/ferric-ci`: bit-string determinants and
their phase convention, the Slater–Condon rules, the direct sigma product
and its Davidson solver, the spin-summed RDMs, the CASSCF orbital gradient
and step, and the SC-NEVPT2 class energies.

Every method here starts from a closed-shell RHF with exact $J$ and $K$, and
uses one restricted orbital set: $n_c$ doubly occupied core orbitals
$i, j$, $n_a$ active orbitals $t, u, v, w$ and virtual orbitals $a$, with
$p, q, r, s$ over all of them. Determinants are α-before-β, which fixes
every phase below. The derivations are in `crates/ferric-ci/theory.md`,
included by heading.
