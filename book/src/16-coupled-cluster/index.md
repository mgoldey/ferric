# 16. Coupled cluster

CCD and CCSD in spin orbitals, the spin-adapted closed-shell CCSD that the
CLI and Python run, the perturbative triples in both forms, EOM-CCSD
singlet excitation energies, the linearized ladder method LinLCCD and the
ωB97X-L-V double hybrid built on it, the local forms of these methods, and
spin-blocked CCSD and CCSD(T) on a UHF reference.

Every solver reads canonical HF orbitals and RI integrals
$(pq|rs) = \sum_P B^P_{pq}B^P_{rs}$ (section 6.1). The solvers read
restricted orbitals, except spin-blocked UCCSD and UCCSD(T) (section 16.8,
UHF) and the library-only open-shell LinLCCD. The
derivations are in `crates/ferric-cc/theory.md`, which the sections include
by heading. Spin-orbital solvers use interleaved spin orbitals (even index
α, odd index β) and $\langle pq\|rs\rangle = (pr|qs) - (ps|qr)$; the code
stores the negative denominator $-\Delta$ and divides by it.
