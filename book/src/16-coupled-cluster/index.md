# Coupled cluster {#coupled-cluster}

MP2 (chapter @sec:mp2) lets each electron pair correlate once, against the
Hartree–Fock background. Coupled cluster lets the pairs feel each other: the
amplitudes are solved self-consistently, and the exponential ansatz builds
products of independent excitations so that the energy of two
non-interacting molecules is the sum of their energies.

The chapter covers:

1. CCD and CCSD in spin orbitals (@sec:ccd-ccsd-spin-orbital), the form closest to the textbook;
2. the spin-adapted closed-shell CCSD that the CLI and Python run (@sec:closed-shell-ccsd);
3. the perturbative triples correction, in both forms (@sec:triples);
4. EOM-CCSD singlet excitation energies (@sec:eom-ccsd);
5. the linearized ladder method LinLCCD (@sec:linlccd) and the ωB97X-L-V double
   hybrid built on it (@sec:wb97x-l-v);
6. the local forms of these methods (@sec:local-cc);
7. spin-blocked CCSD and CCSD(T) on a UHF reference (@sec:uccsd).

Every solver reads canonical HF orbitals and RI integrals
$(pq|rs) = \sum_P B^P_{pq}B^P_{rs}$ (section @sec:factorization-and-metric). The solvers read
restricted orbitals, except spin-blocked UCCSD and UCCSD(T) (section @sec:uccsd,
UHF) and the library-only open-shell LinLCCD. The reference equations are in
`crates/ferric-cc/theory.md`; each section names the headings it follows.

Two conventions hold throughout. Spin-orbital solvers use interleaved spin
orbitals (even index α, odd index β) and
$\langle pq\|rs\rangle = (pr|qs) - (ps|qr)$. The code stores the negative
denominator $-\Delta$ and divides by it. Chapter @sec:second-quantization sets up the second
quantization (@sec:hamiltonian), the Wick contractions the proof scripts use (@sec:wick) and the
spin adaptation of amplitudes (@sec:spin-adaptation) that this chapter relies on.
