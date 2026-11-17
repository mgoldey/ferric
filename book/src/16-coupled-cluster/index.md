# 16. Coupled cluster

MP2 (chapter 10) lets each electron pair correlate once, against the
Hartree–Fock background. Coupled cluster lets the pairs feel each other: the
amplitudes are solved self-consistently, and the exponential ansatz builds
products of independent excitations so that the energy of two
non-interacting molecules is the sum of their energies.

The chapter covers:

1. CCD and CCSD in spin orbitals (16.1), the form closest to the textbook;
2. the spin-adapted closed-shell CCSD that the CLI and Python run (16.2);
3. the perturbative triples correction, in both forms (16.3);
4. EOM-CCSD singlet excitation energies (16.4);
5. the linearized ladder method LinLCCD (16.5) and the ωB97X-L-V double
   hybrid built on it (16.6);
6. the local forms of these methods (16.7);
7. spin-blocked CCSD and CCSD(T) on a UHF reference (16.8).

Every solver reads canonical HF orbitals and RI integrals
$(pq|rs) = \sum_P B^P_{pq}B^P_{rs}$ (section 6.1). The solvers read
restricted orbitals, except spin-blocked UCCSD and UCCSD(T) (section 16.8,
UHF) and the library-only open-shell LinLCCD. The reference equations are in
`crates/ferric-cc/theory.md`; each section names the headings it follows.

Two conventions hold throughout. Spin-orbital solvers use interleaved spin
orbitals (even index α, odd index β) and
$\langle pq\|rs\rangle = (pr|qs) - (ps|qr)$. The code stores the negative
denominator $-\Delta$ and divides by it. Chapter 1 sets up the second
quantization (1.1), the Wick contractions the proof scripts use (1.2) and the
spin adaptation of amplitudes (1.3) that this chapter relies on.
