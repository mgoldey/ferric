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
UHF) and the library-only open-shell LinLCCD.

The solvers fall into three index conventions:

- **Spin orbital** (`ccd`, `ccsd`, `ccsd_t`, `linlccd`, `linlccd_u`,
  `linlccd_exact`, `dlpno_linlccd`): indices run over $2n_o$ occupied and
  $2n_v$ virtual spin orbitals, interleaved (even index α, odd index β), with
  $\langle pq\|rs\rangle = (pr|qs) - (ps|qr)$ built from the dressed RI
  factors (`ferric_mp2::spinorbital::asym_*`).
- **Spin blocks on a UHF reference** (`u_ccsd`, `u_ccsd_t`): separate α and β
  occupied and virtual spaces (section @sec:uccsd).
- **Spatial, closed shell** (`ccsd_closed_shell`, `ccsd_t_closed_shell`,
  `linlccd_amplitude`, `dlpno_ccsd*`): indices run over $n_o$ and $n_v$
  spatial orbitals, and the amplitudes satisfy $t_{ij}^{ab} = t_{ji}^{ba}$.

Denominators are positive for a gapped reference,

$$
\Delta_{ij}^{ab} = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j,
\qquad
\Delta_{ijk}^{abc} = \varepsilon_a + \varepsilon_b + \varepsilon_c
 - \varepsilon_i - \varepsilon_j - \varepsilon_k,
$$

and every energy expression carries its minus sign explicitly. The code
stores the negative $-\Delta$ (the `d` arrays and `pno_denominators`) and
divides by it, so an update reads $t \leftarrow r/(-\Delta)$. Canonical HF
orbitals are assumed: $f_{ia} = 0$ and the occupied and virtual Fock blocks
are diagonal. Chapter @sec:second-quantization sets up the second
quantization (@sec:hamiltonian), the Wick contractions the proof scripts use (@sec:wick) and the
spin adaptation of amplitudes (@sec:spin-adaptation) that this chapter relies on.
