# Determinants, strings and the phase convention {#determinants}

<!-- status: ready; registry: casci -->

Stretch the bond of H2 and the bonding orbital $\sigma_g$ and the
antibonding orbital $\sigma_u$ approach the same energy. The ground state
then needs $|\sigma_g^2\rangle$ and $|\sigma_u^2\rangle$ with nearly equal
weight, which puts one electron near each nucleus. Restricted Hartree–Fock
keeps only $|\sigma_g^2\rangle$, which gives the ionic configurations
H$^+$H$^-$ the same coefficient as the covalent ones at every distance, so
its energy dissociates too high. CAS-CI fixes this by picking a
few orbitals near the Fermi level, the active space, and diagonalizing $H$
over every determinant that places the active electrons in them; the other
orbitals stay doubly occupied (core) or empty (virtual).

H2 in STO-3G with a CAS(2,2), from the `589b50832` build, at four bond
lengths. At 0.74 Å the CAS-CI is 20 mHa below RHF and $|\sigma_g^2\rangle$
carries almost all the weight. By 4 Å the two coefficients are nearly equal
in magnitude, the natural occupations approach 1 and 1, and RHF sits 318 mHa
above the CAS-CI. The mixed coefficient `c12` is zero by symmetry, and the trace of
the 1-RDM (section @sec:rdms) is the electron count:

```python
import numpy as np
import ferric

bs = ferric.BasisSet.bundled("sto-3g")
for r in (0.74, 1.5, 2.5, 4.0):
    mol = ferric.Molecule.from_xyz_string(f"2\nH2\nH 0 0 0\nH 0 0 {r}\n")
    res = ferric.run_casci(mol, bs, n_active=2, n_elec_active=(1, 1))
    c = np.asarray(res.ci_vector)          # (alpha strings, beta strings), strings 0b01, 0b10
    occ = np.linalg.eigvalsh(np.asarray(res.dm1))[::-1]
    print(f"R={r:4.2f} A  E_RHF={res.rhf_energy:.8f}  E_CAS={res.e_total:.8f}  "
          f"n_det={res.n_determinants}  c11={c[0,0]:+.5f} c22={c[1,1]:+.5f}  "
          f"c12={c[0,1]:+.1e}  occupations={occ.round(5)}  tr(dm1)={np.trace(res.dm1):.6f}")
```

```text
R=0.74 A  E_RHF=-1.11675931  E_CAS=-1.13728383  n_det=4  c11=-0.99365 c22=+0.11254  c12=+8.7e-17  occupations=[1.97467 0.02533]  tr(dm1)=2.000000
R=1.50 A  E_RHF=-0.91087356  E_CAS=-0.99814935  n_det=4  c11=-0.93471 c22=+0.35540  c12=+2.1e-16  occupations=[1.74738 0.25262]  tr(dm1)=2.000000
R=2.50 A  E_RHF=-0.70294360  E_CAS=-0.93605492  n_det=4  c11=-0.77099 c22=+0.63685  c12=+1.1e-14  occupations=[1.18884 0.81116]  tr(dm1)=2.000000
R=4.00 A  E_RHF=-0.61486998  E_CAS=-0.93317136  n_det=4  c11=-0.71026 c22=+0.70394  c12=-3.8e-17  occupations=[1.00893 0.99107]  tr(dm1)=2.000000
```

A CAS-CI wavefunction is a linear combination of every determinant with
$n_\alpha$ and $n_\beta$ electrons in $n_a$ active orbitals. Each spin
channel is chosen independently, so the count is a product of two
binomials. H2 in a CAS(2,2) with $(n_\alpha, n_\beta) = (1,1)$ has
$2\times2 = 4$ determinants. N2 in a CAS(6,6) has $20^2 = 400$. Full CI of
water in STO-3G (7 orbitals, 5 α and 5 β electrons) has $21^2 = 441$.

ferric stores a determinant as two `u64` bit strings, one per spin, and the
CI vector as the matrix $C(I_\alpha, I_\beta)$ over α strings and β
strings. A sign convention is part of that representation. It fixes which
matrix elements of $H$ are positive, and a convention mixed up between two
parts of the code gives a wrong Hamiltonian with the right spectrum on
symmetric test molecules.

| Symbol | Meaning | Code name |
|---|---|---|
| $A$, $B$ | α and β occupation strings, bit $p$ = active orbital $p$ | `u64` |
| ${\rm pb}(s, p)$ | $(-1)^{\#\{q<p:\,q\in s\}}$ | `strings::phase_below` |
| $N_\text{det}$ | $\binom{n_a}{n_\alpha}\binom{n_a}{n_\beta}$ | `DeterminantSpace::n_det` |

| Entry point | Location |
|---|---|
| Rust | `ferric-ci/src/strings.rs::{enumerate_strings,phase_below,count_strings}` |
| CLI | used by `method.kind = "casscf"` and `"nevpt2"` |
| Python | `run_casci`, `run_casscf`, `run_nevpt2` |

## The convention {#determinants-convention}

A determinant is a pair of `u64` strings $(A, B)$ over $n$ active spatial
orbitals. Bit $p$ of $A$ set means $p\alpha$ is occupied; bit $p$ of $B$
set means $p\beta$ is occupied. To write the determinant as operators we
need an order for the $2n$ spin orbitals. ferric puts α before β: mode $p$
is $p\alpha$ and mode $n+p$ is $p\beta$. The determinant is

$$
|A,B\rangle = \prod_{p\in A}^{\uparrow} a^\dagger_{p\alpha}
              \prod_{q\in B}^{\uparrow} a^\dagger_{q\beta}\,|0\rangle ,
$$

with the creators in ascending index order from left to right.

Now apply an annihilator $a_P$ to the bit string $s$ of this product. It
has to reach $a^\dagger_P$ before it can remove it, so anticommute it to
the right past every creator to the left of $a^\dagger_P$. Those are the
occupied modes $Q < P$, and each swap costs a factor $-1$. Then
$a_Pa^\dagger_P = 1 - a^\dagger_Pa_P$ acts on a product without mode $P$,
and the second term gives zero. A creator $a^\dagger_P$ moves to its
ascending position past the same occupied modes. Both cases give

$$
a_P\,|s\rangle = (-1)^{\#\{Q<P\,:\,Q\in s\}}\,|s\setminus P\rangle ,
\qquad
a^\dagger_P\,|s\rangle = (-1)^{\#\{Q<P\,:\,Q\in s\}}\,|s\cup P\rangle .
$$

The factor is the Jordan–Wigner sign. For example, the string `0b1011` has
orbitals 0, 1 and 3 occupied. Removing orbital 1 crosses orbital 0, sign
$-1$. Removing orbital 3 crosses orbitals 0 and 1, sign $+1$.

`ferric-ci/src/strings.rs::phase_below` evaluates
$(-1)^{\#\{q<p\,:\,q\in s\}}$ on one spin channel's string; section @sec:hamiltonian-determinant-code
shows its body. For an α operator that is the full Jordan–Wigner sign. A β
operator also has to cross all $N_\alpha$ α creators, which multiplies the
full sign by $(-1)^{N_\alpha}$. Every term of $H$ contains β operators in
annihilator–creator pairs, so this factor appears an even number of times
and cancels. The code therefore never counts α electrons for a β operator.

Other phase conventions are possible. An interleaved order
($p\alpha, p\beta, (p{+}1)\alpha, \dots$) assigns each determinant a sign
$S_I = \pm1$ relative to this one, and changes $H_{IJ}$ to $S_IS_JH_{IJ}$.
That is a diagonal similarity: the same spectrum, different matrix
elements. Sections @sec:slater-condon to @sec:rdms all assume α before β.

> **Verified equation {#eq:ci-strings-convention}** · bit strings are determinants with sign $+1$
>
> On Jordan–Wigner matrices for eight spin orbitals: the annihilators obey
> the canonical anticommutation relations; the ascending product of
> creators applied to $|0\rangle$ is the computational basis vector of the
> bit string with coefficient $+1$; and $a_P$ on it carries
> $(-1)^{\#\{Q<P\}}$ over occupied modes. A single-channel phase is
> therefore the full Jordan–Wigner phase for α, and for β up to a factor
> $(-1)^{N_\alpha}$ that cancels in every term of $H$.
>
> | | |
> |---|---|
> | Proof | `scripts/casci_slater_condon_phases_sympy.py`, claims 1 and 2 |
> | Negative control | claim 4c: a β single excitation with the $(-1)^{N_\alpha}$ applied once |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-ci/src/strings.rs::{phase_below,enumerate_strings}` |
> | Code test | `brute_force_reference_is_hermitian_and_number_conserving` in `crates/ferric-ci/tests/slater_condon_brute_force.rs` |
>
> <!-- verified: id=ci.strings.convention; script=scripts/casci_slater_condon_phases_sympy.py; claim=2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-ci/src/strings.rs::{phase_below,enumerate_strings}; rust_test=ferric-ci/tests/slater_condon_brute_force.rs::brute_force_reference_is_hermitian_and_number_conserving; registry=casci -->

## The Hamiltonian {#determinants-hamiltonian}

Over the active orbitals the Hamiltonian is

$$
H = \sum_{pq,\sigma} h_{pq}\, a^\dagger_{p\sigma} a_{q\sigma}
  + \tfrac12 \sum_{pqrs,\sigma\tau} (pq|rs)\,
    a^\dagger_{p\sigma} a^\dagger_{r\tau} a_{s\tau} a_{q\sigma}
  + E_\text{core} .
$$

Read the two-electron term from the right: it removes electrons from $q$
(spin σ) and $s$ (spin τ) and puts them in $p$ and $r$ with the same spins.
$(pq|rs)$ is in chemists' notation, the Coulomb interaction of the charge
distributions $\phi_p\phi_q$ and $\phi_r\phi_s$, and has the 8-fold
symmetry of real orbitals. $h$ is symmetric. $E_\text{core}$ is a constant
and is added on the diagonal only.

The core electrons are frozen in doubly occupied orbitals. Their Coulomb
and exchange interaction with an active electron depends only on where that
one electron is, so it is a one-electron operator and is folded into $h$.
$E_\text{core}$ is the core's own energy plus the nuclear repulsion.
Section @sec:casscf gives both for a CASSCF orbital set. With no core orbitals,
$h$ is the bare one-electron operator and $E_\text{core}$ the nuclear
repulsion.

## Strings in the code {#determinants-strings-code}

`ferric-ci/src/strings.rs::enumerate_strings` lists all $k$-electron
strings of $n$ orbitals in ascending numeric order with Gosper's hack. The
loop, with the `clippy` attributes and their comment removed:

```rust
let mut x: u64 = (1u64 << k) - 1;        // lowest k bits set: the first string
let limit: u64 = if n_orb >= 64 { u64::MAX } else { 1u64 << n_orb };
while x < limit {
    out.push(x);
    let c = x & x.wrapping_neg();        // lowest set bit of x
    let r = x + c;                       // the carry clears the lowest run of 1s, sets the bit above it
    if c == 0 {
        break;
    }
    x = (((x ^ r) >> 2) / c) | r;        // that run, one bit shorter, refilled at the bottom
}
```

For $k = 3$ it steps from `0b0111` ($c = 1$, $r$ = `0b1000`,
$x \oplus r$ = `0b1111`, shifted and divided: `0b11`) to `0b1011`, the next
larger number with three bits set. The order only indexes the coefficient
array, so any fixed order is correct.

The `u64` holds at most `MAX_ACTIVE_ORBITALS` = 63 active orbitals. CAS-CI
and CASSCF refuse a larger space before any integral work. The real limit
is memory. The Davidson subspace grows with $N_\text{det}$, which is
combinatorial in $n_a$, and the CAS-CI memory plan refuses a space that
does not fit (section @sec:direct-sigma-davidson). A CAS(14,14) singlet has
$3432^2 = 11\,778\,624$ determinants, and a 50-vector Davidson subspace
for it takes 9.4 GB.

**Known limits (2026-11-15).** One fixed convention, α before β. An
interleaved order ($p\alpha, p\beta$) would change $H_{IJ}$ by a diagonal
$\pm1$ similarity; ferric never uses it, and the proof script shows it fails
the element-wise check while keeping the spectrum.
