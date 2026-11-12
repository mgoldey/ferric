# SC-NEVPT2 prototype: derivation, conventions, measurements

`nevpt2_proto.py` computes the strongly contracted NEVPT2 energy (Angeli,
Cimiraglia, Malrieu, JCP 117, 9138 (2002)) class by class on the output of
`ferric.run_casscf`. It is the correctness reference for a Rust port (T048).

## Definition

The zeroth-order Hamiltonian is Dyall's:

    H_D = sum_i e_i E_ii + sum_r e_r E_rr + H_act + const,
    H_act = sum_tu F^I_tu E_tu + 1/2 sum_tuvw (tu|vw) (E_tu E_vw - d_uv E_tw),

with F^I the inactive Fock matrix (h plus the core Coulomb and exchange) and
e_i, e_r the eigenvalues of the core and virtual blocks of the generalized
Fock matrix F = F^I + F^A, F^A built from the active 1-RDM.

For a class with nv virtual particles and nc core holes, and a label l (the
external orbitals: r, rs, i, ij, ir, ...), the strongly contracted perturber is

    Psi_l = P_l H Psi_0,

where P_l projects on the determinants whose external occupation is exactly
l. With N_l = <Psi_l|Psi_l> and the Dyall excitation energy

    E_l - E_0 = Delta_l + K_l / N_l,   Delta_l = sum_(r in l) e_r - sum_(i in l) e_i,
    K_l = <Psi_l| H_act - E_0 |Psi_l>,

the class energy is E_class = - sum_l N_l / (Delta_l + K_l / N_l). Labels of
two orbitals of the same space are unordered pairs (r <= s, i <= j). Labels
with N_l <= 1e-14 are skipped (PySCF's `NUMERICAL_ZERO`).

| class | nv | nc | active change | PySCF |
|---|---|---|---|---|
| ijrs | 2 | 2 | 0 | Sijrs |
| ijr | 1 | 2 | +1 | Sijr |
| rsi | 2 | 1 | -1 | Srsi |
| ij | 0 | 2 | +2 | Sij |
| rs | 2 | 0 | -2 | Srs |
| ir | 1 | 1 | 0 | Sir |
| r | 1 | 0 | -1 | Sr |
| i | 0 | 1 | +1 | Si |

## Construction A: symbolic Wick reduction (the port target)

1. H normal ordered against the core is E_core + sum F^I_pq {a+_p a_q} +
   1/2 sum (pq|rs) {a+_p a+_r a_s a_q}. On the reference (core filled,
   virtuals empty) only strings without core creators and virtual
   annihilators survive, and for those the core-normal-ordered product is the
   plain string. `perturber(nv, nc)` keeps the space assignments of those
   strings with the class's external counts, so the one-body coefficient is
   F^I and no orbital-energy correction is written by hand.
2. P_l is a product of spin-summed number operators on the label slots
   (`projector`): sum_s a+_rs a_rs per virtual label, sum_s a_is a+_is per
   core label, and (1/2) sum_st a+_rs a+_rt a_rt a_rs for a doubly occupied
   virtual label (likewise for a doubly emptied core label).
3. N_l = <V+ P_l V> and K_l = <V+ P_l [H_act, V]>. The commutator uses the
   derivation rule on each active operator of V (`comm_hact`):

       [H_act, a_x]  = -h_xq a_q - 1/2 (xq|rs) a+_r a_s a_q + 1/2 (pq|xs) a+_p a_s a_q,
       [H_act, a+_x] =  h_px a+_p + 1/2 (px|rs) a+_p a+_r a_s - 1/2 (pq|rx) a+_p a+_r a_q,

   with spins following the substituted operator. Using the commutator
   instead of <V+ P H_act V> - E_0 N keeps the RDM rank at 4.
4. `expect` evaluates <ref| string |ref>: the string is sorted by space
   (sign of the permutation), core and virtual operators are fully contracted
   (core: a+ left of a; virtual: a left of a+), active operators get every
   vacuum Wick contraction (a left of a+), and the uncontracted active
   remainder is a spin-free RDM. Each spin id sits on one creator and one
   annihilator, so after contraction the spin graph is a set of paths, each
   joining one uncontracted creator to one uncontracted annihilator, plus
   closed cycles (a factor 2 each). The path pairing fixes the RDM index order
   and its sign.
5. Terms are merged under a canonical renaming of dummy slots and evaluated by
   `numpy.einsum` over MO integral blocks and the RDMs in PySCF's
   normal-ordered convention
   dm_k[p1,q1,...,pk,qk] = sum_spins <a+_p1 .. a+_pk a_qk .. a_q1>.

Delta functions that would equate two different output labels drop the term,
so the r != s (i != j) variants and the r = s (i = j) variants are compiled
separately and summed over their own label ranges.

## Construction B: determinant-space definition

`nevpt2_det` builds Psi_0 in the full space (alpha x beta strings over all
orbitals), applies H with explicit spin-summed E_pq matrices, groups the
determinants of H Psi_0 by external occupation (the labels), and evaluates
N_l, K_l with H_act applied once to the whole perturber vector (H_D does not
couple different external occupations). It shares no algebra with
construction A. N2/STO-3G with 2 core, CAS(10,6) and 2 virtual orbitals has
14400 determinants and every class populated.

## RDMs

The 1- to 4-RDMs come from ferric's CI vector twice:

* `dense_rdms`: the CI vector is embedded in the active Fock space (alpha
  modes 0..n-1, beta modes n..2n-1, Jordan-Wigner signs), annihilation chains
  phi = a_qk .. a_q1 Psi are built sector by sector, and
  dm_k = sum_spins <phi_p|phi_q>.
* `pyscf_rdms`: `pyscf.fci.rdm.make_dm1234` with `reorder_dm1234`.

`build_ref` requires agreement within 100 * ndet * eps * max|dm_k| for every
rank, and `ref_from_casscf` checks the dense 1-/2-RDM against ferric's `dm1`
and `dm2` and E_core + E_0 against ferric's CASSCF energy.

## Orbitals

Core and virtual orbitals are the eigenvectors of the core and virtual blocks
of F. Active orbitals are pseudo-canonical (eigenvectors of the active block
of F); the RDMs are rotated with them, so ferric's CI vector is used
unchanged. Every active index is summed, so E_PT2 does not depend on the
active basis; PySCF uses natural orbitals for its comparison.

## Usage

    cd ~/qc/ferric && PYTHONPATH=<shim with a ferric build> box-lease exec -- \
      env OPENBLAS_NUM_THREADS=1 .venv/bin/python \
      ~/qc/ferric-nevpt2/scripts/prototypes/nevpt2_proto.py [--only STEP]

Steps in order: full, mp2, size, det, ref, mut (cost prints the term counts).
An anchor failure stops the run. The ref step with FD takes about 9 minutes
on this box; `--no-fd` skips the own-CASSCF bar.

## Anchors and checks (measured 2026-11-12, base commit 0ec644103)

All runs: `OPENBLAS_NUM_THREADS=1`, ferric `run_casscf` with
`casscf_grad_tol=1e-8` and its default `ci_conv_thresh=1e-10` (a 1e-12 CI
threshold makes ferric's Davidson stop at residual 7.9e-11 after 7
iterations on the 36-determinant N2/STO-3G CAS(10,6) and raise).

Every reference also passes: dense RDMs vs `pyscf.fci.rdm` within
100 * ndet * eps; dense 1-/2-RDM vs ferric's `dm1`/`dm2` (<= 1.0e-14);
E_core + E_0 vs ferric's CASSCF energy (<= 7.1e-13 Ha).

### 1. Active = full space

H2/6-31G CAS(2,4): no core and no virtual orbitals, every class has an empty
label set, E_PT2 = 0 exactly. This anchor exercises only the bookkeeping.

### 2. Single-determinant active space vs MP2

H2O/6-31G on ferric RHF orbitals, one active orbital (the HOMO, occupation
2), CI vector (1). With the active orbital doubly occupied the RDMs factorize,
so this anchor does not see cumulant terms. Expected per class:

* ijrs: MP2 pair sum over core pairs (i, j), from `ao2mo` and RHF energies;
* rsi: 2 * sum_i e_(i,t), the core-HOMO MP2 pairs (removing one electron from
  the doubly occupied active orbital costs e_t under H_act, so the strongly
  contracted function has the MP2 denominator);
* rs: -sum_ab (ta|tb)^2 / (e_a + e_b - 2 F^I_tt - (tt|tt)). The Dyall
  denominator differs from MP2's e_a + e_b - 2 e_t by (tt|tt), because the
  active vacuum has energy 0 and E_0 = 2 F^I_tt + (tt|tt);
* r, ir: proportional to the RHF Fock elements F_rt and F_ri, zero at SCF
  convergence;
* ijr, ij, i: zero norm (no active holes).

| class | wick (Ha) | expected (Ha) | diff |
|---|---|---|---|
| ijrs | -0.07676409403993 | -0.07676409403993 | 1.4e-17 |
| rsi | -0.04321362382404 | -0.04321362382404 | -2.8e-17 |
| rs | -0.00722987471871 | -0.00722987471871 | -4.3e-18 |
| ijr, ij, ir, r, i | 0 | 0 | 0 |

Bar 1e-12 Ha: rounding of about 1e3 summed label terms of magnitude below
0.1 Ha is below 1e-13.

### 3. Size consistency

H2 (0.74 Angstrom) CAS(2,2)/cc-pVDZ and two such molecules 100 Angstrom
apart, CAS(4,4). Only classes r and rs are populated (no core).

| quantity | value |
|---|---|
| E_PT2(H2) | -0.010688228780 Ha |
| E_PT2(dimer) - 2 E_PT2(H2), total | 1.4e-14 Ha |
| largest per-class difference | below 1e-14 Ha |
| bar | 3.4e-9 Ha |

Bar: E_PT2 is first order in the orbital error, so each run contributes
|dE_PT2/dkappa| * ||g|| / lambda_min (FD over all rotation pairs; monomer
8.87e-3, |g| 2.9e-9, lambda_min 3.09e-2; dimer 1.25e-2, 4.1e-9, 3.09e-2),
the monomer twice.

### Construction A vs construction B

N2/STO-3G at 1.6 Angstrom, 2 core, CAS(10,6), 2 virtual orbitals, 14400
determinants. Natural occupations 2.0000, 2.0000, 1.9998, 1.9998, 1.0002,
1.0002.

| class | wick (Ha) | determinant space (Ha) | diff |
|---|---|---|---|
| ijrs | -0.00003312501989 | -0.00003312501989 | -6.8e-21 |
| ijr | -0.00000000499472 | -0.00000000499472 | -9.3e-21 |
| rsi | -0.00014167398680 | -0.00014167398680 | 8.1e-20 |
| ij | -0.00000000087340 | -0.00000000087340 | -4.4e-20 |
| rs | -0.11365548686815 | -0.11365548686815 | 3.5e-16 |
| ir | -0.00001114475643 | -0.00001114475643 | 1.7e-19 |
| r | -0.00422429955163 | -0.00422429955162 | -2.3e-15 |
| i | -0.00000000389342 | -0.00000000389342 | -9.7e-19 |

Bar 1e-11 Ha (rounding). The determinant construction also reproduces
ferric's CASSCF energy and E_0 to 1e-12 Ha.

## Comparison with pyscf.mrpt.NEVPT on ferric integrals

PySCF runs on ferric's hcore, overlap and ERIs (`casscf_proto.
pyscf_rhf_on_ferric_integrals`). Three comparisons per system:

* shared: PySCF gets ferric's orbitals and CI vector, then canonicalizes them
  itself (natural active orbitals, eigenvectors of the core and virtual Fock
  blocks);
* PySCF orbitals: the prototype evaluated on PySCF's canonical orbitals and
  transformed CI vector, without re-diagonalizing;
* own: PySCF's own CASSCF on ferric's integrals from its own RHF.

Degenerate external orbitals. SC-NEVPT2 is not invariant to rotations inside
a group of degenerate core or virtual orbitals: a label (r, s) contracts the
perturbers of one orbital pair, and N/(Delta + K/N) is not linear in the
rotation. N2 has degenerate pi pairs, so ferric's eigh and PySCF's
canonicalization pick different bases inside them. `degenerate_spread`
measures the largest change of each class under four random rotations inside
every group with |de| < 1e-6. Only rs moves above rounding: 1.7e-7 Ha at N2
equilibrium, 6.2e-10 Ha stretched. The shared comparison of rs therefore
differs by up to the spread, and the PySCF-orbitals comparison removes it.

Measured per class (Ha). "PySCF orbitals" is the prototype on PySCF's
canonical orbitals and CI minus PySCF; "own" is the prototype on ferric's
CASSCF minus PySCF on its own CASSCF; "bar" is the own-CASSCF bar below.

H2O/6-31G CAS(4,4), 3 core, 6 virtual; natural occupations 1.9782, 1.9752,
0.0236, 0.0230. E_CASSCF ferric -76.036969096502, PySCF own -76.036969096501.

| class | E (Ha) | shared | PySCF orbitals | own | bar | \|dE/dk\| |
|---|---|---|---|---|---|---|
| ijrs | -0.023532987576 | 1.4e-16 | 1.9e-16 | -7.8e-09 | 1.9e-06 | 3.06e-02 |
| ijr | -0.000761026978 | 5.4e-19 | 1.4e-18 | -6.2e-11 | 3.4e-07 | 5.56e-03 |
| rsi | -0.007548207494 | -3.0e-17 | -1.6e-17 | -5.2e-09 | 1.9e-06 | 3.16e-02 |
| ij | -0.001315338819 | -1.7e-18 | -2.0e-18 | 1.0e-09 | 3.1e-07 | 5.17e-03 |
| rs | -0.003556132710 | 5.2e-18 | 5.6e-18 | 4.6e-09 | 4.1e-07 | 6.81e-03 |
| ir | -0.035945693465 | -1.2e-14 | -1.2e-14 | 1.8e-08 | 2.2e-06 | 3.54e-02 |
| r | -0.004444028117 | 1.4e-13 | 1.4e-13 | 7.5e-10 | 5.3e-07 | 8.77e-03 |
| i | -0.000148550940 | 2.1e-13 | 2.1e-13 | -1.4e-09 | 7.1e-08 | 1.16e-03 |
| total | -0.077251966099 | 0 | | 1.0e-08 | 3.0e-06 | 4.96e-02 |

N2/6-31G CAS(6,6) at 1.0977 Angstrom, 4 core, 8 virtual; occupations
1.981, 1.935, 1.935, 0.065, 0.065, 0.019.

| class | E (Ha) | shared | PySCF orbitals | own | degenerate spread | bar | \|dE/dk\| |
|---|---|---|---|---|---|---|---|
| ijrs | -0.008358190045 | 2.1e-17 | -3.6e-16 | 7.1e-09 | 9.0e-17 | 7.7e-08 | 1.60e-02 |
| ijr | -0.001854595592 | -1.4e-17 | -1.1e-17 | -2.5e-10 | 3.0e-18 | 3.7e-08 | 7.65e-03 |
| rsi | -0.016120056173 | -3.1e-17 | -5.9e-16 | -3.2e-11 | 1.2e-16 | 9.8e-08 | 2.02e-02 |
| ij | -0.005497423663 | -1.2e-16 | 7.8e-18 | -1.0e-09 | 6.9e-18 | 4.9e-08 | 1.01e-02 |
| rs | -0.009997154852 | 5.0e-09 | -6.9e-18 | -1.7e-07 | 1.7e-07 | 4.4e-07 | 2.17e-02 |
| ir | -0.020429543937 | -1.1e-15 | -4.0e-16 | -6.5e-09 | 1.5e-16 | 1.5e-07 | 3.17e-02 |
| r | -0.003605677340 | -2.8e-13 | -2.8e-13 | 7.3e-10 | 5.3e-16 | 1.4e-08 | 2.87e-03 |
| i | -0.001870567892 | 9.0e-14 | 9.3e-14 | -8.5e-10 | 3.7e-17 | 2.7e-08 | 5.66e-03 |
| total | -0.067733209495 | -5.0e-09 | | -1.7e-07 | | 5.8e-07 | 5.04e-02 |

N2/6-31G CAS(6,6) at 2.2 Angstrom (stretched), 4 core, 8 virtual;
occupations 1.4555, 1.1577, 1.1577, 0.8423, 0.8423, 0.5445, so the 3- and
4-RDM cumulants are large. E_CASSCF ferric -108.766572303401, PySCF own
-108.766572303393.

| class | E (Ha) | shared | PySCF orbitals | own | degenerate spread | bar | \|dE/dk\| |
|---|---|---|---|---|---|---|---|
| ijrs | -0.006499792238 | 5.0e-17 | 1.9e-17 | -2.6e-11 | 1.7e-17 | 1.7e-08 | 9.58e-03 |
| ijr | -0.003495749627 | 5.6e-18 | -4.3e-19 | -1.1e-09 | 1.0e-17 | 2.7e-08 | 1.49e-02 |
| rsi | -0.024768575115 | 1.4e-16 | 3.1e-17 | -3.6e-09 | 4.5e-17 | 4.8e-08 | 2.66e-02 |
| ij | -0.000870897000 | -2.4e-16 | -7.3e-17 | 5.8e-09 | 2.8e-18 | 7.7e-09 | 4.22e-03 |
| rs | -0.011900358517 | -7.5e-11 | 4.2e-17 | 1.1e-08 | 6.2e-10 | 2.3e-08 | 1.19e-02 |
| ir | -0.009315399273 | -6.3e-16 | -1.6e-16 | 1.5e-08 | 6.9e-18 | 7.3e-08 | 4.01e-02 |
| r | -0.003810519454 | 4.6e-15 | 4.8e-15 | 2.2e-08 | 8.6e-17 | 2.4e-08 | 1.32e-02 |
| i | -0.000310065158 | 3.6e-15 | 4.0e-15 | 3.6e-09 | 1.8e-17 | 4.5e-09 | 2.45e-03 |
| total | -0.060971356382 | -7.5e-11 | | 5.2e-08 | | 8.4e-08 | 4.55e-02 |

Bars:

* shared inputs and PySCF orbitals: 1e-10 Ha. The inputs are identical, so
  only rounding and PySCF's exact CI rotation to natural orbitals enter. The
  PySCF-orbitals comparison passes at <= 2.8e-13 Ha on all three systems; the
  shared comparison differs only in rs on N2, by less than the degenerate
  spread.
* own CASSCF: bar_class = |dE_class/dkappa| * (||g_ferric|| + ||g_pyscf||)
  / lambda_min + 2 * spread_class + 1e-10. ||g|| is the 2(F - F^T) gradient
  of `casscf_proto.Cas` evaluated at each code's orbitals, lambda_min the
  lowest eigenvalue of the FD CI-relaxed orbital Hessian at ferric's
  orbitals, dE/dkappa a central difference (h = 1e-4) of each class energy
  over every nonredundant rotation pair with the CI re-solved and the
  orbitals re-canonicalized at each point. The spread enters twice because
  each code picks its own basis inside the degenerate groups.

| system | \|\|g\|\| ferric | \|\|g\|\| PySCF | lambda_min | \|dE_PT2/dk\| |
|---|---|---|---|---|
| H2O | 8.2e-09 | 1.2e-06 | 1.95e-02 | 4.96e-02 |
| N2 | 8.7e-09 | 1.9e-07 | 4.02e-02 | 5.04e-02 |
| N2 stretched | 4.3e-09 | 9.6e-07 | 5.34e-01 | 4.55e-02 |

PySCF's CASSCF (`conv_tol_grad=1e-7` in its own norm) is the less converged
one in every case, so the own-CASSCF bars are set by PySCF. On stretched N2
three classes sit within a factor 1.3 of the bar (r 2.2e-8 vs 2.4e-8, i
3.6e-9 vs 4.5e-9, ij 5.8e-9 vs 7.7e-9), consistent with an error that is
first order in PySCF's orbital gradient.

## Mutations

Each mutation is applied to construction A only and must fail at least one
check. "PySCF" means the PySCF-orbitals comparison (bar 1e-10 Ha).

| mutation | MP2 anchor | size | wick vs det | H2O vs PySCF | stretched N2 vs PySCF |
|---|---|---|---|---|---|
| drop one 3-RDM term of the class-r norm | pass | fail (7.9e-9 > 3.4e-9) | fail (0.88 Ha) | fail (total -0.10530 vs -0.07725) | fail (total +0.04482 vs -0.06097) |
| flip the two-electron part of the class-i Koopmans matrix | pass | pass | fail (3.6e-9) | fail (total -0.077070) | fail (total -0.060566) |
| omit class ir | pass | pass | fail (1.1e-5) | fail (total -0.041306) | fail (total -0.051656) |

The MP2 anchor passes all three: in the single-determinant limit classes r,
i and ir are zero (Brillouin, or no active holes), and the RDMs factorize.
Size consistency holds for any additive formula and catches the first
mutation only through its size. The per-class comparisons on correlated
references catch all three.

## What a Rust port needs

Inputs: canonical core and virtual orbital energies, the inactive Fock matrix
in the MO basis (blocks va, ca, vc), the active F^I block (h_eff), and these
ERI blocks: (aa|aa), (va|aa), (ca|aa), (vc|aa), (va|ac), (vc|vc) (ijrs),
(va|vc), (ca|ca), (va|va), plus the generalized Fock (core and virtual
blocks) for canonicalization.

RDMs (spin-free, PySCF normal-ordered convention) needed per class, counted
from the compiled terms (`--only cost`; "norm" and "K" list the number of
summed terms per RDM rank 0..4 over all label variants):

| class | norm terms by rank | K terms by rank | highest RDM | largest single term (core, active, virtual indices) |
|---|---|---|---|---|
| ijrs | 17, 0, 0, 0, 0 | none | none | (2, 0, 2) |
| ijr | 13, 20, 0, 0, 0 | 20, 100, 40, 0, 0 | 2 | (2, 5, 1) |
| rsi | 0, 20, 0, 0, 0 | 0, 20, 40, 0, 0 | 2 | (1, 5, 2) |
| ij | 8, 20, 6, 0, 0 | 40, 200, 152, 24, 0 | 3 | (2, 7, 0) |
| rs | 0, 0, 6, 0, 0 | 0, 0, 24, 24, 0 | 3 | (0, 7, 2) |
| ir | 1, 24, 16, 0, 0 | 0, 40, 176, 64, 0 | 3 | (1, 7, 1) |
| r | 0, 1, 8, 4, 0 | 0, 1, 38, 84, 24 | 4 | (0, 9, 1) |
| i | 1, 17, 20, 4, 0 | 1, 77, 290, 180, 24 | 4 | (1, 9, 0) |

The term counts come from symbolic expansion and are an upper bound on what
a hand-factored port needs: the hole-type expectations (classes with active
particles: ijr, ij, i, ir) expand into many particle-RDM terms, which a port
evaluates as precomputed hole RDMs instead.

Intermediates a port should form once per reference (PySCF's names in
brackets, n = active orbitals):

1. dm1, dm2, dm3 from the CI vector (n^6 storage; the CI-to-dm3 cost is
   about ndet * n^6 with string-driven sigma).
2. The 4-RDM only enters the Koopmans matrices of classes r and i, always
   contracted with one active ERI (aa|aa). Form the two contracted
   intermediates directly from the CI vector instead of storing dm4 (n^8):
   the 4-RDM contracted with (aa|aa) as n^6 objects [f3ca, f3ac], cost about
   ndet * n^8 at most and n^6 storage.
3. Hole RDMs hdm1, hdm2, hdm3 (n^2..n^6) from dm1..dm3 [make_hdm*], for
   classes i, ij, ijr, ir.
4. Per-class Koopmans matrices built from h_eff, (aa|aa) and the RDMs:
   class r: n^6 (tuv, t'u'v') block plus n^4 and n^2 blocks [a16, a17, a19];
   class i: the same with hole RDMs [a22, a23, a25]; ijr [a3, hdm1];
   rsi [k27]; rs [a7]; ij [a9, hdm1..3]; ir [a12, a13]. f3ca and f3ac
   enter a16 and a22 only. The largest is the n^6 block, costing n^8 to
   build (dm3 and the 4-RDM contracted with (aa|aa)).
5. Per-label contraction: class r and i cost n_v (or n_c) * n^6 (the
   (r tuv) x n^6 x (r t'u'v') product); rs and ij cost n_v^2 n^4
   (n_c^2 n^4); rsi, ijr cost n_v^2 n_c n^2; ir costs n_c n_v n^4;
   ijrs is MP2-like, n_c^2 n_v^2.

The Wick construction (`compile_class`) also serves as a generator: its
compiled term lists are the complete set of contractions per class, and a
Rust implementation can be checked term group by term group against it.

## Not verified here

* The `rot` path of `build_ref` (degenerate rotations) is used only to
  measure the spread; it has no anchor of its own beyond the spread being at
  rounding for nondegenerate systems (H2O: 0).
* lambda_min is taken from an FD Hessian of the prototype gradient at
  ferric's orbitals only. A larger lambda_min makes the bar tighter, so an
  overestimate cannot cause a false pass, but the value is not checked
  independently.
* CI convergence is not in the bar: both codes use CI residuals at or below
  1e-10, and E_PT2 is first order in the CI error.
* Open-shell references (n_alpha != n_beta) are not exercised; the
  construction is spin-free and does not assume a singlet, but ferric's
  run_casscf supports n_alpha == n_beta only.
* Multi-state or quasi-degenerate NEVPT2 and the partially contracted
  variant are out of scope.

