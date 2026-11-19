# Amplitude-threshold local OP-MP3 {#local-opmp3}

<!-- status: gap; registry: local-opmp3 -->

*Notation follows `book/notation.md` and section @sec:opmp3. $c^{(1)}$ and
$c^{(2)}$ are the first- and second-order doubles amplitudes, $W^{(2)}$ and
$W^{(3)}$ the matrices built from them, and $P$ the projector onto the kept
amplitudes of section @sec:amplitude-lmp2.*

OP-MP3 (section @sec:opmp3) sums the MP2 and MP3 energies of each occupied
orbital through a matrix Padé approximant. Its two inputs are the first-
and second-order doubles amplitudes. Both are as short-ranged as the MP2
amplitudes: an electron pair on one end of an alkane does not excite into
virtual orbitals on the other end at second order, and it does not at
third order either. So the same amplitude threshold that makes local MP2
cheap should make OP-MP3 cheap. Local OP-MP3 solves both orders of
amplitudes in localized orbitals, on the kept set of amplitude-threshold
LMP2, and hands them to the unchanged OP-MP3 models.

The plan:

1. Show that OP-MP3 does not depend on which orthonormal occupied and
   virtual orbitals carry the amplitudes. Then "local OP-MP3" has one
   meaning at $\varepsilon = 0$.
2. Choose the truncation of the second-order amplitudes so that the
   third-order energy keeps its Wigner $2n+1$ form.
3. Say what a Kohn-Sham reference adds: the non-Brillouin singles energy and
   the occupied-virtual Fock block, and why the mask leaves both alone.

## The equations in a localized basis {#local-opmp3-equations}

For a closed-shell reference with frozen core, store the alpha-beta
amplitude as $c_{iajb} = c^{\alpha\beta}_{iJaB}$ and write $J_{iajb} = (ia|jb)$.
The swap $(\mathcal S c)_{iajb} = c_{ibja}$ turns it into the same-spin part
$c - \mathcal S c$. In orbitals where the occupied and virtual Fock blocks
$f$ and $F$ need not be diagonal, the Fock superoperator is

$$ (\mathcal A c)_{iajb} = \sum_c F_{ac}\,c_{icjb} + \sum_c c_{iajc}\,F_{cb} - \sum_k f_{ik}\,c_{kajb} - \sum_k c_{iakb}\,f_{kj} . $$

The first- and second-order amplitudes solve

$$ \mathcal A c^{(1)} = -J, \qquad \mathcal A c^{(2)} = -R(c^{(1)}), \label{eq:local-opmp3-orders} $$

with $R$ the closed-shell MP3 residual of section @sec:opmp3 (two ladders,
the rings and the exchange rings). In canonical orbitals $\mathcal A$ is
the diagonal $\Delta$ and both solves are divisions. The energies are
$E^{(n)} = \langle (2-\mathcal S)c^{(n-1)}, J\rangle$ and the alpha-block W
matrices are

$$ W^{\mathrm{OS}}_{ij} = \tfrac14\sum_{kab} c_{iakb}J_{jakb} + (i\leftrightarrow j), \qquad
W^{\mathrm{SS}}_{ij} = \tfrac14\sum_{kab} (c - \mathcal S c)_{iakb}J_{jakb} + (i\leftrightarrow j) . $$

Every model of section @sec:opmp3 is twice the alpha trace of a Padé
$P(A, B) = A(A-B)^{-1}A$ of these matrices.

## Why the localized basis does not matter at $\varepsilon = 0$ {#local-opmp3-invariance}

Localize the active occupied orbitals by an orthogonal $U$ and the
virtuals by an orthogonal $V$. $J$ and $R$ transform as tensors, and
$\mathcal A$ does too as long as the full Fock blocks $f$ and $F$ are kept,
off-diagonal elements included. So the solutions of
(@eq:local-opmp3-orders) transform as tensors. $W$ sums over $a$, $k$ and
$b$, which absorbs $V$ and the $U$ on $k$, and leaves $i$ and $j$ free:

$$ W_{\mathrm{loc}} = U^{\mathsf T} W_{\mathrm{can}}\, U \quad\text{for each order and spin channel.} \label{eq:local-opmp3-covariance} $$

Substitute into the Padé:
$P(U^{\mathsf T}AU, U^{\mathsf T}BU) = U^{\mathsf T}AU\,(U^{\mathsf T}(A-B)U)^{-1}U^{\mathsf T}AU = U^{\mathsf T}P(A,B)\,U$,
and the trace does not change. The same-spin/opposite-spin split is by the
spin of the summed $k$, which a spatial rotation leaves alone. Check the
limit: with $U = 1$ and $V = 1$ the local equations are the canonical ones
term by term.

Drop the off-diagonal Fock elements in the local solve and the argument
fails: $\mathcal A$ is no longer the rotated canonical operator. On propane
in STO-3G that error is $5\times10^{-3}$ Ha, far above the $10^{-10}$ Ha bar of
the anchor.

> **Unverified equation @eq:local-opmp3-covariance** · OP-MP3 is invariant under the occupied and virtual localization
>
> | | |
> |---|---|
> | Proof | `scripts/local_opmp3_sympy.py` claims a1 (the trace Padé under a symbolic rotation) and a2 (the W covariance, every amplitude and integral an independent symbol), and N1 (canonical route vs a rotated route with the full Fock, OP-MP3 and SSD-OPMP3 agree to $3.3\times10^{-16}$); mutations `pade_half_rotated`, `w_virtual_once` and `semicanonical` each turn their check red |
> | Code | not ported yet: the Python prototype `scripts/prototypes/local_opmp3_proto.py` |
> | Code test | `test_local_opmp3_matches_exact_in_the_trivial_limit` in `crates/ferric-python/tests/test_local_opmp3_proto.py` |
>
> <!-- unverified: id=local-opmp3.invariance; todo=cite the Rust local OP-MP3 driver and its eps=0 anchor test once ported -->

## One mask for both orders {#local-opmp3-one-mask}

Keep the amplitudes of the LMP2 kept set $K$ (Eq 8 of section
@sec:amplitude-lmp2, with the Fock-propagated screen of section
@sec:amplitude-lmp2-fock-screen) and write $P$ for the projector onto them.
The truncated equations are

$$ P\mathcal A P\, c^{(1)} = -PJ, \qquad P\mathcal A P\, c^{(2)} = -P\,R(c^{(1)}) . \label{eq:local-opmp3-masked} $$

$R$ is evaluated from the masked $c^{(1)}$ with untruncated integrals. A
dropped amplitude is zero at both orders, and it contributes nothing to
the residual of any other pair.

Why the same $P$ for both orders? Because then the third-order energy
keeps its two forms. Write $G = 2 - \mathcal S$ for the closed-shell energy
weight. $\mathcal A$ commutes with $\mathcal S$ and is symmetric, so $G\mathcal A$
is symmetric, and the kept set is closed under the swap, so $G$ commutes
with $P$. Take the second-order energy, substitute
$PJ = -P\mathcal A P c^{(1)}$, move $\mathcal A$ to the other side, and
substitute $P\mathcal A P c^{(2)} = -PR$:

$$ E^{(3)} = \langle G c^{(2)}, J\rangle = -\langle G c^{(2)}, \mathcal A c^{(1)}\rangle = -\langle G c^{(1)}, \mathcal A c^{(2)}\rangle = \langle G c^{(1)}, R(c^{(1)})\rangle . \label{eq:local-opmp3-2n1} $$

The right-hand form needs only $c^{(1)}$, so the two sides are computed by
different code and checked against each other at every $\varepsilon$. Solve
$c^{(2)}$ on a second pattern $K_2 \ne K$ and the middle step fails: the
operator that maps $c^{(1)}$ to $J$ and the one that maps $c^{(2)}$ to $R$
are then different blocks of $\mathcal A$. Check the limit: at $K$ = everything
the identity reduces to the canonical $2n+1$ identity.

> **Unverified equation @eq:local-opmp3-2n1** · the third-order energy keeps its 2n+1 form on one mask
>
> | | |
> |---|---|
> | Proof | `scripts/local_opmp3_sympy.py` claim b (symbolic, $G\mathcal A$ symmetric and $G$ block-diagonal on $K$), b2 (two masks: a rational counterexample, $-9/7$ against $3/11$) and N2 (closed-shell model at kept fraction 0.65); mutation `two_masks` turns b and N2 red ($-8.2\times10^{-2}$) |
> | Code | not ported yet: `local_run` in `scripts/prototypes/local_opmp3_proto.py` |
> | Code test | `test_e3_identity_holds_on_one_mask_and_fails_on_two` in `crates/ferric-python/tests/test_local_opmp3_proto.py` |
>
> <!-- unverified: id=local-opmp3.2n1; todo=cite the Rust local OP-MP3 driver and its finite-eps E3 identity test once ported -->

## A Kohn-Sham reference {#local-opmp3-ks-reference}

OP-MP3 on a Kohn-Sham reference (section @sec:opmp3) uses the HF Fock
operator built from the Kohn-Sham density. Its occupied and virtual blocks
are diagonalized separately (semicanonical orbitals), and its
occupied-virtual block $F_{ia}$, which is not zero for a Kohn-Sham
density, enters only the non-Brillouin singles energy

$$ E_{\mathrm{NBS}} = -2\sum_{ia}\frac{F_{ia}^2}{\varepsilon_a - \varepsilon_i} \qquad\text{(semicanonical).} $$

The local doubles equations use only the $f$ and $F$ blocks of that same
Fock operator, rotated into the localized orbitals. $F_{ia}$ never enters
them, so the doubles mask cannot change $E_{\mathrm{NBS}}$. And
$E_{\mathrm{NBS}}$ is itself orbital-invariant: in localized orbitals it is
the solution of a singles equation with the non-diagonal blocks, and that
solution is the rotated semicanonical one. Its cost is $O(ov)$, so the
local method keeps it untruncated, a reference-wide term computed once.
The same holds for the HF reference energy of the Kohn-Sham density.

The same-spin damping $x_{ss}$ of SSD-OPMP3 multiplies $W^{(3)\mathrm{SS}}$
after the amplitudes are formed. It does not touch the mask either.

> **Unverified claim** · $E_{\mathrm{NBS}}$ is independent of the doubles mask and of the localization
>
> | | |
> |---|---|
> | Proof | `scripts/local_opmp3_sympy.py` claim N3 (semicanonical formula vs the localized singles solve, $-4.6\times10^{-2}$ Ha model value, agreement $< 10^{-14}$); mutation `nbs_local_diag` (keep only the diagonal of the localized Fock) turns it red |
> | Code | not ported yet: `System` (the `reference` branch) in `scripts/prototypes/local_opmp3_proto.py` |
> | Code test | `test_pbe0_reference_new_maps_match_exact_in_the_trivial_limit` in `crates/ferric-python/tests/test_local_opmp3_proto.py` |
>
> <!-- unverified: id=local-opmp3.nbs; todo=cite the Rust driver's reuse of the canonical E_NBS once ported -->

## What is proven and what is not {#local-opmp3-proven}

Proven, by the sympy script and its mutations: the invariance at
$\varepsilon = 0$, the $2n+1$ identity on one mask and its failure on two,
and the independence of $E_{\mathrm{NBS}}$ from the mask. Not proven: any
bound on the truncation error of the OP-type models. The error of
$E^{(2)}$ is bounded by the Schur norm of the dropped amplitudes (section
@sec:amplitude-lmp2-fock-screen). The Padé passes the errors of $W^{(2)}$
and $W^{(3)}$ through its first derivatives, and nothing here bounds those
derivatives away from a near-singular $A - B$ (a stretched bond). The
method exists only as the Python prototype; this section gains its code
excerpts and measurements when the Rust port lands.
