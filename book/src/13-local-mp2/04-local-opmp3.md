# Amplitude-threshold local OP-MP3 {#local-opmp3}

<!-- status: ready; registry: local-opmp3 -->

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

Notation follows section @sec:opmp3 and the chapter table of section
@sec:local-mp2:

| Symbol | Meaning |
|---|---|
| $c^{(1)}$, $c^{(2)}$ | the first- and second-order alpha-beta doubles amplitudes |
| $J$ | the integrals $(ia\vert jb)$ |
| $R(c^{(1)})$ | the closed-shell MP3 residual |
| $\mathcal A$ | the Fock superoperator in the localized orbitals |
| $W^{(2)}$, $W^{(3)}$ | the occupied matrices built from $c^{(1)}$ and $c^{(2)}$ |
| $K$, $P$ | the kept set of section @sec:amplitude-lmp2 and the projector onto it |
| $G$ | the closed-shell energy weight $2 - \mathcal S$ (written $\mathcal W$ in section @sec:amplitude-lmp2) |
| $\epsilon_p$ | orbital energies; $\varepsilon$ is the amplitude threshold |

## The equations in a localized basis {#local-opmp3-equations}

For a closed-shell reference with frozen core, store the alpha-beta
amplitude as $c_{iajb} = c^{\alpha\beta}_{iJaB}$ and write $J_{iajb} = (ia|jb)$.
The swap $(\mathcal S c)_{iajb} = c_{ibja}$ turns it into the same-spin part
$c - \mathcal S c$. In orbitals where the occupied and virtual Fock blocks
$f$ and $F$ need not be diagonal, the Fock superoperator is

$$ (\mathcal A c)_{iajb} = \sum_c F_{ac}\,c_{icjb} + \sum_c c_{iajc}\,F_{cb} - \sum_k f_{ik}\,c_{kajb} - \sum_k c_{iakb}\,f_{kj} . $$

The first- and second-order amplitudes solve

$$ \mathcal A c^{(1)} = -J, \qquad \mathcal A c^{(2)} = -R(c^{(1)}), \label{eq:local-opmp3-orders} $$

with $R$ the closed-shell MP3 residual: the alpha-beta block of the
residual of section @sec:mp3-u-residual-blocks (two ladders, the rings and
the exchange rings), with both spins carried by the one closed-shell
amplitude. In canonical orbitals $\mathcal A$ is
the diagonal $\Delta$ and both solves are divisions. The energies are
$E^{(n)} = \langle (2-\mathcal S)c^{(n-1)}, J\rangle$ and the alpha-block W
matrices are those of (@eq:opmp3-w-channels), which fixes the prefactor
$\tfrac14$ and the symmetrization in $i$ and $j$; for a closed shell the
same-spin block is built from $c - \mathcal S c$:

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
the anchor. The number is the prototype's `--mutate semicanonical` run,
recorded with the anchors of `scripts/prototypes/local_opmp3_derivation.md`
(smoke runs on propane and butane STO-3G, 2026-11-16); mutation `semicanonical` of the sympy script below is the same
defect on its model problem.

> **Verified equation @eq:local-opmp3-covariance** · OP-MP3 is invariant under the occupied and virtual localization
>
> | | |
> |---|---|
> | Proof | `scripts/local_opmp3_sympy.py` claims a1 (the trace Padé under a symbolic rotation) and a2 (the W covariance, every amplitude and integral an independent symbol), and N1 (canonical route vs a rotated route with the full Fock, OP-MP3 and SSD-OPMP3 agree to $3.3\times10^{-16}$); mutations `pade_half_rotated`, `w_virtual_once` and `semicanonical` each turn their check red |
> | Runner | `crates/ferric-python/tests/test_local_opmp3_sympy.py` |
> | Code | `ferric-mp2/src/local_opmp3.rs::local_opmp3_on` (prototype: `scripts/prototypes/local_opmp3_proto.py`) |
> | Code test | `crates/ferric-mp2/tests/local_opmp3.rs`: at $\varepsilon = 0$ every model, $E_{\mathrm{HF}}$, $E_{\mathrm{NBS}}$ and the $W$ spectra equal `opmp3_on` (same RI basis) to $10^{-10}$ Ha on the HF reference, the PBE0 reference and the PBE0 reference with a frozen core |
>
> <!-- verified: id=local-opmp3.invariance; script=scripts/local_opmp3_sympy.py; claim=a2; runner=crates/ferric-python/tests/test_local_opmp3_sympy.py; code=ferric-mp2/src/local_opmp3.rs::local_opmp3_on; rust_test=ferric-mp2/tests/local_opmp3.rs::local_opmp3_matches_exact_in_the_trivial_limit_pbe0; registry=local-opmp3 -->

## One mask for both orders {#local-opmp3-one-mask}

Keep the amplitudes of the LMP2 kept set $K$ (the Eq-8 mask of section
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
with $P$. Take the third-order energy $\langle Gc^{(2)}, J\rangle$. Since
$c^{(2)}$ lies in $K$ and $G$ commutes with $P$, $J$ may be replaced by
$PJ = -P\mathcal A P c^{(1)}$; move $\mathcal A$ to the other side, and
substitute $P\mathcal A P c^{(2)} = -PR$:

$$ E^{(3)} = \langle G c^{(2)}, J\rangle = -\langle G c^{(2)}, \mathcal A c^{(1)}\rangle = -\langle G c^{(1)}, \mathcal A c^{(2)}\rangle = \langle G c^{(1)}, R(c^{(1)})\rangle . \label{eq:local-opmp3-2n1} $$

The right-hand form needs only $c^{(1)}$, so the two sides are computed by
different code and checked against each other at every $\varepsilon$. Solve
$c^{(2)}$ on a second pattern $K_2 \ne K$ and the middle step fails: the
operator that maps $c^{(1)}$ to $J$ and the one that maps $c^{(2)}$ to $R$
are then different blocks of $\mathcal A$. Check the limit: at $K$ = everything
the identity reduces to the canonical $2n+1$ identity.

> **Verified equation @eq:local-opmp3-2n1** · the third-order energy keeps its 2n+1 form on one mask
>
> | | |
> |---|---|
> | Proof | `scripts/local_opmp3_sympy.py` claim b (symbolic, $G\mathcal A$ symmetric and $G$ block-diagonal on $K$), b2 (two masks: a rational counterexample, $-9/7$ against $3/11$) and N2 (closed-shell model at kept fraction 0.65); mutation `two_masks` turns b and N2 red ($-8.2\times10^{-2}$) |
> | Runner | `crates/ferric-python/tests/test_local_opmp3_sympy.py` |
> | Code | `ferric-mp2/src/local_opmp3.rs::local_opmp3_on` (`e3_2n1`, `e3_identity_gap`) |
> | Code test | `finite_eps_keeps_e_nbs_and_the_2n1_identity` in `crates/ferric-mp2/tests/local_opmp3.rs` (PBE0 propane, $\varepsilon = 10^{-4}$: the two routes agree to $10^{-10}$ Ha) |
>
> <!-- verified: id=local-opmp3.2n1; script=scripts/local_opmp3_sympy.py; claim=b; runner=crates/ferric-python/tests/test_local_opmp3_sympy.py; code=ferric-mp2/src/local_opmp3.rs::local_opmp3_on; rust_test=ferric-mp2/tests/local_opmp3.rs::finite_eps_keeps_e_nbs_and_the_2n1_identity; registry=local-opmp3 -->

## A Kohn-Sham reference {#local-opmp3-ks-reference}

OP-MP3 on a Kohn-Sham reference (section @sec:opmp3) uses the HF Fock
operator built from the Kohn-Sham density. Its occupied and virtual blocks
are diagonalized separately (semicanonical orbitals), and its
occupied-virtual block $F_{ia}$, which is not zero for a Kohn-Sham
density, enters only the non-Brillouin singles energy

$$ E_{\mathrm{NBS}} = -2\sum_{ia}\frac{F_{ia}^2}{\epsilon_a - \epsilon_i} \qquad\text{(semicanonical),} \label{eq:local-opmp3-nbs} $$

the closed-shell form of (@eq:opmp3-5).

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

> **Verified equation @eq:local-opmp3-nbs** · $E_{\mathrm{NBS}}$ is independent of the doubles mask and of the localization
>
> | | |
> |---|---|
> | Proof | `scripts/local_opmp3_sympy.py` claim N3 (semicanonical formula vs the localized singles solve, $-4.6\times10^{-2}$ Ha model value, agreement $< 10^{-14}$); mutation `nbs_local_diag` (keep only the diagonal of the localized Fock) turns it red |
> | Runner | `crates/ferric-python/tests/test_local_opmp3_sympy.py` |
> | Code | `ferric-mp2/src/local_opmp3.rs::local_opmp3_on` takes `Opmp3Reference::e_nbs` unchanged |
> | Code test | `finite_eps_keeps_e_nbs_and_the_2n1_identity` in `crates/ferric-mp2/tests/local_opmp3.rs` ($E_{\mathrm{NBS}}$ of the local run at $\varepsilon = 10^{-4}$ equals the canonical one bit for bit) |
>
> <!-- verified: id=local-opmp3.nbs; script=scripts/local_opmp3_sympy.py; claim=N3; runner=crates/ferric-python/tests/test_local_opmp3_sympy.py; code=ferric-mp2/src/local_opmp3.rs::local_opmp3_on; rust_test=ferric-mp2/tests/local_opmp3.rs::finite_eps_keeps_e_nbs_and_the_2n1_identity; registry=local-opmp3 -->

## What is proven and what is not {#local-opmp3-proven}

Proven, by the sympy script and its mutations: the invariance at
$\varepsilon = 0$, the $2n+1$ identity on one mask and its failure on two,
and the independence of $E_{\mathrm{NBS}}$ from the mask. Not proven: any
bound on the truncation error of the OP-type models. The error of
$E^{(2)}$ is bounded by the Schur norm of the dropped amplitudes (section
@sec:amplitude-lmp2-fock-screen). The Padé passes the errors of $W^{(2)}$
and $W^{(3)}$ through its first derivatives, and nothing here bounds those
derivatives away from a near-singular $A - B$ (a stretched bond).

## The implementation {#local-opmp3-implementation}

| Entry point | Call |
|---|---|
| CLI | `method.kind = "opmp3"` with `[local] scheme = "amplitude-threshold"` |
| Python | `run_opmp3(local="amplitude-threshold", eps=...)` |
| Library | `ferric_mp2::local_opmp3::local_opmp3_on` |

The `[local]` keys `opmp3` reads are `eps`, `reference`, `fock_screen`,
`lambda_boys` and `virtual_space` (with its `virtual_*` thresholds). It has
no integral-direct path, so `integral_direct` and the direct maps of
section @sec:integral-direct are errors, as is `eps_sweep`.

`local_opmp3_on` takes the OP-MP3 reference of
section @sec:opmp3 (HF or a functional), localizes against its HF Fock
matrix, builds the LMP2 pattern with the shared pair builder (the Eq-8 mask plus the
Fock-propagated screen, λ-Boys occupieds, VV-HV virtuals: the local MP2
maps and defaults), solves both orders on that one pattern with the ragged
LMP2 solver, and hands $W^{(2)}$, $W^{(3)}$ to the unchanged OP-MP3 models.
$E_{\mathrm{NBS}}$ and $E_{\mathrm{HF}}$ come from the reference unchanged.
RI integrals only, closed shell only. The two solves and the second
route to $E^{(3)}$:

<!-- excerpt: ferric-mp2/src/local_opmp3.rs::local_opmp3_on -->
```rust
    let rtol = local.cg_rtol.unwrap_or(LOCAL_OPMP3_CG_RTOL);
    let s1 = solve(
        &rg,
        &sp.f_oo,
        &gather(&rg, &k.ab, -1.0),
        rtol,
        local.cg_max_iter,
        "c1",
    )?;
    let c1 = closed_shell_doubles(scatter(&rg, &s1.t, no, nv));
    let res = mp3_residual(&src, ring, &c1)?;
    let s2 = solve(
        &rg,
        &sp.f_oo,
        &gather(&rg, &res.ab, -1.0),
        rtol,
        local.cg_max_iter,
        "c2",
    )?;
    let c2 = closed_shell_doubles(scatter(&rg, &s2.t, no, nv));
```

and, in the result, `e3_2n1: channel_energy(&c1, &res).total(),`.

| Code | Equation |
|---|---|
| `rg` (from `pattern`, swap closure checked) | the kept set $K$ and projector $P$ |
| `solve(.., &gather(&rg, &k.ab, -1.0), ..)` | $P\mathcal AP\,c^{(1)} = -PJ$ (@eq:local-opmp3-masked) |
| `mp3_residual(&src, ring, &c1)` | $R(c^{(1)})$ with untruncated integrals |
| `solve(.., &gather(&rg, &res.ab, -1.0), ..)` | $P\mathcal AP\,c^{(2)} = -PR$ |
| `channel_energy(&c1, &res)` | $\langle Gc^{(1)}, R(c^{(1)})\rangle$, the right side of (@eq:local-opmp3-2n1) |

The residual $R(c^{(1)})$ is the canonical spin-blocked MP3 residual
evaluated over the localized orbitals with dense amplitudes, of which only
the pattern part is kept. That makes the first version a correctness
reference, not a fast method: its cost is canonical RI-MP3 plus the two
pattern solves, and no cost or scaling claim attaches to it. A ragged
residual (ring kernels contracted through the RI factors, ladders through
`ferric_mp2::ladders`) is the next step and must reproduce this one.

### Measured {#local-opmp3-measured}

Measured 2026-11-19, provisional. The script is
`scripts/prototypes/local_opmp3_data/v2/rust_sweep.py`, its output
`rust_sweep.json` beside it. It runs `ferric.run_opmp3` at 6-31G with
cc-pvdz-ri, one frozen core orbital per heavy atom, and SCF converged to
1e-12 Ha. Each error is the local model minus the canonical RI OP-MP3 of the
same reference, in Ha. "Default" means the maps of section
@sec:local-opmp3-implementation: Fock screen κ = 1 and λ-Boys with λ = 10.
"Plain" means `fock_screen = false` and `lambda_boys = 0`. The PBE0 rows use
the HF Fock of the PBE0 density (section @sec:local-opmp3-ks-reference).

| system | reference | maps | $\varepsilon$ | keep | $\Delta E^{(2)}$ | $\Delta E^{(3)}$ | $\Delta$OP-MP3 |
|---|---|---|---|---|---|---|---|
| n-octane | HF | plain | 1e-5 | 0.390 | +8.6e-5 | −1.0e-5 | +7.4e-5 |
| n-octane | HF | default | 1e-5 | 0.441 | +2.1e-5 | −2.7e-6 | +1.8e-5 |
| n-octane | HF | plain | 1e-6 | 0.746 | +2.1e-6 | −1.5e-7 | +2.0e-6 |
| n-octane | HF | default | 1e-6 | 0.793 | +1.9e-7 | −1.5e-8 | +1.7e-7 |
| n-octane | PBE0 | plain | 1e-5 | 0.416 | +7.8e-5 | −9.4e-6 | +6.6e-5 |
| n-octane | PBE0 | default | 1e-5 | 0.468 | +2.2e-5 | −3.2e-6 | +1.8e-5 |
| n-octane | PBE0 | plain | 1e-6 | 0.781 | +2.7e-6 | −5.3e-7 | +2.0e-6 |
| n-octane | PBE0 | default | 1e-6 | 0.824 | +1.8e-7 | −1.4e-8 | +1.7e-7 |
| naphthalene | HF | plain | 1e-5 | 0.629 | +1.4e-4 | −2.2e-5 | +1.2e-4 |
| naphthalene | HF | default | 1e-5 | 0.540 | +1.7e-5 | −7.0e-6 | +9.8e-6 |
| naphthalene | HF | plain | 1e-6 | 0.915 | +2.3e-6 | −1.0e-6 | +1.5e-6 |
| naphthalene | HF | default | 1e-6 | 0.826 | +1.2e-7 | −7.3e-8 | +5.0e-8 |
| naphthalene | PBE0 | plain | 1e-5 | 0.658 | +2.0e-4 | −3.6e-5 | +1.6e-4 |
| naphthalene | PBE0 | default | 1e-5 | 0.557 | +1.6e-5 | −7.5e-6 | +9.1e-6 |
| naphthalene | PBE0 | plain | 1e-6 | 0.925 | +2.5e-5 | −9.6e-6 | +1.6e-5 |
| naphthalene | PBE0 | default | 1e-6 | 0.825 | +1.1e-7 | −4.6e-8 | +7.2e-8 |

What the table shows:

- **The default maps.** At the same $\varepsilon$ the default maps leave
  3.7–12× less OP-MP3 error than the plain ones on octane and 12–230× less on
  naphthalene. On octane they keep 5 percentage points more amplitudes. On naphthalene
  they keep fewer, because λ-Boys rotates the π orbitals (max $||U|-1|$ =
  0.71 in the prototype) and the Eq-8 mask shrinks in that basis.
- **The third order follows the second.** $\Delta E^{(3)}$ has the opposite
  sign to $\Delta E^{(2)}$ in every row and is 8–60% of it, so the OP-MP3
  error is smaller than the $E^{(2)}$ error. The maps reduce both by about
  the same factor.
- **HF and PBE0.** With the default maps the PBE0 OP-MP3 errors are within a
  factor 1.5 of the HF ones.
- **The identity and the reference terms.** On all 48 runs of the sweep
  (butane, octane and naphthalene, $\varepsilon$ = 1e-4 to 1e-7, both maps,
  both references) $|E^{(3)} - E^{(3)}_{2n+1}| \le 6.4\times10^{-14}$,
  $E_{\mathrm{NBS}}$ equals the canonical value bit for bit, and both
  pattern solves reach the 1e-11 relative residual in at most 29 iterations.
- **The prototype.** The numpy prototype
  (`scripts/prototypes/local_opmp3_proto.py`, PySCF integrals and its own
  localization) gives the same kept fraction to within 0.002 in all 48
  cases. With the default maps its errors agree with these to 5% wherever
  the error is above 1e-6 Ha; smaller errors differ by up to a factor 10.
  With the plain maps the agreement is 10% at $\varepsilon \ge 10^{-5}$ and
  up to a factor 3 at $10^{-6}$. One plain case is unexplained: naphthalene
  PBE0 at $\varepsilon = 10^{-7}$ gives $\Delta E^{(2)}$ = +2.1e-6 here
  against +4.1e-8 in the prototype. The plain naphthalene PBE0 row at
  $10^{-6}$ in the table (+2.5e-5, ten times the HF row) looks like the
  same effect. It touches only the plain maps, which are not the default,
  so the plain columns are provisional until it is explained; the default
  columns are not affected.
- **Cost.** The local run takes longer than the canonical one (octane 20 s
  against 14 s), as section @sec:local-opmp3-implementation predicts for a
  dense residual.

## Sources {#local-opmp3-sources}

| Source | What it supplies |
|---|---|
| Section @sec:opmp3 (@eq:opmp3-w-channels, @eq:opmp3-pade, @eq:opmp3-5) | the W matrices, the Padé models and $E_{\mathrm{NBS}}$ |
| Section @sec:mp3-u-residual-blocks | the MP3 residual $R$ |
| `scripts/local_opmp3_sympy.py`, claims a1, a2, b, b2, N1, N2, N3 | verified equations @eq:local-opmp3-covariance, @eq:local-opmp3-2n1 and @eq:local-opmp3-nbs |
| `scripts/prototypes/local_opmp3_proto.py`, `local_opmp3_derivation.md` | the numpy prototype, the semicanonical-Fock error |
| `crates/ferric-mp2/src/local_opmp3.rs` and `tests/local_opmp3.rs` | the implementation, the trivial-limit anchor and the finite-$\varepsilon$ identity test |
| `scripts/prototypes/local_opmp3_data/v2/rust_sweep.py`, `rust_sweep.json`, measured 2026-11-19 | the measured table |
