# Amplitude-threshold local MP2 {#amplitude-lmp2}

<!-- status: ready; registry: local-mp2 -->

The MP2 amplitude $t_{ij}^{ab}$ measures how much two electrons in orbitals
$i$ and $j$ correlate by jumping together into $a$ and $b$. What drives the
jump is the integral $(ia|jb)$, the Coulomb interaction of the charge
distributions $\phi_i\phi_a$ and $\phi_j\phi_b$. In canonical orbitals
every orbital spreads over the whole molecule, so almost every $(ia|jb)$ is
nonzero and canonical MP2 has to sum all of them.

Localize the orbitals and that changes. Each $\phi_i$ now sits on a bond or
a lone pair, and so does each local virtual $\phi_a$. If $a$ is far from
$i$, the product $\phi_i\phi_a$ is close to zero everywhere, so $(ia|jb)$ is
too. In a long molecule most $(i,a)$ combinations are far apart, and most
amplitudes are negligible.

The amplitude-threshold scheme of Wang, Aldossary, Shi, Liu, Li and
Head-Gordon (J. Chem. Theory Comput. 19, 7577, 2023) acts on this directly.
It keeps an amplitude only when the integral that drives it exceeds one
threshold $\varepsilon$, and solves the MP2 equations for the kept set only.
Three facts frame the rest of the section:

- $\varepsilon = 0$ keeps everything and gives canonical RI-MP2 exactly.
- The energy error has one sign: the truncated energy is never below the
  canonical one (§@sec:amplitude-lmp2-amplitude-equations-energy shows why).
- `eps` has no default. It is part of the model, like the basis set.

Notation:

| Symbol | Meaning | In the code |
|---|---|---|
| $i, j, k$ | active occupied orbitals, Boys-localized | `no` of them; `LocalizedSpaces::c_locc` |
| $a, b$ | virtual orbitals of the VV-HV localized space | `nv` of them; `VvHv::c_vloc` |
| $t_{ij}^{ab}$ | first-order amplitudes | one `Array2` block per ordered pair, `PairBlock` |
| $J_{ia,jb}$ | $(ia\vert jb)$ in the localized basis | `PairBlock::j_blk` |
| $f_{ik}$, $F_{ab}$ | occupied and virtual blocks of the Fock matrix in the localized bases; **not diagonal** | `f_oo`, `f_vv` |
| $\Delta_{ij}^{ab}$ | $F_{aa}+F_{bb}-f_{ii}-f_{jj}$, positive | `PairBlock::denom` |
| $\varepsilon$ | the threshold | `AmplitudeLmp2Config::eps` |

Entry points:
- Rust: `ferric-mp2/src/lmp2_amplitude.rs::amplitude_lmp2`. Use
  `ferric-mp2/src/lmp2_amplitude.rs::amplitude_lmp2_with_virtuals` when you
  supply your own virtual space.
- CLI: `method.kind = "rimp2"` with `[local] scheme = "amplitude-threshold"`
  and `eps = ...` (`examples/water-rimp2-local.toml`).
- Python: `run_rimp2(..., local="amplitude-threshold", eps=...)`.

The method is closed shell and computes energies only.

## The amplitude equations and the energy {#amplitude-lmp2-amplitude-equations-energy}

In canonical orbitals MP2 needs no equations: $t_{ij}^{ab} = -J_{ia,jb}/\Delta_{ij}^{ab}$,
one division per amplitude. Localized orbitals are not eigenfunctions of the
Fock operator, so this shortcut fails. The Fock matrix in the local basis
has off-diagonal elements, and they couple amplitudes to each other.

To see the coupling, write the first-order condition for pair $(i,j)$ as a
matrix in $a,b$. The Fock operator acts on each of the four orbital labels:

- on the particle $a$: $F_{ac}$ moves amplitude from $c$ to $a$, giving
  $F\,t_{ij}$;
- on the particle $b$: likewise, giving $t_{ij}\,F$;
- on the hole $i$: $f_{ik}$ moves the hole from $k$ to $i$, so pair
  $(i,j)$ receives amplitude from pair $(k,j)$;
- on the hole $j$: likewise, from $(i,k)$.

Holes enter with the opposite sign to particles. The residual is

$$ (\mathcal A\,t)_{ij} = F\,t_{ij} + t_{ij}\,F - \sum_k\big(f_{ik}\,t_{kj} + f_{kj}\,t_{ik}\big), \qquad
R = J + \mathcal A\,t = 0, \qquad t^{*} = -\mathcal A^{-1}J . \label{eq:amplitude-lmp2-1} $$

Check the canonical limit. With $F$ and $f$ diagonal, $\mathcal A$
multiplies each amplitude by
$F_{aa} + F_{bb} - f_{ii} - f_{jj} = \Delta_{ij}^{ab}$, and $R = 0$ gives back
$t = -J/\Delta$ and the energy of equation (@eq:closed-shell-ri-mp2-3).

The code applies $\mathcal A$ without ever forming it. This is the body of
`ferric-mp2/src/ragged.rs::matvec_indexed`, run in parallel over output
pairs; only the flop counter has been removed:

```rust
let mut r = pb.fvv_aa.dot(&t[p_idx]);        // F t_ij     (particle a)
r += &t[p_idx].dot(&pb.fvv_bb);              //   + t_ij F (particle b)
for &q in &rg.by_j[&pb.j] {                  // pairs (k, j): same j
    let f = f_oo[(pb.i, rg.pairs[q].i)];     //   f_ik
    if f == 0.0 { continue; }
    gather_into(&mut r, &rg.pairs[q], pb, -f, &t[q], ..);   // - f_ik t_kj
}
for &q in &rg.by_i[&pb.i] {                  // pairs (i, k): same i
    let f = f_oo[(rg.pairs[q].j, pb.j)];     //   f_kj
    if f == 0.0 { continue; }
    gather_into(&mut r, &rg.pairs[q], pb, -f, &t[q], ..);   // - f_kj t_ik
}
apply_pattern(&mut r, &pb.pat, nb);          // zero the dropped amplitudes
```

The four lines of (@eq:amplitude-lmp2-1) map onto these lines one for one. Two details are
not in the equation:

- **Pair blocks have different shapes.** Pair $(i,j)$ only stores the
  virtuals in its own domain, `da` × `db`. `gather_into` moves amplitudes
  between two blocks whose virtual lists overlap only in part, matching
  them by index. "Ragged" in the file name refers to these unequal blocks.
- **`apply_pattern` makes the truncation exact.** The operator is projected
  onto the kept amplitudes. Dropped amplitudes are not approximated; they
  are zero, and the equations hold exactly on what remains.

### Two energy formulas, and why the code uses the longer one

With $\tilde t = (2-\mathcal S)\,t$, where $\mathcal S$ swaps
$t^{ab}_{ij} \leftrightarrow t^{ba}_{ij}$ (the same $2(ia|jb) - (ib|ja)$
combination as in equation @eq:closed-shell-ri-mp2-3), there are two ways to write the energy:

$$ E_{\mathrm{lin}}(t) = \langle\tilde t,\,J\rangle = \sum_{ijab}\big(2t_{ij}^{ab} - t_{ij}^{ba}\big)J_{ia,jb}, \qquad
E_H(t) = \langle\tilde t,\;2J + \mathcal A\,t\rangle . \label{eq:lmp2-hylleraas} $$

At the exact solution they agree, since $\mathcal A t^* = -J$ turns
$2J + \mathcal A t^*$ into $J$. They differ in how they respond to an
amplitude that is slightly wrong, and an iterative solver only ever returns
slightly wrong amplitudes.

Put $t = t^* + d$, where $d$ is the solver's error, and expand. For
$E_{\mathrm{lin}}$ this is one line:

$$ E_{\mathrm{lin}}(t^*+d) = E^* + \langle (2-\mathcal S)\,d,\; J\rangle . $$

The error is linear in $d$. For $E_H$, expand the product. Two cross terms
appear: $\langle (2-\mathcal S)t^*, \mathcal A d\rangle$ and
$\langle (2-\mathcal S)d, J\rangle$. Move $\mathcal A$ to the left,
which needs $\mathcal A$ symmetric and commuting with $\mathcal S$, and use
$\mathcal A t^* = -J$. The first cross term becomes
$-\langle (2-\mathcal S) J, d\rangle$, which cancels the second. What is
left is

$$ E_H(t^*+d) = E^* + \langle (2-\mathcal S)\,d,\; \mathcal A\,d\rangle . $$

The error is quadratic in $d$. An amplitude error of order $\delta$ gives
an $E_H$ error of order $\delta^2$ and an $E_{\mathrm{lin}}$ error of order
$\delta$. That is why the code reports $E_H$. It keeps $E_{\mathrm{lin}}$ as an independent check of the
same solution.

The same expansion gives the sign of the truncation error. $(2-\mathcal S)\mathcal A$
is positive definite, so $E_H$ is a convex quadratic whose minimum over all
amplitudes is the canonical energy $E^*$. Truncation minimizes $E_H$ over a
subset of the amplitudes, and a minimum over fewer variables cannot be
lower. So $E_{\varepsilon} \ge E^*$ for every $\varepsilon$.

> **Verified equation @eq:lmp2-hylleraas** · the two energy forms and the stationarity of $E_H$
>
> Along a line $t = t^* + s\,d$ through the solution, $E_H$ has no term linear in
> $s$ while $E_{\mathrm{lin}}$ does, and $E_H(t^*) = E_{\mathrm{lin}}(t^*)$ (claim 1).
> "Residual folded in" is the identity $E_{\mathrm{lin}} + \langle\tilde t,R\rangle = \langle\tilde t, 2J+\mathcal At\rangle$
> (claim 2). On a one-pair model with a swap-closed retained pattern the
> stationarity holds, and on a pattern that keeps $(a,b)$ but drops its
> partner $(b,a)$ it does not: a first-order term survives (claim 5). The
> sign of the exchange term in the code's energy sum is pinned to
> $\sum(2t_{ab}-t_{ba})J_{ab}$, since $2+\mathcal S$ has the same spectrum and would pass
> the rest (claim 4).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_hylleraas_vs_wshg23_sympy.py`, claims 1, 2, 4 and 5 |
> | Negative control | the runner flips the exchange sign of the transcribed energy sum (claim 4) and drops the 2 on $J$ in the Hylleraas form so a closed mask has a first-order term (claim 5) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/lmp2_amplitude.rs::{hylleraas_energy,hylleraas_energy_and_residual,linear_energy}`, `ferric-mp2/src/ragged.rs::check_swap_closed` |
> | Code test | `hylleraas_is_stationary` in `crates/ferric-mp2/tests/lmp2_direct.rs` |
>
> <!-- verified: id=lmp2.hylleraas; script=scripts/lmp2_hylleraas_vs_wshg23_sympy.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/lmp2_amplitude.rs::{hylleraas_energy,hylleraas_energy_and_residual,linear_energy},ferric-mp2/src/ragged.rs::check_swap_closed; rust_test=ferric-mp2/tests/lmp2_direct.rs::hylleraas_is_stationary -->

### The kept set must be closed under the swap

The cancellation above assumed that
$\mathcal S$ commutes with the projection onto the kept amplitudes. Suppose
the mask keeps $t^{ab}_{ij}$ but drops its partner $t^{ba}_{ij}$. Then
$\mathcal S$ maps a kept amplitude to a dropped one, the two cross terms no
longer cancel, and a term linear in $d$ comes back. The energy error goes
from second order to first. Claim 5 of the proof script builds exactly this
case and shows the linear term.

So the retention test has to be symmetric. The test in §@sec:amplitude-lmp2-algorithm is symmetric
by construction, and the code does not rely on that alone:
`check_swap_closed` asserts the property, and `hylleraas_energy` calls it.
On a one-sided pattern it returns an error, never a number
(`hylleraas_energy_rejects_a_one_sided_pattern` in
`crates/ferric-mp2/tests/lmp2_direct.rs`).

## How accurate is the solve? {#amplitude-lmp2-accurate-solve}

The amplitudes come from preconditioned conjugate gradients on
$\mathcal A t = -J$, with the diagonal $\Delta_{ij}^{ab}$ as preconditioner.
CG stops at some residual $R = J + \mathcal A t$. The question is how much
energy error that leaves. The expansion above already answers it.

Since $R = J + \mathcal A(t^* + d) = \mathcal A d$, the error is
$d = \mathcal A^{-1}R$. Substitute into the $E_H$ expansion:

$$ E_H(t) - E^{*} = \langle(2-\mathcal S)\,\mathcal A^{-1}R,\;R\rangle \;\le\; \frac{3\,\lVert R\rVert^2}{\lambda_{\min}(\mathcal A)}, \qquad
\lambda_{\min}(\mathcal A) \ge 2\,(\epsilon_{\mathrm{LUMO}} - \epsilon_{\mathrm{HOMO}}) . \label{eq:lmp2-cg-bound} $$

There are three steps from the identity to the bound:

1. $\lVert 2-\mathcal S\rVert = 3$: $\mathcal S$ is a swap, so its
   eigenvalues are $\pm 1$, and $2-\mathcal S$ has eigenvalues 1 and 3.
2. $\lVert \mathcal A^{-1}\rVert = 1/\lambda_{\min}(\mathcal A)$.
3. Bound $\lambda_{\min}$ without computing it. Before projection,
   $\mathcal A$ is a sum of Fock operators acting on separate indices, so
   its eigenvalues are $\epsilon_a+\epsilon_b-\epsilon_i-\epsilon_j$, with
   the $\epsilon$ taken from diagonalizing `f_vv` and `f_oo`. The smallest is
   $2(\epsilon_{\mathrm{LUMO}} - \epsilon_{\mathrm{HOMO}})$. Projecting onto
   the kept amplitudes compresses the operator, and by Cauchy interlacing a
   compression's smallest eigenvalue can only be larger.

The bound needs only the orbital gap, which is cheap and guaranteed. The CG
iteration also produces Lanczos estimates of $\lambda_{\min}$ for free, but
they converge to it from above. Dividing by an overestimate would make the
"bound" too small, so the code does not use them.

> **Verified equation @eq:lmp2-cg-bound** · the solver-error bound
>
> For a one-pair, two-virtual model with $\mathcal A$ symmetric positive
> definite, $\mathcal S$ commuting with $\mathcal A$ and $(2-\mathcal S)\mathcal A$ symmetric:
> $E_H(t^*+d) - E^* = \langle(2-\mathcal S)\mathcal A^{-1}R,R\rangle$ and
> $\lVert R\rVert^2/\lambda_{\max} \le E_H - E^* \le 3\lVert R\rVert^2/\lambda_{\min}$,
> checked on five directions $d$ of a concrete rational model
> (claim 6, which uses the structure of claim 4).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_hylleraas_vs_wshg23_sympy.py`, claim 6 |
> | Negative control | none for this claim; the claim-4 structure checks it relies on carry the controls of equation @eq:lmp2-hylleraas |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/lmp2_amplitude.rs::{cg_energy_error_bound,operator_lambda_min_lower_bound}` |
> | Code test | `default_rtol_energy_is_within_its_bound` in `crates/ferric-mp2/tests/lmp2_direct.rs` |
>
> <!-- verified: id=lmp2.cg-bound; script=scripts/lmp2_hylleraas_vs_wshg23_sympy.py; claim=6; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/lmp2_amplitude.rs::{cg_energy_error_bound,operator_lambda_min_lower_bound}; rust_test=ferric-mp2/tests/lmp2_direct.rs::default_rtol_energy_is_within_its_bound -->

What is proven and what is not: the script proves the identity and the bound
on a small model. It does not prove the interlacing step for the production
operator. The code applies that step with the gap of the localized-space
Fock blocks, and the code test checks the reported bound against a dense
solve.

**How the code uses the bound.**

1. The solve first stops at a relative residual of $10^{-6}$
   (`LMP2_CG_RTOL_DEFAULT`) and evaluates (@eq:lmp2-cg-bound).
2. If the bound is above $10^{-10}$ Ha (`LMP2_CG_ENERGY_ERROR_TARGET`), CG
   continues on the same Krylov sequence, without restarting. It stops at
   the residual that puts the bound at a quarter of the target, but never
   goes below $10^{-11}$ (`LMP2_CG_RTOL_STRICT`, the floor). That residual is
   `ferric-mp2/src/lmp2_amplitude.rs::tightened_cg_rtol`:

   $$ r_{\mathrm{rel}} = \max\Big(\tfrac12\sqrt{\lambda_{\min}\,10^{-10}/3}\;\big/\;\lVert J\rVert,\ 10^{-11}\Big), $$

   with the factor $\tfrac12$ the constant `CG_TIGHTEN_SAFETY`. The bound
   the code evaluates is
   `ferric-mp2/src/lmp2_amplitude.rs::cg_energy_error_bound`,
   $3(\lVert J\rVert\,r_{\mathrm{rel}})^2/\lambda_{\min}$, which is
   (@eq:lmp2-cg-bound) with $\lVert R\rVert = \lVert J\rVert\,r_{\mathrm{rel}}$.
   Substitute the residual above: the bound is
   $3 \cdot \tfrac14 \cdot \lambda_{\min}10^{-10}/3 \,/\, \lambda_{\min} = \tfrac14\,10^{-10}$ Ha,
   a quarter of the target. The policy is
   `ferric-mp2/src/lmp2_amplitude.rs::solve_under_rtol_policy`.
3. `ferric-mp2/src/lmp2_amplitude.rs::operator_lambda_min_lower_bound`
   returns `NaN` when the gap $2(\epsilon_{\mathrm{LUMO}} - \epsilon_{\mathrm{HOMO}})$
   from the eigenvalues of `f_oo` and `f_vv` is not positive. The bound is
   then undefined, and the solve runs at $10^{-11}$.
4. An explicit `cg_rtol` disables the tightening: CG stops at that value.

Two details make the bound reliable:

- $R$ is the true residual $\lVert J+\mathcal At\rVert$, taken from the
  energy's own matrix-vector product. It is not the residual CG updates
  recursively, which can drift from the true one.
- Running out of iterations at either stage is an error, not a warning.

The bound covers the solver only. It says nothing about the error from
dropping amplitudes with $\varepsilon$, which is the subject of the rest of
this chapter.

## The algorithm {#amplitude-lmp2-algorithm}

Sections @sec:amplitude-lmp2-amplitude-equations-energy and @sec:amplitude-lmp2-accurate-solve solved the equations for a given set of kept
amplitudes. This section builds that set. Three things have to exist first:
occupied and virtual orbitals that are each confined to a small region,
the Fock matrix in those orbitals, and the integrals $J_{ia,jb}$. Then the
code decides which amplitudes to keep, in two stages: a cheap test on whole
pairs before any integral is formed, and the $\varepsilon$ test on each
element.

| Step | Quantity | Function |
|---|---|---|
| 1 | RHF reference: canonical orbitals and Fock matrix | `ferric-scf/src/rhf.rs::solve_rhf` |
| 2 | virtual space: valence virtuals from projected minimal-basis functions, hard virtuals from the rest | `ferric-mp2/src/lmp2_amplitude.rs::build_vvhv` |
| 3 | Boys-localized active occupieds, their centroids and spreads, and the localized Fock blocks $f$ and $F$ | `ferric-mp2/src/lmp2_amplitude.rs::localized_spaces` |
| 4 | three-index factors in the localized basis, and the metric | `ferric-mp2/src/lmp2_amplitude.rs::assemble_basis` |
| 5 | integral-free pair gate | `ferric-mp2/src/lmp2_amplitude.rs::pair_gate_keep` |
| 6 | per unique pair: $J_{ij}$ by a Gram product of whitened factors over Schwarz-screened candidates, or a domain-local fit; the $\varepsilon$ mask; the ragged block | `ferric-mp2/src/lmp2_amplitude.rs::assemble_ragged_direct_aux`, `ferric-mp2/src/ragged.rs::pair_block_from_g_cand` |
| 7 | PCG on the ragged blocks under the tolerance policy | `ferric-mp2/src/lmp2_amplitude.rs::solve_under_rtol_policy` |
| 8 | $E_H$ and the true residual | `ferric-mp2/src/lmp2_amplitude.rs::hylleraas_energy_and_residual` |

### Step 2: why the virtuals must be localized too

Localizing the occupied orbitals alone does not make $J$ sparse. The
product $\phi_i\phi_a$ is small only when $i$ and $a$ sit far apart, and a
canonical virtual sits everywhere. With canonical virtuals, a pair $(i,j)$
couples to every $a$ and $b$, and the pair block stays full. So the
virtuals also have to be built as functions on bonds and atoms. That is
the VV-HV space: valence virtuals (VV) and hard virtuals (HV), the
construction of Wang et al.

**Valence virtuals.** A minimal basis has one function per core and
valence atomic orbital. Its span holds the occupied space plus the
antibonding partners of the bonds, so $n_{\mathrm{val}} = n_{\mathrm{min}} - n_{\mathrm{occ}}$
of its directions are virtual. The code builds them in four operations:

1. Project the STO-3G functions into the working basis. With $S$ the
   working-basis overlap and $S_x$ the cross overlap between the two
   bases (taken from one combined basis set), the projected functions are
   the columns of $T = S^{-1}S_x$.
2. Remove the occupied part, with the projector
   $P_{\mathrm{occ}} = C_{\mathrm{occ}}C_{\mathrm{occ}}^{\mathsf T}S$ over
   all $n_{\mathrm{occ}}$ occupied orbitals: $T_v = (1 - P_{\mathrm{occ}})\,T$.
3. Orthonormalize $T_v$ canonically, keeping $n_{\mathrm{val}}$ columns.
4. Boys-localize them.

The count $n_{\mathrm{min}} - n_{\mathrm{occ}}$ uses an all-electron
minimal basis, so it is wrong on an atom whose core is an ECP. The code
refuses such a molecule (`refuse_ecp_atoms`) instead of counting core
functions as valence virtuals.

**Hard virtuals.** The remaining
$n_{\mathrm{AO}} - n_{\mathrm{occ}} - n_{\mathrm{val}}$ virtuals carry
polarization and diffuse character. Each one is grown from one atomic
orbital, so it starts on one atom:

1. Take every unit AO vector and project out the occupied and valence
   parts, $X = 1 - P_{\mathrm{occ+val}}$. Drop the columns whose squared
   norm in the $S$ metric is at most $10^{-8}$, and normalize the rest.
2. Weight each candidate $k$ by $w_k = 1/\sigma_k^2$, where
   $\sigma_k^2 = \langle r^2\rangle_k - \lvert\langle\mathbf r\rangle_k\rvert^2$
   is its spatial variance (floored at $10^{-6}$), and form the weighted
   overlap $w_rw_c\,S^X_{rc}/w_{\max}^2$.
3. Choose $n_{\mathrm{h}}$ candidates by pivoted Cholesky on that matrix.
   The first pivot is the largest diagonal, which is the most compact
   candidate. Each later pivot is the candidate with the largest weighted
   norm left after the chosen ones are projected out, so the selection
   favours compact functions that add a new direction.
4. Löwdin-orthonormalize the chosen candidates.
5. Pseudo-canonicalize per atom: for each atom $A$, diagonalize the Fock
   block $C_{h,A}^{\mathsf T}FC_{h,A}$ of the hard virtuals grown from $A$'s
   AOs and rotate them among themselves. A rotation inside one atom's set
   cannot move a function onto another atom, which a full
   semicanonicalization would do.

This selection differs from the paper, which orthogonalizes the full
redundant candidate set with a weighted symmetric orthogonalization. The
$n_{\mathrm{val}} + n_{\mathrm{h}}$ functions are orthonormal, orthogonal to
the occupied space and $n_{\mathrm{AO}} - n_{\mathrm{occ}}$ in number, so
they span the whole virtual space whichever candidates are chosen. The
$\varepsilon = 0$ energy is therefore invariant to the selection, and only
the compactness of the pair domains depends on it. The anchor test of
@sec:amplitude-lmp2-numerical-anchors checks the spanning: dropping one hard virtual breaks the
$\varepsilon = 0$ agreement (`mutated_virtual_space_fails_the_anchor`).

### Step 3: localized occupieds and their sizes

`localized_spaces` Boys-localizes the active occupied orbitals and records,
for each, its centroid $\langle\mathbf r\rangle_i$ and spread
$\sigma_i = \sqrt{\langle r^2\rangle_i - \lvert\langle\mathbf r\rangle_i\rvert^2}$.
It also forms the two Fock blocks of @sec:amplitude-lmp2-amplitude-equations-energy, $f = C_{\mathrm{locc}}^{\mathsf T}FC_{\mathrm{locc}}$
and $F = C_{\mathrm{vloc}}^{\mathsf T}FC_{\mathrm{vloc}}$, and the dipole
centroids of the virtuals, which only the integral-direct path of 13.2
uses. No RI work happens here.

### Step 5: skipping whole pairs before any integral

The correlation energy of two well-separated localized orbitals behaves
like a London dispersion energy. It decays as $R_{ij}^{-6}$, with a
prefactor that grows with the orbitals' volumes, for which
$\sigma_i^3\sigma_j^3$ stands in. That estimate costs nothing: it needs only
centroids and spreads. An off-diagonal pair $(i,j)$ is skipped when

$$ \mathrm{cal}\,\frac{(\sigma_i\sigma_j)^3}{R_{ij}^6} < 10^{-2}\,\varepsilon , \label{eq:amplitude-lmp2-4} $$

with $R_{ij}$ the distance between Boys centroids. The code guards the
denominator as $\max(R_{ij}^6, 10^{-12})$. The constant `cal` is
`pair_gate_cal`. The calibrated values from the alkane series are about
0.7 for the Coulomb operator and about 0.02 for erfc at $\omega = 1$.

This is the body of `ferric-mp2/src/lmp2_amplitude.rs::pair_gate_keep_with`,
with the declarations and the gated-pair counter removed:

```rust
let theta = 1e-2 * eps;                      // threshold 10^-2 ε
for i in 0..no {
    for j in 0..no {
        if i == j {
            continue;                        // diagonal pairs are never gated
        }
        let rij2: f64 = rij2(i, j);          // R_ij^2
        let est = cal * (occ_spreads[i] * occ_spreads[j]).powi(3) / rij2.powi(3).max(1e-12);
        if est < theta {                     // cal (σ_i σ_j)^3 / R_ij^6 < 10^-2 ε
            keep[i * no + j] = false;
        }
    }
}
```

Check the limits:

- At $\varepsilon = 0$ the threshold is 0, and the estimate is never
  negative, so no pair is gated.
- The estimate is symmetric in $i$ and $j$, so $(i,j)$ and $(j,i)$ are kept
  or dropped together.
- The gate is off when `pair_gate_cal` is `None`, the default. A
  calibration of 0 would make every off-diagonal estimate 0, which is below
  any positive threshold. The CLI accepts only `gate_cal > 0`.

The gate only removes amplitudes. It never changes a kept $J$ value. By the
argument at the end of @sec:amplitude-lmp2-amplitude-equations-energy, minimizing $E_H$ over fewer amplitudes
cannot give a lower energy, so the gate can only raise the energy. The
estimate itself is calibrated, not a bound: nothing ties the size of the
gated energy to $\varepsilon$.

### Step 6: the integrals and the Eq-8 mask

For each pair that survives the gate, the code needs the block
$J_{ij} = [(ia\vert jb)]_{ab}$. On the default path it is a Gram product of
the whitened three-index factors of section @sec:closed-shell-ri-mp2,
$J_{ia,jb} = \sum_P\tilde B_{P,ia}\tilde B_{P,jb}$ with
$\tilde B = V^{-1/2}B$.

**Schwarz candidates.** Most rows of that block will be dropped by the
$\varepsilon$ test, so the code first finds the virtuals that can appear at
all. Let $q_{ia} = \lVert\tilde B_{:,ia}\rVert$ and
$q^{\max}_j = \max_b q_{jb}$. The Cauchy–Schwarz inequality on the sum over
$P$ gives

$$ \lvert J_{ia,jb}\rvert \le q_{ia}\,q_{jb} \le q_{ia}\,q^{\max}_j . $$

The code keeps virtual $a$ as a candidate when
$q_{ia}q^{\max}_j \ge \varepsilon$ or $q_{ja}q^{\max}_i \ge \varepsilon$.
This loses no retained element. Suppose element $(a,b)$ passes the test
below through $\lvert J_{ia,jb}\rvert > \varepsilon$. Then
$q_{ia}q^{\max}_j \ge q_{ia}q_{jb} > \varepsilon$, so $a$ is a candidate,
and $q_{jb}q^{\max}_i \ge q_{ia}q_{jb} > \varepsilon$, so $b$ is one too.
The case $\lvert J_{ib,ja}\rvert > \varepsilon$ follows with $a$ and $b$
exchanged. The pair's integrals are then formed over the candidate square
only. The screen runs only on the global-fit path with $\varepsilon > 0$;
the domain-local fit (`fit_radius_bohr`, section @sec:integral-direct) forms every column.

**The Eq-8 mask.** Element $(a,b)$ of pair $(i,j)$ is kept when
$\lvert J_{ia,jb}\rvert > \varepsilon$ or $\lvert J_{ib,ja}\rvert > \varepsilon$,
or when $\varepsilon = 0$. The second integral is the exchange partner of
the first, the one the same-spin energy pairs it with. Testing both makes
the rule symmetric in $(a,b)$, so the kept set is closed under the swap
$\mathcal S$ by construction, which is what the stationarity of @sec:amplitude-lmp2-amplitude-equations-energy
requires. A pair block holds the rows and columns that have any kept
element. Those are the pair domain, the same for rows and columns on a
swap-closed pattern, and `check_swap_closed` requires it. The kept values
are the exact $J$ values, so there is no second approximation inside the
block.

The mask is the first loop of the block construction. The excerpt is
copied unchanged:

<!-- excerpt: ferric-mp2/src/ragged.rs::pair_block_from_g_cand_gated -->
```rust
if eps == 0.0 || jd > eps || kd > eps {
    any_a[a] = true;
    any_b[b] = true;
    any = true;
}
```

| Code | Equation |
|---|---|
| `jd` | $\lvert J_{ia,jb}\rvert$, from the gate quantity `gate[(a, b)]` |
| `kd` | $\lvert J_{ib,ja}\rvert$, the same block transposed, `gate[(b, a)]` |
| `eps` | $\varepsilon$ |
| `any_a`, `any_b` | rows and columns that enter the pair domain |

Points about the construction:

- **One integral block serves two pairs.** $J_{ji} = J_{ij}^{\mathsf T}$,
  so the code computes each unique pair $i \le j$ once and builds the
  $(j,i)$ block from the transpose.
- **The Fock blocks follow the domain.** The block also stores the
  restrictions of $F$ to its row and column domains (`fvv_aa`, `fvv_bb`)
  and the diagonal denominators $F_{aa}+F_{bb}-f_{ii}-f_{jj}$ that
  precondition the solve of @sec:amplitude-lmp2-accurate-solve.
- **An empty pattern returns no block.** A pair with no kept element
  disappears from the ragged list, and the occupied coupling of @sec:amplitude-lmp2-amplitude-equations-energy
  never visits it.

**Per-pair auxiliary truncation (opt-in).** With `aux_tail_frac` $= f$ and
$\varepsilon > 0$, the Gram product for pair $(i,j)$ may skip auxiliary
functions. Score each $P$ by $n_{Pi}n_{Pj}$, with
$n_{Pi} = \max_a\lvert\tilde B_{P,ia}\rvert$, and drop the lowest-scoring
functions while their summed score stays at most $f\varepsilon$. Each
element then moves by at most

$$ \Big\lvert\sum_{P\ \mathrm{dropped}}\tilde B_{P,ia}\tilde B_{P,jb}\Big\rvert
\le \sum_{P\ \mathrm{dropped}} n_{Pi}\,n_{Pj} \le f\varepsilon , $$

a rigorous elementwise bound. It is inactive at $\varepsilon = 0$ and on
the domain-fit path, which has its own auxiliary domains.

## Reproducibility and memory {#amplitude-lmp2-reproducibility-memory}

- The ragged matrix-vector product runs in parallel over output pairs with
  rayon. Each block is computed in the same arithmetic order whatever the
  schedule, so the product is bit-identical across thread counts.
- The default assembly builds the global three-index tensor
  $(N_{\mathrm{aux}}, n_on_v)$ once (`ferric-mp2/src/rimp2.rs::eri3_mo_ov_blocked`).
  So amplitude-threshold LMP2 without `integral_direct` is not
  integral-direct and carries no scaling claim. The integral-direct path of
  section @sec:integral-direct removes that tensor.
- The exact reference that measures the truncation error is opt-in
  (`compute_reference`, CLI `[local] reference = true`). It is a full
  canonical RI-MP2 that forms the same global tensor, so a run with it on
  is not reduced-cost. Without it the printout says the canonical energy
  was not computed.

## Options that change the equation {#amplitude-lmp2-options-change-equation}

The options split into two kinds, and the kind decides the sign of the
error.

- **Pattern options** remove amplitudes and leave every kept $J$ value
  unchanged: `eps` itself and `pair_gate_cal`. On a swap-closed pattern,
  the energy is the minimum of $E_H$ over the kept amplitudes (@sec:amplitude-lmp2-amplitude-equations-energy), so
  removing amplitudes can only raise it, up to the solver error bounded in
  @sec:amplitude-lmp2-accurate-solve.
- **Value options** change the $J$ values themselves: `fit_radius_bohr` and
  `aux_tail_frac`. The minimum argument does not cover a changed $J$, and
  their error has no proven sign.

| Option | Effect | Where |
|---|---|---|
| `eps` | the threshold; no default in the CLI or Python, where it is required (the Rust `AmplitudeLmp2Config::default()` sets $10^{-4}$) | this section |
| `frozen_core` | removes the lowest occupied orbitals from $i,j$ through `active_occ` | section @sec:closed-shell-ri-mp2 |
| `pair_gate_cal` | enables the gate (@eq:amplitude-lmp2-4); off by default | this section |
| `fit_radius_bohr` | per-pair domain-local RI fit in place of the global-metric fit | section @sec:integral-direct |
| `aux_tail_frac` | per-pair auxiliary truncation, rigorously bounded per element by $f\varepsilon$; inactive at $\varepsilon = 0$ | `ferric-mp2/src/lmp2_amplitude.rs::AmplitudeLmp2Config` |
| `fock_screen` | Rust only, off by default: also keeps elements whose one-step Fock-propagated integral clears $\kappa\varepsilon$ (section 13.1.3); inert at $\varepsilon = 0$ | `ferric-mp2/src/ragged.rs::pair_blocks_screened` |
| `cg_rtol`, `cg_max_iter` | Rust only: an explicit tolerance disables the tightening; neither is a CLI or Python option | this section |
| operator | Coulomb or erfc, in the metric and the three-index integrals | chapter @sec:operators |

## Numerical anchors {#amplitude-lmp2-numerical-anchors}

The exactness anchor comes first, and it is the only check with a
tolerance. At $\varepsilon = 0$ the localized-orbital ragged solve must
reproduce canonical RI-MP2. The two are independent constructions: one
uses canonical orbitals and closed-form denominators, the other localized
orbitals and a CG solve. They share only the RI integrals. A construction
mutation, dropping one hard virtual, breaks the agreement
(`mutated_virtual_space_fails_the_anchor`).

| Check | Reference | Agreement | Test |
|---|---|---|---|
| $\varepsilon = 0$, every locality map at its no-op limit, in-core (global and per-pair domain fit) and integral-direct: H2O, n-butane / cc-pVDZ, 6-31G (cc-pvdz-ri), frozen core 0 and one per heavy atom | PySCF `DFMP2` with the same auxiliary basis, checked against a dense numpy sum to $10^{-12}$ | $E_{\mathrm{corr}}$ at most $1.38\times10^{-11}$ Ha over 24 comparisons, at the default PCG stop; bar $10^{-10}$ Ha | `crates/ferric-mp2/tests/validation_lmp2_amplitude.rs`, `eps_zero_matches_canonical_ri_mp2` in `crates/ferric-mp2/tests/lmp2_amplitude.rs` |
| finite $\varepsilon$ on n-butane / cc-pVDZ, all electrons | PySCF `DFMP2` (canonical) | recovers 99.98%, 99.67%, 95.0% of the DF-MP2 correlation energy in core at $\varepsilon = 10^{-5}$, $10^{-4}$, $10^{-3}$ with production maps; the error is one-sided and has no tolerance | same |

<!-- figures: site/src/reference/validation.md "Local RI-MP2 (`rimp2` with `[local] scheme = \"amplitude-threshold\"`)" -->

ORCA DLPNO-MP2 on the same molecules is recorded on the validation page as
a ballpark only. It truncates by pair-natural-orbital occupation, not by an
amplitude threshold, so the two recoveries are not compared.

**The error is superlinear in $\varepsilon$.** As $\varepsilon$ grows by
factors of ten, the recovery falls from 99.98% to 99.67% to 95.0%. The
missing correlation energy therefore grows faster than linearly in
$\varepsilon$. There is no tolerance on it. To reach a target accuracy, run
a series of $\varepsilon$ values and choose from it.

n-Butane in cc-pVDZ, all electrons, from the `589b50832` build. Each line
of the output is cut after `keep_fraction`; the full `local` dict also
carries the solver statistics, the timings and the trust text. The
$\varepsilon = 0$ run reproduces the exact RI-MP2 energy to
$4.919\times10^{-13}$ Ha, and the recovery falls as stated above:

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/alkane_4.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

exact = ferric.run_rimp2(mol, bs, aux, frozen_core=0).mp2_corr
print(f"exact RI-MP2 E_corr = {exact:.10f} Ha")
for eps in (0.0, 1e-5, 1e-4, 1e-3):
    r = ferric.run_rimp2(mol, bs, aux, frozen_core=0, local="amplitude-threshold", eps=eps)
    print(f"eps={eps:7.0e}  E_corr={r.mp2_corr:.10f}  recovery={100 * r.mp2_corr / exact:8.4f}%  "
          f"E-E*={r.mp2_corr - exact:+.3e} Ha  local={r.local}")
```

```text
exact RI-MP2 E_corr = -0.5996452251 Ha
eps=  0e+00  E_corr=-0.5996452251  recovery=100.0000%  E-E*=+4.919e-13 Ha  local={'scheme': 'amplitude-threshold', 'eps': 0.0, 'keep_fraction': 1.0, ...}
eps=  1e-05  E_corr=-0.5995279382  recovery= 99.9804%  E-E*=+1.173e-04 Ha  local={'scheme': 'amplitude-threshold', 'eps': 1e-05, 'keep_fraction': 0.6013112181756786, ...}
eps=  1e-04  E_corr=-0.5976711043  recovery= 99.6708%  E-E*=+1.974e-03 Ha  local={'scheme': 'amplitude-threshold', 'eps': 0.0001, 'keep_fraction': 0.22030308815120248, ...}
eps=  1e-03  E_corr=-0.5696280916  recovery= 94.9942%  E-E*=+3.002e-02 Ha  local={'scheme': 'amplitude-threshold', 'eps': 0.001, 'keep_fraction': 0.029930075062173215, ...}
```

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/lmp2_amplitude.rs`, `crates/ferric-mp2/src/ragged.rs` | the algorithm, the policy and the mask |
| `scripts/lmp2_hylleraas_vs_wshg23_sympy.py`, claims 1, 2, 4, 5, 6 | verified equations @eq:lmp2-hylleraas and @eq:lmp2-cg-bound |
| Wang, Aldossary, Shi, Liu, Li and Head-Gordon, J. Chem. Theory Comput. 19, 7577 (2023) | the single-threshold scheme and the Hylleraas stopping rule, as cited in the module documentation |
| `site/src/methods/mp2.md`, "Exact and local MP2" | the user-facing description and the certificate fields |
| `site/src/reference/validation.md` | the anchor row |
