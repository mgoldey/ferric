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
| 3 | λ-Boys-localized active occupieds (section @sec:amplitude-lmp2-lambda-boys), their centroids and spreads, and the localized Fock blocks $f$ and $F$ | `ferric-mp2/src/lmp2_amplitude.rs::localized_spaces_lambda` |
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

`localized_spaces_lambda` localizes the active occupied orbitals by
λ-Boys (Boys at `lambda_boys = 0`; section @sec:amplitude-lmp2-lambda-boys) and records,
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

## What the fixed pattern misses: the Fock-propagated screen {#amplitude-lmp2-fock-screen}

The Eq-8 mask looks at one number per amplitude, its own integral. In a
localized basis that is not enough. The off-diagonal Fock elements pass
amplitude from kept elements to dropped ones, so an element whose integral
is zero, by symmetry for instance, can still carry a sizeable exact
amplitude. Eq 8 cannot keep such an element at any $\varepsilon > 0$, and
the truncation error then stops falling with $\varepsilon$.

**The error of a fixed pattern.** Split the amplitudes of (@eq:amplitude-lmp2-1)
into the kept block $K$, the Eq-8 pattern, and the dropped block $D$. The
masked solve is $t_K = -\mathcal A_{KK}^{-1}J_K$, $t_D = 0$. For the
Hylleraas form with the exchange weighting absorbed,

$$ E_{\mathrm{mask}} - E^{*} = t^{*\mathsf T}_D\big(\mathcal A_{DD} - \mathcal A_{DK}\mathcal A_{KK}^{-1}\mathcal A_{KD}\big)t^{*}_D \;\ge 0 . \label{eq:lmp2-fock-screen-schur} $$

The error is the Schur norm of the dropped exact amplitudes. It is set by
those amplitudes, not by their integrals and not by their own energy
$\langle t^*_D, J_D\rangle$. Write $\mathcal A = \Delta + \mathcal A'$, with
$\mathcal A'$ the off-diagonal Fock coupling, and expand the exact
amplitudes in it:

$$ t^{*} = -\frac{J}{\Delta} + \frac{\mathcal A' J/\Delta}{\Delta} + O(\mathcal A'^2) . $$

An element with $J_D = 0$ still carries a first-order amplitude, so the
masked error has a floor that no threshold removes.

**Measured floor.** The record case is adamantane/6-31G (cc-pvdz-ri, frozen
core 10). At $\varepsilon = 10^{-8}$ the pattern keeps 99.998% of the
amplitudes and the error is still $2.4\times10^{-7}$ Ha. The dropped exact
amplitudes have norm $3.0\times10^{-4}$ against $2.5\times10^{-8}$ for
$J/\Delta$. 99.8% of their weight sits in the mixed valence-virtual /
hard-virtual blocks, where the virtual Fock coupling is largest (0.43 Ha).

**The screen.** `AmplitudeLmp2Config::fock_screen = Some(κ)` changes the
selection (`ferric-mp2/src/ragged.rs::fock_dressed_gate`,
`ferric-mp2/src/ragged.rs::pair_blocks_screened`). For each pair, over the
candidate square $C_{ij}$ whose integrals the assembly already holds, form

$$ \tilde g = g + F' t^{0} + t^{0} F', \qquad t^{0}_{ab} = -\frac{g_{ab}}{\Delta^{ij}_{ab}}, \qquad
F' = F_{vv}[C_{ij}, C_{ij}] - \operatorname{diag} . \label{eq:lmp2-fock-screen-dressed} $$

$-\tilde g/\Delta$ is one Jacobi step from the diagonal estimate with the
virtual Fock coupling switched on, and it is exact to first order in the
coupling. Element $(a,b)$ is kept when Eq 8 keeps it, or when
$\lvert\tilde g_{ab}\rvert > κ\varepsilon$ or $\lvert\tilde g_{ba}\rvert > κ\varepsilon$.
The kept values are still $g$.

The core of `fock_dressed_gate`, after the diagonal of $F'$ has been
removed, copied unchanged:

<!-- excerpt: ferric-mp2/src/ragged.rs::fock_dressed_gate -->
```rust
let t0 = Array2::from_shape_fn((nc, nc), |(r, c)| {
    -g[(r, c)] / (fv[cand[r]] + fv[cand[c]] - fo_i - fo_j)
});
let mut gt = fsub.dot(&t0);
gt += &t0.dot(&fsub);
gt += g;
let inv = 1.0 / kappa;
Array2::from_shape_fn((nc, nc), |(r, c)| {
    g[(r, c)].abs().max(gt[(r, c)].abs() * inv)
})
```

| Code | Equation |
|---|---|
| `fsub` | $F'$, the virtual Fock block on $C_{ij}$ with its diagonal removed |
| `t0` | $t^0 = -g/\Delta$ |
| `gt` | $\tilde g = g + F't^0 + t^0F'$ |
| the returned array | $\max(\lvert g\rvert, \lvert\tilde g\rvert/κ)$, thresholded at $\varepsilon$ by `ferric-mp2/src/ragged.rs::pair_block_from_g_cand_gated` |

Points about the screen:

- **It only adds elements.** The gate is $\max(\lvert g\rvert, \lvert\tilde g\rvert/κ)$,
  so the union never drops an Eq-8 element.
- **The pattern stays swap-closed.** The gate is covariant under the pair
  transpose, $\mathrm{gate}(g^{\mathsf T}) = \mathrm{gate}(g)^{\mathsf T}$, since $\Delta$
  and the two Fock products swap.
- **It is pair-local.** It reads only the integrals already evaluated on the
  pair's candidate square, two $\lvert C\rvert^3$ GEMMs. An element outside
  $C_{ij}$ is not considered, the same scope as the WLLH25 embedding below.
- **The occupied off-diagonal coupling is left out.** It would need other
  pairs' integrals. On the systems measured it ranks the dropped amplitudes
  worse than the virtual part. Section @sec:amplitude-lmp2-combined-truncation
  states what that omission costs.
- **The limits.** `None`, the library default of `AmplitudeLmp2Config`, is
  the plain Eq-8 pattern bitwise. The integral-direct CLI and Python paths
  switch the screen on with κ = 1
  (`ferric-mp2/src/lmp2_direct.rs::DIRECT_FOCK_SCREEN_KAPPA`; section
  @sec:amplitude-lmp2-fock-screen-default). At $\varepsilon = 0$ the screen is inert. A vacuous $κ$ keeps the whole
  candidate square (`crates/ferric-mp2/tests/lmp2_fock_screen.rs`).
- **Memory.** On the integral-direct path the gate holds six
  candidate-square matrices per pair ($F'$, $t^0$, $F't^0$, the $t^0F'$
  temporary, the gate and its transpose; at most four are live at once). The
  per-group scratch bound that sizes the group concurrency against the
  memory budget counts all six when the screen is on
  (`ferric-mp2/src/lmp2_direct.rs::group_scratch_bound`). The kept blocks
  stay inside the candidate square, so the candidate-domain bound of the
  pre-flight still holds, and the charge-as-built gate charges them at their
  exact size.

> **Verified equation @eq:lmp2-fock-screen-schur** · the masked error is the Schur norm of the dropped exact amplitudes
>
> $E_{\mathrm{mask}} - E^*$ equals the Schur-complement norm of $t^*_D$ for
> any SPD $\mathcal A$, in symbols on a 2+1 block model and by exact rational
> arithmetic on a random 6-dimensional model (claim 1). The one-step
> estimate $-(J + \mathcal A't^0)/\Delta$ matches $t^*$ through first order in
> the coupling, and an element with $J_D = 0$ has a first-order $t^*_D$
> (claim 2). The gate is pair-transpose covariant (claim 3).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_fock_screen_sympy.py`, claims 1, 2 and 3 |
> | Negative control | the runner's three recorded mutations, one per claim |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/ragged.rs::{fock_dressed_gate,pair_blocks_screened}` |
> | Code test | `vacuous_screen_at_finite_eps_matches_canonical` in `crates/ferric-mp2/tests/lmp2_fock_screen.rs` |
>
> <!-- verified: id=lmp2.fock-screen; script=scripts/lmp2_fock_screen_sympy.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/ragged.rs::{fock_dressed_gate,pair_blocks_screened}; rust_test=ferric-mp2/tests/lmp2_fock_screen.rs::vacuous_screen_at_finite_eps_matches_canonical; registry=local-mp2 -->

### Relation to prior work

The element-wise mask is the single-threshold local MP2 of Wang, Aldossary,
Shi, Liu, Li and Head-Gordon (JCTC 19, 7577 (2023)), which keeps $(ia|jb)$
by integral magnitude alone. In a fixed, non-canonical localized virtual
basis the off-diagonal virtual Fock block couples kept and dropped
amplitudes. Wang, Ling, Shi, Shen, Wang, Yang, Li and Head-Gordon (JCTC 21,
10910 (2025); ChemRxiv 10.26434/chemrxiv-2025-j0wd1, Eqs. 43–49) identified
that coupling ($\mathcal A_{KD}$ in (@eq:lmp2-fock-screen-schur)) as the
dominant fixed-pattern error. They correct it with a per-pair block Kapuy
solve of the dropped amplitudes and a right-hand-side embedding
$m = \Delta_{ls}\tau_s$ of the kept equations, leaving the kept set
unchanged. The screen changes the kept set instead.

The dressed integral $\tilde g$ is the numerator of the next-order Kapuy
amplitude, the first Jacobi iterate (Kapuy et al., IJQC 23, 981 (1983);
Subotnik and Head-Gordon, JCP 122, 034109 (2005)). Selecting by
perturbative coupling to a zeroth-order vector is the logic of selected CI
and importance truncation (Huron, Malrieu and Rancurel, JCP 58, 5745
(1973); Roth, PRC 79, 064324 (2009)). The closest per-amplitude precedent is
Auer and Nooijen (JCP 125, 024104 (2006)), who screen individual
coupled-cluster amplitudes with complete MP2 information. Iterated LMP2
amplitudes or pair energies also select PNO spaces and pair classes (Werner
et al., JCTC 11, 484 (2015); Schwilk et al., JCP 142, 121102 (2015)).
Pair-natural-orbital methods avoid the problem by selecting in
pseudo-canonical domains, where the in-domain virtual Fock block is
diagonal (Pinski et al., JCP 143, 034108 (2015)). To our knowledge, a cheap
one-step Fock-propagated estimate has not previously been used as the
membership rule of an element-wise amplitude mask in a fixed localized
virtual basis. That combination is all this section adds.

### Measured on a dense oracle {#amplitude-lmp2-fock-screen-measured}

Measured 2026-11-16, provisional. The probe is
`crates/ferric-mp2/tests/lmp2_fock_screen_probe.rs` (`#[ignore]`). It
forms every $(ia|jb)$ of the global RI fit and the whole $\mathcal A$, at
6-31G with cc-pvdz-ri and frozen core one per heavy atom. The reference is
canonical RI-MP2 with the same orbitals and auxiliary basis, the
$\varepsilon = 0$ limit of every arm. Errors are in Ha, from Hylleraas
energies; keep is the retained fraction. "WLLH" is Eqs. 43–49 of WLLH25
transcribed per pair: occupied Fock diagonal, a Sylvester solve over the
pair's whole virtual space, the embedding $m = \Delta_{ls}\tau_s$ added to
the kept right-hand side, and the full $\mathcal A$ on the kept set. That is
its best case, with no integral-map limit.

| system | $\varepsilon$ | plain keep / err | WLLH err | screen κ=1 keep / err | κ=0.3 keep / err | screen+WLLH (κ=1) |
|---|---|---|---|---|---|---|
| adamantane | 1e-6 | 0.9767 / 6.2e-6 | 1.1e-6 | 0.9902 / 1.2e-7 | 0.9982 / 1.1e-8 | 1.0e-7 |
| adamantane | 1e-7 | 0.9992 / 4.2e-7 | 1.0e-7 | 0.9998 / 1.0e-10 | 1.0000 / 4.0e-11 | 4.4e-11 |
| adamantane | 1e-8 | 1.0000 / 2.4e-7 | 8.8e-8 | 1.0000 / 3e-14 | 1.0000 / <1e-13 | <1e-13 |
| n-octane | 1e-5 | 0.3903 / 8.6e-5 | 1.6e-5 | 0.4400 / 2.1e-5 | 0.6269 / 2.6e-6 | 4.4e-6 |
| n-octane | 1e-6 | 0.7458 / 2.1e-6 | 6.4e-7 | 0.7918 / 2.0e-7 | 0.8886 / 2.4e-8 | 7.3e-8 |
| n-octane | 1e-7 | 0.9314 / 6.7e-8 | 2.7e-8 | 0.9487 / 1.1e-9 | 0.9712 / 1.5e-10 | 6.1e-10 |
| naphthalene | 1e-5 | 0.6293 / 1.4e-4 | 6.2e-5 | 0.6821 / 2.1e-5 | 0.8377 / 2.4e-6 | 6.9e-6 |
| naphthalene | 1e-6 | 0.9146 / 2.3e-6 | 1.4e-6 | 0.9379 / 1.3e-7 | 0.9764 / 1.6e-8 | 7.7e-8 |
| naphthalene | 1e-7 | 0.9911 / 6.7e-8 | 1.9e-8 | 0.9949 / 1.9e-10 | 0.9992 / 1.7e-11 | 1.1e-10 |

What the table shows:

- **The adamantane floor.** Plain adamantane does not converge with
  $\varepsilon$: 4.2e-7 at 1e-7, 2.4e-7 at 1e-8. The screen does. n-octane
  and naphthalene show no floor down to 1e-7 (octane 8.7e-9 at 3e-8), only
  the slower plain convergence.
- **Equal kept fraction.** At equal kept fraction the screen is ahead.
  Octane plain at keep 0.869 ($\varepsilon$ = 3e-7) is 3.4e-7 and WLLH at
  that pattern 6.8e-8, against 2.4e-8 for the screen at keep 0.889.
- **What WLLH leaves.** WLLH adds no amplitudes but a per-pair Sylvester
  solve. It removes a factor 2–5, and the remainder is the cross-pair
  occupied Fock coupling it sets to zero. On the same probe with $f_{ik}$
  ($i\ne k$) zeroed, WLLH is exact to $\le 3\times10^{-13}$ on all three
  systems. This is an anchor, not a finding: with a diagonal occupied Fock
  the pairs decouple and the whole-space Sylvester solve is the exact
  solution. The paper's λ-Boys orbitals, which make $f_{oo}$ more
  diagonal, were not tried.
- **Both arms.** Combining the two arms gains a further factor 1.5–5 over
  the screen alone.
- **The candidate restriction.** The production global path, with the
  screen restricted to each pair's Schwarz candidate square, matches the
  unrestricted dense screen to $2\times10^{-13}$ in every row. On these
  systems the candidate restriction leaves no floor.
- **Ranking.** At matched count, ranking the dropped amplitudes by the
  one-step virtual estimate leaves 10–50× less error than a random choice
  of the same size (octane 20%: 9.4e-8 vs 1.07e-6; adamantane
  $\varepsilon$=1e-7, 20%: 3.6e-9 vs 1.7e-7).

Timings on the production path are in section @sec:amplitude-lmp2-fock-screen-default.

### On by default on the integral-direct path {#amplitude-lmp2-fock-screen-default}

`[local] integral_direct = true` (CLI) and `run_rimp2(..., integral_direct=True)`
(Python) run the screen with κ = 1 unless `fock_screen = false` /
`fock_screen=False` is given. The key is an error without `integral_direct`.
The in-core path and the Rust `AmplitudeLmp2Config::default` keep the plain
mask.

The decision rests on criteria fixed before any of the data below was taken.
At matched $\varepsilon$ the wall-time overhead is at most 10% on the
largest systems, and the error is lower on every system. On the n-alkanes
the error grows by less per CH$_2$. On the ACONF20 conformers the error in
the conformer energy differences is smaller. All four hold.

**Construction.** Measured 2026-11-18 with
`crates/ferric-mp2/tests/lmp2_fock_default_probe.rs` (`probe_fock_default`,
`#[ignore]`), library code of commit `ac41c2305`. The setup is 6-31G with
cc-pVDZ-RI, all-electron, Coulomb, and the CLI integral-direct maps
(aux_radius 10, virt_radius 20 Bohr, ao_tail 1e-4, schwarz_skip 1e-5,
batch_merge 4, no pair gate). Only $\varepsilon$ and κ vary. The anchor
`fock_screen_is_inert_at_eps0_butane` passes: at $\varepsilon = 0$, screen
on minus screen off is 0 on the production maps, and with every map inert
both are $+1.5\times10^{-13}$ Ha from canonical RI-MP2.

The error is $E - E_0$, where $E_0$ is the $\varepsilon = 0$ run on the same
path and maps. C24 and C32 at $\varepsilon = 0$ do not fit the memory
budget (C32 needs 51.7 GB resident). For them, and for ACONF20, the error is
taken against canonical RI-MP2 instead (marked \*). That reference
includes the maps' own error, $+0.9$ to $+3.2\times10^{-6}$ Ha on C12–C20
and $\pm1.9\times10^{-6}$ Ha on naphthalene and adamantane. Keep is the
retained fraction; the errors are in Ha.

| system | $\varepsilon$ | keep off / on | error off | error κ = 1 | ratio |
|---|---|---|---|---|---|
| C12 | 1e-4 | 0.0269 / 0.0314 | 4.32e-3 | 2.72e-3 | 0.63 |
| C12 | 1e-6 | 0.2968 / 0.3235 | 9.05e-6 | 2.29e-6 | 0.25 |
| C12 | 1e-7 | 0.5507 / 0.5772 | 7.60e-7 | 5.08e-8 | 0.07 |
| C16 | 1e-4 | 0.0122 / 0.0144 | 6.19e-3 | 3.97e-3 | 0.64 |
| C16 | 1e-6 | 0.1567 / 0.1735 | 1.42e-5 | 3.95e-6 | 0.28 |
| C16 | 1e-7 | 0.3340 / 0.3572 | 1.15e-6 | 9.53e-8 | 0.08 |
| C20 | 1e-4 | 0.0066 / 0.0077 | 8.06e-3 | 5.22e-3 | 0.65 |
| C20 | 1e-6 | 0.0906 / 0.1011 | 1.79e-5 | 5.70e-6 | 0.32 |
| C24 | 1e-4 | 0.0039 / 0.0046 | 9.95e-3\* | 6.48e-3\* | 0.65 |
| C24 | 1e-6 | 0.0565 / 0.0634 | 3.13e-5\* | 1.18e-5\* | 0.38 |
| C32 | 1e-4 | 0.0017 / 0.0020 | 1.37e-2\* | 8.98e-3\* | 0.66 |
| C32 | 1e-5 | 0.0080 / 0.0092 | 8.44e-4\* | 3.29e-4\* | 0.39 |
| naphthalene | 1e-4 | 0.1337 / 0.1538 | 4.17e-3 | 2.15e-3 | 0.51 |
| naphthalene | 1e-6 | 0.7477 / 0.7649 | 3.80e-6 | 8.77e-7 | 0.23 |
| naphthalene | 1e-7 | 0.9319 / 0.9369 | 4.10e-7 | 9.44e-9 | 0.02 |
| adamantane | 1e-4 | 0.1219 / 0.1457 | 5.61e-3 | 3.38e-3 | 0.60 |
| adamantane | 1e-6 | 0.8334 / 0.8466 | 1.03e-5 | 1.49e-6 | 0.14 |
| adamantane | 1e-7 | 0.9728 / 0.9757 | 6.63e-7 | 1.49e-8 | 0.02 |

The full grid also has $\varepsilon$ = 3e-5, 1e-5, 3e-6 and 3e-7, plus
κ = 0.3 on the four smaller systems. Every one of the 38 off/on pairs has
the lower error with the screen, ratios 0.022–0.66. C20 at 1e-7 and C32 at
1e-6 are refused by the memory budget in both arms.

**Not only more amplitudes.** The screen keeps 0.3–19.5% more amplitudes.
Interpolating the plain arm's error to the screen's kept fraction (log-log,
along the $\varepsilon$ series) leaves the screen ahead everywhere. The
ratio at matched keep is 0.73–0.89 at $\varepsilon$ = 1e-4 and 0.09–0.70
from 3e-5 to 3e-7.

**Cost.** Each run is one process, with a 3-index memory budget of 8 GB for
both arms (`FERRIC_MEM_BUDGET_GB = 8`, a 12 GB cgroup cap). There are three
repeats with the κ order alternated. Wall time is the whole
`amplitude_lmp2_direct` call, in seconds, as median (range).

| system | $\varepsilon$ | off | κ = 1 | overhead | peak RSS off / on (GB) |
|---|---|---|---|---|---|
| C24 | 1e-4 | 88.6 (88.6–89.8) | 95.6 (95.2–95.8) | +7.8% | 2.47 / 2.89 |
| C24 | 1e-5 | 118.1 (117.7–118.8) | 128.4 (127.3–128.5) | +8.7% | 2.45 / 2.76 |
| C24 | 1e-6 | 227.2 (225.3–227.6) | 246.4 (245.9–249.2) | +8.5% | 6.68 / 7.27 |
| C32 | 1e-4 | 210.3 (208.9–213.9) | 222.4 (222.3–222.6) | +5.7% | 4.02 / 4.68 |
| C32 | 1e-5 | 258.0 (256.2–258.2) | 275.6 (274.4–276.4) | +6.8% | 3.88 / 4.22 |

Most of the overhead is in the assembly, where `fock_dressed_gate` runs
two candidate-square GEMMs per pair. At C32, $\varepsilon$ = 1e-4, the
assembly goes from 204 s to 214 s and the solve from 6.8 s to 8.1 s. The
peak RSS rises by 0.3–0.7 GB, the six candidate-square matrices per pair
that `group_scratch_bound` charges. At matched final error (the plain arm's
wall time interpolated to the screen's error), the ratio is 0.87 on C24 at
$\varepsilon$ = 1e-5 and 1.03 at 1e-4 on C24 and C32. On naphthalene and
adamantane (one run each, 3–15 s) it is 0.85–1.01. The overhead at matched
$\varepsilon$ there is −2% to +12%, the largest at 1e-4.

**Size extensivity.** Error per added CH$_2$ along the n-alkanes, a linear
fit to the last three sizes (μHa per CH$_2$; \* against canonical RI-MP2):

| $\varepsilon$ | sizes | off | κ = 1 |
|---|---|---|---|
| 1e-4 | C20, C24, C32\* | 468 | 313 |
| 1e-5 | C20, C24, C32\* | 26.5 | 12.3 |
| 1e-6 | C16, C20, C24\* | 1.88 | 0.73 |

The error per carbon at $\varepsilon$ = 1e-5 is 26.0, 27.2 and 26.4 μHa
from C20 to C32 without the screen, and 9.1, 9.6 and 10.3 μHa with it.
The screen is lower at every size, but its per-carbon error has not
levelled off by C32.

**Conformer energies.** These are the 22 ACONF20 C20H42 conformers
(`testdata/molecules/aconfl/`). Each conformer error is the local
correlation energy minus canonical RI-MP2. The tabulated quantity is the
error in the conformer energy relative to A2000, the reactions of
`ref_aconf20.csv`, in kJ/mol. At $\varepsilon$ = 1e-6 the 8 GB budget
refuses 9 of the 22 conformers in both arms, A2000 among them. That row is
over all pairs of the 13 that fit.

| $\varepsilon$ | conformers | max\|Δ\| off | max\|Δ\| κ = 1 | MAD off | MAD κ = 1 |
|---|---|---|---|---|---|
| 1e-4 | 22 | 7.76 | 5.38 | 4.01 | 2.96 |
| 1e-5 | 22 | 0.642 | 0.359 | 0.343 | 0.199 |
| 1e-6 | 13 (all pairs) | 0.0253 | 0.0055 | 0.0104 | 0.0016 |

**Reading (provisional, one basis, one family of chains plus two rigid
molecules).** At the production example $\varepsilon$ = 1e-4 the screen
removes a third to a half of the truncation error for 6–9% more wall time. At
1e-6 and below it removes 60–98%. The 1e-4 conformer errors stay at several
kJ/mol with or without it: $\varepsilon$ = 1e-4 is a loose model whichever
mask is used.

## Combining the screen, the embedding and λ-Boys orbitals {#amplitude-lmp2-combined-truncation}

A combined production scheme would use three pieces: λ-Boys occupied
orbitals (WLLH25 Eqs. 8–9, after Subotnik and Head-Gordon, JCP 122, 034109
(2005)), the WLLH25 block-Kapuy embedding (Eqs. 43–49), and the screen of
section @sec:amplitude-lmp2-fock-screen extended with occupied coupling.
This section proves what each does to the energy. The proofs, and a dense
cross-check on the LMP2 block structure (several pairs, shared virtuals,
non-zero $f_{oo}$ and $F_{vv}$ off-diagonals), are in
`scripts/lmp2_combined_sympy.py`. Each claim carries a status. No
production code implements the occupied extension, the
Hylleraas-evaluated embedding or the embedded Galerkin solve below. They
are statements about the equations.

**Setting and assumptions.**

- (A1) The occupied and virtual orbitals are orthonormal, so the amplitude
  space carries the Euclidean metric and $\mathcal A$ of
  (@eq:amplitude-lmp2-1) is symmetric. With
  $\epsilon_{\mathrm{LUMO}} > \epsilon_{\mathrm{HOMO}}$ it is positive
  definite, its spectrum being $\epsilon_a+\epsilon_b-\epsilon_i-\epsilon_j$.
- (A2) The retained pattern $K$ and its complement $D$ are closed under the
  virtual swap $\mathcal S$ and the pair transpose
  $\mathcal P: (ij,ab)\mapsto(ji,ba)$. Eq 8 and the screen satisfy this by
  construction.
- (A3) The reference $E^*$ is the exact minimum over a fixed universe of
  amplitudes with fixed integrals $J$: the global RI fit, or the union of
  candidate squares. The domain-local fit of section
  @sec:integral-direct-domain-local-fit changes $J$ and is outside these
  statements.

Write $\mathcal W = 2 - \mathcal S$ (eigenvalues 1 and 3, commuting with
$\mathcal A$), $\Delta = \operatorname{diag}\mathcal A$,
$\mathcal A' = \mathcal A - \Delta$, and

$$ E_H(t) = \langle\mathcal W t,\ 2J + \mathcal A t\rangle, \qquad t^* = -\mathcal A^{-1}J, \qquad E_H(t) - E^* = (t-t^*)^{\mathsf T}\mathcal W\mathcal A\,(t-t^*) \ge 0 . $$

$E_H$ is a minimum principle over the whole universe.

### P1. The truncation error (PROVEN under A1–A3) {#amplitude-lmp2-combined-p1}

The masked solve is the Galerkin solution $\mathcal A_{KK}t_K = -J_K$,
$t_D = 0$. Let $R_D = J_D + \mathcal A_{DK}t_K$ be the dropped-block
residual and
$\Sigma = \mathcal A_{DD} - \mathcal A_{DK}\mathcal A_{KK}^{-1}\mathcal A_{KD}$
the Schur complement. By (A2) $\mathcal W$ is block diagonal,
$\mathcal W_D$ commutes with $\Sigma$, and

$$ E_K - E^* = R_D^{\mathsf T}\,\mathcal W_D\,\Sigma^{-1}R_D = t^{*\mathsf T}_D\,\mathcal W_D\,\Sigma\,t^*_D, \qquad t^*_D = -\Sigma^{-1}R_D , \label{eq:lmp2-combined-truncation-error} $$
$$ \frac{\lVert R_D\rVert^2}{\lambda_{\max}(\mathcal A)} \;\le\; E_K - E^* \;\le\; \frac{3\,\lVert R_D\rVert^2}{\lambda_{\min}(\mathcal A)} . $$

The bounds hold because $\Sigma^{-1} = (\mathcal A^{-1})_{DD}$ has its
spectrum in $[\lambda_{\max}^{-1}, \lambda_{\min}^{-1}]$. Two corollaries
follow:

- The masked energy is an upper bound, $E_K \ge E^*$: it recovers less
  correlation energy.
- $K_1 \subseteq K_2 \Rightarrow E^* \le E_{K_2} \le E_{K_1}$. So any screen
  that only adds elements to the Eq-8 pattern can never raise the energy
  error, for every $\kappa$ and with or without the occupied term.

The bracket is a computable a-posteriori certificate for any arm that stops
at a Galerkin solution: $R_D$ costs one matvec over the universe. The CG
error of section @sec:amplitude-lmp2-accurate-solve adds a non-negative term
on top, so the bound survives inexact solves.

> **Verified equation @eq:lmp2-combined-truncation-error** · the truncation error, its two-sided bound and nesting
>
> $E_K - E^* = R_D^{\mathsf T}\mathcal W_D\Sigma^{-1}R_D = t^{*\mathsf T}_D\mathcal W_D\Sigma t^*_D$
> in symbols on a 2+1 block model with a generic symmetric matrix, and
> numerically on random instances with the LMP2 block structure, where the
> two-sided bound holds and a screened pattern is nested in Eq 8 and no
> worse (P1).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_combined_sympy.py`, P1 |
> | Negative control | the runner drops $\mathcal A_{DK}$ from the Schur complement ("P1: residual identity") |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/ragged.rs::pair_blocks_screened` |
>
> <!-- verified: id=lmp2.combined.truncation-error; script=scripts/lmp2_combined_sympy.py; claim=P1: upper bound; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/ragged.rs::pair_blocks_screened; registry=local-mp2 -->

### P2. The screen's estimate is one Jacobi step (PROVEN) {#amplitude-lmp2-combined-p2}

With $t^0 = -J/\Delta$, the occupied-extended dressed integral is

$$ \tilde g_{ij} = g_{ij} + F'\,t^0_{ij} + t^0_{ij}\,F' - \sum_{k\ne i} f_{ik}\,t^0_{kj} - \sum_{k\ne j} t^0_{ik}\,f_{kj} \;=\; \big(J + \mathcal A' t^0\big)_{ij} . $$

Apply $t^* = t^0 - \Delta^{-1}\mathcal A' t^*$ twice. Exactly,

$$ t^* + \Delta^{-1}\tilde g = (\Delta^{-1}\mathcal A')^2\,t^*, \qquad
\big|t^*_e + \tilde g_e/\Delta_e\big| \le \rho^2\,\lVert t^*\rVert, \quad \rho = \lVert\Delta^{-1}\mathcal A'\rVert_2 \le \frac{\lVert\mathcal A'\rVert_2}{\Delta_{\min}} . \label{eq:lmp2-combined-jacobi-remainder} $$

So the estimate $-\tilde g/\Delta$ is exact through first order in the Fock
coupling, and the estimator error $t^* + \Delta^{-1}\tilde g$ is second
order. It has two sources:

- (i) Paths of length two: a dropped element reached from the integrals
  only through another element with $J = 0$. A three-element chain has
  $\tilde g_3 = 0$ and $t^*_3 = O(\rho^2)$, so the estimate is zero there.
- (ii) The deviation of the kept solution from $t^0$,
  $t_K - t^0_K = -\mathcal A_{KK}^{-1}\mathcal A'_{KK}t^0_K$, which enters
  $R_D$ only through $\mathcal A'_{DK}$ and which $\tilde g_D$, built on
  $t^0$, does not contain.

The committed virtual-only screen drops the occupied terms, which are first
order in $\lVert f'_{oo}\rVert$. The error of its estimate is therefore
$O(\rho)$, not $O(\rho^2)$. On the model, halving the coupling divides the
error of the occupied-extended estimate by 4.0 and that of the
virtual-only estimate by 2.0.

**The estimate is second order; the mask's miss is not.** Everything above
is about the estimator. A mask at a fixed threshold drops every element
whose estimate falls below $\kappa\varepsilon$, and those elements carry
exact amplitudes of $O(\mu)$ in the coupling strength $\mu$. Unless the mask covers the
whole support of the occupied coupling term $f'_{oo}\,t$, the norm of the
dropped exact amplitudes therefore stays first order, for both screens.
Measured 2026-11-17 on octane / 6-31G at $\varepsilon = 10^{-6}$, with all
off-diagonal Fock coupling scaled by $\mu$:

| quantity | occupied-extended | virtual-only |
|---|---|---|
| estimator error $\lVert t_{\mathrm{est}} - t^*\rVert$, no threshold | $\approx 0.038\,\mu^2$, slope 1.98 | $\approx 0.049\,\mu$, slope 1.00–1.07 |
| norm of the dropped exact amplitudes, $\kappa$ = 1, 0.1, 0.01 | slope $\approx 1$ | slope $\approx 1$ |
| miss coefficient, occupied-extended relative to virtual-only | 3% lower ($\kappa$ = 1), 12% lower ($\kappa$ = 0.1), 21% lower ($\kappa$ = 0.01) | |
| kept amplitudes at production $\kappa = 1$ | 0.05% more | |

Both estimator errors are exactly 0 at $\mu = 0$. The occupied extension
makes the estimate second order, but at a fixed threshold it lowers the
dropped-amplitude norm only by the coefficients in the table.

Since $\Sigma = \Delta_D + O(\rho\,\Delta)$, the leading-order error
estimate is the Kapuy energy of the dressed integrals over the dropped set,

$$ \delta E_{\mathrm{est}} = \sum_{e\in D}\frac{\tilde g_e\,(\mathcal W\tilde g)_e}{\Delta_e} = \sum_{ij}\sum_{(a,b)\in D_{ij}}\frac{\tilde g^{ij}_{ab}\,(2\tilde g^{ij}_{ab} - \tilde g^{ij}_{ba})}{\Delta^{ij}_{ab}} , $$

with relative error $O(\rho)$ of either sign. It is an estimate, not a
bound, and it sees only the candidate squares the assembly evaluates.

> **Verified equation @eq:lmp2-combined-jacobi-remainder** · the occupied-extended estimate is one Jacobi step with a second-order estimator error
>
> $t^* + \Delta^{-1}\tilde g = (\Delta^{-1}\mathcal A')^2t^*$ in symbols on a
> generic 3×3 model and numerically on the block model, with the elementwise
> bound; the explicit tensor form of $\tilde g$ equals $J + \mathcal A't^0$
> in symbols ($n_o = n_v = 2$) and numerically; halving the coupling divides
> the occupied-extended estimator error by about 4 and the virtual-only one
> by about 2 (P2).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_combined_sympy.py`, P2 |
> | Negative control | the runner removes the occupied term $\sum_{k\ne i} f_{ik}t^0_{kj}$ from $\tilde g$ ("P2: Jacobi") |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/ragged.rs::fock_dressed_gate` (the virtual-only form) |
>
> <!-- verified: id=lmp2.combined.jacobi-remainder; script=scripts/lmp2_combined_sympy.py; claim=P2: remainder identity (numeric); runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/ragged.rs::fock_dressed_gate; registry=local-mp2 -->

### P3. The WLLH25 embedding (identities PROVEN; upper bound of the WLLH energy DISPROVED) {#amplitude-lmp2-combined-p3}

Split $\mathcal A = \mathcal B + \mathcal C$. $\mathcal B$ is $\mathcal A$
with the occupied off-diagonals removed: block diagonal over pairs, the
per-pair Sylvester operator of Eq 45.
$\mathcal C = -(f'_{oo}\oplus f'_{oo})\otimes 1_{vv}$ is the inter-pair
coupling, $\lVert\mathcal C\rVert_2 = 2\lVert f'_{oo}\rVert_2$. WLLH25
solves $\tau = -\mathcal B^{-1}J$ per pair, keeps $\tau_D$, and solves the
kept block with the shifted right-hand side $m = \mathcal B_{KD}\tau_D$:
$\mathcal A_{KK}t_K = -J_K - \mathcal B_{KD}\tau_D$,
$t_{\mathrm{emb}} = (t_K, \tau_D)$. For any split, with
$\varrho = J + \mathcal A\,t_{\mathrm{emb}}$,

$$ \varrho_K = \mathcal C_{KD}\,\tau_D, \qquad
\varrho_D = \mathcal C_{DK}t_K + \mathcal C_{DD}\tau_D - \mathcal B_{DK}\mathcal B_{KK}^{-1}\mathcal C_{KK}t_K . \label{eq:lmp2-combined-embedding-residual} $$

So $\mathcal C = 0 \Rightarrow t_{\mathrm{emb}} = t^*$ exactly, and the
whole residual the embedding leaves is linear in the inter-pair occupied
Fock coupling it sets to zero. That is the probe's "$f_{ik}$ zeroed ⇒ WLLH
exact" anchor of section @sec:amplitude-lmp2-fock-screen-measured, now a
theorem. What it cancels exactly is the intra-pair part $\mathcal B_{DK}$
of the coupling that drives $R_D$ in P1, including the virtual Fock
coupling the screen also targets. Two energies can be formed:

- **Full Hylleraas** $E_H(t_{\mathrm{emb}})$ (the probe's "H" column):
  $E_H(t_{\mathrm{emb}}) - E^* = \varrho^{\mathsf T}\mathcal W\mathcal A^{-1}\varrho \ge 0$,
  an upper bound with error $O(\lVert\mathcal C\rVert^2)$ (ratio 4.03 when
  $f'_{oo}$ is halved), and the certificate
  $E_H(t_{\mathrm{emb}}) - E^* \le 3\lVert\varrho\rVert^2/\lambda_{\min}(\mathcal A)$.
  PROVEN. It costs one matvec over $K\cup D$.
- **WLLH25 Eq 41 at convergence**,
  $E_L = \langle\mathcal W t_{\mathrm{emb}}, J\rangle$ (the "lin" column):
  $E_L - E^* = E_H - E^* - \langle\mathcal W t_{\mathrm{emb}},\varrho\rangle$,
  first order in $\mathcal C$ with no fixed sign (ratio 2.03). On the model
  it fell below $E^*$ in 10 of 28 instances. As an upper bound:
  DISPROVED.

Neither energy is guaranteed to beat the plain mask:
$E_H(t_{\mathrm{emb}}) > E_K$ in 4 of 28 model instances (DISPROVED as a
guarantee). The fix is a Galerkin step over the enlarged space
$K \oplus \operatorname{span}\{\tau^{(ij)}_D\}$, one contracted dropped
vector per pair:

$$ E_G = \min_{t_K,\ \alpha_{ij}} E_H\Big(t_K + \sum_{ij}\alpha_{ij}\,\tau^{(ij)}_D\Big), \qquad E^* \le E_G \le \min\big(E_K,\ E_H(t_{\mathrm{emb}})\big) . $$

The bound holds because the family contains $\alpha = 0$ and, at
$\alpha = 1$, the best $t_K$ for the fixed $\tau_D$. $E_G$ is exact at
$\mathcal C = 0$ and size-consistent (P6). The $\tau^{(ij)}_D$ enter a CG
solve as extra unknowns. PROVEN by construction; its cost in production is
OPEN. One global $\alpha$ instead of one per pair is not size-consistent:
on the model it is off by $+2.2\times10^{-6}$ Ha for two non-interacting
fragments.

> **Verified equation @eq:lmp2-combined-embedding-residual** · the embedding residual is linear in the inter-pair coupling
>
> $\varrho_K = \mathcal C_{KD}\tau_D$ for a generic split, $\mathcal C = 0$
> gives $t_{\mathrm{emb}} = t^*$, $E_H(t_{\mathrm{emb}})$ keeps the upper
> bound and its certificate, $E_L$ undershoots $E^*$ on some instances, and
> halving $\lVert\mathcal C\rVert$ divides the $E_H$ error by about 4 and
> the $E_L$ error by about 2 (P3).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_combined_sympy.py`, P3 |
> | Negative control | the runner flips the sign of the embedding right-hand side ("P3: exact when f_oo diagonal") |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/ragged.rs::matvec_indexed` (the operator $\mathcal A$; the embedding has no production code) |
>
> <!-- verified: id=lmp2.combined.embedding-residual; script=scripts/lmp2_combined_sympy.py; claim=P3: rho_K; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/ragged.rs::matvec_indexed; registry=local-mp2 -->

### P4. λ-Boys (PROVEN; global optimality OPEN) {#amplitude-lmp2-combined-p4}

Maximize WLLH25 Eq 9. Eq 8 of the paper is the same up to the
rotation-invariant $\sum_i\langle i|r^2|i\rangle$ and $\lVert f_{oo}\rVert_F^2$.

$$ G(U) = \sum_{x=1}^{3}\sum_i (r^x_{ii})^2 + \lambda\sum_i f_{ii}^2 , \qquad \Phi = \sum_{i\ne j} f_{ij}^2 = \lVert f_{oo}\rVert_F^2 - \sum_i f_{ii}^2 . $$

For $U = 1 + \kappa$, $\kappa^{\mathsf T} = -\kappa$, and any symmetric $X$,
$\partial_{\kappa_{pq}}\sum_i X_{ii}^2 = 4X_{pq}(X_{qq} - X_{pp})$, so

$$ \frac{\partial G}{\partial\kappa_{pq}} = 4\sum_{X}w_X\,X_{pq}\,(X_{qq} - X_{pp}), \qquad (X, w_X) \in \{(r^x, 1),\ (f_{oo}, \lambda)\} , \label{eq:lmp2-combined-lambda-boys} $$

and stationarity is
$\sum_x r^x_{pq}(r^x_{pp} - r^x_{qq}) + \lambda f_{pq}(f_{pp} - f_{qq}) = 0$
for all $p<q$. A 2×2 rotation by $\theta$ gives, with
$d_X = (X_{pp} - X_{qq})/2$ and $c_X = X_{pq}$,

$$ G(\theta) = \mathrm{const} + a\cos4\theta + b\sin4\theta, \qquad a = \sum_X w_X(d_X^2 - c_X^2), \quad b = 2\sum_X w_X d_Xc_X , $$

maximized at $4\theta = \operatorname{atan2}(b, a)$. λ-Boys is therefore
Boys with a fourth "dipole component" $\sqrt\lambda\,f_{oo}$, and each
Jacobi step is the exact maximum in its plane, so $G$ rises monotonically.
The paper's remark that Jacobi sweeps converge less easily concerns rate,
not the step. Each step maximizes only within its plane, so a converged
sweep is a stationary point that is a maximum in every 2×2 plane, not
necessarily a global maximum (OPEN).

If $U_\lambda$ is a global optimum, summing the optimality inequalities at
$\lambda_1 < \lambda_2$ gives $(\lambda_2 - \lambda_1)(\Phi_1 - \Phi_2) \ge 0$,
so $\Phi$ is non-increasing in $\lambda$ (PROVEN UNDER the global-optimum
assumption; seen on the model with warm starts). λ enters P1–P3 through

$$ \lVert\mathcal C\rVert_2 = 2\lVert f'_{oo}\rVert_2 \le 2\sqrt{\Phi} , $$

the occupied part of $\mathcal A'$ in P2's $\rho$, and the embedding
residual of P3:
$\lVert\varrho\rVert \le 2\sqrt\Phi\,\big[2\lVert\tau_D\rVert + (1+\gamma)\lVert t_K\rVert\big]$,
$\gamma = \lVert\mathcal B_{DK}\mathcal B_{KK}^{-1}\rVert_2$, hence
$E_H(t_{\mathrm{emb}}) - E^* \le 12\,\Phi\,[2\lVert\tau_D\rVert + (1+\gamma)\lVert t_K\rVert]^2/\lambda_{\min}(\mathcal A)$.
This bound is not size-extensive. It fixes the scaling with $\Phi$, not the
magnitude.

> **Verified equation @eq:lmp2-combined-lambda-boys** · the λ-Boys gradient and Jacobi angle
>
> The gradient formula, the 2×2 Jacobi form $G(\theta)$ in symbols, and a
> numerical Jacobi λ-Boys sweep that is stationary and rises monotonically
> (P4).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_combined_sympy.py`, P4 |
> | Negative control | the runner drops the factor 4 from the gradient ("P4: gradient") and sets the $f_{oo}$ weight to 0 ("P4: stationarity") |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/lmp2_amplitude.rs::localized_spaces` (plain Boys, $\lambda = 0$; λ-Boys is section @sec:amplitude-lmp2-lambda-boys) |
>
> <!-- verified: id=lmp2.combined.lambda-boys; script=scripts/lmp2_combined_sympy.py; claim=P4: gradient; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/lmp2_amplitude.rs::localized_spaces; registry=local-mp2 -->

### P5. Composition {#amplitude-lmp2-combined-p5}

| piece | error term it attacks | effect on the bound |
|---|---|---|
| screen (virtual-only) | $\mathcal B$-driven part of $t^*_D$, by moving it into $K$ | $E^* \le E_{K'} \le E_K$ (P1) |
| screen (+ occupied) | also the first-order $\mathcal C$-driven part ($\varrho_D \approx \mathcal C_{DK}t_K$) | same |
| embedding | $\mathcal B_{KD}$ coupling of the remaining $D$, exactly | $E_H(t_{\mathrm{emb}})$ an upper bound; $E_L$ no bound |
| λ-Boys | $\lVert\mathcal C\rVert \le 2\sqrt\Phi$: the embedding's whole residual and the occupied part of the screen's | none directly; it changes $J$, hence $K$ |

The screen and the embedding overlap in what they target: both act on the
large dropped exact amplitudes. The screen removes them from $D$, and the
embedding approximates them by $\tau_D$. After a screen the embedding has
less left to recover, consistent with the probe's further factor 1.5–5
instead of 2–5. The occupied extension and λ-Boys overlap too: both reduce
the $\mathcal C$-driven error, one by keeping those amplitudes and the
other by shrinking $\mathcal C$. At a fixed threshold the occupied
extension lowers the dropped-amplitude norm only modestly (P2).

The combination is an upper bound when every piece ends in a Galerkin
solve over a subspace of the universe (plain, screened or
embedded-Galerkin, $E_G$), or when the embedding is scored with the full
Hylleraas form. With the WLLH25 linear energy it is not, and a combined
method using it needs the a-posteriori estimate of P2 and P3 to be trusted.
λ trades $\Phi$ against locality: larger $\lambda$ spreads $J$ and enlarges
$K$ at fixed $\varepsilon$. Where the optimum lies is OPEN and empirical.

### P6. Limits and invariances {#amplitude-lmp2-combined-p6}

- $\varepsilon \to 0$: Eq 8 keeps everything, the screen is inert, and
  $D = \emptyset$ makes $\tau_D$ and $m$ vanish. Every combination is exact
  RI-MP2 in any orthonormal orbitals. PROVEN.
- $\kappa \to \infty$, embedding off: the screen adds nothing and the
  pattern is the WSHG23 Eq-8 mask. PROVEN.
- $\lambda = 0$: $G$ is the Boys functional. PROVEN (definition).
- The untruncated $E^* = -J^{\mathsf T}\mathcal W\mathcal A^{-1}J$ is
  invariant under orthogonal $U_o$, $U_v$:
  $Q = U_o\otimes U_o\otimes U_v\otimes U_v$ is orthogonal,
  $J\to Q^{\mathsf T}J$, $\mathcal A\to Q^{\mathsf T}\mathcal AQ$, and $Q$
  commutes with $\mathcal S$. The truncated energies are not invariant.
  PROVEN.
- Size-consistency (PROVEN UNDER ASSUMPTION). Take non-interacting
  fragments, with every cross-fragment $(ia|jb)$ and Fock element exactly
  zero, and fragment-pure orbitals identical to the isolated fragments'.
  Then $\mathcal A$ preserves the fragment label of each index, $J$ lives
  only in the all-A and all-B sectors, and the screen's $\mathcal A' t^0$
  and the per-pair $\mathcal B^{-1}$ stay in them. Every arm (plain,
  screen, $E_H(t_{\mathrm{emb}})$, $E_L$, $E_G$ with per-pair $\alpha$)
  gives $E(A{+}B) = E(A) + E(B)$. The localizer half of the assumption
  holds for Jacobi λ-Boys: on a cross-fragment plane $c_X = 0$ for every
  $X$, so $b = 0$, $a > 0$ and $\theta = 0$ exactly. Fragment orbitals are
  never mixed, and the intra-fragment rotations run in the same order as
  in the isolated run, so the result matches it up to the stopping
  tolerance. A localizer that can stop at a saddle (PySCF's default Boys
  optimizer was observed to) gives no such guarantee. The same argument
  needs the VV-HV construction, the candidate squares and the pair gate to
  be fragment-local.

The script checks each limit numerically on the block model: at
$\varepsilon = 0$ every arm gives $E^*$, at $\kappa \to \infty$ the screened
pattern is the Eq-8 pattern, rotating the orbitals leaves $E^*$ unchanged
(mutation "P6: rotational invariance"), and non-interacting fragments give
$E(A{+}B) = E(A) + E(B)$ to a relative $10^{-11}$ for every arm.

## λ-Boys occupied orbitals, the default {#amplitude-lmp2-lambda-boys}

Boys orbitals are as compact as a Jacobi sweep can make them, and nothing
else. In a conjugated molecule several placements of the π bonds have
nearly the same spread, so Boys picks one of them by small differences, and
the one it picks can leave large off-diagonal elements $f_{ik}$ between
neighbouring π orbitals. Those elements drive amplitudes that the
integral-size mask does not see (section @sec:amplitude-lmp2-fock-screen,
P2 and P4 above). λ-Boys adds a penalty on that coupling to the Boys
objective and keeps the rest of the method unchanged.

The local MP2 drivers localize the active occupieds with λ-Boys at
$\lambda = 10$ $\text{Bohr}^2/\text{Ha}^2$. `lambda_boys = 0` gives plain
Boys. The dRPA, LinLCCD and rCCD drivers that share the localized spaces
keep plain Boys, because the measurements below were made on MP2 only.

**The objective and its stationarity condition.** $G$ is the function of
section @sec:amplitude-lmp2-combined-p4. The code rotates orbitals $p<q$ as
$p' = c\,p + s\,q$, $q' = -s\,p + c\,q$ with $c=\cos\theta$, $s=\sin\theta$.
This is $\kappa_{pq} = -\theta$ in the book's $U = 1+\kappa$, so
(@eq:lmp2-combined-lambda-boys) changes sign. With
$d_X = (X_{pp}-X_{qq})/2$, $c_X = X_{pq}$, $a = \sum_X w_X(d_X^2-c_X^2)$ and
$b = 2\sum_X w_X d_Xc_X$:

$$ \left.\frac{\partial G}{\partial\theta_{pq}}\right|_{\theta=0} = 4\sum_X w_X\,(X_{pp}-X_{qq})\,X_{pq} = 4b_{pq}, \qquad G'' (\theta^*) = -16\sqrt{a^2+b^2}\ \text{at}\ \theta^* = \tfrac14\operatorname{atan2}(b, a) . \label{eq:lmp2-lambda-boys-stationarity} $$

A point is stationary when $b_{pq} = 0$ for every pair. There the step
angle is zero exactly when $a_{pq} > 0$, so a sweep that moves nothing ends
on a point that is a maximum in every plane. Two limits:

- $\lambda = 0$: $a$ and $b$ are the $A$ and $B$ of
  `ferric-mp2/src/boys.rs::boys_localize`, which rotates by
  $x' = c\,x - s\,y$ through $\tfrac14\operatorname{atan2}(-B, A)$. That is
  the same rotation, so the λ = 0 step is the Boys step.
- Two orbitals on non-interacting fragments: $c_X = 0$ for every $X$,
  $b = 0$, $a > 0$ and $\theta = 0$. The penalty never mixes fragments.

The energy at $\varepsilon = 0$ does not depend on λ, because $E^*$ is
invariant under any orthogonal occupied rotation (P6). At finite
$\varepsilon$, λ changes $J$ and therefore the kept set, but the error keeps
its sign: P1 holds in any orthonormal occupied basis, and $E^*$ is the same
in all of them.

**The code.** `ferric-mp2/src/boys.rs::lambda_boys_localize_checked` runs
converged Boys first, forms $r^x$ and $f$ in the Boys orbitals, and calls
`ferric-mp2/src/boys.rs::lambda_boys_rotation`. Its sweep, with the setup
and the result struct removed:

<!-- excerpt: ferric-mp2/src/boys.rs::lambda_boys_rotation -->
```rust
for p in 0..n {
    for q in p + 1..n {
        // θ* = atan2(b, a)/4: the maximum of G in the (p, q) plane
        let th = lambda_boys_pair_angle(&x, &f, lambda, p, q);
        if th == 0.0 {
            continue; // b = 0 and a > 0: already the in-plane maximum
        }
        let (c, s) = (th.cos(), th.sin());
        // r^x ← Rᵀ r^x R and f ← Rᵀ f R
        for m in x.iter_mut() {
            rotate_sym_pair(m, p, q, c, s);
        }
        rotate_sym_pair(&mut f, p, q, c, s);
        // U ← U R: p' = c p + s q, q' = −s p + c q
        for k in 0..n {
            let (a, b) = (u[(k, p)], u[(k, q)]);
            u[(k, p)] = c * a + s * b;
            u[(k, q)] = -s * a + c * b;
        }
    }
}
// stop on max over p < q of |∂G/∂θ_pq| after the whole sweep
max_grad = lambda_boys_max_grad(&x, &f, lambda);
```

- The sweep works on $n_o\times n_o$ matrices only. The AO coefficients are
  rotated once at the end, $C_{\mathrm{locc}} = C_{\mathrm{Boys}}U$.
- The pair order is fixed and the loop is serial, so the result does not
  depend on the thread count.
- The stop is $\max_{p<q}\lvert 4b_{pq}\rvert < 10^{-8}$ $\text{Bohr}^2$
  (`LAMBDA_BOYS_GRAD_TOL`), evaluated after a whole sweep. A run that does
  not reach it within `LAMBDA_BOYS_MAX_SWEEPS` sweeps is a `Convergence`
  error, the same rule as for Boys: the orbitals would still span the
  occupied space, but the pair gate, the domains and the mask are built
  from them.
- At $\lambda = 0$ the optimiser is not called and the Boys orbitals are
  returned bit for bit.

**Design choices.** Why start from Boys and not from the canonical
orbitals? $G$ has several local maxima, and on naphthalene the λ = 10
optimum is a different π localization from the Boys one (the largest
element of $\lvert U\rvert - 1$ is 0.707). Starting from converged Boys makes
the λ-Boys result a deterministic function of the Boys result, and the
λ = 0 limit the Boys orbitals themselves. Why 10? It is the smallest value
of the measured grid at which naphthalene moves to the better localization;
on octane larger λ spreads the orbitals and costs accuracy (λ = 100 made
$f_{oo}$ 40% more diagonal and the largest spread 1.5 times larger).

**Measured.** 6-31G, at matched kept fraction, on the dense oracle of
section @sec:amplitude-lmp2-fock-screen-measured:

| System | Effect of λ = 10 against λ = 0 |
|---|---|
| naphthalene | error 10 to 30 times smaller in every arm, plain mask included; at keep 0.83, $6.8\times10^{-7}$ to $3.8\times10^{-8}$ Ha. Largest orbital spread 5.1 to 8.5 $\text{Bohr}^2$ |
| n-octane | at most 2.5% better; 2 to 7% worse with the screen at $\kappa = 0.3$, no change at $\kappa = 1$ |
| adamantane | no change at keep 0.99 |

End to end, from the `76ffd5a79` CLI build: 6-31G, cc-pvdz-ri,
$\varepsilon = 10^{-4}$, frozen core (one orbital per heavy atom), one run
each on a quiet box. The error is against canonical RI-MP2 in the same
basis; $t_{\mathrm{corr}}$ is assembly plus solve. Single runs, so a
difference of a few percent in $t_{\mathrm{corr}}$ is noise:

| System, path | λ-Boys sweeps | λ-Boys time | $t_{\mathrm{corr}}$, λ = 0 / 10 | λ-Boys share of $t_{\mathrm{corr}}$ | error, λ = 0 / 10 (Ha) |
|---|---|---|---|---|---|
| n-octane, in core | 7 | 2.2 ms | 0.87 / 0.88 s | 0.25% | $1.912 / 1.913\times10^{-3}$ |
| naphthalene, in core | 21 | 5.0 ms | 1.60 / 1.33 s | 0.37% | $3.733 / 2.135\times10^{-3}$ |
| n-octane, integral-direct | 7 | 2.2 ms | 1.30 / 1.28 s | 0.17% | $1.913 / 1.916\times10^{-3}$ |
| n-C20H42, integral-direct | 7 | 17.9 ms | 8.53 / 8.87 s | 0.20% | $6.434 / 6.466\times10^{-3}$ |

On n-butane / cc-pVDZ with all electrons the in-core recovery moves from
99.980, 99.671, 94.99% to 99.992, 99.735, 95.20% at
$\varepsilon = 10^{-5}$, $10^{-4}$, $10^{-3}$, and n-octane at
$10^{-3}$ from 93.81 to 94.00%. With frozen core the same n-butane errors
move by 2 to 3% in either direction. The frozen-core alkane errors grow
by 0.05 to 0.5%; the default is taken for the
naphthalene gain at a cost below 0.4% of $t_{\mathrm{corr}}$.

> **Verified equation @eq:lmp2-lambda-boys-stationarity** · the λ-Boys pair gradient, the in-plane maximum and the λ = 0 limit
>
> The pair gradient in the code's rotation convention, the Jacobi form and
> its second derivative at $\theta^*$, the stationarity condition
> $b_{pq} = 0$, and the identity of the λ = 0 step with the Boys step,
> symbolically; numerically, the gradient against a finite difference along
> $\exp(sK)$, and a Jacobi optimum that is stationary, rises monotonically
> and has negative curvature along random directions.
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_lambda_boys_sympy.py`, S1 to S4 |
> | Negative control | the runner drops the factor 4 ("S1: gradient"), takes the minimum branch of the angle ("S2: maximum"), drops λ from the angle ("S3: stationarity") and flips the Boys angle ("S4: Boys") |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/boys.rs::lambda_boys_rotation` |
>
> <!-- verified: id=lmp2.lambda-boys; script=scripts/lmp2_lambda_boys_sympy.py; claim=S3: stationarity; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/boys.rs::{lambda_boys_rotation,lambda_boys_pair_gradient,lambda_boys_pair_angle}; rust_test=ferric-mp2/src/boys.rs::lambda_boys_pair_gradient_matches_central_difference; registry=local-mp2 -->

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
  unchanged: `eps` itself, `pair_gate_cal` and `fock_screen`. On a swap-closed pattern,
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
| `lambda_boys` | λ of the λ-Boys occupied localization, default 10; 0 is plain Boys. Changes the orbitals, so $J$ and the kept set; the error stays one-sided because $E^*$ is rotation invariant; inert at $\varepsilon = 0$ | section @sec:amplitude-lmp2-lambda-boys |
| `fock_screen` | also keeps elements whose one-step Fock-propagated integral clears $\kappa\varepsilon$ (section @sec:amplitude-lmp2-fock-screen); inert at $\varepsilon = 0$. On by default (κ = 1) on the integral-direct CLI and Python paths, `[local] fock_screen = false` to turn it off; off in the in-core path and the Rust default | `ferric-mp2/src/ragged.rs::pair_blocks_screened` |
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
| finite $\varepsilon$ on n-butane / cc-pVDZ, all electrons | PySCF `DFMP2` (canonical) | recovers 99.99%, 99.73%, 95.2% of the DF-MP2 correlation energy in core (λ-Boys, λ = 10) at $\varepsilon = 10^{-5}$, $10^{-4}$, $10^{-3}$ with production maps; the error is one-sided and has no tolerance | same |

<!-- figures: site/src/reference/validation.md "Local RI-MP2 (`rimp2` with `[local] scheme = \"amplitude-threshold\"`)" -->

ORCA DLPNO-MP2 on the same molecules is recorded on the validation page as
a ballpark only. It truncates by pair-natural-orbital occupation, not by an
amplitude threshold, so the two recoveries are not compared.

**The error is superlinear in $\varepsilon$.** As $\varepsilon$ grows by
factors of ten, the recovery falls from 99.99% to 99.73% to 95.2%. The
missing correlation energy therefore grows faster than linearly in
$\varepsilon$. There is no tolerance on it. To reach a target accuracy, run
a series of $\varepsilon$ values and choose from it.

n-Butane in cc-pVDZ, all electrons, from the `76ffd5a79` build (λ-Boys, λ = 10). Each line
of the output is cut after `keep_fraction`; the full `local` dict also
carries the solver statistics, the timings and the trust text. The
$\varepsilon = 0$ run reproduces the exact RI-MP2 energy to
$2.726\times10^{-13}$ Ha, and the recovery falls as stated above:

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
eps=  0e+00  E_corr=-0.5996452251  recovery=100.0000%  E-E*=+2.726e-13 Ha  local={'scheme': 'amplitude-threshold', 'eps': 0.0, 'keep_fraction': 1.0, ...}
eps=  1e-05  E_corr=-0.5995980733  recovery= 99.9921%  E-E*=+4.715e-05 Ha  local={'scheme': 'amplitude-threshold', 'eps': 1e-05, 'keep_fraction': 0.5846693712871353, ...}
eps=  1e-04  E_corr=-0.5980550045  recovery= 99.7348%  E-E*=+1.590e-03 Ha  local={'scheme': 'amplitude-threshold', 'eps': 0.0001, 'keep_fraction': 0.21723822050709232, ...}
eps=  1e-03  E_corr=-0.5708766759  recovery= 95.2024%  E-E*=+2.877e-02 Ha  local={'scheme': 'amplitude-threshold', 'eps': 0.001, 'keep_fraction': 0.02984183343387928, ...}
```

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/lmp2_amplitude.rs`, `crates/ferric-mp2/src/ragged.rs` | the algorithm, the policy and the mask |
| `scripts/lmp2_hylleraas_vs_wshg23_sympy.py`, claims 1, 2, 4, 5, 6 | verified equations @eq:lmp2-hylleraas and @eq:lmp2-cg-bound |
| `scripts/lmp2_fock_screen_sympy.py`, claims 1, 2, 3 | verified equation @eq:lmp2-fock-screen-schur |
| `scripts/lmp2_lambda_boys_sympy.py`, S1 to S4 | verified equation @eq:lmp2-lambda-boys-stationarity |
| `scripts/lmp2_combined_sympy.py`, P1 to P6 | verified equations @eq:lmp2-combined-truncation-error, @eq:lmp2-combined-jacobi-remainder, @eq:lmp2-combined-embedding-residual and @eq:lmp2-combined-lambda-boys |
| `crates/ferric-mp2/tests/lmp2_fock_screen_probe.rs` | the dense-oracle measurements of section @sec:amplitude-lmp2-fock-screen-measured |
| Wang, Ling, Shi, Shen, Wang, Yang, Li and Head-Gordon, JCTC 21, 10910 (2025) | the block-Kapuy embedding compared with the screen |
| Wang, Aldossary, Shi, Liu, Li and Head-Gordon, J. Chem. Theory Comput. 19, 7577 (2023) | the single-threshold scheme and the Hylleraas stopping rule, as cited in the module documentation |
| `site/src/methods/mp2.md`, "Exact and local MP2" | the user-facing description and the certificate fields |
| `site/src/reference/validation.md` | the anchor row |
