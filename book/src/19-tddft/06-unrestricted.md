# Unrestricted references and $\langle S^2\rangle$ {#unrestricted}

<!-- status: ready; registry: tddft -->

A radical such as NH2 has one more $\alpha$ electron than $\beta$. In a UHF
or UKS reference the extra $\alpha$ electron pulls the $\alpha$ orbitals
into a different shape from the $\beta$ ones. Section @sec:triplets split the
response on $X_\alpha \pm X_\beta$, and that needed identical spatial
orbitals for both spins, so the split is gone here. ferric solves the
spin-conserving problem over both spins at once. Its roots are not spin
eigenfunctions: a root can be part doublet and part quartet. So each root
comes with $\langle S^2\rangle$, which says how far it is from a pure spin
state.

| Symbol | Meaning | Code name |
|---|---|---|
| $ia\sigma$ | compound index, $\alpha$ block first, then $\beta$ | `x[i * nva + a]`, `x[na * nva + i * nvb + a]` |
| $\bar\jmath, \bar b$ | $\beta$ occupied and virtual orbitals | |
| $O = C_\alpha^{\mathsf T}SC_\beta$ | $\alpha$–$\beta$ MO overlap | `SpinOverlap::o` |
| $P, Q, R$ | $O^{\mathsf T}O$, $O_{:,\mathrm{occ}\beta}O_{:,\mathrm{occ}\beta}^{\mathsf T}$, $O_{\mathrm{occ}\alpha,:}^{\mathsf T}O_{\mathrm{occ}\alpha,:}$ | `p`, `q`, `r` |
| $M$ | $(n_\alpha - n_\beta)/2$ | `m_terms` returns $M^2 + M$ |

| Entry point | Location |
|---|---|
| Rust | `ferric-tddft/src/tddft.rs::run_tddft` → `ferric-tddft/src/unrestricted.rs::{run_unrestricted,casida_x,SpinOverlap}` |
| CLI | `method.kind = "tda"` or `"tddft"` with multiplicity > 1 |
| Python | `run_tddft(...)` on a UHF or UKS reference; the result carries `s2` per root |

## The two-spin problem {#unrestricted-spin-problem}

The space is every spin-conserving single excitation: the compound indices
$ia\alpha$ ($d_\alpha = n^\alpha_on^\alpha_v$) followed by $ia\beta$
($d_\beta$), of dimension $d_\alpha + d_\beta$. Each orbital belongs to its
own spin, and the matrices are the spin-orbital ones of section @sec:triplets with
that one change:

$$ A_{ia\sigma,jb\tau} = \delta_{\sigma\tau}\delta_{ij}\delta_{ab}(\varepsilon^\sigma_a - \varepsilon^\sigma_i)
+ (i_\sigma a_\sigma|j_\tau b_\tau) + (i_\sigma a_\sigma|f_{\sigma\tau}|j_\tau b_\tau)
- \delta_{\sigma\tau}\,c\,(i_\sigma j_\sigma|a_\sigma b_\sigma), \label{eq:td-unrestricted-matrices} $$

$$ B_{ia\sigma,jb\tau} = (i_\sigma a_\sigma|j_\tau b_\tau) + (i_\sigma a_\sigma|f_{\sigma\tau}|j_\tau b_\tau)
- \delta_{\sigma\tau}\,c\,(i_\sigma b_\sigma|j_\sigma a_\sigma) , $$

with $c(pq|rs)$ the exchange combination of section @sec:rsh (global, HF or
range-separated). These are the blocks of PySCF's `tdscf.uhf.get_ab`:
$\mathbf A_{\alpha\alpha}$, $\mathbf A_{\alpha\beta}$ and
$\mathbf A_{\beta\beta}$, with
$\mathbf A_{\beta\alpha} = \mathbf A_{\alpha\beta}^{\mathsf T}$. TDA, the
Casida reduction and the Davidson solvers of section @sec:response apply unchanged to
the $(d_\alpha + d_\beta)$-dimensional matrices.

### Assembly {#unrestricted-assembly}

Equation @eq:td-unrestricted-matrices reuses the restricted builders. Each same-spin block is the
restricted $\mathbf A$ or $\mathbf B$ of section @sec:response-matrices over that spin's
orbitals and RI tensors, with that spin's exchange and Coulomb coefficient
1 in place of 2: there is one $(ia|jb)$ per spin pair and no spin
adaptation. The cross-spin block is the Coulomb term alone,
$B_{ov}^{\alpha\mathsf T}B_{ov}^\beta$, in $\mathbf A$ and in $\mathbf B$ (the
orbitals are real). There is no cross-spin exchange, because exchange
needs equal spins. `ferric-tddft/src/unrestricted.rs::fill_blocks` places
the pieces:

<!-- excerpt: ferric-tddft/src/unrestricted.rs::fill_blocks -->
```rust
let [ma, mb] = blocks;
m.slice_mut(s![..d, ..d]).assign(&ma);
m.slice_mut(s![d.., d..]).assign(&mb);
let cross = ri[0].b_ov.t().dot(&ri[1].b_ov);
m.slice_mut(s![..d, d..]).assign(&cross);
m.slice_mut(s![d.., ..d]).assign(&cross.t());
```

Here `d` is $d_\alpha$, and `ma`, `mb` come from `build_a_matrix` (or
`build_b_matrix`) called with `c_j = 1.0`. The kernel block is added to the
assembled matrix afterwards.

The matrix-free products use the same pieces. Per spin they take the
$\Delta$ and exchange products of `matvec.rs`, with Coulomb coefficient 0
and no kernel. To those they add the spin-summed Coulomb product
$J(x)_\sigma = B_{ov}^{\sigma\mathsf T}(B_{ov}^\alpha x_\alpha + B_{ov}^\beta x_\beta)$
and the kernel $F$. $(\mathbf A+\mathbf B)x$ takes $2J$ and $2F$, and
$(\mathbf A-\mathbf B)x$ neither, since both terms are equal in
$\mathbf A$ and $\mathbf B$.

### The kernel {#unrestricted-kernel}

`ferric_dft::lr_kernel::UksKernel` is the spin-resolved AO kernel at the
UKS reference $(D^\alpha, D^\beta)$, screened per spin on its own
$\rho_\sigma$ as the UKS $V_{xc}$ is. `UksOvFxc` applies it to one vector
at a time. It forms the AO perturbation of each spin,

$$ \delta D^\sigma = \tfrac12\big(Y_\sigma + Y_\sigma^{\mathsf T}\big),\qquad
Y_\sigma = C^\sigma_{\rm occ}X_\sigma C^{\sigma\mathsf T}_{\rm vir}, $$

then projects the potential back:
$(Kx)_\sigma = C^{\sigma\mathsf T}_{\rm occ}\,\delta V_\sigma[\delta D^\alpha, \delta D^\beta]\,C^\sigma_{\rm vir}$.
A unit vector of spin $\tau$ gives column $jb\tau$ of both spin blocks,
$(ia\sigma|f_{\sigma\tau}|jb\tau)$, with the normalization of the
restricted block. At a restricted point
$K_{\alpha\alpha} \pm K_{\alpha\beta}$ is the singlet or triplet block of
section @sec:triplets. The dense path applies the operator to the identity, checks
the asymmetry against `FXC_BLOCK_ASYMMETRY_TOL` and symmetrizes.

Every functional the restricted path accepts is accepted here, $\tau$-only
meta-GGAs and hybrid meta-GGAs included (`MggaFxcKernel` is polarized).
VV10 stays refused.

### Occupations and refusals {#unrestricted-occupations-refusals}

$n_\alpha$ and $n_\beta$ come from the molecule's charge and multiplicity,
and the occupied orbitals are the lowest $n_\sigma$ of each spin.

- **A non-aufbau (MOM) reference** is refused, because its density is not
  built from those orbitals.
- **A triplet request** is refused, since the roots are not spin-adapted.
  What an open-shell reference gives is every spin-conserving
  ($\Delta M_S = 0$) excitation. For a triplet ground state such as O2
  that means the excited states with the reference's $M_S = 1$; states
  reached by flipping a spin (spin-flip TDDFT) are not computed.
- **An ROHF or ROKS reference** is refused. ROHF orbitals diagonalize one
  Roothaan-coupled Fock operator, not separate $\alpha$ and $\beta$ Fock
  operators, so equation @eq:td-unrestricted-matrices, with orbital-energy differences on its
  diagonal, is not the response of that reference. The ROKS response needs
  the ROHF orbital-rotation Hessian.

### The closed-shell limit {#unrestricted-closed-shell-limit}

For a closed-shell molecule $C_\alpha = C_\beta$, and the problem splits on
$X_\alpha \pm X_\beta$ as in section @sec:triplets. The unrestricted roots are the
union of the restricted singlets and triplets, with $\langle S^2\rangle = 0$
and 2. Only two things differ from the restricted run: rounding, since the
Coulomb term is summed as $B^{\mathsf T}B + B^{\mathsf T}B$ in place of
$2B^{\mathsf T}B$, and the kernel's density screen, applied per spin in
place of on $\rho_\alpha + \rho_\beta$, which acts only at points at the
density floor.

### Transition dipoles {#unrestricted-transition-dipoles}

Each spin is its own block, so there is no $\sqrt2$:

$$ \langle 0|\mathbf r|n\rangle = \sum_\sigma\sum_{ia}(X+Y)_{ia\sigma,n}\langle i_\sigma|\mathbf r|a_\sigma\rangle,\qquad
f_n = \tfrac23\,\Omega_n\,|\langle 0|\mathbf r|n\rangle|^2 , $$

as in PySCF `tdscf.uhf`. For a closed shell, $X_\alpha = X_\beta = X/\sqrt2$
turns the spin sum into equation @eq:td-osc-sum-rule.

> **Verified equation @eq:td-unrestricted-matrices** · the unrestricted matrices
>
> For a random model with 2 $\alpha$ and 1 $\beta$ occupied orbitals,
> different $\alpha$ and $\beta$ orbitals and a spin-dependent kernel, the
> code's dense $\mathbf A$ and $\mathbf B$ (RSH and kernel) equal the
> spin-orbital construction; CIS by brute force over determinants equals
> $\mathbf A$ elementwise; and for $C_\alpha = C_\beta$ the roots are the
> restricted singlets and triplets and the transition moment reduces to the
> $\sqrt2$ form.
>
> | | |
> |---|---|
> | Proof | `scripts/tddft_response_sympy.py`, claim B5 |
> | Negative control | the runner drops the cross-spin Coulomb block of $\mathbf A$ (B5 fails); in the Rust validation, dropping the $\alpha\beta$ kernel block moves the TDA roots by 0.018 to 0.128 eV |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-tddft/src/unrestricted.rs::run_unrestricted` |
> | Code test | `closed_shell_through_the_unrestricted_route_is_singlets_plus_triplets` in `crates/ferric-tddft/src/unrestricted/tests.rs`; `nh2_pbe`, `o2_b3lyp` and the other cases in `crates/ferric-tddft/tests/validation_tddft_open_shell.rs` (against PySCF `tdscf.uhf.get_ab`) |
>
> <!-- verified: id=td.unrestricted.matrices; script=scripts/tddft_response_sympy.py; claim=B5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-tddft/src/unrestricted.rs::run_unrestricted; rust_test=ferric-tddft/src/unrestricted/tests.rs::closed_shell_through_the_unrestricted_route_is_singlets_plus_triplets; registry=tddft -->

## $\langle S^2\rangle$ of each root {#unrestricted-each-root}

### Which wavefunction {#unrestricted-which-wavefunction}

To compute $\langle S^2\rangle$ a root needs a wavefunction. Each root is
given

$$ |\Psi\rangle = \sum_{ia\sigma}X_{ia\sigma}\,a^\dagger_{a\sigma}a_{i\sigma}|\Phi\rangle $$

with unit-normalized $X$. For TDA that is the eigenvector. For Casida it is
the $X$ part, renormalized: $X = \tfrac12\big((X+Y) + (X-Y)\big)$, with
$X - Y = (\mathbf A+\mathbf B)(X+Y)/\Omega$ from the first row-sum equation
of section @sec:response-tda-casida-reduction. Casida response defines no excited-state wavefunction;
$X$ is the particle–hole part of the root. Taking it as the state is
ferric's convention. The validation generator evaluates the same $X$-only
state in numpy, so a Casida $\langle S^2\rangle$ is checked against this
definition, not against an independent Casida $\langle S^2\rangle$.
`ferric-tddft/src/unrestricted.rs::casida_x` computes the unnormalized $X$:

<!-- excerpt: ferric-tddft/src/unrestricted.rs::casida_x -->
```rust
let mut x = xpy.clone();
for (n, mut col) in x.columns_mut().into_iter().enumerate() {
    col += &(&apb_xpy.column(n) / omega[n]);
    col *= 0.5;
}
```

The renormalization happens in `SpinOverlap::excited`, which divides by
$|X|$.

### The operator {#unrestricted-operator}

Write $S^2 = M^2 + M + S_-S_+$ with $M = (n_\alpha - n_\beta)/2$, the
$S_z$ eigenvalue of every determinant in the space. $S_+$ turns a $\beta$
electron into an $\alpha$ one. The $\alpha$ and $\beta$ orbitals differ, so
in the MO basis it carries their overlap. With $O = C_\alpha^{\mathsf T}SC_\beta$
over all MOs ($S$ the AO overlap),

$$ S_+ = \sum_{p\in\alpha,\,q\in\beta}O_{pq}\,a^\dagger_{p\alpha}a_{q\beta} . $$

Form $S_-S_+$ and move the $\beta$ creator past the two $\alpha$
operators. The anticommutator leaves a one-body $\beta$ term:

$$ S_-S_+ = \sum_{sq\in\beta}P_{sq}E^\beta_{sq} - \sum_{pr\in\alpha}\sum_{sq\in\beta}O_{rs}O_{pq}\,E^\alpha_{pr}E^\beta_{sq},
\qquad P = O^{\mathsf T}O . $$

### Matrix elements {#unrestricted-matrix-elements}

Two rules give everything. Within one spin, between single excitations,

$$ \langle\Phi_i^a|E_{pr}|\Phi_j^b\rangle = \delta_{ab}\delta_{ij}\gamma^0_{pr} + \delta_{ij}\delta_{pa}\delta_{rb} - \delta_{ab}\delta_{pj}\delta_{ri} , $$

with $\gamma^0$ the reference density matrix of that spin. Across spins, an
$\alpha$-even times $\beta$-even operator factorizes:
$\langle\Phi_{i\alpha}^{a}|E^\alpha_{pr}E^\beta_{sq}|\Phi_{\bar\jmath}^{\bar b}\rangle = \delta_{pa}\delta_{ri}\,\delta_{s\bar\jmath}\delta_{q\bar b}$,
with overbars marking $\beta$ orbitals.

Collect the terms. Each comes from one piece of $S_-S_+$:

- the one-body $\beta$ term $\sum P_{sq}E^\beta_{sq}$, between $\beta$
  excitations by the same-spin rule and with the reference value on the
  $\alpha$ ones, gives $\sum_{\bar\jmath}P_{\bar\jmath\bar\jmath} + \pi(X_\beta,P)$;
- the two-body term between two $\alpha$ excitations, where $E^\beta_{sq}$
  acts on the $\beta$ reference and gives $\delta_{sq}$ over occupied
  $\bar\jmath$, collapses $\sum_{\bar\jmath}O_{r\bar\jmath}O_{p\bar\jmath}$ to
  $Q_{pr}$ and gives the $Q$ bracket;
- the same term between two $\beta$ excitations, with $E^\alpha_{pr}$ on
  the $\alpha$ reference, collapses to $R_{sq}$ and gives the $R$ bracket;
- the same term between an $\alpha$ and a $\beta$ excitation, by the
  cross-spin rule, gives the last term, with the 2 from the two orders of
  the pair.

Write $X_\alpha$ ($n_\alpha\times n^\alpha_{\rm vir}$)
and $X_\beta$ as matrices, $w_\sigma = |X_\sigma|^2$ (so
$w_\alpha + w_\beta = 1$), and define $Q$ ($\alpha\times\alpha$) and $R$
($\beta\times\beta$) as in the table. For a matrix $N$ over one spin let
$\pi(X, N) = \mathrm{tr}(XN_{vv}X^{\mathsf T}) - \mathrm{tr}(X^{\mathsf T}N_{oo}X)$:
the particle part minus the hole part. Then

$$ \langle S_-S_+\rangle = \sum_{\bar\jmath}P_{\bar\jmath\bar\jmath} + \pi(X_\beta,P)
- \big[w_\alpha\mathrm{tr}\,Q_{oo} + \pi(X_\alpha,Q)\big]
- \big[w_\beta\mathrm{tr}\,R_{oo} + \pi(X_\beta,R)\big]
- 2\,\mathrm{tr}\big(X_\alpha^{\mathsf T}O_{oo}X_\beta O_{vv}^{\mathsf T}\big) , \label{eq:td-s2-per-root} $$

where $O_{oo}$ is the $\alpha$-occupied × $\beta$-occupied block and
$O_{vv}$ the $\alpha$-virtual × $\beta$-virtual block.

Check the limits. With no excitation it reduces to the reference value
$M^2 + M + \sum_{\bar\jmath}P_{\bar\jmath\bar\jmath} - \sum_{i,\bar\jmath}O_{i\bar\jmath}^2$,
PySCF's `UHF.spin_square`. For a closed shell, $O = 1$: singlets
($X_\alpha = X_\beta$) give 0 and triplets ($X_\alpha = -X_\beta$) give 2.
The cross-spin term is what tells them apart; it is the only term that
depends on the relative sign of $X_\alpha$ and $X_\beta$.

`ferric-tddft/src/unrestricted.rs::excited` evaluates equation @eq:td-s2-per-root
term by term:

<!-- excerpt: ferric-tddft/src/unrestricted.rs::excited -->
```rust
let one_body = (wa + wb) * self.beta_occ_weight() + ph(&xb, &p, nb);
let two_alpha = wa * trace(q.slice(s![..na, ..na])) + ph(&xa, &q, na);
let two_beta = wb * trace(r.slice(s![..nb, ..nb])) + ph(&xb, &r, nb);
```

| Term of equation @eq:td-s2-per-root | Code |
|---|---|
| $\sum_{\bar\jmath}P_{\bar\jmath\bar\jmath}$ | `beta_occ_weight` ($w_\alpha + w_\beta = 1$) |
| $\pi(X, N)$ | `ph` |
| the cross-spin term | `cross` |

The function returns `self.m_terms() + one_body - two_alpha - two_beta - cross`,
with `m_terms` the $M^2 + M$ of the first line. `beta_occ_weight` is
$\sum_{\bar\jmath}P_{\bar\jmath\bar\jmath}$, which equals $n_\beta$ when the
$\alpha$ MOs span the $\beta$ ones; its factor $w_\alpha + w_\beta$ is 1
after the normalization.

> **Verified equation @eq:td-s2-per-root** · $\langle S^2\rangle$ of a single-excitation state
>
> The transcribed `SpinOverlap::reference` and `SpinOverlap::excited` equal
> $M^2 + M + \lVert S_+\Psi\rVert^2$ evaluated by Jordan–Wigner on the
> determinant space, for 4 spatial orbitals per spin,
> $(n_\alpha, n_\beta) = (2,1), (3,1), (2,2)$ and 24 random $X$, with a
> general $O$ (the $\alpha$ and $\beta$ MOs span different AO subspaces, so
> $P \ne 1$) and an orthogonal one; for $O = 1$, $X_\alpha = \pm X_\beta$
> gives 0 and 2; `casida_x` equals the $X$ block of the eigenvector of
> equation @eq:response-2 normalized to $X\cdot X - Y\cdot Y = 1$.
>
> | | |
> |---|---|
> | Proof | `scripts/tddft_response_sympy.py`, claims s2ref, s2, casidax |
> | Negative control | the runner drops the 2 on the cross-spin term (D2 fails), drops the occupied–occupied overlap from the reference value (D1 fails), and takes the $Y$ block in `casida_x` (D4 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-tddft/src/unrestricted.rs::{excited,reference,casida_x}` |
> | Code test | `crates/ferric-tddft/tests/validation_tddft_open_shell.rs` (`compare_s2`, against values the generator checks with `pyscf.fci.spin_op.spin_square`) |
>
> <!-- verified: id=td.s2.per-root; script=scripts/tddft_response_sympy.py; claim=s2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-tddft/src/unrestricted.rs::excited; rust_test=ferric-tddft/tests/validation_tddft_open_shell.rs::compare_s2; registry=tddft -->

The generator `scripts/validation/gen_tddft_open_shell_refs.py` evaluates
equation @eq:td-s2-per-root in numpy. For OH and NH2 it also expands the same $X$ into a
full determinant vector over the UHF orbitals and takes $\langle S^2\rangle$
with `pyscf.fci.spin_op.spin_square`, which builds it from the 2-RDMs, an
independent construction. It asserts agreement to $10^{-10}$ on every root,
and the Rust test compares ferric's values with the recorded ones.

NH2 at PBE / 6-31G and closed-shell water through the unrestricted route,
from the `589b50832` build. On NH2 most roots sit near the doublet value
0.75, and roots 3 and 6 are strongly spin-contaminated ($\langle S^2\rangle
\approx 2.7$). On water each root has $\langle S^2\rangle$ 0 or 2, and the
energies are the restricted singlets and triplets of the example in
@sec:triplets-spin-blocks, to $10^{-5}$ eV.

```python
import ferric

HA_TO_EV = 27.211386245988
bs = ferric.BasisSet.bundled("6-31g")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

nh2 = ferric.Molecule.from_xyz("testdata/molecules/nh2.xyz", multiplicity=2)
r = ferric.run_tddft(nh2, bs, aux, functional="PBE", n_roots=8, method="tda")
print("NH2 PBE: <S^2> of the reference", r.s2_reference)
for n in range(r.n_roots):
    print(f"  root {n + 1}: {r.excitation_energies[n] * HA_TO_EV:9.5f} eV  <S^2> {r.s2[n]:.5f}  f {r.oscillator_strengths[n]:.5f}")

# Closed-shell water through the unrestricted route: the restricted singlets and triplets, <S^2> 0 or 2.
water = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
u = ferric.run_tddft(water, bs, aux, n_roots=6, method="tda", reference="unrestricted")
for n in range(u.n_roots):
    print(f"  water UHF root {n + 1}: {u.excitation_energies[n] * HA_TO_EV:9.5f} eV  <S^2> {u.s2[n]:.5f}")
```

```text
NH2 PBE: <S^2> of the reference 0.7516594991540426
  root 1:   2.21686 eV  <S^2> 0.75262  f 0.00508
  root 2:   7.05954 eV  <S^2> 0.75241  f 0.00000
  root 3:   7.49707 eV  <S^2> 2.66798  f 0.00407
  root 4:   8.26795 eV  <S^2> 0.75118  f 0.00684
  root 5:   8.59155 eV  <S^2> 0.83434  f 0.11157
  root 6:   9.19905 eV  <S^2> 2.70998  f 0.00308
  root 7:  10.24358 eV  <S^2> 0.75158  f 0.00000
  root 8:  11.00321 eV  <S^2> 0.79357  f 0.13017
  water UHF root 1:   8.46195 eV  <S^2> 2.00000
  water UHF root 2:   9.40530 eV  <S^2> 0.00000
  water UHF root 3:  10.28941 eV  <S^2> 2.00000
  water UHF root 4:  10.71261 eV  <S^2> 2.00000
  water UHF root 5:  11.34728 eV  <S^2> 0.00000
  water UHF root 6:  11.87564 eV  <S^2> 0.00000
```

Where does 2.7 come from? Excite an electron from a doubly occupied orbital
$i$ to a virtual $a$ of NH2, and three electrons are unpaired: the radical's
own $\alpha$ electron and the two in $i$ and $a$. Three spins with
$M_S = \tfrac12$ span one quartet and two doublets over three determinants.
The spin-conserving singles $i_\alpha\to a_\alpha$ and $i_\beta\to a_\beta$
are two of them. In the third the radical's own electron, in orbital $u$,
is the $\beta$ one; from the reference that is the double excitation
$u_\alpha\to a_\alpha$ with $i_\beta\to u_\beta$, outside the space. The
quartet component projects onto the two available determinants with
weight $\tfrac23$. One combination of them is a pure doublet; the other is
$\tfrac23$ quartet and $\tfrac13$ doublet, with
$\langle S^2\rangle = \tfrac23\cdot\tfrac{15}4 + \tfrac13\cdot\tfrac34 = 2.75$.
Roots 3 and 6 (2.668 and 2.710) are such states, slightly mixed with
others. The contamination comes from the single-excitation space, not from
the functional, and no functional removes it.

## Numerical anchors {#unrestricted-numerical-anchors}

From `site/src/reference/validation.md`, "Spin-conserving TDA and Casida
excitation energies and `<S^2>` on UHF and UKS references".

| Quantity | Value | Test |
|---|---|---|
| lowest eight roots vs PySCF, NH2 and O2 / 6-31G, OH TDA on UHF | $\le 5.0\times10^{-7}$ eV | `crates/ferric-tddft/tests/validation_tddft_open_shell.rs` |
| $\langle S^2\rangle$ per root | $\le 2.5\times10^{-7}$ | the same |
| generator's closed formula vs `spin_square` on the expanded determinant vector (OH, NH2) | $\le 8.4\times10^{-15}$ | `scripts/validation/gen_tddft_open_shell_refs.py` |

The exactness anchor is water through the unrestricted route, which equals
the restricted singlets and triplets elementwise, all roots, with
$\langle S^2\rangle$ 0 or 2, for HF, PBE, B3LYP, CAM-B3LYP and r2SCAN.

## Sources {#unrestricted-sources}

| Source | What it gives |
|---|---|
| section @sec:triplets-spin-blocks | the spin-orbital matrices this page generalizes |
| D. Maurice and M. Head-Gordon, *Int. J. Quantum Chem.* **56** (S29), 361 (1995) | CIS on UHF and ROHF references and $\langle S^2\rangle$ of its states |
| Q. Sun *et al.*, *J. Chem. Phys.* **153**, 024109 (2020) | PySCF, whose `tdscf.uhf.get_ab` blocks are the reference |
