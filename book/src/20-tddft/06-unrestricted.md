# 20.6 Unrestricted references and $\langle S^2\rangle$

<!-- section: 20.6; status: ready; registry: tddft;
     theory: crates/ferric-tddft/theory.md#"10. Unrestricted (UHF / UKS) references" -->

On a UHF or UKS reference the $\alpha$ and $\beta$ orbitals differ, so there
is no singlet/triplet split. ferric solves the spin-conserving problem over
both spins at once and reports $\langle S^2\rangle$ for each root, since the
roots are not spin eigenfunctions.

| Symbol | Meaning | Code name |
|---|---|---|
| $ia\sigma$ | compound index, $\alpha$ block first, then $\beta$ | `x[i * nva + a]`, `x[na * nva + i * nvb + a]` |
| $\bar\jmath, \bar b$ | $\beta$ occupied and virtual orbitals | |
| $O = C_\alpha^{\mathsf T}SC_\beta$ | $\alpha$–$\beta$ MO overlap | `SpinOverlap::o` |
| $P, Q, R$ | $O^{\mathsf T}O$, $O_{:,\mathrm{occ}\beta}O_{:,\mathrm{occ}\beta}^{\mathsf T}$, $O_{\mathrm{occ}\alpha,:}^{\mathsf T}O_{\mathrm{occ}\alpha,:}$ | `p`, `q`, `r` |
| $M$ | $(n_\alpha - n_\beta)/2$ | `m_terms` |

| Entry point | Location |
|---|---|
| Rust | `ferric-tddft/src/tddft.rs::run_tddft` → `ferric-tddft/src/unrestricted.rs::{run_unrestricted,casida_x,SpinOverlap}` |
| CLI | `method.kind = "tda"` or `"tddft"` with multiplicity > 1 |
| Python | `run_tddft(...)` on a UHF or UKS reference; the result carries `s2` per root |

## 20.6.1 The two-spin problem

<!-- include: crates/ferric-tddft/theory.md#"10. Unrestricted (UHF / UKS) references" -->

$$ A_{ia\sigma,jb\tau} = \delta_{\sigma\tau}\delta_{ij}\delta_{ab}(\varepsilon^\sigma_a - \varepsilon^\sigma_i)
+ (i_\sigma a_\sigma|j_\tau b_\tau) + (i_\sigma a_\sigma|f_{\sigma\tau}|j_\tau b_\tau)
- \delta_{\sigma\tau}\,c\,(i_\sigma j_\sigma|a_\sigma b_\sigma), \tag{20.10} $$

and $\mathbf B$ likewise with $(i_\sigma b_\sigma|j_\sigma a_\sigma)$ in the
exchange. Each same-spin block is the restricted builder over that spin's
orbitals with Coulomb coefficient 1; the cross-spin Coulomb block is
$B^{\alpha\mathsf T}_{ov}B^\beta_{ov}$; there is no cross-spin exchange. TDA,
Casida and the Davidson solvers apply unchanged. The transition dipole has no
$\sqrt2$, since each spin is its own block.

The kernel is `UksKernel`, the spin-resolved AO kernel at
$(D^\alpha, D^\beta)$, screened per spin as the UKS $V_{xc}$ is. Every
functional the restricted path accepts is accepted here. A triplet request
on an unrestricted reference is refused, since the roots are not
spin-adapted. An ROHF or ROKS reference is refused, because its orbitals
diagonalize one Roothaan-coupled Fock operator and the orbital-energy
diagonal of equation 20.10 is not its response. A non-aufbau (MOM) reference
is also refused.

For a closed shell with $C_\alpha = C_\beta$ the problem splits on
$X_\alpha \pm X_\beta$, and the unrestricted roots are the restricted
singlets and triplets with $\langle S^2\rangle = 0$ and 2.

> **Verified equation 20.10** · the unrestricted matrices
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
> | Code test | `crates/ferric-tddft/tests/validation_tddft_open_shell.rs` (against PySCF `tdscf.uhf.get_ab`) |
>
> <!-- verified: id=td.unrestricted.matrices; script=scripts/tddft_response_sympy.py; claim=B5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-tddft/src/unrestricted.rs::run_unrestricted; registry=tddft -->

## 20.6.2 $\langle S^2\rangle$ of each root

<!-- include: crates/ferric-tddft/theory.md#"11. ⟨S²⟩ of each root" -->

Each root is given the wavefunction
$|\Psi\rangle = \sum_{ia\sigma}X_{ia\sigma}a^\dagger_{a\sigma}a_{i\sigma}|\Phi\rangle$
with unit-normalized $X$: the TDA eigenvector, or for Casida the $X$ part
$\tfrac12\big((X+Y) + (\mathbf A+\mathbf B)(X+Y)/\Omega\big)$ (`casida_x`),
renormalized. Casida response defines no excited-state wavefunction; this
is the particle–hole part of the root. Then
$\langle S^2\rangle = M^2 + M + \langle S_-S_+\rangle$ with
$S_+ = \sum_{p\in\alpha,q\in\beta}O_{pq}a^\dagger_{p\alpha}a_{q\beta}$, and with
$\pi(X, N) = \mathrm{tr}(XN_{vv}X^{\mathsf T}) - \mathrm{tr}(X^{\mathsf T}N_{oo}X)$,

$$ \langle S_-S_+\rangle = \sum_{\bar\jmath}P_{\bar\jmath\bar\jmath} + \pi(X_\beta,P)
- \big[w_\alpha\mathrm{tr}\,Q_{oo} + \pi(X_\alpha,Q)\big]
- \big[w_\beta\mathrm{tr}\,R_{oo} + \pi(X_\beta,R)\big]
- 2\,\mathrm{tr}\big(X_\alpha^{\mathsf T}O_{oo}X_\beta O_{vv}^{\mathsf T}\big) , \tag{20.11} $$

$w_\sigma = |X_\sigma|^2$. With no excitation it reduces to PySCF's
`UHF.spin_square` of the reference.

<!-- excerpt: ferric-tddft/src/unrestricted.rs::excited -->
```rust
let one_body = (wa + wb) * self.beta_occ_weight() + ph(&xb, &p, nb);
let two_alpha = wa * trace(q.slice(s![..na, ..na])) + ph(&xa, &q, na);
let two_beta = wb * trace(r.slice(s![..nb, ..nb])) + ph(&xb, &r, nb);
```

| Term of equation 20.11 | Code |
|---|---|
| $\sum_{\bar\jmath}P_{\bar\jmath\bar\jmath}$ | `beta_occ_weight` ($w_\alpha + w_\beta = 1$) |
| $\pi(X, N)$ | `ph` |
| the cross-spin term | `cross` |

> **Verified equation 20.11** · $\langle S^2\rangle$ of a single-excitation state
>
> The transcribed `SpinOverlap::reference` and `SpinOverlap::excited` equal
> $M^2 + M + \lVert S_+\Psi\rVert^2$ evaluated by Jordan–Wigner on the
> determinant space, for 4 spatial orbitals per spin,
> $(n_\alpha, n_\beta) = (2,1), (3,1), (2,2)$ and 24 random $X$, with a
> general $O$ (the $\alpha$ and $\beta$ MOs span different AO subspaces, so
> $P \ne 1$) and an orthogonal one; for $O = 1$, $X_\alpha = \pm X_\beta$
> gives 0 and 2; `casida_x` equals the $X$ block of the eigenvector of
> equation 20.2 normalized to $X\cdot X - Y\cdot Y = 1$.
>
> | | |
> |---|---|
> | Proof | `scripts/tddft_response_sympy.py`, claims s2ref, s2, casidax |
> | Negative control | the runner drops the 2 on the cross-spin term (D2 fails), drops the occupied–occupied overlap from the reference value (D1 fails), and takes the $Y$ block in `casida_x` (D4 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-tddft/src/unrestricted.rs::{excited,reference,casida_x}` |
> | Code test | `crates/ferric-tddft/tests/validation_tddft_open_shell.rs` (`compare_s2`, against values the generator checks with `pyscf.fci.spin_op.spin_square`) |
>
> <!-- verified: id=td.s2.per-root; script=scripts/tddft_response_sympy.py; claim=s2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-tddft/src/unrestricted.rs::excited; registry=tddft -->

## 20.6.3 Numerical anchors

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

## 20.6.4 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-tddft/theory.md`, sections 10 and 11 | the two-spin matrices, the kernel, the refusals, the $\langle S^2\rangle$ derivation |
