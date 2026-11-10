# ferric-tddft Mathematical Foundations

Closed-shell singlet linear response: TDA (CIS for an HF reference) and full
Casida TDDFT (TDHF for an HF reference), on RI integrals. The driver is
`run_tddft` (`src/tddft.rs`); the XC kernel block comes from
`ferric_dft::lr_kernel` (equations in `crates/ferric-dft/theory.md`).

---

## 1. Response matrices

Indices $i,j$ run over doubly occupied and $a,b$ over virtual spatial
orbitals of a restricted reference; the compound index $ia$ has dimension
$d = n_{\rm occ} n_{\rm vir}$, stored as `i * nvir + a`. With chemist-notation
integrals over real orbitals:

$$A_{ia,jb} = \delta_{ij}\delta_{ab}(\epsilon_a-\epsilon_i) + 2(ia|jb) - c_{\rm HF}(ij|ab) + K_{ia,jb}$$

$$B_{ia,jb} = 2(ia|bj) - c_{\rm HF}(ib|aj) + K_{ia,jb}$$

- The factor 2 on the Coulomb term is the singlet spin adaptation.
- $c_{\rm HF}$ is the exact-exchange fraction: 1 for HF, the functional's
  global fraction for a hybrid, 0 for LDA/GGA/meta-GGA.
- $K_{ia,jb} = (ia|f_{\alpha\alpha}+f_{\alpha\beta}|jb)$ is the singlet XC-kernel
  block built by `singlet_fxc_ov_block`. With real orbitals
  $(ia|f_{xc}|jb) = (ia|f_{xc}|bj)$, so the same block enters $A$ and $B$.
  It is zero for an HF reference. Its normalization equals PySCF
  `tdscf.rhf.get_ab`'s `iajb`; that factor is pinned by comparison with
  PySCF, not derived here (§5).
- $(ia|bj) = (ia|jb)$ for real orbitals, so $A$ and $B$ share the Coulomb
  block $2\,B_{ov}^{\mathsf T}B_{ov}$. The exchange terms differ:
  $(ij|ab)$ in $A$ and $(ib|aj)$ in $B$, where
  $(ib|aj) = (ib|ja)$ is the $(ib, ja)$ element of the same Coulomb product.

**RI.** Every integral is factorized in the Coulomb metric,
$(pq|rs) \approx \sum_P B^P_{pq}B^P_{rs}$ with
$B^P_{pq} = \sum_Q [(\mathbf{J})^{-1/2}]_{PQ}(Q|pq)$ and $J_{PQ} = (P|Q)$ over
the auxiliary basis (`metric_inverse_sqrt`, `stream_dressed_mo_band`). The
driver builds the $ov$, $oo$ and $vv$ blocks $B_{ov}$, $B_{oo}$, $B_{vv}$.

---

## 2. Tamm-Dancoff approximation

Setting $B = 0$ leaves the Hermitian eigenproblem

$$A\,X_n = \omega_n X_n,\qquad X_n^{\mathsf T}X_n = 1.$$

For an HF reference ($c_{\rm HF}=1$, $K=0$) this is CIS. `solve_tda` diagonalizes
the dense $A$.

---

## 3. Casida equations

Full linear response solves

$$\begin{pmatrix}A & B\\ B & A\end{pmatrix}\begin{pmatrix}X\\ Y\end{pmatrix}
= \Omega\begin{pmatrix}1 & 0\\ 0 & -1\end{pmatrix}\begin{pmatrix}X\\ Y\end{pmatrix}.$$

Adding and subtracting the two block rows gives
$(A+B)(X+Y) = \Omega(X-Y)$ and $(A-B)(X-Y) = \Omega(X+Y)$, so
$(A-B)(A+B)(X+Y) = \Omega^2(X+Y)$. When $A-B$ is positive definite, the
symmetric form with $T = (A-B)^{1/2}$ is

$$M Z = \Omega^2 Z,\qquad M = T(A+B)T,$$

and $X+Y = c\,T Z$ solves the non-symmetric equation. The normalization
$|X|^2-|Y|^2 = (X+Y)^{\mathsf T}(X-Y) = (X+Y)^{\mathsf T}(A+B)(X+Y)/\Omega
= c^2 Z^{\mathsf T}MZ/\Omega = c^2\Omega$ for unit $Z$, so

$$(X+Y)_n = \Omega_n^{-1/2}\,T Z_n.$$

`solve_casida` builds $T$ from the eigendecomposition of $A-B$. A negative
eigenvalue of $A-B$ (an instability of the reference) is returned as an
error, not clamped. For an HF reference this is TDHF.

**Roots.** Both paths return the lowest `n_roots` eigenvalues above
$10^{-10}$ Ha, in ascending order (`lowest_positive_roots`).

---

## 4. Transition dipoles and oscillator strengths

In the length gauge, with the singlet spin-adaptation factor $\sqrt2$:

$$\langle 0|\mathbf r|n\rangle = \sqrt2\sum_{ia}(X+Y)_{ia,n}\langle i|\mathbf r|a\rangle,\qquad
f_n = \tfrac23\,\Omega_n\,|\langle 0|\mathbf r|n\rangle|^2,$$

with $Y = 0$ and $|X| = 1$ in TDA. This is PySCF's convention
(`oscillator_strengths`, dipole origin at the coordinate origin; the result
is origin-independent because $\langle i|a\rangle = 0$).

---

## 5. Functional gate and kernel

`ferric_dft::lr_kernel::resolve_singlet_response_xc` maps the functional to
$c_{\rm HF}$ and a kernel, and refuses functionals whose response would be
incomplete:

| Functional class | Kernel | Status |
|---|---|---|
| HF (no functional) | none | CIS / TDHF; a `c_hf` override other than 1 is refused |
| LDA, GGA, global hybrid | `GgaFxcKernel` | accepted |
| τ-only meta-GGA (SCAN, r2SCAN, TPSS) | `MggaFxcKernel` | accepted |
| VV10-carrying functionals | none | refused (nonlocal response not in the kernel) |
| range-separated hybrids | none | refused (long-range exchange needs attenuated integrals) |

The kernel is evaluated on `TddftConfig.grid`, the same grid as the reference
SCF. For meta-GGAs the default is 150 × 434 with NWChem radial pruning
(`ferric_dft::grid::response_default_grid`), because the τ kernel is more
grid-sensitive than LDA/GGA.

Open-shell references are refused: there is no spin-resolved $A$/$B$
assembly, and triplet ($f_{\alpha\alpha}-f_{\alpha\beta}$) blocks are not built.

---

## 6. Cost

$A$, $B$, $A\pm B$, $T$ and $M$ are dense $d\times d$ with $d = n_{\rm occ}n_{\rm vir}$,
so memory grows as $N^4$ and each `eigh` as $N^6$; Casida performs two. There
is no iterative (Davidson) solver. The memory plan counts the co-resident
$d^2$ matrices and the RI tensors before allocating, and a run over budget is
an error.

---

## 7. Validation

- Cross-code: `tests/validation_tddft.rs` against PySCF `TDA`/`TDDFT`
  (reference generator `scripts/validation/gen_tddft_refs.py`); grades and
  measured errors are in `site/src/reference/validation.md`, rows `tda`
  and `tddft`.
- Library anchor: TDA equals `ferric_gw::tddft::run_tda_dft`, which shares
  the kernel block; with $K = 0$ and $c_{\rm HF} = 1$ it reduces to CIS.
- Prototype: `scripts/proto_tddft.py` (hand-built TDA/Casida matrices from
  PySCF integrals against PySCF `tdscf`, water/cc-pVDZ).
- Refusals: `tests/refusals.rs`; memory plan: `tests/memory_budget.rs`.
