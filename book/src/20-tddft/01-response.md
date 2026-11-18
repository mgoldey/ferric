# Response matrices, TDA and the Casida equations {#response}

<!-- status: ready; registry: tddft -->

An excited state of a closed-shell molecule is, to first approximation, an
electron moved from an occupied orbital $i$ to a virtual orbital $a$. The
moved electron leaves a hole behind and the two attract; the other
electrons rearrange around both. So the excitation energy is the orbital
gap $\varepsilon_a - \varepsilon_i$ plus couplings between the different
$i \to a$ moves. Linear response collects those couplings in two
$d\times d$ matrices, $d = n_on_v$. $\mathbf A$ couples one excitation to
another. $\mathbf B$ couples an excitation to a de-excitation: the
perturbed ground state is not the bare reference determinant, so the field
can also take an electron from $a$ back to $i$.

The plan:

1. Write $\mathbf A$ and $\mathbf B$ in spatial orbitals (@sec:response-matrices).
2. Derive the Casida eigenproblem from the response equations, reduce it to
   a symmetric one, and drop $\mathbf B$ to get the Tamm–Dancoff
   approximation (@sec:response-tda-casida-reduction).
3. Solve it without forming the matrices (@sec:response-matrix-free-solvers).

The Tamm–Dancoff approximation (TDA) solves with $\mathbf A$ alone; full
response (Casida) keeps $\mathbf B$. On an HF reference the two are CIS and
TDHF.

| Symbol | Meaning | Code name |
|---|---|---|
| $ia$ | compound index, `i * nvir + a` | `ia` |
| $\Delta_i^a$ | $\varepsilon_a - \varepsilon_i$ | the diagonal of `a`, `diag` |
| $c_J$ | Coulomb coefficient, 2 singlet, 0 triplet | `coulomb_factor` |
| $c_{\rm sr}, c_{\rm lr}, \omega$ | exact-exchange fractions and range separation | `ExchangeMix` |
| $K_{ia,jb}$ | XC kernel block | added by `add_fxc_block` |
| $B^P_{pq}$ | Coulomb-metric RI tensors ($ov$, $oo$, $vv$) | `RiTensors { b_ov, .. }` |
| $\Omega_n$, $X_n$, $(X+Y)_n$ | excitation energy and amplitudes | `omega`, `amplitudes` |

| Entry point | Location |
|---|---|
| Rust | `ferric-tddft/src/tddft.rs::{run_tddft,build_a_matrix,build_b_matrix,solve_tda,solve_casida}`, `ferric-tddft/src/davidson.rs::{davidson_tda,davidson_casida}` |
| CLI | `method.kind = "tda"` or `"tddft"`, `[tddft]` section |
| Python | `run_tddft(method="tda" \| "casida")` |

## The matrices {#response-matrices}

Indices $i, j$ run over the doubly occupied and $a, b$ over the virtual
spatial orbitals of a restricted reference. The orbitals are real, so the
chemists'-notation integrals carry no complex conjugates and
$(pq|rs) = (qp|rs)$. For a singlet excitation the matrices are

$$ A_{ia,jb} = \delta_{ij}\delta_{ab}\Delta_i^a + 2(ia|jb) - c_{\rm HF}(ij|ab) + K_{ia,jb},\qquad
B_{ia,jb} = 2(ia|jb) - c_{\rm HF}(ib|ja) + K_{ia,jb}. \label{eq:td-matrices-spin-adaptation} $$

Read the terms of $\mathbf A$ one at a time.

- $\Delta_i^a = \varepsilon_a - \varepsilon_i$ is the cost of the move with
  every other electron frozen. It sits on the diagonal only.
- $2(ia|jb)$ is the Coulomb interaction of the transition densities
  $\phi_i\phi_a$ and $\phi_j\phi_b$. A singlet moves both spins together,
  so this term counts once for each spin.
- $-c_{\rm HF}(ij|ab)$ is exact exchange between the moved electron and its
  hole, weighted by the exact-exchange fraction: 1 for HF, the functional's
  global fraction for a hybrid, 0 for LDA, GGA and meta-GGA.
- $K_{ia,jb} = (ia|f_{\alpha\alpha}+f_{\alpha\beta}|jb)$ is the response of
  the exchange-correlation potential, the singlet kernel block of section
  @sec:fxc-singlet-triplet-block. It is zero for an HF reference.

Where does the 2 come from? In spin orbitals each excitation carries a spin
label, $ia\alpha$ or $ia\beta$. The Coulomb integral couples $ia\alpha$ to
$jb\alpha$ and to $jb\beta$ alike, since the Coulomb operator sees the total
density. Exchange couples only equal spins. A singlet has
$X_{ia\alpha} = X_{ia\beta}$, so its matrix is the sum of the $\alpha\alpha$
and $\alpha\beta$ blocks: the Coulomb term appears twice and the exchange
once. Section @sec:triplets does this in full and gets the triplet from the
difference.

$\mathbf B$ has the same Coulomb and kernel terms; only the exchange
integral changes. The derivation produces $(ia|bj)$ and $(ib|aj)$ in
$\mathbf B$; for real orbitals they equal $(ia|jb)$ and $(ib|ja)$. The same
block $K$ goes into both matrices.

Every integral is fitted in the Coulomb metric,
$(pq|rs) \approx \sum_P B^P_{pq}B^P_{rs}$ with
$B^P_{pq} = \sum_Q M_{PQ}(Q|pq)$, where $M^{\mathsf T}M = V^{-1}$ and
$V_{PQ} = (P|Q)$ over the auxiliary basis.
`ferric-mp2/src/rimp2.rs::metric_inverse_sqrt` returns the inverse Cholesky
factor for the Coulomb metric, and
`ferric-mp2/src/rimp2.rs::stream_dressed_mo_band` forms the MO blocks. The driver builds three of
them, $B_{ov}$, $B_{oo}$ and $B_{vv}$. The Coulomb part of both matrices is
then $2\,B_{ov}^{\mathsf T}B_{ov}$. The exchange in $\mathbf A$ is read from
$B_{oo}^{\mathsf T}B_{vv}$, and the exchange in $\mathbf B$ is the
$(ib, ja)$ element of the same $B_{ov}^{\mathsf T}B_{ov}$ that gives the
Coulomb term.

`ferric-tddft/src/tddft.rs::build_a_matrix` writes $\mathbf A$ in that
order. This is its body without the signature; `c_j` is $c_J$, 2 here:

<!-- excerpt: ferric-tddft/src/tddft.rs::build_a_matrix -->
```rust
// Diagonal: orbital energy differences
for i in 0..nocc {
    for a_idx in 0..nvir {
        let ia = i * nvir + a_idx;
        a[(ia, ia)] = eps[nocc + a_idx] - eps[i];
    }
}

// Coulomb: c_J (ia|jb) = c_J B_ovᵀ B_ov.
if c_j != 0.0 {
    let coulomb = ri.b_ov.t().dot(&ri.b_ov);
    a += &(c_j * &coulomb);
}
if mix.c_sr != 0.0 {
    subtract_exchange_a(&mut a, ri.tensors(), nocc, nvir, mix.c_sr);
}
if let (Some(lr), Some(dc)) = (ri.lr.as_deref(), mix.long_range()) {
    subtract_exchange_a(&mut a, lr.tensors(), nocc, nvir, dc);
}
```

| Lines | Term of equation @eq:td-matrices-spin-adaptation |
|---|---|
| the double loop | $\delta_{ij}\delta_{ab}\Delta_i^a$; `eps` holds occupied then virtual energies |
| `c_j * B_ovᵀ B_ov` | $2(ia|jb)$ |
| `subtract_exchange_a(.., mix.c_sr)` | $-c_{\rm HF}(ij|ab)$; for a global hybrid $c_{\rm sr} = c_{\rm HF}$ |
| the `lr` branch | the long-range exchange of a range-separated hybrid (section @sec:rsh); absent otherwise |

The kernel is not here. `ferric-tddft/src/tddft.rs::add_fxc_block` adds the
one block $K$ to $\mathbf A$ and, for Casida, to $\mathbf B$ after both are
built. In `ferric-tddft/src/tddft.rs::build_b_matrix` the exchange
subtraction reads the transposed index pair of the Coulomb product:

<!-- excerpt: ferric-tddft/src/tddft.rs::subtract_exchange_b -->
```rust
let ib = i * nvir + b_idx;
let ja = j * nvir + a_idx;
b[(ia, jb)] -= coef * k_ov[(ib, ja)];
```

Here `k_ov` is $B_{ov}^{\mathsf T}B_{ov}$, so `k_ov[(ib, ja)]` is
$(ib|ja)$. No second integral set is needed for $\mathbf B$.

> **Verified equation @eq:td-matrices-spin-adaptation** · the singlet matrices are the spin adaptation of the spin-orbital ones
>
> Symbolically, $U = \tfrac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ on
> the spin index block-diagonalizes the closed-shell spin-orbital $\mathbf A$
> and $\mathbf B$ into singlet ($c_J = 2$, kernel $f_{\alpha\alpha}+f_{\alpha\beta}$)
> and triplet ($c_J = 0$, kernel $f_{\alpha\alpha}-f_{\alpha\beta}$) blocks.
> Numerically, on a random closed-shell model with RI tensors for Coulomb and
> erf and a grid kernel, the spin-orbital $\mathbf A$ and $\mathbf B$ built
> from the textbook definition equal, after $U$, the code's singlet and
> triplet blocks for HF, a global hybrid and an RSH; and for an HF model,
> $\langle\Phi_i^a|H - E_0|\Phi_j^b\rangle$ evaluated over Jordan–Wigner
> determinants equals the spin-orbital $\mathbf A$ elementwise.
>
> | | |
> |---|---|
> | Proof | `scripts/tddft_response_sympy.py`, claims A1, B1, B2 |
> | Negative control | the runner sets the singlet `coulomb_factor` to 1 and reads $(ia|jb)$ for the $\mathbf B$ exchange (B1 fails both times) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-tddft/src/tddft.rs::{build_a_matrix,build_b_matrix,coulomb_factor}` |
> | Code test | `crates/ferric-tddft/tests/validation_tddft.rs` (against PySCF `TDA` and `TDDFT`) |
>
> <!-- verified: id=td.matrices.spin-adaptation; script=scripts/tddft_response_sympy.py; claim=B1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-tddft/src/tddft.rs::build_a_matrix; rust_test=ferric-tddft/tests/validation_tddft.rs::tda_matches_the_validated_library_tda_path; registry=tddft -->

## TDA and the Casida reduction {#response-tda-casida-reduction}

### From the response to an eigenproblem

Perturb the reference with a real one-electron operator oscillating at
frequency $\omega$, with occupied–virtual elements $v_{ia}$. To first order
the density changes by amplitudes $X_{ia}$ on the $i\to a$ products and
$Y_{ia}$ on the $a\to i$ products, and linear response gives them as the
solution of

$$ \left[\begin{pmatrix}\mathbf A & \mathbf B\\ \mathbf B & \mathbf A\end{pmatrix}
- \omega\begin{pmatrix}1 & 0\\ 0 & -1\end{pmatrix}\right]\begin{pmatrix}X\\ Y\end{pmatrix}
= -\begin{pmatrix}v\\ v\end{pmatrix} . $$

For most $\omega$ this has a finite solution. It fails to have one where the
matrix on the left is singular, and there the response diverges. Setting
$v = 0$ and asking for a nonzero solution gives those frequencies:

$$ \begin{pmatrix}\mathbf A & \mathbf B\\ \mathbf B & \mathbf A\end{pmatrix}\begin{pmatrix}X\\ Y\end{pmatrix}
= \Omega\begin{pmatrix}1 & 0\\ 0 & -1\end{pmatrix}\begin{pmatrix}X\\ Y\end{pmatrix} \label{eq:response-2} $$

Equation @eq:response-2 is the Casida equation. Its positive roots $\Omega_n$ are the
excitation energies.

The matrix on the right is not the identity, so equation @eq:response-2 is not a
symmetric eigenproblem. Add and subtract its two block rows:

$$ (\mathbf A+\mathbf B)(X+Y) = \Omega(X-Y),\qquad (\mathbf A-\mathbf B)(X-Y) = \Omega(X+Y). $$

Take $X-Y = (\mathbf A+\mathbf B)(X+Y)/\Omega$ from the first, substitute it
into the second and multiply by $\Omega$:

$$ (\mathbf A-\mathbf B)(\mathbf A+\mathbf B)(X+Y) = \Omega^2(X+Y). $$

The product of two symmetric matrices is not symmetric either. When
$\mathbf A-\mathbf B$ is positive definite it has a symmetric square root
$T = (\mathbf A-\mathbf B)^{1/2}$. Write $X+Y = c\,TZ$, multiply on the
left by $T^{-1}$, and the problem becomes symmetric:

$$ T(\mathbf A+\mathbf B)T\,Z = \Omega^2 Z,\qquad T = (\mathbf A-\mathbf B)^{1/2},\qquad
(X+Y)_n = \Omega_n^{-1/2}\,TZ_n . \label{eq:td-casida-reduction} $$

The factor $c = \Omega_n^{-1/2}$ comes from the normalization. Excited states
in response theory are normalized to $|X|^2 - |Y|^2 = 1$. Expand it with the
first row-sum equation and $M = T(\mathbf A+\mathbf B)T$:

$$ |X|^2-|Y|^2 = (X+Y)^{\mathsf T}(X-Y) = \frac{(X+Y)^{\mathsf T}(\mathbf A+\mathbf B)(X+Y)}{\Omega}
= \frac{c^2\,Z^{\mathsf T}MZ}{\Omega} = c^2\,\Omega $$

for unit $Z$. So $c^2 = 1/\Omega$.

Check the scalar case, one excitation with $\mathbf A = a$ and
$\mathbf B = b$. The roots of equation @eq:response-2 are
$\pm\sqrt{(a-b)(a+b)} = \pm\sqrt{a^2 - b^2}$. With $b = 0$ the root is $a$,
the TDA value. For small $b$ it is $a - b^2/(2a)$ to second order, below
the TDA root: coupling to the de-excitation lowers the excitation energy.

`ferric-tddft/src/tddft.rs::solve_casida` builds $T$ from the
eigendecomposition of $\mathbf A-\mathbf B$, then forms the symmetric
matrix and diagonalizes it:

<!-- excerpt: ferric-tddft/src/tddft.rs::solve_casida -->
```rust
// M = (A−B)^{1/2} (A+B) (A−B)^{1/2}
let m = amb_sqrt.dot(&a_plus_b).dot(&amb_sqrt);
```

The columns it returns are $(X+Y)_n$: `amb_sqrt.dot(&z)` scaled column by
column by $1/\sqrt{\Omega_n}$.

> **Verified equation @eq:td-casida-reduction** · the Casida reduction and its normalization
>
> Symbolically, for the scalar case the eigenvalues of
> $\begin{pmatrix}a&b\\-b&-a\end{pmatrix}$ are $\pm\sqrt{(a-b)(a+b)}$ and
> $X+Y = \Omega^{-1/2}\sqrt{a-b}\,z$ gives $X^2 - Y^2 = 1$. Numerically, the
> positive eigenvalues of the full non-Hermitian matrix equal the code's
> $\Omega$, its eigenvector normalized to $X\cdot X - Y\cdot Y = 1$ gives the
> code's $X+Y$ up to sign, and the TDA roots are the $X$-block roots with
> $\mathbf B = 0$.
>
> | | |
> |---|---|
> | Proof | `scripts/tddft_response_sympy.py`, claims A3, B3, B4 |
> | Negative control | the runner scales $X+Y$ by $1/\Omega$ instead of $\Omega^{-1/2}$ in `solve_casida` (B4 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-tddft/src/tddft.rs::{solve_tda,solve_casida}` |
> | Code test | `casida_matches_dense_solve` in `crates/ferric-tddft/src/davidson.rs`; `crates/ferric-tddft/tests/validation_tddft.rs` |
>
> <!-- verified: id=td.casida.reduction; script=scripts/tddft_response_sympy.py; claim=B4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-tddft/src/tddft.rs::solve_casida; rust_test=ferric-tddft/src/davidson.rs::casida_matches_dense_solve; registry=tddft -->

### What the roots mean

The same reduction solves the driven problem. Add and subtract its rows
and eliminate $X-Y$:

$$ \big[(\mathbf A+\mathbf B) - \omega^2(\mathbf A-\mathbf B)^{-1}\big]{(X+Y)} = -2v . $$

Put $X+Y = Tz$ and multiply on the left by $T$; this gives
$(M - \omega^2)z = -2Tv$. Expand $z$ in the eigenvectors $Z_n$ of $M$, and
use $TZ_n = \Omega_n^{1/2}(X+Y)_n$ from equation @eq:td-casida-reduction:

$$ v\cdot(X+Y) = \sum_n \frac{2\,\Omega_n\,\big(v\cdot(X+Y)_n\big)^2}{\omega^2 - \Omega_n^2} . $$

The left side is proportional to the first-order change of the expectation
value of the perturbation, and the right side writes it as a sum over
states. Each root contributes a pole at $\omega = \pm\Omega_n$, with a
weight set by how strongly the perturbation couples to $(X+Y)_n$. When $v$
is a dipole operator, the weight is set by the transition dipole of
section @sec:oscillator-strengths. At $\omega = 0$ the sum is
$-2\,v^{\mathsf T}(\mathbf A+\mathbf B)^{-1}v$, the static response, and
$\mathbf A + \mathbf B$ is the orbital Hessian of section @sec:second-order-orbital-gradient-hessian at canonical
orbitals.

The positive-definiteness conditions have a meaning too. $T$ is symmetric
and invertible, so $M = T(\mathbf A+\mathbf B)T$ is congruent to
$\mathbf A+\mathbf B$, and by Sylvester's law of inertia the two have the
same number of negative eigenvalues. A negative $\Omega^2$ therefore means
$\mathbf A+\mathbf B$ has a negative eigenvalue: the reference is a saddle
point of the energy, the instability that section @sec:stability detects. ferric
treats both cases as errors. A negative eigenvalue of
$\mathbf A-\mathbf B$, or an $\Omega^2$ below $-10^{-10}$ Ha$^2$, is an
instability of the reference, and the run stops; the root is not clamped
to zero. Both solvers return the lowest `n_roots` roots above
$10^{-10}$ Ha, in ascending order
(`ferric-tddft/src/tddft.rs::lowest_positive_roots`).

### The Tamm–Dancoff approximation

Set $\mathbf B = 0$. Equation @eq:response-2 splits into $\mathbf AX = \Omega X$ and
$\mathbf AY = -\Omega Y$, and the excitation block is the Hermitian
eigenproblem

$$ \mathbf A\,X_n = \Omega_n X_n,\qquad X_n^{\mathsf T}X_n = 1 . $$

`ferric-tddft/src/tddft.rs::solve_tda` diagonalizes the dense $\mathbf A$.
For an HF reference ($c_{\rm HF} = 1$, $K = 0$), $\mathbf A$ is the CIS
Hamiltonian minus the reference energy, and TDA is CIS.

TDA drops the de-excitation coupling. It is cheaper, one symmetric
diagonalization in place of two, and it never needs $\mathbf A-\mathbf B$ to
be positive definite, so it runs on references where Casida fails. The
cost is the $\mathbf B$ shift of each root, $-b^2/(2a)$ in the scalar case.

TDA and Casida side by side for water / 6-31G, from the `589b50832` build,
at HF (CIS and TDHF), PBE and B3LYP. Every Casida root lies below its TDA
partner, by 0.003 to 0.09 eV. At B3LYP the second and third roots change
order between the two: the bright one ($f \approx 0.1$) is the third TDA
root and the second Casida root.

```python
import ferric

HA_TO_EV = 27.211386245988
mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("6-31g")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

for functional in (None, "PBE", "B3LYP"):
    tda = ferric.run_tddft(mol, bs, aux, functional=functional, n_roots=5, method="tda")
    cas = ferric.run_tddft(mol, bs, aux, functional=functional, n_roots=5, method="casida")
    label = functional or "HF (CIS / TDHF)"
    print(label)
    for n in range(5):
        e_t = tda.excitation_energies[n] * HA_TO_EV
        e_c = cas.excitation_energies[n] * HA_TO_EV
        print(f"  root {n + 1}: TDA {e_t:9.5f} eV  Casida {e_c:9.5f} eV  "
              f"Casida - TDA {e_c - e_t:+.5f} eV  "
              f"f(TDA) {tda.oscillator_strengths[n]:.5f}  f(Casida) {cas.oscillator_strengths[n]:.5f}")
```

```text
HF (CIS / TDHF)
  root 1: TDA   9.40529 eV  Casida   9.34854 eV  Casida - TDA -0.05675 eV  f(TDA) 0.01503  f(Casida) 0.01456
  root 2: TDA  11.34728 eV  Casida  11.27372 eV  Casida - TDA -0.07356 eV  f(TDA) 0.00000  f(Casida) 0.00000
  root 3: TDA  11.87564 eV  Casida  11.79286 eV  Casida - TDA -0.08277 eV  f(TDA) 0.12122  f(Casida) 0.11301
  root 4: TDA  13.95825 eV  Casida  13.86799 eV  Casida - TDA -0.09026 eV  f(TDA) 0.10455  f(Casida) 0.09618
  root 5: TDA  15.50651 eV  Casida  15.45368 eV  Casida - TDA -0.05283 eV  f(TDA) 0.47123  f(Casida) 0.44163
PBE
  root 1: TDA   7.53950 eV  Casida   7.51906 eV  Casida - TDA -0.02044 eV  f(TDA) 0.01227  f(Casida) 0.01164
  root 2: TDA   9.68326 eV  Casida   9.59501 eV  Casida - TDA -0.08825 eV  f(TDA) 0.10769  f(Casida) 0.09779
  root 3: TDA   9.73256 eV  Casida   9.72987 eV  Casida - TDA -0.00270 eV  f(TDA) 0.00000  f(Casida) 0.00000
  root 4: TDA  12.22189 eV  Casida  12.15415 eV  Casida - TDA -0.06774 eV  f(TDA) 0.09875  f(Casida) 0.09006
  root 5: TDA  14.72545 eV  Casida  14.64695 eV  Casida - TDA -0.07851 eV  f(TDA) 0.45631  f(Casida) 0.39791
B3LYP
  root 1: TDA   7.82610 eV  Casida   7.80519 eV  Casida - TDA -0.02090 eV  f(TDA) 0.01221  f(Casida) 0.01157
  root 2: TDA   9.95555 eV  Casida   9.93123 eV  Casida - TDA -0.02432 eV  f(TDA) 0.00000  f(Casida) 0.09675
  root 3: TDA  10.01225 eV  Casida   9.95087 eV  Casida - TDA -0.06139 eV  f(TDA) 0.10640  f(Casida) 0.00000
  root 4: TDA  12.45036 eV  Casida  12.39400 eV  Casida - TDA -0.05636 eV  f(TDA) 0.09492  f(Casida) 0.08685
  root 5: TDA  14.80836 eV  Casida  14.74155 eV  Casida - TDA -0.06682 eV  f(TDA) 0.46594  f(Casida) 0.41112
```

## The matrix-free solvers {#response-matrix-free-solvers}

The dense path holds $\mathbf A$, $\mathbf B$, $\mathbf A\pm\mathbf B$, $T$
and $M$, all $d\times d$ with $d = n_on_v$. Memory grows as $N^4$ and each
`eigh` as $N^6$; Casida performs two. Before it allocates, the memory plan
counts the co-resident $d^2$ matrices and the RI tensors, and a run over
budget is an error.

The Davidson solvers in `ferric-tddft/src/davidson.rs` hold only a
subspace. They need products by $\mathbf A$ (TDA) or by
$\mathbf A\pm\mathbf B$ (Casida), which `matvec.rs` supplies for a closed
shell and the unrestricted operator for an open shell (section @sec:unrestricted). The
preconditioner is the diagonal $D_{ia} = \Delta_i^a$.

`TddftSolver::Auto` keeps the dense path when $d \le 200$
(`ferric-tddft/src/tddft.rs::AUTO_DENSE_MAX_DIM`) or $4n_{\rm roots} \ge d$, and runs Davidson
otherwise. The settings `dense` and `davidson` force one.

### TDA

Rayleigh–Ritz on an orthonormal subspace $V$ diagonalizes
$H = V^{\mathsf T}\mathbf AV$ and gives Ritz pairs $(\theta, x = Vy)$ with
residual $r = \mathbf Ax - \theta x$. The correction added to the subspace
is Olsen's,

$$ t = Pr - \epsilon Px,\qquad P = (\theta - D)^{-1},\qquad \epsilon = \frac{x^{\mathsf T}Pr}{x^{\mathsf T}Px}, \label{eq:response-4} $$

with $\epsilon$ chosen so that
$x^{\mathsf T}t = x^{\mathsf T}Pr - \epsilon\,x^{\mathsf T}Px = 0$.

Why not the plain $Pr$? Take a root whose eigenvector lies on a block of
equal orbital differences, all equal to $\theta$, such as a coupling-free
combination of a degenerate set of $\pi\to\pi^*$ excitations. There
$Pr \approx -x$, and almost every new direction is one the subspace already
has; the solver stalls. Removing the $x$ part first keeps the step in the
complement. `ferric-tddft/src/davidson.rs::diagonal_correction` is the
equation:

<!-- excerpt: ferric-tddft/src/davidson.rs::diagonal_correction -->
```rust
let p = inverse_shifted_diag(theta, diag);
let pr = &p * r;
let px = &p * x;
let den = x.dot(&px);
if den.is_finite() && den != 0.0 {
    let eps = x.dot(&pr) / den;
```

`inverse_shifted_diag` floors $|\theta - D|$ at `PRECOND_FLOOR`, and
$\epsilon$ is skipped when $x^{\mathsf T}Px$ is not finite or vanishes.

### Casida

One subspace $V$ carries both $X+Y$ and $X-Y$. Project the two products,
$M_\pm = V^{\mathsf T}(\mathbf A\pm\mathbf B)V$, and write the subspace
coefficients of $X+Y$ and $X-Y$ as $u$ and $v$. The Galerkin conditions are
the two row-sum equations of section @sec:response-tda-casida-reduction in the subspace,
$M_+u = \Omega v$ and $M_-v = \Omega u$, so $M_-M_+u = \Omega^2u$. Put
$L = M_-^{1/2}$ and $u = Lz$ to get the subspace image of equation @eq:td-casida-reduction,

$$ LM_+L\,z = \Omega^2z,\qquad u = \Omega^{-1/2}Lz,\qquad v = M_+u/\Omega . \label{eq:response-5} $$

For unit $z$,
$u^{\mathsf T}v = z^{\mathsf T}LM_+Lz/\Omega^2 = 1$, which is the
normalization $(X+Y)\cdot(X-Y) = 1$ of section @sec:response-tda-casida-reduction. In
`ferric-tddft/src/davidson.rs::casida_ritz`:

<!-- excerpt: ferric-tddft/src/davidson.rs::casida_ritz -->
```rust
let om = om2[j].sqrt();
let uk = l_half.dot(&z.column(j)) / om.sqrt();
let vk = mp.dot(&uk) / om;
```

`l_half` is $L$ and `mp` is $M_+$. A non-positive eigenvalue of $M_-$ or a
negative $\Omega^2$ is an RPA instability and an error, as on the dense
path.

The residuals are the two row-sum equations,

$$ r_+ = (\mathbf A+\mathbf B)(X+Y) - \Omega(X-Y),\qquad
r_- = (\mathbf A-\mathbf B)(X-Y) - \Omega(X+Y), $$

and the reported residual is
$\sqrt{(\lVert r_+\rVert^2 + \lVert r_-\rVert^2)/2}$. That is the residual of
equation @eq:response-2 itself: its $X$ and $Y$ rows are $r_X = (r_+ + r_-)/2$ and
$r_Y = (r_+ - r_-)/2$, and
$\lVert r_X\rVert^2 + \lVert r_Y\rVert^2 = (\lVert r_+\rVert^2 + \lVert r_-\rVert^2)/2$.
The corrections solve the per-element model
$\begin{pmatrix}D & -\Omega\\ -\Omega & D\end{pmatrix}\delta = -r$ with the
Olsen projection against $q = (X-Y;\,X+Y)$
(`ferric-tddft/src/davidson.rs::casida_correction`), and both
$\delta_+$ and $\delta_-$ join the subspace.

### Guesses and stopping

The guesses are unit vectors on the lowest
$\max(2n_{\rm roots}, n_{\rm roots}+8)$ diagonal elements, extended over any
element tied with the last within $10^{-8}$ Ha, so a degenerate pair is
never split. Each guess gets a small deterministic dense part so that the
subspace reaches every point-group irrep
(`ferric-tddft/src/davidson.rs::{guess_indices,guess_vectors}`). The targets are the lowest $n_{\rm roots}$ Ritz values
above $10^{-10}$ Ha, the dense path's cut.

| Setting | Default |
|---|---|
| guesses | lowest $\max(2n_{\rm roots}, n_{\rm roots}+8)$ diagonal elements, plus any tied with the last, each with a small dense part |
| `conv_tol` | $10^{-6}$ on the residual norm |
| `max_iter` | 100; an unconverged root is an error |
| subspace before collapse | $\max(n_{\rm guess} + 10n_{\rm roots}, 50)$ |

These solvers have no proof script. The unit tests `tda_matches_dense_eigh`,
`casida_matches_dense_solve`, `root_in_a_block_no_guess_touches_is_found`,
`unconverged_root_is_an_error` and
`non_positive_a_minus_b_is_an_rpa_instability` in
`crates/ferric-tddft/src/davidson.rs` compare them with the dense solve,
and the range-separated, triplet and open-shell validation tests run both
solvers against PySCF.

## Numerical anchors {#response-numerical-anchors}

From `site/src/reference/validation.md`, "TDA and Casida TDDFT excitation energies".

| Quantity | Value | Test |
|---|---|---|
| lowest five roots vs PySCF, HF | $\le 2\times10^{-6}$ eV | `crates/ferric-tddft/tests/validation_tddft.rs` |
| LDA, PBE | $\le 1.2\times10^{-5}$ eV | the same |
| B3LYP, default pruned grid | $\le 8.1\times10^{-4}$ eV | the same |
| B3LYP, the reference's flat grid | $\le 2.8\times10^{-6}$ eV | the same |
| r2SCAN, water/6-31G, flat 75 × 110 grid | $2.1\times10^{-6}$ eV | the same |

The B3LYP rows differ by grid alone; section @sec:kernel-grid-sensitivity explains why.

Two more anchors sit inside the library. TDA equals
`ferric-gw/src/tddft.rs::run_tda_dft`, which shares the kernel block, and with
$K = 0$ and $c_{\rm HF} = 1$ it reduces to CIS. The prototype
`scripts/proto_tddft.py` builds the TDA and Casida matrices by hand from
PySCF integrals and compares them with PySCF `tdscf` on water/cc-pVDZ.

### What is proven and what is not {#response-proven}

The chapter's tests and their reference data:

| Check | Where | Reference |
|---|---|---|
| TDA and Casida roots against PySCF `TDA`/`TDDFT` | `crates/ferric-tddft/tests/validation_tddft.rs` | generator `scripts/validation/gen_tddft_refs.py`; grades in `site/src/reference/validation.md`, rows `tda` and `tddft` |
| range-separated hybrids and triplets against PySCF `get_ab` | `crates/ferric-tddft/tests/validation_tddft_rsh.rs` | generator `scripts/validation/gen_tddft_rsh_refs.py`; RHF blocks for singlets, the UKS spin blocks for triplets; the response integrals, the SCF fit and the grid are matched to PySCF |
| the exact limits of section @sec:rsh and the singlet + triplet identity of section @sec:triplets | unit tests in `crates/ferric-tddft/src` | exact identities |
| refusals | `crates/ferric-tddft/tests/refusals.rs` | |
| memory plan | `crates/ferric-tddft/tests/memory_budget.rs` | |

`scripts/tddft_response_sympy.py` proves, symbolically: the singlet and
triplet spin adaptation of $\mathbf A$ and $\mathbf B$, the kernel factor 2,
the scalar Casida reduction and its normalization, and the range-separated
exchange split. Numerically, on random models with RI tensors and a grid
kernel, it checks the spin-orbital $\mathbf A$ and $\mathbf B$ against the
code's singlet and triplet blocks; CIS by brute force over determinants
(Jordan–Wigner) against $\mathbf A$, closed and open shell; the Casida roots
and $X+Y$ against the full non-Hermitian problem; the oscillator-strength sum
rule; the unrestricted blocks and their closed-shell limit; the
$\langle S^2\rangle$ formula against a Fock-space $\lVert S_+\Psi\rVert^2$;
and the Casida $X$ extraction against the full eigenvector. With
`DERIV_CROSSCHECK=1` and PySCF importable, it also compares
H$_2$O/STO-3G RHF (TDA and TDHF, singlet and triplet) and
H$_2$O$^+$/STO-3G UHF (TDA and TDHF) energies and oscillator strengths with
PySCF; without that variable this part is skipped with a message.

## Sources {#response-sources}

| Source | What it gives |
|---|---|
| section @sec:fxc-linearized-potential | the kernel block $K$ (section @sec:fxc) |
