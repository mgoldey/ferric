# Static and dynamic polarizabilities from the dielectric modes {#polarizabilities-from-dielectric-modes}

<!-- status: ready; registry: rpa-polarizability-c6 -->

Put a molecule in a weak uniform field and its electron cloud shifts against
the field. The induced dipole is proportional to the field, and the
proportionality constant is the polarizability tensor $\alpha$: three by
three, because a field along $x$ can push charge along $y$ in a molecule
without that symmetry. A field that oscillates at an imaginary frequency
$iu$ gives $\alpha(iu)$. No laboratory field does that, but the diagonal
elements of $\alpha(iu)$ are positive and fall smoothly with $u$
(@sec:polarizabilities-from-dielectric-modes-dielectric-form shows why), and the Casimir–Polder integral of chapter @sec:dispersion turns
$\alpha(iu)$ into the $C_6$ coefficient of the dispersion energy. The static
value is the $u = 0$ point of the same function.

In direct RPA the electrons respond to the field and to the Coulomb potential
of each other's induced density, with no exchange kernel. That is the same
screening that chapter @sec:rpa builds into the dielectric matrix for the
correlation energy, so the polarizability comes out of that matrix. Each
frequency costs one linear solve against $\tilde{\boldsymbol\varepsilon}(iu)$
per field direction, in the auxiliary basis. Nothing is solved in the
$n_on_v$ space.

Section @sec:rpax-static-polarizability also computes a static polarizability, from a statically
screened exchange-type kernel, and that route is graded Smoke. This section
is the direct-RPA response. Its static tensor is the $u = 0$ value of the same
$\alpha(iu)$ that section @sec:casimir-polder-c6 integrates into $C_6$.

| Symbol | Meaning | Code name |
|---|---|---|
| $\Delta_i^a$ | $\epsilon_a - \epsilon_i$ | `e_ia` |
| $g_{ia}(u)$ | $\Delta_i^a/(u^2 + (\Delta_i^a)^2)$; $1/\Delta_i^a$ at $u = 0$ | `g`, `inv_de` |
| $\mu_{d,ia}$ | MO dipole block, $d = x,y,z$, about the origin $\mathbf 0$ | `mu_flat` |
| $B^P_{ia}$ | dressed three-index tensor, occupied–virtual block | `b_ov` |
| $\tilde{\boldsymbol\varepsilon}(iu)$ | $\mathbf 1 + 4\,B\,\mathrm{diag}(g)\,B^{\mathsf T}$ (closed shell) | `eps_mat` |
| $w_d$ | $B\,(g\circ\mu_d)$ | `w_vec` |
| $\lambda_\xi$, $U_\xi$ | dielectric eigenvalues and eigenvectors at $iu$ | `ResponseModes::eigenvalues`, the coefficients |
| $s_{\xi,d}$ | field coupling of mode $\xi$, $4\,U_\xi\!\cdot w_d$ | `field_couplings` |

| Entry point | Location |
|---|---|
| Static | `ferric-rpa/src/properties.rs::{pdep_polarizability_static,pdep_polarizability_static_unrestricted}` |
| Dynamic | `ferric-rpa/src/properties.rs::molecular_dynamic_polarizability` |
| Modes | `ferric-rpa/src/response_modes.rs::{response_modes,polarizability_from_modes}` |
| CLI | `method.kind = "pdep-rpa"` with `[rpa] compute_polarizability` (on by default under `export_npz`); `[rpa] compute_c6` for the frequency grid |
| Python | `run_pdep_rpa(properties=True)` giving `alpha_tensor`; `response_modes(..., n_modes=, omega=)` |

## The dielectric form {#polarizabilities-from-dielectric-modes-dielectric-form}

Start from one orbital pair. A field along $d$ couples occupied $i$ to
virtual $a$ through $\mu_{d,ia}$, and an independent pair answers at two
poles, $\Delta_i^a$ and $-\Delta_i^a$. At imaginary frequency the two poles
add to a real number:

$$ \frac{1}{\Delta_i^a - iu} + \frac{1}{\Delta_i^a + iu} = \frac{2\Delta_i^a}{(\Delta_i^a)^2 + u^2} = 2\,g_{ia}(u) . $$

So $g_{ia}(u)$ is half the pair's response. It equals $1/\Delta_i^a$ at
$u = 0$, falls as $\Delta_i^a/u^2$ at large $u$, and is positive everywhere.

Now couple the pairs. In singlet direct RPA the orbital Hessian blocks are
$\mathbf A - \mathbf B = \Delta$ and $\mathbf A + \mathbf B = \Delta + 4\mathbf K$,
with $K_{ia,jb} = (ia|jb) = \sum_PB^P_{ia}B^P_{jb}$, so $\mathbf K = B^{\mathsf T}B$.
The response formula of section @sec:rpax-static-polarizability with these blocks gives

$$ \alpha_{de}(iu) = 4\,\mu_d^{\mathsf T}\big[\Delta(\Delta + 4\mathbf K) + u^2\big]^{-1}\Delta\,\mu_e . \label{eq:pol-dielectric-form} $$

The prefactor 4 is 2 for spin and 2 for the pair of poles. Two steps take
(@eq:pol-dielectric-form) to the form the code solves. First, factor $\Delta$ out of the bracket
on the left, $\Delta(\Delta + 4\mathbf K) + u^2 = \Delta\big[\Delta + u^2\Delta^{-1} + 4\mathbf K\big]$,
and cancel it against the trailing $\Delta$. Since
$\Delta + u^2\Delta^{-1} = \mathrm{diag}(g)^{-1}$,

$$ \alpha_{de}(iu) = 4\,\mu_d^{\mathsf T}\big[\mathrm{diag}(g)^{-1} + 4B^{\mathsf T}B\big]^{-1}\mu_e . $$

Second, apply the Woodbury identity to the inverse, with $G = \mathrm{diag}(g)$:

$$ \big[G^{-1} + 4B^{\mathsf T}B\big]^{-1} = G - 4\,GB^{\mathsf T}\big[\mathbf 1 + 4BGB^{\mathsf T}\big]^{-1}BG . $$

The bracket on the right is $\tilde{\boldsymbol\varepsilon}(iu)$. Sandwich
between $\mu_d$ and $\mu_e$ and collect $BG\mu_d = w_d$:

$$ \alpha_{de}(iu) = 4\,\mu_d^{\mathsf T}\mathrm{diag}(g)\,\mu_e - 16\,w_d^{\mathsf T}\,\tilde{\boldsymbol\varepsilon}(iu)^{-1}\,w_e , \qquad w_d = B\,(g\circ\mu_d) . \label{eq:pol-kernel-vs-casida} $$

The $n_on_v$-sized inverse has become an $N_{\rm aux}$-sized one. The first
term is the polarizability of independent particles. The second is the
screening: the induced density makes a Coulomb potential that pushes back on
the electrons. $\tilde{\boldsymbol\varepsilon}$ is positive definite, so the
term is nonnegative on the diagonal and lowers each $\alpha_{dd}$.

The two equation labels read the other way round from their names: the label
`pol-dielectric-form` sits on the $n_on_v$ response formula, and
`pol-kernel-vs-casida` on the dielectric form that the code solves. Every
reference below follows that.

Check two limits. At $u = 0$ every $g_{ia}$ is $1/\Delta_i^a$ and (@eq:pol-kernel-vs-casida) is
the static polarizability. As $u \to \infty$, $g_{ia} \to \Delta_i^a/u^2$, so
$w_d$ falls as $1/u^2$ and the screening term as $1/u^4$; the tensor tends to
$4\,\mu_d^{\mathsf T}\Delta\,\mu_e/u^2$, and the coupling $\mathbf K$ drops
out. The source comment above the loop below writes out the same two steps.
`scripts/rpa_response_modes_sympy.py` claim M5 checks the large-$u$ limit on
a random model: $u^2\alpha$ approaches $4\,\mu^{\mathsf T}\Delta\,\mu$, not
$4\,\mu^{\mathsf T}(\Delta+4\mathbf K)\mu$, and the screening term falls by
16 when $u$ doubles. The verified boxes cover (@eq:pol-kernel-vs-casida) itself.

<!-- excerpt: ferric-rpa/src/properties.rs::pdep_polarizability_static -->
```rust
    let mut tensor = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let bare = mu_flat[i].dot(&mu_flat_inv[j]); // = μ^i^T D^{-1} μ^j
            let coupled = w_vec[i].dot(&y_vec[j]);
            tensor[i][j] = 4.0 * bare - 16.0 * coupled;
        }
    }
```

| Symbol | Variable |
|---|---|
| $\mu_d^{\mathsf T}\mathrm{diag}(g)\,\mu_e$ | `bare` (here with $g = 1/\Delta$) |
| $w_d^{\mathsf T}\tilde{\boldsymbol\varepsilon}^{-1}w_e$ | `coupled`, with `y_vec` $= \tilde{\boldsymbol\varepsilon}^{-1}w$ from `solve_dielectric_3` |
| 4, 16 | the prefactor of the independent-particle term (2 for spin, 2 for the pair of poles), and 16 = 4 (that prefactor) $\times$ 4 (the dielectric's $4B\Delta^{-1}B^{\mathsf T}$) |

Three points about this loop:

- **One factorization, three right-hand sides.** `solve_dielectric_3`
  Cholesky-factors $\tilde{\boldsymbol\varepsilon}$ once (it is symmetric
  positive definite) and back-solves for $w_x$, $w_y$, $w_z$.
- **The static driver precomputes $\Delta^{-1}\mu$.** `mu_flat_inv` holds
  $\mu_d/\Delta$, so `bare` is the $u = 0$ independent-particle term. The
  source comment writes $D^{-1}$ for $\Delta^{-1}$.
- **The tensor is then symmetrized.** The result also carries the
  isotropic average $\tfrac13\operatorname{Tr}\alpha$ and the principal values,
  the eigenvalues of the symmetrized tensor sorted ascending.

The dipole on both sides is the whole-molecule lab-frame dipole about
$\mathbf 0$. Moving the origin by $\mathbf c$ adds $c_d\langle i|a\rangle$ to
$\mu_{d,ia}$, which is zero because occupied and virtual orbitals are
orthogonal, so the tensor does not depend on the origin.

The dynamic driver evaluates the same expression with $g_{ia}(u)$ in place of
$1/\Delta_i^a$. Its per-frequency closure in
`ferric-rpa/src/properties.rs::molecular_dynamic_polarizability` starts by
building $g$:

<!-- excerpt: ferric-rpa/src/properties.rs::molecular_dynamic_polarizability -->
```rust
                    let omega2 = omega * omega;
                    let mut g = ndarray::Array1::<f64>::zeros(nov);
                    for ia in 0..nov {
                        let e = e_ia[ia];
                        g[ia] = e / (omega2 + e * e);
                    }
```

At $u = 0$ this gives $\Delta/\Delta^2$, which can differ from the static
driver's $1/\Delta$ in the last bit. The two drivers therefore agree to
rounding at $u = 0$, not bit for bit.

> **Verified equation @eq:pol-dielectric-form** · the dielectric form equals the direct response solve
>
> The dielectric form (@eq:pol-kernel-vs-casida) equals an independent direct solve of (@eq:pol-dielectric-form) in
> the $n_on_v$ space, for the full tensor, at $u=0$ and $u=0.9$.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_response_modes_sympy.py`, claim M1 |
> | Negative control | the runner changes the screening prefactor 16 to 4 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/properties.rs::pdep_polarizability_static` |
> | Code test | `alpha_row` in `crates/ferric-rpa/tests/validation_static_alpha.rs` |
>
> <!-- verified: id=pol.dielectric-form; script=scripts/rpa_response_modes_sympy.py; claim=M1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/properties.rs::pdep_polarizability_static; rust_test=ferric-rpa/tests/validation_static_alpha.rs::alpha_row; registry=rpa-polarizability-c6 -->

A second, independent route to the same response is the sum over the
direct-RPA (Casida) excitations. Diagonalize
$\Delta^{1/2}(\Delta+4\mathbf K)\Delta^{1/2}\mathbf Z_n = \Omega_n^2\mathbf Z_n$;
then

$$ \alpha(iu) = \sum_n\frac{f_n}{\Omega_n^2 + u^2}, \qquad f_n = 4\,(m^{\mathsf T}\Delta^{1/2}\mathbf Z_n)(\mathbf Z_n^{\mathsf T}\Delta^{1/2}\mu) , $$

with $m$ the bra and $\mu$ the ket. For a diagonal molecular element the bra
is $\mu_d$ too, every $f_n$ is a square, and $\alpha_{dd}(iu)$ is a sum of
positive Lorentzians: positive and decreasing in $u$. The validation below
compares against this route.

> **Verified equation @eq:pol-kernel-vs-casida** · the kernel against the sum over excitations
>
> The coded kernel $4\,m\cdot(g\circ\mu) - 16\,(B(g\circ m))^{\mathsf T}\tilde{\boldsymbol\varepsilon}^{-1}(B(g\circ\mu))$
> equals the sum over the closed-shell dRPA excitations for any bra $m$, at
> $u = 0$, $0.4$ and $1.7$.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_c6_polarizability_sympy.py`, check `kernel-vs-casida` |
> | Negative control | the runner changes the kernel's screening term 16 to 4 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/properties/per_atom.rs::closed_per_atom_alpha` |
> | Code test | `pdep_c6_per_atom_becke_h2o_aug_cc_pvdz_vs_numpy` in `crates/ferric-rpa/tests/validation_pdep_c6.rs` |
>
> <!-- verified: id=pol.kernel-vs-casida; script=scripts/rpa_c6_polarizability_sympy.py; claim=kernel-vs-casida; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/properties/per_atom.rs::closed_per_atom_alpha; rust_test=ferric-rpa/tests/validation_pdep_c6.rs::pdep_c6_per_atom_becke_h2o_aug_cc_pvdz_vs_numpy; registry=rpa-polarizability-c6 -->

The same script checks that the static form $g = 1/\Delta$ equals the
$u \to 0$ limit of the dynamic one (check `static-limit`), and
`scripts/rpa_response_modes_sympy.py` claim M3 checks the same limit for the
molecular form.

## The mode form {#polarizabilities-from-dielectric-modes-mode-form}

The dielectric matrix is symmetric, so it has an eigendecomposition,
$\tilde{\boldsymbol\varepsilon}(iu) = \sum_\xi\lambda_\xi U_\xi U_\xi^{\mathsf T}$,
and its inverse is $\sum_\xi U_\xi U_\xi^{\mathsf T}/\lambda_\xi$. Substitute
the inverse into (@eq:pol-kernel-vs-casida) and write $16\,w_d^{\mathsf T}U_\xi U_\xi^{\mathsf T}w_e = s_{\xi,d}s_{\xi,e}$:

$$ \alpha(iu) = \alpha_{\rm bare}(iu) - \sum_\xi\frac{s_\xi s_\xi^{\mathsf T}}{\lambda_\xi},\qquad s_{\xi,d} = 4\,U_\xi\cdot w_d,\quad \alpha_{\rm bare} = 4\,\mu^{\mathsf T}\mathrm{diag}(g)\,\mu . \label{eq:pol-mode-expansion} $$

Each mode is a charge density in the auxiliary basis. $s_\xi$ is its Coulomb
overlap with the density that a unit field induces in independent particles,
and $s_\xi s_\xi^{\mathsf T}/\lambda_\xi$ is how much that mode screens the
field. Every $\lambda_\xi \ge 1$ (section @sec:pdep-dielectric-matrix-static-modes), so mode $\xi$ removes at
most $s_{\xi,d}^2$ from $\alpha_{dd}$. With every mode kept, (@eq:pol-mode-expansion) is (@eq:pol-kernel-vs-casida). Keeping
only the leading modes truncates the screening sum, and the modes are
ordered by $\lambda_\xi$, the strength of the screening.
`ferric-rpa/src/response_modes.rs::polarizability_from_modes` evaluates it:

<!-- excerpt: ferric-rpa/src/response_modes.rs::polarizability_from_modes -->
```rust
        let mut a = self.alpha_bare;
        for (k, &lam) in self.eigenvalues.iter().enumerate() {
            for d in 0..3 {
                for e in 0..3 {
                    a[d][e] -= self.field_couplings[(k, d)] * self.field_couplings[(k, e)] / lam;
                }
            }
        }
        a
```

`field_couplings` is stored as an $M\times3$ array, row $k$ holding
$s_{k,x}, s_{k,y}, s_{k,z}$ for the $k$-th kept mode.

> **Verified equation @eq:pol-mode-expansion** · the mode expansion
>
> With every mode kept, $\alpha_{\rm bare} - \sum_\xi s_\xi s_\xi^{\mathsf T}/\lambda_\xi$
> equals (@eq:pol-kernel-vs-casida); keeping 3 of 7 modes does not.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_response_modes_sympy.py`, claim M2 |
> | Negative control | the 3-of-7 truncation must differ; the runner flips the sign of the mode sum |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/response_modes.rs::polarizability_from_modes` |
> | Code test | `full_rank_modes_reconstruct_the_static_polarizability` in `crates/ferric-rpa/tests/response_modes.rs` |
>
> <!-- verified: id=pol.mode-expansion; script=scripts/rpa_response_modes_sympy.py; claim=M2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/response_modes.rs::polarizability_from_modes; rust_test=ferric-rpa/tests/response_modes.rs::full_rank_modes_reconstruct_the_static_polarizability; registry=rpa-polarizability-c6 -->

Water / cc-pVDZ, from the `589b50832` build: the dielectric form of
`run_pdep_rpa` and the mode form of `response_modes` with all 84 modes
agree to $4\times10^{-16}$ Bohr$^3$. The leading 10 modes alone leave
$\alpha_{\rm iso}$ 1% high, since the dropped modes screen too, and the
unscreened $\alpha_{\rm bare}$ is 33% high. The last four lines are
$\alpha_{\rm iso}(iu)$ along the imaginary axis.

```python
import numpy as np
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

r = ferric.run_pdep_rpa(mol, bs, aux, properties=True)
alpha = np.asarray(r.alpha_tensor)
print("alpha_tensor (Bohr^3):\n", alpha)
print("alpha_iso = Tr/3 =", np.trace(alpha) / 3)

rhf = ferric.run_rhf(mol, bs)
# A large n_modes is clamped to the dressed auxiliary dimension: every mode kept.
modes = ferric.response_modes(mol, bs, aux, rhf, n_modes=10_000, omega=0.0)
full = np.asarray(modes.polarizability_from_modes())
print("n_modes kept:", modes.n_modes, "of naux", modes.naux)
print("lambda_min (>= 1):", np.asarray(modes.eigenvalues).min())
print("max |mode form - dielectric form|:", np.abs(full - alpha).max())

# Truncation: the leading 10 modes only.
lead = ferric.response_modes(mol, bs, aux, rhf, n_modes=10, omega=0.0)
print("alpha_iso, 10 modes:", np.trace(np.asarray(lead.polarizability_from_modes())) / 3)
print("alpha_bare iso (no screening):", np.trace(np.asarray(lead.alpha_bare)) / 3)

# Dynamic: alpha(iu) along the imaginary axis falls with u.
for u in (0.0, 0.5, 1.0, 2.0):
    m = ferric.response_modes(mol, bs, aux, rhf, n_modes=10_000, omega=u)
    print(f"u = {u:4.1f}  alpha_iso(iu) = {np.trace(np.asarray(m.polarizability_from_modes())) / 3:.6f}")
```

```text
alpha_tensor (Bohr^3):
 [[ 2.03909217e+00  1.41768136e-15 -4.84218500e-16]
 [ 1.41768136e-15  4.09661992e+00 -1.93510960e-14]
 [-4.84218500e-16 -1.93510960e-14  3.19683930e+00]]
alpha_iso = Tr/3 = 3.110850466940373
n_modes kept: 84 of naux 84
lambda_min (>= 1): 1.0000000000000002
max |mode form - dielectric form|: 4.440892098500626e-16
alpha_iso, 10 modes: 3.1460968880402302
alpha_bare iso (no screening): 4.149029589112157
u =  0.0  alpha_iso(iu) = 3.110850
u =  0.5  alpha_iso(iu) = 2.721055
u =  1.0  alpha_iso(iu) = 2.110306
u =  2.0  alpha_iso(iu) = 1.261344
```

At $u=0$ the modes are the PDEP eigenvectors of section @sec:pdep (test
`leading_static_modes_are_the_pdep_eigenpotentials`). They live in the
dressed auxiliary basis of section @sec:factorization-and-metric. Let $V$ be
the auxiliary Coulomb metric $(P|Q)$, $M$ the metric factor with
$MVM^{\mathsf T} = \mathbf 1$ that dresses the three-index integrals,
$B = M\,b$ for the undressed $b^P_{ia} = (P|ia)$, and
$\Pi(iu) = 4\,b\,\mathrm{diag}(g)\,b^{\mathsf T}$ the undressed response.
The physical coefficients $c_\xi = M^{\mathsf T}U_\xi$ obey
$(V + \Pi(iu))c_\xi = \lambda_\xi Vc_\xi$ with
$c_\xi^{\mathsf T}Vc_\eta = \delta_{\xi\eta}$. Read that equation physically:
add the density $\rho_\xi$ to the molecule and the electrons rearrange until
the net density is $\rho_\xi/\lambda_\xi$. A mode is the charge density
$\sum_Pc_P\chi_P(\mathbf r)$, and its potential is the Coulomb potential of
that density. Reading the coefficients as a potential is wrong.

## Open shell {#polarizabilities-from-dielectric-modes-open-shell}

With different $\alpha$ and $\beta$ orbitals each spin has its own pairs,
gaps and dipole block, but both spins see the same induced Coulomb potential.
So the screening is shared and the bare terms are per spin. With one spin per
pair the spin factor 2 is gone from each prefactor:

$$ \tilde{\boldsymbol\varepsilon}_U = \mathbf 1 + 2\sum_\sigma B_\sigma\,\mathrm{diag}(g_\sigma)B_\sigma^{\mathsf T},\qquad \alpha_{de} = \sum_\sigma 2\,\mu_{\sigma,d}^{\mathsf T}\mathrm{diag}(g_\sigma)\mu_{\sigma,e} - 4\,w_d^{\mathsf T}\tilde{\boldsymbol\varepsilon}_U^{-1}w_e,\quad w_d = \sum_\sigma B_\sigma(g_\sigma\circ\mu_{\sigma,d}) . \label{eq:pol-open-shell} $$

Check the closed-shell limit. When the spins coincide the two terms of
$\tilde{\boldsymbol\varepsilon}_U$ add to $4BGB^{\mathsf T}$, the bare terms
add to $2 + 2 = 4$, and $w$ doubles, so the screening term becomes
$4\,(2w_\sigma)^{\mathsf T}\tilde{\boldsymbol\varepsilon}^{-1}(2w_\sigma) = 16\,w_\sigma^{\mathsf T}\cdots$.
That is (@eq:pol-kernel-vs-casida). ROHF and ROKS references are semi-canonicalized per spin
first by `ferric-scf/src/semicanonical.rs::unrestricted_reference`, as in
section @sec:open-shell-gw; a UHF or UKS reference passes through unchanged.

> **Verified equation @eq:pol-open-shell** · the open-shell form against the spin-orbital response
>
> For unequal $\alpha$ and $\beta$ spaces (2 occupied and 4 virtual, 1
> occupied and 5 virtual), (@eq:pol-open-shell) equals a direct solve of the spin-orbital
> direct-RPA response, $2\mu_{\rm so}^{\mathsf T}[\Delta(\Delta+2\mathbf K_{\rm so})+u^2]^{-1}\Delta\mu_{\rm so}$,
> at $u = 0$ and $0.9$.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_response_modes_sympy.py`, claim M4 |
> | Negative control | leaving the $\beta$ channel out of $w$ must miss; the runner changes the screening prefactor 4 to 16 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/properties.rs::molecular_dynamic_polarizability` |
> | Code test | `u_polarizability_oh_matches_pyscf` in `crates/ferric-rpa/tests/u_pdep_rpa.rs` |
>
> <!-- verified: id=pol.open-shell; script=scripts/rpa_response_modes_sympy.py; claim=M4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/properties.rs::molecular_dynamic_polarizability; rust_test=ferric-rpa/tests/u_pdep_rpa.rs::u_polarizability_oh_matches_pyscf; registry=rpa-polarizability-c6 -->

## The algorithm {#polarizabilities-from-dielectric-modes-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | reference (RHF or RKS), all occupied active (no frozen core: the response is on every occupied orbital) | | the caller |
| 2 | $B^P_{ia}$ and the gaps, the same intermediates as RPA | as RI-MP2 | `compute_rpa_intermediates` (ferric-mp2) |
| 3 | $\mu_{d,ia}$ from the AO dipole and the MO coefficients | $O(N^2 n_o + N n_on_v)$ per component | `oneelectron::dipole` |
| 4 | $\tilde{\boldsymbol\varepsilon}(iu)$, one dense matrix per frequency | $N_{\rm aux}^2n_on_v$ | `pdep_polarizability_static`, `molecular_dynamic_polarizability` |
| 5 | one SPD solve for the three right-hand sides $w_d$ | $N_{\rm aux}^3$ | `solve_dielectric_3` |
| 6 | the tensor, symmetrized; principal values by a $3\times3$ eigensolve | $O(1)$ | `eig3_sym` |
| 6' | the mode form: a dense eigensolve of $\tilde{\boldsymbol\varepsilon}$, the leading $M$ modes and their couplings | $N_{\rm aux}^3$ | `response_modes` |

Frequencies are independent, so the dynamic driver loops over the
imaginary-frequency nodes in parallel with one BLAS thread per worker. Its
memory pre-flight charges `b_ov`, the scaled copy `b_scaled` and `eps_mat`
per spin. The open-shell branch also charges the second $N_{\rm aux}^2$
buffer that the sum of the two spins' products needs. A solve of a
non-positive-definite $\tilde{\boldsymbol\varepsilon}$ (a NaN dielectric)
returns an error. It never panics inside the parallel region.

`response_modes` clamps `n_modes` to the dressed auxiliary dimension.

## Options and limits {#polarizabilities-from-dielectric-modes-options-limits}

| Option | Effect |
|---|---|
| `[rpa] n_quad`, `quadrature`, `u0` | the frequency grid of the dynamic tensor, and of $C_6$ |
| `memory_budget_gb` | the pre-flight ceiling |
| `n_modes`, `omega` (`response_modes`) | how many leading modes, and the imaginary frequency |

**Known limits (2026-11-15).** The response is direct RPA: the kernel has no
exchange, so the static tensor is not the CPHF polarizability.
`response_modes` is closed shell only, and refuses a negative or non-finite
$u$ and `n_modes = 0`.

## Numerical anchors {#polarizabilities-from-dielectric-modes-numerical-anchors}

From `site/src/reference/validation.md` (the static $\alpha$ and the dynamic
$\alpha(iu)$ rows of the properties table):

| Quantity | Systems | Error | Test |
|---|---|---|---|
| static $\alpha$ against a numpy linear solve on PySCF `int3c2e`, same auxiliary basis | H2O / aug-cc-pVDZ; CH3OH / cc-pVDZ | $4.8\times10^{-14}$ relative on the same orbitals; $5.3\times10^{-10}$ with ferric's own SCF | `crates/ferric-rpa/tests/validation_static_alpha.rs` |
| $\alpha(iu)$ against a numpy sum over Casida excitations at ferric's 20 Gauss–Legendre nodes | H2O, N2 / aug-cc-pVDZ | $4.9\times10^{-14}$ relative on the same orbitals | `crates/ferric-rpa/tests/validation_pdep_c6.rs` |

## Sources {#polarizabilities-from-dielectric-modes-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-rpa/src/response_modes.rs`, module documentation | the mode form (@eq:pol-mode-expansion) and the meaning of a mode |
| `scripts/rpa_response_modes_sympy.py`, claims M1 to M4 | (@eq:pol-kernel-vs-casida) to (@eq:pol-open-shell) |
| `scripts/rpa_c6_polarizability_sympy.py`, check `kernel-vs-casida` | the sum-over-excitations route |
