# Hartree–Fock nuclear gradient {#hf-gradient}

<!-- status: ready; registry: hf-gradient -->

Move one nucleus a little and the SCF energy changes. The slope of that
change is the gradient, and minus the gradient is the force on the nucleus.
You could get it by running two SCFs and differencing, at two SCFs per
coordinate. The analytic route costs about one Fock build, because a
converged SCF energy does not change to first order when its orbitals do,
so the orbitals' own derivative never has to be computed. What remains is
the derivative of the integrals, plus one term that exists because the basis
functions ride on the nuclei and must stay orthonormal as they move.

The plan: write the gradient from a Lagrangian, read off the
energy-weighted density $W$ for each reference, then look at the loops that
contract it.

| Symbol | Meaning | Code name |
|---|---|---|
| $x$ | one nuclear Cartesian coordinate | column of the `(natoms, 3)` result |
| $D$ | total AO density, $D = 2\,C_o C_o^{\mathsf T}$ (closed shell) | `result.density_r()`, `d` |
| $W$ | energy-weighted density, the AO form of the orthonormality multiplier | `w`, `build_energy_weighted_density` |
| $\Gamma_{\mu\nu\lambda\sigma}$ | two-particle density, $\sum\Gamma(\mu\nu\vert\lambda\sigma) = E_{2e}$ | `gamma`, `gamma_uhf` |
| $h^x$, $S^x$, $(\mu\nu\vert\lambda\sigma)^x$ | integral derivatives at fixed basis coefficients | derivative engines of `ferric-integrals` |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/gradient.rs::{rhf_gradient,uhf_gradient,rohf_gradient}`; the Kohn–Sham routes of section @sec:xc-gradient call the same pieces |
| CLI | `task = "optimize"` for `rhf`, `uhf`, `rohf`, `ksdft` (`examples/h2-lda-opt.toml`) |
| Python | `run_rhf(..., with_gradient=True)`, `run_optimize`, `run_frequencies` |

This section covers the exact four-centre route. When the SCF density-fitted
J or K, the two-electron term differentiates the fitted energy instead
(section @sec:df-jk-gradient). COSX exchange has its own gradient
(`ferric-scf/src/gradient.rs::rhf_gradient_cosx`) and is not derived here.

## The gradient {#hf-gradient-gradient}

The energy depends on $x$ in two ways. The integrals $h(x)$,
$(\mu\nu|\lambda\sigma)(x)$, $S(x)$ and $E_{nn}(x)$ change because the basis
functions sit on the nuclei (section @sec:integrals-derivatives differentiates them). The MO
coefficients $C(x)$ change because the SCF re-solves at the new geometry.
The second dependence is the expensive one: $\mathrm dC/\mathrm dx$ would need
a linear solve per coordinate. The SCF conditions remove it.

**The Lagrangian.** The SCF minimizes $E[C]$ subject to orthonormal occupied
orbitals, $C_o^{\mathsf T} S C_o = 1$. Attach a symmetric multiplier
$\varepsilon'$ (one per occupied pair and spin):

$$ L = E[C] - \mathrm{tr}\big[\varepsilon'\,(C_o^{\mathsf T} S C_o - 1)\big] . $$

At the SCF solution $L$ is stationary in $C$ and the constraint holds, so
$L = E$ along the path of solutions and the chain-rule term through $C$
drops out:

$$ \frac{\mathrm dE}{\mathrm dx} = \frac{\partial L}{\partial x}\Big|_{C\ \text{fixed}} . $$

For RHF the multiplier is fixed by the Fock equations. With
$D = 2C_oC_o^{\mathsf T}$ the energy is
$E = \sum D h + \tfrac12\sum D\,G(D) + E_{nn}$, and differentiating in
$C_o$ gives $\partial E/\partial C_o = 4FC_o$. Stationarity of $L$ is then

$$ 4FC_o - 2SC_o\varepsilon' = 0 . $$

Compare with the Roothaan equations $FC_o = SC_o\varepsilon$ in canonical
orbitals: $\varepsilon' = 2\,\mathrm{diag}(\varepsilon_o)$. Now differentiate
$L$ at fixed $C$. The energy gives the integral derivatives (the skeleton).
The constraint gives
$-\mathrm{tr}[\varepsilon' C_o^{\mathsf T} S^x C_o] = -\mathrm{tr}[W S^x]$ with
$W = 2\,C_o\,\mathrm{diag}(\varepsilon_o)\,C_o^{\mathsf T}$. Collect the terms:

$$ E^x = E_{nn}^x + \sum_{\mu\nu} D_{\mu\nu} h^x_{\mu\nu}
+ \sum_{\mu\nu\lambda\sigma} \Gamma_{\mu\nu\lambda\sigma} (\mu\nu|\lambda\sigma)^x
- \sum_{\mu\nu} W_{\mu\nu} S^x_{\mu\nu} . $$

The derivatives $h^x$, $(\mu\nu|\lambda\sigma)^x$ and $S^x$ are taken at fixed
basis coefficients. $\Gamma$ is the two-particle density, defined so that
$\sum\Gamma(\mu\nu|\lambda\sigma) = E_{2e}$, and $W$ is the orthonormality
multiplier in the AO basis. Each reference has its own pair:

| reference | $\Gamma_{\mu\nu\lambda\sigma}$ (`gamma`, `gamma_uhf`) | $W$ |
|---|---|---|
| RHF | $\tfrac12 D_{\mu\nu}D_{\lambda\sigma} - \tfrac14 D_{\mu\lambda}D_{\nu\sigma}$ | $2\,C_o\,\mathrm{diag}(\varepsilon_o)\,C_o^{\mathsf T}$ (`build_energy_weighted_density`) |
| UHF | $\tfrac12 D_{\mu\nu}D_{\lambda\sigma} - \tfrac12\sum_\tau D^\tau_{\mu\lambda}D^\tau_{\nu\sigma}$ | $\sum_\sigma C_{\sigma}\,\mathrm{diag}(f_\sigma \varepsilon_\sigma)\,C_\sigma^{\mathsf T}$ (`build_energy_weighted_density_uhf`) |
| ROHF | UHF form at the ROHF $D_\alpha, D_\beta$ | $\mathrm{sym}(D_\alpha F_\alpha D_\alpha + D_\alpha F_\beta D_\beta)$ (`rohf_energy_weighted_density`) |

For RHF and UHF the canonical orbitals diagonalize the multiplier, so $W$ is
the occupation-weighted sum over eigenvalues. The UHF $f_\sigma$ are the
occupations recovered from the stored density: 1 for aufbau, fractions under
`fractional_occ` (section @sec:hf-gradient-reproducibility-refusals says how they are recovered). For open
shells $D = D_\alpha + D_\beta$ in the one-electron term.

**Check a limit: a basis that does not move.** If the basis functions stayed
in place while the nucleus moved, $S^x = 0$, $(\mu\nu|\lambda\sigma)^x = 0$,
and $h^x$ would be the derivative of the nuclear-attraction operator alone.
The gradient would collapse to $E_{nn}^x + \sum D\,(\partial V/\partial x)$,
the Hellmann–Feynman force: the classical pull of the electron density on
the nucleus. The $W$ term (the Pulay force) and the parts of $h^x$ and
$(\mu\nu|\lambda\sigma)^x$ that come from moving basis centres all exist
because the basis functions are centred on the nuclei.

**Check translational invariance.** Move every nucleus by the same vector.
Each integral is unchanged, so the derivatives of one integral with respect
to all its centres sum to zero (section @sec:integrals-derivatives), and $E_{nn}^x$ sums to zero
pair by pair. Summing $E^x$ over all atoms along one axis sums each
integral's derivative over all its centres, at fixed $D$, $W$ and $\Gamma$.
Every term vanishes, and the forces on all atoms add to zero. This holds
without external point charges or a field, which stay fixed while the
molecule moves. A gradient of an isolated molecule whose rows do not sum to
zero, up to the integral screening, has a contraction error.

> **Verified equation {#eq:scf-grad-hf}** · RHF, UHF and ROHF gradients equal finite differences
>
> On a toy system whose integrals are smooth functions of two parameters,
> with a non-orthogonal overlap, the gradient assembled from the code's
> $\Gamma$ and $W$ equals a fourth-order finite difference of the
> re-solved SCF energy: RHF, UHF and ROHF each.
>
> | | |
> |---|---|
> | Proof | `scripts/hf_gradient_sympy.py`, claims G1 (RHF), G2 (UHF), G3 (ROHF) |
> | Negative control | the eigenvalue form $C\,\mathrm{diag}(2\varepsilon_c,\varepsilon_o)\,C^{\mathsf T}$ of the ROHF $W$ must miss the finite difference (claim G5); mutations in the runner halve the RHF $W$, change the ROHF $W$ term and the UHF exchange weight |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/gradient.rs::{build_energy_weighted_density,build_energy_weighted_density_uhf,rohf_energy_weighted_density,gamma,gamma_uhf}` |
> | Code test | `test_gradient_h2o_sto3g_vs_finite_diff` in `crates/ferric-scf/src/gradient.rs` |
>
> <!-- verified: id=scf.grad.hf; script=scripts/hf_gradient_sympy.py; claim=G1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/gradient.rs::{build_energy_weighted_density,build_energy_weighted_density_uhf,rohf_energy_weighted_density,gamma,gamma_uhf}; rust_test=ferric-scf/src/gradient.rs::test_gradient_h2o_sto3g_vs_finite_diff; registry=hf-gradient -->

The toy system has integrals quadratic in two parameters. The script
re-solves the SCF at displaced parameters and differences the energy with a
fourth-order stencil. The analytic formula with the code's $W$ and $\Gamma$
matches to $3\times10^{-11}$ (RHF), $2\times10^{-11}$ (UHF) and
$2\times10^{-11}$ (ROHF).

Both checks on a real molecule, from the `589b50832` build: water in
cc-pVDZ, pushed off its minimum so the forces are not zero. The rows of the
analytic gradient sum to zero at the level of rounding, and two components
match a central difference of the energy ($h = 10^{-3}$ Bohr) to about
$10^{-7}$ Ha/Bohr, the $O(h^2)$ error of the stencil.

```python
import numpy as np
import ferric

# Water, distorted from the HF/cc-pVDZ minimum so the forces are not zero (Bohr).
sym = ["O", "H", "H"]
x0 = np.array([[0.0, 0.0, 0.25], [0.0, 1.50, -0.85], [0.0, -1.35, -0.95]])
bs = ferric.BasisSet.bundled("cc-pvdz")


def run(x, grad=False):
    mol = ferric.Molecule.from_arrays(sym, x, units="bohr")
    return ferric.run_rhf(mol, bs, energy_conv=1e-11, density_conv=1e-9, with_gradient=grad)


g = np.asarray(run(x0, grad=True).gradient())
print("analytic gradient (Ha/Bohr):\n", g)
print("sum over atoms (translational invariance):", g.sum(axis=0))
h = 1e-3
for atom, axis in ((0, 2), (1, 1)):
    xp = x0.copy(); xp[atom, axis] += h
    xm = x0.copy(); xm[atom, axis] -= h
    fd = (run(xp).energy - run(xm).energy) / (2 * h)
    print(f"atom {atom} axis {axis}: analytic {g[atom, axis]:.10f}  FD {fd:.10f}  diff {abs(fd - g[atom, axis]):.1e}")
```

```text
analytic gradient (Ha/Bohr):
 [[ 4.22569852e-16 -2.39602914e-02  3.31319737e-02]
 [-2.26713528e-16  2.89788597e-02 -2.47658342e-02]
 [-1.95856324e-16 -5.01856830e-03 -8.36613955e-03]]
sum over atoms (translational invariance): [-3.94430453e-30 -2.77555756e-17 -1.33226763e-15]
atom 0 axis 2: analytic 0.0331319737  FD 0.0331318859  diff 8.8e-08
atom 1 axis 1: analytic 0.0289788597  FD 0.0289787254  diff 1.3e-07
```

> **Verified equation {#eq:scf-grad-pulay}** · the Pulay term is $-\mathrm{tr}[W\,S^x]$
>
> For one doubly occupied orbital in a two-function basis, at fixed
> normalized direction, $\mathrm dE/\mathrm dx$ equals the skeleton term
> minus $\mathrm{tr}[W S^x]$ with $W = 2\varepsilon\,c\,c^{\mathsf T}$, in
> symbols.
>
> | | |
> |---|---|
> | Proof | `scripts/hf_gradient_sympy.py`, claim S1 |
> | Negative control | the runner flips the sign of the Pulay term (claim G1 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/gradient.rs::oneelectron_gradient` |
>
> <!-- verified: id=scf.grad.pulay; script=scripts/hf_gradient_sympy.py; claim=S1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/gradient.rs::oneelectron_gradient; registry=hf-gradient -->

### Why $W$ for ROHF is not the eigenvalue form

ROHF has one orbital set and two Fock matrices, and its orbitals are
eigenvectors of neither. Stationarity of $L$ gives

$$ \frac{\partial E}{\partial C} = \sum_\sigma 2 F_\sigma C\,\mathrm{diag}(n_\sigma) = 2 S C \varepsilon , $$

so over occupied orbitals the multiplier is the full matrix
$\varepsilon_{ij} = \sum_\sigma n_{\sigma j} (f_\sigma)_{ij}$, with $f_\sigma$ the
spin Fock matrices in the MO basis. Transform to the AO basis. Every occupied
orbital holds an alpha electron, so
$C_{\mathrm{occ}} C_{\mathrm{occ}}^{\mathsf T} = D_\alpha$, and

$$ W = C_{\mathrm{occ}}\,\varepsilon\,C_{\mathrm{occ}}^{\mathsf T} = \sum_\sigma D_\alpha F_\sigma D_\sigma . $$

The antisymmetric part of $\varepsilon$ is
$\varepsilon_{tc} - \varepsilon_{ct} = (f_\beta)_{oc}$, the closed-open block
of the beta Fock matrix, which is zero at convergence. The code's `sym()`
only removes convergence noise.

<!-- excerpt: ferric-scf/src/gradient.rs::rohf_energy_weighted_density -->
```rust
let w = d_a.dot(f_a).dot(d_a) + d_a.dot(f_b).dot(d_b);
Ok(0.5 * (&w + &w.t()))
```

The eigenvalue form $C\,\mathrm{diag}(2\varepsilon_c, \varepsilon_o)\,C^{\mathsf T}$
of the ROHF effective Fock matrix is not this multiplier. It drops the
closed-open coupling, and the script's negative control misses by
$5\times10^{-2}$. It agrees only when symmetry decouples the open shell from
the closed shells, as it does for the $\pi$ SOMO of OH, so an OH-only test
cannot tell the two apart.

## The algorithm {#hf-gradient-algorithm}

| Step | Quantity | Function |
|---|---|---|
| 1 | $E_{nn}^x$, with `effective_z()` for ECP atoms | `ferric-scf/src/gradient.rs::oneelectron_gradient` |
| 2 | $\sum D\,h^x$: kinetic, nuclear attraction, and, when present, point-charge and field terms | same |
| 3 | $-\sum W\,S^x$ with $W$ from the converged orbitals and eigenvalues | `ferric-scf/src/gradient.rs::build_energy_weighted_density` |
| 4 | $\sum\Gamma\,(\mu\nu\vert\lambda\sigma)^x$ over Schwarz-screened shell quartets | `ferric-scf/src/gradient.rs::twoelectron_gradient`, `ferric-scf/src/gradient.rs::twoelectron_gradient_uhf` |
| 5 | $\sum D\,V_{\mathrm{ECP}}^x$, a no-op without ECPs | `ferric-scf/src/gradient.rs::ecp_gradient` |

Only converged-SCF quantities enter: no response equations are solved. The
gradient costs one pass of derivative integrals, comparable to one Fock build
over the same quartets.

Step 3 scales the occupied columns by their eigenvalues once and forms $W$
with one matrix product.

<!-- excerpt: ferric-scf/src/gradient.rs::build_energy_weighted_density -->
```rust
let c_occ = c.slice(ndarray::s![.., ..nocc]);
let eps_occ = ndarray::ArrayView1::from(&eps[..nocc]);
let cw = &c_occ * &eps_occ;
c_occ.dot(&cw.t()) * 2.0
```

| Code | Equation |
|---|---|
| `c_occ` | $C_o$ |
| `eps_occ` | $\varepsilon_i$ over occupied orbitals |
| `cw` | $\varepsilon_i\,C_{\mu i}$, columns scaled |
| the returned matrix | $W = 2\,C_o\,\mathrm{diag}(\varepsilon_o)\,C_o^{\mathsf T}$ |

The Pulay term itself is a loop over canonical shell pairs $s_1 \ge s_2$.
The inner body in `oneelectron_gradient` runs after the overlap derivative
block for the pair has been computed:

<!-- excerpt: ferric-scf/src/gradient.rs::oneelectron_gradient -->
```rust
let wval = if s1 == s2 {
    w[(mu, nu)]
} else {
    2.0 * w[(mu, nu)]
};
for c in 0..3 {
    // deriv layout: [dx1, dy1, dz1, dx2, dy2, dz2]
    let d1 = deriv[c * block_sz + idx];
    let d2 = deriv[(3 + c) * block_sz + idx];
    local[(a1, c)] -= wval * d1;
    local[(a2, c)] -= wval * d2;
}
```

| Code | Equation |
|---|---|
| `2.0 * w[(mu, nu)]` | $W_{\mu\nu} + W_{\nu\mu}$ for a pair visited once; $W$ is symmetric |
| `d1`, `d2` | $\partial S_{\mu\nu}/\partial A_c$ and $\partial S_{\mu\nu}/\partial B_c$, the bra and ket centre blocks |
| `local[(a1, c)] -= wval * d1` | $-W_{\mu\nu}S^x_{\mu\nu}$ added to the atom that carries $\mu$ |

Two points about this loop:

- **Each derivative block lands on the atom that carries its centre.** The
  derivative with respect to atom $A$ collects every centre that sits on
  $A$. That is the bookkeeping behind the translational-invariance check
  above: the two blocks of one pair sum to zero when both shells sit on one
  atom.
- **The factor 2 assumes a symmetric $W$.** The RHF and UHF forms are
  symmetric by construction, and the ROHF form is symmetrized before it gets
  here.

The closed-shell two-particle density is one line.

<!-- excerpt: ferric-scf/src/gradient.rs::gamma -->
```rust
0.5 * d[(mu, nu)] * d[(la, sg)] - 0.25 * d[(mu, la)] * d[(nu, sg)]
```

With `d` the total density $D$, the two products are the Coulomb weight
$\tfrac12 D_{\mu\nu}D_{\lambda\sigma}$ and the exchange weight
$-\tfrac14 D_{\mu\lambda}D_{\nu\sigma}$ of the table.

## Reproducibility and refusals {#hf-gradient-reproducibility-refusals}

- The gradient is bit-identical across rayon thread counts
  (`rhf_gradient_bit_identical_across_thread_counts` in
  `crates/ferric-scf/src/gradient.rs`).
- Fractional and non-aufbau occupations: the UHF $W$ uses the occupations
  that built the stored density. The code recovers them by projecting the
  density on each orbital, $f_p = c_p^{\mathsf T} S D S c_p$, and snaps the
  result to the aufbau pattern, the `fractional_occ` ensemble, or the
  nearest integer occupation with the right electron count (a MOM state),
  each within `ferric-scf/src/gradient.rs::W_OCCUPATION_SNAP_TOL` (0.1). A
  density that matches none of these is refused, because a Pulay term built
  on the wrong occupations is silently wrong.
- Ghost atoms are refused: derivatives with respect to a ghost centre need a
  different one-electron derivative structure.
- A relativistic (X2C) SCF is refused by
  `ferric-scf/src/x2c.rs::refuse_relativistic`; its gradient is not
  implemented.

## Options that change the equation {#hf-gradient-options-change-equation}

| Option | Effect | Where |
|---|---|---|
| ECPs | adds $\sum D V^x_{\mathrm{ECP}}$ and uses $Z_{\mathrm{eff}}$ in $E_{nn}^x$ | this section |
| external point charges and field | folded into the one-electron term; the field-density term is a finite difference | `ferric-scf/src/gradient.rs::oneelectron_gradient` |
| density-fitted J and K | the two-electron term differentiates the fitted energy | section @sec:df-jk-gradient |
| COSX exchange | exchange differentiated on the COSX grid | `ferric-scf/src/gradient.rs::rhf_gradient_cosx` |
| Kohn–Sham | adds the XC gradient and scales exchange by the hybrid fraction | section @sec:xc-gradient |
| solvent (PCM) | adds the analytic IEF-PCM nuclear gradient | chapter @sec:implicit-solvation |

## Numerical anchors {#hf-gradient-numerical-anchors}

| Check | Reference | Agreement | Test |
|---|---|---|---|
| RHF, UHF, ROHF and KS gradients, including density-fitted J/K: water, OH, HO2 / cc-pVDZ, 6-31G | finite differences of the energy; PySCF `df.grad` | 1e-7 to 3e-7 Ha/Bohr (FD); about 1e-10 (PySCF), bar 1e-6 Ha/Bohr | `crates/ferric-scf/tests/df_jk_gradient.rs` |
| RHF and UHF with def2 ECPs: HI, CH3I, RbH, I atom / def2-SVP, def2-TZVP; SnH4 / def2-TZVP; HBr (all-electron control) | PySCF analytic gradient and ORCA, both fed ferric's basis and ECP | at most 2.3e-8 Ha/Bohr, bar 1e-7 Ha/Bohr | `crates/ferric-scf/tests/validation_ecp.rs` |
| Geometry optimization of H2O, NH3, CH2O from distorted starts, RHF and B3LYP | PySCF gradient driven to $10^{-6}$ Ha/Bohr | distances at most 2.6e-6 Bohr, bar 2e-5 Bohr | `crates/ferric-scf/tests/validation_geometry_optimization.rs` |

<!-- figures: site/src/reference/validation.md "RHF/UHF/ROHF/KS gradients, including density-fitted J/K"; site/src/reference/validation.md "RHF and UHF with def2 ECPs: energies and analytic gradients"; site/src/reference/validation.md "Geometry optimization, RHF and RKS" -->

The registry's `hf-gradient` entry is the toy-model proof above; the
finite-difference rows are the end-to-end check of the same assembly on real
integrals.

## Sources

| Source | Used for |
|---|---|
| `scripts/hf_gradient_sympy.py`, claims G1, G2, G3, G5, S1 | verified equations @eq:scf-grad-hf and @eq:scf-grad-pulay |
| `crates/ferric-scf/src/gradient.rs` | the algorithm, refusals and options |
| `site/src/reference/validation.md` | the anchor rows |
