# 21.2 Atomic partitions: Becke, Hirshfeld, Löwdin, ESP fits

<!-- section: 21.2; status: edit; registry: atomic-charges;
     theory: crates/ferric-dft/theory.md#"Becke partition and its nuclear derivative"; crates/ferric-scf/theory.md#"Atomic charges" -->

A molecule has one density and no atoms. Every "atomic" number ferric
exports, a charge or a per-atom polarizability, rests on a rule that splits
the molecular density, or a molecular response, among the nuclei. This
section states the four rules, derives what each conserves, and says which
exported array uses which. Section 21.1 supplies the molecular response;
section 27.2 is the consumer's view of the same charges, with the agreement
figures.

| Rule | Splits | Weight | Used for |
|---|---|---|---|
| Becke | space | geometry-only fuzzy cells $w^A(\mathbf r)$ | `alpha_atomic` and `alpha_ct`, the per-atom $C_6$ with `c6_partition = "becke"` (the default for the TS and MBD routes), the integration grid, `becke_charges` |
| Hirshfeld | space | proatom ratio $\rho^0_A/\sum_B\rho^0_B$ | `hirshfeld_charges`, `alpha_atomic` with `c6_partition = "hirshfeld"` |
| Löwdin | basis functions | symmetric orthogonalization $S^{1/2}$ | `lowdin_charges` |
| ESP fit (CHELPG, RESP) | the potential outside the molecule | none; a least-squares fit | `chelpg_charges`, `resp_charges` |

| Entry point | Location |
|---|---|
| Rust (charges) | `ferric-scf/src/properties.rs::{becke_charges,lowdin_charges,mulliken_charges,chelpg_charges,resp_charges,chelpg_and_resp_charges}`, `ferric-scf/src/properties/hirshfeld_grid.rs::hirshfeld_charges` |
| Rust (weights) | `ferric-dft/src/becke.rs::{becke_weight,becke_weights_all}` |
| Rust (per-atom bras) | `ferric-rpa/src/properties/per_atom.rs::{becke_bra,hirshfeld_bra}` |
| CLI | `compute_hirshfeld_charges`, `compute_lowdin_charges`, `compute_mulliken_charges`, `compute_chelpg_charges`, `compute_resp_charges`, `c6_partition` under `[rpa]` |
| Python | `hirshfeld_charges`, `lowdin_charges`, `mulliken_charges`, `chelpg_charges`, `resp_charges` (`crates/ferric-python/ferric.pyi`) |

## 21.2.1 What a partition must do

Write the electron count of atom $A$ as $N_A$ and its charge as
$q_A = Z_A - N_A$. Two requirements fix the algebra, and both are enforced in
the code rather than assumed.

1. **The parts sum to the whole.** $\sum_A N_A = N_e$, so $\sum_A q_A$ is the
   molecular charge. A space partition gets this from a partition of unity,
   $\sum_A w^A(\mathbf r) = 1$. A basis partition gets it from a trace
   identity. A fit gets it from a constraint.
2. **The parts do not depend on where the molecule sits.** This holds for the
   Becke, Hirshfeld and Löwdin rules by construction. It holds for the
   fits only up to the lattice they sample (section 21.2.5).

The three space and basis rules carry the grid or basis error into the
individual charges only: each is rescaled so that requirement 1 holds exactly
(the code does this for Becke and Hirshfeld; Löwdin and Mulliken satisfy it
without rescaling).

## 21.2.2 The Becke partition

Becke's fuzzy cells are built from the pair coordinates
$\mu_{AB}=(r_A-r_B)/R_{AB}$, a size correction from the Bragg–Slater radii, and
three applications of $f(x)=\tfrac12(3x-x^3)$. The cell function of atom $A$ is
$P_A=\prod_{B\neq A}s_{AB}$ and the weight is $w^A=P_A/\sum_C P_C$. Dividing by
the sum is what makes $\sum_A w^A=1$ exact at every point. The derivation of
the weight and of its nuclear derivative, which the grid-response gradient of
chapter 14 uses, is the following text from `ferric-dft/theory.md`.

<!-- include: crates/ferric-dft/theory.md#"Becke partition and its nuclear derivative" -->

The partition is proved in section 9.2.2, verified equation 9.3: the loop's
assembly of $\partial w^A/\partial\mathbf R_C$ equals sympy's derivative of the
definition, and the derivatives sum to zero. That verified box is the
machine check of the weights. Nothing in it concerns the charge.

**Becke charges.** `becke_charges` evaluates

$$ N_A = \int w^A(\mathbf r)\,\rho(\mathbf r)\,d\mathbf r \approx \sum_{g:\,\mathrm{home}(g)=A} w_g\,\rho(\mathbf r_g), \qquad q_A = Z_A - s\,N_A, \tag{21.5} $$

on the default flat 75 × 110 Becke–Lebedev grid. The weight $w^A$ is already
inside the point weight $w_g = w^{\rm rad}w^\Omega w^A(\mathbf r_g)$ of the
grid built around the home atom $A$ (section 9.2.3), so each point contributes
only to its home atom. The factor $s=N_e/\sum_AN_A$ removes the quadrature
error of the electron count. The Becke partition has no proatom model.

**Becke bra for the polarizability.** The per-atom polarizability of section
27.3 needs the atomic dipole matrix elements
$m^A_{d,ia}=\int w^A(\mathbf r)\,(\mathbf r-\mathbf R_A)_d\,\phi_i\phi_a\,d\mathbf r$,
which `becke_bra` accumulates on the same grid in memory-bounded chunks. The
operator is $(\mathbf r-\mathbf R_A)$, not $\mathbf r$, so that $\alpha^A$ does
not depend on the origin.

## 21.2.3 What the partition of unity buys the polarizability

Because $\sum_Aw^A=1$, the bras of all atoms add up to
$\sum_A m^A_d=\mu_d-\sum_A R_{A,d}\,w_A$, and
$\sum_A\langle i|w_A|a\rangle=0$ for $i\ne a$. Substituting into the linear
response gives the charge-transfer remainder of the polarizability in terms of
the weights alone:

$$ \alpha_{\rm CT} = \alpha_{\rm mol}-\sum_A\operatorname{sym}\alpha^A = \operatorname{sym}\,\Big\langle \sum_A \mathbf R_A\,w_A\Big|R\Big|\mu\Big\rangle . \tag{21.6} $$

The first term is the molecular tensor of section 21.1. The atoms do not add
up to it because a field moves charge between atoms, and an atom-centred
dipole cannot see that. The remainder is exported as `alpha_ct`.

> **Verified equation 21.6** · the charge-transfer remainder follows from the partition of unity
>
> For weights with $\sum_Aw_A=1$ and bras $m^A=\langle i|(\mathbf r-\mathbf R_A)w_A|a\rangle$,
> $\alpha_{\rm mol}-\sum_A\operatorname{sym}\alpha^A=\operatorname{sym}\langle\sum_A\mathbf R_Aw_A|R|\mu\rangle$,
> static and at an imaginary frequency, to $10^{-12}$ (the script's check).
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_c6_polarizability_sympy.py`, claim ct-sum-rule (check 5) |
> | Negative control | none in the runner's mutation list. The check compares two evaluations of one response function, so it tests the algebra of the sum rule, not the kernel |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/properties/per_atom.rs::charge_transfer_remainder` |
>
> <!-- verified: id=part.ct-sum-rule; script=scripts/rpa_c6_polarizability_sympy.py; claim=ct-sum-rule; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/properties/per_atom.rs::charge_transfer_remainder; registry=rpa-polarizability-c6 -->

This is the same proof that section 27.3 cites for the exported array. The
identity holds for any partition of unity, so it holds for the Hirshfeld
weights as well as the Becke ones.

## 21.2.4 Hirshfeld, Löwdin and Mulliken

**Hirshfeld.** Split the density in proportion to free atoms placed at the
nuclei:

$$ w^A(\mathbf r)=\frac{\rho^0_A(\mathbf r-\mathbf R_A)}{\sum_B\rho^0_B(\mathbf r-\mathbf R_B)+\varepsilon},\qquad N_A=\int w^A\rho,\qquad q_A=Z_A-s\,N_A, \tag{21.7} $$

with $\varepsilon=10^{-12}$ guarding the far tail. Without $\varepsilon$ the
weights also sum to one; with it they sum to one within $10^{-12}/\sum_B\rho^0_B$,
which the factor $s$ absorbs. The proatom $\rho^0_A$ is a spherically averaged
free-atom SCF density in the molecule's own basis, tabulated on a radial grid
and interpolated by a cubic spline of $\ln\rho^0$, which keeps it positive and
even at the nucleus. When no such density is supplied, an atom falls back to a
single Slater function $Z\xi^3\pi^{-1}e^{-2\xi r}$, and the code warns.

<!-- excerpt: ferric-scf/src/properties/hirshfeld_grid.rs::hirshfeld_charges -->
```rust
    for a in 0..natoms {
        let mut acc = 0.0;
        for g in 0..npts {
            let w = rho_free[a][g] / (rho_sum[g] + eps_floor);
            acc += rho[g] * w * gw[g];
        }
        n_e_grid[a] = acc;
    }
```

| Symbol | Code |
|---|---|
| $\rho^0_A(\mathbf r_g)$ | `rho_free[a][g]` |
| $\sum_B\rho^0_B(\mathbf r_g)$ | `rho_sum[g]` |
| $\varepsilon$ | `eps_floor` |
| $\rho(\mathbf r_g)$ | `rho[g]` |
| quadrature weight | `gw[g]` |
| $N_A$ | `n_e_grid[a]` |

The grid is the atom-centred Becke–Lebedev one, not a Cartesian lattice. A
Cartesian lattice does not carry the molecule's point group, and
symmetry-equivalent atoms came out with different charges on it; the code
comments record the failure on a set of 500 molecules.

The polarizability bra for `c6_partition = "hirshfeld"` is different. It is
the dipole integral $\int\chi_\mu\chi_\nu\,w^A(\mathbf r-\mathbf R_A)_d$ with
the weights of (21.7), by quadrature on a bounding-box lattice
(`ferric-rpa/src/properties/per_atom.rs::hirshfeld_bra`), and no renormalization to the lab-frame dipole is applied, because that would
break the origin independence. So the Hirshfeld charge and the Hirshfeld
polarizability use two different quadratures of the same weights.

**Löwdin.** Orthogonalize the atomic orbitals symmetrically, $\chi'=S^{-1/2}\chi$.
The density matrix in the new basis is $D'=S^{1/2}DS^{1/2}$, and the
population of an atom is the diagonal of $D'$ over its functions:

$$ q_A=Z_A-\sum_{\mu\in A}\big(S^{1/2}DS^{1/2}\big)_{\mu\mu}, \qquad \sum_\mu D'_{\mu\mu}=\operatorname{Tr}(S^{1/2}DS^{1/2})=\operatorname{Tr}(DS)=N_e . \tag{21.8} $$

The trace identity is why no rescaling is needed. `lowdin_charges` forms
$S^{1/2}$ from the eigendecomposition of $S$ and returns an error if any
eigenvalue is not positive. Mulliken is the same sum with $D S$ in place of
$S^{1/2}DS^{1/2}$: $q_A=Z_A-\sum_{\mu\in A}(DS)_{\mu\mu}$, again with trace
$N_e$. Both are partitions of the basis, not of space, so they change with the
basis set and, for Löwdin, with the function ordering convention of the
basis.

<!-- include: crates/ferric-scf/theory.md#"Atomic charges" -->

> **Verified equation 21.8** · the population sums and the grid rescale
>
> For an idempotent density with $\operatorname{Tr}(DS) = N_e$ on a random overlap,
> the transcribed Mulliken and Löwdin charges sum to $Q$ exactly and the Löwdin
> populations are nonnegative (claim A1). The Becke and Hirshfeld charges of
> (21.5) and (21.7) sum to $Q$ exactly after the rescale $s$, whatever the
> quadrature error and the $\varepsilon$ floor (claim A2).
>
> | | |
> |---|---|
> | Proof | `scripts/atomic_charges_sympy.py`, claims A1 and A2 |
> | Negative control | the runner builds Löwdin with $S$ in place of $S^{1/2}$ (claim A1 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/properties.rs::lowdin_charges` |
>
> <!-- verified: id=charges.population.sums; script=scripts/atomic_charges_sympy.py; claim=A1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/properties.rs::lowdin_charges; registry=atomic-charges -->

## 21.2.5 ESP-fitted charges

CHELPG and RESP do not partition the density. They choose point charges
$q_A$ at the nuclei whose potential reproduces the exact electrostatic
potential $V(\mathbf r_g)$ of the reference density on a set of points
outside the molecule (section 27.2.2 defines $V$). Minimize
$\chi^2(\mathbf q)=\sum_g\big(V_g-\sum_Aq_A/r_{gA}\big)^2$ over the points $g$
with the charge fixed, $\sum_Aq_A=Q$, through a multiplier $\lambda$. The
stationarity conditions are the bordered normal equations

$$ \begin{pmatrix} A & \mathbf 1\\ \mathbf 1^{\mathsf T} & 0\end{pmatrix}\begin{pmatrix}\mathbf q\\ \lambda\end{pmatrix} = \begin{pmatrix}\mathbf b\\ Q\end{pmatrix}, \qquad A_{AB}=\sum_g\frac1{r_{gA}r_{gB}},\quad b_A=\sum_g\frac{V_g}{r_{gA}}, \tag{21.9} $$

where the factor 2 of $\partial\chi^2/\partial q_A$ is absorbed into $\lambda$.
This is the system `solve_chelpg_normal_equations` assembles and solves.

> **Verified equation 21.9** · the CHELPG bordered system
>
> The transcribed bordered normal equations give the same charges as an
> independent null-space solve of the constrained least-squares problem, and
> the charges sum to $Q$ (claim A3).
>
> | | |
> |---|---|
> | Proof | `scripts/atomic_charges_sympy.py`, claim A3 |
> | Negative control | the runner puts 0 in place of $Q$ on the border (claim A3 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/properties.rs::solve_chelpg_normal_equations` |
>
> <!-- verified: id=charges.chelpg.bordered; script=scripts/atomic_charges_sympy.py; claim=A3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/properties.rs::solve_chelpg_normal_equations; registry=atomic-charges -->

<!-- excerpt: ferric-scf/src/properties.rs::solve_chelpg_normal_equations -->
```rust
        for a in 0..natoms {
            rhs[a] += pt.v * inv_r[a];
            for b in 0..=a {
                let contrib = inv_r[a] * inv_r[b];
                mat[(a, b)] += contrib;
                if a != b {
                    mat[(b, a)] += contrib;
                }
            }
        }
```

| Symbol | Code |
|---|---|
| $1/r_{gA}$ | `inv_r[a]` |
| $V_g$ | `pt.v` |
| $b_A$ (running sum) | `rhs[a]` |
| $A_{AB}$ (running sum) | `mat[(a, b)]` |
| $Q$ | `q_total`, from `mol.charge` |

**The points.** A cubic lattice of spacing 0.3 Bohr in the lab frame,
extending 2.8 Bohr beyond the atoms. A point is kept if it lies outside every
atom's Bondi sphere and within 2.8 Bohr of some atom's sphere surface. The
four numbers are read from `FERRIC_CHELPG_SPACING`, `FERRIC_CHELPG_MARGIN`,
`FERRIC_CHELPG_VDW_SCALE` and `FERRIC_CHELPG_OUTER_CUTOFF`; section 27.2.4
lists the defaults and why a dataset must record them.

**RESP** adds a hyperbolic restraint on the non-hydrogen charges,
$a\sum_{A\notin\mathrm H}\big(\sqrt{q_A^2+b^2}-b\big)$ with $a=0.0005$ and
$b=0.1\,e$. Its derivative with respect to $q_A$ is
$a\,q_A/\sqrt{q_A^2+b^2}$, which is linear in $q_A$ with a coefficient that
depends on $q_A$. Freezing the coefficient at the previous iterate gives a
linear system of the form (21.9) with
$A_{AA}\to A_{AA}+a/\sqrt{q_A^2+b^2}$ for non-hydrogen atoms
(`solve_resp_restrained`), solved repeatedly until the largest charge change is
below $10^{-8}$ or 50 iterations pass. The iteration starts from $\mathbf q = 0$,
so its first solve already carries the slope $a/b$, and it returns its last
iterate without an error if it reaches 50 iterations. It is a fixed-point
(iteratively reweighted least-squares) iteration, not Newton's method. Its
fixed point is the minimizer of

$$ \tfrac12\chi^2(\mathbf q) + a\sum_{A\notin\mathrm H}\Big(\sqrt{q_A^2+b^2}-b\Big), \qquad \sum_Aq_A=Q, \tag{21.10} $$

with the $\tfrac12$ on $\chi^2$: the normal equations (21.9) are those of
$\tfrac12\chi^2$, so the restraint enters with weight $a$ relative to that.
The functional without the $\tfrac12$ has a different minimizer, that of the
code with $a/2$.

> **Verified equation 21.10** · the RESP fixed point
>
> The transcribed `solve_resp_restrained` iteration converges to the
> minimizer of (21.10), found independently by a constrained quasi-Newton
> minimization; its projected gradient vanishes within the stopping bound,
> while that of the functional without the $\tfrac12$ does not; and $a\to0$
> gives the CHELPG charges (claim A4).
>
> | | |
> |---|---|
> | Proof | `scripts/atomic_charges_sympy.py`, claim A4 |
> | Negative control | the runner puts $a/(2\sqrt{q^2+b^2})$ on the diagonal (claim A4 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/properties.rs::solve_resp_restrained` |
>
> <!-- verified: id=charges.resp.fixed.point; script=scripts/atomic_charges_sympy.py; claim=A4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/properties.rs::solve_resp_restrained; registry=atomic-charges -->

 It is a single-stage fit: there is no
second stage that equivalences methyl hydrogens, and no multi-conformer
averaging.

The lattice is fixed in the lab frame, so the fitted charges change a little
when the molecule is rotated. The Löwdin, Mulliken, Becke and Hirshfeld
charges do not have that dependence on a lattice, though the first of these is
only invariant for spherical-harmonic shells (section 27.2.4).

## 21.2.6 Which rule for which job

| Need | Use | Why |
|---|---|---|
| A charge that tracks the electron density's own geometry and is stable under a change of basis | Hirshfeld, or Becke | integrals over space of the density, not a property of the basis |
| A cheap baseline with no grid | Löwdin | a trace identity, no quadrature |
| Charges that reproduce the electrostatics outside the molecule | CHELPG or RESP | the only rules fitted to the potential |
| A partition of a molecular response | the same weights as the bra: Becke by default for TS and MBD, Hirshfeld for PDEP | partition of unity gives (21.6) |

The five charge rules are five definitions, not five estimates of one number
(Figure 27.1 shows water). Choose one for a dataset and record it with the
arrays; `c6_partition` is written into the file for the per-atom
polarizability arrays and no equivalent tag exists for the charges.

## 21.2.7 How well these are checked

The figures are in section 27.2.5 and are not repeated here. In summary: the
potential and field at nuclei and the Hirshfeld charges are compared with
numpy and PySCF contractions of a shared density; Mulliken and Löwdin are part
of the all-keys NPZ comparison; the CHELPG and RESP checks are invariants only
(total charge, mirror symmetry of water, reproduction of the fitted surface
ESP) and no external code supplies a reference value for them. The Becke
weights have the machine-checked proof of section 9.2.2, and the partition
sum rule (21.6) the proof above.

**Known limits (2026-11-15).**

- **What the proofs cover.** The verified boxes check the algebra of the
  charge rules on transcriptions of the code: the sum rules, the bordered
  system and the RESP fixed point. They do not check the ESP on the points or
  the Hirshfeld proatoms; those are the comparisons of section 27.2.5.
- **Effective nuclear charges.** The charge rules use $Z^{\mathrm{eff}}_A$
  (2.1): the valence charge on an ECP atom and zero on a ghost (checked
  against PySCF on HI and CH3I, section 27.2.6). Hirshfeld on an ECP atom
  still divides by a proatom built for the full $Z$; that pairing is
  unchecked.
- **Spin.** Every rule takes the total spin-summed density. The Mulliken and
  Löwdin doc comments say closed-shell only, but the formulas have no
  closed-shell assumption.
- **Orientation.** CHELPG and RESP depend on the molecule's orientation
  through the lab-frame lattice.
- **Löwdin ordering.** The Löwdin split depends on the basis-function
  ordering convention, so values from two codes are not comparable even on the
  same density.

## 21.2.8 Sources

| Source | Used for |
|---|---|
| `crates/ferric-dft/theory.md`, "Becke partition and its nuclear derivative" | the Becke weight and its proof |
| `crates/ferric-rpa/theory.md`, "Dispersion: Casimir-Polder C6" | the per-atom bra and equation (21.6) |
| `crates/ferric-scf/theory.md`, "Atomic charges" | the derivation of the charge rules and the ESP fits |
| `scripts/atomic_charges_sympy.py` | the proofs, claims A1 to A4 |
| `crates/ferric-scf/src/properties.rs` | (21.5), (21.8), (21.9), (21.10), the lattice and restraint parameters |
| `crates/ferric-scf/src/properties/hirshfeld_grid.rs` | (21.7), the proatom and the grid |
| `crates/ferric-rpa/src/properties/per_atom.rs` | the Becke and Hirshfeld bras |
| `site/src/reference/validation.md`, "Anchors" | the agreement figures, through section 27.2.5 |
