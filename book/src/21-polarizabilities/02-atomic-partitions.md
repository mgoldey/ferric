# Atomic partitions: Becke, Hirshfeld, Löwdin, ESP fits {#atomic-partitions}

<!-- status: edit; registry: atomic-charges -->

A molecule has one electron density and no atoms. Chemists still want
per-atom numbers: a partial charge on the carbonyl oxygen, a polarizability
for each atom that a force field or a machine-learned model can transfer to
the next molecule. Every such number ferric exports, a charge or a per-atom
polarizability, rests on a rule that splits the molecular density, or a
molecular response, among the nuclei. Different rules give different
numbers for the same density, and none of them is the true atomic charge,
because there is none.

This section states four rules, derives what each conserves, and says which
exported array uses which. Under them sit six charge models in Rust (Becke,
Hirshfeld, Löwdin, Mulliken, CHELPG and RESP); five of them are exported to
Python, and Becke charges are Rust only. The polarizability split of @sec:atomic-partitions-partition-unity-buys-polarizability is where the
choice of rule meets the response of section @sec:polarizabilities-from-dielectric-modes. Section @sec:electrostatics-and-charges is the
consumer's view of the same charges, with the agreement figures.

| Rule | Splits | Weight | Used for |
|---|---|---|---|
| Becke | space | geometry-only fuzzy cells $w^A(\mathbf r)$ | `alpha_atomic` and `alpha_ct`, the per-atom $C_6$ with `c6_partition = "becke"` (the default for the TS and MBD routes), the integration grid, `becke_charges` |
| Hirshfeld | space | proatom ratio $\rho^0_A/\sum_B\rho^0_B$ | `hirshfeld_charges`; `alpha_atomic_dynamic`, `alpha_ct_dynamic` and the per-atom $C_6$ with `c6_partition = "hirshfeld"` (the default for `c6_source = "pdep"`); `hirshfeld_polarizability` |
| Löwdin | basis functions | symmetric orthogonalization $S^{1/2}$ | `lowdin_charges` |
| ESP fit (CHELPG, RESP) | the potential outside the molecule | none; a least-squares fit | `chelpg_charges`, `resp_charges` |

| Entry point | Location |
|---|---|
| Rust (charges) | `ferric-scf/src/properties.rs::{becke_charges,lowdin_charges,mulliken_charges,chelpg_charges,resp_charges,chelpg_and_resp_charges}`, `ferric-scf/src/properties/hirshfeld_grid.rs::hirshfeld_charges` |
| Rust (weights) | `ferric-dft/src/becke.rs::{becke_weight,becke_weights_all}` |
| Rust (per-atom bras) | `ferric-rpa/src/properties/per_atom.rs::{becke_bra,hirshfeld_bra}` |
| CLI | `compute_hirshfeld_charges`, `compute_lowdin_charges`, `compute_mulliken_charges`, `compute_chelpg_charges`, `compute_resp_charges`, `c6_partition` under `[rpa]` |
| Python | `hirshfeld_charges`, `lowdin_charges`, `mulliken_charges`, `chelpg_charges`, `resp_charges` (`crates/ferric-python/python/ferric/ferric.pyi`) |

## What a partition must do {#atomic-partitions-partition-must-do}

`properties.rs` and `properties/hirshfeld_grid.rs` compute six charge
models: Becke, Hirshfeld, Löwdin, Mulliken, CHELPG and RESP. Write the
electron count of atom $A$ as $N_A$ and its charge as
$q_A = Z^{\rm eff}_A - N_A$. Here $Z^{\rm eff}_A$ is `Atom::effective_z`,
the valence charge on an ECP atom (equation @eq:molecule-1). The density is always the
total spin-summed one, $\mathbf P$. The electron count
$N = \sum_AZ^{\rm eff}_A - Q$, with $Q$ the molecular charge, is what
`Molecule::nelec` returns.

Two requirements fix the algebra, and the code enforces both.

1. **The parts sum to the whole.** $\sum_A N_A = N$, so $\sum_A q_A = Q$. A
   space partition gets this from a partition of unity,
   $\sum_A w^A(\mathbf r) = 1$. A basis partition gets it from a trace
   identity. A fit gets it from a constraint.
2. **The parts do not depend on where the molecule sits.** This holds for the
   Becke, Hirshfeld and Löwdin rules by construction. It holds for the fits
   only up to the lattice they sample (section @sec:atomic-partitions-esp-fitted-charges).

A space partition integrated on a grid satisfies requirement 1 only up to the
quadrature error. The code rescales the Becke and Hirshfeld populations so
that it holds exactly; the grid error then shows up only in the individual
charges. Löwdin and Mulliken satisfy it without rescaling.

## The Becke partition {#atomic-partitions-becke-partition}

Becke's partition gives every point of space a set of weights, one per atom,
that sum to one. Section @sec:grids-becke-partition builds it in four steps, and
`ferric-dft/src/becke.rs::becke_weights_and_grad` codes it. With
$r_X = |\mathbf r-\mathbf R_X|$ and $R_{AB} = |\mathbf R_A-\mathbf R_B|$:

$$ \mu_{AB} = \frac{r_A-r_B}{R_{AB}},\qquad \nu_{AB} = \mu_{AB} + a_{AB}(1-\mu_{AB}^2),\qquad s_{AB} = \tfrac12\bigl(1 - f^{(3)}(\nu_{AB})\bigr),\quad f(x)=\tfrac12(3x-x^3), $$

$$ a_{AB} = \operatorname{clip}_{[-1/2,\,1/2]}\frac{u}{u^2-1},\quad u=\frac{\chi-1}{\chi+1},\ \chi = R^{BS}_A/R^{BS}_B,\qquad P_A = \prod_{B\ne A}s_{AB},\qquad w^A = P_A/T,\ T=\sum_C P_C . $$

$\mu_{AB}$ is an elliptical coordinate, $-1$ at nucleus $A$ and $+1$ at $B$.
The size correction $a_{AB}$ shifts the dividing surface by the ratio of the
Bragg–Slater radii, so unequal atoms do not split at the midpoint. Three applications of $f$ sharpen the
step into the cell function $s_{AB}$. Dividing $P_A$ by the sum $T$ is what
makes $\sum_A w^A = 1$ exact at every point. At a nucleus, $w^A(\mathbf R_A) = 1$.

Gradients need the nuclear derivatives of $w^A$, because moving a nucleus
moves the cells. Section @sec:grids-becke-partition derives them, and
verified equation @eq:ks-grid-becke records their proof
(`scripts/dft_becke_weight_grad_sympy.py`). A charge needs only the weights.

**Becke charges.** `becke_charges` integrates the density inside each cell:

$$ N_A = \int w^A(\mathbf r)\,\rho(\mathbf r)\,d\mathbf r \approx \sum_{g:\,\mathrm{home}(g)=A} w_g\,\rho(\mathbf r_g), \qquad q_A = Z^{\rm eff}_A - s\,N_A, \qquad s = \frac{N}{\sum_BN_B}, \label{eq:atomic-partitions-1} $$

with $\rho_g = \sum_{\mu\nu}P_{\mu\nu}\chi_\mu(\mathbf r_g)\chi_\nu(\mathbf r_g)$,
on the default flat 75 × 110 Becke–Lebedev grid (`build_atomic_grid` with
the default configuration). Why is the sum only over the home atom's points?
The weight $w^A$ is already inside the point weight
$w_g = w^{\rm rad}w^\Omega w^A(\mathbf r_g)$ of the grid built around $A$
(section @sec:grids-building-molecular-grid), so each point carries exactly its home atom's share. The
rescale $s$ removes the quadrature error of the electron count; the doc
comment puts that error at about 0.003 e on water. The Becke partition has no
proatom model: the cells depend on the geometry and the Bragg–Slater radii
alone.

**Becke bra for the polarizability.** The per-atom polarizability of @sec:atomic-partitions-partition-unity-buys-polarizability
needs the atomic dipole matrix elements

$$ m^A_{d,ia}=\int w^A(\mathbf r)\,(\mathbf r-\mathbf R_A)_d\,\phi_i(\mathbf r)\phi_a(\mathbf r)\,d\mathbf r , $$

which `ferric-rpa/src/properties/per_atom.rs::becke_bra` accumulates in the
AO basis on the same grid, in memory-bounded chunks, symmetrizes, and
transforms to the occupied–virtual block. The operator is
$(\mathbf r-\mathbf R_A)$, not $\mathbf r$, so that $\alpha^A$ does not depend
on the origin.

## What the partition of unity buys the polarizability {#atomic-partitions-partition-unity-buys-polarizability}

What should "the polarizability of atom $A$" mean? The field acts on the
whole molecule, so the perturbation is still the molecular dipole. What
belongs to atom $A$ is the part of the induced dipole that sits in $A$'s
cell, measured from $A$'s nucleus. Krishtal, Senet and Van Alsenoy (J. Chem.
Phys. 125, 034312 (2006)) define the intrinsic atomic polarizability that
way. Write the response of section @sec:polarizabilities-from-dielectric-modes as an operator $R(iu)$, so that
$\alpha_{\rm mol} = \langle\mu|R|\mu\rangle$. Then

$$ \alpha^A_{dj}(iu) = \langle m^A_d|R(iu)|\mu_j\rangle , $$

with the atom-centred dipole $m^A_d = w^A(\mathbf r)(r-R_A)_d$ on the bra and
the molecular dipole on the field-side ket. In the auxiliary basis this is
(@eq:pol-kernel-vs-casida) with the bra $\mu_d$ replaced by $m^A_d$:

$$ \alpha^A_{dj}(iu) = 4\,m^A_d\cdot(g\circ\mu_j) - 16\,\big(B(g\circ m^A_d)\big)^{\mathsf T}\tilde{\boldsymbol\varepsilon}^{-1}\big(B(g\circ\mu_j)\big). $$

The dielectric and the solve against the molecular dipole are shared by all
atoms; only the bra changes. The loop over atoms in
`ferric-rpa/src/properties/per_atom.rs::closed_per_atom_alpha` is:

<!-- excerpt: ferric-rpa/src/properties/per_atom.rs::closed_per_atom_alpha -->
```rust
    Ok(bra
        .iter()
        .map(|atom| {
            let mut tensor = [[0.0_f64; 3]; 3];
            for d in 0..3 {
                let w_ai = b_ov.dot(&(&atom[d] * &g));
                for j in 0..3 {
                    let bare = atom[d].dot(&mu_g[j]);
                    let coupled = w_ai.dot(&y_mol[j]);
                    tensor[d][j] = 4.0 * bare - 16.0 * coupled;
                }
            }
            symmetrize_tensor(&mut tensor);
            tensor
        })
        .collect())
```

| Symbol | Variable |
|---|---|
| $m^A_d$ | `atom[d]`, the bra of one atom |
| $g\circ\mu_j$ | `mu_g[j]` |
| $B(g\circ m^A_d)$ | `w_ai` |
| $\tilde{\boldsymbol\varepsilon}^{-1}B(g\circ\mu_j)$ | `y_mol[j]`, solved once before the atom loop |
| $\tfrac12(\alpha^A+\alpha^{A\mathsf T})$ | `symmetrize_tensor` |

The same function serves the static drivers, with $g = 1/\Delta$, and the
dynamic ones, with $g_{ia}(u)$. It also serves both partitions: the Becke
bra of @sec:atomic-partitions-becke-partition and the Hirshfeld bra of @sec:atomic-partitions-hirshfeld-l-wdin-mulliken.

**The sum over atoms.** Add the bras. Because $\sum_Aw^A = 1$,

$$ \sum_A m^A_d = \sum_A w^A(r_d - R_{A,d}) = r_d - \sum_A R_{A,d}\,w^A , $$

and in the occupied–virtual block $r_d$ is $\mu_d$. The response is linear in
the bra, so $\sum_A\alpha^A = \alpha_{\rm mol} - \langle\sum_AR_Aw^A|R|\mu\rangle$.
After symmetrization this gives the charge-transfer remainder in terms of the
weights alone:

$$ \alpha_{\rm CT} = \alpha_{\rm mol}-\sum_A\operatorname{sym}\alpha^A = \operatorname{sym}\,\Big\langle \sum_A \mathbf R_A\,w^A\Big|R\Big|\mu\Big\rangle . \label{eq:part-ct-sum-rule} $$

The first term is the molecular tensor of section @sec:polarizabilities-from-dielectric-modes. The atoms do not add
up to it because a field moves charge between atoms, and an atom-centred
dipole cannot see that. $\langle w^A|R|\mu\rangle$ is the electron population
that the field moves into or out of cell $A$, so the right side of (@eq:part-ct-sum-rule) is
the dipole of those population changes placed at the nuclei.

Check the limits. For one atom, $w^A = 1$ everywhere and
$\langle i|1|a\rangle = 0$, so $\alpha_{\rm CT} = 0$ and $\alpha^A = \alpha_{\rm mol}$.
For any molecule, shifting the origin by $\mathbf c$ shifts every
$\mathbf R_A$ by $\mathbf c$ and adds $\mathbf c\sum_A\langle i|w^A|a\rangle = \mathbf c\,\langle i|a\rangle = 0$,
so $\alpha_{\rm CT}$ does not depend on the origin either. For H2 along $z$,
charge can flow only along the bond: the test
`becke_dynamic_at_zero_frequency_is_the_static_per_atom_alpha` asserts that
$\alpha_{{\rm CT},zz}$ exceeds 5% of $\alpha_{zz}$ and that
$\alpha_{{\rm CT},xx}$ is below $10^{-3}\alpha_{xx}$. For water and N2 at
aug-cc-pVDZ the isotropic $\alpha_{\rm CT}(0)$ is 27.0% and 22.7% of
$\alpha_{\rm iso}$ (section @sec:polarizability-pdep-c6-per-atom-split-charge).

The code computes the remainder as the difference, accumulated in atom order
from 0.0:

<!-- excerpt: ferric-rpa/src/properties/per_atom.rs::charge_transfer_remainder -->
```rust
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let sum: f64 = per_atom.iter().map(|t| t[i][j]).sum();
            molecular[i][j] - sum
        })
    })
```

So `alpha_ct + alpha_atomic.sum(0) == alpha_tensor` holds by construction,
up to rounding. It means something only when the molecular and per-atom
tensors come from matching paths: `pdep_polarizability_static` with
`pdep_polarizability_becke`, or `molecular_dynamic_polarizability` with
`pdep_polarizability_becke_dynamic` or `_hirshfeld_dynamic` at the same
frequencies. `charge_transfer_remainder_dynamic` applies it frequency by
frequency and returns an error if an atom's frequency count differs from the
molecule's.

> **Verified equation @eq:part-ct-sum-rule** · the charge-transfer remainder follows from the partition of unity
>
> For weights with $\sum_Aw^A=1$ and bras $m^A=\langle i|(\mathbf r-\mathbf R_A)w^A|a\rangle$,
> $\alpha_{\rm mol}-\sum_A\operatorname{sym}\alpha^A=\operatorname{sym}\langle\sum_A\mathbf R_Aw^A|R|\mu\rangle$,
> static and at an imaginary frequency, to $10^{-12}$ (the script's check).
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_c6_polarizability_sympy.py`, claim ct-sum-rule (check 5) |
> | Negative control | the runner leaves the first atom out of $\sum_A\alpha^A$. The check compares two evaluations of one response function, so it tests the algebra of the sum rule, not the kernel |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/properties/per_atom.rs::charge_transfer_remainder` |
>
> <!-- verified: id=part.ct-sum-rule; script=scripts/rpa_c6_polarizability_sympy.py; claim=ct-sum-rule; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/properties/per_atom.rs::charge_transfer_remainder; registry=rpa-polarizability-c6 -->

Section @sec:polarizability-pdep-c6 cites the same proof for the exported array. The
identity holds for any partition of unity, so it holds for the Hirshfeld
weights as well as the Becke ones.

**Static and dynamic agree at $u = 0$.** The static per-atom driver uses
$g = 1/\Delta$; the dynamic one, called at $u = 0$, uses $\Delta/\Delta^2$.
The two are one definition evaluated two ways, and two tests hold them
together: `becke_dynamic_at_zero_frequency_is_the_static_per_atom_alpha`
(H2, relative difference below $10^{-11}$, plus the sum rule
$\sum_A\alpha^A + \alpha_{\rm CT} = \alpha_{\rm mol}$ to $10^{-13}\alpha_{zz}$) and
`bundle_alpha_atomic_is_the_static_becke_and_dynamic_at_zero_matches_it` in
`crates/ferric-rpa/tests/property_bundle.rs` (water, absolute difference
below $10^{-10}$, and the exported `alpha_atomic` equal bit for bit to the
static Becke call).

## Hirshfeld, Löwdin and Mulliken {#atomic-partitions-hirshfeld-l-wdin-mulliken}

**Hirshfeld.** Place a free, spherical atom at each nucleus, and give each
point of the molecule's density to the atoms in proportion to how much
density each free atom would have there:

$$ w^A(\mathbf r)=\frac{\rho^0_A(\mathbf r-\mathbf R_A)}{\sum_B\rho^0_B(\mathbf r-\mathbf R_B)+\varepsilon},\qquad N_A=\int w^A\rho,\qquad q_A=Z^{\rm eff}_A-s\,N_A, \label{eq:atomic-partitions-3} $$

with $\varepsilon=10^{-12}$ guarding the far tail, where every proatom
density underflows. Without $\varepsilon$ the weights sum to one. With it
they sum to $1-\varepsilon/(\sum_B\rho^0_B+\varepsilon)$, slightly less than
one where every proatom density is tiny, and the rescale $s = N/\sum_BN_B$
absorbs that deficit together with the quadrature error.

The proatom $\rho^0_A$ comes from the proatom provider: a spherically averaged
free-atom SCF density in the molecule's own basis, tabulated on a radial grid
and interpolated by a cubic spline of $\ln\rho^0$. The logarithm keeps the
interpolant positive in the exponential tail, and mirroring the table about
$r = 0$ keeps it even at the nucleus. When no such density is supplied, an
atom falls back to a single Slater function $Z_A\xi^3\pi^{-1}e^{-2\xi r}$
(`slater_xi_for_z`), and the code warns. Both proatoms are built for the full
nuclear charge $Z_A$. The Python `hirshfeld_charges` takes `proatom="scf"`
(the default, free-atom SCFs solved with the same SCF settings as the
molecule) or `proatom="slater"`, and the CLI's `compute_hirshfeld_charges`
uses the SCF proatoms.

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

The loop sums over every grid point, not only atom $A$'s. The point weight
`gw[g]` already contains the Becke share of the point's home atom, so
summing it over the whole grid integrates $\rho$ once; the Hirshfeld weight
then redistributes each point's share between atoms. In formulas,

$$ N^{\rm Hirsh}_A = \sum_g w_g\rho_g\,\frac{\rho^0_A(g)}{\sum_B\rho^0_B(g) + 10^{-12}} . $$

The grid is the atom-centred Becke–Lebedev one (default configuration), not a
Cartesian lattice. A Cartesian lattice does not carry the molecule's point
group and cannot resolve the nuclear cusp. It gives symmetry-equivalent atoms
different charges: more than 0.05 e in 52 of 500 QM9 molecules, per the code
comment.

The polarizability bra for `c6_partition = "hirshfeld"` is different. It is
the dipole integral $\int\chi_\mu\chi_\nu\,w^A(\mathbf r-\mathbf R_A)_d$ with
the weights of (@eq:atomic-partitions-3), by quadrature on a bounding-box lattice of 0.20 Bohr
spacing and 6 Bohr margin (`FERRIC_HIRSHFELD_SPACING`,
`FERRIC_HIRSHFELD_MARGIN`; `ferric-rpa/src/properties/per_atom.rs::hirshfeld_bra`).
No renormalization to the lab-frame dipole is applied, because that would
break the origin independence. So the Hirshfeld charge and the Hirshfeld
polarizability use two different quadratures of the same weights. The cusp
objection does not apply to the bra in the same way: its integrand carries
$(\mathbf r-\mathbf R_A)$, which vanishes at nucleus $A$, and $w^A$ is close to
zero at every other nucleus, so the sharp part of each $\chi_\mu\chi_\nu$ sits
where the operator is small. On water/cc-pVDZ (RHF, cc-pVDZ-RI, build `3f612e0b3`)
the oxygen's isotropic $\alpha$ is 2.340961, 2.341371, 2.341391 and 2.341412
bohr$^3$ at spacings 0.30, 0.20, 0.14 and 0.10 Bohr, and each hydrogen's
stays at 0.115181 from 0.20 down. At the default 0.20 Bohr the per-atom
values are within $2\times10^{-5}$ relative of the finest lattice.

**Löwdin.** The overlap matrix $S$ mixes basis functions on different atoms,
so the density matrix alone does not say which electrons belong where.
Löwdin removes the overlap symmetrically, treating every function alike:
$\chi'=S^{-1/2}\chi$ is orthonormal. The density matrix in that basis is
$D'=S^{1/2}\mathbf PS^{1/2}$, and the population of an atom is the diagonal of
$D'$ over its functions:

$$ q_A=Z^{\rm eff}_A-\sum_{\mu\in A}\big(S^{1/2}\mathbf PS^{1/2}\big)_{\mu\mu}, \qquad \sum_\mu D'_{\mu\mu}=\operatorname{Tr}(S^{1/2}\mathbf PS^{1/2})=\operatorname{Tr}(\mathbf PS)=N . \label{eq:charges-population-sums} $$

The middle step is the cyclic property of the trace. The charges therefore
sum to $\sum_AZ^{\rm eff}_A - \operatorname{Tr}[\mathbf P S]$, which is $Q$ for
an SCF density, and no rescaling is needed. $D'$ is positive semidefinite
because $\mathbf P$ is, so every Löwdin population is nonnegative.
`ferric-scf/src/properties.rs::lowdin_charges` forms $S^{1/2}$ from the
eigendecomposition of $S$ and returns an error if any eigenvalue is not
positive.

**Mulliken** is the same sum with $\mathbf PS$ in place of
$S^{1/2}\mathbf PS^{1/2}$: $q_A=Z^{\rm eff}_A-\sum_{\mu\in A}(\mathbf PS)_{\mu\mu}$,
again with trace $N$. Löwdin and Mulliken partition the basis, not space, so
they change with the basis set. The Löwdin split also changes under any
non-orthogonal change of the functions on an atom: Cartesian against
spherical $d$ shells, or a different normalization. It does not depend on the
order of the functions: a permutation $\Pi$ gives
$S'^{1/2} = \Pi S^{1/2}\Pi^{\mathsf T}$, which only permutes the diagonal.

> **Verified equation @eq:charges-population-sums** · the population sums and the grid rescale
>
> For an idempotent density with $\operatorname{Tr}(DS) = N_e$ on a random overlap,
> the transcribed Mulliken and Löwdin charges sum to $Q$ exactly and the Löwdin
> populations are nonnegative (claim A1). The Becke and Hirshfeld charges of
> (@eq:atomic-partitions-1) and (@eq:atomic-partitions-3) sum to $Q$ exactly after the rescale $s$, whatever the
> quadrature error and the $\varepsilon$ floor (claim A2).
>
> | | |
> |---|---|
> | Proof | `scripts/atomic_charges_sympy.py`, claims A1 and A2 |
> | Negative control | the runner builds Löwdin with $S$ in place of $S^{1/2}$ (claim A1 fails), and drops the rescale $s$ (claim A2 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/properties.rs::lowdin_charges` |
>
> <!-- verified: id=charges.population.sums; script=scripts/atomic_charges_sympy.py; claim=A1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/properties.rs::lowdin_charges; registry=atomic-charges -->

## ESP-fitted charges {#atomic-partitions-esp-fitted-charges}

CHELPG and RESP do not partition the density. They answer a different
question: which point charges at the nuclei reproduce the electrostatic
potential that another molecule would feel outside this one? That is what a
classical force field needs from its charges.

The potential of the reference density at a point $\mathbf r_g$ is

$$ V_g = \sum_BZ^{\rm eff}_B/|\mathbf{r}_g-\mathbf{R}_B| - \sum_{\mu\nu}P_{\mu\nu}\langle\mu|1/|\mathbf{r}-\mathbf{r}_g||\nu\rangle $$

(section @sec:electrostatics-and-charges-potential-arbitrary-points-surface). Choose the charges to minimize
$\tfrac12\chi^2(\mathbf q)=\tfrac12\sum_g\big(V_g-\sum_Aq_A/r_{gA}\big)^2$ over the
points $g$, with the total fixed, $\sum_Aq_A=Q$, through a multiplier
$\lambda$. Differentiate $\tfrac12\chi^2 + \lambda(\sum_Aq_A - Q)$ with respect
to $q_A$ and set it to zero:
$\sum_B\big(\sum_g 1/(r_{gA}r_{gB})\big)q_B + \lambda = \sum_g V_g/r_{gA}$.
Together with the constraint, these are the bordered normal equations

$$ \begin{pmatrix} A & \mathbf 1\\ \mathbf 1^{\mathsf T} & 0\end{pmatrix}\begin{pmatrix}\mathbf q\\ \lambda\end{pmatrix} = \begin{pmatrix}\mathbf b\\ Q\end{pmatrix}, \qquad A_{AB}=\sum_g\frac1{r_{gA}r_{gB}},\quad b_A=\sum_g\frac{V_g}{r_{gA}} . \label{eq:charges-chelpg-bordered} $$

`ferric-scf/src/properties.rs::solve_chelpg_normal_equations` assembles
this system and solves it once by LU. The points carry no weights.

> **Verified equation @eq:charges-chelpg-bordered** · the CHELPG bordered system
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

The assembly is one pass over the points, reusing each point's inverse
distances for both $A$ and $\mathbf b$:

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

After the loop, row and column `natoms` are set to ones and the right-hand
side's last entry to `q_total`: that is the border of (@eq:charges-chelpg-bordered).

**The points.** A cubic lattice of spacing 0.3 Bohr in the lab frame,
extending 2.8 Bohr beyond the atoms. `chelpg_in_shell` keeps a point if it
lies outside every atom's Bondi sphere, scaled by `vdw_scale`, and within
`outer_cutoff` = 2.8 Bohr of some scaled sphere's surface. Points inside the
molecule are dropped because there the potential is not what a neighbour
feels, and far points because there every charge set looks like the total
charge. The four numbers are read from `FERRIC_CHELPG_SPACING`,
`FERRIC_CHELPG_MARGIN`, `FERRIC_CHELPG_VDW_SCALE` (default 1.0) and
`FERRIC_CHELPG_OUTER_CUTOFF`; section @sec:electrostatics-and-charges-five-charge-schemes says why a
dataset must record them.

**RESP.** The points see a buried atom mostly through its neighbours, so
its fitted CHELPG charge is poorly determined. RESP adds a hyperbolic restraint toward zero on every
atom with $Z \ne 1$:
$a\sum_{A\notin\mathrm H}\big(\sqrt{q_A^2+b^2}-b\big)$, with $a=0.0005$ and
$b=0.1\,e$ (`FERRIC_RESP_RESTRAINT_WEIGHT`, `FERRIC_RESP_RESTRAINT_B`). Its
derivative with respect to $q_A$ is $a\,q_A/\sqrt{q_A^2+b^2}$. Write that as
$[a/\sqrt{q_A^2+b^2}]\,q_A$ and freeze the bracket at the previous iterate.
Then each step is a linear system of the form (@eq:charges-chelpg-bordered) with a heavier diagonal,
which `solve_resp_restrained` iterates:

$$ \Big(\mathbf{A} + \mathrm{diag}_{A\,\in\,\rm heavy}\frac{a}{\sqrt{(q^{(k)}_A)^2 + b^2}}\Big)\mathbf{q}^{(k+1)} + \lambda\mathbf{1} = \mathbf{b},\qquad \sum_Aq^{(k+1)}_A = Q . $$

The iteration starts from $\mathbf q^{(0)} = 0$, so its first solve already
carries the finite slope $a/b$; a bare $|q|$ restraint would be singular
there. It stops when $\max_A|q^{(k+1)}_A - q^{(k)}_A| < 10^{-8}$, and if that
has not happened after 50 iterations it returns an error, not the last
iterate. It is a fixed-point (iteratively reweighted least-squares) iteration,
not Newton's method. At its fixed point,
$(\mathbf{A}\mathbf{q} - \mathbf{b})_A + a\,q_A/\sqrt{q_A^2+b^2} + \lambda = 0$,
which is the stationarity condition of

$$ \tfrac12\chi^2(\mathbf q) + a\sum_{A\notin\mathrm H}\Big(\sqrt{q_A^2+b^2}-b\Big), \qquad \sum_Aq_A=Q . \label{eq:charges-resp-fixed-point} $$

The $\tfrac12$ on $\chi^2$ sets the meaning of $a$: the normal equations
(@eq:charges-chelpg-bordered) are those of $\tfrac12\chi^2$, so the restraint enters with weight $a$
relative to that. The functional without the $\tfrac12$ has a different
minimizer, that of the code with $a/2$. As $a\to0$ the restraint vanishes and
RESP becomes CHELPG.

> **Verified equation @eq:charges-resp-fixed-point** · the RESP fixed point
>
> The transcribed `solve_resp_restrained` iteration converges to the
> minimizer of (@eq:charges-resp-fixed-point), found independently by a constrained quasi-Newton
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

It is a single-stage fit: there is no second stage that equivalences methyl
hydrogens, and no multi-conformer averaging. `chelpg_and_resp_charges`
evaluates the potential on the points once and runs both solves, since the
two fits differ only in the solve.

The lattice is fixed in the lab frame, so the fitted charges change a little
when the molecule is rotated. The Löwdin, Mulliken, Becke and Hirshfeld
charges do not depend on a lattice, though Löwdin is invariant only for
spherical-harmonic shells (section @sec:electrostatics-and-charges-five-charge-schemes).

## Which rule for which job {#atomic-partitions-rule-job}

| Need | Use | Why |
|---|---|---|
| A charge that tracks the electron density's own geometry and is stable under a change of basis | Hirshfeld, or Becke | integrals over space of the density, not a property of the basis |
| A cheap baseline with no grid | Löwdin | a trace identity, no quadrature |
| Charges that reproduce the electrostatics outside the molecule | CHELPG or RESP | the only rules fitted to the potential |
| A partition of a molecular response | the same weights as the bra: Becke by default for TS and MBD, Hirshfeld for PDEP | partition of unity gives (@eq:part-ct-sum-rule) |

The five charge models reachable from Python are five definitions of an
atomic charge, and they disagree by design (the figure in section
@sec:electrostatics-and-charges-five-charge-schemes shows water). Choose one for a dataset and
record it with the arrays. `c6_partition` tags the dispersion block
(`alpha_atomic_dynamic`, `alpha_ct_dynamic`, `c6_iso`, `c6_aniso`). The static
`alpha_atomic` and `alpha_ct` are always Becke and carry no tag, and no
equivalent tag exists for the charges.

Both halves on one water / cc-pVDZ density, from the `589b50832` build. The
first block is the Becke split of $\alpha$ and its charge-transfer remainder
(@eq:part-ct-sum-rule), here 37.0% of $\alpha_{\rm iso}$ at cc-pVDZ; the residual of
the sum rule prints as 0.0. The second block is the five Python charge models: each sums
to zero within $10^{-14}$, and the oxygen charge ranges from $-0.31$
(Mulliken) to $-0.74$ (CHELPG).

```python
import numpy as np
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

r = ferric.run_pdep_rpa(mol, bs, aux, properties=True)
alpha = np.asarray(r.alpha_tensor)
atomic = np.asarray(r.alpha_atomic)  # (N, 3, 3), Becke
ct = np.asarray(r.alpha_ct)
print("alpha_iso       :", np.trace(alpha) / 3)
for sym, a in zip(mol.symbols(), atomic):
    print(f"alpha_atomic {sym:2s}: iso {np.trace(a) / 3:.6f}")
print("alpha_ct iso    :", np.trace(ct) / 3, f"({100 * np.trace(ct) / np.trace(alpha):.1f}% of alpha)")
print("max |sum_A alpha^A + alpha_ct - alpha|:", np.abs(atomic.sum(0) + ct - alpha).max())

rhf = ferric.run_rhf(mol, bs)
rules = {
    "hirshfeld": ferric.hirshfeld_charges(mol, bs, rhf),
    "lowdin": ferric.lowdin_charges(mol, bs, rhf),
    "mulliken": ferric.mulliken_charges(mol, bs, rhf),
    "chelpg": ferric.chelpg_charges(mol, bs, rhf),
    "resp": ferric.resp_charges(mol, bs, rhf),
}
print("rule       " + "  ".join(f"{s:>9s}" for s in mol.symbols()) + "       sum")
for name, q in rules.items():
    q = np.asarray(q)
    print(f"{name:9s}  " + "  ".join(f"{v:9.5f}" for v in q) + f"  {q.sum():9.2e}")
```

```text
alpha_iso       : 3.110850466940373
alpha_atomic O : iso 1.185806
alpha_atomic H : iso 0.386672
alpha_atomic H : iso 0.386672
alpha_ct iso    : 1.1517009421877453 (37.0% of alpha)
max |sum_A alpha^A + alpha_ct - alpha|: 0.0
rule               O          H          H       sum
hirshfeld   -0.32158    0.16079    0.16079   5.55e-16
lowdin      -0.48085    0.24042    0.24042   9.66e-15
mulliken    -0.30539    0.15269    0.15269   1.55e-15
chelpg      -0.73999    0.37000    0.37000   0.00e+00
resp        -0.73997    0.36998    0.36998   0.00e+00
```

## How well these are checked {#atomic-partitions-well-these-checked}

The figures are in section @sec:electrostatics-and-charges-well-these-checked. In summary: the potential and field at
nuclei and the Hirshfeld charges are compared with numpy and PySCF
contractions of a shared density; Mulliken and Löwdin are part of the
all-keys NPZ comparison; the CHELPG and RESP checks are invariants only
(total charge, mirror symmetry of water, reproduction of the fitted surface
ESP), and no external code supplies a reference value for them. The Becke
weights have the machine-checked proof of section @sec:grids-becke-partition, and the partition
sum rule (@eq:part-ct-sum-rule) the proof above.

**Known limits (2026-11-15).**

- **What the proofs cover.** The verified boxes check the algebra of the
  charge rules on transcriptions of the code: the sum rules, the bordered
  system and the RESP fixed point. They do not check the ESP on the points or
  the Hirshfeld proatoms; those are the comparisons of section @sec:electrostatics-and-charges-well-these-checked.
- **Effective nuclear charges.** The charge rules use $Z^{\mathrm{eff}}_A$
  (@eq:molecule-1): the valence charge on an ECP atom and zero on a ghost (checked
  against PySCF on HI and CH3I, section @sec:electrostatics-and-charges-known-limits-2026-11). Hirshfeld on an ECP atom
  still divides by a proatom built for the full $Z$; that pairing is
  unchecked.
- **Spin.** Every rule takes the total spin-summed density; none assumes a
  closed shell.
- **Orientation.** CHELPG and RESP depend on the molecule's orientation
  through the lab-frame lattice.
- **Löwdin and Cartesian shells.** The Löwdin split is invariant to the order
  of the functions and to orthogonal mixing on one atom. It differs between
  codes only when the basis differs (Cartesian against spherical shells,
  normalization). On a pure-function basis ferric matches PySCF
  (`crates/ferric-scf/tests/properties_effective_z.rs`).

## Sources {#atomic-partitions-sources}

| Source | Used for |
|---|---|
| section @sec:grids-becke-partition | the Becke weight and its proof |
| section @sec:polarizabilities-from-dielectric-modes-dielectric-form | the per-atom bra and equation (@eq:part-ct-sum-rule) |
| `scripts/atomic_charges_sympy.py` | the proofs, claims A1 to A4 |
| `crates/ferric-scf/src/properties.rs` | (@eq:atomic-partitions-1), (@eq:charges-population-sums), (@eq:charges-chelpg-bordered), (@eq:charges-resp-fixed-point), the lattice and restraint parameters |
| `crates/ferric-scf/src/properties/hirshfeld_grid.rs` | (@eq:atomic-partitions-3), the proatom and the grid |
| `crates/ferric-rpa/src/properties/per_atom.rs` | the Becke and Hirshfeld bras |
| `site/src/reference/validation.md`, "Anchors" | the agreement figures, through section @sec:electrostatics-and-charges-well-these-checked |
