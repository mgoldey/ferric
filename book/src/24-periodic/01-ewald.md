# Lattices, Ewald summation and the Madelung constant {#ewald}

<!-- status: ready; registry: pbc-ewald-pairft-stress -->

Put one nucleus in every cell of a crystal and ask for its electrostatic
energy. A shell of cells at distance $R$ holds about $R^2$ copies, each
contributing about $1/R$, so the shells do not shrink: the sum over a charged
lattice diverges, and the sum over a neutral one converges only
conditionally, to a value that depends on the order in which you add the
shells. The Ewald method splits $1/r$ into a short-range part, which decays
fast enough to sum in real space, and a smooth long-range part, whose Fourier
series converges fast in reciprocal space. Two constants, a self term and a
uniform background, account for what the split moves between the two sums.
The total does not depend on where you put the split, and the code is tested
on that independence and against PySCF.

Section @sec:ewald-ewald-sum-independence derives the sum, its independence of the split, the Madelung
constant and the nuclear gradient; @sec:ewald-algorithm shows the code and @sec:ewald-numerical-anchors the
anchors.

| Symbol | Meaning | Code name |
|---|---|---|
| $\mathbf a_i$, $\mathbf b_i$ | lattice and reciprocal rows, $\mathbf a_i\cdot\mathbf b_j = 2\pi\delta_{ij}$ | `Cell` lattice, reciprocal |
| $\Omega = \det(\mathbf a)$ | cell volume | `cell.volume()` |
| $\mathbf L = \sum_in_i\mathbf a_i$, $\mathbf G = \sum_im_i\mathbf b_i$ | translations and reciprocal vectors ($m_i$ the Miller index) | |
| $\eta$ | Ewald splitting parameter | `omega` |
| $Z_j$, $\mathbf R_j$ | point charges and positions | nuclear charges |
| $Q = \sum_jZ_j$ | net charge of the set | `zsum` |
| $S(\mathbf G)$ | structure factor | |
| $v_M$ | Madelung constant | `madelung` |

| Entry point | Location |
|---|---|
| Rust | `ferric-pbc/src/ewald.rs::{ewald_point_charges,ewald_nuclear_repulsion,madelung_constant}` |
| Mesh constant | `ferric-pbc/src/kpts.rs::madelung` |
| Python | every periodic result exposes `e_nuc` and `madelung` |

## The Ewald sum and its independence of $\eta$ {#ewald-ewald-sum-independence}

**The split.** For any $\eta > 0$,

$$ \frac1r = \frac{\operatorname{erfc}(\eta r)}{r} + \frac{\operatorname{erf}(\eta r)}{r} . $$

The first term decays as a Gaussian, because $\operatorname{erfc}(x) < e^{-x^2}$.
The second is the potential of a unit Gaussian charge
$\rho_\eta(r) = (\eta^2/\pi)^{3/2}e^{-\eta^2r^2}$: it is finite at $r = 0$,
where it equals $2\eta/\sqrt\pi$, and smooth everywhere. A smooth function has
Fourier coefficients that fall fast. Subtract the transform of the erfc part
(verified equation @eq:pbc-hcore-split in section @sec:nuclear-attraction) from the bare $4\pi/G^2$:

$$ \int\frac{\operatorname{erf}(\eta r)}{r}\,e^{-i\mathbf G\cdot\mathbf r}\,d^3r = \frac{4\pi}{G^2}\,e^{-G^2/4\eta^2} . $$

**The short-range sum.** Give every pair of charges, in every relative
translation, its erfc interaction:

$$ E_\text{SR} = \tfrac12\sum_{\mathbf L}{\sum_{i,j}}'Z_iZ_j\frac{\operatorname{erfc}(\eta d_{ij\mathbf L})}{d_{ij\mathbf L}}, \qquad d_{ij\mathbf L} = |\mathbf R_i - \mathbf R_j - \mathbf L| . $$

The prime drops the self pair $i = j$ at $\mathbf L = 0$. The terms decay as a
Gaussian in $d$, so the sum converges absolutely.

**The long-range sum.** Sum the erf interaction over all pairs and all
translations, the self pair included for now. Poisson summation turns a
lattice sum into a sum over the reciprocal lattice,
$\sum_{\mathbf L}g(\mathbf d - \mathbf L) = \Omega^{-1}\sum_{\mathbf G}\hat g(\mathbf G)\,e^{i\mathbf G\cdot\mathbf d}$.
Apply it with $g = \operatorname{erf}(\eta r)/r$ and
$\mathbf d = \mathbf R_i - \mathbf R_j$. The double sum over charges becomes
$\sum_{ij}Z_iZ_je^{i\mathbf G\cdot(\mathbf R_i - \mathbf R_j)} = |S(\mathbf G)|^2$:

$$ \tfrac12\sum_{\mathbf L}\sum_{i,j}Z_iZ_j\frac{\operatorname{erf}(\eta d_{ij\mathbf L})}{d_{ij\mathbf L}} = \frac{1}{2\Omega}\sum_{\mathbf G}\frac{4\pi}{G^2}e^{-G^2/4\eta^2}\,|S(\mathbf G)|^2, \qquad S(\mathbf G) = \sum_jZ_je^{-i\mathbf G\cdot\mathbf R_j} . $$

The $\mathbf G \neq 0$ terms are the long-range energy,

$$ E_\text{LR} = \frac{2\pi}{\Omega}\sum_{\mathbf G\neq 0}\frac{e^{-G^2/4\eta^2}}{G^2}\,|S(\mathbf G)|^2 . $$

**The self term.** The left side above contains the self pair, which the
physical energy does not. Its value is $\tfrac12Z_j^2$ times
$\lim_{r\to 0}\operatorname{erf}(\eta r)/r = 2\eta/\sqrt\pi$. Subtract it:

$$ E_\text{self} = -\frac{\eta}{\sqrt\pi}\sum_jZ_j^2 . $$

**The background.** At $\mathbf G = 0$ the kernel $1/G^2$ diverges and
$S(0) = Q$. A charged set of nuclei needs a neutralizing partner. ferric uses
PySCF's convention (`Cell.energy_nuc()`): a uniform background of density
$-Q/\Omega$ in every cell. A uniform density has only a $\mathbf G = 0$
component, so it cancels the $\mathbf G = 0$ charge of the long-range sum,
and the reciprocal sum runs over $\mathbf G \neq 0$ only. What remains are the
background's short-range interactions, through
$\int\operatorname{erfc}(\eta r)/r\,d^3r = \pi/\eta^2$ (verified equation
@eq:pbc-hcore-g0). The charges with the background give
$\sum_iZ_i(-Q/\Omega)\,\pi/\eta^2 = -\pi Q^2/(\Omega\eta^2)$. The background
with itself gives half of $(Q/\Omega)^2\,\Omega\,\pi/\eta^2$. Add them:

$$ E_\text{bg} = -\frac{\pi Q^2}{2\Omega\eta^2} . $$

It vanishes for a neutral set. With $c_0 = \pi/(\eta^2\Omega)$, the constant
that sections @sec:nuclear-attraction and @sec:rsgdf use, it is $-\tfrac12c_0Q^2$.

**The total.** For point charges $Z_j$ at $\mathbf R_j$ and any $\eta > 0$:

$$
\begin{aligned}
E ={}& \tfrac12 \sum_{\mathbf L}{\sum_{i,j}}' Z_i Z_j
      \frac{\operatorname{erfc}(\eta d_{ij\mathbf L})}{d_{ij\mathbf L}}
+ \frac{2\pi}{\Omega}\sum_{\mathbf G\neq 0}
      \frac{e^{-G^2/4\eta^2}}{G^2}\,|S(\mathbf G)|^2
- \frac{\eta}{\sqrt\pi}\sum_j Z_j^2
 - \frac{\pi Q^2}{2\Omega\eta^2} .
\end{aligned}
$$

This is `ferric-pbc/src/ewald.rs::ewald_point_charges`. Each of the four
pieces is finite and converges absolutely, so the result no longer depends on
the order of summation.

**Independence of $\eta$.** The split point is arbitrary, so the total must
not depend on it. Differentiate each piece. Three identities are needed:

$$
\frac{\partial}{\partial\eta}\frac{\operatorname{erfc}(\eta r)}{r}
 = -\frac{2}{\sqrt\pi} e^{-\eta^2 r^2},
\qquad
\int e^{-\eta^2 r^2} e^{-i\mathbf G\cdot\mathbf r}\,d^3r
 = \Big(\frac{\pi}{\eta^2}\Big)^{3/2} e^{-G^2/4\eta^2},
$$

$$
\frac{\partial}{\partial\eta}\Big[\frac{4\pi}{G^2}e^{-G^2/4\eta^2}\Big]
 = \frac{2}{\sqrt\pi}\Big(\frac{\pi}{\eta^2}\Big)^{3/2} e^{-G^2/4\eta^2}.
$$

Use the first to differentiate $E_\text{SR}$, then the Poisson sum
$\sum_{\mathbf L} e^{-\eta^2|\mathbf d - \mathbf L|^2} = \Omega^{-1}\sum_{\mathbf G}(\pi/\eta^2)^{3/2} e^{-G^2/4\eta^2}e^{i\mathbf G\cdot\mathbf d}$
to move it to reciprocal space. It becomes a sum over $|S(\mathbf G)|^2$
plus the excluded self pair. Now compare term by term:

- per $\mathbf G \neq 0$, it cancels $\partial_\eta E_\text{LR}$ (third identity);
- its $\mathbf G = 0$ term, $-\pi Q^2/(\Omega\eta^3)$, cancels $\partial_\eta E_\text{bg} = +\pi Q^2/(\Omega\eta^3)$;
- the excluded self pair, $+\pi^{-1/2}\sum Z_j^2$, cancels $\partial_\eta E_\text{self}$.

Hence $dE/d\eta = 0$ exactly. Truncating the sums is the only source of
$\eta$-dependence. The background term is needed for this on a charged set:
without it the $\mathbf G = 0$ derivative is left over.

> **Verified equation {#eq:pbc-ewald-omega-independence}** · the Ewald sum is independent of $\eta$
>
> The $\eta$-derivatives of the short-range, long-range, self and background
> terms cancel per $\mathbf G \neq 0$ and at $\mathbf G = 0$; a transcribed
> Ewald sum on a charged triclinic toy cell is $\eta$-independent to
> $10^{-12}$, and without the background term it is not.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, part S1, claim `ewald-omega` |
> | Negative control | the runner's mutation: background $-\pi Q^2/(2\Omega\eta^2)$ without the 2 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/ewald.rs::ewald_point_charges` |
> | Code test | `ewald_matches_pyscf_and_is_omega_independent` in `crates/ferric-pbc/tests/pbc_stage0.rs` |
>
> <!-- verified: id=pbc.ewald.omega-independence; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=ewald-omega; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/ewald.rs::ewald_point_charges; rust_test=ferric-pbc/tests/pbc_stage0.rs::ewald_matches_pyscf_and_is_omega_independent; registry=pbc-ewald-pairft-stress -->

**The two extremes.** A large $\eta$ makes the erfc part short: the real-space
sum needs few translations and the reciprocal sum many $\mathbf G$. A small
$\eta$ does the opposite. Neither changes the value, only where the work
goes. `ferric-pbc/src/ewald.rs::default_ewald_omega` returns
$\eta = \sqrt\pi/\Omega^{1/3}$, which balances the two for a roughly isotropic
cell; the periodic one-electron builders pass it to
`ferric-pbc/src/ewald.rs::ewald_nuclear_repulsion`.

**The cutoffs.** Both tails are Gaussian. $\operatorname{erfc}(x) < e^{-x^2}$,
and $e^{-G^2/4\eta^2} = e^{-x^2}$ at $x = G/2\eta$. Both fall below a target
$\epsilon$ once $x \ge \sqrt{\ln(1/\epsilon)}$. With a margin of 1,

$$ r_\text{cut} = s/\eta, \qquad G_\text{cut} = 2\eta s, \qquad s = \sqrt{\ln(1/\epsilon)} + 1 . $$

The default target is `DEFAULT_EWALD_PRECISION` $= 10^{-16}$.

**The Madelung constant.** Take one unit point charge per cell with its
background. Its energy is half the charge times the potential $\psi$ that its
images and the background create at its own site, so $E = \tfrac12\psi$.
Define

$$ v_M = -2\,E[\text{one unit charge per cell, with background}] = -\psi . $$

$v_M$ depends on the lattice only. For ordinary cells $\psi < 0$: the
background attracts the charge more than its images repel it, and $v_M > 0$.
For a simple cubic lattice $v_Ma = 2.8372974794806$. A neutral set with mixed
signs is the textbook rock-salt case: the eight ions of the conventional cell
give $E = -8M/a$ with $M = 1.747564594633182$.

Check the large-cell limit. $v_M$ is homogeneous of degree $-1$ in lengths
(verified equation @eq:pbc-stress-madelung), so $v_M \propto 1/L$ for a cell of edge $L$. The
energy $-v_M/2$ of one charge with its images and background goes to zero as
the cell grows, as it must for an isolated charge.

H2 in STO-3G in cubic cells of 6, 8 and 10 Bohr, from the `589b50832`
build. $v_Ma$ stays at the simple-cubic constant, so $v_M$ falls as $1/a$, and
the `none` and `ewald` energies differ by $Nv_M/2$ with $N = 2$ electrons:

```python
import ferric

BOHR = 0.529177210903
mol = ferric.Molecule.from_xyz_string("2\nH2\nH 0 0 0\nH 0 0 0.74\n")
basis = ferric.BasisSet.bundled("sto-3g")
for a_bohr in (6.0, 8.0, 10.0):
    a = a_bohr * BOHR
    lat = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    ew = ferric.run_rhf_gamma(mol, lat, basis, exxdiv="ewald")
    no = ferric.run_rhf_gamma(mol, lat, basis, exxdiv="none")
    vm = ew.madelung
    print(f"a={a_bohr:5.1f} Bohr  e_nuc={ew.e_nuc:.10f}  v_M={vm:.10f}  "
          f"v_M*a={vm * a_bohr:.10f} (simple cubic 2.8372974794806)  "
          f"E_none-E_ewald={no.energy - ew.energy:.10f}  N v_M/2={vm:.10f}")
```

```text
a=  6.0 Bohr  e_nuc=-0.2101515940  v_M=0.4728829133  v_M*a=2.8372974796 (simple cubic 2.8372974794806)  E_none-E_ewald=0.4728829133  N v_M/2=0.4728829133
a=  8.0 Bohr  e_nuc=0.0141443639  v_M=0.3546621849  v_M*a=2.8372974796 (simple cubic 2.8372974794806)  E_none-E_ewald=0.3546621849  N v_M/2=0.3546621849
a= 10.0 Bohr  e_nuc=0.1518598009  v_M=0.2837297480  v_M*a=2.8372974796 (simple cubic 2.8372974794806)  E_none-E_ewald=0.2837297480  N v_M/2=0.2837297480
```

`ferric-pbc/src/ewald.rs::madelung_constant` evaluates $v_M$ at
$\eta = \sqrt\pi/\Omega^{1/3}$. A $k$-point mesh $N_1\times N_2\times N_3$ uses
the Gamma constant of the $\operatorname{diag}(N)$ supercell
(`ferric-pbc/src/kpts.rs::madelung`, PySCF `tools.pbc.madelung(cell, kpts)`).
Sections @sec:exchange-divergence and @sec:kpoint-scf explain why exchange needs it.

> **Verified equation {#eq:pbc-ewald-madelung}** · the Madelung constant
>
> $v_M = -2E$ for one unit charge per cell with its background reproduces the
> rock-salt constant $1.747564594633182$ and the simple-cubic
> $v_Ma = 2.8372974794806$, independent of $\eta$.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, part S1, claim `madelung` |
> | Negative control | the runner's mutation $v_M = -E$ in place of $-2E$ fails the simple-cubic constant |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/ewald.rs::madelung_constant` |
> | Code test | `madelung_matches_pyscf` in `crates/ferric-pbc/tests/pbc_stage0.rs` |
>
> <!-- verified: id=pbc.ewald.madelung; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=madelung; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/ewald.rs::madelung_constant; rust_test=ferric-pbc/tests/pbc_stage0.rs::madelung_matches_pyscf; registry=pbc-ewald-pairft-stress -->

**The gradient.** The forces of section @sec:forces need $\partial E/\partial\mathbf R_i$.
The self and background terms do not depend on positions and contribute
nothing. The short-range term differentiates through its radial function:

$$ f'(d) = \frac{d}{dd}\frac{\operatorname{erfc}(\eta d)}{d} = -\Big[\frac{\operatorname{erfc}(\eta d)}{d^2} + \frac{2\eta}{\sqrt\pi}\frac{e^{-\eta^2d^2}}{d}\Big], $$

times the unit vector $(\mathbf R_i - \mathbf R_j - \mathbf L)/d$, summed
over every partner $j$ and translation $\mathbf L$ (the pair appears twice in
$E_\text{SR}$, which cancels the $\tfrac12$). The long-range term
differentiates through the phase of $S(\mathbf G)$:
$\partial S/\partial\mathbf R_i = -i\mathbf G\,Z_ie^{-i\mathbf G\cdot\mathbf R_i}$,
so
$\partial|S|^2/\partial\mathbf R_i = 2\operatorname{Re}[S^*\,\partial S/\partial\mathbf R_i] = -2Z_i\mathbf G\operatorname{Im}[S(\mathbf G)e^{i\mathbf G\cdot\mathbf R_i}]$.
With the prefactor of $E_\text{LR}$:

$$ \frac{\partial E_\text{LR}}{\partial\mathbf R_i} = -\frac{4\pi}{\Omega}Z_i\sum_{\mathbf G\neq0}\frac{e^{-G^2/4\eta^2}}{G^2}\,\mathbf G\,\operatorname{Im}\big[S(\mathbf G)\,e^{i\mathbf G\cdot\mathbf R_i}\big] . \label{eq:pbc-ewald-lr-gradient} $$

Check the limit of a rigid translation. Summed over $i$,
$\sum_iZ_ie^{i\mathbf G\cdot\mathbf R_i} = S^*$, and $\operatorname{Im}[SS^*] = 0$,
so the long-range rows sum to zero; the short-range rows cancel in pairs. The
total force on the nuclei is zero, as it must be for an energy that depends on
differences of positions only.

> **Verified equation {#eq:pbc-ewald-gradient}** · the Ewald nuclear gradient
>
> The short-range radial derivative and the long-range row
> (@eq:pbc-ewald-lr-gradient) reproduce a central difference of the Ewald
> energy on a charged triclinic toy cell to $2.4\times10^{-11}$ Ha/Bohr, and
> sum to zero over the charges.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, part S1, claim `ewald-gradient` |
> | Negative control | the runner's mutation: the long-range row with the wrong sign |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/ewald.rs::ewald_nuclear_gradient_parts` |
> | Code test | `ewald_gradient_matches_pyscf_grad_nuc_and_fd` in `crates/ferric-pbc/tests/pbc_grad.rs` |
>
> <!-- verified: id=pbc.ewald.gradient; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=ewald-gradient; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/ewald.rs::ewald_nuclear_gradient_parts; rust_test=ferric-pbc/tests/pbc_grad.rs::ewald_gradient_matches_pyscf_grad_nuc_and_fd; registry=pbc-ewald-pairft-stress -->

`ferric-pbc/src/ewald.rs::ewald_nuclear_gradient_parts` returns the two parts
separately, so a mutation test can drop one; both are checked against PySCF
`grad_nuc` and finite differences in
`ewald_gradient_matches_pyscf_grad_nuc_and_fd`
(`crates/ferric-pbc/tests/pbc_grad.rs`).

## The algorithm {#ewald-algorithm}

| Step | Quantity | Cost |
|---|---|---|
| 1 | cutoffs $r_\text{cut} = s/\eta$, $G_\text{cut} = 2\eta s$, $s = \sqrt{\ln(1/\epsilon)} + 1$ | |
| 2 | real-space sum of $Z_iZ_j\operatorname{erfc}(\eta d)/d$ over pairs and translations within $r_\text{cut}$ | $N_\text{atom}^2 N_{\mathbf L}$ |
| 3 | reciprocal sum of $e^{-G^2/4\eta^2}|S(\mathbf G)|^2/G^2$ within $G_\text{cut}$ | $N_\text{atom}N_{\mathbf G}$ |
| 4 | self term $-\eta\pi^{-1/2}\sum Z_j^2$ and background $-\pi Q^2/(2\Omega\eta^2)$ | |

The reciprocal sum and the two constants, from
`ferric-pbc/src/ewald.rs::ewald_point_charges`:

<!-- excerpt: ferric-pbc/src/ewald.rs::ewald_point_charges -->
```rust
    for g in cell.gvectors(gcut)? {
        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        if g2 < 1e-24 {
            continue; // G = 0 handled by the background term
        }
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for (z, r) in charges.iter().zip(positions) {
            let ph = g[0] * r[0] + g[1] * r[1] + g[2] * r[2];
            re += z * ph.cos();
            im -= z * ph.sin();
        }
        lr.add((-g2 / (4.0 * omega * omega)).exp() / g2 * (re * re + im * im));
    }
    let e_lr = 2.0 * std::f64::consts::PI / vol * lr.value();

    let z2: f64 = charges.iter().map(|z| z * z).sum();
    let zsum: f64 = charges.iter().sum();
    let e_self = -omega / std::f64::consts::PI.sqrt() * z2;
    let e_g0 = -std::f64::consts::PI * zsum * zsum / (2.0 * vol * omega * omega);
```

| Code | Equation |
|---|---|
| `re`, `im` | $\operatorname{Re}S(\mathbf G)$, $\operatorname{Im}S(\mathbf G)$ with $S = \sum_jZ_je^{-i\mathbf G\cdot\mathbf R_j}$ |
| `lr.add(...)` | $e^{-G^2/4\eta^2}|S(\mathbf G)|^2/G^2$, one term of $E_\text{LR}$ |
| `e_lr` | $E_\text{LR}$ with its prefactor $2\pi/\Omega$ |
| `e_self` | $E_\text{self}$ |
| `e_g0` | $E_\text{bg}$ |

- **$\mathbf G = 0$ is skipped, not regularized.** The `continue` is the
  background convention: the uniform background removes that term, and its
  finite remainder is `e_g0`.
- **Compensated sums.** `lr` and the real-space accumulator are Neumaier
  sums, and so is the final total. The module documentation records why, from
  a numpy measurement that no test pins (the test evidence is the
  $\eta$-spread below $10^{-12}$ in
  `ewald_matches_pyscf_and_is_omega_independent`): at
  $\eta = 3$ the self and long-range terms of H$_2$O in a 7-Bohr cell are each
  about $10^2$ Ha and cancel to about $10^1$, and naive summation of the
  roughly $10^5$ $\mathbf G$ terms left a $2.6\times10^{-11}$ Ha
  $\eta$-dependent residual, measured in numpy against PySCF and the same at
  precision targets $10^{-16}$, $10^{-20}$ and $10^{-24}$, so it was
  rounding, not truncation. Exact summation (`math.fsum`) reduced it to
  $3\times10^{-14}$.
- **Coincident charges are an error.** Two charges at the same point under
  some translation have an infinite pair energy; the function returns an
  error naming them. A zero charge (a ghost atom) is skipped, so it may sit
  on a real nucleus.

## Numerical anchors {#ewald-numerical-anchors}

From the tests named in section @sec:ewald-ewald-sum-independence.

| Quantity | Reference | Bar | Test |
|---|---|---|---|
| $E_{nn}$ on three charged nuclear cells | PySCF `energy_nuc` | $10^{-10}$ Ha; $\eta$-spread $< 10^{-12}$ at $\eta = 0.8, 1.5, 3$ | `crates/ferric-pbc/tests/pbc_stage0.rs` `ewald_matches_pyscf_and_is_omega_independent` |
| $v_M$ | PySCF and the simple-cubic textbook value | $10^{-11}$ | `madelung_matches_pyscf` |
| rock-salt $E = -8M/a$ | $M = 1.747564594633182$ | $\eta$-spread $< 10^{-12}$ | `ewald_rock_salt_madelung_is_textbook_and_omega_independent` |
| Ewald gradient | PySCF `grad_nuc` and finite differences; the script's central difference | $2.4\times10^{-11}$ Ha/Bohr (script) | `crates/ferric-pbc/tests/pbc_grad.rs` `ewald_gradient_matches_pyscf_grad_nuc_and_fd` |

## Sources {#ewald-sources}

| Source | What it gives |
|---|---|
| `scripts/pbc_ewald_pairft_stress_sympy.py`, part S1 | the proof |
| P. P. Ewald, *Ann. Phys.* **369**, 253 (1921) | the method |
| `site/src/methods/pbc.md` | the user guide |
