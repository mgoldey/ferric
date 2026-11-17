# The stress tensor {#stress}

<!-- status: ready; registry: pbc-ewald-pairft-stress -->

The force asks how the energy changes when one atom moves. The stress asks
how it changes when the whole cell is deformed: the lattice vectors and the
atoms are strained together, with fractional coordinates and Miller indices
fixed. It is the quantity a lattice optimizer would follow, and its trace
gives the pressure. ferric computes it analytically. There is no lattice
relaxation driver: the stress is reported, not used.

A strain touches every term of the periodic energy in one of three ways:
through the volume $\Omega$, through $G^2$ in a reciprocal kernel, or through
a real-space distance or pair separation. Section @sec:stress-strain-moves derives how those
three quantities move; @sec:stress-ewald-energy-madelung-constant applies it to the Ewald energy and the
Madelung constant, @sec:stress-pair-transform-under-strain to the pair transform, and @sec:stress-assembling-electronic-stress assembles the
electronic stress. The overlap-coupled terms reuse the force machinery of
section @sec:forces.

| Symbol | Meaning | Code name |
|---|---|---|
| $\varepsilon_{ab}$ | homogeneous strain, $\mathbf r \to (1 + \varepsilon)\mathbf r$ | |
| $\sigma_{ab} = \Omega^{-1}dE/d\varepsilon_{ab}$ | stress, Hartree/Bohr$^3$, not symmetrized | `stress()` |
| $p = -\operatorname{tr}\sigma/3$ | pressure | |
| $k(G^2) = e^{-G^2/4\eta^2}/G^2$ | Ewald LR kernel | `k` |

| Entry point | Location |
|---|---|
| Ewald strain | `ferric-pbc/src/ewald.rs::{ewald_nuclear_strain,madelung_strain}` |
| Pair-transform strain | `ferric-pbc/src/pair_ft.rs::pair_ft_strain_chunked` |
| Gamma assembly | `ferric-pbc/src/stress.rs::gamma_rhf_stress` and siblings |
| $k$-point assembly | `ferric-pbc/src/kstress.rs::kpoint_rhf_stress` and siblings |
| Python | `with_stress=True` on the SCF entry points |
| CLI | `[cell] stress = true`, which requires `kmesh` |

## What a strain moves {#stress-strain-moves}

Strain every lattice row and every atom, $\mathbf r \to (1 + \varepsilon)\mathbf r$.
Three consequences follow to first order in $\varepsilon$.

**The volume.** $\Omega \to \Omega\det(1 + \varepsilon)$, and
$\det(1 + \varepsilon) = 1 + \operatorname{tr}\varepsilon + O(\varepsilon^2)$, so

$$ \frac{d\Omega}{d\varepsilon_{ab}} = \Omega\,\delta_{ab} . $$

**The reciprocal vectors.** $\mathbf a_i\cdot\mathbf b_j = 2\pi\delta_{ij}$
must survive the strain, so $\mathbf G \to (1 + \varepsilon)^{-\mathsf T}\mathbf G \approx (1 - \varepsilon^{\mathsf T})\mathbf G$
at a fixed Miller index. Then $G_c \to G_c - \varepsilon_{dc}G_d$, and

$$ \frac{dG^2}{d\varepsilon_{ab}} = -2G_aG_b . $$

**A distance.** A separation vector strains like a position,
$d_c \to d_c + \varepsilon_{cd}d_d$, so $|\mathbf d|^2$ gains
$2d_c\varepsilon_{cd}d_d$ and

$$ \frac{d|\mathbf d|}{d\varepsilon_{ab}} = \frac{d_ad_b}{|\mathbf d|} . $$

**What does not move.** A phase $\mathbf G\cdot\mathbf X$ becomes
$\mathbf G^{\mathsf T}(1 + \varepsilon)^{-1}(1 + \varepsilon)\mathbf X = \mathbf G\cdot\mathbf X$.
So the structure factor $S(\mathbf G) = \sum_AZ_Ae^{-i\mathbf G\cdot\mathbf R_A}$
and every phase in a pair transform are strain-invariant.

With these, any reciprocal-space energy
$E = \Omega^{-1}\sum_{\mathbf G}v(G^2)\Phi(\mathbf G)$ differentiates by the
product rule:

$$
\frac{dE}{d\varepsilon_{ab}} = -\delta_{ab}E
 + \frac{1}{\Omega}\sum_{\mathbf G}\Big[v'(G^2)\,(-2G_aG_b)\,\Phi + v\,\frac{d\Phi}{d\varepsilon_{ab}}\Big] .
$$

The first term is the $1/\Omega$; it puts $-E$ on the diagonal. Every
periodic stress term below has this shape.

## The Ewald energy and the Madelung constant {#stress-ewald-energy-madelung-constant}

Apply the three rules to the four Ewald parts of section @sec:ewald.

**Short range.** $E_\text{SR} = \tfrac12\sum'Z_iZ_jf(d)$ with
$f(d) = \operatorname{erfc}(\eta d)/d$. The chain rule and the distance
rule give

$$ \frac{dE_\text{SR}}{d\varepsilon_{ab}} = \tfrac12\sum{}'Z_iZ_jf'(d)\frac{d_ad_b}{d},\qquad
f'(d) = -\frac{\operatorname{erfc}(\eta d)}{d^2} - \frac{2\eta}{\sqrt\pi}\frac{e^{-\eta^2d^2}}{d} . $$

It has the virial form: a pair force $f'$ times the outer product of the
separation.

**Long range.** $E_\text{LR} = (2\pi/\Omega)\sum_{\mathbf G\neq0}|S(\mathbf G)|^2k(G^2)$.
$S$ does not move, so only the volume and the kernel do. The derivative of
$k(x) = e^{-x/4\eta^2}/x$ with respect to $x = G^2$ is

$$ k'(G^2) = -k\Big(\frac{1}{4\eta^2} + \frac{1}{G^2}\Big), $$

and the product rule gives

$$ \frac{dE_\text{LR}}{d\varepsilon_{ab}} = -\delta_{ab}E_\text{LR}
 + \frac{2\pi}{\Omega}\sum_{\mathbf G\neq0}|S(\mathbf G)|^2\,k'(G^2)\,(-2G_aG_b) . $$

**Background and self.** $E_\text{bg} = -\pi Q^2/(2\Omega\eta^2)$ depends on
the cell only through $1/\Omega$, so its derivative is
$-\delta_{ab}E_\text{bg}$. $E_\text{self} = -(\eta/\sqrt\pi)\sum Z^2$ does
not depend on the cell, so its derivative is zero.

**Check: η-independence.** $E_{nn}$ does not depend on $\eta$ at any strain,
so neither does its strain derivative. The individual parts do. The code
test evaluates the total at $\eta = 1$ and $2.3$ and requires agreement to
$10^{-10}$.

**The Madelung constant.** $v_M = -2E$ of one unit charge in its neutralizing
background (section @sec:ewald), so $dv_M/d\varepsilon = -2$ times the strain of
that one-charge Ewald sum (`madelung_strain`). There is an independent check.
$v_M$ has units of inverse length and depends on nothing but the lattice, so
it is homogeneous of degree $-1$: scaling every length by $\lambda$ gives
$v_M/\lambda$. An isotropic strain $\varepsilon = t\,\mathbb 1$ is that
scaling with $\lambda = 1 + t$, and
$\frac{d}{dt}\big[v_M/(1 + t)\big]_{t=0} = -v_M$ is the trace:

$$ \operatorname{tr}\frac{dv_M}{d\varepsilon} = -v_M . $$

**What `exxdiv` does to the pressure.** With `exxdiv = ewald` the energy has
the Madelung term $-\tfrac\alpha2v_M\sum_\sigma\operatorname{tr}(D_\sigma SD_\sigma S)$.
The densities are the same as with `none` (section @sec:exchange-divergence), and the part of its
strain derivative that goes through $S$ cancels against the shift inside $W$
exactly as in the force (section @sec:forces-madelung-term-cancels). What remains is the derivative of
$v_M$ at $\operatorname{tr}(D_\sigma SD_\sigma S) = N_\sigma$:

$$ \sigma^\text{ewald} - \sigma^\text{none} = -\frac{\alpha N}{2\Omega}\frac{dv_M}{d\varepsilon},\qquad
p^\text{ewald} - p^\text{none} = -\frac{\alpha N v_M}{6\Omega}, $$

the second from the Euler identity. The test
`h2_rhf_ewald_minus_none_is_the_madelung_strain` checks the stress
difference against the Madelung strain term to $10^{-9}$.

> **Verified equation {#eq:pbc-stress-ewald}** · the Ewald strain
>
> The SR virial, the LR term $-\delta_{ab}E_\text{LR}$ plus the $G^2$
> derivative of the kernel, and $-\delta_{ab}E_\text{bg}$ equal central
> finite differences of a pure-Python Ewald at frozen index sets.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, claim `ewald-strain` |
> | Negative control | the runner's mutation: the $1/G^2$ term dropped from $k'$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/ewald.rs::ewald_nuclear_strain` |
> | Code test | `ewald_strain_matches_fd_is_omega_independent_and_madelung_is_homogeneous` in `crates/ferric-pbc/tests/pbc_stress.rs` |
>
> <!-- verified: id=pbc.stress.ewald; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=ewald-strain; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/ewald.rs::ewald_nuclear_strain; rust_test=ferric-pbc/tests/pbc_stress.rs::ewald_strain_matches_fd_is_omega_independent_and_madelung_is_homogeneous; registry=pbc-ewald-pairft-stress -->

> **Verified equation {#eq:pbc-stress-madelung}** · the Madelung constant is homogeneous
>
> $v_M$ is homogeneous of degree $-1$ in lengths, so
> $\operatorname{tr}\,dv_M/d\varepsilon = -v_M$.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, claim `madelung-euler` |
> | Negative control | none in the script |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/ewald.rs::madelung_strain` |
> | Code test | `h2_rhf_ewald_minus_none_is_the_madelung_strain` in `crates/ferric-pbc/tests/pbc_stress.rs` |
>
> <!-- verified: id=pbc.stress.madelung; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=madelung-euler; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/ewald.rs::madelung_strain; rust_test=ferric-pbc/tests/pbc_stress.rs::h2_rhf_ewald_minus_none_is_the_madelung_strain; registry=pbc-ewald-pairft-stress -->

The long-range strain is this part of
`ferric-pbc/src/ewald.rs::ewald_point_charges_strain`, inside the loop over
$\mathbf G$ and after it:

<!-- excerpt: ferric-pbc/src/ewald.rs::ewald_point_charges_strain -->
```rust
        let s2 = re * re + im * im;
        let k = (-g2 / (4.0 * omega * omega)).exp() / g2;
        e_lr.add(k * s2);
        let kp = -k * (1.0 / (4.0 * omega * omega) + 1.0 / g2);
        let f = s2 * kp * -2.0;
        for a in 0..3 {
            for b in 0..3 {
                lr_g[a][b].add(f * g[a] * g[b]);
            }
        }
    }
    let e_lr = tp_vol * e_lr.value();
    let mut lr = fold9(&lr_g);
    for (a, row) in lr.iter_mut().enumerate() {
        for v in row.iter_mut() {
            *v *= tp_vol;
        }
        row[a] -= e_lr;
    }
```

| Code | Equation |
|---|---|
| `s2` | $\lvert S(\mathbf G)\rvert^2$ |
| `k`, `kp` | $k(G^2)$, $k'(G^2)$ |
| `f * g[a] * g[b]` | $\lvert S\rvert^2k'(-2G_aG_b)$ |
| `tp_vol` | $2\pi/\Omega$ |
| `row[a] -= e_lr` | $-\delta_{ab}E_\text{LR}$ |

The energy and its strain are accumulated in the same pass, so the
$-\delta_{ab}E_\text{LR}$ term uses exactly the energy that was summed. Both
go through Neumaier sums, as the energy does.

## The pair transform under strain {#stress-pair-transform-under-strain}

For one primitive pair, the transform of section @sec:pair-ft is a product of three
factors: the Hermite polynomial $\prod_d\sum_tE_t(-iG_d)^t$, whose $E$
coefficients depend on the separation $\mathbf A - \mathbf B'$ (with
$\mathbf B'$ the image centre) and on the exponents; the Gaussian
$e^{-G^2/4p}$; and the phase $e^{-i\mathbf G\cdot\mathbf P_c}$. Differentiate
each:

- The phase is invariant (@sec:stress-strain-moves).
- The Gaussian: $\frac{d}{d\varepsilon_{ab}}e^{-G^2/4p} = -\frac{1}{4p}(-2G_aG_b)e^{-G^2/4p}$,
  which gives $\frac{G_aG_b}{2p}P$.
- The polynomial through $\mathbf G$: $dG_c/d\varepsilon_{ab} = -\delta_{cb}G_a$,
  so it gives $-G_aP^g_b$, with $P^g_b$ the $G_b$-derivative of the
  polynomial factor.
- The separation: $d(\mathbf A - \mathbf B')_c/d\varepsilon_{ab} = \delta_{ca}(\mathbf A - \mathbf B')_b$.
  The bra-centre derivative $Q_a = \partial P/\partial A_a$ contains the
  separation derivative and the motion of $\mathbf P_c = (\alpha\mathbf A + \beta\mathbf B')/p$
  in the phase, $-iG_a(\alpha/p)P$. The phase does not move under strain, so
  that piece is removed: $Q_a + iG_a(\alpha/p)P$.

Together:

$$
\frac{dP}{d\varepsilon_{ab}} = \Big[Q_a + iG_a\frac{\alpha}{p}P\Big](\mathbf A - \mathbf B')_b
 + \frac{G_aG_b}{2p}P - G_aP^g_b .
$$

> **Verified equation {#eq:pbc-stress-pairft}** · the strain derivative of a pair transform
>
> $dP/d\varepsilon_{ab} = [Q_a + iG_a(\alpha/p)P](\mathbf A - \mathbf B')_b
> + G_aG_bP/(2p) - G_aP^g_b$ equals the direct derivative of the strained
> closed form for s|s and p$_x$|s at exact rational points.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, claim `pairft-strain` |
> | Negative control | none in the script; the code test compares against finite differences at fixed Miller index |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/pair_ft.rs::pair_ft_strain_chunked` |
> | Code test | `pair_ft_strain_matches_fd_at_fixed_miller_index` in `crates/ferric-pbc/tests/pbc_stress.rs` |
>
> <!-- verified: id=pbc.stress.pairft; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=pairft-strain; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/pair_ft.rs::pair_ft_strain_chunked; rust_test=ferric-pbc/tests/pbc_stress.rs::pair_ft_strain_matches_fd_at_fixed_miller_index; registry=pbc-ewald-pairft-stress -->

## Assembling the electronic stress {#stress-assembling-electronic-stress}

| Step | Term | Function |
|---|---|---|
| 1 | Ewald nuclear strain | `ferric-pbc/src/ewald.rs::ewald_nuclear_strain` |
| 2 | each reciprocal energy: $-\delta_{ab}E + \Omega^{-1}\sum_{\mathbf G}[v'(-2G_aG_b)\Phi + v\,d\Phi]$, with $d\Phi$ from the pair-transform strain | `ferric-pbc/src/pair_ft.rs::pair_ft_strain_chunked` |
| 3 | overlap-coupled terms: the force's matrix $M$ contracted with per-image pair vectors | `ferric-pbc/src/stress.rs::gamma_rhf_stress` |
| 4 | Madelung: $-\tfrac\alpha2\sum_\sigma\operatorname{tr}(D_\sigma SD_\sigma S)\,dv_M/d\varepsilon$ | `ferric-pbc/src/ewald.rs::madelung_strain` |
| 5 | XC, ECP and RS-GDF terms | listed in the `stress.rs` module documentation |

Step 3 is the force's $M$ (Pulay, the $c_0Z_\text{tot}$ term and the Madelung
$S$ term) with a different derivative of $S$: an overlap element between a
function at $\mathbf A$ and an image at $\mathbf B + \mathbf L$ depends on the
separation $\mathbf A - \mathbf B - \mathbf L$, which strains by the
distance rule. The $\mathbf G = 0$ energy $c_0Z_\text{tot}\operatorname{tr}(DS)$
also depends on the volume through $c_0 = \pi/(\eta^2\Omega)$; at
$\operatorname{tr}(DS) = N$ that adds $-\delta_{ab}c_0Z_\text{tot}N$ (its $S$
part is in $M$).

**Frozen index sets.** Every truncated sum (real-space images, $\mathbf G$
spheres, screened triplets) is frozen as integer indices at the reference
cell (`Cell::strained`). Re-selecting the sphere at the strained cell would
add or drop vectors at the cutoff, a step change in the energy that no
derivative describes. With the sets frozen, the stress is the exact
derivative of the energy that was computed. The same reasoning makes the
finite-difference anchors clean: they strain a frozen cell.

## Options {#stress-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `with_stress=` (Python) | `True`, `False` | `False` | Gamma point or mesh |
| `[cell] stress` (CLI) | `true`, `false` | `false` | prints the stress; requires `kmesh` |

## Numerical anchors {#stress-numerical-anchors}

| Quantity | Bar | Test |
|---|---|---|
| Ewald strain against finite differences; at $\eta = 1$ and $2.3$; Euler identity | $10^{-8}$; $10^{-10}$; $10^{-12}$ | `crates/ferric-pbc/tests/pbc_stress.rs` `ewald_strain_matches_fd_is_omega_independent_and_madelung_is_homogeneous` |
| H2 RHF stress, all nine components, both `exxdiv` | $10^{-7}$ Ha/Bohr$^3$ against finite differences of ferric's own energy | `h2_rhf_stress_matches_fd_of_own_energy_all_nine_both_exxdiv` |
| `ewald` minus `none` stress equals the Madelung strain term | $10^{-9}$ | `h2_rhf_ewald_minus_none_is_the_madelung_strain` |
| pressure against finite differences of an isotropic scaling (a consistency check, not independent of the nine-component anchor) | $10^{-9}$ | `h2_rhf_pressure_matches_isotropic_scaling` |
| one-point mesh stress against the Gamma stress | | `crates/ferric-pbc/tests/pbc_kstress.rs` `one_point_mesh_rhf_stress_is_the_gamma_stress` |
| each stress mutant (`NoPulay`, `FtNoCentres`, `GUnstrained`, `NoVolume`, `NoEwaldLr`, `NoEwaldBg`, `NoC0Volume`, `NoSrImages` with `exxdiv = none`; `NoMadelung`, `MadelungS` with `exxdiv = ewald`; the variants of `ferric-pbc/src/stress.rs::StressMutation`) misses finite differences on at least one component; the volume-only mutants are diagonal by construction | the FD anchor of the row above | `crates/ferric-pbc/tests/pbc_stress.rs` `h2_fd_anchor_catches_each_mutant` |

The scripted proof of the Ewald strain (part S5 of
`scripts/pbc_ewald_pairft_stress_sympy.py`) agrees with central finite
differences of a pure-Python Ewald sum at frozen index sets to
$1.5\times10^{-10}$, and with the Euler identity to $10^{-12}$. The
$k$-point, open-shell, hybrid and meta-GGA stresses are tested in
`crates/ferric-pbc/tests/pbc_kstress.rs` and the `validation_*stress*.rs`
files of `crates/ferric-pbc/tests/`.

## Limits {#stress-limits}

- No lattice-vector relaxation.
- Gamma-point stress is Python-only; the CLI `stress = true` requires
  `kmesh`.
- VV10 is refused. Per-functional support is listed in the input reference.

## Sources {#stress-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-pbc/theory.md`, §5 | the strain derivatives |
| `scripts/pbc_ewald_pairft_stress_sympy.py` | the proof (S5) |
| O. H. Nielsen and R. M. Martin, *Phys. Rev. B* **32**, 3780 (1985) | the stress theorem |
