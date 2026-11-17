# Ewald-split nuclear attraction {#nuclear-attraction}

<!-- status: ready; registry: pbc-ewald-pairft-stress -->

An electron in a crystal is attracted by every nucleus in every cell, and
that sum has the problem of section @sec:ewald: it diverges, here with the
opposite sign to the electron repulsion. In a neutral cell the two
divergences are the same $\mathbf G = 0$ term with opposite signs, so ferric
drops $\mathbf G = 0$ from both. What is left is evaluated with the same
Ewald split as the nuclear repulsion: a short-range erfc part in real space
through the three-centre integral engine, a long-range part in reciprocal
space through the pair transform of section @sec:pair-ft, and one constant that
removes the $\mathbf G = 0$ average the real-space part carries.

Section @sec:nuclear-attraction-three-parts-matrix derives the three parts and shows that their sum does not
depend on the split; @sec:nuclear-attraction-algorithm gives the algorithm and the code, @sec:nuclear-attraction-numerical-anchors the
anchors.

| Symbol | Meaning | Code name |
|---|---|---|
| $V_\text{SR}$, $V_\text{LR}$, $V_\text{G0}$ | the three parts | `periodic_hcore` |
| $\eta$ | the nuclear split | `PeriodicHcoreConfig::omega` |
| $c_0 = \pi/(\eta^2\Omega)$ | the $\mathbf G = 0$ constant | `c0` |
| $\zeta = 10^{16}$ | exponent of the Gaussian nuclei in $V_\text{SR}$ | `GAUSSIAN_NUCLEUS_EXPONENT` |
| $S^\text{latt}_{\mu\nu} = P_{\mu\nu}(0)$ | lattice overlap | `s` |

| Entry point | Location |
|---|---|
| Rust | `ferric-pbc/src/hcore.rs::{periodic_hcore,reciprocal_nuclear}` |
| Oracle | `ferric-pbc/src/hcore.rs::pure_aft_nuclear` ($\eta \to \infty$) |
| $k$-points | `crates/ferric-pbc/src/hcore/kpoint.rs` |

## Three parts, one matrix {#nuclear-attraction-three-parts-matrix}

The nuclear attraction of the AO pair $\mu\nu$ is

$$ V = V_\text{SR}(\eta) + V_\text{LR}(\eta) + V_\text{G0}(\eta) . $$

**The short-range part.** Split each nucleus's $1/r$ as in section @sec:ewald and
keep the erfc piece in real space:

$$ V_\text{SR} = -\sum_{\mathbf L}\sum_{C,\mathbf M} Z_C \Big\langle \mu_0 \Big| \frac{\operatorname{erfc}(\eta|\mathbf r - \mathbf R_C - \mathbf M|)} {|\mathbf r - \mathbf R_C - \mathbf M|}\Big| \nu_{\mathbf L}\Big\rangle . $$

The sum over $\mathbf L$ places the partner function in every cell; the sum
over $\mathbf M$ places every nucleus $C$ in every cell. Both are short
because erfc decays as a Gaussian.

**The long-range part.** The erf piece of all the nuclei is a smooth
periodic potential. Poisson summation over $\mathbf M$, with the transform of
section @sec:ewald, writes it as a Fourier series:

$$ \phi_\text{LR}(\mathbf r) = \frac{1}{\Omega}\sum_{\mathbf G}\frac{4\pi}{G^2}e^{-G^2/4\eta^2}\,S(\mathbf G)\,e^{i\mathbf G\cdot\mathbf r}, \qquad S(\mathbf G) = \sum_CZ_Ce^{-i\mathbf G\cdot\mathbf R_C} . $$

Integrate it against the lattice-summed pair density
$\sum_{\mathbf L}\phi_\mu\phi_{\nu,\mathbf L}$. For real AOs,
$\int\phi_\mu\phi_{\nu\mathbf L}\,e^{+i\mathbf G\cdot\mathbf r} = P_{\mu\nu}(\mathbf G)^*$,
so each $\mathbf G$ contributes $-\Omega^{-1}v_\eta(G)P^*_{\mu\nu}S$. The
terms at $\mathbf G$ and $-\mathbf G$ are complex conjugates, and their sum
is twice the real part. Drop $\mathbf G = 0$:

$$ V_\text{LR} = -\frac{1}{\Omega}\sum_{\mathbf G\neq 0}\frac{4\pi}{G^2}e^{-G^2/4\eta^2} \operatorname{Re}\big[P^*_{\mu\nu}(\mathbf G)\,S(\mathbf G)\big] . $$

**Per $\mathbf G \neq 0$, the two parts add to the bare kernel.** The
transform of the erfc kernel is

$$
\int\frac{\operatorname{erfc}(\eta r)}{r}e^{-i\mathbf G\cdot\mathbf r}d^3r
 = \frac{4\pi}{G^2}\big(1 - e^{-G^2/4\eta^2}\big) .
$$

Add the erf transform $(4\pi/G^2)e^{-G^2/4\eta^2}$: the Gaussian factors
cancel and the sum is the bare $4\pi/G^2$. So per $\mathbf G \neq 0$,
$V_\text{SR} + V_\text{LR}$ is the bare $-4\pi Z/(\Omega G^2)$ kernel, for
every $\eta$.

> **Verified equation {#eq:pbc-hcore-split}** · the erfc and erf parts add to the bare kernel
>
> $\operatorname{FT}[\operatorname{erfc}(\eta r)/r] = (4\pi/G^2)(1 - e^{-G^2/4\eta^2})$,
> so per $\mathbf G \neq 0$ the short- and long-range parts are the bare
> $-4\pi Z/(\Omega G^2)$ kernel.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, part S3, claim `hcore-split` |
> | Negative control | none in the script; the code test checks that $V_\text{G0}$ is visible |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/hcore.rs::reciprocal_nuclear` |
> | Code test | `periodic_hcore_is_omega_independent_triclinic_sp` in `crates/ferric-pbc/tests/pbc_hcore.rs` |
>
> <!-- verified: id=pbc.hcore.split; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=hcore-split; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/hcore.rs::reciprocal_nuclear; rust_test=ferric-pbc/tests/pbc_hcore.rs::periodic_hcore_is_omega_independent_triclinic_sp; registry=pbc-ewald-pairft-stress -->

**At $\mathbf G = 0$, the real-space part carries an average.** The erfc
transform is finite at $G \to 0$:

$$ \lim_{G\to 0}\frac{4\pi}{G^2}\big(1 - e^{-G^2/4\eta^2}\big) = \int\frac{\operatorname{erfc}(\eta r)}{r}\,d^3r = \frac{\pi}{\eta^2} . $$

The real-space sum is a full lattice sum, so it contains this term: a
constant potential $-\pi Z_\text{tot}/(\eta^2\Omega)$, which acts on the
pair as that constant times the overlap. The convention of the chapter drops
$\mathbf G = 0$, so this average has to come out again:

$$ V_\text{G0} = +\frac{\pi Z_\text{tot}}{\eta^2\Omega}\,S^\text{latt}_{\mu\nu} = c_0Z_\text{tot}S^\text{latt}_{\mu\nu} . $$

With it, $V$ is independent of $\eta$ and equals the pure reciprocal-space
attraction with $\mathbf G = 0$ dropped. That is the convention that cancels
against the same term of $J$ in a neutral cell. It is the same constant
$c_0$ as the background of section @sec:ewald and the subtraction of section @sec:rsgdf.

> **Verified equation {#eq:pbc-hcore-g0}** · the $\mathbf G = 0$ constant
>
> $\int\operatorname{erfc}(\eta r)/r\,d^3r = \pi/\eta^2$, so the real-space
> erfc sum carries the average $-\pi Z_\text{tot}/(\eta^2\Omega)$ times the
> overlap, which $V_\text{G0} = +c_0Z_\text{tot}S$ removes; the total is the
> pure reciprocal attraction with $\mathbf G = 0$ dropped.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, part S3, claim `hcore-g0` |
> | Negative control | none in the script |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/hcore.rs::periodic_hcore` |
> | Code test | `periodic_hcore_is_omega_independent_triclinic_sp` in `crates/ferric-pbc/tests/pbc_hcore.rs` |
>
> <!-- verified: id=pbc.hcore.g0; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=hcore-g0; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/hcore.rs::periodic_hcore; rust_test=ferric-pbc/tests/pbc_hcore.rs::periodic_hcore_is_omega_independent_triclinic_sp; registry=pbc-ewald-pairft-stress -->

**Check the limit $\eta \to \infty$.** The erfc part shrinks to nothing,
$c_0 \to 0$, and the Gaussian damping $e^{-G^2/4\eta^2}$ goes to 1. What is
left is the pure reciprocal sum
$-(1/\Omega)\sum_{\mathbf G\neq0}(4\pi/G^2)\operatorname{Re}[P^*S]$. That is
the oracle `pure_aft_nuclear`: no libint2, no real-space sum and no
$\mathbf G = 0$ bookkeeping, a construction independent of the split one.
The test compares the split $V$ at three values of $\eta$ with it.

**Gaussian nuclei.** The short-range part needs
$\langle\mu|\operatorname{erfc}(\eta r_C)/r_C|\nu\rangle$ for a point
nucleus. libint2's `erf_nuclear` and `erfc_nuclear` kernels are wrong in the
libint2 versions ferric builds against, so ferric does not use them. Each
nucleus is instead a unit-normalized s Gaussian of exponent
$\zeta = 10^{16}$ (PySCF's `fakemol_for_charges` value), and the integral
comes from the ordinary erfc three-centre engine, with the nucleus translated
by $\mathbf M$ and the partner by $\mathbf L$
(`Engine::compute_eri3_shifted`). A Gaussian charge has potential
$\operatorname{erf}(\sqrt\zeta r)/r$, which differs from $1/r$ only within
about $1/\sqrt\zeta$ of the nucleus. For a pair density $\rho$ the error is

$$ Z\int\rho\,\frac{\operatorname{erfc}(\sqrt\zeta r)}{r} \approx Z\rho(\mathbf R_C)\int\frac{\operatorname{erfc}(\sqrt\zeta r)}{r}\,d^3r = \frac{\pi Z\rho(\mathbf R_C)}{\zeta}, $$

by the same integral as above with $\eta = \sqrt\zeta$. The module
documentation evaluates it for an O 1s AO ($\rho(0) \approx Z^3/\pi \approx 163$):
$4\times10^3/\zeta$, which is $4\times10^{-13}$ Ha at $\zeta = 10^{16}$.

## The algorithm {#nuclear-attraction-algorithm}

| Step | Quantity | Cost |
|---|---|---|
| 1 | $S$, $T$ by lattice-summed shifted one-electron blocks | $N_\text{shell}^2N_{\mathbf L}$ |
| 2 | $V_\text{SR}$: erfc three-centre integrals of $\langle\mu_0|\cdot|\nu_{\mathbf L}\rangle$ with Gaussian nuclei at $\mathbf R_C + \mathbf M$, screened per triplet | $N_\text{shell}^2N_{\mathbf L}N_\text{nuc}$ |
| 3 | $V_\text{LR}$ on the half $\mathbf G$ sphere (weight 2): $-(2/\Omega)\sum v_\eta(\mathbf G)\operatorname{Re}[P^*_{\mu\nu}S(\mathbf G)]$ | $N_\text{pair}N_{\mathbf G}$ |
| 4 | $V_\text{G0} = c_0Z_\text{tot}S$ | $n_\text{ao}^2$ |

Step 3 is the closure that `ferric-pbc/src/hcore.rs::reciprocal_nuclear`
hands to the chunked pair transform, once per $\mathbf G$ of a chunk:

<!-- excerpt: ferric-pbc/src/hcore.rs::reciprocal_nuclear -->
```rust
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let mut kern = 4.0 * PI / g2;
                if let Some(w) = omega {
                    kern *= (-g2 / (4.0 * w * w)).exp();
                }
                // S(G) = Σ_C Z_C e^{−iG·R_C}
                let (mut sre, mut sim) = (0.0_f64, 0.0_f64);
                for (zc, r) in z.iter().zip(&pos) {
                    let ph = gvec[0] * r[0] + gvec[1] * r[1] + gvec[2] * r[2];
                    sre += zc * ph.cos();
                    sim -= zc * ph.sin();
                }
                let f = -2.0 / vol * kern;
                for m in 0..n {
                    for k in 0..n {
                        let pz = p[[m, k, g]];
                        // Re[conj(P) S] = P.re S.re + P.im S.im
                        v[(m, k)] += f * (pz.re * sre + pz.im * sim);
                    }
                }
```

| Code | Equation |
|---|---|
| `kern` | $v_\eta(G) = (4\pi/G^2)e^{-G^2/4\eta^2}$; `omega = None` gives the undamped oracle kernel |
| `sre`, `sim` | $\operatorname{Re}S(\mathbf G)$, $\operatorname{Im}S(\mathbf G)$ |
| `f` | $-2/\Omega$ times the kernel: the half-sphere weight 2 and the attraction sign |
| `pz` | $P_{\mu\nu}(\mathbf G)$ from section @sec:pair-ft |

Step 4 is two lines of `ferric-pbc/src/hcore.rs::periodic_hcore_impl`:

<!-- excerpt: ferric-pbc/src/hcore.rs::periodic_hcore_impl -->
```rust
    let c0 = PI / (omega * omega * cell.volume());
    let v_g0 = (c0 * ztot) * &s;
```

**Why split at all?** The oracle needs no split, but its $\mathbf G$ sphere
must resolve the steepest product exponent, $|\mathbf G| \le 2\sqrt{p_\text{max}\ln(1/\epsilon)}$,
where $p_\text{max}$ is twice the largest exponent. With the split, the
steep part of each pair is handled in real space and the sphere only has to
resolve $e^{-G^2/4\eta^2}$: $|\mathbf G| \le \min(2\eta, 2\sqrt{p_\text{max}})\sqrt{\ln(1/\epsilon)}$.

**The default split.** $\eta = \min(2.5\sqrt\pi/\Omega^{1/3}, \eta_\text{cap})$
(`ferric-pbc/src/hcore.rs::default_hcore_omega`; `site/src/methods/pbc.md`).
Any value gives the same matrix up to truncation; the rule only moves cost
between the real-space triplets and the $\mathbf G$ sphere. The cap
$\eta_\text{cap} = \eta_\text{gdf}\sqrt{\ln(1/p_\text{gdf})/\ln(1/\epsilon)}$,
with the RS-GDF defaults $\eta_\text{gdf} = 1$ and $p_\text{gdf} = 10^{-13}$,
is 0.9636 at $\epsilon = 10^{-14}$. It keeps every nuclear-attraction
$\mathbf G$ inside the sphere that the density fit of section @sec:rsgdf already
walks. A run that sets its own `gdf_omega` caps with that value instead.

## Numerical anchors {#nuclear-attraction-numerical-anchors}

| Quantity | Reference | Test |
|---|---|---|
| $V(\eta)$ at three $\eta$ | the pure-AFT oracle `pure_aft_nuclear` | `crates/ferric-pbc/tests/pbc_hcore.rs` `periodic_hcore_is_omega_independent_triclinic_sp` |
| one s primitive | the closed form $V_\text{erf} = -(2/\sqrt\pi)(1/\eta^2 + 1/p + 1/\zeta)^{-1/2}$ | `single_s_primitive_matches_closed_form` |
| erf + erfc on a Gaussian nucleus (`ferric-pbc/src/hcore.rs::gaussian_nucleus_attraction`) | the point nucleus | `gaussian_nucleus_erf_plus_erfc_is_point_nuclear` |
| H$_2$/STO-3G at $a = 4$ | the Python prototype | `periodic_hcore_matches_prototype_h2_sto3g_a4` |

## Sources {#nuclear-attraction-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-pbc/theory.md`, §3 | the split and its constant |
| `scripts/pbc_ewald_pairft_stress_sympy.py`, part S3 | the proof |
| Q. Sun *et al.*, *J. Chem. Phys.* **153**, 024109 (2020) | PySCF's periodic conventions |
