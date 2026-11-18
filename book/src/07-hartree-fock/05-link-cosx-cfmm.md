# LinK, seminumerical exchange (COSX) and CFMM {#link-cosx-cfmm}

<!-- status: ready; registry: rhf, uhf, rohf -->

In a large molecule most four-centre contributions to $J$ and $K$ are
negligible, and each builder of this section uses a different reason. In a
molecule with a gap the density matrix decays with distance, and exchange
couples two functions only through the density between their partners, so
LinK keeps the exact exchange integrals and visits fewer of them. One of the
two electron integrations of an exchange integral can be done on a grid,
which COSX uses to turn the four-index problem into three-index potentials.
Two distant charge clouds interact through their multipoles, so CFMM splits
the Coulomb matrix into a near field computed exactly and a far field from
multipoles.

This section derives LinK's pair lists and COSX's overlap fit (@sec:link-cosx-cfmm-link-cosx), states
what CFMM computes (@sec:link-cosx-cfmm-cfmm), and then gives the measured errors and costs. The
three builders differ in what they give up:

| Builder | Replaces | Exact? | `[scf]` setting |
|---|---|---|---|
| LinK | $K$ | yes, to the screening threshold | `k_builder = "link"` |
| COSX | $K$ | no: grid error, reduced by the overlap fit and the final pass | `k_builder = "cosx"` |
| CFMM | $J$ | exact in the all-near-field limit; multipole error otherwise | library only (feature `cfmm-incomplete`) |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/link_k.rs::LinkK`, `ferric-scf/src/cosx_k.rs::CosxK`, `ferric-scf/src/cfmm.rs::CfmmJ` |
| CLI | `[scf] k_builder = "link"` or `"cosx"`, `cosx_grid`, `cosx_final_pass`, `cosx_final_grid` |
| Python | `k_builder=` and the `cosx_*` keywords on the SCF entry points (`crates/ferric-python/ferric.pyi`) |

## LinK and COSX {#link-cosx-cfmm-link-cosx}

### LinK

LinK (`LinkK`, `crates/ferric-scf/src/link_k.rs`) uses the same quartets and
scatter as the direct builder, for $K$ only, with two pair lists in front of
the quartet test. Write $Q_{ij}$ for the Schwarz factor of shells $i, j$ and
$q_x = \max_yQ_{xy}$ for the largest factor of shell $x$. The lists are

- significant pairs (geometry only): $j \in \mathrm{sp}(i)$ iff $Q_{ij}\,Q_{\max} > \tau$;
- density pairs (every iteration): $\sigma \in \mathrm{dp}(j)$ iff
  $D^{\rm blk}_{j\sigma}\,q_j\,q_\sigma > \tau$, with $D^{\rm blk}$ the
  shell-block maximum of $|D|$.

The bra runs over $(i, j)$ with $j\in\mathrm{sp}(i)$, the third shell over
$\mathrm{dp}(i)\cup\mathrm{dp}(j)$, the fourth over $\mathrm{sp}$ of the
third. A quartet, canonicalized and deduplicated, is skipped when

$$ Q_{s_1s_2}Q_{s_3s_4}\,\max\big(D^{\rm blk}_{13}, D^{\rm blk}_{14}, D^{\rm blk}_{23}, D^{\rm blk}_{24}\big) < \tau, $$

the four exchange pairings only (`DensityScreen::FourPairK`). The direct
builder's test uses six pairings because $J$ also reads $D_{12}$ and $D_{34}$;
$K$ does not, so LinK's test is tighter.

Why can the lists drop nothing the test would keep? Take one exchange pairing,
say $D^{\rm blk}_{23}$. Since $Q_{s_1s_2} \le q_{s_2}$ and
$Q_{s_3s_4} \le q_{s_3}$ by the definition of $q$,

$$ Q_{s_1s_2}Q_{s_3s_4}D^{\rm blk}_{23} \le q_{s_2}\,q_{s_3}\,D^{\rm blk}_{23} . $$

So a quartet survives on that pairing only if $s_3 \in \mathrm{dp}(s_2)$. The
same holds for the other three pairings. The lists are necessary conditions of
this test when the bound factorizes as $Q_{12}Q_{34}$ (Schwarz), so LinK is
direct $K$ to the threshold. The skip rule is exact; the asymptotic count of
surviving quartets is a property of the density, not of the code. For a
density that decays with distance, $\mathrm{dp}(j)$ holds the shells near $j$,
and the third-shell loop shrinks from all shells to a neighbourhood.

On water in cc-pVDZ (the `589b50832` build) LinK reproduces direct $K$ to
rounding, while COSX at its default grids carries a grid error of a few
times $10^{-8}$ Ha:

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")

e_direct = ferric.run_rhf(mol, bs).energy
e_link = ferric.run_rhf(mol, bs, k_builder="link").energy
e_cosx = ferric.run_rhf(mol, bs, k_builder="cosx").energy
print(f"direct         {e_direct:.10f}")
print(f"LinK - direct  {e_link - e_direct:.1e}")
print(f"COSX - direct  {e_cosx - e_direct:.1e}")
```

```text
direct         -76.0267679974
LinK - direct  9.9e-14
COSX - direct  -5.2e-08
```

### COSX

COSX (`CosxK`, `crates/ferric-scf/src/cosx_k.rs`) integrates one electron of
each exchange integral on a Becke–Lebedev grid and the other analytically.
Put the grid on electron 1, with points $\mathbf r_g$ and weights $w_g$:

$$ K_{\mu\nu} = \sum_{\lambda\sigma}D_{\lambda\sigma}\int\!\!\int\frac{\chi_\mu(\mathbf r_1)\chi_\sigma(\mathbf r_1)\,\chi_\nu(\mathbf r_2)\chi_\lambda(\mathbf r_2)}{r_{12}} \approx \sum_gX_{\mu g}\,G_{\nu g}, $$

$$ X_{\mu g} = \sqrt{w_g}\,\chi_\mu(\mathbf r_g), \quad F_{\lambda g} = \sum_\sigma D_{\lambda\sigma}X_{\sigma g}, \quad A_{\nu\lambda}(g) = \int\frac{\chi_\nu\chi_\lambda}{|\mathbf r - \mathbf r_g|}\,d\mathbf r, \quad G_{\nu g} = \sum_\lambda A_{\nu\lambda}(g)F_{\lambda g} . $$

Read it right to left. $F_{\lambda g}$ contracts the density with the
electron-1 function $\chi_\sigma$ at the point. $A(g)$ is the potential at
$\mathbf r_g$ of the electron-2 product $\chi_\nu\chi_\lambda$, a
three-centre one-electron integral (section @sec:md-three-center). $G$ applies that potential,
and the outer sum over points finishes the electron-1 integral. The weight is
split as $\sqrt{w_g}$ on each side because $F$ contains one factor of $X$.

$A(g)$ comes from the McMurchie–Davidson kernel (`cosx_backend = "md3c1e"`)
and is never assembled as a matrix: each shell pair's block is added into
$G$ for both orderings of the pair (`accumulate_pair`). The plain result is
$K = \tfrac12(\tilde K + \tilde K^{\mathsf T})$ with $\tilde K = XG^{\mathsf T}$.

**The overlap fit.** The grid makes an error even in the simplest case. Let
the potential be constant over the molecule, $A_{\nu\lambda}(g) = cS_{\nu\lambda}$.
Then $G = cSF = cSDX$, and

$$ \tilde K = XG^{\mathsf T} = c\,XX^{\mathsf T}DS = c\,S_{\rm num}DS, \qquad S_{\rm num} = XX^{\mathsf T}, $$

where $S_{\rm num}$ is the overlap matrix integrated on the grid. The exact
answer is $cSDS$, so the plain result carries the grid's overlap error. The
overlap fit (`cosx_overlap_fit`, default on) replaces the left factor by its
projection that makes the numerical overlap exact,

$$ K = \tfrac12\big(Q\tilde K + (Q\tilde K)^{\mathsf T}\big), \qquad Q = S\,S_{\rm num}^{-1}, $$

computed with a Cholesky factor of $S_{\rm num}$ (built once per geometry).
For the constant potential $Q\tilde K = SS_{\rm num}^{-1}cS_{\rm num}DS = cSDS$,
exact on any grid: the fit removes the error of the grid's overlap. As the
grid becomes exact $S_{\rm num}\to S$, $Q \to 1$, and both forms tend to the
exact $K$.

> **Verified equation {#eq:scf-cosx-overlap-fit}** · the COSX overlap fit
>
> For a potential constant over the molecule, $A_{\nu\lambda}(g) = cS_{\nu\lambda}$,
> the fitted $K = \operatorname{sym}(SS_{\rm num}^{-1}\tilde K)$ equals the exact
> $cSDS$ on any grid, while the plain $\operatorname{sym}(\tilde K)$ equals
> $cS_{\rm num}DS$. On one $s$ Gaussian the grid sum tends to
> $(ss|ss) = 2\sqrt{\alpha/\pi}$.
>
> | | |
> |---|---|
> | Proof | `scripts/scf_jk_builders_sympy.py`, claims C1 and C2 |
> | Negative control | the fit replaced by the plain form (`qk = k_tilde`) must fail C2 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/cosx_k.rs::{CosxK,finalize_fitted}` |
>
> <!-- verified: id=scf.cosx-overlap-fit; script=scripts/scf_jk_builders_sympy.py; claim=C2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/cosx_k.rs::{CosxK,finalize_fitted}; registry=rhf -->

**Screening.** A shell pair is dropped for a sub-batch of points when
$\mathrm{est}\cdot\max(f_{s_1}, f_{s_2}) < \tau_{\rm COSX}$. Here est is an
upper bound on $\max_g|A^{(g)}|$ over the region (the minimum of a sphere and
a box bound, both never below the true value), and $f_s = \max|F_{\mu g}|$
over the shell and the points. Each dropped pair perturbs an element of $G$ by
at most $\tau_{\rm COSX}$ per component it contributes, so the bound on one
element is the sum over the dropped partners. $\tau_{\rm COSX} = 0$ is bitwise
the unscreened build. The default sparse half transform keeps only the shells
whose $|X|$ and $|D|$ blocks exceed $10^{-10}$; zero thresholds are bitwise
the dense build.

**Final pass.** With `cosx_final_pass` the converged energy is corrected
once, non-self-consistently, on a larger grid:

$$ E = E_{\rm SCF} + \sum_sc_s\big(\operatorname{tr}[D_sK_{\rm final}(D_s)] - \operatorname{tr}[D_sK_{\rm SCF}(D_s)]\big), $$

with $c = -c_x/4$ for the closed shell and $-c_x/2$ per spin. These are the
exchange weights of the energy functionals of section @sec:energy-functionals, so the correction
replaces the exchange energy of the SCF grid by that of the final grid at the
same density. The SCF iterates on the cheaper grid, and the reported energy
uses the larger one.

`scripts/scf_jk_builders_sympy.py` proves claims C1 (one $s$ Gaussian: the
grid sum tends to $(ss|ss) = 2\sqrt{\alpha/\pi}$) and C2 (the fit is exact for
a constant potential, the plain form is not).

### The code

The fitted COSX exchange matrix is two triangular solves against the
Cholesky factor of $S_{\rm num}$ and one product with $S$:

<!-- excerpt: ferric-scf/src/cosx_k.rs::finalize_fitted -->
```rust
let qk = s_ao.dot(&z);
    k.assign(&qk);
    *k += &qk.t();
    *k *= 0.5;
```

| Code | Equation |
|---|---|
| `z` | $S_{\rm num}^{-1}\tilde K$, from `fac.l` and `fac.lt` |
| `s_ao` | the analytic overlap $S$ |
| `qk` | $Q\tilde K = SS_{\rm num}^{-1}\tilde K$ |
| `k` | $\tfrac12(Q\tilde K + (Q\tilde K)^{\mathsf T})$ |

The two triangular solves apply $S_{\rm num}^{-1}$ without forming it.

LinK's per-quartet screen uses the exchange pairings only:

<!-- excerpt: ferric-scf/src/link_k.rs::build -->
```rust
let screen = crate::quartet_scatter::DensityScreen::FourPairK(&d_max_shell);
```

`d_max_shell` is the same shell-blocked table of $\max|D|$ that the direct
builders use; only the set of pairings differs. The density pair list that
bounds the ket loop (`ferric-scf/src/pairs.rs::DensityPairs`) uses the same
threshold as that screen, so it drops only pairs no surviving quartet could
need. `scatter_bra_pair_probed` debug-asserts that `FourPairK` is never paired
with a mode that accumulates $J$, since it omits the two pairings the
Coulomb contraction reads.

## CFMM {#link-cosx-cfmm-cfmm}

This subsection states what `CfmmJ` (`crates/ferric-scf/src/cfmm.rs`,
feature `cfmm-incomplete`) computes.

**The idea.** Two charge clouds far apart interact through their multipoles.
The direct build costs $O(N^4)$ in the basis size; the far field is where a
multipole method saves. The textbook form is the spherical multipole
expansion of a cloud's potential,
$J(\mathbf r) \approx \sum_{l,m} M_{lm}\,Y_{lm}(\theta,\phi)/r^{l+1}$.
`cfmm.rs` does not use spherical harmonics: it expands in primitive
Cartesian multipoles (below). The two carry the same terms order by order,
because the degree-$n$ part of the Taylor expansion of
$1/\lvert\mathbf R-\mathbf r'\rvert$ in $\mathbf r'$ is
$r'^{\,n}P_n(\cos\gamma)/R^{n+1}$, the $l=n$ spherical term. A hierarchical FMM (M2M/L2L translations) uses that to make the far-field
Coulomb build $O(N)$. ferric's module is behind the off-by-default
`cfmm-incomplete` feature: its far field is a flat leaf-to-leaf M2L with no
M2M/L2L pass, so it does not have that asymptotic scaling, and it is slower
than `DirectJ`. It reproduces the direct $J$ to about $10^{-14}$ in the
all-near-field limit.

**Shell pairs.** Every ordered shell pair $(s_1, s_2)$ is placed at the
weighted centre of its primitive products, $\sum w\,\mathbf P/\sum w$ with
$w = |c_1c_2|\,e^{-\alpha\beta|\mathbf{AB}|^2/p}$ (`pair_center`). Its extent
is $\max\sqrt{-\ln\epsilon/p}$ over primitives whose prefactor is at least
$\epsilon$ (`extent_thresh`, default $10^{-10}$): the radius beyond which a
Gaussian of exponent $p$ has fallen below $\epsilon$. The pairs are binned
into the leaves of an octree of uniform depth `max_level` (default 3) whose
root is the bounding cube of the pair centres enlarged by 1.1.

**Near and far.** A pair of leaves $(a, b)$ is far when

$$ |\mathbf C_a - \mathbf C_b| > e_a + e_b + w_s\,h, $$

with $e_x$ the largest pair extent in leaf $x$, $h$ the leaf width and
$w_s$ = `ws_factor` (default 2). The extents keep the two clouds from
overlapping, which a multipole expansion requires; $w_sh$ adds a margin in
units of the box size. A near pair is contracted exactly,
$J_{\mu\nu} \mathrel{+}= \sum(\mu\nu|\lambda\sigma)D_{\lambda\sigma}$ over
every bra pair in $a$ and ket pair in $b$, with no permutational folding and
no screening. A far pair goes through Cartesian multipoles about the leaf
centres:

$$ M^b_{ijk} = \sum_{\lambda\sigma\in b}D_{\lambda\sigma}\int\chi_\lambda\chi_\sigma\,(\mathbf r - \mathbf C_b)^{ijk}\,d\mathbf r, $$

$$ L^a_{ijk} = \frac{1}{i!\,j!\,k!}\sum_{i'+j'+k' \le l_{\max}}M^b_{i'j'k'}\,\frac{(-1)^{i'+j'+k'}}{i'!\,j'!\,k'!}\,\partial_x^{i+i'}\partial_y^{j+j'}\partial_z^{k+k'}\frac{1}{r}\Big|_{\mathbf C_a - \mathbf C_b}, $$

$$ J_{\mu\nu} \mathrel{+}= \sum_{ijk}L^a_{ijk}\,M^{\mu\nu}_{ijk}(\mathbf C_a) $$

(`add_m2l`, $l_{\max}$ default 6). $M^b$ are the multipoles of the density
in leaf $b$, $L^a$ the Taylor coefficients of their potential about the
centre of leaf $a$, and the last line takes that potential between
$\chi_\mu$ and $\chi_\nu$. This is one flat level of leaf-to-leaf M2L; there
is no M2M or L2L pass and the far-field work is $O(n_{\mathrm{leaf}}^2)$.

**Exact limit.** Each pair sits in exactly one leaf, so with every leaf pair
treated as near (`force_all_near_field`) every quartet is counted once and $J$
is the exact Coulomb matrix (`cfmm_matches_direct_j_in_the_trivial_limit`,
$10^{-10}$). With the far field engaged the error falls with $l_{\max}$
(`cfmm_far_field_engaged_matches_direct_j_on_alkane10`). `CfmmConfig::thresh`
is not read by the code: the near field is unscreened.

The near/far decision is one comparison per leaf pair:

<!-- excerpt: ferric-scf/src/cfmm.rs::is_far_ext -->
```rust
dist > ext_a + ext_b + self.cfg.ws_factor * a.width
```

| Code | Equation |
|---|---|
| `dist` | $\lvert\mathbf C_a - \mathbf C_b\rvert$ |
| `ext_a`, `ext_b` | the largest pair extents $e_a$, $e_b$ in each leaf |
| `ws_factor` | $w_s$, default 2 |
| `a.width` | the leaf width $h$ |

With `force_all_near_field` the function returns `false` before this line,
which is the exactness anchor.

CFMM is not reachable from the CLI or Python. It has its exactness anchor
(`cfmm_matches_direct_j_in_the_trivial_limit` in `crates/ferric-scf/src/cfmm.rs`)
and one far-field test, and no production caller.

## Numerical anchors {#link-cosx-cfmm-numerical-anchors}

| Check | Reference | Agreement | Test |
|---|---|---|---|
| LinK SCF energy, butane / def2-SVP | direct $K$ | 9e-12 Ha | `crates/ferric-scf/tests/link_scf_anchor.rs` |
| COSX, dense-grid limit, water / cc-pVDZ | direct $K$ | 3.3e-7 | `crates/ferric-scf/tests/cosx_k_anchors.rs` |

<!-- figures: site/src/reference/validation.md "COSX exchange, dense-grid limit"; site/src/methods/scf.md "Choosing how exchange is built" -->

The energy error of COSX at its defaults, from
`site/src/methods/scf.md`, "Choosing how exchange is built":

| Grid | System / basis | Error against exact exchange |
|---|---|---:|
| default (sgx (35,194), final pass on sgx (50,302)) | water / aug-cc-pVDZ | −8.1e-9 Ha |
| default | butane / def2-SVP | −3.4e-5 Ha |
| default | butane / def2-TZVP | −4.3e-6 Ha |
| flat (50,110) | butane / def2-SVP | 1.7e-4 Ha |

and the measured cost on butane, one thread: one COSX $K$ build takes 90 s at
def2-QZVP against 400 s for one direct $J$ + $K$ sweep, and a full COSX SCF
takes 358 s at def2-TZVP against 98 s for the direct SCF. COSX pays only where
its per-build saving outweighs its larger iteration count and grid cost; the
same page records the LinK timings as being re-measured.

## Options that change the equation {#link-cosx-cfmm-options-change-equation}

| Option | Effect |
|---|---|
| `k_builder` | `"link"`, `"cosx"`; unset is the direct builder |
| `cosx_grid` | the SCF grid |
| `cosx_final_pass`, `cosx_final_grid` | one non-self-consistent energy correction on a larger grid |
| `cosx_overlap_fit` | the $S S_{\rm num}^{-1}$ fit, on by default |

## Doc fixes needed {#link-cosx-cfmm-doc-fixes-needed}

| Text | Code |
|---|---|
| `LinkK` struct doc: ket shells are the "intersection" of the two lists | the third shell runs over the union $\mathrm{dp}(i) \cup \mathrm{dp}(j)$ |
| the COSX pair-screen doc bounds each element's error by the threshold | the bound is the sum over the dropped partners |
| `CfmmConfig::thresh` documented as a near-field screen | the field is never read |
| the CFMM separation criterion in its docs | the code adds both extents and $w_s h$ (excerpt above) |

## Sources

| Source | Used for |
|---|---|
| `scripts/scf_jk_builders_sympy.py` | verified equation @eq:scf-cosx-overlap-fit |
| `site/src/methods/scf.md`, "Choosing how exchange is built" | the measured errors and times |
| `site/src/reference/validation.md` | the COSX anchor |

The plain-form step $\tilde K = cS_{\rm num}DS$ and the pairing inequality of
the LinK lists are written out here from the cited claims and definitions.
