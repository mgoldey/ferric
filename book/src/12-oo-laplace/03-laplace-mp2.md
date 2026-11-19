# Laplace RI-MP2 and Laplace SOS-MP2 {#laplace-mp2}

<!-- status: ready; registry: laplace-mp2, laplace-sos-mp2 -->

*Notation follows chapter @sec:notation. $t$ is the Laplace variable (chapter @sec:oo-laplace).*

Every MP2 term divides by $\Delta_{ij}^{ab} = (\varepsilon_a - \varepsilon_i) + (\varepsilon_b - \varepsilon_j)$,
a sum of two pair energies, while the fitted numerator is a product of an
$(i,a)$ factor and a $(j,b)$ factor (section @sec:closed-shell-ri-mp2-density-fitting-making-affordable). The sum ties the two
pairs together, which is why RI-MP2 visits every $(ia,jb)$ combination. An
exponential turns the sum into a product,
$e^{-t\Delta} = e^{-t(\varepsilon_a - \varepsilon_i)}\,e^{-t(\varepsilon_b - \varepsilon_j)}$,
so writing $1/\Delta$ as an integral of $e^{-t\Delta}$ over $t$ splits every
term at each $t$. A quadrature of three to seven points then turns the
energy into a few RI-MP2-like contractions with weighted factors, and the
opposite-spin energy no longer needs any object that carries both pairs.
The price is a quadrature error, and one pass over the factors per point.

The code uses the rewrite in two ways:

- **Exactly**, in the MO and AO bases (this section). It is the graded
  method.
- **Truncated**, with localized orbitals (section @sec:local-laplace-sos).

Both kinds are closed shell (RHF) and energy only.

The plan:

1. Build the quadrature for $1/x$ and bound its error (@sec:laplace-mp2-quadrature).
2. Factorize the energy at fixed $t$, in the MO basis (@sec:laplace-mp2-factorized-energy) and in the
   AO basis (@sec:laplace-mp2-ao-form-coulomb-term).
3. Drop the same-spin term and see what that saves (@sec:laplace-mp2-sos-mp2).
4. Read the algorithm, the kernel, the memory rules and the anchors.

| Index | Range | Code name |
|---|---|---|
| $i, j$; $a, b$ | active occupied; virtual | `0..nocc` offset `frozen_core`; `0..nvir` offset `nocc_total` |
| $\mu, \nu$; $P, Q$ | AO; auxiliary | `0..nbas`; `0..naux` |
| $k$ | quadrature point | `points`, `weights` of `LaplaceMp2` |

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/laplace.rs::laplace_ri_mp2` (full MP2); `ferric-mp2/src/laplace.rs::laplace_sos_mp2`, `laplace_sos_mp2_with_basis` (SOS) |
| CLI | `method.kind = "laplace-mp2"` (`examples/water-laplace-rimp2.toml`); `method.kind = "laplace-sos-mp2"` (`examples/water-laplace-sos-mp2.toml`); keys `n_quad`, `frozen_core`, `c_os`, `sos_formulation`, `domain_cutoff_bohr` in `[mp2]`, `[memory] budget_gb` |
| Python | `run_laplace_mp2`, `run_laplace_sos_mp2` (`crates/ferric-python/ferric.pyi`) |

## The quadrature {#laplace-mp2-quadrature}

For $x > 0$, integrate the exponential: $\int_0^\infty e^{-tx}\,dt = \big[-e^{-tx}/x\big]_0^\infty = 1/x$.
Replace the integral by a sum over $n_q$ points:

$$ \frac1x = \int_0^\infty e^{-tx}\,dt \;\approx\; \sum_{k=1}^{n_q} w_k\,e^{-t_kx}, \qquad x \in [y_{\min}, y_{\max}] . \label{eq:laplace-identity} $$

The rule has to serve every denominator at once, and the denominators
fill a whole interval. The **minimax** rule chooses the $n_q$ nodes and
$n_q$ weights to minimize the largest error over that interval.

**The tables.** The nodes and weights come from the tables of Takatsuka,
Ten-no and Hackbusch, for $1/y$ on $[1, R]$
(`ferric-quadrature/src/minimax.rs::select_minimax_points`). ferric
fits no minimax rule itself: it selects a precomputed entry at run time.
Each entry gives
exponents $\tau_k$ and weights $\omega_k$ that minimize

$$ \delta(R) = \max_{y\in[1,R]}\Big|\frac1y - \sum_k\omega_ke^{-\tau_ky}\Big| . $$

The error curve of a minimax fit alternates in sign, with $2n_q + 1$ extrema
of equal size $\delta(R)$. Every shipped entry shows this. For example, the
$n_q = 7$ entry with $R = 50$ has $\delta = 3.1\times10^{-6}$. The script
below counts 169 and 17 extrema for the two smallest-$R$ entries of
$n_q = 7$, where $\delta$ is below $10^{-7}$. Those counts come from the
count, not the fit. On the 200 001-point grid the step-to-step change of
the error near an extremum falls to the $10^{-16}$ rounding of the f64
evaluation, so the sign test picks up spurious changes. The tables carry 17
significant digits, and the same two curves evaluated in 40-digit
arithmetic on 20 001 points have 15 extrema each.

**The rescaling.** The tables cover $[1, R]$, and the problem needs
$[y_{\min}, y_{\max}]$. Substitute $x = y_{\min}\,y$:

$$ \frac1x = \frac1{y_{\min}}\cdot\frac1y \approx \sum_k\frac{\omega_k}{y_{\min}}\,e^{-\tau_k x/y_{\min}} . $$

So `LaplaceQuadrature::new` sets

$$ R = \frac{y_{\max}}{y_{\min}}, \qquad t_k = \frac{\tau_k}{y_{\min}}, \qquad w_k = \frac{\omega_k}{y_{\min}} . \label{eq:laplace-mp2-2} $$

The same substitution scales the error: the largest absolute error of the
rescaled rule on $[y_{\min}, R\,y_{\min}]$ is $\delta(R)/y_{\min}$. The
relative error $x\,\lvert 1/x - \sum_kw_ke^{-t_kx}\rvert$ is largest at the
top of the interval, where it reaches $R\,\delta(R)$. Only the ratio $R$
decides which table entry is needed.

Both claims can be checked from the tables alone. The script below needs
numpy and the source tree, not the extension; it reads `minimax.rs` at the
`589b50832` commit, counts the extrema of each error curve on a fine grid,
and compares the rescaled error with $1/y_{\min}$ for $y_{\min} = 0.37$.
The two smallest-$R$ entries of $n_q = 7$ show the counting artifact:

```python
# Standalone: needs only numpy and the ferric source tree (reads crates/ferric-quadrature/src/minimax.rs).
import re
import numpy as np

src = open("crates/ferric-quadrature/src/minimax.rs").read()

def table(name):
    blk = src.split(f"static {name}")[1].split("];\n\n")[0]
    num = lambda s: np.array([float(x) for x in re.findall(r"[0-9.]+e[+-]\d+", s)])
    return [(float(r), num(t), num(w))
            for r, t, w in re.findall(r"\(\s*([0-9.e+-]+),\s*&\[(.*?)\],\s*&\[(.*?)\],\s*\)", blk, re.S)]

for name, nq in (("MINIMAX_K3", 3), ("MINIMAX_K5", 5), ("MINIMAX_K7", 7)):
    for R, tau, om in table(name):
        y = np.geomspace(1.0, R, 200001)
        err = 1.0 / y - (om[None, :] * np.exp(-np.outer(y, tau))).sum(1)
        d = np.diff(err)
        ext = np.r_[0, np.where(np.sign(d[1:]) != np.sign(d[:-1]))[0] + 1, len(err) - 1]
        print(f"n_q={nq} R={R:7.1f} delta={abs(err).max():.2e} extrema={len(ext)} (expect {2*nq+1})")

R, tau, om = table("MINIMAX_K7")[3]
ymin = 0.37
x = np.geomspace(ymin, R * ymin, 20001)
e_x = abs(1 / x - (om / ymin * np.exp(-np.outer(x, tau / ymin))).sum(1)).max()
y = x / ymin
e_y = abs(1 / y - (om * np.exp(-np.outer(y, tau))).sum(1)).max()
print(f"rescaled max error / table max error = {e_x / e_y:.6f}  (1/y_min = {1 / ymin:.6f})")
```

```text
n_q=3 R=    5.0 delta=1.50e-04 extrema=7 (expect 7)
n_q=3 R=   10.0 delta=7.15e-04 extrema=7 (expect 7)
n_q=3 R=   20.0 delta=1.82e-03 extrema=7 (expect 7)
n_q=3 R=   50.0 delta=3.66e-03 extrema=7 (expect 7)
n_q=3 R=  100.0 delta=4.79e-03 extrema=7 (expect 7)
n_q=5 R=    5.0 delta=2.54e-07 extrema=11 (expect 11)
n_q=5 R=   10.0 delta=4.24e-06 extrema=11 (expect 11)
n_q=5 R=   20.0 delta=2.52e-05 extrema=11 (expect 11)
n_q=5 R=   50.0 delta=1.11e-04 extrema=11 (expect 11)
n_q=5 R=  100.0 delta=2.27e-04 extrema=11 (expect 11)
n_q=5 R=  500.0 delta=5.52e-04 extrema=11 (expect 11)
n_q=5 R= 1000.0 delta=6.38e-04 extrema=11 (expect 11)
n_q=7 R=    5.0 delta=4.01e-10 extrema=169 (expect 15)
n_q=7 R=   10.0 delta=2.34e-08 extrema=17 (expect 15)
n_q=7 R=   20.0 delta=3.25e-07 extrema=15 (expect 15)
n_q=7 R=   50.0 delta=3.12e-06 extrema=15 (expect 15)
n_q=7 R=  100.0 delta=9.84e-06 extrema=15 (expect 15)
n_q=7 R=  500.0 delta=4.84e-05 extrema=15 (expect 15)
n_q=7 R= 1000.0 delta=7.15e-05 extrema=15 (expect 15)
rescaled max error / table max error = 2.702703  (1/y_min = 2.702703)
```

**The range.** Every driver sets

$$ y_{\min} = 2(\varepsilon_{\mathrm{LUMO}} - \varepsilon_{\mathrm{HOMO}}), \qquad y_{\max} = 2(\varepsilon_{n_{\mathrm{mo}}} - \varepsilon_1) . $$

The smallest denominator excites two electrons from the HOMO into the
LUMO, which gives $y_{\min}$. No denominator exceeds twice the full orbital
energy spread, which gives $y_{\max}$. $\varepsilon_1$ is the lowest orbital
energy, a frozen-core orbital when `frozen_core > 0`. The range is then
wider than the active $\Delta_{ij}^{ab}$ need, which is safe.

**The error in the energy.** Take $E_{\mathrm{OS}} = -\sum(ia|jb)^2/\Delta$.
Every numerator is non-negative, so the triangle inequality bounds the
quadrature error of $E_{\mathrm{OS}}$ by

$$ \big|E_{\mathrm{OS}}^{\mathrm{quad}} - E_{\mathrm{OS}}\big| \le \frac{\delta(R_{\mathrm{tab}})}{y_{\min}}\sum_{ijab}(ia|jb)^2 , $$

provided every $\Delta_{ij}^{ab}$ lies in the fitted range
$[y_{\min}, R_{\mathrm{tab}}\,y_{\min}]$ of the chosen table entry. A small gap
makes $y_{\min}$ small and loosens the bound. A wide orbital spectrum makes
$R$ large and forces a table entry with a larger $\delta$.

The rules for the table:

- `n_quad` must be 3, 5 or 7. Any other value is an error, not a rounding.
- The table entry used is the first with
  $R_{\mathrm{tab}} \ge 0.99\,R$. An entry up to 1% short of the true range
  is accepted, to absorb round-off. Denominators in that last 1% lie outside
  the fitted range, and the bound above does not cover them.
- $R$ beyond the largest entry is an error. The largest entry is 100 for
  `n_quad` = 3 and 1000 for 5 and 7.
- The CLI default is `n_quad` = 7.

> **Verified equation @eq:laplace-identity** · Laplace identity
>
> $\int_0^\infty e^{-tx}dt = 1/x$ for $x > 0$. The energy assembled from
> (@eq:laplace-factorization), integrated over $t$ by an adaptive quadrature, equals the
> brute-force spin-orbital MP2 energy (to about $2\times10^{-15}$ on the
> script's model).
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_family_sympy.py`, part F |
> | Negative control | the half-exponent of $B(t)$ changed from $\tfrac12$ to 1 must fail ("FAIL F") |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/laplace.rs::weighted_b_mo` |
> | Code test | `laplace_mp2_h2o_ccpvdz_vs_pyscf_dfmp2` in `crates/ferric-mp2/tests/validation_laplace_mp2.rs` (and its four siblings): the production energy against exact DF-MP2 and against a quadrature-isolated numpy reference (section @sec:laplace-mp2-anchors) |
>
> <!-- verified: id=laplace.identity; script=scripts/mp2_family_sympy.py; claim=F; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/laplace.rs::weighted_b_mo; rust_test=ferric-mp2/tests/validation_laplace_mp2.rs::laplace_mp2_h2o_ccpvdz_vs_pyscf_dfmp2 -->

The proof covers the exact integral. The minimax weights are tabulated data
and no committed script proves them. Their error in the energy is measured
instead (section @sec:laplace-mp2-anchors). The equioscillation, the $\delta = 3.1\times10^{-6}$
figure and the rescaled error $\delta/y_{\min}$ above were checked
numerically on the tables in `minimax.rs` for this section.

## The factorized energy {#laplace-mp2-factorized-energy}

At a fixed $t$, split the exponential:
$e^{-t\Delta_{ij}^{ab}} = e^{-t(\varepsilon_a - \varepsilon_i)}\,e^{-t(\varepsilon_b - \varepsilon_j)}$.
Give half of each pair's factor to each fitted factor $B^P_{ia}$ of (@eq:closed-shell-ri-mp2-4).
Then a product $B^P_{ia}(t)B^P_{ia}(t)$ carries the whole factor
$e^{-t(\varepsilon_a - \varepsilon_i)}$:

$$ B^P_{ia}(t) = B^P_{ia}\,e^{-t(\varepsilon_a - \varepsilon_i)/2}, \qquad J_{PQ}(t) = \sum_{ia} B^P_{ia}(t)B^Q_{ia}(t), \qquad G_{PQ}[i,j]{}(t) = \sum_a B^P_{ia}(t)\,B^Q_{ja}(t). $$

Now expand the two sums of (@eq:closed-shell-ri-mp2-3) at fixed $t$.

- **The Coulomb term.** Write $(ia|jb)^2 = \sum_{PQ}B^P_{ia}B^P_{jb}B^Q_{ia}B^Q_{jb}$
  and attach the exponentials. The $(i,a)$ factors collect into $J_{PQ}(t)$
  and the $(j,b)$ factors into a second copy of it:
  $\sum_{ijab}(ia|jb)^2e^{-t\Delta} = \sum_{PQ}J_{PQ}(t)^2$.
- **The exchange term.** In $(ia|jb)(ib|ja)$ the occupied index $i$ meets
  $a$ in the first integral and $b$ in the second. Collect the sum over $a$
  first: it pairs $i$ with $j$ through $a$, which is $G_{PQ}[i,j]$. The sum
  over $b$ gives $G_{PQ}[j,i]$. So
  $\sum_{ijab}(ia|jb)(ib|ja)e^{-t\Delta} = \sum_{PQ}\sum_{ij}G_{PQ}[i,j]\,G_{PQ}[j,i]$.

Integrate over $t$ with the rule (@eq:laplace-identity):

$$ E^{(2)} \approx -\sum_k w_k\Big[\,2\sum_{PQ}J_{PQ}(t_k)^2 - \sum_{PQ}\sum_{ij} G_{PQ}[i,j]{}(t_k)\,G_{PQ}[j,i]{}(t_k)\Big], \qquad E_{\mathrm{OS}} \approx -\sum_k w_k\sum_{PQ}J_{PQ}(t_k)^2 . \label{eq:laplace-factorization} $$

In the split of (@eq:mp2-cs-spin-adaptation), $E_{\mathrm{OS}}$ is the $J^2$ term and
$E_{\mathrm{SS}}$ is $J^2$ minus the $G\cdot G$ term.

Check the limit $t = 0$. All exponentials are 1, $B(0) = B$, and
$\sum_{PQ}J_{PQ}(0)^2 = \sum_{ijab}(ia|jb)^2$: the numerator of
$E_{\mathrm{OS}}$ summed without its denominator. The Laplace weights then
supply $1/\Delta$ point by point.

> **Verified equation @eq:laplace-factorization** · factorization at fixed $t$
>
> At any fixed $t$ and for any coefficients,
> $\sum_{PQ}J_{PQ}(t)^2 = \sum_{ijab}(ia|jb)^2e^{-t\Delta_{ij}^{ab}}$ and
> $\sum_{PQ}\sum_{ij}G_{PQ}[i,j]G_{PQ}[j,i] = \sum_{ijab}(ia|jb)(ib|ja)e^{-t\Delta_{ij}^{ab}}$.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 3 |
> | Negative control | the half-exponent $-\tfrac12t \to -t$ must fail |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/laplace.rs::sos_j_square_trace`, `ferric-mp2/src/laplace.rs::laplace_exchange_energy` |
> | Code test | `laplace_mp2_h2o_ccpvdz_vs_pyscf_dfmp2` in `crates/ferric-mp2/tests/validation_laplace_mp2.rs`: the MO energy, Coulomb and exchange, against the quadrature-isolated numpy reference, at most $3.1\times10^{-11}$ Ha |
>
> <!-- verified: id=laplace.factorization; script=scripts/mp2_variant_identities.py; claim=3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/laplace.rs::sos_j_square_trace, ferric-mp2/src/laplace.rs::laplace_exchange_energy; rust_test=ferric-mp2/tests/validation_laplace_mp2.rs::laplace_mp2_h2o_ccpvdz_vs_pyscf_dfmp2 -->

## The AO form of the Coulomb term {#laplace-mp2-ao-form-coulomb-term}

$J_{PQ}(t)$ can be computed without MO factors at all. Expand
$B^P_{ia} = \sum_{\mu\nu}C_{\mu i}B^P_{\mu\nu}C_{\nu a}$ in
$J_{PQ}(t) = \sum_{ia}B^P_{ia}B^Q_{ia}e^{t\varepsilon_i}e^{-t\varepsilon_a}$.
The sum over $i$ now meets only two coefficients and one exponential, and so
does the sum over $a$. Do those sums first. They define the occupied and
virtual **pseudo-densities**:

$$ \underline P(t) = \sum_i C_{\cdot i}\,e^{+t\varepsilon_i}\,C_{\cdot i}^{\mathsf T}, \qquad \underline Q(t) = \sum_a C_{\cdot a}\,e^{-t\varepsilon_a}\,C_{\cdot a}^{\mathsf T}, \qquad J_{PQ}(t) = \operatorname{Tr}\big[B^P\,\underline P(t)\,B^Q\,\underline Q(t)\big], \label{eq:laplace-ao-trace} $$

with $B^P_{\mu\nu}$ the dressed AO factor. They are underlined here to keep
them apart from the auxiliary index $P$ and the MP2 density.

At $t = 0$ and with no frozen core, $\underline P(0) = C_oC_o^{\mathsf T}$ is
half the Hartree–Fock density, hence the name. At $t > 0$ each orbital is
weighted by its own exponential. The AO form is the one a reduced-scaling
method would sparsify (section @sec:local-laplace-sos); as written here it is dense.
The same factorization, with the Laplace variable read as an imaginary
time, splits the independent-particle
polarizability $\chi_0(iu)$ of RPA into occupied and virtual products
(section @sec:laplace-ao-chi0). That formulation reaches $\mathcal O(N^3)$
only with AO sparsity; `crates/ferric-rpa/src/ao_rpa.rs` implements the
MO-basis version, which keeps the dense $\mathcal O(N^4)$ scaling.

`build_pseudo_density_occ` and `build_pseudo_density_vir` form the
pseudo-densities, and `laplace_ao_coulomb_energy` forms the trace in panels
of $\mu$ rows.

The exchange term has no AO trace of this shape. The script's negative
control shows that the natural candidate
$\sum_{PQ}\operatorname{Tr}(B^P\underline P B^Q\underline Q B^Q\underline P B^P\underline Q)$
is not it: it contracts the wrong index pairs. The AO path therefore still
forms the exchange term from the MO factors $B(t)$.

> **Verified equation @eq:laplace-ao-trace** · AO trace of the Coulomb term
>
> $\operatorname{Tr}[B^P\underline P(t)B^Q\underline Q(t)] = J_{PQ}(t)$
> elementwise.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 3 |
> | Negative control | the four-factor AO trace used as the exchange term must differ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/laplace.rs::laplace_ao_coulomb_energy`, `build_pseudo_density_occ`, `build_pseudo_density_vir` |
> | Code test | `laplace_mp2_h2o_ccpvdz_vs_pyscf_dfmp2` in `crates/ferric-mp2/tests/validation_laplace_mp2.rs`: the AO energy against the MO energy on the same grid, at most $3.4\times10^{-13}$ Ha |
>
> <!-- verified: id=laplace.ao-trace; script=scripts/mp2_variant_identities.py; claim=3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/laplace.rs::laplace_ao_coulomb_energy; rust_test=ferric-mp2/tests/validation_laplace_mp2.rs::laplace_mp2_h2o_ccpvdz_vs_pyscf_dfmp2 -->

## SOS-MP2 {#laplace-mp2-sos-mp2}

Section @sec:closed-shell-ri-mp2-energy-splits showed that about three quarters of the MP2 correlation
energy of water is opposite-spin. Scaled opposite-spin MP2 keeps only that
part and scales it:

$$ E_{\mathrm{SOS}} = c_{\mathrm{OS}}\,E_{\mathrm{OS}}, \qquad c_{\mathrm{OS}} = 1.3 \text{ by default (Jung, Lochan, Dutoi, Head-Gordon)}. \label{eq:laplace-mp2-5} $$

There is no $c_{\mathrm{SS}}$, by design. Dropping the same-spin term is what
lets the denominator factorize into independent $(i,a)$ and $(j,b)$
contractions. Compare the two intermediates of @sec:laplace-mp2-factorized-energy:

- In the opposite-spin numerator $(ia|jb)^2$, the $(i,a)$ and $(j,b)$
  pairings match the split of $e^{-t\Delta}$. The intermediate $J_{PQ}(t)$
  carries no orbital index at all. Forming it from $B(t)$, an
  $N_{\mathrm{aux}}\times n_on_v$ matrix, takes of order
  $N_{\mathrm{aux}}^2n_on_v$ multiply-adds per point: fourth order.
- In the same-spin exchange product $(ia|jb)(ib|ja)$ the pairings cross.
  $i$ shares $a$ with $j$, so the intermediate $G_{PQ}[i,j]$ keeps both
  occupied indices. Forming it takes of order $N_{\mathrm{aux}}^2n_o^2n_v$
  multiply-adds per point:
  fifth order, like RI-MP2 itself. No change to the denominator undoes the
  crossing, because it is in the numerator. Over $n_q$ points that is
  $n_qN_{\mathrm{aux}}/n_v$ times the canonical $n_o^2n_v^2N_{\mathrm{aux}}$,
  about 30 for water in cc-pVDZ at $n_q = 7$ ($N_{\mathrm{aux}} = 84$,
  $n_v = 19$). The kind `laplace-mp2` is therefore not a faster MP2. It is
  the quadrature and AO-against-MO anchor for the SOS energy.

So the SOS energy is the part of MP2 whose Laplace form costs less than the
canonical sum. Two input rules follow:

- The CLI warns if `[mp2] c_ss` is set.
- A non-finite $c_{\mathrm{OS}}$ is an error.

Check the limit $c_{\mathrm{OS}} = 1$. Then $E_{\mathrm{SOS}} = E_{\mathrm{OS}}$,
and the Laplace result must reproduce the opposite-spin energy of
`scs_mp2` to within the quadrature error. Section @sec:laplace-mp2-anchors gives the measured
agreement.

For water in cc-pVDZ (the `589b50832` build) the error against the
canonical RI value falls from $3.72\times10^{-4}$ to $4.94\times10^{-8}$ Ha
as `n_quad` goes from 3 to 7:

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

ref = ferric.run_scs_mp2(mol, bs, aux, c_os=1.0, c_ss=1.0).e_os
print(f"E_OS (canonical RI) = {ref:.10f}")
for n in (3, 5, 7):
    r = ferric.run_laplace_sos_mp2(mol, bs, aux, c_os=1.0, n_quad=n)
    print(f"n_quad={n}: E_OS = {r.e_os:.10f}  error = {r.e_os - ref:+.2e}")
```

```text
E_OS (canonical RI) = -0.1524402132
n_quad=3: E_OS = -0.1528123252  error = -3.72e-04
n_quad=5: E_OS = -0.1524478826  error = -7.67e-06
n_quad=7: E_OS = -0.1524402626  error = -4.94e-08
```

## The algorithm {#laplace-mp2-algorithm}

| Step | Quantity | Function |
|---|---|---|
| 1 | $y_{\min}$, $y_{\max}$, the grid (@eq:laplace-mp2-2) | `LaplaceMp2::init_quadrature`, `LaplaceQuadrature::new` |
| 2 | $V$, $M = L^{-1}$ | `coulomb_metric_2c`, `cholesky_inverse_sqrt` |
| 3 | MO path: $B^P_{ia}$, blocked transform then dressing | `eri3_mo_ov_blocked`, `dress_with_metric` |
| 3′ | AO path: dressed $B^P_{\mu\nu}$, stored sparse (threshold $10^{-12}$) | `compute_ao`, `compute_sos_ao` |
| 4 | per point, in parallel over $k$: $B(t_k)$, then $\sum J^2$ and (full MP2) $\sum G\cdot G$ | `weighted_b_mo`, `sos_j_square_trace`, `laplace_exchange_energy` |
| 4′ | AO path, per point: $\underline P(t_k)$, $\underline Q(t_k)$, the trace (@eq:laplace-ao-trace) | `laplace_ao_coulomb_energy` |
| 5 | $-\sum_k w_k(\ldots)$ | per-point terms collected, then summed in point order |

Which path each entry point takes:

| Entry | Coulomb term | Exchange term |
|---|---|---|
| `laplace-mp2` (`laplace_ri_mp2` → `compute_ao(.., None)`) | AO trace (@eq:laplace-ao-trace) | MO $G\cdot G$ |
| `laplace-sos-mp2`, `mo` (default) | MO $J = B(t)B(t)^{\mathsf T}$ (`compute_sos_mo`) | none |
| `laplace-sos-mp2`, `ao` | AO trace (`compute_sos_ao(.., None)`) | none |
| `LaplaceMp2::compute_mo` | MO | MO; library and tests only, the MO side of the MO ≡ AO checks |

Only the scaling depends on $t$. $B^P_{ia}$ is built once (step 3); each
point rescales a copy by a diagonal factor, which takes $n_o + n_v$
exponentials and one pass over the tensor, before the contractions of
step 4.

## The kernel {#laplace-mp2-kernel}

The per-point MO energy is the `map` closure of the quadrature loop in
`ferric-mp2/src/laplace.rs::LaplaceMp2::compute_mo`, the library and test
driver, with the code's comments replaced by the equation terms:

```rust
.map(|(&t, &w)| {
    let b_t = weighted_b_mo(&b_flat, &occ_scale, &vir_scale, nocc, nvir, t); // B^P_ia(t_k)
    let e_coul = sos_j_square_trace(&b_t);                                   // sum_PQ J_PQ(t_k)^2
    let e_exch = laplace_exchange_energy(&b_t, naux, nocc, nvir, k_budget); // sum G[i,j] G[j,i]
    -w * (2.0 * e_coul - e_exch)                                             // -w_k [2 J^2 - G.G]
})
.sum();
```

| Code | Equation |
|---|---|
| `t`, `w` | $t_k$, $w_k$ of (@eq:laplace-mp2-2) |
| `b_t` | $B^P_{ia}(t_k)$, scaled by $e^{t\varepsilon_i/2}e^{-t\varepsilon_a/2}$ |
| `e_coul` | $\sum_{PQ}J_{PQ}(t_k)^2$ |
| `e_exch` | $\sum_{PQ}\sum_{ij}G_{PQ}[i,j]G_{PQ}[j,i]$ |

`compute_mo` ends in a rayon `.sum()`, which adds the per-point terms in a
tree whose shape depends on the thread count. The production drivers do
not. `ferric-mp2/src/laplace.rs::LaplaceMp2::compute_sos_mo` drops the exchange term, collects
the terms in point order and adds them on one thread, for the reason of
section @sec:pair-kernel-parallel-loop-fixed-summation:

```rust
.map(|(&t, &w)| {
    let b_t = weighted_b_mo(&b_flat, &occ_scale, &vir_scale, nocc, nvir, t); // B^P_ia(t_k)
    -w * sos_j_square_trace(&b_t)                                            // -w_k J^2
})
.collect();
let e_os: f64 = per_point.iter().sum();                                      // in point order
```

`compute_ao` and `compute_sos_ao` collect the same way. With three to seven
points the tree of `compute_mo` happens not to split differently in
practice, so no bit moves today.

Three points about it:

- **The half exponent.** `weighted_b_mo` multiplies by
  $e^{t\varepsilon_i/2}e^{-t\varepsilon_a/2}$, the outer product of a per-occupied
  and a per-virtual vector. Each $J$ element is a product of two $B(t)$, so
  the halves recombine into $e^{-t(\varepsilon_a - \varepsilon_i)}$. The
  negative control of verified equation @eq:laplace-identity changes this $\tfrac12$ to 1.
- **The symmetric $J$.** `sos_j_square_trace` forms only the lower triangle
  of $J = B(t)B(t)^{\mathsf T}$ and counts each off-diagonal square twice.
- **The exchange blocks.** `laplace_exchange_energy` reads `b_t` as an
  $(N_{\mathrm{aux}}n_o)\times n_v$ matrix $X$. It forms
  $Y = X_{\mathrm{blk}}X^{\mathsf T}$ in blocks of whole auxiliary indices $P$,
  so that each $n_o\times n_o$ block $G_{PQ}$ lies inside one $Y$ block.

## Reproducibility and memory {#laplace-mp2-reproducibility-memory}

- **Fixed blocking.** The $J$ lower triangle is built in fixed blocks of
  `SOS_J_COL_BLOCK` = 128 columns, and the block sums are folded in order.
  The per-task memory share is the budget divided by the compile-time
  constant `NOMINAL_TASK_WORKERS` = 8, not by the live pool size. Block
  widths do not depend on `RAYON_NUM_THREADS`, so the Laplace contractions
  add no thread-count dependence. With `[parallel] blas_threads = 1` every
  bit of the energy is then thread-independent, which
  `crates/ferric-mp2/tests/laplace_energy_is_thread_independent.rs` checks.
  With the default serial BLAS count, the pool size, the metric factor and
  the SCF move in the last bits with the pool size (section
  @sec:determinism-and-memory-blas-thread-counts).
- **Wide pools.** A pool wider than 8 can exceed the budget by
  $n_{\mathrm{workers}}/8$. The code warns when that happens
  (`warn_if_task_workers_exceed_nominal`). The per-task share has a
  64 MiB floor (`MIN_PER_TASK`), which depends on nothing ambient.
- **MO path.** It holds $B_{ov}$ ($N_{\mathrm{aux}}n_on_v$) and one weighted
  copy per point in flight.
- **AO path.** It holds two $N_{\mathrm{aux}}N^2$ tensors at once, plus the
  sparsified copy. It fails fast when `laplace_dressing_peak_bytes` exceeds
  the budget.
- **Choosing a path.** The AO peak grows as $N_{\mathrm{aux}}N^2$ against
  $N_{\mathrm{aux}}n_on_v$ for `mo`. For `laplace-sos-mp2` the two give the
  same energy (section @sec:laplace-mp2-anchors), so `mo` is the path to use when memory is
  short. The kind `laplace-mp2` always takes the AO Coulomb path
  (`laplace_ri_mp2` calls `compute_ao(.., None)`) and has no MO route from
  the CLI.

## Anchors {#laplace-mp2-anchors}

| Check | Reference | Agreement | Test |
|---|---|---|---|
| `laplace-mp2`, five systems, $R$ = 19.0 to 45.7, `n_quad` = 7 | exact DF-MP2 (PySCF `DFMP2`) | the minimax quadrature error, at most $2.2\times10^{-7}$ Ha (H2O/aug-cc-pVDZ), bar $2\times10^{-6}$ | `laplace_mp2_*_vs_pyscf_dfmp2` in `crates/ferric-mp2/tests/validation_laplace_mp2.rs` |
| same five systems | quadrature-isolated reference: a dense numpy $(i,a,j,b)$ sum on PySCF fitted integrals with $1/\Delta$ replaced by the same minimax nodes, rescaled as in (@eq:laplace-mp2-2) | at most $3.1\times10^{-11}$ Ha | same tests |
| MO against AO, same grid | — | at most $3.4\times10^{-13}$ Ha | same tests |
| `laplace-sos-mp2` $E_{\mathrm{OS}}$, five closed-shell systems, all electrons | PySCF `DFMP2` `e_corr_os`, same auxiliary basis | the quadrature error, at most $1.4\times10^{-7}$ Ha at `n_quad` = 7 | `sos_mp2_*_vs_pyscf_dfmp2` in `crates/ferric-mp2/tests/validation_sos_mp2.rs` |
| SOS, MO against AO | — | $1.6\times10^{-14}$ measured, bar $10^{-10}$ | same tests |

<!-- figures: site/src/reference/validation.md rows "laplace-mp2", "laplace-sos-mp2"; crates/ferric-mp2/tests/validation_laplace_mp2.rs module doc table (MEASURED 2026-10-03); crates/ferric-mp2/tests/validation_sos_mp2.rs module doc -->

The rows separate two sources of error. The first row compares with exact
DF-MP2, so it measures the quadrature error and any implementation error
together. The second row uses the same quadrature in an independent numpy
code, so it isolates the implementation. `validation_laplace_mp2.rs`
asserts the second row so that a disagreement with DF-MP2 can be
attributed. Its "vs exact DF-MP2" column reproduces each system's
independently computed quadrature error to the last printed digit. The MO
and AO rows compare two independent constructions of the same number.

The test's negative controls are recorded mutations:

| Mutation | Error produced | Bar |
|---|---|---|
| drop one weight | $1.30\times10^{-2}$ | $10^{-9}$ |
| take $y_{\max}$ from $\varepsilon_{\mathrm{HOMO}}$ | $1.31\times10^{-3}$ | $10^{-9}$ |
| frozen core off by one | $2.46\times10^{-3}$ | $10^{-9}$ |

Each produces an error far above its bar against the quadrature-isolated
reference.

Both kinds are graded Proven. For SOS the grade covers the `mo` and `ao`
formulations only. Frozen core is not covered by the PySCF reference.

**Known limits (2026-11-15).**

- Only `n_quad` = 3, 5, 7 are tabulated. There is no error-controlled choice
  of the number of points.
- $y_{\max}$ includes frozen-core energies, which widens $R$.
- The AO path is dense; no reduced scaling has been measured for it.
- `build_pseudo_density_occ` and `build_pseudo_density_vir` are scalar
  triple loops, not GEMMs.

## Sources

| Source | Used for |
|---|---|
| `scripts/mp2_family_sympy.py`, part F; `scripts/mp2_variant_identities.py`, claim 3 | verified equations @eq:laplace-identity, @eq:laplace-factorization, @eq:laplace-ao-trace |
| M. Häser and J. Almlöf, J. Chem. Phys. 96, 489 (1992) (as listed in `site/src/reference/references.md`) | the Laplace-transformed denominator |
| A. Takatsuka, S. Ten-no and W. Hackbusch, J. Chem. Phys. 129, 044112 (2008) | the minimax tables |
| Y. Jung, R. C. Lochan, A. D. Dutoi and M. Head-Gordon, J. Chem. Phys. 121, 9793 (2004) | SOS-MP2 and $c_{\mathrm{OS}} = 1.3$ |
| `site/src/reference/validation.md`; the two validation test files | the anchor figures |

Changes from the sources:

- The pseudo-densities are underlined.
- The quadrature range is stated with the frozen-core caveat and the 1%
  table slack.
- The minimax error statements (equioscillation, $\delta/y_{\min}$, the
  energy bound) and the per-point operation counts of @sec:laplace-mp2-sos-mp2
  were checked numerically for this section.
- The `local` formulation is treated in full in section @sec:local-laplace-sos.
