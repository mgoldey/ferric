# Distance-dependent bounds (QQR) {#qqr}

<!-- status: edit; registry: none -->

Two charge clouds far apart repel like two point charges: the product of
their charges over the distance between them. A Coulomb integral between
well-separated pairs therefore falls as $1/R$. The Schwarz bound of section
@sec:schwarz cannot see this. Its factors are self-interactions, so it gives the same
number for two distant pairs as for two touching ones. QQR multiplies Schwarz
by an envelope that falls as $1/R$ with the distance between the pair
centres, the leading behaviour of the Coulomb interaction between two compact
charge clouds.

The plan: define the envelope, prove it for primitive $s$ pairs, read the
code, and then see where the envelope changes which quartets are kept.

| Entry point | Location |
|---|---|
| four-index | `ferric-scf/src/qqr.rs::QqrBounds` (implements `ferric-scf/src/screening.rs::Bound`) |
| three-index | `ferric-integrals/src/qqr3.rs::QqrBounds3` (`estimate3`) |

## The envelope {#qqr-envelope}

The estimate is $Q_{ij}Q_{kl}$ times a distance envelope. It is a proven
bound for primitive $s$ pairs under the Coulomb kernel
(section @sec:qqr-s-pair-proof) and a measured one otherwise
(section @sec:qqr-numerical-anchors):

$$ |(ij|kl)| \le Q_{ij}Q_{kl}\,\min\Big(1, \frac{1.10\,(\epsilon_{ij} + \epsilon_{kl})}{R_{\mathrm{eff}}}\Big), \qquad R_{\mathrm{eff}} = \max(0, R - \epsilon_{ij} - \epsilon_{kl}), $$

where $R$ is the distance between the two pair centres. Each shell pair gets
a centre and an extent from the most diffuse exponent $\alpha$ of each of its
two shells:

$$ \mathbf R_{ij} = \frac{\alpha_i\mathbf R_i + \alpha_j\mathbf R_j}{\alpha_i + \alpha_j}, \qquad \epsilon_{ij} = (\alpha_i + \alpha_j)^{-1/2} . $$

The centre is the centre of the Gaussian product (section @sec:two-electron-engine-computes-engine-serves), and the
extent is the width of that product Gaussian. $R_{\mathrm{eff}}$ is the
edge-to-edge gap between the two clouds. Check the limits. While the clouds
overlap, $R \le \epsilon_{ij} + \epsilon_{kl}$, so $R_{\mathrm{eff}} = 0$, the
envelope is $1$ and the bound is Schwarz. The envelope stays at $1$ until
$R_{\mathrm{eff}} = 1.10(\epsilon_{ij} + \epsilon_{kl})$, that is until
$R = 2.10(\epsilon_{ij} + \epsilon_{kl})$. At large $R$ it falls as $1/R$, the
decay of the true Coulomb integral.

The three-index version replaces one pair by an auxiliary shell with its own
centre (its atom) and extent, $(2\alpha_{\min}(P))^{-1/2}$
(`ferric-integrals/src/qqr3.rs::QqrBounds3::estimate3`).

## The proof for primitive $s$ pairs {#qqr-s-pair-proof}

Take two primitive $s$ pair densities $\rho_1$, $\rho_2$ with exponents
$p_1$, $p_2$ and charges $q_1$, $q_2$, at distance $R$. Their Coulomb
integral and their Schwarz factors are closed forms
(section @sec:schwarz-bound-admitted-kernels):

$$ (\rho_1|\rho_2) = q_1q_2\,\frac{\operatorname{erf}(\theta R)}{R}, \qquad Q_n^2 = q_n^2\,\frac{2\theta_n}{\sqrt\pi},\quad \theta_n^2 = \frac{p_n}{2} . $$

Since $\operatorname{erf} \le 1$, the integral is at most the point-charge
value $q_1q_2/R$. Multiply the two factors:

$$ Q_1Q_2 = q_1q_2\sqrt{\frac 2\pi}\,(p_1p_2)^{1/4} . $$

Divide, and write the extents as $\epsilon = p^{-1/2}$, so that
$(p_1p_2)^{-1/4} = \sqrt{\epsilon_1\epsilon_2}$. The charges cancel:

$$ \frac{|(\rho_1|\rho_2)|}{Q_1Q_2} \le \sqrt{\frac{\pi}{2}}\, \frac{\sqrt{\epsilon_1\epsilon_2}}{R} . $$

The geometric mean is at most the arithmetic mean,
$\sqrt{\epsilon_1\epsilon_2} \le (\epsilon_1 + \epsilon_2)/2$, because
$\big((\epsilon_1 + \epsilon_2)/2\big)^2 - \epsilon_1\epsilon_2 = (\epsilon_1 - \epsilon_2)^2/4 \ge 0$.
That turns the product of extents into the sum the code uses. Finally
$\sqrt{\pi/8} = 0.627 < 1.10$ and $R_{\mathrm{eff}} < R$:

$$ \frac{|(\rho_1|\rho_2)|}{Q_1Q_2} \le \sqrt{\frac{\pi}{2}}\, \frac{\sqrt{\epsilon_1\epsilon_2}}{R} \le \sqrt{\frac{\pi}{8}}\, \frac{\epsilon_1 + \epsilon_2}{R} \le \frac{1.10\,(\epsilon_1 + \epsilon_2)}{R_{\mathrm{eff}}} . $$

Schwarz alone bounds the ratio by $1$, so the minimum with $1$ is also a
bound. For these pairs the envelope is a proven bound. For contracted and
higher-$l$ pairs the argument does not cover the dipole and quadrupole terms
the monopole picture neglects, and the factor $1.10$ is a measured margin.
Without it the three-index envelope undershoots by up to 5.9% (worst
true/bound 1.059 over 797,568 shell triples of alkane_6, water and benzene,
cc-pVDZ / cc-pVDZ-RI, erfc with $\omega = 0.222$); with it the worst ratio
is 0.9998 (`ferric-integrals/src/qqr3.rs::QqrBounds3`).

> **Verified equation {#eq:screening-qqr-s-pairs}** · the envelope bounds primitive $s$ pairs
>
> For primitive $s$ pairs, $|(ab|cd)|/(Q_{ab}Q_{cd}) \le \sqrt{\pi/8}\,(\epsilon_{ab} + \epsilon_{cd})/R$,
> so the coded envelope with factor $1.10$ holds; the same envelope over erfc
> Schwarz factors holds on the same sweep.
>
> | | |
> |---|---|
> | Proof | `scripts/screening_bounds_sympy.py`, claims qqr-s and qqr-erfc |
> | Negative control | factor $1.10 \to 0.50$ must fail (recorded mutation) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/qqr.rs::QqrBounds` |
> | Code test | `crates/ferric-scf/tests/qqr_diagonal_noop.rs` |
>
> <!-- verified: id=screening.qqr-s-pairs; script=scripts/screening_bounds_sympy.py; claim=qqr-s; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/qqr.rs::QqrBounds; rust_test=ferric-scf/tests/qqr_diagonal_noop.rs::qqr_equals_schwarz_on_every_diagonal_quartet -->

## No operator factor {#qqr-no-operator-factor}

The envelope is the Coulomb one for every kernel. Attenuation enters only
through the erfc Schwarz factors, so QQR shares the blindness of section
@sec:schwarz-bound-admitted-kernels. The erfc case of the verification script is a sweep over
$\omega \in \{0.2, 1, 5\}$, a measured bound, not a proof.

Why is there no erfc factor? The erfc integral between two $s$ pairs decays
as $\operatorname{erfc}(\phi R)$ with $\phi^2 = 1/(1/p_1 + 1/p_2 + 1/\omega^2)$
(section @sec:schwarz-bound-admitted-kernels), so a valid factor depends on
the pair exponents, and through $\phi < \omega$ on the most diffuse pair in
the block. $e^{-\omega^2R^2}$ is not one: it decays faster than the
integral. On benzene / cc-pVDZ at $\omega = 1$ it fell below every one of
30,000 sampled quartets, by up to 12 orders of magnitude. ferric has no
kernel-aware envelope. For compact pairs the Coulomb envelope is therefore
far from tight under erfc: the Schwarz factors keep none of the attenuation,
and the gain over Schwarz is the factor $1/R$ and no more. For two pairs of
exponent 20 at $\omega = 0.7$ and $R = 8/\omega$, the true erfc integral is
$7.15\times10^{-30}$ of the Schwarz product (`screening_bounds_sympy.py`,
claim blind), where the envelope is 0.045. The worst ratio of 0.998 on the
script's sweep comes from near-concentric pairs, where the envelope is 1.
Thompson and Ochsenfeld (2017), cited in section @sec:schwarz, derive
distance-including bounds for short-range kernels.

## The code {#qqr-code}

The four-index estimate is `ferric-scf/src/qqr.rs::QqrBounds`'s `estimate`,
with the centre lookup elided and the comments replaced:

```rust
let schwarz_est = self.schwarz.estimate(sh1, sh2, sh3, sh4); // Q_ij Q_kl
// ... look up c_bra, c_ket = pair centres R_ij, R_kl ...
let r = (dx * dx + dy * dy + dz * dz).sqrt();                 // R
if r < 1e-14 {
    return schwarz_est;                                       // concentric pairs: Schwarz
}
let ext_sum = self.pair_extents[idx_bra] + self.pair_extents[idx_ket]; // eps_ij + eps_kl
let r_eff = (r - ext_sum).max(0.0);                           // edge-to-edge gap
let decay = if r_eff > 0.0 {
    (SAFETY_FACTOR * ext_sum / r_eff).min(1.0)                // min(1, 1.10 eps_sum / R_eff)
} else {
    1.0                                                       // overlapping clouds
};
schwarz_est * decay
```

Three points about it:

- **The centres and extents are tables.** `QqrBounds::try_new` builds them
  once over all ordered shell pairs, dense $n_{\mathrm{sh}}^2$, from the
  smallest exponent of each shell. It checks the 32 bytes per pair (a centre
  and an extent) against the memory budget before allocating.
- **The operator is stored and not read.** `QqrBounds` keeps `op`, but
  `estimate` never uses it, and never reads $\omega$.
- **The safety factor is a constant.** `SAFETY_FACTOR` is $1.10$ in both
  `qqr.rs` and `qqr3.rs`. The three-index code seeds its Schwarz factors at
  engine precision $10^{-30}$ (`QQR3_SEED_PREC`), for the reason section @sec:schwarz-building-table-never-undershoots
  builds its table at precision $0$: a diffuse pair's self-integral must not
  be rounded to zero.

## Where QQR is used {#qqr-qqr-used}

| Consumer | Effect |
|---|---|
| LinK pair lists | none: the lists use diagonal estimates, where QQR equals Schwarz |
| per-quartet screens built on `Bound` | the envelope removes distant quartets |
| three-index builds (`build_band_screened`) | QQR-3 removes distant $(P\vert\mu\nu)$ triples; the screen is fixed for the life of a source (section @sec:three-index-sources) |

Why does QQR equal Schwarz on the pair lists? A pair list asks about the
diagonal quartet $(ij|ij)$, where bra and ket are the same pair, so $R = 0$
and the code returns the Schwarz product. Section @sec:pair-lists shows that LinK's
density lists already restrict the ket loop to pairs the density couples, so
the per-quartet envelope inside LinK finds few distant quartets left to
remove. On alkane_16 / cc-pVDZ at $10^{-8}$, QQR removes 3.83% more of all
unique quartets than Schwarz, 0.273% of those the pair lists admit, and
0.009% at the innermost test (`crates/ferric-scf/tests/qqr_population_gap.rs`).
One composition on one linear topology was measured; it says nothing about
builders without a pair-list prune, such as the three-index QQR-3.

`solve_rhf` builds `SchwarzBounds`, and no production path uses the
four-index `QqrBounds`; a caller can pass it in as the `Bound`. The envelope never reads $\omega$, so for erfc the attenuation enters
only through the Schwarz factors (section @sec:schwarz), and the gain is geometric.

## Numerical anchors {#qqr-numerical-anchors}

| What | Test | Bar |
|---|---|---|
| QQR bounds every quartet on water and benzene, Coulomb and erfc | `ferric-scf/src/qqr.rs` unit tests (`test_estimate_is_valid_upper_bound_*`) | bound/true $\ge 1$ |
| QQR equals Schwarz on every diagonal quartet | `crates/ferric-scf/tests/qqr_diagonal_noop.rs` | exact |
| LinK K at a threshold sweep with QQR against the dense K | `crates/ferric-scf/tests/screening_exactness.rs` (`link_k_thresh_sweep_alkane_8_qqr`) | per test |
| QQR-3 on terfc three-index tensors | `crates/ferric-integrals/tests/qqr3_terfc_screening.rs` | per test |
| four-index envelope against PySCF quartets, benzene / cc-pVDZ, 30,000 quartets (`scripts/qqr4_bound_validity.py`) | recorded in `ferric-scf/src/qqr.rs::QqrBounds` | worst true/bound 0.92 (Coulomb), 0.77 (erfc, $\omega = 1$) |
| three-index envelope with and without 1.10, 797,568 triples | recorded in `ferric-integrals/src/qqr3.rs::QqrBounds3` | worst 0.9998 with, 1.059 without |

The proof covers primitive $s$ pairs and the Coulomb kernel. Contracted and
higher-$l$ pairs, and the erfc kernel, rest on the measured anchors above.

## Sources

- S. A. Maurer, D. S. Lambrecht, D. Flaig and C. Ochsenfeld, *J. Chem. Phys.*
  **136**, 144107 (2012).
