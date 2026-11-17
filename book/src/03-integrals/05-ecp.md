# 3.5 Effective core potentials

<!-- section: 3.5; status: edit; registry: none;
     theory: crates/ferric-integrals/theory.md#"Effective core potentials" -->

The core electrons of a heavy atom take no part in bonding, but they are
expensive. They need tight basis functions, and in an atom like iodine they
move fast enough for relativity to matter. An effective core potential (ECP)
removes them. The atom's nuclear charge drops by the number of core electrons,
and an operator is added to the core Hamiltonian that does two jobs: it keeps
the valence electrons out of the region the core used to occupy, and it
carries the scalar relativistic effect of the core.

The operator cannot be a plain function of $r$. A valence $s$ electron and a
valence $p$ electron of the same atom see different cores, because each must
stay orthogonal to the core shells of its own angular momentum. So the ECP
acts differently on each angular momentum channel about its centre. That is
what "semi-local" means, and it is why ECP integrals need their own code.

ferric evaluates the operator by its own quadrature, with libecpint kept as a
cross-check. The plan: write the operator, expand each basis function in
angular momentum channels about the ECP centre, combine the radial factors so
nothing overflows, choose a radial quadrature window, and take derivatives.

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/ecp.rs::{ecp_matrix_spherical,ecp_matrix_deriv_spherical}`, `ferric-integrals/src/oneelectron.rs::hcore_ecp` |
| Backend | `FERRIC_ECP_BACKEND` = `quadrature` (default) or `libecpint` |
| Bases | def2 sets (`def2-ecp.json`), cc-pV$n$Z-PP for $Z \ge 37$ |

### The operator

`crates/ferric-integrals/src/ecp.rs` evaluates the semi-local ECP operator on
centre $C$,

$$\hat U = U_L(r) + \sum_{l=0}^{L-1}\sum_{m=-l}^{l} |lm\rangle\,\bigl[U_l(r) - U_L(r)\bigr]\,\langle lm|,\qquad U_l(r) = \sum_t d_t\, r^{n_t-2} e^{-\zeta_t r^2},$$

with $r = |\mathbf r - \mathbf C|$. Read it in two parts. $U_L$ acts on
every channel alike; it is local, an ordinary potential (the type-1 term).
Each channel $l < L$ gets the correction $U_l - U_L$, and the projector
$|lm\rangle\langle lm|$ applies it only to the part of the function that has
angular momentum $l$ about $C$ (the type-2 terms). Check the limit where all
$U_l$ equal $U_L$: the projector sum vanishes and $\hat U$ is the local
potential $U_L(r)$.

There are two backends (`FERRIC_ECP_BACKEND`): `quadrature`, the default
(`crates/ferric-integrals/src/ecp_quad/`, pure Rust), or `libecpint` (C++ via
`crates/ferric-integrals/shim/ecp_shim.cc`, kept as a cross-check). Any other
value is an error. Both return Cartesian blocks that
`crates/ferric-integrals/src/ecp.rs` transforms to spherical harmonics.

### Angular projection

The projector needs each basis function written in angular momentum channels
about $C$. A Gaussian on centre $A$, with $\mathbf a = \mathbf A - \mathbf C$,
has $|\mathbf r - \mathbf a|^2 = r^2 + a^2 - 2\mathbf r\cdot\mathbf a$
(coordinates now relative to $C$). The first two terms are spherical about
$C$. The cross term gives $e^{2\alpha\mathbf r\cdot\mathbf a}$, an exponential
of $z\cos\theta$ with $z = 2\alpha a r$ and $\theta$ the angle between
$\hat{\mathbf r}$ and $\hat{\mathbf a}$. That exponential has the Legendre
expansion
$e^{z\cos\theta} = \sum_\lambda (2\lambda+1)\, i_\lambda(z)\,P_\lambda(\cos\theta)$,
and the addition theorem
$P_\lambda(\hat{\mathbf r}\cdot\hat{\mathbf a}) = \frac{4\pi}{2\lambda+1}\sum_\mu Y_{\lambda\mu}(\hat{\mathbf r})Y_{\lambda\mu}(\hat{\mathbf a})$
splits it into a product of the two directions. Together they give the
plane-wave (Bessel) identity

$$e^{-\alpha|\mathbf r-\mathbf a|^2} = 4\pi\, e^{-\alpha(r^2+a^2)} \sum_{\lambda}\, i_\lambda(2\alpha a r)\sum_{\mu} Y_{\lambda\mu}(\hat{\mathbf r})\,Y_{\lambda\mu}(\hat{\mathbf a}),$$

where $i_\lambda$ is the modified spherical Bessel function of the first kind
and $Y_{\lambda\mu}$ are real spherical harmonics. Check the on-centre limit
$\mathbf a = 0$: $i_\lambda(0) = \delta_{\lambda 0}$ and
$Y_{00}^2 = 1/(4\pi)$, so the sum collapses to $e^{-\alpha r^2}$.

The semi-local (type-2) term needs the channel projection

$$ F_{lm}(r) = \int Y_{lm}(\Omega)\,\chi(\mathbf C + r\Omega)\,d\Omega . $$

Insert the identity and expand the polynomial factor of $\chi$ about $C$. The
angular integrals then involve only products of spherical harmonics and
monomials in the direction $\Omega$, none of which depend on $r$. So the
angular factor is built once per shell pair, centre and channel
(`crates/ferric-integrals/src/ecp_quad/type2.rs`), and only a radial integral is
left. The local (type-1) term expands the bra-ket product Gaussian the same
way (`crates/ferric-integrals/src/ecp_quad/type1.rs`).

### Radial integrals

A type-2 radial integrand contains, for a bra exponent $\alpha$ at distance
$a$ from $C$, a ket exponent $\beta$ at distance $b$ and an ECP exponent
$\zeta$,

$$ e^{-\alpha(r^2+a^2)}\,i_\lambda(2\alpha a r)\; e^{-\beta(r^2+b^2)}\,i_{\lambda'}(2\beta b r)\; r^{n}e^{-\zeta r^2} . $$

Taken factor by factor this overflows: $i_\lambda(z)$ grows as $e^z/(2z)$,
and $z = 2\alpha a r$ is large for a tight function far from the ECP. The
code uses the exponentially scaled
$\tilde k_\lambda(z) = e^{-z} i_\lambda(z)$
(`crates/ferric-integrals/src/ecp_quad/bessel.rs`), which lies in $[0, 1]$, and
moves the $e^{z}$ into the Gaussian. Completing the square in $r$:

$$ -\alpha(r^2+a^2) + 2\alpha a r - \beta(r^2+b^2) + 2\beta b r - \zeta r^2 = -p(r-r_0)^2 - K, $$

$$ p = \alpha+\beta+\zeta, \qquad r_0 = \frac{\alpha a + \beta b}{p}, \qquad K = \alpha a^2 + \beta b^2 - p r_0^2 . $$

Multiply $K$ by $p$ and expand:
$pK = \alpha\beta(a-b)^2 + \alpha\zeta a^2 + \beta\zeta b^2$. Every term is a
product of positives and squares, so $K \ge 0$ and no factor overflows at long
range. What is left is a Gaussian envelope $e^{-p(r-r_0)^2}$ centred at $r_0$
with width $1/\sqrt p$, times a slowly varying polynomial-like factor.

Each (primitive pair, ECP term) radial integral runs on its own window:
Gauss-Hermite away from $r = 0$, adaptive Gauss-Legendre otherwise
(`crates/ferric-integrals/src/ecp_quad/radial.rs`). Here is the body of
`ferric-integrals/src/ecp_quad/radial.rs::nodes`, with comments added:

```rust
let sp = p.sqrt();                                   // 1/width of the envelope
if r0 * sp >= RAD_TAU {                              // envelope far from r = 0
    let (x, wt) = gh_table(rule.gh);
    let isp = 1.0 / sp;
    for i in 0..rule.gh {
        r[i] = r0 + x[i] * isp;                      // Gauss-Hermite about r0
        w[i] = wt[i] * isp;
    }
    return rule.gh;
}
let lo = (r0 - RAD_T / sp).max(0.0);                 // window clipped at r = 0
let rpk = 0.5 * (r0 + (r0 * r0 + 2.0 * f64::from(deg.max(0)) / p).sqrt());
let hi = r0.max(rpk) + RAD_T / sp;                   // past the peak of r^deg e^{-p(r-r0)^2}
let want = (RAD_PER_SIGMA * (hi - lo) * sp).ceil();  // nodes per width
let n = if want.is_finite() {
    (want as usize).clamp(rule.gl_min, rule.gl_max)
} else {
    rule.gl_max
};
// ... map the n Gauss-Legendre nodes from [-1, 1] onto [lo, hi]
```

Three points about it:

- **When the envelope is far from the origin, Gauss-Hermite is the natural
  rule.** With `RAD_TAU` = 6.5, the envelope sits at least 6.5 widths from
  $r = 0$, and the `RAD_NH` = 20 nodes $r = r_0 + x/\sqrt p$ all lie at
  $r > 0$. The mass the rule would miss below $r = 0$ is at most $e^{-42}$.
- **Otherwise the window is cut at $r = 0$ and stretched past the peak.** The
  factor $r^{\mathrm{deg}}$ moves the maximum of the integrand outward. Setting
  the derivative of $r^{\mathrm{deg}}e^{-p(r-r_0)^2}$ to zero gives
  $2pr^2 - 2pr_0 r - \mathrm{deg} = 0$, whose positive root is `rpk`. The
  window runs `RAD_T` = 6 widths beyond the larger of $r_0$ and `rpk`.
- **The node count follows the window length.** `RAD_PER_SIGMA` = 3.4 nodes per
  width, clamped to between `RAD_NMIN` = 16 and `RAD_N` = 40. The module records
  a maximum relative error of $5.2 \times 10^{-14}$ per window over 42.8k
  windows, with 27.7 nodes on average.

### Derivatives

Shell derivatives use raised and lowered shells, the formula of section 3.3:

$$ \frac{\partial\chi}{\partial A_x} = 2\alpha\,\chi(i+1) - i\,\chi(i-1), $$

where $\chi(i\pm1)$ is the function with the $x$ exponent raised or lowered by
one. The code builds the raised companion of a shell with coefficients
$2\alpha c$ and the lowered one with coefficients $c$, evaluates the ordinary
ECP block for each, and combines them. The ECP-centre derivative follows from
translation invariance (section 3.3) as $-(\partial_A + \partial_B)$; in
`ferric-integrals/src/ecp_quad/mod.rs::block_deriv_cart` that is the line
`cen[k].1[x][i] -= da[x][i] + db[x][i]`. No separate derivative of the
projector is needed.

## 3.5.1 The order of operations

| Step | What happens | Where |
|---|---|---|
| 1 | `mol.apply_ecp(bs)` records $n_{\mathrm{core}}$ per atom | `ferric-core/src/mol.rs` |
| 2 | electron count and nuclear repulsion use $Z - n_{\mathrm{core}}$ | `Molecule::nelec`, `Molecule::nuclear_repulsion` |
| 3 | $V$ uses the same effective charges | `ferric-integrals/src/oneelectron.rs::nuclear` |
| 4 | $h = T + V + \hat U$ | `ferric-integrals/src/oneelectron.rs::hcore_ecp` |

Calling step 2 before step 1 counts the core electrons twice; the order is
part of the API.

### Why a quadrature backend

The libecpint backend's projector values carry a non-smooth error of about
$10^{-7}$ to $3.5 \times 10^{-5}$ (the doc comment of
`ferric-integrals/src/ecp.rs::EcpBackend`). The quadrature's radial rule is
accurate to $5.2 \times 10^{-14}$ per window, and its angular factors are
analytic. libecpint stays in the build as an independent implementation that the
quadrature is tested against.

## 3.5.2 Numerical anchors

| What | Test | Bar |
|---|---|---|
| RHF and UHF energies and gradients with def2 ECPs (HI, CH3I, RbH, I, SnH4) against PySCF and ORCA | `crates/ferric-scf/tests/validation_ecp.rs` (row in `site/src/reference/validation.md`) | measured $\le 2.9\times10^{-8}$ Ha, $\le 2.3\times10^{-8}$ Ha/Bohr |
| spherical ECP blocks against PySCF `ECPscalar` | `crates/ferric-integrals/tests/ecp_matrix.rs` | per test |
| quadrature against libecpint, with broken constructions that must fail | `crates/ferric-integrals/tests/ecp_quadrature.rs` | per test |
| derivative blocks | `crates/ferric-integrals/tests/ecp_matrix_deriv.rs`, `crates/ferric-integrals/tests/ecp_deriv_block.rs` | per test |

No proof script covers this section. The Bessel identity, the completed
square with $pK \ge 0$ and the peak position `rpk` were checked numerically
and symbolically for this section.

## Sources

- L. E. McMurchie and E. R. Davidson, *J. Comput. Phys.* **44**, 289 (1981).
- R. A. Shaw and J. G. Hill, *J. Chem. Phys.* **147**, 074108 (2017).
