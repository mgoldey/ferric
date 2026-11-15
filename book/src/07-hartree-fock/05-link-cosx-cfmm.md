# 7.5 LinK, seminumerical exchange (COSX) and CFMM

<!-- section: 7.5; status: ready; registry: rhf, uhf, rohf;
     theory: crates/ferric-scf/theory.md#"LinK and COSX exchange", "CFMM: the near and far fields as coded" -->

Three builders replace parts of the direct four-centre build for large
systems. LinK keeps the exact exchange integrals and visits fewer of them.
COSX does one of the two electron integrations of each exchange integral on a
grid. CFMM splits the Coulomb matrix into a near field computed exactly and a
far field from multipoles. They differ in what they give up:

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

## 7.5.1 LinK and COSX

<!-- include: crates/ferric-scf/theory.md#"LinK and COSX exchange" -->

> **Verified equation 7.7** · the COSX overlap fit
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

LinK's per-quartet screen uses the exchange pairings only:

<!-- excerpt: ferric-scf/src/link_k.rs::build -->
```rust
let screen = crate::quartet_scatter::DensityScreen::FourPairK(&d_max_shell);
```

The density pair list that bounds the ket loop
(`ferric-scf/src/pairs.rs::DensityPairs`) uses the same threshold as that
screen, so it drops only pairs no surviving quartet could need.

## 7.5.2 CFMM

<!-- include: crates/ferric-scf/theory.md#"CFMM: the near and far fields as coded" -->

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

CFMM is not reachable from the CLI or Python. It has its exactness anchor
(`cfmm_matches_direct_j_in_the_trivial_limit` in `crates/ferric-scf/src/cfmm.rs`)
and one far-field test, and no production caller.

## 7.5.3 Numerical anchors

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

## 7.5.4 Options that change the equation

| Option | Effect |
|---|---|
| `k_builder` | `"link"`, `"cosx"`; unset is the direct builder |
| `cosx_grid` | the SCF grid |
| `cosx_final_pass`, `cosx_final_grid` | one non-self-consistent energy correction on a larger grid |
| `cosx_overlap_fit` | the $S S_{\rm num}^{-1}$ fit, on by default |

## 7.5.5 Doc fixes needed

| Text | Code |
|---|---|
| `LinkK` struct doc: ket shells are the "intersection" of the two lists | the third shell runs over the union $\mathrm{dp}(i) \cup \mathrm{dp}(j)$ |
| the COSX pair-screen doc bounds each element's error by the threshold | the bound is the sum over the dropped partners |
| `CfmmConfig::thresh` documented as a near-field screen | the field is never read |
| the CFMM separation criterion in its docs | the code adds both extents and $w_s h$ (excerpt above) |

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "LinK and COSX exchange", "CFMM: the near and far fields as coded" | the equations |
| `scripts/scf_jk_builders_sympy.py` | verified equation 7.7 |
| `site/src/methods/scf.md`, "Choosing how exchange is built" | the measured errors and times |
| `site/src/reference/validation.md` | the COSX anchor |
