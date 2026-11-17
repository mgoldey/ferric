# Two-, three- and four-center integrals {#two-electron}

<!-- status: edit; registry: none;
     theory: crates/ferric-integrals/theory.md#"Two-, three- and four-center integrals" -->

Two electrons repel. In a basis, the repulsion between the charge cloud
$\chi_\mu\chi_\nu$ of one electron and the cloud $\chi_\lambda\chi_\sigma$ of
another is the four-center integral $(\mu\nu|\lambda\sigma)$. Exact Coulomb
and exchange matrices (J and K) contract these with the density. There are
$O(N^4)$ of them, which is why so much of this book is about not computing
most of them.

Density fitting (chapter @sec:density-fitting) replaces each cloud $\chi_\mu\chi_\nu$ by a sum of
auxiliary functions $\chi_P$. It then needs only the three-center integrals
$(P|\mu\nu)$, the repulsion between one auxiliary function and one orbital
product, and the two-center metric $(P|Q)$. Many methods also replace
$1/r_{12}$ by a modified kernel $w(r_{12})$: erf or erfc for range
separation, terf and terfc for attenuated MP2, Yukawa and Slater geminals for
F12. One `Engine` type serves all three integral classes and every kernel.

The plan: derive the simplest four-center integral to see what an engine
computes, then list which engine serves which kernel, then read how the
engine returns a block and what the caller must check.

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/engine.rs::Engine` (`new_2e`, `compute_quartet`, `new_3center`, `compute_eri3`, `new_2center`, `compute_eri2`) |
| Rust, whole tensors | `ferric-integrals/src/threeindex.rs::{eri3_tensor,coulomb_metric_2c}` |
| Python | `compute_eri3`, `compute_metric_2c` (`crates/ferric-python/ferric.pyi`) |

## What an engine computes, and which engine serves which kernel {#two-electron-engine-computes-engine-serves}

### What a four-center integral is

Take four unnormalized $s$ primitives with exponents $a, b$ on $\mathbf A,
\mathbf B$ and $c, d$ on $\mathbf C, \mathbf D$. Apply the Gaussian product
theorem (section @sec:one-electron) to each electron's pair. Electron 1 sees one Gaussian
of exponent $p = a + b$ on $\mathbf P$, scaled by
$K_{AB} = e^{-ab|\mathbf A - \mathbf B|^2/p}$; electron 2 sees one of exponent
$q = c + d$ on $\mathbf Q$, scaled by $K_{CD}$. The Coulomb repulsion of two
spherical Gaussians is a closed form:

$$ (ab|cd) = \frac{2\pi^{5/2}}{pq\sqrt{p+q}}\, K_{AB}K_{CD}\, F_0\!\Big(\frac{pq}{p+q}\,|\mathbf P - \mathbf Q|^2\Big), \qquad F_0(T) = \int_0^1 e^{-Tt^2}\,dt . $$

$F_0$ is the zeroth Boys function (section @sec:md-three-center evaluates it). Check two
limits. When $\mathbf P = \mathbf Q$ the argument is zero and $F_0(0) = 1$:
two concentric clouds repel by a finite amount. When the clouds are far apart,
$F_0(T) \to \tfrac12\sqrt{\pi/T}$, and the integral becomes
$(\pi/p)^{3/2}(\pi/q)^{3/2}K_{AB}K_{CD}/|\mathbf P - \mathbf Q|$: the
charges of the two clouds, divided by their distance. That is Coulomb's law
for two point charges.

A third limit connects this to section @sec:md-three-center. Normalize the second cloud by
$(q/\pi)^{3/2}$ and let $q \to \infty$: it becomes a unit point charge at
$\mathbf Q$, and the formula goes to $(2\pi/p)\,K_{AB}\,F_0(p|\mathbf P - \mathbf Q|^2)$,
the potential of the first cloud at $\mathbf Q$. Higher angular momentum
adds polynomial factors, which recurrence relations handle; the Boys function
of higher order $F_n$ appears there.

The integrals have the permutational symmetry of real functions,

$$ (\mu\nu|\lambda\sigma) = (\nu\mu|\lambda\sigma) = (\mu\nu|\sigma\lambda) = (\lambda\sigma|\mu\nu), $$

which makes eight orderings of each quartet equal. The engine computes one
quartet at a time; using the symmetry is the caller's job.

### The engines

`ferric-integrals/src/engine.rs` wraps one libint2 engine per kernel and
integral class: `Engine::new_2e` (four-center), `Engine::new_3center` with
`Engine::compute_eri3` for $(P|\mu\nu)$, and `Engine::new_2center` with
`Engine::compute_eri2` for $(P|Q)$. Libint2 serves the Coulomb, erf, erfc,
Yukawa and Slater-geminal kernels. The terf and terfc kernels go to the table
engine of this crate (two-, three- and four-center: `scf_compute_terfc_eri2`,
`_eri3`, `_eri4` in the shim). A libint2-native terf route exists behind the
`libint2_terf` feature and a patched libint2 (section @sec:terf-table-engine).

| Kernel | `OperatorKind` | Engine |
|---|---|---|
| $1/r$ | `Coulomb` | libint2 |
| $\operatorname{erf}(\omega r)/r$, $\operatorname{erfc}(\omega r)/r$ | `ErfCoulomb`, `ErfcCoulomb` | libint2 |
| $e^{-\zeta r}/r$, $e^{-\zeta r}$ | `Yukawa`, `SlaterGeminal` | libint2 (section @sec:yukawa-geminal) |
| fitted geminals | `Cgtg`, `CgtgCoulomb`, `Delcgtg2` | libint2, `Engine::new_2e_geminal` |
| $\operatorname{terf}/r$, $\operatorname{terfc}/r$ | `Terf`, `Terfc` | the table engine (section @sec:terf-table-engine) |

A linear combination of kernels
(`ferric-integrals/src/operator.rs::Operator::composite`) is a sum of engines,
each with its weight. Integrals are linear in the kernel, so
$(\mu\nu|\sum_k c_k \hat w_k|\lambda\sigma) = \sum_k c_k(\mu\nu|\hat w_k|\lambda\sigma)$,
and the engine holds one libint2 handle per term. The accumulation is in
`ferric-integrals/src/engine.rs::Engine::compute_quartet_checked`; here it is
with the terf and terfc branches of the call and the `SAFETY` comment removed
and the argument and message lists shortened:

```rust
let mut max_written = 0;
self.buf[..n].fill(0.0);
for &(coeff, h) in &self.handles {            // one handle per kernel term w_k
    let written = unsafe {
        ffi::scf_compute_eri_quartet(h, prep.handle(), /* sh1..sh4 */ ..,
                                     self.scratch.as_mut_ptr())
    };                                        // (sh1 sh2|w_k|sh3 sh4) in scratch
    if written < 0 && self.can_refuse() {
        return Err(written);                  // table engine refused the quartet
    }
    assert!(written >= 0, /* .. */);          // a libint2 failure is a bug
    if written > 0 {
        let w = written as usize;
        max_written = max_written.max(w);
        for i in 0..w {
            self.buf[i] += coeff * self.scratch[i];   // sum_k c_k (..|w_k|..)
        }
    }
}
if max_written == 0 { Ok(None) } else { Ok(Some(&self.buf[..max_written])) }
```

Three points about it:

- **`written == 0` from every term means screened.** Libint2 skipped the
  quartet, and the result is `None`, not a block of zeros.
- **A single unit-weight libint2 term skips this loop.** When the engine is
  one libint2 operator with weight 1, `compute_quartet_checked` returns a view
  of libint2's own result buffer instead of copying it into `buf`. Adding
  `0 + 1·x` gives `x` for every finite `x`, so the values are the same and
  the copy is saved.
- **The two error paths differ.** A negative status from a table engine is a
  refusal and comes back as `Err`. A negative status from libint2 is a caught
  C++ exception, which means a bug, and the code asserts.

## What a caller must respect {#two-electron-caller-must-respect}

An engine is constructed with a precision. Libint2 skips a shell quartet
whose estimated magnitude is below it, and the engine returns no block
(`None`). Most callers pass $10^{-14}$; the Schwarz table builder passes $0$,
which disables that skip (section @sec:schwarz).

A block for shells $(s_1 s_2|s_3 s_4)$ is row-major over the shells'
functions, $n_1 n_2 n_3 n_4$ doubles. A table engine can refuse a quartet
(total angular momentum past the table depth, tables not loaded);
`compute_quartet_checked`, `compute_eri3_checked` and `compute_eri2_checked`
return that refusal as an error, distinct from a screened block.

| Rule | Why |
|---|---|
| A `None` block means screened, not zero by symmetry | libint2 skips a quartet below the engine precision |
| Use the `_checked` calls on a table engine | a refusal (angular momentum past the table depth) is an error, not a zero block |
| Build bound tables with precision $0$ | a skipped diagonal would make a bound lower than its integral (section @sec:schwarz) |
| One engine per thread | libint2 engines hold scratch; the code builds a pool (`crates/ferric-integrals/src/engine_pool.rs`) |
| Exceptions never cross the C ABI | every `scf_compute_*` in `crates/ferric-integrals/shim/shim.cc` catches and returns a negative status; the Rust side asserts it |

Why does a refusal have to be an error? The Schwarz bound (section @sec:schwarz)
uses the diagonal integrals $(\mu\nu|\mu\nu)$ to decide what to skip. If a
refused quartet were read as zero, its bound would be zero, and every
integral it bounds would be dropped without anyone being told.

The last rule is about C++ exceptions. A C++ exception that unwinds through a
Rust frame is undefined behaviour, so every shim entry point catches
everything. This is `scf_compute_eri_quartet` in
`crates/ferric-integrals/shim/shim.cc`, unchanged:

```cpp
int scf_compute_eri_quartet(scf_engine *eng, const scf_basis *bs,
                              int sh1, int sh2, int sh3, int sh4, double *out) {
    FERRIC_TERF_HOOK_SCOPE(eng);
  try {
    const double *result = compute_eri_quartet_raw(eng, bs, sh1, sh2, sh3, sh4);
    if (result == nullptr) {
        return 0;  // libint screened the quartet (all zero).
    }
    int n = bs->nfunc[sh1] * bs->nfunc[sh2] * bs->nfunc[sh3] * bs->nfunc[sh4];
    for (int i = 0; i < n; ++i) out[i] = result[i];
    return n;
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "%s: %s\n", __func__, ex.what());
    return SCF_EINTERNAL;
  } catch (...) {
    return SCF_EINTERNAL;
  }
}
```

The return value carries three cases in one integer: $n > 0$ doubles
written, 0 for screened, and `SCF_EINTERNAL` ($-3$) for a caught exception.
The Rust wrapper above maps them to `Some`, `None` and an assertion.

## Numerical anchors {#two-electron-numerical-anchors}

| What | Test | Bar |
|---|---|---|
| terf and terfc $(P\vert\mu\nu)$, $(P\vert Q)$ against an independent Fourier-space construction (H2O, NH3 / cc-pVDZ; H2O / aug-cc-pVDZ) | `crates/ferric-integrals/tests/validation_terfc_integrals.rs` (`scs-mp2-2terfc` row of `site/src/reference/validation.md`) | $2\times10^{-12}$ (3-index), $3\times10^{-11}$ (2-index) |
| the four-center table engine against libint2 Coulomb | `crates/ferric-integrals/tests/eri4_vs_libint.rs` | per test |
| permutational symmetry of the quartets | `crates/ferric-integrals/tests/eri_permutational_symmetry.rs` | rounding |
| a C++ throw returns a status, not undefined behaviour | `crates/ferric-integrals/tests/ffi_exception_safety.rs` | — |

The $(ss|ss)$ closed form of @sec:two-electron-engine-computes-engine-serves is a textbook result (Helgaker,
Jørgensen and Olsen, chapter @sec:kohn-sham). No committed script covers it, and ferric does
not evaluate it itself; libint2 does. The closed form was checked against a
direct radial quadrature of the Coulomb integral for this section.

## Sources

- E. F. Valeev, Libint, <https://github.com/evaleev/libint>.
- L. E. McMurchie and E. R. Davidson, *J. Comput. Phys.* **26**, 218 (1978).
- T. Helgaker, P. Jørgensen and J. Olsen, *Molecular Electronic-Structure
  Theory* (Wiley, 2000), chapter 9.
