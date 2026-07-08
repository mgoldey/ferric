# `terfc` Integral Implementation Plan

**Goal:** Exact `terfc(r,r₀)/r` two-electron integrals via pre-computed 2D interpolation tables and a standalone C++ Obara-Saika engine — no fitting, no libint2 modification.

---

## STATUS (2026-07-07, PM) — table path RESOLVED against the actual Dutoi paper

The Dutoi–Head-Gordon paper is now local (`../dutoi_a-study-of-the-effect-of-attenuation-curvature...pdf`,
JPCA 2008, 112, 2110–2119). It resolves the ingestion recipe (Eqs 6–11). Earlier "off by 1.37 /
need a hidden contraction / use base_terfc_closed.py as oracle" bullets are superseded and removed.

- **Task 0 (tables): DONE + self-verified.** `generate_tables.py --check` passes 3 anchors
  (G_00(0,0)=1; Boys face G_(m,0)(S,0)=F_m(S); interior full-series slice). Break-on-zero bug fixed
  (memory `terfc-table-break-bug`). Generator MATH confirmed vs paper: `Σ_i df(2i)·gS[m+1][i]·gs[n][i]`
  with `df(2i)=(2i)!!/(2i+1)!!` **is** Dutoi Eq 19 for `G_m^(n)`.
- **INGESTION RECIPE (verified, `terfc_lookup_reference.py` → worst rel diff 1.4e-15):**
  the shipped `G_0` is NOT the terfc base alone; **terfc = Coulomb − terf** with TWO reduced exponents:
  ```
  θ = (1/p+1/q)^{-1/2},  φ = (1/p+1/q+1/ω²)^{-1/2},  ω = 1/(r0√2)   (φ folds in 1/ω² — the key)
  T = (θR)²,  S = (φR)²,  s = (φr0)²                                 (R = |R_PQ|)
  I_pq[terfc] = (2θ/√π)·F_0(T)  −  (2φ/√π)·G_0(S,s)
  ```
  Two DIFFERENT prefactors and arguments; their difference is the base. Convert to ferric's
  `[0|op|0]/pref_boys` units by the Coulomb-anchor rescale (reference lines 141–145). Higher L:
  `F_m(T)`, `G_m^(n)(S,s)` (m≤4·l_max) as Boys replacements in STANDARD Coulomb OS; n-index = s-deriv
  order (0 for energies). 10×10 poly interp → ~1e-15 (bilinear only ~1e-4).
- **CORRECT terfc operator** (paper Eq 1): `terf=½[erf(x+y)+erf(x−y)]`, `terfc=1−terf`, so
  `terfc(r,r0)/r = [1 − ½(erf(ω(r+r0))+erf(ω(r−r0)))]/r` (positive, in (0,1/r); the ½ is essential).
- **RETRACTED: `base_terfc_closed.py` is NOT a valid oracle** — it validated against a wrong,
  sign-changing operator `1/r−(erf(ω(r−r0))+erf(ω(r+r0)))/r` (no ½; goes negative). Its "1e-60"
  match is vacuous (matches its own wrong operator). Use `terfc_lookup_reference.py`'s oracle
  (`gt_check.gt_phi` with the correct operator) instead.
- **Gate:** terfc→Coulomb only as **O(1/r0)** (correct); validate against the reference, not the
  retracted closed form.
- **shim.cc:** contains PARTIAL, WRONG table code from a dead agent (`terfc_aux` = F_m−Σc_n G —
  fabricated). Remove it; reimplement as a mechanical port of `terfc_lookup_reference.py` +
  standard Coulomb OS for angular momentum. shim.h has the 4 correct C ABI decls. No production
  Rust wiring yet (Task 2+).

---

## Background: Why Tables Work

The `terfc` primitive integral cannot be reduced to the standard Boys function $F_m(T)$. Instead, the "base integral" is a two-variable function $G_{m,n}(S, s)$:

$$G_{m,n}(S,s) = \sum_{i=0}^{\infty} \underbrace{\left(\int_0^1 (1-u^2)^i du\right)}_{\text{df}(2i)} \cdot g_S[m][i] \cdot g_s[n][i]$$

where:
- $g_x[0][i] = \sum_{j=0}^{i} e^{-x} x^j/j!$ (Poisson CDF)
- $g_x[1][i] = e^{-x} x^i/i!$ (Poisson PMF)
- $g_x[k][i] = g_x[k-1][i] - g_x[k-1][i-1]$ for $k \ge 2$ (forward differences)

The arguments map from GTO shell-pair parameters:
- $S = \theta \cdot |\vec{R}_{PQ}|^2$ (same as the Boys function argument $T$, using $\theta = pq/(p+q)$)
- $s = \frac{\theta\,\omega^2}{\theta + \omega^2} \cdot r_0^2$ (screened exponent × cutoff squared)
- For the curvature-constrained case: $\omega = 1/(r_0\sqrt{2})$, giving $s \in [0, 0.5]$

$G_{m,n}$ satisfies the **same** Obara-Saika vertical recurrence as $F_m(T)$, but with two indices tracking the attenuation. This means our custom engine can use standard OS transfer equations — we only need to provide the correct $G_{m,n}$ starting integrals.

---

## Files Changed / Created

| File | Action |
|------|--------|
| `terf-tables/generate_tables.py` | **New**: fast Python 3 generator (mpmath + multiprocessing + binary output) |
| `crates/ferric-integrals/shim/shim.h` | Add `scf_engine_create_terfc`, `scf_terfc_eri3`, `scf_terfc_eri2` declarations |
| `crates/ferric-integrals/shim/shim.cc` | Add `TerfcEngine` C++ class + C ABI wrappers |
| `crates/ferric-integrals/src/ffi.rs` | Add `scf_engine_create_terfc` binding |
| `crates/ferric-integrals/src/operator.rs` | Add `OperatorKind::Terfc { r0 }` and `Operator::terfc(r0)` |
| `crates/ferric-integrals/src/engine.rs` | Add `Engine::new_terfc_3center`, `Engine::new_terfc_2center` |
| `crates/ferric-mp2/src/scs.rs` | Promote `scs_mp2_2terfc` from deprecated |
| `testdata/terf-tables/` | Pre-generated binary table files (ship with repo) |

---

## Task 0 — Generate the Binary Tables

```bash
cd terf-tables
pip install mpmath numpy
python3 generate_tables.py   # ~10 min on 8 cores
```

Produces four files (`16_4_2.bin`, `8_10_5.bin`, `4_20_20.bin`, `2_20_80.bin`). These are ~60 MB total and can be committed to `testdata/terf-tables/` via LFS or distributed separately.

Validate with:
```bash
python3 generate_tables.py --check
```

---

## Task 1 — C++ `TerfcEngine` in the Shim

### Table loading

```cpp
struct TerfcTable {
    int nS, ns, dimm, dimn;
    double delta_S, delta_s;
    double S_max, s_max;
    std::vector<double> data;  // shape [nS][ns][dimm][dimn], C-order

    double G(int iS, int is, int m, int n) const {
        return data[((iS * ns + is) * dimm + m) * dimn + n];
    }
};
```

### `G_m^{(n)}(S, s)` via polynomial interpolation

For a query `(S, s)`, pick the finest covering table (`16_4_2` → `8_10_5` → `4_20_20` → `2_20_80`),
then 10×10 Lagrange interpolation on the consecutive-integer grid nodes (→ ~1e-15; bilinear only ~1e-4).
See `terfc_lookup_reference.py::interp_G`.

### Base integral: terfc = Coulomb − terf (two reduced exponents)

The `(s|terfc|s)` primitive base is **not** a single `G` lookup — it is the Coulomb base minus the
terf base, each with its own reduced exponent (paper Eqs 6/7/9/11; verified in
`terfc_lookup_reference.py` to 1.4e-15):

$$\theta=\left(\tfrac1p+\tfrac1q\right)^{-1/2},\quad \varphi=\left(\tfrac1p+\tfrac1q+\tfrac1{\omega^2}\right)^{-1/2},\quad \omega=\tfrac1{r_0\sqrt2}$$
$$T=(\theta R)^2,\quad S=(\varphi R)^2,\quad s=(\varphi r_0)^2,\quad R=|\vec R_{PQ}|$$
$$I_{pq}[\mathrm{terfc}](R)=\frac{2\theta}{\sqrt\pi}F_0(T)\;-\;\frac{2\varphi}{\sqrt\pi}G_0(S,s)$$

in the paper's average-interaction normalization (Eq 2). Rescale to ferric's `[0|op|0]/pref_boys`
units via the Coulomb anchor (reference lines 141–145). The two pieces carry **different** prefactors
(`2θ/√π` vs `2φ/√π`) and **different** arguments (`T` vs `S,s`) — do not collapse them.

### OS Vertical Recurrences for terfc 3-center `(P|terfc|mn)`

Angular momentum is built by the STANDARD Coulomb OS transfer equations applied to **each piece
separately**: the Coulomb piece uses `F_m(T)` (its own θ, T), the terf piece uses `G_m^{(n)}(S,s)`
(its own φ, S, s) as the Boys replacement; subtract at the end. The `n`-index is the s-derivative
order and stays 0 for energies (nonzero only for terfc gradients). $K_P, K_{mn}$ are the Gaussian
overlap prefactors as in the standard Coulomb 3-center engine.

### C ABI Functions Added

```c
// Create a terfc 3-center engine. table_dir: path to the directory with *.bin files.
scf_engine* scf_engine_create_terfc_3center(double r0, double omega,
    int max_nprim, int max_L, double precision, const char* table_dir);

// Compute (P|terfc|mn) for one shell triple. Same signature as scf_compute_eri3.
int scf_compute_terfc_eri3(scf_engine* eng, const scf_basis* obs,
    const scf_basis* dfbs, int shP, int sh1, int sh2, double* out);

// Compute (P|terfc|Q) for one shell pair (2-center metric for terfc RI).
int scf_compute_terfc_eri2(scf_engine* eng, const scf_basis* dfbs,
    int shP, int shQ, double* out);
```

---

## Task 2 — Rust `OperatorKind::Terfc`

```rust
// In operator.rs
pub enum OperatorKind {
    Coulomb,
    ErfCoulomb,
    ErfcCoulomb,
    Terfc,          // ← NEW: exact terfc via interpolation table
    // ...
}

impl Operator {
    pub fn terfc(r0: f64) -> Self {
        // omega from the curvature constraint: r0 * omega = 1/sqrt(2)
        let omega = 1.0 / (r0 * std::f64::consts::SQRT_2);
        Self { kind: OperatorKind::Terfc, omega, distance: r0 }
    }
}
```

The `Engine::new_3center` match gains a `Terfc` arm that calls `scf_engine_create_terfc_3center`.

---

## Task 3 — SCS-MP2(2terfc) Promotion

Move `scs_mp2_2terfc` from `deprecated/scs_2terfc.rs` into `crates/ferric-mp2/src/scs.rs`, replacing the `Operator::erfc(omega)` calls with `Operator::terfc(r0)`:

```rust
// Before (erfc approximation):
let op1 = Operator::erfc(omega1);

// After (exact terfc via interpolation table):
let op1 = Operator::terfc(config.r0_bonded);
```

No other changes needed — `ri_mp2_spin_components` already accepts any `Operator`.

---

## Task 4 — Validation Tests

```rust
#[test]
fn terfc_plus_terf_equals_coulomb() {
    // (ia|terfc|jb) + (ia|terf|jb) = (ia|jb) exactly
    // terfc + terf = 1/r, so terfc_corr + terf_corr = full_mp2_corr
    // terf = erf(ω(r-r₀))/r + erf(ω(r+r₀))/r  -- uses ErfCoulomb twice
    // This is the key identity test.
}

#[test]
fn terfc_large_r0_approaches_coulomb() {
    // As r₀ → ∞, terfc(r, r₀)/r → 1/r
    // RI-MP2 with terfc operator should → standard RI-MP2 energy
}

#[test]
fn terfc_known_value_h2_ccpvdz() {
    // Compare against reference from Q-Chem for H₂/cc-pVDZ at r₀=1.05 Å
}
```

---

## Notes

- **Table directory**: configurable via `FERRIC_TERF_TABLE_DIR` env var, falling back to `$FERRIC_DATA_DIR/terf-tables/` then the compile-time path.
- **2-center terfc metric** `(P|terfc|Q)`: needed for the RI auxiliary metric. Uses the same engine with a different BraKet (xs_xs equivalent for terfc).
- **Thread safety**: each thread creates its own `TerfcEngine` (same pattern as existing `Engine` usage). Tables are loaded once into a process-global `shared_ptr<TerfcTable[]>`.
- **Angular momentum coverage**: the tables support up to `DIMM=24` m-indices, covering angular momenta through at least `h` functions via the OS recurrences — sufficient for cc-pVDZ through cc-pVQZ.

