# Moving-grid (Becke weight response) XC Hessian: Python prototype

`proto_grid_hessian.py` is the specification of the opt-in moving-grid
closed-shell KS Hessian (`ferric_scf::hessian::rks_moving_grid_hessian`,
issue #490). It works out, at a FIXED AO density `D`, the second nuclear
derivative of

```text
E_xc(R) = Σ_g W_g(R) f(ρ(r_g(R); R)),   r_g = R_home(g) + u_g,
W_g = s_g · P_home(r_g) / Σ_C P_C(r_g)
```

(Becke 1988 cells, three smoothing iterations, Bragg radii, size adjustment
clamped to |a| ≤ 1/2: the partition `ferric_dft::becke` uses), with every
grid point riding its home atom as ferric's production KS grid does.

The Hessian is taken by second-order forward-mode automatic differentiation
(value, gradient and Hessian carried per point), a construction independent
of finite differences. It needs only `numpy` and `pyscf`:

```
OPENBLAS_NUM_THREADS=1 python proto_grid_hessian.py      # QUICK=1 skips checks A and B
```

## What it checks (H2O / STO-3G, LDA VWN, 40x110 grid)

| Check | Result |
|---|---|
| A. AD moving-grid Hessian vs central FD of the AD moving-grid gradient | 5.6e-8 Ha/Bohr² at h = 5e-4 (falls as h²); the fixed-grid Hessian misses the same FD by 1.35e-3 |
| B. AD Hessian vs second differences of the plain (no AD) moving-grid energy, Richardson | 9e-10 |
| C. translation sum rule `Σ_B H[(B,α),:]` | 3e-15 moving, 3.8e-4 fixed grid |
| D. mutants (drop `W''`; drop the `W'·f'` cross terms; freeze the weights; no point motion) | each moves H by 1.2 to 2.6 (sum rule 1e-14 for all but "no point motion", 0.98) |
| E. `Σ_h T_h G_h T_hᵀ` (point-fixed per-home Hessian over atoms ≠ h, home rows/columns filled by translation invariance) vs the full AD Hessian | 4.9e-15 |

Check C is blind to the weight-term mutants (they keep translation
invariance), which is why the Rust tests anchor on finite differences of the
moving-grid gradient, not on the sum rule. The individual terms are O(1) and
cancel to O(1e-3), so precision matters in any reimplementation.

Check E is the structure the Rust code implements: for the points of home
atom `h`, the point-fixed Hessian `G_h` (the fixed-grid AO skeleton plus the
point-fixed Becke weight terms `W'' f + W' f'ᵀ + f' W'ᵀ`) is kept on the atoms
`B, C ≠ h`, and the `h` rows and columns are filled by translation invariance
(minus the sum of the others). The first-order XC Fock matrix gets the same
treatment, `V^h = −Σ_{B≠h} V^B` per home batch; the CPKS kernel is unchanged.
