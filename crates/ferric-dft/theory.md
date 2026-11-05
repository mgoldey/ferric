# Density Functional Theory (DFT)

This module implements the Kohn-Sham DFT method with numerical integration on a grid.

## Kohn-Sham Equations
The DFT energy is:
$$E_{DFT} = T_s + V_{ext} + J + E_{xc}$$
where $E_{xc}$ is the exchange-correlation functional.

## Numerical Integration
The exchange-correlation energy and potential are computed on a grid of points $\{\mathbf{r}_k, w_k\}$:
$$E_{xc} = \int \rho(\mathbf{r}) \epsilon_{xc}(\rho, \nabla \rho, \dots) d\mathbf{r} \approx \sum_k w_k \rho(\mathbf{r}_k) \epsilon_{xc}(\rho_k, \gamma_k, \dots)$$
where $\gamma = |\nabla \rho|^2$.

## Functionals
We support LDA and GGA functionals via an interface to `libxc`.
- **LDA**: $\epsilon_{xc} = \epsilon_{xc}(\rho)$
- **GGA**: $\epsilon_{xc} = \epsilon_{xc}(\rho, \gamma)$

## Integration Grid
Each atom carries a Treutler-Ahlrichs radial grid times a Lebedev angular grid; atomic grids are combined with Becke partition weights $w_A(\mathbf{r})$, so the weight of a point on atom $A$ is $w_r\,w_\Omega\,w_A(\mathbf{r})$.

**Default grid.** A Kohn-Sham calculation with no grid set (`RhfConfig::dft_grid = None`, CLI `[dft]` grid keys absent, Python grid kwargs `None`) runs on `AtomicGridConfig::production()`: 75 radial $\times$ 110 angular points per atom, pruned with `NwchemRadial` (NWChem angular regions; 50 radial shells for H/He and 66 for Li-Ne), with points of final weight below $10^{-14}$ dropped (`DEFAULT_WEIGHT_CUT`, ORCA's `WeightCut`). Every KS path (SCF, gradients, Hessians, response, TDDFT) resolves `None` to this grid, so they integrate on the same points.

`AtomicGridConfig::default()` is the flat, uncut 75 $\times$ 110 grid. It is the quadrature for analysis code (Hirshfeld volumes, partitions, property grids) and for tests that pin flat-grid numbers; it is not the KS default. The VV10 nonlocal grid is a separate flat 50 $\times$ 50 grid (`default_nlc()`); pruning has no region table at angular order 50, so it is never pruned.

**Config keys.** CLI `[dft]`: `grid_radial`, `grid_angular`, `grid_prune`, `grid_weight_cut`. Python (`run_dft`, `run_ksdft`, ...): the same names as keyword arguments. Setting only a size keeps the default scheme and cutoff at that size; `grid_prune = "none"` with `grid_weight_cut = 0` is the flat grid. See `site/src/reference/input.md` for the value tables.

**Gradients.** The analytic gradient differentiates the energy on the grid the SCF ran on, including pruning and the weight cut, by dropping exactly the same points.
