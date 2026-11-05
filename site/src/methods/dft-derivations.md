# DFT derivations

The exchange-correlation pieces of the Kohn–Sham solver, each stated in the
code's conventions and proved by a sympy script that exits nonzero on a
mismatch. The full equations, function locations and test cross-references
are in
[`crates/ferric-dft/theory.md`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-dft/theory.md).
Each script also checks mutated formulas (a dropped factor of 2,
a wrong libxc packing slot) and requires them to fail.

## Conventions

Closed shell, total density matrix \\( D \\), real AOs \\( \chi_\mu \\):

\\[ \rho = \sum_{\mu\nu} D_{\mu\nu}\chi_\mu\chi_\nu,\qquad
\sigma = |\nabla\rho|^2,\qquad
\tau = \tfrac12 \sum_b \sum_{\mu\nu} D_{\mu\nu}\,\partial_b\chi_\mu\,\partial_b\chi_\nu . \\]

Spin-polarized variables use libxc's packing:
\\( (\rho_\alpha,\rho_\beta) \\),
\\( (\sigma_{\alpha\alpha},\sigma_{\alpha\beta},\sigma_{\beta\beta}) \\) with
\\( \sigma_{\alpha\beta} = \nabla\rho_\alpha\cdot\nabla\rho_\beta \\), and
\\( (\tau_\alpha,\tau_\beta) \\). \\( v_\sigma = \partial(\rho\epsilon)/\partial\sigma \\),
not twice it. Meta-GGAs are \\( \tau \\)-only (no density Laplacian).

## What is proved

| Piece | Identity | Script |
|---|---|---|
| Becke partition | \\( \partial w^A/\partial \mathbf R_C \\) as coded, and \\( \nabla_{\mathbf r} w^A = -\sum_C \partial w^A/\partial\mathbf R_C \\) (used to build the grid-weight response) | `scripts/dft_becke_weight_grad_sympy.py` |
| Semilocal \\( V_{xc} \\) | \\( \partial E_{xc}/\partial D_{\mu\nu} = \sum_g w_g [v_\rho\chi_\mu\chi_\nu + 2v_\sigma\nabla\rho\cdot\nabla(\chi_\mu\chi_\nu) + \tfrac12 v_\tau\nabla\chi_\mu\cdot\nabla\chi_\nu] \\); polarized cross term \\( v_{\sigma_{\alpha\beta}}\nabla\rho_\beta \\) with coefficient 1; equal to \\( v_\rho - 2\nabla\cdot(v_\sigma\nabla\rho) \\) up to a divergence | `scripts/dft_vxc_matrix_elements_sympy.py` |
| \\( f_{xc} \\) kernels | GGA response \\( (u^\sigma, \mathbf w^\sigma) \\) is the linearization of \\( (v_{\rho_\sigma}, 2v_{\sigma_{\sigma\sigma}}\nabla\rho_\sigma + v_{\sigma_{\alpha\beta}}\nabla\rho_{\bar\sigma}) \\), with libxc's packed second derivatives | `scripts/dft_fxc_kernel_sympy.py` |
| VV10 | per-point \\( v_\rho, v_\sigma \\) are \\( \partial E_{nl}/\partial\rho_i, \partial E_{nl}/\partial\sigma_i \\) (self pair included) | `scripts/dft_vv10_potential_sympy.py` |
| XC nuclear gradient | \\( \partial\tau/\partial R_{A,\alpha} = -\sum_b\sum_{\mu\in A}\partial^2_{\alpha b}\chi_\mu (D\partial_b\chi)_\mu \\): the \\( \tfrac12 \\) in \\( \tau \\) cancels against the \\( \mu\leftrightarrow\nu \\) doubling; grid-response terms \\( \partial_\alpha\nabla\rho \\), \\( \partial_\alpha\tau \\) | `scripts/dft_mgga_tau_gradient_sympy.py` |

The numerical checks of the same identities (finite differences of the
energy, PySCF references) are listed per section in `theory.md`.

## Known open issue

The VV10 AO matrix \\( V_{nl} \\) is not exactly \\( \partial E_{nl}/\partial D \\):
on NH₂/6-31G the directional derivative misses by \\( 3.6\times10^{-6} \\)
relative (\\( -1.31\times10^{-7} \\) absolute), independent of the step size,
while ωB97X without VV10 agrees to \\( 8.6\times10^{-14} \\). The per-point
partials are proved exact, so the residual lies elsewhere on the path from
\\( D \\) to \\( V_{nl} \\); the cause is not identified. The test is a strict
xfail (`crates/ferric-python/tests/test_xc_blocks.py`).
