"""Frequency-quadrature error of ferric's DEFAULT grid on the periodic toy cells.  ferric PdepRpaConfig
default: QuadratureScheme::MiniMax, n_points 20 -> GL mapped with u0 = optimized_u0(20) = 0.5, i.e. the
same nodes as gl_quadrature(20, 0.5).  Reference: plasmon (no quadrature).  Also min e_ia (gap)."""

import sys

sys.path.insert(0, ".")
from pbc_gamma import Cell, build_integrals, rhf
from pbc_mp2 import denominators, ovov_from_eri
from pbc_rpa import drpa_plasmon, drpa_quad, ov_factor
from run_mp2_fit_error import SYS

for name in sys.argv[1:] or list(SYS):
    a, atoms, basis = SYS[name]
    nel = len(atoms)
    nocc = nel // 2
    ints = build_integrals(Cell(a, atoms, basis), None, exxdiv="ewald", verbose=False)
    e_n, eps, _, C = rhf(
        ints["S"], ints["h"], ints["I"], ints["enn"], nel, conv=1e-12, return_mo=True
    )
    ov = ovov_from_eri(ints["I"], C[:, :nocc], C[:, nocc:])
    for conv in ("shifted", "unshifted"):
        eo, ev = denominators(eps, nocc, ints["madelung"], conv)
        ex = drpa_plasmon(ov, eo, ev)
        errs = "  ".join(
            f"n={n}: {drpa_quad(ov_factor(ov), eo, ev, n=n) - ex:+.2e}"
            for n in (10, 20, 40, 80)
        )
        print(
            f"{name} {conv:9s} gap {(ev.min() - eo.max()):.4f} max e_ia {(ev.max() - eo.min()):.3f}  E {ex:.10e}  GL x0=0.5 {errs}",
            flush=True,
        )
