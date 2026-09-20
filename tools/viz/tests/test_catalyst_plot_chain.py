"""The four catalyst figures, driven by real ferric output rather than literals.

Each plot has unit tests against hand-built inputs. None of them checks that
the SHAPES ferric actually returns are the shapes the plots accept -- and that
is where this session's bugs lived: `imaginary_mode` is `None` unless there is
exactly one, `IrcResult.forward_barrier` is a method not a property,
`link_atom_positions()` is Angstrom while `point_charges()` is Bohr.

A unit test with a literal `[0.1, 0.2, ...]` cannot catch any of those. This
runs the chain.

Marked slow: it does four SCF-driven calculations. Worth the seconds, because
the alternative is discovering the mismatch by hand a fifth time.
"""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")
pytest.importorskip("matplotlib")

from tools.viz.energy_plots import (  # noqa: E402
    close,
    imaginary_mode,
    optimization_trace,
    qmmm_partition,
    reaction_path,
)

_BOHR = 0.52917721
_ETHANE = (
    ["C", "H", "H", "H", "C", "H", "H", "H"],
    [
        (0.000, 0.000, 0.000),
        (-0.363, 1.027, 0.000),
        (-0.363, -0.513, 0.889),
        (-0.363, -0.513, -0.889),
        (1.540, 0.000, 0.000),
        (1.903, -1.027, 0.000),
        (1.903, 0.513, -0.889),
        (1.903, 0.513, 0.889),
    ],
    [0.0, 0.0, 0.0, 0.0, -0.27, 0.09, 0.09, 0.09],
)


def test_qmmm_partition_and_trace_accept_what_ferric_returns():
    """Setup -> partition -> relax -> trace, with no hand-built numbers."""
    symbols, coords, charges = _ETHANE
    z1 = (
        ferric.QmmmSystem(symbols, coords, charges, qm_indices=[0, 1, 2, 3])
        .with_link_atoms([(0, 4)])
        .with_boundary_charges([(0, 4)], "delete-host")
    )
    pc = z1.point_charges()

    # The unit cross-check is the point: pass the accessor's own value and the
    # plot refuses coordinates that disagree with it.
    fig = qmmm_partition(
        symbols,
        coords,
        [0, 1, 2, 3],
        link_positions_angstrom=[tuple(p) for p in z1.link_atom_positions()],
        charge_positions_angstrom=[
            (c[1] * _BOHR, c[2] * _BOHR, c[3] * _BOHR) for c in pc
        ],
        charge_values=[c[0] for c in pc],
        expect_min_angstrom=z1.min_link_to_charge_distance(),
    )
    assert "1.3" in fig.axes[0].get_title(), fig.axes[0].get_title()
    close(fig)

    opt = ferric.run_optimize(
        z1.qm_molecule(), "sto-3g", point_charges=pc, max_steps=60
    )
    assert opt.converged
    fig = optimization_trace(opt.energy_trace, converged=opt.converged)
    assert fig is not None
    close(fig)


def test_imaginary_mode_and_reaction_path_accept_what_ferric_returns():
    """Saddle -> mode -> IRC -> barriers, in an MM field, no literals."""
    mol = ferric.Molecule.from_xyz_string(
        "4\nnh3\nN 0 0 0\nH 0.0 1.01 0.1\nH 0.8747 -0.505 0.1\nH -0.8747 -0.505 0.1\n"
    )
    field = [(-0.4, 0.0, 0.0, 6.0), (-0.4, 0.0, 0.0, -6.0)]

    sad = ferric.run_saddle(mol, "sto-3g", max_steps=40, point_charges=field)
    assert sad.converged and sad.n_imaginary == 1
    # `imaginary_mode` is None unless n_imaginary == 1 -- asserted above, so
    # this is the case where the plot must accept it.
    fig = imaginary_mode(sad.symbols, sad.imaginary_mode)
    assert fig is not None
    close(fig)

    xyz = f"{len(sad.symbols)}\nsaddle\n" + "".join(
        f"{a} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n"
        for a, c in zip(sad.symbols, sad.coords)
    )
    irc = ferric.run_irc(
        ferric.Molecule.from_xyz_string(xyz),
        "sto-3g",
        mode=sad.imaginary_mode,
        max_steps=120,
        step=0.15,
        point_charges=field,
    )
    # These are METHODS, not properties. A plot fed the bound method instead of
    # the float draws a TypeError, and that mistake was made once already.
    fig = reaction_path(
        saddle_energy=irc.saddle_energy,
        forward_energy=irc.forward.energy,
        reverse_energy=irc.reverse.energy,
        forward_converged=irc.forward.converged,
        reverse_converged=irc.reverse.converged,
    )
    assert fig is not None
    close(fig)
    assert irc.forward_barrier() > 0 and irc.reverse_barrier() > 0
