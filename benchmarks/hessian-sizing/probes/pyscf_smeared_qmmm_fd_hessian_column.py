import numpy as np
from pyscf import gto, qmmm, scf

mm_xyz = np.array([[1.3, -0.4, 2.1], [-2.0, 1.0, 1.5]])  # Bohr
mm_q = np.array([0.4, -0.6])
radii = np.array([1.2, 0.9])  # Bohr widths


def grad(dz):
    mol = gto.M(
        atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        basis="sto-3g",
        verbose=0,
    )
    mol.set_geom_(
        [
            ["O", (0, 0, 0.1173 / 0.529177210903 * 0.529177210903)],
            ["H", (0, 0.7572, -0.4692)],
            ["H", (0, -0.7572, -0.4692 + dz)],
        ],
        unit="Angstrom",
    )
    mf = qmmm.mm_charge(scf.RHF(mol), mm_xyz, mm_q, radii, unit="Bohr")
    mf.conv_tol = 1e-12
    mf.kernel()
    g = mf.nuc_grad_method().kernel()
    return g


h = 1e-3  # Angstrom
g = (grad(h) - grad(-h)) / (2 * h * 1.8897261246)  # d g / d z (Bohr)
print("PySCF smeared QM/MM FD-of-gradient column (d g / d z_H2) Ha/Bohr^2:")
print(g)
