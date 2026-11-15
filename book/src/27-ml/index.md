# 27. ML integration touchpoints

This chapter is for someone wiring ferric outputs into a machine-learning
model: as input features, as training labels, or as provenance for a dataset.
It describes what each exported quantity is, as computed by the code, and how
far it can be trusted. It does not describe models.

| § | Read it when you need |
|---|---|
| 27.1 | the NPZ feature bundle: every key, its definition, shape, units and frame behaviour |
| 27.2 | the electrostatic features: ESP, field, multipoles and the five charge schemes |
| 27.3 | polarizabilities (molecular, per-atom, charge transfer), PDEP modes and $C_6$ |
| 27.4 | the Python surface, cube files and the certificate as a provenance label |
| 27.5 | bit-reproducibility rules and which quantity carries which validation grade |
| 27.6 | energies, gradients, Hessians and response properties as training labels |
| 27.7 | one worked consumer, the esp-conditioning dataset builder |

Units are atomic units throughout: Bohr, Hartree, $e$, and $e\,a_0$ for
dipoles. The user guide pages `site/src/using/export.md`,
`site/src/using/ase.md` and `site/src/using/qcschema.md` cover the same
outputs from the user's side.
