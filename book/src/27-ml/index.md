# 27. ML integration touchpoints

A model trained on quantum-chemical labels inherits every convention the
labels were computed with. If one sample's charges are Löwdin and the next
one's are Hirshfeld, or one polarizability was converged and the next was
not, the model fits the difference as if it were chemistry. So a dataset
builder needs four things from the code: the definition of each array, its
units and shape, what produced it, and how far it has been checked.

This chapter supplies those for ferric's outputs, read from the code. It is
for someone wiring ferric into a machine-learning model as input features,
as training labels, or as the provenance record of a dataset. It does not
describe models. The derivations behind the polarizabilities and charges are
in chapter 21, and those behind $C_6$ in chapter 25. Here the subject is the
arrays a consumer receives.

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
