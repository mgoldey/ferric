# ML integration touchpoints {#ml}

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
in chapter @sec:polarizabilities, and those behind $C_6$ in chapter @sec:dispersion. Here the subject is the
arrays a consumer receives.

| Section | Read it when you need |
|---|---|
| @sec:npz-bundle | the NPZ feature bundle: every key, its definition, shape, units and frame behaviour |
| @sec:electrostatics-and-charges | the electrostatic features: ESP, field, multipoles and the five charge schemes |
| @sec:polarizability-pdep-c6 | polarizabilities (molecular, per-atom, charge transfer), PDEP modes and $C_6$ |
| @sec:python-cube-provenance | the Python surface, cube files and the certificate as a provenance label |
| @sec:reproducibility-and-grades | bit-reproducibility rules and which quantity carries which validation grade |
| @sec:training-labels | energies, gradients, Hessians and response properties as training labels |
| @sec:esp-conditioning | one worked consumer, the esp-conditioning dataset builder |

Units are atomic units throughout: Bohr, Hartree, $e$, and $e\,a_0$ for
dipoles. The user guide pages `site/src/using/export.md`,
`site/src/using/ase.md` and `site/src/using/qcschema.md` cover the same
outputs from the user's side.
