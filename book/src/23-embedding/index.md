# Embedding and QM/MM {#embedding}

A reaction in an enzyme or a solute in water involves thousands of atoms, but
the electrons that rearrange belong to a few dozen of them. The rest of the
structure acts on that region mainly through its electric field. This chapter
treats the reactive region quantum mechanically and its environment
classically.

Section @sec:external-potentials is the operator everything else reuses: fixed point charges,
Gaussian-smeared charges and a uniform field added to the one-electron
Hamiltonian. Section @sec:qmmm-boundaries partitions a large structure into a QM region and
charges, caps the cut bonds and assembles the gradient over the full
structure. Section @sec:polarizable-embedding lets the environment respond, with Thole-damped
induced dipoles. Section @sec:molecular-mechanics is the AMBER-form force field that supplies the
MM terms.

A run with no environment is bit-identical to the gas-phase run in every
case in this chapter: an empty charge list, an empty site list and a `None`
topology each take the original code path.
