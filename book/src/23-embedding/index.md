# 23. Embedding and QM/MM

The environment of a quantum region treated classically. Section 23.1 is the
operator everything else reuses: fixed point charges, Gaussian-smeared
charges and a uniform field added to the one-electron Hamiltonian. Section
23.2 partitions a large structure into a QM region and charges, caps the cut
bonds and assembles the gradient over the full structure. Section 23.3 lets
the environment respond, with Thole-damped induced dipoles. Section 23.4 is
the AMBER-form force field that supplies the MM terms.

A run with no environment is bit-identical to the gas-phase run in every
case in this chapter: an empty charge list, an empty site list and a `None`
topology each take the original code path.
