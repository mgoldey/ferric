# Molecules and numerics {#molecules-and-numerics}

Every method in this book takes the same two inputs: a molecule and a basis
set. This chapter describes both, and the numerical rules the later chapters
assume: lengths are in Bohr, a sum is added in a fixed order, a large
allocation is charged against one memory budget, and a tensor contraction is
a matrix product.

Sections:

- [@sec:molecule Molecule, charge, multiplicity and units](01-molecule.md)
- [@sec:basis-sets Contracted Gaussians, normalization and parsers](02-basis-sets.md)
- [@sec:determinism-and-memory Determinism, threads and the memory budget](03-determinism-and-memory.md)
- [@sec:tensors Dense tensors and the einsum macro](04-tensors.md)
