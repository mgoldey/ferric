# 2. Molecules and numerics

Every method in this book takes the same two inputs: a molecule and a basis
set. This chapter describes both, and the numerical rules the later chapters
assume: lengths are in Bohr, a sum is added in a fixed order, a large
allocation is charged against one memory budget, and a tensor contraction is
a matrix product.

Sections:

- [2.1 Molecule, charge, multiplicity and units](01-molecule.md)
- [2.2 Contracted Gaussians, normalization and parsers](02-basis-sets.md)
- [2.3 Determinism, threads and the memory budget](03-determinism-and-memory.md)
- [2.4 Dense tensors and the einsum macro](04-tensors.md)
