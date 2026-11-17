# 4. Range-separated and attenuated operators

The two-electron operators other than $1/r$ that ferric evaluates, and the
parameters that tie them together. Notation follows chapter 0, section 8.

Most of the book uses the Coulomb kernel $1/r_{12}$. This chapter covers
the kernels that replace it. Section 4.1 splits $1/r$ into a short-range
and a long-range part, with the error function or with the tempered pair
terf and terfc, and derives the curvature constraint that fixes the terf
sharpness. Section 4.2 shows how ferric computes terf integrals, which
libint2 cannot: a McMurchie–Davidson pass with the Boys function replaced
by a two-variable function read from tables. Section 4.3 covers the
exponentially decaying kernels, Yukawa and the Slater geminal, which come
from libint2.
