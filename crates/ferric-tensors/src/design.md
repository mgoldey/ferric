# Design Note: Perturbation Theory with Modified Coulomb Operators in Coupled Cluster

## 1. Context: Modified Coulomb Operators
Modified operators are typically used in Range-Separated Hybrid (RSH) or Attenuated methods:
- **Long-range (LR)**: $v_{lr}(r) = \frac{\text{erf}(\omega r)}{r}$
- **Short-range (SR)**: $v_{sr}(r) = \frac{\text{erfc}(\omega r)}{r}$
- **Screened**: $v_{sc}(r) = \frac{e^{-\alpha r}}{r}$

## 2. Brainstorming: CC Integration

### A. Range-Separated Coupled Cluster (RS-CC)
The Hamiltonian is partitioned: $\hat{H} = \hat{h} + \hat{W}_{sr} + \hat{W}_{lr}$.
- **Method**: Solve the CC equations using ONLY the short-range operator $\hat{W}_{sr}$. This captures the complex, local dynamic correlation.
- **Correction**: Use Perturbation Theory (e.g., MP2 or a custom PT) for the long-range part $\hat{W}_{lr}$.
- **Benefit**: Long-range correlation is often smoother and better described by simpler methods or even long-range DFT. This reduces the basis set incompleteness error (BSIE) since the SR part converges faster with basis size.

### B. Attenuated (T) Correction: CCSD(att-T)
Standard CCSD(T) is $O(N^7)$.
- **Idea**: Compute the (T) correction using an attenuated Coulomb operator.
- **Mechanism**: The triples correction is predominantly local. By using an attenuated operator, we can effectively "screen" distant triple excitations, potentially allowing for a reduced-scaling triples implementation without the complexity of full local correlation methods (like PNO-LCCSD(T)).
- **Hybrid Approach**: Use full 1/r for local triples and attenuated for mid-range triples.

### C. Perturbatively Corrected Modified-CC
Start with a Coupled Cluster calculation using a modified operator $v_{mod}$ (which might be easier to compute or screen).
- Use PT to "add back" the difference $\Delta v = 1/r - v_{mod}$.
- This is analogous to how RI-MP2 uses an approximate metric, but here we approximate the operator itself in the iterative CC part.

## 3. Implementation in Ferric
- `ferric-integrals`: Needs support for `erfc` attenuated operators (libint2 supports this via the `erfc_coulomb` operator).
- `ferric-tensors`: Must handle the different sets of integrals (SR vs LR) efficiently.
- `ferric-mp2`: Already has Att-MP2; this can be the starting point for RS-CC.
