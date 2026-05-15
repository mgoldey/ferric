# Coupled Cluster Theory

This module implements Coupled Cluster (CC) methods, including CCD, CCSD, and CCSD(T).

## Exponential Ansatz
The Coupled Cluster wavefunction is given by:
$$|\Psi_{CC}\rangle = e^{\hat{T}} |\Phi_0\rangle$$
where $\hat{T} = \hat{T}_1 + \hat{T}_2 + \dots$.

## CCD Equations
For CCD, we solve for $T_2$ amplitudes:
$$\langle ab | \hat{H}_N e^{\hat{T}_2} | ij \rangle_C = 0$$
The correlation energy is:
$$E_{corr} = \sum_{ijab} (ia|jb) (2 T_{ij}^{ab} - T_{ji}^{ab})$$

## CCSD Residuals
The residuals $R_1$ and $R_2$ are solved iteratively:
$$R_i^a = \langle a | \hat{H}_N e^{\hat{T}_1 + \hat{T}_2} | i \rangle_C = 0$$
$$R_{ij}^{ab} = \langle ab | \hat{H}_N e^{\hat{T}_1 + \hat{T}_2} | ij \rangle_C = 0$$

### Particle-Hole Intermediates
To implement CCSD efficiently, we use intermediates:
- $F_{vv}, F_{oo}, F_{ov}$ (Effective Fock matrices)
- $W_{oooo}, W_{vvvv}, W_{ovov}$ (Effective 2-electron integrals)

## CCSD(T)
The perturbative triples correction is computed after CCSD convergence:
$$E_{(T)} = \sum_{ijkabc} \frac{t_{ijk}^{abc} (4 w_{ijk}^{abc} + w_{ikj}^{bca} + w_{ikj}^{abc})}{\epsilon_i + \epsilon_j + \epsilon_k - \epsilon_a - \epsilon_b - \epsilon_c}$$
where $t_{ijk}^{abc}$ and $w_{ijk}^{abc}$ are triples amplitudes formed from $T_1$ and $T_2$.
