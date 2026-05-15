# RI-MP2 Theory

This module implements the Resolution of Identity (RI) Møller-Plesset Perturbation Theory of second order (MP2).

## Canonical MP2
The second-order correlation energy for a closed-shell system is given by:
$$E_{corr} = -\sum_{ijab} \frac{(ia|jb)[2(ia|jb) - (ib|ja)]}{\epsilon_a + \epsilon_b - \epsilon_i - \epsilon_j}$$
where $i, j$ are occupied orbitals and $a, b$ are virtual orbitals.

## Resolution of Identity (RI)
The 4-center electron repulsion integrals (ERIs) are approximated using a 3-center auxiliary basis set $\{P\}$:
$$(ia|jb) \approx \sum_{PQ} (ia|P) [V^{-1}]_{PQ} (Q|jb)$$
where $V_{PQ} = (P|Q)$ is the Coulomb metric of the auxiliary basis.
Defining the "dressed" 3-index amplitudes:
$$B^P_{ia} = \sum_Q (ia|Q) [V^{-1/2}]_{PQ}$$
the ERIs become:
$$(ia|jb) \approx \sum_P B^P_{ia} B^P_{jb}$$

## Spin-Component Scaling (SCS-MP2)
The energy is decomposed into opposite-spin (OS) and same-spin (SS) parts:
$$E_{OS} = \sum_{ijab} \frac{(ia|jb)^2}{\Delta_{ijab}}, \quad E_{SS} = \sum_{ijab} \frac{(ia|jb)[(ia|jb) - (ib|ja)]}{\Delta_{ijab}}$$
$$E_{SCS} = c_{OS} E_{OS} + c_{SS} E_{SS}$$
Standard parameters: $c_{OS} = 1.2$, $c_{SS} = 0.33$.

## RI-Laplace MP2
Using the Laplace transform:
$$\frac{1}{x} = \int_0^\infty e^{-tx} dt \approx \sum_k w_k e^{-t_k x}$$
The energy is evaluated in the AO basis using pseudo-densities $P(t)$ and $Q(t)$:
$$P(t)_{\mu\nu} = \sum_i C_{\mu i} e^{t \epsilon_i} C_{\nu i}, \quad Q(t)_{\mu\nu} = \sum_a C_{\mu a} e^{-t \epsilon_a} C_{\nu a}$$
$$E_{corr} \approx -\sum_k w_k \sum_{PQ} \left[ 2 \text{Tr}(M^P N^Q) \text{Tr}(M^Q N^P) - \text{Tr}(M^P N^Q M^Q N^P) \right]$$
where $M^P = B^P P(t)$ and $N^P = B^P Q(t)$.
