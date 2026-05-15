# Self-Consistent Field (SCF) Theory

This module implements the Restricted Hartree-Fock (RHF) method and associated optimization techniques.

## Hartree-Fock Equations
The RHF energy is given by:
$$E_{HF} = 2 \sum_i H_{ii} + \sum_{ij} [2(ii|jj) - (ij|ij)] + V_{nn}$$
where $H$ is the core Hamiltonian.

The Fock matrix in the AO basis is:
$$F_{\mu\nu} = H_{\mu\nu} + \sum_{\lambda\sigma} D_{\lambda\sigma} [2(\mu\nu|\lambda\sigma) - (\mu\lambda|\nu\sigma)]$$
where $D_{\lambda\sigma} = \sum_i C_{\lambda i} C_{\sigma i}$ is the density matrix.

## Convergence Acceleration (DIIS)
Direct Inversion in the Iterative Subspace (DIIS) minimizes the error vector:
$$\mathbf{e}_k = \mathbf{F}_k \mathbf{D}_k \mathbf{S} - \mathbf{S} \mathbf{D}_k \mathbf{F}_k$$
A linear combination of past Fock matrices is formed:
$$\mathbf{F}_{k+1} = \sum_i c_i \mathbf{F}_i$$
where the coefficients $c_i$ are found by solving the DIIS system.

## Continuous Fast Multipole Method (CFMM)
To avoid the $O(N^4)$ cost of direct ERI evaluation, CFMM treats the long-range Coulomb interaction using multipole expansions:
$$J(\mathbf{r}) \approx \sum_{l,m} M_{lm} \frac{Y_{lm}(\theta, \phi)}{r^{l+1}}$$
This allows for $O(N)$ scaling in the Coulomb build.
