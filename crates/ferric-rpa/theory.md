# Random Phase Approximation (RPA) Theory

This module implements the direct Random Phase Approximation (dRPA) correlation energy using the Projective Dielectric Eigenpotential (PDEP) approach.

## RPA Correlation Energy

The RPA correlation energy is commonly evaluated using the Adiabatic-Connection Fluctuation-Dissipation (ACFD) theorem via an imaginary frequency integration:

$$ E_c^{\text{RPA}} = \frac{1}{2\pi} \int_0^\infty \text{Tr}\left[ \ln(\mathbf{1} - \mathbf{\chi}_0(i\omega) \mathbf{v}) + \mathbf{\chi}_0(i\omega) \mathbf{v} \right] d\omega $$

where $\mathbf{\chi}_0(i\omega)$ is the independent-particle polarizability and $\mathbf{v}$ is the Coulomb interaction. The term $\mathbf{\epsilon}(i\omega) = \mathbf{1} - \mathbf{\chi}_0(i\omega) \mathbf{v}$ is the interacting dielectric matrix.

## The PDEP Eigenpotential Approach

Instead of explicitly constructing and diagonalizing the large dense dielectric matrix $\mathbf{\epsilon}(i\omega)$ at each frequency point, the PDEP approach exploits the fact that the spectrum of the dielectric operator decays rapidly.

1. **Static Dielectric Eigenpotentials:** The static ($\omega=0$) dielectric operator $\mathbf{\epsilon}(0)$ is diagonalized iteratively using the Davidson algorithm to find the dominant eigenpotentials (often called plasmon density eigenpotentials) $|V_\alpha\rangle$:
   $$ \mathbf{\epsilon}(0) |V_\alpha\rangle = \lambda_\alpha(0) |V_\alpha\rangle $$
   Because the eigenvalues decay steeply, only a small number of eigenpotentials with eigenvalues $\lambda_\alpha(0) > \epsilon_{\text{trunc}}$ need to be retained. This forms a highly compressed, system-adapted optimal basis.

2. **Sternheimer Linear Response:** The action of the dielectric matrix on a trial potential is computed efficiently without an explicit summation over all unoccupied (virtual) states. This is achieved by solving the Sternheimer linear response equation for each occupied orbital $i$:
   $$ (\hat{H}_0 - \epsilon_i \mathbf{1} + i\omega \mathbf{1}) |\Delta \Psi_i(i\omega, V)\rangle = - P_{\text{virt}} \hat{V} |\Psi_i\rangle $$
   where $P_{\text{virt}}$ is the projector onto the virtual space.

3. **Frequency-Dependent Eigenvalues:** Assuming the static eigenpotential basis $|V_\alpha\rangle$ is adequate for all frequencies, the frequency-dependent eigenvalues $\lambda_\alpha(i\omega)$ are evaluated by projecting the frequency-dependent dielectric operator onto the fixed static PDEP basis:
   $$ \lambda_\alpha(i\omega) = \langle V_\alpha | \mathbf{\epsilon}(i\omega) | V_\alpha \rangle $$

4. **Energy Integration:** The correlation energy trace is then evaluated via numerical quadrature (e.g., MiniMax or mapped Gauss-Legendre) over the imaginary frequency grid:
   $$ E_c^{\text{RPA}} = \frac{1}{2\pi} \sum_k w_k \sum_\alpha \left[ \ln(\lambda_\alpha(i\omega_k)) + (1 - \lambda_\alpha(i\omega_k)) \right] $$

This low-scaling formulation avoids the heavy $\mathcal{O}(N^6)$ cost or massive $\mathcal{O}(N^4)$ memory overheads typically associated with forming and manipulating $\mathbf{\chi}_0(i\omega)$ across all frequencies.

## Literature Citations

1. **PDEP Eigenpotentials:** F. Gygi and G. Galli, *Phys. Rev. B* **65**, 220102(R) (2002).
2. **PDEP-RPA Correlation Energy:** D. Rocca, D. Lu, and G. Galli, *J. Chem. Phys.* **133**, 164109 (2010).
3. **MiniMax Quadrature:** A. Takatsuka, S. Ten-no, and W. Hackbusch, *J. Chem. Phys.* **129**, 044112 (2008).
4. **RI-RPA Review:** H. Eshuis, J. E. Bates, and F. Furche, *Theor. Chem. Acc.* **131**, 1084 (2012).
