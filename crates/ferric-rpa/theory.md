# Random Phase Approximation (RPA) Theory

This module implements the direct Random Phase Approximation (dRPA) correlation energy using the Projective Dielectric Eigenpotential (PDEP) approach.

## RPA Correlation Energy

The RPA correlation energy is commonly evaluated using the Adiabatic-Connection Fluctuation-Dissipation (ACFD) theorem via an imaginary frequency integration:

$$ E_c^{\text{RPA}} = \frac{1}{2\pi} \int_0^\infty \text{Tr}\left[ \ln(\mathbf{1} - \mathbf{\chi}_0(i\omega) \mathbf{v}) + \mathbf{\chi}_0(i\omega) \mathbf{v} \right] d\omega $$

where $\mathbf{\chi}_0(i\omega)$ is the independent-particle polarizability and $\mathbf{v}$ is the Coulomb interaction. The term $\mathbf{\epsilon}(i\omega) = \mathbf{1} - \mathbf{\chi}_0(i\omega) \mathbf{v}$ is the interacting dielectric matrix.

## The PDEP Eigenpotential Approach

Instead of explicitly constructing and diagonalizing the large dense dielectric matrix $\mathbf{\epsilon}(i\omega)$ at each frequency point, the PDEP approach uses the rapid decay of the dielectric operator's spectrum.

1. **Static Dielectric Eigenpotentials:** The static ($\omega=0$) dielectric operator $\mathbf{\epsilon}(0)$ is diagonalized iteratively using the Davidson algorithm to find the dominant eigenpotentials (often called plasmon density eigenpotentials) $|V_\alpha\rangle$:
   $$ \mathbf{\epsilon}(0) |V_\alpha\rangle = \lambda_\alpha(0) |V_\alpha\rangle $$
   Because the eigenvalues decay steeply, only a small number of eigenpotentials with eigenvalues $\lambda_\alpha(0) > \epsilon_{\text{trunc}}$ need to be retained. They form a compressed, system-adapted basis.

2. **Sternheimer Linear Response:** The action of the dielectric matrix on a trial potential is written in the Sternheimer form, which avoids an explicit sum over virtual states in a plane-wave or real-space implementation. The Sternheimer equation for each occupied orbital $i$ is the Sternheimer linear response equation for each occupied orbital $i$:
   $$ (\hat{H}_0 - \epsilon_i \mathbf{1} + i\omega \mathbf{1}) |\Delta \Psi_i(i\omega, V)\rangle = - P_{\text{virt}} \hat{V} |\Psi_i\rangle $$
   where $P_{\text{virt}}$ is the projector onto the virtual space.

   ferric's RI-MO implementation does sum over every occupied-virtual pair: `sternheimer::chi_from_trial_potential` (`crates/ferric-rpa/src/sternheimer.rs:330`) evaluates $\chi_0$ as $-2\sum_{ia} e_{ia}/(\omega^2+e_{ia}^2)\,|\sum_P V_P B^P_{ia}|^2$. The Sternheimer form is the formulation; PDEP compresses the result of the sum.

3. **Frequency-Dependent Eigenvalues:** Assuming the static eigenpotential basis $|V_\alpha\rangle$ is adequate for all frequencies, the frequency-dependent eigenvalues $\lambda_\alpha(i\omega)$ are evaluated by projecting the frequency-dependent dielectric operator onto the fixed static PDEP basis:
   $$ \lambda_\alpha(i\omega) = \langle V_\alpha | \mathbf{\epsilon}(i\omega) | V_\alpha \rangle $$

4. **Energy Integration:** The correlation energy trace is then evaluated via numerical quadrature (e.g., MiniMax or mapped Gauss-Legendre) over the imaginary frequency grid:
   $$ E_c^{\text{RPA}} = \frac{1}{2\pi} \sum_k w_k \sum_\alpha \left[ \ln(\lambda_\alpha(i\omega_k)) + (1 - \lambda_\alpha(i\omega_k)) \right] $$

This formulation avoids the $\mathcal{O}(N^6)$ cost and $\mathcal{O}(N^4)$ memory associated with forming and manipulating $\mathbf{\chi}_0(i\omega)$ across all frequencies.

## Open-Shell (Spin-Summed) Dielectric

`run_u_pdep_rpa` (`crates/ferric-rpa/src/lib.rs`) takes a UHF reference, or a ROHF reference semi-canonicalized per spin. Each spin $\sigma$ has its own RI tensor $B^P_{i_\sigma a_\sigma}$ and gaps $e_{ia}^\sigma = \epsilon_a^\sigma - \epsilon_i^\sigma$ in one shared aux basis, and the symmetrized dielectric is

$$ \tilde{\mathbf{\epsilon}}_U(i\omega) = \mathbf{1} + \mathbf{\Pi}_\alpha(i\omega) + \mathbf{\Pi}_\beta(i\omega), \qquad \Pi^\sigma_{PQ}(i\omega) = 2 \sum_{ia} \frac{e^\sigma_{ia}}{\omega^2 + (e^\sigma_{ia})^2} B^P_{i_\sigma a_\sigma} B^Q_{i_\sigma a_\sigma}. $$

The closed-shell $\mathbf{\Pi}$ carries the prefactor 4; with identical spin channels $\mathbf{\Pi}_\alpha + \mathbf{\Pi}_\beta$ equals it, so a UHF singlet reproduces `run_pdep_rpa`. The PDEP eigensolve, truncation, frequency projection and trace-log integration are those of the closed-shell path applied to $\tilde{\mathbf{\epsilon}}_U$; the matvec $\tilde{\mathbf{\epsilon}}_U \mathbf{V} = \mathbf{V} + \mathbf{\Pi}_\alpha \mathbf{V} + \mathbf{\Pi}_\beta \mathbf{V}$ is two closed-shell-shaped applications (`sternheimer::dielectric_apply_unrestricted`). The Laplace $\chi_0$ backend builds one Laplace quadrature per spin on that spin's gap range. `run_diagnostics` assembles $\tilde{\mathbf{\epsilon}}_U(i\omega_k)$ densely at every quadrature node (`diagnostics::u_ri_drpa_energy`).

The Boys-screened $\chi_0$ is closed-shell only: its screened tiles and localized orbital energies are built from the restricted MOs and Fock matrix.

## Literature Citations

1. **PDEP Eigenpotentials:** F. Gygi and G. Galli, *Phys. Rev. B* **65**, 220102(R) (2002).
2. **PDEP-RPA Correlation Energy:** D. Rocca, D. Lu, and G. Galli, *J. Chem. Phys.* **133**, 164109 (2010).
3. **MiniMax Quadrature:** A. Takatsuka, S. Ten-no, and W. Hackbusch, *J. Chem. Phys.* **129**, 044112 (2008).
4. **RI-RPA Review:** H. Eshuis, J. E. Bates, and F. Furche, *Theor. Chem. Acc.* **131**, 1084 (2012).
