# Short-range MP2 with long-range RPA {#rs-mp2-rpa}

This chapter is about one method: second-order perturbation theory at short
range, combined with the direct random-phase approximation at long range.
MP2 is accurate where electrons are close together. At long range its
unscreened ring diagrams overbind polarizable systems, because nothing
weakens the interaction between two distant fragments. The dRPA resums
those rings and screens the interaction. Splitting the Coulomb operator lets
each method handle the range it is good at. The split has to keep the
second-order term in which one interaction is short-range and the other
long-range, and must count nothing twice.

The chapter covers only what the code commits:

- section @sec:cross-term, second order in a split interaction;
- section @sec:two-formulations, the two formulations `rs_mp2_lr_rpa` evaluates, and
  the terf long range, whose kernel is not positive definite;
- section @sec:shared-intermediates, the intermediates the two kernels share.

Sections @sec:ring-series, @sec:coloring-the-rings, @sec:omega-scale-law and @sec:measurements are deferred until the work they draw on is
published.

Notation follows chapter @sec:notation. $\Delta_{ij}^{ab} = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j > 0$,
$K_{ia,jb} = 2(ia|jb)$ as in section @sec:drpa-three-ways, and $\omega$ is the
range-separation parameter, never a frequency. $v_{\mathrm{sr}} + v_{\mathrm{lr}} = 1/r$
is the operator split; C, sr and lr label integrals and energies of the
Coulomb, short-range and long-range operators.
