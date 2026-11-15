# 26. Short-range MP2 with long-range RPA

This chapter is about one method: second-order perturbation theory at short
range, combined with the direct random-phase approximation at long range.
MP2 is accurate where electrons are close together. It fails at long range,
where its unscreened ring diagrams overbind polarizable systems. The dRPA
resums those rings. Splitting the Coulomb operator lets each method handle
the range it is good at, but the split has to be done so that nothing is
counted twice.

The chapter covers only what the code commits:

- section 26.1, second order in a split interaction;
- section 26.4, the two formulations `rs_mp2_lr_rpa` evaluates;
- section 26.6, the intermediates the two kernels share.

Sections 26.2, 26.3, 26.5 and 26.7 are deferred until the work they draw on is
published.

Notation follows chapter 0. $\Delta_{ij}^{ab} = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j > 0$,
$K_{ia,jb} = 2(ia|jb)$ as in section 18.1, and $\omega$ is the
range-separation parameter, never a frequency. $v_{\mathrm{sr}} + v_{\mathrm{lr}} = 1/r$
is the operator split; C, sr and lr label integrals and energies of the
Coulomb, short-range and long-range operators.
