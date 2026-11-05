# D3(BJ) Dispersion: Energy and Analytic Gradient

Two-body Grimme D3 with Becke-Johnson (rational) damping, as implemented in
`src/lib.rs`. Zero damping, the three-body (ATM, $s_9$) term and periodic
boundary conditions are not implemented. Coordinates are in Bohr, energies in
Hartree.

The identities below are proved by
`scripts/d3_cn_chain_rule_gradient_sympy.py` (exits nonzero on mismatch; one
negative control per claim), and the full gradient is checked against finite
differences of the energy in `tests/gradient_vs_fd.rs`.

## Coordination number

For atoms $A \neq B$ with $R_{AB} = |\mathbf{R}_A - \mathbf{R}_B|$,

$$g_{AB}(R_{AB}) = \frac{1}{1 + \exp\!\left[-k\left(\frac{R^{\mathrm{cov}}_{AB}}{R_{AB}} - 1\right)\right]},
\qquad
\mathrm{CN}_A = \sum_{B \neq A} g_{AB}(R_{AB}),$$

with $k = 16$ (`KCN`), $R^{\mathrm{cov}}_{AB} = R^{\mathrm{cov}}_A + R^{\mathrm{cov}}_B$ and
$R^{\mathrm{cov}}_A = \tfrac{4}{3}\,r^{\mathrm{cov}}_A$ (2009 covalent radii, converted to
Bohr; `rcov`). Implemented in `coordination_numbers`. Each pair term $g_{AB}$
enters both $\mathrm{CN}_A$ and $\mathrm{CN}_B$.

$$\frac{\partial g_{AB}}{\partial R_{AB}} = -\frac{k\,R^{\mathrm{cov}}_{AB}}{R_{AB}^2}\,g_{AB}\,(1 - g_{AB}).$$

## C6 interpolation

Element $A$ has $N_A$ reference systems with coordination numbers
$\mathrm{CN}^{\mathrm{ref}}_{A,i}$ and tabulated $C_6^{\mathrm{ref}}(A_i, B_j)$. With
Gaussian weights ($w_f = 4$, `WF`; `reference_weights`)

$$W_{A,i}(\mathrm{CN}_A) = \exp\!\left[-w_f\left(\mathrm{CN}_A - \mathrm{CN}^{\mathrm{ref}}_{A,i}\right)^2\right],$$

the interpolated coefficient is (`c6_interpolated`)

$$C_6^{AB} = \frac{N_{AB}}{D_{AB}},\qquad
N_{AB} = \sum_{i,j} C_6^{\mathrm{ref}}(A_i,B_j)\,W_{A,i}W_{B,j},\qquad
D_{AB} = \sum_{i,j} W_{A,i}W_{B,j}.$$

When $D_{AB} \le 10^{-300}$ (all weights underflow) the nearest reference on
each side is used; that branch is piecewise constant in CN.

$$\frac{\partial W_{A,i}}{\partial \mathrm{CN}_A} = -2w_f\left(\mathrm{CN}_A - \mathrm{CN}^{\mathrm{ref}}_{A,i}\right)W_{A,i},
\qquad
\frac{\partial C_6^{AB}}{\partial \mathrm{CN}_A} = \frac{1}{D_{AB}}\left(\frac{\partial N_{AB}}{\partial \mathrm{CN}_A} - C_6^{AB}\,\frac{\partial D_{AB}}{\partial \mathrm{CN}_A}\right),$$

and likewise for $\mathrm{CN}_B$ (`c6_interpolated_derivatives`; it returns
$(0,0)$ in the underflow branch).

## C8 and the damping radius

$$Q_A = \tfrac12\sqrt{Z_A}\,\frac{\langle r^4\rangle_A}{\langle r^2\rangle_A},\qquad
q_{AB} = \sqrt{Q_A Q_B},\qquad
C_8^{AB} = 3\,C_6^{AB}\,q_{AB}$$

(`sqrt_q` returns $\sqrt{Q_A}$). The BJ radius is

$$R_0^{AB} = \sqrt{C_8^{AB}/C_6^{AB}} = \sqrt{3\,q_{AB}},\qquad f_{AB} = a_1 R_0^{AB} + a_2 .$$

$C_6$ cancels in $C_8/C_6$, so $f_{AB}$ depends only on the two elements and
contributes no gradient term.

## Energy

$$E = -\sum_{A<B}\left[\frac{s_6\,C_6^{AB}}{R_{AB}^6 + f_{AB}^6} + \frac{s_8\,C_8^{AB}}{R_{AB}^8 + f_{AB}^8}\right]
\equiv \sum_{A<B} E_{AB}\!\left(R_{AB},\,C_6^{AB}(\mathrm{CN}_A,\mathrm{CN}_B)\right).$$

Implemented in `d3bj_energy`; parameters $(s_6, s_8, a_1, a_2)$ per
functional in `params.rs`. Fewer than two atoms gives exactly $0$.

## Gradient

$E$ depends on $\mathbf{R}_X$ explicitly through every $R_{AB}$ and implicitly
through every $\mathrm{CN}_A$, because $C_6^{AB}$ depends on $\mathrm{CN}_A$ and
$\mathrm{CN}_B$, and every CN depends on every distance:

$$\frac{dE}{d\mathbf{R}_X}
= \underbrace{\sum_{A<B}\frac{\partial E_{AB}}{\partial R_{AB}}\frac{\partial R_{AB}}{\partial \mathbf{R}_X}}_{\text{explicit}}
+ \underbrace{\sum_{A}\frac{\partial E}{\partial \mathrm{CN}_A}\frac{\partial \mathrm{CN}_A}{\partial \mathbf{R}_X}}_{\text{CN chain rule}} .$$

Explicit part, with $d_6 = R^6 + f^6$, $d_8 = R^8 + f^8$:

$$\frac{\partial E_{AB}}{\partial R_{AB}} = \frac{6\,s_6\,C_6^{AB}R_{AB}^5}{d_6^2} + \frac{8\,s_8\,C_8^{AB}R_{AB}^7}{d_8^2},
\qquad
\frac{\partial R_{AB}}{\partial \mathbf{R}_A} = -\frac{\partial R_{AB}}{\partial \mathbf{R}_B} = \frac{\mathbf{R}_A - \mathbf{R}_B}{R_{AB}} .$$

Chain part. $E_{AB}$ is linear in $C_6^{AB}$ (including through $C_8$):

$$\frac{\partial E_{AB}}{\partial C_6^{AB}} = -\left(\frac{s_6}{d_6} + \frac{3\,q_{AB}\,s_8}{d_8}\right),
\qquad
\frac{\partial E}{\partial \mathrm{CN}_A} = \sum_{B \neq A}\frac{\partial E_{AB}}{\partial C_6^{AB}}\,\frac{\partial C_6^{AB}}{\partial \mathrm{CN}_A}.$$

Since $\mathrm{CN}_A = \sum_{C\neq A} g_{AC}(R_{AC})$ and $g_{AB}$ appears in both
$\mathrm{CN}_A$ and $\mathrm{CN}_B$,

$$\sum_{A}\frac{\partial E}{\partial \mathrm{CN}_A}\frac{\partial \mathrm{CN}_A}{\partial \mathbf{R}_X}
= \sum_{A<B}\left(\frac{\partial E}{\partial \mathrm{CN}_A} + \frac{\partial E}{\partial \mathrm{CN}_B}\right)
\frac{\partial g_{AB}}{\partial R_{AB}}\,\frac{\partial R_{AB}}{\partial \mathbf{R}_X}.$$

`d3bj_gradient` evaluates this in two $O(N^2)$ passes: the first adds the
explicit term and accumulates the $N$ scalars $\partial E/\partial \mathrm{CN}_A$
(`de_dcn`); the second contracts them with $\partial g_{AB}/\partial R_{AB}$.
Moving atom $X$ changes $C_6^{AB}$ for pairs that do not contain $X$; the chain
part carries that.

## Verification

| Claim | Check |
|---|---|
| $\partial g/\partial R$, $\partial W/\partial\mathrm{CN}$, quotient-rule $\partial C_6/\partial\mathrm{CN}$, $\partial E_{AB}/\partial R$, $\partial E_{AB}/\partial C_6$, $R_0 = \sqrt{3q}$ | sympy, symbolic |
| assembled gradient (4 atoms, two elements with 3 and 2 references, $k=16$, $w_f=4$) = sympy $dE/d\mathbf{R}$ = 60-digit central FD | sympy script, claim 5 |
| one reference per element ⇒ chain term exactly $0$ | sympy script, claim 6 |
| `d3bj_gradient` = central FD of `d3bj_energy` ($<10^{-7}$ Ha/Bohr) on Ar$_2$, water, a C/H cluster and a C/F/O/N/H fragment | `tests/gradient_vs_fd.rs::analytic_gradient_matches_finite_difference` |
| dropping the chain term, or only its $\mathrm{CN}_B$ half, or halving $\partial W/\partial\mathrm{CN}$ fails that test (water: $1.3\times10^{-6}$, $5.3\times10^{-7}$, $4.0\times10^{-7}$ Ha/Bohr) | mutation |
| energy vs simple-dftd3 1.6.0 ($<10^{-12}$ Ha, ATM off) | `tests/vs_reference_dftd3.rs` |
