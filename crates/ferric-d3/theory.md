# D3(BJ) Dispersion: Energy, Analytic Gradient and Hessian

Grimme D3 with Becke-Johnson (rational) damping and the optional
Axilrod-Teller-Muto (ATM) three-body term, as implemented in `src/lib.rs`.
Zero damping of the two-body term and periodic boundary conditions are not
implemented. Coordinates are in Bohr, energies in Hartree.

The two-body identities below are proved by
`scripts/d3_cn_chain_rule_gradient_sympy.py`, the three-body ones by
`scripts/d3_atm_gradient_sympy.py`, the Hessian by
`scripts/d3_hessian_sympy.py` (all exit nonzero on mismatch, with
negative controls), and the full gradient is checked against finite
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
functional in `params.rs`, equal to simple-dftd3 1.6.0's rational-damping
table: BLYP, BP86, B97-D, PBE, PBE0, revPBE, B3LYP, TPSS, TPSS0, B2PLYP,
BHLYP, SCAN, r2SCAN, ωB97X. M06-2X has a zero-damping D3(0) fit only and is
refused. The ωB97X row is ωB97X-D3(BJ) of Najibi and Goerigk, JCTC 14, 5725
(2018), doi 10.1021/acs.jctc.8b00842 (D3(BJ) in place of ωB97X-V's VV10), not
the zero-damping ωB97X-D3 of Lin, Li, Mao and Chai, JCTC 9, 263 (2013). Fewer than two atoms gives exactly $0$.

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

## Hessian

`d3bj_hessian` is the analytic second derivative of the two-body energy. The
three-body term has no Hessian; $s_9 \neq 0$ is refused.

Write the energy as a sum over pairs $P = (A, B)$ of $-C_6^P\,G_P(R_P)$, with

$$G(R) = \frac{s_6}{R^6 + f^6} + \frac{3\,q\,s_8}{R^8 + f^8},$$

$$G' = -\frac{6 s_6 R^5}{d_6^2} - \frac{8 S_8 R^7}{d_8^2},\qquad
G'' = s_6\left(-\frac{30R^4}{d_6^2} + \frac{72R^{10}}{d_6^3}\right)
    + S_8\left(-\frac{56R^6}{d_8^2} + \frac{128R^{14}}{d_8^3}\right),$$

where $S_8 = 3 q s_8$. The damping radius $f$ does not depend on the geometry.

**Pair block.** For any $\phi(R_{AB})$ with $\mathbf{n} = (\mathbf{R}_A - \mathbf{R}_B)/R_{AB}$,

$$K = \phi''\,\mathbf{n}\mathbf{n}^\mathsf{T} + \frac{\phi'}{R}\left(\mathbf{1} - \mathbf{n}\mathbf{n}^\mathsf{T}\right),$$

and $\partial^2\phi/\partial\mathbf{R}_A\partial\mathbf{R}_A = \partial^2\phi/\partial\mathbf{R}_B\partial\mathbf{R}_B = K$,
$\partial^2\phi/\partial\mathbf{R}_A\partial\mathbf{R}_B = -K$.

**Second derivatives of $C_6$.** With $C_6 = N/D$, $N = \sum_{ab} C_{6,ab}^\mathrm{ref} W_a W_b$
and $D = \sum_{ab} W_a W_b$ (subscripts after a comma are derivatives with
respect to $\mathrm{CN}_A$ and $\mathrm{CN}_B$):

$$C_{6,A} = \frac{N_{,A} - C_6 D_{,A}}{D},\qquad
C_{6,AA} = \frac{N_{,AA} - 2C_{6,A}D_{,A} - C_6 D_{,AA}}{D},$$

$$C_{6,AB} = \frac{N_{,AB} - C_{6,A}D_{,B} - C_{6,B}D_{,A} - C_6 D_{,AB}}{D}.$$

The Gaussian weights give $W' = \alpha W$ and $W'' = (\alpha^2 - 2w_f)W$, with
$\alpha = -2w_f(\mathrm{CN} - \mathrm{CN}^\mathrm{ref})$.

**Counting function.** With $a = -k\,r_c/R^2$, $g' = a\,g(1-g)$ and

$$g'' = -\frac{2}{R}\,g' + a\,(1 - 2g)\,g'.$$

**Assembly.** Let $J_{A} = \partial\mathrm{CN}_A/\partial\mathbf{x}$ (a row of
length $3N$), $e_A = \partial E/\partial\mathrm{CN}_A$ (the gradient's
`de_dcn`), $M_{AB} = \partial^2 E/\partial\mathrm{CN}_A\partial\mathrm{CN}_B$,
and $u_A = \partial e_A/\partial\mathbf{x}$ at fixed CN. Then

$$H = \sum_P K_P\!\left[-C_6^P G_P\right]
    + \sum_A \left(u_A^\mathsf{T} J_A + J_A^\mathsf{T} u_A\right)
    + J^\mathsf{T} M J
    + \sum_{P=(A,B)} (e_A + e_B)\,K_P[g_P].$$

The four terms are the direct pair blocks at fixed $C_6$, the mixed
distance/CN term, the CN-CN term and the curvature of the CNs. Per pair,
$u_A$ gets $-G'_P C_{6,A}^P\,\partial R_P/\partial\mathbf{x}$ and
$M_{AA}$, $M_{AB}$, $M_{BB}$ get $-G_P C_{6,AA}^P$, $-G_P C_{6,AB}^P$, $-G_P C_{6,BB}^P$.
The cost is $O(N^2)$ for the pair loops and $O(N^3)$ for $J^\mathsf{T} M J$.

With one reference per element, $C_6$ is constant, $e_A = 0$, $u_A = 0$ and
$M = 0$, so only the direct pair blocks remain. A dimer of atoms with
$\mathrm{CN} \approx 0$ is close to this limit, which is why the FD test runs
on fragments with mixed coordination.

**Frequencies.** D3(BJ) depends on the geometry only. The frequency driver
adds this Hessian to the analytic KS Hessian wherever the run without
dispersion uses an analytic Hessian
(`harmonic_frequencies_with_geometric_correction`); otherwise it central-differences
the corrected gradient.

## Three-body term (ATM)

Enabled by $s_9 \neq 0$ (`D3Params::s9`, `src/params.rs:39`); named
functionals load $s_9 = 0$ (`src/params.rs:74`), the reference default. With
$s_9 = 0$ the triple sum is skipped (`src/lib.rs:363-367`, `src/lib.rs:501-506`),
so energy and gradient are the two-body ones bit for bit.

For atoms $i>j>k$ let $x = R_{ij}^2$, $y = R_{jk}^2$, $w = R_{ik}^2$ and
$P = R_{ij}R_{jk}R_{ik}$. The angular factor (`atm_triple_factor`,
`src/lib.rs:606`) is

$$a_{ijk} = \frac{3}{8}\,\frac{(x+y-w)(x-y+w)(-x+y+w)}{P^5} + \frac{1}{P^3}
= \frac{3\cos\theta_i\cos\theta_j\cos\theta_k + 1}{P^3},$$

with $\theta$ the interior angles of the triangle. The damping is the zero
damping simple-dftd3 uses for the three-body term of rational damping:

$$f_{ijk} = \frac{1}{1 + 6\,(r_0/P)^{\alpha/3}},\qquad
r_0 = \prod_{(ab)\in\{ij,jk,ik\}} r_s\,R^{\mathrm{vdW}}_{ab},\qquad
r_s = \tfrac43,\quad \alpha = 16,$$

$r_s$ = `ATM_RS9`, $\alpha$ = `ATM_ALPHA` ($=$ alp $+\,2$ with alp $= 14$;
`src/lib.rs:125,128`), $R^{\mathrm{vdW}}_{ab}$ the tabulated pairwise radii
(`vdw_rad`, `src/lib.rs:590`). The BJ $a_1, a_2$ do not enter. The energy
(`atm_energy`, `src/lib.rs:657`) is

$$E_{\mathrm{ATM}} = s_9 \sum_{i>j>k} \sqrt{C_6^{ij}C_6^{ik}C_6^{jk}}\; f_{ijk}\,a_{ijk},$$

with the same CN-interpolated $C_6$ as the two-body term (upstream writes
$C_9 = -s_9\sqrt{\cdots}$ and $E = -C_9 f a$).

**Gradient** (`atm_gradient`, `src/lib.rs:683`). With $F = f\,a$ and
$N = (x+y-w)(x-y+w)(-x+y+w)$, for $q \in \{x,y,w\}$ ($\partial P/\partial q = P/2q$):

$$\frac{\partial a}{\partial q} = \frac{3}{8}\frac{\partial N/\partial q}{P^5}
- \frac{15}{16}\frac{N}{q\,P^5} - \frac{3}{2\,q\,P^3},\qquad
\frac{\partial f}{\partial q} = \frac{\alpha\,(r_0/P)^{\alpha/3} f^2}{q},$$

and $\partial q/\partial \mathbf{R}_a = 2(\mathbf{R}_a - \mathbf{R}_b)$ for the
pair $(a,b)$ that $q$ belongs to (`src/lib.rs:703-709`, `src/lib.rs:722-724`). The $C_6$ dependence is

$$\frac{\partial E_{ijk}}{\partial C_6^{ab}} = \frac{E_{ijk}}{2\,C_6^{ab}},\qquad
\frac{\partial E}{\partial \mathrm{CN}_i} \mathrel{+}= \frac{E_{ijk}}{2C_6^{ij}}\frac{\partial C_6^{ij}}{\partial \mathrm{CN}_i}
+ \frac{E_{ijk}}{2C_6^{ik}}\frac{\partial C_6^{ik}}{\partial \mathrm{CN}_i}$$

and likewise for $j$, $k$ (`src/lib.rs:726-731`). These go into the same
`de_dcn` as the two-body term before the contraction pass, so one
$\sum_{A<B}(\partial E/\partial\mathrm{CN}_A + \partial E/\partial\mathrm{CN}_B)\,\partial g_{AB}/\partial R_{AB}$
carries both chain rules.

The triple sum is $O(N^3)$. Unlike simple-dftd3 there is no real-space
cutoff (upstream skips triples with any distance above 40 Bohr).

## Verification

| Claim | Check |
|---|---|
| $\partial g/\partial R$, $\partial W/\partial\mathrm{CN}$, quotient-rule $\partial C_6/\partial\mathrm{CN}$, $\partial E_{AB}/\partial R$, $\partial E_{AB}/\partial C_6$, $R_0 = \sqrt{3q}$ | sympy, symbolic |
| assembled gradient (4 atoms, two elements with 3 and 2 references, $k=16$, $w_f=4$) = sympy $dE/d\mathbf{R}$ = 60-digit central FD | sympy script, claim 5 |
| one reference per element ⇒ chain term exactly $0$ | sympy script, claim 6 |
| `d3bj_gradient` = central FD of `d3bj_energy` ($<10^{-7}$ Ha/Bohr) on Ar$_2$, water, a C/H cluster and a C/F/O/N/H fragment | `tests/gradient_vs_fd.rs::analytic_gradient_matches_finite_difference` |
| dropping the chain term, or only its $\mathrm{CN}_B$ half, or halving $\partial W/\partial\mathrm{CN}$ fails that test (water: $1.3\times10^{-6}$, $5.3\times10^{-7}$, $4.0\times10^{-7}$ Ha/Bohr) | mutation |
| energy vs simple-dftd3 1.6.0 ($<10^{-12}$ Ha, ATM off) | `tests/vs_reference_dftd3.rs` |
| every tabulated name's $(s_6, s_8, a_1, a_2)$ bit-identical to simple-dftd3 1.6.0, name set equal, M06-2X refused; snapshot from `scripts/gen_d3bj_param_snapshot.py`, whose TOML values are cross-checked against the compiled library | `tests/params_snapshot.rs` |
| $a_{ijk}$ = cosine form; $\partial a/\partial q$, $\partial f/\partial q$, $\partial E/\partial C_6 = E/2C_6$; assembled Cartesian gradient = sympy $dE/d\mathbf{R}$ | `scripts/d3_atm_gradient_sympy.py` |
| energy and gradient with $s_9 = 0$ and $s_9 = 1$ vs simple-dftd3 1.6.0, 9 systems × 3 functionals: totals $\le 1.2\times10^{-13}$ Ha, $5.5\times10^{-13}$ Ha/Bohr; three-body part $2.6\times10^{-15}$ Ha, $5.9\times10^{-16}$ Ha/Bohr | `tests/vs_reference_dftd3_atm.rs` |
| three-body gradient alone = central FD to $<10^{-7}$ relative (measured $\le 4\times10^{-8}$) on Ar$_3$, water, C/H, C/F/O/N/H; full D3(BJ)-ATM to $10^{-7}$ Ha/Bohr | `tests/gradient_vs_fd.rs::atm_gradient_matches_finite_difference` |
| $s_9 = 0$ energy and gradient bit-identical to the two-body code (4 functionals × 4 systems, checked against the pre-ATM build); linear in $s_9$; no effect on dimers | `tests/vs_reference_dftd3_atm.rs::three_body_term_is_linear_in_s9_and_absent_for_dimers` |
| mutants: three-body CN chain dropped, $\partial E/\partial C_6 = E/C_6$, the $15/16$ term dropped, $\alpha/6$ in $\partial f/\partial q$ (each fails the FD test); $\alpha = 14$, $r_s = 1$ (each fails the reference test). Removing the $s_9 \neq 0$ guard is an equivalent mutant (adds $+0.0$) | mutation |
| $g''$, $W''$, the quotient-rule $C_{6,AA}$, $C_{6,AB}$ (abstract $N$, $D$), $G'$, $G''$, the pair block $K$ | `scripts/d3_hessian_sympy.py`, claims 1-5 |
| assembled Hessian (the gradient proof's 4-atom model) = 60-digit FD of the transcribed gradient ($2\times10^{-42}$) and = 60-digit second difference of the energy ($4\times10^{-32}$); each CN part is 2-69% of max$\lvert H\rvert$; one reference per element ⇒ CN parts exactly 0 | sympy script, claims 6-7 |
| `d3bj_hessian` = central FD of `d3bj_gradient` on Ar$_2$, water, a C/H cluster and a C/F/O/N/H fragment, inside twice the truncation estimate from steps $h = 2\times10^{-3}$ and $h/2$ (error ratio 4.00 per halving on every system; water $3.7\times10^{-10}$, mixed $1.6\times10^{-8}$ Ha/Bohr²); symmetric and translation sums $\le 10^{-18}$ | `tests/hessian_vs_fd.rs` |
| `d3bj_hessian` = central FD of the simple-dftd3 1.6.0 gradient (water, C/H cluster, C/F/O/N/H, CHBr$_3$), bar from the same $h$, $h/2$ data | `tests/hessian_vs_dftd3.rs` |
| mutants: the mixed term without its transpose, $72\to36$ in $G''$, the off-diagonal $M_{AB}$ dropped, the $(1-2g)$ factor of $g''$ dropped (sympy); the $e_A\,\partial^2\mathrm{CN}_A$ term dropped, the mixed transpose dropped (Rust FD test) | mutation |
