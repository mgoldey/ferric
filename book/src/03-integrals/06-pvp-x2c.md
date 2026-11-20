# The pVp integral and spin-free X2C-1e {#pvp-x2c}

<!-- status: edit; registry: pvp-integrals, x2c1e -->

Electrons near a heavy nucleus move fast. In a heavy atom the inner
electrons reach a sizeable fraction of the speed of light, and a
nonrelativistic Hamiltonian describes them poorly. Part of the relativistic
effect is scalar (spin-free), and that part can enter a nonrelativistic code
through the core Hamiltonian alone.

The spin-free exact two-component Hamiltonian (X2C-1e) does that. It solves
the one-electron Dirac equation in the basis, separates the electronic
solutions from the positronic ones, and returns a matrix that replaces
$h = T + V$. The Dirac equation needs one integral block beyond $S$, $T$ and
$V$: the pVp matrix $W$. This section derives $W$, shows how the code obtains
it from libint2's nuclear second derivatives, and states the X2C-1e
transformation that consumes it.

| Entry point | Location |
|---|---|
| Rust, $W$ | `ferric-integrals/src/pvp.rs::{pvp,pvp_nuclear}` |
| Rust, X2C-1e | `ferric-scf/src/x2c.rs::{x2c1e_transform,x2c1e_hcore,reference_hcore}` |
| CLI | `[scf] relativistic = "x2c1e"` |
| Python | `AoIntegrals.pvp()`, `AoIntegrals.hcore_x2c(c=None)`, `relativistic=` on the SCF drivers |

## The pVp integral {#pvp-x2c-pvp-integral}

### What $W$ is

In the Dirac equation the small component of an electronic state is, to
leading order, $(\boldsymbol\sigma\cdot\hat{\mathbf p}/2c)$ times the large
component. The nuclear potential acting on the small component therefore
appears as $\langle\boldsymbol\sigma\cdot\hat{\mathbf p}\,\chi_\mu|V|\boldsymbol\sigma\cdot\hat{\mathbf p}\,\chi_\nu\rangle/(4c^2)$.
Its spin-free part is the block

$$ W_{\mu\nu} = \sum_{a \in \{x,y,z\}} \langle \partial_a \chi_\mu | V | \partial_a \chi_\nu \rangle = \langle \chi_\mu | \hat{\mathbf p}\cdot V \hat{\mathbf p} | \chi_\nu\rangle, \qquad V(\mathbf r) = -\sum_C \frac{Z_C}{|\mathbf r - \mathbf C|} , \label{eq:pvp-x2c-w} $$

(PySCF's `int1e_pnucp`), with $Z_C$ the effective charge $Z^{\mathrm{eff}}_C$ of
(@eq:molecule-1), where $\hat{\mathbf p} = -i\nabla$ and the two factors
of $-i$ and $+i$ from the bra and ket cancel. $W$ is the potential energy
weighted by the gradient of the basis functions. It is large where the
functions are steep and the potential deep, which is near the nucleus, where
the relativistic correction lives.

### $W$ from second derivatives of $V$

libint2 has no pVp operator, but it has second derivatives of the nuclear
attraction integral. With $\chi_\mu$ on centre $\mathbf A$ and $\chi_\nu$ on
centre $\mathbf B$, section @sec:integrals-derivatives-basis-function gives
$\partial\chi_\mu/\partial A_a = -\partial_a\chi_\mu$. Differentiate
$V_{\mu\nu} = \langle\chi_\mu|V|\chi_\nu\rangle$ once with respect to
$A_a$ and once with respect to $B_a$. Each derivative lands on one function
and brings one minus sign, and the two signs cancel:

$$ \frac{\partial^2 V_{\mu\nu}}{\partial A_a\, \partial B_a} = \langle \partial_a\chi_\mu | V | \partial_a\chi_\nu \rangle , \qquad W_{\mu\nu} = \sum_a \frac{\partial^2 V_{\mu\nu}}{\partial A_a\, \partial B_a} . $$

The derivative with respect to the charge positions does not enter. Libint2
reports the bra and ket centres separately even when both functions sit on
the same atom, so the identity holds for same-centre pairs as well. In the
second derivative engine's coordinate list (bra $0..2$, ket $3..5$, then the
charges), $W$ is the sum of the three blocks $(a, 3 + a)$. Here is the inner
part of `ferric-integrals/src/pvp.rs::pvp`, with comments added:

```rust
for chunk in charges.chunks(CHARGE_CHUNK) {               // W is linear in the charges
    eng.set_point_charges_only(chunk)?;
    let ncoord = 3 * (2 + chunk.len());                   // bra, ket, each charge
    let blocks = [0usize, 1, 2].map(|a| deriv2_pair_index(ncoord, a, 3 + a));
    for s1 in 0..nsh {                                    //   (d/dA_a, d/dB_a), a = x, y, z
        for s2 in 0..=s1 {
            // ...
            let d2 = eng.compute_1e_deriv2_block(prep, s1, s2, chunk.len());
            for i in 0..n1 {
                for j in 0..n2 {
                    let k = i * n2 + j;
                    let v: f64 = blocks.iter().map(|&blk| d2[blk * nn + k]).sum();  // sum_a
                    out[(offs[s1] + i, offs[s2] + j)] += v;
                    if s1 != s2 {
                        out[(offs[s2] + j, offs[s1] + i)] += v;   // W symmetric
                    }
                }
            }
        }
    }
}
```

Two points about it:

- **The charges come in chunks.** The result is linear in the charges, so they
  are passed to the engine in chunks of `CHARGE_CHUNK` = 8 and the chunk
  results added. A second-derivative engine with $n_c$ charges has
  $3(2 + n_c)$ coordinates and returns a block for every pair of them,
  $O(n_c^2)$ blocks. Chunking keeps that at $O(8^2)$ instead of
  $O(n_{\mathrm{atoms}}^2)$.
- **Only three of the blocks are used.** The pair indices are those of
  section @sec:integrals-derivatives-second: for $n_c = 1$, `ncoord` is 9 and the blocks $(0,3)$, $(1,4)$,
  $(2,5)$ sit at indices 3, 12 and 20 of 45. Everything else libint2
  computes, including all charge derivatives, is discarded.

`ferric-integrals/src/pvp.rs::pvp_nuclear` uses the same effective charges as
`nuclear`, $Z^{\mathrm{eff}}_C$ of (@eq:molecule-1).

### A closed form to test against

A test needs a case with a closed form, and a wrong construction that the
case rejects. Two primitive $s$ functions on one centre give both. Take
$\chi_a = N_a e^{-ar^2}$ and $\chi_b = N_b e^{-br^2}$, with $p = a + b$. The gradients are
$\nabla\chi_a = -2a\mathbf r\,\chi_a$ and likewise for $b$, so

$$ \nabla\chi_a\cdot\nabla\chi_b = 4ab\,r^2 e^{-pr^2}N_aN_b . $$

Write $r^2e^{-pr^2} = -\partial_p e^{-pr^2}$. The potential of a spherical
Gaussian at distance $D$ is
$\int e^{-pr^2}/|\mathbf r - \mathbf C|\,d\mathbf r = (\pi/p)^{3/2}\operatorname{erf}(\sqrt p D)/D$,
and differentiating it with respect to $p$ gives $W$:

$$ W = -4ab\,Z\,N_aN_b\, I_2, \qquad I_2 = \frac{3\pi^{3/2}\operatorname{erf}(\sqrt p D)}{2Dp^{5/2}} - \frac{\pi e^{-pD^2}}{p^2}, \qquad p = a + b , $$

for a charge $Z$ at distance $D$. Check the limit $D \to 0$. With
$\operatorname{erf}(x) \approx 2x/\sqrt\pi$, the potential becomes
$(\pi/p)^{3/2}\cdot 2\sqrt{p/\pi} = 2\pi/p$, and $-\partial_p(2\pi/p) = 2\pi/p^2$.
So $I_2 \to 2\pi/p^2$ at $D = 0$, the value for a charge on the centre.
`scripts/pvp_ss_sympy.py` derives it, checks it against a radial quadrature
of the definition and checks the $p$-shell trace;
`crates/ferric-integrals/tests/pvp.rs` checks the code against these closed
forms.

The wrong construction is the bra-bra form
$\langle\nabla^2\chi_\mu|V|\chi_\nu\rangle$, which looks like a
shortcut to $W$ because $\hat p^2 = -\nabla^2$. Integrate by parts once,
moving one derivative off $\nabla^2\chi_\mu$ onto the product $V\chi_\nu$:

$$ -\langle\nabla^2\chi_\mu|V|\chi_\nu\rangle = W_{\mu\nu} + \langle\nabla\chi_\mu\cdot\nabla V\,|\,\chi_\nu\rangle . $$

So the bra-bra block is close to $-W$, not $W$, and the gradient in the
remainder sits on $\chi_\mu$, the function that carried the Laplacian. The
remainder does not vanish for any charge position. For the two $s$
primitives, use $\nabla\chi_a = -2a\mathbf r\,\chi_a$ again; the average of
$\mathbf r\cdot\nabla V$ over a sphere of radius $r$ about the centre is
$Z/r$ for $r > D$ and $0$ inside, so

$$ \langle\nabla\chi_a\cdot\nabla V\,|\,\chi_b\rangle = -2aZN_aN_b\,4\pi\int_D^\infty r\,e^{-pr^2}\,dr = -\frac{4\pi ZaN_aN_b}{p}\,e^{-pD^2} . $$

With the charge on the centre it is $-4\pi ZaN_aN_b/p$, not zero. So the
code has to take the mixed derivative. The script checks the
integration-by-parts identity against a quadrature of the bra-bra block at
$D = 0$, $0.9$ and $1.7$ Bohr, and keeps the bra-bra block as a negative
control.

> **Verified equation {#eq:integrals-pvp-ss-closed-form}** · the same-centre $s$ closed form of $W$
>
> For primitive $s$ functions on one centre and a charge $Z$ at distance $D$,
> $W = -4abZN_aN_bI_2$ with $I_2$ as above, including the limit
> $I_2 \to 2\pi/p^2$ at $D = 0$.
>
> | | |
> |---|---|
> | Proof | `scripts/pvp_ss_sympy.py`, claim 1 (against a 30-digit radial quadrature of the definition) |
> | Negative control | dropping one axis, the opposite sign, and the bra-bra Laplacian block must not reproduce the quadrature |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-integrals/src/pvp.rs::pvp` |
> | Code test | `crates/ferric-integrals/tests/pvp.rs` (closed forms, mixed finite difference of $V$, symmetry, linearity across a charge chunk) |
>
> <!-- verified: id=integrals.pvp.ss-closed-form; script=scripts/pvp_ss_sympy.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-integrals/src/pvp.rs::pvp; rust_test=ferric-integrals/tests/pvp.rs::same_centre_s_and_p_match_the_closed_forms; registry=pvp-integrals -->

> **Verified equation {#eq:integrals-pvp-p-trace}** · the $p$-shell trace of $W$
>
> For $p$ shells on one centre with the charge on the centre,
> $\operatorname{tr} W = -ZN_aN_b\,4\pi\,[3/(2p) - 1/p + 4ab/p^3]$, the same in
> a Cartesian and a pure shell.
>
> | | |
> |---|---|
> | Proof | `scripts/pvp_ss_sympy.py`, claim 2 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-integrals/src/pvp.rs::pvp` |
>
> <!-- verified: id=integrals.pvp.p-trace; script=scripts/pvp_ss_sympy.py; claim=2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-integrals/src/pvp.rs::pvp; rust_test=ferric-integrals/tests/pvp.rs::same_centre_s_and_p_match_the_closed_forms; registry=pvp-integrals -->

The mixed-derivative identity itself, $W = \sum_a \partial^2V/\partial A_a\partial B_a$,
is checked numerically in `crates/ferric-integrals/tests/pvp.rs`
(`pvp_matches_mixed_fd_of_nuclear_attraction`) against a mixed central
difference of $V$, with a bar derived from the step and the rounding of a
second difference.

## Spin-free X2C-1e {#pvp-x2c-spin-free-x2c-1e}

### The modified Dirac equation {#pvp-x2c-modified-dirac}

Measure energies from the rest mass, $E \to E - c^2$. The one-electron Dirac
equation in the potential $V$ couples a large component $\psi_L$ and a small
component $\psi_S$, each a two-spinor:

$$ V\psi_L + c\,(\boldsymbol\sigma\cdot\hat{\mathbf p})\,\psi_S = E\psi_L, \qquad c\,(\boldsymbol\sigma\cdot\hat{\mathbf p})\,\psi_L + (V - 2c^2)\,\psi_S = E\psi_S . $$

Check the free particle at low energy. With $V = 0$ and $|E| \ll 2c^2$ the
second equation gives $\psi_S \approx (\boldsymbol\sigma\cdot\hat{\mathbf p}/2c)\,\psi_L$.
Put that into the first and use $(\boldsymbol\sigma\cdot\hat{\mathbf p})^2 = \hat p^2$:
$\hat p^2\psi_L/2 = E\psi_L$, the Schrödinger equation of a free electron.

The limit suggests a change of variable. Write the small component through
a pseudo-large function $\phi$,

$$ \psi_S = \frac{\boldsymbol\sigma\cdot\hat{\mathbf p}}{2c}\,\phi . $$

This is a substitution, not an approximation. Put it into the first
equation. $c\,(\boldsymbol\sigma\cdot\hat{\mathbf p})\psi_S = \hat p^2\phi/2 = \hat T\phi$, so

$$ V\psi_L + \hat T\phi = E\psi_L . $$

Put it into the second equation and apply
$\boldsymbol\sigma\cdot\hat{\mathbf p}/2c$ from the left. Term by term:
$c\,(\boldsymbol\sigma\cdot\hat{\mathbf p})\psi_L$ becomes $\hat T\psi_L$,
$V\psi_S$ becomes
$(\boldsymbol\sigma\cdot\hat{\mathbf p})V(\boldsymbol\sigma\cdot\hat{\mathbf p})\phi/4c^2$,
$-2c^2\psi_S$ becomes $-\hat T\phi$, and $E\psi_S$ becomes
$E\,\hat p^2\phi/4c^2 = E\,\hat T\phi/2c^2$:

$$ \hat T\psi_L + \Big[\frac{(\boldsymbol\sigma\cdot\hat{\mathbf p})V(\boldsymbol\sigma\cdot\hat{\mathbf p})}{4c^2} - \hat T\Big]\phi = \frac{E}{2c^2}\,\hat T\phi . $$

The only operator here that a nonrelativistic code lacks is
$(\boldsymbol\sigma\cdot\hat{\mathbf p})V(\boldsymbol\sigma\cdot\hat{\mathbf p})$.
Expand both dot products and use $\sigma_a\sigma_b = \delta_{ab} + i\epsilon_{abc}\sigma_c$:

$$ (\boldsymbol\sigma\cdot\hat{\mathbf p})\,V\,(\boldsymbol\sigma\cdot\hat{\mathbf p}) = \hat{\mathbf p}\cdot V\hat{\mathbf p} + i\boldsymbol\sigma\cdot\big[(\hat{\mathbf p}V)\times\hat{\mathbf p}\big], $$

where $\hat{\mathbf p}V = -i\nabla V$ acts on $V$ alone. In the $\epsilon$
term the part with both momenta acting on the function cancels, because
$\hat p_a\hat p_b$ is symmetric. The second term is the spin-orbit
coupling. Drop it, and no Pauli matrix is left: both spinor components obey
the same scalar equation. That is what spin-free means.

Now expand $\psi_L = \sum_\mu\chi_\mu (C_L)_\mu$ and
$\phi = \sum_\mu\chi_\mu (C_S)_\mu$ in the same real basis and project both
equations onto $\chi_\mu$. The kinetic and potential terms give $T$ and $V$,
$E$ gives $S$, and $\hat{\mathbf p}\cdot V\hat{\mathbf p}$ gives $W$ of
(@eq:pvp-x2c-w). The result is a real generalized eigenproblem of dimension
$2n$:

$$ \begin{pmatrix} V & T \\ T & \dfrac{W}{4c^2} - T \end{pmatrix} \begin{pmatrix} C_L \\ C_S \end{pmatrix} = \begin{pmatrix} S & 0 \\ 0 & \dfrac{T}{2c^2} \end{pmatrix} \begin{pmatrix} C_L \\ C_S \end{pmatrix} E . \label{eq:pvp-x2c-modified-dirac} $$

Its $n$ highest solutions are the electronic ones; the $n$ lowest are the
positronic states, below $-2c^2$.

Check the limit $c \to \infty$ on one basis function, where $S$, $T$, $V$
and $W$ are numbers $s$, $t$, $v$, $w$. Expand the electronic root of the
$2 \times 2$ problem in $c^{-2}$:

$$ E = E_0 + \frac{1}{c^2}\,\frac{w/4 - E_0\,t/2}{s} + O(c^{-4}), \qquad E_0 = \frac{t + v}{s} . $$

At $c^{-2} = 0$ it is the nonrelativistic energy of the one function, and the
first correction is of order $c^{-2}$.

> **Verified equation {#eq:integrals-x2c-modified-dirac}** · the spin-free modified Dirac equation
>
> With $\psi_S = (\boldsymbol\sigma\cdot\hat{\mathbf p}/2c)\phi$, the two
> Dirac equations become $V\psi_L + \hat T\phi = E\psi_L$ and
> $\hat T\psi_L + [(\boldsymbol\sigma\cdot\hat{\mathbf p})V(\boldsymbol\sigma\cdot\hat{\mathbf p})/4c^2 - \hat T]\phi = E\hat T\phi/2c^2$,
> and $(\boldsymbol\sigma\cdot\hat{\mathbf p})V(\boldsymbol\sigma\cdot\hat{\mathbf p}) = \hat{\mathbf p}\cdot V\hat{\mathbf p} + i\boldsymbol\sigma\cdot[(\hat{\mathbf p}V)\times\hat{\mathbf p}]$.
> Their projection is (@eq:pvp-x2c-modified-dirac).
>
> | | |
> |---|---|
> | Proof | `scripts/x2c_sympy.py`, claims 1 and 2 (operators on a two-spinor of arbitrary functions); claim 4 (the $c^{-2}$ expansion on one function) |
> | Negative control | the spin term without its $i$; the metric $T/4c^2$ in place of $T/2c^2$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/x2c.rs::x2c1e_transform` |
> | Code test | `crates/ferric-scf/tests/x2c.rs` ($c^{-2}$ fall-off of the correction) |
>
> <!-- verified: id=integrals.x2c.modified-dirac; script=scripts/x2c_sympy.py; claim=2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/x2c.rs::x2c1e_transform; rust_test=ferric-scf/tests/x2c.rs::x2c_correction_falls_as_c_minus_two; registry=x2c1e -->

### Decoupling and renormalization {#pvp-x2c-decoupling}

For the electronic solutions write $C_S = XC_L$, with
$X = C_SC_L^{-1}$ and $E$ now the diagonal matrix of the $n$ electronic
eigenvalues. The two block rows of (@eq:pvp-x2c-modified-dirac) read

$$ (V + TX)\,C_L = S\,C_L E, \qquad \Big(T + \Big(\frac{W}{4c^2} - T\Big)X\Big)C_L = \frac{TX}{2c^2}\,C_L E . $$

Multiply the second row by $X^{\mathsf T}$ from the left and add the first:

$$ \Big(V + TX + X^{\mathsf T}T - X^{\mathsf T}TX + \frac{X^{\mathsf T}WX}{4c^2}\Big)C_L = \Big(S + \frac{X^{\mathsf T}TX}{2c^2}\Big)C_L E . $$

Call the bracket on the left $L$ and the one on the right $\tilde S$. Both
are symmetric, and the $n$ electronic eigenvalues are exactly the
eigenvalues of $L$ in the metric $\tilde S$. $\tilde S$ is not $S$: the norm
of an electronic state includes its small component. The renormalization

$$ R = S^{-1/2}\big(S^{-1/2}\tilde S S^{-1/2}\big)^{-1/2}S^{1/2} $$

gives the two-component core Hamiltonian

$$ h^{\mathrm{X2C}} = R^{\mathsf T}\Big(V + TX + X^{\mathsf T}T - X^{\mathsf T}TX + \frac{X^{\mathsf T}WX}{4c^2}\Big)R . \label{eq:pvp-x2c-hcore} $$

What each piece does:

- $X$ expresses the small-component coefficients of the electronic states
  through the large-component ones, so the problem can be written in the
  large-component space alone.
- The bracket $L$ is the electronic block of the Dirac Hamiltonian in that
  space, and $\tilde S$ is its metric.
- $R$ maps back to the original metric. Write
  $M = S^{-1/2}\tilde SS^{-1/2}$, so $R = S^{-1/2}M^{-1/2}S^{1/2}$ and
  $R^{\mathsf T} = S^{1/2}M^{-1/2}S^{-1/2}$. Then
  $R^{\mathsf T}\tilde SR = S^{1/2}M^{-1/2}MM^{-1/2}S^{1/2} = S$.

What does the transformation guarantee? Take an eigenvector of
$h^{\mathrm{X2C}}$ in the metric $S$, $h^{\mathrm{X2C}}u = Su\,\varepsilon$.
Replace $S$ by $R^{\mathsf T}\tilde SR$ and cancel the invertible
$R^{\mathsf T}$: $L(Ru) = \tilde S(Ru)\,\varepsilon$. So the eigenvalues of
$h^{\mathrm{X2C}}$ in the metric $S$ are exactly the $n$ electronic
eigenvalues of the $2n$ problem (@eq:pvp-x2c-modified-dirac), with no
expansion in $c^{-2}$, and $h^{\mathrm{X2C}}$ is an ordinary core
Hamiltonian that the SCF uses in place of $T + V$ with nothing else changed.
Any $R$ with $R^{\mathsf T}\tilde SR = S$ keeps this spectrum; the form above
is the one of Liu and Peng, which PySCF also uses.

Check the nonrelativistic limit $c \to \infty$. The second block row becomes
$T(1 - X)C_L = 0$, so $X \to 1$ (the small component equals the pseudo-large
one), $\tilde S \to S$, $R \to 1$, and the bracket becomes
$V + T + T - T + 0 = T + V$. The correction falls off as $c^{-2}$, which the
tests check.

> **Verified equation {#eq:integrals-x2c-decoupling}** · the decoupled electronic block
>
> With $C_S = XC_L$, $X^{\mathsf T}$ times the second block row of
> (@eq:pvp-x2c-modified-dirac) plus the first is $LC_L - \tilde SC_LE$, with
> $L$ the bracket of (@eq:pvp-x2c-hcore) and
> $\tilde S = S + X^{\mathsf T}TX/2c^2$.
>
> | | |
> |---|---|
> | Proof | `scripts/x2c_sympy.py`, claim 3 (noncommuting matrix symbols) |
> | Negative control | $X^{\mathsf T}TX$ entering $L$ with $+$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/x2c.rs::x2c1e_transform` |
>
> <!-- verified: id=integrals.x2c.decoupling; script=scripts/x2c_sympy.py; claim=3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/x2c.rs::x2c1e_transform; registry=x2c1e -->

> **Verified equation {#eq:integrals-x2c-spectrum}** · the spectrum of $h^{\mathrm{X2C}}$
>
> $R^{\mathsf T}\tilde SR = S$, and the eigenvalues of $h^{\mathrm{X2C}}$ of
> (@eq:pvp-x2c-hcore) in the metric $S$ are the $n$ highest eigenvalues of
> (@eq:pvp-x2c-modified-dirac).
>
> | | |
> |---|---|
> | Proof | `scripts/x2c_sympy.py`, claim 5: on a random model ($n = 7$, $c = 137.036$) $h^{\mathrm{X2C}}$ built step by step from the equations above is compared with a direct diagonalization of the $2n$ problem (relative $1.5\times10^{-12}$); $\vert R^{\mathsf T}\tilde SR - S\vert \le 4.4\times10^{-15}$; $\Vert h^{\mathrm{X2C}}(c) - (T+V)\Vert$ falls by 98.2, 99.98 and 100.00 per decade of $c$ |
> | Negative control | $W/2c^2$ in place of $W/4c^2$ in the bracket; $R$ without the inner inverse square root |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/x2c.rs::x2c1e_transform` |
> | Code test | `crates/ferric-scf/tests/x2c.rs` (eigenvalues of the contracted $h^{\mathrm{X2C}}$ against PySCF `sfx2c1e`) |
>
> <!-- verified: id=integrals.x2c.spectrum; script=scripts/x2c_sympy.py; claim=5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/x2c.rs::x2c1e_transform; rust_test=ferric-scf/tests/x2c.rs::hcore_eigenvalues_match_pyscf_sfx2c1e; registry=x2c1e -->

### The code

Here is the core of `ferric-scf/src/x2c.rs::x2c1e_transform`, with the input
checks removed and the solve for $X$ shortened:

```rust
h4.slice_mut(s![..n, ..n]).assign(v);                       // [ V  T          ]
h4.slice_mut(s![..n, n..]).assign(t);
h4.slice_mut(s![n.., ..n]).assign(t);
h4.slice_mut(s![n.., n..]).assign(&(w / (4.0 * c2) - t));  // [ T  W/4c^2 - T ]
m4.slice_mut(s![..n, ..n]).assign(s);                       // [ S  0          ]
m4.slice_mut(s![n.., n..]).assign(&(t / (2.0 * c2)));       // [ 0  T/2c^2     ]
let (_, (vec, _)) = (h4, m4)
    .eigh(UPLO::Lower)
    .map_err(|e| lapack("Dirac eigenproblem", e))?;
// Electronic solutions: the n highest eigenvalues (ascending order).
let cl = vec.slice(s![..n, n..]).to_owned();                // C_L
let cs = vec.slice(s![n.., n..]).to_owned();                // C_S
// ... X = C_S C_L^-1 by an LU solve of C_L^T, row by row
let tx = t.dot(&x);
let xt_tx = x.t().dot(&tx);
let s_tilde = s + &(&xt_tx / (2.0 * c2));                   // S~
let lag = v + &tx + tx.t() - &xt_tx + &(x.t().dot(w).dot(&x) / (4.0 * c2));
let s_half = sym_pow(s, 0.5, "S^1/2")?;
let s_mhalf = sym_pow(s, -0.5, "S^-1/2")?;
let inner = sym_pow(&s_mhalf.dot(&s_tilde).dot(&s_mhalf), -0.5, "R")?;
let r = s_mhalf.dot(&inner).dot(&s_half);                   // R
let h = r.t().dot(&lag).dot(&r);                            // R^T (..) R
Ok(0.5 * (&h + &h.t()))
```

Three points about it:

- **No matrix is inverted.** $X$ comes from an LU solve with $C_L^{\mathsf T}$,
  and the powers of $S$ from an eigendecomposition (`sym_pow`), which also
  refuses a matrix that is not positive definite.
- **`tx.t()` stands for $X^{\mathsf T}T$.** $T$ is symmetric, so
  $(TX)^{\mathsf T} = X^{\mathsf T}T$, and one product serves both terms.
- **The result is symmetrized.** In exact arithmetic $h^{\mathrm{X2C}}$ is
  symmetric; the last line removes the rounding asymmetry.

| Step | What | Function |
|---|---|---|
| 1 | uncontract the basis: one shell per unique exponent and $l$ per element | `ferric-scf/src/x2c.rs::uncontracted_basis` |
| 2 | $S$, $T$, $V$, $W$ in the uncontracted basis | `ferric-integrals/src/oneelectron.rs::{overlap,kinetic,nuclear}`, `ferric-integrals/src/pvp.rs::pvp_nuclear` |
| 3 | solve the eigenproblem, form $X$, $R$, $h^{\mathrm{X2C}}$ | `ferric-scf/src/x2c.rs::x2c1e_transform` |
| 4 | contract back: $h = C^{\mathsf T}h^{\mathrm{X2C}}C$ | `ferric-scf/src/x2c.rs::contraction_matrix` |

Why uncontract? The decoupling needs a basis flexible enough to represent
the small component, so $X$, the bracket and $R$ are built in the uncontracted
basis. The contracted functions are exact linear combinations of the
uncontracted ones, so step 4 is a change of basis, not an approximation.

$c = 137.03599967994$ (`ferric-scf/src/x2c.rs::LIGHT_SPEED`, PySCF's value).
These are the settings of PySCF `scf.RHF(mol).sfx2c1e()` with `approx = '1e'`
and `xuncontract = True`. Only the core Hamiltonian is transformed; the
two-electron interaction stays nonrelativistic.

| Restriction | Reason |
|---|---|
| ECP atoms are refused | the ECP already carries the scalar relativistic effect of its core |
| energies only | gradients, Hessians, Z-vector response and post-SCF paths that rebuild $h$ refuse a relativistic reference (`ferric-scf/src/x2c.rs::refuse_relativistic`) |

`scripts/x2c_sympy.py` proves the steps from the Dirac equation to
(@eq:pvp-x2c-hcore) and the spectrum of $h^{\mathrm{X2C}}$. It does not cover
the uncontraction and the contraction back (steps 1 and 4), which are a
change of basis; `crates/ferric-scf/tests/x2c.rs` checks that the contraction
matrix maps the uncontracted $S$, $T$ and $V$ onto the contracted ones. The
numpy reference that compares the whole route with PySCF is
`scripts/prototypes/x2c_proto.py`.

## Numerical anchors {#pvp-x2c-numerical-anchors}

From the X2C-1e row of `site/src/reference/validation.md`:

| What | Reference | Measured |
|---|---|---|
| RHF on HBr and H2Se, cc-pVDZ and uncontracted cc-pVDZ | PySCF `sfx2c1e()` | $6.7\times10^{-11}$ to $1.6\times10^{-10}$ Ha |
| the SCF option against the block route (constant `fock_hook`), RHF, UHF, ROHF, UKS/PBE on HBr and HBr$^+$ | ferric itself | $4.6\times10^{-13}$ to $9.1\times10^{-13}$ Ha |
| RI-MP2 on the X2C reference, HBr / cc-pVDZ | PySCF DF-MP2 | $1.3\times10^{-11}$ Ha |

Tests: `crates/ferric-scf/tests/x2c.rs` (the Hamiltonian, including the
$c^{-2}$ fall-off of the correction), `crates/ferric-scf/tests/x2c_scf.rs`
(the SCF option).

## Sources

- K. G. Dyall, *J. Chem. Phys.* **106**, 9618 (1997).
- W. Kutzelnigg and W. Liu, *J. Chem. Phys.* **123**, 241102 (2005).
- W. Liu and D. Peng, *J. Chem. Phys.* **131**, 031104 (2009).
