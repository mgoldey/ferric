# 3.6 The pVp integral and spin-free X2C-1e

<!-- section: 3.6; status: edit; registry: pvp-integrals;
     theory: crates/ferric-integrals/theory.md#"The pVp integral" -->

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

## 3.6.1 The pVp integral

### What $W$ is

In the Dirac equation the small component of an electronic state is, to
leading order, $(\boldsymbol\sigma\cdot\hat{\mathbf p}/2c)$ times the large
component. The nuclear potential acting on the small component therefore
appears as $\langle\boldsymbol\sigma\cdot\hat{\mathbf p}\,\chi_\mu|V|\boldsymbol\sigma\cdot\hat{\mathbf p}\,\chi_\nu\rangle/(4c^2)$.
Its spin-free part is the block

$$ W_{\mu\nu} = \sum_{a \in \{x,y,z\}} \langle \partial_a \chi_\mu | V | \partial_a \chi_\nu \rangle = \langle \chi_\mu | \hat{\mathbf p}\cdot V \hat{\mathbf p} | \chi_\nu\rangle, \qquad V(\mathbf r) = -\sum_C \frac{q_C}{|\mathbf r - \mathbf C|} , $$

(PySCF's `int1e_pnucp`), where $\hat{\mathbf p} = -i\nabla$ and the two factors
of $-i$ and $+i$ from the bra and ket cancel. $W$ is the potential energy
weighted by the gradient of the basis functions. It is large where the
functions are steep and the potential deep, which is near the nucleus, where
the relativistic correction lives.

### $W$ from second derivatives of $V$

libint2 has no pVp operator, but it has second derivatives of the nuclear
attraction integral. With $\chi_\mu$ on centre $\mathbf A$ and $\chi_\nu$ on
centre $\mathbf B$, section 3.3 gives
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
  section 3.3: for $n_c = 1$, `ncoord` is 9 and the blocks $(0,3)$, $(1,4)$,
  $(2,5)$ sit at indices 3, 12 and 20 of 45. Everything else libint2
  computes, including all charge derivatives, is discarded.

`ferric-integrals/src/pvp.rs::pvp_nuclear` uses the same effective charges as
`nuclear`: zero for ghosts, $Z - n_{\mathrm{core}}$ for ECP atoms.

### A closed form to test against

The bra-bra form $\langle \nabla^2\chi_\mu|V|\chi_\nu\rangle$ is not $W$: it
differs by $\langle\chi_\mu|\nabla V\cdot\nabla\chi_\nu\rangle$, which is
nonzero whenever a charge is away from the centre. So the code has to take
the mixed derivative, and a test needs a case where the two forms disagree.

Take primitive $s$ functions on one centre, $\chi_a = N_a e^{-ar^2}$ and
$\chi_b = N_b e^{-br^2}$, with $p = a + b$. The gradients are
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

> **Verified equation 3.1** · the same-centre $s$ closed form of $W$
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

> **Verified equation 3.2** · the $p$-shell trace of $W$
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

## 3.6.2 Spin-free X2C-1e

### The modified Dirac equation

Write the small component as $(\boldsymbol\sigma\cdot\hat{\mathbf p}/2c)$
times a pseudo-large function, expand both in the same basis, and drop the
spin-dependent part. The one-electron Dirac equation of the nuclear potential
becomes a real generalized eigenproblem of dimension $2n$, with the blocks
$S$, $T$, $V$ and $W$:

$$ \begin{pmatrix} V & T \\ T & \dfrac{W}{4c^2} - T \end{pmatrix} \begin{pmatrix} C_L \\ C_S \end{pmatrix} = \begin{pmatrix} S & 0 \\ 0 & \dfrac{T}{2c^2} \end{pmatrix} \begin{pmatrix} C_L \\ C_S \end{pmatrix} E . $$

Its $n$ highest solutions are the electronic ones; the $n$ lowest are the
positronic states, below $-2c^2$. The electronic solutions define the
decoupling matrix and the renormalization

$$ X = C_S C_L^{-1}, \qquad \tilde S = S + \frac{X^{\mathsf T}TX}{2c^2}, \qquad R = S^{-1/2}\big(S^{-1/2}\tilde S S^{-1/2}\big)^{-1/2}S^{1/2}, $$

and the two-component core Hamiltonian

$$ h^{\mathrm{X2C}} = R^{\mathsf T}\Big(V + TX + X^{\mathsf T}T - X^{\mathsf T}TX + \frac{X^{\mathsf T}WX}{4c^2}\Big)R . $$

What each piece does:

- $X$ expresses the small-component coefficients of the electronic states
  through the large-component ones, so the problem can be written in the
  large-component space alone.
- The bracket is the electronic block of the Dirac Hamiltonian in that space,
  and $\tilde S$ is its metric: the electronic state's norm includes its small
  component.
- $R$ maps back to the original metric. Write
  $M = S^{-1/2}\tilde SS^{-1/2}$, so $R = S^{-1/2}M^{-1/2}S^{1/2}$ and
  $R^{\mathsf T} = S^{1/2}M^{-1/2}S^{-1/2}$. Then
  $R^{\mathsf T}\tilde SR = S^{1/2}M^{-1/2}MM^{-1/2}S^{1/2} = S$. So
  $h^{\mathrm{X2C}}$ is an ordinary core Hamiltonian in the metric $S$, and
  the SCF uses it in place of $T + V$ with nothing else changed.

Check the nonrelativistic limit $c \to \infty$. Then $X \to 1$ (the small
component equals the pseudo-large one), $\tilde S \to S$, $R \to 1$, and the
bracket becomes $V + T + T - T + 0 = T + V$. The correction falls off as
$c^{-2}$, which the tests check.

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

The X2C-1e equations have no derivation in a committed `theory.md` and no
proof script. The numpy reference is `scripts/prototypes/x2c_proto.py`, and
its equations are written out in `scripts/prototypes/x2c_derivation.md`. The
derivation belongs in `crates/ferric-scf/theory.md`. Until it is there, this
subsection is the derivation of record and the section stays at `edit`.

## 3.6.3 Numerical anchors

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
