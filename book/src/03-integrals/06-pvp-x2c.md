# 3.6 The pVp integral and spin-free X2C-1e

<!-- section: 3.6; status: edit; registry: pvp-integrals;
     theory: crates/ferric-integrals/theory.md#"The pVp integral" -->

Scalar relativistic effects enter a nonrelativistic code through the core
Hamiltonian. The spin-free exact two-component Hamiltonian (X2C-1e) needs one
integral block beyond $S$, $T$ and $V$: the pVp matrix $W$. This section
derives $W$, shows how the code obtains it from libint2's nuclear second
derivatives, and states the X2C-1e transformation that consumes it.

| Entry point | Location |
|---|---|
| Rust, $W$ | `ferric-integrals/src/pvp.rs::{pvp,pvp_nuclear}` |
| Rust, X2C-1e | `ferric-scf/src/x2c.rs::{x2c1e_transform,x2c1e_hcore,reference_hcore}` |
| CLI | `[scf] relativistic = "x2c1e"` |
| Python | `AoIntegrals.pvp()`, `AoIntegrals.hcore_x2c(c=None)`, `relativistic=` on the SCF drivers |

## 3.6.1 The pVp integral

<!-- include: crates/ferric-integrals/theory.md#"The pVp integral" -->

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

The one-electron Dirac equation of the nuclear potential, in the modified
(kinetically balanced) form and a basis with $S$, $T$, $V$ and $W$, is the
generalized eigenproblem

$$ \begin{pmatrix} V & T \\ T & \dfrac{W}{4c^2} - T \end{pmatrix} \begin{pmatrix} C_L \\ C_S \end{pmatrix} = \begin{pmatrix} S & 0 \\ 0 & \dfrac{T}{2c^2} \end{pmatrix} \begin{pmatrix} C_L \\ C_S \end{pmatrix} E . $$

Its $n$ highest solutions are the electronic ones. They define the decoupling
matrix and the renormalization

$$ X = C_S C_L^{-1}, \qquad \tilde S = S + \frac{X^{\mathsf T}TX}{2c^2}, \qquad R = S^{-1/2}\big(S^{-1/2}\tilde S S^{-1/2}\big)^{-1/2}S^{1/2}, $$

and the two-component core Hamiltonian

$$ h^{\mathrm{X2C}} = R^{\mathsf T}\Big(V + TX + X^{\mathsf T}T - X^{\mathsf T}TX + \frac{X^{\mathsf T}WX}{4c^2}\Big)R . $$

| Step | What | Function |
|---|---|---|
| 1 | uncontract the basis: one shell per unique exponent and $l$ per element | `ferric-scf/src/x2c.rs::uncontracted_basis` |
| 2 | $S$, $T$, $V$, $W$ in the uncontracted basis | `ferric-integrals/src/oneelectron.rs::{overlap,kinetic,nuclear}`, `ferric-integrals/src/pvp.rs::pvp_nuclear` |
| 3 | solve the eigenproblem, form $X$, $R$, $h^{\mathrm{X2C}}$ | `ferric-scf/src/x2c.rs::x2c1e_transform` |
| 4 | contract back: $h = C^{\mathsf T}h^{\mathrm{X2C}}C$ | `ferric-scf/src/x2c.rs::contraction_matrix` |

$c = 137.03599967994$ (`ferric-scf/src/x2c.rs::LIGHT_SPEED`, PySCF's value).
These are the settings of PySCF `scf.RHF(mol).sfx2c1e()` with `approx = '1e'`
and `xuncontract = True`. Only the core Hamiltonian is transformed; the
two-electron interaction stays nonrelativistic.

| Restriction | Reason |
|---|---|
| ECP atoms are refused | the ECP already carries the scalar relativistic effect of its core |
| energies only | gradients, Hessians, Z-vector response and post-SCF paths that rebuild $h$ refuse a relativistic reference (`ferric-scf/src/x2c.rs::refuse_relativistic`) |

The X2C-1e equations have no derivation in a committed `theory.md` and no
proof script. The numpy reference is `scripts/prototypes/x2c_proto.py`. The
home for the derivation is `crates/ferric-scf/theory.md`, which this lane does
not edit; until it lands, this subsection is the derivation of record and the
section stays at `edit`.

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
