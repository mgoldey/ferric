# The reaction field in the Fock matrix {#reaction-field}

<!-- status: ready; registry: pcm -->

The apparent charges of section @sec:ief-operators act back on the solute as a set of
classical charges outside the molecule. An electron feels them as one more
one-electron potential, and that potential depends on the density through
the charges. So the SCF re-evaluates the tessera potential, re-solves the
charges and rebuilds the potential every iteration, and the outer SCF and
DIIS loop carries the fixed point. An external point charge (section @sec:external-potentials)
is different: its potential is folded into the core Hamiltonian once,
before the loop.

The plan: write the potential at the tesserae and the operator, show that
the operator is the density derivative of the energy (@sec:reaction-field-potential-operator-energy), then the two
probe conventions (@sec:reaction-field-probe-conventions) and the per-iteration algorithm (@sec:reaction-field-algorithm).

| Symbol | Meaning | Code name |
|---|---|---|
| $P$ | total AO density, $D_\alpha + D_\beta$ (for closed shell the doubly occupied density, $2\mathbf C_o\mathbf C_o^{\mathsf T}$) | `density`, `d_total` |
| $v_k$ | solute potential at tessera $k$ | `v` |
| $B^k_{\mu\nu}$ | $\langle\mu\vert -1/\vert\mathbf r-\mathbf r_k\vert\vert\nu\rangle$ | the integrals of the kernels below |
| $V^{\mathrm{pcm}}$ | reaction-field operator, $\sum_k q_kB^k$ | `v_pcm` |
| $\xi_k$ | tessera Gaussian width (smeared probe only) | `charge_exp` |

| Entry point | Location |
|---|---|
| Rust | `ferric-pcm/src/lib.rs::pcm_step`, `ferric-pcm/src/potential.rs::{solute_potential_at_tesserae,build_reaction_field_operator,solute_potential_at_tesserae_smeared,build_reaction_field_operator_smeared}`, `ferric-scf/src/driver.rs::solvent_terms` |
| CLI | `[pcm]` with `kind` = `rhf`, `uhf`, `rohf`, `ksdft` or `pdep-rpa` for `task = "energy"`, and the first four for `task = "optimize"` |
| Python | `solvent=` |

## The potential, the operator and the energy {#reaction-field-potential-operator-energy}

The potential at tessera $k$ has a nuclear and an electronic part. The
nuclei are point charges $Z_A$. The electrons are a charge density
$-\rho(\mathbf r)$ with $\rho = \sum_{\mu\nu}P_{\mu\nu}\chi_\mu\chi_\nu$, so
their potential at $\mathbf r_k$ is
$-\int\rho(\mathbf r)/|\mathbf r-\mathbf r_k|\,d\mathbf r$. Expand $\rho$ in
the basis:

$$ v_k = \sum_A \frac{Z_A}{|\mathbf r_k-\mathbf R_A|} + \sum_{\mu\nu} P_{\mu\nu} B^k_{\mu\nu}, \qquad
   B^k_{\mu\nu} = \left\langle \mu \middle| -\frac{1}{|\mathbf r-\mathbf r_k|} \middle| \nu \right\rangle . $$

$B^k$ is libint's nuclear-attraction integral for a $+1$ charge at
$\mathbf r_k$. The default `ProbeKind::Point` evaluates it
(`ferric-pcm/src/potential.rs::solute_potential_at_tesserae`).

A charge $q_k$ at $\mathbf r_k$ acts on an electron with the potential energy
$-q_k/|\mathbf r-\mathbf r_k|$, whose matrix is $q_kB^k$. Sum over the
tesserae (`ferric-pcm/src/potential.rs::build_reaction_field_operator`):

$$ V^{\mathrm{pcm}}_{\mu\nu} = \sum_k q_k\, B^k_{\mu\nu} . $$

Is this the right Fock contribution? The Fock matrix must be the derivative
of the energy with respect to the density, so check that
$\partial E_{\mathrm{pcm}}/\partial P_{\mu\nu} = V^{\mathrm{pcm}}_{\mu\nu}$.
Write the energy with the symmetrized response of section @sec:ief-operators,
$Q_{\mathrm{sym}} = \tfrac12(Q + Q^{\mathsf T})$:

$$ E_{\mathrm{pcm}}(P) = \tfrac12\, v(P)^{\mathsf T} Q_{\mathrm{sym}}\, v(P) . $$

$v$ is affine in $P$, with $\partial v_k/\partial P_{\mu\nu} = B^k_{\mu\nu}$.
Apply the chain rule. The derivative of $\tfrac12 v^{\mathsf T}Mv$ is
$\tfrac12 v^{\mathsf T}(M + M^{\mathsf T})\,\partial v$, which for the
symmetric $M = Q_{\mathrm{sym}}$ is $(Q_{\mathrm{sym}}v)\cdot\partial v$.
With $q = Q_{\mathrm{sym}}v$,

$$ \frac{\partial E_{\mathrm{pcm}}}{\partial P_{\mu\nu}} = \sum_k q_k B^k_{\mu\nu} = V^{\mathrm{pcm}}_{\mu\nu} . $$

Now try the raw charges $Qv$. The energy is the same function of $P$
(section @sec:ief-operators), so its derivative is still $\sum_k (Q_{\mathrm{sym}}v)_kB^k$.
A Fock matrix built from $\sum_k (Qv)_kB^k$ would not be that derivative,
and the converged SCF density would not be a stationary point of its own
energy. Section @sec:gradient relies on that stationarity, and section @sec:ief-operators
symmetrizes the charges to keep it.

> **Verified equation {#eq:pcm-fock-reaction-field}** · the reaction-field operator is the density derivative of the PCM energy
>
> With $v_k(P) = v^{\mathrm{nuc}}_k + \sum_{\mu\nu}P_{\mu\nu}B^k_{\mu\nu}$ and
> $E_{\mathrm{pcm}}(P) = \tfrac12 v^{\mathsf T}Q_{\mathrm{sym}}v$, the
> derivative is $\partial E_{\mathrm{pcm}}/\partial P_{\mu\nu} =
> \sum_k q_kB^k_{\mu\nu} = V^{\mathrm{pcm}}_{\mu\nu}$, with $q =
> Q_{\mathrm{sym}}v$. The raw $Qv$ would not be this derivative.
>
> | | |
> |---|---|
> | Proof | `scripts/pcm_ief_sympy.py`, claim 6 (symbolic density, exact rationals) |
> | Negative control | $q_{\mathrm{raw}} = Qv$ is not the gradient when $Q$ is not symmetric |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pcm/src/potential.rs::build_reaction_field_operator`, `ferric-pcm/src/lib.rs::pcm_step` |
> | Code test | `scf_water_sto3g_on_pyscf_cavity` in `crates/ferric-scf/tests/validation_pcm.rs` |
>
> <!-- verified: id=pcm.fock.reaction-field; script=scripts/pcm_ief_sympy.py; claim=6; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pcm/src/potential.rs::build_reaction_field_operator; rust_test=ferric-scf/tests/validation_pcm.rs::scf_water_sto3g_on_pyscf_cavity; registry=pcm -->

$E_{\mathrm{pcm}}$ depends on the spins only through $P$, so the same
$V^{\mathrm{pcm}}$ goes into every spin Fock matrix. In vacuum,
$\varepsilon\to 1$, the charges vanish and so does $V^{\mathrm{pcm}}$: the
SCF is the gas-phase one.

One PCM step, as the SCF calls it, is `pcm_step`. It evaluates $v$, solves
for $q$ (section @sec:ief-operators) and builds the operator. The function body,
unchanged:

<!-- excerpt: ferric-pcm/src/lib.rs::pcm_step -->
```rust
let v = ctx
    .probe
    .potential_at_tesserae(mol, prep, density, &ctx.tess)?;
let v_arr = ndarray::Array1::from_vec(v);
let PcmChargeResult { q, e_pcm } = solver::solve_pcm_charges(&ctx.k, &ctx.r, &v_arr)?;
let q = q
    .as_slice()
    .expect("solve_pcm_charges returns a contiguous vector");
let v_pcm = ctx.probe.reaction_field_operator(prep, &ctx.tess, q)?;
Ok((v_pcm, e_pcm))
```

The driver adds the operator to every spin Fock matrix and passes the
energy back as a separate term (`solvent_terms`):

<!-- excerpt: ferric-scf/src/driver.rs::solvent_terms -->
```rust
let (v_pcm, e_pcm) = ferric_pcm::pcm_step(pctx, mol, prep, d_total)?;
for f in focks.iter_mut() {
    **f += &v_pcm;
}
e_pcm
```

The energy of the iteration is the trace plus the standalone terms, and the
PCM energy is one of them:

<!-- excerpt: ferric-scf/src/rhf.rs::RhfState::energy -->
```rust
Ok(e_elec_no_xc + e_xc + e_hook + e_cosmo + e_pcm + e_pol + env.vnn)
```

| Symbol | Code |
|---|---|
| $\tfrac12\operatorname{Tr}[P(h+F)]$, before any solvent or XC term is in $F$ | `e_elec_no_xc` |
| $E_{xc}$ | `e_xc` |
| $E_{\mathrm{pcm}}$ | `e_pcm` |
| $E_{\mathrm{cosmo}}$, $E_{\mathrm{pol}}$ | `e_cosmo`, `e_pol` (sections @sec:conductor-limit-and-cosmo and @sec:polarizable-embedding) |
| $V_{nn}$ plus any external-charge nuclear energy | `env.vnn` |

Why is $E_{\mathrm{pcm}}$ a separate term and not part of the trace
(section @sec:energy-functionals)? `e_elec_no_xc` is formed before `solvent_terms` adds
$V^{\mathrm{pcm}}$ to $F$, so $E_{\mathrm{pcm}}$ is not counted twice. It
also cannot be folded into the trace. Contract the operator with the
density: $\tfrac12\operatorname{Tr}[PV^{\mathrm{pcm}}] = \tfrac12\sum_kq_k\sum_{\mu\nu}P_{\mu\nu}B^k_{\mu\nu} = \tfrac12 q\cdot v^{\mathrm{el}}$.
That lacks the nuclear part $\tfrac12 q\cdot v^{\mathrm{nuc}}$ of
$E_{\mathrm{pcm}}$, because $V^{\mathrm{pcm}}$ is a derivative of the energy
and not an energy density. The standalone term is PySCF's `e_solvent`
convention (`solve_rhf` in `ferric-scf/src/rhf.rs`).

The charge solve itself is direct. The self-consistency between $q$ and $P$
is carried by the outer SCF and DIIS loop.

## The two probe conventions {#reaction-field-probe-conventions}

| `ProbeKind` | $v_k$ electronic kernel | $V^{\mathrm{pcm}}$ kernel | Integrals |
|---|---|---|---|
| `Point` (default) | $\langle\mu\vert 1/\vert\mathbf r-\mathbf r_k\vert\vert\nu\rangle$ | the same, weighted by $q_k$ | the batched McMurchie–Davidson kernel of `ferric-integrals/src/md3c1e.rs`, or libint nuclear attraction for a basis the kernel refuses |
| `GaussianSmeared` | $\langle\mu\vert \operatorname{erf}(\xi_kr)/r\vert\nu\rangle$ | the same, weighted by $q_k$ | the three-index Coulomb integral against a normalized s Gaussian, `ferric-integrals/src/site_basis.rs::SiteBasis` |

`Point` treats each tessera charge as a point in the potential and as a
Gaussian in $S$ and $D$. `GaussianSmeared`
(`solute_potential_at_tesserae_smeared`, `build_reaction_field_operator_smeared`)
uses the Gaussian everywhere: it replaces every $1/r$ by
$\operatorname{erf}(\xi_kr)/r$, which is PySCF's `pcm.py` convention. The
classical nuclear term of $v_k$ takes the same erf form under the smeared
probe. Section @sec:gradient compares against PySCF with Gaussian probes for this
reason.

## The algorithm {#reaction-field-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | $v^{\mathrm{nuc}}_k = \sum_B Z_B/\lvert\mathbf r_k-\mathbf R_B\rvert$ | $N N_{\mathrm{at}}$ | `ferric-pcm/src/potential.rs::nuclear_potential_at` |
| 2 | $v_k = v^{\mathrm{nuc}}_k + \sum_{\mu\nu}P_{\mu\nu}B^k_{\mu\nu}$, batched over tesserae, parallel over batches | $N\,N^2_{\mathrm{bf}}$ | `solute_potential_at_tesserae` |
| 3 | $q$, $E_{\mathrm{pcm}}$ (section @sec:ief-operators) | $N^3$ | `solve_pcm_charges` |
| 4 | $V^{\mathrm{pcm}} = \sum_kq_kB^k$, parallel over shell pairs | $N\,N^2_{\mathrm{bf}}$ | `build_reaction_field_operator` |
| 5 | $V^{\mathrm{pcm}}$ added to every spin Fock matrix; $E_{\mathrm{pcm}}$ to the energy | $N^2_{\mathrm{bf}}$ | `solvent_terms` |

Steps 2 and 4 are bit-identical for every thread-pool size: each output
block is independent of the others. A tessera that coincides with a nucleus
(a degenerate cavity) is an error in step 1.

## Options {#reaction-field-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `PcmConfig::probe` | `Point`, `GaussianSmeared` | `Point` | the probe of the table above; library only |
| `[pcm]` absent | | | a vacuum run, byte-identical to a build with no solvent code (`pcm_none_matches_vacuum_exactly` in `ferric-scf/src/rhf.rs`) |

## Numerical anchors {#reaction-field-numerical-anchors}

| Quantity | System | Error | Source |
|---|---|---|---|
| total SCF energy on PySCF's injected cavity | water, NH3 / STO-3G, cc-pVDZ | $4.8\times10^{-12}$ Ha | `site/src/reference/validation.md`, IEF-PCM row |

## Sources {#reaction-field-sources}

| Source | What it gives |
|---|---|
| `scripts/pcm_ief_sympy.py`, claim 6 | $V^{\mathrm{pcm}} = \partial E_{\mathrm{pcm}}/\partial P$ |
| `crates/ferric-scf/tests/validation_pcm.rs` | the converged SCF against PySCF |

**Known limits (2026-11-15).**

- The CLI refuses `[pcm]` together with `[cosmo]`: they are two models of one
  solvent and would be added.
- The CLI wires `[pcm]` for the energy task on the listed kinds, and for
  `task = "optimize"` on a subset (`PCM_KINDS`, `PCM_OPTIMIZE_KINDS` in
  `crates/ferric-cli/src/config.rs`); frequencies with `[pcm]` are refused.
