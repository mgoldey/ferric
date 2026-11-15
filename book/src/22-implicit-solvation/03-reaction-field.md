# 22.3 The reaction field in the Fock matrix

<!-- section: 22.3; status: ready; registry: pcm;
     theory: crates/ferric-pcm/theory.md#"7. Solute potential, Fock contribution, total energy" -->

The apparent charges of section 22.2 act back on the solute as a set of
classical charges outside the molecule. An electron feels them as one more
one-electron potential, and that potential depends on the density through the
charges. The SCF therefore re-evaluates the tessera potential, re-solves the
charges and rebuilds the potential every iteration, and the outer SCF and DIIS
loop carries the fixed point.

This differs from an external point charge (section 23.1), whose potential is
folded into the core Hamiltonian once before the loop.

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

## 22.3.1 The potential, the operator and the energy

<!-- include: crates/ferric-pcm/theory.md#"7. Solute potential, Fock contribution, total energy" -->

One PCM step, as the SCF calls it, is `ferric-pcm/src/lib.rs::pcm_step`:

<!-- excerpt: ferric-pcm/src/lib.rs::pcm_step -->
```rust
let v_pcm = ctx.probe.reaction_field_operator(prep, &ctx.tess, q)?;
Ok((v_pcm, e_pcm))
```

The driver adds the operator to every spin Fock matrix and passes the energy
back as a separate term (`ferric-scf/src/driver.rs::solvent_terms`):

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
| $E_{\mathrm{cosmo}}$, $E_{\mathrm{pol}}$ | `e_cosmo`, `e_pol` (sections 22.5 and 23.3) |
| $V_{nn}$ plus any external-charge nuclear energy | `env.vnn` |

`e_elec_no_xc` is formed before `solvent_terms` adds $V^{\mathrm{pcm}}$ to $F$,
so $E_{\mathrm{pcm}}$ is not counted twice. It cannot be folded into the trace:
$\tfrac12\operatorname{Tr}[PV^{\mathrm{pcm}}] = \tfrac12q\cdot v^{\mathrm{el}}$ lacks
the nuclear part $\tfrac12q\cdot v^{\mathrm{nuc}}$ of $E_{\mathrm{pcm}}$,
because $V^{\mathrm{pcm}}$ is a derivative of the energy and not an
energy density. The standalone term is PySCF's `e_solvent` convention.

> **Verified equation 22.6** · the reaction-field operator is the density derivative of the PCM energy
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

## 22.3.2 The two probe conventions

| `ProbeKind` | $v_k$ electronic kernel | $V^{\mathrm{pcm}}$ kernel | Integrals |
|---|---|---|---|
| `Point` (default) | $\langle\mu\vert 1/\vert\mathbf r-\mathbf r_k\vert\vert\nu\rangle$ | the same, weighted by $q_k$ | the batched McMurchie–Davidson kernel of `ferric-integrals/src/md3c1e.rs`, or libint nuclear attraction for a basis the kernel refuses |
| `GaussianSmeared` | $\langle\mu\vert \operatorname{erf}(\xi_kr)/r\vert\nu\rangle$ | the same, weighted by $q_k$ | the three-index Coulomb integral against a normalized s Gaussian, `ferric-integrals/src/site_basis.rs::SiteBasis` |

`Point` treats each tessera charge as a point in the potential and as a
Gaussian in $S$ and $D$. `GaussianSmeared` uses the Gaussian everywhere, which
is PySCF's `pcm.py` convention. The classical nuclear term of $v_k$ takes the
same erf form under the smeared probe.

## 22.3.3 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | $v^{\mathrm{nuc}}_k = \sum_B Z_B/\lvert\mathbf r_k-\mathbf R_B\rvert$ | $N N_{\mathrm{at}}$ | `ferric-pcm/src/potential.rs::nuclear_potential_at` |
| 2 | $v_k = v^{\mathrm{nuc}}_k + \sum_{\mu\nu}P_{\mu\nu}B^k_{\mu\nu}$, batched over tesserae, parallel over batches | $N\,N^2_{\mathrm{bf}}$ | `solute_potential_at_tesserae` |
| 3 | $q$, $E_{\mathrm{pcm}}$ (section 22.2) | $N^3$ | `solve_pcm_charges` |
| 4 | $V^{\mathrm{pcm}} = \sum_kq_kB^k$, parallel over shell pairs | $N\,N^2_{\mathrm{bf}}$ | `build_reaction_field_operator` |
| 5 | $V^{\mathrm{pcm}}$ added to every spin Fock matrix; $E_{\mathrm{pcm}}$ to the energy | $N^2_{\mathrm{bf}}$ | `solvent_terms` |

Steps 2 and 4 are bit-identical for every thread-pool size: each output block
is independent of the others. A tessera that coincides with a nucleus (a
degenerate cavity) is an error in step 1.

## 22.3.4 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `PcmConfig::probe` | `Point`, `GaussianSmeared` | `Point` | the probe of the table above; library only |
| `[pcm]` absent | | | a vacuum run, byte-identical to a build with no solvent code (`pcm_none_matches_vacuum_exactly` in `ferric-scf/src/rhf.rs`) |

## 22.3.5 Numerical anchors

| Quantity | System | Error | Source |
|---|---|---|---|
| total SCF energy on PySCF's injected cavity | water, NH3 / STO-3G, cc-pVDZ | $4.8\times10^{-12}$ Ha | `site/src/reference/validation.md`, IEF-PCM row |

## 22.3.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-pcm/theory.md`, section 7 | the potential, the operator and the energy bookkeeping |
| `scripts/pcm_ief_sympy.py`, claim 6 | $V^{\mathrm{pcm}} = \partial E_{\mathrm{pcm}}/\partial P$ |
| `crates/ferric-scf/tests/validation_pcm.rs` | the converged SCF against PySCF |

**Known limits (2026-11-15).**

- The CLI refuses `[pcm]` together with `[cosmo]`: they are two models of one
  solvent and would be added.
- The CLI wires `[pcm]` for the energy task on the listed kinds, and for
  `task = "optimize"` on a subset (`PCM_KINDS`, `PCM_OPTIMIZE_KINDS` in
  `crates/ferric-cli/src/config.rs`); frequencies with `[pcm]` are refused.
