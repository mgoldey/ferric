# The exchange divergence {#exchange-divergence}

<!-- status: ready; registry: pbc-hf;
     theory: crates/ferric-pbc/theory.md#"4. Exchange divergence (`exxdiv`)" -->

Dropping $\mathbf G = 0$ is exact for the Coulomb energy of a neutral cell:
the electronic and nuclear divergences cancel. Exchange has no such partner,
because each electron's exchange hole holds one unit of charge, so its
$\mathbf G = 0$ term has nothing to cancel against. In the infinite crystal
that term is integrable, but a finite sampling of it (one cell at $\Gamma$,
or a mesh) misses a finite piece that vanishes only as $1/L$ in the cell or
supercell size. The Madelung correction adds that piece back for a probe
charge in a uniform background, and the error then falls as $L^{-3}$.

Section @sec:exchange-divergence-exchange-diverges shows where the missing piece comes from, @sec:exchange-divergence-term-does derives what
the correction does to the energy and the orbitals, and @sec:exchange-divergence-options to @sec:exchange-divergence-sources
give the options, anchors and sources.

| Symbol | Meaning | Code name |
|---|---|---|
| $v_M$ | Madelung constant of the cell, or of the $\operatorname{diag}(N)$ supercell for a mesh | `madelung`, `KPointMesh::madelung` |
| `exxdiv` | `"ewald"` (correction on) or `"none"` | `ExxDiv::{Ewald,None}` |
| $\alpha$ | exact-exchange fraction, 1 for HF | `exact_exchange` |

| Entry point | Location |
|---|---|
| Rust | `ferric-pbc/src/kpts.rs::madelung`, `ferric-pbc/src/ewald.rs::madelung_constant` |
| K builders that carry the term | `ferric-pbc/src/rsgdf.rs::k_builder`, `ferric-pbc/src/kdense_aft.rs::build` |
| CLI | `[cell] exxdiv = "ewald"` (default) or `"none"` |
| Python | `exxdiv=` on every periodic SCF entry point |

## Why exchange diverges {#exchange-divergence-exchange-diverges}

**The dense integrals.** The simplest periodic electron repulsion puts all of
the Coulomb operator in reciprocal space, with the pair transforms of section
@sec:pair-ft and $\mathbf G = 0$ dropped:

$$
(\mu\nu|\lambda\sigma) = \frac{1}{\Omega}\sum_{\mathbf G\neq 0}\frac{4\pi}{G^2}
 \operatorname{Re}\big[P^*_{\mu\nu}(\mathbf G)P_{\lambda\sigma}(\mathbf G)\big].
$$

`crates/ferric-pbc/src/dense_aft.rs` stores this tensor. It
holds $8n_\text{ao}^4$ bytes and its $\mathbf G$ count grows as
$\Omega p_\text{max}^{3/2}$, so it is a correctness oracle for toy cells,
never a production builder. Its size is capped by `max_eri_gb` (default
512 MiB, `DEFAULT_DENSE_AFT_MAX_BYTES`, about $n_\text{ao} = 90$), and a
tensor over the cap is an error, not a warning.

**Coulomb.** $J$ contracts the pair $\lambda\sigma$ with the total density.
The $\mathbf G = 0$ component of $\sum_{\lambda\sigma}D_{\lambda\sigma}P_{\lambda\sigma}(0)$
is $\operatorname{tr}(DS)$, the number of electrons per cell. The nuclear
attraction of section @sec:nuclear-attraction drops the matching $\mathbf G = 0$ term of the
nuclei, and in a neutral cell the two are equal and opposite. Dropping both
is exact.

**Exchange.** $K_{\mu\nu} = \sum_{\lambda\sigma}(\mu\lambda|\nu\sigma)D_{\lambda\sigma}$
on the total density, and $F = h + J - \tfrac12K$. In orbitals, the exchange
energy contains $(ii|ii)$, the interaction of $|\phi_i|^2$ with itself. That
density holds one electron, so its $\mathbf G = 0$ component is 1, and no
other term of the energy has the opposite $\mathbf G = 0$ charge. Dropping
it is not exact.

**Why the missing piece is finite and shrinks as $1/L$.** In the infinite
crystal, the exchange pair densities of section @sec:kpoint-scf carry crystal momentum
$q$, and the kernel near the singularity is $4\pi/q^2$. Averaged over the
Brillouin zone this is finite, because in three dimensions

$$ \int_{|q| < q_0}\frac{4\pi}{q^2}\,d^3q = 16\pi^2q_0 . $$

A mesh of $N_k$ points samples the average at points and drops the one at
$q = 0$. The region it stands for has volume $V_\text{BZ}/N_k$, a radius
$q_0 \propto N_k^{-1/3}$, and so a contribution proportional to $q_0$. With
$L$ the edge of the supercell the mesh represents, $q_0 \propto 1/L$: the
error of `exxdiv = none` falls only as $1/L$. A Gamma calculation is the
$N_k = 1$ case, with $L$ the cell itself.

**The correction.** `exxdiv = ewald` treats each exchange hole as a point
probe charge in a uniform background and adds the Ewald energy that charge
has with its own images. This is PySCF's `exxdiv='ewald'`. It is one term in
the exchange builder,

$$ K \mathrel{+}= v_M\,SDS , $$

with $v_M$ the Madelung constant of section @sec:ewald. The point-charge part of
the missing piece is then restored; what remains depends on the shape of the
hole and falls as $L^{-3}$.

## What the term does {#exchange-divergence-term-does}

**The energy shift.** The energy is $\tfrac12\operatorname{tr}[(h + F)D]$, so
the $-\tfrac12K$ in $F$ contributes $-\tfrac14\operatorname{tr}(KD)$. Insert the
new term:

$$ \Delta E = -\tfrac14v_M\operatorname{tr}(SDSD) . $$

A closed-shell density is idempotent in the $S$ metric, $DSD = 2D$. Then
$\operatorname{tr}(SDSD) = 2\operatorname{tr}(SD) = 2N$, and

$$
\Delta E = -\tfrac14 v_M \operatorname{tr}(DSDS) = -\tfrac12 N v_M,
\qquad
\frac{\partial\Delta E}{\partial D} = -\tfrac12 v_M SDS .
$$

Read it per electron: $-v_M/2$ is the Ewald energy of one unit charge with
its images and background (the definition of $v_M$ in section @sec:ewald). Each
electron's exchange hole receives that energy once.

> **Verified equation {#eq:pbc-exxdiv-energy-shift}** · the closed-shell energy shift
>
> For an idempotent closed-shell density, $DSD = 2D$, so
> $\Delta E = -\tfrac14v_M\operatorname{tr}(DSDS) = -\tfrac12Nv_M$ and
> $\partial\Delta E/\partial D = -\tfrac12v_MSDS$.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, part S4, claim `exxdiv-shift` |
> | Negative control | none in the script |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/rsgdf.rs::k_builder` |
> | Code test | `h2_rhf_ewald_minus_none_is_the_madelung_strain` in `crates/ferric-pbc/tests/pbc_stress.rs` |
>
> <!-- verified: id=pbc.exxdiv.energy-shift; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=exxdiv-shift; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/rsgdf.rs::k_builder; rust_test=ferric-pbc/tests/pbc_stress.rs::h2_rhf_ewald_minus_none_is_the_madelung_strain; registry=pbc-hf -->

**The orbitals do not move.** Take $S$-orthonormal orbitals $C$ at any $k$,
$C^{\mathsf H}SC = 1$, with $D = 2C_oC_o^{\mathsf H}$. Then
$C^{\mathsf H}SC_o$ is the occupied block of the unit matrix, and the Fock
term $-\tfrac12v_MSDS$ in the MO basis is

$$ C^{\mathsf H}\big(-\tfrac12v_MSDS\big)C = -v_M\operatorname{diag}(1_\text{occ}, 0_\text{virt}) . $$

It is diagonal: every occupied level moves down by $v_M$, every virtual
level stays, and the eigenvectors do not change. The energy per cell moves
by $-v_MN_e/2$.

> **Verified equation {#eq:pbc-exxdiv-level-shift}** · the occupied levels move, the orbitals do not
>
> For S-orthonormal orbitals at any $k$,
> $C^{\mathsf H}(-\tfrac12v_MSDS)C = -v_M\operatorname{diag}(1_\text{occ}, 0_\text{virt})$,
> and the energy per cell moves by $-v_MN_e/2$.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `madelung-shift` |
> | Negative control | the runner's mutation: the Madelung term divided by $N_k$ (fails `krhf-supercell`) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/kpts.rs::madelung`, `ferric-pbc/src/kdense_aft.rs::build` |
> | Code test | `mutant_primitive_madelung_is_caught_and_leaves_exxdiv_none_exact` in `crates/ferric-pbc/tests/pbc_krhf.rs` |
>
> <!-- verified: id=pbc.exxdiv.level-shift; script=scripts/pbc_kpoint_corr_sympy.py; claim=madelung-shift; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/kpts.rs::madelung; rust_test=ferric-pbc/tests/pbc_krhf.rs::mutant_primitive_madelung_is_caught_and_leaves_exxdiv_none_exact; registry=pbc-hf -->

**Open shells and Kohn–Sham.** The builder is linear in the density it is
given. Per spin it carries $v_MSD_\sigma S$ with coefficient 1, and the energy
term is $-\tfrac{\alpha v_M}{2}\sum_\sigma\operatorname{tr}(D_\sigma SD_\sigma S)$,
with $\alpha$ the exact-exchange fraction (1 for HF). The occupied levels of
each spin move by $-v_M$ (times $\alpha$ in Kohn–Sham), and the energy by
$-\alpha v_M(N_\alpha + N_\beta)/2$. Writing the term on the total density
instead, $-\tfrac{\alpha v_M}{2}DSD$, agrees only for a closed shell; section
@sec:forces shows where the difference would show up. The RS-GDF builders of
section @sec:rsgdf use the same constant. A range-separated hybrid on a mesh uses
the mixed supercell constant $c_\text{sr}v_M^\text{full} + (c_\text{lr} - c_\text{sr})v_M^\text{LR}(\omega)$
(`ferric-pbc/src/rsh.rs::kpoint_madelung`).

The consequences, each proven here or in section @sec:kpoint-scf:

| Consequence | Where |
|---|---|
| At fixed orbitals every occupied level moves by $-v_M$ (times $\alpha$ in Kohn–Sham), virtual levels and orbitals do not move; $\Delta E = -\alpha v_MN_e/2$ per cell | Verified equations @eq:pbc-exxdiv-energy-shift and @eq:pbc-exxdiv-level-shift |
| On a mesh, the constant is the supercell's and is added at every $k$ with no $1/N_k$ | Verified equation @eq:pbc-kscf-supercell |
| Per spin it is $v_MSD_\sigma S$ with coefficient 1; the total-density form is wrong for open shells | Verified equations @eq:pbc-kscf-uhf and @eq:pbc-grad-madelung |

**Why open shells start without the term.** Because the orbitals do not
move, `none` and `ewald` share their stationary densities. But the shift
changes which states are self-consistent under aufbau. A state with a hole
shallower than $v_M$ below the Fermi level can be self-consistent under
`ewald` only. The open-shell drivers therefore converge with `none` first and
continue with `ewald` (`EwaldStart::Staged`, the default), and report each
spin's gap against $v_M$. The $k$-point UHF module documentation records the
measurement behind the default: on a triclinic four-hydrogen s+p triplet a
direct `ewald` start stopped 15 mHa high on a $1\times1\times1$ mesh and
93 mHa high on $1\times1\times2$, and was not trapped on $1\times1\times3$.
In the measured staged run, the `ewald` continuation from the `none` density
took one iteration.

**The code.** In the RS-GDF exchange builder the term is the last step of
every build. These are the last lines of `RsGdfK::build_untimed` in
`crates/ferric-pbc/src/rsgdf.rs`, with the shape check, the scratch-budget
check and the return value removed:

```rust
        exchange_aux_grouped(&self.gdf.b, d, n, k)?;
        if self.madelung != 0.0 {
            let sds = self.gdf.s.dot(d).dot(&self.gdf.s);
            k.scaled_add(self.madelung, &sds);
        }
```

`exchange_aux_grouped` forms the fitted $\sum_kB_kDB_k^{\mathsf T}$ of section
@sec:rsgdf, and `madelung` is $v_M$ for `ewald` and 0 for `none`. The builder
does not override the trait's occupied-orbital path, which rebuilds
$D = CC^{\mathsf T}$ and calls this function, so the term is kept there
too; a test guards that (`k_builders_keep_madelung_on_the_occupied_path` in
`crates/ferric-pbc/tests/pbc_uhf.rs`).

## Options {#exchange-divergence-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[cell] exxdiv`, `exxdiv=` | `"ewald"`, `"none"` (strict) | `"ewald"` | adds $v_MSDS$ to every exchange build |
| `ewald_start=` (open-shell Python drivers) | `"staged"`, `"direct"` | `"staged"` | converge with `none` first, then `ewald` |

Correlation methods read the reference's `exxdiv` as a stated input, because
the occupied orbital energies differ by $v_M$ between the two (section @sec:mp2-drpa).

## Numerical anchors {#exchange-divergence-numerical-anchors}

| Quantity | Test |
|---|---|
| $E_\text{none} - E_\text{ewald} - n_\text{occ}v_M$ on the $k$-point supercell anchors, $< 10^{-11}$ | `crates/ferric-pbc/tests/pbc_krhf.rs` (inside `supercell_anchor`) |
| PySCF energies with both `exxdiv` values | `crates/ferric-pbc/tests/pbc_dense_aft_scf.rs` |
| `ewald` minus `none` stress equals the Madelung strain term | `crates/ferric-pbc/tests/pbc_stress.rs` `h2_rhf_ewald_minus_none_is_the_madelung_strain` |

## Sources {#exchange-divergence-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-pbc/theory.md`, §4 and §8 | the correction and its per-spin and per-$k$ forms |
| `scripts/pbc_ewald_pairft_stress_sympy.py` part S4, `scripts/pbc_kpoint_corr_sympy.py` | the proofs |
| J. Paier, R. Hirschl, M. Marsman, G. Kresse, *J. Chem. Phys.* **122**, 234102 (2005) | the probe-charge correction |
