# Nuclear forces {#forces}

<!-- status: edit; registry: pbc-hf, pbc-ks;
     theory: crates/ferric-pbc/theory.md#"6. Refusal gates and tolerances of the open-shell force and stress paths" -->

Moving atom $A$ in a periodic cell moves its nucleus, every basis function
centred on it and, for Kohn–Sham, every grid point homed on it, in every
lattice image at once. The force is the derivative of the energy of sections
@sec:ewald to @sec:kpoint-scf at the converged density, and each term of the energy gives one
term of the force. The Ewald and $\mathbf G = 0$ bookkeeping carries over
unchanged; the new feature is that two of its constants enter through the
overlap matrix. The forces drive `task = "optimize"`: atoms move, the lattice
is fixed.

Section @sec:forces-force-term-term derives the force term by term, @sec:forces-madelung-term-cancels shows why the
Madelung term drops out of it, and @sec:forces-refusal-gates lists the gates that refuse a state
the force formula does not apply to; @sec:forces-numerical-anchors to @sec:forces-sources give anchors, limits
and sources.

| Symbol | Meaning | Code name |
|---|---|---|
| $D_\sigma$, $D = D_\alpha + D_\beta$ | converged spin densities | |
| $W = \sum_\sigma D_\sigma F_\sigma D_\sigma$ | energy-weighted density | |
| $M$ | the matrix contracted with $\partial S/\partial R_A$ | `m` |
| $\alpha$ | exact-exchange fraction (1 for HF) | `alpha` |
| $Q_{\mu\nu}(\mathbf G)$ | bra-centre derivative of the pair transform | `pair_ft_deriv_chunked` |

| Entry point | Location |
|---|---|
| Gamma, RHF and RKS | `ferric-pbc/src/grad.rs::gamma_rhf_gradient` |
| Gamma, UHF and UKS | `ferric-pbc/src/grad.rs::gamma_uhf_gradient` |
| $k$-points | `ferric-pbc/src/kgrad.rs::kpoint_rhf_gradient` and siblings |
| CLI | `task = "optimize"` with a `[cell]` section |
| Python | `with_gradient=True` on the SCF entry points; `gradient()` in Hartree/Bohr per cell |

## The force, term by term {#forces-force-term-term}

**The energy being differentiated.** At the Gamma point, with $\mathbf G = 0$
dropped everywhere (`grad.rs` module documentation):

$$ E = \sum D\,h + \sum\Gamma_{\mu\nu\lambda\sigma}I_{\mu\nu\lambda\sigma} - \frac{\alpha v_M}{2}\sum_\sigma\operatorname{tr}(D_\sigma SD_\sigma S) + E_{xc} + E_{nn}, $$

with $\Gamma = \tfrac12D_{\mu\nu}D_{\lambda\sigma} - \tfrac\alpha2\sum_\sigma D^\sigma_{\mu\lambda}D^\sigma_{\nu\sigma}$,
$h = T + V_\text{SR} + V_\text{LR} + c_0Z_\text{tot}S$ (section @sec:nuclear-attraction) and $I$
the dense AFT tensor of section @sec:exchange-divergence. For RHF, $D_\sigma = D/2$ and
$\Gamma = \tfrac12DD - \tfrac14DD$.

**Fixed density, moving basis.** Differentiate at fixed converged $D_\sigma$.
The energy is stationary with respect to orbital rotations, so the only
orbital response that survives is the one forced by the orthonormality
constraint $C_\sigma^{\mathsf H}SC_\sigma = 1$. It gives the Pulay term
$-\sum W\,\partial S/\partial R_A$ with $W = \sum_\sigma D_\sigma F_\sigma D_\sigma$,
as in the molecular gradient of section @sec:hf-gradient. For RHF,
$\sum_\sigma D_\sigma FD_\sigma = 2(D/2)F(D/2) = \tfrac12DFD$.

**Each energy term, differentiated.**

| Energy term | Force term |
|---|---|
| kinetic, ECP | lattice-summed shifted one-electron derivative blocks |
| SR nuclear attraction (erfc) | three-centre derivative; the nuclear centre by translation invariance, $-(\text{bra} + \text{ket})$ |
| LR nuclear attraction | basis: $-(2/\Omega)\sum_{\mathbf G}v_\eta\operatorname{Re}[\rho_A'^*S(\mathbf G)]$ with $\rho'_A = 2\sum_{\mu\in A,\nu}D_{\mu\nu}Q_{\mu\nu}$; nucleus: the structure-factor derivative |
| two-electron (pure AFT) | $(8/\Omega)\sum_{\mathbf G}(4\pi/G^2)\operatorname{Re}\sum_{\mu\in A,\nu}Q^*_{\mu\nu}Z_{\mu\nu}$, $Z = \tfrac12D\rho(\mathbf G) - \tfrac\alpha2\sum_\sigma D_\sigma P(\mathbf G)D_\sigma$ |
| overlap coupling | $\sum M\,\partial S/\partial R_A$, $M = -W + c_0Z_\text{tot}D - \alpha v_M\sum_\sigma D_\sigma SD_\sigma$ |
| Ewald $E_{nn}$ | SR erfc plus LR structure factor (section @sec:ewald) |
| XC | AO term, grid-point motion and the partition-weight derivative |

The $\mathbf G$ sums run over the half sphere. $Q_{\mu\nu} = \partial P_{\mu\nu}/\partial A$
is the bra-centre derivative of the pair transform
(`ferric-pbc/src/pair_ft.rs::pair_ft_deriv_chunked`). The ket derivative is
$Q_{\nu\mu}$, because $P_{\mu\nu} = P_{\nu\mu}$ at reciprocal-lattice
$\mathbf G$ (section @sec:pair-ft). $Q$ is three times the size of the pair
transform, so it is streamed per $\mathbf G$ chunk together with $P$ and
contracted at once.

**The nuclear centre by translation invariance.** The three-centre integral
depends on the bra, ket and nuclear centres only through their differences,
so its three derivatives sum to zero. The code computes the bra and ket
derivatives and takes the nuclear one as minus their sum. It does not use
libint2's `erf_nuclear`/`erfc_nuclear` derivative operators, for the same
reason as the energy (section @sec:nuclear-attraction).

**The overlap-coupled matrix.** Every term whose geometry dependence is only
through $S$ goes into one matrix $M$:

- $-W$ is the Pulay term above.
- $c_0Z_\text{tot}D$: the $\mathbf G = 0$ term of the nuclear attraction,
  $c_0Z_\text{tot}S$ (section @sec:nuclear-attraction), depends on the geometry only through
  $S$. Its energy is $c_0Z_\text{tot}\operatorname{tr}(DS)$, and its derivative
  is $c_0Z_\text{tot}\sum D\,\partial S$.
- $-\alpha v_M\sum_\sigma D_\sigma SD_\sigma$: the explicit $S$ dependence of
  the Madelung energy $-\tfrac{\alpha v_M}{2}\sum_\sigma\operatorname{tr}(D_\sigma SD_\sigma S)$,
  per spin. The trace contains $S$ twice, which cancels the $\tfrac12$.

**The XC term.** On the periodic SSF/Becke grid the force has an AO term,
the motion of the grid points homed on $A$, and the derivative of the
partition weights over the image neighbour list. With the full grid response
the force is translation invariant. The module documentation records that the
response is about $2$–$3\times10^{-3}$ Ha/Bohr at a $(40, 50)$ grid on the
prototype cells, above optimization thresholds, so it is not optional.

**Consistency with the energy.** Every lattice sum in the force uses the
energy's truncation: the same pair images, nucleus candidates and screens,
the same $\mathbf G$ spheres, the same XC grid. The spheres do not depend on
the geometry. The force is therefore the derivative of the truncated energy
up to the primitive screens ($\le$ precision) and the AO extent threshold, and the finite-difference
anchors below test it to $10^{-7}$ Ha/Bohr.

## The Madelung term cancels {#forces-madelung-term-cancels}

Look at what `exxdiv = ewald` does to $M$. The Fock matrix of spin $\sigma$
contains $-\alpha v_MSD_\sigma S$ (section @sec:exchange-divergence), so $W$ contains

$$ D_\sigma\big(-\alpha v_MSD_\sigma S\big)D_\sigma = -\alpha v_M\,D_\sigma SD_\sigma SD_\sigma = -\alpha v_MD_\sigma , $$

using $D_\sigma SD_\sigma = D_\sigma$ twice. In $M$ this appears as $+\alpha v_MD_\sigma$.
The explicit term is $-\alpha v_MD_\sigma SD_\sigma = -\alpha v_MD_\sigma$. They
cancel, spin by spin. Forces for `ewald` and `none` therefore agree, for any
$\alpha$ and any spin state. This is consistent with section @sec:exchange-divergence: the
correction moves occupied levels but not orbitals, so it cannot move atoms.

Why per spin? The total-density form $-\tfrac{\alpha v_M}2DSD$ is the same
thing for a closed shell, where $D = 2D_\sigma$ and
$DSD = 4D_\sigma = 2\sum_\sigma D_\sigma$. On an open shell
$DSD$ contains the cross terms $D_\alpha SD_\beta + D_\beta SD_\alpha$, and the
cancellation fails. The module documentation records the size from the
prototype: $1.7\times10^{-2}$ to $1.5\times10^{-1}$ Ha/Bohr on open shells with
$\alpha > 0$. Closed shells, `exxdiv = none` and the sum of forces do not see
that error.

> **Verified equation {#eq:pbc-grad-madelung}** · the Madelung terms cancel in the force
>
> With the per-spin overlap term $-\alpha v_M\sum_\sigma D_\sigma SD_\sigma$,
> the overlap-coupled force is the same for `exxdiv = ewald` and `none`, for
> closed and open shells; with the total-density form it differs on an open
> shell.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `madelung-force` |
> | Negative control | the claim itself evaluates the total-density form and requires it to fail on the open shell |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/grad.rs::gamma_uhf_gradient` |
> | Code test | `h2_ewald_and_none_forces_agree` in `crates/ferric-pbc/tests/pbc_grad.rs` |
>
> <!-- verified: id=pbc.grad.madelung; script=scripts/pbc_kpoint_corr_sympy.py; claim=madelung-force; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/grad.rs::gamma_uhf_gradient; rust_test=ferric-pbc/tests/pbc_grad.rs::h2_ewald_and_none_forces_agree; registry=pbc-hf -->

The matrix is built in `ferric-pbc/src/grad.rs::assemble`:

<!-- excerpt: ferric-pbc/src/grad.rs::assemble -->
```rust
    let c0 = PI / (omega * omega * vol);
    let mut m = if mutation == Some(GradMutation::WSign) {
        w.clone()
    } else {
        -&w
    };
    m.scaled_add(c0 * ztot, &d);
    match mutation {
        Some(GradMutation::NoMadelungS) => {}
        Some(GradMutation::MadelungTotal) => {
            let dsd = d.dot(s_mat).dot(&d);
            m.scaled_add(-0.5 * alpha * vm, &dsd);
        }
        _ => {
            if alpha != 0.0 && vm != 0.0 {
                let dsd = spins.sandwich(s_mat);
                m.scaled_add(-alpha * vm, &dsd);
            }
        }
    }
```

| Code | Equation |
|---|---|
| `-&w` | $-W$ |
| `c0 * ztot`, `d` | $c_0Z_\text{tot}D$, with `omega` the nuclear split of section @sec:nuclear-attraction |
| `spins.sandwich(s_mat)` | $\sum_\sigma D_\sigma SD_\sigma$ |
| `-alpha * vm` | $-\alpha v_M$ |

The three `GradMutation` arms are the test mutants: the wrong sign of $W$,
the Madelung $S$ term left out, and the total-density form. Each is detected
by `h2_fd_anchor_catches_each_mutant`.

The Ewald nuclear gradient is part of section @sec:ewald's
`ferric-pbc/src/ewald.rs::ewald_nuclear_gradient_parts`, checked against
PySCF and finite differences in `ewald_gradient_matches_pyscf_grad_nuc_and_fd`.

There is no committed derivation of the full electronic force (the
two-electron, nuclear-attraction and XC terms above) in `theory.md` and no
proof script for it. The forces are anchored by finite differences of
ferric's own energy instead (@sec:forces-numerical-anchors). That is why this section's status is
`edit`.

## Refusal gates {#forces-refusal-gates}

A force or stress is first order in the departure of the SCF state from
stationarity. The derivation above assumed a stationary state, so the
open-shell entry points (Gamma and $k$-point ROHF/ROKS/UHF/UKS) refuse a
state that is not one, instead of returning a number. The table gives each
gate with its provenance, taken from the code and `git log -S`:
**measured** means calibrated against a force error, **roundoff** a decade or
more above the observed residual for matching inputs, **arbitrary** no
derivation or measurement recorded, and **definitional** no tolerance. No
threshold is a function of system size.

| Gate | Value | Where | Provenance |
|---|---|---|---|
| ROHF/ROKS orbital gradient $g = \max(\lvert f^\beta_{co}\rvert,\lvert f^\alpha_{ov}\rvert,\lvert (f^\alpha{+}f^\beta)_{cv}\rvert)$ | $g \le 10^{-5}$ Ha (NaN refused) | `grad.rs` `RO_ORBITAL_GRADIENT_TOL`, `ro_gate`; Gamma force and stress | measured: force error $\approx 0.1$–$0.74\,g$ (first order), so $10^{-5}$ caps it near $7\times10^{-6}$ Ha/Bohr, below optimizer thresholds ($\sim 4.5\times10^{-4}$); not a precision guarantee |
| Same, $k$-point | same number on the largest spectral norm per block, maximised over $k$ | `kuxcgrad.rs` `ro_orbital_gradient_k`, `kuxcstress.rs` | measured on H3/STO-3G 1x1x1 only (rank-1 blocks); the spectral norm is $\ge$ the Gamma max-element, so the $k$ gate can refuse a state the Gamma gate accepts. Constant shared |
| UKS spin commutator $\max_{\sigma,k}\lvert F_\sigma D_\sigma S - S D_\sigma F_\sigma\rvert$ | $\le 10^{-5}$ Ha | `kuxcgrad.rs` `UKS_COMMUTATOR_TOL`; $k$-point UKS force and stress | measured: commutator $1.6\times10^{-3}\to$ force error $7.2\times10^{-4}$, $2.7\times10^{-5}\to1.4\times10^{-5}$, $2.6\times10^{-6}\to5\times10^{-7}$. Gamma UKS has no commutator gate (only `converged`) |
| MO occupation classification (closed 1,1 / open 1,0 / virtual 0,0) | $\lvert n_i - \mathrm{round}(n_i)\rvert < 10^{-6}$ | `grad.rs` `RO_OCC_TOL`, `rohf_orbital_gradient` | arbitrary |
| Class counts vs $\mathrm{tr}(D_\sigma S)$ | $\lvert\Delta\rvert \le 10^{-6}$ electrons | `grad.rs` `rohf_orbital_gradient` | arbitrary |
| $\beta$-occupied, $\alpha$-empty MO | none | same | definitional |
| $k$ densities nested orthogonal projectors in the MO space ($D_\alpha^2{=}D_\alpha$, $D_\beta^2{=}D_\beta$, $D_\beta D_\alpha{=}D_\beta$) | max element deviation $\le 10^{-6}$ | `kuxcgrad.rs` `PROJECTOR_TOL` | arbitrary |
| Stored Fock $=$ Fock rebuilt from $D$ and the J/K source (all force/stress entries, per spin) | $\le 10^{-8}$ Ha (also for the Madelung-shift form) | `kgrad.rs` `FOCK_CONSISTENCY_BAR` | measured separation: matching inputs $2\times10^{-16}$; wrong `exxdiv` $0.79$; KS result to an HF entry $0.34$; wrong functional $5.9\times10^{-3}$ |
| $k$ ECP: $\lvert h - T - V - V_{\mathrm{ECP}}\rvert$ (or $T{+}V$) | $\le 10^{-10}$ Ha | `kgrad.rs` `HK_ECP_CONSISTENCY_BAR` | roundoff ($h$ is assembled from exactly these pieces) |
| ECP strain/force screen keeps the same triple count as hcore | exact integer equality | `stress.rs` (Gamma), `kgrad.rs` | definitional (same ECP precision/config and geometry) |
| SCF $k$-points equal the mesh | $\max\lvert\Delta k\rvert \le 10^{-12}$ | `kgrad.rs` `check_inputs_rsh` | roundoff |
| RS-GDF overlap equals `hc.s` (J3 $G{=}0$ term) | $\max\lvert\Delta S\rvert \le 10^{-12}\max(1,\max\lvert S\rvert)$ | `grad.rs` `check_rsgdf_inputs` | roundoff |
| `hcore.omega == hcore_cfg.omega` | exact `f64` equality | `grad.rs`, `kgrad.rs`, `kstress.rs` | definitional (the strain differentiates the build's own partition) |
| Range split on the RS-GDF stress | $\lambda \le 1$ | `kstress.rs` `refuse_split_mismatch` | measured: $\lambda\le1$ matches FD to $\sim10^{-11}$, $\lambda=2$ to $\sim10^{-8}$ (the Fock-guard bar); also refused: build/config disagreeing on whether a split ran |
| Exact-exchange fraction | $\in[0,1]$ | every `*stress.rs`, `*grad.rs` entry | definitional |
| SCF `converged` | boolean | every gradient entry (stress follows the force checks) | definitional |
| Non-finite $E_{xc}$, $V_{xc}$, force, strain derivative | any NaN/Inf | each XC and total assembly | definitional |

Three groups of gates stand out. The orbital-gradient and commutator gates
are the stationarity condition itself, and their values come from measured
force errors. The Fock-consistency gate catches a mismatched input, such as a
force requested with the wrong `exxdiv` or functional, by rebuilding the Fock
matrix: a matching input reproduces it to $2\times10^{-16}$, the nearest
mismatch measured differs by $5.9\times10^{-3}$, and the bar sits between. The
rest are definitional or arbitrary bounds on the shape of the input.

Two tolerances in the code are not force or stress gates. `IMAG_TOL` $= 10^{-9}$
in `gamma_rsh.rs` (the imaginary part of the $k = 0$ exchange of a real
density) belongs to the Gamma range-separated energy. The memory pre-flights
(`Ledger::check`) are budget refusals, not tolerances.

Scope gates have no tolerance. They are functional and route allow-lists,
each with a named-refusal test: VV10 and double hybrids on every derivative
path; meta-GGA derivatives only for pure SCAN/r2SCAN (TPSS closed-shell dense
only; ROKS meta-GGA on RS-GDF refused); range-separated hybrids only for the
validated HSE06/CAM-B3LYP set; Gamma ROKS range-separated stress.

The tests: `crates/ferric-pbc/tests/pbc_gate_tolerances.rs` brackets each
Gamma occupation and count tolerance; `kuxcgrad.rs`
`gate_tests::projector_tolerance_is_one_micro`;
`crates/ferric-pbc/tests/pbc_kuxcgrad_ro.rs`
`scf_kpoints_displaced_from_the_mesh_are_refused`. The orbital-gradient,
commutator, Fock-consistency, $\lambda$ and exchange-fraction gates are pinned
by the existing `pbc_kuxcgrad*.rs`, `pbc_grad_ro.rs`, `pbc_kstress*.rs` and
`validation_*stress*.rs` files. Unit tests with crafted inputs reach three more
refusals: `grad.rs` `gate_tests` (RS-GDF overlap, $\max\lvert\Delta S\rvert$
at $5\times10^{-13}$ accepted, $2\times10^{-12}$ and NaN refused), `kgrad.rs`
`gate_tests` (`ecp_presence`: the $10^{-10}$ bar with and without an ECP at
$5\times10^{-11}$ / $2\times10^{-10}$ / NaN, and the kept-triple count
mismatch), and `crates/ferric-pbc/tests/pbc_grad_ecp.rs` (Gamma force and
stress triple-count mismatch). Every max-reduction feeding a refusal
propagates NaN (`grad::max_nan`), and every gate is written `!(x <= bar)`, so
a NaN is refused.

## Numerical anchors {#forces-numerical-anchors}

| Quantity | Bar | Test |
|---|---|---|
| H2 force against finite differences of ferric's own energy, both `exxdiv` | $10^{-7}$ Ha/Bohr; net force below $10^{-10}$ | `crates/ferric-pbc/tests/pbc_grad.rs` `h2_force_matches_fd_of_own_energy_both_exxdiv` |
| `ewald` against `none` forces | $10^{-12}$ Ha/Bohr | `h2_ewald_and_none_forces_agree` |
| each deliberate force mutant | detected above $10^{-3}$ | `h2_fd_anchor_catches_each_mutant` |
| triclinic s/p cell against finite differences | $1.5\times10^{-8}$ Ha/Bohr | `triclinic_sp_force_matches_fd_of_own_energy` |
| $1\times1\times3$ mesh force against the supercell force on one copy | | `crates/ferric-pbc/tests/pbc_kgrad.rs` `h2_1x1x3_force_is_the_supercell_force_on_one_copy` |

## Limits {#forces-limits}

- The lattice is fixed; there is no lattice-vector relaxation.
  `task = "frequencies"` is an error for a cell.
- VV10 is refused on every periodic derivative path.
- `range_split` is refused with `task = "optimize"`.

## Sources {#forces-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-pbc/src/grad.rs`, module documentation | the force terms |
| `crates/ferric-pbc/theory.md`, §1 (Gradient), §6 | the Ewald gradient, the refusal gates |
| `scripts/pbc_kpoint_corr_sympy.py` | the Madelung cancellation |
