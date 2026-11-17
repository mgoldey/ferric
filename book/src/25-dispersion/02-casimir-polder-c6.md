# Casimir–Polder $C_6$ from dynamic polarizabilities {#casimir-polder-c6}

<!-- status: ready; registry: rpa-polarizability-c6 -->

D3 reads its $C_6$ from a table. Where does a $C_6$ come from in the first
place? Two neutral fragments $A$ and $B$ far apart attract as
$-C_6^{AB}/R^6$, and that coefficient measures how easily each fragment's
charge fluctuates, at every frequency, not only how much it polarizes in a
static field. The Casimir–Polder formula makes this exact: $C_6^{AB}$ is an
integral over imaginary frequency of the two dynamic polarizabilities.
ferric evaluates that integral for per-atom polarizabilities from three
sources: the Tkatchenko–Scheffler single pole, the direct-RPA response itself
(PDEP-RPA), and the full-range self-consistently screened TS model of section
@sec:mbd.

The coefficients are exported as data (section @sec:polarizability-pdep-c6); no energy in ferric
sums them. The plan: the integral and its pole form (@sec:casimir-polder-c6-casimir-polder-integral), the three
sources of $\alpha^A(iu)$ (@sec:casimir-polder-c6-three-sources), the direct-RPA per-atom kernel
(@sec:casimir-polder-c6-direct-rpa-per-atom), then the algorithm, options and anchors.

| Symbol | Meaning | Code name |
|---|---|---|
| $A, B$ | atoms | `a`, `b` |
| $u_k, w_k$ | frequency node and weight (section @sec:quadratures) | `freqs`, `weights` |
| $\alpha^A_{ij}(iu)$ | per-atom polarizability tensor | `DynamicPolarizability::per_atom[a][k]` |
| $\bar\alpha = \tfrac13\operatorname{tr}\alpha$ | isotropic average | `iso[a][k]` |
| $g_{ia}(u)$ | $\Delta_i^a/(u^2 + (\Delta_i^a)^2)$ | `Frequency::weights` |

| Entry point | Location |
|---|---|
| Rust | `ferric-rpa/src/dispersion.rs::{casimir_polder_c6,ts_dynamic_polarizability,pdep_dynamic_polarizability}`, `ferric-rpa/src/dispersion/mbd.rs::mbd_dynamic_polarizability` |
| CLI | `method.kind = "pdep-rpa"` with `[rpa] export_npz`, `compute_c6`, `c6_source = "ts" \| "pdep" \| "mbd"`, `c6_partition` |
| Python | `run_pdep_rpa(..., properties=True, c6_source=..., c6_partition=...)`; arrays `c6_iso`, `c6_aniso`, `c6_molecular_iso` on the result |

## The Casimir–Polder integral {#casimir-polder-c6-casimir-polder-integral}

Start with the pole form of a polarizability. A response function is a sum
over excitations $n$ with energies $\Omega_n$ and strengths $f_n$, and on
the imaginary axis it is real and smooth:
$\alpha(iu) = \sum_n f_n/(\Omega_n^2 + u^2)$. London's second-order
dispersion coefficient is a sum over pairs of excitations, one on each
fragment, (@eq:disp-c6-pole-sum) below. Its energy denominator contains
$\Omega_m + \Omega_n$, which does not factorize into an $A$ part and a $B$
part. The identity

$$ \int_0^\infty\frac{du}{(\Omega_m^2 + u^2)(\Omega_n^2 + u^2)} = \frac{\pi}{2\,\Omega_m\Omega_n(\Omega_m + \Omega_n)} $$

turns it into an integral whose integrand does factorize, one pole from
each fragment. Multiply by $\tfrac3\pi f^A_mf^B_n$, sum over $m$ and $n$,
and pull the sums inside the integral. The sums are the two
polarizabilities, and the result is the Casimir–Polder formula with its
quadrature on the nodes $u_k$:

$$ C_6^{AB} = \frac{3}{\pi}\int_0^\infty \bar\alpha^A(iu)\,\bar\alpha^B(iu)\,du \approx \frac{3}{\pi}\sum_k w_k\,\bar\alpha^A(iu_k)\,\bar\alpha^B(iu_k) . \label{eq:casimir-polder-c6-1} $$

The closed double sum is

$$ C_6^{AB} = \frac32\sum_{mn}\frac{f^A_m f^B_n}{\Omega_m\Omega_n(\Omega_m + \Omega_n)} . \label{eq:disp-c6-pole-sum} $$

Check it on one pole per atom. Write
$\bar\alpha_A(iu) = \bar\alpha_A/(1 + (u/\omega_A)^2) = \bar\alpha_A\omega_A^2/(\omega_A^2 + u^2)$,
so $f_A = \bar\alpha_A\omega_A^2$ and $\Omega_A = \omega_A$. Substitute into
(@eq:disp-c6-pole-sum) and cancel one power of each frequency. The result is the London formula

$$ C_6^{AB} = \frac32\,\frac{\bar\alpha_A\bar\alpha_B\,\omega_A\omega_B}{\omega_A + \omega_B} . $$

For two identical atoms it is $C_{6,A} = \tfrac34\bar\alpha_A^2\omega_A$, so
an atom's static polarizability and homonuclear $C_6$ fix its frequency,
$\omega_A = \tfrac43 C_{6,A}/\bar\alpha_A^2$. Section @sec:casimir-polder-c6-three-sources uses exactly
this.

> **Verified equation @eq:disp-c6-pole-sum** · the pole double sum
>
> $C_6$ from the coded kernel (@eq:disp-c6-kernel-vs-casida) on a 200-point grid equals the analytic
> double sum over the dRPA excitations of a random closed-shell model.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_c6_polarizability_sympy.py`, check `c6-pole-sum` |
> | Negative control | the prefactor $3/\pi \to 3/(2\pi)$ turns `c6-quadrature-converged` red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/dispersion.rs::casimir_polder_c6` |
> | Code test | `pdep_c6_molecular_n2_aug_cc_pvdz_vs_numpy` in `crates/ferric-rpa/tests/validation_pdep_c6.rs` |
>
> <!-- verified: id=disp.c6.pole-sum; script=scripts/rpa_c6_polarizability_sympy.py; claim=c6-pole-sum; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/dispersion.rs::casimir_polder_c6; rust_test=ferric-rpa/tests/validation_pdep_c6.rs::pdep_c6_molecular_n2_aug_cc_pvdz_vs_numpy; registry=rpa-polarizability-c6 -->

`casimir_polder_c6` forms three things from the per-atom tensors on the
grid. The isotropic pair matrix is (@eq:casimir-polder-c6-1) for every ordered pair, including
$A = B$. Its core, unchanged:

<!-- excerpt: ferric-rpa/src/dispersion.rs::casimir_polder_c6 -->
```rust
let mut s_iso = 0.0;
for k in 0..nfreq {
    s_iso += dyn_pol.weights[k] * iso[a][k] * iso[b][k];
}
iso_row[b] = pref * s_iso;
```

`pref` is $3/\pi$ and `iso[a][k]` is $\bar\alpha^A(iu_k)$. The rows are
independent, so the function runs them in parallel over `a`; each pair
writes its own cells, and the result equals the serial loop.

The anisotropic coefficient is element-wise,
$C^{AB}_{6,ij} = \frac3\pi\sum_k w_k\,\alpha^A_{ij}(iu_k)\,\alpha^B_{ij}(iu_k)$,
not a tensor contraction (section @sec:polarizability-pdep-c6). The molecular $C_{6,\mathrm{mol}}$
comes from the molecular $\bar\alpha(iu)$. When the molecular tensor is the
sum of the per-atom ones, expand the square of the sum: the molecular
coefficient is then exactly the sum of the full pair matrix,
$\sum_{A,B}C_6^{AB}$. Once charge transfer between atoms is present
(@sec:casimir-polder-c6-direct-rpa-per-atom) it is not. When no molecular tensor was filled in, the function
returns the pair sum.

## Three sources of $\alpha^A(iu)$ {#casimir-polder-c6-three-sources}

| `c6_source` | $\alpha^A(iu)$ | Function |
|---|---|---|
| `ts` (default) | single pole $\bar\alpha_A/(1 + (u/\omega_A)^2)$ with $\bar\alpha_A = r_A\alpha^{\mathrm{free}}$, $C_{6,A} = r_A^2C_6^{\mathrm{free}}$, $\omega_A = \tfrac43C_{6,A}/\bar\alpha_A^2$; the static per-atom tensor supplies only the shape (@eq:ml-c6-ts-pole) | `ts_dynamic_polarizability`, `mbd::ts_atom_params` |
| `pdep` | the direct-RPA per-atom response (@eq:casimir-polder-c6-3) at each node | `pdep_dynamic_polarizability` |
| `mbd` | the TS poles screened by full-range coupled dipoles (section @sec:mbd-full-range-screening) | `mbd_dynamic_polarizability` |

The Tkatchenko–Scheffler (TS) model starts from a free atom and asks how much
the molecule has squeezed it. The measure is the Hirshfeld volume ratio
$r_A = V_A/V_A^{\mathrm{free}}$, where $V_A = \int w_A\rho\,r^3$ weights the
density by the atom's Hirshfeld share. Polarizability scales like a volume,
so $\bar\alpha_A = r_A\alpha^{\mathrm{free}}$, and $C_6$ is a product of two
polarizabilities, so $C_{6,A} = r_A^2C_6^{\mathrm{free}}$. The frequency then
follows from the homonuclear London identity of @sec:casimir-polder-c6-casimir-polder-integral, and the single pole
is the whole frequency dependence. The tensor is $\bar\alpha_A(iu)$ times
the static per-atom tensor normalized to unit isotropic average (an
isotropic shape when that average is below $10^{-12}$). Only the shape comes
from the molecular calculation. The TS atoms are independent oscillators,
so the molecular tensor is the sum of the per-atom ones.

Where do the free-atom numbers come from? $\alpha^{\mathrm{free}}$ and
$C_6^{\mathrm{free}}$ are Tkatchenko and Scheffler (PRL 102, 073005 (2009),
Table I) for $Z \le 18$ and Gould and Bučko (JCTC 12, 3603 (2016)) for
$Z = 19$–$54$, except Pd. Gould and Bučko's Pd is not the closed-shell
$4d^{10}$ ground state; ferric takes $\alpha = 26.14$ a.u. from Jerabek,
Schwerdtfeger and Nagle (PRA 98, 012508 (2018)) and $C_6 = 157.5$ a.u. from
Gobre's compilation. Any other element ($Z > 54$) is a hard error, never a
substituted value such as hydrogen's frequency. $V_A^{\mathrm{free}}$ has no
table at all. It
comes only from a live free-atom SCF with the same basis and SCF settings,
integrated on the same atom-centred Becke–Lebedev quadrature as the
molecular volumes, so the ratio compares like with like. If that SCF fails,
or gives a free volume at or below $10^{-10}$, TS $C_6$ is skipped with a
warning; no denominator is fabricated. The volume ratios are always
Hirshfeld, whatever `c6_partition` says, because TS was parametrized with
Hirshfeld volumes.

## The direct-RPA per-atom polarizability {#casimir-polder-c6-direct-rpa-per-atom}

The TS pole is a model. The `pdep` source computes $\alpha^A(iu)$ from the
same direct-RPA response that screens the correlation energy in chapter @sec:rpa.
The question is how to give each atom its share of a molecular response.
ferric uses the Krishtal–Senet–Van Alsenoy intrinsic polarizability (JCP
125, 034312 (2006)): the bra is the atom-weighted dipole
$m^A_d = w_A(\mathbf r)(r - R_A)_d$ of a Becke or Hirshfeld partition
(section @sec:atomic-partitions), the ket the molecular dipole $\mu_j$:

$$ \alpha^A_{dj}(iu) = \langle m^A_d|\,\mathbf R(iu)\,|\mu_j\rangle, \qquad \mathbf R = \big(\tfrac14\mathbf G^{-1} + \mathbf K\big)^{-1}, \quad \mathbf G = \operatorname{diag}(g_{ia}), \quad \mathbf K = \tilde{\mathbf B}^{\mathsf T}\tilde{\mathbf B}, \label{eq:casimir-polder-c6-3} $$

with $\tilde B^P_{ia}$ the dressed RI factor (section @sec:factorization-and-metric) and the vectors
$m^A_{d,ia} = \langle i|m^A_d|a\rangle$, $\mu_{j,ia} = \langle i|r_j|a\rangle$.

Is this the usual singlet response? Substitute
$\mathbf G^{-1} = (u^2 + \boldsymbol\Delta^2)\boldsymbol\Delta^{-1}$ and
factor out $\boldsymbol\Delta^{-1}$:
$\mathbf R = 4\,[\boldsymbol\Delta(\boldsymbol\Delta + 4\mathbf K) + u^2]^{-1}\boldsymbol\Delta$,
the direct-RPA response of section @sec:polarizabilities-from-dielectric-modes. The 4 is 2 for spin and 2 for the
pair of poles.

$\mathbf R$ is an $n_on_v\times n_on_v$ inverse, too large to form. The
Woodbury identity moves it into the auxiliary basis. With
$\mathbf A = \tfrac14\mathbf G^{-1}$, $\mathbf A^{-1} = 4\mathbf G$ and
$\mathbf K = \tilde{\mathbf B}^{\mathsf T}\mathbf 1\tilde{\mathbf B}$,

$$ \mathbf R = 4\mathbf G - 4\mathbf G\tilde{\mathbf B}^{\mathsf T}\big(\mathbf 1 + 4\tilde{\mathbf B}\mathbf G\tilde{\mathbf B}^{\mathsf T}\big)^{-1}\tilde{\mathbf B}\,4\mathbf G . $$

Sandwich it between $m^A_d$ and $\mu_j$ and use that $\mathbf G$ is
diagonal, so $\mathbf G\mu_j = g\circ\mu_j$:

$$ \alpha^A_{dj}(iu) = 4\,m^A_d\cdot(g\circ\mu_j) - 16\,\big(\tilde{\mathbf B}(g\circ m^A_d)\big)^{\mathsf T}\,\tilde{\boldsymbol\varepsilon}^{-1}\,\big(\tilde{\mathbf B}(g\circ\mu_j)\big), \qquad \tilde{\boldsymbol\varepsilon} = \mathbf 1 + 4\tilde{\mathbf B}\mathbf G\tilde{\mathbf B}^{\mathsf T}, \label{eq:disp-c6-kernel-vs-casida} $$

the same dielectric matrix as the dRPA energy (@eq:drpa-three-ways-3). The first term is the
bare, unscreened response; the second is the screening by the other
electrons. The 16 is the outer 4 times the 4 of the dielectric.

Check the static limit. At $u = 0$, $g_{ia} = 1/\Delta_i^a$, and (@eq:disp-c6-kernel-vs-casida) is
the static per-atom tensor that section @sec:atomic-partitions exports. The dynamic and static
paths share one definition, so the dynamic value at $u = 0$ equals the
static one. The code then symmetrizes each tensor,
$\alpha^A \to \tfrac12(\alpha^A + \alpha^{A\mathsf T})$.

> **Verified equation @eq:disp-c6-kernel-vs-casida** · the aux-basis kernel is the dRPA response
>
> The coded kernel equals the independent sum over closed-shell dRPA (Casida)
> excitations, $\alpha(iu) = \sum_n f_n/(\Omega_n^2 + u^2)$ with
> $\Omega_n^2$ the eigenvalues of $\Delta^{1/2}(\Delta + 4K)\Delta^{1/2}$,
> on a random model; its static form is the $u \to 0$ limit.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_c6_polarizability_sympy.py`, checks `kernel-vs-casida` and `static-limit` |
> | Negative control | screening term $16 \to 4$ turns `kernel-vs-casida` red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/properties/per_atom.rs::closed_per_atom_alpha` |
> | Code test | `pdep_c6_per_atom_becke_h2o_aug_cc_pvdz_vs_numpy` in `crates/ferric-rpa/tests/validation_pdep_c6.rs` |
>
> <!-- verified: id=disp.c6.kernel-vs-casida; script=scripts/rpa_c6_polarizability_sympy.py; claim=kernel-vs-casida; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/properties/per_atom.rs::closed_per_atom_alpha; rust_test=ferric-rpa/tests/validation_pdep_c6.rs::pdep_c6_per_atom_becke_h2o_aug_cc_pvdz_vs_numpy; registry=rpa-polarizability-c6 -->

The Casida check in the box also gives the pole strengths of (@eq:disp-c6-pole-sum) for
this source,
$f_n = 4\,(m^{\mathsf T}\boldsymbol\Delta^{1/2}\mathbf Z_n)(\mathbf Z_n^{\mathsf T}\boldsymbol\Delta^{1/2}\mu)$
with $\mathbf Z_n$ the eigenvectors of
$\boldsymbol\Delta^{1/2}(\boldsymbol\Delta + 4\mathbf K)\boldsymbol\Delta^{1/2}$.

### Reading the kernel

`ferric-rpa/src/properties/per_atom.rs::closed_per_atom_alpha` builds
$\tilde{\boldsymbol\varepsilon}$ once per node and solves it for the three
molecular right-hand sides before it touches any atom. The per-atom loop is
then cheap. Its body, unchanged:

<!-- excerpt: ferric-rpa/src/properties/per_atom.rs::closed_per_atom_alpha -->
```rust
for d in 0..3 {
    let w_ai = b_ov.dot(&(&atom[d] * &g));
    for j in 0..3 {
        let bare = atom[d].dot(&mu_g[j]);
        let coupled = w_ai.dot(&y_mol[j]);
        tensor[d][j] = 4.0 * bare - 16.0 * coupled;
    }
}
```

| Symbol | Code |
|---|---|
| $m^A_d$, $g$ | `atom[d]`, `g` |
| $g\circ\mu_j$ | `mu_g[j]` |
| $\tilde{\mathbf B}(g\circ m^A_d)$ | `w_ai` |
| $\tilde{\boldsymbol\varepsilon}^{-1}\tilde{\mathbf B}(g\circ\mu_j)$ | `y_mol[j]`, one solve per direction (`solve_dielectric_3`) |

Why put the atom on the bra and the molecule on the ket? The solve sits on
the ket side, so the three molecular solves serve every atom: three
$N_{\mathrm{aux}}$-sized solves per node instead of three per atom.

The partition weights sum to one, so the bras sum to
$\sum_A m^A_d = \mu_d - \sum_A R_{A,d}w_A$. By linearity of (@eq:casimir-polder-c6-3) in the
bra, the charge-transfer remainder
$\alpha_{\mathrm{CT}} = \alpha_{\mathrm{mol}} - \sum_A\alpha^A$ is
$\operatorname{sym}\langle\sum_A R_Aw_A|\mathbf R|\mu\rangle$: the response
of the field-induced charge flow between atoms (check `ct-sum-rule` of the
same script). It is what makes the `pdep` molecular $C_6$ differ from the
pair sum.

## The algorithm {#casimir-polder-c6-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | RHF; $\tilde B^P_{ia}$; the partition's $m^A_d$ and $\mu_j$ in the $ov$ space | as PDEP-RPA | `pdep_dynamic_polarizability` |
| 2 | per node $u_k$ (parallel over nodes): $\tilde{\boldsymbol\varepsilon}(iu_k)$, three solves, (@eq:disp-c6-kernel-vs-casida) for every atom | $N_{\mathrm{aux}}^2n_on_v + N_{\mathrm{aux}}^3 + N_{\mathrm{atom}}N_{\mathrm{aux}}n_on_v$ | `closed_per_atom_alpha` |
| 3 | molecular $\alpha(iu_k)$ from the lab-frame dipole | $N_{\mathrm{aux}}n_on_v$ | `ferric-rpa/src/properties.rs::molecular_dynamic_polarizability` |
| 4 | (@eq:casimir-polder-c6-1) for every pair, element-wise tensors, molecular $C_6$ | $N_{\mathrm{atom}}^2 n_u$ | `casimir_polder_c6` |

The TS source skips steps 1–3 apart from the static tensor and the volumes.
The grid is the run's RPA quadrature: 20-point Gauss–Legendre with
$u_0 = 0.5$ by default. On the single-pole model with
$\omega_{A,B} \approx 0.43\ E_h$ it reproduces the London value to
$3\times10^{-14}$ relative (`scripts/rpa_c6_polarizability_sympy.py`, check
`c6-default-grid`); the mapped rule converges geometrically on this
rational integrand.

## Options {#casimir-polder-c6-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[rpa] c6_source`, `c6_source=` | `ts`, `pdep`, `mbd` | `ts` | the source of $\alpha^A(iu)$ |
| `[rpa] c6_partition`, `c6_partition=` | `becke`, `hirshfeld` | `hirshfeld` for `pdep`, `becke` otherwise | the bra $m^A_d$; TS volume ratios are Hirshfeld whatever this says |
| `[rpa] n_quad`, `quadrature`, `u0` | section @sec:quadratures | 20, Gauss–Legendre, 0.5 | the nodes of (@eq:casimir-polder-c6-1) |

An open-shell reference with `pdep` and `hirshfeld` falls back to the Becke
partition, which has the per-spin branch. An unknown `c6_source` or
`c6_partition` string is an error.

## Numerical anchors {#casimir-polder-c6-numerical-anchors}

From `site/src/reference/validation.md` (rows "Dynamic α(iω) and
Casimir–Polder C6" and "TS C6 and MBD@TS").

| Quantity | Systems | Reference | Error |
|---|---|---|---|
| molecular $\alpha(iu)$, $C_6$ | H2O, N2 / aug-cc-pVDZ | numpy Casida sum on PySCF integrals, same orbitals | $4.9\times10^{-14}$, $6.2\times10^{-14}$ rel. |
| per-atom Becke $\alpha^A(iu)$ | same | same | $3.4\times10^{-14}$ rel. |
| TS $\bar\alpha_A$, $\omega_A$, pair and molecular $C_6$ | H2O, CO, CH3OH, CH4, benzene | pymbd 0.15.0 given the same volume ratios | $\le 7.5\times10^{-16}$ rel. |

The proofs cover the identities (@eq:disp-c6-pole-sum)–(@eq:disp-c6-kernel-vs-casida) on random models. The
free-atom tables, the partitions and the volume ratios are checked against
the references in this table, not by a proof script.

## Known limits (2026-11-15)

- The Hirshfeld per-atom path is closed-shell only (open shells fall back to
  Becke).
- TS and MBD cover $Z \le 54$.
- These coefficients feed exports and analysis; no dispersion energy in
  ferric sums $-C_6^{AB}/R^6$ from them.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-rpa/theory.md`, "Dispersion: Casimir-Polder C6" | (@eq:casimir-polder-c6-1)–(@eq:disp-c6-kernel-vs-casida) |
| `crates/ferric-rpa/src/dispersion.rs`, `crates/ferric-rpa/src/properties/per_atom.rs`, `crates/ferric-rpa/src/dispersion/free_atom_ref.rs`, `crates/ferric-rpa/src/property_bundle.rs` | the loops, the free-atom data, the TS volume ratios |
| `scripts/rpa_c6_polarizability_sympy.py` | verified boxes |
| book section @sec:polarizability-pdep-c6 | the exported arrays and their conventions |

Changes from the sources: none to notation; theory.md and this section both
write the imaginary frequency $u$ and the orbital-energy gap $\Delta_i^a$.
theory.md writes the TS frequency as $u_A$; this section writes $\omega_A$,
as the code does.
