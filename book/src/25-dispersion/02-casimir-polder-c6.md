# 25.2 Casimir–Polder $C_6$ from dynamic polarizabilities

<!-- section: 25.2; status: ready; registry: rpa-polarizability-c6;
     theory: crates/ferric-rpa/theory.md#"Dispersion: Casimir-Polder C6" -->

Two neutral fragments $A$ and $B$ far apart attract as $-C_6^{AB}/R^6$, and
$C_6^{AB}$ is an integral over imaginary frequency of their two dynamic
polarizabilities. ferric evaluates that integral on the RPA frequency grid for
per-atom polarizabilities from three sources: the Tkatchenko–Scheffler single
pole, the direct-RPA response itself (PDEP-RPA), and the full-range
self-consistently screened TS model of section 25.3. The coefficients are
exported as data (section 27.3); no energy in ferric sums them.

| Symbol | Meaning | Code name |
|---|---|---|
| $A, B$ | atoms | `a`, `b` |
| $u_k, w_k$ | frequency node and weight (section 18.2) | `freqs`, `weights` |
| $\alpha^A_{ij}(iu)$ | per-atom polarizability tensor | `DynamicPolarizability::per_atom[a][k]` |
| $\bar\alpha = \tfrac13\operatorname{tr}\alpha$ | isotropic average | `iso[a][k]` |
| $g_{ia}(u)$ | $\Delta_i^a/(u^2 + (\Delta_i^a)^2)$ | `Frequency::weights` |

| Entry point | Location |
|---|---|
| Rust | `ferric-rpa/src/dispersion.rs::{casimir_polder_c6,ts_dynamic_polarizability,pdep_dynamic_polarizability}`, `ferric-rpa/src/dispersion/mbd.rs::mbd_dynamic_polarizability` |
| CLI | `method.kind = "pdep-rpa"` with `[rpa] export_npz`, `compute_c6`, `c6_source = "ts" \| "pdep" \| "mbd"`, `c6_partition` |
| Python | `run_pdep_rpa(..., properties=True, c6_source=..., c6_partition=...)`; arrays `c6_iso`, `c6_aniso`, `c6_molecular_iso` on the result |

## 25.2.1 The Casimir–Polder integral

$$ C_6^{AB} = \frac{3}{\pi}\int_0^\infty \bar\alpha^A(iu)\,\bar\alpha^B(iu)\,du \approx \frac{3}{\pi}\sum_k w_k\,\bar\alpha^A(iu_k)\,\bar\alpha^B(iu_k) . \tag{25.8} $$

`casimir_polder_c6` also forms an element-wise
$C^{AB}_{6,ij} = \frac3\pi\sum_k w_k\,\alpha^A_{ij}(iu_k)\,\alpha^B_{ij}(iu_k)$
(not a tensor contraction; see section 27.3) and the molecular
$C_{6,\mathrm{mol}}$ from the molecular $\bar\alpha(iu)$, which is not the sum of
the pair coefficients once charge transfer between atoms is present.

For a polarizability with poles,
$\alpha(iu) = \sum_n f_n/(\Omega_n^2 + u^2)$, (25.8) is a closed double sum,

$$ C_6^{AB} = \frac32\sum_{mn}\frac{f^A_m f^B_n}{\Omega_m\Omega_n(\Omega_m + \Omega_n)}, \tag{25.9} $$

and for one pole per atom it is the London formula
$\tfrac32\bar\alpha_A\bar\alpha_B\omega_A\omega_B/(\omega_A + \omega_B)$.

> **Verified equation 25.9** · the pole double sum
>
> $C_6$ from the coded kernel (25.11) on a 200-point grid equals the analytic
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

## 25.2.2 Three sources of $\alpha^A(iu)$

| `c6_source` | $\alpha^A(iu)$ | Function |
|---|---|---|
| `ts` (default) | single pole $\bar\alpha_A/(1 + (u/\omega_A)^2)$ with $\bar\alpha_A = r_A\alpha^{\mathrm{free}}$, $C_{6,A} = r_A^2C_6^{\mathrm{free}}$, $\omega_A = \tfrac43C_{6,A}/\bar\alpha_A^2$; the static per-atom tensor supplies only the shape (27.11) | `ts_dynamic_polarizability`, `mbd::ts_atom_params` |
| `pdep` | the direct-RPA per-atom response (25.10) at each node | `pdep_dynamic_polarizability` |
| `mbd` | the TS poles screened by full-range coupled dipoles (section 25.3.5) | `mbd_dynamic_polarizability` |

$r_A$ is the Hirshfeld volume ratio $V_A/V_A^{\mathrm{free}}$, with
$V_A^{\mathrm{free}}$ from a live free-atom SCF on the same quadrature. The
free-atom $\alpha^{\mathrm{free}}$ and $C_6^{\mathrm{free}}$ are Tkatchenko and
Scheffler (PRL 102, 073005 (2009)) for $Z \le 18$ and Gould and Bučko (JCTC
12, 3603 (2016)) for $Z = 19$–$54$; any other element is an error, never a
substituted value.

## 25.2.3 The direct-RPA per-atom polarizability

The per-atom tensor is the Krishtal–Senet–Van Alsenoy intrinsic
polarizability (JCP 125, 034312 (2006)): the bra is the atom-weighted dipole
$m^A_d = w_A(\mathbf r)(r - R_A)_d$ of a Becke or Hirshfeld partition, the ket
the molecular dipole $\mu_j$:

$$ \alpha^A_{dj}(iu) = \langle m^A_d|\,\mathbf R(iu)\,|\mu_j\rangle, \qquad \mathbf R = \big(\tfrac14\mathbf G^{-1} + \mathbf K\big)^{-1}, \quad \mathbf G = \operatorname{diag}(g_{ia}), \quad \mathbf K = \tilde{\mathbf B}^{\mathsf T}\tilde{\mathbf B}, \tag{25.10} $$

with $\tilde B^P_{ia}$ the dressed RI factor (section 6.1) and the vectors
$m^A_{d,ia} = \langle i|m^A_d|a\rangle$, $\mu_{j,ia} = \langle i|r_j|a\rangle$.
Woodbury turns the $n_on_v$-sized inverse into an auxiliary-basis solve,

$$ \alpha^A_{dj}(iu) = 4\,m^A_d\cdot(g\circ\mu_j) - 16\,\big(\tilde{\mathbf B}(g\circ m^A_d)\big)^{\mathsf T}\,\tilde{\boldsymbol\varepsilon}^{-1}\,\big(\tilde{\mathbf B}(g\circ\mu_j)\big), \qquad \tilde{\boldsymbol\varepsilon} = \mathbf 1 + 4\tilde{\mathbf B}\mathbf G\tilde{\mathbf B}^{\mathsf T}, \tag{25.11} $$

the same dielectric matrix as the dRPA energy (18.3). At $u = 0$,
$g_{ia} = 1/\Delta_i^a$ gives the static per-atom tensor. The code then
symmetrizes each tensor.

> **Verified equation 25.11** · the aux-basis kernel is the dRPA response
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

The inner loop, quoted from `ferric-rpa/src/properties/per_atom.rs::closed_per_atom_alpha` (not a checked excerpt):
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

Because the partition weights sum to one, the bras sum to
$\mu_d - \sum_A R_{A,d}w_A$, so the charge-transfer remainder
$\alpha_{\mathrm{CT}} = \alpha_{\mathrm{mol}} - \sum_A\alpha^A$ is
$\operatorname{sym}\langle\sum_A R_Aw_A|\mathbf R|\mu\rangle$, the response of
the field-induced charge flow between atoms (check `ct-sum-rule` of the same
script).

## 25.2.4 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | RHF; $\tilde B^P_{ia}$; the partition's $m^A_d$ and $\mu_j$ in the $ov$ space | as PDEP-RPA | `pdep_dynamic_polarizability` |
| 2 | per node $u_k$ (parallel over nodes): $\tilde{\boldsymbol\varepsilon}(iu_k)$, three solves, (25.11) for every atom | $N_{\mathrm{aux}}^2n_on_v + N_{\mathrm{aux}}^3 + N_{\mathrm{atom}}N_{\mathrm{aux}}n_on_v$ | `closed_per_atom_alpha` |
| 3 | molecular $\alpha(iu_k)$ from the lab-frame dipole | $N_{\mathrm{aux}}n_on_v$ | `ferric-rpa/src/properties.rs::molecular_dynamic_polarizability` |
| 4 | (25.8) for every pair, element-wise tensors, molecular $C_6$ | $N_{\mathrm{atom}}^2 n_u$ | `casimir_polder_c6` |

The TS source skips steps 1–3 apart from the static tensor and the volumes.
The grid is the run's RPA quadrature: 20-point Gauss–Legendre with
$u_0 = 0.5$ by default, which reproduces the London single-pole value to
$3\times10^{-14}$ relative (`scripts/rpa_c6_polarizability_sympy.py`, check
`c6-default-grid`).

## 25.2.5 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[rpa] c6_source`, `c6_source=` | `ts`, `pdep`, `mbd` | `ts` | the source of $\alpha^A(iu)$ |
| `[rpa] c6_partition`, `c6_partition=` | `becke`, `hirshfeld` | `hirshfeld` for `pdep`, `becke` otherwise | the bra $m^A_d$; TS volume ratios are Hirshfeld whatever this says |
| `[rpa] n_quad`, `quadrature`, `u0` | section 18.2 | 20, Gauss–Legendre, 0.5 | the nodes of (25.8) |

An open-shell reference with `pdep` and `hirshfeld` falls back to the Becke
partition, which has the per-spin branch.

## 25.2.6 Numerical anchors

From `site/src/reference/validation.md` (rows "Dynamic α(iω) and
Casimir–Polder C6" and "TS C6 and MBD@TS").

| Quantity | Systems | Reference | Error |
|---|---|---|---|
| molecular $\alpha(iu)$, $C_6$ | H2O, N2 / aug-cc-pVDZ | numpy Casida sum on PySCF integrals, same orbitals | $4.9\times10^{-14}$, $6.2\times10^{-14}$ rel. |
| per-atom Becke $\alpha^A(iu)$ | same | same | $3.4\times10^{-14}$ rel. |
| TS $\bar\alpha_A$, $\omega_A$, pair and molecular $C_6$ | H2O, CO, CH3OH, CH4, benzene | pymbd 0.15.0 given the same volume ratios | $\le 7.5\times10^{-16}$ rel. |

## Known limits (2026-11-15)

- The Hirshfeld per-atom path is closed-shell only (open shells fall back to
  Becke).
- TS and MBD cover $Z \le 54$.
- These coefficients feed exports and analysis; no dispersion energy in
  ferric sums $-C_6^{AB}/R^6$ from them.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-rpa/theory.md`, "Dispersion: Casimir-Polder C6" | (25.8)–(25.11) |
| `crates/ferric-rpa/src/dispersion.rs`, `crates/ferric-rpa/src/properties/per_atom.rs` | the loops |
| `scripts/rpa_c6_polarizability_sympy.py` | verified boxes |
| book section 27.3 | the exported arrays and their conventions |

Changes from the sources: theory.md writes the imaginary frequency as
$\omega$ and the orbital-energy gap as $e_{ia}$; this section writes $u$ and
$\Delta_i^a$.
