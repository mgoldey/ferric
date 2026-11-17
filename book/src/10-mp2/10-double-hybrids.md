# 10.10 MP2-based double hybrids

<!-- section: 10.10; status: ready; registry: double-hybrid;
     theory: crates/ferric-mp2/theory.md#"16. Double hybrids" -->

A semilocal correlation functional sees the density and its gradient at
one point at a time, so it has no term for two electrons correlating
across a gap, which is what dispersion is. MP2 has such a term: its pair
amplitudes couple any two occupied orbitals through the virtual space. A
double hybrid is a hybrid density functional with a scaled MP2 term
evaluated on its own Kohn–Sham orbitals, and it turns down the semilocal
correlation to make room for it. The MP2 part is section 10.3's
spin-component sum; what is new is the reference.

The plan: the energy and its parameters (10.10.1), what is anchored
(10.10.2), and the derivation of the two MP2 terms on a Kohn–Sham
determinant (10.10.3).

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/double_hybrid.rs::mp2_double_hybrid` (RKS), `ferric-mp2/src/double_hybrid.rs::u_mp2_double_hybrid` (UKS), `DoubleHybridKind` |
| CLI | `method.kind = "b2plyp"` (`examples/water-b2plyp.toml`, `examples/nh2-b2plyp.toml`), `method.kind = "dsd-pbep86"` (`examples/water-dsd-pbep86.toml`, `examples/nh2-dsd-pbep86-d3bj.toml`); `[dft] dispersion = "d3bj"` |
| Python | `run_double_hybrid(kind="b2plyp")`, `run_double_hybrid(kind="dsd-pbep86")`, with `dispersion="d3bj"` and `unrestricted=` |

## 10.10.1 The energy

$$ E_{\mathrm{DH}} = E_{\mathrm{KS}}\big[a_x E_x^{\mathrm{HF}} + (1 - a_x) E_x^{\mathrm{DFA}} + a_c\,E_c^{\mathrm{DFA}}\big] + c_{\mathrm{OS}}\,E_{\mathrm{OS}} + c_{\mathrm{SS}}\,E_{\mathrm{SS}} . \tag{10.25} $$

Read it in two parts. $E_{\mathrm{KS}}$ is the converged energy of a hybrid
functional with exact-exchange fraction $a_x$, semilocal exchange fraction
$b_x = 1 - a_x$ and semilocal correlation scaled by $a_c$. Written out for
a closed-shell density $D$,

$$ E_{\mathrm{KS}} = E_{nn} + \operatorname{Tr}[Dh] + \tfrac12\operatorname{Tr}\big[DJ[D]\big] - \tfrac{a_x}{4}\operatorname{Tr}\big[DK[D]\big] + b_x E_x^{\mathrm{DFA}}[\rho] + a_c E_c^{\mathrm{DFA}}[\rho] . $$

$E_{\mathrm{OS}}$ and $E_{\mathrm{SS}}$ are the RI-MP2 components (10.2)
computed with the Kohn–Sham orbitals and orbital energies in place of
Hartree–Fock ones, with the RI integrals of section 10.1 and the frozen
core of the input. The SCF never sees the MP2 term: the orbitals are those
of $E_{\mathrm{KS}}$ alone, and the MP2 term is added afterwards.

The two published sets in ferric:

| Kind | Exchange | Semilocal correlation | $c_{\mathrm{OS}}$ | $c_{\mathrm{SS}}$ |
|---|---|---|---|---|
| B2PLYP (Grimme, J. Chem. Phys. 124, 034108 (2006)) | 0.53 HF + 0.47 B88 | 0.73 LYP | 0.27 | 0.27 |
| DSD-PBEP86, no dispersion (Kozuch and Martin, J. Comput. Chem. 34, 2327 (2013)) | 0.72 HF + 0.28 PBE | 0.44 P86 | 0.51 | 0.36 |

Both have $a_x + b_x = 1$: the exchange fractions add to one. B2PLYP
also has $a_c = 1 - c_{\mathrm{OS}}$ (0.73 + 0.27 = 1), so its semilocal and
MP2 correlation fractions add to one as well, with a single MP2 scale for
both spin channels. DSD-PBEP86 has no such link: its two spin scales differ, and
$a_c = 0.44$ is a separate parameter.

The exchange fraction $a_x$ is the `XcDef` field `b3lyp_mix`; the DFA
exchange and correlation fractions are its per-component `weights`, which
multiply the whole libxc exchange and correlation components, and
$c_{\mathrm{OS}}, c_{\mathrm{SS}}$ come from `DoubleHybridKind::mp2_config`:

<!-- excerpt: ferric-dft/src/libxc.rs::b2plyp_def -->
```rust
b3lyp_mix: Some(0.53),
weights: Some(vec![0.47, 0.73]),
```

> **Verified table** · double-hybrid parameters
>
> | | |
> |---|---|
> | Claim | the fractions parsed from the Rust sources equal the cited B2PLYP and no-dispersion DSD-PBEP86 values; $a_x + b_x = 1$ for both; B2PLYP's $a_c = 1 - c_{\mathrm{OS}}$ |
> | Proof | claim P of `scripts/double_hybrid_sympy.py` |
> | Code | `ferric-dft/src/libxc.rs::{b2plyp_def,dsd_pbep86_def}`, `ferric-mp2/src/double_hybrid.rs::{mp2_config,DoubleHybridKind}` |
> | Code test | `dsd_pbep86_h2o_cc_pvdz_vs_pyscf` in `crates/ferric-mp2/tests/validation_dsd_pbep86_pyscf.rs` |
>
> <!-- verified: id=dh.parameters; script=scripts/double_hybrid_sympy.py; claim=P; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/libxc.rs::{b2plyp_def,dsd_pbep86_def}; rust_test=ferric-mp2/tests/validation_dsd_pbep86_pyscf.rs::dsd_pbep86_h2o_cc_pvdz_vs_pyscf; registry=double-hybrid -->

### The D3(BJ) sets

`dispersion = "d3bj"` (CLI `[dft]`, Python `dispersion=`) adds the damped
two-body dispersion sum of Grimme, Ehrlich and Goerigk (J. Comput. Chem.
32, 1456 (2011)),

$$ E_{\mathrm{D3(BJ)}} = -\sum_{A<B}\Big[ s_6\frac{C_6^{AB}}{R_{AB}^6 + f_{AB}^6} + s_8\frac{C_8^{AB}}{R_{AB}^8 + f_{AB}^8}\Big],\qquad f_{AB} = a_1\sqrt{C_8^{AB}/C_6^{AB}} + a_2 . $$

It depends on the geometry only, not on the orbitals, so it adds to the
total and does not enter the SCF. Each damping row was fitted together
with a set of functional coefficients and travels with it:

| Set | $a_x$ | $b_x$ | $a_c$ | $c_{\mathrm{OS}}$ | $c_{\mathrm{SS}}$ | $(s_6, s_8, a_1, a_2)$ |
|---|---|---|---|---|---|---|
| B2PLYP-D3(BJ) (Goerigk and Grimme, Phys. Chem. Chem. Phys. 13, 6670 (2011)) | 0.53 | 0.47 B88 | 0.73 LYP | 0.27 | 0.27 | (0.64, 0.9147, 0.3065, 5.0570) |
| DSD-PBEP86-D3(BJ) (Kozuch and Martin 2013, the D3(BJ) refit) | 0.69 | 0.31 PBE | 0.44 P86 | 0.52 | 0.22 | (0.48, 0, 0, 5.6) |

B2PLYP-D3(BJ) keeps the B2PLYP coefficients unchanged. DSD-PBEP86 was
refitted with its dispersion term, so `dispersion = "d3bj"` on
`dsd-pbep86` selects the refit set as a whole; adding the D3(BJ) row to
the no-dispersion coefficients would mix two fits. With $a_1 = 0$ and
$s_8 = 0$ each DSD-PBEP86 pair term is $-s_6 C_6/(R^6 + a_2^6)$. The rows
are in `ferric-d3/src/params.rs` (`b2plyp`, `dsdpbep86`) and equal
simple-dftd3's rational-damping rows (`crates/ferric-d3/testdata/d3bj_params_snapshot.json`).
The 2011 DSD-PBEP86-D2 set is not offered.

### The code

The MP2 step is a few lines on top of RI-MP2. This is
`mp2_double_hybrid` with the `RiMp2Config` construction and the result
struct trimmed:

```rust
if !ks.converged {
    return Err(FerricError::ScfConvergence {                 // unconverged KS: refused
        iterations: ks.iterations,
        last_energy: ks.energy,
    });
}
let op = Operator::coulomb();                                // full 1/r, no attenuation
// ... ri_cfg carries frozen_core and the memory budget ...
let (sc, _b_ov) = ri_mp2_spin_components(mol, obs, dfbs, op, ks, &ri_cfg)?;   // E_OS, E_SS on KS orbitals
let e_corr_scaled = cfg.c_os * sc.e_os + cfg.c_ss * sc.e_ss; // c_OS E_OS + c_SS E_SS
// total_energy: ks.energy + e_corr_scaled                   // equation 10.25
```

- **Nothing in the MP2 code knows about DFT.** `ri_mp2_spin_components` is
  the routine of section 10.1, handed a Kohn–Sham `ScfResult`. Its orbital
  energies are the Kohn–Sham eigenvalues.
- **`ks.energy` already is $E_{\mathrm{KS}}$.** The semilocal and exact
  exchange fractions were applied inside the SCF; the driver adds only the
  scaled PT2 term (and, when requested, the D3(BJ) energy).
- **An unconverged reference is an error.** The CLI kinds converge their
  own KS reference before the MP2 step, and `mp2_double_hybrid` refuses an
  unconverged one.
- **`u_mp2_double_hybrid` is the same with `u_ri_mp2`.** It takes
  $E_{\mathrm{OS}} = E_{\alpha\beta}$ and
  $E_{\mathrm{SS}} = E_{\alpha\alpha} + E_{\beta\beta}$.

The CLI and Python converge a closed-shell molecule through the KS
convergence ladder, and a molecule with `multiplicity > 1` (or Python
`unrestricted=True`) as a UKS. Both use RI-J and RI-K on
def2-universal-jkfit unless the input names another fit.

The KS orbitals are not Hartree–Fock orbitals, so the Brillouin condition
fails and the singles term of MP2 is not zero. Standard double hybrids,
B2PLYP included, drop it; so does ferric (10.10.3).

## 10.10.2 What is anchored

| Kind | Grade | Evidence |
|---|---|---|
| `b2plyp` | Proven | total energy vs PySCF built from the definition (`dft.RKS` with 0.53 HF + 0.47 B88, 0.73 LYP, then `mp.dfmp2.DFMP2` on the KS orbitals): ≤ 1.3e-11 Ha; E_KS ≤ 7.1e-13, E_OS ≤ 4.1e-11, E_SS ≤ 1.5e-11 Ha (H2O, NH3, CH4 / cc-pVDZ and def2-SVP, frozen core 0 and 1); test `crates/ferric-mp2/tests/validation_b2plyp_pyscf.rs` |
| `dsd-pbep86` | Proven | total energy vs PySCF built from the definition (`dft.RKS` with 0.72 HF + 0.28 PBE, 0.44 P86, then `mp.dfmp2.DFMP2` scaled 0.51/0.36): ≤ 1.9e-11 Ha; E_KS ≤ 6.5e-13, E_OS ≤ 2.2e-11, E_SS ≤ 2.3e-11 Ha (same cases as B2PLYP); test `crates/ferric-mp2/tests/validation_dsd_pbep86_pyscf.rs`, which also asserts that the superseded 0.69/0.31/0.56/0.29 set misses by at least 100 bars |
| UKS reference and the D3(BJ) sets | in the `dsd-pbep86` row | B2PLYP, DSD-PBEP86 and DSD-PBEP86-D3(BJ) on a UKS reference, plus B2PLYP-D3(BJ), vs PySCF `dft.UKS` + `mp.dfump2.DFUMP2` and simple-dftd3: total ≤ 3.2e-11 Ha, E_KS ≤ 3.1e-13, E_OS ≤ 7.6e-11, E_SS ≤ 1.8e-11, D3(BJ) ≤ 3.6e-16 Ha (NH2, CH3 doublets / cc-pVDZ and def2-SVP, frozen core 0 and 1); a singlet through the UKS route equals the RKS route to ≤ 9.9e-14 Ha; test `crates/ferric-mp2/tests/validation_udh_pyscf.rs` |

Both closed-shell references run a flat (75, 110) grid with RI-J in f64.
The CLI default (pruned grid, f32 RI-J) moves the total by up to 1.0e-6 Ha
(B2PLYP) and 1.2e-6 Ha (DSD-PBEP86), so a default run carries no anchor and
the CLI prints a notice. Only Grimme's 2006 B2PLYP parameters, the 2013
no-dispersion DSD-PBEP86 set and the D3(BJ) sets above are checked; no
comparison of B2PLYP with experiment is claimed. Figures from
`site/src/reference/validation.md`.

## 10.10.3 Derivation

Which two sums are $E_{\mathrm{OS}}$ and $E_{\mathrm{SS}}$ on a Kohn–Sham
determinant, and what is left out?

Start with the SCF. It minimizes $E_{\mathrm{KS}}$ alone. Differentiate
$E_{\mathrm{KS}}$ with respect to $D$: the Coulomb term gives $J[D]$, the
exact-exchange term $-\tfrac{a_x}{4}\operatorname{Tr}[DK[D]]$ gives
$-\tfrac{a_x}{2}K[D]$ (it is quadratic in $D$ and $K$ is linear), and the
semilocal terms give their potentials. The Fock matrix is

$$ F = h + J[D] - \tfrac{a_x}{2}K[D] + b_x V_x + a_c V_c . $$

$E_{\mathrm{OS}}$ and $E_{\mathrm{SS}}$ are the closed-shell sums as
`pair_energy` codes them, with the code's negative denominator and no
leading minus sign,

$$ E_{\mathrm{OS}} = \sum_{ijab}\frac{(ia|jb)^2}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}, \qquad
E_{\mathrm{SS}} = \sum_{ijab}\frac{(ia|jb)\big[(ia|jb) - (ib|ja)\big]}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}, $$

on the KS orbitals and KS orbital energies. Where do they come from? Use
Møller–Plesset partitioning on the KS determinant,
$H_0 = \sum_p \varepsilon_p\,a_p^\dagger a_p$. The second-order energy is
$\sum_{D\neq0}|\langle D|H|0\rangle|^2/(E^{(0)}_0 - E^{(0)}_D)$ over singly
and doubly excited determinants. Split the doubles by the spins of the two
holes: the $\alpha\beta$ doubles sum to $E_{\mathrm{OS}}$ and the
$\alpha\alpha$ and $\beta\beta$ doubles to $E_{\mathrm{SS}}$, as in
section 10.1.

The singles $\langle\Phi_i^a|H|0\rangle$ do not vanish on KS orbitals,
because the KS orbitals do not diagonalize the Hartree–Fock Fock operator.
The code, like the published functionals, keeps only the doubles.

Check the plain-MP2 limit
$(a_x, b_x, a_c, c_{\mathrm{OS}}, c_{\mathrm{SS}}) = (1, 0, 0, 1, 1)$. The
Fock matrix becomes $h + J - \tfrac12 K$, so the SCF is RHF; Brillouin's
theorem removes the singles, and (10.25) is
$E_{\mathrm{HF}} + E_{\mathrm{MP2}}$.

For a UKS reference with spin densities $D^\sigma$ and
$D = D^\alpha + D^\beta$, the exact-exchange term is
$-\tfrac{a_x}{2}\sum_\sigma\operatorname{Tr}[D^\sigma K[D^\sigma]]$, the
Fock matrix of spin $\sigma$ is
$F^\sigma = h + J[D] - a_xK[D^\sigma] + b_xV_x^\sigma + a_cV_c^\sigma$, and
the PT2 term is split by the spins of the two holes:
$E_{\mathrm{OS}} = E_{\alpha\beta}$ and
$E_{\mathrm{SS}} = E_{\alpha\alpha} + E_{\beta\beta}$ from `u_ri_mp2`. Two
checks fix the convention. For a closed shell
$E_{\alpha\alpha} = E_{\beta\beta}$ is half the closed-shell $E_{\mathrm{SS}}$
and $E_{\alpha\beta}$ is the closed-shell $E_{\mathrm{OS}}$, so a singlet
through the UKS route equals the RKS route. And the plain-MP2 limit is UHF,
where $E_{\mathrm{DH}} = E_{\mathrm{UHF}} + E_{\mathrm{UMP2}}$. Singles stay
out, as in the closed-shell definition.

> **Verified equation 10.25** · double-hybrid assembly
>
> | | |
> |---|---|
> | Claim | on a model with a semilocal functional on a grid, the plain-MP2 limit of (10.25) equals $\langle0|H|0\rangle + E^{(2)}$ with every singles element zero; with the B2PLYP and DSD-PBEP86 fractions, $E_{\mathrm{OS}}$ and $E_{\mathrm{SS}}$ on the KS orbitals equal the αβ and same-spin doubles sums of a determinant-space route (second quantization on bit strings), and the singles are not zero |
> | Proof | claim E of `scripts/double_hybrid_sympy.py` |
> | Code | `ferric-mp2/src/double_hybrid.rs::mp2_double_hybrid` |
> | Code test | `b2plyp_h2o_cc_pvdz_vs_pyscf` in `crates/ferric-mp2/tests/validation_b2plyp_pyscf.rs` |
>
> <!-- verified: id=dh.assembly; script=scripts/double_hybrid_sympy.py; claim=E; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/double_hybrid.rs::mp2_double_hybrid; rust_test=ferric-mp2/tests/validation_b2plyp_pyscf.rs::b2plyp_h2o_cc_pvdz_vs_pyscf; registry=double-hybrid -->

The recorded mutations of the proof (a perturbed cited parameter, a dropped
exchange term in $E_{\mathrm{SS}}$, swapped $c_{\mathrm{OS}}$ and
$c_{\mathrm{SS}}$, a doubled exact-exchange term in the Fock matrix) are in
`crates/ferric-python/tests/test_derivation_scripts.py`; each fails an
assertion. The same script's claim U checks the unrestricted assembly
(a model with 3 α and 2 β electrons: the UHF limit, $E_{\mathrm{OS}}$ and
$E_{\mathrm{SS}}$ for each set, the singlet UKS against the RKS route, and
invariance under relabelling α and β), and claim D checks that $a_1 = 0$,
$s_8 = 0$ reduce the D3(BJ) pair energy to $-s_6C_6/(R^6+a_2^6)$ and that
the `dftd3` package's rows equal the cited parameters.

## Known limits (2026-11-15)

- No gradient or Hessian: the MP2 term is not differentiated (the KS
  Hessian refuses double hybrids with a typed error).
- A ROKS reference is not routed: the CLI and Python run a UKS for an open
  shell.
- The 2011 DSD-PBEP86-D2 set is not offered.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/double_hybrid.rs` | parameters, reference check, the RKS and UKS drivers |
| `site/src/reference/validation.md`, rows `b2plyp` and `dsd-pbep86` and the UKS / D3(BJ) row | grades, anchors |
| `crates/ferric-mp2/theory.md`, "16. Double hybrids" | the energy, the parameter tables, the D3(BJ) term, the unrestricted assembly, the plain-MP2 limit |
| `scripts/double_hybrid_sympy.py` | claims P, E, U and D |
