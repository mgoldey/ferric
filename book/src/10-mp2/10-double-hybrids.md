# 10.10 MP2-based double hybrids

<!-- section: 10.10; status: ready; registry: double-hybrid;
     theory: crates/ferric-mp2/theory.md#"16. Double hybrids" -->

A double hybrid is a hybrid density functional with a scaled MP2 term
evaluated on its own Kohn–Sham orbitals. The MP2 part is section 10.3's
spin-component sum; what is new is the reference.

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/double_hybrid.rs::mp2_double_hybrid`, `DoubleHybridKind` |
| CLI | `method.kind = "b2plyp"` (`examples/water-b2plyp.toml`), `method.kind = "dsd-pbep86"` |
| Python | `run_double_hybrid(kind="b2plyp")`, `run_double_hybrid(kind="dsd-pbep86")` |

## 10.10.1 The energy

$$ E_{\mathrm{DH}} = E_{\mathrm{KS}}\big[a_x E_x^{\mathrm{HF}} + (1 - a_x) E_x^{\mathrm{DFA}} + a_c\,E_c^{\mathrm{DFA}}\big] + c_{\mathrm{OS}}\,E_{\mathrm{OS}} + c_{\mathrm{SS}}\,E_{\mathrm{SS}} . \tag{10.25} $$

$E_{\mathrm{KS}}$ is the converged energy of a hybrid functional with
exact-exchange fraction $a_x$ and semilocal correlation scaled by $a_c$.
For B2PLYP $a_c = 1 - c_{\mathrm{OS}}$; DSD-PBEP86 has no such link.
$E_{\mathrm{OS}}$ and $E_{\mathrm{SS}}$ are the RI-MP2 components (10.2)
computed with the Kohn–Sham orbitals and orbital energies in place of
Hartree–Fock ones. The SCF never sees the MP2 term.

| Kind | Exchange | Semilocal correlation | $c_{\mathrm{OS}}$ | $c_{\mathrm{SS}}$ |
|---|---|---|---|---|
| B2PLYP (Grimme, J. Chem. Phys. 124, 034108 (2006)) | 0.53 HF + 0.47 B88 | 0.73 LYP | 0.27 | 0.27 |
| DSD-PBEP86, no dispersion (Kozuch and Martin, J. Comput. Chem. 34, 2327 (2013)) | 0.72 HF + 0.28 PBE | 0.44 P86 | 0.51 | 0.36 |

The exchange fraction $a_x$ is the `XcDef` field `b3lyp_mix`; the DFA
exchange and correlation fractions are its per-component `weights`, and
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

The KS orbitals are not Hartree–Fock orbitals, so the Brillouin condition
fails and the singles term of MP2 is not zero. Standard double hybrids,
B2PLYP included, drop it; so does ferric.

The CLI kinds converge their own KS reference before the MP2 step, and
`mp2_double_hybrid` refuses an unconverged reference.

## 10.10.2 What is anchored

| Kind | Grade | Evidence |
|---|---|---|
| `b2plyp` | Proven | total energy vs PySCF built from the definition (`dft.RKS` with 0.53 HF + 0.47 B88, 0.73 LYP, then `mp.dfmp2.DFMP2` on the KS orbitals): ≤ 1.3e-11 Ha; E_KS ≤ 7.1e-13, E_OS ≤ 4.1e-11, E_SS ≤ 1.5e-11 Ha (H2O, NH3, CH4 / cc-pVDZ and def2-SVP, frozen core 0 and 1); test `crates/ferric-mp2/tests/validation_b2plyp_pyscf.rs` |
| `dsd-pbep86` | Proven | total energy vs PySCF built from the definition (`dft.RKS` with 0.72 HF + 0.28 PBE, 0.44 P86, then `mp.dfmp2.DFMP2` scaled 0.51/0.36): ≤ 1.9e-11 Ha; E_KS ≤ 6.5e-13, E_OS ≤ 2.2e-11, E_SS ≤ 2.3e-11 Ha (same cases as B2PLYP); test `crates/ferric-mp2/tests/validation_dsd_pbep86_pyscf.rs`, which also asserts that the superseded 0.69/0.31/0.56/0.29 set misses by at least 100 bars |

Both references run a flat (75, 110) grid with RI-J in f64. The CLI
default (pruned grid, f32 RI-J) moves the total by up to 1.0e-6 Ha (B2PLYP)
and 1.2e-6 Ha (DSD-PBEP86), so a default run carries no anchor and the CLI
prints a notice. Only Grimme's 2006 B2PLYP parameters and the 2013
no-dispersion DSD-PBEP86 set are checked; the -D3BJ variant is not offered; no comparison of B2PLYP with experiment is
claimed. Figures from `site/src/reference/validation.md`.

## 10.10.3 Derivation

The SCF minimizes $E_{\mathrm{KS}}$ alone, with Fock matrix
$F = h + J[D] - \tfrac{a_x}{2}K[D] + b_x V_x + a_c V_c$. $E_{\mathrm{OS}}$
and $E_{\mathrm{SS}}$ are the closed-shell sums as `pair_energy` codes them,

$$ E_{\mathrm{OS}} = \sum_{ijab}\frac{(ia|jb)^2}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}, \qquad
E_{\mathrm{SS}} = \sum_{ijab}\frac{(ia|jb)\big[(ia|jb) - (ib|ja)\big]}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}, $$

on the KS orbitals and KS orbital energies. In Møller–Plesset partitioning
on the KS determinant, $H_0 = \sum_p \varepsilon_p\,a_p^\dagger a_p$, these are
the αβ and the same-spin (αα + ββ) parts of the second-order doubles sum.
The singles $\langle\Phi_i^a|H|0\rangle$ do not vanish on KS orbitals; the
code, like the published functionals, keeps only the doubles. In the
plain-MP2 limit $(a_x, b_x, a_c, c_{\mathrm{OS}}, c_{\mathrm{SS}}) = (1, 0, 0, 1, 1)$
the SCF is RHF, Brillouin's theorem removes the singles, and (10.25) is
$E_{\mathrm{HF}} + E_{\mathrm{MP2}}$.

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
assertion.

## Known limits (2026-11-15)

- Closed-shell (RKS) only.
- No gradient or Hessian: the MP2 term is not differentiated (the KS
  Hessian refuses double hybrids with a typed error).
- DSD-PBEP86 is the no-dispersion 2013 set only; DSD-PBEP86-D3BJ (0.69/0.31/0.44, 0.52/0.22 plus D3BJ) is not offered.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/double_hybrid.rs` | parameters, reference check |
| `site/src/reference/validation.md`, rows `b2plyp` and `dsd-pbep86` | grades, anchors |
| `crates/ferric-mp2/theory.md`, "16. Double hybrids" | the energy, the parameter table, the plain-MP2 limit |
| `scripts/double_hybrid_sympy.py` | claims P and E |
