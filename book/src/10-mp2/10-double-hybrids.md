# 10.10 MP2-based double hybrids

<!-- section: 10.10; status: edit; registry: double-hybrid;
     theory: crates/ferric-mp2/theory.md#"4. Spin-component scaling" -->

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
| DSD-PBEP86 | 0.69 HF + 0.31 PBE | 0.44 P86 | 0.56 | 0.29 |

The KS orbitals are not Hartree–Fock orbitals, so the Brillouin condition
fails and the singles term of MP2 is not zero. Standard double hybrids,
B2PLYP included, drop it; so does ferric.

The CLI kinds converge their own KS reference before the MP2 step, and
`mp2_double_hybrid` refuses an unconverged reference.

## 10.10.2 What is anchored

| Kind | Grade | Evidence |
|---|---|---|
| `b2plyp` | Proven | total energy vs PySCF built from the definition (`dft.RKS` with 0.53 HF + 0.47 B88, 0.73 LYP, then `mp.dfmp2.DFMP2` on the KS orbitals): ≤ 1.3e-11 Ha; E_KS ≤ 7.1e-13, E_OS ≤ 4.1e-11, E_SS ≤ 1.5e-11 Ha (H2O, NH3, CH4 / cc-pVDZ and def2-SVP, frozen core 0 and 1); test `crates/ferric-mp2/tests/validation_b2plyp_pyscf.rs` |
| `dsd-pbep86` | Spike | compared with no reference code; ferric's parameters are tied to neither of the two published DSD-PBEP86 sets (2011, 2013) |

The B2PLYP reference runs a flat (75, 110) grid with RI-J in f64. The CLI
default (pruned grid, f32 RI-J) moves the total by up to 1.0e-6 Ha, so a
default run carries no anchor and the CLI prints a notice. Only Grimme's
2006 parameters are checked; no comparison of B2PLYP with experiment is
claimed. Figures from `site/src/reference/validation.md`.

## 10.10.3 Derivation status

(10.25) is a sum of an SCF energy and the SCS energy of section 10.3, so it
adds no new identity. The derivation in `crates/ferric-mp2/theory.md` is
being written on the double-hybrid grading branch; the registry lists
`double-hybrid` as a gap until it lands.

> **Unverified equation 10.25** · double-hybrid assembly
>
> <!-- unverified: id=dh.assembly; todo=derivation and proof owned by the double-hybrid grading branch (registry gap double-hybrid); link its theory.md heading and sympy claim here when merged -->

## Known limits (2026-11-15)

- Closed-shell (RKS) only.
- No gradient or Hessian: the MP2 term is not differentiated (the KS
  Hessian refuses double hybrids with a typed error).
- DSD-PBEP86 parameters are not tied to a published set.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/double_hybrid.rs` | parameters, reference check |
| `site/src/reference/validation.md`, rows `b2plyp` and `dsd-pbep86` | grades, anchors |
| `crates/ferric-mp2/theory.md`, "4. Spin-component scaling" | the MP2 part |
