# 10.6 Attenuation and the basis-set limit

<!-- section: 10.6; status: edit; registry: attenuated-mp2;
     theory: crates/ferric-mp2/theory.md#"5. Attenuated MP2" -->

Attenuated MP2 works partly by cancelling basis-set superposition error,
so its one parameter, $r_0$, belongs to a basis. This section records how
the optimal $r_0$ moves as the basis grows toward the complete-basis-set
(CBS) limit. It is a summary of the Goldey dissertation, chapter 6. No new
code is involved: every row below is a terfc run of section 10.4 at a
different `att_r0`.

## 10.6.1 Setup

| Choice | Value (dissertation §6.2) |
|---|---|
| Kernel | terfc (10.12), one parameter $r_0$ |
| Training set | S66 interaction energies, CCSD(T)/CBS reference |
| Counterpoise | none, so the result transfers to intramolecular interactions, where counterpoise is not defined |
| MP2 | RI, frozen core |
| Quadruple zeta | dual-basis SCF |
| CBS estimate | correlation energy extrapolated from cardinal numbers $X < Y$, HF at the larger basis |

The two-point extrapolation (dissertation Eq. 6.1) is

$$ E^{\mathrm{corr}}_{XY} = \frac{E^{\mathrm{corr}}_X X^3 - E^{\mathrm{corr}}_Y Y^3}{X^3 - Y^3} . \tag{10.17} $$

ferric has no built-in extrapolation; (10.17) is applied to two runs by
hand.

## 10.6.2 Optimal $r_0$ by basis

S66 RMSD in kcal/mol at the optimal $r_0$, no counterpoise (dissertation
Tables 6.2–6.4). TQZ and aTQZ are (10.17) applied to the triple- and
quadruple-zeta correlation energies. "jun-", "jul-", "may-" and "apr-" are
the calendar bases of Papajak et al., with diffuse functions removed from
the aug- sets.

| Family | Basis | $r_0$ (Å) | RMSD |
|---|---|---|---|
| Dunning DZ | cc-pVDZ | 1.55 | 1.283 |
| | jun-cc-pVDZ | 1.50 | 0.687 |
| | jul-cc-pVDZ | 1.25 | 0.644 |
| | aug-cc-pVDZ | 1.05 | 0.426 |
| Dunning TZ | cc-pVTZ | 1.50 | 0.604 |
| | may-cc-pVTZ | 1.60 | 0.369 |
| | jun-cc-pVTZ | 1.45 | 0.388 |
| | jul-cc-pVTZ | 1.45 | 0.378 |
| | aug-cc-pVTZ | 1.35 | 0.251 |
| Dunning QZ | cc-pVQZ | 1.55 | 0.379 |
| | apr-cc-pVQZ | 1.65 | 0.301 |
| | may-cc-pVQZ | 1.60 | 0.309 |
| | jun-cc-pVQZ | 1.55 | 0.313 |
| | jul-cc-pVQZ | 1.55 | 0.315 |
| | aug-cc-pVQZ | 1.50 | 0.265 |
| Extrapolated | TQZ | 1.55 | 0.366 |
| | aTQZ | 1.65 | 0.304 |
| Pople | 6-31G* | 1.75 | 1.063 |
| | 6-31+G* | 1.45 | 0.916 |
| | 6-31++G** | 1.35 | 0.720 |
| | 6-311++G** | 1.35 | 0.741 |
| Karlsruhe | def2-SVPD | 0.75 | 0.493 |
| | def2-TZVPD | 1.30 | 0.439 |
| | def2-TZVPPD | 1.35 | 0.375 |

## 10.6.3 What the table says

- **$r_0$ grows toward the CBS limit.** In the fully augmented series it
  goes 1.05 → 1.35 → 1.50 → 1.65 Å (aDZ, aTZ, aQZ, aTQZ). A larger basis
  has less superposition error to cancel, so less correlation is cut off.
- **It does not go to infinity.** At aTQZ attenuated MP2 (0.304 kcal/mol)
  still beats MP2/CBS (0.73 kcal/mol, dissertation Table 2.1): part of what
  attenuation removes is MP2's intrinsic overestimate of dispersion, which
  no basis fixes.
- **Augmentation matters more than cardinal number.** aug-cc-pVTZ at
  0.251 kcal/mol is the best entry in the table, better than aug-cc-pVQZ
  and the extrapolations. The dissertation recommends MP2(terfc, aTZ).
- **Without diffuse functions the fit is poor at double zeta** (cc-pVDZ,
  1.283 kcal/mol).

## 10.6.4 Running these in ferric

| Basis in the table | Bundled in ferric |
|---|---|
| cc-pV{D,T,Q}Z, aug-cc-pV{D,T,Q}Z | yes |
| 6-31G*, 6-31+G*, 6-31++G**, 6-311++G** | yes |
| def2-SVPD, def2-TZVPD, def2-TZVPPD | yes, with matching `-RIFIT` sets |
| calendar sets (jun-, jul-, may-, apr-) | no; load one from a Basis Set Exchange JSON file (`BasisSet.from_bse_json` in Python) |

To use a row: `method.kind = "att-rimp2"`, `[mp2] att_operator = "terfc"`,
`att_r0` = the table's value in Å. The 1.05 Å default is right only for
aug-cc-pVDZ.

These are dissertation numbers, run in Q-Chem. ferric has not re-run the
S66 fits, and none of the figures above is a ferric anchor. The terfc
integrals and energies ferric computes are anchored in section 10.5.

## Known limits (2026-11-15)

- No dual-basis SCF and no built-in CBS extrapolation.
- Calendar bases are not bundled.

## Sources

| Source | Used for |
|---|---|
| Goldey dissertation, chapter 6 (§6.2, Eq. 6.1, Tables 6.2–6.4) | setup, equation 10.17, the table, the conclusions |
| `crates/ferric-core/src/basis/bundled` | which bases ship |
