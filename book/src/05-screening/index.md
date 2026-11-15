# 5. Screening

Most two-electron integrals in a large molecule are negligible, and every fast
algorithm in ferric rests on knowing which ones before computing them. A
screen is safe only if its estimate is an upper bound, and a bound is useful
only if it is tight where the integrals are small. This chapter derives the
bounds ferric uses, states for which kernels each holds, and shows why the
standard bound cannot see the attenuation of a short-range kernel.

| Section | Bound | Valid for |
|---|---|---|
| 5.1 | Schwarz $Q_{\mu\nu}Q_{\lambda\sigma}$; CSB | positive-definite kernels; CSB's $MM$ terms for pointwise-positive kernels |
| 5.2 | QQR, Schwarz times a distance envelope | Coulomb and the kernels Schwarz admits; proven for primitive $s$ pairs |
| 5.3 | pair lists for LinK | whatever bound they are built from |
