# Screening {#screening}

A molecule with $N$ basis functions has $O(N^4)$ two-electron integrals, but
in a large molecule most of them are negligible. An integral
$(\mu\nu|\lambda\sigma)$ is the repulsion between the charge clouds
$\chi_\mu\chi_\nu$ and $\chi_\lambda\chi_\sigma$. If $\chi_\mu$ and $\chi_\nu$
sit on atoms far apart, their product is close to zero everywhere, and so is
the integral. Every fast algorithm in ferric depends on finding these
integrals before it computes them.

A screen decides from a cheap estimate. Two properties make the estimate
useful. It must be an upper bound: if it ever falls below the integral, the
screen drops an integral that matters and nothing reports it. And it should be
tight where the integrals are small, or it keeps work it could skip. This
chapter derives the bounds ferric uses, states for which kernels each one
holds, and shows why the standard bound cannot see the attenuation of a
short-range kernel.

| Section | Bound | Valid for |
|---|---|---|
| @sec:schwarz | Schwarz $Q_{\mu\nu}Q_{\lambda\sigma}$; CSB | positive-definite kernels; CSB's $MM$ terms for pointwise-positive kernels |
| @sec:qqr | QQR, Schwarz times a distance envelope | Coulomb and the kernels Schwarz admits; proven for primitive $s$ pairs |
| @sec:pair-lists | pair lists for LinK | whatever bound they are built from |
