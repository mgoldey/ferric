# k-point Hartree-Fock and Kohn-Sham {#kpoint-scf}

<!-- status: edit; registry: pbc-hf, pbc-ks -->

A Gamma-point calculation treats the cell as if every copy of every orbital
had the same sign in every cell. That is exact only when the cell is large
enough that the bands are flat. A crystal's Hamiltonian commutes with lattice
translations, so by Bloch's theorem its orbitals can be labelled by a crystal
momentum $k$, and orbitals with different $k$ do not mix: one infinite
problem becomes one finite problem at each $k$. A $k$-point mesh samples
$N_k$ of them, each with its own complex Fock matrix, coupled to the others
through the Coulomb and exchange terms.

Section @sec:kpoint-scf-bloch-functions-mesh-supercell derives the mesh, the matrices and the identity that tests
the $k$-point code: a Gamma-centred mesh is the Gamma calculation of the
$\operatorname{diag}(N)$ supercell, term by term, per primitive cell. @sec:kpoint-scf-algorithm
gives the SCF, @sec:kpoint-scf-options to @sec:kpoint-scf-sources the options, anchors, limits and sources.

| Symbol | Meaning | Code name |
|---|---|---|
| $k$, $k'$ | mesh points | `KPointMesh::kpts` |
| $N_k$ | mesh size | `nk` |
| $q = k' - k$ | momentum transfer of an exchange pair | `q` class |
| $\chi_{\mu k}$ | Bloch AO | |
| $D(k)$ | density, $2C_o(k)C_o(k)^{\mathsf H}$ | `dms[k]` |
| $v_M$ | Madelung constant of the $\operatorname{diag}(N)$ supercell | `KPointMesh::madelung` |

| Entry point | Location |
|---|---|
| Mesh | `ferric-pbc/src/kpts.rs::{monkhorst_pack,madelung}` |
| RHF | `ferric-pbc/src/kscf.rs::{solve_krhf,solve_k_closed_shell}` |
| UHF | `ferric-pbc/src/kuscf.rs::solve_kuhf` |
| ROHF | `ferric-pbc/src/kroks.rs::solve_krohf` |
| ROKS | `ferric-pbc/src/kroks.rs::solve_kroks` |
| RKS | `ferric-pbc/src/kdft.rs::solve_krks` |
| UKS | `ferric-pbc/src/kuks.rs::solve_kuks` |
| CLI | `[cell] kmesh = [n1, n2, n3]` with `method.kind` `rhf`, `uhf`, `rohf` or `ksdft` |
| Python | `run_rhf_kpts`, `run_uhf_kpts`, `run_rohf_kpts`, `run_rks_kpts`, `run_uks_kpts`, `run_roks_kpts` |

## Bloch functions, the mesh and the supercell {#kpoint-scf-bloch-functions-mesh-supercell}

**The mesh.** A mesh $N_1\times N_2\times N_3$ (`ferric-pbc/src/kpts.rs::monkhorst_pack`,
PySCF `cell.make_kpts`) has fractional points $f_i = \mathrm{num}_i/(2N_i)$
along $\mathbf b_i$, with integer numerators

$$ \mathrm{num}_i = 2m_i \ \text{(Gamma-centred)}, \qquad \mathrm{num}_i = 2m_i + 1 - N_i \ \text{(Monkhorst–Pack)}, \qquad m_i = 0, \dots, N_i - 1 . $$

For odd $N_i$ the two are the same set modulo the reciprocal lattice. For
even $N_i$ the Monkhorst–Pack set is shifted by half a step and does not
contain $\Gamma$.

**Bloch AOs.** Build one function per AO and per $k$ by summing the copies
with a phase:

$$
\chi_{\mu k}(\mathbf r) = \sum_{\mathbf L}e^{ik\cdot\mathbf L}\phi_\mu(\mathbf r - \mathbf L) .
$$

Translating $\chi_{\mu k}$ by a lattice vector multiplies it by a phase, which
is Bloch's theorem for a basis function. Now take the overlap of two of them
over a Born–von Kármán supercell of $N_k$ cells. Write $\mathbf L' = \mathbf L + \mathbf M$
and use translation invariance,
$\langle\mu_{\mathbf L}|\nu_{\mathbf L+\mathbf M}\rangle = \langle\mu_0|\nu_{\mathbf M}\rangle$:

$$ \langle\chi_{\mu k}|\chi_{\nu k'}\rangle = \sum_{\mathbf L}e^{i(k'-k)\cdot\mathbf L}\sum_{\mathbf M}e^{ik'\cdot\mathbf M}\langle\mu_0|\nu_{\mathbf M}\rangle = N_k\,\delta_{kk'}\,S_{\mu\nu}(k') . $$

The sum over $\mathbf L$ of the mesh phase is $N_k\delta_{kk'}$ (claim `mesh`
of the proof below). Every one-electron matrix is block-diagonal in $k$ the
same way:

$$ S_{\mu\nu}(k) = \sum_{\mathbf L}e^{ik\cdot\mathbf L}\langle\mu_0|\nu_{\mathbf L}\rangle , $$

and $h(k)$ likewise. Real AOs give real lattice blocks, so
$X(-k) = X(k)^*$ for every one-particle matrix. The code computes every
phase from the integer $\sum_i\mathrm{num}_in_iD/(2N_i) \bmod D$,
$D = \prod_i 2N_i$, because $k\cdot\mathbf L = 2\pi$ times that integer over
$D$. The reduction is exact, so $X(-k) = X(k)^*$ holds bit for bit.

> **Verified equation {#eq:pbc-kscf-mesh}** · the mesh and its phases
>
> The Gamma-centred and Monkhorst-Pack numerators give the PySCF fractional
> points; for odd $N$ the two meshes are the same set modulo the reciprocal
> lattice. Phases built from the integer numerators satisfy
> $X(-k) = X(k)^*$ for every one-particle matrix.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `mesh` |
> | Negative control | none in the script; the code test pins the points to PySCF |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/kpts.rs::monkhorst_pack` |
> | Code test | `monkhorst_pack_points_and_mesh_madelung_match_pyscf` in `crates/ferric-pbc/tests/pbc_krhf.rs` |
>
> <!-- verified: id=pbc.kscf.mesh; script=scripts/pbc_kpoint_corr_sympy.py; claim=mesh; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/kpts.rs::monkhorst_pack; rust_test=ferric-pbc/tests/pbc_krhf.rs::monkhorst_pack_points_and_mesh_madelung_match_pyscf; registry=pbc-hf -->

**Pair densities carry momentum.** The two-electron terms contract pair
densities $\chi^*_{\mu k}\chi_{\nu k'}$. Translate one by $\mathbf L$ and it
picks up $e^{i(k'-k)\cdot\mathbf L}$. Its Fourier transform is therefore
nonzero only on $\mathbf K = \mathbf G + q$, $q = k' - k$:

$$ P^{k'}_{\mu\nu}(\mathbf K) = \sum_{\mathbf L}e^{ik'\cdot\mathbf L}\int\phi_\mu\,\phi_\nu(\cdot - \mathbf L)\,e^{-i\mathbf K\cdot\mathbf r} , $$

the transform of section @sec:pair-ft with a phase in the image sum
(`ferric-pbc/src/kdense_aft.rs::build`). Only the ket's image sum carries a
phase: the bra's Bloch sum has been removed by translation invariance, as in
the overlap above, which leaves the factor $N_k\delta$ of the mesh identity. Coulomb pairs a function with its
own $k$, so $q = 0$ and the kernel is taken at $\mathbf G$. Exchange pairs
$k$ with every $k'$, so the kernel is taken at $\mathbf G + q$. The only term
dropped is $\mathbf K = 0$.

**Energy and Fock matrix.** With orbitals normalized per cell and
$D(k) = 2C_o(k)C_o(k)^{\mathsf H}$ (`ferric-pbc/src/kscf.rs::solve_k_closed_shell`):

$$
\begin{aligned}
F(k) &= h(k) + J(k) - \tfrac12K(k), \\
J_{\mu\nu}(k) &= \frac{1}{N_k}\sum_{k'}\frac{1}{\Omega}\sum_{\mathbf G\neq0}v(\mathbf G)\,
 P^{k}_{\mu\nu}(\mathbf G)\sum_{\lambda\sigma}P^{k'}_{\lambda\sigma}(\mathbf G)^*D_{\lambda\sigma}(k'), \\
K_{\mu\nu}(k) &= \frac{1}{N_k}\sum_{k'}\frac{1}{\Omega}\sum_{\mathbf K\in\mathbf G+q,\ \mathbf K\neq0}
 v(\mathbf K)\sum_{\lambda\sigma}P^{k'}_{\mu\lambda}(\mathbf K)P^{k'}_{\nu\sigma}(\mathbf K)^*D_{\lambda\sigma}(k')
 + v_M\,S(k)D(k)S(k), \\
E/\text{cell} &= \frac{1}{N_k}\sum_k\tfrac12\operatorname{Re}\operatorname{tr}\big[(h(k) + F(k))D(k)\big] + E_{nn}.
\end{aligned}
$$

Every $1/N_k$ is an average over the mesh: the density at $k'$ contributes
to the potential at $k$ with weight $1/N_k$, and the energy per cell is the
mesh average of the per-$k$ traces. The Madelung term is the exception. $v_M$
is the constant of the $\operatorname{diag}(N)$ supercell
(`ferric-pbc/src/kpts.rs::madelung`), added at every $k$ with no $1/N_k$
(PySCF `_ewald_exxdiv_for_G0`). The supercell is the cell whose $\Gamma$
calculation the mesh reproduces, so the probe charge of section @sec:exchange-divergence has the
supercell's images.

**The mesh is the supercell.** For a Gamma-centred mesh, $\{\mathbf G + q\}$ is
the reciprocal lattice of the $\operatorname{diag}(N)$ supercell. Build the
supercell density

$$ D^\text{sc}_{(\mu\mathbf L_1),(\nu\mathbf L_2)} = \frac{1}{N_k}\sum_k e^{ik\cdot(\mathbf L_1 - \mathbf L_2)}D_{\mu\nu}(k) . $$

Then the $J$, $K$ and energy above are, term by term, the Gamma ones of the
supercell per cell, and
$X(k) = \sum_{\mathbf L}e^{ik\cdot\mathbf L}X^\text{sc}_{(\mu0),(\nu\mathbf L)}$
maps the supercell Fock matrix to $F(k)$. A shifted mesh is a
twisted-boundary supercell and has no such anchor.

The anchor in numbers, from the `589b50832` build: H2/STO-3G in a 5 Bohr cubic
cell on a $1\times1\times3$ Gamma-centred mesh, against the tripled cell at
$\Gamma$, per primitive cell. The two agree to $5\times10^{-14}$ Ha under both
`exxdiv` conventions:

```python
import ferric

BOHR = 0.529177210903
basis = ferric.BasisSet.bundled("sto-3g")
a = 5.0 * BOHR
prim = ferric.Molecule.from_xyz_string("2\nH2\nH 0 0 0\nH 0 0 0.74\n")
lat = [[a, 0, 0], [0, a, 0], [0, 0, a]]
sc_xyz = "6\nH2x3\n" + "".join(f"H 0 0 {i * a:.10f}\nH 0 0 {i * a + 0.74:.10f}\n" for i in range(3))
sc = ferric.Molecule.from_xyz_string(sc_xyz)
lat3 = [[a, 0, 0], [0, a, 0], [0, 0, 3 * a]]
for exx in ("ewald", "none"):
    k = ferric.run_rhf_kpts(prim, lat, basis, (1, 1, 3), exxdiv=exx)
    g = ferric.run_rhf_gamma(sc, lat3, basis, exxdiv=exx)
    print(f"exxdiv={exx:5s}  E(1x1x3 mesh)/cell={k.energy:.12f}  "
          f"E(supercell)/3={g.energy / 3:.12f}  diff={k.energy - g.energy / 3:.2e}")
```

```text
exxdiv=ewald  E(1x1x3 mesh)/cell=-1.228803447495  E(supercell)/3=-1.228803447495  diff=-4.88e-14
exxdiv=none   E(1x1x3 mesh)/cell=-1.077069004631  E(supercell)/3=-1.077069004631  diff=-4.91e-14
```

**Check the Gamma limit.** With $N_k = 1$ the only point is $k = 0$, every
phase is 1, $q = 0$, $v_M$ is the cell's own constant, and the formulas are
those of sections @sec:exchange-divergence and @sec:rsgdf. The force and stress tests in sections @sec:forces
and @sec:stress use this: a one-point mesh must give the Gamma numbers.

> **Verified equation {#eq:pbc-kscf-supercell}** · the closed-shell k-mesh is the Gamma supercell
>
> $J(k)$ with the kernel at $\mathbf G$, $K(k)$ with the kernel at
> $\mathbf G + q$ and the factor $1/N_k$, the Madelung term
> $v_MS(k)D(k)S(k)$ with no $1/N_k$, and the two-electron energy
> $N_k^{-1}\sum_k\tfrac12\operatorname{Re}\operatorname{tr}[(J - \tfrac12K)(k)D(k)]$
> reproduce the supercell's Gamma $J$, $K$ and two-electron energy per cell.
> The one-electron term maps the same way through
> $X(k) = \sum_{\mathbf L}e^{ik\cdot\mathbf L}X^\text{sc}_{(\mu0),(\nu\mathbf L)}$
> (claim `mesh`).
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `krhf-supercell` |
> | Negative control | the runner's two mutations: the Madelung term divided by $N_k$, and the exchange kernel taken at $\mathbf G$ instead of $\mathbf G + q$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/kscf.rs::solve_k_closed_shell` |
> | Code test | `h2_1x1x3_mesh_is_the_gamma_supercell` in `crates/ferric-pbc/tests/pbc_krhf.rs` |
>
> <!-- verified: id=pbc.kscf.supercell; script=scripts/pbc_kpoint_corr_sympy.py; claim=krhf-supercell; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/kscf.rs::solve_k_closed_shell; rust_test=ferric-pbc/tests/pbc_krhf.rs::h2_1x1x3_mesh_is_the_gamma_supercell; registry=pbc-hf -->

The proof script works in one dimension with s Gaussians. The $k$-point
bookkeeping it checks (Bloch phases, the momentum transfer $q$, the $1/N_k$
normalizations, the Madelung constant per $k$) does not depend on the
dimension.

**Open shells.** Per spin, $D_\sigma(k) = C_{\sigma,o}C_{\sigma,o}^{\mathsf H}$
and (`ferric-pbc/src/kuscf.rs::solve_kuhf`)

$$
F_\sigma(k) = h(k) + J[D_\alpha + D_\beta]{}(k) - K[D_\sigma]{}(k),\qquad
E/\text{cell} = \frac{1}{N_k}\sum_{k,\sigma}\tfrac12\operatorname{Re}\operatorname{tr}\big[(h + F_\sigma)D_\sigma\big] + E_{nn},
$$

where $K[D_\sigma]$ carries $v_MSD_\sigma S$ with coefficient 1: the builder is
linear in the density, and there is no per-spin $\tfrac12$. Occupations are a
global aufbau per spin over the whole mesh. ROHF and ROKS combine the same
per-spin Fock matrices into the Guest–Saunders (Roothaan) effective Fock on
one orbital set: at Gamma through `ferric_scf::rohf`, on a mesh in
`ferric-pbc/src/kroks.rs::solve_krohf` (PySCF `get_roothaan_fock`). The
effective Fock matrix couples the closed, open and virtual blocks of one
orbital set (verified equation @eq:scf-rohf-gradient). On a mesh that coupling is formed per $k$
from $F_\alpha(k)$ and $F_\beta(k)$ in the MO basis of that $k$; the
occupations, which decide which orbitals are closed, open or virtual, come
from one aufbau per spin over the whole mesh.

> **Verified equation {#eq:pbc-kscf-uhf}** · the open-shell k-mesh is the Gamma supercell
>
> Per spin, $F_\sigma(k) = h + J[D_\alpha + D_\beta] - K[D_\sigma]$ with the
> Madelung term at coefficient 1 inside $K[D_\sigma]$, and the per-spin
> energy sum, reproduce the supercell's unrestricted Gamma energy.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `kuhf-supercell` |
> | Negative control | the `krhf-supercell` mutations act on the shared builder |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/kuscf.rs::solve_kuhf` |
> | Code test | `tri_triplet_1x1x3_is_the_gamma_supercell_uhf` in `crates/ferric-pbc/tests/pbc_kuhf.rs` |
>
> <!-- verified: id=pbc.kscf.uhf; script=scripts/pbc_kpoint_corr_sympy.py; claim=kuhf-supercell; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/kuscf.rs::solve_kuhf; rust_test=ferric-pbc/tests/pbc_kuhf.rs::tri_triplet_1x1x3_is_the_gamma_supercell_uhf; registry=pbc-hf -->

The Madelung shift of the occupied levels, which explains why `none` and
`ewald` share their densities, is box @eq:pbc-exxdiv-level-shift in section @sec:exchange-divergence. Per spin the
energy moves by $-v_M(N_\alpha + N_\beta)/2$.

**Kohn–Sham.** With an exact-exchange fraction $a$
(`ferric-pbc/src/kscf.rs::solve_krks_injected`):

$$
F(k) = h(k) + J(k) - \tfrac a2K(k) + V_{xc}(k),\qquad
E/\text{cell} = \frac{1}{N_k}\sum_k\operatorname{Re}\operatorname{tr}\big[D(k)\big(h + \tfrac12J - \tfrac a4K\big)(k)\big] + E_{xc} + E_{nn},
$$

$$
\rho(\mathbf r) = \frac{1}{N_k}\sum_k\sum_{\mu\nu}D_{\mu\nu}(k)\,\chi_{\mu k}(\mathbf r)\chi_{\nu k}(\mathbf r)^*,\qquad
V_{xc}(k)_{\mu\nu} = \int_\text{cell}\chi_{\mu k}^*\,v_\rho\,\chi_{\nu k} + \text{(GGA terms)} .
$$

$V_{xc}(k)$ is the $D(k)$ derivative of $E_{xc}$ times $N_k$, the same
convention as $h$, $J$ and $K$. The injected $K$ carries the Madelung term,
so it is scaled by $a$ with the rest of the exchange. The GGA terms need
$\nabla\chi_{\mu k}$. In this gauge,
$\chi_{\mu k}(\mathbf r) = \sum_{\mathbf L}e^{ik\cdot\mathbf L}\phi_\mu(\mathbf r - \mathbf L)$,
the phase does not depend on $\mathbf r$, so
$\nabla\chi_{\mu k} = \sum_{\mathbf L}e^{ik\cdot\mathbf L}\nabla\phi_\mu(\mathbf r - \mathbf L)$
with no extra $ik$ term; the molecular GGA and meta-GGA kernels apply to the
Bloch sums unchanged. UKS and ROKS use the per-spin densities and the
open-shell Fock matrices above, with $V_{xc,\sigma}(k)$ added per spin
(`ferric-pbc/src/kuks.rs::solve_kuks`, `ferric-pbc/src/kroks.rs::solve_kroks`).

**The cell integral.** The grid is the molecular atom-centred grid on the
cell's atoms only, each point weighted by its home atom's partition weight
evaluated in the infinite crystal, with every image atom within a cutoff $D$
of the point as a competitor (`ferric-pbc/src/dft.rs::PeriodicGrid`; SSF by
default). Write $w_{A,\mathbf L}$ for the weight of image $A + \mathbf L$.
Translation covariance gives $w_{A,\mathbf L}(\mathbf r) = w_{A,0}(\mathbf r - \mathbf L)$
and the weights sum to one over all atoms and images. For a lattice-periodic
$f$,

$$ \int_\text{cell}f = \sum_{\mathbf L}\sum_{A\in\text{cell}}\int_\text{cell}w_{A,\mathbf L}f = \sum_{A\in\text{cell}}\int_{\mathbb R^3}w_{A,0}\,f , $$

because the translated cells tile space and $f$ is periodic. So the full
atomic grid of each cell atom, with no cut at the cell boundary, integrates
over one cell. The cutoff $D$ keeps an exact partition of unity only when
every point has an atom within $D$, so `PeriodicGrid::build` refuses a $D$
below a bound on the covering radius. The range-separated exchange $K$ of a
CAM hybrid uses the kernel of section @sec:exchange-divergence. The periodic
quadrature, its partition-weight derivatives in the force and stress, and the
meta-GGA $\tau$ terms have no proof script. That is why `pbc-ks` stays a
registry gap and this section's status is `edit`.

## The algorithm {#kpoint-scf-algorithm}

| Step | What | Function |
|---|---|---|
| 1 | mesh, phases, supercell Madelung constant | `ferric-pbc/src/kpts.rs::{monkhorst_pack,madelung}` |
| 2 | $S(k)$, $h(k)$, orthogonalizer: eigenvectors of $S(k)$ with eigenvalue $\ge$ `lindep` ($10^{-6}$), scaled by $\lambda^{-1/2}$ | `ferric-pbc/src/kscf.rs::complex_canonical_orthogonalizer` |
| 3 | $J(k)$, $K(k)$ from the pair transforms (dense) or the $k$-point fit (`rsgdf`) | `ferric-pbc/src/kdense_aft.rs::build`, `ferric-pbc/src/rsgdf/kpoint.rs::KRsGdf` |
| 4 | diagonalize one of each $\{k, -k\}$; $C(-k) = C(k)^*$, $\varepsilon(-k) = \varepsilon(k)$ | `ferric-pbc/src/kscf.rs::diagonalize_all` |
| 5 | occupy the $N_kN_e/2$ lowest levels over the whole mesh (PySCF `KRHF.get_occ`); a converged gap below `min_gap` is an error | `ferric-pbc/src/kscf.rs::aufbau` |
| 6 | DIIS on $X^{\mathsf H}(FDS - SDF)X$ stacked over $k$, Gram $\operatorname{Re}\sum_k\langle e_i, e_j\rangle$, real weights | method `step` of `ferric-pbc/src/kscf.rs::KDiis` |

Step 4 halves the diagonalizations: $F(-k) = F(k)^*$, so the eigenvectors at
$-k$ are the conjugates of those at $k$. Step 5 fills levels over the whole
mesh, not per $k$, so two points can hold different numbers of occupied
levels; the correlation methods of section @sec:mp2-drpa refuse such a reference.

The energy is assembled in one loop of
`ferric-pbc/src/kscf.rs::solve_k_closed_shell`:

<!-- excerpt: ferric-pbc/src/kscf.rs::solve_k_closed_shell -->
```rust
        let mut e_elec = 0.0;
        for k in 0..nk {
            let hf = &inj.h[k] + &f_nox[k];
            // Re tr[(h + F) D] = Re Σ_mn (h+F)_mn D_nm
            let mut tr = 0.0;
            for m in 0..n {
                for nu in 0..n {
                    tr += (hf[(m, nu)] * dm[k][(nu, m)]).re;
                }
            }
            e_elec += 0.5 * tr;
        }
        let energy = e_elec * inv_nk + e_xc + inj.vnn;
```

| Symbol | Code |
|---|---|
| $h(k) + F(k)$, $F$ without $V_{xc}$ | `hf` (`f_nox`) |
| $E_{xc}$ (0 for HF) | `e_xc` |
| $D(k)$ | `dm[k]` |
| $N_k^{-1}$ | `inv_nk` |
| $E_{nn}$ | `inj.vnn` |

The Fock matrix in the trace has no $V_{xc}$: for Kohn–Sham,
$\tfrac12\operatorname{tr}[(h + F_\text{no xc})D]$ is the
$h + \tfrac12J - \tfrac a4K$ part of the energy, and $E_{xc}$ is added as a
number.

## Options {#kpoint-scf-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[cell] kmesh`, `kmesh=` | three positive integers | none (Gamma point) | the mesh |
| `[cell] centring` | `"gamma"`, `"mp"` | `"gamma"` | Gamma-centred or Monkhorst-Pack; only the Gamma-centred mesh has the supercell anchor |
| `[cell] exxdiv` | `"ewald"`, `"none"` | `"ewald"` | the Madelung term of section @sec:exchange-divergence |
| `[cell] ewald_start` | `"staged"`, `"direct"` | `"staged"` | open shells: converge with `none`, then restart with `ewald` |
| `[cell] jk` | `"dense"`, `"rsgdf"` | `"dense"` | section @sec:rsgdf |
| `min_gap` (`KScfConfig`) | Ha | $10^{-6}$ | a converged HOMO–LUMO gap on the mesh below it is an error |

## Numerical anchors {#kpoint-scf-numerical-anchors}

| Quantity | Bar | Test |
|---|---|---|
| $1\times1\times3$ and $2\times2\times2$ meshes against the explicit supercell, both `exxdiv` | $10^{-11}$ Ha | `crates/ferric-pbc/tests/pbc_krhf.rs` `h2_1x1x3_mesh_is_the_gamma_supercell`, `h2_2x2x2_mesh_is_the_gamma_supercell` |
| PySCF `KRHF` with AFTDF | $10^{-9}$ Ha | `h2_1x1x2_matches_pinned_pyscf_krhf_aftdf` |
| exchange kernel without the $q$ shift, a phase sign flip, the primitive-cell Madelung constant | each detected | `mutant_kernel_without_q_shift_is_caught`, `mutant_phase_sign_in_pair_ft_is_caught`, `mutant_primitive_madelung_is_caught_and_leaves_exxdiv_none_exact` |
| time reversal and Hermiticity of the Bloch matrices | | `bloch_convention_time_reversal_and_hermiticity` |

## Limits {#kpoint-scf-limits}

- Insulators only: the global aufbau refuses a converged gap below
  `min_gap` ($10^{-6}$ Ha). There is no smearing.
- A shifted (Monkhorst-Pack, even $N$) mesh is a twisted-boundary supercell
  and has no supercell anchor; it is pinned to PySCF instead
  (`h2_shifted_mp_1x1x2_matches_pinned_pyscf_krhf_aftdf` in
  `crates/ferric-pbc/tests/pbc_krhf.rs`; `shifted_mesh_triclinic_krks_matches_pinned_pyscf`
  and `shifted_mesh_cubic_krks_matches_pinned_pyscf` in
  `crates/ferric-pbc/tests/pbc_krks_shifted.rs`).

## Sources {#kpoint-scf-sources}

| Source | What it gives |
|---|---|
| `scripts/pbc_kpoint_corr_sympy.py` | the proof |
| PySCF `pyscf.pbc.scf.khf` | the conventions (`get_occ`, `_ewald_exxdiv_for_G0`) |
