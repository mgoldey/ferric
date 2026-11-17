# SC-NEVPT2 {#nevpt2}

<!-- status: ready; registry: nevpt2 -->

A CASSCF wavefunction gets the near-degenerate configurations right, but
every electron outside the active space stays in a fixed core or virtual
orbital, so the short-range avoidance that MP2 describes for a single
determinant (chapter @sec:mp2) is missing. Strongly contracted NEVPT2 adds it by
second-order perturbation theory on top of the CAS root. Its zeroth-order
Hamiltonian is Dyall's, which treats the core and virtual electrons with
orbital energies but keeps the full two-electron interaction inside the
active space, so the CAS root is an exact eigenfunction of it. One
perturber per external label, $\Psi_l = P_lH\Psi_0$, makes the energy a sum
over eight excitation classes, each a ratio of a norm to a shifted
denominator.

| Symbol | Meaning | Code name |
|---|---|---|
| $l$ | label: the external (core and virtual) orbitals of a class | |
| $N_l$, $K_l$ | $\langle\Psi_l|\Psi_l\rangle$, $\langle\Psi_l|H_\text{act} - E_0|\Psi_l\rangle$ | |
| $\Delta_l$ | $\sum_{r\in l}\varepsilon_r - \sum_{i\in l}\varepsilon_i$ | |
| classes | `Sr`, `Si`, `Sijrs`, `Sijr`, `Srsi`, `Srs`, `Sij`, `Sir` | `sc_nevpt2` result |

| Entry point | Location |
|---|---|
| Rust | `ferric-ci/src/nevpt2.rs::sc_nevpt2` |
| CLI | `method.kind = "nevpt2"` with `[cas]` (`examples/n2-nevpt2.toml`); `[cas] reference = "casscf"` (default) or `"casci"` |
| Python | `run_nevpt2(reference=)` |

## The method {#nevpt2-method}

This is strongly contracted NEVPT2 of Angeli, Cimiraglia and Malrieu, JCP
117, 9138 (2002), on one CAS root. The definitions, conventions and the
prototype's measurements are in `scripts/prototypes/nevpt2_derivation.md`
("the derivation" below), and its numpy construction
`scripts/prototypes/nevpt2_proto.py` is the reference implementation. In
this section $i, j$ are core, $r, s$ virtual, and active orbitals keep the
active letters.

### Dyall's Hamiltonian

The zeroth-order Hamiltonian (derivation, "Definition") is

$$
H_D = \sum_i \varepsilon_i E_{ii} + \sum_r \varepsilon_r E_{rr} + H_\text{act},\qquad
H_\text{act} = \sum_{tu} F^I_{tu}E_{tu} + \tfrac12\sum_{tuvw}(tu|vw)(E_{tu}E_{vw} - \delta_{uv}E_{tw}).
$$

An electron in a core or virtual orbital contributes its orbital energy and
nothing else. $H_\text{act}$ is the active-space Hamiltonian of section
@sec:determinants-hamiltonian, with $F^I$ as the one-electron part and the two-electron term in
the form of section @sec:direct-sigma-davidson-split. So $H_D$ acting on $\Psi_0$, whose core is
full and virtuals empty, gives the CAS-CI problem back: $\Psi_0$ is an
eigenfunction with eigenvalue $E_0$ in the active part. This holds only if
$\Psi_0$ is an exact CI root, which is why the residual is checked below.

### One perturber per label

A label $l$ is the set of external orbitals of a class with $n_v$ virtual
particles and $n_c$ core holes. The perturber is $\Psi_l = P_l H\Psi_0$,
the part of $H\Psi_0$ with exactly that external occupation. Define

$$
N_l = \langle\Psi_l|\Psi_l\rangle,\qquad
K_l = \langle\Psi_l|H_\text{act} - E_0|\Psi_l\rangle,\qquad
\Delta_l = \sum_{r\in l}\varepsilon_r - \sum_{i\in l}\varepsilon_i .
$$

Second-order perturbation theory with one normalized function
$\Psi_l/\sqrt{N_l}$ per label gives
$-|\langle\Psi_l|H|\Psi_0\rangle|^2/N_l/(E_l - E_0)$. Two facts simplify it.

- $P_l$ is a projector, so
  $\langle\Psi_l|H|\Psi_0\rangle = \langle P_lH\Psi_0|H\Psi_0\rangle =
  \langle P_lH\Psi_0|P_lH\Psi_0\rangle = N_l$.
- $E_l$ is the $H_D$ expectation value of the perturber. Its external part
  adds $\varepsilon_r$ for each virtual particle and removes $\varepsilon_i$
  for each core hole relative to $\Psi_0$, and its active part is
  $H_\text{act}$, so
  $\langle\Psi_l|H_D|\Psi_l\rangle/N_l - E_0 = \Delta_l + K_l/N_l$.

Substitute both:

$$
E_\text{class} = -\sum_l \frac{N_l}{\Delta_l + K_l/N_l},
$$

over labels with $|N_l| > 10^{-14}$ (PySCF `NUMERICAL_ZERO`, `NORM_ZERO` in
the code). The denominator is the cost of the excitation: orbital energies
for the external part, plus the average active-space energy the perturber
pays above $E_0$.

`ferric-ci/src/nevpt2.rs::norm_to_energy` loops over the labels, one
$(N_l, K_l, \Delta_l)$ triple each:

```rust
for (n, h, d) in labels {             // (N_l, K_l, Δ_l)
    nt += n;                          // Σ_l N_l, the class norm
    if n.abs() > NORM_ZERO {
        e -= n / (d + h / n);         // − N_l / (Δ_l + K_l/N_l)
    }
}
```

### The eight classes

The eight classes are those of the derivation's table: `Sr` (r), `Si` (i),
`Sijrs`, `Sijr`, `Srsi`, `Srs`, `Sij` and `Sir`. The name lists the
external labels. Counting electrons gives the change of the active-space
electron number: `Sijrs` and `Sir` leave it unchanged, `Sijr`, `Si` add
one, `Sij` adds two, `Srsi`, `Sr` remove one and `Srs` removes two.

### Orbitals

$F^I = h + J[D^c] - \tfrac12 K[D^c]$ with $D^c = 2C_cC_c^\mathsf{T}$, and
$F = F^I + F^A$ with $F^A$ from $D^a = C_a\gamma C_a^\mathsf{T}$, as in
section @sec:casscf. The core and virtual orbitals are the eigenvectors of the core
and virtual blocks of $F$, and $\varepsilon$ their eigenvalues. This is
PySCF `mcscf.CASCI.canonicalize`, which `mrpt.NEVPT` calls (derivation,
"Orbitals"). The active orbitals are used as given: every active index is
summed, so $E$ does not depend on the active basis, and the RDMs are built
in that basis from the reference CI vector. PySCF rotates to natural
orbitals instead. With `canonicalize = false` the given core and virtual
orbitals are used, $\varepsilon = \operatorname{diag}F$, and an off-diagonal
element above $10^{-8}$ Ha (`CANONICAL_TOL`) is an error.

Inside a group of degenerate core or virtual orbitals the eigenvectors are
not unique, and the pair classes are not invariant to rotations in it
(derivation, "Comparison with pyscf.mrpt.NEVPT"). N2's $\pi$ pairs move
`Srs` by up to 1.7e-7 Ha.

### Evaluation

$N_l$ and $K_l$ reduce (derivation, "Construction A", steps 1-5; $K_l$
through $\langle V^\dagger P_l [H_\text{act}, V]\rangle$) to contractions
of integrals with RDMs up to rank 4, the 4-RDM only contracted with
$(aa|aa)$. The module evaluates them with the factorization of PySCF
`mrpt.nevpt2` (`Sr` ... `Sir`, `make_a16`, `make_a17`, `make_a19`,
`make_a22`, `make_a23`, `make_a25`, `make_a3`, `make_k27`, `make_a7`,
`make_a9`, `make_a12`, `make_a13`), transcribed spec for spec into an
`einsum` over owned arrays.

The RDM inputs are PySCF's (section @sec:rdms-nevpt-rdms): the E-product
$\langle E_{pq}E_{rs}\rangle$ and $\langle E_{pq}E_{rs}E_{tu}\rangle$ (from
the normal-ordered ones by `rdm3::e_product_rdms`, the inverse of the
reordering there), the hole RDMs, and $f^\text{ca}$, $f^\text{ac}$. In
PySCF's layout `h1e` is the active block of $F^I$ and
$\texttt{h2e}_{pqrs} = (pr|qs)$. The external integral blocks are
$(va|aa)$, $(ac|aa)$, $(cv|cv)$, $(ca|cv)$, $(va|cv)$, $(va|va)$, $(ca|ca)$,
$(vc|aa)$, $(va|ca)$ and the $F^I$ blocks $va$, $ac$, $vc$ (derivation,
"What a Rust port needs"), each transformed from the dense AO block with its
narrowest orbital set contracted first.

**Reference checks before any class.** $\lVert Hc - Ec\rVert \le$
`max_ci_residual` (default $10^{-8}$; the formulas assume
$H_\text{act}\Psi_0 = E_0\Psi_0$), $\lVert c\rVert = 1$ to $10^{-10}$,
$n_\alpha = n_\beta$ (open-shell active spaces are not anchored and are
refused), and a converged CASSCF or CAS-CI run.

**Cost.** $O(n^8)$ for the $a_{16}$/$a_{22}$ intermediates, $O(n_v n^6)$ (r),
$O(n_c n^6)$ (i), $O(n_v^2 n^4)$ (rs), $O(n_c^2 n^4)$ (ij), $O(n_c n_v n^4)$
(ir) and $O(n_c^2 n_v^2)$ (ijrs). The `einsum` is a direct loop over every
label combination (one sequential sum per output element, parallel over
output elements), so its cost is the product of all label extents of a spec.

**Memory.** The dense AO block $n_\text{bas}^4$, the MO blocks (twice,
transformed and permuted), twelve $n^6$ intermediates and the per-label
tensors, reserved through one `MemoryPlan` before the AO block is built.

### The classes as coded

The notation below is the code's: $\gamma$ = `dm1`; $G_2$, $G_3$ = `dm2`,
`dm3` of `Inputs`, the E-product RDMs $\langle E_{pq}E_{rs}\rangle$ and
$\langle E_{pq}E_{rs}E_{tu}\rangle$; $\bar\gamma$, $\bar G_2$, $\bar G_3$
the hole RDMs; $f^{ca}$, $f^{ac}$ the $(aa|aa)$-contracted 4-RDMs of section
@sec:rdms-nevpt-rdms; $h$ = `h1e` $= F^I_{\rm act}$ and $g_{pqrs}$ = `h2e` $= (pr|qs)$
over active orbitals; $i, j$ core, $r, s$ virtual, every other index
active. Specs are `einsum` strings over the named operands; a sum without a
repeated label is a product.

| Class | Labels $l$ (weight) | $\Delta_l$ | $N_l$ | $K_l$ |
|---|---|---|---|---|
| `Sr` | $r$ | $\varepsilon_r$ | `ipqr,rpqbac,iabc` $(V,G_3,V)$ $+\,2$ `ipqr,rpqa,ia` $(V,G_2,h^{\rm sr})$ $+$ `ip,pa,ia` $(h^{\rm sr},\gamma,h^{\rm sr})$ | the same with $A_{16}$ (`pqrabc`), $A_{17}$ (`pqra`), $A_{19}$ (`pa`) |
| `Si` | $i$ | $-\varepsilon_i$ | `qpir,rpqbac,baic` $(V,G_3^h,V)$ $+\,2$ `qpir,rpqa,ai` $(V,G_2^h,h^{\rm si})$ $+$ `pi,pa,ai` $(h^{\rm si},\gamma^h,h^{\rm si})$ | the same with $A_{22}$, $A_{23}$, $A_{25}$ |
| `Sijrs` | $(i,r,j,s)$, all ordered | $\varepsilon_r + \varepsilon_s - \varepsilon_i - \varepsilon_j$ | $v\,\theta$ | 0 |
| `Sijr` | $r$, $i \ge j$ ($\tfrac12$ at $i = j$) | $\varepsilon_r - \varepsilon_i - \varepsilon_j$ | $T[\bar\gamma]$ | $T[A_3]$ |
| `Srsi` | $s \ge r$ ($\tfrac12$ at $r = s$), $i$ | $\varepsilon_r + \varepsilon_s - \varepsilon_i$ | $U[\gamma]$ | $U[K_{27}]$ |
| `Srs` | $(r,s)$, all ordered | $\varepsilon_r + \varepsilon_s$ | $\tfrac12$ `rsqp,rsba,pqba` $(V,V,R_2)$ | $\tfrac12$ `rsqp,rsba,pqab` $(V,V,A_7)$ |
| `Sij` | $(i,j)$, all ordered | $-(\varepsilon_i + \varepsilon_j)$ | $\tfrac12$ `qpij,baij,pqab` $(V,V,\bar G_2)$ | $\tfrac12$ `qpij,baij,pqab` $(V,V,A_9)$ |
| `Sir` | $(i,r)$ | $\varepsilon_r - \varepsilon_i$ | nine terms below | four terms below |

The $\Delta_l$ column follows from the definition: a virtual label adds
$\varepsilon_r$, a core label subtracts $\varepsilon_i$.

**Integral blocks** (chemist $(pq|rs)$ over MO blocks, laid out as in
PySCF): `Sr` $V_{rqps} = (rp|qs)$, $h^{\rm sr}_{rp} = F^I_{rp} - \sum_b (rb|bp)$;
`Si` $V_{pqir} = (pi|qr)$, $h^{\rm si}_{pi} = F^I_{pi}$;
`Sijrs` $v = (ir|js)$, $\theta = 2(ir|js) - (is|jr)$;
`Sijr` $V_{rpij} = (jp|ir)$; `Srsi` $V_{rsip} = (sp|ir)$; `Srs`
$V_{rspq} = (rp|sq)$; `Sij` $V_{pqij} = (ip|jq)$; `Sir`
$V^1_{rpiq} = (ri|pq)$, $V^2_{rqpi} = (rp|iq)$, $h^{\rm ir}_{ri} = F^I_{ri}$.

**`Sijrs` is closed-shell MP2.** Two core electrons go to two virtuals and
the active part of each label is $\Psi_0$ itself, so $K_l = 0$. The class
is the direct sum
$\sum_{irjs} v\theta/(\varepsilon_i + \varepsilon_j - \varepsilon_r - \varepsilon_s)$,
with no $|N_l|$ test, which is equation (@eq:closed-shell-ri-mp2-3) restricted to core-to-virtual
excitations. `ferric-ci/src/nevpt2.rs::sijrs` is that loop:

```rust
let v = g[[i, a, j, bb]];                                               // v = (ir|js)
let theta = 2.0 * v - g[[i, bb, j, a]];                                 // θ = 2(ir|js) − (is|jr)
norm += v * theta;                                                      // Σ N_l
e += v * theta / (x.eps_c[i] + x.eps_c[j] - x.eps_v[a] - x.eps_v[bb]); // vθ/(ε_i + ε_j − ε_r − ε_s)
```

Here `a` and `bb` are the virtual indices $r$ and $s$. As in section @sec:closed-shell-ri-mp2,
the denominator is negative and carries the sign.

**Hole-type inputs of `Si`.** $G^h_3 = 2\,G_2\otimes\delta$
(`abef,cd->abcdef`) $-\,G_3$ with axes $(0,1,3,2,4,5)$;
$G^h_2 = 2\,\gamma\otimes\delta - G_2$ with axes $(0,1,3,2)$;
$\gamma^h = 2\delta - \gamma^{\sf T}$.

**Pair symmetrization.** `Sijr` and `Srsi` are pair-symmetrized: with $M$
the active matrix, $t = 2\,s_1(V,V,M) - s_2(V,V,M)$ and $T = t + t^{\sf T}$
over the pair axes, with $s_1$ and $s_2$ the contractions `s1` and `s2`.
`Sijr`: `s1` = `rpji,raji,pa->rji`, `s2` = `rpji,raij,pa->rji`, pair axes
$(i, j)$. `Srsi`: `s1` = `rsip,rsia,pa->rsi`, `s2` = `rsip,sria,pa->rsi`,
pair axes $(r, s)$. The pair-diagonal label carries $\tfrac12$ of both $N$
and $K$. `Srs` and `Sij` sum every ordered pair with $\tfrac12$ instead,
which is the same sum.

**`Sir`.** The norm is
$N_{ir} = 2$`rpiq,raib,qpab`$(V^1,V^1,G_2)$ $-$ `rpiq,rabi,qpab`$(V^1,V^2,G_2)$
$-$ `rpqi,raib,qpab`$(V^2,V^1,G_2)$ $+\,2$`raqi,rabi,qb`$(V^2,V^2,\gamma)$
$-$ `rpqi,rabi,qbap`$(V^2,V^2,G_2)$ $+$ `rpqi,raai,qp`$(V^2,V^2,\gamma)$
$+\,4$`rpiq,ri,qp`$(V^1,h^{\rm ir},\gamma)$ $-\,2$`rpqi,ri,qp`$(V^2,h^{\rm ir},\gamma)$
$+\,2\,h^{\rm ir}_{ri}h^{\rm ir}_{ri}$, and
$K_{ir} = 2$`rpiq,raib,pqab`$(V^1,V^1,A_{12})$ $-$ `rpiq,rabi,pqab`$(V^1,V^2,A_{12})$
$-$ `rpqi,raib,pqab`$(V^2,V^1,A_{12})$ $+$ `rpqi,rabi,pqab`$(V^2,V^2,A_{13})$.
$K$ has no $h^{\rm ir}$ term. The part of $\Psi_l$ that is
$F^I_{ri}E_{ri}\Psi_0$ leaves the active space untouched, and $E_{ri}$
commutes with $H_{\rm act}$, so its contribution to
$\langle\Psi_l|H_{\rm act} - E_0|\Psi_l\rangle$ is
$\langle\Psi_0|(H_{\rm act} - E_0)\cdots\rangle = 0$.

**The intermediates**, term by term. $\delta$ is the active identity, and
$h'$, the shifted one-electron matrix of `make_a17`/`make_a19`, is $h$
minus the contraction `mjjn->mn` of $g$. The 3-RDM enters through $G_3$ (and $\bar G_3$ in
$A_9$); the 4-RDM only through $f^{ca}$, $f^{ac}$, permuted. A spec without
`->` produces the intermediate's own index string; an axes tuple is a numpy
transpose.

| Intermediate | Terms (coefficient, spec, operands) |
|---|---|
| $A_{16}$ `pqrabc` | $-$`ib,rpqiac`$(h,G_3)$; $+$`ia,rpqbic`$(h,G_3)$; $-$`ci,rpqbai`$(h,G_3)$; $-f^{ca}$ axes $(1,4,0,2,5,3)$; $-$`kbia,rpqcki`$(g,G_3)$; $-$`kbaj,rpqjkc`$(g,G_3)$; $+$`cbij,rpqjai`$(g,G_3)$; $+$`prab,qc`$(F,\delta)$ with $F$ = `kbij,rpajki->prab`$(g,G_3)$; $+f^{ac}$ axes $(1,2,0,4,3,5)$; $-f^{ca}$ axes $(1,2,0,4,3,5)$; $+$`jbij,rpqiac`$(g,G_3)$; $-$`cjka,rpqbjk`$(g,G_3)$; $+$`jcij,rpqbai`$(g,G_3)$ |
| $A_{17}$ `abcp` | $-$`pi,cabi`$(h',G_2)$; $-$`kpij,cabjki`$(g,G_3)$ |
| $A_{19}$ `ap` | $-$`pi,ai`$(h',\gamma)$; $-$`kpij,ajki`$(g,G_2)$ |
| $A_{22}$ `ijkabc` | $-$`pb,kipjac`$(h,G_3)$; $-$`pa,kibjpc`$(h,G_3)$; $+$`cp,kibjap`$(h,G_3)$; $+$`cqra,kibjqr`$(g,G_3)$; $-$`qcpq,kibjap`$(g,G_3)$; $-f^{ac}$ axes $(1,5,0,2,4,3)$; $-$`ikbc,ja`$(F,\delta)$ with $F$ = `pqrb,kiqcpr->ikbc`$(g,G_3)$; $-$`pqab,kiqjpc`$(g,G_3)$; $+$`pcrb,kiajpr`$(g,G_3)$; $+$`cqrb,kiqjar`$(g,G_3)$; $-f^{ac}$ axes $(1,3,0,4,2,5)$; $+f^{ca}$ axes $(1,3,0,4,2,5)$; $+2$`jb,kiac`$(h,G_2)$; $+2$`pjrb,kiprac`$(g,G_3)$; $+2$`ikac,jb`$(F',\delta)$ with $F'$ = `pa,kipc`$(h,G_2)$ $-$ `cp,kiap`$(h,G_2)$ $-$ `cqra,kiqr`$(g,G_2)$ $+$ `qcpq,kiap`$(g,G_2)$ $+$ `pqra,kiqcpr`$(g,G_3)$ $-$ `rcpq,kiaqrp`$(g,G_3)$ |
| $A_{23}$ `abcp` | $-$`ip,caib`$(h,G_2)$; $-$`pijk,cajbik`$(g,G_3)$; $+2$`bp,ca`$(h,\gamma)$; $+2$`pibk,caik`$(g,G_2)$ |
| $A_{25}$ `ap` | $-$`pi,ai`$(h,\gamma)$; $-$`pijk,jaik`$(g,G_2)$; $+2h^{\sf T}$; $+2$`piaj,ij`$(g,\gamma)$ |
| $A_3$ `pa` | $+$`ia,ip`$(h,\bar\gamma)$; $+2$`ijka,pj,ik`$(g,\delta,\gamma)$; $-$`ijka,jpik`$(g,G_2)$ |
| $K_{27}$ `pa` | $-$`ai,pi`$(h,\gamma)$; $-$`iajk,pkij`$(g,G_2)$; $+$`iaji,pj`$(g,\gamma)$ |
| $R_2$, $A_7$ `pqab` | $R_2$ = `iljk->ijkl`$(G_2)$ $-$ `ik,jl`$(\gamma,\delta)$; $R_3$ = `injmkl->ijklmn`$(G_3)$ $-$ `jn,imkl`$(\delta,G_2)$ $-$ `km,ijln`$(\delta,R_2)$ $-$ `kn,ijml`$(\delta,R_2)$; $A_7$ = $-$`bi,pqia`$(h,R_2)$ $-$ `ai,pqbi`$(h,R_2)$ $-$ `kbij,pqkija`$(g,R_3)$ $-$ `kaij,pqkibj`$(g,R_3)$ $-$ `baij,pqij`$(g,R_2)$ |
| $A_9$ `pqab` | $+$`ib,pqai`$(h,\bar G_2)$; $+2$`ijib,pqaj`$(g,\bar G_2)$; $-$`ijjb,pqai`$(g,\bar G_2)$; $-$`ijkb,pkqaij`$(g,\bar G_3)$; $+$`ia,pqib`$(h,\bar G_2)$; $-$`ijja,pqib`$(g,\bar G_2)$; $-$`ijba,pqji`$(g,\bar G_2)$; $+2$`ijia,pqjb`$(g,\bar G_2)$; $-$`ijka,pqkjbi`$(g,\bar G_3)$ |
| $A_{12}$ `pqab` | $+$`ia,qpib`$(h,G_2)$; $-$`bi,qpai`$(h,G_2)$; $+$`ijka,qpjbik`$(g,G_3)$; $-$`kbij,qpajki`$(g,G_3)$; $-$`bjka,qpjk`$(g,G_2)$; $+$`jbij,qpai`$(g,G_2)$ |
| $A_{13}$ `pqab` | $-$`ia,qbip`$(h,G_2)$; $+2$`pa,qb`$(h,\gamma)$; $+$`bi,qiap`$(h,G_2)$; $-2$`pa,bi,qi`$(\delta,h,\gamma)$; $-$`ijka,qbjpik`$(g,G_3)$; $+$`kbij,qjapki`$(g,G_3)$; $+$`blma,qmlp`$(g,G_2)$; $+2$`kpma,qbkm`$(g,G_2)$; $-2$`bpma,qm`$(g,\gamma)$; $-$`lbkl,qkap`$(g,G_2)$; $-2$`ap,mbkl,qlmk`$(\delta,g,G_2)$; $+2$`ap,lbkl,qk`$(\delta,g,\gamma)$ |

### What the proofs cover

[`scripts/nevpt2_sympy.py`](https://github.com/mgoldey/ferric/blob/main/scripts/nevpt2_sympy.py)
proves four things. (D, sympy) the single-perturber energy above. (R) on a
CAS(4,4) model, the normal-ordered and E-product 1- to 3-RDMs and the
reorder formula, the hole RDMs of `hole_rdms` against
$\langle aa\,a^\dagger a^\dagger\rangle$ and
$\langle aaa\,a^\dagger a^\dagger a^\dagger\rangle$ from created vectors,
and `f3ca`/`f3ac` by the `rdm3.rs` construction against a stored E-product
4-RDM. (T) that its transcription of every function in the tables uses the
specs and axes of `nevpt2.rs`, in order, parsed from the file. (C) that
each of the eight class norms and energies of that transcription equals the
definition in the full 4900-determinant space ($P_lH\Psi_0$ grouped by
external occupation, $K_l$ from Dyall's $H_{\rm act}$ on active indices
only), for the lowest singlet and the lowest $M_s = 0$ triplet root, with
$\langle\Psi_0|H|\Psi_0\rangle = E_{\rm core} + E_0$ and
$\sum_l N_l = |QH\Psi_0|^2$ as anchors. Its recorded mutations are in
`crates/ferric-python/tests/test_derivation_scripts.py`.

> **Verified equation {#eq:ci-nevpt2-classes}** · the eight class energies
>
> On a CAS(4,4) model (two core, four active, two virtual orbitals, random
> positive semidefinite $(pq|rs)$, core and virtual orbitals canonical in
> $F^I + F^A$), a numpy transcription of every class and intermediate
> function of `nevpt2.rs` gives each class norm $\sum_l N_l$ and energy
> $E_\text{class} = -\sum_l N_l/(\Delta_l + K_l/N_l)$ equal to the
> definition evaluated in the full 4900-determinant space:
> $\Psi_l = P_lH\Psi_0$ grouped by external occupation, $K_l$ from Dyall's
> $H_\text{act}$ on active indices only. It holds on the lowest singlet and on
> the lowest $M_s = 0$ triplet root of the $(2, 2)$ sector. Anchors:
> $\langle\Psi_0|H|\Psi_0\rangle = E_\text{core} + E_0$ and
> $\sum_l N_l = |QH\Psi_0|^2$. The transcription is pinned to the Rust file:
> every `einsum` specification and permutation of those functions, in order,
> is parsed from `nevpt2.rs` and compared (claim T).
>
> | | |
> |---|---|
> | Proof | claims C and T of `scripts/nevpt2_sympy.py` (claim D: the single-perturber energy, sympy) |
> | Negative control | the runner's mutations: a sign flip of the `f3ca` term of `make_a16` (`Sr`) and of one `make_a12` term (`Sir`), the `Sijr` pair diagonal not halved, $\theta$ of `Sijrs` replaced by $v$, the sign of the `Sij` denominator, an $i \leftrightarrow j$ swap of the `Sijr` block (seen by claim T only: `Sijr` is symmetric in the pair) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-ci/src/nevpt2.rs::{sr,si,sijrs,sijr,srsi,srs,sij,sir,norm_to_energy}` |
> | Code test | `matches_prototype_h2o_cas44` in `crates/ferric-ci/tests/nevpt2.rs` |
>
> <!-- verified: id=ci.nevpt2.classes; script=scripts/nevpt2_sympy.py; claim=C; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-ci/src/nevpt2.rs::{sr,si,sijrs,sijr,srsi,srs,sij,sir,norm_to_energy}; rust_test=ferric-ci/tests/nevpt2.rs::matches_prototype_h2o_cas44; registry=nevpt2 -->

> **Verified equation {#eq:ci-nevpt2-rdms}** · the RDM inputs
>
> On the same model's CI vector, each input of the class formulas by two
> routes: the normal-ordered 1- to 3-RDMs (Gram matrices of annihilated
> vectors) against the E-product $\langle EE\rangle$, $\langle EEE\rangle$
> through the reorder formula and `e_product_rdms`; `hole_rdms` against
> $\langle aa\,a^\dagger a^\dagger\rangle$ and
> $\langle aaa\,a^\dagger a^\dagger a^\dagger\rangle$ from created vectors;
> $f^\text{ca}$, $f^\text{ac}$ by the `rdm3.rs` construction
> ($\langle E_{dc}X_{ba}|R_{ef}\rangle$) against the contraction of a stored
> E-product 4-RDM with $(aa|aa)$.
>
> | | |
> |---|---|
> | Proof | claim R of `scripts/nevpt2_sympy.py` |
> | Negative control | the runner's mutations: the $\delta_{qt}$ term of `e_product_rdms` dropped; a coefficient 2 → 1 in `hole_rdms`; $R'$ built with the index order of $R$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-ci/src/rdm3.rs::{e_product_rdms,hole_rdms,nevpt_rdms}` |
> | Code test | `rdm3_and_f3_match_pyscf_h2o_cas44` in `crates/ferric-ci/tests/casci_rdm3.rs` |
>
> <!-- verified: id=ci.nevpt2.rdms; script=scripts/nevpt2_sympy.py; claim=R; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-ci/src/rdm3.rs::{e_product_rdms,hole_rdms,nevpt_rdms}; rust_test=ferric-ci/tests/casci_rdm3.rs::rdm3_and_f3_match_pyscf_h2o_cas44; registry=nevpt2 -->

The proof checks the factorization against the definition on a model; the
Rust tests below check the Rust code against the numpy prototype's Wick
construction and against PySCF on molecules.

## The algorithm {#nevpt2-algorithm}

| Step | Quantity | Function |
|---|---|---|
| 1 | a converged CASSCF or CAS-CI root with $n_\alpha = n_\beta$; $\lVert Hc - Ec\rVert \le$ `max_ci_residual`, $\lVert c\rVert = 1$ | `sc_nevpt2` |
| 2 | canonical core and virtual orbitals in $F^I + F^A$ (PySCF `CASCI.canonicalize`); active orbitals as given | |
| 3 | NEVPT RDMs: E-product 2- and 3-RDMs, $f^\text{ca}$, $f^\text{ac}$, hole RDMs | `ferric-ci/src/rdm3.rs::{nevpt_rdms,hole_rdms,e_product_rdms}` |
| 4 | external integral blocks from the dense AO block (or the RI $B$) | |
| 5 | the eight classes, $E_\text{class} = -\sum_l N_l/(\Delta_l + K_l/N_l)$ over $|N_l| > 10^{-14}$ | `make_a16`, `make_a22`, … |

The memory of the AO block, the MO blocks, twelve $n_a^6$ intermediates and
the per-label tensors is reserved through one memory plan before the AO
block is built.

## Numerical anchors {#nevpt2-numerical-anchors}

| Quantity | Systems | Error | Test |
|---|---|---|---|
| class energies vs the numpy prototype on the same orbitals and CI vector | H2O CAS(4,4), N2 CAS(6,6) at 1.0977 and 2.2 Å / 6-31G | $\le 6.2\times10^{-14}$ Ha | `crates/ferric-ci/tests/nevpt2.rs` |
| vs PySCF `mrpt.NEVPT` on PySCF's own CASSCF | N2 (H2O $1.3\times10^{-8}$, stretched N2 $2.1\times10^{-8}$) | $\le 3.2\times10^{-8}$ Ha | `crates/ferric-ci/tests/nevpt2.rs` |
| one closed-shell active orbital vs MP2 pair sums | H2O / 6-31G | $8.1\times10^{-15}$ Ha | `crates/ferric-ci/tests/nevpt2.rs` |
| classes with an empty external space | H2, LiH | exactly 0 | `crates/ferric-ci/tests/nevpt2.rs` |

The third row is the single-determinant limit. With one doubly occupied
active orbital on RHF orbitals the reference is the RHF determinant, and
the test compares the classes with MP2 pair sums and the closed-form `Srs`
(`Sijrs` is MP2 by construction, as shown above). The fourth row is the
limit in which a class has no labels: when the core or the virtual space is
empty, every class that needs a label from it is zero exactly, while the
others stay nonzero.

The PySCF bar propagates each code's CASSCF convergence into the class
energies; inside a degenerate group of core or virtual orbitals the pair
classes depend on the eigenvectors chosen, which moves N2's `Srs` by up to
$1.7\times10^{-7}$ Ha between codes.

### What is proven and what is not {#nevpt2-proven}

The details of `crates/ferric-ci/tests/nevpt2.rs`:

| Check | Systems | Measured | Bar |
|---|---|---|---|
| exact zeros of every class whose external space is empty, the non-empty classes nonzero | H2 CAS(2,4) and CAS(2,2); LiH with 1 core orbital, CAS(2,5) | exactly 0 | |
| one-orbital closed-shell active space on RHF orbitals vs the MP2 pair sums and the closed-form `Srs` | H2O / 6-31G | $8.1\times10^{-15}$ Ha | $10^{-10}$ |
| the numpy prototype's Wick construction on the same ferric CASSCF orbitals and CI vector | H2O CAS(4,4); N2 CAS(6,6) at 1.0977 and 2.2 Å / 6-31G | $6.2\times10^{-14}$ Ha | $2.3\times10^{-10}$, $4.6\times10^{-9}$ |
| `pyscf.mrpt.NEVPT` on PySCF's own CASSCF | H2O, N2, stretched N2 | $1.3\times10^{-8}$, $3.2\times10^{-8}$, $2.1\times10^{-8}$ Ha | the propagated CASSCF-convergence bar of each class; largest ratio 0.88 (stretched N2 `Sr`, where PySCF's gradient is $9.8\times10^{-7}$) |

Five mutations fail it: a sign flip in `make_a12` (caught through `Sir`); a
$p\leftrightarrow q$ swap in one `make_a12` RDM index; the dropped `f3ca`
term of `make_a16` (caught through `Sr`); $F^I$ in place of $F^I + F^A$ for
the orbital energies (caught by the MP2 anchor, the prototype and PySCF);
and `Sr` gated on the core space (caught by the zeros anchor).

## Sources {#nevpt2-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-ci/theory.md`, "SC-NEVPT2" | definitions, orbitals, the classes and intermediates as coded, cost |
| `scripts/nevpt2_sympy.py` | claims D, R, T and C |
| Angeli, Cimiraglia and Malrieu, JCP 117, 9138 (2002) | the method |
| `scripts/prototypes/nevpt2_derivation.md`, `nevpt2_proto.py` | the derivation note and the Wick-construction reference |
| PySCF `mrpt/nevpt2.py` | the factorization transcribed |

**Known limits (2026-11-15).** Single root, closed-shell active spaces only
($n_\alpha = n_\beta$; open-shell active spaces are refused). On a CAS-CI
reference the certificate is not graded. Dense $O(n_a^8)$ intermediates
bound the active-space size.
