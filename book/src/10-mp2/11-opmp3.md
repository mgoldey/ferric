# 10.11 Orbital-partitioned Padé MP3

<!-- section: 10.11; status: ready; registry: opmp3;
     theory: crates/ferric-mp2/theory.md#"6.2 Orbital-partitioned Padé MP3 (OP-MP3)" -->

MP3 overshoots where MP2 already overshoots, and the scalar Padé
$E_2^2/(E_2 - E_3)$ that tames it is not size-consistent. OP-MP3 (Bystrom,
Berkelbach and Hait, arXiv:2609.25330) applies the Padé to matrices instead:
the second- and third-order energies are written as traces of symmetric
matrices over occupied spin orbitals, and the resummation is a matrix
function of those. A matrix function of a block-diagonal pair is
block-diagonal, so the energy of two non-interacting fragments is the sum
of their energies. The amplitudes are the spin-blocked ones of section
10.9.3; this section adds the matrices, the Padé and the reference.

| Symbol | Meaning | Code name |
|---|---|---|
| $c^{(1)}, c^{(2)}$ | first- and second-order doubles, spin blocked ($\alpha\alpha$, $\beta\beta$, $\alpha\beta$ stored $[i,J,a,B]$) | `UMp3Result::{c1,c2}` |
| $K$ | $\langle ij\Vert ab\rangle$ ($\alpha\alpha$, $\beta\beta$), $(ia\vert JB)$ ($\alpha\beta$) | `UMp3Result::ints` |
| $W^{(n)}_\sigma$ | occupied-space matrix of order $n$, spin $\sigma$; SS and OS channels | `WChannels::{ss,os}` |
| $P(A,B)$ | matrix Padé $A(A-B)^{-1}A$ | `pade` |
| $x$, $x_{ss}$, $x_{os}$ | scale on $W^{(3)}$; same- and opposite-spin scales | `x_ss` |
| $E_{\mathrm{NBS}}$ | non-Brillouin singles energy | `e_nbs` |

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/opmp3_driver.rs::{run_opmp3,opmp3_on_reference}`; matrices and models in `ferric-mp2/src/opmp3.rs::{w_channels,pade,op_mp3,sop_mp3,ssd_opmp3}`; reference in `ferric-mp2/src/opmp3_reference.rs::opmp3_reference` |
| CLI | `method.kind = "opmp3"` with `[opmp3] reference` (`examples/oh-opmp3-pbe0.toml`) |
| Python | `run_opmp3(reference=, x_ss=, auxbasis=)` |

## 10.11.1 The occupied-space matrices

In spin orbitals, with the amplitudes $t^{(1)} = c^{(1)}$ and
$t^{(2)} = c^{(2)}$,

$$ W^{(n)}_{ij} = \tfrac18\sum_{kab}\Big(t^{(n-1)}_{ikab}\langle jk\Vert ab\rangle + t^{(n-1)}_{jkab}\langle ik\Vert ab\rangle\Big), \qquad \operatorname{Tr}W^{(n)} = E^{(n)} . \tag{10.26} $$

Spin is conserved along the pair, so $W$ has no $\alpha\beta$ block. Each
spin block splits by the spin of the summed $k$ into a same-spin (SS) and
an opposite-spin (OS) channel, which in the stored blocks of section 10.9.3
read

$$ W^{SS}_{\sigma,ij} = \tfrac18\sum_{kab\in\sigma}c^{\sigma\sigma}_{ikab}K^{\sigma\sigma}_{jkab} + (i\leftrightarrow j), $$
$$ W^{OS}_{\alpha,ij} = \tfrac14\sum_{KaB}c^{\alpha\beta}_{iKaB}K^{\alpha\beta}_{jKaB} + (i\leftrightarrow j), \qquad W^{OS}_{\beta,IJ} = \tfrac14\sum_{kaB}c^{\alpha\beta}_{kIaB}K^{\alpha\beta}_{kJaB} + (I\leftrightarrow J). \tag{10.27} $$

The OS prefactor is $\tfrac14$ because, with $k$ of the other spin, the two
orderings $(a\bar B)$ and $(\bar B a)$ of the virtual pair map onto the one
stored $\alpha\beta$ block with the same sign. The traces are the channel
energies of (10.24): $\operatorname{Tr}W^{SS} = E_{\alpha\alpha} + E_{\beta\beta}$
and $\operatorname{Tr}W^{OS} = E_{\alpha\beta}$, half of it in each spin's OS
block.

<!-- excerpt: ferric-mp2/src/opmp3.rs::w_channels -->
```rust
WChannels {
    ss: [same(&c.aa, &k.aa), same(&c.bb, &k.bb)],
    os: [
        symmetrized_contraction(&c.ab, &k.ab, 0.25),
        symmetrized_contraction(&beta_first(&c.ab), &beta_first(&k.ab), 0.25),
    ],
}
```

| Symbol | Code |
|---|---|
| $f\,(Y + Y^{\mathsf T})$, $Y_{ij} = \sum_{\text{rest}}c_{i,\text{rest}}K_{j,\text{rest}}$ | `symmetrized_contraction(c, k, f)` |
| $\tfrac18$ SS, $\tfrac14$ OS | `0.125` inside `same`, `0.25` |
| $c^{\alpha\beta}_{kIaB}$ read with $I$ first | `beta_first` (axes $[1,0,2,3]$) |

A spin with fewer than two occupied orbitals has no same-spin pair; its
$W^{SS}$ is set to exact zeros, since its $c^{\sigma\sigma}_{iiab}$ vanish only
to rounding.

> **Verified equation 10.27** · the SS/OS split of (10.26)
>
> In a spin-orbital model whose amplitudes and integrals are independent
> symbols, constrained only by antisymmetry and spin conservation, the
> $k$-in-$\bar\sigma$ part of each spin block of (10.26) equals the OS form of
> (10.27) with the $\tfrac14$ prefactor and the plain $\alpha\beta$ integrals;
> the $k$-in-$\sigma$ part equals the SS form; the $\alpha\beta$ block of $W$ is
> zero; and the traces are the `u_mp3` channel energies.
>
> | | |
> |---|---|
> | Proof | `scripts/opmp3_identities_sympy.py`, checks a1–a4 (claim a3) |
> | Negative control | mutation `os_eighth` ($\tfrac18$ on the OS channel) turns a3 and a4 red; `w_unsymmetrized` turns a2–a4 red |
> | Runner | `crates/ferric-python/tests/test_opmp3_identities_sympy.py` |
> | Code | `ferric-mp2/src/opmp3.rs::w_channels` |
> | Code test | `eq18_channels_equal_the_k_spin_split_of_spin_orbital_eq11a` in `crates/ferric-mp2/tests/opmp3.rs` (OH UHF: the blocks expanded to full spin-orbital tensors) |
>
> <!-- verified: id=opmp3.w.channels; script=scripts/opmp3_identities_sympy.py; claim=a3; runner=crates/ferric-python/tests/test_opmp3_identities_sympy.py; code=ferric-mp2/src/opmp3.rs::{w_channels,symmetrized_contraction,beta_first}; rust_test=ferric-mp2/tests/opmp3.rs::eq18_channels_equal_the_k_spin_split_of_spin_orbital_eq11a; registry=opmp3 -->

> **Verified equation 10.26** · $\operatorname{Tr}W^{(n)} = E^{(n)}$
>
> Symbolically, $\operatorname{Tr}W = \tfrac14\sum c\,\langle ij\Vert ab\rangle$
> for any antisymmetric amplitudes (check d1). Numerically, on a random
> antisymmetrized Hamiltonian in six spin orbitals with three occupied,
> $\operatorname{Tr}W^{(2)}$ and $\operatorname{Tr}W^{(3)}$ built from $c^{(1)}$ and
> $c^{(2)}$ equal the brute-force Rayleigh–Schrödinger $E_2$ and
> $E_3 = \langle\psi_1|V - E_1|\psi_1\rangle$ in the 20-determinant space to
> $10^{-12}$ relative (check d2), an independent route.
>
> | | |
> |---|---|
> | Proof | `scripts/opmp3_identities_sympy.py`, checks d1, d2 (claim d2) |
> | Negative control | mutation `w_unsymmetrized` (the $(i\leftrightarrow j)$ half dropped) turns d1 and d2 red |
> | Runner | `crates/ferric-python/tests/test_opmp3_identities_sympy.py` |
> | Code | `ferric-mp2/src/opmp3.rs::{w_matrices,WChannels}` |
> | Code test | `series_limit_equals_mp3_and_traces_equal_channel_energies` in `crates/ferric-mp2/tests/opmp3.rs` |
>
> <!-- verified: id=opmp3.w.trace; script=scripts/opmp3_identities_sympy.py; claim=d2; runner=crates/ferric-python/tests/test_opmp3_identities_sympy.py; code=ferric-mp2/src/opmp3.rs::{w_matrices,WChannels}; rust_test=ferric-mp2/tests/opmp3.rs::series_limit_equals_mp3_and_traces_equal_channel_energies; registry=opmp3 -->

## 10.11.2 The matrix Padé and the models

For symmetric $A = W^{(2)}$ and $B = xW^{(3)}$,

$$ P(A,B) = A\,(A-B)^{-1}A, \qquad \operatorname{Tr}P(A,tB) = \operatorname{Tr}A + t\operatorname{Tr}B + t^2\operatorname{Tr}\big[BA^{-1}B\big] + O(t^3) . \tag{10.28} $$

The first two Taylor coefficients are $E^{(2)}$ and $E^{(3)}$, so the series
limit of every model is MP3. For a $1\times 1$ block $P$ is the scalar Padé
$a^2/(a-b)$; for commuting $A$ and $B$ each common eigenvector gets it. The
code diagonalizes the symmetrized $A - B = V\Lambda V^{\mathsf T}$ and forms
$P = (AV)\Lambda^{-1}(AV)^{\mathsf T}$:

<!-- excerpt: ferric-mp2/src/opmp3.rs::pade -->
```rust
let av = ak.dot(&v);
let mut scaled = av.clone();
for (mut col, &l) in scaled.columns_mut().into_iter().zip(lam.iter()) {
    col /= l;
}
let pk = scaled.dot(&av.t());
```

| Symbol | Code |
|---|---|
| $A$ on its non-zero rows | `ak` |
| $V$, $\Lambda$ | `v`, `lam` (from `eigh` of $\tfrac12[(A-B) + (A-B)^{\mathsf T}]$) |
| $P$ on the kept rows | `pk` |

Rows on which $A$ is identically zero (a channel with no pairs) are left
out, so $P(0,0) = 0$. An eigenvalue of $A - B$ below $10^{-12}\max|\lambda|$
is a singular denominator and returns an error.

The models are traces over both spins:

$$ E_c^{\text{OP-MP3}(x)} = \sum_\sigma\operatorname{Tr}P\big(W^{(2)}_\sigma,\;xW^{(3)}_\sigma\big), \qquad W_\sigma = W^{SS}_\sigma + W^{OS}_\sigma , $$
$$ E_c^{\text{SOP-MP3}(x_{ss},x_{os})} = \sum_\sigma\Big[\operatorname{Tr}P\big(W^{(2),SS}_\sigma, x_{ss}W^{(3),SS}_\sigma\big) + \operatorname{Tr}P\big(W^{(2),OS}_\sigma, x_{os}W^{(3),OS}_\sigma\big)\Big], \qquad E_c^{\text{SSD-OPMP3}(x_{ss})} = E_c^{\text{SOP-MP3}(x_{ss},1)} . \tag{10.29} $$

`Opmp3Models::evaluate` reports MP2 ($\operatorname{Tr}W^{(2)}$), MP3, the
scalar Padé (P-MP3), OP-MP3 and SOP-MP3 at $x = 1$, and SSD-OPMP3 at the
run's $x_{ss}$. Unless the input sets it, $x_{ss}$ is the paper's printed
value for the reference (`PAPER_X_SS`):

| Reference | hf | pbe0 | pbe | spw92 | r2scan | wb97x-v | wb97m-v | koomp2 |
|---|---|---|---|---|---|---|---|---|
| $x_{ss}$ | 0 | 0.50 | 0.70 | 0.75 | 0.60 | 0.50 | 0.50 | 0.45 |

A reference with no printed value and no explicit `x_ss` is an error.

> **Verified equation 10.28** · Padé series, scalar limit, size consistency
>
> For symbolic symmetric $2\times2$ $A$, $B$, $P(A,tB) = A + tB + t^2BA^{-1}B + O(t^3)$
> element by element (b1); the $1\times1$ case is $a^2/(a-b)$ (b2); commuting
> blocks give the per-eigenvalue scalar Padé (b3). For a block-diagonal pair
> $\operatorname{Tr}P$ is the sum of the blocks' traces (c1), while the scalar
> Padé of the summed traces is not (c2: $-8/3$ against $-3$ on the
> counterexample $A_1 = A_2 = -1$, $B_1 = -\tfrac12$, $B_2 = 0$), except for
> identical copies (c3). Numerically, $\operatorname{Tr}P$ equals the paper's
> eigen route $\sum_i(S^{-1}AS)_{ii}/(1-\zeta_i)$, $\zeta = \operatorname{eig}(A^{-1}B)$,
> to $10^{-12}$ (check N).
>
> | | |
> |---|---|
> | Proof | `scripts/opmp3_identities_sympy.py`, checks b1–b3, c1–c3, N (claim b1) |
> | Negative control | `pade_swapped` ($(A-B)A^{-1}(A-B)$) turns b1–b3 and N red; `scalar_in_block` turns b1, b3, c1 and N red |
> | Runner | `crates/ferric-python/tests/test_opmp3_identities_sympy.py` |
> | Code | `ferric-mp2/src/opmp3.rs::{pade,pade_trace,op_mp3,sop_mp3,ssd_opmp3,scalar_pade}` |
> | Code test | `pade_taylor_coefficients_are_e2_e3_and_the_second_order_term` and `models_are_size_consistent_and_the_scalar_pade_is_not` in `crates/ferric-mp2/tests/opmp3.rs` |
>
> <!-- verified: id=opmp3.pade; script=scripts/opmp3_identities_sympy.py; claim=b1; runner=crates/ferric-python/tests/test_opmp3_identities_sympy.py; code=ferric-mp2/src/opmp3.rs::{pade,pade_trace,op_mp3,sop_mp3,ssd_opmp3,scalar_pade}; rust_test=ferric-mp2/tests/opmp3.rs::pade_taylor_coefficients_are_e2_e3_and_the_second_order_term; registry=opmp3 -->

## 10.11.3 The reference

The perturbation theory is built on the Hartree–Fock Fock operator of the
reference density, also for a Kohn–Sham reference. From the converged spin
densities $D_\sigma$,

$$ F_\sigma = h + J[D_\alpha + D_\beta] - K[D_\sigma], \qquad E_{\mathrm{HF}}[\rho] = E_{nn} + \tfrac12\sum_\sigma\operatorname{Tr}\big[D_\sigma(h + F_\sigma)\big], $$
$$ E_{\mathrm{NBS}} = \sum_\sigma\sum_{i\,\text{active},\,a}\frac{(F^\sigma_{ia})^2}{\varepsilon_{i\sigma} - \varepsilon_{a\sigma}}, \qquad E = E_{\mathrm{HF}}[\rho] + E_{\mathrm{NBS}} + E_c . \tag{10.30} $$

$h$ is the SCF's own one-electron operator. The occupied and virtual blocks
of each $F_\sigma$ are diagonalized separately in the SCF orbitals
(semicanonicalization), which leaves the density and $E_{\mathrm{HF}}[\rho]$
unchanged and gives the orbital energies of the denominators of (10.24);
what remains of $F^\sigma_{ia}$ enters through $E_{\mathrm{NBS}}$. The driver
runs RHF or RKS for a singlet and ROHF or ROKS otherwise, with exact
$J$ and $K$ and an unconverged SCF as an error; `opmp3_on_reference` also
takes a converged UHF, for which $F^\sigma_{ia} = 0$ and $E_{\mathrm{NBS}} = 0$.
The amplitude stage uses exact four-centre integrals by default, or RI when
an auxiliary basis is given.

<!-- excerpt: ferric-mp2/src/opmp3_reference.rs::singles_energy -->
```rust
for i in first..self.nocc {
    for (a, &f) in self.f_ov.row(i).iter().enumerate() {
        e += f * f / (self.eps[i] - self.eps[self.nocc + a]);
    }
}
```

No proof script covers (10.30). It is checked against independent codes
instead: PySCF's ROHF $E_2$ and $E_3$ channels on OH, and the Q-Chem 6.4
values of the paper's own tests for $E_{\mathrm{HF}}[\rho]$, $E_{\mathrm{NBS}}$,
$\operatorname{Tr}W^{(2)}$ and $\operatorname{Tr}W^{(3)}$ (table below).

## 10.11.4 What is anchored

| Quantity | Systems | Agreement | Bar | Test |
|---|---|---|---|---|
| $\operatorname{Tr}W^{(2)} + \operatorname{Tr}W^{(3)}$ vs MP2 + MP3 of `mp3.rs` (RI); channel traces vs `u_mp3` channel energies | H2O RHF, OH UHF | rounding | 1e-12 Ha (traces), 1e-10 Ha (vs `mp3.rs`) | `crates/ferric-mp2/tests/opmp3.rs` |
| Taylor coefficients of OP-MP3($x$): $E_2$, $E_3$, $\sum_\sigma\operatorname{Tr}W_3W_2^{-1}W_3$ | H2O | central-difference truncation | derived in the test | `crates/ferric-mp2/tests/opmp3.rs` |
| size consistency: every model additive, the scalar Padé not | N2 ⋯ CH4 at 100 Å / cc-pVDZ | SCF floor | 1e-9 Ha; scalar Padé must miss by > 1e-6 | `crates/ferric-mp2/tests/opmp3.rs` |
| reference path: $E_{\mathrm{HF}}[\rho]$, $E_{\mathrm{NBS}}$ (zero for RHF, UHF), ROHF $E_2$, $E_3$ channels vs PySCF | OH / cc-pVDZ | | derived in the test | `crates/ferric-mp2/tests/opmp3.rs` |
| $E_{\mathrm{HF}}[\rho]$, $E_{\mathrm{NBS}}$, $\operatorname{Tr}W^{(2)}$ (SS, OS), $\operatorname{Tr}W^{(3)}$ vs Q-Chem 6.4 | CH2 ³B1, CH3, CN, HCO, H2O, He, H / cc-pVDZ; HF and PBE0 references | HF: printed digits; PBE0: grid difference | half the last printed digit (HF); 1e-5 / 1e-4 Ha (PBE0) | `crates/ferric-mp2/tests/opmp3_qchem.rs` |
| resummation on the paper's own $W$ matrices vs its per-reaction W4-17 TAE errors | 12 reactions, HF, PBE0, κOOMP2 references | printed 1e-6 kcal/mol | 1e-5 kcal/mol | `crates/ferric-mp2/tests/opmp3_paper.rs` |

The grade in `site/src/reference/validation.md` is Smoke: no committed test
pins an end-to-end OP-MP3 total against an independent code.

## Known limits (2026-11-15)

- Energy only: no gradient.
- The reference is exact-$J/K$ only; the amplitude stage holds the spin-blocked
  MP3 tensors of section 10.9.3 and their memory peak.
- `PAPER_OP_MP3_X` (0.7890, the paper's fitted OP-MP3 scale on κOOMP2
  references) is defined in `opmp3.rs` but no run uses it: OP-MP3 is always
  reported at $x = 1$.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/theory.md`, "6.2 Orbital-partitioned Padé MP3 (OP-MP3)" | equations 10.26–10.30 |
| Bystrom, Berkelbach and Hait, arXiv:2609.25330 | the method, Eqs. 11a, 13, 14, 18–22, the $x_{ss}$ table |
| `crates/ferric-mp2/src/opmp3.rs`, `opmp3_reference.rs`, `opmp3_driver.rs` | implementation |
| `scripts/opmp3_identities_sympy.py` | proofs |
| `crates/ferric-mp2/tests/opmp3.rs`, `opmp3_qchem.rs`, `opmp3_paper.rs` | anchors |
