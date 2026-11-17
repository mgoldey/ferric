# Open-shell dielectric {#open-shell-dielectric}

<!-- status: ready; registry: u-pdep-rpa;
     theory: crates/ferric-rpa/theory.md#"Open-Shell (Spin-Summed) Dielectric" -->

In a radical or a triplet the $\alpha$ and $\beta$ electrons have different
orbitals, orbital energies and occupations, so they respond differently to
a potential. Both still feel the same potential. Direct RPA has no exchange,
so the two spins couple only through the Coulomb interaction of their
density responses. The dielectric matrix is then a sum of one response per
spin in one shared auxiliary basis, and everything in section @sec:pdep carries
over with the per-spin prefactor 2 in place of the closed-shell 4.

The plan: the spin-summed matrix and why its prefactor is 2 (@sec:open-shell-dielectric-spin-summed-dielectric), what
changes in the algorithm (@sec:open-shell-dielectric-algorithm), then options and anchors.

| Index | Range | Code name |
|---|---|---|
| $\sigma$ | spin, $\alpha$ or $\beta$ | |
| $i, a$ | $\alpha$ occupied, virtual | |
| $\bar\imath, \bar a$ | $\beta$ occupied, virtual | |
| $P, Q$ | auxiliary functions, shared by both spins | `naux` |
| $\xi, \eta$ | PDEP modes of the spin-summed matrix | `n_keep` |

| Entry point | Location |
|---|---|
| Rust | `ferric-rpa/src/lib.rs::{run_u_pdep_rpa,run_u_pdep_rpa_from_parts,u_pdep_rpa_core}` |
| Dense check | `ferric-rpa/src/diagnostics.rs::u_ri_drpa_energy` |
| Kernel | `ferric-rpa/src/sternheimer.rs::{dielectric_apply_unrestricted,dielectric_matrix_unrestricted}` |
| Frequency loop | `ferric-rpa/src/energy.rs::eval_eigenvalues_at_frequencies_unrestricted` |
| CLI | `method.kind = "pdep-rpa"` with multiplicity > 1 (UHF reference, energy only) |
| Python | none; `run_pdep_rpa` is closed shell only |

## The spin-summed dielectric {#open-shell-dielectric-spin-summed-dielectric}

Start from spin orbitals. Each excitation $i_\sigma \to a_\sigma$ responds
to a frequency $iu$ with the factor $\Delta/(u^2 + \Delta^2)$, times 2 for
the pair of poles at $\pm\Delta$. Its coupling to every other excitation,
of either spin, is $(i_\sigma a_\sigma|j_\tau b_\tau)$, which factors through
the one auxiliary basis as $\sum_P B^P_{i_\sigma a_\sigma}B^P_{j_\tau b_\tau}$.
Sum the response over the excitations of each spin:

$$ \tilde{\boldsymbol\varepsilon}_U(iu) = \mathbf 1 + \boldsymbol\Pi_\alpha(iu) + \boldsymbol\Pi_\beta(iu), \qquad \Pi^\alpha_{PQ}(iu) = 2\sum_{ia}B^P_{ia}\,\frac{\Delta_i^a}{u^2 + (\Delta_i^a)^2}\,B^Q_{ia}, \label{eq:open-shell-dielectric-1} $$

and $\boldsymbol\Pi_\beta$ the same over $\bar\imath\bar a$ with the $\beta$
factor $B^P_{\bar\imath\bar a}$ and gaps $\Delta_{\bar\imath}^{\bar a}$. Both
factors are built with one dressing $M$. The prefactor 2 is the pair of
poles at $\pm\Delta$ alone; the closed-shell 4 of (@eq:pdep-1) is this 2 times the
spin sum.

Check the closed-shell limit. With identical $\alpha$ and $\beta$ channels,
$\boldsymbol\Pi_\alpha + \boldsymbol\Pi_\beta = 2\boldsymbol\Pi_\alpha$, which
is the closed-shell $\boldsymbol\Pi$ with its 4. So a UHF singlet reproduces
section @sec:pdep.

The energy is the frequency integral of section @sec:drpa-three-ways with the spin-summed
matrix:

$$ E_c = \frac{1}{2\pi}\int_0^\infty\Big[\ln\det\tilde{\boldsymbol\varepsilon}_U(iu) - \operatorname{tr}\big(\boldsymbol\Pi_\alpha + \boldsymbol\Pi_\beta\big)\Big]\,du \label{eq:rpa-updep-spin-summed} $$

It equals the spin-orbital plasmon definition
$\tfrac12\sum_n(\Omega_n - A_{nn})$ with $\mathbf A = \boldsymbol\Delta + \mathbf K$,
$\mathbf B = \mathbf K$ over both spin blocks and
$K_{i_\sigma a_\sigma, j_\tau b_\tau} = (i_\sigma a_\sigma|j_\tau b_\tau)$.
The four steps of @sec:drpa-three-ways-imaginary-frequency go through with spin-orbital indices, the per-spin
prefactor 2 in $g$, and the spin-orbital $\mathbf K$ in place of
$\tfrac12\mathbf K$. The open-shell matrix has no
singlet/triplet split, because the $\alpha$ and $\beta$ blocks differ.

> **Verified equation @eq:rpa-updep-spin-summed** · the spin-summed dielectric
>
> With different $\alpha$ and $\beta$ orbitals, gaps and occupations, (@eq:rpa-updep-spin-summed)
> equals the spin-orbital definition; with identical channels the per-spin
> prefactor 2 gives the closed-shell $\boldsymbol\Pi$.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_drpa_pdep_sympy.py`, checks `u-freq-vs-definition` and `u-closed-shell-limit` |
> | Negative control | the runner's mutation of the per-spin prefactor $2 \to 4$ turns `u-freq-vs-definition` red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/sternheimer.rs::{dielectric_apply_unrestricted,build_scale_factors_with_prefactor}` |
> | Code test | `urpa_oh_vs_pyscf_urpa` in `crates/ferric-rpa/tests/validation_urpa.rs` |
>
> <!-- verified: id=rpa.updep.spin-summed; script=scripts/rpa_drpa_pdep_sympy.py; claim=u-freq-vs-definition; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/sternheimer.rs::{dielectric_apply_unrestricted,build_scale_factors_with_prefactor}; rust_test=ferric-rpa/tests/validation_urpa.rs::urpa_oh_vs_pyscf_urpa; registry=u-pdep-rpa -->

The code applies (@eq:open-shell-dielectric-1) to a block of vectors $\mathbf V$ one spin at a
time. The loop of `ferric-rpa/src/sternheimer.rs::dielectric_apply_unrestricted`,
unchanged:

<!-- excerpt: ferric-rpa/src/sternheimer.rs::dielectric_apply_unrestricted -->
```rust
let mut out: Array2<f64> = v_mat.to_owned();

for chan in [chan_a, chan_b] {
    let RpaChannel {
        b_ov,
        eps_occ,
        eps_vir,
    } = *chan;
    let scale = build_scale_factors_with_prefactor(eps_occ, eps_vir, omega, 2.0);
    let nov = scale.len();
    assert_eq!(b_ov.shape()[1], nov);

    let mut y: Array2<f64> = v_mat.t().dot(b_ov);
    let scale_row = scale.view().insert_axis(Axis(0));
    Zip::from(&mut y)
        .and_broadcast(scale_row)
        .for_each(|x, &s| *x *= s * s);

    // out += B_ov · y^T
    general_mat_mul(1.0, b_ov, &y.t(), 1.0, &mut out);
}
```

| Code | Equation |
|---|---|
| `out` before the loop | $\mathbf 1\,\mathbf V$ |
| `chan_a`, `chan_b` | $\sigma = \alpha, \beta$ |
| `scale` | $\sqrt{2\Delta/(u^2 + \Delta^2)}$ per pair of spin $\sigma$ |
| `y` after the `Zip` | $\mathbf V^{\mathsf T}B_\sigma$ times $2\Delta/(u^2 + \Delta^2)$ |
| `general_mat_mul` | $+\boldsymbol\Pi_\sigma\mathbf V$ |

- **The 2.0 is the per-spin prefactor.** The closed-shell path passes 4.
  The proof script's mutation $2 \to 4$ makes the open-shell energy wrong,
  so the constant is pinned.
- **The two spins may have different lengths.** `b_ov` is
  $N_{\mathrm{aux}} \times n_o^\sigma n_v^\sigma$, and the occupied and
  virtual counts differ between spins. Only the auxiliary dimension is
  shared, and that is the dimension of `out`.

## The algorithm {#open-shell-dielectric-algorithm}

| Step | Quantity | Difference from section @sec:pdep |
|---|---|---|
| 1 | reference | UHF; or ROHF, semicanonicalized per spin so each spin's occupied and virtual Fock blocks are diagonal |
| 2 | $B_\alpha$, $B_\beta$ in one auxiliary basis | one intermediates build per spin, the same frozen-core count in each |
| 3 | static modes of $\tilde{\boldsymbol\varepsilon}_U(0)$ | the dense full-rank `eigh` (default `Eigensolver::Lanczos`), or Davidson from the identity seed; the matrix-vector product is two closed-shell-shaped applications, $\mathbf v + \boldsymbol\Pi_\alpha\mathbf v + \boldsymbol\Pi_\beta\mathbf v$, prefactor 2 each |
| 4 | truncation | the same rule (@eq:pdep-3) |
| 5 | frequency loop | always the eigenvalue route: per node, the eigenvalues of $\mathbf U^{\mathsf T}\tilde{\boldsymbol\varepsilon}_U\mathbf U$ and (@eq:rpa-pdep-projection); no log-det branch |

Step 1 semicanonicalizes the ROHF orbitals because the kernel uses one gap
$\Delta_i^a = \varepsilon_a - \varepsilon_i$ per pair, and that scalar is
the right denominator only in a basis where the Fock blocks are diagonal.

The open-shell driver, `u_pdep_rpa_core`, always takes the static-eigenbasis
route: there is no aux-basis log-det route and no LU log-det branch on the
open-shell path.
The Laplace $\chi_0$ backend (section @sec:laplace-ao-chi0) builds one Laplace quadrature per
spin on that spin's gap range. `run_diagnostics` assembles
$\tilde{\boldsymbol\varepsilon}_U(iu_k)$ densely at every node as a check
(`diagnostics::u_ri_drpa_energy`).

`run_u_pdep_rpa_from_parts` starts from pre-built per-spin intermediates
instead of a molecule. It accepts only full rank (`trunc_thresh` $= 0$), the
dense $\chi_0$ (`chi0_sparsity` and `chi0_backend` both `Dense`) and
`Eigensolver::Lanczos`, and it refuses $\alpha$ and $\beta$ intermediates
with different auxiliary dimensions or different metric factors
`v_inv_sqrt`: the eigenpotential back-transform reads the $\alpha$ factor,
so both spins must share one $M$.

**Refused inputs.**

- A restricted (closed-shell) result passed to the unrestricted driver.
- A spin whose highest occupied orbital is not below its lowest virtual.
  Some $\Delta_i^a \le 0$ then, and the frequency integral is undefined.
- The Boys-screened $\chi_0$ (`chi0_sparsity = "boys"`, or `"auto"` when it
  resolves to Boys), whose screened tiles are built from restricted orbitals.

## Options {#open-shell-dielectric-options}

The `[rpa]` options of section @sec:pdep apply. The CLI runs UHF with maximum
overlap after five DIIS iterations before the RPA step, to keep the
occupation pattern of an open-shell atom from reordering.

## Numerical anchors {#open-shell-dielectric-numerical-anchors}

From `site/src/reference/validation.md` (open-shell PDEP-RPA row):

| Quantity | Systems | Error | Test |
|---|---|---|---|
| $E_c$ vs PySCF `gw.urpa.URPA`, same UHF, aux basis and 40-node grid | OH, CH3, NH2 (doublets), O2 (triplet) / cc-pVDZ; OH / aug-cc-pVDZ | $\le 9.2\times10^{-11}$ Ha | `crates/ferric-rpa/tests/validation_urpa.rs` |
| Truncated, against a numpy emulation | OH | $6.4\times10^{-11}$ Ha | `urpa_truncation_control_oh` |
| Unrestricted vs restricted path at the closed-shell limit | H2O / cc-pVDZ | $3.2\times10^{-12}$ Ha | `urpa_closed_shell_limit_h2o_vs_pyscf_rpa` |

The last row is the closed-shell limit of @sec:open-shell-dielectric-spin-summed-dielectric run through the real code.

## Sources {#open-shell-dielectric-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-rpa/theory.md`, "Open-Shell (Spin-Summed) Dielectric" | (@eq:open-shell-dielectric-1), (@eq:rpa-updep-spin-summed) |
| `scripts/rpa_drpa_pdep_sympy.py`, check group 8 | the proof |

**Known limits (2026-11-15).** Energy only; no gradient. CLI only. The
open-shell driver never takes a log-det branch (`theory.md` states this).
