# 18.4 Open-shell dielectric

<!-- section: 18.4; status: ready; registry: u-pdep-rpa;
     theory: crates/ferric-rpa/theory.md#"Open-Shell (Spin-Summed) Dielectric" -->

For an open-shell reference the $\alpha$ and $\beta$ electrons have different
orbitals, orbital energies and occupations, but direct RPA has no exchange,
so the two spins couple only through the Coulomb interaction of their
density responses. The dielectric matrix is then a sum of one response per
spin in one shared auxiliary basis, and everything in section 18.3 carries
over with the per-spin prefactor 2 in place of the closed-shell 4.

| Index | Range | Code name |
|---|---|---|
| $\sigma$ | spin, $\alpha$ or $\beta$ | |
| $i, a$ | $\alpha$ occupied, virtual | |
| $\bar\imath, \bar a$ | $\beta$ occupied, virtual | |
| $P, Q$ | auxiliary functions, shared by both spins | `naux` |
| $\xi, \eta$ | PDEP modes of the spin-summed matrix | `n_keep` |

| Entry point | Location |
|---|---|
| Rust | `ferric-rpa/src/lib.rs::run_u_pdep_rpa` |
| Kernel | `ferric-rpa/src/sternheimer.rs::{dielectric_apply_unrestricted,dielectric_matrix_unrestricted}` |
| Frequency loop | `ferric-rpa/src/energy.rs::eval_eigenvalues_at_frequencies_unrestricted` |
| CLI | `method.kind = "pdep-rpa"` with multiplicity > 1 (UHF reference, energy only) |
| Python | none; `run_pdep_rpa` is closed shell only |

## 18.4.1 The spin-summed dielectric

$$ \tilde{\boldsymbol\varepsilon}_U(iu) = \mathbf 1 + \boldsymbol\Pi_\alpha(iu) + \boldsymbol\Pi_\beta(iu), \qquad \Pi^\alpha_{PQ}(iu) = 2\sum_{ia}B^P_{ia}\,\frac{\Delta_i^a}{u^2 + (\Delta_i^a)^2}\,B^Q_{ia}, \tag{18.12} $$

and $\boldsymbol\Pi_\beta$ the same over $\bar\imath\bar a$ with the $\beta$
factor $B^P_{\bar\imath\bar a}$ and gaps $\Delta_{\bar\imath}^{\bar a}$. The
prefactor 2 is the pair of poles at $\pm\Delta$ alone; the closed-shell 4 of
(18.7) is this 2 times the spin sum. With identical $\alpha$ and $\beta$
channels $\boldsymbol\Pi_\alpha + \boldsymbol\Pi_\beta$ is the closed-shell
$\boldsymbol\Pi$, so a UHF singlet reproduces section 18.3.

$$ E_c = \frac{1}{2\pi}\int_0^\infty\Big[\ln\det\tilde{\boldsymbol\varepsilon}_U(iu) - \operatorname{tr}\big(\boldsymbol\Pi_\alpha + \boldsymbol\Pi_\beta\big)\Big]\,du \tag{18.13} $$

equals the spin-orbital plasmon definition
$\tfrac12\sum_n(\Omega_n - A_{nn})$ with $\mathbf A = \boldsymbol\Delta + \mathbf K$,
$\mathbf B = \mathbf K$ over both spin blocks and
$K_{i_\sigma a_\sigma, j_\tau b_\tau} = (i_\sigma a_\sigma|j_\tau b_\tau)$.

> **Verified equation 18.13** · the spin-summed dielectric
>
> With different $\alpha$ and $\beta$ orbitals, gaps and occupations, (18.13)
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

## 18.4.2 The algorithm

| Step | Quantity | Difference from section 18.3 |
|---|---|---|
| 1 | reference | UHF; or ROHF, semicanonicalized per spin so each spin's occupied and virtual Fock blocks are diagonal |
| 2 | $B_\alpha$, $B_\beta$ in one auxiliary basis | one intermediates build per spin, the same frozen-core count in each |
| 3 | static modes of $\tilde{\boldsymbol\varepsilon}_U(0)$ | the matrix-vector product is two closed-shell-shaped applications, $\mathbf v + \boldsymbol\Pi_\alpha\mathbf v + \boldsymbol\Pi_\beta\mathbf v$, prefactor 2 each |
| 4 | truncation | the same rule (18.9) |
| 5 | frequency loop | always the eigenvalue route: per node, the eigenvalues of $\mathbf U^{\mathsf T}\tilde{\boldsymbol\varepsilon}_U\mathbf U$ and (18.11); no log-det branch |

The Laplace $\chi_0$ backend (section 18.5) builds one Laplace quadrature per
spin on that spin's gap range. `run_diagnostics` assembles
$\tilde{\boldsymbol\varepsilon}_U(iu_k)$ densely at every node as a check.

**Refused inputs.** A restricted (closed-shell) result passed to the
unrestricted driver; a spin whose highest occupied orbital is not below its
lowest virtual (the frequency integral is undefined); and the Boys-screened
$\chi_0$ (`chi0_sparsity = "boys"`, or `"auto"` when it resolves to Boys),
whose screened tiles are built from restricted orbitals.

## 18.4.3 Options

The `[rpa]` options of section 18.3 apply. The CLI runs UHF with maximum
overlap after five DIIS iterations before the RPA step, to keep the
occupation pattern of an open-shell atom from reordering.

## 18.4.4 Numerical anchors

From `site/src/reference/validation.md` (open-shell PDEP-RPA row):

| Quantity | Systems | Error | Test |
|---|---|---|---|
| $E_c$ vs PySCF `gw.urpa.URPA`, same UHF, aux basis and 40-node grid | OH, CH3, NH2 (doublets), O2 (triplet) / cc-pVDZ; OH / aug-cc-pVDZ | $\le 9.2\times10^{-11}$ Ha | `crates/ferric-rpa/tests/validation_urpa.rs` |
| Truncated, against a numpy emulation | OH | $6.4\times10^{-11}$ Ha | `urpa_truncation_control_oh` |
| Unrestricted vs restricted path at the closed-shell limit | H2O / cc-pVDZ | $3.2\times10^{-12}$ Ha | `urpa_closed_shell_limit_h2o_vs_pyscf_rpa` |

## 18.4.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-rpa/theory.md`, "Open-Shell (Spin-Summed) Dielectric" | (18.12), (18.13) |
| `scripts/rpa_drpa_pdep_sympy.py`, check group 8 | the proof |

**Known limits (2026-11-15).** Energy only; no gradient. CLI only. The
`theory.md` section says the frequency projection and trace-log integration
are "those of the closed-shell path"; they are, except that the open-shell
driver never takes the log-det branch.
