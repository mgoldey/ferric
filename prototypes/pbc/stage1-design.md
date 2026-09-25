# Stage 1 design: Gamma-point periodic RHF in Rust (2026-09-23)

Scope: closed-shell Gamma HF, all-electron, `exxdiv='ewald'` default, reusing `solve_rhf`. Lines = `spike/pbc-prototype`. Design only; nothing built.

## 1. SCF entry: inject S, h, E_nn

`solve_rhf(ctx, mol, prep, op, bounds, config)` (rhf.rs:753) gets S/h/V_nn only through
`driver::prepare` (rhf.rs:816 → driver.rs:85). It builds `overlap(prep)`, `hcore_ecp_with_external`
and `mol.nuclear_repulsion()` at driver.rs:92-99. UHF (uhf.rs:438) and ROHF (rohf.rs:273) call the same function.

**Proposal (smallest diff):**
```rust
pub struct PeriodicInjection<'a> {          // ferric-scf, pub
    pub s: Array2<f64>, pub h: Array2<f64>, pub vnn: f64,
    pub j: Box<dyn JBuilder + 'a>, pub k: Box<dyn KBuilder + 'a>,
}
pub fn solve_rhf_injected(ctx, mol, prep, op, bounds, config, inj: PeriodicInjection) -> Result<ScfResult>
```
`solve_rhf` body → `solve_rhf_impl(.., Option<PeriodicInjection>)`; `solve_rhf` passes `None`. `prepare` gains
`pre: Option<(S, h, vnn)>` replacing driver.rs:92-99 when present. `None` runs identical code ⇒ molecular
byte-identity. `mol`=`cell.mol()`, `prep`=cell-0 basis, `bounds`=cell-0 Schwarz (never read when injected).

Every site reading `mol`/`prep`/`bounds` or rebuilding molecular integrals:

| site | line | injected-path action |
|---|---|---|
| `mol.nelec()` | rhf.rs:819 | OK (cell electrons) |
| MINAO guess (molecular cross-overlap) | rhf.rs:839, guess.rs:808 | force `hcore_guess(&s,&h)` or `init_guess_density`; SAD/MINAO later |
| XC `KsXc::new…(mol,…)` | rhf.rs:767 | **reject** `config.xc` (Stage 2) |
| DF-J/K auto build, `k_builder` | rhf.rs:906, 976-1030 | **reject** `df_j_aux`/`df_k_aux`/`k_builder`; skip construction |
| `DirectJ/DirectK/DirectJK::new` | rhf.rs:1124-1162 | gate on `inj.is_none()` |
| incremental Fock | rhf.rs:1186 | off (`direct_jk` is None) |
| J/K branch | rhf.rs:1229-1305 | new first arm: `inj.j.build(&d,&mut j_buf)`, `inj.k.build(…)` |
| `solvent_terms(mol,prep,…)` | rhf.rs:1364 | no-op if cosmo/pcm/polarizable None → **reject** them |
| `external_potential` in hcore/vnn | driver.rs:94-99 | **reject** |
| stability `stability_rhf(mol,prep,bounds)` | rhf.rs:1533 | **reject** `check_stability` |
| TRAH / Newton (`rhf_newton` uses molecular JK + `FxcKernelStore(mol,prep)`) | rhf.rs:1604, 1690-1712, 1886-1940 | **reject** `trah_trigger`, `newton_trigger>0` |
| AURORA `AuroraState::new(mol,prep)` | rhf.rs:1836-1860 | **reject** `aurora.enabled` |
| level shift (`s·C_vir`) | rhf.rs:1970-1985 | OK (uses injected S) |
| MOM `mom_reorder(&c,&s,…)` | rhf.rs:1992 | OK |
| smearing (eps only) | rhf.rs:2010 | OK |
| canonical orthogonalizer | rhf.rs:1041 | OK, and it does the lindep filtering diffuse PBC bases need |
| `link_debug` DirectK cross-check | rhf.rs:1279 | unreachable (k_builder rejected) |
| gradients `rhf_gradient(mol,prep,…)` | lib.rs:128 | out of scope; the periodic entry never calls it |

Rejections = one `validate_injected(config)` erroring by field name (config-honesty: no silent no-ops).

## 2. J/K injection

`JBuilder` (fock.rs:10-13) and `KBuilder` (fock.rs:16-52, default `build_from_occ`, `update_density`) are `pub`
(lib.rs:53); ferric-pbc implements them (add `ferric-scf` dep; no cycle). J is concrete: `DfJ` / `DirectJ` /
combined `DirectJK` (rhf.rs:1124-1162). K is name-resolved: `resolve_k_builder` (fock_assembly.rs:199,
"direct|link|cosx") → `build_pluggable_k` (fock_assembly.rs:251).

Inject boxes directly (§1) rather than extend the name registry — keeps lattice state out of `RhfConfig`.
Madelung lives in the periodic `KBuilder`: `K += v_M·S D S` (`madelung_constant`, ewald.rs:92), so the SCF knows no exxdiv.

Later sites: UHF uhf.rs:630/684/703-714/858, ROHF rohf.rs:445/473/492-503/615, KS `KMix` scaling rhf.rs:1330-1341. **Stage 1: RHF only.**

## 3. Periodic one-electron integrals

`S_latt[m,n] = Σ_L ⟨m_0|n_L⟩` (T likewise). `compute_1e_block` (engine.rs:423) → `scf_compute_1e_block` (shim.cc:462)
indexes one `scf_basis` (shim.cc:31) whose centres are fixed at `PreparedBasis::new` (basis_bridge.rs:59); unchanged, the only
route is a ghost-image `PreparedBasis` per L — the supercell pattern FINDINGS rejects.

**Add one shim entry point** that follows the shim.cc try/catch + status convention (model: ecp_shim.cc):
```c
/* shell sh2 translated by shift[3] (Bohr); SCF_EINVAL on non-finite shift. */
int scf_compute_1e_block_shifted(scf_engine*, const scf_basis*, int sh1, int sh2,
                                 const double shift[3], double *out);
```
Copy `bs->bs[sh2]` into a per-engine scratch `Shell`, `move({O+shift})`, compute, inside `try{}catch(...){return SCF_EINTERNAL;}`.
Rust: `Engine::compute_1e_block_shifted` beside engine.rs:423; ferric-pbc loops `cell.translations(r)` with per-shell-pair
image lists (extent as pair_ft.rs:197). Per-L blocks are not symmetric: fill full blocks, symmetrise only Σ_L.

**V_ne** = SR `Σ_{L,M,A}⟨m_0|erfc(ωr_{A,M})/r|n_L⟩` + LR (`pair_ft` × structure factor, prototype `Vlr`) + G=0 `π Z_tot S/(ω²Ω)`. `new_1e` hardcodes ω=0 (engine.rs:226).

| option | pros | cons |
|---|---|---|
| **(a) op code 103 = `Operator::erfc_nuclear`** | libint2 has it natively (engine.impl.h:181, 284); exact point nuclei; derivative engine comes free for Stage-3 gradients; `set_point_charges_extra` (engine.rs:305) already takes arbitrary image positions | shim change |
| (b) Gaussian nucleus via eri3 + `Operator::erfc` (`smeared_attraction` pattern, oneelectron.rs:159) | no new C++ | ζ≈1e16 point approximation; `norm_int` scaling; slower 3c engine; still needs pair shifts |

**Recommend (a).** The exact changes:
- `op_for_kind` (shim.cc:328): add `case 103: return Operator::erfc_nuclear;`.
- `scf_engine_create` (shim.cc:397): for 103, `set_params(std::make_tuple(omega, q_default))`. The
  param type is `tuple<scalar, vector<pair<double,array<double,3>>>>` (libint2 engine.h:225-229).
- New `int scf_engine_set_point_charges_erfc(scf_engine*, double omega, const scf_atom*, int)`.
  It mirrors shim.cc:442 and sets the tuple, because the existing setter would throw `bad_any_cast`
  for op 103. It gets the same try/catch.
- `ffi.rs`: add `OP_ERFC_NUCLEAR = 103` next to ffi.rs:376. `Engine::new_1e_attenuated(op_kind, omega, prep, precision)`
  goes next to engine.rs:219 so that `new_1e` stays byte-identical.
- Test: at ω→0 (tiny ω), `erfc_nuclear` must equal `nuclear`, and an erf+erfc split must equal
  `nuclear` (anchor). Mutation: drop `set_params`, and the test must fail.

## 4. Two-basis 3c2e for RS-GDF

`scf_compute_eri3(eng, obs, dfbs, shP, sh1, sh2)` (shim.h:132, shim.cc:835) takes fixed centres. RS-GDF needs
`(μ_0 ν_L | P_M)` with erfc (SR part) plus a lattice-summed aux metric `(P_0|Q_M)`.

**Smallest C-ABI addition:**
```c
int scf_compute_eri3_shifted(scf_engine*, const scf_basis *obs, const scf_basis *dfbs,
                             int shP, int sh1, int sh2, const double shifts[9], double *out);
int scf_compute_eri2_shifted(scf_engine*, const scf_basis *dfbs, int shP, int shQ,
                             const double shiftQ[3], double *out);
```
`shifts` = (P, sh1, sh2) translations (callers pass shift1=0). Same scratch-shell `move()`; eri3 bypasses the
quartet-only ShellPair cache (shim.cc:549), so no invalidation hazard. No shifted quartet (4-centre SR skipped per FINDINGS).
Rejected: a combined cell0⊕image `scf_basis` with index ranges — supercell again, memory ∝ n_img.

**Screening.** Schwarz is blind to erfc attenuation: an erfc table (schwarz.rs:157) bounds by self-interaction
and never sees separation. Distance-aware SR 3c bound:

```
|(μ_0 ν_L | P_M)|_erfc ≤ Q^erfc_{μν_L} · Q^erfc_P · g(R),  g(R) = erfc(ω_eff R)/R ÷ erfc-self term
R = |centroid(μν_L) − (R_P+M)| − ext_{μν} − ext_P,   1/ω_eff² = 1/ω² + 1/p_min + 1/q_min
```
Pair pre-screen: drop (μ,ν_L) when `c·exp(−αβ/(α+β)|A−B−L|²) < thresh` (pair_ft's criterion). `g` is a hypothesis:
first commit here is a **screening study** (bound vs true max|integral| over ω, L, basis; threshold derived between
measured sides; anchor thresh→0 ≡ unscreened bitwise).

**Feeding DfJ/DfK.** At Gamma RS-GDF yields real `(P|mn)` and `(P|Q)` (SR lattice + LR aux-FT); the algebra is
molecular. Add `DfJ::from_parts(source, metric)` / `DfK::from_parts(source, v_inv_sqrt)` (`DfJ::from_source`,
df_j.rs:132, recomputes a molecular `coulomb_metric_2c`) and `ThreeIndexSource::from_blocks`. Main physics
risk: G=0 of charged aux functions — follow PySCF RSGDF, anchor to pure-AFT.

## 5. Memory gating

Budget: `resolve_three_index_budget` (rhf.rs:625). Plumbing ≠ enforcement — each buffer needs an erroring pre-flight:

| buffer | size | gate |
|---|---|---|
| `pair_ft` output `P[m,n,g]` | 16·nao²·nG B (nao=100, nG=1e5 → 16 GB) | chunk G; gate the chunk; never materialise all G |
| dense oracle `I[nao⁴]` | 8·nao⁴ | test-only; hard error above a small cap (Si8/cc-pVDZ = 4.3 GB) |
| RS-GDF raw/dressed B | 8·naux·nao² | the existing `ThreeIndexSource` budget/spill path (three_index_source.rs:341 `preflight_spill`) |
| per-shell-pair image lists | Σ pairs·n_img | small, but assert against the budget |
| LR J per iteration `ρ(G)` | 16·nG | stream |

Plus `driver::warn_if_rss_over_at_stage` (driver.rs:278) after the periodic integral build.

## 6. Implementation order (one reviewable commit each)

| # | commit | test gate (anchor first) |
|---|---|---|
| 1 | `solve_rhf_impl` + `PeriodicInjection` + `validate_injected` | the full RHF suite is byte-identical; injecting the molecular S,h,V_nn,DirectJ,DirectK reproduces `solve_rhf` bit-for-bit; each rejected field has its own error test |
| 2 | shim `1e_block_shifted` + `Engine` wrapper | shift=0 ≡ `compute_1e_block` bitwise; `S_latt` ≡ `pair_ft(G=0)` (independent construction) and ≡ prototype/`pbc_intor` 1e-13 |
| 3 | op 103 `erfc_nuclear` + setter | erf+erfc ≡ nuclear; mutation on set_params |
| 4 | `PeriodicHcore` (T + V_SR + V_LR + G=0) | the prototype `h` at the same ω ≤1e-10; ω-sweep independence ≤ SR-cutoff error (artifact hypothesis: a G=0 bug would not shrink with cutoff) |
| 5 | test-only `DenseAftJK` (nao⁴ pure-AFT, size-capped) + Madelung in K | H2/STO-3G a=4: E = −1.658327061049 (ewald) / −0.949002691179 (none) vs prototype and PySCF AFTDF; the triclinic 4H s+p test |
| 6 | ferric-vs-ferric molecular limit | box sweep a=16/20/24 ewald: E−E_mol(ferric `solve_rhf`) fits c3/a³ with c3 = −(4π/3)σ² to 1% (FINDINGS) |
| 7 | shim `eri3/eri2_shifted` | shift=0 ≡ `compute_eri3` bitwise; translation invariance (shift all by L ≡ unshifted) |
| 8 | **screening study** (measurement commit, no production path) | thresh→0 ≡ unscreened; measured bound-vs-true table; mutated bound shown to fail |
| 9 | RS-GDF build → `DfJ/DfK::from_parts` → periodic J/K builders | vs step-5 DenseAftJK on toys (fitting error only, trending to 0 with a bigger aux); vs PySCF GDF (correct the PySCF 2.13 RSJK E_nn bug) |
| 10 | memory gates + `warn_if_rss` | tiny budget → error; ample → unchanged energy |
| 11 | Python `run_rhf_gamma(cell,…)` surface (CLI later) | the pytest suite runs step 5 via Python; a charged cell is a hard error |

Out of Stage 1: UHF/ROHF/KS, ECP lattice sums, gradients/stress, low-dimensional systems, k-points.
