//! VALIDATION tier — VALIDATION.md rows "G0W0", "G0W0@HF+ECP", "U-G0W0" and
//! "COHSEX/evGW₀/evGW".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-gw --release \
//!     --test validation_gw --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric quasiparticle energies for a window HOMO−2 … LUMO+2 (ferric's default
//! QP range; six states per spin) against PySCF 2.13 `gw_ac` / `ugw_ac`, fed
//! ferric's OWN bundled orbital JSON, RI-aux JSON, ECP and Bohr geometry
//! (`scripts/validation/common.py`):
//!
//! | test | systems / basis (aux) | reference |
//! |---|---|---|
//! | G0W0@HF | H2O, NH3, N2 / cc-pVDZ (cc-pvdz-ri), aug-cc-pVDZ (aug-cc-pvdz-rifit) | `gw_ac` |
//! | G0W0@PBE | H2O / cc-pVDZ, aug-cc-pVDZ; N2 / cc-pVDZ | `gw_ac`, `vhf_df = True` |
//! | G0W0@HF, frozen core 1 | H2O / cc-pVDZ | `gw_ac`, `frozen = 1` |
//! | G0W0@HF + ECP | I2, Xe, Ag2 / aug-cc-pVDZ-PP (def2-tzvp-rifit) | `gw_ac`, same inline ECP |
//! | U-G0W0@UHF | OH, CH3, NH2 / cc-pVDZ, aug-cc-pVDZ; O2 (³Σg⁻), CH2 (³B1) / aug-cc-pVDZ | `ugw_ac` on a stability-checked UHF |
//! | COHSEX@HF | H2O, N2 / cc-pVDZ | numpy on PySCF's Lpq and Π(0) |
//! | evGW₀@HF, evGW@HF | H2O / cc-pVDZ | PySCF `gw_ac.get_sigma` iterated as ferric iterates |
//!
//! PySCF has no evGW; the generator iterates PySCF's own `get_sigma` (it takes
//! separate G and W orbital energies) with ferric's update rule: Jacobi
//! updates of the QP window only, ef and ε_mf held at the mean-field values, W
//! frozen (evGW₀) or rebuilt (evGW).
//!
//! References: `scripts/validation/gen_gw.py` →
//! `testdata/reference/validation/gw/<system>_<basis>.json`.
//!
//! # The like-for-like recipe (read from the code)
//!
//! * W: `trunc_thresh = 0` with the Lanczos eigensolver is the full-rank
//!   dielectric eigenbasis (one dense eigh of I + Π(0)), and `sigma_c_at_z`
//!   contracts the FULL inverse-dielectric matrix in it, so ferric's W is the
//!   ordinary RI W that PySCF builds in the Cholesky basis.
//! * W frequency integral: Gauss–Legendre, ω = u₀(1+x)/(1−x), u₀ = 0.5,
//!   100 points on both sides (`N_QUAD`; PySCF `nw = 100`).
//! * Padé: ferric samples Σc(ef + iω) on [0] + GL(100, u₀ = 0.5), picks 18
//!   nodes with PySCF's `_get_ac_idx` (step ratio 2/3) and fits Thiele. PySCF
//!   does the same on its W grid, but cuts it at `ac_iw_cutoff` (5 Ha by
//!   default); the generator sets the cut to `None`, so the nodes coincide,
//!   and so do the Thiele coefficients (same recursion; the generator asserts
//!   it). The EVALUATION of the fraction does not.
//! * PySCF DEFECT (PySCF 2.13.1, `pyscf/gw/utils/ac_grid.py`,
//!   `pade_thiele_ndarray`, reached through `PadeAC.ac_eval`): it seeds
//!   `X = coeff[-1] * (freqs - zn[-2])` and then loops `idx = ncoeff-1 … 1`
//!   with `X = coeff[idx] * (freqs - zn[idx-1]) / (1 + X)`. The first pass
//!   applies `coeff[-1]` a second time, so the innermost level is
//!   1 + a_{N−1}(z−z_{N−2})/(1 + a_{N−1}(z−z_{N−2})) instead of
//!   1 + a_{N−1}(z−z_{N−2}), and the fraction does not interpolate the data it
//!   was fitted to (up to 1.8e-3 Ha off at the nodes, H2O/cc-pVDZ). ferric's
//!   `pade.rs` `PadeCF::eval` is the standard Thiele fraction
//!   a₀/(1 + a₁(z−z₀)/(1 + … a_{N−1}(z−z_{N−2}))), which reproduces its nodes
//!   to 1e-16. MEASURED effect on Σc(ε_qp) with the same Σc(iω) and the same
//!   coefficients: @HF ≤1.2e-7 Ha (H2O/cc-pVDZ), so the @HF, ECP, U-G0W0 and
//!   ev references keep PySCF's evaluation; @PBE up to 7.14e-3 Ha
//!   (H2O/cc-pVDZ HOMO−2; 1.1e-6 at HOMO−1/HOMO, 1.16e-5 at LUMO+2; N2 LUMO+2
//!   1.4e-5), because the KS roots sit further from ef. THEREFORE every
//!   `g0w0_pbe` reference uses the standard Thiele fraction, re-implemented in
//!   numpy on PySCF's own Σc(ef + iω) (`gen_gw.py` `textbook_continuation`);
//!   PySCF's numbers are kept as `eps_qp_pyscf_pade` /
//!   `sigma_c_at_qp_pyscf_pade`.
//! * Imaginary axis, no continuation: Σc(ef + iω) at the 18 Padé nodes is
//!   stored under `g0w0_pbe/ac` (and `g0w0_hf/ac` for H2O and N2 cc-pVDZ), and
//!   `sigma_c_imag_axis_*` compares ferric's `sigma::sigma_c_at_z` there
//!   directly — the like-for-like check of Σc itself, independent of how
//!   either code continues it (`TOL_SIGMA_C_IW`).
//! * QP equation: the full root of ω − ε_mf − (Σx − v_xc) − ReΣc(ω) = 0 on
//!   both sides (PySCF's secant answer is re-solved at 1e-12 on the Padé —
//!   PySCF's object for @HF, the textbook fraction for @PBE). On PySCF's
//!   Σc(iω), ferric's own Newton (linearized start, 4-point FD slope, h = 0.05)
//!   lands on the reference root to ≤9e-10 Ha for every H2O @PBE state.
//! * Σx is density-fitted with the GW aux in ferric; the @PBE reference uses
//!   `vhf_df = True` so the static shift Σx − v_xc is the same object. For an
//!   HF reference ferric applies no shift and PySCF's vk − v_mf is 0 (the
//!   generator records it at ≤1e-13 Ha).
//! * SCF: exact 4-index integrals on both sides (PBE: `df_j_aux = Some("")`;
//!   grid (75,110) unpruned Becke, ferric's default).
//! * Open shell: ferric uses a PER-SPIN mid-gap Fermi level (u_sigma.rs)
//!   where `ugw_ac` uses one common ef; the generator runs `ugw_ac` once per
//!   spin with ef overridden. MEASURED: the two conventions differ by ≤4 µeV
//!   on every system here, so the choice cannot hide a defect at these bars.
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric's GW is right: with every discretization matched, the only
//!   differences left are the integral engines (libint2 vs libcint), SCF
//!   convergence and the eigensolver, so QP energies agree at the µeV level
//!   (Padé extrapolation amplifies Σc(iω) noise, so not at 1e-10).
//! * If W is wrong (a factor in Π, a missing spin channel, the diagonal-only
//!   W̃ defect that once over-screened the virtuals by 1 eV): Σc moves by
//!   tens to hundreds of meV. The virtual states in the window see this first.
//! * If the analytic continuation differs (node selection, ef, the Padé grid):
//!   the @PBE states move most, because the small KS gap puts ε_HOMO far from
//!   ef. Measured sensitivities (H2O/cc-pVDZ @PBE HOMO, textbook evaluation):
//!   16 instead of 18 Padé nodes 0.97 meV; PySCF's default 5 Ha cut 0.51 meV;
//!   exact instead of DF Σx 1.14 meV; a 16-point W integral 34 meV (at @HF
//!   only 10 µeV); PySCF's evaluation of the same fraction 0.03 meV on the
//!   HOMO but 194 meV on HOMO−2.
//! * If the static shift is misapplied at a KS reference (post-hoc instead of
//!   inside the QP equation — the 0.76 eV defect class): @PBE misses by
//!   ≫ 0.1 eV while @HF passes.
//! * If the frozen core leaks into W or Σ: the frozen-core case misses its
//!   reference by the measured core effect (2.7 meV on the H2O HOMO).
//! * If the HARNESS is wrong (geometry, basis, aux, ECP core count, a
//!   mislabelled file): E_nuc, the AO and aux counts, the SCF energy or the
//!   Σx anchor fails before any QP energy is compared.
//!
//! # Exactness anchors (checked before any QP energy)
//!
//! 1. SCF: energy and mean-field orbital energies equal PySCF's.
//! 2. Σx: ferric's DF exchange diagonal equals −Σ_{P,i} L_{P,mi}² from PySCF's
//!    Lpq with the same aux — an independent construction of the B tensor.
//! 3. QP residual: ferric's returned ε_qp, Σc(ε_qp) and shift satisfy the QP
//!    equation, and every Newton solve reports convergence.
//!
//! # TOLERANCES
//!
//! Each bar is 3–25× the worst |d| measured over every orbital in the window
//! and every system (`validation_gw`, 2026-09-25):
//!
//! | quantity | measured max \|d\| (Ha) | bar |
//! |---|---:|---:|
//! | QP energies, closed shell @HF | 1.32e-7 | `TOL_QP` 1e-6 |
//! | QP energies, @PBE | 6.2e-8 | `TOL_QP_PBE` 1e-6 |
//! | QP energies, ECP | 2.5e-6 | `TOL_QP_ECP` 2e-5 |
//! | QP energies, U-G0W0 | 7.0e-6 (OH/cc-pVDZ, marginal); others ≤ 8.7e-7 | `TOL_QP_U` 3e-5 |
//! | COHSEX | 9.4e-10 | `TOL_COHSEX` 1e-8 |
//! | evGW₀ / evGW | 5.7e-7 | `TOL_EV` 5e-6 |
//! | Σx (DF) | 4.9e-9 | `TOL_SX` 3e-8 |
//! | Σx (DF), ECP | 4.6e-6 | `TOL_SX_ECP` 2e-5 |
//! | Σc(ef + iω) at the Padé nodes (no continuation) | 8.6e-11 | `TOL_SIGMA_C_IW` 1e-9 |
//! | SCF energy | 4.7e-11 (ECP: 5.1e-8, Ag2) | `TOL_E_SCF` 1e-9, `TOL_E_SCF_ECP` 2.5e-7 |
//! | ε_mf | 3.6e-9 | `TOL_EPS_MF` 3e-8 |
//! | ε_mf, ECP | 2.9e-6 | `TOL_EPS_MF_ECP` 1e-5 |
//!
//! Mutation: evaluating the Padé continuation the way PySCF's
//! `pade_thiele_ndarray` does (last coefficient applied twice) fails both
//! @PBE tests, by 7.14e-3 Ha (H2O MO 2) and 1.4e-5 Ha (N2 MO 9).
//!
//! # NEGATIVE CONTROLS (asserted in the tests)
//!
//! * Basis: ferric's QP energies at basis A miss basis B's reference.
//! * Starting point: ferric G0W0@PBE misses the @HF reference.
//! * W quadrature: ferric @PBE with a 16-point W integral must MISS the
//!   100-point reference and MATCH PySCF's 16-point value (recorded in
//!   `diagnostics.pbe_recipe_sensitivity.nw_16`, Padé grid kept at 101
//!   points as ferric does). This proves `n_points` reaches W.
//! * Frozen core: ferric at frozen core 1 matches the frozen-core reference
//!   and misses the all-electron one.
//! * Spin: ferric's α HOMO misses the reference β HOMO.
//! * Self-consistency: evGW₀ misses G0W0 and evGW; COHSEX misses G0W0.
//!
//! MUTATIONS to run once and record here: (A) `STEP_RATIO` in
//! `sigma::solve_qp_for_mo` from 2/3 to 5/6 — every @PBE case must fail;
//! (B) drop the α or β Π in `run_u_pdep_rpa` — every U-G0W0 case must fail;
//! (C) in `run_evgw0`, update `eps_prop` for all states instead of the window —
//! the evGW₀ case must fail or this file must say it cannot see that choice.
//!
//! # Provenance of older numbers
//!
//! `h2o_g0w0_cohsex.rs` compares H2O/cc-pVDZ G0W0@HF against 11.97 eV,
//! attributed to MOLGW (van Setten et al. 2015). That value is unreproduced:
//! no MOLGW input, basis or settings for it exist in the repository, and the
//! GW100 paper, as far as we can tell, reports G0W0@PBE with def2 bases, not
//! an HF-started cc-pVDZ value. The PySCF reference on the same geometry,
//! basis and aux is 12.160022 eV (`h2o_cc-pvdz.json`), which reproduces the
//! 12.160 eV PySCF number `h2o_g0w0_cohsex.rs` also quotes. This file does
//! not use 11.97 eV.

use std::path::{Path, PathBuf};

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::vxc_mo::vxc_diagonal_mo;
use ferric_gw::{run_gw, run_u_gw, GwConfig, GwMethod, GwResult, UGwResult};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_scf::ladder::{default_ladder_from, solve_rhf_ladder};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use num_complex::Complex64;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/gw";
const MOL_DIR: &str = "testdata/molecules/validation";
const HA_TO_EV: f64 = 27.211_386_245_988;
/// W frequency points; PySCF `nw` in the generator.
const N_QUAD: usize = 100;

// Measured floors (worst |d| over every orbital in the window, all systems).
// Closed-shell G0W0@HF: 1.32e-7 Ha (H2O/cc-pVDZ MO 7; the Padé continuation
// and QP root, since Σc on the imaginary axis agrees to 1e-10).
const TOL_QP: f64 = 1e-6;
// U-G0W0@UHF: 7.0e-6 Ha on OH/cc-pVDZ β MO 7 (OH's π hole is MARGINAL: the
// lowest orbital-Hessian eigenvalue is zero). Every other open-shell case,
// OH/aug-cc-pVDZ included, is ≤ 8.7e-7.
const TOL_QP_U: f64 = 3e-5;
// G0W0@PBE: 6.2e-8 Ha (N2/cc-pVDZ MO 9), references with ferric's XC density
// floor and the textbook Thiele continuation (see gen_gw.py).
const TOL_QP_PBE: f64 = 1e-6;
// ECP: 2.5e-6 Ha (I2 MO 26), from the ε_mf offset below.
const TOL_QP_ECP: f64 = 2e-5;
// COHSEX is closed form (no Padé, no quadrature): 9.4e-10 Ha.
const TOL_COHSEX: f64 = 1e-8;
// evGW₀/evGW: 5.7e-7 Ha (evGW, H2O MO 7); PySCF's iteration stops at 1e-7 Ha
// per sweep and ferric's at `EV_CONV`.
const TOL_EV: f64 = 5e-6;
// Σx (DF exchange diagonal, same aux): 4.9e-9 Ha (CH2 triplet).
const TOL_SX: f64 = 3e-8;
// SCF energy: 4.7e-11 Ha all-electron (H2O/aug-cc-pVDZ PBE); 5.1e-8 Ha with
// ECPs (Ag2), the libecpint-vs-PySCF ECP integral offset.
const TOL_E_SCF: f64 = 1e-9;
const TOL_E_SCF_ECP: f64 = 2.5e-7;
// ε_mf: 3.6e-9 Ha (CH3/aug-cc-pVDZ). Degenerate pairs are compared index by
// index, which this floor already includes.
const TOL_EPS_MF: f64 = 3e-8;
/// ECP systems (basis names ending in -pp): the mean-field orbital energies
/// and Σx inherit the ECP SCF difference between libecpint and PySCF (the ECP
/// row measured ~1e-8 Ha in the energy; orbital energies are first order).
/// Measured: ε_mf 2.9e-6 (I2), Σx 4.6e-6 (Ag2).
const TOL_EPS_MF_ECP: f64 = 1e-5;
const TOL_SX_ECP: f64 = 2e-5;
/// Newton step tolerance in `solve_qp_for_mo` is 1e-7 Ha.
const TOL_RESID: f64 = 1e-6;
// Σc(ef + iω) at the 18 Padé nodes, no continuation, same z on both sides:
// 8.6e-11 Ha (H2O/cc-pVDZ @PBE). PySCF against itself moves these by 1.7e-9 Ha
// when its PBE SCF restarts from another guess at the same thresholds.
const TOL_SIGMA_C_IW: f64 = 1e-9;
const TOL_ENUC: f64 = 1e-9;
/// A reference ferric must MISS is missed by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 10.0;
/// The frozen-core effect is small: the reference blocks differ by at most
/// 1.07e-4 Ha over the window (107 x `TOL_QP`); this control asks for 5x.
const FC_MISS_FACTOR: f64 = 5.0;
/// ferric's evGW/evGW₀ outer-loop threshold (Ha).
const EV_CONV: f64 = 1e-7;

// ---------------------------------------------------------------------------
// Reference plumbing
// ---------------------------------------------------------------------------

/// Workspace root, found by walking up from the CWD (nextest sets the CWD to
/// the package dir); `CARGO_MANIFEST_DIR` is only a fallback because it is
/// baked in at compile time and is wrong inside a nextest archive.
fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("testdata").is_dir() && p.join("crates").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return p.to_path_buf();
            }
            here = p.parent();
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-gw manifest dir should be <root>/crates/ferric-gw")
        .to_path_buf()
}

/// Load a reference JSON. Missing or unparsable is a HARD failure.
fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_gw.py — a missing reference is a failure, never a skip",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn num(v: &Value, ptr: &str, ctx: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not a number"))
}

fn vec_f64(v: &Value, ptr: &str, ctx: &str) -> Vec<f64> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"))
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn vec_usize(v: &Value, ptr: &str, ctx: &str) -> Vec<usize> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"))
        .iter()
        .map(|x| x.as_u64().expect("index") as usize)
        .collect()
}

/// The reference QP window as a contiguous range (ferric's `qp_mos` type).
fn window(orbs: &[usize], ctx: &str) -> std::ops::Range<usize> {
    assert!(!orbs.is_empty(), "{ctx}: empty QP window");
    for w in orbs.windows(2) {
        assert_eq!(
            w[1],
            w[0] + 1,
            "{ctx}: QP window {orbs:?} is not contiguous"
        );
    }
    orbs[0]..orbs[orbs.len() - 1] + 1
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!(
        "{ctx}: {what:<28} ferric {got:+.10} ref {want:+.10} |d| {d:.2e} ({:.4} meV)",
        d * HA_TO_EV * 1e3
    );
    assert!(
        d < tol,
        "{ctx}: {what}: ferric {got:.10} vs reference {want:.10} (|d| {d:.2e} Ha, {:.4} meV) \
         exceeds {tol:.1e} Ha",
        d * HA_TO_EV * 1e3
    );
}

/// Print every state's difference FIRST (the whole window is the measurement
/// the bars are set from), then assert the worst one.
fn check_vec(ctx: &str, what: &str, orbs: &[usize], got: &[f64], want: &[f64], tol: f64) -> f64 {
    assert_eq!(got.len(), want.len(), "{ctx}: {what} length");
    assert_eq!(got.len(), orbs.len(), "{ctx}: {what} vs window length");
    let mut worst = (0.0_f64, 0usize, 0.0_f64, 0.0_f64);
    for ((&p, &g), &w) in orbs.iter().zip(got).zip(want) {
        let d = (g - w).abs();
        assert!(
            d.is_finite(),
            "{ctx}: {what}[{p}] is not finite: ferric {g} ref {w}"
        );
        eprintln!(
            "{ctx}: {:<28} ferric {g:+.10} ref {w:+.10} |d| {d:.2e} ({:.4} meV)",
            format!("{what}[{p}]"),
            d * HA_TO_EV * 1e3
        );
        if d > worst.0 {
            worst = (d, p, g, w);
        }
    }
    let (d, p, g, w) = worst;
    eprintln!(
        "{ctx}: {what} WORST |d| {d:.3e} Ha ({:.4} meV) at MO {p}",
        d * HA_TO_EV * 1e3
    );
    assert!(
        d < tol,
        "{ctx}: {what}[{p}]: ferric {g:.10} vs reference {w:.10} (|d| {d:.2e} Ha, {:.4} meV) \
         exceeds {tol:.1e} Ha",
        d * HA_TO_EV * 1e3
    );
    d
}

/// max_p |got_p − want_p| must exceed `factor × tol`: the comparison responds
/// to the thing the control changes.
fn assert_misses(ctx: &str, what: &str, got: &[f64], want: &[f64], tol: f64, factor: f64) {
    let d = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "{ctx}: control {what}: max |d| {d:.3e} Ha ({:.3} meV), must exceed {:.1e}",
        d * HA_TO_EV * 1e3,
        factor * tol
    );
    assert!(
        d > factor * tol,
        "{ctx}: negative control '{what}' did not miss: max |d| {d:.3e} Ha <= {factor} x {tol:.1e} \
         — the comparison does not respond to what the control changes"
    );
}

// ---------------------------------------------------------------------------
// System setup
// ---------------------------------------------------------------------------

struct Sys {
    label: String,
    r: Value,
    mol: Molecule,
    obs_bs: BasisSet,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

fn load_system(system: &str, xyz_rel: &str, basis_name: &str, ecp: bool) -> Sys {
    let r = reference(system, basis_name);
    let label = format!("{system}/{basis_name}");
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(xyz_rel);
    let mut mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let obs_bs = basis::bundled(basis_name).unwrap();
    if ecp {
        // apply_ecp BEFORE nelec() / nuclear_repulsion() (CLAUDE.md).
        mol.apply_ecp(&obs_bs);
    }
    // Geometry / ECP core like-for-like FIRST.
    check_close(
        &label,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &label),
        TOL_ENUC,
    );
    assert_eq!(
        mol.nelec() as i64,
        r["nelectron"].as_i64().unwrap(),
        "{label}: electron count (ECP core?)"
    );
    let aux_name = r["aux"].as_str().expect("aux").to_string();
    let aux_bs = basis::bundled(&aux_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    assert_eq!(
        obs.nbasis() as i64,
        r["nao"].as_i64().unwrap(),
        "{label}: AO count differs from the reference's"
    );
    assert_eq!(
        dfbs.nbasis() as i64,
        r["naux"].as_i64().unwrap(),
        "{label}: aux count differs from the reference's"
    );
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    Sys {
        label,
        r,
        mol,
        obs_bs,
        obs,
        dfbs,
        bounds,
        ctx: ParallelContext::default(),
    }
}

fn rhf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-10,
        ..Default::default()
    }
}

fn uhf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        check_stability: true,
        scf_stability_descent: true,
        ..Default::default()
    }
}

/// Closed-shell HF through the level-shift ladder (Ag2 needs it; for the
/// others rung 0 converges), then pinned to the reference state by energy.
fn rhf(sys: &Sys, tol_e: f64) -> ScfResult {
    let lr = solve_rhf_ladder(
        &sys.ctx,
        &sys.mol,
        &sys.obs,
        Operator::coulomb(),
        &sys.bounds,
        &default_ladder_from(&rhf_config()),
    )
    .unwrap_or_else(|e| panic!("{}: RHF ladder failed: {e:?}", sys.label));
    assert!(lr.converged, "{}: RHF did not converge", sys.label);
    let scf = lr.result;
    check_close(
        &sys.label,
        "E_RHF",
        scf.energy,
        num(&sys.r, "/rhf/energy", &sys.label),
        tol_e,
    );
    scf
}

/// UHF on the reference's stable state. First the stability-following
/// config (the UHF/ROHF row's), then the level-shift + MOM steering that
/// NH2/cc-pVDZ and OH/aug-cc-pVTZ are known to need; the state is accepted
/// only if its energy equals the stability-checked reference's.
fn uhf(sys: &Sys) -> ScfResult {
    let e_ref = num(&sys.r, "/uhf/energy", &sys.label);
    let steered = RhfConfig {
        level_shift: 0.5,
        mom_after_iter: 5,
        ..uhf_config()
    };
    let mut tried = Vec::new();
    for (name, cfg) in [("stability-descent", uhf_config()), ("ls0.5+MOM5", steered)] {
        let res = solve_uhf(&sys.ctx, &sys.mol, &sys.obs, &sys.bounds, &cfg)
            .unwrap_or_else(|e| panic!("{}: solve_uhf failed: {e:?}", sys.label));
        let d = (res.energy - e_ref).abs();
        tried.push(format!("{name}: E {:.10} |d| {d:.2e}", res.energy));
        if res.converged && d < TOL_E_SCF {
            eprintln!("{}: UHF state reached with {name}", sys.label);
            let st = res.stability.as_ref().unwrap_or_else(|| {
                panic!(
                    "{}: no stability analysis for the accepted UHF state",
                    sys.label
                )
            });
            // OH's degenerate pi hole gives an exact zero mode, which
            // ferric reports as MARGINAL (lambda_min ~1e-11); that is the
            // correct verdict. Unstable or indeterminate still fails.
            assert!(
                matches!(
                    st.verdict(),
                    StabilityVerdict::Stable | StabilityVerdict::Marginal
                ),
                "{}: ferric UHF state not stable: {}",
                sys.label,
                st.summary()
            );
            check_close(&sys.label, "E_UHF", res.energy, e_ref, TOL_E_SCF);
            return res;
        }
    }
    panic!(
        "{}: ferric did not reach the reference UHF state {e_ref:.10}: {tried:?}",
        sys.label
    );
}

fn pdep_cfg(n_quad: usize, frozen_core: usize) -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core,
        // Full rank: the dense Lanczos path keeps every dielectric mode.
        trunc_thresh: 0.0,
        eigensolver: Eigensolver::Lanczos,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: n_quad,
            u0: 0.5,
        },
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        ..Default::default()
    }
}

fn gw_cfg(method: GwMethod, qp: std::ops::Range<usize>, frozen_core: usize) -> GwConfig {
    GwConfig {
        method,
        qp_mos: Some(qp),
        frozen_core,
        max_ev_iter: 60,
        ev_conv_thresh: EV_CONV,
        ..Default::default()
    }
}

fn run_closed(
    sys: &Sys,
    scf: &ScfResult,
    method: GwMethod,
    qp: std::ops::Range<usize>,
    n_quad: usize,
    frozen_core: usize,
    vxc: Option<&ndarray::Array1<f64>>,
) -> GwResult {
    let res = run_gw(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        scf,
        &pdep_cfg(n_quad, frozen_core),
        &gw_cfg(method, qp, frozen_core),
        vxc,
    )
    .unwrap_or_else(|e| panic!("{}: run_gw({method:?}) failed: {e:?}", sys.label));
    assert!(
        res.qp_converged.iter().all(|&c| c),
        "{}: {method:?} QP Newton did not converge: {:?}",
        sys.label,
        res.qp_converged
    );
    assert!(
        res.outer_converged,
        "{}: {method:?} outer loop not converged",
        sys.label
    );
    res
}

/// Exactness anchors + QP comparison for one closed-shell G0W0 block.
/// `vxc` is the absolute-MO v_xc diagonal for a KS reference.
fn check_g0w0_block(
    sys: &Sys,
    block: &str,
    res: &GwResult,
    vxc: Option<&ndarray::Array1<f64>>,
    tol_qp: f64,
) -> f64 {
    let ctx = format!("{} {block}", sys.label);
    let orbs = vec_usize(&sys.r, &format!("/{block}/orbs"), &ctx);
    assert_eq!(res.mo_indices, orbs, "{ctx}: QP window");
    // Anchor 1: mean-field orbital energies.
    check_vec(
        &ctx,
        "eps_mf",
        &orbs,
        res.eps_mf.as_slice().unwrap(),
        &vec_f64(&sys.r, &format!("/{block}/eps_mf"), &ctx),
        if ctx.contains("-pp") {
            TOL_EPS_MF_ECP
        } else {
            TOL_EPS_MF
        },
    );
    // Anchor 2: DF exchange diagonal vs PySCF's Lpq with the same aux.
    check_vec(
        &ctx,
        "sigma_x(DF)",
        &orbs,
        res.sigma_x.as_slice().unwrap(),
        &vec_f64(&sys.r, &format!("/{block}/sigma_x_df"), &ctx),
        if ctx.contains("-pp") {
            TOL_SX_ECP
        } else {
            TOL_SX
        },
    );
    // Anchor 3: ferric's own QP equation is satisfied.
    for (k, &p) in orbs.iter().enumerate() {
        let shift = vxc.map(|v| res.sigma_x[k] - v[p]).unwrap_or(0.0);
        let resid = res.eps_qp[k] - res.eps_mf[k] - shift - res.sigma_c[k];
        assert!(
            resid.abs() < TOL_RESID,
            "{ctx}: QP residual {resid:.2e} Ha at MO {p} (ε_qp {:.10}, Σc {:.10}, shift {shift:.10})",
            res.eps_qp[k],
            res.sigma_c[k]
        );
    }
    // The headline comparison.
    check_vec(
        &ctx,
        "sigma_c(eps_qp)",
        &orbs,
        res.sigma_c.as_slice().unwrap(),
        &vec_f64(&sys.r, &format!("/{block}/sigma_c_at_qp"), &ctx),
        tol_qp,
    );
    check_vec(
        &ctx,
        "eps_qp",
        &orbs,
        res.eps_qp.as_slice().unwrap(),
        &vec_f64(&sys.r, &format!("/{block}/eps_qp"), &ctx),
        tol_qp,
    )
}

// ---------------------------------------------------------------------------
// Closed-shell G0W0@HF and @PBE
// ---------------------------------------------------------------------------

const CLOSED_BASES: [&str; 2] = ["cc-pvdz", "aug-cc-pvdz"];

/// G0W0@HF at both bases, then the cross-basis control.
fn g0w0_hf_row(system: &str) {
    let mut qp = Vec::new();
    for basis_name in CLOSED_BASES {
        let sys = load_system(system, &format!("{system}.xyz"), basis_name, false);
        let scf = rhf(&sys, TOL_E_SCF);
        let orbs = vec_usize(&sys.r, "/g0w0_hf/orbs", &sys.label);
        let res = run_closed(
            &sys,
            &scf,
            GwMethod::G0W0,
            window(&orbs, &sys.label),
            N_QUAD,
            0,
            None,
        );
        let worst = check_g0w0_block(&sys, "g0w0_hf", &res, None, TOL_QP);
        eprintln!(
            "{}: G0W0@HF worst |d| {:.3e} Ha ({:.4} meV)",
            sys.label,
            worst,
            worst * HA_TO_EV * 1e3
        );
        qp.push(res.eps_qp.to_vec());
    }
    // Control: each basis's ferric numbers miss the OTHER basis's reference.
    for (i, basis_name) in CLOSED_BASES.iter().enumerate() {
        let other = CLOSED_BASES[1 - i];
        let r_other = reference(system, other);
        assert_misses(
            &format!("{system}/{basis_name}"),
            &format!("vs the {other} reference"),
            &qp[i],
            &vec_f64(&r_other, "/g0w0_hf/eps_qp", system),
            TOL_QP,
            MUST_MISS_FACTOR,
        );
    }
}

#[test]
#[ignore = "validation: G0W0"]
fn g0w0_hf_h2o_vs_pyscf_gw_ac() {
    g0w0_hf_row("h2o");
}

#[test]
#[ignore = "validation: G0W0"]
fn g0w0_hf_nh3_vs_pyscf_gw_ac() {
    g0w0_hf_row("nh3");
}

#[test]
#[ignore = "validation: G0W0"]
fn g0w0_hf_n2_vs_pyscf_gw_ac() {
    g0w0_hf_row("n2");
}

fn pbe_config() -> RhfConfig {
    RhfConfig {
        xc: Some("pbe".into()),
        // Exact J, as the reference (an empty name is the explicit opt-out of
        // the RI-J auto-default in rhf.rs `resolve_aux`).
        df_j_aux: Some(String::new()),
        // KS densities floor above 1e-10 on the grid (KS-DFT row uses 1e-8).
        density_conv: 1e-10,
        ..rhf_config()
    }
}

fn g0w0_pbe_case(system: &str, basis_name: &str, quadrature_control: bool) {
    let sys = load_system(system, &format!("{system}.xyz"), basis_name, false);
    let ctx = format!("{} @PBE", sys.label);
    let scf = solve_rhf(
        &sys.ctx,
        &sys.mol,
        &sys.obs,
        Operator::coulomb(),
        &sys.bounds,
        &pbe_config(),
    )
    .unwrap_or_else(|e| panic!("{ctx}: RKS failed: {e:?}"));
    assert!(scf.converged, "{ctx}: RKS did not converge");
    check_close(
        &ctx,
        "E_RKS",
        scf.energy,
        num(&sys.r, "/g0w0_pbe/rks_energy", &ctx),
        TOL_E_SCF,
    );
    let (vxc, _) = vxc_diagonal_mo(&sys.mol, &sys.obs_bs, "pbe", &scf)
        .unwrap_or_else(|e| panic!("{ctx}: vxc_diagonal_mo: {e:?}"));
    let orbs = vec_usize(&sys.r, "/g0w0_pbe/orbs", &ctx);
    // The static shift ferric builds must be the reference's Σx(DF) − v_mf.
    let v_mf_ref = vec_f64(&sys.r, "/g0w0_pbe/v_mf", &ctx);
    for (k, &p) in orbs.iter().enumerate() {
        check_close(&ctx, &format!("v_xc[{p}]"), vxc[p], v_mf_ref[k], TOL_EPS_MF);
    }
    let qp = window(&orbs, &ctx);
    let res = run_closed(
        &sys,
        &scf,
        GwMethod::G0W0,
        qp.clone(),
        N_QUAD,
        0,
        Some(&vxc),
    );
    check_g0w0_block(&sys, "g0w0_pbe", &res, Some(&vxc), TOL_QP_PBE);
    // Control: the starting point matters — @PBE misses the @HF reference.
    assert_misses(
        &ctx,
        "@PBE vs the @HF reference",
        res.eps_qp.as_slice().unwrap(),
        &vec_f64(&sys.r, "/g0w0_hf/eps_qp", &ctx),
        TOL_QP_PBE,
        MUST_MISS_FACTOR,
    );
    if quadrature_control {
        // Control: a 16-point W integral must MISS the 100-point reference and
        // MATCH PySCF's 16-point value on the HOMO (Padé grid held at 101
        // points on both sides), i.e. `n_points` really reaches W.
        let res16 = run_closed(&sys, &scf, GwMethod::G0W0, qp, 16, 0, Some(&vxc));
        let nocc = (sys.mol.nelec() as usize) / 2;
        let k = orbs.iter().position(|&p| p == nocc - 1).expect("HOMO");
        let want16 = num(&sys.r, "/diagnostics/pbe_recipe_sensitivity/nw_16", &ctx) / HA_TO_EV;
        check_close(
            &ctx,
            "HOMO eps_qp, n_quad 16",
            res16.eps_qp[k],
            want16,
            TOL_QP_PBE,
        );
        assert_misses(
            &ctx,
            "n_quad 16 vs the 100-point reference",
            &[res16.eps_qp[k]],
            &[vec_f64(&sys.r, "/g0w0_pbe/eps_qp", &ctx)[k]],
            TOL_QP_PBE,
            MUST_MISS_FACTOR,
        );
    }
}

#[test]
#[ignore = "validation: G0W0"]
fn g0w0_pbe_h2o_vs_pyscf_gw_ac() {
    g0w0_pbe_case("h2o", "cc-pvdz", true);
    g0w0_pbe_case("h2o", "aug-cc-pvdz", false);
}

#[test]
#[ignore = "validation: G0W0"]
fn g0w0_pbe_n2_vs_pyscf_gw_ac() {
    g0w0_pbe_case("n2", "cc-pvdz", false);
}

// ---------------------------------------------------------------------------
// Σc on the imaginary axis (no analytic continuation)
// ---------------------------------------------------------------------------

/// Nested `[[f64]]` from the reference JSON.
fn mat_f64(v: &Value, ptr: &str, ctx: &str) -> Vec<Vec<f64>> {
    v.pointer(ptr)
        .unwrap_or_else(|| panic!("{ctx}: missing {ptr}"))
        .as_array()
        .unwrap_or_else(|| panic!("{ctx}: {ptr} is not an array"))
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap_or_else(|| panic!("{ctx}: {ptr} row is not an array"))
                .iter()
                .map(|x| x.as_f64().expect("number"))
                .collect()
        })
        .collect()
}

/// ferric Σc(ef + iω) at the reference's 18 Padé nodes, one row per window
/// state, as (Re, Im). Rebuilds the projected tensor from the GwResult's own
/// PDEP result exactly as `run_gw` does (build_full_b → redress → project) and
/// calls `sigma::sigma_c_at_z` — no Padé anywhere. z uses the REFERENCE ef, so
/// both codes are evaluated at the same complex energies; ferric's own ef is
/// checked against it separately.
fn ferric_sigma_c_iw(
    sys: &Sys,
    scf: &ScfResult,
    res: &GwResult,
    block: &str,
    ctx: &str,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let orbs = vec_usize(&sys.r, &format!("/{block}/orbs"), ctx);
    let omegas = vec_f64(&sys.r, &format!("/{block}/ac/ac_nodes_omega"), ctx);
    let node_idx = vec_usize(&sys.r, &format!("/{block}/ac/ac_nodes_index"), ctx);
    let ef_ref = num(&sys.r, &format!("/{block}/ef"), ctx);
    assert_eq!(res.mo_indices, orbs, "{ctx}: QP window");

    // The stored nodes are ferric's own AC grid: [0] + GL(100, u0 = 0.5)
    // sampled at `ac_nodes_index` (sigma::solve_qp_for_mo).
    let (leg, _) = ferric_rpa::quadrature::gauss_legendre_nodes(100, 0.5);
    for (&i, &w) in node_idx.iter().zip(&omegas) {
        let mine = if i == 0 { 0.0 } else { leg[i - 1] };
        assert!(
            (mine - w).abs() <= 1e-12 * w.abs().max(1.0),
            "{ctx}: AC node {i}: ferric grid {mine:.15e} vs reference {w:.15e}"
        );
    }

    let mo_b = ferric_gw::mo_b::build_full_b(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        scf,
        0,
        None,
    )
    .unwrap_or_else(|e| panic!("{ctx}: build_full_b: {e:?}"));
    let n_occ = mo_b.n_occ_act;
    let ef_ferric = 0.5 * (mo_b.eps_act[n_occ - 1] + mo_b.eps_act[n_occ]);
    check_close(ctx, "ef (mid-gap)", ef_ferric, ef_ref, TOL_EPS_MF);
    let (v_dressed, _) =
        ferric_gw::w_pdep::redress_with_check(&mo_b.v_inv_sqrt, &res.pdep.eigenpotentials)
            .unwrap_or_else(|e| panic!("{ctx}: redress: {e:?}"));
    let m_proj = ferric_gw::cohsex::project_b_into_pdep(&mo_b, &v_dressed, None)
        .unwrap_or_else(|e| panic!("{ctx}: project_b_into_pdep: {e:?}"));
    let inv = res
        .pdep
        .inv_dielectric_freq
        .as_ref()
        .expect("GW keeps inv_dielectric_freq");
    let mut re = Vec::with_capacity(orbs.len());
    let mut im = Vec::with_capacity(orbs.len());
    for &p in &orbs {
        let m_loc = p - mo_b.first_act;
        let vals: Vec<Complex64> = omegas
            .iter()
            .map(|&w| {
                ferric_gw::sigma::sigma_c_at_z(
                    m_loc,
                    Complex64::new(ef_ref, w),
                    &m_proj,
                    inv,
                    &res.pdep.quad_weights,
                    &res.pdep.quad_freqs,
                    &mo_b.eps_act,
                )
            })
            .collect();
        re.push(vals.iter().map(|z| z.re).collect());
        im.push(vals.iter().map(|z| z.im).collect());
    }
    (re, im)
}

/// Worst |ferric − reference| over every state and node, Re and Im; prints
/// each state's worst node first.
fn sigma_c_iw_worst(
    ctx: &str,
    orbs: &[usize],
    got: &(Vec<Vec<f64>>, Vec<Vec<f64>>),
    want: &(Vec<Vec<f64>>, Vec<Vec<f64>>),
) -> f64 {
    let mut worst = 0.0_f64;
    for (k, &p) in orbs.iter().enumerate() {
        let mut w_state = 0.0_f64;
        for part in [(&got.0, &want.0), (&got.1, &want.1)] {
            assert_eq!(part.0[k].len(), part.1[k].len(), "{ctx}: node count");
            for (g, r) in part.0[k].iter().zip(&part.1[k]) {
                w_state = w_state.max((g - r).abs());
            }
        }
        eprintln!("{ctx}: Σc(ef+iω)[{p}] worst |d| over 18 nodes (Re, Im) {w_state:.2e} Ha");
        worst = worst.max(w_state);
    }
    worst
}

fn sigma_c_iw_case(system: &str, basis_name: &str, block: &str) {
    let sys = load_system(system, &format!("{system}.xyz"), basis_name, false);
    let ctx = format!("{} {block} Σc(iω)", sys.label);
    let (scf, vxc) = if block == "g0w0_pbe" {
        let scf = solve_rhf(
            &sys.ctx,
            &sys.mol,
            &sys.obs,
            Operator::coulomb(),
            &sys.bounds,
            &pbe_config(),
        )
        .unwrap_or_else(|e| panic!("{ctx}: RKS failed: {e:?}"));
        assert!(scf.converged, "{ctx}: RKS did not converge");
        let (vxc, _) = vxc_diagonal_mo(&sys.mol, &sys.obs_bs, "pbe", &scf)
            .unwrap_or_else(|e| panic!("{ctx}: vxc_diagonal_mo: {e:?}"));
        (scf, Some(vxc))
    } else {
        (rhf(&sys, TOL_E_SCF), None)
    };
    let orbs = vec_usize(&sys.r, &format!("/{block}/orbs"), &ctx);
    let res = run_closed(
        &sys,
        &scf,
        GwMethod::G0W0,
        window(&orbs, &ctx),
        N_QUAD,
        0,
        vxc.as_ref(),
    );
    let got = ferric_sigma_c_iw(&sys, &scf, &res, block, &ctx);
    let want = (
        mat_f64(&sys.r, &format!("/{block}/ac/sigma_c_iw_nodes_re"), &ctx),
        mat_f64(&sys.r, &format!("/{block}/ac/sigma_c_iw_nodes_im"), &ctx),
    );
    let worst = sigma_c_iw_worst(&ctx, &orbs, &got, &want);
    eprintln!(
        "{ctx}: worst |d| {worst:.3e} Ha ({:.6} meV)",
        worst * HA_TO_EV * 1e3
    );
    assert!(
        worst < TOL_SIGMA_C_IW,
        "{ctx}: Σc(ef + iω) differs from PySCF by {worst:.2e} Ha (bar {TOL_SIGMA_C_IW:.0e})"
    );
    // Control: the other starting point's stored nodes (same file, same
    // nodes) must be MISSED, so this comparison can resolve the reference.
    let other = if block == "g0w0_pbe" {
        "g0w0_hf"
    } else {
        "g0w0_pbe"
    };
    if sys.r.pointer(&format!("/{other}/ac")).is_some() {
        let want_other = (
            mat_f64(&sys.r, &format!("/{other}/ac/sigma_c_iw_nodes_re"), &ctx),
            mat_f64(&sys.r, &format!("/{other}/ac/sigma_c_iw_nodes_im"), &ctx),
        );
        let miss = sigma_c_iw_worst(&format!("{ctx} vs {other}"), &orbs, &got, &want_other);
        assert!(
            miss > MUST_MISS_FACTOR * TOL_SIGMA_C_IW,
            "{ctx}: control: ferric also matches the {other} nodes ({miss:.2e} Ha)"
        );
    }
}

#[test]
#[ignore = "validation: G0W0"]
fn sigma_c_imag_axis_h2o_pbe_and_hf_vs_pyscf() {
    sigma_c_iw_case("h2o", "cc-pvdz", "g0w0_pbe");
    sigma_c_iw_case("h2o", "cc-pvdz", "g0w0_hf");
}

#[test]
#[ignore = "validation: G0W0"]
fn sigma_c_imag_axis_n2_pbe_vs_pyscf() {
    sigma_c_iw_case("n2", "cc-pvdz", "g0w0_pbe");
}

#[test]
#[ignore = "validation: G0W0"]
fn g0w0_hf_frozen_core_h2o_vs_pyscf_gw_ac() {
    let sys = load_system("h2o", "h2o.xyz", "cc-pvdz", false);
    let scf = rhf(&sys, TOL_E_SCF);
    let orbs = vec_usize(&sys.r, "/g0w0_hf_fc1/orbs", &sys.label);
    assert_eq!(
        sys.r["g0w0_hf_fc1"]["frozen"].as_u64(),
        Some(1),
        "{}: reference block is not frozen-core",
        sys.label
    );
    let res = run_closed(
        &sys,
        &scf,
        GwMethod::G0W0,
        window(&orbs, &sys.label),
        N_QUAD,
        1,
        None,
    );
    check_g0w0_block(&sys, "g0w0_hf_fc1", &res, None, TOL_QP);
    // Control: frozen-core ferric misses the all-electron reference.
    assert_misses(
        &sys.label,
        "frozen core 1 vs the all-electron reference",
        res.eps_qp.as_slice().unwrap(),
        &vec_f64(&sys.r, "/g0w0_hf/eps_qp", &sys.label),
        TOL_QP,
        FC_MISS_FACTOR,
    );
}

// ---------------------------------------------------------------------------
// ECP
// ---------------------------------------------------------------------------

const ECP_BASIS: &str = "aug-cc-pvdz-pp";

fn g0w0_ecp_case(system: &str) {
    let sys = load_system(system, &format!("ecp/{system}.xyz"), ECP_BASIS, true);
    let scf = rhf(&sys, TOL_E_SCF_ECP);
    let nocc = (sys.mol.nelec() as usize) / 2;
    assert!(
        scf.eps_r()[nocc - 1] < 0.0,
        "{}: unbound HOMO — unphysical RHF basin",
        sys.label
    );
    let orbs = vec_usize(&sys.r, "/g0w0_hf/orbs", &sys.label);
    let res = run_closed(
        &sys,
        &scf,
        GwMethod::G0W0,
        window(&orbs, &sys.label),
        N_QUAD,
        0,
        None,
    );
    check_g0w0_block(&sys, "g0w0_hf", &res, None, TOL_QP_ECP);
}

#[test]
#[ignore = "validation: G0W0@HF+ECP"]
fn g0w0_hf_ecp_i2_vs_pyscf_gw_ac() {
    g0w0_ecp_case("i2");
}

#[test]
#[ignore = "validation: G0W0@HF+ECP"]
fn g0w0_hf_ecp_xe_vs_pyscf_gw_ac() {
    g0w0_ecp_case("xe");
}

#[test]
#[ignore = "validation: G0W0@HF+ECP"]
fn g0w0_hf_ecp_ag2_vs_pyscf_gw_ac() {
    g0w0_ecp_case("ag2");
}

// ---------------------------------------------------------------------------
// U-G0W0@UHF
// ---------------------------------------------------------------------------

fn check_u_spin(ctx: &str, r: &Value, spin: &str, orbs: &[usize], res: &UGwResult) -> Vec<f64> {
    let (eps_mf, eps_qp, sx, sc, conv) = if spin == "alpha" {
        (
            &res.eps_mf_a,
            &res.eps_qp_a,
            &res.sigma_x_a,
            &res.sigma_c_a,
            &res.qp_converged_a,
        )
    } else {
        (
            &res.eps_mf_b,
            &res.eps_qp_b,
            &res.sigma_x_b,
            &res.sigma_c_b,
            &res.qp_converged_b,
        )
    };
    let ctx = format!("{ctx} {spin}");
    assert!(conv.iter().all(|&c| c), "{ctx}: QP Newton not converged");
    let p = format!("/u_g0w0_uhf/{spin}");
    check_vec(
        &ctx,
        "eps_mf",
        orbs,
        eps_mf.as_slice().unwrap(),
        &vec_f64(r, &format!("{p}/eps_mf"), &ctx),
        TOL_EPS_MF,
    );
    check_vec(
        &ctx,
        "sigma_x(DF)",
        orbs,
        sx.as_slice().unwrap(),
        &vec_f64(r, &format!("{p}/sigma_x_df"), &ctx),
        TOL_SX,
    );
    for (k, &p) in orbs.iter().enumerate() {
        let resid = eps_qp[k] - eps_mf[k] - sc[k];
        assert!(
            resid.abs() < TOL_RESID,
            "{ctx}: QP residual {resid:.2e} Ha at MO {p}"
        );
    }
    check_vec(
        &ctx,
        "sigma_c(eps_qp)",
        orbs,
        sc.as_slice().unwrap(),
        &vec_f64(r, &format!("{p}/sigma_c_at_qp"), &ctx),
        TOL_QP_U,
    );
    check_vec(
        &ctx,
        "eps_qp",
        orbs,
        eps_qp.as_slice().unwrap(),
        &vec_f64(r, &format!("{p}/eps_qp"), &ctx),
        TOL_QP_U,
    );
    eps_qp.to_vec()
}

fn u_g0w0_case(system: &str, basis_name: &str) -> Vec<f64> {
    let sys = load_system(system, &format!("{system}.xyz"), basis_name, false);
    let scf = uhf(&sys);
    let orbs = vec_usize(&sys.r, "/u_g0w0_uhf/orbs", &sys.label);
    let res = run_u_gw(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        &scf,
        &pdep_cfg(N_QUAD, 0),
        &gw_cfg(GwMethod::G0W0, window(&orbs, &sys.label), 0),
    )
    .unwrap_or_else(|e| panic!("{}: run_u_gw failed: {e:?}", sys.label));
    assert_eq!(res.mo_indices, orbs, "{}: QP window", sys.label);
    let qa = check_u_spin(&sys.label, &sys.r, "alpha", &orbs, &res);
    check_u_spin(&sys.label, &sys.r, "beta", &orbs, &res);
    // Control: the α channel misses the β reference (spin channels are not
    // interchangeable in the comparison).
    assert_misses(
        &sys.label,
        "alpha QP vs the beta reference",
        &qa,
        &vec_f64(&sys.r, "/u_g0w0_uhf/beta/eps_qp", &sys.label),
        TOL_QP_U,
        MUST_MISS_FACTOR,
    );
    qa
}

fn u_g0w0_two_bases(system: &str) {
    let qa = [
        u_g0w0_case(system, "cc-pvdz"),
        u_g0w0_case(system, "aug-cc-pvdz"),
    ];
    for (i, basis_name) in CLOSED_BASES.iter().enumerate() {
        let other = CLOSED_BASES[1 - i];
        assert_misses(
            &format!("{system}/{basis_name}"),
            &format!("alpha QP vs the {other} reference"),
            &qa[i],
            &vec_f64(
                &reference(system, other),
                "/u_g0w0_uhf/alpha/eps_qp",
                system,
            ),
            TOL_QP_U,
            MUST_MISS_FACTOR,
        );
    }
}

#[test]
#[ignore = "validation: U-G0W0"]
fn u_g0w0_oh_vs_pyscf_ugw_ac() {
    u_g0w0_two_bases("oh");
}

#[test]
#[ignore = "validation: U-G0W0"]
fn u_g0w0_ch3_vs_pyscf_ugw_ac() {
    u_g0w0_two_bases("ch3");
}

#[test]
#[ignore = "validation: U-G0W0"]
fn u_g0w0_nh2_vs_pyscf_ugw_ac() {
    u_g0w0_two_bases("nh2");
}

#[test]
#[ignore = "validation: U-G0W0"]
fn u_g0w0_triplets_o2_ch2_vs_pyscf_ugw_ac() {
    u_g0w0_case("o2", "aug-cc-pvdz");
    u_g0w0_case("ch2_triplet", "aug-cc-pvdz");
}

// ---------------------------------------------------------------------------
// COHSEX, evGW₀, evGW
// ---------------------------------------------------------------------------

fn cohsex_case(system: &str) {
    let sys = load_system(system, &format!("{system}.xyz"), "cc-pvdz", false);
    let scf = rhf(&sys, TOL_E_SCF);
    let orbs = vec_usize(&sys.r, "/cohsex_hf/orbs", &sys.label);
    let res = run_closed(
        &sys,
        &scf,
        GwMethod::Cohsex,
        window(&orbs, &sys.label),
        N_QUAD,
        0,
        None,
    );
    let ctx = format!("{} COHSEX", sys.label);
    // COHSEX's "sigma_c" is ΔΣ_SEX + Σ_COH (cohsex.rs).
    let dsex = vec_f64(&sys.r, "/cohsex_hf/delta_sigma_sex", &ctx);
    let coh = vec_f64(&sys.r, "/cohsex_hf/sigma_coh", &ctx);
    let sc_ref: Vec<f64> = dsex.iter().zip(&coh).map(|(a, b)| a + b).collect();
    check_vec(
        &ctx,
        "dSEX+COH",
        &orbs,
        res.sigma_c.as_slice().unwrap(),
        &sc_ref,
        TOL_COHSEX,
    );
    check_vec(
        &ctx,
        "eps_qp",
        &orbs,
        res.eps_qp.as_slice().unwrap(),
        &vec_f64(&sys.r, "/cohsex_hf/eps_qp", &ctx),
        TOL_COHSEX,
    );
    assert_misses(
        &ctx,
        "COHSEX vs the G0W0 reference",
        res.eps_qp.as_slice().unwrap(),
        &vec_f64(&sys.r, "/g0w0_hf/eps_qp", &ctx),
        TOL_COHSEX,
        MUST_MISS_FACTOR,
    );
}

#[test]
#[ignore = "validation: COHSEX/evGW0/evGW"]
fn cohsex_hf_h2o_n2_vs_numpy() {
    cohsex_case("h2o");
    cohsex_case("n2");
}

#[test]
#[ignore = "validation: COHSEX/evGW0/evGW"]
fn evgw0_evgw_hf_h2o_vs_iterated_pyscf() {
    let sys = load_system("h2o", "h2o.xyz", "cc-pvdz", false);
    let scf = rhf(&sys, TOL_E_SCF);
    let mut got = Vec::new();
    for (method, block) in [(GwMethod::EvGw0, "evgw0_hf"), (GwMethod::EvGw, "evgw_hf")] {
        let ctx = format!("{} {block}", sys.label);
        let orbs = vec_usize(&sys.r, &format!("/{block}/orbs"), &ctx);
        let res = run_closed(&sys, &scf, method, window(&orbs, &ctx), N_QUAD, 0, None);
        eprintln!(
            "{ctx}: ferric {} outer iterations (reference {})",
            res.n_ev_iter, sys.r[block]["iterations"]
        );
        check_vec(
            &ctx,
            "eps_mf",
            &orbs,
            res.eps_mf.as_slice().unwrap(),
            &vec_f64(&sys.r, &format!("/{block}/eps_mf"), &ctx),
            TOL_EPS_MF,
        );
        check_vec(
            &ctx,
            "eps_qp",
            &orbs,
            res.eps_qp.as_slice().unwrap(),
            &vec_f64(&sys.r, &format!("/{block}/eps_qp"), &ctx),
            TOL_EV,
        );
        got.push(res.eps_qp.to_vec());
    }
    let ctx = &sys.label;
    // Controls: evGW₀ is not G0W0 (G really iterated) and not evGW (W really
    // frozen); evGW is not evGW₀ (W really rebuilt).
    assert_misses(
        ctx,
        "evGW0 vs the G0W0 reference",
        &got[0],
        &vec_f64(&sys.r, "/g0w0_hf/eps_qp", ctx),
        TOL_EV,
        MUST_MISS_FACTOR,
    );
    assert_misses(
        ctx,
        "evGW0 vs the evGW reference",
        &got[0],
        &vec_f64(&sys.r, "/evgw_hf/eps_qp", ctx),
        TOL_EV,
        MUST_MISS_FACTOR,
    );
    assert_misses(
        ctx,
        "evGW vs the evGW0 reference",
        &got[1],
        &vec_f64(&sys.r, "/evgw0_hf/eps_qp", ctx),
        TOL_EV,
        MUST_MISS_FACTOR,
    );
}
