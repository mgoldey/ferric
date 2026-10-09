//! Stage 6: Gamma-point closed-shell MP2 on the periodic integrals (Rust
//! port of `reference/pbc/pbc_mp2.py`, FINDINGS "Iteration 3 (Python, Gamma
//! MP2)" and its "For the Rust port (stage 6)" recommendation).
//!
//! ```text
//! E_MP2 = Σ_{ijab} (ia|jb) [2 (ia|jb) − (ib|ja)] / (ε_i + ε_j − ε_a − ε_b)
//! ```
//!
//! At Gamma the orbitals are real. `(ia|jb)` is taken in the SAME G = 0-dropped
//! kernel as the HF ERI (no G = 0 correction is needed: ov pair densities are
//! neutral, `C_oᵀ S C_v = 0`).
//!
//! # Integral sources ([`GammaMp2Integrals`])
//!
//! * [`GammaMp2Integrals::RsGdf`] — production: the SAME metric-dressed
//!   `B[k, μν]` the SCF used for J/K, transformed to `B[k, ia]`
//!   ([`b_ov_from_ao_b`]) and handed to ferric-mp2's existing kernel
//!   [`ferric_mp2::rimp2::spin_components_from_b_ov`] — the very function the
//!   molecular `ri_mp2_spin_components` calls after forming its own `b_flat`
//!   (through the κ = None branch of `spin_components_from_b_ov_kappa`). No
//!   ferric-mp2 code was changed for this stage, so the molecular path is
//!   byte-identical by construction.
//! * [`GammaMp2Integrals::DenseAft`] — TEST/ORACLE ONLY: the exact dense
//!   `(ia|jb) = Wᵀ I W` (`W = C_o ⊗ C_v`) from the pure-AFT tensor, summed by
//!   [`ferric_mp2::rimp2::spin_components_from_g`] — a different MO transform
//!   AND a different energy loop from the B path, which is what makes the
//!   trivial-aux anchor a test of construction (`tests/pbc_mp2.rs`).
//!
//! # Denominators: the exchange divergence (config honesty)
//!
//! The exchange G = 0 divergence enters MP2 ONLY through the occupied orbital
//! energies: at Gamma, `exxdiv = ewald` (`K += v_M S D S`) leaves C unchanged
//! and moves every occupied level by `−v_M`. Two conventions exist
//! ([`Mp2Denominators`]); nothing else differs between them.
//!
//! * `MadelungShifted` (the physical one): occupied ε from the ewald Fock.
//!   The prototype's box-limit sweep measured it converging to the molecular
//!   MP2 as a⁻³ with a coefficient predicted from molecular moments;
//!   `Unshifted` converges as 1/a (still 4.5–5.7 % of E_corr off at a = 40).
//!   PySCF pbc CC always uses it; PySCF `pbc.mp.RMP2` uses it only when the
//!   HF ran with `exxdiv='ewald'`.
//! * `Unshifted`: occupied ε from the `exxdiv = None` Fock (what PySCF RMP2
//!   gives after an `exxdiv=None` HF). Kept for oracle comparisons and the
//!   1/a demonstration; not a production choice.
//!
//! The reference's exxdiv cannot be read off an `ScfResult`, so the caller
//! must STATE it ([`GammaMp2Config::reference_exxdiv`]); there is no
//! `Default` for the config. The driver then applies exactly the occupied
//! shift that turns the stated reference into the requested convention:
//!
//! | reference exxdiv | denominators      | shift added to ε_occ |
//! |------------------|-------------------|----------------------|
//! | ewald            | MadelungShifted   | 0                    |
//! | none             | MadelungShifted   | −v_M (PySCF-CC style)|
//! | none             | Unshifted         | 0                    |
//! | ewald            | Unshifted         | +v_M                 |
//!
//! Choice for an `exxdiv = none` reference: APPLY the shift (not refuse). It
//! is exact at Gamma (C is identical under none/ewald, the Madelung term is
//! fit-independent), it is what PySCF CC does, and it is never silent: the
//! reference convention is a required input and the applied shift is
//! reported ([`GammaMp2Result::occ_shift`]). A caller that lies about the
//! reference exxdiv gets a result off by exactly ±v_M in every occupied
//! denominator — detectable against the box limit (`tests/pbc_mp2.rs`).
//!
//! # Frozen core
//!
//! `frozen_core` goes through [`ferric_mp2::rimp2::active_occ`] (errors
//! instead of underflowing). No periodic frozen-core measurement exists yet.
//!
//! # Memory
//!
//! Every buffer (MO-transform intermediate, `B[k,ia]` or `(ia|jb)`, one
//! per-worker `G_i` block of the kernel) is reserved on a
//! [`crate::budget`] ledger before allocation against
//! [`GammaMp2Config::budget_bytes`]. The kernel's `G_i` is held once per rayon
//! worker; the ledger counts ONE (a gate must not read the thread count).
//!
//! Units: Bohr and Hartree.

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::{DenseAftEri, ExxDiv};
use crate::ewald::madelung_constant;
use crate::lattice::Cell;
use crate::rsgdf::RsGdf;
use ferric_core::FerricError;
pub use ferric_mp2::rimp2::SpinComponents;
use ferric_mp2::rimp2::{active_occ, spin_components_from_b_ov, spin_components_from_g};
use ferric_scf::result::{ScfResult, Spin};
use ndarray::{s, Array2, ArrayView2};

/// Which occupied orbital energies the MP2 denominators use (module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mp2Denominators {
    /// Occupied ε from the `exxdiv = ewald` Fock (= none-Fock ε_occ − v_M).
    /// The physical convention: residual vs the molecule is O(a⁻³).
    MadelungShifted,
    /// Occupied ε from the `exxdiv = None` Fock. Residual O(1/a).
    Unshifted,
}

impl Mp2Denominators {
    /// Strict parse: `"shifted"` or `"unshifted"` (case-insensitive);
    /// anything else is an error (no silent default).
    pub fn parse_config_str(s: &str) -> Result<Self, FerricError> {
        match s.to_ascii_lowercase().as_str() {
            "shifted" => Ok(Mp2Denominators::MadelungShifted),
            "unshifted" => Ok(Mp2Denominators::Unshifted),
            other => Err(FerricError::General(format!(
                "MP2 denominators must be \"shifted\" or \"unshifted\", got {other:?}"
            ))),
        }
    }
}

/// The shift added to every occupied ε so that a reference converged with
/// `reference_exxdiv` yields `denominators` (the module-doc table). Shared
/// by [`gamma_mp2`] and [`crate::drpa::gamma_drpa`].
pub fn occupied_shift(
    reference_exxdiv: ExxDiv,
    denominators: Mp2Denominators,
    madelung: f64,
) -> f64 {
    match (reference_exxdiv, denominators) {
        (ExxDiv::Ewald, Mp2Denominators::MadelungShifted) => 0.0,
        (ExxDiv::None, Mp2Denominators::MadelungShifted) => -madelung,
        (ExxDiv::None, Mp2Denominators::Unshifted) => 0.0,
        (ExxDiv::Ewald, Mp2Denominators::Unshifted) => madelung,
    }
}

/// Settings for [`gamma_mp2`]. Deliberately no `Default`: the reference
/// exxdiv and the denominator convention must be stated.
#[derive(Debug, Clone, Copy)]
pub struct GammaMp2Config {
    /// Core orbitals excluded from correlation (validated by `active_occ`).
    pub frozen_core: usize,
    /// The `exxdiv` the RHF REFERENCE was converged with (its ε are read
    /// as-is and shifted per the module-doc table). This describes the SCF,
    /// not the integral object passed to [`gamma_mp2`] (the ov integrals do
    /// not depend on exxdiv).
    pub reference_exxdiv: ExxDiv,
    /// Denominator convention.
    pub denominators: Mp2Denominators,
    /// Memory budget in bytes (`None` = ferric's unified budget).
    pub budget_bytes: Option<usize>,
}

impl GammaMp2Config {
    /// The physical convention (Madelung-shifted occupied energies), no
    /// frozen core, ferric's unified budget.
    pub fn shifted(reference_exxdiv: ExxDiv) -> Self {
        Self {
            frozen_core: 0,
            reference_exxdiv,
            denominators: Mp2Denominators::MadelungShifted,
            budget_bytes: None,
        }
    }
}

/// Where `(ia|jb)` comes from.
#[derive(Debug, Clone, Copy)]
pub enum GammaMp2Integrals<'a> {
    /// Production: the fitted periodic `B` (the same one the SCF used).
    RsGdf(&'a RsGdf),
    /// TEST/ORACLE ONLY: exact dense pure-AFT `(ia|jb)`.
    DenseAft(&'a DenseAftEri),
}

/// Gamma-point MP2 result.
#[derive(Debug, Clone)]
#[must_use]
pub struct GammaMp2Result {
    /// Opposite-/same-spin split (`e_total` = the correlation energy).
    pub components: SpinComponents,
    /// MP2 correlation energy (= `components.e_total`).
    pub mp2_corr: f64,
    /// `rhf.energy + mp2_corr`. NOTE: the RHF energy is the reference's own
    /// (it carries `−v_M N_e/2` under ewald vs none); only the correlation
    /// part is made convention-consistent here.
    pub total_energy: f64,
    /// Gamma-point Madelung constant `v_M` of the cell.
    pub madelung: f64,
    /// Shift actually added to every active occupied ε (0, −v_M or +v_M).
    pub occ_shift: f64,
    /// Correlated occupied / virtual counts.
    pub nocc_active: usize,
    pub nvir: usize,
    /// Rows of the fitted B (`None` for the dense oracle).
    pub naux: Option<usize>,
}

/// `B[k, μ·nao+ν]` (row k) → `B[k, i·nvir+a] = Σ_μν C_o[μ,i] B_k[μ,ν] C_v[ν,a]`,
/// the `(naux, nocc·nvir)` layout ferric-mp2's `spin_components_from_b_ov`
/// consumes. Result is standard layout; every intermediate is read by index
/// or slice only (a `dot` output may be column-major when `nvir = 1`, e.g.
/// H2/minimal — never flat-indexed here).
pub fn b_ov_from_ao_b(
    b: &Array2<f64>,
    nao: usize,
    c_occ: ArrayView2<'_, f64>,
    c_vir: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, FerricError> {
    let naux = b.nrows();
    if b.ncols() != nao * nao || c_occ.nrows() != nao || c_vir.nrows() != nao {
        return Err(FerricError::General(format!(
            "b_ov_from_ao_b: B {:?}, C_occ {:?}, C_vir {:?} inconsistent with nao = {nao}",
            b.dim(),
            c_occ.dim(),
            c_vir.dim()
        )));
    }
    let (nocc, nvir) = (c_occ.ncols(), c_vir.ncols());
    let b_std = b.as_standard_layout();
    let flat = b_std
        .as_slice()
        .ok_or_else(|| FerricError::General("b_ov_from_ao_b: B not contiguous".into()))?;
    // Rows (k, μ), columns ν.
    let bv = ArrayView2::from_shape((naux * nao, nao), flat)
        .map_err(|e| FerricError::General(format!("b_ov_from_ao_b: reshape: {e}")))?;
    let x = bv.dot(&c_vir); // (naux·nao, nvir): X[(k,μ), a] = Σ_ν B_k[μ,ν] C_v[ν,a]
    let mut out = Array2::<f64>::zeros((naux, nocc * nvir));
    for k in 0..naux {
        let xk = x.slice(s![k * nao..(k + 1) * nao, ..]); // (nao, nvir)
        let ov = c_occ.t().dot(&xk); // (nocc, nvir)
        for i in 0..nocc {
            for a in 0..nvir {
                out[(k, i * nvir + a)] = ov[(i, a)];
            }
        }
    }
    Ok(out)
}

/// Exact `(ia|jb)` as an `(nocc·nvir, nocc·nvir)` matrix from a dense
/// `(nao², nao²)` ERI (the [`DenseAftEri::eri`] layout): `Wᵀ I W`,
/// `W[μ·nao+ν, i·nvir+a] = C_o[μ,i] C_v[ν,a]`. Test/oracle scale only.
pub fn ovov_from_dense(
    eri: &Array2<f64>,
    nao: usize,
    c_occ: ArrayView2<'_, f64>,
    c_vir: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, FerricError> {
    let n2 = nao * nao;
    if eri.dim() != (n2, n2) || c_occ.nrows() != nao || c_vir.nrows() != nao {
        return Err(FerricError::General(format!(
            "ovov_from_dense: ERI {:?}, C_occ {:?}, C_vir {:?} inconsistent with nao = {nao}",
            eri.dim(),
            c_occ.dim(),
            c_vir.dim()
        )));
    }
    let (nocc, nvir) = (c_occ.ncols(), c_vir.ncols());
    let mut w = Array2::<f64>::zeros((n2, nocc * nvir));
    for m in 0..nao {
        for n in 0..nao {
            for i in 0..nocc {
                for a in 0..nvir {
                    w[(m * nao + n, i * nvir + a)] = c_occ[(m, i)] * c_vir[(n, a)];
                }
            }
        }
    }
    let iw = eri.dot(&w);
    Ok(w.t().dot(&iw))
}

/// Gamma-point closed-shell MP2 from a converged Gamma RHF `rhf` (from
/// `solve_rhf_injected` with RS-GDF or dense J/K) on `cell`. See the module
/// doc for the integral sources and the denominator convention.
pub fn gamma_mp2(
    cell: &Cell,
    rhf: &ScfResult,
    ints: GammaMp2Integrals<'_>,
    cfg: &GammaMp2Config,
) -> Result<GammaMp2Result, FerricError> {
    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "gamma_mp2: closed-shell RHF reference required (got a non-restricted result)".into(),
        ));
    }
    if !rhf.converged {
        return Err(FerricError::General(
            "gamma_mp2: the RHF reference did not converge".into(),
        ));
    }
    let nelec = cell.mol().nelec();
    if nelec <= 0 || nelec % 2 != 0 {
        return Err(FerricError::General(format!(
            "gamma_mp2: closed shell needs an even, positive electron count (got {nelec})"
        )));
    }
    let nocc_total = (nelec / 2) as usize;
    let nocc = active_occ(nocc_total, cfg.frozen_core)?;
    let first_occ = cfg.frozen_core;
    let c = rhf.mos_r();
    let eps_in = rhf.eps_r();
    let (nao, nmo) = c.dim();
    if eps_in.len() != nmo || nmo <= nocc_total {
        return Err(FerricError::General(format!(
            "gamma_mp2: {nmo} MOs / {} orbital energies with {nocc_total} occupied: no virtuals \
             or inconsistent reference",
            eps_in.len()
        )));
    }
    let nvir = nmo - nocc_total;

    let madelung = madelung_constant(cell)?;
    let occ_shift = occupied_shift(cfg.reference_exxdiv, cfg.denominators, madelung);
    let mut eps = eps_in.to_vec();
    for e in eps.iter_mut().take(nocc_total) {
        *e += occ_shift;
    }

    let c_occ = c.slice(s![.., first_occ..first_occ + nocc]);
    let c_vir = c.slice(s![.., nocc_total..]);
    let nov = nocc * nvir;
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));

    let (components, naux) = match ints {
        GammaMp2Integrals::RsGdf(gdf) => {
            let b = gdf.b();
            let naux = b.nrows();
            if b.ncols() != nao * nao {
                return Err(FerricError::General(format!(
                    "gamma_mp2: RS-GDF B has {} columns, the reference has nao = {nao}",
                    b.ncols()
                )));
            }
            ledger.reserve(
                &format!("Gamma MP2 half-transformed B[k,μ,a] (naux = {naux}, nao = {nao}, nvir = {nvir})"),
                bytes_of((naux as u64).saturating_mul(nao as u64), nvir.saturating_mul(8)),
            )?;
            ledger.reserve(
                &format!("Gamma MP2 B[k,ia] (naux = {naux}, nocc = {nocc}, nvir = {nvir})"),
                bytes_of(naux as u64, nov.saturating_mul(8)),
            )?;
            ledger.reserve(
                &format!(
                    "Gamma MP2 kernel G_i block per worker (nvir = {nvir}, nocc·nvir = {nov})"
                ),
                bytes_of(nvir as u64, nov.saturating_mul(8)),
            )?;
            let b_ov = b_ov_from_ao_b(b, nao, c_occ, c_vir)?;
            (
                spin_components_from_b_ov(&b_ov, &eps, nocc, nvir, first_occ, nocc_total),
                Some(naux),
            )
        }
        GammaMp2Integrals::DenseAft(eri) => {
            let n2 = nao * nao;
            ledger.reserve(
                &format!("Gamma MP2 dense W = C_o⊗C_v + I·W (nao = {nao}, nocc·nvir = {nov})"),
                bytes_of(n2 as u64, nov.saturating_mul(16)),
            )?;
            ledger.reserve(
                &format!("Gamma MP2 dense (ia|jb) (nocc·nvir = {nov})"),
                bytes_of(nov as u64, nov.saturating_mul(8)),
            )?;
            let g = ovov_from_dense(eri.eri(), nao, c_occ, c_vir)?;
            (
                spin_components_from_g(&g, &eps, nocc, nvir, first_occ, nocc_total),
                None,
            )
        }
    };
    if !components.e_total.is_finite() {
        return Err(FerricError::General(format!(
            "gamma_mp2: non-finite correlation energy {} (zero denominator? occupied/virtual \
             degeneracy after a {occ_shift:+} occupied shift)",
            components.e_total
        )));
    }
    let mp2_corr = components.e_total;
    Ok(GammaMp2Result {
        components,
        mp2_corr,
        total_energy: rhf.energy + mp2_corr,
        madelung,
        occ_shift,
        nocc_active: nocc,
        nvir,
        naux,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denominators_parse_is_strict() {
        assert_eq!(
            Mp2Denominators::parse_config_str("Shifted").unwrap(),
            Mp2Denominators::MadelungShifted
        );
        assert_eq!(
            Mp2Denominators::parse_config_str("unshifted").unwrap(),
            Mp2Denominators::Unshifted
        );
        for bad in ["", "ewald", "shifted ", "madelung"] {
            assert!(Mp2Denominators::parse_config_str(bad).is_err(), "{bad:?}");
        }
    }

    /// B from an exact factorization of a random PSD (ia|jb): the two
    /// transforms must agree (dense `Wᵀ I W` vs per-k `C_oᵀ B_k C_v`), with
    /// nvir = 1 (the column-major `dot` layout case) and nvir > 1.
    #[test]
    fn b_ov_and_dense_transforms_agree_on_an_exact_factorization() {
        for (nao, nocc, nvir, naux) in [(2usize, 1usize, 1usize, 3usize), (4, 2, 2, 7)] {
            let mut seed = 12345u64;
            let mut rnd = || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
            };
            let mut b = Array2::<f64>::zeros((naux, nao * nao));
            for k in 0..naux {
                for m in 0..nao {
                    for n in 0..=m {
                        let v = rnd();
                        b[(k, m * nao + n)] = v;
                        b[(k, n * nao + m)] = v;
                    }
                }
            }
            let c = Array2::from_shape_fn((nao, nocc + nvir), |_| rnd());
            let (co, cv) = (c.slice(s![.., ..nocc]), c.slice(s![.., nocc..]));
            let eri = b.t().dot(&b);
            let g_dense = ovov_from_dense(&eri, nao, co, cv).unwrap();
            let b_ov = b_ov_from_ao_b(&b, nao, co, cv).unwrap();
            let g_b = b_ov.t().dot(&b_ov);
            let d = g_dense
                .iter()
                .zip(g_b.iter())
                .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
            assert!(d < 1e-13, "nao {nao}: {d:.3e}");
            // Same energy through the two ferric-mp2 loops.
            let eps: Vec<f64> = (0..nocc + nvir)
                .map(|p| {
                    if p < nocc {
                        -1.0 - p as f64
                    } else {
                        0.5 + p as f64
                    }
                })
                .collect();
            let e1 = spin_components_from_g(&g_dense, &eps, nocc, nvir, 0, nocc);
            let e2 = spin_components_from_b_ov(&b_ov, &eps, nocc, nvir, 0, nocc);
            assert!((e1.e_total - e2.e_total).abs() < 1e-13);
            assert!((e1.e_os - e2.e_os).abs() < 1e-13);
        }
    }
}
