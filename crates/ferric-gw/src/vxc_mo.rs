//! Mean-field exchange-correlation potential in the MO basis, for GW on a
//! Kohn–Sham reference (closed and open shell).
//!
//! At a KS reference, the QP equation reads
//!   ε^QP = ε_KS + (Σ_x^GW − v_mf) + Σ_c
//! where v_mf is everything in the KS Fock besides h + J: the semilocal
//! v_xc (+ VV10) AND, for a hybrid or range-separated hybrid, the exact
//! exchange the SCF put there,
//!   `v_mf,σ = v_xc,σ − [c_sr K_erfc(ω)[D_σ] + c_lr K_erf(ω)[D_σ]]`
//! (global hybrid: ω = 0, c_sr = c_lr = c_x, one Coulomb K). Σ_x^GW is the
//! FULL exchange, so leaving the K term out counts the exact-exchange
//! fraction twice (book/src/20-gw-bse/05-continuation-and-qp-equation.md,
//! "Hybrid references"). For
//! UHF/ROHF references the whole shift is zero (Σ_x^GW = −K) and callers
//! pass no v_mf at all.
//!
//! The K term is built with exact four-centre integrals through the SCF's
//! own `build_jk`, on the operator(s) the SCF's `k_mix` names. A reference
//! SCF that used RI-K differs from that by its fitting error only.

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::ks::{KsXc, KsXcUks};
use ferric_dft::xc_trait::{KMix, UksXcContribution, XcContribution};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::{ScfResult, Spin};
use ndarray::{Array1, Array2};

/// Diagonal of the KS mean-field potential v_mf (semilocal V_xc + VV10 +
/// the hybrid's exact-exchange term) in the MO basis.
/// Returns (vxc_diag_α, vxc_diag_β), each of length `nmo` (equal for a
/// restricted reference).
///
/// Builds the AO-basis V_xc (+ V_nl/VV10) matrix via the same
/// `XcContribution`/`UksXcContribution::add_xc[_uks]` entry point the SCF
/// itself uses to build the KS Fock — this automatically stays consistent
/// with whichever grid-cache strategy `KsXc`/`KsXcUks` picked internally
/// (the full cached path, or the budget-triggered batched fallback), rather
/// than reaching into the AO/χ tensors directly (which are a private
/// implementation detail of `KsXc`/`KsXcUks`, not guaranteed to be
/// eagerly-materialized).
///
/// For an SCF on the default grids (`AtomicGridConfig::production()` and
/// `default_nlc()`) and the default `RhfConfig::gc_rotation` (unset); a
/// reference SCF on other grids or with another `gc_rotation` must use
/// [`vxc_diagonal_mo_on_grids`] with them, or V_xc is not the potential that
/// entered its Fock matrix (to the grid's accuracy, and to rounding for the
/// rotation).
pub fn vxc_diagonal_mo(
    mol: &Molecule,
    bs: &BasisSet,
    xc_name: &str,
    scf: &ScfResult,
) -> Result<(Array1<f64>, Array1<f64>), FerricError> {
    vxc_diagonal_mo_on_grids(
        mol,
        bs,
        xc_name,
        scf,
        &AtomicGridConfig::production(),
        &AtomicGridConfig::default_nlc(),
        None,
        None,
    )
}

/// [`vxc_diagonal_mo`] on the reference SCF's own main and VV10 grids, with
/// the exact exchange's four-centre K built under the reference SCF's
/// `RhfConfig::gc_rotation` (`ferric_scf::direct_jk::gc_rotation`; `None` =
/// unset: `FERRIC_GC_ROTATE`, then `auto`) and `RhfConfig::jk_digest`
/// (`ferric_scf::quartet_scatter::JkDigest`; `None` = `FERRIC_JK_DIGEST`,
/// then `pair`).
#[allow(clippy::too_many_arguments)]
pub fn vxc_diagonal_mo_on_grids(
    mol: &Molecule,
    bs: &BasisSet,
    xc_name: &str,
    scf: &ScfResult,
    main_grid: &AtomicGridConfig,
    nlc_grid: &AtomicGridConfig,
    gc_rotation: Option<ferric_scf::direct_jk::GcRotate>,
    jk_digest: Option<ferric_scf::quartet_scatter::JkDigest>,
) -> Result<(Array1<f64>, Array1<f64>), FerricError> {
    if matches!(scf.spin, Spin::Restricted) {
        // Full closed-shell density: solve_rhf stores density_alpha = 0.5·D_full,
        // so the physical density matrix is 2·density_alpha (rhf.rs).
        let d_full = 2.0 * &scf.density_alpha;
        let ks_xc = KsXc::new(mol, bs, xc_name, main_grid, nlc_grid)
            .map_err(|e| FerricError::General(format!("KsXc::new: {e:?}")))?;
        let mut vxc_ao = ndarray::Array2::<f64>::zeros((d_full.nrows(), d_full.ncols()));
        let _e_xc_plus_nl = ks_xc.add_xc(&d_full, &mut vxc_ao);
        // Closed shell: F = … − ½ K_mix[D] = … − K_mix[D/2] = − K_mix[D_α].
        add_exact_exchange(
            mol,
            bs,
            ks_xc.k_mix(),
            gc_rotation,
            jk_digest,
            &mut [(&scf.density_alpha, &mut vxc_ao)],
        )?;
        let diag = mo_diagonal(&vxc_ao, scf.mos_r());
        return Ok((diag.clone(), diag));
    }
    let d_a = &scf.density_alpha;
    let d_b = scf
        .density_beta
        .as_ref()
        .ok_or_else(|| FerricError::General("vxc_diagonal_mo: missing density_beta".into()))?;

    // Build the same grid + AO values KsXcUks uses for SCF — guarantees the
    // v_xc we subtract matches what entered the KS Fock.
    // The SCF's grids, so V_xc is evaluated on the quadrature the density was
    // converged on.
    let ks_xc = KsXcUks::new(mol, bs, xc_name, main_grid, nlc_grid)
        .map_err(|e| FerricError::General(format!("KsXcUks::new: {e:?}")))?;

    let mut vxc_a = ndarray::Array2::<f64>::zeros((d_a.nrows(), d_a.ncols()));
    let mut vxc_b = ndarray::Array2::<f64>::zeros((d_b.nrows(), d_b.ncols()));
    let _e_xc_plus_nl = ks_xc.add_xc_uks(d_a, d_b, &mut vxc_a, &mut vxc_b);
    add_exact_exchange(
        mol,
        bs,
        ks_xc.k_mix(),
        gc_rotation,
        jk_digest,
        &mut [(d_a, &mut vxc_a), (d_b, &mut vxc_b)],
    )?;

    // Transform to MO basis and take the diagonal: (vxc_σ)_pp = c_σ_p^T v_xc^AO c_σ_p.
    let (mos_a_arr, mos_b_arr) = (scf.mos_a(), scf.mos_b());
    let diag_a = mo_diagonal(&vxc_a, mos_a_arr);
    let diag_b = mo_diagonal(&vxc_b, mos_b_arr);
    Ok((diag_a, diag_b))
}

/// `v_σ −= c_sr K_erfc(ω)[D_σ] + c_lr K_erf(ω)[D_σ]` (ω = 0: `c_sr K[D_σ]`)
/// for each `(D_σ, v_σ)`, with `K[D]_μν = Σ_λσ (μλ|νσ) D_λσ` — the
/// exact-exchange term of the KS Fock for the mix `k_mix` (rhf.rs/uhf.rs
/// Fock assembly). A pure functional (`k_mix` all zero) builds nothing and
/// leaves `v` untouched bit for bit.
fn add_exact_exchange(
    mol: &Molecule,
    bs: &BasisSet,
    k_mix: KMix,
    gc_rotation: Option<ferric_scf::direct_jk::GcRotate>,
    jk_digest: Option<ferric_scf::quartet_scatter::JkDigest>,
    pairs: &mut [(&Array2<f64>, &mut Array2<f64>)],
) -> Result<(), FerricError> {
    let terms: Vec<(Operator, f64)> = if k_mix.omega > 0.0 {
        vec![
            (Operator::erfc(k_mix.omega), k_mix.sr), // c_sr K^erfc_ω
            (Operator::erf(k_mix.omega), k_mix.lr),  // c_lr K^erf_ω
        ]
    } else {
        vec![(Operator::coulomb(), k_mix.sr)] // c_x K
    };
    let terms: Vec<_> = terms.into_iter().filter(|&(_, c)| c != 0.0).collect();
    if terms.is_empty() {
        return Ok(());
    }
    let prep = PreparedBasis::new(mol, bs)?;
    let ctx = ferric_core::parallel::ParallelContext::default();
    // RhfConfig's default integral screening threshold.
    let thresh = ferric_scf::rhf::RhfConfig::default().integral_thresh;
    for (op, c) in terms {
        let bounds = SchwarzBounds::compute(op, &prep)?;
        // The reference SCF's contraction rotation.
        bounds.gc.set_request(gc_rotation);
        // ... and its J/K digest.
        bounds.digest.set(jk_digest);
        for (d, v) in pairs.iter_mut() {
            let n = d.nrows();
            let (mut j, mut k) = (Array2::<f64>::zeros((n, n)), Array2::<f64>::zeros((n, n)));
            ferric_scf::rhf::build_jk(&ctx, &prep, &bounds, thresh, d, &mut j, &mut k)?;
            v.scaled_add(-c, &k); // v_σ −= c K[D^σ]
        }
    }
    Ok(())
}

fn mo_diagonal(ao: &ndarray::Array2<f64>, c: &ndarray::Array2<f64>) -> Array1<f64> {
    let nmo = c.ncols();
    let mut diag = Array1::<f64>::zeros(nmo);
    let ac = ao.dot(c);
    for p in 0..nmo {
        let mut acc = 0.0;
        for mu in 0..c.nrows() {
            acc += c[(mu, p)] * ac[(mu, p)];
        }
        diag[p] = acc;
    }
    diag
}
