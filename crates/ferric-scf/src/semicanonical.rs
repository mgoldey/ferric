//! Semi-canonicalization of ROHF/ROKS orbitals for unrestricted post-SCF methods.
//!
//! # Why
//!
//! ROHF converges a *single* set of spatial orbitals against an effective (Roothaan)
//! Fock operator. The true spin Fock operators `F_α` and `F_β` are **not** diagonal in
//! that basis, so a correlated method fed ROHF orbitals directly has no valid orbital
//! energies to build denominators from.
//!
//! Semi-canonicalization fixes this: build `F_α`/`F_β` once from the converged ROHF
//! density, then diagonalize each **within the occupied and virtual blocks separately**.
//! The result is a per-spin orbital set in which each spin's Fock matrix is diagonal
//! inside each block, so `ε_i + ε_j − ε_a − ε_b` denominators are again meaningful.
//!
//! It is "semi"-canonical because the occupied–virtual block of `F_α`/`F_β` is left
//! untouched — non-zero, since ROHF stationarity only annihilates the *effective* Fock's
//! ov block, not each spin's. Those elements do not enter doubles denominators. They DO
//! mean the reference is not a UHF stationary point, so any method with singles
//! amplitudes must include the `f_ia` terms explicitly.
//!
//! This is the standard prescription for ROHF-based UCC/UMP2, and is exactly what
//! Ransford & Carter-Fenk specify for ωB97X-L-V (PCCP 2026, 28, 14428): *"we use
//! restricted open-shell orbitals to converge the self-consistent field procedure,
//! followed by a single unrestricted (Kohn–Sham) Fock build and semi-canonicalization
//! routine prior to inputting the Kohn–Sham orbitals into an unrestricted coupled-cluster
//! code."*
//!
//! # Kohn–Sham references
//!
//! Passing an [`crate::semicanonical::XcSpec`] builds an unrestricted **Kohn–Sham** `F_σ`: semilocal `V^σ_xc`
//! (plus VV10) from the named functional, with exact exchange following that
//! functional's own `KMix` — full `K` for Hartree–Fock, scaled `K` for a plain hybrid,
//! and the split `c_sr·K_SR + c_lr·K_LR` for a range-separated one such as ωB97X-L-V.
//! Passing `None` gives the plain Hartree–Fock `F_σ = h + J[D] − K[D_σ]` appropriate to
//! a ROHF reference.

use crate::result::{ScfResult, Spin};
use crate::screening::SchwarzBounds;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{Array1, Array2};
use ndarray_linalg::Eigh;
use ndarray_linalg::UPLO;

/// Which density functional to use when building the unrestricted Kohn–Sham `F_σ`.
///
/// Mirrors how `solve_uhf` constructs its own XC contribution (`uhf.rs:119-129`), so a
/// semi-canonicalized ROKS reference sees the same potential the SCF converged against.
#[derive(Debug, Clone)]
pub struct XcSpec<'a> {
    /// Functional name, e.g. `"wB97X-L-V"` — resolved by `ferric_dft::libxc`.
    pub name: &'a str,
    /// Main XC quadrature grid. `None` uses the same default as the SCF drivers.
    pub grid: Option<ferric_dft::grid::AtomicGridConfig>,
    /// Non-local (VV10) grid. `None` uses the drivers' `{n_radial: 50, n_angular: 50}`.
    pub nlc_grid: Option<ferric_dft::grid::AtomicGridConfig>,
}

impl<'a> XcSpec<'a> {
    /// Convenience constructor using the default grids.
    pub fn new(name: &'a str) -> Self {
        Self { name, grid: None, nlc_grid: None }
    }

    fn build(
        &self,
        mol: &ferric_core::mol::Molecule,
        prep: &PreparedBasis,
    ) -> Result<Box<dyn ferric_dft::xc_trait::UksXcContribution>, FerricError> {
        let main = self.grid.clone().unwrap_or_default();
        let nlc = self.nlc_grid.clone().unwrap_or(ferric_dft::grid::AtomicGridConfig {
            n_radial: 50,
            n_angular: 50,
            ..Default::default()
        });
        let ks = ferric_dft::ks::KsXcUks::new(mol, prep.basis_set(), self.name, &main, &nlc)
            .map_err(|e| {
                FerricError::General(format!("KsXcUks init for {}: {e:?}", self.name))
            })?;
        Ok(Box::new(ks) as Box<dyn ferric_dft::xc_trait::UksXcContribution>)
    }
}

/// A semi-canonical open-shell orbital set derived from a ROHF reference.
#[derive(Debug, Clone)]
pub struct SemicanonicalOrbitals {
    /// α MO coefficients (nbasis, nmo), semi-canonical within occ and virt blocks.
    pub mos_alpha: Array2<f64>,
    /// β MO coefficients (nbasis, nmo).
    pub mos_beta: Array2<f64>,
    /// α orbital energies — diagonal of `F_α` in the new basis.
    pub eps_alpha: Vec<f64>,
    /// β orbital energies — diagonal of `F_β` in the new basis.
    pub eps_beta: Vec<f64>,
    /// Number of occupied α orbitals.
    pub nocc_alpha: usize,
    /// Number of occupied β orbitals.
    pub nocc_beta: usize,
    /// Largest |F_α| occupied–virtual element in the semi-canonical basis.
    ///
    /// Non-zero by construction (see the module docs). Reported so callers can judge how
    /// far the reference is from a UHF stationary point: a large value means singles
    /// contributions matter.
    pub max_ov_alpha: f64,
    /// Largest |F_β| occupied–virtual element.
    pub max_ov_beta: f64,
}

/// Diagonalize `f_mo` within one index block, returning the rotation and eigenvalues.
///
/// Takes the block `[range, range]` of the MO-basis Fock matrix, symmetrizes it (guarding
/// against accumulated asymmetry), and returns its eigen-decomposition.
fn diagonalize_block(
    f_mo: &Array2<f64>,
    start: usize,
    end: usize,
) -> Result<(Array2<f64>, Array1<f64>), FerricError> {
    let n = end - start;
    if n == 0 {
        return Ok((Array2::zeros((0, 0)), Array1::zeros(0)));
    }
    let mut block = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            // Symmetrize: F should be symmetric, but round-off in the AO->MO transform
            // can leave a tiny asymmetry that `eigh` would silently ignore half of.
            block[[i, j]] = 0.5 * (f_mo[[start + i, start + j]] + f_mo[[start + j, start + i]]);
        }
    }
    let (evals, evecs) = block
        .eigh(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("semi-canonical block eigh failed: {e}")))?;
    Ok((evecs, evals))
}

/// Rotate one spin's MOs into its semi-canonical basis.
///
/// Returns the rotated coefficients, the new orbital energies (block-diagonal entries of
/// `F` in that basis), and the largest surviving occ–virt element.
fn semicanonicalize_spin(
    c: &Array2<f64>,
    f_ao: &Array2<f64>,
    nocc: usize,
) -> Result<(Array2<f64>, Vec<f64>, f64), FerricError> {
    let nmo = c.ncols();
    let f_mo = c.t().dot(f_ao).dot(c);

    let (u_occ, e_occ) = diagonalize_block(&f_mo, 0, nocc)?;
    let (u_vir, e_vir) = diagonalize_block(&f_mo, nocc, nmo)?;

    // Block-diagonal rotation: occupied and virtual spaces rotate independently, so the
    // occupied SPAN is preserved and the reference determinant is unchanged.
    let mut c_new = Array2::<f64>::zeros((c.nrows(), nmo));
    if nocc > 0 {
        c_new
            .slice_mut(ndarray::s![.., ..nocc])
            .assign(&c.slice(ndarray::s![.., ..nocc]).dot(&u_occ));
    }
    if nmo > nocc {
        c_new
            .slice_mut(ndarray::s![.., nocc..])
            .assign(&c.slice(ndarray::s![.., nocc..]).dot(&u_vir));
    }

    let mut eps = Vec::with_capacity(nmo);
    eps.extend(e_occ.iter().copied());
    eps.extend(e_vir.iter().copied());

    // The occ-virt block in the NEW basis: non-zero, and worth reporting.
    let f_new = c_new.t().dot(f_ao).dot(&c_new);
    let mut max_ov = 0.0f64;
    for i in 0..nocc {
        for a in nocc..nmo {
            max_ov = max_ov.max(f_new[[i, a]].abs());
        }
    }

    Ok((c_new, eps, max_ov))
}

/// Build `F_α`/`F_β` from a converged ROHF density and semi-canonicalize its orbitals.
///
/// Implements the standard ROHF → unrestricted post-SCF bridge: one unrestricted Fock
/// build, then independent occ-occ and virt-virt diagonalizations per spin.
///
/// # Errors
///
/// * The reference is not `Spin::RestrictedOpen`. A UHF result is *already*
///   semi-canonical (its `F_σ` are diagonal by construction) and a restricted result has
///   no open shell, so neither needs this.
/// * The reference did not converge. Semi-canonicalizing a garbage density produces
///   garbage orbital energies that look perfectly well-formed.
///
/// # Kohn–Sham references (ROKS)
///
/// Pass the functional name in `xc` to build an unrestricted **Kohn–Sham** `F_σ`:
/// the semilocal `V^σ_xc` is added via [`ferric_dft::ks::KsXcUks`], and the exact-exchange
/// fraction follows the functional's own `KMix` — full `K` for Hartree–Fock, a scaled
/// `K` for a plain hybrid, and the split `c_sr·K_SR + c_lr·K_LR` for a range-separated
/// one like ωB97X-L-V. Pass `None` for a plain ROHF reference.
///
/// Getting this wrong is silent: an HF `F_σ` built on a ROKS density yields orbital
/// energies that look reasonable but are not the Kohn–Sham ones, which is why the
/// functional must be named explicitly rather than guessed.
pub fn semicanonicalize(
    ctx: &ParallelContext,
    mol: &ferric_core::mol::Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    rohf: &ScfResult,
    integral_thresh: f64,
    xc: Option<&XcSpec<'_>>,
) -> Result<SemicanonicalOrbitals, FerricError> {
    if !matches!(rohf.spin, Spin::RestrictedOpen) {
        return Err(FerricError::General(format!(
            "semicanonicalize expects a ROHF/ROKS reference, got {:?}; UHF orbitals are \
             already semi-canonical and restricted ones have no open shell",
            rohf.spin
        )));
    }
    if !rohf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: rohf.iterations,
            last_energy: rohf.energy,
        });
    }

    let d_a = &rohf.density_alpha;
    let d_b = rohf
        .density_beta
        .as_ref()
        .ok_or_else(|| FerricError::General("ROHF result carries no beta density".into()))?;
    let d_total = d_a + d_b;

    let n = prep.nbasis();
    let h = ferric_integrals::oneelectron::hcore(prep);

    // Build the XC contribution first: its k_mix decides how exchange is assembled.
    let xc_contrib = match xc {
        Some(spec) => Some(spec.build(mol, prep)?),
        None => None,
    };
    let k_mix = xc_contrib.as_ref().map(|c| c.k_mix()).unwrap_or_default();

    // Coulomb from the TOTAL density (one build).
    let (mut j_tot, mut k_scratch) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    crate::rhf::build_jk(ctx, prep, bounds, integral_thresh, &d_total, &mut j_tot, &mut k_scratch)?;

    let mut f_a = &h + &j_tot;
    let mut f_b = &h + &j_tot;

    // Exact exchange, per spin, following the functional's k_mix:
    //   HF / no XC  -> full K[D_sigma]
    //   plain hybrid -> k_mix.sr * K[D_sigma]
    //   RSH          -> c_sr*K_SR[D_sigma] + c_lr*K_LR[D_sigma]
    if k_mix.omega > 0.0 {
        let ooc_budget = ferric_core::memory::resolve_budget_bytes(None);
        let (mut dfk_sr, mut dfk_lr) = crate::fock_assembly::build_rsh_dfk_pair(
            ctx,
            mol,
            prep,
            None,
            k_mix.omega,
            ooc_budget,
        )?;
        // occ_factor/scale = 1.0: these are per-spin Focks built from per-spin
        // densities, matching how solve_uhf calls this (uhf.rs:314-319).
        crate::fock_assembly::subtract_rsh_exchange(
            &mut dfk_sr, &mut dfk_lr, d_a, None, 1.0, &mut f_a, k_mix.sr, k_mix.lr, 1.0,
        )?;
        crate::fock_assembly::subtract_rsh_exchange(
            &mut dfk_sr, &mut dfk_lr, d_b, None, 1.0, &mut f_b, k_mix.sr, k_mix.lr, 1.0,
        )?;
    } else {
        // Full K for a pure-HF reference; k_mix.sr for a plain hybrid. A pure
        // (non-hybrid) functional has k_mix.sr == 0 and needs no exchange at all.
        let c_k = if xc_contrib.is_some() { k_mix.sr } else { 1.0 };
        if c_k != 0.0 {
            let (mut j_scratch, mut k_a) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
            crate::rhf::build_jk(ctx, prep, bounds, integral_thresh, d_a, &mut j_scratch, &mut k_a)?;
            let mut k_b = Array2::zeros((n, n));
            j_scratch.fill(0.0);
            crate::rhf::build_jk(ctx, prep, bounds, integral_thresh, d_b, &mut j_scratch, &mut k_b)?;
            f_a.scaled_add(-c_k, &k_a);
            f_b.scaled_add(-c_k, &k_b);
        }
    }

    // Semilocal V^sigma_xc (+ VV10), added in place. This is the piece whose absence
    // made ROKS references wrong before.
    if let Some(c) = xc_contrib.as_ref() {
        c.add_xc_uks(d_a, d_b, &mut f_a, &mut f_b);
    }

    // ROHF stores one spatial MO set in mos_alpha; both spins start from it.
    let c = &rohf.mos_alpha;
    let (nocc_a, nocc_b) = rohf_occupations(mol)?;

    let (c_a, eps_a, max_ov_a) = semicanonicalize_spin(c, &f_a, nocc_a)?;
    let (c_b, eps_b, max_ov_b) = semicanonicalize_spin(c, &f_b, nocc_b)?;

    Ok(SemicanonicalOrbitals {
        mos_alpha: c_a,
        mos_beta: c_b,
        eps_alpha: eps_a,
        eps_beta: eps_b,
        nocc_alpha: nocc_a,
        nocc_beta: nocc_b,
        max_ov_alpha: max_ov_a,
        max_ov_beta: max_ov_b,
    })
}

impl SemicanonicalOrbitals {
    /// Repackage as an unrestricted [`ScfResult`], suitable for any consumer that
    /// expects UHF-shaped input.
    ///
    /// This is the practical payoff of semi-canonicalization. ferric's open-shell
    /// post-SCF code detects a ROHF result and falls back to α orbitals with the
    /// *effective* Fock's eigenvalues for BOTH spins — see the comment at
    /// `u_rimp2.rs:97` ("ROHF has no eps_beta — fall back to eps_alpha"). Feeding it
    /// the result of this conversion instead supplies genuine, distinct per-spin
    /// orbitals and orbital energies.
    ///
    /// `energy` is carried over from the ROHF reference unchanged: the block-diagonal
    /// rotation preserves the occupied span, so the reference determinant — and hence
    /// the SCF energy — is identical.
    ///
    /// The Fock matrices are NOT stored (`ScfResult` would need the AO-basis ones, which
    /// callers can rebuild); `fock_alpha` carries the ROHF effective Fock unchanged.
    /// Consumers of this conversion want the MOs and eigenvalues.
    pub fn to_unrestricted_result(&self, rohf: &ScfResult) -> ScfResult {
        let occ_dens = |c: &Array2<f64>, nocc: usize| -> Array2<f64> {
            let occ = c.slice(ndarray::s![.., ..nocc]);
            occ.dot(&occ.t())
        };
        let d_a = occ_dens(&self.mos_alpha, self.nocc_alpha);
        let d_b = occ_dens(&self.mos_beta, self.nocc_beta);
        ScfResult {
            spin: Spin::Unrestricted,
            energy: rohf.energy,
            density_total: &d_a + &d_b,
            density_alpha: d_a,
            density_beta: Some(d_b),
            mos_alpha: self.mos_alpha.clone(),
            mos_beta: Some(self.mos_beta.clone()),
            eps_alpha: self.eps_alpha.clone(),
            eps_beta: Some(self.eps_beta.clone()),
            fock_alpha: rohf.fock_alpha.clone(),
            fock_beta: None,
            converged: rohf.converged,
            exit: rohf.exit,
            iterations: rohf.iterations,
            computed_quartets: rohf.computed_quartets,
            induced_dipoles: rohf.induced_dipoles.clone(),
        }
    }
}

/// Derive (nocc_α, nocc_β) from the molecule's electron count and multiplicity.
///
/// Same derivation `solve_uhf` uses (`uhf.rs:161-168`), taken from the `Molecule` rather
/// than inferred from the density: `ScfResult` does not record occupations, and
/// reconstructing them from `tr(D_σ S)` would need the overlap matrix that
/// `ScfResult` also does not carry.
fn rohf_occupations(mol: &ferric_core::mol::Molecule) -> Result<(usize, usize), FerricError> {
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    if two_s < 0 || (nelec - two_s) < 0 || (nelec + two_s) % 2 != 0 {
        return Err(FerricError::General(format!(
            "semicanonicalize: inconsistent electron count {nelec} and multiplicity {}",
            mol.multiplicity
        )));
    }
    Ok((((nelec + two_s) / 2) as usize, ((nelec - two_s) / 2) as usize))
}
