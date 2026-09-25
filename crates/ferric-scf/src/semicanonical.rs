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
        Self {
            name,
            grid: None,
            nlc_grid: None,
        }
    }

    fn build(
        &self,
        mol: &ferric_core::mol::Molecule,
        prep: &PreparedBasis,
    ) -> Result<Box<dyn ferric_dft::xc_trait::UksXcContribution>, FerricError> {
        let main = self.grid.clone().unwrap_or_default();
        let nlc = self
            .nlc_grid
            .clone()
            .unwrap_or(ferric_dft::grid::AtomicGridConfig {
                n_radial: 50,
                n_angular: 50,
                ..Default::default()
            });
        let ks = ferric_dft::ks::KsXcUks::new(mol, prep.basis_set(), self.name, &main, &nlc)
            .map_err(|e| FerricError::General(format!("KsXcUks init for {}: {e:?}", self.name)))?;
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
    // V_ECP included, as the SCF's hcore has it (driver::prepare); an external
    // potential is not available here, so an embedded reference must use
    // `semicanonicalize_from_spin_focks`, which reuses the SCF's own F_σ.
    let h = ferric_integrals::oneelectron::hcore_ecp(prep, mol, prep.basis_set());

    // Build the XC contribution first: its k_mix decides how exchange is assembled.
    let xc_contrib = match xc {
        Some(spec) => Some(spec.build(mol, prep)?),
        None => None,
    };
    let k_mix = xc_contrib.as_ref().map(|c| c.k_mix()).unwrap_or_default();

    // Coulomb from the TOTAL density (one build).
    let (mut j_tot, mut k_scratch) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    crate::rhf::build_jk(
        ctx,
        prep,
        bounds,
        integral_thresh,
        &d_total,
        &mut j_tot,
        &mut k_scratch,
    )?;

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
            &mut dfk_sr,
            &mut dfk_lr,
            d_a,
            None,
            1.0,
            &mut f_a,
            k_mix.sr,
            k_mix.lr,
            1.0,
        )?;
        crate::fock_assembly::subtract_rsh_exchange(
            &mut dfk_sr,
            &mut dfk_lr,
            d_b,
            None,
            1.0,
            &mut f_b,
            k_mix.sr,
            k_mix.lr,
            1.0,
        )?;
    } else {
        // Full K for a pure-HF reference; k_mix.sr for a plain hybrid. A pure
        // (non-hybrid) functional has k_mix.sr == 0 and needs no exchange at all.
        let c_k = if xc_contrib.is_some() { k_mix.sr } else { 1.0 };
        if c_k != 0.0 {
            let (mut j_scratch, mut k_a) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
            crate::rhf::build_jk(
                ctx,
                prep,
                bounds,
                integral_thresh,
                d_a,
                &mut j_scratch,
                &mut k_a,
            )?;
            let mut k_b = Array2::zeros((n, n));
            j_scratch.fill(0.0);
            crate::rhf::build_jk(
                ctx,
                prep,
                bounds,
                integral_thresh,
                d_b,
                &mut j_scratch,
                &mut k_b,
            )?;
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
    /// This is the practical payoff of semi-canonicalization: genuine, distinct
    /// per-spin orbitals and orbital energies for any UHF-shaped consumer. The
    /// unrestricted correlated entry points (U-RI-MP2, U-PDEP-RPA, U-GW, open-shell
    /// PDEP polarizabilities) apply it themselves to a ROHF input, through
    /// [`unrestricted_reference`].
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
            stability: None,
            // Same energy as the ROHF reference, so the same J/K builders.
            df_jk: rohf.df_jk.clone(),
            rohf_spin_focks: rohf.rohf_spin_focks.clone(),
        }
    }
}

/// Semi-canonicalize a ROHF/ROKS result with the spin Fock matrices its own SCF
/// converged against ([`ScfResult::rohf_spin_focks`]).
///
/// Same block-diagonal rotation as [`semicanonicalize`], but no Fock rebuild: the
/// stored `(F_α, F_β)` already carry everything the SCF put in them — XC (so a ROKS
/// reference gets its Kohn–Sham `F_σ` without naming the functional again), the
/// density-fitting route, external point charges and solvent reaction fields. The
/// stored Focks are those of the final SCF iteration, i.e. built from the density one
/// diagonalization before the returned MOs; at convergence the two differ at the
/// density-convergence level.
///
/// Unlike [`semicanonicalize`] this does not refuse an unconverged reference: it is
/// the bridge every unrestricted correlated method takes on a ROHF input, and those
/// methods already report (and the drivers already warn on) the reference's
/// convergence themselves.
///
/// # Errors
///
/// * `rohf.spin` is not `Spin::RestrictedOpen`.
/// * `rohf.rohf_spin_focks` is `None` (a hand-built result): there are no `F_σ` to
///   diagonalize, and falling back to the effective Fock's eigenvalues is exactly the
///   defect this bridge removes.
pub fn semicanonicalize_from_spin_focks(
    mol: &ferric_core::mol::Molecule,
    rohf: &ScfResult,
) -> Result<SemicanonicalOrbitals, FerricError> {
    if !matches!(rohf.spin, Spin::RestrictedOpen) {
        return Err(FerricError::General(format!(
            "semicanonicalize_from_spin_focks expects a ROHF/ROKS reference, got {:?}",
            rohf.spin
        )));
    }
    let (f_a, f_b) = rohf.rohf_spin_focks.as_ref().ok_or_else(|| {
        FerricError::General(
            "ROHF/ROKS result carries no spin Fock matrices (ScfResult::rohf_spin_focks is \
             None), so it cannot be semi-canonicalized for an unrestricted correlated method; \
             use a result from solve_rohf, or semicanonicalize() to rebuild F_alpha/F_beta"
                .into(),
        )
    })?;
    let c = &rohf.mos_alpha;
    let (nocc_a, nocc_b) = rohf_occupations(mol)?;
    let (c_a, eps_a, max_ov_a) = semicanonicalize_spin(c, f_a, nocc_a)?;
    let (c_b, eps_b, max_ov_b) = semicanonicalize_spin(c, f_b, nocc_b)?;
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

/// The reference an unrestricted correlated method (U-RI-MP2, U-PDEP-RPA, U-GW, the
/// open-shell PDEP polarizabilities) should run on.
///
/// * `Unrestricted` / `Restricted`: the input itself, borrowed — bit-identical.
/// * `RestrictedOpen`: the semi-canonical UHF-shaped result of
///   [`semicanonicalize_from_spin_focks`], with `fock_alpha`/`fock_beta` set to the
///   spin Focks `F_α`/`F_β`. Its per-spin orbital energies are the occ–occ and
///   virt–virt eigenvalues of each spin's own Fock operator; the ROHF result's own
///   `eps_alpha` are eigenvalues of the Roothaan EFFECTIVE Fock, which belongs to
///   neither spin and (for α and β alike) shifts U-MP2/U-RPA correlation energies by
///   mEh. The occupied span of each spin is unchanged, so the reference determinant,
///   density and energy are the ROHF ones.
///
/// Methods with single excitations must still add the non-zero `f_ia` of each spin
/// (ROHF is not a UHF stationary point); the doubles-only methods that call this
/// (U-RI-MP2, RPA, GW) do not include singles.
pub fn unrestricted_reference<'a>(
    mol: &ferric_core::mol::Molecule,
    scf: &'a ScfResult,
) -> Result<std::borrow::Cow<'a, ScfResult>, FerricError> {
    if !matches!(scf.spin, Spin::RestrictedOpen) {
        return Ok(std::borrow::Cow::Borrowed(scf));
    }
    let sc = semicanonicalize_from_spin_focks(mol, scf)?;
    let mut u = sc.to_unrestricted_result(scf);
    if let Some((f_a, f_b)) = scf.rohf_spin_focks.as_ref() {
        u.fock_alpha = f_a.clone();
        u.fock_beta = Some(f_b.clone());
    }
    Ok(std::borrow::Cow::Owned(u))
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
    Ok((
        ((nelec + two_s) / 2) as usize,
        ((nelec - two_s) / 2) as usize,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::operator::Operator;

    fn oh() -> Molecule {
        Molecule::parse_xyz("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap()
    }

    fn setup(mol: &Molecule) -> (PreparedBasis, SchwarzBounds, ParallelContext) {
        let prep = PreparedBasis::new(mol, &basis::bundled("6-31g").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        (prep, bounds, ParallelContext::default())
    }

    fn cfg() -> crate::rhf::RhfConfig {
        crate::rhf::RhfConfig {
            energy_conv: 1e-11,
            density_conv: 1e-10,
            max_iter: 300,
            ..Default::default()
        }
    }

    /// Exactness anchor: a UHF reference is passed through untouched (borrowed,
    /// so every UHF consumer stays bit-identical).
    #[test]
    fn unrestricted_reference_borrows_a_uhf_result() {
        let mol = oh();
        let (prep, bounds, ctx) = setup(&mol);
        let uhf = crate::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &cfg()).unwrap();
        let view = unrestricted_reference(&mol, &uhf).unwrap();
        assert!(
            matches!(view, std::borrow::Cow::Borrowed(p) if std::ptr::eq(p, &uhf)),
            "a UHF reference must be returned as-is"
        );
    }

    /// With an ECP, the rebuild path's hcore must carry V_ECP as the SCF's does:
    /// the stored-Fock view and the rebuilt-Fock construction agree on HI+
    /// (doublet, def2-SVP ECP on I). A bare T + V_nuc hcore misses by the size
    /// of V_ECP's diagonal (Hartrees).
    #[test]
    fn rebuilt_fock_carries_the_ecp() {
        let bs = ferric_core::basis::bundled("def2-svp").unwrap();
        let mut mol = Molecule::parse_xyz("2\n\nH 0.0 0.0 0.0\nI 0.0 0.0 1.61\n", 1, 2).unwrap();
        mol.apply_ecp(&bs);
        assert!(
            mol.atoms[1].n_core_ecp > 0,
            "def2-SVP must carry an ECP for I"
        );
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let ctx = ParallelContext::default();
        let rohf = crate::rohf::solve_rohf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, &cfg())
            .unwrap();
        assert!(rohf.converged);
        let view = unrestricted_reference(&mol, &rohf).unwrap();
        let rebuilt = semicanonicalize(&ctx, &mol, &prep, &bounds, &rohf, 1e-12, None).unwrap();
        for (a, b) in [
            (view.eps_a(), rebuilt.eps_alpha.as_slice()),
            (view.eps_b(), rebuilt.eps_beta.as_slice()),
        ] {
            let d = a
                .iter()
                .zip(b)
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
            assert!(
                d < 1e-7,
                "stored-Fock vs rebuilt-Fock eps differ by {d:.2e}"
            );
        }
    }

    /// On a ROHF reference the view is semi-canonical: each spin's Fock is diagonal
    /// inside its occ–occ and virt–virt blocks with those diagonals as `eps_σ`, the
    /// per-spin densities (hence the reference determinant) are the ROHF ones, and
    /// the stored-Fock construction agrees with the independent J/K-rebuild
    /// construction of [`semicanonicalize`].
    #[test]
    fn rohf_view_is_semicanonical_and_matches_the_rebuilt_fock_construction() {
        let mol = oh();
        let (prep, bounds, ctx) = setup(&mol);
        let rohf = crate::rohf::solve_rohf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, &cfg())
            .unwrap();
        assert!(rohf.converged);
        let view = unrestricted_reference(&mol, &rohf).unwrap();
        let u: &ScfResult = &view;
        assert_eq!(u.spin, Spin::Unrestricted);
        let (nocc_a, nocc_b) = rohf_occupations(&mol).unwrap();
        let (f_a, f_b) = rohf.rohf_spin_focks.as_ref().unwrap();

        for (c, f, eps, nocc, d_ref, d_view) in [
            (
                u.mos_a(),
                f_a,
                u.eps_a(),
                nocc_a,
                &rohf.density_alpha,
                &u.density_alpha,
            ),
            (
                u.mos_b(),
                f_b,
                u.eps_b(),
                nocc_b,
                rohf.density_beta.as_ref().unwrap(),
                u.density_beta.as_ref().unwrap(),
            ),
        ] {
            let f_mo = c.t().dot(f).dot(c);
            let nmo = c.ncols();
            let mut worst_off = 0.0f64;
            let mut worst_diag = 0.0f64;
            for p in 0..nmo {
                worst_diag = worst_diag.max((f_mo[[p, p]] - eps[p]).abs());
                for q in 0..nmo {
                    let same_block = (p < nocc) == (q < nocc);
                    if same_block && p != q {
                        worst_off = worst_off.max(f_mo[[p, q]].abs());
                    }
                }
            }
            assert!(worst_off < 1e-10, "in-block off-diagonal F {worst_off:.2e}");
            assert!(worst_diag < 1e-10, "diag F vs eps {worst_diag:.2e}");
            let dd = (d_ref - d_view).iter().fold(0.0f64, |m, x| m.max(x.abs()));
            assert!(dd < 1e-10, "occupied span changed: max |dD| {dd:.2e}");
        }

        // Independent construction: J/K rebuilt from the final density.
        let rebuilt = semicanonicalize(&ctx, &mol, &prep, &bounds, &rohf, 1e-12, None).unwrap();
        for (a, b) in [
            (u.eps_a(), rebuilt.eps_alpha.as_slice()),
            (u.eps_b(), rebuilt.eps_beta.as_slice()),
        ] {
            let d = a
                .iter()
                .zip(b)
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
            assert!(
                d < 1e-7,
                "stored-Fock vs rebuilt-Fock eps differ by {d:.2e}"
            );
        }

        // Non-vacuity: the view's beta energies are NOT the ROHF effective-Fock
        // eigenvalues the old fallback used for both spins.
        let d = rohf
            .eps_a()
            .iter()
            .zip(u.eps_b())
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        assert!(d > 1e-3, "beta eps equal the effective-Fock ones ({d:.2e})");
    }

    /// A RestrictedOpen result without spin Focks is refused, not silently served
    /// the effective-Fock eigenvalues.
    #[test]
    fn rohf_without_spin_focks_is_refused() {
        let mol = oh();
        let (prep, bounds, ctx) = setup(&mol);
        let mut rohf =
            crate::rohf::solve_rohf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, &cfg())
                .unwrap();
        rohf.rohf_spin_focks = None;
        let err = unrestricted_reference(&mol, &rohf).unwrap_err();
        assert!(format!("{err}").contains("rohf_spin_focks"), "{err}");
    }
}
