//! The OP-MP3 reference determinant: from any converged SCF density, the
//! Hartree–Fock spin Fock matrices `F_σ = h + J[D] − K[D_σ]`, semicanonical
//! orbitals per spin, `E_HF[ρ]` and the non-Brillouin singles energy.
//!
//! This is the paper's reference path (Bystrom, Berkelbach, Hait,
//! arXiv:2609.25330, Computational methods; their `wmatrix_code` README §1):
//! the reference determinant (RHF, ROHF, RKS or ROKS, or UHF) keeps its
//! occupied space; each spin's HF Fock matrix is diagonalized separately in
//! the occupied and in the virtual space, the MP denominators use that
//! diagonal, and the occupied–virtual block enters only through
//!
//! ```text
//! E_NBS = Σ_σ Σ_{i active, a} |F^σ_ia|² / (ε_iσ − ε_aσ)
//! E_HF  = E_nuc + ½ Σ_σ Tr[D_σ (h + F_σ)]
//! ```
//!
//! A Kohn–Sham orbital energy never enters. For a HF reference at
//! convergence the rotation only resolves degeneracies and `E_NBS` vanishes
//! with the orbital gradient (UHF, RHF); ROHF keeps a non-zero `F_ia`.
//! Equations: `book/src/10-mp2/11-opmp3.md#opmp3`.

use crate::rimp2::active_occ;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::{ScfResult, Spin};
use ndarray::{s, Array2};
use ndarray_linalg::{Eigh, UPLO};

/// One spin of the semicanonical reference.
#[derive(Debug, Clone)]
pub struct SemicanonicalSpin {
    /// MO coefficients `(nbas, nmo)`: occupied block, then virtual block,
    /// each diagonalizing this spin's HF Fock matrix.
    pub mos: Array2<f64>,
    /// Diagonal of the HF Fock matrix in [`Self::mos`] (ascending per block).
    pub eps: Vec<f64>,
    /// Occupied orbitals (frozen core included).
    pub nocc: usize,
    /// Occupied–virtual block `F_ia` in [`Self::mos`], `(nocc, nvir)`.
    pub f_ov: Array2<f64>,
}

impl SemicanonicalSpin {
    /// `max |F_ia|` of this spin.
    pub fn max_ov(&self) -> f64 {
        self.f_ov.iter().fold(0.0f64, |m, x| m.max(x.abs()))
    }

    /// `Σ_{i ≥ frozen_core, a} F_ia² / (ε_i − ε_a)`.
    pub fn singles_energy(&self, frozen_core: usize) -> Result<f64, FerricError> {
        let first = self.nocc - active_occ(self.nocc, frozen_core)?;
        let mut e = 0.0;
        for i in first..self.nocc {
            for (a, &f) in self.f_ov.row(i).iter().enumerate() {
                e += f * f / (self.eps[i] - self.eps[self.nocc + a]);
            }
        }
        Ok(e)
    }
}

/// The OP-MP3 reference of a converged SCF result.
#[derive(Debug, Clone)]
pub struct Opmp3Reference {
    /// `[α, β]` semicanonical orbitals.
    pub spins: [SemicanonicalSpin; 2],
    /// HF energy of the reference density, `E_HF[ρ]` (Hartree).
    pub e_hf: f64,
    /// The SCF's own energy (the KS energy for a DFT reference).
    pub e_scf: f64,
}

impl Opmp3Reference {
    /// Non-Brillouin singles energy over both spins, active occupied only.
    pub fn e_nbs(&self, frozen_core: usize) -> Result<f64, FerricError> {
        Ok(self.spins[0].singles_energy(frozen_core)?
            + self.spins[1].singles_energy(frozen_core)?)
    }
}

/// `(nocc_α, nocc_β)` from the electron count and multiplicity.
fn occupations(mol: &Molecule) -> Result<(usize, usize), FerricError> {
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    if two_s < 0 || nelec < two_s || (nelec + two_s) % 2 != 0 {
        return Err(FerricError::General(format!(
            "opmp3 reference: {nelec} electrons cannot have multiplicity {}",
            mol.multiplicity
        )));
    }
    Ok((
        ((nelec + two_s) / 2) as usize,
        ((nelec - two_s) / 2) as usize,
    ))
}

/// The symmetrized `[r, r]` block of `f` over `range`, diagonalized.
fn diagonalize_block(
    f: &Array2<f64>,
    range: std::ops::Range<usize>,
) -> Result<(ndarray::Array1<f64>, Array2<f64>), FerricError> {
    let n = range.len();
    if n == 0 {
        return Ok((ndarray::Array1::zeros(0), Array2::zeros((0, 0))));
    }
    let b = f.slice(s![range.clone(), range]);
    let sym = (&b + &b.t()) * 0.5;
    sym.eigh(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("opmp3 semicanonical eigh failed: {e}")))
}

/// Semicanonicalize one spin: `c` its reference MOs, `f_ao` its HF Fock.
fn semicanonical_spin(
    c: &Array2<f64>,
    f_ao: &Array2<f64>,
    nocc: usize,
) -> Result<SemicanonicalSpin, FerricError> {
    let nmo = c.ncols();
    if nocc > nmo {
        return Err(FerricError::General(format!(
            "opmp3 reference: {nocc} occupied orbitals but {nmo} MOs"
        )));
    }
    let f_mo = c.t().dot(f_ao).dot(c);
    let (e_o, u_o) = diagonalize_block(&f_mo, 0..nocc)?;
    let (e_v, u_v) = diagonalize_block(&f_mo, nocc..nmo)?;
    let mut mos = Array2::zeros(c.raw_dim());
    mos.slice_mut(s![.., ..nocc])
        .assign(&c.slice(s![.., ..nocc]).dot(&u_o));
    mos.slice_mut(s![.., nocc..])
        .assign(&c.slice(s![.., nocc..]).dot(&u_v));
    let f_ov = mos
        .slice(s![.., ..nocc])
        .t()
        .dot(f_ao)
        .dot(&mos.slice(s![.., nocc..]));
    let eps = e_o.iter().chain(e_v.iter()).copied().collect();
    Ok(SemicanonicalSpin {
        mos,
        eps,
        nocc,
        f_ov,
    })
}

/// `(J[D_total], K[D_α], K[D_β])`; one build when `D_α = D_β`.
fn coulomb_and_exchange(
    ctx: &ParallelContext,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    thresh: f64,
    d: [&Array2<f64>; 2],
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), FerricError> {
    let n = prep.nbasis();
    let build = |dm: &Array2<f64>| -> Result<(Array2<f64>, Array2<f64>), FerricError> {
        let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
        ferric_scf::rhf::build_jk(ctx, prep, bounds, thresh, dm, &mut j, &mut k)?;
        Ok((j, k))
    };
    let (j_a, k_a) = build(d[0])?;
    if d[0] == d[1] {
        return Ok((&j_a * 2.0, k_a.clone(), k_a));
    }
    let (j_b, k_b) = build(d[1])?;
    Ok((j_a + j_b, k_a, k_b))
}

/// Per-spin reference MOs and densities `[α, β]` of `scf`.
fn spin_mos_and_densities(scf: &ScfResult) -> ([&Array2<f64>; 2], [&Array2<f64>; 2]) {
    let mo_b = match scf.spin {
        Spin::Unrestricted => scf.mos_b(),
        Spin::Restricted | Spin::RestrictedOpen => &scf.mos_alpha,
    };
    let d_b = scf.density_beta.as_ref().unwrap_or(&scf.density_alpha);
    ([&scf.mos_alpha, mo_b], [&scf.density_alpha, d_b])
}

/// Build the OP-MP3 reference of a converged `scf` on `mol`.
///
/// `bounds` gives the operator (Coulomb) and Schwarz screening of the
/// exact four-centre J/K build, `thresh` its screening threshold. The core
/// Hamiltonian is the SCF's (`T + V + V_ECP`, or X2C-1e for a relativistic
/// reference); an external potential is not supported here.
///
/// # Errors
///
/// An unconverged reference, or one whose spin densities are inconsistent
/// with the molecule's multiplicity.
pub fn opmp3_reference(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    thresh: f64,
    scf: &ScfResult,
) -> Result<Opmp3Reference, FerricError> {
    if !scf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: scf.iterations,
            last_energy: scf.energy,
        });
    }
    let (na, nb) = occupations(mol)?;
    let (mos, d) = spin_mos_and_densities(scf);
    let h = ferric_scf::x2c::reference_hcore(prep, mol, prep.basis_set(), None, scf.relativistic)?;
    let (j, k_a, k_b) = coulomb_and_exchange(ctx, prep, bounds, thresh, d)?;
    let f_a = &h + &j - &k_a;
    let f_b = &h + &j - &k_b;
    let tr = |x: &Array2<f64>, y: &Array2<f64>| (x * y).sum();
    let e_hf = mol.nuclear_repulsion() + 0.5 * (tr(d[0], &(&h + &f_a)) + tr(d[1], &(&h + &f_b)));
    Ok(Opmp3Reference {
        spins: [
            semicanonical_spin(mos[0], &f_a, na)?,
            semicanonical_spin(mos[1], &f_b, nb)?,
        ],
        e_hf,
        e_scf: scf.energy,
    })
}
