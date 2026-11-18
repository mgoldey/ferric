//! The f_xc kernel of an UNRESTRICTED (UKS) reference in the two-spin
//! particle–hole space: the `(ia σ| f_στ |jb τ)` blocks spin-conserving
//! TDA-DFT and Casida TDDFT add to A (and B) on a UKS reference
//! (`book/src/20-tddft/06-unrestricted.md#unrestricted-spin-problem`).
//!
//! The space is the concatenation of the α pairs (`ia = i·nvir_α + a`) and
//! the β pairs (offset by `nocc_α·nvir_α`). For a block of vectors `x`, the
//! product is the AO route of [`fxc_ov_block`](super::fxc_ov_block) taken
//! per spin: `δD_σ = ½(M_σ + M_σᵀ)` with `M_σ = C^σ_occ X_σ C^σᵀ_vir`, the
//! spin-resolved AO kernel gives `(δV_α, δV_β)`, and
//! `(K x)_σ = C^σᵀ_occ δV_σ C^σ_vir`. Feeding a unit vector of spin τ gives
//! column `jb τ` of both spin blocks: `δV_σ = ∫ f_στ φ_j φ_b χχ`.
//!
//! The kernel is built at the UKS reference `(D_α, D_β)` with the per-spin
//! density screen the UKS V_xc applies (each spin block on its own `ρ_σ`):
//! it is the derivative of the V_xc the UKS SCF converged. At a restricted
//! point (D_α = D_β) it differs from [`SingletKernel`](super::SingletKernel),
//! which screens on `ρ_α + ρ_β` as the RKS V_xc does, only at points where
//! `ρ_σ` sits at the density floor.

use crate::fxc::{GgaFxcKernel, MggaFxcKernel, MggaFxcReference};
use crate::grid::AtomicGridConfig;
use crate::libxc::{FunctionalFamily, XcDef};
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ndarray::{s, Array2};

/// The spin-resolved f_xc kernel at an unrestricted reference `(D_α, D_β)`:
/// GGA-family (LDA / GGA / global or range-separated hybrid) or meta-GGA
/// (τ-only, including hybrid meta-GGAs).
#[derive(Debug)]
pub enum UksKernel {
    /// No meta-GGA component.
    Gga {
        /// The kernel and its grid / AO data.
        kernel: GgaFxcKernel,
        /// Reference α/β densities.
        reference: Box<crate::density_on_grid::UksDensityGrid>,
    },
    /// At least one meta-GGA component.
    Mgga {
        /// The kernel and its grid / AO data.
        kernel: MggaFxcKernel,
        /// Reference per-point Hessians, gradients and per-spin screen.
        reference: Box<MggaFxcReference>,
    },
}

impl UksKernel {
    /// Build the kernel for `def` (resolved with `nspin = 2`, e.g. by
    /// [`resolve_response_xc`](super::resolve_response_xc)) on `grid`, the
    /// reference SCF's grid, at `(d_a, d_b)`.
    ///
    /// # Errors
    ///
    /// Kernel construction (grid, AO-on-grid budget) or a libxc meta-GGA
    /// second-derivative failure.
    pub fn new(
        mol: &Molecule,
        bs: &BasisSet,
        def: XcDef,
        grid: &AtomicGridConfig,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> Result<Self, String> {
        if def
            .funcs
            .iter()
            .any(|f| f.family() == FunctionalFamily::MetaGga)
        {
            let kernel = MggaFxcKernel::new(mol, bs, def, grid)?;
            let reference = Box::new(kernel.reference(d_a, d_b)?);
            Ok(Self::Mgga { kernel, reference })
        } else {
            let kernel = GgaFxcKernel::new(mol, bs, def, grid)?;
            let reference = Box::new(kernel.reference_density(d_a, d_b));
            Ok(Self::Gga { kernel, reference })
        }
    }

    /// `(δV_α, δV_β)` for the perturbation `(δD_α, δD_β)`.
    pub fn apply(&self, dd_a: &Array2<f64>, dd_b: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        match self {
            Self::Gga { kernel, reference } => kernel.apply_with_ref(reference, dd_a, dd_b),
            Self::Mgga { kernel, reference } => kernel.apply_with_ref(reference, dd_a, dd_b),
        }
    }

    /// Grid points the kernel integrates over.
    pub fn npts(&self) -> usize {
        match self {
            Self::Gga { kernel, .. } => kernel.grid.len(),
            Self::Mgga { kernel, .. } => kernel.grid.len(),
        }
    }
}

/// Occupied and virtual MO coefficients of one spin, `(nbf, nocc)` and
/// `(nbf, nvir)`.
#[derive(Debug, Clone)]
pub struct SpinOrbitals {
    /// Occupied MOs of this spin.
    pub c_occ: Array2<f64>,
    /// Virtual MOs of this spin.
    pub c_vir: Array2<f64>,
}

impl SpinOrbitals {
    fn dim(&self) -> usize {
        self.c_occ.ncols() * self.c_vir.ncols()
    }
}

/// Matrix-free `y = K x` over the two-spin `(ia σ)` space of a UKS
/// reference (module docs). Rows of `x` and `y`: the α pairs, then the β
/// pairs.
#[derive(Debug)]
pub struct UksOvFxc<'k> {
    kernel: &'k UksKernel,
    spins: [SpinOrbitals; 2],
}

impl<'k> UksOvFxc<'k> {
    /// The operator of `kernel` over the pairs of `spins = [α, β]`.
    pub fn new(kernel: &'k UksKernel, spins: [SpinOrbitals; 2]) -> Self {
        Self { kernel, spins }
    }

    /// `nocc_α·nvir_α + nocc_β·nvir_β`.
    pub fn dim(&self) -> usize {
        self.spins[0].dim() + self.spins[1].dim()
    }

    /// `K x` for every column of `x` (`(dim, nvec)`): one AO kernel
    /// application per column.
    pub fn apply(&self, x: &Array2<f64>) -> Array2<f64> {
        assert_eq!(x.nrows(), self.dim(), "UksOvFxc: x has the wrong length");
        let d_a = self.spins[0].dim();
        let mut y = Array2::<f64>::zeros(x.raw_dim());
        for n in 0..x.ncols() {
            let col = x.column(n);
            let dd_a = transition_density(&self.spins[0], col.slice(s![..d_a]));
            let dd_b = transition_density(&self.spins[1], col.slice(s![d_a..]));
            let (dv_a, dv_b) = self.kernel.apply(&dd_a, &dd_b);
            let mut out = y.column_mut(n);
            project(&self.spins[0], &dv_a, out.slice_mut(s![..d_a]));
            project(&self.spins[1], &dv_b, out.slice_mut(s![d_a..]));
        }
        y
    }
}

/// `δD = ½(M + Mᵀ)`, `M = C_occ X C_virᵀ` with `X[i, a] = x[i·nvir + a]`:
/// symmetrized because the GGA terms contract ∇δD (see `fxc_ov_block`).
fn transition_density(o: &SpinOrbitals, x: ndarray::ArrayView1<f64>) -> Array2<f64> {
    let (nocc, nvir) = (o.c_occ.ncols(), o.c_vir.ncols());
    let xm = Array2::from_shape_fn((nocc, nvir), |(i, a)| x[i * nvir + a]);
    let m = o.c_occ.dot(&xm).dot(&o.c_vir.t());
    0.5 * (&m + &m.t())
}

/// `out[i·nvir + a] = (C_occᵀ δV C_vir)[i, a]`.
fn project(o: &SpinOrbitals, dv: &Array2<f64>, mut out: ndarray::ArrayViewMut1<f64>) {
    let blk = o.c_occ.t().dot(dv).dot(&o.c_vir);
    let nvir = o.c_vir.ncols();
    for ((i, a), v) in blk.indexed_iter() {
        out[i * nvir + a] = *v;
    }
}

#[cfg(test)]
mod tests {
    use super::super::{fxc_ov_block, resolve_response_xc, ResponseSpin, SingletKernel};
    use super::*;
    use ferric_core::basis;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;

    /// At a restricted point the UKS operator's spin blocks reproduce the
    /// restricted singlet and triplet blocks: `K_αα + K_αβ = K_S` and
    /// `K_αα − K_αβ = K_T` (`fxc_ov_block`, which feeds δD_β = ±δD_α to the
    /// restricted-screened kernel), water/6-31G at an HF density, LDA, PBE,
    /// B3LYP and r2SCAN; and the operator is symmetric with `K_ββ = K_αα`.
    ///
    /// Bar: `2·FXC_BLOCK_ASYMMETRY_TOL·max|K|` plus the screening
    /// difference, which is zero here: no grid point of water has `ρ_σ` at
    /// the density floor while `ρ_α + ρ_β` is above it in a way that moves a
    /// block element (asserted by the measured difference sitting under the
    /// rounding bar).
    ///
    /// Mutation (applied and reverted, 2026-11-14): the α-β kernel block
    /// dropped in `UksOvFxc::apply` (δV_α from (δD_α, 0), δV_β from
    /// (0, δD_β)) fails the visibility assertion ("LDA: cross-spin block
    /// invisible"); the same mutation fails the unrestricted anchor
    /// (`ferric-tddft`) and the PySCF comparison (NH2/PBE).
    #[test]
    fn spin_blocks_at_a_restricted_point_are_the_singlet_and_triplet_blocks() {
        let _env_guard = crate::TEST_BUDGET_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let xyz = format!(
            "{}/../../testdata/molecules/water.xyz",
            env!("CARGO_MANIFEST_DIR")
        );
        let mol = Molecule::load_xyz(&xyz).unwrap();
        let bs = basis::bundled("6-31g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = ferric_scf::screening::SchwarzBounds::compute(op, &prep).unwrap();
        let scf = ferric_scf::rhf::solve_rhf(
            &ParallelContext::default(),
            &mol,
            &prep,
            op,
            &bounds,
            &ferric_scf::rhf::RhfConfig::default(),
        )
        .unwrap();
        let nocc = (mol.nelec() / 2) as usize;
        let c = scf.mos_r();
        let nvir = scf.eps_r().len() - nocc;
        let n = nocc * nvir;
        let spin = SpinOrbitals {
            c_occ: c.slice(s![.., ..nocc]).to_owned(),
            c_vir: c.slice(s![.., nocc..nocc + nvir]).to_owned(),
        };
        let d = &scf.density_alpha;
        let grid = AtomicGridConfig::default();
        for name in ["LDA", "PBE", "B3LYP", "r2SCAN"] {
            let restricted = SingletKernel::new(
                &mol,
                &bs,
                resolve_response_xc(name, "t").unwrap().xc_kernel,
                &grid,
                d,
            )
            .unwrap();
            let (ks, _) = fxc_ov_block(&restricted, ResponseSpin::Singlet, c, 0, nocc, nocc, nvir);
            let (kt, _) = fxc_ov_block(&restricted, ResponseSpin::Triplet, c, 0, nocc, nocc, nvir);
            let uks = UksKernel::new(
                &mol,
                &bs,
                resolve_response_xc(name, "t").unwrap().xc_kernel,
                &grid,
                d,
                d,
            )
            .unwrap();
            let op = UksOvFxc::new(&uks, [spin.clone(), spin.clone()]);
            let k = op.apply(&Array2::eye(2 * n));
            let aa = k.slice(s![..n, ..n]);
            let ab = k.slice(s![..n, n..]);
            let ba = k.slice(s![n.., ..n]);
            let bb = k.slice(s![n.., n..]);
            let maxabs =
                |m: ndarray::ArrayView2<f64>| m.iter().fold(0.0_f64, |x, &y| x.max(y.abs()));
            let scale = maxabs(ks.view());
            let bar = 2.0 * super::super::FXC_BLOCK_ASYMMETRY_TOL * scale;
            let ds = maxabs((&aa + &ab - &ks).view());
            let dt = maxabs((&aa - &ab - &kt).view());
            let dsym = maxabs((&ab - &ba.t()).view()).max(maxabs((&aa - &aa.t()).view()));
            let dbb = maxabs((&bb - &aa).view());
            eprintln!(
                "{name}: |K_aa+K_ab-K_S| {ds:.2e}, |K_aa-K_ab-K_T| {dt:.2e}, asym {dsym:.2e}, \
                 |K_bb-K_aa| {dbb:.2e} (bar {bar:.2e}, |K_ab| {:.2e})",
                maxabs(ab)
            );
            assert!(maxabs(ab) > 1e3 * bar, "{name}: cross-spin block invisible");
            for (what, v) in [
                ("singlet", ds),
                ("triplet", dt),
                ("symmetry", dsym),
                ("bb", dbb),
            ] {
                assert!(v <= bar, "{name}: {what} differs by {v:e} (bar {bar:e})");
            }
        }
    }
}
