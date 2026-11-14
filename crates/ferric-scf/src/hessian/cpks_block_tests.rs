//! Exactness anchors of the Kohn-Sham CPKS: the XC kernel in the MO form the
//! solver iterates in ([`crate::ks_fixed_grid::XcMoResponse`] and its
//! polarized twin) against the AO route it replaced (the oracle), and the
//! lockstep blocks (the MO kernel and the fitted Coulomb matrix
//! [`crate::df_j::DfJ::build_many`] of every coordinate of a block in one
//! pass) against one-coordinate blocks.
//!
//! Bars:
//! - kernel products: the MO and AO routes are the same sums regrouped, so
//!   they differ by rounding; each element is a point sum of length `npts`
//!   of terms that are themselves sums over the basis, so the bar is
//!   `(npts + nbf)·ε` times the largest element (`γ_n` of the longest sum,
//!   relative to the result scale);
//! - response terms: the sum of the two compared terms' consistency probes
//!   `max |R − Rᵀ|`, which measure the CPKS solve error at the production
//!   tolerance (as for the Hartree-Fock anchors
//!   `rhf_/uhf_cphf_block_matches_one_coordinate_blocks`): the two solves run
//!   the same PCG recurrence on operators equal to rounding.
//!
//! The Hessian's kernel covers closed-shell LDA and GGA, pure or global
//! hybrid (`FixedGridKs` refuses meta-GGAs), so there is no τ term to anchor.

use super::*;
use crate::ks_fixed_grid::{FixedGridKs, MoRows, XcGeometry, XcMutants};
use ferric_dft::grid::AtomicGridConfig;

const AUX: &str = "def2-universal-jkfit";

fn grid_cfg() -> AtomicGridConfig {
    AtomicGridConfig {
        n_radial: 40,
        n_angular: 110,
        prune: None,
        weight_cut: None,
    }
}

/// Distorted water in 6-31G* (no symmetry zeros in the response).
fn distorted_water(charge: i32, mult: usize) -> (Molecule, PreparedBasis, SchwarzBounds) {
    let xyz = "3\nwater\nO 0.02 -0.01 0.1173\nH 0.0 0.80 -0.45\nH 0.05 -0.74 -0.50\n";
    let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    let prep = PreparedBasis::new(&mol, &ferric_core::basis::bundled("6-31g*").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    (mol, prep, bounds)
}

/// The production SCF of `xc` with RI-J over [`AUX`] in f64, tight enough
/// for the CPKS.
fn scf_config(xc: &str) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.into()),
        df_j_aux: Some(AUX.into()),
        df_k_aux: Some(String::new()),
        ri_j_precision: Some(crate::df_j::RiJPrecision::F64),
        dft_grid: Some(grid_cfg()),
        energy_conv: 1e-12,
        density_conv: 1e-10,
        ..Default::default()
    }
}

/// The SCF's RI-J for the CPKS `G`, on a tier the lockstep pass reads (so
/// the anchors exercise it).
fn fitted_j(mol: &Molecule, prep: &PreparedBasis) -> crate::df_j::DfJ<'static> {
    let aux = PreparedBasis::new(mol, &ferric_core::basis::bundled(AUX).unwrap()).unwrap();
    let df = crate::df_j::DfJ::new_with_precision(
        Operator::coulomb(),
        prep,
        &aux,
        1 << 30,
        crate::df_j::RiJPrecision::F64,
    )
    .unwrap();
    assert!(
        df.lockstep_supported(),
        "the anchor must reach the lockstep RI-J pass"
    );
    df
}

fn asymmetry(r: &Array2<f64>) -> f64 {
    max_abs(&(r - &r.t()))
}

/// Assert two response terms agree within the sum of their asymmetries.
fn assert_responses_agree(label: &str, a: &Array2<f64>, b: &Array2<f64>) {
    let diff = max_abs(&(a - b));
    let bar = asymmetry(a) + asymmetry(b);
    eprintln!("{label}: {diff:.3e} (bar {bar:.3e})");
    assert!(diff <= bar, "{label}: differ by {diff:.3e} > {bar:.3e}");
}

/// A deterministic, unstructured `(rows, cols)` matrix.
fn probe(rows: usize, cols: usize, seed: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        ((i * 37 + j * 11) as f64 * seed).sin()
    })
}

/// Assert an MO product equals the projected AO product within the
/// contraction-length bar (module doc).
fn assert_products_agree(label: &str, mo: &Array2<f64>, ao: &Array2<f64>, len: usize) {
    let diff = max_abs(&(mo - ao));
    let bar = len as f64 * f64::EPSILON * max_abs(ao);
    eprintln!(
        "{label}: MO vs projected AO {diff:.3e} (bar {bar:.3e}, scale {:.3e})",
        max_abs(ao)
    );
    assert!(
        diff <= bar,
        "{label}: MO vs projected AO {diff:.3e} > {bar:.3e}"
    );
}

/// A converged closed-shell KS reference with its CPKS inputs.
struct RksCase {
    ctx: ParallelContext,
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    rks: ScfResult,
    grid: FixedGridKs,
    nocc: usize,
}

impl RksCase {
    fn new(xc: &str) -> Self {
        let ctx = ParallelContext::default();
        let (mol, prep, bounds) = distorted_water(0, 1);
        let rks = crate::rhf::solve_rhf(
            &ctx,
            &mol,
            &prep,
            Operator::coulomb(),
            &bounds,
            &scf_config(xc),
        )
        .unwrap();
        assert!(rks.converged);
        let grid = FixedGridKs::new(&mol, xc, &grid_cfg()).unwrap();
        let nocc = (mol.nelec() / 2) as usize;
        RksCase {
            ctx,
            mol,
            prep,
            bounds,
            rks,
            grid,
            nocc,
        }
    }

    fn frame(&self) -> MoFrame<'_> {
        MoFrame::new(self.rks.mos_r(), self.rks.eps_r(), self.nocc, 2.0)
    }

    /// `(h1, s1)` in the MO basis (pure-functional Fock derivative; any
    /// right-hand side serves the anchors).
    fn first_order(&self) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let d = self.rks.density_r();
        let spec = FockSpec {
            coulomb: d,
            exchange: d,
            k_scale: 0.0,
        };
        let natoms = self.mol.atoms.len();
        let first = first_order_ao(
            &self.prep,
            Operator::coulomb(),
            &self.bounds,
            natoms,
            &[],
            &[spec],
        )
        .unwrap();
        let frame = self.frame();
        (frame.to_mo_all(&first.f1[0]), frame.to_mo_all(&first.s1))
    }

    fn geo(&self) -> XcGeometry<'_> {
        XcGeometry::new(&self.mol, &self.prep).unwrap()
    }

    fn mo_kernel(&self) -> crate::ks_fixed_grid::XcMoResponse {
        let mos = (self.rks.mos_r(), self.nocc);
        self.grid
            .response_mo(&self.geo(), self.rks.density_r(), mos, 1.0, 1.0)
            .unwrap()
    }

    fn ao_kernel(&self) -> crate::ks_fixed_grid::XcResponse {
        self.grid
            .response(&self.geo(), self.rks.density_r(), 1.0, 1.0)
            .unwrap()
    }

    /// The CPKS response term with `g`, in lockstep blocks of `block`.
    fn response(&self, g: &TwoElectronBuilder<'_>, block: usize) -> Array2<f64> {
        let frame = self.frame();
        let (h1, s1) = self.first_order();
        let sols = solve_cphf_blocks_of(g, &frame, &h1, &s1, block).unwrap();
        assemble_response(&h1, &s1, &sols, &frame)
    }

    fn builder(&self) -> TwoElectronBuilder<'_> {
        TwoElectronBuilder::new(&self.ctx, &self.prep, &self.bounds)
            .unwrap()
            .fitted(Some(fitted_j(&self.mol, &self.prep)), None, false)
    }
}

/// The closed-shell MO product (rotations of the occupied or the virtual MOs,
/// results on all MOs) equals `Cᵀ V1[D1] C_o` of the AO route at
/// `D1 = 2 (C_r U C_oᵀ + C_o Uᵀ C_rᵀ)`, for an LDA and a GGA, three
/// rotations at once.
#[test]
fn mo_product_equals_the_projected_ao_product() {
    for xc in ["LDA", "PBE"] {
        let case = RksCase::new(xc);
        let (mo, ao) = (case.mo_kernel(), case.ao_kernel());
        let frame = case.frame();
        let len = case.grid.npts() + case.prep.nbasis();
        for (input, c_r) in [
            (MoRows::Virtual, frame.c_vir),
            (MoRows::Occupied, frame.c_occ),
        ] {
            let us: Vec<Array2<f64>> = (1..=3)
                .map(|k| probe(c_r.ncols(), case.nocc, 0.3 * k as f64))
                .collect();
            let refs: Vec<&Array2<f64>> = us.iter().collect();
            let got = mo.apply_many(&refs, input, MoRows::All);
            let dens: Vec<Array2<f64>> = us
                .iter()
                .map(|u| {
                    let half = c_r.dot(u).dot(&frame.c_occ.t());
                    2.0 * (&half + &half.t())
                })
                .collect();
            let want = ao.apply_many(&dens.iter().collect::<Vec<_>>());
            for (k, (g, v)) in got.iter().zip(&want).enumerate() {
                let projected = frame.c.t().dot(v).dot(&frame.c_occ);
                assert_products_agree(&format!("{xc} {input:?} #{k}"), g, &projected, len);
            }
        }
    }
}

/// Closed-shell PBE with RI-J: the CPKS response through the MO kernel equals
/// the AO-route (oracle) response.
#[test]
fn rks_cpks_mo_route_matches_the_ao_route() {
    let case = RksCase::new("PBE");
    let (mo, ao) = (case.mo_kernel(), case.ao_kernel());
    let ao_fn = |d1s: &[&Array2<f64>]| ao.apply_many(d1s);
    let g_mo = case.builder().with_xc_mo_response(&mo, 0.0);
    let g_ao = case.builder().with_xc_response(Some(&ao_fn), 0.0);
    let block = case.mol.atoms.len() * 3;
    let (r_mo, r_ao) = (case.response(&g_mo, block), case.response(&g_ao, block));
    assert_responses_agree("RKS CPKS MO vs AO route", &r_mo, &r_ao);
}

/// Closed-shell PBE with RI-J: the MO kernel and fitted J of a block of all 9
/// coordinates equal one-coordinate blocks.
#[test]
fn rks_cpks_block_matches_one_coordinate_blocks() {
    let case = RksCase::new("PBE");
    let mo = case.mo_kernel();
    let g = case.builder().with_xc_mo_response(&mo, 0.0);
    let ncoord = case.mol.atoms.len() * 3;
    let (all, one) = (case.response(&g, ncoord), case.response(&g, 1));
    assert_responses_agree("RKS CPKS block vs one-coordinate blocks", &all, &one);
}

/// A converged UKS reference (H2O+ doublet, PBE, RI-J) with its CPKS inputs.
struct UksCase {
    ctx: ParallelContext,
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    uks: ScfResult,
    grid: FixedGridKs,
    nocc: [usize; 2],
}

impl UksCase {
    fn new() -> Self {
        let ctx = ParallelContext::default();
        let (mol, prep, bounds) = distorted_water(1, 2);
        let uks = crate::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &scf_config("PBE")).unwrap();
        assert!(uks.converged);
        let grid = FixedGridKs::new(&mol, "PBE", &grid_cfg()).unwrap();
        let nocc = uhf_occupations(&mol);
        UksCase {
            ctx,
            mol,
            prep,
            bounds,
            uks,
            grid,
            nocc,
        }
    }

    fn frames(&self) -> [MoFrame<'_>; 2] {
        [
            MoFrame::new(self.uks.mos_a(), self.uks.eps_a(), self.nocc[0], 1.0),
            MoFrame::new(self.uks.mos_b(), self.uks.eps_b(), self.nocc[1], 1.0),
        ]
    }

    fn kernels(
        &self,
    ) -> (
        crate::ks_fixed_grid::PolarizedXcMoResponse,
        crate::ks_fixed_grid::PolarizedXcResponse,
    ) {
        let (da, db) = uhf_spin_densities(&self.uks).unwrap();
        let geo = XcGeometry::new(&self.mol, &self.prep).unwrap();
        let pol = self.grid.polarized().unwrap();
        let m = XcMutants::default();
        let mos = ([self.uks.mos_a(), self.uks.mos_b()], self.nocc);
        (
            pol.response_mo(&geo, [da, db], mos, 1.0, &m).unwrap(),
            pol.response(&geo, [da, db], 1.0, &m).unwrap(),
        )
    }

    fn response(&self, g: &TwoElectronBuilder<'_>, block: usize) -> Array2<f64> {
        let (da, db) = uhf_spin_densities(&self.uks).unwrap();
        let d = da + db;
        let specs = [da, db].map(|ds| FockSpec {
            coulomb: &d,
            exchange: ds,
            k_scale: 0.0,
        });
        let natoms = self.mol.atoms.len();
        let first = first_order_ao(
            &self.prep,
            Operator::coulomb(),
            &self.bounds,
            natoms,
            &[],
            &specs,
        )
        .unwrap();
        let frames = self.frames();
        let h1 = [0, 1].map(|sp| frames[sp].to_mo_all(&first.f1[sp]));
        let s1 = [0, 1].map(|sp| frames[sp].to_mo_all(&first.s1));
        let sols = solve_ucphf_blocks_of(g, &frames, &h1, &s1, block).unwrap();
        assemble_response(&h1[0], &s1[0], &sols[0], &frames[0])
            + assemble_response(&h1[1], &s1[1], &sols[1], &frames[1])
    }

    fn builder(&self) -> TwoElectronBuilder<'_> {
        TwoElectronBuilder::new(&self.ctx, &self.prep, &self.bounds)
            .unwrap()
            .fitted(Some(fitted_j(&self.mol, &self.prep)), None, false)
    }
}

/// The polarized MO product equals each spin's `C_sᵀ V1^s C_o^s` of the AO
/// route, two rotation pairs at once.
#[test]
fn polarized_mo_product_equals_the_projected_ao_product() {
    let case = UksCase::new();
    let (mo, ao) = case.kernels();
    let frames = case.frames();
    let len = case.grid.npts() + case.prep.nbasis();
    let us: Vec<[Array2<f64>; 2]> = (1..=2)
        .map(|k| {
            [0, 1].map(|sp| {
                let nvir = frames[sp].c_vir.ncols();
                probe(nvir, case.nocc[sp], 0.2 * k as f64 + 0.1 * sp as f64)
            })
        })
        .collect();
    let got = mo.apply_many(&pair_refs(&us), MoRows::Virtual, MoRows::All);
    let dens: Vec<[Array2<f64>; 2]> = us
        .iter()
        .map(|u| [0, 1].map(|sp| frames[sp].response_density(&u[sp])))
        .collect();
    let want = ao.apply_many(&pair_refs(&dens));
    for (k, (g, v)) in got.iter().zip(&want).enumerate() {
        for sp in 0..2 {
            let projected = frames[sp].c.t().dot(&v[sp]).dot(&frames[sp].c_occ);
            assert_products_agree(&format!("UKS pair {k} spin {sp}"), &g[sp], &projected, len);
        }
    }
}

/// The UKS CPKS response through the MO kernel equals the AO-route (oracle)
/// response.
#[test]
fn uks_cpks_mo_route_matches_the_ao_route() {
    let case = UksCase::new();
    let (mo, ao) = case.kernels();
    let ao_fn = |d1s: &[[&Array2<f64>; 2]]| ao.apply_many(d1s);
    let g_mo = case.builder().with_uks_xc_mo_response(&mo, 0.0);
    let g_ao = case.builder().with_uks_xc_response(&ao_fn, 0.0);
    let block = case.mol.atoms.len() * 3;
    let (r_mo, r_ao) = (case.response(&g_mo, block), case.response(&g_ao, block));
    assert_responses_agree("UKS CPKS MO vs AO route", &r_mo, &r_ao);
}

/// [`rks_cpks_block_matches_one_coordinate_blocks`] for the spin-coupled UKS
/// CPKS: the lockstep polarized MO kernel takes every coordinate's rotation
/// pair at once.
#[test]
fn uks_cpks_block_matches_one_coordinate_blocks() {
    let case = UksCase::new();
    let (mo, _) = case.kernels();
    let g = case.builder().with_uks_xc_mo_response(&mo, 0.0);
    let ncoord = case.mol.atoms.len() * 3;
    let (all, one) = (case.response(&g, ncoord), case.response(&g, 1));
    assert_responses_agree("UKS CPKS block vs one-coordinate blocks", &all, &one);
}
