//! CPU tests of the cuEST SCF route (`ferric_scf::cuest_route`), with no GPU.
//!
//! A stand-in backend serves libint's S, T, V and ferric's exact DirectJ /
//! DirectK, but permuted into a FOREIGN AO order (every l>=1 shell reversed,
//! alternating signs). The route must calibrate that order away and reproduce
//! ferric's own exact-J/K SCF; this is the exactness anchor for everything the
//! route does besides the GPU arithmetic (calibration, density and J/K
//! reordering, Fock assembly through the injected path, guess, XC).
//!
//! Mutations (each must be caught):
//! - J/K returned in an order other than the one-electron one: calibration
//!   cannot see it, the energy comparison must.
//! - a 5% normalization error on a d shell: calibration must refuse to run.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_cuest::layout::{AoLayout, AoMap};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron as oe;
use ferric_integrals::operator::Operator;
use ferric_scf::cuest_route::{solve_rhf_backend, solve_uhf_backend, AoBackend};
use ferric_scf::direct_j::DirectJ;
use ferric_scf::direct_k::DirectK;
use ferric_scf::fock::{JBuilder, KBuilder};
use ferric_scf::rhf::{resolve_three_index_budget, solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

/// Converged-energy agreement bar. Both runs are the same exact-J/K SCF to a
/// 1e-9 density threshold (energy error ~ its square); only the AO order and
/// the guess path differ.
const BAR_E: f64 = 1e-9;

struct Setup {
    ctx: ParallelContext,
    mol: Molecule,
    bs: basis::BasisSet,
    prep: PreparedBasis,
    op: Operator,
    bounds: SchwarzBounds,
}

fn setup(basis_name: &str, charge: i32, mult: usize) -> Setup {
    let mol = Molecule::parse_xyz(WATER, charge, mult).expect("water");
    let bs = basis::bundled(basis_name).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    Setup {
        ctx: ParallelContext::default(),
        mol,
        bs,
        prep,
        op,
        bounds,
    }
}

/// Every l>=1 shell reversed with alternating signs; s shells unchanged.
fn foreign(layout: &AoLayout) -> AoMap {
    AoMap::from_components(layout, |sh| {
        let n = sh.nfunc();
        (0..n)
            .map(|k| (n - 1 - k, if k % 2 == 1 { -1.0 } else { 1.0 }))
            .collect()
    })
}

/// libint one-electron + ferric DirectJ/DirectK, presented in the foreign
/// order (J/K in `jk_map`'s, which differs only in the mutation test).
struct FakeBackend<'a> {
    layout: AoLayout,
    jk_map: AoMap,
    prep: &'a PreparedBasis,
    j: DirectJ<'a>,
    k: DirectK<'a>,
    /// Multiply one d shell's rows/columns of S, T, V by this (mutation).
    d_scale: f64,
}

impl<'a> FakeBackend<'a> {
    fn new(su: &'a Setup, map_of: impl Fn(&AoLayout) -> AoMap) -> Self {
        let layout = AoLayout::new(&su.mol, &su.bs).expect("layout");
        let jk_map = map_of(&layout);
        let budget = resolve_three_index_budget(0);
        let thresh = RhfConfig::default().integral_thresh;
        Self {
            j: DirectJ::new(&su.ctx, &su.prep, &su.bounds, thresh, budget),
            k: DirectK::new(&su.ctx, &su.prep, &su.bounds, thresh, budget),
            layout,
            jk_map,
            prep: &su.prep,
            d_scale: 1.0,
        }
    }
}

impl FakeBackend<'_> {
    /// libint S, T, V of `prep` in the foreign order (and, for the mutation,
    /// with the FIRST d shell of each matrix scaled).
    fn present(&self, prep: &PreparedBasis, layout: &AoLayout) -> [Array2<f64>; 3] {
        let map = foreign(layout);
        let mut m =
            [oe::overlap(prep), oe::kinetic(prep), oe::nuclear(prep)].map(|x| map.to_cuest(&x));
        if self.d_scale != 1.0 {
            for sh in layout.shells.iter().filter(|s| s.l == 2) {
                for x in &mut m {
                    for i in sh.offset..sh.offset + sh.nfunc() {
                        x.row_mut(i).mapv_inplace(|v| v * self.d_scale);
                        x.column_mut(i).mapv_inplace(|v| v * self.d_scale);
                    }
                }
            }
        }
        m
    }
}

impl AoBackend for FakeBackend<'_> {
    fn name(&self) -> &str {
        "fake"
    }
    fn layout(&self) -> &AoLayout {
        &self.layout
    }
    fn one_electron(&mut self) -> Result<[Array2<f64>; 3], ferric_core::FerricError> {
        Ok(self.present(self.prep, &self.layout.clone()))
    }
    fn one_electron_at(
        &mut self,
        mol: &Molecule,
    ) -> Result<[Array2<f64>; 3], ferric_core::FerricError> {
        let bs = self.prep.basis_set().clone();
        let prep = PreparedBasis::new(mol, &bs)?;
        let layout = AoLayout::new(mol, &bs).expect("probe layout");
        Ok(self.present(&prep, &layout))
    }
    fn coulomb(&mut self, d: &Array2<f64>) -> Result<Array2<f64>, ferric_core::FerricError> {
        let df = self.jk_map.to_ferric(d);
        let mut j = Array2::zeros(df.dim());
        self.j.build(&df, &mut j)?;
        Ok(self.jk_map.to_cuest(&j))
    }
    fn exchange(&mut self, d: &Array2<f64>) -> Result<Array2<f64>, ferric_core::FerricError> {
        let df = self.jk_map.to_ferric(d);
        let mut k = Array2::zeros(df.dim());
        self.k.update_density(&df);
        self.k.build(&df, &mut k)?;
        Ok(self.jk_map.to_cuest(&k))
    }
}

fn config(xc: Option<&str>) -> RhfConfig {
    let mut c = RhfConfig::default()
        .with_density_conv(1e-9)
        // Exact four-centre J/K on the reference side, as the backend's.
        .with_df_j_aux(String::new())
        .with_df_k_aux(String::new());
    if let Some(x) = xc {
        c = c.with_xc(x);
    }
    c
}

fn closed_shell_matches(basis_name: &str, xc: Option<&str>) {
    let su = setup(basis_name, 0, 1);
    let cfg = config(xc);
    let reference = solve_rhf(&su.ctx, &su.mol, &su.prep, su.op, &su.bounds, &cfg).expect("ref");
    let run = solve_rhf_backend(
        &su.ctx,
        &su.mol,
        &su.prep,
        su.op,
        &su.bounds,
        &cfg,
        Box::new(FakeBackend::new(&su, foreign)),
    )
    .expect("backend SCF");
    assert!(reference.converged && run.result.converged);
    assert!(
        !run.calibration.matches_hypothesis,
        "the foreign order should not match the built-in guess"
    );
    let de = (run.result.energy - reference.energy).abs();
    println!(
        "{basis_name} {xc:?}: E = {:.12}, |dE| = {de:.2e}; {}",
        run.result.energy,
        run.calibration.summary()
    );
    assert!(de < BAR_E, "{basis_name} {xc:?}: |dE| = {de:.3e}");
}

#[test]
fn rhf_through_a_foreign_ao_order_matches_ferric() {
    closed_shell_matches("6-31g*", None); // Cartesian d
    closed_shell_matches("cc-pvdz", None); // pure d (p sent pure too)
}

#[test]
fn rks_through_a_foreign_ao_order_matches_ferric() {
    closed_shell_matches("cc-pvdz", Some("PBE"));
    closed_shell_matches("cc-pvdz", Some("PBE0"));
}

#[test]
fn uhf_and_uks_through_a_foreign_ao_order_match_ferric() {
    let su = setup("cc-pvdz", 1, 2);
    for xc in [None, Some("PBE0")] {
        let cfg = config(xc);
        let reference = solve_uhf(&su.ctx, &su.mol, &su.prep, &su.bounds, &cfg).expect("ref");
        let run = solve_uhf_backend(
            &su.ctx,
            &su.mol,
            &su.prep,
            &su.bounds,
            &cfg,
            Box::new(FakeBackend::new(&su, foreign)),
        )
        .expect("backend UHF");
        let de = (run.result.energy - reference.energy).abs();
        println!("H2O+ {xc:?}: |dE| = {de:.2e}");
        assert!(reference.converged && run.result.converged);
        assert!(de < BAR_E, "{xc:?}: |dE| = {de:.3e}");
    }
}

/// Mutation: J/K in the hypothesis order while S/T/V are foreign. The
/// calibration passes (it only sees S/T/V); the energy must be visibly wrong.
#[test]
fn jk_in_a_different_order_is_caught_by_the_energy() {
    let su = setup("cc-pvdz", 0, 1);
    let cfg = config(None);
    let reference = solve_rhf(&su.ctx, &su.mol, &su.prep, su.op, &su.bounds, &cfg).expect("ref");
    let mut b = FakeBackend::new(&su, foreign);
    b.jk_map = AoMap::hypothesis(&b.layout);
    let run = solve_rhf_backend(
        &su.ctx,
        &su.mol,
        &su.prep,
        su.op,
        &su.bounds,
        &cfg,
        Box::new(b),
    );
    match run {
        Ok(r) => {
            let de = (r.result.energy - reference.energy).abs();
            assert!(de > 1e-3, "mis-ordered J/K went unnoticed: |dE| = {de:.3e}");
        }
        Err(e) => println!("mis-ordered J/K: SCF failed ({e}), also a detection"),
    }
}

/// Mutation: a normalization disagreement on the d shells must stop the run
/// before the SCF, naming the factor.
#[test]
fn a_normalization_error_is_refused() {
    let su = setup("cc-pvdz", 0, 1);
    let mut b = FakeBackend::new(&su, foreign);
    b.d_scale = 1.05;
    let err = solve_rhf_backend(
        &su.ctx,
        &su.mol,
        &su.prep,
        su.op,
        &su.bounds,
        &config(None),
        Box::new(b),
    )
    .expect_err("a scaled d shell must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("calibration failed") && msg.contains("1.05"),
        "{msg}"
    );
}
