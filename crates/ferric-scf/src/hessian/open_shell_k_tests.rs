//! Exactness anchors of the factored RI-K of the open-shell CPHF kernel
//! ([`TwoElectronBuilder::uhf_fitted_exchanges`]): each spin's fitted K of
//! every UHF/UKS response density, built from that spin's `L_P = B_P C_σ,o`
//! by [`crate::df_k::DfK::build_factored_many`], against the density path it
//! replaced (the oracle), elementwise on the K matrices themselves; the
//! closed-shell limit (a closed-shell UHF through the per-spin route equals
//! the RHF factored route); and the UKS CPKS response through the factored
//! route against the density-path one. ROHF's factors are anchored in
//! `hessian::rohf` (`factored_k_tests`), which sees its `Frame`.
//!
//! Bars:
//! - K matrices: the factored product and the density path compute the same
//!   `K = Σ_P B_P (X C_oᵀ + C_o Xᵀ) B_P` with the sums regrouped; the longest
//!   sum either forms is `L = 2n + naux·o + 2` terms over terms bounded
//!   elementwise by `A = Σ_P |B_P| (|X||C_o|ᵀ + |C_o||X|ᵀ) |B_P|`, so the two
//!   differ by at most `2 γ_L A` (`γ_L = L u / (1 − L u)`, `u = ε/2`; the bar
//!   of `crate::df_k::response`'s `factored_k_matches_the_density_path`);
//! - closed-shell limit: the RHF route builds `K(2 D1_σ)` from `X = 2 X_σ`,
//!   an exact scaling by two of every product, so `K_RHF = 2 K_σ` bit for bit;
//! - response terms: the sum of the two compared terms' consistency probes
//!   `max |R − Rᵀ|` (as in `cpks_block_tests`).

use super::*;

const AUX: &str = "def2-universal-jkfit";

/// Distorted water in 6-31G* (no symmetry zeros in the response).
fn distorted_water(charge: i32, mult: usize) -> (Molecule, PreparedBasis, SchwarzBounds) {
    let xyz = "3\nwater\nO 0.02 -0.01 0.1173\nH 0.0 0.80 -0.45\nH 0.05 -0.74 -0.50\n";
    let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    let prep = PreparedBasis::new(&mol, &ferric_core::basis::bundled("6-31g*").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    (mol, prep, bounds)
}

/// The SCF's RI-K (packed f64, the production tier the bar reads).
pub(crate) fn fitted_k(mol: &Molecule, prep: &PreparedBasis) -> crate::df_k::DfK<'static> {
    let aux = PreparedBasis::new(mol, &ferric_core::basis::bundled(AUX).unwrap()).unwrap();
    let df = crate::df_k::DfK::new(Operator::coulomb(), prep, &aux, 1 << 30).unwrap();
    assert!(df.is_packed(), "the anchor must reach the packed f64 tier");
    df
}

/// `A = Σ_P |B_P| E |B_P|`, `E = |X||C_o|ᵀ + |C_o||X|ᵀ` (module bar).
fn abs_bound(dfk: &crate::df_k::DfK<'_>, x: &Array2<f64>, c_o: ArrayView2<'_, f64>) -> Array2<f64> {
    let n = c_o.nrows();
    let (xa, ca) = (x.mapv(f64::abs), c_o.mapv(f64::abs));
    let e = xa.dot(&ca.t()) + ca.dot(&xa.t());
    let rows = dfk.dressed_rows_for_test().expect("packed rows");
    let mut buf = vec![0.0; n * n];
    let mut a = Array2::<f64>::zeros((n, n));
    for p in 0..rows.nrows() {
        rows.unpack_lower_into(p, &mut buf, n);
        let b = Array2::from_shape_fn((n, n), |(i, j)| buf[i.max(j) * n + i.min(j)].abs());
        a += &b.dot(&e).dot(&b);
    }
    a
}

/// Assert `got` (factored) and `want` (density path) agree within the module
/// bar `2 γ_L A` elementwise, and return the worst `|diff| / bar`.
pub(crate) fn assert_k_within_bar(
    label: &str,
    dfk: &crate::df_k::DfK<'_>,
    got: &Array2<f64>,
    want: &Array2<f64>,
    x: &Array2<f64>,
    c_o: ArrayView2<'_, f64>,
) -> f64 {
    let (n, o) = c_o.dim();
    let naux = dfk.dressed_rows_for_test().expect("packed rows").nrows();
    let len = (2 * n + naux * o + 2) as f64;
    let u = f64::EPSILON / 2.0;
    let gamma = len * u / (1.0 - len * u);
    let bar = abs_bound(dfk, x, c_o) * (2.0 * gamma);
    let (mut worst, mut ratio) = (0.0f64, 0.0f64);
    for ((a, b), t) in got.iter().zip(want).zip(&bar) {
        worst = worst.max((a - b).abs());
        ratio = ratio.max((a - b).abs() / t.max(f64::MIN_POSITIVE));
    }
    eprintln!(
        "{label}: max |K_fac − K_dens| {worst:.3e} (scale {:.3e}), max diff/bar {ratio:.3e}",
        max_abs(want)
    );
    assert!(ratio <= 1.0, "{label}: exceeds the bar by {ratio:.3e}x");
    ratio
}

/// A deterministic, unstructured `(rows, cols)` matrix.
fn probe(rows: usize, cols: usize, seed: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        ((i * 37 + j * 11) as f64 * seed).sin()
    })
}

/// A converged UHF with RI-J and RI-K over [`AUX`] (`charge`, `mult`).
struct UhfCase {
    ctx: ParallelContext,
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    uhf: ScfResult,
    nocc: [usize; 2],
}

impl UhfCase {
    fn new(charge: i32, mult: usize) -> Self {
        let ctx = ParallelContext::default();
        let (mol, prep, bounds) = distorted_water(charge, mult);
        let cfg = RhfConfig {
            df_j_aux: Some(AUX.into()),
            df_k_aux: Some(AUX.into()),
            ri_j_precision: Some(crate::df_j::RiJPrecision::F64),
            energy_conv: 1e-12,
            density_conv: 1e-10,
            ..Default::default()
        };
        let uhf = crate::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
        assert!(uhf.converged);
        let nocc = uhf_occupations(&mol);
        UhfCase {
            ctx,
            mol,
            prep,
            bounds,
            uhf,
            nocc,
        }
    }

    fn frames(&self) -> [MoFrame<'_>; 2] {
        [
            MoFrame::new(self.uhf.mos_a(), self.uhf.eps_a(), self.nocc[0], 1.0),
            MoFrame::new(self.uhf.mos_b(), self.uhf.eps_b(), self.nocc[1], 1.0),
        ]
    }

    /// RI-K only (the J side is not under test), factored route unless
    /// `density`.
    fn builder(&self, density: bool) -> TwoElectronBuilder<'_> {
        TwoElectronBuilder::new(&self.ctx, &self.prep, &self.bounds)
            .unwrap()
            .fitted(None, Some(fitted_k(&self.mol, &self.prep)), false)
            .with_density_path_k_if(density)
    }
}

/// UHF (H2O+ doublet): the factored K of each spin of three rotation pairs
/// and two occupied-block pairs, through the production factors
/// ([`MoFrame::response_factor`], [`MoFrame::occupied_factor`],
/// [`spin_factors`]) and per-spin half transforms, equals the density-path K
/// of the same densities within `2 γ_L A`, spin by spin.
#[test]
fn uhf_factored_k_matches_the_density_path_per_spin_density() {
    let case = UhfCase::new(1, 2);
    assert_ne!(case.nocc[0], case.nocc[1], "an open shell");
    let frames = case.frames();
    let g = case.builder(false);
    let df = g.df_k.as_ref().expect("fitted K");
    for occupied in [false, true] {
        let mut ds: Vec<[Array2<f64>; 2]> = Vec::new();
        let mut xs: [Vec<Array2<f64>>; 2] = Default::default();
        for k in 1..=3 {
            let pair = [0, 1].map(|sp| {
                let (fr, no) = (&frames[sp], case.nocc[sp]);
                let seed = 0.17 * k as f64 + 0.05 * sp as f64;
                if occupied {
                    let s = probe(no, no, seed);
                    let s = 0.5 * (&s + &s.t());
                    (fr.occupied_density(&s), fr.occupied_factor(&s))
                } else {
                    let u = probe(fr.c_vir.ncols(), no, seed);
                    (fr.response_density(&u), fr.response_factor(&u))
                }
            });
            let [(da, xa), (db, xb)] = pair;
            ds.push([da, db]);
            xs[0].push(xa);
            xs[1].push(xb);
        }
        let got = g
            .uhf_fitted_exchanges(df, &ds, Some(spin_factors(&frames, &xs)))
            .unwrap();
        for (k, (pair, d)) in got.iter().zip(&ds).enumerate() {
            let want = g.fitted_exchange(df, d).unwrap();
            for sp in 0..2 {
                assert_k_within_bar(
                    &format!("UHF occupied={occupied} #{k} spin {sp}"),
                    &df.borrow(),
                    &pair[sp],
                    &want[sp],
                    &xs[sp][k],
                    frames[sp].c_occ,
                );
            }
        }
    }
}

/// Closed-shell limit: a closed-shell water through the per-spin factored
/// route (`occ` 1 per spin, `X_σ = C_v U`) equals the RHF factored route
/// (`occ` 2, `X = 2 C_v U`) as `K_RHF = 2 K_σ` bit for bit (module bar), and
/// both spins (the same orbitals, each with its own half transform) give the
/// same matrix bit for bit.
#[test]
fn closed_shell_uhf_factored_k_equals_the_rhf_factored_k() {
    let case = UhfCase::new(0, 1);
    assert_eq!(case.nocc[0], case.nocc[1]);
    // The closed-shell limit is `C_α = C_β` exactly; the converged UHF spin
    // sets agree only to the SCF tolerance, so both spins take the α set.
    let mo = |_| MoFrame::new(case.uhf.mos_a(), case.uhf.eps_a(), case.nocc[0], 1.0);
    let frames: [MoFrame<'_>; 2] = [0, 1].map(mo);
    let rhf_frame = MoFrame::new(case.uhf.mos_a(), case.uhf.eps_a(), case.nocc[0], 2.0);
    let g = case.builder(false);
    let df = g.df_k.as_ref().expect("fitted K");
    let us: Vec<Array2<f64>> = (1..=3)
        .map(|k| probe(frames[0].c_vir.ncols(), case.nocc[0], 0.23 * k as f64))
        .collect();
    let ds: Vec<[Array2<f64>; 2]> = us
        .iter()
        .map(|u| [0, 1].map(|sp| frames[sp].response_density(u)))
        .collect();
    let xs: [Vec<Array2<f64>>; 2] =
        [0, 1].map(|sp| us.iter().map(|u| frames[sp].response_factor(u)).collect());
    let open = g
        .uhf_fitted_exchanges(df, &ds, Some(spin_factors(&frames, &xs)))
        .unwrap();
    let rd: Vec<Array2<f64>> = us.iter().map(|u| rhf_frame.response_density(u)).collect();
    let rx: Vec<Array2<f64>> = us.iter().map(|u| rhf_frame.response_factor(u)).collect();
    let refs: Vec<&Array2<f64>> = rd.iter().collect();
    let closed = g
        .rhf_fitted_exchanges(&refs, Some((rhf_frame.c_occ, &rx)))
        .unwrap()
        .unwrap();
    for (k, (pair, kr)) in open.iter().zip(&closed).enumerate() {
        assert_eq!(pair[0], pair[1], "#{k}: the two spins differ");
        let diff = max_abs(&(kr - &(2.0 * &pair[0])));
        eprintln!("closed-shell #{k}: max |K_RHF − 2 K_σ| {diff:.3e}");
        assert_eq!(kr, &(2.0 * &pair[0]), "#{k}: K_RHF != 2 K_σ");
    }
}

/// Open-shell B3LYP (H2O+ doublet) with RI-J and RI-K: the UKS CPKS response
/// through the per-spin factored exchange equals the density-path (oracle)
/// response, in lockstep blocks of all 9 coordinates.
#[test]
fn uks_cpks_factored_k_matches_the_density_path() {
    use crate::ks_fixed_grid::{FixedGridKs, XcGeometry, XcMutants};
    let grid_cfg = ferric_dft::grid::AtomicGridConfig {
        n_radial: 40,
        n_angular: 110,
        prune: None,
        weight_cut: None,
    };
    let ctx = ParallelContext::default();
    let (mol, prep, bounds) = distorted_water(1, 2);
    let cfg = RhfConfig {
        xc: Some("B3LYP".into()),
        df_j_aux: Some(AUX.into()),
        df_k_aux: Some(AUX.into()),
        ri_j_precision: Some(crate::df_j::RiJPrecision::F64),
        dft_grid: Some(grid_cfg.clone()),
        energy_conv: 1e-12,
        density_conv: 1e-10,
        ..Default::default()
    };
    let uks = crate::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
    assert!(uks.converged);
    let nocc = uhf_occupations(&mol);
    let grid = FixedGridKs::new(&mol, "B3LYP", &grid_cfg).unwrap();
    let (da, db) = uhf_spin_densities(&uks).unwrap();
    let geo = XcGeometry::new(&mol, &prep).unwrap();
    let pol = grid.polarized().unwrap();
    let mo = pol
        .response_mo(
            &geo,
            [da, db],
            ([uks.mos_a(), uks.mos_b()], nocc),
            1.0,
            &XcMutants::default(),
        )
        .unwrap();
    let exx = grid.exact_exchange();
    assert!(exx > 0.0, "a hybrid, so K is built");
    let frames = [
        MoFrame::new(uks.mos_a(), uks.eps_a(), nocc[0], 1.0),
        MoFrame::new(uks.mos_b(), uks.eps_b(), nocc[1], 1.0),
    ];
    let d = da + db;
    let specs = [da, db].map(|ds| FockSpec {
        coulomb: &d,
        exchange: ds,
        k_scale: 0.0,
    });
    let first = first_order_ao(
        &prep,
        Operator::coulomb(),
        &bounds,
        mol.atoms.len(),
        &[],
        &specs,
    )
    .unwrap();
    let h1 = [0, 1].map(|sp| frames[sp].to_mo_all(&first.f1[sp]));
    let s1 = [0, 1].map(|sp| frames[sp].to_mo_all(&first.s1));
    let aux = PreparedBasis::new(&mol, &ferric_core::basis::bundled(AUX).unwrap()).unwrap();
    let response = |density: bool| {
        let df_j = crate::df_j::DfJ::new_with_precision(
            Operator::coulomb(),
            &prep,
            &aux,
            1 << 30,
            crate::df_j::RiJPrecision::F64,
        )
        .unwrap();
        let g = TwoElectronBuilder::new(&ctx, &prep, &bounds)
            .unwrap()
            .fitted(Some(df_j), Some(fitted_k(&mol, &prep)), false)
            .with_uks_xc_mo_response(&mo, exx)
            .with_density_path_k_if(density);
        let sols = solve_ucphf_blocks_of(&g, &frames, &h1, &s1, 3 * mol.atoms.len()).unwrap();
        assemble_response(&h1[0], &s1[0], &sols[0], &frames[0])
            + assemble_response(&h1[1], &s1[1], &sols[1], &frames[1])
    };
    let (r_fac, r_den) = (response(false), response(true));
    let asym = |r: &Array2<f64>| max_abs(&(r - &r.t()));
    let diff = max_abs(&(&r_fac - &r_den));
    let bar = asym(&r_fac) + asym(&r_den);
    eprintln!("UKS CPKS factored vs density-path K: {diff:.3e} (bar {bar:.3e})");
    assert!(diff <= bar, "differ by {diff:.3e} > {bar:.3e}");
}
