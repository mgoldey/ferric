//! Exactness anchors of the unrestricted response
//! (`book/src/20-tddft/06-unrestricted.md#unrestricted-spin-problem`,
//! `book/src/20-tddft/06-unrestricted.md#unrestricted-each-root`).

use super::*;
use crate::tddft::{add_fxc_block, build_kernel, coulomb_factor};
use crate::{ResponseSpin, TddftConfig};
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray_linalg::{Eigh, UPLO};

struct Case {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    scf: ScfResult,
}

fn prepare(xyz: &str, mult: usize) -> (Molecule, PreparedBasis, PreparedBasis) {
    let path = format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    );
    let mol = Molecule::load_xyz_with_charge(&path, 0, mult).unwrap();
    let obs = PreparedBasis::new(&mol, &ferric_core::basis::bundled("6-31g").unwrap()).unwrap();
    let dfbs =
        PreparedBasis::new(&mol, &ferric_core::basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    (mol, obs, dfbs)
}

fn scf_config(xc: Option<&str>) -> RhfConfig {
    RhfConfig {
        xc: xc.map(str::to_string),
        density_conv: 1e-9,
        max_iter: 300,
        ..Default::default()
    }
}

/// Water/6-31G, restricted.
fn water(xc: Option<&str>) -> Case {
    let (mol, obs, dfbs) = prepare("water.xyz", 1);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let scf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &scf_config(xc),
    )
    .unwrap();
    assert!(scf.converged);
    Case {
        mol,
        obs,
        dfbs,
        scf,
    }
}

/// OH/6-31G, unrestricted. Its ²Π ground state is doubly degenerate, so the
/// β π→π rotation between the two components is a zero mode (Ω² at the
/// noise floor, either sign): fine for products, not for Casida roots.
fn oh(xc: Option<&str>) -> Case {
    open_shell("oh.xyz", xc)
}

/// NH2/6-31G (²B₁, non-degenerate), unrestricted.
fn nh2(xc: Option<&str>) -> Case {
    open_shell("nh2.xyz", xc)
}

fn open_shell(xyz: &str, xc: Option<&str>) -> Case {
    let (mol, obs, dfbs) = prepare(xyz, 2);
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let scf = ferric_scf::uhf::solve_uhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        &bounds,
        &scf_config(xc),
    )
    .unwrap();
    assert!(scf.converged);
    Case {
        mol,
        obs,
        dfbs,
        scf,
    }
}

/// The restricted reference written as an unrestricted one with
/// `C_α = C_β`, `ε_α = ε_β`, `D_α = D_β`.
fn as_unrestricted(r: &ScfResult) -> ScfResult {
    let mut u = r.clone();
    u.spin = ferric_scf::Spin::Unrestricted;
    u.mos_beta = Some(r.mos_alpha.clone());
    u.eps_beta = Some(r.eps_alpha.clone());
    u.density_beta = Some(r.density_alpha.clone());
    u.fock_beta = Some(r.fock_alpha.clone());
    u
}

fn maxabs(m: &Array2<f64>) -> f64 {
    m.iter().fold(0.0_f64, |x, &y| x.max(y.abs()))
}

fn frobenius(m: &Array2<f64>) -> f64 {
    m.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn spectral(m: &Array2<f64>) -> f64 {
    let (w, _) = (0.5 * (m + &m.t())).eigh(UPLO::Lower).unwrap();
    w.iter().fold(0.0_f64, |x, &y| x.max(y.abs()))
}

/// Elementwise rounding bound of the RI terms of one spin's tensors with
/// exchange `mix` and Coulomb coefficient `c_j`: each element is an
/// `naux`-term dot product, within `γ_n Σ|terms|` of the exact sum, and two
/// routes that order it differently are within twice that. `Σ|terms|` is
/// bounded by the largest element of `|B|ᵀ|B|`, weighted by the coefficient
/// each term enters `A+B` with.
fn ri_bar(ri: &RiTensors, mix: ExchangeMix, c_j: f64) -> f64 {
    let naux = ri.b_ov.nrows() as f64;
    let gamma = naux * f64::EPSILON / (1.0 - naux * f64::EPSILON);
    let sums = |t: &RiTensors| {
        let s_j = maxabs(&t.b_ov.mapv(f64::abs).t().dot(&t.b_ov.mapv(f64::abs)));
        let s_k = maxabs(&t.b_oo.mapv(f64::abs).t().dot(&t.b_vv.mapv(f64::abs))).max(s_j);
        (s_j, s_k)
    };
    let (s_j, s_k) = sums(ri);
    let mut bar = 2.0 * gamma * (2.0 * c_j * s_j + 2.0 * mix.c_sr.abs() * s_k);
    if let (Some(lr), Some(dc)) = (ri.lr.as_deref(), mix.long_range()) {
        bar += 2.0 * gamma * 2.0 * dc.abs() * sums(lr).1;
    }
    bar
}

/// Everything the dense and matrix-free comparisons need for one run.
struct Built {
    sp: USpace,
    ri: [RiTensors; 2],
    mix: ExchangeMix,
    kernel: Option<UksKernel>,
}

fn build(c: &Case, xc: Option<&str>) -> Built {
    let (na, nb) = occupations(&c.mol, &c.scf).unwrap();
    let (mix, def) = resolve_response_xc(xc, None).unwrap();
    let sp = space(&c.scf, na, nb).unwrap();
    let cfg = TddftConfig::default();
    let mut plan = MemoryPlan::resolve(None, "test");
    let mut acc = PhaseAccumulator::new();
    let ri = [
        build_response_tensors(
            &c.obs,
            &c.dfbs,
            &sp.spins[0],
            mix,
            &cfg,
            &mut plan,
            &mut acc,
        )
        .unwrap(),
        build_response_tensors(
            &c.obs,
            &c.dfbs,
            &sp.spins[1],
            mix,
            &cfg,
            &mut plan,
            &mut acc,
        )
        .unwrap(),
    ];
    let kernel = def.map(|d| {
        UksKernel::new(
            &c.mol,
            c.obs.basis_set(),
            d,
            &AtomicGridConfig::production(),
            &c.scf.density_alpha,
            c.scf.density_beta.as_ref().unwrap(),
        )
        .unwrap()
    });
    Built {
        sp,
        ri,
        mix,
        kernel,
    }
}

/// The restricted dense `(A, B)` of `spin` on `c`'s orbitals.
fn restricted_dense(c: &Case, xc: Option<&str>, spin: ResponseSpin) -> (Array2<f64>, Array2<f64>) {
    let nocc = (c.mol.nelec() / 2) as usize;
    let (mix, def) = resolve_response_xc(xc, None).unwrap();
    let space = OvSpace::from_mos(c.scf.mos_r(), c.scf.eps_r().len(), nocc).unwrap();
    let mut plan = MemoryPlan::resolve(None, "test");
    let mut acc = PhaseAccumulator::new();
    let cfg = TddftConfig::default();
    let ri =
        build_response_tensors(&c.obs, &c.dfbs, &space, mix, &cfg, &mut plan, &mut acc).unwrap();
    let kernel =
        build_kernel(&c.mol, &c.obs, def, &AtomicGridConfig::production(), &c.scf).unwrap();
    let c_j = coulomb_factor(spin);
    let mut a = build_a_matrix(c.scf.eps_r(), nocc, space.nvir, &ri, mix, c_j);
    let mut b = build_b_matrix(nocc, space.nvir, &ri, mix, c_j);
    add_fxc_block(
        kernel.as_ref(),
        spin,
        &c.scf,
        nocc,
        space.nvir,
        &mut [&mut a, &mut b],
    )
    .unwrap();
    (a, b)
}

/// `Uᵀ M U` for `U = [[1, 1], [1, −1]]/√2` on the `(α, β)` halves: the
/// singlet / triplet blocks of a closed-shell unrestricted matrix.
fn rotate_to_spin_adapted(m: &Array2<f64>) -> Array2<f64> {
    let d = m.nrows() / 2;
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let mut u = Array2::<f64>::zeros((2 * d, 2 * d));
    for k in 0..d {
        u[(k, k)] = h;
        u[(k + d, k)] = h;
        u[(k, k + d)] = h;
        u[(k + d, k + d)] = -h;
    }
    u.t().dot(m).dot(&u)
}

fn blockdiag(s: &Array2<f64>, t: &Array2<f64>) -> Array2<f64> {
    let d = s.nrows();
    let mut m = Array2::<f64>::zeros((2 * d, 2 * d));
    m.slice_mut(s![..d, ..d]).assign(s);
    m.slice_mut(s![d.., d..]).assign(t);
    m
}

fn run(
    c: &Case,
    xc: Option<&str>,
    method: TddftMethod,
    solver: TddftSolver,
    n: usize,
) -> TddftResult {
    let cfg = TddftConfig {
        n_roots: n,
        method,
        xc: xc.map(str::to_string),
        grid: AtomicGridConfig::production(),
        solver,
        ..TddftConfig::default()
    };
    crate::run_tddft(&c.mol, &c.obs, &c.dfbs, &c.scf, &cfg).unwrap()
}

const CLOSED_CASES: [Option<&str>; 5] = [
    None,
    Some("PBE"),
    Some("B3LYP"),
    Some("HYB_GGA_XC_CAM_B3LYP"),
    Some("r2SCAN"),
];

/// The exactness anchor (`book/src/20-tddft/06-unrestricted.md#unrestricted-spin-problem`):
/// water/6-31G, the RHF/RKS reference
/// written as an unrestricted one (`C_α = C_β`), HF, PBE, B3LYP, CAM-B3LYP
/// and r2SCAN.
///
/// 1. Matrices: `Uᵀ A_u U` (and `B`) equals `blockdiag(A_S, A_T)` of the
///    restricted builders elementwise, within `E_bar` = the RI rounding bar
///    of the two routes (`ri_bar` with the singlet's Coulomb coefficient 2,
///    times 2 for the four-element rotation) + `3·FXC_BLOCK_ASYMMETRY_TOL·max|K|`
///    (the symmetrization of the restricted block and, through the
///    rotation, of two unrestricted elements). The kernels differ also by
///    their density screen (per spin against total), which is not bounded
///    a priori; the measured difference sitting under the rounding bar
///    shows it is not reached on this grid.
/// 2. Roots: all `2d` unrestricted roots of `run_tddft` equal the sorted
///    union of the restricted singlets and triplets, within Weyl's
///    `‖E‖₂ ≤ ‖E‖_F ≤ n·E_bar` plus `2·n·ε·‖A‖₂` for the two `eigh` (TDA),
///    and the Casida perturbation bound of
///    `tddft::tests::davidson_roots_equal_the_dense_roots` with
///    `e± = n·E_bar` (Casida).
/// 3. `<S²>`: 0 on the roots matched to singlets and 2 on the triplets,
///    within `2(‖E‖₂/gap)² + 1e-10`, `gap` the distance to the nearest root
///    of the other spin (Davis–Kahan: the eigenvector leaks `sin θ ≤ ‖E‖/gap`
///    into the other spin, and `<S²>` is quadratic in it).
///
/// Measured (water/6-31G): matrices within 7.1e-12 (r2SCAN; 7.2e-14 or less
/// for the rest) against bars of 2e-13 (HF) to 3.9e-9; roots within
/// 9.1e-12 Ha; `<S²>` within 8.4e-15 of 0 or 2.
///
/// Mutations (applied and reverted, 2026-11-14): the α-β f_xc block dropped
/// (`UksOvFxc::apply`) fails the PBE matrix check (|dA| 8.2e-3 against
/// 2.8e-9); the cross-spin Coulomb block zeroed fails the HF matrix check
/// (|dA| 0.159); the `<S²>` cross term with its sign flipped fails the `<S²>`
/// check (a triplet root reports 0).
#[test]
fn closed_shell_through_the_unrestricted_route_is_singlets_plus_triplets() {
    for xc in CLOSED_CASES {
        let mut c = water(xc);
        let (a_s, b_s) = restricted_dense(&c, xc, ResponseSpin::Singlet);
        let (a_t, b_t) = restricted_dense(&c, xc, ResponseSpin::Triplet);
        c.scf = as_unrestricted(&c.scf);
        let u = build(&c, xc);
        let fxc = u.kernel.as_ref().map(|k| UksOvFxc::new(k, u.sp.orbitals()));
        let m = dense_matrices(
            TddftMethod::Casida,
            &u.sp,
            &c.scf,
            &u.ri,
            u.mix,
            fxc.as_ref(),
        )
        .unwrap();
        let b_u = m.b.as_ref().unwrap();
        let k_scale = maxabs(
            &(&a_s
                - &build_a_matrix(
                    c.scf.eps_a(),
                    u.sp.spins[0].nocc,
                    u.sp.spins[0].nvir,
                    &u.ri[0],
                    u.mix,
                    2.0,
                )),
        );
        let e_bar = 2.0 * ri_bar(&u.ri[0], u.mix, 2.0)
            + 3.0 * ferric_dft::lr_kernel::FXC_BLOCK_ASYMMETRY_TOL * k_scale;
        let da = maxabs(&(&rotate_to_spin_adapted(&m.a) - &blockdiag(&a_s, &a_t)));
        let db = maxabs(&(&rotate_to_spin_adapted(b_u) - &blockdiag(&b_s, &b_t)));
        eprintln!("{xc:?}: matrices |dA| {da:.2e}, |dB| {db:.2e} (bar {e_bar:.2e})");
        assert!(
            da <= e_bar && db <= e_bar,
            "{xc:?}: A {da:e} / B {db:e} > {e_bar:e}"
        );

        let n = m.a.nrows();
        let d = n / 2;
        let e_norm = n as f64 * e_bar;
        let floor = 2.0 * n as f64 * f64::EPSILON * spectral(&m.a);
        // TDA roots.
        let ur = run(&c, xc, TddftMethod::Tda, TddftSolver::Dense, n);
        let cs = water(xc);
        let singlets = run(&cs, xc, TddftMethod::Tda, TddftSolver::Dense, d);
        let triplets = {
            let cfg = TddftConfig {
                n_roots: d,
                method: TddftMethod::Tda,
                xc: xc.map(str::to_string),
                grid: AtomicGridConfig::production(),
                solver: TddftSolver::Dense,
                spin: ResponseSpin::Triplet,
                ..TddftConfig::default()
            };
            crate::run_tddft(&cs.mol, &cs.obs, &cs.dfbs, &cs.scf, &cfg).unwrap()
        };
        let mut merged: Vec<(f64, f64)> = singlets
            .excitation_energies
            .iter()
            .map(|&w| (w, 0.0))
            .chain(triplets.excitation_energies.iter().map(|&w| (w, 2.0)))
            .collect();
        merged.sort_by(|x, y| x.0.total_cmp(&y.0));
        assert_eq!(ur.excitation_energies.len(), merged.len());
        let mut worst = (0.0_f64, 0.0_f64);
        for (k, (&w, &(w_r, s2_r))) in ur.excitation_energies.iter().zip(&merged).enumerate() {
            let dw = (w - w_r).abs();
            assert!(
                dw <= e_norm + floor,
                "{xc:?} TDA root {k}: {dw:e} > {:e}",
                e_norm + floor
            );
            let gap = merged
                .iter()
                .filter(|(_, s)| *s != s2_r)
                .map(|(x, _)| (x - w_r).abs())
                .fold(f64::INFINITY, f64::min);
            let s2_bar = 2.0 * (e_norm / gap).powi(2) + 1e-10;
            let ds2 = (ur.s2[k] - s2_r).abs();
            assert!(
                ds2 <= s2_bar,
                "{xc:?} TDA root {k}: <S2> {} vs {s2_r} (bar {s2_bar:e})",
                ur.s2[k]
            );
            worst = (worst.0.max(dw), worst.1.max(ds2));
        }
        assert!(
            ur.s2_reference.abs() < 1e-10,
            "closed-shell <S2>_0 {}",
            ur.s2_reference
        );
        eprintln!(
            "{xc:?}: TDA max |dw| {:.2e}, max |d<S2>| {:.2e}",
            worst.0, worst.1
        );
        casida_anchor(&c, &cs, xc, &m.a, b_u, e_norm);
    }
}

/// Part 2 and 3 of the anchor for Casida.
fn casida_anchor(c: &Case, cs: &Case, xc: Option<&str>, a: &Array2<f64>, b: &Array2<f64>, e: f64) {
    let n = a.nrows();
    let d = n / 2;
    let ur = run(c, xc, TddftMethod::Casida, TddftSolver::Dense, n);
    let singlets = run(cs, xc, TddftMethod::Casida, TddftSolver::Dense, d);
    let triplets = {
        let cfg = TddftConfig {
            n_roots: d,
            method: TddftMethod::Casida,
            xc: xc.map(str::to_string),
            grid: AtomicGridConfig::production(),
            solver: TddftSolver::Dense,
            spin: ResponseSpin::Triplet,
            ..TddftConfig::default()
        };
        crate::run_tddft(&cs.mol, &cs.obs, &cs.dfbs, &cs.scf, &cfg).unwrap()
    };
    let mut merged: Vec<(f64, f64)> = singlets
        .excitation_energies
        .iter()
        .map(|&w| (w, 0.0))
        .chain(triplets.excitation_energies.iter().map(|&w| (w, 2.0)))
        .collect();
    merged.sort_by(|x, y| x.0.total_cmp(&y.0));
    let apb = a + b;
    let amb = a - b;
    let (lam, _) = amb.eigh(UPLO::Lower).unwrap();
    let (lmin, lmax) = (lam[0], lam[lam.len() - 1]);
    let cond = (lmax / lmin).sqrt();
    let n_p = spectral(&apb);
    let pert = cond * (e * n_p + lmax * e + e * e);
    let floor = 2.0 * n as f64 * f64::EPSILON * lmax * n_p;
    let mut worst = (0.0_f64, 0.0_f64);
    for (k, (&w, &(w_r, s2_r))) in ur.excitation_energies.iter().zip(&merged).enumerate() {
        let bar = (pert + floor) / (w + w_r);
        let dw = (w - w_r).abs();
        assert!(dw <= bar, "{xc:?} Casida root {k}: {dw:e} > {bar:e}");
        let gap = merged
            .iter()
            .filter(|(_, s)| *s != s2_r)
            .map(|(x, _)| (x - w_r).abs())
            .fold(f64::INFINITY, f64::min);
        let s2_bar = 2.0 * (bar.max(e) / gap).powi(2) + 1e-10;
        let ds2 = (ur.s2[k] - s2_r).abs();
        assert!(
            ds2 <= s2_bar,
            "{xc:?} Casida root {k}: <S2> {} vs {s2_r}",
            ur.s2[k]
        );
        worst = (worst.0.max(dw), worst.1.max(ds2));
    }
    eprintln!(
        "{xc:?}: Casida max |dW| {:.2e}, max |d<S2>| {:.2e}",
        worst.0, worst.1
    );
}

const OPEN_CASES: [Option<&str>; 4] = [
    None,
    Some("PBE"),
    Some("HYB_GGA_XC_CAM_B3LYP"),
    Some("r2SCAN"),
];

/// The matrix-free products, applied to the identity, reproduce the dense
/// A, A+B and A−B elementwise on OH/6-31G UHF/UKS (HF, PBE, CAM-B3LYP,
/// r2SCAN). Bar: the larger spin's `ri_bar` with Coulomb coefficient 2
/// (the spin-summed Coulomb term of `(A+B)`) plus `2·FXC_BLOCK_ASYMMETRY_TOL·max|K|`
/// (the dense block is the matrix-free one symmetrized, entering `A+B`
/// twice).
///
/// Mutation (applied and reverted, 2026-11-14): the Coulomb term of
/// `apply_pm` taken once instead of twice fails the `A+B` check (0.158
/// against 9e-14) while `A` and `A−B` stay at 4e-16.
#[test]
fn matrix_free_products_equal_the_dense_matrices() {
    for xc in OPEN_CASES {
        let c = oh(xc);
        let u = build(&c, xc);
        let fxc = || u.kernel.as_ref().map(|k| UksOvFxc::new(k, u.sp.orbitals()));
        let m = dense_matrices(
            TddftMethod::Casida,
            &u.sp,
            &c.scf,
            &u.ri,
            u.mix,
            fxc().as_ref(),
        )
        .unwrap();
        let b = m.b.unwrap();
        let k_max = fxc().map_or(0.0, |f| maxabs(&f.apply(&Array2::eye(f.dim()))));
        let bar = ri_bar(&u.ri[0], u.mix, 2.0).max(ri_bar(&u.ri[1], u.mix, 2.0))
            + 2.0 * ferric_dft::lr_kernel::FXC_BLOCK_ASYMMETRY_TOL * k_max;
        let op = u_operator(&u.sp, &c.scf, &u.ri, u.mix, fxc());
        let eye = Array2::<f64>::eye(u.sp.dim());
        let mut acc = PhaseAccumulator::new();
        let a_mf = op.apply_a(&eye, &mut acc).unwrap();
        let (p_mf, m_mf) = op.apply_pm(&eye, &mut acc).unwrap();
        let da = maxabs(&(&a_mf - &m.a));
        let dp = maxabs(&(&p_mf - &(&m.a + &b)));
        let dm = maxabs(&(&m_mf - &(&m.a - &b)));
        eprintln!("{xc:?}: A {da:.2e}, A+B {dp:.2e}, A-B {dm:.2e} (bar {bar:.2e})");
        assert!(
            da <= bar && dp <= bar && dm <= bar,
            "{xc:?}: {da:e} {dp:e} {dm:e} > {bar:e}"
        );
    }
}

/// `run_tddft` through Davidson equals the dense route on NH2/6-31G (HF,
/// PBE, CAM-B3LYP, r2SCAN), TDA and Casida, 5 roots, energies and `<S²>`.
///
/// Energy bars as `tddft::tests::davidson_roots_equal_the_dense_roots`
/// (TDA: `‖r‖²/gap + ‖Ã − A‖_F + dim·ε·‖A‖₂`; Casida: the perturbation of
/// `(A−B)(A+B)` plus the Davidson residual term). `<S²>` is compared on
/// roots isolated by more than 1e-3 Ha from their neighbours, with bar
/// `8 (r / gap)`: a residual `r` leaves the eigenvector `sin θ ≤ r/gap` off,
/// and `<S²>` moves by at most `2·‖S² − <S²>‖·sin θ`, `‖S²‖ ≤ 4` here.
#[test]
fn davidson_roots_equal_the_dense_roots() {
    for xc in OPEN_CASES {
        let c = nh2(xc);
        let u = build(&c, xc);
        let fxc = || u.kernel.as_ref().map(|k| UksOvFxc::new(k, u.sp.orbitals()));
        let m = dense_matrices(
            TddftMethod::Casida,
            &u.sp,
            &c.scf,
            &u.ri,
            u.mix,
            fxc().as_ref(),
        )
        .unwrap();
        let b = m.b.unwrap();
        let op = u_operator(&u.sp, &c.scf, &u.ri, u.mix, fxc());
        let eye = Array2::<f64>::eye(u.sp.dim());
        let mut acc = PhaseAccumulator::new();
        let a_mf = op.apply_a(&eye, &mut acc).unwrap();
        let (p_mf, m_mf) = op.apply_pm(&eye, &mut acc).unwrap();
        let dim = m.a.nrows() as f64;
        let eps = f64::EPSILON;

        let dense = run(&c, xc, TddftMethod::Tda, TddftSolver::Dense, 5);
        let dav = run(&c, xc, TddftMethod::Tda, TddftSolver::Davidson, 5);
        assert_eq!(dav.solver, TddftSolver::Davidson);
        let (w_all, _) = solve_tda(&m.a).unwrap();
        let e_a = frobenius(&(&a_mf - &m.a));
        let floor = dim * eps * spectral(&m.a);
        for k in 0..5 {
            let r = dav.residuals[k];
            let gap = (0..w_all.len())
                .filter(|&j| j != k)
                .map(|j| (w_all[j] - w_all[k]).abs())
                .fold(f64::INFINITY, f64::min);
            let bar = r * r / gap + e_a + floor;
            let de = (dav.excitation_energies[k] - dense.excitation_energies[k]).abs();
            eprintln!("{xc:?} TDA root {k}: |dw| {de:.2e} (bar {bar:.2e})");
            assert!(de <= bar, "{xc:?} TDA root {k}: {de:e} > {bar:e}");
            if k < 4 {
                compare_s2(&dense, &dav, k, 8.0 * r);
            }
        }

        let dense = run(&c, xc, TddftMethod::Casida, TddftSolver::Dense, 5);
        let dav = run(&c, xc, TddftMethod::Casida, TddftSolver::Davidson, 5);
        let apb = &m.a + &b;
        let amb = &m.a - &b;
        let (lam, _) = amb.eigh(UPLO::Lower).unwrap();
        let (lmin, lmax) = (lam[0], lam[lam.len() - 1]);
        let cond = (lmax / lmin).sqrt();
        let e_p = frobenius(&(&p_mf - &apb));
        let e_m = frobenius(&(&m_mf - &amb));
        let n_p = spectral(&apb);
        let (lam_p, _) = apb.eigh(UPLO::Lower).unwrap();
        let pert = cond * (e_m * n_p + lmax * e_p + e_m * e_p);
        let floor = dim * eps * lmax * n_p;
        for k in 0..5 {
            let r = dav.residuals[k];
            let (om, om_d) = (dav.excitation_energies[k], dense.excitation_energies[k]);
            let solve = 2f64.sqrt() * r * (om + n_p) * cond * (om / lam_p[0]).sqrt();
            let bar = (pert + solve + floor) / (om + om_d);
            let de = (om - om_d).abs();
            eprintln!("{xc:?} Casida root {k}: |dW| {de:.2e} (bar {bar:.2e})");
            assert!(de <= bar, "{xc:?} Casida root {k}: {de:e} > {bar:e}");
            if k < 4 {
                compare_s2(&dense, &dav, k, 8.0 * solve.max(r) * cond);
            }
        }
    }
}

/// `<S²>` of root `k` of two solves agree within `scale / gap` when the
/// root is isolated (gap > 1e-3 Ha) in the dense spectrum.
fn compare_s2(dense: &TddftResult, dav: &TddftResult, k: usize, scale: f64) {
    let w = &dense.excitation_energies;
    let gap = (0..w.len())
        .filter(|&j| j != k)
        .map(|j| (w[j] - w[k]).abs())
        .fold(f64::INFINITY, f64::min);
    if gap > 1e-3 {
        let ds2 = (dense.s2[k] - dav.s2[k]).abs();
        assert!(
            ds2 <= scale / gap + 1e-10,
            "root {k}: <S2> dense {} vs Davidson {} (bar {:e})",
            dense.s2[k],
            dav.s2[k],
            scale / gap
        );
    }
}
