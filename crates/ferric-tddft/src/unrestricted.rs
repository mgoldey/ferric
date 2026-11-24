//! Spin-conserving linear response of an unrestricted (UHF / UKS) reference:
//! TDA and Casida over the two-spin particle–hole space
//! (`book/src/19-tddft/06-unrestricted.md#unrestricted-spin-problem`), and
//! `<S²>` of each root (§11).
//!
//! The space is the α pairs (`ia = i·nvir_α + a`) followed by the β pairs.
//! Per spin the same-spin blocks are the restricted builders with the
//! Coulomb coefficient 1 (`(iaσ|jbσ)` once, not the singlet 2) and the
//! spin's own exchange; the Coulomb coupling `(iaα|jbβ) = B_ovαᵀ B_ovβ`
//! links the spins, and the f_xc kernel `(iaσ|f_στ|jbτ)` fills all four
//! blocks ([`ferric_dft::lr_kernel::UksOvFxc`]).

use crate::davidson::{davidson_casida, davidson_tda};
use crate::matvec::ResponseOperator;
use crate::tddft::{
    build_a_matrix, build_b_matrix, build_response_tensors, casida_residuals, reserve_davidson,
    reserve_dense_response, resolve_response_xc, select_roots, solve_casida, solve_tda,
    tda_residuals, ExchangeMix, OvSpace, RiTensors, Roots,
};
use crate::{TddftConfig, TddftMethod, TddftResult, TddftSolver};
use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use ferric_core::timing::PhaseAccumulator;
use ferric_core::FerricError;
use ferric_dft::lr_kernel::{SpinOrbitals, UksKernel, UksOvFxc, FXC_BLOCK_ASYMMETRY_TOL};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_scf::ScfResult;
use ndarray::{s, Array1, Array2, Axis};

/// Largest `|D_σ − C_occ C_occᵀ|` accepted as "the occupied orbitals are the
/// lowest `n_σ`": a converged density agrees with its own orbitals to the
/// SCF density threshold, and a non-aufbau occupation differs at O(1).
const AUFBAU_DENSITY_TOL: f64 = 1e-4;

/// The `(ia)` spaces of the two spins and their offsets.
pub(crate) struct USpace {
    pub(crate) spins: [OvSpace; 2],
    pub(crate) n_alpha: usize,
    pub(crate) n_beta: usize,
}

impl USpace {
    fn dims(&self) -> [usize; 2] {
        [self.spins[0].dim(), self.spins[1].dim()]
    }

    fn dim(&self) -> usize {
        self.spins[0].dim() + self.spins[1].dim()
    }

    fn orbitals(&self) -> [SpinOrbitals; 2] {
        self.spins.each_ref().map(|o| SpinOrbitals {
            c_occ: o.c_occ.clone(),
            c_vir: o.c_vir.clone(),
        })
    }
}

/// `(n_α, n_β)` from the molecule's charge and multiplicity, with the
/// reference's densities checked against its lowest `n_σ` orbitals.
fn occupations(mol: &Molecule, scf: &ScfResult) -> Result<(usize, usize), FerricError> {
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    let (na, nb) = ((nelec + two_s) / 2, (nelec - two_s) / 2);
    if na < 0 || nb < 0 || na + nb != nelec {
        return Err(FerricError::General(format!(
            "run_tddft: {nelec} electrons with multiplicity {} has no (n_alpha, n_beta)",
            mol.multiplicity
        )));
    }
    let (na, nb) = (na as usize, nb as usize);
    let beta = scf
        .density_beta
        .as_ref()
        .ok_or_else(|| FerricError::General("run_tddft: UHF reference without D_beta".into()))?;
    for (label, c, d, n) in [
        ("alpha", scf.mos_a(), &scf.density_alpha, na),
        ("beta", scf.mos_b(), beta, nb),
    ] {
        let occ = c.slice(s![.., ..n]);
        let dev = (&occ.dot(&occ.t()) - d)
            .iter()
            .fold(0.0_f64, |m, &x| m.max(x.abs()));
        if dev > AUFBAU_DENSITY_TOL {
            return Err(FerricError::General(format!(
                "run_tddft: the {label} density is not built from the lowest {n} {label} \
                 orbitals (max deviation {dev:.2e}); a non-aufbau (e.g. MOM) occupation is \
                 not supported by the response"
            )));
        }
    }
    Ok((na, nb))
}

fn space(scf: &ScfResult, na: usize, nb: usize) -> Result<USpace, FerricError> {
    let per_spin = |c: &Array2<f64>, nmo: usize, n: usize, label: &str| {
        OvSpace::from_mos(c, nmo, n).map_err(|_| {
            FerricError::General(format!(
                "run_tddft: the {label} spin has no particle-hole pair ({n} occupied of {nmo} \
                 orbitals); the unrestricted response needs at least one per spin"
            ))
        })
    };
    Ok(USpace {
        spins: [
            per_spin(scf.mos_a(), scf.eps_a().len(), na, "alpha")?,
            per_spin(scf.mos_b(), scf.eps_b().len(), nb, "beta")?,
        ],
        n_alpha: na,
        n_beta: nb,
    })
}

/// The unrestricted response operator: per-spin Δ and exchange
/// ([`ResponseOperator`] with no Coulomb term and no kernel), the
/// spin-summed Coulomb term and the two-spin f_xc kernel.
struct UOperator<'a> {
    spins: [ResponseOperator<'a>; 2],
    b_ov: [&'a Array2<f64>; 2],
    fxc: Option<UksOvFxc<'a>>,
    dims: [usize; 2],
}

impl UOperator<'_> {
    fn diag(&self) -> Array1<f64> {
        ndarray::concatenate(
            Axis(0),
            &[self.spins[0].diag.view(), self.spins[1].diag.view()],
        )
        .expect("1-D")
    }

    fn split(&self, x: &Array2<f64>) -> [Array2<f64>; 2] {
        let d = self.dims[0];
        [
            x.slice(s![..d, ..]).to_owned(),
            x.slice(s![d.., ..]).to_owned(),
        ]
    }

    fn join(a: Array2<f64>, b: Array2<f64>) -> Array2<f64> {
        ndarray::concatenate(Axis(0), &[a.view(), b.view()]).expect("same columns")
    }

    /// `J(x)_σ = B_ovσᵀ (B_ovα x_α + B_ovβ x_β)`.
    fn coulomb(&self, x: &[Array2<f64>; 2]) -> Array2<f64> {
        let w = self.b_ov[0].dot(&x[0]) + self.b_ov[1].dot(&x[1]);
        Self::join(self.b_ov[0].t().dot(&w), self.b_ov[1].t().dot(&w))
    }

    fn apply_a(
        &self,
        x: &Array2<f64>,
        acc: &mut PhaseAccumulator,
    ) -> Result<Array2<f64>, FerricError> {
        let xs = self.split(x);
        let ya = self.spins[0].apply_a(&xs[0], acc)?;
        let yb = self.spins[1].apply_a(&xs[1], acc)?;
        let mut y = Self::join(ya, yb);
        y += &acc.time("product: RI Coulomb", || self.coulomb(&xs));
        if let Some(f) = &self.fxc {
            y += &acc.time("product: f_xc", || f.apply(x));
        }
        Ok(y)
    }

    fn apply_pm(
        &self,
        x: &Array2<f64>,
        acc: &mut PhaseAccumulator,
    ) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
        let xs = self.split(x);
        let (pa, ma) = self.spins[0].apply_pm(&xs[0], acc)?;
        let (pb, mb) = self.spins[1].apply_pm(&xs[1], acc)?;
        let mut p = Self::join(pa, pb);
        p.scaled_add(2.0, &acc.time("product: RI Coulomb", || self.coulomb(&xs)));
        if let Some(f) = &self.fxc {
            p.scaled_add(2.0, &acc.time("product: f_xc", || f.apply(x)));
        }
        Ok((p, Self::join(ma, mb)))
    }
}

/// The per-spin operator: Δ and exchange only.
fn spin_operator<'a>(
    o: &OvSpace,
    eps: &[f64],
    ri: &'a RiTensors,
    mix: ExchangeMix,
) -> ResponseOperator<'a> {
    ResponseOperator {
        diag: o.orbital_differences(eps),
        coul: ri.tensors(),
        nocc: o.nocc,
        nvir: o.nvir,
        c_j: 0.0,
        c_hf: mix.c_sr,
        long_range: ri
            .lr
            .as_deref()
            .zip(mix.long_range())
            .map(|(lr, dc)| (lr.tensors(), dc)),
        fxc: None,
    }
}

fn u_operator<'a>(
    sp: &USpace,
    scf: &ScfResult,
    ri: &'a [RiTensors; 2],
    mix: ExchangeMix,
    fxc: Option<UksOvFxc<'a>>,
) -> UOperator<'a> {
    UOperator {
        spins: [
            spin_operator(&sp.spins[0], scf.eps_a(), &ri[0], mix),
            spin_operator(&sp.spins[1], scf.eps_b(), &ri[1], mix),
        ],
        b_ov: [&ri[0].b_ov, &ri[1].b_ov],
        fxc,
        dims: sp.dims(),
    }
}

/// Dense A (and B for Casida) over the two-spin space, f_xc included.
struct DenseU {
    a: Array2<f64>,
    b: Option<Array2<f64>>,
}

/// Write the per-spin blocks and the cross-spin Coulomb block of `m`.
fn fill_blocks(m: &mut Array2<f64>, blocks: [Array2<f64>; 2], ri: &[RiTensors; 2], d: usize) {
    let [ma, mb] = blocks;
    m.slice_mut(s![..d, ..d]).assign(&ma);
    m.slice_mut(s![d.., d..]).assign(&mb);
    let cross = ri[0].b_ov.t().dot(&ri[1].b_ov);
    m.slice_mut(s![..d, d..]).assign(&cross);
    m.slice_mut(s![d.., ..d]).assign(&cross.t());
}

/// The dense f_xc block `K` (the operator applied to the identity), checked
/// for symmetry and symmetrized.
fn dense_fxc(f: &UksOvFxc<'_>) -> Result<Array2<f64>, FerricError> {
    let k = f.apply(&Array2::eye(f.dim()));
    let scale = k.iter().fold(0.0_f64, |m, &x| m.max(x.abs())).max(1e-30);
    let asym = (&k - &k.t()).iter().fold(0.0_f64, |m, &x| m.max(x.abs())) / scale;
    if asym > FXC_BLOCK_ASYMMETRY_TOL {
        return Err(FerricError::General(format!(
            "run_tddft: unrestricted f_xc block asymmetry {asym:e} exceeds the grid-noise \
             tolerance {FXC_BLOCK_ASYMMETRY_TOL:e}"
        )));
    }
    Ok(0.5 * (&k + &k.t()))
}

fn dense_matrices(
    method: TddftMethod,
    sp: &USpace,
    scf: &ScfResult,
    ri: &[RiTensors; 2],
    mix: ExchangeMix,
    fxc: Option<&UksOvFxc<'_>>,
) -> Result<DenseU, FerricError> {
    let (n, d) = (sp.dim(), sp.dims()[0]);
    let [oa, ob] = &sp.spins;
    let mut a = Array2::<f64>::zeros((n, n));
    let blocks = [
        build_a_matrix(scf.eps_a(), oa.nocc, oa.nvir, &ri[0], mix, 1.0),
        build_a_matrix(scf.eps_b(), ob.nocc, ob.nvir, &ri[1], mix, 1.0),
    ];
    fill_blocks(&mut a, blocks, ri, d);
    let mut b = (method == TddftMethod::Casida).then(|| {
        let mut b = Array2::<f64>::zeros((n, n));
        let blocks = [
            build_b_matrix(oa.nocc, oa.nvir, &ri[0], mix, 1.0),
            build_b_matrix(ob.nocc, ob.nvir, &ri[1], mix, 1.0),
        ];
        fill_blocks(&mut b, blocks, ri, d);
        b
    });
    if let Some(f) = fxc {
        let k = dense_fxc(f)?;
        a += &k;
        if let Some(b) = b.as_mut() {
            *b += &k;
        }
    }
    Ok(DenseU { a, b })
}

/// Roots of one solve plus, for `<S²>`, the `X` part of each root.
struct URoots {
    roots: Roots,
    x: Array2<f64>,
}

/// `X = ((X+Y) + (X−Y))/2` with `X−Y = (A+B)(X+Y)/Ω`, from `(A+B)(X+Y)`.
fn casida_x(xpy: &Array2<f64>, apb_xpy: &Array2<f64>, omega: &Array1<f64>) -> Array2<f64> {
    let mut x = xpy.clone();
    for (n, mut col) in x.columns_mut().into_iter().enumerate() {
        col += &(&apb_xpy.column(n) / omega[n]);
        col *= 0.5;
    }
    x
}

fn solve_dense_u(
    config: &TddftConfig,
    sp: &USpace,
    scf: &ScfResult,
    ri: &[RiTensors; 2],
    mix: ExchangeMix,
    fxc: Option<&UksOvFxc<'_>>,
    acc: &mut PhaseAccumulator,
) -> Result<URoots, FerricError> {
    let n_roots = config.n_roots.min(sp.dim());
    let m = acc.time("dense: A/B assembly", || {
        dense_matrices(config.method, sp, scf, ri, mix, fxc)
    })?;
    match &m.b {
        None => {
            let (w, x) = acc.time("dense: eigh", || solve_tda(&m.a))?;
            let (omega, amplitudes) = select_roots(&w, &x, n_roots);
            let residuals = tda_residuals(&m.a, &omega, &amplitudes);
            Ok(URoots {
                x: amplitudes.clone(),
                roots: Roots {
                    omega,
                    amplitudes,
                    residuals,
                    iterations: 0,
                },
            })
        }
        Some(b) => {
            let (w, xpy) = acc.time("dense: eigh", || solve_casida(&m.a, b))?;
            let (omega, amplitudes) = select_roots(&w, &xpy, n_roots);
            let residuals = casida_residuals(&m.a, b, &omega, &amplitudes);
            let x = casida_x(&amplitudes, &(&m.a + b).dot(&amplitudes), &omega);
            Ok(URoots {
                x,
                roots: Roots {
                    omega,
                    amplitudes,
                    residuals,
                    iterations: 0,
                },
            })
        }
    }
}

fn solve_davidson_u(
    config: &TddftConfig,
    op: &UOperator<'_>,
    acc: &mut PhaseAccumulator,
) -> Result<URoots, FerricError> {
    let diag = op.diag();
    let s = &config.davidson;
    let out = match config.method {
        TddftMethod::Tda => davidson_tda(|x| op.apply_a(x, acc), &diag, config.n_roots, s)?,
        TddftMethod::Casida => davidson_casida(|x| op.apply_pm(x, acc), &diag, config.n_roots, s)?,
    };
    let x = match config.method {
        TddftMethod::Tda => out.vectors.clone(),
        TddftMethod::Casida => {
            let (p, _) = op.apply_pm(&out.vectors, acc)?;
            casida_x(&out.vectors, &p, &out.eigenvalues)
        }
    };
    Ok(URoots {
        x,
        roots: Roots {
            omega: out.eigenvalues,
            amplitudes: out.vectors,
            residuals: out.residuals,
            iterations: out.iterations,
        },
    })
}

/// The dense plan: Casida holds the nine `dim²` matrices of the restricted
/// path (the same `solve_casida`); TDA holds `A` plus the largest of the
/// per-spin build scratch (`A_σσ` and the builder's own `2·dim_σ²`), the
/// f_xc identity and block (`2·dim²`, KS only) and the eigenvectors.
fn reserve_dense_u(plan: &mut MemoryPlan, method: TddftMethod, sp: &USpace, has_kernel: bool) {
    let n = sp.dim();
    let n2 = n.saturating_mul(n);
    let dmax = sp.dims()[0].max(sp.dims()[1]);
    match method {
        TddftMethod::Casida => reserve_dense_response(plan, method, n),
        TddftMethod::Tda => {
            plan.reserve("A (ia sigma, jb tau)", n2, Lifetime::Resident);
            plan.reserve(
                "per-spin A build",
                3usize.saturating_mul(dmax.saturating_mul(dmax)),
                Lifetime::Transient,
            );
            plan.reserve("eigh eigenvectors", n2, Lifetime::Transient);
        }
    }
    if has_kernel {
        plan.reserve(
            "f_xc identity and block",
            2usize.saturating_mul(n2),
            Lifetime::Transient,
        );
    }
}

/// Transition dipoles `<0|r|n> = Σ_σ Σ_ia (X+Y)_iaσ <iσ|r|aσ>` and
/// `f_n = (2/3) Ω_n |<0|r|n>|²` (no spin-adaptation factor: each spin is
/// its own block).
fn transition_properties(
    obs: &PreparedBasis,
    sp: &USpace,
    roots: &Roots,
) -> Result<(Vec<f64>, Vec<[f64; 3]>), FerricError> {
    let mu = oneelectron::dipole(obs, [0.0; 3])?;
    let d = sp.dims()[0];
    let mut tdips = vec![[0.0_f64; 3]; roots.omega.len()];
    for (o, off) in [(&sp.spins[0], 0), (&sp.spins[1], d)] {
        for (xyz, m) in mu.iter().enumerate() {
            let mu_ov = o.c_occ.t().dot(m).dot(&o.c_vir);
            let flat = Array1::from_iter(mu_ov.iter().copied());
            let amps = roots.amplitudes.slice(s![off..off + o.dim(), ..]);
            for (n, t) in tdips.iter_mut().enumerate() {
                t[xyz] += amps.column(n).dot(&flat);
            }
        }
    }
    let osc = tdips
        .iter()
        .zip(&roots.omega)
        .map(|(t, &w)| (2.0 / 3.0) * w * (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]))
        .collect();
    Ok((osc, tdips))
}

/// The spin-overlap data of `<S²>` (`book/src/19-tddft/06-unrestricted.md#unrestricted-each-root`):
/// `O = C_αᵀ S C_β` over
/// all MOs, with the reference occupations.
pub(crate) struct SpinOverlap {
    o: Array2<f64>,
    na: usize,
    nb: usize,
}

impl SpinOverlap {
    pub(crate) fn new(
        s_ao: &Array2<f64>,
        ca: &Array2<f64>,
        cb: &Array2<f64>,
        na: usize,
        nb: usize,
    ) -> Self {
        Self {
            o: ca.t().dot(s_ao).dot(cb),
            na,
            nb,
        }
    }

    fn m_terms(&self) -> f64 {
        let m = 0.5 * (self.na as f64 - self.nb as f64);
        m * m + m
    }

    /// `Σ_{S∈β occ} (OᵀO)_SS`: `n_β` when the α MOs span the β ones.
    fn beta_occ_weight(&self) -> f64 {
        let ob = self.o.slice(s![.., ..self.nb]);
        ob.iter().map(|v| v * v).sum()
    }

    /// `<S²>` of the reference determinant.
    pub(crate) fn reference(&self) -> f64 {
        let oo = self.o.slice(s![..self.na, ..self.nb]);
        self.m_terms() + self.beta_occ_weight() - oo.iter().map(|v| v * v).sum::<f64>()
    }

    /// `<S²>` of `Σ X_iaσ |Φ_iσ^aσ>` for the unit-normalized `x` (α pairs,
    /// then β pairs).
    pub(crate) fn excited(&self, x: ndarray::ArrayView1<f64>) -> f64 {
        let (na, nb) = (self.na, self.nb);
        let (nva, nvb) = (self.o.nrows() - na, self.o.ncols() - nb);
        let norm2 = x.dot(&x);
        let xa = Array2::from_shape_fn((na, nva), |(i, a)| x[i * nva + a] / norm2.sqrt());
        let xb =
            Array2::from_shape_fn((nb, nvb), |(i, a)| x[na * nva + i * nvb + a] / norm2.sqrt());
        let o = &self.o;
        // P = OᵀO (β × β), Q = O_{:,occβ} O_{:,occβ}ᵀ (α × α),
        // R = O_{occα,:}ᵀ O_{occα,:} (β × β).
        let p = o.t().dot(o);
        let q = o.slice(s![.., ..nb]).dot(&o.slice(s![.., ..nb]).t());
        let r = o.slice(s![..na, ..]).t().dot(&o.slice(s![..na, ..]));
        let (wa, wb) = (
            xa.iter().map(|v| v * v).sum::<f64>(),
            xb.iter().map(|v| v * v).sum::<f64>(),
        );
        let trace = |m: ndarray::ArrayView2<f64>| m.diag().sum();
        // tr(X M_vv Xᵀ) − tr(Xᵀ M_oo X) for X (nocc, nvir) and M over that spin.
        let ph = |xs: &Array2<f64>, m: &Array2<f64>, nocc: usize| {
            trace(xs.dot(&m.slice(s![nocc.., nocc..])).dot(&xs.t()).view())
                - trace(xs.t().dot(&m.slice(s![..nocc, ..nocc])).dot(xs).view())
        };
        let one_body = (wa + wb) * self.beta_occ_weight() + ph(&xb, &p, nb);
        let two_alpha = wa * trace(q.slice(s![..na, ..na])) + ph(&xa, &q, na);
        let two_beta = wb * trace(r.slice(s![..nb, ..nb])) + ph(&xb, &r, nb);
        // 2 Σ Xα_ia Xβ_JB O_iJ O_aB = 2 tr(Xαᵀ O_oo Xβ O_vvᵀ).
        let cross = 2.0
            * trace(
                xa.t()
                    .dot(&o.slice(s![..na, ..nb]))
                    .dot(&xb)
                    .dot(&o.slice(s![na.., nb..]).t())
                    .view(),
            );
        self.m_terms() + one_body - two_alpha - two_beta - cross
    }
}

/// [`crate::run_tddft`] on an unrestricted reference (already checked to be
/// converged).
pub(crate) fn run_unrestricted(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    scf: &ScfResult,
    config: &TddftConfig,
) -> Result<TddftResult, FerricError> {
    if config.spin != crate::ResponseSpin::Singlet {
        return Err(FerricError::General(
            "run_tddft: spin = triplet on an unrestricted reference. The UHF/UKS response is \
             spin-conserving and its roots are not spin-adapted, so there is no separate \
             triplet problem; run the default and read <S^2> of each root."
                .to_string(),
        ));
    }
    let mut acc = PhaseAccumulator::new().report_on_drop("TDDFT (unrestricted)");
    let (na, nb) = occupations(mol, scf)?;
    let (mix, xc_def) = resolve_response_xc(config.xc.as_deref(), config.c_hf_override)?;
    let sp = space(scf, na, nb)?;
    let solver = config
        .solver
        .resolve(sp.dim(), config.n_roots.min(sp.dim()));

    let label = match config.method {
        TddftMethod::Tda => "TDDFT/TDA (unrestricted)",
        TddftMethod::Casida => "TDDFT/Casida (unrestricted)",
    };
    let mut plan = MemoryPlan::resolve(config.memory_budget_bytes, label);
    match solver {
        TddftSolver::Dense => reserve_dense_u(&mut plan, config.method, &sp, xc_def.is_some()),
        _ => reserve_davidson(&mut plan, config, sp.dim()),
    }
    let ri = [
        build_response_tensors(obs, dfbs, &sp.spins[0], mix, config, &mut plan, &mut acc)?,
        build_response_tensors(obs, dfbs, &sp.spins[1], mix, config, &mut plan, &mut acc)?,
    ];
    let kernel = acc.time("f_xc kernel setup", || {
        xc_def
            .map(|def| {
                let db = scf.density_beta.as_ref().expect("checked in occupations");
                UksKernel::new(
                    mol,
                    obs.basis_set(),
                    def,
                    &config.grid,
                    &scf.density_alpha,
                    db,
                )
                .map_err(|e| {
                    FerricError::General(format!("run_tddft: f_xc kernel construction: {e}"))
                })
            })
            .transpose()
    })?;
    let fxc = kernel.as_ref().map(|k| UksOvFxc::new(k, sp.orbitals()));
    let ur = match solver {
        TddftSolver::Davidson => {
            let op = u_operator(&sp, scf, &ri, mix, fxc);
            solve_davidson_u(config, &op, &mut acc)?
        }
        _ => solve_dense_u(config, &sp, scf, &ri, mix, fxc.as_ref(), &mut acc)?,
    };
    let roots = &ur.roots;
    let (oscillator_strengths, transition_dipoles) = acc.time("oscillator strengths", || {
        transition_properties(obs, &sp, roots)
    })?;
    let so = SpinOverlap::new(
        &oneelectron::overlap(obs),
        scf.mos_a(),
        scf.mos_b(),
        sp.n_alpha,
        sp.n_beta,
    );
    let s2 = ur.x.columns().into_iter().map(|x| so.excited(x)).collect();

    Ok(TddftResult {
        excitation_energies: ur.roots.omega.to_vec(),
        oscillator_strengths,
        transition_dipoles,
        method: config.method,
        c_hf: mix.c_sr,
        c_lr: mix.c_lr,
        omega: mix.omega,
        spin: None,
        reference: ferric_scf::Spin::Unrestricted,
        s2,
        s2_reference: so.reference(),
        fxc_included: kernel.is_some(),
        solver,
        residuals: ur.roots.residuals,
        iterations: ur.roots.iterations,
    })
}

#[cfg(test)]
mod tests;
