//! SCF with the one-electron matrices and exact J/K from an external AO
//! backend: NVIDIA cuEST on the GPU ([`solve_rhf_cuest`], [`solve_uhf_cuest`]).
//!
//! The SCF loop itself is ferric's injected path ([`solve_rhf_injected`],
//! [`solve_uhf_injected`]): the backend supplies `S`, `h = T + V` and
//! `J[D]`, `K[D]` (exact four-centre, no density fitting); XC for KS-DFT stays
//! on ferric's CPU grid. `F = h + J − ½·a·K + V_xc` (RKS) and the per-spin UKS
//! form are as documented on [`PeriodicInjection`].
//!
//! **AO order.** A backend computes in its own AO order. Before the SCF the
//! route computes S, T, V through the backend and through libint on a probe
//! copy of the molecule ([`probe_molecule`]: rotated, jittered, plus two H
//! atoms, so no symmetry hides a component), infers the backend→ferric map per
//! shell class (permutation and phase; [`ferric_cuest::layout::infer_map`]) and
//! refuses to run unless every class is a clean signed permutation. It then
//! checks that the molecule's own S, T, V, mapped, agree with libint to
//! [`ONE_ELECTRON_BAR`]. The same map then carries every density to the
//! backend and every J/K back. A wrong guess about cuEST's order therefore
//! cannot produce a wrong energy: it either calibrates away or refuses.
//! What calibration does NOT check is J/K; the energy comparison against
//! ferric's own exact J/K (tests/cuest_route.rs, scripts/cuest/run_gpu_check.sh)
//! does.
//!
//! Not supported here (each a hard error): ECPs, ghost atoms (rejected by the
//! backend), range-separated hybrids (the injected XC carries one global
//! exact-exchange fraction), and every config feature
//! [`validate_injected`](crate::rhf::validate_injected)
//! rejects (external potentials, solvation, symmetry, Newton/TRAH, ...).

use crate::fock::{JBuilder, KBuilder};
use crate::result::ScfResult;
use crate::rhf::{solve_rhf_injected, PeriodicInjection, RhfConfig, XcBuilder};
use crate::screening::SchwarzBounds;
use crate::uhf::solve_uhf_injected;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_cuest::layout::{classify, infer_map, max_abs_diff, AoLayout, AoMap, ClassVerdict};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::Array2;
use serde::Serialize;
use std::cell::RefCell;
use std::rc::Rc;

/// Inference tolerance: a class is a clean signed permutation when every
/// AO's residual and `||scale| − 1|` are below this.
pub const CALIBRATION_TOL: f64 = 1e-8;
/// Max |backend − libint| allowed on S, T and V after reordering. Both codes
/// are fp64 (~1e-12 here); an order, phase or normalization defect is ≥ 1e-3.
pub const ONE_ELECTRON_BAR: f64 = 1e-8;

/// An AO-integral backend, in its own AO order (described by `layout`).
pub trait AoBackend {
    fn name(&self) -> &str;
    fn layout(&self) -> &AoLayout;
    /// `[S, T, V]` in backend order, V including the sign of `−Z/r`.
    fn one_electron(&mut self) -> Result<[Array2<f64>; 3], FerricError>;
    /// `[S, T, V]` for another molecule in the same basis (the calibration
    /// probe, see [`probe_molecule`]), in backend order.
    fn one_electron_at(&mut self, mol: &Molecule) -> Result<[Array2<f64>; 3], FerricError>;
    /// `J[D]_pq = Σ_rs (pq|rs) D_rs`, backend order.
    fn coulomb(&mut self, d: &Array2<f64>) -> Result<Array2<f64>, FerricError>;
    /// `K[D]_pr = Σ_qs (pq|rs) D_qs`, backend order.
    fn exchange(&mut self, d: &Array2<f64>) -> Result<Array2<f64>, FerricError>;
    /// One line on where the backend spent its time.
    fn timing_summary(&self) -> String {
        String::new()
    }
}

/// What the AO-order calibration found.
#[derive(Debug, Clone, Serialize)]
pub struct Calibration {
    pub classes: Vec<ClassVerdict>,
    /// True when the inferred order equals `layout::hypothesis_components`.
    pub matches_hypothesis: bool,
    /// Max |backend − libint| on S, T, V after reordering with the inferred map.
    pub max_err: [f64; 3],
}

impl Calibration {
    /// One line for a log.
    pub fn summary(&self) -> String {
        format!(
            "AO order {} the built-in guess; mapped one-electron max|Δ| vs libint: \
             S {:.1e}, T {:.1e}, V {:.1e}",
            if self.matches_hypothesis {
                "matches"
            } else {
                "was inferred and DIFFERS from"
            },
            self.max_err[0],
            self.max_err[1],
            self.max_err[2]
        )
    }
}

/// An SCF run through a backend.
#[derive(Debug)]
pub struct BackendRun {
    pub result: ScfResult,
    pub calibration: Calibration,
    /// The backend's [`AoBackend::timing_summary`] after the SCF.
    pub timings: String,
}

/// The calibration probe: `mol` rotated to a generic orientation, each atom
/// nudged by a different small offset, and two H atoms added at generic points
/// outside it. AO conventions are per shell class, not per geometry, so the map
/// inferred here applies to `mol`; the probe only makes sure no symmetry zeroes
/// the s-column fingerprints the inference matches on (a planar molecule zeroes
/// every out-of-plane p against every s, and a lone atom has no other centre).
pub fn probe_molecule(mol: &Molecule) -> Molecule {
    let (a, b, c) = (0.3f64, 0.7f64, 1.1f64);
    let rz = |t: f64, v: [f64; 3]| {
        [
            t.cos() * v[0] - t.sin() * v[1],
            t.sin() * v[0] + t.cos() * v[1],
            v[2],
        ]
    };
    let ry = |t: f64, v: [f64; 3]| {
        [
            t.cos() * v[0] + t.sin() * v[2],
            v[1],
            -t.sin() * v[0] + t.cos() * v[2],
        ]
    };
    let rot = |v: [f64; 3]| rz(c, ry(b, rz(a, v)));
    let mut p = mol.clone();
    for (i, at) in p.atoms.iter_mut().enumerate() {
        let f = i as f64 + 1.0;
        let r = rot([at.x, at.y, at.zpos]);
        at.x = r[0] + 0.05 * (1.3 * f).sin();
        at.y = r[1] + 0.05 * (2.1 * f).cos();
        at.zpos = r[2] + 0.05 * (0.7 * f).sin();
    }
    let n = p.atoms.len() as f64;
    let cen = p.atoms.iter().fold([0.0; 3], |m, a| {
        [m[0] + a.x / n, m[1] + a.y / n, m[2] + a.zpos / n]
    });
    let radius = p
        .atoms
        .iter()
        .map(|a| {
            ((a.x - cen[0]).powi(2) + (a.y - cen[1]).powi(2) + (a.zpos - cen[2]).powi(2)).sqrt()
        })
        .fold(0.0, f64::max);
    let template = p.atoms[0].clone();
    for dir in [[0.62f64, 0.41, -0.67], [-0.33, 0.78, 0.53]] {
        let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        let d = radius + 2.0;
        let mut h = template.clone();
        h.symbol = "H".into();
        h.z = 1;
        h.ghost = false;
        h.n_core_ecp = 0;
        h.x = cen[0] + d * dir[0] / norm;
        h.y = cen[1] + d * dir[1] / norm;
        h.zpos = cen[2] + d * dir[2] / norm;
        p.atoms.push(h);
    }
    p
}

/// Infer the backend→ferric class map on the probe (`probe_layout`,
/// `[S, T, V]` both ways) and apply it to `layout`; refuse anything but clean
/// signed permutations. The molecule's own matrices are checked by the caller.
pub fn calibrate(
    probe_layout: &AoLayout,
    probe_backend: &[Array2<f64>; 3],
    probe_libint: &[Array2<f64>; 3],
    layout: &AoLayout,
) -> Result<(AoMap, Vec<ClassVerdict>), FerricError> {
    let b: [&Array2<f64>; 3] = [&probe_backend[0], &probe_backend[1], &probe_backend[2]];
    let f: [&Array2<f64>; 3] = [&probe_libint[0], &probe_libint[1], &probe_libint[2]];
    let inferred = infer_map(probe_layout, &b, &f);
    let classes = classify(probe_layout, &inferred, CALIBRATION_TOL);
    let bad: Vec<String> = classes
        .iter()
        .filter(|c| c.problem.is_some() || c.components.is_none())
        .map(|c| {
            format!(
                "l={} cuest_pure={} ferric_pure={}: {}",
                c.l,
                c.cuest_pure,
                c.ferric_pure,
                c.problem.as_deref().unwrap_or("no consistent map")
            )
        })
        .collect();
    if !bad.is_empty() {
        return Err(FerricError::General(format!(
            "cuEST AO-order calibration failed (the backend's one-electron matrices are not a \
             signed permutation of libint's): {}",
            bad.join("; ")
        )));
    }
    let missing = std::cell::Cell::new(None);
    let map = AoMap::from_components(layout, |sh| {
        let c = classes
            .iter()
            .find(|c| (c.l, c.cuest_pure, c.ferric_pure) == (sh.l, sh.cuest_pure, sh.ferric_pure))
            .and_then(|c| c.components.clone());
        c.unwrap_or_else(|| {
            missing.set(Some(sh.l));
            (0..sh.nfunc()).map(|k| (k, 1.0)).collect()
        })
    });
    if let Some(l) = missing.get() {
        return Err(FerricError::General(format!(
            "cuEST AO-order calibration: the probe has no l={l} shell of the molecule's kind"
        )));
    }
    Ok((map, classes))
}

type Shared<'a> = Rc<RefCell<Box<dyn AoBackend + 'a>>>;

struct BackendJ<'a> {
    b: Shared<'a>,
    map: Rc<AoMap>,
}

impl JBuilder for BackendJ<'_> {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError> {
        let jc = self.b.borrow_mut().coulomb(&self.map.to_cuest(d))?;
        j.assign(&self.map.to_ferric(&jc));
        Ok(0)
    }
    fn reset(&mut self) {}
}

struct BackendK<'a> {
    b: Shared<'a>,
    map: Rc<AoMap>,
}

impl KBuilder for BackendK<'_> {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        let kc = self.b.borrow_mut().exchange(&self.map.to_cuest(d))?;
        k.assign(&self.map.to_ferric(&kc));
        Ok(0)
    }
    fn update_density(&mut self, _d: &Array2<f64>) {}
    fn reset(&mut self) {}
}

struct ClosedXc(ferric_dft::ks::KsXc);

impl XcBuilder for ClosedXc {
    fn build(&mut self, d: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError> {
        use ferric_dft::xc_trait::XcContribution;
        let mut v = Array2::<f64>::zeros(d.dim());
        let e = self.0.add_xc(d, &mut v);
        Ok((e, v))
    }
    fn exact_exchange_fraction(&self) -> f64 {
        use ferric_dft::xc_trait::XcContribution;
        self.0.k_mix().sr
    }
}

struct OpenXc(Box<dyn ferric_dft::xc_trait::UksXcContribution>);

impl XcBuilder for OpenXc {
    fn build(&mut self, _d: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError> {
        Err(FerricError::General(
            "cuest_route: the UKS XC builder has no closed-shell build".into(),
        ))
    }
    fn exact_exchange_fraction(&self) -> f64 {
        self.0.k_mix().sr
    }
    fn build_polarized(
        &mut self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError> {
        let mut v_a = Array2::<f64>::zeros(d_a.dim());
        let mut v_b = Array2::<f64>::zeros(d_a.dim());
        let e = self.0.add_xc_uks(d_a, d_b, &mut v_a, &mut v_b);
        Ok((e, v_a, v_b))
    }
    fn supports_polarized(&self) -> bool {
        true
    }
}

fn refuse_range_separated(k: ferric_dft::xc_trait::KMix, name: &str) -> Result<(), FerricError> {
    if k.omega != 0.0 || k.sr != k.lr {
        return Err(FerricError::General(format!(
            "the cuEST SCF route does not run range-separated hybrids yet ({name}: ω = {}, \
             sr = {}, lr = {}); use a pure or global-hybrid functional",
            k.omega, k.sr, k.lr
        )));
    }
    Ok(())
}

/// The config the injected solver accepts, plus the XC builder taken out of it.
///
/// J/K selections (`k_builder`, `df_j_aux`, `df_k_aux`) are cleared: the
/// backend's exact J/K replace them, which the caller reports. A SAD guess is
/// computed here on the CPU and passed as `init_guess_density` (the injected
/// path cannot project an atomic guess itself).
fn injected_config<'x>(
    mol: &Molecule,
    prep: &PreparedBasis,
    config: &RhfConfig,
    open_shell: bool,
) -> Result<(RhfConfig, Option<Box<dyn XcBuilder + 'x>>), FerricError> {
    let xc: Option<Box<dyn XcBuilder + 'x>> = match config.xc.as_deref() {
        None => None,
        Some(name) if open_shell => {
            let mut x =
                crate::driver::uks_xc_contribution(mol, prep, config)?.expect("config.xc is set");
            crate::driver::commit_xc(Some(&mut *x))?;
            refuse_range_separated(x.k_mix(), name)?;
            Some(Box::new(OpenXc(x)))
        }
        Some(name) => {
            let x = crate::rhf::molecular_ks_xc(mol, prep.basis_set(), config, false)?
                .expect("config.xc is set");
            {
                use ferric_dft::xc_trait::XcContribution;
                refuse_range_separated(x.k_mix(), name)?;
            }
            Some(Box::new(ClosedXc(x)))
        }
    };
    let mut cfg = config.clone();
    cfg.xc = None;
    cfg.k_builder = None;
    cfg.df_j_aux = None;
    cfg.df_k_aux = None;
    if cfg.use_sad_guess && cfg.init_guess_density.is_none() {
        cfg.init_guess_density = Some(crate::guess::sad_guess(mol, prep, prep.basis_set())?);
    }
    cfg.use_sad_guess = false;
    Ok((cfg, xc))
}

/// `(S, h, map, calibration)` in ferric order from the backend's one-electron
/// matrices, with the AO order calibrated against libint on the probe.
fn one_electron(
    backend: &mut dyn AoBackend,
    mol: &Molecule,
    prep: &PreparedBasis,
) -> Result<(Array2<f64>, Array2<f64>, AoMap, Calibration), FerricError> {
    use ferric_integrals::oneelectron as oe;
    let libint_of = |p: &PreparedBasis| [oe::overlap(p), oe::kinetic(p), oe::nuclear(p)];
    if backend.layout().nao != prep.nbasis() {
        return Err(FerricError::General(format!(
            "{} has {} AOs, ferric's basis {}",
            backend.name(),
            backend.layout().nao,
            prep.nbasis()
        )));
    }
    let probe = probe_molecule(mol);
    let probe_prep = PreparedBasis::new(&probe, prep.basis_set())?;
    let probe_layout = AoLayout::new(&probe, prep.basis_set())
        .map_err(|e| FerricError::General(format!("cuEST probe layout: {e}")))?;
    let probe_b = backend.one_electron_at(&probe)?;
    let (map, classes) = calibrate(
        &probe_layout,
        &probe_b,
        &libint_of(&probe_prep),
        backend.layout(),
    )?;

    let raw = backend.one_electron()?;
    let libint = libint_of(prep);
    let max_err = [0, 1, 2].map(|i| max_abs_diff(&map.to_ferric(&raw[i]), &libint[i]));
    let cal = Calibration {
        matches_hypothesis: classes.iter().all(|c| c.matches_hypothesis),
        classes,
        max_err,
    };
    if max_err.iter().any(|e| !(*e < ONE_ELECTRON_BAR)) {
        return Err(FerricError::General(format!(
            "cuEST one-electron matrices disagree with libint after reordering (bar \
             {ONE_ELECTRON_BAR:.0e}): {}",
            cal.summary()
        )));
    }
    let s = map.to_ferric(&raw[0]);
    let h = map.to_ferric(&raw[1]) + map.to_ferric(&raw[2]);
    Ok((s, h, map, cal))
}

fn injection<'a>(
    backend: Box<dyn AoBackend + 'a>,
    mol: &Molecule,
    prep: &PreparedBasis,
    xc: Option<Box<dyn XcBuilder + 'a>>,
) -> Result<(PeriodicInjection<'a>, Calibration, Shared<'a>), FerricError> {
    let shared: Shared<'a> = Rc::new(RefCell::new(backend));
    let (s, h, map, cal) = one_electron(&mut **shared.borrow_mut(), mol, prep)?;
    let map = Rc::new(map);
    let inj = PeriodicInjection {
        s,
        h,
        vnn: mol.nuclear_repulsion(),
        j: Box::new(BackendJ {
            b: shared.clone(),
            map: map.clone(),
        }),
        k: Box::new(BackendK {
            b: shared.clone(),
            map,
        }),
        xc,
    };
    Ok((inj, cal, shared))
}

/// Closed-shell RHF/RKS (`config.xc`) on `backend`'s S, h, J and K.
pub fn solve_rhf_backend<'a>(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    backend: Box<dyn AoBackend + 'a>,
) -> Result<BackendRun, FerricError> {
    let (cfg, xc) = injected_config(mol, prep, config, false)?;
    let (inj, calibration, shared) = injection(backend, mol, prep, xc)?;
    let result = solve_rhf_injected(ctx, mol, prep, op, bounds, &cfg, inj)?;
    let timings = shared.borrow().timing_summary();
    Ok(BackendRun {
        result,
        calibration,
        timings,
    })
}

/// UHF/UKS (`config.xc`) on `backend`'s S, h, J and K.
pub fn solve_uhf_backend<'a>(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    backend: Box<dyn AoBackend + 'a>,
) -> Result<BackendRun, FerricError> {
    let (cfg, xc) = injected_config(mol, prep, config, true)?;
    let (inj, calibration, shared) = injection(backend, mol, prep, xc)?;
    let result = solve_uhf_injected(ctx, mol, prep, bounds, &cfg, inj, None)?;
    let timings = shared.borrow().timing_summary();
    Ok(BackendRun {
        result,
        calibration,
        timings,
    })
}

/// Whether this build links cuEST.
pub fn cuest_available() -> bool {
    cfg!(feature = "cuest")
}

#[cfg(not(feature = "cuest"))]
fn device_backend<'a>(
    _mol: &Molecule,
    _prep: &PreparedBasis,
) -> Result<Box<dyn AoBackend + 'a>, FerricError> {
    Err(FerricError::General(
        "this ferric was built without cuEST; rebuild with the `cuest` feature \
         (e.g. `cargo build --release -p ferric-cli --features cuest`, with CUEST_ROOT set; \
         see crates/ferric-cuest/README.md)"
            .into(),
    ))
}

#[cfg(feature = "cuest")]
mod device {
    use super::AoBackend;
    use ferric_core::FerricError;
    use ferric_cuest::context::{ContextOptions, CuestContext};
    use ferric_cuest::layout::AoLayout;
    use ndarray::Array2;

    pub(super) struct CuestDevice(
        pub(super) CuestContext,
        pub(super) ferric_core::basis::BasisSet,
    );

    impl AoBackend for CuestDevice {
        fn name(&self) -> &str {
            "cuEST"
        }
        fn layout(&self) -> &AoLayout {
            &self.0.layout
        }
        fn one_electron(&mut self) -> Result<[Array2<f64>; 3], FerricError> {
            Ok(self.0.one_electron()?)
        }
        fn one_electron_at(
            &mut self,
            mol: &ferric_core::mol::Molecule,
        ) -> Result<[Array2<f64>; 3], FerricError> {
            let m = ferric_cuest::oe::compute_one_electron(mol, &self.1)?;
            Ok([m.s, m.t, m.v])
        }
        fn coulomb(&mut self, d: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
            Ok(self.0.coulomb(d)?)
        }
        fn exchange(&mut self, d: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
            Ok(self.0.exchange(d)?)
        }
        fn timing_summary(&self) -> String {
            let t = self.0.timings;
            format!(
                "cuEST time: setup {:.3}s, one-electron {:.3}s, J {:.3}s ({} builds), \
                 K {:.3}s ({} builds)",
                t.setup, t.one_electron, t.coulomb, t.n_coulomb, t.exchange, t.n_exchange
            )
        }
    }

    pub(super) fn options() -> ContextOptions {
        ContextOptions::default()
    }
}

#[cfg(feature = "cuest")]
fn device_backend<'a>(
    mol: &Molecule,
    prep: &PreparedBasis,
) -> Result<Box<dyn AoBackend + 'a>, FerricError> {
    let ctx = ferric_cuest::context::CuestContext::new(mol, prep.basis_set(), device::options())?;
    Ok(Box::new(device::CuestDevice(ctx, prep.basis_set().clone())))
}

/// Closed-shell RHF/RKS with S, h, J and K from cuEST on GPU 0.
/// Errors when the build lacks the `cuest` feature or the device is unsupported.
pub fn solve_rhf_cuest(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
) -> Result<BackendRun, FerricError> {
    let backend = device_backend(mol, prep)?;
    solve_rhf_backend(ctx, mol, prep, op, bounds, config, backend)
}

/// UHF/UKS with S, h, J and K from cuEST on GPU 0.
pub fn solve_uhf_cuest(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
) -> Result<BackendRun, FerricError> {
    let backend = device_backend(mol, prep)?;
    solve_uhf_backend(ctx, mol, prep, bounds, config, backend)
}
