//! Analytic stress tensor of the k-point RHF and UHF (Gamma-centred or
//! shifted meshes) on the dense pure-AFT J/K — the strain twin of
//! [`crate::kgrad`], built from the same phase-folded weights.
//!
//! # Convention (the Gamma stress's, [`crate::stress`])
//!
//! ```text
//! σ_ab = (1/Ω) dE/dε_ab,   a_i → (1 + ε) a_i,  R_A → (1 + ε) R_A,
//! G = Σ n_i b_i → (1 + ε)^{−T} G  (Miller index FIXED),  k = Σ f_i b_i (f_i FIXED)
//! ```
//!
//! # Why no "k strain" term
//!
//! A k-point is held at FIXED fractional coordinates, `k(ε) = (1 + ε)^{−T} k₀`
//! (exactly what `KPointMesh::gamma_centred(&cell.strained(ε), n)` builds from
//! the same integer numerators). The Bloch phase `e^{ik·L}` depends only on
//! those numerators and on the integer lattice coordinates of `L`, so it is
//! strain-INVARIANT, and so is the residue binning of the images. Every
//! per-k weight the forces build (`Mt(L)`, `Δ(L)`, `Yt_r`, ...) is therefore
//! reused unchanged: the stress multiplies the same per-image / per-triplet
//! derivative by the PAIR VECTOR instead of by a unit displacement, exactly
//! as the Gamma stress does with the Gamma force walks.
//!
//! # Terms (per cell; `D(k)`, `F(k)` fixed)
//!
//! 1. overlap / kinetic virial: `Σ_L Σ_mn w[r(L)]_{mn} ½(∂_bra − ∂_ket)⟨m|op|n_L⟩ (A − B − L)_b`,
//!    weights `w_s` (`M(k) = −W + c0 Z_tot D − v_M Σ_s D_s S D_s`) and `w_t`
//!    of [`crate::kgrad`].
//! 2. `V_SR`: the SR attraction walk's per-triplet `∂_A (A − X)_b + ∂_B′ (B′ − X)_b`
//!    with the force's image weight `Re Δ(L)`.
//! 3. `V_LR`: the full G sphere, `E_LR = −(1/Ω) Σ_G v_ω Re[S̄ ρ]`, with the
//!    RESIDUE-resolved pair-FT strain kernel
//!    ([`crate::pair_ft::residues::pair_ft_strain_residues_chunked`]); `−δ E_LR`
//!    volume term.
//! 4. `c0` volume `−δ c0 Z_tot N`, `N = (1/N_k) Σ_k tr[D(k) S(k)]`.
//! 5. Ewald `E_nn` ([`crate::ewald::ewald_nuclear_strain`]).

use crate::budget::{bytes_of, Ledger};
use crate::ewald::{default_ewald_omega, ewald_nuclear_strain, DEFAULT_EWALD_PRECISION};
use crate::grad::GRAD_NUCLEUS_EXPONENT;
use crate::hcore::kpoint::PeriodicHcoreK;
use crate::hcore::{
    gvector_list_bytes, hcore_pair_images, lr_gcut, sr_attraction_deriv_visit, PeriodicHcoreConfig,
    G_CHUNK_BYTES, ONE_E_ENGINE_PRECISION,
};
use crate::kgrad::{cr, fold_complex, fold_real};
use crate::kpts::{lattice_coords, KPointMesh};
use crate::lattice::Cell;
use crate::pair_ft::residues::{pair_ft_strain_residues_chunked, residue_coords, residue_index};
use crate::pair_ft::PairFtStrainTerms;
use crate::stress::{add_into, add_pair_virial, diag, Mat3, ZERO3};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ndarray::{Array2, Array3};
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// Deliberate defects for the mutation tests (`tests/pbc_kstress.rs`; the
/// Gamma [`crate::stress::StressMutation`] and [`crate::kgrad::KGradMutation`]
/// seams, same names where the defect is the same). Never set in production.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KStressMutation {
    /// Bloch phase dropped on every derivative weight (`Mt`, `Δ`, `Yt`; the
    /// energy-side `ρ` and `P^{k'}` keep theirs).
    NoPhase,
    /// `D_s(k)`, `F_s(k)` conjugated inside the stress. IDENTITY on
    /// TRIM-only meshes.
    ConjD,
    /// Exchange derivative pairs `k'` with `k = k' + q` instead of `k' − q`.
    /// IDENTITY on TRIM-only meshes.
    WrongQ,
    /// Missing `1/N_k` in the overlap / kinetic weights (blind at 1×1×1).
    NoInvNk,
    /// Primitive-cell Gamma `v_M` in the overlap term `−v_M Σ_s D_s S D_s`
    /// (blind for `exxdiv = none` and at 1×1×1).
    GammaMadelung,
    /// Primitive-cell `dv_M/dε` in the Madelung strain (blind for
    /// `exxdiv = none` and at 1×1×1).
    GammaMadelungStrain,
    /// Drop `Σ M dS` (overlap / Pulay term).
    NoPulay,
    /// Drop `c0 Z_tot D` from `M` (the c0 overlap part).
    NoC0Overlap,
    /// Drop `−v_M Σ_s D_s S D_s` from `M`.
    MadelungS,
    /// Drop `dv_M/dε` (exxdiv = ewald only).
    NoMadelung,
    /// Pair FT: drop the basis-centre (pair-vector) part of `dP`.
    FtNoCentres,
    /// G treated as unstrained: drop the G parts of `dP` and `dv`.
    GUnstrained,
    /// Drop the `−δ E_LR` volume terms (V_LR, J, K; diagonal only).
    NoVolume,
    /// Drop `−δ c0 Z_tot N` (diagonal only).
    NoC0Volume,
    /// SR / overlap virials with the image translations dropped from the
    /// pair vectors.
    NoSrImages,
    /// Drop the reciprocal part of the Ewald `E_nn` strain.
    NoEwaldLr,
    /// Drop the Ewald background volume term (diagonal only).
    NoEwaldBg,
}

/// Settings for the k-point stress entry points.
#[derive(Debug, Clone, Copy, Default)]
pub struct KStressConfig {
    /// Memory budget (bytes); `None` = ferric's unified budget.
    pub budget_bytes: Option<usize>,
    /// TEST ONLY: a deliberate defect ([`KStressMutation`]).
    #[doc(hidden)]
    pub mutation: Option<KStressMutation>,
    /// Gaussian-nucleus exponent for the SR attraction strain only (`None` =
    /// [`GRAD_NUCLEUS_EXPONENT`], never tighter than the energy's).
    pub nucleus_exponent: Option<f64>,
}

/// Bloch phases per `(k, residue)`: the energy's (`true_`) and the
/// derivative-side ones (`d`; all 1 under [`KStressMutation::NoPhase`]).
pub(crate) struct Phases {
    pub(crate) moduli: [usize; 3],
    pub(crate) true_: Vec<Vec<C64>>,
    pub(crate) d: Vec<Vec<C64>>,
}

impl Phases {
    pub(crate) fn new(mesh: &KPointMesh, mutation: Option<KStressMutation>) -> Self {
        let moduli = mesh.residue_moduli();
        let nr = moduli[0] * moduli[1] * moduli[2];
        let nk = mesh.nk();
        let true_: Vec<Vec<C64>> = (0..nk)
            .map(|k| {
                (0..nr)
                    .map(|r| mesh.phase(k, residue_coords(r, moduli)))
                    .collect()
            })
            .collect();
        let d = if mutation == Some(KStressMutation::NoPhase) {
            vec![vec![cr(1.0); nr]; nk]
        } else {
            true_.clone()
        };
        Self { moduli, true_, d }
    }
}

/// `Σ_r Σ_mn dP[r][ab]_mn(g) w[r]_nm` (the strain analogue of
/// `kgrad::residue_trace`).
pub(crate) fn residue_trace_d(
    dp: &[[Array3<C64>; 9]],
    ab: usize,
    gi: usize,
    w: &[Array2<C64>],
) -> C64 {
    let n = w[0].nrows();
    let mut acc = C64::new(0.0, 0.0);
    for (dr, wr) in dp.iter().zip(w) {
        let a = &dr[ab];
        for m in 0..n {
            for nn in 0..n {
                acc += a[[m, nn, gi]] * wr[(nn, m)];
            }
        }
    }
    acc
}

/// The one-electron and nuclear pieces of the k stress (`dE/dε`, Hartree per
/// cell, NOT divided by Ω).
#[derive(Debug, Clone, Default)]
pub struct KOneElectronStress {
    /// `Σ_k (1/N_k) tr[D dT]`.
    pub kinetic: Mat3,
    /// `Σ D dV_SR` (basis centres and nuclei).
    pub vsr: Mat3,
    /// `Σ D dV_LR`, including its `−δ E_LR` volume term.
    pub vlr: Mat3,
    /// `−δ c0 Z_tot N`.
    pub c0_volume: Mat3,
    /// Virial of each overlap weight set handed in (`Σ w dS`), in order.
    pub overlap: Vec<Mat3>,
    /// Ewald `E_nn`: real space, reciprocal (with volume), background.
    pub nn_sr: Mat3,
    pub nn_lr: Mat3,
    pub nn_background: Mat3,
    /// `Σ D V_LR` recomputed from the strain kernel's `P`.
    pub e_vlr: f64,
    /// `N = (1/N_k) Σ_k tr[D(k) S(k)]`.
    pub electrons: f64,
    /// Pair images, SR triplets, V_LR G count and pair-FT chunks.
    pub n_images: usize,
    pub n_sr_triplets: usize,
    pub n_g_lr: usize,
    pub n_chunks: usize,
}

/// Resolved ledger for the stress passes (per-k matrices and residue weights).
pub(crate) fn open_ledger(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    cfg: &KStressConfig,
    unrestricted: bool,
) -> Result<Ledger, FerricError> {
    let n = prep.nbasis();
    let nk = mesh.nk();
    let m = mesh.residue_moduli();
    let nr = m[0] * m[1] * m[2];
    let _ = cell;
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    let per_k = if unrestricted { 16 } else { 10 };
    ledger.reserve(
        &format!("k stress per-k complex n×n matrices (n = {n}, N_k = {nk})"),
        bytes_of((nk * n * n) as u64, 16 * per_k),
    )?;
    ledger.reserve(
        &format!("k stress residue-folded weights (R = {nr}, n = {n})"),
        bytes_of((nr * n * n) as u64, 16 * 2 + 8 * 6),
    )?;
    Ok(ledger)
}

/// The one-electron + nuclear strain pass at the total density `dtot(k)`.
///
/// `overlap_sets` are real per-residue weight sets (`w[r][m, n]`, block
/// `(m_0, n_L)`, `L ≡ r`, as [`fold_real`]) whose overlap virials `Σ w dS`
/// are returned in order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn one_electron_pass(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    dtot: &[Array2<C64>],
    ph: &Phases,
    overlap_sets: &[&[Array2<f64>]],
    cfg: &KStressConfig,
    ledger: &mut Ledger,
) -> Result<KOneElectronStress, FerricError> {
    let mutation = cfg.mutation;
    let is = |m: KStressMutation| mutation == Some(m);
    let n = prep.nbasis();
    let nk = mesh.nk();
    let natoms = cell.positions().len();
    let _ = natoms;
    let inv_nk = 1.0 / nk as f64;
    let moduli = ph.moduli;
    let b = cell.reciprocal();
    let nrm1 = if is(KStressMutation::NoInvNk) {
        1.0
    } else {
        inv_nk
    };
    let w_t = fold_real(dtot, &ph.d, nrm1);
    let w_d = fold_real(dtot, &ph.d, inv_nk);
    let delta = fold_complex(dtot, &ph.d, inv_nk);
    let delta_true = fold_complex(dtot, &ph.true_, inv_nk);
    let mut out = KOneElectronStress {
        overlap: vec![ZERO3; overlap_sets.len()],
        ..Default::default()
    };

    // --- N = (1/N_k) Σ_k Re tr[D(k) S(k)].
    let mut electrons = 0.0_f64;
    for (d, s) in dtot.iter().zip(&hk.s) {
        for m in 0..n {
            for nn in 0..n {
                electrons += (d[(m, nn)] * s[(nn, m)]).re;
            }
        }
    }
    electrons *= inv_nk;
    out.electrons = electrons;

    // --- dS, dT: per-image derivative blocks weighted by the pair vector.
    let images = hcore_pair_images(cell, prep, hcore_cfg.precision, ledger)?;
    let dims = prep.shell_dims().to_vec();
    let offs = prep.shell_offsets().to_vec();
    let centers: Vec<[f64; 3]> = prep.located_shells().iter().map(|s| s.center).collect();
    let nsh = prep.nshells();
    let drop_images = is(KStressMutation::NoSrImages);
    {
        let mut eng_s = Engine::new_1e_deriv(ffi::OP_OVERLAP, prep, ONE_E_ENGINE_PRECISION)?;
        let mut eng_t = Engine::new_1e_deriv(ffi::OP_KINETIC, prep, ONE_E_ENGINE_PRECISION)?;
        for l in &images {
            let r = residue_index(lattice_coords(&b, l), moduli);
            let li = if drop_images { [0.0; 3] } else { *l };
            for s1 in 0..nsh {
                for s2 in 0..nsh {
                    let (a, bc) = (centers[s1], centers[s2]);
                    let rel = [
                        a[0] - bc[0] - li[0],
                        a[1] - bc[1] - li[1],
                        a[2] - bc[2] - li[2],
                    ];
                    if let Some(blk) = eng_s.compute_1e_deriv_block_shifted(prep, s1, s2, *l)? {
                        for (acc, set) in out.overlap.iter_mut().zip(overlap_sets) {
                            add_pair_virial(acc, blk, &set[r], &dims, &offs, s1, s2, rel);
                        }
                    }
                    if let Some(blk) = eng_t.compute_1e_deriv_block_shifted(prep, s1, s2, *l)? {
                        add_pair_virial(&mut out.kinetic, blk, &w_t[r], &dims, &offs, s1, s2, rel);
                    }
                }
            }
        }
    }
    out.n_images = images.len();

    // --- V_SR: the force's image weight Re Δ(L), strain virial per triplet.
    let sr_cfg = PeriodicHcoreConfig {
        nucleus_exponent: cfg
            .nucleus_exponent
            .unwrap_or(GRAD_NUCLEUS_EXPONENT)
            .min(hcore_cfg.nucleus_exponent),
        ..*hcore_cfg
    };
    let mut vsr = ZERO3;
    out.n_sr_triplets = sr_attraction_deriv_visit(cell, prep, &sr_cfg, ledger, false, |t| {
        let wd = &w_d[residue_index(lattice_coords(&b, &t.l), moduli)];
        let (bb, xx) = if drop_images {
            (
                [
                    t.b_image[0] - t.l[0],
                    t.b_image[1] - t.l[1],
                    t.b_image[2] - t.l[2],
                ],
                [
                    t.nucleus[0] - t.m[0],
                    t.nucleus[1] - t.m[1],
                    t.nucleus[2] - t.m[2],
                ],
            )
        } else {
            (t.b_image, t.nucleus)
        };
        let ra = [
            t.a_center[0] - xx[0],
            t.a_center[1] - xx[1],
            t.a_center[2] - xx[2],
        ];
        let rb = [bb[0] - xx[0], bb[1] - xx[1], bb[2] - xx[2]];
        let mut ga = [0.0_f64; 3];
        let mut gb = [0.0_f64; 3];
        for i in 0..t.dim1 {
            for j in 0..t.dim2 {
                let coeff = t.f * wd[(t.off1 + i, t.off2 + j)];
                if coeff == 0.0 {
                    continue;
                }
                let idx = i * t.dim2 + j;
                let idx_swapped = j * t.dim1 + i;
                for c in 0..3 {
                    ga[c] += coeff * t.ket_a[c * t.bs + idx_swapped];
                    gb[c] += coeff * t.ket_b[c * t.bs + idx];
                }
            }
        }
        for a in 0..3 {
            for bq in 0..3 {
                vsr[a][bq] += ga[a] * ra[bq] + gb[a] * rb[bq];
            }
        }
    })?;
    out.vsr = vsr;

    // --- V_LR: full G sphere, residue pair-FT strain kernel.
    let zs = cell.nuclear_charges();
    let ztot: f64 = zs.iter().sum();
    let omega = hcore_cfg.omega;
    let vol = cell.volume();
    let pos = cell.positions();
    let ft_terms = PairFtStrainTerms {
        centres: !is(KStressMutation::FtNoCentres),
        g_shape: !is(KStressMutation::GUnstrained),
    };
    let gcut_lr = lr_gcut(prep, omega, hcore_cfg.precision);
    ledger.reserve(
        &format!("k stress V_LR G list (|G| <= {gcut_lr:.3})"),
        gvector_list_bytes(cell, gcut_lr)?,
    )?;
    let gv_lr: Vec<[f64; 3]> = cell
        .gvectors(gcut_lr)?
        .into_iter()
        .filter(|g| g[0] * g[0] + g[1] * g[1] + g[2] * g[2] > 0.0)
        .collect();
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    let mut e_vlr = 0.0_f64;
    let mut s_vlr = ZERO3;
    out.n_chunks = pair_ft_strain_residues_chunked(
        cell,
        prep,
        &gv_lr,
        moduli,
        0.1 * hcore_cfg.precision,
        chunk_budget,
        0,
        ft_terms,
        |_g0, gs, p, dp| {
            let fac = 1.0 / vol;
            for (gi, gvec) in gs.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let v = 4.0 * PI / g2 * (-g2 / (4.0 * omega * omega)).exp();
                let dvdg2 = -v * (1.0 / g2 + 1.0 / (4.0 * omega * omega));
                let (mut sre, mut sim) = (0.0_f64, 0.0_f64);
                for (z, r) in zs.iter().zip(&pos) {
                    let a = gvec[0] * r[0] + gvec[1] * r[1] + gvec[2] * r[2];
                    sre += z * a.cos();
                    sim -= z * a.sin();
                }
                let rho = crate::kgrad::residue_trace(p, gi, &delta_true);
                let phi = -(rho.re * sre + rho.im * sim);
                e_vlr += fac * v * phi;
                for a in 0..3 {
                    for bq in 0..3 {
                        let dz = residue_trace_d(dp, 3 * a + bq, gi, &delta);
                        let dv = if ft_terms.g_shape {
                            dvdg2 * (-2.0 * gvec[a] * gvec[bq])
                        } else {
                            0.0
                        };
                        let dphi = -(dz.re * sre + dz.im * sim);
                        s_vlr[a][bq] += fac * (dv * phi + v * dphi);
                    }
                }
            }
            Ok(())
        },
    )?;
    if !is(KStressMutation::NoVolume) {
        add_into(&mut s_vlr, &diag(-e_vlr), 1.0);
    }
    out.vlr = s_vlr;
    out.e_vlr = e_vlr;
    out.n_g_lr = gv_lr.len();

    // --- h's G = 0 bookkeeping: E_c0 = c0 Z_tot N ∝ 1/Ω.
    let c0 = PI / (omega * omega * vol);
    if !is(KStressMutation::NoC0Volume) {
        out.c0_volume = diag(-c0 * ztot * electrons);
    }

    // --- Ewald E_nn (the energy's ω; the result is ω-independent).
    let nn = ewald_nuclear_strain(cell, default_ewald_omega(cell), DEFAULT_EWALD_PRECISION)?;
    out.nn_sr = nn.sr;
    if !is(KStressMutation::NoEwaldLr) {
        out.nn_lr = nn.lr;
    }
    if !is(KStressMutation::NoEwaldBg) {
        out.nn_background = nn.background;
    }
    Ok(out)
}

impl KOneElectronStress {
    /// `dE/dε` of `(1/N_k) Σ_k tr[D h(k)] + E_nn` at FIXED `D` when the
    /// single overlap set handed to the pass was `c0 Z_tot D` (the `Σ D dS`
    /// part of `h`'s G = 0 term): kinetic + V_SR + V_LR + c0 (volume and
    /// overlap) + Ewald. The Pulay `−W dS` is NOT part of it.
    pub fn hcore_total(&self) -> Mat3 {
        let mut t = ZERO3;
        for m in [
            &self.kinetic,
            &self.vsr,
            &self.vlr,
            &self.c0_volume,
            &self.nn_sr,
            &self.nn_lr,
            &self.nn_background,
        ]
        .into_iter()
        .chain(self.overlap.iter())
        {
            add_into(&mut t, m, 1.0);
        }
        t
    }
}

/// TEST/DIAGNOSTIC: the one-electron + `E_nn` strain of the k-point energy at
/// a FIXED total density `D(k)` (`dtot`, one `n × n` matrix per mesh point),
/// i.e. `dE/dε` of `(1/N_k) Σ_k tr[D(k) h(k)] + E_nn` with `D` held fixed.
/// No SCF, no two-electron terms: the hcore gate that must pass before the
/// ERI terms are trusted. `hcore_total()` of the result is the quantity.
#[doc(hidden)]
pub fn kpoint_hcore_stress_fixed_d(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    dtot: &[Array2<C64>],
    cfg: &KStressConfig,
) -> Result<KOneElectronStress, FerricError> {
    if hk.v_ecp.is_some() {
        return Err(FerricError::General(
            "kpoint stress: periodic ECP strain at k is not implemented (the k energy supports \
             ECPs; the k forces and stress do not)"
                .into(),
        ));
    }
    let mut ledger = open_ledger(cell, prep, mesh, cfg, false)?;
    let ph = Phases::new(mesh, cfg.mutation);
    let omega = hcore_cfg.omega;
    let c0 = PI / (omega * omega * cell.volume());
    let ztot: f64 = cell.nuclear_charges().iter().sum();
    let w_c0 = fold_real(
        &dtot
            .iter()
            .map(|d| d.mapv(|z| z * c0 * ztot))
            .collect::<Vec<_>>(),
        &ph.d,
        if cfg.mutation == Some(KStressMutation::NoInvNk) {
            1.0
        } else {
            1.0 / mesh.nk() as f64
        },
    );
    one_electron_pass(
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        dtot,
        &ph,
        &[&w_c0],
        cfg,
        &mut ledger,
    )
}
