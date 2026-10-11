//! Analytic ROHF Hessian: the orbital response (term 5) and the assembly of
//! the full matrix (issue #291). [`rohf_hessian`] is the public entry point;
//! `rhf_hessian`/`uhf_hessian` refuse ROHF results, and `frequencies`
//! dispatches ROHF here. A semicanonicalized result (`Spin::Unrestricted` with
//! `rohf_spin_focks` set, which `uhf_hessian` refuses) is mapped back to ROHF
//! orbitals first ([`rohf_from_semicanonical`]).
//!
//! # Setup
//!
//! One orbital set `C` (columns closed `c` | open `o` | virtual `v`), spin
//! occupation vectors `n_α = (1_c, 1_o, 0_v)`, `n_β = (1_c, 0_o, 0_v)`,
//! `D_σ = C n_σ Cᵀ`, spin Focks `F_σ = h + J(D_α+D_β) − K(D_σ)`, MO Focks
//! `f_σ = Cᵀ F_σ C` (NOT diagonal: only the Roothaan combination below is
//! zero at convergence). ROHF stationarity is
//!
//! ```text
//!   g_vc = (f_α + f_β)_vc = 0,   g_vo = (f_α)_vo = 0,   g_oc = (f_β)_oc = 0.
//! ```
//!
//! The energy gradient (what `gradient::rohf_gradient` evaluates) is, with
//! `X^x = ∂X/∂x`,
//!
//! ```text
//!   E^x = V_nn^x + Σ_σ Tr[D_σ h^x] + ½ Σ_σ Tr[D_σ G_σ^x(D)] − Tr[W S^x],
//!   G_σ^x(D) = J^x(D_α+D_β) − K^x(D_σ),   W = sym(D_α F_α D_α + D_α F_β D_β).
//! ```
//!
//! # The Hessian
//!
//! Differentiating `E^x` along `y` (everything is a function of `D_α, D_β`
//! and the integrals; the two `G^x` halves are equal by the 8-fold symmetry of
//! the integrals) gives
//!
//! ```text
//!   H_xy = [V_nn + Σ D h'' + Γ(D)·(μν|λσ)'' − Tr[W S'']]_xy      (skeleton, terms 1-4)
//!        + Σ_σ Tr[D_σ^y F_σ^x(D fixed)]  −  Tr[W^y S^x]           (response, term 5)
//!   F_σ^x(D fixed) = h^x + G_σ^x(D),
//!   W^y = sym( D_α^y F_α D_α + D_α F_α^y D_α + D_α F_α D_α^y
//!            + D_α^y F_β D_β + D_α F_β^y D_β + D_α F_β D_β^y ),
//!   F_σ^y = F_σ^y(D fixed) + G_σ(D^y)  (total derivative; G_σ(D) = J(D_α+D_β) − K(D_σ)).
//! ```
//!
//! This is NOT the UHF `assemble_response`: that form eliminates `W^y` using
//! spin-diagonal MO Focks (`ε_ij^x = F^x_ij + G(D^x)_ij − ½S_ij(ε_i+ε_j)`),
//! which a single ROHF orbital set with two different Focks does not have.
//! Here `W^y` is built from the AO densities and Focks directly.
//!
//! # First-order orbitals and the densities
//!
//! `C^y = C U^y` with `U = −½ S^y_MO + κ`, `S^y_MO = Cᵀ S^y C`, `κ` real
//! antisymmetric. The symmetric part is fixed by orthonormality
//! (`U + Uᵀ + S^y_MO = 0`); the closed-closed, open-open and virtual-virtual
//! antisymmetric rotations leave `D_α, D_β` unchanged (the `α` occupied space
//! is `c+o`, the `β` one is `c`), so `κ` has only the three physical blocks
//! `κ_vc, κ_vo, κ_oc`. Then, in the MO basis,
//!
//! ```text
//!   D_σ^y = U n_σ + n_σ Uᵀ = D_σ^S + D_σ^κ,
//!   (D_σ^S)_pq = −½ S^y_pq (n_σp + n_σq),     (D_σ^κ)_pq = (n_σq − n_σp) κ_pq.
//! ```
//!
//! Zero blocks by symmetry: `D_α^y` has no `vv` block; `D_β^y` has no `oo`,
//! `ov`, `vv` block (the open orbitals are β-unoccupied); `D^κ_α` has no `cc`,
//! `oo`, `oc` or `vv` block (they cancel in `κ n − n κ`: both indices occupied
//! in α, or both empty), `D^κ_β` is non-zero only in `vc` and `oc`. The overlap part `D^S_σ` is non-zero in every occupied-
//! occupied block and in the occupied-virtual blocks (half weight): it is the
//! generalisation of RHF/UHF's `D_oo^x` (there the `−½S_vo` part is absorbed
//! into `U_vo`; here it is kept explicit so that `κ` is the pure rotation).
//!
//! # The CPHF equation (derivation)
//!
//! The condition `g = 0` holds along the whole family of solutions, so
//! `d g/dy = 0`. With `d f_σ/dy = Uᵀ f_σ + f_σ U + Cᵀ(F_σ^y(fixed) + G_σ(D^y))C`
//! and `Uᵀ f + f U = −½(S f + f S) + [f, κ]`:
//!
//! ```text
//!   Φ_σ(κ) = Cᵀ F_σ^y(fixed) C + Cᵀ G_σ(D^S) C − ½(S f_σ + f_σ S)   (right-hand side B)
//!          + Cᵀ G_σ(D^κ) C + [f_σ, κ]                               (operator A κ)
//!   (Φ_α+Φ_β)_vc = 0,   (Φ_α)_vo = 0,   (Φ_β)_oc = 0     ⇒   A κ = −B.
//! ```
//!
//! `A` is the commutator-form orbital Hessian (half the true Hessian, the
//! normalisation of `rohf_newton::hessian_matvec`); its `[f, κ]` has the
//! off-diagonal terms (`vc↔vo` through `(f_α)_oc`, `vo↔oc` through `(f_β)_vo`
//! ...) that the diagonal-gap form of `hessian_matvec` omits (measured by
//! `tests/rohf_hessian_matvec_fd.rs`). It is solved per nuclear coordinate by
//! the same Jacobi-preconditioned CG as the UHF response, in the stacked
//! `(κ_vc, κ_vo, κ_oc)` vector.
//!
//! # What is detected, and what is not
//!
//! A typed error is returned when a CG search direction has non-positive
//! curvature `pᵀAp ≤ 0`, when CG does not converge, or when a solved rotation
//! exceeds `KAPPA_MAX` (a nearly singular mode that the nuclear displacement
//! drives). These are properties of the directions the nuclear perturbations
//! generate, NOT a stability analysis: a saddle point of the ROHF energy whose
//! unstable orbital mode is not driven by any nuclear displacement (for example
//! one of a different symmetry from every perturbation) returns the valid
//! Hessian of that stationary state with no warning. Use the ROHF stability
//! analysis, not this function, to establish that the state is a minimum.
//!
//! # Trust in the Focks
//!
//! The response uses the spin Focks REBUILT from the final densities
//! (`h + G_σ(D)`), and `W` is rebuilt from them too, so the Hessian is a
//! function of the final densities alone; the stored `rohf_spin_focks` (from
//! the last SCF iteration's density) are only compared with them and refused
//! if they differ by more than `FOCK_TRUST_TOL` (a ROKS, COSX or otherwise
//! different Hamiltonian). COSX-converged results, external potentials
//! (point charges, smeared charges, a field) and ECPs are refused: no ROHF
//! Hessian validation covers them.
//!
//! # Assembly
//!
//! For each `y` the solved `κ^y` gives `D_σ^y`, `G_σ(D^y)` (one `J/K` pair),
//! `F_σ^y`, `W^y`, and the whole column `H[·, y]` of the response
//! `Σ_σ Tr[D^y_σ F^x_σ] − Tr[W^y S^x]` follows by contraction with every
//! first-order AO matrix `F^x_σ`, `S^x`. The result is symmetric to the CPHF
//! tolerance; the unsymmetrised asymmetry is reported as in RHF/UHF.

use super::{
    add_extra_first_order, extra_charges, first_order_ao, max_abs, parts_from_terms, pcg,
    rohf_support::rohf_skeleton_parts, uhf_occupations, uhf_spin_densities_for, FirstOrderAo,
    FockSpec, Reference, ResponseTerm, RhfHessianParts, TwoElectronBuilder, OCC_CONSISTENCY_TOL,
};
use crate::result::{ScfResult, Spin};
use crate::rhf::RhfConfig;
use crate::screening::SchwarzBounds;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};
use std::ops::Range;

/// Deliberate defects of the response, for mutation tests only
/// ([`rohf_hessian_parts_mutated`]); production uses [`RohfMutation::None`].
/// Reachable only through `hessian::rohf_support` (hidden, not API).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RohfMutation {
    /// The correct response.
    #[default]
    None,
    /// `κ_oc` forced to zero (the closed-open rotation is dropped).
    DropOcBlock,
    /// The operator carries the diagonal Fock gap `f_pp − f_qq` instead of the
    /// commutator `[f, κ]` (the form of `rohf_newton::hessian_matvec`).
    DiagonalGapOperator,
    /// `−Tr[W^y S^x]` omitted.
    DropWOverlap,
    /// `F_α^y` used in the `D_α F_β^y D_β` term of `W^y`.
    OneSpinFockInW,
    /// `D_α F_β D_β → D_β F_β D_β` in `W` (and its derivative): an IDENTITY at
    /// convergence, because the difference is the closed-open block of `F_β`,
    /// which the `oc` stationarity condition keeps zero along every
    /// perturbation. Recorded as blind, not counted as a mutant.
    DbFbDbInW,
    /// `F_α^x` used for both spins (response contraction and right-hand side).
    OneSpinFockX,
    /// `F_α^x ↔ F_β^x` swapped (response contraction and right-hand side).
    SwapSpinFockX,
    /// `G(D^S)` (the overlap part of the density change) dropped from the
    /// right-hand side AND from `G(D^y)` in `F^y` (so `W^y` sees it too).
    DropOverlapDensityG,
    /// Not a defect: the stored `rohf_spin_focks` instead of the Focks rebuilt
    /// from the final densities (used to measure the effect of the SCF
    /// tolerance on the stored ones).
    UseStoredFocks,
    /// Not a defect: the rotation bound lowered to 0.1, to show the bound is
    /// wired to a typed error (no real system exceeds the production bound).
    ProbeKappaBound,
    /// Sign of `−½(S f + f S)` in the right-hand side flipped.
    FlipOverlapFockSign,
}

/// Largest first-order rotation `|κ|` (radians per Bohr of nuclear displacement)
/// accepted. Measured on the validation systems: <= 0.75 (NH2 0.74, CH2, HO2, OH 0.45).
/// A gap `Δ` in the orbital Hessian gives `κ ~ F^x/Δ` with `|F^x| ~ 0.5`, so
/// `KAPPA_MAX = 10` flags effective gaps below ~0.05 Ha, where the harmonic
/// (linear-response) picture needs displacements far below the FD step anyway.
const KAPPA_MAX: f64 = 10.0;

/// Largest `max|F_stored − F_rebuilt|` (Ha) for which the stored spin Focks are
/// trusted to be those of the final densities.
const FOCK_TRUST_TOL: f64 = 1e-4;

/// Row/column ranges of the closed, open and virtual MOs.
struct Blocks {
    c: Range<usize>,
    o: Range<usize>,
    v: Range<usize>,
}

/// The converged ROHF state the response is built on.
struct Frame {
    c: Array2<f64>,
    blocks: Blocks,
    nmo: usize,
    /// Spin occupation vectors over the MOs: α, β.
    n: [Array1<f64>; 2],
    /// Converged AO spin Focks and spin densities.
    f_ao: [Array2<f64>; 2],
    d_ao: [Array2<f64>; 2],
    /// MO spin Focks `Cᵀ F_σ C`.
    f_mo: [Array2<f64>; 2],
}

impl Frame {
    fn new(
        mol: &Molecule,
        rohf: &ScfResult,
        focks: [Array2<f64>; 2],
    ) -> Result<Frame, FerricError> {
        let fail = |what: &str| {
            FerricError::General(format!(
                "analytic ROHF Hessian: {what} (the response needs closed|open|virtual \
                 ordered orbitals of an aufbau ROHF)"
            ))
        };
        let [na, nb] = uhf_occupations(mol);
        let c = rohf.mos_alpha.clone();
        let nmo = c.ncols();
        if na < nb || na > nmo {
            return Err(fail("occupations do not fit the MO count"));
        }
        let (da, db) = uhf_spin_densities_for(Reference::Rohf, rohf)?;
        let [fa, fb] = &focks;
        let occ_density = |k: usize| {
            let co = c.slice(ndarray::s![.., ..k]);
            co.dot(&co.t())
        };
        if max_abs(&(occ_density(na) - da)) > OCC_CONSISTENCY_TOL
            || max_abs(&(occ_density(nb) - db)) > OCC_CONSISTENCY_TOL
        {
            return Err(fail("the densities are not the first MOs' projectors"));
        }
        let occ = |k: usize| Array1::from_shape_fn(nmo, |p| f64::from(p < k));
        let to_mo = |m: &Array2<f64>| c.t().dot(m).dot(&c);
        Ok(Frame {
            blocks: Blocks {
                c: 0..nb,
                o: nb..na,
                v: na..nmo,
            },
            nmo,
            n: [occ(na), occ(nb)],
            f_mo: [to_mo(fa), to_mo(fb)],
            f_ao: [fa.clone(), fb.clone()],
            d_ao: [da.clone(), db.clone()],
            c,
        })
    }

    fn to_mo(&self, m: &Array2<f64>) -> Array2<f64> {
        self.c.t().dot(m).dot(&self.c)
    }

    fn to_ao(&self, m: &Array2<f64>) -> Array2<f64> {
        self.c.dot(m).dot(&self.c.t())
    }

    fn sizes(&self) -> [usize; 3] {
        let (nc, no, nv) = (
            self.blocks.c.len(),
            self.blocks.o.len(),
            self.blocks.v.len(),
        );
        [nv * nc, nv * no, no * nc]
    }

    /// Roothaan projection of a pair of per-spin MO matrices onto the stacked
    /// `(vc, vo, oc)` column: `vc = α+β`, `vo = α`, `oc = β`.
    fn project(&self, ma: &Array2<f64>, mb: &Array2<f64>) -> Array2<f64> {
        let b = &self.blocks;
        let mut v = Vec::with_capacity(self.sizes().iter().sum());
        for p in b.v.clone() {
            v.extend(b.c.clone().map(|q| ma[(p, q)] + mb[(p, q)]));
        }
        for p in b.v.clone() {
            v.extend(b.o.clone().map(|q| ma[(p, q)]));
        }
        for p in b.o.clone() {
            v.extend(b.c.clone().map(|q| mb[(p, q)]));
        }
        let n = v.len();
        Array2::from_shape_vec((n, 1), v).expect("a Vec of length n forms an (n, 1) column")
    }

    /// The antisymmetric `nmo × nmo` rotation of a stacked `(vc, vo, oc)` column.
    fn kappa_matrix(&self, x: &Array2<f64>) -> Array2<f64> {
        let b = &self.blocks;
        let mut k = Array2::<f64>::zeros((self.nmo, self.nmo));
        let mut it = x.iter().copied();
        let mut put = |rows: &Range<usize>, cols: &Range<usize>| {
            for p in rows.clone() {
                for q in cols.clone() {
                    let val = it.next().expect("stacked rotation has the block sizes");
                    k[(p, q)] = val;
                    k[(q, p)] = -val;
                }
            }
        };
        put(&b.v, &b.c);
        put(&b.v, &b.o);
        put(&b.o, &b.c);
        k
    }

    /// Zero the `oc` segment (the `DropOcBlock` mutant).
    fn drop_oc(&self, x: &mut Array2<f64>) {
        let [vc, vo, _] = self.sizes();
        x.slice_mut(ndarray::s![vc + vo.., ..]).fill(0.0);
    }

    /// Per-spin density change of a rotation in the MO basis,
    /// `(n_σq − n_σp) κ_pq`.
    fn kappa_densities(&self, kappa: &Array2<f64>) -> [Array2<f64>; 2] {
        [0, 1].map(|s| {
            Array2::from_shape_fn(kappa.dim(), |(p, q)| {
                (self.n[s][q] - self.n[s][p]) * kappa[(p, q)]
            })
        })
    }

    /// Preconditioner: per-block Fock gap of the spin(s) that change occupation,
    /// floored away from zero (a preconditioner only, never the operator).
    fn precond(&self) -> Array2<f64> {
        let b = &self.blocks;
        let (fa, fb) = (&self.f_mo[0], &self.f_mo[1]);
        let gap = |p: usize, q: usize, sa: bool, sb: bool| {
            let mut g = 0.0;
            if sa {
                g += fa[(p, p)] - fa[(q, q)];
            }
            if sb {
                g += fb[(p, p)] - fb[(q, q)];
            }
            g.abs().max(0.05)
        };
        let mut v = Vec::new();
        for p in b.v.clone() {
            v.extend(b.c.clone().map(|q| gap(p, q, true, true)));
        }
        for p in b.v.clone() {
            v.extend(b.o.clone().map(|q| gap(p, q, true, false)));
        }
        for p in b.o.clone() {
            v.extend(b.c.clone().map(|q| gap(p, q, false, true)));
        }
        let n = v.len();
        Array2::from_shape_vec((n, 1), v).expect("a Vec of length n forms an (n, 1) column")
    }
}

/// `[f, κ] = f κ − κ f`, or its diagonal-gap truncation (a mutant).
fn commutator(f: &Array2<f64>, k: &Array2<f64>, diagonal_only: bool) -> Array2<f64> {
    if diagonal_only {
        Array2::from_shape_fn(k.dim(), |(p, q)| (f[(p, p)] - f[(q, q)]) * k[(p, q)])
    } else {
        f.dot(k) - k.dot(f)
    }
}

/// The per-spin first-order Fock `F_σ^x(D fixed)` seen through a mutation.
fn fx(first: &FirstOrderAo, spin: usize, x: usize, mutation: RohfMutation) -> &Array2<f64> {
    let s = match mutation {
        RohfMutation::OneSpinFockX => 0,
        RohfMutation::SwapSpinFockX => 1 - spin,
        _ => spin,
    };
    &first.f1[s][x]
}

struct Response<'a> {
    frame: &'a Frame,
    g: &'a TwoElectronBuilder<'a>,
    first: &'a FirstOrderAo,
    mutation: RohfMutation,
}

impl Response<'_> {
    /// `A κ`: the J/K response of `D^κ` plus the Fock commutator, projected.
    fn apply(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        let fr = self.frame;
        let mut x = x.clone();
        if self.mutation == RohfMutation::DropOcBlock {
            fr.drop_oc(&mut x);
        }
        let kappa = fr.kappa_matrix(&x);
        let dk = fr.kappa_densities(&kappa).map(|m| fr.to_ao(&m));
        let v = self.g.g_uhf(&dk)?;
        let diag = self.mutation == RohfMutation::DiagonalGapOperator;
        let m = [0, 1].map(|s| fr.to_mo(&v[s]) + commutator(&fr.f_mo[s], &kappa, diag));
        let mut out = fr.project(&m[0], &m[1]);
        if self.mutation == RohfMutation::DropOcBlock {
            fr.drop_oc(&mut out);
        }
        Ok(out)
    }

    /// `S^x_MO` and the AO overlap-part densities `D^S_σ`.
    fn overlap_densities(&self, x: usize) -> (Array2<f64>, [Array2<f64>; 2]) {
        let fr = self.frame;
        let s_mo = fr.to_mo(&self.first.s1[x]);
        let ds = [0, 1].map(|s| {
            let n = &fr.n[s];
            fr.to_ao(&Array2::from_shape_fn(s_mo.dim(), |(p, q)| {
                -0.5 * s_mo[(p, q)] * (n[p] + n[q])
            }))
        });
        (s_mo, ds)
    }

    /// The solve right-hand side `−B` for perturbation `x`.
    fn rhs(&self, x: usize) -> Result<Array2<f64>, FerricError> {
        let fr = self.frame;
        let (s_mo, ds) = self.overlap_densities(x);
        let gs = if self.mutation == RohfMutation::DropOverlapDensityG {
            [0, 1].map(|_| Array2::zeros(ds[0].dim()))
        } else {
            self.g.g_uhf(&ds)?
        };
        let sign = if self.mutation == RohfMutation::FlipOverlapFockSign {
            -1.0
        } else {
            1.0
        };
        let phi = [0, 1].map(|s| {
            let f = &fr.f_mo[s];
            fr.to_mo(fx(self.first, s, x, self.mutation)) + fr.to_mo(&gs[s])
                - sign * 0.5 * (s_mo.dot(f) + f.dot(&s_mo))
        });
        let mut rhs = -fr.project(&phi[0], &phi[1]);
        if self.mutation == RohfMutation::DropOcBlock {
            fr.drop_oc(&mut rhs);
        }
        Ok(rhs)
    }

    /// Column `H[·, y]` of the response and the CPHF statistics.
    fn column(&self, y: usize) -> Result<(Array1<f64>, usize, f64), FerricError> {
        let fr = self.frame;
        let rhs = self.rhs(y)?;
        let (kvec, iterations, residual) = pcg(|v| self.apply(v), &rhs, &fr.precond())?;
        let mut kvec = kvec;
        if self.mutation == RohfMutation::DropOcBlock {
            fr.drop_oc(&mut kvec);
        }
        let kmax = max_abs(&kvec);
        let bound = if self.mutation == RohfMutation::ProbeKappaBound {
            0.1
        } else {
            KAPPA_MAX
        };
        if kmax > bound {
            return Err(FerricError::Convergence(format!(
                "first-order orbital rotation is not small (max |κ| = {kmax:.3e} > {bound}): \
                 the orbital Hessian is nearly singular along a mode the nuclear displacement \
                 drives (a near-degenerate open shell or a near-instability); the Hessian of \
                 this state is not meaningful"
            )));
        }
        let kappa = fr.kappa_matrix(&kvec);
        let (s_mo, _) = self.overlap_densities(y);
        let u = &kappa - &(0.5 * &s_mo);
        let dy = [0, 1].map(|s| {
            let n = &fr.n[s];
            fr.to_ao(&Array2::from_shape_fn(u.dim(), |(p, q)| {
                u[(p, q)] * n[q] + n[p] * u[(q, p)]
            }))
        });
        let gy = if self.mutation == RohfMutation::DropOverlapDensityG {
            let dk = fr.kappa_densities(&kappa).map(|m| fr.to_ao(&m));
            self.g.g_uhf(&dk)?
        } else {
            self.g.g_uhf(&dy)?
        };
        let fy = [0, 1].map(|s| &self.first.f1[s][y] + &gy[s]);
        let w = self.w_response(&dy, &fy);

        let n3 = self.first.s1.len();
        let col = Array1::from_shape_fn(n3, |x| {
            let resp: f64 = (0..2)
                .map(|s| (&dy[s] * fx(self.first, s, x, self.mutation)).sum())
                .sum();
            let ovl = if self.mutation == RohfMutation::DropWOverlap {
                0.0
            } else {
                (&w * &self.first.s1[x]).sum()
            };
            resp - ovl
        });
        Ok((col, iterations, residual))
    }

    /// `W^y` (unsymmetrised; contracted with the symmetric `S^x` only).
    fn w_response(&self, dy: &[Array2<f64>; 2], fy: &[Array2<f64>; 2]) -> Array2<f64> {
        let fr = self.frame;
        let (da, db) = (&fr.d_ao[0], &fr.d_ao[1]);
        let (fa, fb) = (&fr.f_ao[0], &fr.f_ao[1]);
        let tri = |a: &Array2<f64>, b: &Array2<f64>, c: &Array2<f64>| a.dot(b).dot(c);
        let fy_b = if self.mutation == RohfMutation::OneSpinFockInW {
            &fy[0]
        } else {
            &fy[1]
        };
        let first = tri(&dy[0], fa, da) + tri(da, &fy[0], da) + tri(da, fa, &dy[0]);
        if self.mutation == RohfMutation::DbFbDbInW {
            return first + tri(&dy[1], fb, db) + tri(db, &fy[1], db) + tri(db, fb, &dy[1]);
        }
        first + tri(&dy[0], fb, db) + tri(da, fy_b, db) + tri(da, fb, &dy[1])
    }
}

/// A `Spin::Unrestricted` result that kept the ROHF spin Focks: the output of
/// [`crate::semicanonical::unrestricted_reference`].
fn is_semicanonicalized(res: &ScfResult) -> bool {
    res.spin == Spin::Unrestricted && res.rohf_spin_focks.is_some()
}

/// Rebuild the ROHF result (one orbital set `closed | open | virtual`) from a
/// semicanonicalized one. The block-diagonal rotations of semicanonicalization
/// keep the spans: the closed orbitals are the occupied β orbitals, the
/// virtuals are the virtual α orbitals, and the open ones are the part of the
/// occupied α span S-orthogonal to the closed span. The Hessian depends on the
/// orbitals only through these spans (it is invariant to rotations within each
/// block), so the particular open orbitals are immaterial.
fn rohf_from_semicanonical(
    mol: &Molecule,
    prep: &PreparedBasis,
    semi: &ScfResult,
) -> Result<ScfResult, FerricError> {
    let fail = |what: &str| {
        FerricError::General(format!(
            "analytic ROHF Hessian: cannot map a semicanonicalized result back to ROHF              orbitals: {what}"
        ))
    };
    let cb = semi
        .mos_beta
        .as_ref()
        .ok_or_else(|| fail("no β orbitals"))?;
    let ca = &semi.mos_alpha;
    let [na, nb] = uhf_occupations(mol);
    if na < nb || na > ca.ncols() || nb > cb.ncols() || ca.ncols() != cb.ncols() {
        return Err(fail("occupations do not fit the MO count"));
    }
    let (nmo, no) = (ca.ncols(), na - nb);
    let s = ferric_integrals::oneelectron::overlap(prep);
    let c_closed = cb.slice(ndarray::s![.., ..nb]).to_owned();
    let c_occ_a = ca.slice(ndarray::s![.., ..na]).to_owned();
    // The α-occupied span with its closed component removed.
    let x = &c_occ_a - &c_closed.dot(&c_closed.t().dot(&s).dot(&c_occ_a));
    let gram = x.t().dot(&s).dot(&x);
    let (evals, evecs) = gram
        .eigh(UPLO::Lower)
        .map_err(|e| fail(&format!("overlap eigendecomposition failed: {e}")))?;
    let mut c = Array2::<f64>::zeros((c_closed.nrows(), nmo));
    c.slice_mut(ndarray::s![.., ..nb]).assign(&c_closed);
    for k in 0..no {
        // The top `no` eigenvalues (ascending order) span the open orbitals.
        let idx = na - no + k;
        let lam = evals[idx];
        if lam < 1e-6 {
            return Err(fail(&format!(
                "the α-occupied span has no {no} orbitals outside the closed span \
                 (eigenvalue {lam:.1e}); the result does not come from an aufbau ROHF"
            )));
        }
        let v = evecs.column(idx).to_owned();
        c.column_mut(nb + k).assign(&(x.dot(&v) / lam.sqrt()));
    }
    c.slice_mut(ndarray::s![.., na..])
        .assign(&ca.slice(ndarray::s![.., na..]));
    let mut out = semi.clone();
    out.spin = Spin::RestrictedOpen;
    out.mos_alpha = c;
    out.mos_beta = None;
    out.eps_beta = None;
    Ok(out)
}

/// `W = sym(D_α F_α D_α + D_α F_β D_β)` from the given Focks.
fn fock_energy_weighted_density(
    da: &Array2<f64>,
    db: &Array2<f64>,
    f: &[Array2<f64>; 2],
) -> Array2<f64> {
    let w = da.dot(&f[0]).dot(da) + da.dot(&f[1]).dot(db);
    0.5 * (&w + &w.t())
}

/// The spin Focks `h + J(D) − K(D_σ)` rebuilt from the final densities. The
/// stored `rohf_spin_focks` are accepted only if they agree with these to
/// [`FOCK_TRUST_TOL`] (they are refused for a result converged with another
/// Hamiltonian: ROKS, COSX, a different operator, ...), and the REBUILT ones
/// are used in the response (the stored ones come from the last SCF
/// iteration's density, which differs from the final one by the SCF
/// tolerance).
fn trusted_focks(
    g: &TwoElectronBuilder<'_>,
    prep: &PreparedBasis,
    rohf: &ScfResult,
    dens: &super::rohf_support::RohfSkeletonDensities,
    mutation: RohfMutation,
) -> Result<[Array2<f64>; 2], FerricError> {
    let (sa, sb) = rohf
        .rohf_spin_focks
        .as_ref()
        .ok_or_else(|| FerricError::General("analytic ROHF Hessian: no spin Focks".into()))?;
    let h = ferric_integrals::oneelectron::hcore(prep);
    let v = g.g_uhf(&[dens.d_alpha.clone(), dens.d_beta.clone()])?;
    let rebuilt = [&h + &v[0], &h + &v[1]];
    let dev = max_abs(&(&rebuilt[0] - sa)).max(max_abs(&(&rebuilt[1] - sb)));
    if dev > FOCK_TRUST_TOL {
        return Err(FerricError::General(format!(
            "analytic ROHF Hessian: the stored spin Focks differ from h + G(D) of the final \
             densities by {dev:.3e} (> {FOCK_TRUST_TOL:.0e}); the result was not converged with \
             the exact-exchange Hamiltonian this Hessian differentiates"
        )));
    }
    Ok(if mutation == RohfMutation::UseStoredFocks {
        [sa.clone(), sb.clone()]
    } else {
        rebuilt
    })
}

/// The full ROHF Hessian, (3N, 3N) Hartree/Bohr², from a converged
/// [`crate::rohf::solve_rohf`] result (or its semicanonicalized form).
///
/// `bounds` is the Schwarz table for `op` on `prep`; `config` is the
/// configuration the SCF was converged with, read to refuse what the Hessian
/// does not implement: Kohn-Sham, RI, COSX, solvation, an external potential,
/// an ECP, non-Coulomb operators. See `hessian::rohf` (module doc) for the
/// equations and [`analytic_rohf_hessian_available`](super::analytic_rohf_hessian_available)
/// for the pre-SCF probe.
pub fn rohf_hessian(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rohf: &ScfResult,
    config: &RhfConfig,
) -> Result<Array2<f64>, FerricError> {
    Ok(rohf_hessian_parts(ctx, mol, prep, op, bounds, rohf, config)?.total())
}

/// [`rohf_hessian`] with the per-term breakdown and CPHF diagnostics.
pub fn rohf_hessian_parts(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rohf: &ScfResult,
    config: &RhfConfig,
) -> Result<RhfHessianParts, FerricError> {
    rohf_hessian_parts_mutated(ctx, mol, prep, op, bounds, rohf, config, RohfMutation::None)
}

/// [`rohf_hessian_parts`] with a deliberate defect in the response (mutation
/// tests); [`RohfMutation::None`] is the production path.
#[allow(clippy::too_many_arguments)]
pub fn rohf_hessian_parts_mutated(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rohf: &ScfResult,
    config: &RhfConfig,
    mutation: RohfMutation,
) -> Result<RhfHessianParts, FerricError> {
    let mapped;
    let rohf = if is_semicanonicalized(rohf) {
        mapped = rohf_from_semicanonical(mol, prep, rohf)?;
        &mapped
    } else {
        rohf
    };
    let mut skel = rohf_skeleton_parts(mol, prep, op, bounds, rohf, config)?;
    let g = TwoElectronBuilder::new(ctx, prep, bounds)?;
    let dens = &skel.densities;
    let focks = trusted_focks(&g, prep, rohf, dens, mutation)?;
    if mutation != RohfMutation::UseStoredFocks {
        // W of the rebuilt Focks, so the whole Hessian is a function of the
        // final densities alone.
        let w = fock_energy_weighted_density(&dens.d_alpha, &dens.d_beta, &focks);
        skel.overlap = super::skeleton_hess_overlap(prep, mol.atoms.len(), &w)?;
    }
    let frame = Frame::new(mol, rohf, focks)?;
    let natoms = mol.atoms.len();
    let dens = &skel.densities;
    let specs = [&dens.d_alpha, &dens.d_beta].map(|ds| FockSpec {
        coulomb: &dens.d_total,
        exchange: ds,
        k_scale: 1.0,
    });
    let mut first = first_order_ao(prep, op, bounds, natoms, extra_charges(config), &specs)?;
    add_extra_first_order(&mut first, mol, prep, config)?;

    let resp = Response {
        frame: &frame,
        g: &g,
        first: &first,
        mutation,
    };
    let n3 = first.s1.len();
    let mut raw = Array2::<f64>::zeros((n3, n3));
    let (mut iterations, mut max_residual) = (Vec::with_capacity(n3), 0.0f64);
    for y in 0..n3 {
        let (col, its, res) = resp.column(y).map_err(|e| super::at_coordinate(y, e))?;
        raw.column_mut(y).assign(&col);
        iterations.push(its);
        max_residual = max_residual.max(res);
    }
    let term = ResponseTerm {
        raw,
        iterations,
        max_residual,
    };
    Ok(parts_from_terms(
        [
            skel.nuclear,
            skel.one_electron,
            skel.overlap,
            skel.two_electron,
        ],
        term,
    ))
}
