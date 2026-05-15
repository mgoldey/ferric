//! # RI-Laplace MP2
//!
//! This module implements the Resolution of Identity (RI) Laplace-transform MP2 method.
//!
//! ## Theory
//! The canonical MP2 correlation energy is given by:
//! $$E_{corr} = -\sum_{iajb} \frac{(ia|jb)[2(ia|jb) - (ib|ja)]}{\epsilon_a + \epsilon_b - \epsilon_i - \epsilon_j}$$
//!
//! Using the Laplace transform identity for the denominator:
//! $$\frac{1}{x} = \int_0^\infty e^{-tx} dt \approx \sum_k w_k e^{-t_k x}$$
//!
//! The correlation energy can be expressed as an integral over the Laplace parameter $t$.
//! In the RI approximation, the energy factorizes into Coulomb ($J$) and Exchange ($K$)
//! traces that can be evaluated efficiently in either the MO or AO basis.
//!
//! ## Implementation
//! This module provides two implementations:
//! 1. **MO-based (`compute_mo`)**: Transforms the 3-center integrals to the MO basis.
//!    This is $O(N^4)$ and used primarily for validating the quadrature convergence.
//! 2. **AO/MO hybrid (`compute_ao`)**: J term in AO basis via pseudo-densities (supports
//!    future sparse path); K term in MO basis via Gram matrix (cheaper than AO Gram).
//!
//! ## Reference
//! Häser & Almlöf, Chem. Phys. Lett. 191, 299 (1992).
//! Takatsuka, Ten-no, Hackbusch, J. Chem. Phys. 129, 044112 (2008).

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfResult;
use ndarray::{Array1, Array2};

/// Row-sparse representation of a B^P slice (nbas × nbas matrix).
///
/// For each row μ, stores only the column indices and values with |B^P_{μν}| > threshold.
/// Enables O(nnz) sparse-dense matrix products M^P = B^P @ P(t) when B^P is sparse.
struct SparseBSlice {
    /// For each row μ: (col_indices, values)
    rows: Vec<(Vec<u16>, Vec<f64>)>,
    nbas: usize,
}

impl SparseBSlice {
    fn from_dense(b: &Array2<f64>, thresh: f64) -> Self {
        let nbas = b.nrows();
        let rows = (0..nbas).map(|mu| {
            let row = b.row(mu);
            let mut cols = Vec::new();
            let mut vals = Vec::new();
            for (nu, &v) in row.iter().enumerate() {
                if v.abs() > thresh {
                    cols.push(nu as u16);
                    vals.push(v);
                }
            }
            (cols, vals)
        }).collect();
        Self { rows, nbas }
    }

    /// Compute M = self @ rhs (dense nbas×nbas), return as flattened row-major Vec.
    /// Output M[μ,ν] = Σ_{σ∈nnz(row μ)} B^P_{μσ} rhs_{σν}.
    fn mat_mul_flat(&self, rhs: &Array2<f64>) -> Vec<f64> {
        let nbas = self.nbas;
        let mut out = vec![0.0f64; nbas * nbas];
        for (mu, (cols, vals)) in self.rows.iter().enumerate() {
            let out_row = &mut out[mu * nbas..(mu + 1) * nbas];
            for (&nu, &b_val) in cols.iter().zip(vals.iter()) {
                let rhs_row = rhs.row(nu as usize);
                for (&r, o) in rhs_row.iter().zip(out_row.iter_mut()) {
                    *o += b_val * r;
                }
            }
        }
        out
    }
}

pub struct LaplaceMp2Result {
    pub total_energy: f64,
    pub mp2_corr: f64,
    pub e_os: f64,
    pub e_ss: f64,
}

pub fn laplace_ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &RhfResult,
    n_quad: usize,
    frozen_core: usize,
) -> Result<LaplaceMp2Result, FerricError> {
    let mut laplace = LaplaceMp2::new(n_quad);
    let (mp2_corr, e_os, e_ss) = laplace.compute_ao(mol, obs, dfbs, op, rhf, frozen_core)?;
    Ok(LaplaceMp2Result {
        total_energy: rhf.energy + mp2_corr,
        mp2_corr,
        e_os,
        e_ss,
    })
}

/// Laplace-transform MP2 energy builder.
pub struct LaplaceMp2 {
    pub n_quad: usize,
    pub points: Vec<f64>,
    pub weights: Vec<f64>,
}

use rayon::prelude::*;

impl LaplaceMp2 {
    /// Create a Laplace-MP2 builder from orbital energies.
    ///
    /// Selects minimax quadrature points for the range [ymin, ymax] where
    /// ymin = 2*(LUMO - HOMO) and ymax = 2*(eps_max - eps_min).
    /// Points are from Takatsuka, Ten-no, Hackbusch (JCP 129, 044112, 2008)
    /// via the Helmich-Paris laplace-minimax library.
    pub fn new(n_quad: usize) -> Self {
        // Default: will be reinitialized by compute() using actual orbital energies.
        LaplaceMp2 { n_quad, points: vec![], weights: vec![] }
    }

    /// Initialize quadrature for orbital energy range [ymin, ymax].
    ///
    /// ymin = 2*(LUMO - HOMO), ymax = 2*(eps_max_vir - eps_min_occ).
    /// The exponents (t) and weights (w) approximate 1/x on [ymin, ymax] as
    /// 1/x ≈ Σ_k w_k exp(-t_k x). Points are scaled: t_actual = t_table / ymin,
    /// w_actual = w_table / ymin.
    fn init_quadrature(&mut self, ymin: f64, ymax: f64) {
        let r = ymax / ymin;
        let (raw_t, raw_w) = select_minimax_points(self.n_quad, r);
        self.points = raw_t.iter().map(|&t| t / ymin).collect();
        self.weights = raw_w.iter().map(|&w| w / ymin).collect();
    }

    /// Compute the MP2 energy using MO-based RI-Laplace transform.
    ///
    /// This method transforms the 3-center integrals to the MO basis and is
    /// useful for verifying the accuracy of the Laplace quadrature against
    /// canonical RI-MP2 results.
    pub fn compute_mo(
        &mut self,
        mol: &Molecule,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
        rhf: &RhfResult,
        frozen_core: usize,
    ) -> Result<f64, FerricError> {
        let nbas = obs.nbasis();
        let nocc_total = mol.nelec() as usize / 2;
        let nocc = nocc_total - frozen_core;
        let nvir = nbas - nocc_total;
        let naux = dfbs.nbasis();

        let eps = &rhf.orbital_energies;
        let nmo = eps.len();
        let ymin = 2.0 * (eps[nocc_total] - eps[nocc_total - 1]);
        let ymax = 2.0 * (eps[nmo - 1] - eps[0]);
        self.init_quadrature(ymin, ymax);

        let c = &rhf.mos;
        let c_occ = c.slice(ndarray::s![.., frozen_core..nocc_total]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

        // 1. Get (P|ia) RI amplitudes
        let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
        let v_inv_sqrt = crate::rimp2::cholesky_inverse_sqrt(&v2c)?;
        let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;
        let eri3_mo = crate::mo_transform::transform_3center_ov(&eri3_ao, &c_occ, &c_vir);
        let b_flat = v_inv_sqrt.dot(&eri3_mo.into_shape_with_order((naux, nocc * nvir)).unwrap());

        // 2. Parallel quadrature over points
        let e_corr: f64 = self.points.par_iter().zip(self.weights.par_iter()).map(|(&t, &w)| {
            // Weighted amplitudes: B_ia(t) = B_ia * exp(-t * (eps_a - eps_i) / 2)
            let mut b_t = b_flat.clone();
            for i in 0..nocc {
                let e_i = eps[frozen_core + i];
                for a in 0..nvir {
                    let e_a = eps[nocc_total + a];
                    let factor = (-0.5 * t * (e_a - e_i)).exp();
                    for p in 0..naux {
                        b_t[(p, i * nvir + a)] *= factor;
                    }
                }
            }

            // J_PQ = sum_{ia} B_ia^P B_ia^Q
            let j_mat = b_t.dot(&b_t.t());
            let e_coul = j_mat.iter().map(|&x| x * x).sum::<f64>();

            // Exchange part: sum_{Pi, Qj} G_{Pi, Qj} G_{Pj, Qi}
            let b_reshape = b_t.into_shape_with_order((naux, nocc, nvir)).unwrap();
            let mut g = Array2::<f64>::zeros((naux * nocc, naux * nocc));
            for a in 0..nvir {
                for i in 0..nocc {
                    for p in 0..naux {
                        let val_pi = b_reshape[(p, i, a)];
                        for j in 0..nocc {
                            for q in 0..naux {
                                g[(p * nocc + i, q * nocc + j)] += val_pi * b_reshape[(q, j, a)];
                            }
                        }
                    }
                }
            }

            let mut e_exch: f64 = 0.0;
            for p in 0..naux {
                for i in 0..nocc {
                    for q in 0..naux {
                        for j in 0..nocc {
                            let g_pi_qj: f64 = g[(p * nocc + i, q * nocc + j)];
                            let g_pj_qi: f64 = g[(p * nocc + j, q * nocc + i)];
                            e_exch += g_pi_qj * g_pj_qi;
                        }
                    }
                }
            }

            -w * (2.0 * e_coul - e_exch)
        }).sum();

        Ok(e_corr)
    }

    /// Compute the MP2 energy using a hybrid AO/MO Laplace-transform approach.
    ///
    /// - J term: AO pseudo-density formulation — O(naux × nbas²) per quad point.
    ///   Foundation for future sparse path when P(t), Q(t) become localized.
    /// - K term: MO Gram matrix — O(naux² × nocc² × nvir), much cheaper than
    ///   the AO exchange Gram O(naux × nbas⁴).
    pub fn compute_ao(
        &mut self,
        mol: &Molecule,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
        rhf: &RhfResult,
        frozen_core: usize,
    ) -> Result<(f64, f64, f64), FerricError> {
        let nbas = obs.nbasis();
        let nocc_total = mol.nelec() as usize / 2;
        let nocc = nocc_total - frozen_core;
        let nvir = nbas - nocc_total;
        let naux = dfbs.nbasis();

        let eps = &rhf.orbital_energies;
        let nmo = eps.len();
        let ymin = 2.0 * (eps[nocc_total] - eps[nocc_total - 1]);
        let ymax = 2.0 * (eps[nmo - 1] - eps[0]);
        self.init_quadrature(ymin, ymax);

        // Build RI-fitted 3-center integrals: b_ao[P, μ, ν] = Σ_Q V^{-1/2}_{PQ} (Q|μν)
        let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
        let v_inv_sqrt = crate::rimp2::cholesky_inverse_sqrt(&v2c)?;
        let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;
        let eri3_flat = eri3_ao.into_shape_with_order((naux, nbas * nbas)).unwrap();
        let b_flat_ao = v_inv_sqrt.dot(&eri3_flat);
        let b_ao = b_flat_ao.into_shape_with_order((naux, nbas, nbas)).unwrap();

        let c = &rhf.mos;
        let c_occ = c.slice(ndarray::s![.., frozen_core..nocc_total]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

        // Build sparse B^P slices for AO J computation.
        // Threshold 1e-12 retains all numerically significant RI integrals.
        // With localized orbitals and diffuse bases this will be much sparser.
        // Sparse B^P representation: threshold small elements of each B^P slice.
        // sto-3g decane: ~32% fill at 1e-12 (14% at 1e-6 with <1e-6 Ha error).
        // True linear scaling requires sparse P(t)/Q(t), which needs localized MOs.
        let b_sparse: Vec<SparseBSlice> = (0..naux)
            .map(|p| SparseBSlice::from_dense(&b_ao.slice(ndarray::s![p, .., ..]).to_owned(), 1e-12))
            .collect();

        // MO-basis integrals for K: b_mo[P, i*nvir+a] = (P|ia)
        let eri3_mo = crate::mo_transform::transform_3center_ov(&b_ao, &c_occ, &c_vir);
        let b_mo_flat = eri3_mo.into_shape_with_order((naux, nocc * nvir)).unwrap();

        // Parallel over quadrature points — each point is independent.
        // Inner BLAS calls use multithreaded DGEMM; no nested rayon.
        let (e_os, e_ss): (f64, f64) = self.points.par_iter().zip(self.weights.par_iter()).map(|(&t, &w)| {
            // --- J term in AO basis (sparse B^P path) ---
            let pt = build_pseudo_density_occ(c, eps, t, nocc, frozen_core);
            let qt = build_pseudo_density_vir(c, eps, t, nvir, nocc_total);

            // M^P = B^P_sparse @ P,  N^P = B^P_sparse @ Q
            // Sparse mat-mul avoids multiplying the zero/tiny elements of B^P.
            // Pack into (naux, nbas²) buffers for the final J = M @ N^T DGEMM.
            let mut m_buf = Array2::<f64>::zeros((naux, nbas * nbas));
            let mut n_t_buf = Array2::<f64>::zeros((naux, nbas * nbas));
            for p in 0..naux {
                let ms = b_sparse[p].mat_mul_flat(&pt);
                let ns = b_sparse[p].mat_mul_flat(&qt);
                m_buf.row_mut(p).assign(&Array1::from(ms));
                // Transpose N^P when packing: n_t_buf[p, ν*nbas+μ] = N^P[μ,ν]
                for mu in 0..nbas {
                    for nu in 0..nbas {
                        n_t_buf[(p, nu * nbas + mu)] = ns[mu * nbas + nu];
                    }
                }
            }
            let j_mat = m_buf.dot(&n_t_buf.t());
            let e_os_k: f64 = j_mat.iter().map(|&x| x * x).sum();

            // --- K term in MO basis ---
            // Apply Laplace weights: B_ia(t) = B_ia * exp(-t*(ε_a - ε_i)/2)
            let mut b_t = b_mo_flat.clone();
            for i in 0..nocc {
                let e_i = eps[frozen_core + i];
                for a in 0..nvir {
                    let factor = (-0.5 * t * (eps[nocc_total + a] - e_i)).exp();
                    for p in 0..naux {
                        b_t[(p, i * nvir + a)] *= factor;
                    }
                }
            }
            // e_exch = Σ_{Pi,Qj} G[Pi,Qj]*G[Pj,Qi]
            //        = Σ_{P,Q} Σ_{ij} G_PQ[i,j] * G_PQ[j,i]
            //        = Σ_{P,Q} Tr(G_PQ²)   where G_PQ = B_P @ B_Q^T (nocc×nocc)
            // Memory: O(naux × nocc × nvir) — no large Gram matrix.
            let b_3d = b_t.into_shape_with_order((naux, nocc, nvir)).unwrap();
            let b_slices_mo: Vec<Array2<f64>> = (0..naux)
                .map(|p| b_3d.slice(ndarray::s![p, .., ..]).to_owned())
                .collect();
            let e_exch_k: f64 = (0..naux).map(|p| {
                (0..naux).map(|q| {
                    let g_pq = b_slices_mo[p].dot(&b_slices_mo[q].t()); // nocc × nocc
                    // Tr(G_PQ²) = Σ_{ij} G_PQ[i,j] * G_PQ[j,i]
                    let gs = g_pq.as_slice().unwrap();
                    (0..nocc).map(|i| (0..nocc).map(|j| gs[i*nocc+j] * gs[j*nocc+i]).sum::<f64>()).sum::<f64>()
                }).sum::<f64>()
            }).sum();

            let e_ss_k = e_os_k - e_exch_k;
            (-w * e_os_k, -w * e_ss_k)
        }).reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

        Ok((e_os + e_ss, e_os, e_ss))
    }
}

/// Build the occupied pseudo-density P(t)_{μν} = Σ_i C_{μi} exp(t ε_i) C_{νi}.
pub fn build_pseudo_density_occ(
    c: &Array2<f64>,
    eps: &[f64],
    t: f64,
    nocc: usize,
    first_occ: usize,
) -> Array2<f64> {
    let n = c.nrows();
    let mut p = Array2::zeros((n, n));
    for i in 0..nocc {
        let factor = (t * eps[first_occ + i]).exp();
        for mu in 0..n {
            let c_mu_i = c[(mu, first_occ + i)] * factor;
            for nu in 0..n {
                p[(mu, nu)] += c_mu_i * c[(nu, first_occ + i)];
            }
        }
    }
    p
}

/// Build the virtual pseudo-density Q(t)_{μν} = Σ_a C_{μa} exp(-t ε_a) C_{νa}.
pub fn build_pseudo_density_vir(
    c: &Array2<f64>,
    eps: &[f64],
    t: f64,
    nvir: usize,
    nocc_total: usize,
) -> Array2<f64> {
    let n = c.nrows();
    let mut q = Array2::zeros((n, n));
    for a in 0..nvir {
        let factor = (-t * eps[nocc_total + a]).exp();
        for mu in 0..n {
            let c_mu_a = c[(mu, nocc_total + a)] * factor;
            for nu in 0..n {
                q[(mu, nu)] += c_mu_a * c[(nu, nocc_total + a)];
            }
        }
    }
    q
}


/// Select minimax quadrature exponents and weights for 1/x on [1, R].
///
/// Data from Takatsuka, Ten-no, Hackbusch (JCP 129, 044112, 2008)
/// via Helmich-Paris laplace-minimax library.
/// Returns (exponents, weights) for the unnormalized interval [1, R].
fn select_minimax_points(k: usize, r: f64) -> (Vec<f64>, Vec<f64>) {
    // Table: (R_threshold, exponents, weights) for each k
    // We pick the entry with the closest R >= r.
    let table: &[MinimaxEntry] = match k {
        3 => MINIMAX_K3,
        5 => MINIMAX_K5,
        7 => MINIMAX_K7,
        _ => MINIMAX_K7,
    };

    // Pick smallest tabulated R that covers our range (R_tab >= R)
    for &(r_tab, ref t, ref w) in table.iter() {
        if r_tab >= r * 0.99 {
            return (t.to_vec(), w.to_vec());
        }
    }
    // Largest R entry as fallback
    let (_, ref t, ref w) = table[table.len() - 1];
    (t.to_vec(), w.to_vec())
}

type MinimaxEntry = (f64, &'static [f64], &'static [f64]);

static MINIMAX_K3: &[MinimaxEntry] = &[
    (5.0,   &[1.6607313750141492e-01, 9.7843720854537763e-01, 3.0530159808682455e+00],
            &[4.3657921050336840e-01, 1.2723305210744182e+00, 3.2151540832501007e+00]),
    (10.0,  &[1.0644554843011034e-01, 6.7919336467354152e-01, 2.4024163101166760e+00],
            &[2.8473486758368682e-01, 9.5831151180023555e-01, 2.8443772254878890e+00]),
    (20.0,  &[6.7901368940589263e-02, 4.8979488750896610e-01, 1.9962015809195739e+00],
            &[1.8718659453518208e-01, 7.6338571884375950e-01, 2.6169711142660255e+00]),
    (50.0,  &[3.9082098432306915e-02, 3.5092306837783149e-01, 1.6921947262166477e+00],
            &[1.1533107566965783e-01, 6.1942325781831142e-01, 2.4405036326787770e+00]),
    (100.0, &[2.8403804819953280e-02, 2.9983005709471117e-01, 1.5761907592561200e+00],
            &[8.9215707908966990e-02, 5.6479425010878515e-01, 2.3699428171677330e+00]),
];

static MINIMAX_K5: &[MinimaxEntry] = &[
    (5.0,   &[1.0333512117595546e-01, 5.6589985547987387e-01, 1.5002594864446179e+00, 3.1563889823905420e+00, 6.1935841722085909e+00],
            &[2.6751449382571008e-01, 6.7316481854106480e-01, 1.2341495375414908e+00, 2.1730749987606481e+00, 4.2400491045957720e+00]),
    (10.0,  &[6.4797316294974178e-02, 3.6329423979113412e-01, 1.0098084995269310e+00, 2.2813077174805523e+00, 4.8771539519428151e+00],
            &[1.6861003324210627e-01, 4.4474622263913965e-01, 8.9143002224546830e-01, 1.7540776690871602e+00, 3.7773593670209080e+00]),
    (20.0,  &[3.9791035518631973e-02, 2.3131332253827694e-01, 6.8992455340431735e-01, 1.7098435082055961e+00, 4.0149225768154402e+00],
            &[1.0434198456045442e-01, 2.9555856650050433e-01, 6.6783133886464374e-01, 1.4788816616452081e+00, 3.4728708916802091e+00]),
    (50.0,  &[2.0598516936418058e-02, 1.2946507144298772e-01, 4.3979723967050899e-01, 1.2499372840249323e+00, 3.2997722744581939e+00],
            &[5.4928968471960854e-02, 1.7972260846046623e-01, 4.8760926843571956e-01, 1.2421875649935035e+00, 3.2030368198904093e+00]),
    (100.0, &[1.2552727540467989e-02, 8.6549394877289257e-02, 3.3100124293713873e-01, 1.0393832416814095e+00, 2.9584699518773347e+00],
            &[3.4210303487463151e-02, 1.3026229608206730e-01, 4.0415841564587901e-01, 1.1232181675451096e+00, 3.0638169728620763e+00]),
    (500.0, &[4.4700910206532437e-03, 4.3042233518954816e-02, 2.1332569755781333e-01, 7.9513224013521722e-01, 2.5436793036816292e+00],
            &[1.3537186918918044e-02, 7.8245171363032009e-02, 3.0503984426641567e-01, 9.7044129821768366e-01, 2.8813614470814399e+00]),
    (1000.0,&[3.4004600489073895e-03, 3.7149816796614998e-02, 1.9603210903786736e-01, 7.5684838765183138e-01, 2.4761708329681671e+00],
            &[1.0838272657023281e-02, 7.0764497808673207e-02, 2.8913820910474880e-01, 9.4454172629160815e-01, 2.8500005633404064e+00]),
];

static MINIMAX_K7: &[MinimaxEntry] = &[
    (5.0,   &[7.5178012053935581e-02, 4.0394057377108084e-01, 1.0300953839123785e+00, 2.0257208844822334e+00, 3.5300416178849785e+00, 5.8202600945046035e+00, 9.5718793022987914e+00],
            &[1.9380100796794852e-01, 4.6924355863005290e-01, 7.9463765966869560e-01, 1.2188241997775222e+00, 1.8330663504453952e+00, 2.8441417219183061e+00, 5.0081428000286365e+00]),
    (10.0,  &[4.6784971919695058e-02, 2.5413109177030713e-01, 6.6202063661281685e-01, 1.3468236945914096e+00, 2.4608175652259985e+00, 4.3018718689144126e+00, 7.5591271744403050e+00],
            &[1.2090244093952099e-01, 2.9921214711072491e-01, 5.2876236081329564e-01, 8.6523254061261390e-01, 1.4096599453909606e+00, 2.3721982300004751e+00, 4.4850705608595405e+00]),
    (20.0,  &[2.8397875019357810e-02, 1.5669465451979245e-01, 4.2128353664098478e-01, 9.0071033948500667e-01, 1.7551920884002583e+00, 3.2943879648814645e+00, 6.2167521169331161e+00],
            &[7.3640943793741087e-02, 1.8809607758877431e-01, 3.5358561229255181e-01, 6.3102636982655691e-01, 1.1268723268357688e+00, 2.0532076958089367e+00, 4.1312146140908652e+00]),
    (50.0,  &[1.4346616954119040e-02, 8.1686636965124057e-02, 2.3392918256659598e-01, 5.4781736961611505e-01, 1.1828967187208399e+00, 2.4526048384096861e+00, 5.0658409539581051e+00],
            &[3.7455686176677001e-02, 1.0185840534484672e-01, 2.1485110258922235e-01, 4.3851262496240967e-01, 8.8110126016728485e-01, 1.7621025260651733e+00, 3.8032676122312763e+00]),
    (100.0, &[8.4890821340208086e-03, 5.0116642014961708e-02, 1.5366998285929265e-01, 3.9160355318450513e-01, 9.1783364668505174e-01, 2.0447798621093103e+00, 4.4889127432163773e+00],
            &[2.2333642551176373e-02, 6.5162010730403855e-02, 1.5346948447776387e-01, 3.4692296202841277e-01, 7.5471793703896095e-01, 1.6044221299260411e+00, 3.6234684847661183e+00]),
    (500.0, &[2.5357978643651921e-03, 1.7578759091398184e-02, 6.7613168610773614e-02, 2.1262834171859676e-01, 5.9156023005630420e-01, 1.5113889410817196e+00, 3.7022956650360443e+00],
            &[6.9203048135253430e-03, 2.6567898179860355e-02, 8.2863747425280715e-02, 2.2897941405638794e-01, 5.7693244244983499e-01, 1.3704988341020230e+00, 3.3539045747733187e+00]),
    (1000.0,&[1.5465907100920744e-03, 1.2056643172650844e-02, 5.1852489933254427e-02, 1.7661795467136937e-01, 5.2036352547560127e-01, 1.3877091798262136e+00, 3.5126938632453744e+00],
            &[4.3590321081121135e-03, 1.9739937643250896e-02, 6.8465314092830382e-02, 2.0200683741399486e-01, 5.3305670896838087e-01, 1.3100924544159886e+00, 3.2836589740207804e+00]),
];

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    use ferric_integrals::operator::Operator;

    #[test]
    fn test_laplace_mp2_mo_vs_ao() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        ).unwrap();

        let mut laplace = LaplaceMp2::new(3);
        let e_mo = laplace.compute_mo(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();
        let (e_ao, _, _) = laplace.compute_ao(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();

        eprintln!("Laplace RI-MP2 MO: {e_mo:.10}");
        eprintln!("Laplace RI-MP2 AO: {e_ao:.10}");

        assert!((e_mo - e_ao).abs() < 1e-8,
            "MO and AO Laplace methods should give identical results: {e_mo} vs {e_ao}");
    }

    #[test]
    fn test_laplace_mp2_water_ccpvdz() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        ).unwrap();

        let mut laplace = LaplaceMp2::new(7);
        let e_mo = laplace.compute_mo(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();
        let (e_ao, _, _) = laplace.compute_ao(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();

        eprintln!("H2O Laplace RI-MP2 MO: {e_mo:.10}");
        eprintln!("H2O Laplace RI-MP2 AO: {e_ao:.10}");

        assert!((e_mo - e_ao).abs() < 1e-8);

        // Reference RI-MP2 for H2O/cc-pVDZ is -0.20403347
        let ri_mp2_ref = -0.20403347;
        assert!((e_mo - ri_mp2_ref).abs() < 1e-3,
            "Laplace RI-MP2 ({e_mo:.6}) should be close to RI-MP2 ({ri_mp2_ref:.6})");
    }

    #[test]
    fn test_laplace_quadrature_convergence() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig { energy_conv: 1e-10, ..Default::default() },
        ).unwrap();

        // 3 points vs 5 points vs 7 points
        let mut lap3 = LaplaceMp2::new(3);
        let mut lap5 = LaplaceMp2::new(5);
        let mut lap7 = LaplaceMp2::new(7);

        let e3 = lap3.compute_mo(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();
        let e5 = lap5.compute_mo(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();
        let e7 = lap7.compute_mo(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();

        eprintln!("H2/cc-pVDZ Laplace MP2: k=3: {e3:.10}, k=5: {e5:.10}, k=7: {e7:.10}");

        // They should all be within ~0.001 Ha of each other for H2
        assert!((e3 - e5).abs() < 1e-3);
        assert!((e5 - e7).abs() < 1e-4);
    }
}
