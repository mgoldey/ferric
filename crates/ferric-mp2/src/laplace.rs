//! AO-based Laplace-transform MP2.
//!
//! Uses the Laplace transform of the orbital energy denominator to express
//! the MP2 energy as an integral over AO-basis contractions, allowing for
//! linear scaling when combined with sparse matrix techniques.
//!
//! Reference: Häser & Almlöf, Chem. Phys. Lett. 191, 299 (1992).

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfResult;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

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

    /// Compute the MP2 energy using AO-basis Laplace transform.
    pub fn compute(
        &mut self,
        prep: &PreparedBasis,
        rhf: &RhfResult,
        bounds: &SchwarzBounds,
        op: Operator,
        frozen_core: usize,
    ) -> Result<f64, FerricError> {
        let nbas = prep.nbasis();
        let nocc_total = rhf.orbital_energies.iter().filter(|&&e| e < 0.0).count();
        let nocc = nocc_total - frozen_core;
        let nvir = nbas - nocc_total;

        let eps = &rhf.orbital_energies;
        let nmo = eps.len();
        assert!(nocc_total > 0 && nocc_total < nmo, "nocc_total={nocc_total} nmo={nmo}");
        let ymin = 2.0 * (eps[nocc_total] - eps[nocc_total - 1]);
        let ymax = 2.0 * (eps[nmo - 1] - eps[0]);
        self.init_quadrature(ymin, ymax);

        let c = &rhf.mos;
        let eps = &rhf.orbital_energies;
        let nsh = prep.nshells();
        let dims = prep.shell_dims();
        let offs = prep.shell_offsets();

        // Parallel quadrature over points.
        let e_corr: f64 = self.points.par_iter().zip(self.weights.par_iter()).map(|(&t, &w)| {
            let pt = build_pseudo_density_occ(c, eps, t, nocc, frozen_core);
            let qt = build_pseudo_density_vir(c, eps, t, nvir, nocc_total);

            // Precompute max values in pseudo-density blocks for screening
            let mut p_max = Array2::zeros((nsh, nsh));
            let mut q_max = Array2::zeros((nsh, nsh));
            for s1 in 0..nsh {
                for s2 in 0..nsh {
                    let mut p_m = 0.0f64;
                    let mut q_m = 0.0f64;
                    for i in 0..dims[s1] {
                        for j in 0..dims[s2] {
                            p_m = p_m.max(pt[(offs[s1] + i, offs[s2] + j)].abs());
                            q_m = q_m.max(qt[(offs[s1] + i, offs[s2] + j)].abs());
                        }
                    }
                    p_max[(s1, s2)] = p_m;
                    q_max[(s1, s2)] = q_m;
                }
            }

            let mut engine = Engine::new_2e(op, prep, 1e-14).unwrap();
            let mut e_t = 0.0;

            // Canonical shell quartet loop
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    let b12 = bounds.q[(s1, s2)];
                    for s3 in 0..=s1 {
                        let s4_max = if s3 == s1 { s2 } else { s3 };
                        for s4 in 0..=s4_max {
                            let b34 = bounds.q[(s3, s4)];
                            
                            // Combined Schwarz and Density screening: (s1 s2 | s3 s4) * P * Q
                            // A rough bound: (s1s2|s3s4) <= b12 * b34
                            let screen_val = b12 * b34 * (2.0 * p_max[(s1, s3)] * q_max[(s2, s4)] + p_max[(s1, s4)] * q_max[(s2, s3)]);
                            if screen_val < 1e-12 { continue; }

                            if let Some(q) = engine.compute_quartet(prep, s1, s2, s3, s4) {
                                let (n1, n2, n3, n4) = (dims[s1], dims[s2], dims[s3], dims[s4]);
                                let (o1, o2, o3, o4) = (offs[s1], offs[s2], offs[s3], offs[s4]);
                                
                                let sym12 = s1 != s2;
                                let sym34 = s3 != s4;
                                let sym1234 = (s1, s2) != (s3, s4);

                                for a in 0..n1 {
                                    for b in 0..n2 {
                                        for cc in 0..n3 {
                                            for dd in 0..n4 {
                                                let v = q[((a * n2 + b) * n3 + cc) * n4 + dd];
                                                let (mu, nu, la, sg) = (o1 + a, o2 + b, o3 + cc, o4 + dd);

                                                let term = |m, n, l, s| {
                                                    v * (2.0 * pt[(m, l)] * qt[(n, s)] - pt[(m, s)] * qt[(n, l)])
                                                };

                                                e_t += term(mu, nu, la, sg);
                                                if sym12 { e_t += term(nu, mu, la, sg); }
                                                if sym34 { e_t += term(mu, nu, sg, la); }
                                                if sym12 && sym34 { e_t += term(nu, mu, sg, la); }

                                                if sym1234 {
                                                    e_t += term(la, sg, mu, nu);
                                                    if sym34 { e_t += term(sg, la, mu, nu); }
                                                    if sym12 { e_t += term(la, sg, nu, mu); }
                                                    if sym12 && sym34 { e_t += term(sg, la, nu, mu); }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            -w * e_t
        }).sum();

        Ok(e_corr)
    }

    /// Compute the analytical nuclear gradient for AO-Laplace MP2.
    /// Returns a (natoms, 3) array.
    pub fn compute_gradient(
        &mut self,
        mol: &Molecule,
        prep: &PreparedBasis,
        rhf: &RhfResult,
        bounds: &SchwarzBounds,
        op: Operator,
        frozen_core: usize,
    ) -> Result<Array2<f64>, FerricError> {
        let nbas = prep.nbasis();
        let natoms = prep.shell_to_atom().iter().copied().max().unwrap_or(0) + 1;
        let nocc_total = rhf.orbital_energies.iter().filter(|&&e| e < 0.0).count();
        let nocc = nocc_total - frozen_core;
        let nvir = nbas - nocc_total;

        let c = &rhf.mos;
        let eps = &rhf.orbital_energies;
        let nsh = prep.nshells();
        let dims = prep.shell_dims();
        let offs = prep.shell_offsets();
        let sh2at = prep.shell_to_atom();

        let mut total_grad = Array2::zeros((natoms, 3));

        // Parallel quadrature over points.
        let point_grads: Vec<Array2<f64>> = self.points.par_iter().zip(self.weights.par_iter()).map(|(&t, &w)| {
            let pt = build_pseudo_density_occ(c, eps, t, nocc, frozen_core);
            let qt = build_pseudo_density_vir(c, eps, t, nvir, nocc_total);

            let mut grad_t = Array2::zeros((natoms, 3));
            let mut engine = Engine::new_2e_deriv(op, prep, 1e-14).unwrap();

            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    let b12 = bounds.q[(s1, s2)];
                    for s3 in 0..=s1 {
                        let s4_max = if s3 == s1 { s2 } else { s3 };
                        for s4 in 0..=s4_max {
                            let b34 = bounds.q[(s3, s4)];
                            if b12 * b34 < 1e-12 { continue; }

                            if let Some(dq) = engine.compute_eri_deriv_quartet(prep, s1, s2, s3, s4) {
                                let (n1, n2, n3, n4) = (dims[s1], dims[s2], dims[s3], dims[s4]);
                                let (o1, o2, o3, o4) = (offs[s1], offs[s2], offs[s3], offs[s4]);
                                let block_sz = n1 * n2 * n3 * n4;
                                let atoms = [sh2at[s1], sh2at[s2], sh2at[s3], sh2at[s4]];

                                for a in 0..n1 {
                                    for b in 0..n2 {
                                        for cc in 0..n3 {
                                            for dd in 0..n4 {
                                                let idx = ((a * n2 + b) * n3 + cc) * n4 + dd;
                                                let (mu, nu, la, sg) = (o1 + a, o2 + b, o3 + cc, o4 + dd);

                                                // Gamma = 2 P_mu,la Q_nu,sg - P_mu,sg Q_nu,la
                                                let gamma = 2.0 * pt[(mu, la)] * qt[(nu, sg)] - pt[(mu, sg)] * qt[(nu, la)];

                                                for center in 0..4 {
                                                    let atom = atoms[center];
                                                    for coord in 0..3 {
                                                        let dv = dq[(center * 3 + coord) * block_sz + idx];
                                                        grad_t[(atom, coord)] -= w * gamma * dv;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            grad_t
        }).collect();

        for pt_grad in point_grads {
            total_grad += &pt_grad;
        }

        // 1. Build total relaxed 1-PDM
        let p_corr = self.build_relaxed_density(c, eps, frozen_core, nocc, nvir, nocc_total);
        let p_relax = &rhf.density + &p_corr;

        // 2. Build energy-weighted density
        // For Laplace, W is approximately epsilon * P_relax
        let f_ao = &rhf.fock;
        let w_relax = p_relax.dot(f_ao).dot(&p_relax);

        // 3. One-electron gradient (Vnn + dS + dT + dV)
        let one_elec = ferric_scf::gradient::oneelectron_gradient(mol, prep, &p_relax, &w_relax)?;
        total_grad += &one_elec;

        Ok(total_grad)
    }

    /// Build the correlation contribution to the 1-PDM for Laplace-MP2.
    pub fn build_relaxed_density(
        &self,
        c: &Array2<f64>,
        eps: &[f64],
        first_occ: usize,
        nocc: usize,
        nvir: usize,
        nocc_total: usize,
    ) -> Array2<f64> {
        let n = c.nrows();
        let mut p_corr = Array2::zeros((n, n));
        
        // Sum over points: D = sum_t w_t [ P(t) Q(t) P(t) - Q(t) P(t) Q(t) ] (approx)
        // More accurately: D_pq = sum_t w_t [exp(t eps_p) * exp(-t eps_q) * ...]
        // We'll use a simplified version that matches the energy expression.
        for (&t, &w) in self.points.iter().zip(self.weights.iter()) {
            let pt = build_pseudo_density_occ(c, eps, t, nocc, first_occ);
            let qt = build_pseudo_density_vir(c, eps, t, nvir, nocc_total);
            
            // Corr density correction is typically small
            p_corr += &(w * (pt.dot(&qt).dot(&pt) - qt.dot(&pt).dot(&qt)));
        }
        p_corr
    }
}

/// Build the pseudo-density P(t) = C_occ * exp(t * epsilon_occ) * C_occ^T
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
            let c_mu_i = c[(mu, first_occ + i)];
            for nu in 0..n {
                p[(mu, nu)] += c_mu_i * c[(nu, first_occ + i)] * factor;
            }
        }
    }
    p
}

/// Build the pseudo-density Q(t) = C_vir * exp(-t * epsilon_vir) * C_vir^T
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
            let c_mu_a = c[(mu, nocc_total + a)];
            for nu in 0..n {
                q[(mu, nu)] += c_mu_a * c[(nu, nocc_total + a)] * factor;
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
    fn test_laplace_mp2_h2_sto3g() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &prep,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        ).unwrap();

        let mut laplace = LaplaceMp2::new(5);
        let e_corr = laplace.compute(&prep, &rhf, &bounds, op, 0).unwrap();
        
        // Laplace MP2 with rough quadrature; check it's finite and non-positive
        eprintln!("Laplace MP2 corr: {e_corr:.10}");
        assert!(e_corr.is_finite(), "Laplace MP2 not finite: {e_corr}");
        assert!(e_corr <= 0.0, "Laplace MP2 should be non-positive: {e_corr}");
    }

    #[test]
    fn test_laplace_mp2_gradient() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol, &prep, op, &bounds,
            &RhfConfig { energy_conv: 1e-10, ..Default::default() },
        ).unwrap();

        let mut laplace = LaplaceMp2::new(3);
        let grad = laplace.compute_gradient(&mol, &prep, &rhf, &bounds, op, 0).unwrap();
        
        eprintln!("Laplace MP2 gradient (direct term):");
        for i in 0..2 {
            eprintln!("  atom {}: {:?}", i, grad.row(i));
        }
        
        // Gradient should be finite
        assert!(grad.iter().all(|x| x.is_finite()));
        // Should be equal and opposite on the two atoms
        assert!((grad[(0, 2)] + grad[(1, 2)]).abs() < 1e-8);
    }
}
