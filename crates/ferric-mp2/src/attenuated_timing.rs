//! Step-by-step timing of attenuated RI-MP2 on decane.
//!
//! Run with:
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-mp2 attenuated_timing \
//!     --release -- --nocapture

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_integrals::qqr3::QqrBounds3;
    use ferric_integrals::threeindex;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    use crate::attenuated::BOHR_INV_PER_ANG_INV;
    use crate::mo_transform::transform_3center_ov;
    use crate::rimp2::{cholesky_inverse_sqrt, SpinComponents};

    fn run_decane(obs_name: &str, aux_name: &str) {
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_10.xyz").unwrap();
        let obs_bs  = basis::bundled(obs_name).unwrap();
        let dfbs_bs = basis::bundled(aux_name).unwrap();
        let op_c = Operator::coulomb();
        let omega = 0.420 * BOHR_INV_PER_ANG_INV; // 0.420 Å⁻¹
        let op_e = Operator::erfc(omega);

        let obs  = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();

        let nbf  = obs.nbasis();
        let naux = dfbs.nbasis();
        let nocc_total = mol.nelec() as usize / 2;
        let nvir = nbf - nocc_total;
        let nocc = nocc_total; // no frozen core in this spike

        // ── Step 0: RHF ────────────────────────────────────────────────────
        let t = Instant::now();
        let bounds_rhf = SchwarzBounds::compute(op_c, &obs).unwrap();
        let rhf = solve_rhf(
            &ParallelContext::default(), &mol, &obs, op_c, &bounds_rhf,
            &RhfConfig::default(),
        ).unwrap();
        let t_rhf = t.elapsed();
        let c = rhf.mos_r();
        let c_occ = c.slice(ndarray::s![.., ..nocc]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();
        let eps = rhf.eps_r();

        println!("\n{obs_name}/{aux_name}  nbf={nbf} naux={naux} nocc={nocc} nvir={nvir}");
        println!("Step 0  RHF:                    {:>8.1} ms", t_rhf.as_secs_f64()*1e3);
        println!();
        println!("{:<30} {:>12} {:>12} {:>12}", "Step", "Coulomb", "erfc(dense)", "erfc(screen)");
        println!("{}", "-".repeat(68));

        // Helper: run all 5 steps for one operator and screening choice.
        let run = |op: Operator, screen_thresh: Option<f64>| -> [f64; 5] {
            let mut ts = [0.0f64; 5];

            // Step 1: 2-center metric + Cholesky
            let t1 = Instant::now();
            let v2c = threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
            let v_inv_sqrt = cholesky_inverse_sqrt(&v2c).unwrap();
            ts[0] = t1.elapsed().as_secs_f64() * 1e3;

            // Step 2: 3-center AO integrals (dense or screened)
            let t2 = Instant::now();
            let (eri3_ao, n_kept, n_total) = if let Some(thresh) = screen_thresh {
                let bounds = QqrBounds3::new(op, &mol, &obs, &dfbs).unwrap();
                threeindex::eri3_tensor_screened_qqr(op, &obs, &dfbs, &bounds, thresh).unwrap()
            } else {
                let full = threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
                let nt = dfbs.nshells() * obs.nshells() * obs.nshells();
                (full, nt, nt)
            };
            ts[1] = t2.elapsed().as_secs_f64() * 1e3;
            let pct = 100.0 * n_kept as f64 / n_total as f64;

            // Step 3: MO transform (P|μν) → (P|ia)
            let t3 = Instant::now();
            let eri3_mo = transform_3center_ov(&eri3_ao, &c_occ, &c_vir);
            ts[2] = t3.elapsed().as_secs_f64() * 1e3;

            // Step 4: Metric contraction  B̃ = V^{-1/2} (P|ia)
            let t4 = Instant::now();
            let eri3_flat = eri3_mo.into_shape_with_order((naux, nocc * nvir)).unwrap();
            let b_flat = v_inv_sqrt.dot(&eri3_flat);
            ts[3] = t4.elapsed().as_secs_f64() * 1e3;

            // Step 5: Energy assembly  Σ_P B̃^P_ia B̃^P_jb / Δε
            let t5 = Instant::now();
            let mut e_os = 0.0f64;
            let mut e_ss = 0.0f64;
            for i in 0..nocc {
                for j in 0..nocc {
                    for a in 0..nvir {
                        for b in 0..nvir {
                            let ia = i*nvir+a; let jb = j*nvir+b;
                            let ib = i*nvir+b; let ja = j*nvir+a;
                            let iajb: f64 = (0..naux).map(|p| b_flat[(p,ia)]*b_flat[(p,jb)]).sum();
                            let ibja: f64 = (0..naux).map(|p| b_flat[(p,ib)]*b_flat[(p,ja)]).sum();
                            let d = eps[i]+eps[j]-eps[nocc_total+a]-eps[nocc_total+b];
                            e_os += iajb*iajb/d;
                            e_ss += iajb*(iajb-ibja)/d;
                        }
                    }
                }
            }
            ts[4] = t5.elapsed().as_secs_f64() * 1e3;
            let _ = SpinComponents { e_os, e_ss, e_total: e_os+e_ss };

            if screen_thresh.is_some() {
                eprintln!("  3-center kept {n_kept}/{n_total} ({pct:.0}%)");
            }
            ts
        };

        let tc = run(op_c, None);
        let te = run(op_e, None);
        let ts = run(op_e, Some(1e-10));

        let labels = [
            "Step 1  2c metric+Cholesky",
            "Step 2  3-center AO build",
            "Step 3  MO transform →(P|ia)",
            "Step 4  Metric contraction B̃",
            "Step 5  Energy assembly",
        ];
        for (i, lbl) in labels.iter().enumerate() {
            println!("{:<30} {:>10.1}ms {:>10.1}ms {:>10.1}ms",
                lbl, tc[i], te[i], ts[i]);
        }
        let sum_c: f64 = tc.iter().sum();
        let sum_e: f64 = te.iter().sum();
        let sum_s: f64 = ts.iter().sum();
        println!("{:<30} {:>10.1}ms {:>10.1}ms {:>10.1}ms",
            "TOTAL (post-RHF)", sum_c, sum_e, sum_s);
        println!("{:<30} {:>10}    {:>10.2}x {:>10.2}x",
            "Speedup vs Coulomb", "1.00x",
            sum_c/sum_e, sum_c/sum_s);
    }

    #[test]
    #[ignore = "benchmark: run with --release --ignored --nocapture"]
    fn decane_step_timing_sto3g() {
        run_decane("sto-3g", "cc-pvdz-ri");
    }

    #[test]
    #[ignore = "benchmark: run with --release --ignored --nocapture"]
    fn decane_step_timing_ccpvdz() {
        run_decane("cc-pvdz", "cc-pvdz-ri");
    }
}
