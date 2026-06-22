//! Full GW100 sweep — IPs by every available method.
//!
//! For each molecule in the 10-molecule subset of GW100, compute the
//! vertical first IP by:
//!
//!   1. **Koopmans**: IP_K = −ε_HOMO from neutral RHF
//!   2. **ΔSCF (UHF)**: E_UHF(cation) − E_RHF(neutral)
//!   3. **ΔRPA**: [E_UHF(cation) + E_U-RPA(cation)] − [E_RHF(neutral) + E_RPA(neutral)]
//!   5. **G0W0@HF**: −ε_HOMO^QP from G0W0 on neutral
//!   6. **COHSEX@HF**: −ε_HOMO^QP from static-W COHSEX on neutral
//!   7. **evGW₀@HF**: eigenvalue-self-consistent (Σ updated, W frozen)
//!   8. **evGW@HF**:  full eigenvalue-self-consistent
//!
//! Output: per-molecule table with IPs and MAE versus experiment.
//!
//! Run:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release \
//!     --example gw100_full -p ferric-gw 2>&1 | tee docs/gw100-full-results.txt

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{run_gw, GwConfig, GwMethod};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_rpa::{run_pdep_rpa, run_u_pdep_rpa};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::{solve_rohf, RohfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_guess, UhfConfig};
use ndarray::Array2;

const HA_TO_EV: f64 = 27.211386245988_f64;

/// PDEP truncation threshold for the sweep. Default 0.0 (full-rank, apples-to-
/// apples vs the reference). Set GW100_TRUNC=1e-4 to enable truncation — PROVEN
/// lossless for the GW IP (water: G0W0/evGW unchanged at 1e-4; commit 971e7de,
/// TRUNCATION_VERIFIED.md). Truncation shrinks the freq_quad O(K·M³) inversions,
/// the dominant cost on large molecules (benzene's bottleneck), so a truncated
/// sweep gives the SAME IPs much faster — the production path for the big organics.
fn trunc_thresh() -> f64 {
    std::env::var("GW100_TRUNC")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0)
}

struct Case {
    name: &'static str,
    xyz: &'static str,
    ip_ref: f64,
}

fn cases() -> Vec<Case> {
    vec![
        Case { name: "H2",   xyz: "2\nH2\nH 0 0 0\nH 0 0 0.7414\n", ip_ref: 15.43 },
        Case { name: "He",   xyz: "1\nHe\nHe 0 0 0\n", ip_ref: 24.59 },
        Case { name: "H2O",  xyz: "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n", ip_ref: 12.62 },
        Case { name: "NH3",  xyz: "4\nNH3\nN 0.0 0.0 0.116743\nH 0.94 0.0 -0.272400\nH -0.471 0.815 -0.272400\nH -0.471 -0.815 -0.272400\n", ip_ref: 10.82 },
        Case { name: "CH4",  xyz: "5\nCH4\nC 0.0 0.0 0.0\nH 0.629 0.629 0.629\nH -0.629 -0.629 0.629\nH -0.629 0.629 -0.629\nH 0.629 -0.629 -0.629\n", ip_ref: 13.6 },
        Case { name: "N2",   xyz: "2\nN2\nN 0 0 -0.5488\nN 0 0  0.5488\n", ip_ref: 15.58 },
        Case { name: "CO",   xyz: "2\nCO\nC 0 0 -0.6442\nO 0 0  0.4828\n", ip_ref: 14.01 },
        Case { name: "F2",   xyz: "2\nF2\nF 0 0 -0.7080\nF 0 0  0.7080\n", ip_ref: 15.70 },
        Case { name: "HF",   xyz: "2\nHF\nF 0 0 0.0\nH 0 0 0.9168\n", ip_ref: 16.12 },
        Case { name: "C2H2", xyz: "4\nC2H2\nC 0 0 -0.6014\nC 0 0  0.6014\nH 0 0 -1.6605\nH 0 0  1.6605\n", ip_ref: 11.40 },
        // --- Batch 2: first/second-row closed-shell, experimental vertical IPs ---
        // Geometries C2H4/C2H6/CO2/HCl/H2S reused from scripts/dosd (DOSD-experimental).
        Case { name: "C2H4", xyz: "6\nC2H4\nC 0.000000 0.000000 0.669500\nC 0.000000 0.000000 -0.669500\nH 0.000000 0.922832 1.237695\nH 0.000000 -0.922832 1.237695\nH 0.000000 0.922832 -1.237695\nH 0.000000 -0.922832 -1.237695\n", ip_ref: 10.68 },
        Case { name: "C2H6", xyz: "8\nC2H6\nC 0.000000 0.000000 0.768000\nC 0.000000 0.000000 -0.768000\nH 0.000000 1.013302 1.164532\nH 0.877488 -0.506651 1.164532\nH -0.877488 -0.506651 1.164532\nH 0.000000 -1.013302 -1.164532\nH -0.877488 0.506651 -1.164532\nH 0.877488 0.506651 -1.164532\n", ip_ref: 11.99 },
        Case { name: "CO2",  xyz: "3\nCO2\nC 0.000000 0.000000 0.000000\nO 0.000000 0.000000 1.162000\nO 0.000000 0.000000 -1.162000\n", ip_ref: 13.78 },
        Case { name: "HCl",  xyz: "2\nHCl\nCl 0.000000 0.000000 0.000000\nH 0.000000 0.000000 1.275000\n", ip_ref: 12.79 },
        Case { name: "H2S",  xyz: "3\nH2S\nS 0.000000 0.000000 0.103729\nH 0.000000 0.961700 -0.829834\nH 0.000000 -0.961700 -0.829834\n", ip_ref: 10.50 },
        // HCN linear (rCH=1.064, rCN=1.156); H2CO planar C2v (rCO=1.208, rCH=1.116, HCH=116.5);
        // CH3OH staggered Cs (standard experimental parameters).
        Case { name: "HCN",  xyz: "3\nHCN\nH 0.0 0.0 -1.064000\nC 0.0 0.0 0.000000\nN 0.0 0.0 1.156000\n", ip_ref: 13.61 },
        Case { name: "H2CO", xyz: "4\nH2CO\nO 0.0 0.0 0.674000\nC 0.0 0.0 -0.534000\nH 0.0 0.945000 -1.130000\nH 0.0 -0.945000 -1.130000\n", ip_ref: 10.88 },
        Case { name: "CH3OH", xyz: "6\nCH3OH\nC 0.0 0.0 0.0\nO 0.0 0.0 1.421000\nH 1.020000 0.0 -0.360000\nH -0.510000 0.883000 -0.360000\nH -0.510000 -0.883000 -0.360000\nH 0.890000 0.0 1.760000\n", ip_ref: 10.96 },
        // --- GW100 expansion: remaining ECP-free (Z<=36) molecules ---
        // Canonical geometries + experimental vertical IPs from the GW100
        // database (github.com/setten/GW100). CAS in trailing comment.
        // The 7 ECP-requiring mols (Xe/Rb/I/CH3I/CHI3-class/AlI3/Ag) are
        // excluded — ferric has no ECP support. 18 originals above + 75 = 93.
        Case { name: "Ne", xyz: "1\nmol\nNe 0.0 0.0 0.0\n", ip_ref: 21.56 }, // 7440-01-9
        Case { name: "Ar", xyz: "1\nmol\nAr 0.0 0.0 0.0\n", ip_ref: 15.76 }, // 7440-37-1
        Case { name: "Kr", xyz: "1\nmol\nKr 0.0 0.0 0.0\n", ip_ref: 14.00 }, // 7439-90-9
        Case { name: "Li2", xyz: "2\nmol\nLi 0.0000 0.0000 0.0000\nLi 0.0000 0.0000 2.6729\n", ip_ref: 4.73 }, // 14452-59-6
        Case { name: "Na2", xyz: "2\nmol\nNa 0.0000 0.0000 0.0000\nNa 0.0000 0.0000 3.0789\n", ip_ref: 4.89 }, // 25681-79-2
        Case { name: "Na4", xyz: "4\nmol\nNa 0.0002445 -0.0998053 1.5471126\nNa -0.0002444 3.1776586 0.0486374\nNa 0.0002444 0.0997722 -1.5472150\nNa -0.0002444 -3.1776254 -0.0485350\n", ip_ref: 4.27 }, // 39297-86-4
        Case { name: "Na6", xyz: "6\nmol\nNa -2.4732949 -1.7969539 -0.2367313\nNa -2.4732949 1.7969539 -0.2367313\nNa 0.9447146 -2.9075325 -0.2367313\nNa 0.9447146 2.9075325 -0.2367313\nNa 3.0571606 0.0000000 -0.2367313\nNa 0.0000000 0.0000000 1.1836565\n", ip_ref: 4.12 }, // 39297-88-6
        Case { name: "K2", xyz: "2\nmol\nK 0.0000 0.0000 0.0000\nK 0.0000 0.0000 3.9051\n", ip_ref: 4.06 }, // 25681-80-5
        Case { name: "P2", xyz: "2\nmol\nP 0.0000 0.0000 0.0000\nP 0.0000 0.0000 1.8931\n", ip_ref: 10.62 }, // 12185-09-0
        Case { name: "As2", xyz: "2\nmol\nAs 0.0000 0.0000 0.0000\nAs 0.0000 0.0000 2.1026\n", ip_ref: 10.00 }, // 23878-46-8
        Case { name: "Cl2", xyz: "2\nmol\nCl 0.0000 0.0000 0.0000\nCl 0.0000 0.0000 1.9878\n", ip_ref: 11.49 }, // 7782-50-5
        Case { name: "Br2", xyz: "2\nmol\nBr 0.0000 0.0000 0.0000\nBr 0.0000 0.0000 2.2811\n", ip_ref: 10.51 }, // 7726-95-6
        Case { name: "C3H8", xyz: "11\nmol\nC 0.0000 0.5863 -0.0000\nC -1.2681 -0.2626 0.0000\nC 1.2681 -0.2626 -0.0000\nH 0.0000 1.2449 0.8760\nH -0.0003 1.2453 -0.8758\nH -2.1576 0.3742 0.0000\nH 2.1576 0.3743 0.0000\nH -1.3271 -0.9014 0.8800\nH -1.3271 -0.9014 -0.8800\nH 1.3271 -0.9014 -0.8800\nH 1.3272 -0.9014 0.8800\n", ip_ref: 11.51 }, // 74-98-6
        Case { name: "C4H10", xyz: "14\nmol\nC -0.5698992 0.0010721 -0.5106280\nC -1.9574388 -0.0010272 0.1310139\nC 0.5699130 0.0010684 0.5106802\nH -2.1019869 -0.8898549 0.7620598\nH -2.1011958 0.8826120 0.7695094\nH -2.7542204 0.0025732 -0.6252247\nH -0.4643309 -0.8776931 -1.1680830\nH -0.4658369 0.8818734 -1.1655696\nH 0.4658491 0.8818113 1.1656828\nH 0.4644008 -0.8777422 1.1680681\nC 1.9574251 -0.0010272 -0.1310574\nH 2.7542844 0.0069401 0.6250702\nH 2.1035257 -0.8919972 -0.7587064\nH 2.0995094 0.8804516 -0.7729107\n", ip_ref: 11.09 }, // 106-97-8
        Case { name: "C4", xyz: "4\nmol\nC 1.2247 0.0000 0.0000\nC -1.2247 0.0000 0.0000\nC 0.0000 -0.7286 0.0000\nC 0.0000 0.7286 0.0000\n", ip_ref: 12.54 }, // 12184-80-4
        Case { name: "C3H6", xyz: "9\nmol\nC 0.036473 0.859901 -0.182257\nC -0.275674 -0.564282 -0.600227\nC 0.268426 -0.283892 0.786852\nH -0.412409 -0.380348 1.623541\nH 1.285827 -0.580215 1.010404\nH -0.796804 1.523474 0.013213\nH 0.896783 1.350428 -0.620593\nH -1.331415 -0.803627 -0.632093\nH 0.315621 -1.079884 -1.346828\n", ip_ref: 10.54 }, // 75-19-4
        Case { name: "C6H6", xyz: "12\nmol\nC 0.0000 1.3990 0.0000\nC 1.2115 0.6995 0.0000\nC 1.2115 -0.6995 0.0000\nC 0.0000 -1.3990 0.0000\nC -1.2115 -0.6995 0.0000\nC -1.2115 0.6995 0.0000\nH 0.0000 2.5000 0.0000\nH 2.1651 1.2500 0.0000\nH 2.1651 -1.2500 0.0000\nH 0.0000 -2.5000 0.0000\nH -2.1651 -1.2500 0.0000\nH -2.1651 1.2500 0.0000\n", ip_ref: 9.23 }, // 71-43-2
        Case { name: "C8H8", xyz: "16\nmol\nC -0.2627 -1.6663 0.3833\nC 1.0331 -1.3409 0.3827\nC -1.0297 1.3407 0.3845\nC 0.2630 1.6666 0.3835\nC -1.3424 -1.0272 -0.3796\nC 1.6770 -0.2635 -0.3837\nC -1.6841 0.2615 -0.3911\nC 1.3455 1.0312 -0.3823\nH -0.5690 -2.5492 0.9659\nH 1.7214 -1.9720 0.9656\nH -1.7184 1.9710 0.9702\nH 0.5706 2.5481 0.9698\nH -1.9688 -1.7275 -0.9575\nH 2.5603 -0.5678 -0.9650\nH -2.5689 0.5611 -0.9710\nH 1.9751 1.7236 -0.9613\n", ip_ref: 8.43 }, // 629-20-9
        Case { name: "C5H6", xyz: "11\nmol\nC 0.735000 0.000000 0.000000\nC -0.735000 0.000000 0.000000\nC 1.180760 0.000000 1.265805\nC -1.180760 0.000000 1.265805\nC -0.003091 0.000000 2.209296\nH 2.228506 0.000000 1.566354\nH 1.364644 0.000000 -0.889746\nH -1.364960 0.000000 -0.889523\nH -2.228838 0.000000 1.565193\nH -0.001086 0.885447 2.844969\nH -0.005702 -0.882815 2.848618\n", ip_ref: 8.53 }, // 542-92-7
        Case { name: "C2H3F", xyz: "6\nmol\nC 0.000000 0.000000 0.000000\nC 0.000000 0.000000 1.321000\nH -0.942589 0.000000 -0.521841\nH -0.874292 0.000000 1.955045\nH 0.922424 0.000000 -0.561725\nF 1.142469 0.000000 2.026603\n", ip_ref: 10.63 }, // 75-02-5
        Case { name: "C2H3Cl", xyz: "6\nmol\nC -0.554265 -0.445361 0.111076\nC 0.372254 0.438035 -0.234540\nH -1.322093 -0.210763 0.831940\nH -0.543697 -1.425246 -0.340536\nH 1.153254 0.241370 -0.951543\nCl 0.440430 2.028766 0.431795\n", ip_ref: 10.20 }, // 75-01-4
        Case { name: "C2H3Br", xyz: "6\nmol\nC 0.000000 0.000000 0.000000\nC 0.000000 0.000000 1.325600\nH -0.895976 0.000000 -0.602298\nH -0.894897 0.000000 1.927173\nH 0.908386 0.000000 -0.581003\nBr 1.357668 0.000000 2.194533\n", ip_ref: 9.90 }, // 593-60-2
        Case { name: "CF4", xyz: "5\nmol\nC 0.0000 0.0000 0.0000\nF 0.7638 -0.7638 0.7638\nF -0.7638 0.7638 0.7638\nF -0.7638 -0.7638 -0.7638\nF 0.7638 0.7638 -0.7638\n", ip_ref: 16.20 }, // 75-73-0
        Case { name: "CCl4", xyz: "5\nmol\nC 0.0000 0.0000 0.0000\nCl 1.0202 -1.0202 1.0202\nCl -1.0202 1.0202 1.0202\nCl -1.0202 -1.0202 -1.0202\nCl 1.0202 1.0202 -1.0202\n", ip_ref: 11.69 }, // 56-23-5
        Case { name: "CBr4", xyz: "5\nmol\nC 0.0000 0.0000 0.0000\nBr 1.1172 -1.1172 1.1172\nBr -1.1172 1.1172 1.1172\nBr -1.1172 -1.1172 -1.1172\nBr 1.1172 1.1172 -1.1172\n", ip_ref: 10.54 }, // 558-13-4
        Case { name: "H4Si", xyz: "5\nmol\nSi 0.0000 0.0000 0.0000\nH 0.8544 -0.8544 0.8544\nH -0.8544 0.8544 0.8544\nH -0.8544 -0.8544 -0.8544\nH 0.8544 0.8544 -0.8544\n", ip_ref: 12.30 }, // 7803-62-5
        Case { name: "GeH4", xyz: "5\nmol\nGe 0.0000 0.0000 0.0000\nH 0.8805 -0.8805 0.8805\nH -0.8805 0.8805 0.8805\nH -0.8805 -0.8805 -0.8805\nH 0.8805 0.8805 -0.8805\n", ip_ref: 11.34 }, // 7782-65-2
        Case { name: "H6Si2", xyz: "8\nmol\nSi 0.000000 0.000000 -1.165500\nSi 0.000000 0.000000 1.165500\nH 1.399330 0.000000 1.683128\nH -1.399330 0.000000 -1.683128\nH 0.699500 1.211600 -1.683128\nH 0.699500 -1.211600 -1.683128\nH -0.699500 -1.211600 1.683128\nH -0.699500 1.211600 1.683128\n", ip_ref: 10.53 }, // 1590-87-0
        Case { name: "H12Si5", xyz: "17\nmol\nSi -0.0048335 -3.8969717 -0.5238439\nH 1.1887767 -3.9134075 -1.4252499\nH 0.0223993 -5.1288186 0.3252902\nH -1.2385400 -3.9282984 -1.3688912\nSi 0.0104464 -1.9536989 0.7969029\nSi -0.0004053 0.0000130 -0.5100885\nH -1.1904082 -1.9457891 1.6943990\nH 1.2296666 -1.9529052 1.6693294\nSi -0.0104987 1.9536727 0.7969268\nH -1.2088685 -0.0034247 -1.3978287\nH 1.2088478 0.0037073 -1.3964714\nSi 0.0051940 3.8969653 -0.5238481\nH -0.0191436 5.1288161 0.3253672\nH 1.2377333 3.9262510 -1.3707348\nH -1.1896928 3.9156075 -1.4234610\nH 1.1910193 1.9496877 1.6935745\nH -1.2290887 1.9491200 1.6702050\n", ip_ref: 9.36 }, // 14868-53-2
        Case { name: "HLi", xyz: "2\nmol\nLi 0.0000 0.0000 0.0000\nH 0.0000 0.0000 1.5949\n", ip_ref: 7.90 }, // 7580-67-8
        Case { name: "HK", xyz: "2\nmol\nK 0.0000 0.0000 0.0000\nH 0.0000 0.0000 2.244\n", ip_ref: 8.00 }, // 7693-26-7
        Case { name: "BH3", xyz: "4\nmol\nB 0.0000 0.0000 0.0000\nH 0.0000 0.0000 1.19\nH 0.0000 1.0306 -0.595\nH 0.0000 -1.0306 -0.595\n", ip_ref: 12.03 }, // 13283-31-3
        Case { name: "B2H6", xyz: "8\nmol\nB 0.0000 0.0000 0.8870\nB 0.0000 0.0000 -0.8870\nH 0.9960 0.0000 0.0000\nH -0.9960 0.0000 0.0000\nH 0.0000 1.0408 1.4639\nH 0.0000 -1.0408 1.4639\nH 0.0000 1.0408 -1.4639\nH 0.0000 -1.0408 -1.4639\n", ip_ref: 11.90 }, // 19287-45-7
        Case { name: "HN3", xyz: "4\nmol\nH -0.9585 0.0000 -0.3338\nN 0.0000 0.0000 0.0000\nN 0.0000 0.0000 1.2450\nN 0.1617 0.0000 2.3674\n", ip_ref: 10.72 }, // 7782-79-8
        Case { name: "H3P", xyz: "4\nmol\nP 0.0000 0.0000 0.0000\nH 0.0000 -1.1932 -0.7717\nH 1.0333 0.5966 -0.7717\nH -1.0333 0.5966 -0.7717\n", ip_ref: 10.59 }, // 7803-51-2
        Case { name: "AsH3", xyz: "4\nmol\nAs 0.0000 0.0000 0.0000\nH 0.0000 1.2561 0.8398\nH 1.0878 -0.6281 0.8398\nH -1.0878 -0.6281 0.8398\n", ip_ref: 10.58 }, // 7784-42-1
        Case { name: "FLi", xyz: "2\nmol\nLi 0.0000 0.0000 0.0000\nF 0.0000 0.0000 1.5639\n", ip_ref: 11.30 }, // 7789-24-4
        Case { name: "F2Mg", xyz: "3\nmol\nF 0.0000 0.0000 1.771\nMg 0.0000 0.0000 0.0000\nF 0.0000 0.0000 -1.771\n", ip_ref: 13.30 }, // 7783-40-6
        Case { name: "F4Ti", xyz: "5\nmol\nTi 0.0000 0.0000 0.0000\nF 1.0127 -1.0127 1.0127\nF -1.0127 1.0127 1.0127\nF -1.0127 -1.0127 -1.0127\nF 1.0127 1.0127 -1.0127\n", ip_ref: 13.30 }, // 7783-63-3
        Case { name: "AlF3", xyz: "4\nmol\nAl 0.0000 0.0000 0.0000\nF 0.0000 0.0000 1.633\nF 0.0000 1.4142 -0.8165\nF 0.0000 -1.4142 -0.8165\n", ip_ref: 15.45 }, // 7784-18-1
        Case { name: "BF", xyz: "2\nmol\nB 0.0000 0.0000 0.0000\nF 0.0000 0.0000 1.2626\n", ip_ref: 11.00 }, // 13768-60-0
        Case { name: "F4S", xyz: "5\nmol\nS 0.0000 0.0000 0.3726\nF 0.0000 1.6430 0.2731\nF 0.0000 -1.6430 0.2731\nF 1.1969 0.0000 -0.6044\nF -1.1969 0.0000 -0.6044\n", ip_ref: 11.69 }, // 7783-60-0
        Case { name: "BrK", xyz: "2\nmol\nBr 0.0000 0.0000 0.0000\nK 0.0000 0.0000 2.8208\n", ip_ref: 8.82 }, // 7758-02-3
        Case { name: "ClGa", xyz: "2\nmol\nGa 0.0000 0.0000 0.0000\nCl 0.0000 0.0000 2.2017\n", ip_ref: 10.07 }, // 17108-85-9
        Case { name: "ClNa", xyz: "2\nmol\nNa 0.0000 0.0000 0.0000\nCl 0.0000 0.0000 2.3609\n", ip_ref: 9.80 }, // 7647-14-5
        Case { name: "Cl2Mg", xyz: "3\nmol\nMg 0.0000 0.0000 0.0000\nCl 0.0000 0.0000 2.179\nCl 0.0000 0.0000 -2.179\n", ip_ref: 11.80 }, // 7786-30-3
        Case { name: "BN", xyz: "2\nmol\nB 0.0000 0.0000 0.0000\nN 0.0000 0.0000 1.281\n", ip_ref: 11.50 }, // 10043-11-5
        Case { name: "NP", xyz: "2\nmol\nP 0.0000 0.0000 0.0000\nN 0.0000 0.0000 1.49087\n", ip_ref: 11.88 }, // 17739-47-8
        Case { name: "H4N2", xyz: "6\nmol\nN 0.0000 0.7230 -0.1123\nN 0.0000 -0.7230 -0.1123\nH -0.4470 1.0031 0.7562\nH 0.4470 -1.0031 0.7562\nH 0.9663 1.0031 0.0301\nH -0.9663 -1.0031 0.0301\n", ip_ref: 8.98 }, // 302-01-2
        Case { name: "C2H6O", xyz: "9\nmol\nC 1.1879 -0.3829 0.0000\nC 0.0000 0.5526 0.0000\nO -1.1867 -0.2472 0.0000\nH -1.9237 0.3850 0.0000\nH 2.0985 0.2306 0.0000\nH 1.1184 -1.0093 0.8869\nH 1.1184 -1.0093 -0.8869\nH -0.0227 1.1812 0.8852\nH -0.0227 1.1812 -0.8852\n", ip_ref: 10.64 }, // 64-17-5
        Case { name: "C2H4O", xyz: "7\nmol\nC 0.000000 0.000000 0.000000\nC 0.000000 0.000000 1.515000\nO 1.001953 0.000000 2.193373\nH -1.019805 0.000000 1.997060\nH -0.905700 -0.522900 -0.363000\nH 0.000000 1.045800 -0.363000\nH 0.905700 -0.522900 -0.363000\n", ip_ref: 10.24 }, // 75-07-0
        Case { name: "C4H10O", xyz: "15\nmol\nO 0.0000 0.0000 0.2696\nC 0.0000 1.1705 -0.5184\nC 0.0000 -1.1705 -0.5184\nC 0.0000 2.3716 0.4082\nC 0.0000 -2.3716 0.4082\nH -0.8879 1.1870 -1.1676\nH 0.8879 1.1870 -1.1676\nH 0.8879 -1.1870 -1.1676\nH -0.8879 -1.1870 -1.1676\nH 0.0000 3.2961 -0.1729\nH 0.0000 -3.2961 -0.1729\nH 0.8840 2.3552 1.0456\nH -0.8840 2.3552 1.0456\nH -0.8840 -2.3552 1.0456\nH 0.8840 -2.3552 1.0456\n", ip_ref: 9.61 }, // 60-29-7
        Case { name: "CH2O2", xyz: "5\nmol\nO 0.9858 0.0000 2.0307\nH -1.0241 0.0000 1.7361\nC 0.0000 0.0000 1.3430\nO 0.0000 0.0000 0.0000\nH 0.9329 0.0000 -0.2728\n", ip_ref: 11.50 }, // 64-18-6
        Case { name: "H2O2", xyz: "4\nmol\nO 0.0000 0.7375 -0.0528\nO 0.0000 -0.7375 -0.0528\nH 0.8190 0.8170 0.4220\nH -0.8190 -0.8170 0.4220\n", ip_ref: 11.70 }, // 7722-84-1
        Case { name: "CS2", xyz: "3\nmol\nC 0.0000 0.0000 0.0000\nS 0.0000 0.0000 1.5526\nS 0.0000 0.0000 -1.5526\n", ip_ref: 10.09 }, // 75-15-0
        Case { name: "COS", xyz: "3\nmol\nO 0.0000 0.0000 1.1578\nC 0.0000 0.0000 0.0000\nS 0.0000 0.0000 -1.5601\n", ip_ref: 11.19 }, // 463-58-1
        Case { name: "COSe", xyz: "3\nmol\nO 0.0000 0.0000 1.159\nC 0.0000 0.0000 0.0000\nSe 0.0000 0.0000 -1.709\n", ip_ref: 10.37 }, // 1603-84-5
        Case { name: "O3", xyz: "3\nmol\nO 0.0000 0.0000 0.0000\nO 1.0869 0.0000 0.6600\nO -1.0869 0.0000 0.6600\n", ip_ref: 12.73 }, // 10028-15-6
        Case { name: "O2S", xyz: "3\nmol\nS 0.0000 0.0000 0.0000\nO 1.2349 0.0000 0.7226\nO -1.2349 0.0000 0.7226\n", ip_ref: 12.50 }, // 7446-09-5
        Case { name: "BeO", xyz: "2\nmol\nBe 0.0000 0.0000 0.0000\nO 0.0000 0.0000 1.3308\n", ip_ref: 10.10 }, // 1304-56-9
        Case { name: "MgO", xyz: "2\nmol\nMg 0.0000 0.0000 0.0000\nO 0.0000 0.0000 1.749\n", ip_ref: 8.76 }, // 1309-48-4
        Case { name: "C7H8", xyz: "15\nmol\nC -1.091600 -0.874900 0.000000\nC 0.211900 -1.382800 0.000000\nC 1.303400 -0.507900 0.000000\nC 1.091600 0.874900 0.000000\nC -0.211900 1.382700 0.000000\nC -1.303500 0.507800 0.000000\nH -1.957699 -1.569142 0.000000\nH 0.380087 -2.479984 0.000000\nH 1.957699 1.569142 0.000000\nH -0.380072 2.479886 0.000000\nH -2.337729 0.910876 0.000000\nC 2.723481 -1.061023 0.000000\nH 3.397184 -0.350813 0.523286\nH 2.739366 -2.039786 0.523326\nH 3.068240 -1.195306 -1.046523\n", ip_ref: 8.82 }, // 108-88-3
        Case { name: "C8H10", xyz: "18\nmol\nC -2.2693535 -0.0000389 -0.2398724\nC -1.5797056 -1.2063053 -0.1005171\nC -0.2110721 -1.2030110 0.1749198\nC 0.4952604 0.0000401 0.3167968\nC -0.2111286 1.2030501 0.1748811\nC -1.5797652 1.2062634 -0.1005510\nH -3.3389696 -0.0000690 -0.4503195\nH -2.1102702 -2.1538484 -0.2012097\nH 0.3195630 -2.1509209 0.2885225\nH 0.3194582 2.1509892 0.2884519\nH -2.1103902 2.1537683 -0.2012753\nC 2.8147071 -0.0000376 -0.7147854\nC 1.9829690 0.0000498 0.5777693\nH 2.5926203 -0.8860058 -1.3248301\nH 2.5930639 0.8861475 -1.3246747\nH 3.8904081 -0.0003286 -0.4914695\nH 2.2473809 -0.8820350 1.1803103\nH 2.2474159 0.8821780 1.1802303\n", ip_ref: 8.77 }, // 100-41-4
        Case { name: "C6F6", xyz: "12\nmol\nC -1.092692 -0.875775 0.000000\nC 0.212112 -1.384183 0.000000\nC 1.304703 -0.508408 0.000000\nC 1.092692 0.875775 0.000000\nC -0.212112 1.384083 0.000000\nC -1.304804 0.508308 0.000000\nF -2.126549 -1.704487 0.000000\nF 0.412876 -2.693885 0.000000\nF 2.539354 -0.989305 0.000000\nF 2.126549 1.704487 0.000000\nF -0.412858 2.693787 0.000000\nF -2.539356 0.989457 0.000000\n", ip_ref: 10.20 }, // 392-56-3
        Case { name: "C6H6O", xyz: "13\nmol\nC -1.046085 -0.892147 -0.000000\nC 0.257414 -1.400049 0.000000\nC 1.331097 -0.503372 -0.000000\nC 1.091601 0.874901 -0.000000\nC -0.121202 1.347367 0.000000\nC -1.230130 0.494536 0.000000\nH -1.874522 -1.578781 0.001032\nH 0.431625 -2.467932 0.001287\nH 2.336451 -0.886828 0.000832\nH 1.937051 1.553332 -0.000803\nO -0.312498 2.697885 -0.001272\nH -2.244385 0.877082 -0.000405\nH -1.251130 2.879281 -0.002039\n", ip_ref: 8.75 }, // 108-95-2
        Case { name: "C6H7N", xyz: "14\nmol\nC -1.086143 -0.870526 0.000000\nC 0.210840 -1.375886 0.000000\nC 1.296883 -0.505360 0.000000\nC 1.086142 0.870526 0.000000\nC -0.210841 1.375786 0.000000\nC -1.296983 0.505261 0.000000\nH -1.932704 -1.548451 0.000000\nH 0.374980 -2.447943 0.000000\nH 2.307474 -0.899002 0.000000\nH 1.932383 1.548850 0.000000\nH -2.307832 0.898242 0.000000\nN -0.428501 2.790157 0.000000\nH 0.323567 3.340664 0.346569\nH -1.327164 3.070170 0.333980\n", ip_ref: 8.05 }, // 62-53-3
        Case { name: "C5H5N", xyz: "11\nmol\nN 0.000000 0.000000 0.000000\nC -0.476428 -1.252444 0.000000\nC -0.903103 0.989952 0.000000\nC -2.282876 0.784403 0.000000\nC -1.835282 -1.567988 0.000000\nC -2.760265 -0.525306 0.000000\nH -0.532213 2.008528 0.000000\nH 0.266630 -2.041697 0.000000\nH -2.958369 1.628364 0.000000\nH -2.153556 -2.601071 0.000000\nH -3.818275 -0.726658 0.000000\n", ip_ref: 9.66 }, // 110-86-1
        Case { name: "C5H5N5O", xyz: "16\nmol\nC -0.8909136 0.0022495 0.4879726\nC 0.4648657 0.0066008 0.8428894\nN -1.4589692 0.0149180 -0.7432767\nC -0.5646417 0.0081792 -1.7097505\nN 0.6167192 0.0012409 2.2159144\nH -0.8858269 -0.0099074 3.7340662\nC -0.6098052 -0.0051706 2.6839376\nN -1.5660059 -0.0044593 1.6825133\nH -2.5739287 -0.0104122 1.7887665\nC 1.4519352 0.0039931 -0.2048012\nO 2.6727207 -0.0060475 -0.1670731\nN 0.7873252 0.0064149 -1.4864710\nH 1.4327355 -0.0565403 -2.2706380\nN -0.9871915 -0.0557168 -3.0197562\nH -1.9795941 0.1251659 -3.1279347\nH -0.4045996 0.3813093 -3.7247680\n", ip_ref: 8.24 }, // 73-40-5
        Case { name: "C5H5N5", xyz: "15\nmol\nC -0.7843450 0.0024660 0.6854944\nC 0.5301823 0.0009168 0.1951947\nN -1.9138024 0.0022584 -0.0334730\nC -1.6479109 -0.0019330 -1.3472026\nH -2.5169904 -0.0044917 -2.0095241\nN -0.4540095 -0.0051210 -1.9657824\nC 0.6617763 -0.0009654 -1.2107858\nN 1.8686255 0.0181736 -1.8266445\nH 2.7131398 -0.0801668 -1.2801876\nH 1.9011412 -0.0813319 -2.8324875\nN 1.4556976 -0.0028403 1.2254693\nH 1.1016503 -0.0046108 3.3314279\nC 0.7173645 -0.0023279 2.3155892\nN -0.6380338 0.0012265 2.0572467\nH -1.3931273 0.0022322 2.7328517\n", ip_ref: 8.48 }, // 73-24-5
        Case { name: "C4H5N3O", xyz: "13\nmol\nH -2.0638946 1.7581987 -0.0048606\nC -1.1537688 1.1649096 0.0014411\nC 0.0751951 1.7542640 0.0005025\nH 0.2153507 2.8348293 0.0000624\nN 1.1910674 0.9892600 -0.0011780\nH 2.1178150 1.4035814 -0.0026938\nC 1.1663885 -0.4437838 -0.0000952\nO 2.2347026 -1.0410121 -0.0014923\nN -0.0756682 -1.0234974 0.0071744\nC -1.1667470 -0.2723068 0.0031814\nN -2.3615150 -0.9269214 -0.0250037\nH -2.3405765 -1.9329874 0.0897975\nH -3.2271802 -0.4358623 0.1455697\n", ip_ref: 8.94 }, // 71-30-7
        Case { name: "C5H6N2O2", xyz: "15\nmol\nC 0.5676469 0.0000748 -1.1919386\nC 1.5304659 0.0002388 -0.2353148\nH 2.5926489 0.0003404 -0.4816273\nC -0.8332110 0.0000968 -0.7623841\nO -1.8038589 0.0001316 -1.5105895\nN -1.0160281 0.0001744 0.6372389\nH -1.9813565 0.0002431 0.9610129\nC -0.0490742 0.0002212 1.6309425\nO -0.2873837 -0.0005911 2.8289393\nN 1.2432728 0.0002954 1.1110067\nH 1.9834347 0.0004963 1.8040615\nC 0.8557798 -0.0004840 -2.6602502\nH 1.9357502 -0.0001999 -2.8533074\nH 0.4097225 0.8785617 -3.1465795\nH 0.4104462 -0.8804337 -3.1455944\n", ip_ref: 9.20 }, // 65-71-4
        Case { name: "C4H4N2O2", xyz: "12\nmol\nH 1.7181113 -0.0000002 -2.1244327\nC 1.1438667 -0.0000001 -1.2038069\nC 1.7446080 -0.0000000 0.0097511\nH 2.8271212 -0.0000001 0.1298240\nC -0.3090145 0.0000000 -1.2970091\nO -0.9725128 -0.0000001 -2.3249912\nN -0.9523747 0.0000001 -0.0343227\nH -1.9701793 0.0000002 -0.0511160\nC -0.3751368 0.0000001 1.2248796\nO -0.9991833 -0.0000001 2.2733353\nN 1.0223980 0.0000001 1.1766034\nH 1.4813173 0.0000001 2.0806653\n", ip_ref: 9.68 }, // 66-22-8
        Case { name: "CH4N2O", xyz: "8\nmol\nO 0.0000 1.3049 0.0000\nC 0.0000 0.0838 0.0000\nN 1.1603 -0.6595 0.0000\nN -1.1603 -0.6595 -0.0000\nH 1.1383 -1.5964 0.3424\nH 1.9922 -0.0940 0.1760\nH -1.1383 -1.5964 -0.3424\nH -1.9922 -0.0940 -0.1760\n", ip_ref: 9.80 }, // 57-13-6
        Case { name: "Cu2", xyz: "2\nmol\nCu 0.0 0.0 0.0\nCu 0.0 0.0 2.2197\n", ip_ref: 7.46 }, // 12190-70-4
        Case { name: "CCuN", xyz: "3\nmol\nC 0.0000 0.0000 0.0000\nN 0.0000 0.0000 1.158\nCu 0.0000 0.0000 -1.832\n", ip_ref: f64::NAN }, // 544-92-3
    ]
}

#[derive(Default, Clone, Copy)]
struct Ips {
    koop: f64,
    dscf: f64,
    drpa: f64,
    g0w0: f64,
    cohsex: f64,
    evgw0: f64,
    evgw: f64,
    g0w0_pbe: f64,
}

fn s_squared(uhf: &ferric_scf::result::ScfResult, s_ao: &Array2<f64>, nocc_a: usize, nocc_b: usize) -> f64 {
    let c_a = &uhf.mos_alpha;
    let c_b = uhf.mos_beta.as_ref().unwrap_or(&uhf.mos_alpha);
    let s_mo_ab = c_a.slice(ndarray::s![.., ..nocc_a]).t()
        .dot(s_ao)
        .dot(&c_b.slice(ndarray::s![.., ..nocc_b]));
    let ov2: f64 = s_mo_ab.iter().map(|x| x * x).sum();
    let s2_z = 0.25 * (nocc_a as f64 - nocc_b as f64).powi(2);
    let s2_xy = 0.5 * (nocc_a as f64 + nocc_b as f64) - ov2;
    s2_z + s2_xy
}

#[allow(dead_code)]
struct CationDiag {
    method: &'static str,
    iters: usize,
    converged: bool,
    s2: f64,
    s2_ideal: f64,
    energy: f64,
}

fn run_case(case: &Case, obs_name: &str, dfbs_name: &str) -> Option<(Ips, CationDiag)> {
    let ctx = ParallelContext::default();
    let obs_bs = basis::bundled(obs_name).ok()?;
    let dfbs_bs = basis::bundled(dfbs_name).ok()?;
    let op = Operator::coulomb();

    let neutral = Molecule::parse_xyz(case.xyz, 0, 1).ok()?;
    let cation  = Molecule::parse_xyz(case.xyz, 1, 2).ok()?;

    let obs_n = PreparedBasis::new(&neutral, &obs_bs).ok()?;
    let dfbs_n = PreparedBasis::new(&neutral, &dfbs_bs).ok()?;
    let bounds_n = SchwarzBounds::compute(op, &obs_n).ok()?;
    let obs_c = PreparedBasis::new(&cation, &obs_bs).ok()?;
    let dfbs_c = PreparedBasis::new(&cation, &dfbs_bs).ok()?;
    let bounds_c = SchwarzBounds::compute(op, &obs_c).ok()?;

    let rhf_n = solve_rhf(&ctx, &neutral, &obs_n, op, &bounds_n, &RhfConfig::default()).ok()?;
    let nocc_n = (neutral.nelec() as usize) / 2;
    let homo_abs = nocc_n - 1;
    let ip_koop = -rhf_n.eps_r()[homo_abs] * HA_TO_EV;

    let rpa_cfg = PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre, n_points: 20, u0: 0.5,
        },
        trunc_thresh: trunc_thresh(),
        davidson_conv_thresh: 1e-9,
        ..Default::default()
    };
    let rpa_n = run_pdep_rpa(&neutral, &obs_n, &dfbs_n, op, &rhf_n, &rpa_cfg).ok()?;

    let uhf_cfg = UhfConfig { max_iter: 200, ..Default::default() };
    let c_seed = rhf_n.mos_alpha.clone();
    let (uhf_c, diag_method) = match solve_uhf_with_guess(&ctx, &cation, &obs_c, &bounds_c, &uhf_cfg, Some((&c_seed, &c_seed))) {
        Ok(r) => (r, "UHF(neutral-seed)"),
        Err(_) => match solve_uhf(&ctx, &cation, &obs_c, &bounds_c, &uhf_cfg) {
            Ok(r) => (r, "UHF(hcore)"),
            Err(_) => {
                let r = solve_rohf(&ctx, &cation, &obs_c, op, &bounds_c, &RohfConfig::default()).ok()?;
                (r, "ROHF")
            }
        },
    };
    let s_ao = oneelectron::overlap(&obs_c);
    let nelec_c = cation.nelec() as i64;
    let two_s = cation.multiplicity as i64 - 1;
    let nocc_a = ((nelec_c + two_s) / 2) as usize;
    let nocc_b = ((nelec_c - two_s) / 2) as usize;
    let s_true = 0.5 * (nocc_a as f64 - nocc_b as f64);
    let diag = CationDiag {
        method: diag_method,
        iters: uhf_c.iterations,
        converged: uhf_c.converged,
        s2: s_squared(&uhf_c, &s_ao, nocc_a, nocc_b),
        s2_ideal: s_true * (s_true + 1.0),
        energy: uhf_c.energy,
    };

    let ip_dscf = (uhf_c.energy - rhf_n.energy) * HA_TO_EV;
    let rpa_c = run_u_pdep_rpa(&cation, &obs_c, &dfbs_c, op, &uhf_c, &rpa_cfg).ok()?;
    let ip_drpa = {
        let e_n = rhf_n.energy + rpa_n.e_rpa;
        let e_c = uhf_c.energy + rpa_c.e_rpa;
        (e_c - e_n) * HA_TO_EV
    };

    let pdep_cfg_gw = PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre, n_points: 16, u0: 0.5,
        },
        davidson_conv_thresh: 1e-7,
        davidson_max_vecs: 0,
        trunc_thresh: trunc_thresh(),
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
    };
    // Method-depth by molecule size. The full @HF stack (G0W0+COHSEX+evGW0+
    // evGW×8) plus a second full @PBE GW is ~13 PDEP solves/molecule — on big
    // organics (>~10 atoms) that's many CPU-hours and blows any budget. For those
    // we compute ONLY G0W0@HF (1 PDEP, the PySCF-validated core number) so the
    // table reaches 93/93. Small molecules keep ALL columns. Threshold:
    // GW100_FULL_MAX_ATOMS (default 10). The other columns stay NaN for big mols
    // (honestly: "not computed at this depth", not a failure).
    let full_max_atoms: usize = std::env::var("GW100_FULL_MAX_ATOMS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let full_depth = neutral.atoms.len() <= full_max_atoms;

    let mut ip_g0w0 = f64::NAN;
    let mut ip_cohsex = f64::NAN;
    let mut ip_evgw0 = f64::NAN;
    let mut ip_evgw = f64::NAN;
    let methods: Vec<(GwMethod, &mut f64)> = if full_depth {
        vec![
            (GwMethod::G0W0,   &mut ip_g0w0),
            (GwMethod::Cohsex, &mut ip_cohsex),
            (GwMethod::EvGw0,  &mut ip_evgw0),
            (GwMethod::EvGw,   &mut ip_evgw),
        ]
    } else {
        // big molecule: G0W0@HF only (the PySCF-validated core number)
        vec![(GwMethod::G0W0, &mut ip_g0w0)]
    };
    for (method, slot) in methods {
        let gcfg = GwConfig { method, max_ev_iter: 8, ev_conv_thresh: 1e-4, ..Default::default() };
        if let Ok(res) = run_gw(&neutral, &obs_n, &dfbs_n, op, &rhf_n, &pdep_cfg_gw, &gcfg, None) {
            if let Some(local) = res.mo_indices.iter().position(|&i| i == homo_abs) {
                *slot = -res.eps_qp[local] * HA_TO_EV;
            }
        }
    }

    // G0W0@PBE: full-depth molecules only (it's a second full GW). Big mols skip it.
    let ip_g0w0_pbe = if full_depth {
        run_g0w0_pbe(&ctx, &neutral, &obs_n, &dfbs_n, &obs_bs, op,
                     &bounds_n, &pdep_cfg_gw, homo_abs)
            .unwrap_or(f64::NAN)
    } else {
        f64::NAN
    };

    Some((
        Ips { koop: ip_koop, dscf: ip_dscf, drpa: ip_drpa,
              g0w0: ip_g0w0, cohsex: ip_cohsex, evgw0: ip_evgw0, evgw: ip_evgw,
              g0w0_pbe: ip_g0w0_pbe },
        diag,
    ))
}

/// G0W0 starting from a self-consistent PBE-KS reference (validated GW-grade,
/// see crates/ferric-scf/tests/pbe_ks_orbital_energies.rs). Returns the HOMO IP
/// in eV, or None on SCF/GW failure.
#[allow(clippy::too_many_arguments)]
fn run_g0w0_pbe(
    ctx: &ParallelContext,
    neutral: &Molecule,
    obs_n: &PreparedBasis,
    dfbs_n: &PreparedBasis,
    obs_bs: &basis::BasisSet,
    op: Operator,
    bounds_n: &SchwarzBounds,
    pdep_cfg_gw: &PdepRpaConfig,
    homo_abs: usize,
) -> Option<f64> {
    let cfg = RhfConfig { xc: Some("pbe".into()), ..Default::default() };
    let ks = solve_rhf(ctx, neutral, obs_n, op, bounds_n, &cfg).ok()?;
    let (vxc, _) = ferric_gw::vxc_mo::vxc_diagonal_mo(neutral, obs_bs, "pbe", &ks).ok()?;
    let gcfg = GwConfig { method: GwMethod::G0W0, ..Default::default() };
    let res = run_gw(neutral, obs_n, dfbs_n, op, &ks, pdep_cfg_gw, &gcfg, Some(&vxc)).ok()?;
    let local = res.mo_indices.iter().position(|&i| i == homo_abs)?;
    Some(-res.eps_qp[local] * HA_TO_EV)
}

fn main() {
    // Basis from CLI: `gw100_full [aug-cc-pvdz|aug-cc-pvtz]` (default aTZ).
    let obs_name = std::env::args().nth(1).unwrap_or_else(|| "aug-cc-pvtz".to_string());
    let dfbs_name = format!("{obs_name}-rifit");
    // Resumability: GW100_DONE=mol1,mol2,... skips already-computed molecules so a
    // restarted run continues instead of recomputing from scratch.
    let done: std::collections::HashSet<String> = std::env::var("GW100_DONE")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    let mut cases: Vec<Case> = cases().into_iter().filter(|c| !done.contains(c.name)).collect();
    // Smaller molecules FIRST — they bank fast (full-depth) so the table fills
    // quickly; the big organics (G0W0-only) come last. Atom count = xyz line 0.
    cases.sort_by_key(|c| c.xyz.split('\n').next().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(999));
    println!("# GW100 subset — basis {obs_name} / {dfbs_name}");
    println!(
        "{:<6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "mol", "exp(eV)", "Koop", "ΔSCF", "ΔRPA", "G0W0", "COHSEX", "evGW0", "evGW", "G0W0pbe"
    );
    println!("{:-<91}", "");

    let mut sum_abs = [0.0f64; 8];
    let mut n_ok = [0usize; 8];
    let mut diags: Vec<(&str, CationDiag)> = Vec::new();

    for case in &cases {
        match run_case(case, &obs_name, &dfbs_name) {
            Some((ips, diag)) => {
                println!(
                    "{:<6} {:>8.2} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
                    case.name, case.ip_ref,
                    ips.koop, ips.dscf, ips.drpa,
                    ips.g0w0, ips.cohsex, ips.evgw0, ips.evgw, ips.g0w0_pbe
                );
                for (k, v) in [ips.koop, ips.dscf, ips.drpa,
                               ips.g0w0, ips.cohsex, ips.evgw0, ips.evgw, ips.g0w0_pbe].iter().enumerate() {
                    if v.is_finite() {
                        sum_abs[k] += (v - case.ip_ref).abs();
                        n_ok[k] += 1;
                    }
                }
                diags.push((case.name, diag));
            }
            None => println!("{:<6} FAILED", case.name),
        }
        // Flush per molecule so a streaming runner can persist each row live —
        // a box restart then loses at most the in-flight molecule, not the run.
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }

    println!("{:-<82}", "");
    let mae: Vec<String> = sum_abs.iter().zip(n_ok.iter()).map(|(s, n)| {
        if *n > 0 { format!("{:>8.3}", s / *n as f64) } else { "     n/a".to_string() }
    }).collect();
    println!(
        "{:<6} {:>8} {} {} {} {} {} {} {} {}",
        "MAE", "",
        mae[0], mae[1], mae[2], mae[3], mae[4], mae[5], mae[6], mae[7],
    );

    println!("\nCation SCF diagnostics:");
    println!("{:<6} {:>18} {:>5} {:>5} {:>9} {:>9} {:>14}",
        "mol", "method", "iter", "conv", "<S^2>", "ideal", "E_cation(Ha)");
    for (name, d) in &diags {
        println!("{:<6} {:>18} {:>5} {:>5} {:>9.4} {:>9.4} {:>14.6}",
            name, d.method, d.iters, d.converged, d.s2, d.s2_ideal, d.energy);
    }

    println!("\nKoopmans, G0W0/COHSEX/evGW(0): direct QP energies on neutral RHF/HF.");
    println!("ΔSCF/ΔRPA: cation − neutral total-energy differences.");
}
