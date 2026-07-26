//! libxc external-parameter FFI, and the ωB97X-L-V definition built on it.
//!
//! ωB97X-L-V (Ransford & Carter-Fenk, PCCP 2026, 28, 14428) is stock ωB97X-V's
//! functional FORM with re-fitted coefficients, so it is expressible through libxc's
//! external-parameter interface. That means libxc supplies the analytic vrho/vsigma
//! derivatives instead of a hand-written B97 kernel.

use ferric_dft::libxc::{
    wb97x_l_v_def, xc_def_from_name, XcFunctional, WB97X_L_V_EXT_PARAMS, WB97X_L_V_LAMBDA,
};

/// Evaluate exc on an unpolarized GGA grid, returning the energy density per point.
fn exc_of(f: &XcFunctional, rho: &[f64], sigma: &[f64]) -> Vec<f64> {
    let n = rho.len();
    let (mut exc, mut vrho, mut vsigma) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    f.eval_gga_unpolarized(rho, sigma, &mut exc, &mut vrho, &mut vsigma);
    exc
}

/// The parameter names and their ORDER are the contract for positional application.
/// If libxc ever reorders them, positional writes would silently scramble the
/// coefficients -- so pin the order here.
#[test]
fn wb97xv_exposes_the_expected_ext_params_in_order() {
    let f = XcFunctional::new("HYB_GGA_XC_WB97X_V", 1).unwrap();
    let names = f.ext_param_names();

    assert_eq!(
        names.len(),
        18,
        "expected 18 external parameters (5 exchange + 5 same-spin + 5 opposite-spin \
         + alpha/beta/omega), got {}: {names:?}",
        names.len()
    );
    let expected: Vec<&str> = WB97X_L_V_EXT_PARAMS.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, expected, "libxc parameter ordering changed");
}

/// Setting parameters must actually change what libxc evaluates.
#[test]
fn set_ext_params_changes_evaluated_energy() {
    let rho = [0.31_f64, 0.12, 0.07];
    let sigma = [0.021_f64, 0.009, 0.004];

    let stock = XcFunctional::new("HYB_GGA_XC_WB97X_V", 1).unwrap();
    let e_stock = exc_of(&stock, &rho, &sigma);

    let mut custom = XcFunctional::new("HYB_GGA_XC_WB97X_V", 1).unwrap();
    let values: Vec<f64> = WB97X_L_V_EXT_PARAMS.iter().map(|(_, v)| *v).collect();
    custom.set_ext_params(&values).unwrap();
    let e_custom = exc_of(&custom, &rho, &sigma);

    eprintln!("stock  wB97X-V  exc = {e_stock:?}");
    eprintln!("custom wB97X-L-V exc = {e_custom:?}");
    for (i, (a, b)) in e_stock.iter().zip(e_custom.iter()).enumerate() {
        assert!(
            (a - b).abs() > 1e-10,
            "point {i}: ext params had no effect ({a} vs {b}) -- the override is a no-op"
        );
        assert!(a.is_finite() && b.is_finite(), "point {i}: non-finite exc");
    }
}

/// Round-trip: what we set is what libxc reports back as applied.
#[test]
fn applied_ext_params_round_trip() {
    let mut f = XcFunctional::new("HYB_GGA_XC_WB97X_V", 1).unwrap();
    assert!(f.applied_ext_params().is_none(), "fresh handle should report no overrides");

    let values: Vec<f64> = WB97X_L_V_EXT_PARAMS.iter().map(|(_, v)| *v).collect();
    f.set_ext_params(&values).unwrap();
    assert_eq!(f.applied_ext_params().unwrap(), values.as_slice());

    // Defaults must differ from our overrides, else the test above proves nothing.
    let defaults = f.ext_param_defaults();
    assert_ne!(defaults, values, "wB97X-L-V params coincide with stock wB97X-V defaults");
}

/// A wrong-length slice must be rejected, NOT passed to libxc.
///
/// `xc_func_set_ext_params` takes no length argument and does no bounds checking,
/// so a short slice would be an out-of-bounds read inside libxc. The length check
/// in the wrapper is a memory-safety guard, not ergonomics.
#[test]
fn wrong_length_ext_params_is_rejected() {
    let mut f = XcFunctional::new("HYB_GGA_XC_WB97X_V", 1).unwrap();
    assert!(f.set_ext_params(&[0.4, 0.154]).is_err(), "short slice must be rejected");
    assert!(f.set_ext_params(&[0.0; 19]).is_err(), "long slice must be rejected");
    assert!(f.applied_ext_params().is_none(), "rejected calls must not record params");
}

/// THE PARALLEL-CLONE TRAP.
///
/// `par_chunks` builds a fresh libxc handle per rayon worker via `from_id`, and a
/// fresh handle carries libxc's BUILT-IN defaults. If the overrides were not
/// replayed onto those clones, a parallel evaluation would silently use stock
/// wB97X-V coefficients while a serial one used wB97X-L-V -- a wrong energy with no
/// error raised anywhere.
///
/// This drives enough points to cross the parallel threshold and checks the result
/// against the serial path.
#[test]
fn ext_params_survive_the_parallel_worker_clone() {
    const N: usize = 20_000; // comfortably above PAR_MIN_PTS
    let rho: Vec<f64> = (0..N).map(|i| 0.05 + (i % 97) as f64 * 0.004).collect();
    let sigma: Vec<f64> = (0..N).map(|i| 0.001 + (i % 53) as f64 * 0.0007).collect();

    let values: Vec<f64> = WB97X_L_V_EXT_PARAMS.iter().map(|(_, v)| *v).collect();

    let mut custom = XcFunctional::new("HYB_GGA_XC_WB97X_V", 1).unwrap();
    custom.set_ext_params(&values).unwrap();
    let e_par = exc_of(&custom, &rho, &sigma);

    // Serial reference: evaluate a few points on a handle that never goes parallel.
    let probe: Vec<usize> = vec![0, 1, N / 3, N / 2, N - 1];
    let rho_s: Vec<f64> = probe.iter().map(|&i| rho[i]).collect();
    let sigma_s: Vec<f64> = probe.iter().map(|&i| sigma[i]).collect();
    let e_ser = exc_of(&custom, &rho_s, &sigma_s);

    // And the stock functional, to prove the comparison can distinguish them.
    let stock = XcFunctional::new("HYB_GGA_XC_WB97X_V", 1).unwrap();
    let e_stock = exc_of(&stock, &rho_s, &sigma_s);

    for (k, &i) in probe.iter().enumerate() {
        assert!(
            (e_par[i] - e_ser[k]).abs() < 1e-14,
            "point {i}: parallel {} != serial {} -- worker clones lost the ext params",
            e_par[i],
            e_ser[k]
        );
        assert!(
            (e_par[i] - e_stock[k]).abs() > 1e-12,
            "point {i}: custom result equals STOCK wB97X-V ({}), so the override was \
             silently dropped somewhere",
            e_stock[k]
        );
    }
}

/// The functional resolves by name, with the paper's CAM and VV10 values.
#[test]
fn wb97x_l_v_resolves_with_paper_parameters() {
    for name in ["wB97X-L-V", "WB97X_L_V", "wb97xlv"] {
        let def = xc_def_from_name(name)
            .unwrap_or_else(|e| panic!("{name} should resolve: {e}"));

        let cam = def.cam.expect("wB97X-L-V is range-separated; CAM must be present");
        assert!((cam.omega - 0.1).abs() < 1e-12, "{name}: omega {} != 0.1 a0^-1", cam.omega);
        // eqn (27): full long-range HF exchange, lambda-scaled short-range.
        assert!((cam.c_lr - 1.0).abs() < 1e-12, "{name}: c_lr {} != 1.0", cam.c_lr);
        assert!(
            (cam.c_sr - WB97X_L_V_LAMBDA).abs() < 1e-12,
            "{name}: c_sr {} != lambda {WB97X_L_V_LAMBDA}",
            cam.c_sr
        );

        let vv10 = def.vv10.expect("wB97X-L-V includes VV10 nonlocal correlation");
        assert!((vv10.b - 10.0).abs() < 1e-12, "{name}: VV10 b {} != 10.0", vv10.b);
        assert!((vv10.c - 0.01).abs() < 1e-12, "{name}: VV10 C {} != 0.01", vv10.c);

        assert_eq!(def.funcs.len(), 1);
        let applied = def.funcs[0]
            .applied_ext_params()
            .expect("the re-fitted coefficients must be applied, not left at stock values");
        let expected: Vec<f64> = WB97X_L_V_EXT_PARAMS.iter().map(|(_, v)| *v).collect();
        assert_eq!(applied, expected.as_slice(), "{name}: wrong coefficients applied");
    }
}

/// The uniform-electron-gas constraints from the paper must hold exactly.
///
/// These three coefficients are FIXED by lambda rather than fitted (paper: "we fix the
/// zero-order exchange coefficient cxs,0 = 1 - l and the corresponding zero-order
/// same-spin and opposite-spin correlation coefficients to css,0 = cab,0 = 1 - l^2 to
/// satisfy the uniform electron gas limit"). A typo in the table transcription would
/// break this relation while leaving every other test green.
#[test]
fn constrained_coefficients_satisfy_the_ueg_limit() {
    let get = |name: &str| -> f64 {
        WB97X_L_V_EXT_PARAMS
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("missing parameter {name}"))
            .1
    };
    let lambda = WB97X_L_V_LAMBDA;

    assert!(
        (get("_cx0") - (1.0 - lambda)).abs() < 1e-12,
        "_cx0 {} != 1 - lambda = {}",
        get("_cx0"),
        1.0 - lambda
    );
    for n in ["_css0", "_cos0"] {
        assert!(
            (get(n) - (1.0 - lambda * lambda)).abs() < 1e-12,
            "{n} {} != 1 - lambda^2 = {}",
            get(n),
            1.0 - lambda * lambda
        );
    }

    // Exchange mixing must also be consistent with lambda: c_sr = alpha + beta = lambda.
    let def = wb97x_l_v_def(1).unwrap();
    let cam = def.cam.unwrap();
    assert!(
        (cam.c_sr - lambda).abs() < 1e-12,
        "c_sr = alpha + beta = {} != lambda {lambda}",
        cam.c_sr
    );
}
