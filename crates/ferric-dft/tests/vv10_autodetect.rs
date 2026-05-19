//! libxc's xc_nlc_coef should return (b=6.0, C=0.01) for wB97X-V and
//! zeros for VV10-free functionals.

use ferric_dft::libxc::{xc_def_from_name, XcFunctional};

#[test]
fn wb97xv_carries_vv10() {
    let def = xc_def_from_name("wB97X-V").unwrap();
    let v = def.vv10.expect("wB97X-V must carry VV10");
    eprintln!("wB97X-V vv10 (from libxc xc_nlc_coef): b={}, C={}", v.b, v.c);
    assert!((v.b - 6.0).abs() < 1e-10, "b should be 6.0, got {}", v.b);
    assert!((v.c - 0.01).abs() < 1e-10, "C should be 0.01, got {}", v.c);
}

#[test]
fn raw_wb97x_v_libxc_name_carries_vv10() {
    let def = xc_def_from_name("HYB_GGA_XC_WB97X_V").unwrap();
    let v = def.vv10.expect("HYB_GGA_XC_WB97X_V must carry VV10");
    assert!((v.b - 6.0).abs() < 1e-10);
    assert!((v.c - 0.01).abs() < 1e-10);
}

#[test]
fn b3lyp_does_not_carry_vv10() {
    let def = xc_def_from_name("B3LYP").unwrap();
    assert!(def.vv10.is_none(), "B3LYP should not carry VV10");
}

#[test]
fn pbe_does_not_carry_vv10() {
    let def = xc_def_from_name("PBE").unwrap();
    assert!(def.vv10.is_none(), "PBE should not carry VV10");
}

#[test]
fn direct_xc_functional_query() {
    let f = XcFunctional::new("HYB_GGA_XC_WB97X_V", 1).unwrap();
    let v = f.vv10_coeffs().expect("wB97X-V should have VV10");
    eprintln!("direct vv10 query on HYB_GGA_XC_WB97X_V: b={}, C={}", v.b, v.c);
    assert!((v.b - 6.0).abs() < 1e-10);
    assert!((v.c - 0.01).abs() < 1e-10);
}
