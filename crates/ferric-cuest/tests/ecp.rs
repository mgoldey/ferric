use ferric_core::basis::bundled;
use ferric_cuest::ecp::convert_def;

#[test]
fn ecp_conversion_roundtrips_metadata() {
    let bs = bundled("cc-pvdz-pp").unwrap();
    // find an element with ECP
    let mut found = false;
    for z in bs.elements() {
        if let Some(def) = bs.ecp_for_element(z) {
            let c = convert_def(def).unwrap();
            assert_eq!(c.n_core, def.n_core);
            assert_eq!(c.shells.len(), def.shells.len());
            assert_eq!(c.max_angular_momentum(), def.max_angular_momentum());
            let expected: usize = def.shells.iter().map(|s| s.terms.len()).sum();
            assert_eq!(c.term_count(), expected);
            found = true;
            break;
        }
    }
    assert!(found, "no ECP in cc-pvdz-pp");
}

#[test]
fn ecp_conversion_rejects_bad_exponent() {
    use ferric_core::ecp::{EcpDef, EcpShell, EcpTerm};
    use ferric_cuest::ecp::convert_def;
    let def = EcpDef {
        n_core: 10,
        shells: vec![EcpShell {
            angular_momentum: 0,
            terms: vec![EcpTerm {
                coef: 1.0,
                r_exp: 0,
                gexp: -1.0,
            }],
        }],
    };
    assert!(convert_def(&def).is_err());
}
