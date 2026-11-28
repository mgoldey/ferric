use ferric_cuest::grid::GridPlan;
use ferric_cuest::pairlist::PairListPlan;

#[test]
fn grid_plan_validate() {
    let g = GridPlan::new(75, 110);
    assert!(g.validate().is_ok());
    assert_eq!(g.slot_count(2), 2 * 75 * 110);
    let bad = GridPlan::new(0, 110);
    assert!(bad.validate().is_err());
}

#[test]
fn pairlist_plan_validate() {
    let p = PairListPlan::new(1e-12);
    assert!(p.validate().is_ok());
    let bad = PairListPlan::new(-1.0);
    assert!(bad.validate().is_err());
}
