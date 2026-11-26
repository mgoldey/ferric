use ferric_cuest::workspace::WorkspacePlan;

#[test]
fn workspace_plan_total_and_budget() {
    let w = WorkspacePlan::new(100, 50);
    assert_eq!(w.total_bytes(), 150);
    assert!(w.fits_budget(200));
    assert!(!w.fits_budget(100));
}
