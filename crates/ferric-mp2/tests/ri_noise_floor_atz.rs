//! Empirical RI-fit noise-floor measurement for RS-MP2-RPA formulation B's
//! aug-cc-pVTZ "criterion met" result (benchmarks/a24-subset/README.md).
//!
//! Formulation B's headline number rests on a 0.143 -> 0.139 kcal/mol MAE
//! shift (0.004 kcal/mol) on a 4-dimer A24 subset at aug-cc-pVTZ, driven by
//! B's E_MP2[Coulomb] term (RI-fit, aug-cc-pvtz-rifit aux). This test asks:
//! how big is the RI-fit error on E_MP2[Coulomb] itself, on the SAME orbital
//! basis, relative to that 0.004 kcal/mol effect?
//!
//! Two probes, both isolating pure RI-fit error (no DF-JK SCF confound --
//! RHF is solved once with exact 4-index J/K and shared across all MP2
//! variants):
//!   (1) canonical (exact 4-index, O(N^5)) MP2[Coulomb] vs RI-MP2[Coulomb]
//!       with the aux basis the A24-subset aTZ sweep actually used
//!       (aug-cc-pvtz-rifit).
//!   (2) RI-MP2[Coulomb] with aug-cc-pvtz-rifit vs RI-MP2[Coulomb] with a
//!       different, independently-optimized aux basis for the same orbital
//!       level (def2-tzvpp-rifit) -- an aux-basis-swap noise probe.
//!
//! Two probe systems are run, both at aug-cc-pVTZ, same operator/aux as the
//! A24-subset aTZ sweep:
//!   - H2/aug-cc-pVTZ: cheap (small nbas), run first -- establishes whether
//!     the RI-fit floor is already visible at minimal molecule size.
//!   - water/aug-cc-pVTZ (~92 bf): the size-representative probe, closer to
//!     an actual A24 fragment; canonical MP2 here is expensive (the
//!     canonical_mp2 doc-comment warns "O(N^5) or worse... not intended for
//!     production use on large molecules" -- the AO->MO transform loop is a
//!     naive quadruple loop per shell quartet, so wall time is much worse
//!     than a clean O(N^5) scaling suggests). This is a shared, contended
//!     box, so run it separately from the H2 probe.
//!
//! Run with:
//!   OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo test -p ferric-mp2 --release --test ri_noise_floor_atz -- --ignored --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::canonical::canonical_mp2;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const WATER_XYZ: &str = "3\nwater optimized HF/cc-pVDZ\nO   0.000000   0.000000   0.117790\nH   0.000000   0.755453  -0.471161\nH   0.000000  -0.755453  -0.471161\n";
const H2_XYZ: &str = "2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n";

// kcal/mol per Hartree, matching benchmarks/a24-subset/run_a24.py's K constant.
const K_KCAL_PER_HA: f64 = 627.509474;

/// Shared probe: canonical (exact) vs RI-MP2[Coulomb] with two aux bases, at
/// aug-cc-pVTZ, for the given system. Prints results with kcal/mol context
/// against the 0.004 kcal/mol A24-subset aTZ effect size. No hard assertion
/// beyond finiteness -- this is a measurement test, not a regression gate.
fn run_probe(label: &str, xyz: &str) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("aug-cc-pvtz").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();

    // Exact 4-index J/K SCF (no DF-JK confound), tight convergence -- mirrors
    // the A24-subset aTZ sweep's SCF caveat #2 (exact 4-index, not RI-JK).
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ctx,
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            density_conv: 1e-9,
            ..Default::default()
        },
    )
    .unwrap();
    eprintln!("{label}/aug-cc-pVTZ RHF energy = {:.10}", rhf.energy);

    let cfg = RiMp2Config {
        frozen_core: 0,
        memory_budget_bytes: None,
        ..Default::default()
    };

    // Probe 1: canonical (exact, no RI) vs RI-MP2 with the aux basis the
    // A24-subset aTZ sweep used.
    let e_canonical = canonical_mp2(&mol, &obs, op, &rhf, 0).unwrap();

    let aux_atz = basis::bundled("aug-cc-pvtz-rifit").unwrap();
    let dfbs_atz = PreparedBasis::new(&mol, &aux_atz).unwrap();
    let ri_atz = ri_mp2(&mol, &obs, &dfbs_atz, op, &rhf, &cfg).unwrap();

    // Probe 2: RI-MP2 with a *different* reasonable aux basis for the same
    // orbital level (def2-TZVPP-rifit is an independently-optimized RI-MP2
    // aux set for triple-zeta orbital bases).
    let aux_def2 = basis::bundled("def2-tzvpp-rifit").unwrap();
    let dfbs_def2 = PreparedBasis::new(&mol, &aux_def2).unwrap();
    let ri_def2 = ri_mp2(&mol, &obs, &dfbs_def2, op, &rhf, &cfg).unwrap();

    let diff_canonical_vs_atzrifit_ha = (e_canonical - ri_atz.mp2_corr).abs();
    let diff_aux_swap_ha = (ri_atz.mp2_corr - ri_def2.mp2_corr).abs();
    let diff_canonical_vs_atzrifit_kcal = diff_canonical_vs_atzrifit_ha * K_KCAL_PER_HA;
    let diff_aux_swap_kcal = diff_aux_swap_ha * K_KCAL_PER_HA;

    eprintln!("=== RI-fit noise floor, {label}/aug-cc-pVTZ, E_MP2[Coulomb] ===");
    eprintln!("canonical (exact 4-index) MP2 corr = {e_canonical:.10} Ha");
    eprintln!(
        "RI-MP2 corr (aug-cc-pvtz-rifit)    = {:.10} Ha",
        ri_atz.mp2_corr
    );
    eprintln!(
        "RI-MP2 corr (def2-tzvpp-rifit)     = {:.10} Ha",
        ri_def2.mp2_corr
    );
    eprintln!(
        "|canonical - RI(aTZ-rifit)| = {diff_canonical_vs_atzrifit_ha:.3e} Ha = {diff_canonical_vs_atzrifit_kcal:.4} kcal/mol"
    );
    eprintln!(
        "|RI(aTZ-rifit) - RI(def2-tzvpp-rifit)| = {diff_aux_swap_ha:.3e} Ha = {diff_aux_swap_kcal:.4} kcal/mol"
    );
    eprintln!(
        "For reference, the A24-subset aTZ 'criterion met' effect size was 0.004 kcal/mol \
         (MP2 MAE 0.143 vs B MAE 0.139, benchmarks/a24-subset/README.md)."
    );

    // No hard assertion: this test's purpose is to PRINT the empirical noise
    // floor for human/report consumption, not to gate CI. It is #[ignore]d.
    // A soft sanity check that both diffs are finite guards against a
    // silently broken measurement.
    assert!(diff_canonical_vs_atzrifit_ha.is_finite());
    assert!(diff_aux_swap_ha.is_finite());
}

#[test]
#[ignore] // release-mode only; a few seconds of dense O(N^5) work
fn ri_fit_noise_floor_h2_augccpvtz() {
    run_probe("H2", H2_XYZ);
}

#[test]
#[ignore] // expensive: O(N^5)-or-worse canonical MP2 at aug-cc-pVTZ, water-sized
fn ri_fit_noise_floor_water_augccpvtz() {
    run_probe("water", WATER_XYZ);
}
