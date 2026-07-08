//! SR-MP2 three-index sparsity spike (corrected target).
//!
//! The first spike measured the *pair list* (Q(i,j), overlap-driven, operator-
//! blind) and correctly showed erfc ≈ Coulomb — the locality is NOT there.
//!
//! This one measures the real SR-MP2 hot tensor: the three-index integrals
//! (P|op|μν). Locality here lives in the *bra-ket distance* between aux shell P
//! and orbital pair (μν), which only the distance-aware QQR-3 bound
//! (Schwarz × min(1,ext·ext/R) × op_decay(R)) can see. The bra-only Schwarz
//! bound (Q3[P]·Q(μ,ν)) is blind to it — confirmed on decane: Schwarz keeps the
//! identical fraction for erfc and Coulomb; QQR3 separates them.
//!
//! Falsifier: walk alkanes C1..C20 and count surviving shell-triples under the
//! QQR-3 bound for Coulomb vs erfc(ω=0.222). If the erfc *kept-per-aux-shell*
//! count PLATEAUS (each P sees a fixed-radius set of pairs ⇒ O(N) total) while
//! Coulomb's keeps climbing, the O(N) SR-MP2 premise HOLDS. If erfc's kept count
//! also grows linearly in nsh, it's O(N²) with a smaller prefactor — a constant
//! win, not a scaling one.
//!
//! Counting only walks the bound (estimate3) — no integral evaluation — so the
//! whole series runs in seconds.
//!
//! Run: cargo run --release -p ferric-integrals --example sr_eri3_scaling

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::qqr3::QqrBounds3;

const OMEGA_BOHR: f64 = 0.222; // 0.420 Å⁻¹, production erfc-optimal
const THRESH: f64 = 1e-10; // production 3-index screening threshold

/// Count shell-triples (P, s1, s2≤s1) whose QQR-3 bound clears `thresh`.
/// This mirrors the loop in eri3_tensor_screened_qqr exactly, minus the
/// integral evaluation — so the count is the production-screened triple count.
fn count_triples(
    op: Operator,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
) -> (usize, usize) {
    let bounds = QqrBounds3::new(op, mol, obs, dfbs).unwrap();
    let nsh_obs = obs.nshells();
    let nsh_df = dfbs.nshells();
    let mut kept = 0usize;
    let mut total = 0usize;
    for sp in 0..nsh_df {
        for s1 in 0..nsh_obs {
            for s2 in 0..=s1 {
                total += 1;
                if bounds.estimate3(sp, s1, s2) >= THRESH {
                    kept += 1;
                }
            }
        }
    }
    (kept, total)
}

fn main() {
    let bs = basis::bundled("cc-pvdz").unwrap();
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let coul = Operator::coulomb();
    let erfc = Operator::erfc(OMEGA_BOHR);

    println!(
        "# SR 3-index scaling — cc-pVDZ/cc-pVDZ-RI, QQR-3 bound, ω={OMEGA_BOHR} Bohr⁻¹, thresh={THRESH:.0e}\n\
         # kept/aux = surviving (P,μν) shell-triples per aux shell. erfc PLATEAU ⇒ O(N); CLIMB ⇒ O(N²).\n"
    );
    println!(
        "{:>4} {:>5} {:>6} | {:>11} {:>8} {:>7} | {:>11} {:>8} {:>7}",
        "C", "nshA", "nshO", "coul_kept", "c/aux", "c_pct", "erfc_kept", "e/aux", "e_pct"
    );
    println!("{}", "-".repeat(86));

    for n in 1..=20usize {
        let path = format!("testdata/molecules/alkane_{n}.xyz");
        let mol = match Molecule::load_xyz(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux).unwrap();
        let nsh_aux = dfbs.nshells();
        let nsh_obs = obs.nshells();

        let (ck, ct) = count_triples(coul, &mol, &obs, &dfbs);
        let (ek, _) = count_triples(erfc, &mol, &obs, &dfbs);

        println!(
            "{:>4} {:>5} {:>6} | {:>11} {:>8.1} {:>6.1}% | {:>11} {:>8.1} {:>6.1}%",
            n, nsh_aux, nsh_obs,
            ck, ck as f64 / nsh_aux as f64, 100.0 * ck as f64 / ct as f64,
            ek, ek as f64 / nsh_aux as f64, 100.0 * ek as f64 / ct as f64,
        );
    }

    println!(
        "\n# Read the c/aux and e/aux columns: a CONSTANT e/aux past ~C10 (while\n\
         # c/aux keeps rising) is the O(N) signature. Both rising ⇒ O(N²).\n\
         #\n\
         # MEASURED VERDICT (valid ext_sum/R_eff bound, 2026-06-17): e/aux RISES\n\
         # monotonically (171→13490 over C1..C20, no plateau) and erfc keeps\n\
         # 99.0%% of the Coulomb triples at C20. The erfc 1st difference grows\n\
         # ~linearly (2nd diff ~+44k, NOT ~0), so the erfc 3-index count is\n\
         # O(N²) — the same scaling as Coulomb with a ~1%% smaller prefactor.\n\
         # Under a VALID bound, linear-SR-MP2 locality at ω=0.222 is DEAD: erfc\n\
         # attenuation buys a constant factor, not a scaling win. (The earlier\n\
         # O(N) / 201472-per-CH2 claim was an artifact of the invalid bound that\n\
         # over-dropped long-range triples.)"
    );
}
