//! SR-MP2 three-index *wall-time* falsifier.
//!
//! The companion counting spike (`sr_eri3_scaling`) proved the erfc(ω=0.222)
//! QQR-3 triple count grows O(N) while Coulomb grows O(N²). That is a count of
//! bound evaluations only — no integrals built. This example answers the
//! question Task 2 actually hinges on: *does screening the SR three-index build
//! buy real seconds?*
//!
//! For each alkane C8..C20 we time three integral builds at production
//! thresh=1e-10, cc-pVDZ/cc-pVDZ-RI:
//!   1. dense_coul   — eri3_tensor(Coulomb)               (the current SR path's cost shape)
//!   2. dense_erfc   — eri3_tensor(erfc)                  (the honest baseline for the Task-2 swap)
//!   3. qqr_erfc     — eri3_tensor_screened_qqr(erfc)     (bounds build + screened eval)
//!
//! Two speedup numbers are reported:
//!   - dense_coul / qqr_erfc : the headline "dense vs screened" the brief asked for.
//!   - dense_erfc / qqr_erfc : the *real* Task-2 swap (same operator, screened vs not).
//!     This is the number that decides whether wiring QQR3 into the erfc arm helps.
//!
//! The qqr_erfc column includes QqrBounds3::new (the bound-build cost is part of
//! what the production path would pay), so the ratio is honest end-to-end.
//!
//! Run:
//!   OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo run --release -p ferric-integrals --example sr_eri3_timing

use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::qqr3::QqrBounds3;
use ferric_integrals::threeindex::{eri3_tensor, eri3_tensor_screened_qqr};

const OMEGA_BOHR: f64 = 0.222; // 0.420 Å⁻¹, production erfc-optimal
const THRESH: f64 = 1e-10; // production 3-index screening threshold

fn main() {
    let bs = basis::bundled("cc-pvdz").unwrap();
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let coul = Operator::coulomb();
    let erfc = Operator::erfc(OMEGA_BOHR);

    // C-range: start at C8 (screening barely fires) through C20 (asymptote).
    // Default 8..=20; override with an env var if a shorter sweep is wanted.
    let (lo, hi): (usize, usize) = {
        let lo = std::env::var("SR_LO").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
        let hi = std::env::var("SR_HI").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
        (lo, hi)
    };

    println!(
        "# SR 3-index TIMING — cc-pVDZ/cc-pVDZ-RI, ω={OMEGA_BOHR} Bohr⁻¹, thresh={THRESH:.0e}\n\
         # times in seconds; qqr_erfc includes QqrBounds3::new. Run at 1 thread each.\n"
    );
    println!(
        "{:>4} {:>5} {:>5} | {:>10} {:>10} | {:>10} {:>10} | {:>9} {:>9}",
        "C", "nbas", "naux", "qqr_kept", "qqr_pct",
        "dense_coul", "dense_erfc", "qqr_erfc", "—",
    );
    println!(
        "{:>4} {:>5} {:>5} | {:>10} {:>10} | {:>10} {:>10} | {:>9} {:>9}",
        "", "", "", "(triples)", "", "t_coul[s]", "t_erfc[s]", "t_qqr[s]", "spdup*",
    );
    println!("{}", "-".repeat(96));

    for n in lo..=hi {
        let path = format!("testdata/molecules/alkane_{n}.xyz");
        let mol = match Molecule::load_xyz(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux).unwrap();
        let nbas = obs.nbasis();
        let naux = dfbs.nbasis();

        // 1. dense Coulomb
        let t0 = Instant::now();
        let dense_c = eri3_tensor(coul, &obs, &dfbs).unwrap();
        let t_coul = t0.elapsed().as_secs_f64();
        std::hint::black_box(&dense_c);
        drop(dense_c);

        // 2. dense erfc
        let t0 = Instant::now();
        let dense_e = eri3_tensor(erfc, &obs, &dfbs).unwrap();
        let t_erfc = t0.elapsed().as_secs_f64();
        std::hint::black_box(&dense_e);
        drop(dense_e);

        // 3. QQR3-screened erfc — bounds build + screened eval, both timed.
        let t0 = Instant::now();
        let bounds = QqrBounds3::new(erfc, &mol, &obs, &dfbs).unwrap();
        let (qqr_t, n_kept, n_total) =
            eri3_tensor_screened_qqr(erfc, &obs, &dfbs, &bounds, THRESH).unwrap();
        let t_qqr = t0.elapsed().as_secs_f64();
        std::hint::black_box(&qqr_t);
        drop(qqr_t);

        let spdup_vs_coul = t_coul / t_qqr.max(1e-9);
        let spdup_vs_erfc = t_erfc / t_qqr.max(1e-9);

        println!(
            "{:>4} {:>5} {:>5} | {:>10} {:>9.1}% | {:>10.3} {:>10.3} | {:>9.3} {:>9.2}",
            n, nbas, naux, n_kept, 100.0 * n_kept as f64 / n_total as f64,
            t_coul, t_erfc, t_qqr, spdup_vs_erfc,
        );
        // spdup* in the table = dense_erfc / qqr_erfc (the real Task-2 swap).
        // Also print the dense_coul/qqr_erfc number on a trailing comment line.
        println!(
            "{:>59}coul/qqr = {:.2}x",
            "", spdup_vs_coul,
        );
    }

    println!(
        "\n# spdup* column = dense_erfc / qqr_erfc (same operator, the Task-2 swap).\n\
         # Decision rule: wire QQR3 into the SR-MP2 erfc arm only if spdup* >= 1.3x at C16+."
    );
}
