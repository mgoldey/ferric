//! Projector-form Q̃ sparsity re-probe.
//!
//! The campaign found canonical-MO pseudo-densities P̃/Q̃ are dense (~50% nbas²),
//! killing the AO-time cubic-RPA route. The projector-form virtual propagator
//!   Q̃(τ) = exp(−τ S⁻¹F) · (S⁻¹ − C_occ C_occᵀ)
//! is built from the AO Fock and the occupied density alone (no explicit virtuals
//! — the Sternheimer-in-imaginary-time idea). It is MATHEMATICALLY identical to the
//! explicit-virtual Q̃ (unit-tested), so this probe does NOT ask "is it a different
//! matrix" — it asks the only question that matters for cubic scaling:
//!
//!   Does Q̃(τ) DECAY with AO-pair distance |R_μ − R_ν|?
//!
//! If Q̃ entries fall off with distance, a distance-truncated build stays sparse
//! and the AO-time route is cubic for plain Coulomb RPA. If Q̃ stays flat at long
//! range (like the canonical explicit-virtual sum), the reformulation buys nothing
//! and the lane parks. We bin |Q̃_μν| by inter-atomic distance and report the decay,
//! alongside P̃ (occupied) as the reference, at a few τ on the minimax grid.
//!
//! Usage: OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!          cargo run --release -p ferric-rpa --example projector_q_sparsity

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_rpa::ao_rpa::{pseudo_density_occ, pseudo_density_vir, pseudo_density_vir_projector};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

// 6-31G (split-valence) keeps SCF cheap enough to reach C8, while being a more
// honest locality test than minimal STO-3G (which is artificially compact). The
// Q̃-decay question is basis-QUALITATIVE — we want the trend across chain length,
// not cc-pVDZ-accurate magnitudes.
const BASIS: &str = "6-31g";
const SYSTEMS: &[(&str, &str, usize)] = &[
    ("C2", "testdata/molecules/alkane_2.xyz", 2),
    ("C4", "testdata/molecules/alkane_4.xyz", 4),
    ("C6", "testdata/molecules/alkane_6.xyz", 6),
    ("C8", "testdata/molecules/alkane_8.xyz", 8),
];

/// Map each AO basis function index to the atom (coordinate) it sits on, via the
/// shell→atom map and per-shell AO counts.
fn ao_atom_centers(mol: &Molecule, obs: &PreparedBasis) -> Vec<[f64; 3]> {
    let shell_to_atom = obs.shell_to_atom();
    let shell_dims = obs.shell_dims();
    let mut centers = Vec::new();
    for (sh, &atom_idx) in shell_to_atom.iter().enumerate() {
        let a = &mol.atoms[atom_idx];
        let c = [a.x, a.y, a.zpos];
        for _ in 0..shell_dims[sh] {
            centers.push(c);
        }
    }
    centers
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Per-distance-bin (Bohr) summary of |M_μν|: peak-normalized MEAN and MAX, plus
/// the count. The MAX matters more than the mean for truncatability — a truncated
/// build is lossless only if the LARGEST dropped entry is negligible, not the average.
fn decay_envelope(m: &Array2<f64>, centers: &[[f64; 3]]) -> Vec<(f64, f64, f64, usize)> {
    let n = m.dim().0;
    let bin_w = 1.0_f64; // Bohr
    let mut sums: std::collections::BTreeMap<i64, (f64, f64, usize)> = std::collections::BTreeMap::new();
    let mut peak = 0.0_f64;
    for mu in 0..n {
        for nu in 0..n {
            let v = m[(mu, nu)].abs();
            peak = peak.max(v);
            let d = dist(&centers[mu], &centers[nu]);
            let b = (d / bin_w).floor() as i64;
            let e = sums.entry(b).or_insert((0.0, 0.0, 0));
            e.0 += v;
            e.1 = e.1.max(v);
            e.2 += 1;
        }
    }
    let p = peak.max(1e-300);
    sums.into_iter()
        .map(|(b, (s, mx, c))| (b as f64 * bin_w, (s / c as f64) / p, mx / p, c))
        .collect()
}

/// Fraction of total |M| Frobenius mass beyond a distance cutoff (Bohr). This is the
/// DIRECT truncatability metric: it is the relative error of a build that drops all
/// pairs farther than `r_cut`. Small ⇒ distance truncation is lossless ⇒ sparse.
fn tail_mass_beyond(m: &Array2<f64>, centers: &[[f64; 3]], r_cut: f64) -> f64 {
    let n = m.dim().0;
    let (mut total, mut beyond) = (0.0_f64, 0.0_f64);
    for mu in 0..n {
        for nu in 0..n {
            let v = m[(mu, nu)] * m[(mu, nu)];
            total += v;
            if dist(&centers[mu], &centers[nu]) > r_cut {
                beyond += v;
            }
        }
    }
    (beyond / total.max(1e-300)).sqrt()
}

fn main() {
    let ctx = ParallelContext::default();
    let obs_bs = basis::bundled(BASIS).unwrap();

    println!("# projector-form Q̃ sparsity re-probe (decay of |Q̃_μν| vs atom-pair distance)");
    println!("# 6-31G; P̃=occ pseudo-density, Q̃_proj=projector virtual propagator");
    for &(label, path, _ncarb) in SYSTEMS {
        let mol = match Molecule::load_xyz(path) {
            Ok(m) => m,
            Err(e) => { eprintln!("skip {label}: {e}"); continue; }
        };
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let n = obs.nbasis();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        // Small basis ⇒ conventional J/K is fast; looser conv is fine (the
        // pseudo-density decay study doesn't need 1e-9 SCF).
        let rhf_cfg = RhfConfig {
            max_iter: 400,
            energy_conv: 1e-7,
            density_conv: 1e-6,
            level_shift: 0.2, // help the alkane SCF converge in 6-31G
            ..Default::default()
        };
        let rhf = match solve_rhf(&ctx, &mol, &obs, op, &bounds, &rhf_cfg) {
            Ok(r) => r,
            Err(e) => { eprintln!("skip {label}: SCF {e}"); continue; }
        };

        let nocc = mol.nelec() as usize / 2;
        let c = rhf.mos_r();
        let c_occ = c.slice(ndarray::s![.., ..nocc]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc..]).to_owned();
        let eps = rhf.eps_r();
        let eps_occ: Vec<f64> = eps[..nocc].to_vec();
        let eps_vir: Vec<f64> = eps[nocc..].to_vec();

        // AO Fock and overlap.
        let s = oneelectron::overlap(&obs);
        // F = S C diag(ε) Cᵀ S (reconstruct AO Fock from MOs; Cᵀ S C = I).
        let eps_all: Vec<f64> = eps.to_vec();
        let f_ao = s.dot(c).dot(&Array2::from_diag(&ndarray::arr1(&eps_all))).dot(&c.t()).dot(&s);

        let centers = ao_atom_centers(&mol, &obs);

        // Representative τ from a small minimax-ish set (small/med/large).
        for &tau in &[0.05_f64, 1.0, 6.0] {
            let p_occ = pseudo_density_occ(&c_occ, &eps_occ, tau);
            let q_expl = pseudo_density_vir(&c_vir, &eps_vir, tau);
            let q_proj = pseudo_density_vir_projector(&f_ao, &s, &c_occ, tau);

            // Sanity: projector ≡ explicit (should be ~1e-9).
            let agree = q_expl.iter().zip(q_proj.iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);

            let env_q = decay_envelope(&q_proj, &centers);
            // DIRECT truncatability: relative Frobenius mass of Q̃ beyond r_cut.
            // This is exactly the error of a distance-truncated Q̃ build.
            let q_tail_5 = tail_mass_beyond(&q_proj, &centers, 5.0);
            let q_tail_8 = tail_mass_beyond(&q_proj, &centers, 8.0);
            let p_tail_5 = tail_mass_beyond(&p_occ, &centers, 5.0);

            println!(
                "{label} n={n} τ={tau:.2} | proj≡expl {agree:.0e} | \
                 P̃ massbeyond5={p_tail_5:.3} | Q̃ massbeyond5={q_tail_5:.3} massbeyond8={q_tail_8:.3}"
            );
            // Q̃ MAX-entry envelope (peak-normalized max |Q̃_μν| per distance bin).
            let curve: Vec<String> = env_q.iter()
                .map(|(r, _mean, mx, _)| format!("{r:.0}:{mx:.2}")).collect();
            println!("    Q̃ max-env  {}", curve.join(" "));
        }
        println!();
    }
    println!("# READ: if Q̃ envelope DROPS toward 0 at large r ⇒ projector Q̃ is atom-pair-local");
    println!("#       ⇒ distance-truncated build is sparse ⇒ cubic non-RS RPA reachable.");
    println!("#       if it FLATTENS at a finite rel value ⇒ same dense problem reformulated ⇒ park.");
}
