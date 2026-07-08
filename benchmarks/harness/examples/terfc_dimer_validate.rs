//! Validate MP2(terfc) interaction energies for noncovalent dimers against the
//! Goldey/Dutoi/Head-Gordon attenuated-MP2 paper (PCCP 2013) regime.
//!
//! Interaction energy WITHOUT counterpoise correction (the paper's variant I is
//! parameterized non-CP — attenuation cancels BSSE, so CP would double-correct):
//!   E_int = E(dimer) - E(monomerA) - E(monomerB)     [each in its own basis]
//! computed for (a) standard RI-MP2 (Coulomb) and (b) MP2(terfc) at r0=1.05 Å
//! (the aDZ-optimal cutoff). terfc should reduce the MP2 overbinding toward the
//! CCSD(T)/CBS benchmark. (Goldey thesis 2014: "No counterpoise corrections are
//! performed.")
//!
//! Run (aug-cc-pVDZ is the paper's aDZ basis):
//!   OPENBLAS_NUM_THREADS=1 FERRIC_TERF_TABLE_DIR=$PWD/terf-tables \
//!     cargo run --release --example terfc_dimer_validate -p ferric-benchmarks

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::io::Write;

const HA_TO_KCAL: f64 = 627.509_474_06;
const ANG2BOHR: f64 = 1.889_725_988_6;

macro_rules! say {
    ($($a:tt)*) => {{ println!($($a)*); let _ = std::io::stdout().flush(); }};
}

/// One dimer: full XYZ, and the split index (first `n_a` atoms = monomer A).
struct Dimer {
    name: &'static str,
    xyz: &'static str,
    n_a: usize,
    /// CCSD(T)/CBS benchmark interaction energy (kcal/mol), from S66.
    ref_kcal: f64,
}

fn dimers() -> Vec<Dimer> {
    vec![
        // S66 #1 water-water at equilibrium (S66x8 factor 1.00). Ref: -4.97 kcal/mol.
        Dimer {
            name: "water-water (S66 #1)",
            xyz: "6
waterwater
O   -0.702196054  -0.056060256   0.009942262
H   -1.022193224   0.846775782  -0.011488714
H    0.257521062   0.042121496   0.005218999
O    2.268880784   0.026340101   0.000508029
H    2.645502399  -0.412039965   0.766632411
H    2.641145101  -0.449872874  -0.744894473
",
            n_a: 3,
            ref_kcal: -4.97,
        },
    ]
}

/// Build a sub-molecule from the given atom index range (NON-CP: dropped atoms
/// are removed entirely, each fragment in its own basis). `None` = full dimer.
fn subsystem_xyz(xyz: &str, keep: Option<(usize, usize)>) -> String {
    let lines: Vec<&str> = xyz.lines().collect();
    let atom_lines = &lines[2..];
    let mut kept: Vec<&str> = Vec::new();
    for (i, l) in atom_lines.iter().enumerate() {
        let l = l.trim();
        if l.is_empty() {
            continue;
        }
        let real = match keep {
            None => true,
            Some((lo, hi)) => i >= lo && i < hi,
        };
        if real {
            kept.push(l);
        }
    }
    let mut out = format!("{}\nsub\n", kept.len());
    for l in kept {
        out.push_str(l);
        out.push('\n');
    }
    out
}

/// Total energy (RHF + RI-MP2 with the given operator) for one geometry.
fn energy(
    ctx: &ParallelContext,
    xyz: &str,
    obs_name: &str,
    dfbs_name: &str,
    op: Operator,
) -> Option<f64> {
    let mut mol = Molecule::parse_xyz(xyz, 0, 1).ok()?;
    let obs_bs = basis::bundled(obs_name).ok()?;
    mol.apply_ecp(&obs_bs);
    let dfbs_bs = basis::bundled(dfbs_name).ok()?;
    let obs = PreparedBasis::new(&mol, &obs_bs).ok()?;
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).ok()?;
    // SCF always uses the plain Coulomb operator; only the MP2 correlation is attenuated.
    let coul = Operator::coulomb();
    let bounds = SchwarzBounds::compute(coul, &obs).ok()?;
    let cfg = RhfConfig { max_iter: 300, ..Default::default() };
    let rhf = solve_rhf(ctx, &mol, &obs, coul, &bounds, &cfg).ok()?;
    let mp2 = ri_mp2(&mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default()).ok()?;
    Some(mp2.total_energy)
}

fn interaction_kcal(
    ctx: &ParallelContext,
    d: &Dimer,
    obs: &str,
    dfbs: &str,
    op: Operator,
) -> Option<f64> {
    let n_atoms = d.xyz.lines().count() - 2;
    let e_dim = energy(ctx, &subsystem_xyz(d.xyz, None), obs, dfbs, op)?;
    let e_a = energy(ctx, &subsystem_xyz(d.xyz, Some((0, d.n_a))), obs, dfbs, op)?;
    let e_b = energy(ctx, &subsystem_xyz(d.xyz, Some((d.n_a, n_atoms))), obs, dfbs, op)?;
    Some((e_dim - e_a - e_b) * HA_TO_KCAL)
}

fn main() {
    let ctx = ParallelContext::new();
    let obs = "aug-cc-pvdz";
    let dfbs = "aug-cc-pvdz-rifit";
    let r0 = 1.05 * ANG2BOHR; // aDZ-optimal terfc cutoff (paper)

    say!("MP2(terfc) dimer validation — {obs} / r0=1.05 A / NON-CP (paper variant I)");
    say!("{:<24} {:>10} {:>10} {:>10} {:>10}", "system", "MP2", "terfc", "CCSD(T)ref", "MP2err");
    for d in dimers() {
        let e_mp2 = interaction_kcal(&ctx, &d, obs, dfbs, Operator::coulomb());
        let e_terfc = interaction_kcal(&ctx, &d, obs, dfbs, Operator::terfc(r0));
        match (e_mp2, e_terfc) {
            (Some(m), Some(t)) => {
                say!(
                    "{:<24} {:>10.3} {:>10.3} {:>10.3} {:>+10.3}",
                    d.name, m, t, d.ref_kcal, m - d.ref_kcal
                );
                say!(
                    "  -> terfc err vs CCSD(T): {:+.3} kcal/mol  (MP2 err {:+.3})",
                    t - d.ref_kcal, m - d.ref_kcal
                );
            }
            _ => say!("{:<24} FAILED (SCF/MP2 did not converge)", d.name),
        }
    }
}
