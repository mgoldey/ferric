//! RRHO thermochemistry and IR intensities of a `task = "frequencies"` run:
//! the `[frequencies] temperature`, `pressure`, `symmetry_number` and
//! `ir_intensities` keys, the printed thermochemistry block, and the
//! `thermochemistry` / `ir_intensities_km_mol` fields of the `frequencies`
//! run-log record.

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{FrequencyReference, FrequencyResult};
use ferric_scf::infrared::{ir_spectrum, IrSpectrum, DEFAULT_IR_FIELD};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::runlog::json_f64;
use ferric_scf::thermo::{ThermoConfig, Thermochemistry};

use crate::config::Config;

/// Hartree to kcal/mol (CODATA 2018 E_h times N_A over 4184 J/kcal).
const HARTREE_TO_KCAL_PER_MOL: f64 = 627.509_474_063_1;

/// What a frequency run adds after the Hessian: thermochemistry, IR
/// intensities (when computed) and their run-log fields.
pub(crate) struct FrequencyExtras {
    pub(crate) thermo: Option<Thermochemistry>,
    pub(crate) ir: Option<IrSpectrum>,
    /// Why there are no intensities (a correlated kind, `ir_intensities =
    /// false`, or a failed field SCF), printed under the table.
    pub(crate) ir_note: Option<String>,
    pub(crate) json: serde_json::Map<String, serde_json::Value>,
}

/// `[frequencies] temperature`, `pressure`, `symmetry_number`; exits on a bad
/// value before any SCF.
pub(crate) fn thermo_config(cfg: &Config) -> ThermoConfig {
    let f = &cfg.frequencies;
    let d = ThermoConfig::default();
    let tc = ThermoConfig {
        temperature: f.temperature.unwrap_or(d.temperature),
        pressure: f.pressure.unwrap_or(d.pressure),
        symmetry_number: f.symmetry_number.unwrap_or(d.symmetry_number),
    };
    if let Err(e) = tc.validate() {
        eprintln!("error: [frequencies]: {e}");
        std::process::exit(1);
    }
    tc
}

/// Whether to compute IR intensities. `reference` is `None` for a correlated
/// kind, where `ir_intensities = true` exits with an error before any SCF.
pub(crate) fn ir_requested(
    method: &str,
    cfg: &Config,
    reference: Option<FrequencyReference>,
) -> bool {
    match (reference, cfg.frequencies.ir_intensities) {
        (Some(_), want) => want.unwrap_or(true),
        (None, Some(true)) => {
            eprintln!(
                "error: [frequencies] ir_intensities = true: IR intensities need the dipole \
                 derivative, which ferric computes for the SCF kinds (rhf, uhf, rohf, ksdft) \
                 only; method.kind = \"{method}\" has none"
            );
            std::process::exit(1);
        }
        (None, _) => false,
    }
}

/// Thermochemistry and (for an SCF `reference`, unless switched off) IR
/// intensities of `res`, computed at `mol`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn frequency_extras(
    method: &str,
    cfg: &Config,
    ctx: &ParallelContext,
    mol: &Molecule,
    bs: &BasisSet,
    op: Operator,
    rhf_config: &RhfConfig,
    reference: Option<FrequencyReference>,
    res: &FrequencyResult,
) -> FrequencyExtras {
    let mut json = serde_json::Map::new();
    let thermo = match res.thermochemistry(mol, &thermo_config(cfg)) {
        Ok(th) => {
            for w in &th.warnings {
                eprintln!("warning: {w}");
            }
            json.insert("thermochemistry".into(), th.to_json());
            Some(th)
        }
        Err(e) => {
            eprintln!("warning: thermochemistry not computed: {e}");
            json.insert("thermochemistry".into(), serde_json::Value::Null);
            None
        }
    };
    let (ir, ir_note) = match (ir_requested(method, cfg, reference), reference) {
        (true, Some(r)) => {
            match ir_spectrum(ctx, mol, &bs.name, op, rhf_config, r, res, DEFAULT_IR_FIELD) {
                Ok(s) => (Some(s), None),
                Err(e) if cfg.frequencies.ir_intensities == Some(true) => {
                    eprintln!("error computing IR intensities: {e}");
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("warning: IR intensities not computed: {e}");
                    (None, Some(format!("not computed ({e})")))
                }
            }
        }
        (false, Some(_)) => (
            None,
            Some("off ([frequencies] ir_intensities = false)".into()),
        ),
        (_, None) => (
            None,
            Some(format!(
                "not available for method.kind = \"{method}\" (SCF kinds only)"
            )),
        ),
    };
    let ir_json = ir.as_ref().map_or(serde_json::Value::Null, |s| {
        serde_json::json!(s
            .intensities_km_mol
            .iter()
            .copied()
            .map(json_f64)
            .collect::<Vec<_>>())
    });
    json.insert("ir_intensities_km_mol".into(), ir_json);
    if let Some(s) = &ir {
        json.insert(
            "dipole_derivatives".into(),
            serde_json::json!(s
                .dipole_derivatives
                .iter()
                .copied()
                .map(json_f64)
                .collect::<Vec<_>>()),
        );
        json.insert(
            "ir_sum_rule_residual".into(),
            s.sum_rule_residual
                .map_or(serde_json::Value::Null, json_f64),
        );
    }
    FrequencyExtras {
        thermo,
        ir,
        ir_note,
        json,
    }
}

impl FrequencyExtras {
    /// Intensities in mode order, when computed.
    pub(crate) fn intensities(&self) -> Option<&[f64]> {
        self.ir.as_ref().map(|s| s.intensities_km_mol.as_slice())
    }

    /// The thermochemistry block and the IR note.
    pub(crate) fn print(&self) {
        if let Some(note) = &self.ir_note {
            println!("\n  IR intensities: {note}");
        } else if let Some(r) = self.ir.as_ref().and_then(|s| s.sum_rule_residual) {
            println!(
                "\n  IR intensities: field route (+-{DEFAULT_IR_FIELD:.0e} a.u.), dipole \
                 sum-rule residual {r:.2e} e"
            );
        }
        let Some(th) = &self.thermo else { return };
        let c = &th.config;
        println!(
            "\n  Thermochemistry (RRHO): T = {:.2} K, P = {:.1} Pa, symmetry number {}{}",
            c.temperature,
            c.pressure,
            c.symmetry_number,
            if c.symmetry_number == 1 {
                " (default; set [frequencies] symmetry_number)"
            } else {
                ""
            }
        );
        let consts: Vec<String> = th
            .rotational_constants_ghz
            .iter()
            .map(|b| format!("{b:.5}"))
            .collect();
        println!(
            "    rotor = {}, rotational constants (GHz) = [{}]",
            th.rotor.label(),
            consts.join(", ")
        );
        println!(
            "    zero-point energy = {:.10} Hartree ({:.4} kcal/mol)",
            th.zpe,
            th.zpe * HARTREE_TO_KCAL_PER_MOL
        );
        println!(
            "    {:<14}{:>16}{:>16}{:>16}{:>16}",
            "", "E (Hartree)", "H (Hartree)", "S (cal/mol/K)", "G (Hartree)"
        );
        let s_cal = |s: f64| s * HARTREE_TO_KCAL_PER_MOL * 1000.0;
        for (name, t) in [
            ("electronic", &th.electronic),
            ("translational", &th.translational),
            ("rotational", &th.rotational),
            ("vibrational", &th.vibrational),
            ("total", &th.total),
        ] {
            println!(
                "    {:<14}{:>16.10}{:>16.10}{:>16.4}{:>16.10}",
                name,
                t.energy,
                t.enthalpy,
                s_cal(t.entropy),
                t.gibbs
            );
        }
        println!(
            "    thermal correction to H = {:.10} Hartree, to G = {:.10} Hartree",
            th.total.enthalpy - th.e0,
            th.total.gibbs - th.e0
        );
        if th.n_imaginary_excluded > 0 {
            println!(
                "    {} imaginary mode(s) excluded from the vibrational terms",
                th.n_imaginary_excluded
            );
        }
    }
}
