//! The headline energy anchor of each method behind a Proven CLI
//! `method.kind`, and the validation envelope it defines.
//!
//! Each [`EnergyAnchor`] is one row of the Anchors table on
//! [`GRADE_SOURCE_PAGE`](super::GRADE_SOURCE_PAGE): the stated agreement of
//! the method's energy with an independent reference, the test that pins it,
//! and the elements, basis sets and functionals that test runs. A certificate
//! carries the anchor's value as its
//! [`measured_error`](super::Certificate::measured_error)
//! ([`Certificate::with_energy_anchor`]); [`Certificate::with_envelope`] adds
//! one warning per dimension on which a run lies outside the anchor's set.
//!
//! The envelope is a lookup of each dimension on its own. A run whose
//! elements and basis are both covered can still be a combination the test
//! never ran (the RHF anchor runs each basis on one molecule), and other rows
//! of the page may cover a run this anchor does not. Warnings never change a
//! grade.
//!
//! The `anchors_match_the_page` test checks every entry against the page and
//! derives the covered sets from the reference data the evidence test reads.
//!
//! Each entry also names a production-code mutation that its evidence test
//! fails on by an assertion: a patch under `testdata/mutations/` with a row in
//! `testdata/mutations/ledger.json`. `scripts/check_anchor_mutations.py` runs
//! the patch and records the result; the `mutation_ledger_matches_the_registry`
//! test checks each row's evidence path and patch hash against this registry
//! and the file on disk.

use super::{Certificate, MeasuredError};
use crate::basis::BasisSet;
use crate::elements::{symbol_to_z, z_to_symbol};

/// The heading of the table on the grade page that holds the anchors.
pub const ANCHOR_SOURCE_HEADING: &str = "## Anchors";

/// Where an anchor's covered systems and basis sets are recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorData {
    /// `testdata/reference/basis_coverage_pyscf.json`, the cases of one
    /// `kind` (`"rhf"` or `"rimp2"`).
    BasisCoverage {
        /// The `kind` field of the cases the anchor covers.
        kind: &'static str,
    },
    /// A directory of `<system>_<basis>.json` references; each system's
    /// geometry is `testdata/molecules/validation/<system>.xyz`.
    Directory {
        /// The reference directory, relative to the workspace root.
        dir: &'static str,
        /// The systems of this anchor (a directory may hold several anchors).
        systems: &'static [&'static str],
    },
    /// Named reference files `<dir>/<file>.json`. Each file records its
    /// orbital basis (`basis`), its auxiliary basis (`aux_basis`, `auxbasis`
    /// or `aux`) and its geometry (`geometry_stem`, else `system`, naming
    /// `testdata/molecules/validation/<stem>.xyz`). For a directory whose
    /// systems do not all run every basis.
    Files {
        /// The reference directory, relative to the workspace root.
        dir: &'static str,
        /// File stems in `dir` (without `.json`).
        files: &'static [&'static str],
    },
    /// Reference files that carry their geometry inline, relative to the
    /// workspace root. Each file is one record, or holds its records in a
    /// `systems` array; a record names its `system` and `basis` and holds an
    /// XYZ block (count line, comment line, atoms) in `xyz`. For methods
    /// without an auxiliary basis.
    InlineGeometry {
        /// The reference files, relative to the workspace root.
        files: &'static [&'static str],
    },
}

/// One method's headline energy anchor; see the module doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnergyAnchor {
    /// The solver the anchor covers, e.g. `"RHF"`, `"UKS"`, `"RI-MP2"`,
    /// `"CCSD"`. When several Anchors rows link the same test file, the row
    /// is the first whose Capability cell contains this name.
    pub method: &'static str,
    /// The CLI `method.kind` the anchor validates. The certificates of
    /// `rhf`, `uhf`, `rohf`, `ksdft` and `rimp2` carry it as their measured
    /// error.
    pub kind: &'static str,
    /// What was compared, e.g. `"energy"`.
    pub quantity: &'static str,
    /// The stated agreement, spelled as on the page, e.g. `"1.8e-12"`.
    pub value: &'static str,
    /// Units of `value`.
    pub units: &'static str,
    /// The reference code, as named in the row's Reference cell.
    pub reference: &'static str,
    /// The test file the row's Proof cell links, relative to the workspace
    /// root.
    pub evidence: &'static str,
    /// Where the covered systems and basis sets are recorded.
    pub data: AnchorData,
    /// Element symbols of the covered systems.
    pub elements: &'static [&'static str],
    /// Orbital basis sets the test runs, as the test names them.
    pub basis_sets: &'static [&'static str],
    /// Auxiliary basis sets the test runs (correlated methods only).
    pub aux_basis_sets: &'static [&'static str],
    /// Functionals the test runs (KS only), lower case.
    pub functionals: &'static [&'static str],
}

const BASIS_COVERAGE: &str = "crates/ferric-mp2/tests/validation_basis_coverage.rs";
const OPEN_SHELL_SCF: &str = "crates/ferric-scf/tests/validation_open_shell_scf.rs";
const KS_ENERGIES: &str = "crates/ferric-scf/tests/validation_ks_energies.rs";
const OPEN_SHELL_SYSTEMS: &[&str] = &["ho2", "no2", "ch2_triplet", "allyl"];
const OPEN_SHELL_ELEMENTS: &[&str] = &["H", "C", "N", "O"];
const OPEN_SHELL_BASES: &[&str] = &["6-31g", "def2-svp"];
const CC: &str = "crates/ferric-cc/tests/validation_cc.rs";
const CC_DIR: &str = "testdata/reference/validation/cc";
const ATT_MP2_DIR: &str = "testdata/reference/validation/attenuated_mp2";

const CASSCF_REFS: &[&str] = &[
    "testdata/reference/casscf_h2o_cas44_631g_pyscf.json",
    "testdata/reference/casscf_n2_cas66_631g_pyscf.json",
    "testdata/reference/casscf_n2_r1.6_cas66_631g_pyscf.json",
    "testdata/reference/casscf_o2_triplet_cas86_631g_pyscf.json",
];

/// The headline energy anchor of every method that has one.
pub const ENERGY_ANCHORS: &[EnergyAnchor] = &[
    EnergyAnchor {
        method: "RHF",
        kind: "rhf",
        quantity: "energy",
        value: "1.8e-12",
        units: "Ha",
        reference: "PySCF 2.13.1",
        evidence: BASIS_COVERAGE,
        data: AnchorData::BasisCoverage { kind: "rhf" },
        elements: &["H", "C", "O", "Ne", "Cl", "Br", "Ag", "I"],
        basis_sets: &[
            "sto-3g",
            "6-31g",
            "6-31g*",
            "6-31g**",
            "6-31+g*",
            "6-31+g**",
            "6-31++g**",
            "6-311g**",
            "6-311++g**",
            "cc-pvdz",
            "cc-pvtz",
            "cc-pvqz",
            "aug-cc-pvdz",
            "aug-cc-pvtz",
            "aug-cc-pvqz",
            "cc-pvdz-pp",
            "cc-pvtz-pp",
            "cc-pvqz-pp",
            "aug-cc-pvdz-pp",
            "aug-cc-pvtz-pp",
            "aug-cc-pvqz-pp",
            "cc-pvdz-f12",
            "def2-svp",
            "def2-svpd",
            "def2-tzvp",
            "def2-tzvpd",
            "def2-tzvpp",
            "def2-tzvppd",
            "def2-qzvp",
            "def2-qzvpp",
            "def2-qzvppd",
            "pcseg-0",
            "pcseg-1",
            "pcseg-2",
            "aug-pcseg-1",
            "aug-pcseg-2",
        ],
        aux_basis_sets: &[],
        functionals: &[],
    },
    EnergyAnchor {
        method: "UHF",
        kind: "uhf",
        quantity: "energy",
        value: "4.0e-12",
        units: "Ha",
        reference: "PySCF",
        evidence: OPEN_SHELL_SCF,
        data: AnchorData::Directory {
            dir: "testdata/reference/validation/uhf_rohf",
            systems: OPEN_SHELL_SYSTEMS,
        },
        elements: OPEN_SHELL_ELEMENTS,
        basis_sets: OPEN_SHELL_BASES,
        aux_basis_sets: &[],
        functionals: &[],
    },
    EnergyAnchor {
        method: "ROHF",
        kind: "rohf",
        quantity: "energy",
        value: "4.0e-12",
        units: "Ha",
        reference: "PySCF",
        evidence: OPEN_SHELL_SCF,
        data: AnchorData::Directory {
            dir: "testdata/reference/validation/uhf_rohf",
            systems: OPEN_SHELL_SYSTEMS,
        },
        elements: OPEN_SHELL_ELEMENTS,
        basis_sets: OPEN_SHELL_BASES,
        aux_basis_sets: &[],
        functionals: &[],
    },
    EnergyAnchor {
        method: "RKS",
        kind: "ksdft",
        quantity: "energy",
        value: "4.2e-12",
        units: "Ha",
        reference: "PySCF",
        evidence: KS_ENERGIES,
        data: AnchorData::Directory {
            dir: "testdata/reference/validation/ks_energies",
            systems: &["h2s", "hcl", "sih4"],
        },
        elements: &["H", "Si", "S", "Cl"],
        basis_sets: &["def2-svp", "def2-tzvp"],
        aux_basis_sets: &[],
        functionals: &["pbe", "b3lyp", "hse06"],
    },
    EnergyAnchor {
        method: "UKS",
        kind: "uhf",
        quantity: "energy",
        value: "6e-6",
        units: "Ha",
        reference: "PySCF",
        evidence: KS_ENERGIES,
        data: AnchorData::Directory {
            dir: "testdata/reference/validation/ks_energies",
            systems: &["nh2", "ch3", "ho2", "o2"],
        },
        elements: &["H", "C", "N", "O"],
        basis_sets: &["6-31g", "def2-svp"],
        aux_basis_sets: &[],
        functionals: &["pbe", "b3lyp", "wb97x-v"],
    },
    EnergyAnchor {
        method: "ROKS",
        kind: "rohf",
        quantity: "energy",
        value: "8.0e-7",
        units: "Ha",
        reference: "PySCF",
        evidence: "crates/ferric-scf/tests/validation_roks_energies.rs",
        data: AnchorData::Directory {
            dir: "testdata/reference/validation/roks_energies",
            systems: &["nh2", "ch3", "ho2", "oh"],
        },
        elements: &["H", "C", "N", "O"],
        basis_sets: &["6-31g", "def2-svp"],
        aux_basis_sets: &[],
        functionals: &["pbe", "b3lyp"],
    },
    EnergyAnchor {
        method: "RI-MP2",
        kind: "rimp2",
        quantity: "correlation energy",
        value: "2.4e-10",
        units: "Ha",
        reference: "PySCF 2.13.1",
        evidence: BASIS_COVERAGE,
        data: AnchorData::BasisCoverage { kind: "rimp2" },
        elements: &["H", "O", "Pd"],
        basis_sets: &[
            "cc-pvdz",
            "cc-pvtz",
            "cc-pvqz",
            "aug-cc-pvdz",
            "aug-cc-pvtz",
            "aug-cc-pvqz",
            "cc-pvdz-pp",
            "cc-pvtz-pp",
            "aug-cc-pvdz-pp",
            "aug-cc-pvtz-pp",
            "cc-pvdz-f12",
            "def2-svp",
            "def2-svpd",
            "def2-tzvp",
            "def2-tzvpd",
            "def2-tzvpp",
            "def2-tzvppd",
            "def2-qzvp",
            "def2-qzvpp",
            "def2-qzvppd",
        ],
        aux_basis_sets: &[
            "cc-pvdz-rifit",
            "cc-pvtz-rifit",
            "cc-pvqz-rifit",
            "aug-cc-pvdz-rifit",
            "aug-cc-pvtz-rifit",
            "aug-cc-pvqz-rifit",
            "cc-pvdz-pp-rifit",
            "cc-pvtz-pp-rifit",
            "cc-pvqz-pp-rifit",
            "aug-cc-pvdz-pp-rifit",
            "aug-cc-pvtz-pp-rifit",
            "aug-cc-pvqz-pp-rifit",
            "cc-pvtz-jkfit",
            "cc-pvqz-jkfit",
            "aug-cc-pvtz-jkfit",
            "cc-pvdz-f12-optri",
            "def2-svp-rifit",
            "def2-svpd-rifit",
            "def2-tzvp-rifit",
            "def2-tzvpd-rifit",
            "def2-tzvpp-rifit",
            "def2-tzvppd-rifit",
            "def2-qzvp-rifit",
            "def2-qzvpp-rifit",
            "def2-qzvppd-rifit",
            "def2-universal-jkfit",
        ],
        functionals: &[],
    },
    EnergyAnchor {
        method: "U-RI-MP2",
        kind: "rimp2",
        quantity: "correlation energy",
        value: "9.2e-10",
        units: "Ha",
        reference: "PySCF DFUMP2",
        evidence: "crates/ferric-mp2/tests/validation_u_rimp2.rs",
        data: AnchorData::Directory {
            dir: "testdata/reference/validation/u_rimp2",
            systems: &["oh", "ch3", "nh2", "o2", "ho2"],
        },
        elements: &["H", "C", "N", "O"],
        basis_sets: &["cc-pvdz"],
        aux_basis_sets: &["cc-pvdz-ri"],
        functionals: &[],
    },
    EnergyAnchor {
        method: "MP3",
        kind: "mp3",
        quantity: "correlation energy",
        value: "7.6e-12",
        units: "Ha",
        reference: "numpy",
        evidence: CC,
        data: AnchorData::Files {
            dir: CC_DIR,
            files: &["h2o_cc-pvdz", "h2o_def2-svp", "nh3_cc-pvdz", "nh3_def2-svp"],
        },
        elements: &["H", "N", "O"],
        basis_sets: &["cc-pvdz", "def2-svp"],
        aux_basis_sets: &["cc-pvdz-ri", "def2-svp-rifit"],
        functionals: &[],
    },
    EnergyAnchor {
        method: "CCSD",
        kind: "ccsd",
        quantity: "correlation energy",
        value: "4.7e-11",
        units: "Ha",
        reference: "PySCF CCSD",
        evidence: CC,
        data: AnchorData::Files {
            dir: CC_DIR,
            files: &[
                "h2o_cc-pvdz",
                "h2o_def2-svp",
                "h2o_aug-cc-pvdz",
                "nh3_cc-pvdz",
                "nh3_def2-svp",
                "hcn_cc-pvdz",
                "c2h6_cc-pvdz",
            ],
        },
        elements: &["H", "C", "N", "O"],
        basis_sets: &["cc-pvdz", "def2-svp", "aug-cc-pvdz"],
        aux_basis_sets: &["cc-pvdz-ri", "def2-svp-rifit", "aug-cc-pvdz-rifit"],
        functionals: &[],
    },
    EnergyAnchor {
        method: "Attenuated RI-MP2",
        kind: "att-rimp2",
        quantity: "correlation energy",
        value: "5.9e-12",
        units: "Ha",
        reference: "numpy",
        evidence: "crates/ferric-mp2/tests/validation_attenuated_mp2.rs",
        data: AnchorData::Directory {
            dir: ATT_MP2_DIR,
            systems: &["h2o", "nh3"],
        },
        elements: &["H", "N", "O"],
        basis_sets: &["aug-cc-pvdz"],
        aux_basis_sets: &["aug-cc-pvdz-rifit"],
        functionals: &[],
    },
    EnergyAnchor {
        method: "SCS-MP2",
        kind: "scs-mp2",
        quantity: "correlation energy",
        value: "3.5e-12",
        units: "Ha",
        reference: "numpy",
        evidence: "crates/ferric-mp2/tests/validation_scs_mp2.rs",
        data: AnchorData::Directory {
            dir: ATT_MP2_DIR,
            systems: &["h2o", "nh3"],
        },
        elements: &["H", "N", "O"],
        basis_sets: &["aug-cc-pvdz"],
        aux_basis_sets: &["aug-cc-pvdz-rifit"],
        functionals: &[],
    },
    EnergyAnchor {
        method: "Laplace RI-MP2",
        kind: "laplace-mp2",
        quantity: "correlation energy at matched quadrature",
        value: "3.1e-11",
        units: "Ha",
        reference: "numpy",
        evidence: "crates/ferric-mp2/tests/validation_laplace_mp2.rs",
        data: AnchorData::Files {
            dir: "testdata/reference/validation/laplace_mp2",
            files: &[
                "h2o_cc-pvdz",
                "h2o_aug_aug-cc-pvdz",
                "ch4_cc-pvdz",
                "butadiene_cc-pvdz",
                "alkane_8_cc-pvdz",
            ],
        },
        elements: &["H", "C", "O"],
        basis_sets: &["cc-pvdz", "aug-cc-pvdz"],
        aux_basis_sets: &["cc-pvdz-ri", "aug-cc-pvdz-rifit"],
        functionals: &[],
    },
    EnergyAnchor {
        method: "Laplace SOS-MP2",
        kind: "laplace-sos-mp2",
        quantity: "opposite-spin energy at matched quadrature",
        value: "1.6e-11",
        units: "Ha",
        reference: "PySCF",
        evidence: "crates/ferric-mp2/tests/validation_sos_mp2.rs",
        data: AnchorData::Files {
            dir: "testdata/reference/validation/sos_mp2",
            files: &[
                "h2o_cc-pvdz",
                "h2o_aug_aug-cc-pvdz",
                "ch4_cc-pvdz",
                "butadiene_cc-pvdz",
                "alkane_8_cc-pvdz",
            ],
        },
        elements: &["H", "C", "O"],
        basis_sets: &["cc-pvdz", "aug-cc-pvdz"],
        aux_basis_sets: &["cc-pvdz-ri", "aug-cc-pvdz-rifit"],
        functionals: &[],
    },
    EnergyAnchor {
        method: "PDEP-RPA",
        kind: "pdep-rpa",
        quantity: "correlation energy",
        value: "1.5e-13",
        units: "Ha",
        reference: "gw.rpa.RPA",
        evidence: "crates/ferric-rpa/tests/validation_rpa_gradient.rs",
        data: AnchorData::Files {
            dir: "testdata/reference/validation/rpa_gradient",
            files: &["h2o_distorted_cc-pvdz", "nh3_distorted_cc-pvdz"],
        },
        elements: &["H", "N", "O"],
        basis_sets: &["cc-pvdz"],
        aux_basis_sets: &["cc-pvdz-ri"],
        functionals: &[],
    },
    EnergyAnchor {
        method: "CASSCF",
        kind: "casscf",
        quantity: "energy",
        value: "1.5e-11",
        units: "Ha",
        reference: "PySCF",
        evidence: "crates/ferric-ci/tests/casscf_vs_pyscf.rs",
        data: AnchorData::InlineGeometry { files: CASSCF_REFS },
        elements: &["H", "N", "O"],
        basis_sets: &["6-31g"],
        aux_basis_sets: &[],
        functionals: &[],
    },
    EnergyAnchor {
        method: "SC-NEVPT2",
        kind: "nevpt2",
        quantity: "class energies",
        value: "3.2e-8",
        units: "Ha",
        reference: "PySCF",
        evidence: "crates/ferric-ci/tests/nevpt2.rs",
        data: AnchorData::InlineGeometry {
            files: &["testdata/reference/nevpt2_refs.json"],
        },
        elements: &["H", "N", "O"],
        basis_sets: &["6-31g"],
        aux_basis_sets: &[],
        functionals: &[],
    },
];

/// The anchor of `method` (see [`EnergyAnchor::method`]), if it has one.
pub fn energy_anchor(method: &str) -> Option<&'static EnergyAnchor> {
    ENERGY_ANCHORS.iter().find(|a| a.method == method)
}

/// Whether `run` is one of the bundled sets in `list` on the elements `zs`:
/// equal shells and ECP for every element of `zs`. The run's name only
/// orders the search (a listed set of the same canonical name is tried
/// first); a loaded file named like a bundled set but with other contractions
/// is not covered, and a registry alias (`cc-pvdz-ri` for `cc-pvdz-rifit`)
/// is.
fn basis_listed(list: &[&str], run: &BasisSet, zs: &[i32]) -> bool {
    let cn = crate::basis_util::canonical_name(&run.name);
    let same = |listed: &str| {
        crate::basis::bundled(listed).is_ok_and(|s| {
            zs.iter().all(|&z| {
                s.for_element(z) == run.for_element(z)
                    && s.ecp_for_element(z) == run.ecp_for_element(z)
            })
        })
    };
    list.iter().filter(|b| **b == cn).any(|b| same(b))
        || list.iter().filter(|b| **b != cn).any(|b| same(b))
}

impl EnergyAnchor {
    /// The anchor as a [`MeasuredError`]. `value` is a finite, non-negative
    /// number for every entry of [`ENERGY_ANCHORS`] (checked by a test).
    pub fn measured_error(&self) -> MeasuredError {
        let v: f64 = self
            .value
            .parse()
            .unwrap_or_else(|e| panic!("{}: value {:?}: {e}", self.method, self.value));
        MeasuredError::new(self.quantity, v, self.units, self.reference, self.evidence)
            .unwrap_or_else(|e| panic!("{}: {e}", self.method))
    }

    /// The first atomic number of `zs` whose element is not in
    /// [`Self::elements`], in input order.
    pub fn first_uncovered_element(&self, zs: &[i32]) -> Option<i32> {
        zs.iter()
            .copied()
            .find(|&z| !self.elements.iter().any(|s| symbol_to_z(s) == Some(z)))
    }

    /// Whether the test runs orbital basis `run` on the elements `zs`: its
    /// shells and ECPs there equal those of a listed set (aliases included).
    pub fn covers_basis(&self, run: &BasisSet, zs: &[i32]) -> bool {
        basis_listed(self.basis_sets, run, zs)
    }

    /// Whether the test runs auxiliary basis `run` on the elements `zs` (see
    /// [`Self::covers_basis`]).
    pub fn covers_aux_basis(&self, run: &BasisSet, zs: &[i32]) -> bool {
        basis_listed(self.aux_basis_sets, run, zs)
    }

    /// Whether the test runs functional `name` (case-insensitive).
    pub fn covers_functional(&self, name: &str) -> bool {
        let n = name.trim().to_ascii_lowercase();
        self.functionals.iter().any(|f| *f == n)
    }
}

/// What a run used, for [`Certificate::with_envelope`]. A `None` dimension
/// is not checked.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunEnvelope<'a> {
    /// Atomic number of every atom.
    pub atomic_numbers: &'a [i32],
    /// The orbital basis, compared by content on the run's elements.
    pub basis: Option<&'a BasisSet>,
    /// The auxiliary basis of the correlation method, compared by content.
    pub aux_basis: Option<&'a BasisSet>,
    /// The XC functional name.
    pub functional: Option<&'a str>,
}

impl Certificate {
    /// Attach `anchor`'s value as the measured error
    /// ([`Self::with_measured_error`]). The grade is not changed.
    pub fn with_energy_anchor(self, anchor: &EnergyAnchor) -> Self {
        self.with_measured_error(anchor.measured_error())
    }

    /// Add one warning for each dimension of `run` outside `anchor`'s
    /// covered set: the first uncovered element, the basis, the auxiliary
    /// basis, the functional. The grade is not changed.
    pub fn with_envelope(mut self, anchor: &EnergyAnchor, run: &RunEnvelope<'_>) -> Self {
        let outside = |what: String| outside_anchor_text(anchor, &what);
        if let Some(z) = anchor.first_uncovered_element(run.atomic_numbers) {
            let sym = z_to_symbol(z).map_or_else(|| format!("Z = {z}"), str::to_string);
            self.push_warning(outside(format!(
                "element {sym} (covered: {})",
                anchor.elements.join(", ")
            )));
        }
        let zs = run.atomic_numbers;
        if let Some(b) = run.basis.filter(|b| !anchor.covers_basis(b, zs)) {
            self.push_warning(outside(format!("basis {}", b.name)));
        }
        if let Some(b) = run.aux_basis.filter(|b| !anchor.covers_aux_basis(b, zs)) {
            self.push_warning(outside(format!("auxiliary basis {}", b.name)));
        }
        if let Some(f) = run.functional.filter(|f| !anchor.covers_functional(f)) {
            self.push_warning(outside(format!(
                "functional {f} (covered: {})",
                anchor.functionals.join(", ")
            )));
        }
        self
    }
}

fn outside_anchor_text(anchor: &EnergyAnchor, what: &str) -> String {
    format!(
        "{what} is outside the {} energy anchor ({})",
        anchor.method, anchor.evidence
    )
}

#[cfg(test)]
mod tests {
    use super::super::{workspace_root, GRADE_SOURCE_PAGE};
    use super::*;
    use std::collections::BTreeSet;
    use std::path::Path;

    const PROOF_PREFIX: &str = "https://github.com/mgoldey/ferric/blob/main/";

    /// Cells of one markdown table row (`\|` is a literal bar).
    fn cells(row: &str) -> Vec<String> {
        row.trim()
            .trim_matches('|')
            .replace("\\|", "\u{0}")
            .split('|')
            .map(|c| c.replace('\u{0}', "|").trim().to_string())
            .collect()
    }

    /// Header and rows of the table under [`ANCHOR_SOURCE_HEADING`].
    fn anchor_table(page: &str) -> (Vec<String>, Vec<Vec<String>>) {
        let start = page.find(ANCHOR_SOURCE_HEADING).expect("Anchors heading");
        let mut rows = page[start..]
            .lines()
            .skip_while(|l| !l.trim_start().starts_with('|'))
            .take_while(|l| l.trim_start().starts_with('|'));
        let header = cells(rows.next().expect("header row"));
        (header, rows.skip(1).map(cells).collect())
    }

    fn column(header: &[String], name: &str) -> usize {
        header
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("no {name:?} column in {header:?}"))
    }

    /// Every anchor-table disagreement of `a`, as messages.
    fn page_disagreements(a: &EnergyAnchor, root: &Path, page: &str) -> Vec<String> {
        let mut bad = Vec::new();
        if !root.join(a.evidence).is_file() {
            bad.push(format!(
                "{}: evidence {} does not exist",
                a.method, a.evidence
            ));
        }
        let (header, rows) = anchor_table(page);
        let (capability, proof, stated, reference) = (
            column(&header, "Capability"),
            column(&header, "Proof"),
            column(&header, "Stated agreement"),
            column(&header, "Reference"),
        );
        let link = format!("]({PROOF_PREFIX}{})", a.evidence);
        let hits: Vec<&Vec<String>> = rows.iter().filter(|r| r[proof].contains(&link)).collect();
        // Several rows can link one test file (MP3 and CCSD share
        // validation_cc.rs): take the row that names the method.
        let named = hits.iter().find(|r| r[capability].contains(a.method));
        let Some(row) = named.or(hits.first()) else {
            bad.push(format!(
                "{}: no Anchors row links {} from its Proof cell",
                a.method, a.evidence
            ));
            return bad;
        };
        let want = format!("{} {}", a.value, a.units);
        if !row[stated].contains(&want) {
            bad.push(format!(
                "{}: the Stated agreement cell lacks {want:?}: {:?}",
                a.method, row[stated]
            ));
        }
        if !row[reference].replace('`', "").contains(a.reference) {
            bad.push(format!(
                "{}: the Reference cell lacks {:?}: {:?}",
                a.method, a.reference, row[reference]
            ));
        }
        bad
    }

    fn set(xs: &[&str]) -> BTreeSet<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    /// `(elements, bases, aux bases)` of the `kind` cases of the basis
    /// coverage reference file.
    fn basis_coverage_sets(
        root: &Path,
        kind: &str,
    ) -> (BTreeSet<String>, BTreeSet<String>, BTreeSet<String>) {
        let path = root.join("testdata/reference/basis_coverage_pyscf.json");
        let text = std::fs::read_to_string(&path).expect("basis coverage reference");
        let refs: serde_json::Value = serde_json::from_str(&text).unwrap();
        let (mut el, mut bs, mut aux) = (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
        for c in refs["cases"].as_array().unwrap() {
            if c["kind"] != kind {
                continue;
            }
            for atom in c["atoms_angstrom"].as_str().unwrap().split(';') {
                el.insert(atom.split_whitespace().next().unwrap().to_string());
            }
            bs.insert(c["basis"].as_str().unwrap().to_string());
            if let Some(x) = c["auxbasis"].as_str() {
                aux.insert(x.to_string());
            }
        }
        (el, bs, aux)
    }

    /// Element symbols of `testdata/molecules/validation/<system>.xyz`.
    fn xyz_elements(root: &Path, system: &str) -> BTreeSet<String> {
        let path = root.join(format!("testdata/molecules/validation/{system}.xyz"));
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        text.lines()
            .skip(2)
            .filter_map(|l| l.split_whitespace().next())
            .map(str::to_string)
            .collect()
    }

    /// `(elements, bases, aux bases, systems)` recorded by the reference
    /// files `<dir>/<file>.json`; a missing file or field is pushed to `bad`.
    fn files_sets(
        root: &Path,
        dir: &str,
        files: &[&str],
        bad: &mut Vec<String>,
        method: &str,
    ) -> (
        BTreeSet<String>,
        BTreeSet<String>,
        BTreeSet<String>,
        BTreeSet<String>,
    ) {
        let (mut el, mut bs, mut aux, mut systems) = (
            BTreeSet::new(),
            BTreeSet::new(),
            BTreeSet::new(),
            BTreeSet::new(),
        );
        for f in files {
            let path = root.join(dir).join(format!("{f}.json"));
            let Ok(text) = std::fs::read_to_string(&path) else {
                bad.push(format!("{method}: no reference {dir}/{f}.json"));
                continue;
            };
            let r: serde_json::Value = serde_json::from_str(&text).unwrap();
            let field =
                |keys: &[&str]| keys.iter().find_map(|k| r[*k].as_str()).map(str::to_string);
            let (Some(system), Some(basis), Some(aux_basis)) = (
                field(&["system"]),
                field(&["basis"]),
                field(&["aux_basis", "auxbasis", "aux"]),
            ) else {
                bad.push(format!(
                    "{method}: {dir}/{f}.json lacks system, basis or aux basis"
                ));
                continue;
            };
            if *f != format!("{system}_{basis}") {
                bad.push(format!(
                    "{method}: {dir}/{f}.json records {system} / {basis}"
                ));
            }
            let stem = field(&["geometry_stem"]).unwrap_or_else(|| system.clone());
            el.extend(xyz_elements(root, &stem));
            bs.insert(basis);
            aux.insert(aux_basis);
            systems.insert(system);
        }
        (el, bs, aux, systems)
    }

    /// `(elements, bases, systems)` of the inline-geometry reference
    /// `files`; a missing file or field is pushed to `bad`.
    fn inline_sets(
        root: &Path,
        files: &[&str],
        bad: &mut Vec<String>,
        method: &str,
    ) -> (BTreeSet<String>, BTreeSet<String>, BTreeSet<String>) {
        let (mut el, mut bs, mut systems) = (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
        for f in files {
            let Ok(text) = std::fs::read_to_string(root.join(f)) else {
                bad.push(format!("{method}: no reference {f}"));
                continue;
            };
            let r: serde_json::Value = serde_json::from_str(&text).unwrap();
            let records = match r["systems"].as_array() {
                Some(a) => a.clone(),
                None => vec![r],
            };
            for rec in &records {
                let (Some(system), Some(basis), Some(xyz)) = (
                    rec["system"].as_str(),
                    rec["basis"].as_str(),
                    rec["xyz"].as_str(),
                ) else {
                    bad.push(format!(
                        "{method}: a record of {f} lacks system, basis or xyz"
                    ));
                    continue;
                };
                el.extend(
                    xyz.lines()
                        .skip(2)
                        .filter_map(|l| l.split_whitespace().next())
                        .map(str::to_string),
                );
                bs.insert(basis.to_string());
                systems.insert(system.to_string());
            }
        }
        (el, bs, systems)
    }

    /// Every disagreement between `a`'s covered sets and the data its
    /// evidence test reads.
    fn data_disagreements(a: &EnergyAnchor, root: &Path) -> Vec<String> {
        let mut bad = Vec::new();
        let mut files_bad = Vec::new();
        let mut same = |what: &str, ours: BTreeSet<String>, theirs: BTreeSet<String>| {
            if ours != theirs {
                bad.push(format!(
                    "{}: {what} {ours:?} differ from the test data {theirs:?}",
                    a.method
                ));
            }
        };
        match a.data {
            AnchorData::BasisCoverage { kind } => {
                let (el, bs, aux) = basis_coverage_sets(root, kind);
                same("elements", set(a.elements), el);
                same("basis sets", set(a.basis_sets), bs);
                same("aux basis sets", set(a.aux_basis_sets), aux);
            }
            AnchorData::Files { dir, files } => {
                let (el, bs, aux, systems) = files_sets(root, dir, files, &mut files_bad, a.method);
                same("elements", set(a.elements), el);
                same("basis sets", set(a.basis_sets), bs);
                same("aux basis sets", set(a.aux_basis_sets), aux);
                let test = std::fs::read_to_string(root.join(a.evidence)).unwrap_or_default();
                for w in systems
                    .iter()
                    .map(String::as_str)
                    .chain(a.functionals.iter().copied())
                {
                    if !test.contains(&format!("\"{w}\"")) {
                        files_bad.push(format!("{}: {} never names \"{w}\"", a.method, a.evidence));
                    }
                }
            }
            AnchorData::InlineGeometry { files } => {
                let (el, bs, systems) = inline_sets(root, files, &mut files_bad, a.method);
                same("elements", set(a.elements), el);
                same("basis sets", set(a.basis_sets), bs);
                same("aux basis sets", set(a.aux_basis_sets), BTreeSet::new());
                let test = std::fs::read_to_string(root.join(a.evidence)).unwrap_or_default();
                for w in &systems {
                    if !test.contains(&format!("\"{w}\"")) {
                        files_bad.push(format!("{}: {} never names \"{w}\"", a.method, a.evidence));
                    }
                }
            }
            AnchorData::Directory { dir, systems } => {
                let el = systems.iter().flat_map(|s| xyz_elements(root, s)).collect();
                same("elements", set(a.elements), el);
                for s in systems {
                    for b in a.basis_sets {
                        if !root.join(dir).join(format!("{s}_{b}.json")).is_file() {
                            bad.push(format!("{}: no reference {dir}/{s}_{b}.json", a.method));
                        }
                    }
                }
                let test = std::fs::read_to_string(root.join(a.evidence)).unwrap_or_default();
                let words = systems
                    .iter()
                    .chain(a.basis_sets)
                    .chain(a.aux_basis_sets)
                    .chain(a.functionals);
                for w in words {
                    if !test.contains(&format!("\"{w}\"")) {
                        bad.push(format!("{}: {} never names \"{w}\"", a.method, a.evidence));
                    }
                }
            }
        }
        bad.extend(files_bad);
        bad
    }

    #[test]
    fn anchors_match_the_page() {
        let root = workspace_root();
        let page = std::fs::read_to_string(root.join(GRADE_SOURCE_PAGE)).unwrap();
        let bad: Vec<String> = ENERGY_ANCHORS
            .iter()
            .flat_map(|a| {
                let mut b = page_disagreements(a, &root, &page);
                b.extend(data_disagreements(a, &root));
                b
            })
            .collect();
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }

    /// Each reference directory is split exactly among its anchors: a system
    /// or basis added to the data without a registry entry fails here.
    #[test]
    fn anchor_directories_are_covered_exactly() {
        let root = workspace_root();
        let mut dirs: std::collections::BTreeMap<&str, BTreeSet<String>> = Default::default();
        for a in ENERGY_ANCHORS {
            match a.data {
                AnchorData::Directory { dir, systems } => {
                    let e = dirs.entry(dir).or_default();
                    for s in systems {
                        for b in a.basis_sets {
                            e.insert(format!("{s}_{b}.json"));
                        }
                    }
                }
                AnchorData::Files { dir, files } => {
                    let e = dirs.entry(dir).or_default();
                    e.extend(files.iter().map(|f| format!("{f}.json")));
                }
                AnchorData::BasisCoverage { .. } | AnchorData::InlineGeometry { .. } => {}
            }
        }
        for (dir, ours) in dirs {
            let on_disk: BTreeSet<String> = std::fs::read_dir(root.join(dir))
                .unwrap_or_else(|e| panic!("{dir}: {e}"))
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .filter(|n| n.ends_with(".json"))
                .collect();
            assert_eq!(ours, on_disk, "{dir}");
        }
    }

    #[test]
    fn the_five_kinds_have_anchors_and_values_parse() {
        for kind in ["rhf", "uhf", "rohf", "ksdft", "rimp2"] {
            assert!(
                ENERGY_ANCHORS.iter().any(|a| a.kind == kind),
                "no anchor for {kind}"
            );
        }
        let mut methods = BTreeSet::new();
        for a in ENERGY_ANCHORS {
            assert!(methods.insert(a.method), "{} twice", a.method);
            let e = a.measured_error();
            assert_eq!(e.value, a.value.parse::<f64>().unwrap());
            assert_eq!(e.evidence, a.evidence);
            for s in a.elements {
                assert!(symbol_to_z(s).is_some(), "{}: {s}", a.method);
            }
            for b in a.basis_sets.iter().chain(a.aux_basis_sets) {
                assert!(crate::basis::bundled(b).is_ok(), "{}: {b}", a.method);
            }
        }
    }

    /// The rule behind the mutation ledger: no Proven grade without an anchor.
    #[test]
    fn every_proven_kind_has_an_anchor() {
        let missing: Vec<&str> = super::super::METHOD_KIND_GRADES
            .iter()
            .filter(|(_, g)| *g == super::super::Grade::Proven)
            .map(|(k, _)| *k)
            .filter(|k| !ENERGY_ANCHORS.iter().any(|a| a.kind == *k))
            .collect();
        assert!(
            missing.is_empty(),
            "Proven kinds without an anchor: {missing:?}"
        );
        for a in ENERGY_ANCHORS {
            assert_eq!(
                super::super::method_kind_grade(a.kind),
                super::super::Grade::Proven,
                "{}: kind {} is not graded Proven",
                a.method,
                a.kind
            );
        }
    }

    #[test]
    fn envelope_warns_per_dimension_and_never_changes_the_grade() {
        let rks = energy_anchor("RKS").unwrap();
        let base = Certificate::for_method_kind("ksdft").with_energy_anchor(rks);
        assert_eq!(base.measured_error.as_ref().unwrap().value, 4.2e-12);
        // Inside: H2S / def2-SVP / PBE, any spelling of the names.
        let bundled = |n: &str| crate::basis::bundled(n).unwrap();
        let (svp, b631g, ri) = (bundled("DEF2-SVP"), bundled("6-31g"), bundled("cc-pvdz-ri"));
        let inside = base.clone().with_envelope(
            rks,
            &RunEnvelope {
                atomic_numbers: &[16, 1, 1],
                basis: Some(&svp),
                aux_basis: None,
                functional: Some("PBE"),
            },
        );
        assert_eq!(inside, base, "a covered run gets no warning");
        // Outside on every dimension: water / 6-31G / SCAN.
        let out = base.clone().with_envelope(
            rks,
            &RunEnvelope {
                atomic_numbers: &[1, 8, 7],
                basis: Some(&b631g),
                aux_basis: Some(&ri),
                functional: Some("SCAN"),
            },
        );
        assert_eq!(out.grade, base.grade);
        assert_eq!(out.measured_error, base.measured_error);
        let ev = KS_ENERGIES;
        assert_eq!(
            out.warnings,
            [
                format!(
                    "element O (covered: H, Si, S, Cl) is outside the RKS energy anchor ({ev})"
                ),
                format!("basis 6-31g is outside the RKS energy anchor ({ev})"),
                format!("auxiliary basis cc-pvdz-ri is outside the RKS energy anchor ({ev})"),
                format!(
                    "functional SCAN (covered: pbe, b3lyp, hse06) is outside the RKS energy \
                     anchor ({ev})"
                ),
            ]
        );
    }

    /// Where each anchor's production-code mutation is recorded; written by
    /// `scripts/check_anchor_mutations.py --write`.
    const MUTATION_LEDGER: &str = "testdata/mutations/ledger.json";

    /// Anchors whose mutation has no recorded result yet. Each one is printed
    /// by `mutation_ledger_matches_the_registry`.
    const PENDING_MUTATIONS: &[&str] = &[];

    /// SHA-256 of `data` as lower-case hex (FIPS 180-4).
    fn sha256_hex(data: &[u8]) -> String {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let mut msg = data.to_vec();
        msg.push(0x80);
        while msg.len() % 64 != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
        for block in msg.as_chunks::<64>().0 {
            let mut w = [0u32; 64];
            for (i, b) in block.as_chunks::<4>().0.iter().enumerate() {
                w[i] = u32::from_be_bytes(*b);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ (!e & g);
                let t1 = hh
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let t2 = s0.wrapping_add(maj);
                (hh, g, f, e, d, c, b, a) =
                    (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
            }
            for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
                *x = x.wrapping_add(y);
            }
        }
        h.iter().map(|x| format!("{x:08x}")).collect()
    }

    #[test]
    fn sha256_matches_the_fips_vectors() {
        let cases = [
            (
                "",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", // pragma: allowlist secret
            ),
            (
                "abc",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", // pragma: allowlist secret
            ),
            (
                "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", // pragma: allowlist secret
            ),
        ];
        for (input, want) in cases {
            assert_eq!(sha256_hex(input.as_bytes()), want, "{input:?}");
        }
    }

    /// Every disagreement of one ledger row with the registry and the files
    /// on disk, pushed to `bad`. Returns the anchor when the row carries a
    /// recorded result.
    fn ledger_row_disagreements(
        row: &serde_json::Value,
        root: &Path,
        bad: &mut Vec<String>,
    ) -> Option<&'static str> {
        let field = |k: &str| row[k].as_str().unwrap_or_default();
        let name = field("anchor");
        let Some(a) = ENERGY_ANCHORS.iter().find(|a| a.method == name) else {
            bad.push(format!("ledger row {name:?} is not an anchor"));
            return None;
        };
        let mut push = |m: String| bad.push(format!("{name}: {m}"));
        if field("evidence") != a.evidence {
            push(format!(
                "ledger evidence {:?}, registry {:?}",
                field("evidence"),
                a.evidence
            ));
        }
        let test = field("test");
        let src = std::fs::read_to_string(root.join(a.evidence)).unwrap_or_default();
        if test.is_empty() || !src.contains(&format!("fn {test}(")) {
            push(format!("{} has no test fn {test:?}", a.evidence));
        }
        let patch = field("patch");
        let bytes = std::fs::read(root.join(patch)).unwrap_or_else(|e| {
            push(format!("patch {patch:?}: {e}"));
            Vec::new()
        });
        let text = String::from_utf8_lossy(&bytes);
        let changed: Vec<&str> = text
            .lines()
            .filter_map(|l| l.strip_prefix("diff --git a/"))
            .filter_map(|l| l.split_whitespace().next())
            .collect();
        let production = |p: &&str| {
            let parts: Vec<&str> = p.split('/').collect();
            parts.len() >= 4 && parts[0] == "crates" && parts[2] == "src"
        };
        if !patch.starts_with("testdata/mutations/") || changed.is_empty() {
            push(format!(
                "patch {patch:?} is not a diff under testdata/mutations/"
            ));
        }
        if let Some(p) = changed.iter().find(|p| !production(p)) {
            push(format!(
                "patch {patch:?} changes {p}, outside crates/*/src/"
            ));
        }
        let res = &row["result"];
        if res.is_null() {
            return None;
        }
        let r = |k: &str| res[k].as_str().unwrap_or_default();
        if r("patch_sha256") != sha256_hex(&bytes) {
            push(format!(
                "recorded patch sha256 {} differs from {patch} ({})",
                r("patch_sha256"),
                sha256_hex(&bytes)
            ));
        }
        let commit = r("base_commit");
        if commit.len() != 40 || !commit.bytes().all(|b| b.is_ascii_hexdigit()) {
            push(format!("base_commit {commit:?} is not a commit hash"));
        }
        if r("failing_test") != test {
            push(format!(
                "failing_test {:?} is not {test:?}",
                r("failing_test")
            ));
        }
        let tests_dir = a.evidence.rsplit_once('/').map_or("", |(d, _)| d);
        if !r("assertion").starts_with(&format!("{tests_dir}/")) {
            push(format!(
                "assertion {:?} is outside {tests_dir}/",
                r("assertion")
            ));
        }
        Some(a.method)
    }

    /// The mutation ledger agrees with [`ENERGY_ANCHORS`] and with the patch
    /// files: every anchor has a recorded result (patch hash, base commit,
    /// failing test) or is listed in [`PENDING_MUTATIONS`].
    #[test]
    fn mutation_ledger_matches_the_registry() {
        let root = workspace_root();
        let text = std::fs::read_to_string(root.join(MUTATION_LEDGER)).expect("mutation ledger");
        let ledger: serde_json::Value = serde_json::from_str(&text).expect("ledger JSON");
        let mut bad = Vec::new();
        let mut recorded = BTreeSet::new();
        let mut rows = BTreeSet::new();
        for row in ledger["rows"].as_array().expect("rows") {
            let name = row["anchor"].as_str().unwrap_or_default().to_string();
            if !rows.insert(name.clone()) {
                bad.push(format!("{name}: two ledger rows"));
            }
            if let Some(m) = ledger_row_disagreements(row, &root, &mut bad) {
                recorded.insert(m);
            }
        }
        for a in ENERGY_ANCHORS {
            let pending = PENDING_MUTATIONS.contains(&a.method);
            match (recorded.contains(a.method), pending) {
                (false, false) => bad.push(format!(
                    "{}: no recorded mutation in {MUTATION_LEDGER} and not pending",
                    a.method
                )),
                (true, true) => bad.push(format!("{}: recorded and also pending", a.method)),
                _ => {}
            }
        }
        for p in PENDING_MUTATIONS {
            if energy_anchor(p).is_none() {
                bad.push(format!("pending {p:?} is not an anchor"));
            }
            println!("mutation pending: {p}");
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }

    #[test]
    fn registry_aliases_are_covered() {
        let b = |n: &str| crate::basis::bundled(n).unwrap();
        let (water, nh3) = ([8, 1, 1], [7, 1, 1, 1]);
        let mp2 = energy_anchor("RI-MP2").unwrap();
        assert!(mp2.covers_aux_basis(&b("cc-pvdz-ri"), &water));
        assert!(mp2.covers_aux_basis(&b("CC-PVDZ/C"), &water));
        assert!(energy_anchor("U-RI-MP2")
            .unwrap()
            .covers_aux_basis(&b("cc-pvdz-rifit"), &nh3));
        assert!(energy_anchor("RHF")
            .unwrap()
            .covers_basis(&b("6-31G(d)"), &water));
        // A different set is not an alias.
        assert!(!mp2.covers_aux_basis(&b("6-31g"), &water));
        assert!(!mp2.covers_basis(&b("sto-3g"), &water));
    }

    /// Coverage is by content: a set carrying a covered set's name but one
    /// changed contraction coefficient on an element of the run is outside,
    /// a change on an element the run does not use is not, and a covered
    /// set under another name is inside.
    #[test]
    fn a_loaded_basis_with_a_bundled_name_is_covered_only_if_its_contents_match() {
        let rhf = energy_anchor("RHF").unwrap();
        let water = [8, 1, 1];
        let sto = crate::basis::bundled("sto-3g").unwrap();
        assert!(
            rhf.covers_basis(&sto, &water),
            "precondition: sto-3g is covered"
        );
        let mut edited = sto.clone();
        edited.shells.get_mut(&8).unwrap()[0].coefficients[0] *= 1.01;
        assert!(!rhf.covers_basis(&edited, &water), "edited O contraction");
        assert!(rhf.covers_basis(&edited, &[1, 1]), "O is not in the run");
        let mut renamed = sto.clone();
        renamed.name = "my-minimal-basis".to_string();
        assert!(
            rhf.covers_basis(&renamed, &water),
            "same contents, other name"
        );
        let base = Certificate::for_method_kind("rhf").with_energy_anchor(rhf);
        let out = base.clone().with_envelope(
            rhf,
            &RunEnvelope {
                atomic_numbers: &water,
                basis: Some(&edited),
                ..Default::default()
            },
        );
        assert_eq!(out.warnings.len(), base.warnings.len() + 1);
        assert!(
            out.warnings
                .last()
                .unwrap()
                .starts_with("basis sto-3g is outside"),
            "{:?}",
            out.warnings
        );
    }
}
