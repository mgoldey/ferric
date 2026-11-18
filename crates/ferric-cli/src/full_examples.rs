//! `examples/full/`: one input per `method.kind` with every key written out.
//!
//! Each file is a minimal input (the keys that have no default) expanded so
//! that every key of every section the run reads is either set to the value
//! the CLI would use when the key is omitted, or listed in a comment line
//! `# key: ...` that says why no value is written (the default is "unset", or
//! the key is read only under another key's value or by another kind).
//!
//! The files are generated from the table in this module:
//!
//! ```text
//! FERRIC_WRITE_FULL_EXAMPLES=1 cargo test -p ferric-cli --release full_example
//! ```
//!
//! and three tests hold them to it:
//!
//! * `full_examples_are_generated`: each file on disk equals what the table
//!   generates now, and no other file sits in `examples/full/`;
//! * `full_examples_name_every_key_of_their_sections`: in each file, every
//!   serde field of every section (read from serde's own `FIELDS` table, so a
//!   new config field fails here until it is classified) appears as a key or
//!   as a `# key:` comment line, and the file has every section its kind
//!   reads;
//! * `full_examples_load`: each file passes `load_config`.
//!
//! That a written value IS the default is checked by running the CLI: the
//! integration test `tests/full_examples_anchor.rs` runs each file and the
//! minimal input in its header and requires identical run logs.
//!
//! Which sections a kind reads, and which keys of a section are active on its
//! default branch, are hand-listed here: the CLI refuses only some unread
//! keys, so they cannot all be derived from its refusal logic. The CAS and
//! OP-MP3 kinds are the exception: their section and `[scf]` key lists are
//! the CLI's own (`crate::cas::READ_SECTIONS`, `crate::opmp3::READ_SECTIONS`).

use super::*;
use crate::cas::{CasCfg, CasCiCfg};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

/// One fully specified file: its name, and the minimal input it expands.
struct Target {
    name: &'static str,
    minimal: String,
}

const W_STO: &str =
    "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n[basis]\nname = \"sto-3g\"\n";
const W_631: &str =
    "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n[basis]\nname = \"6-31g\"\n";
const OH_STO: &str =
    "[molecule]\nxyz = \"testdata/molecules/oh.xyz\"\nmultiplicity = 2\n[basis]\nname = \"sto-3g\"\n";
const H2_631: &str = "[molecule]\nxyz = \"testdata/molecules/h2.xyz\"\n[basis]\nname = \"6-31g\"\n";

/// The fully specified files. `minimal` holds only keys with no default
/// (and the method), so the run it describes is the defaults' run.
fn targets() -> Vec<Target> {
    let t = |name: &'static str, mol: &str, rest: &str| Target {
        name,
        minimal: format!("{mol}{rest}"),
    };
    vec![
        t("rhf", W_STO, "[method]\nkind = \"rhf\"\n"),
        t("rhf-optimize", H2_631, "[method]\nkind = \"rhf\"\ntask = \"optimize\"\n"),
        t("rhf-frequencies", W_STO, "[method]\nkind = \"rhf\"\ntask = \"frequencies\"\n"),
        t("uhf", OH_STO, "[method]\nkind = \"uhf\"\n"),
        t("rohf", OH_STO, "[method]\nkind = \"rohf\"\n"),
        t("ksdft", W_STO, "[method]\nkind = \"ksdft\"\n"),
        t("ksdft-optimize", H2_631, "[method]\nkind = \"ksdft\"\ntask = \"optimize\"\n"),
        t("ksdft-frequencies", W_STO, "[method]\nkind = \"ksdft\"\ntask = \"frequencies\"\n"),
        t("rimp2", W_STO, "[method]\nkind = \"rimp2\"\n"),
        t("rimp2-optimize", H2_631, "[method]\nkind = \"rimp2\"\ntask = \"optimize\"\n"),
        t("rimp2-frequencies", H2_631, "[method]\nkind = \"rimp2\"\ntask = \"frequencies\"\n"),
        t("mp3", W_STO, "[method]\nkind = \"mp3\"\n"),
        t("oo-rimp2", W_STO, "[method]\nkind = \"oo-rimp2\"\n"),
        t("att-rimp2", W_STO, "[method]\nkind = \"att-rimp2\"\n"),
        t("mp2-v", W_STO, "[method]\nkind = \"mp2-v\"\n"),
        t("scs-mp2", W_STO, "[method]\nkind = \"scs-mp2\"\n"),
        // The four SCS-MP2(2terfc) parameters are fitted in aug-cc-pVTZ only;
        // in any other basis they are required.
        t(
            "scs-mp2-2terfc",
            W_STO,
            "[method]\nkind = \"scs-mp2-2terfc\"\n[mp2]\nr0_bonded = 0.75\nr0_nonbonded = 1.05\nc_os = 1.27\nc_ss = 4.05\n",
        ),
        t("laplace-mp2", W_STO, "[method]\nkind = \"laplace-mp2\"\n"),
        t("laplace-sos-mp2", W_STO, "[method]\nkind = \"laplace-sos-mp2\"\n"),
        t("pdep-rpa", W_STO, "[method]\nkind = \"pdep-rpa\"\n"),
        t("rs-mp2-rpa", W_STO, "[method]\nkind = \"rs-mp2-rpa\"\n"),
        t("gw", W_631, "[method]\nkind = \"gw\"\n"),
        t("bse-tda", W_631, "[method]\nkind = \"bse-tda\"\n"),
        t(
            "tdhf-static-polarizability",
            W_631,
            "[method]\nkind = \"tdhf-static-polarizability\"\n[rpa]\nxc = \"PBE\"\n[gw]\nscissor = 0.36\n",
        ),
        t("ccsd", W_STO, "[method]\nkind = \"ccsd\"\n"),
        t("ccd", W_STO, "[method]\nkind = \"ccd\"\n"),
        t("ccsd(t)", W_STO, "[method]\nkind = \"ccsd(t)\"\n"),
        t("eom-ccsd", W_STO, "[method]\nkind = \"eom-ccsd\"\n"),
        t("linlccd", W_STO, "[method]\nkind = \"linlccd\"\n"),
        t("drpa", W_STO, "[method]\nkind = \"drpa\"\n"),
        t("wb97x-l-v", W_STO, "[method]\nkind = \"wb97x-l-v\"\n"),
        t("b2plyp", W_STO, "[method]\nkind = \"b2plyp\"\n"),
        t("dsd-pbep86", W_STO, "[method]\nkind = \"dsd-pbep86\"\n"),
        t("tda", W_631, "[method]\nkind = \"tda\"\n[tddft]\nxc = \"PBE\"\n"),
        t("tddft", W_631, "[method]\nkind = \"tddft\"\n[tddft]\nxc = \"PBE\"\n"),
        t(
            "casscf",
            W_STO,
            "[method]\nkind = \"casscf\"\n[cas]\nn_active = 4\nn_elec_active = [2, 2]\n",
        ),
        t(
            "nevpt2",
            W_STO,
            "[method]\nkind = \"nevpt2\"\n[cas]\nn_active = 4\nn_elec_active = [2, 2]\n",
        ),
        t("opmp3", W_631, "[method]\nkind = \"opmp3\"\n"),
    ]
}

/// What a file says about one key.
enum Line {
    /// Written with this TOML value: the value an omitted key resolves to.
    Set(String),
    /// Not written, because the default is "absent" (no value expresses it).
    Unset(&'static str),
    /// Not written: read only under another key's value or by another kind.
    Off(&'static str),
}
use Line::{Off, Set, Unset};

fn s(v: &str) -> Line {
    Set(format!("{v:?}"))
}
fn n(v: &str) -> Line {
    Set(v.to_string())
}

/// The run a minimal input describes, as the rules below need it.
struct Ctx {
    kind: String,
    task: String,
    mult: usize,
    basis: String,
}

impl Ctx {
    fn is(&self, kinds: &[&str]) -> bool {
        kinds.contains(&self.kind.as_str())
    }
    fn energy(&self) -> bool {
        self.task == "energy"
    }
    /// CASSCF, SC-NEVPT2 and OP-MP3 run their own reference SCF.
    fn own_start(&self) -> bool {
        self.is(&["casscf", "nevpt2", "opmp3"])
    }
    /// The SCF-only kinds.
    fn scf_kind(&self) -> bool {
        self.is(&["rhf", "uhf", "rohf", "ksdft"])
    }
    /// `[scf] jk` governs the reference and resolves to RI-JK.
    fn fitted_jk(&self) -> bool {
        scf_jk_governs(&self.kind, &self.task) && self.mult == 1
    }
    /// Kinds whose SCF defaults to RI-J/RI-K on `def2-universal-jkfit`.
    fn ri_default(&self) -> bool {
        self.is(&[
            "ksdft",
            "pdep-rpa",
            "gw",
            "bse-tda",
            "tdhf-static-polarizability",
            "tda",
            "tddft",
        ])
    }
    /// The double hybrids, which build their own Kohn-Sham reference.
    fn double_hybrid(&self) -> bool {
        self.is(&["wb97x-l-v", "b2plyp", "dsd-pbep86"])
    }
    /// The run builds a Kohn-Sham grid (gw excepted: it refuses grid keys).
    fn ks_grid(&self) -> bool {
        self.is(&["ksdft", "tda", "tddft"]) || self.double_hybrid()
    }
    /// The reference SCF builds four-centre K.
    fn exact_k(&self) -> bool {
        !self.own_start() && !self.fitted_jk() && !self.ri_default() && !self.double_hybrid()
    }
    /// The closed-shell, non-laddered SCF path that runs the DF-guess
    /// pre-stage.
    fn df_guess_path(&self) -> bool {
        self.mult == 1
            && !self.own_start()
            && !self.is(&["rhf", "ksdft"])
            && !self.fitted_jk()
            && !self.double_hybrid()
    }
}

/// One key: its section, name, one-line meaning, and what a file says.
struct Key {
    section: &'static str,
    key: &'static str,
    doc: &'static str,
    line: fn(&Ctx) -> Line,
}

macro_rules! key {
    ($sec:literal, $key:literal, $doc:literal, $line:expr) => {
        Key {
            section: $sec,
            key: $key,
            doc: $doc,
            line: $line,
        }
    };
}

const OWN_START: &str = "not read: this kind runs its own reference SCF, which reads only \
                         max_iter, energy_conv and density_conv";
const COSX: &str = "only with k_builder = \"cosx\" or jk = \"rijcosx\"";
const FOUR_CENTRE: &str = "only on a four-centre exchange route; this run fits K";
const NO_GRID: &str = "only on a run with a Kohn-Sham grid";
const NPZ: &str = "only with export_npz";

/// Every key the files may mention, in file order within each section.
fn keys() -> Vec<Key> {
    vec![
        // ---- [molecule] / [basis] / [method]: the required keys come from
        // the minimal input; the rest are listed here.
        key!("molecule", "xyz", "XYZ geometry in Angstrom, relative to the working directory (required)", |_| Off("required")),
        key!("molecule", "charge", "total charge", |_| n("0")),
        key!("molecule", "multiplicity", "2S + 1", |c| Set(c.mult.to_string())),
        key!("basis", "name", "bundled basis name, case-insensitive (required unless path is set)", |_| Off("required")),
        key!("basis", "path", "Gaussian-94 basis file", |_| Off("read only when name is absent")),
        key!("method", "kind", "the method (required)", |_| Off("required")),
        key!("method", "task", "energy | optimize | frequencies", |c| s(&c.task)),
        // ---- [scf]
        key!("scf", "max_iter", "SCF iteration cap", |_| n("200")),
        key!("scf", "energy_conv", "energy-change bound, a sanity bound and not the convergence signal", |c| {
            n(if c.is(&["opmp3"]) { "1e-10" } else { "1e-3" })
        }),
        key!("scf", "density_conv", "density-change threshold, the convergence signal", |c| {
            n(if c.is(&["opmp3"]) { "1e-8" } else { "1e-6" })
        }),
        key!("scf", "diis_size", "DIIS subspace size", |_| n("8")),
        key!("scf", "diis", "pulay | adiis | ediis", |c| if c.mult == 1 { s("pulay") } else { Off("the open-shell SCF runs Pulay DIIS; adiis/ediis are refused") }),
        key!("scf", "diis_switch_thresh", "error level where ADIIS/EDIIS hand over to Pulay", |_| Off("only with diis = \"adiis\" or \"ediis\" (default 1e-1)")),
        key!("scf", "smearing_sigma", "Fermi-Dirac smearing width, Hartree", |c| if c.mult == 1 { Unset("integer occupations") } else { Off("closed-shell SCF only") }),
        key!("scf", "guess", "minao | sad (alias of minao) | hcore", |_| s("minao")),
        key!("scf", "guess_basis", "secondary-basis guess: converge in this basis first, then project", |_| Unset("no secondary-basis guess")),
        key!("scf", "guess_projection", "fock | overlap", |_| Off("only with guess_basis (default \"fock\")")),
        key!("scf", "soscf", "second-order (Newton) step in the SCF tail", |_| n("false")),
        key!("scf", "aurora", "AURORA auxiliary-curvature accelerator", |c| if c.is(&["rhf"]) && c.mult == 1 { n("false") } else { Off("closed-shell rhf only") }),
        key!("scf", "trah_trigger", "DIIS error below which the TRAH solver takes over", |c| if c.is(&["rhf", "uhf"]) { Unset("TRAH off") } else { Off("rhf and uhf only") }),
        key!("scf", "integral_thresh", "integral screening threshold", |_| n("1e-12")),
        key!("scf", "eri_precision", "libint primitive-screening precision, 0 to 1e-8 (omitted: FERRIC_ERI_PRECISION, else 1e-20)", |_| n("1e-20")),
        key!("scf", "jk_storage", "RI tensor storage: auto | memory | disk | direct | split", |c| if c.fitted_jk() || c.ri_default() || c.double_hybrid() { s("auto") } else { Off("only with RI-J or RI-K") }),
        key!("scf", "ri_j_precision", "RI-J tensor storage precision: f32 | f64 (omitted: f32 for energy, f64 for optimize/frequencies)", |c| {
            if c.fitted_jk() || c.ri_default() || c.double_hybrid() { s(if c.energy() { "f32" } else { "f64" }) } else { Off("only with RI-J") }
        }),
        key!("scf", "ri_k_precision", "RI-K tensor storage precision: f64 | f32", |c| if c.fitted_jk() || c.ri_default() || c.double_hybrid() { s("f64") } else { Off("only with RI-K") }),
        key!("scf", "k_builder", "exchange builder: direct | link | cosx", |c| if c.exact_k() { s("direct") } else { Off(FOUR_CENTRE) }),
        key!("scf", "screening", "four-centre screening: schwarz | csb | csam", |c| if c.exact_k() { s("schwarz") } else { Off(FOUR_CENTRE) }),
        key!("scf", "eri", "four-centre J/K build: auto | direct | incore", |c| if c.exact_k() { s("auto") } else { Off(FOUR_CENTRE) }),
        key!("scf", "relativistic", "x2c1e: spin-free X2C-1e core Hamiltonian (energy only)", |c| {
            if c.energy() && X2C_KINDS.contains(&c.kind.as_str()) { Unset("nonrelativistic") } else { Off("energies of the X2C kinds only") }
        }),
        key!("scf", "cosx_grid", "COSX SCF grid", |_| Off(COSX)),
        key!("scf", "cosx_final_pass", "COSX final pass on a larger grid", |_| Off(COSX)),
        key!("scf", "cosx_final_grid", "COSX final-pass grid", |_| Off(COSX)),
        key!("scf", "cosx_grid_schedule", "coarse-then-production COSX grid", |_| Off(COSX)),
        key!("scf", "cosx_overlap_fit", "COSX overlap fitting", |_| Off(COSX)),
        key!("scf", "cosx_backend", "COSX kernel", |_| Off(COSX)),
        key!("scf", "cosx_screen_thresh", "COSX screen", |_| Off(COSX)),
        key!("scf", "cosx_half_transform", "COSX half transform", |_| Off(COSX)),
        key!("scf", "cosx_fp64_multiplier", "COSX precision-router threshold", |_| Off(COSX)),
        key!("scf", "df_j_aux", "RI-J auxiliary basis; \"\" is exact J", |c| {
            if c.ri_default() { s(DEFAULT_SCF_JK_AUX) } else if c.fitted_jk() { Off("jk sets the fit on this kind; the two spellings cannot be combined") } else if c.double_hybrid() { Off("the double hybrid fixes its own RI-JK reference") } else { Unset("exact four-centre J") }
        }),
        key!("scf", "df_k_aux", "RI-K auxiliary basis; \"\" is exact K", |c| {
            if c.ri_default() { s(DEFAULT_SCF_JK_AUX) } else if c.fitted_jk() { Off("jk sets the fit on this kind; the two spellings cannot be combined") } else if c.double_hybrid() { Off("the double hybrid fixes its own RI-JK reference") } else { Unset("exact four-centre K") }
        }),
        key!("scf", "jk", "reference J/K route of the MP2 family: exact | rijk | rijcosx", |c| if c.fitted_jk() { s("rijk") } else { Off("MP2-family kinds only") }),
        key!("scf", "jk_aux", "JK-fit basis of jk = rijk/rijcosx", |c| {
            if !c.fitted_jk() { return Off("only with jk"); }
            if c.is(&["rs-mp2-rpa"]) { s(DEFAULT_SCF_JK_AUX) } else { s(ferric_scf::scf_jk::matched_jk_aux(&c.basis)) }
        }),
        key!("scf", "level_shift", "virtual-block level shift, Hartree", |c| if c.double_hybrid() { Off("the double hybrid's own SCF") } else { n("0.0") }),
        key!("scf", "mom_after_iter", "maximum-overlap pinning after this many iterations; 0 = aufbau", |_| n("0")),
        key!("scf", "mom_reference", "previous | initial", |c| if c.is(&["uhf"]) { s("previous") } else { Off("UHF/UKS only") }),
        key!("scf", "verbose", "one line per SCF iteration", |_| n("false")),
        key!("scf", "df_guess", "two-stage SCF: DF first, then exact", |c| {
            if c.df_guess_path() { n("true") } else if c.is(&["rhf", "ksdft"]) { Off("not composed with the rhf/ksdft convergence ladder") } else if c.fitted_jk() { Off("skipped: the reference already fits J and K") } else { Off("closed-shell correlated kinds only") }
        }),
        key!("scf", "df_guess_aux", "aux basis of the DF-guess pre-stage", |c| if c.df_guess_path() { s(DEFAULT_SCF_JK_AUX) } else { Off("only with df_guess") }),
        key!("scf", "df_increments", "DF-corrected incremental Fock SCF", |c| if c.df_guess_path() { n("false") } else { Off("same scope as df_guess") }),
        key!("scf", "df_increments_aux", "aux basis of df_increments", |_| Off("only with df_increments = true (default def2-universal-jkfit)")),
        key!("scf", "check_stability", "diagnostic orbital-Hessian check of the converged SCF", |c| if c.double_hybrid() { Off("the double hybrid's own SCF") } else { n("false") }),
        key!("scf", "stability_descent", "follow a downhill mode at a saddle and re-converge", |c| if c.is(&["rhf", "uhf", "ksdft"]) && c.energy() { n("false") } else { Off("rhf, uhf and ksdft energies only") }),
        key!("scf", "symmetry", "point-group-blocked closed-shell RHF/RKS", |c| if c.is(&["rhf", "ksdft"]) && c.energy() && c.mult == 1 { n("false") } else { Off("closed-shell rhf/ksdft energies only") }),
        key!("scf", "irrep_nelec", "electrons per irrep", |_| Off("only with symmetry = true")),
        key!("scf", "ladder", "[[scf.ladder]] convergence-ladder rungs", |c| if c.is(&["rhf", "ksdft"]) { Unset("the built-in ladder; [[scf.ladder]] rungs replace it") } else { Off("rhf and ksdft only") }),
        // ---- [dft]
        key!("dft", "functional", "XC functional name (LDA, PBE, B3LYP, ...) or libxc name", |c| {
            if c.is(&["ksdft"]) { s("LDA") } else if c.scf_kind() { Unset("Hartree-Fock; a functional makes it Kohn-Sham") } else { Off("this kind sets its functional elsewhere") }
        }),
        key!("dft", "lambda", "wB97X-L-V double-hybrid lambda", |c| if c.is(&["wb97x-l-v"]) { n("0.6") } else { Off("wb97x-l-v only") }),
        key!("dft", "omega", "wB97X-L-V double-hybrid omega, Bohr^-1", |c| if c.is(&["wb97x-l-v"]) { n("0.1") } else { Off("wb97x-l-v only") }),
        key!("dft", "grid_prune", "none | nwchem | nwchem-radial | defgrid1..3", |c| if c.ks_grid() { s("nwchem-radial") } else { Off(NO_GRID) }),
        key!("dft", "grid_radial", "radial points per atom", |c| if c.ks_grid() { n("75") } else { Off(NO_GRID) }),
        key!("dft", "grid_angular", "Lebedev order", |c| if c.ks_grid() { n("110") } else { Off(NO_GRID) }),
        key!("dft", "grid_weight_cut", "drop grid points with |weight| below this; 0 = off", |c| if c.ks_grid() { n("1e-14") } else { Off(NO_GRID) }),
        key!("dft", "dispersion", "d3bj | d3bj(<functional>) | mbd | mbd(<functional>)", |c| {
            if c.is(&["ksdft", "b2plyp", "dsd-pbep86"]) && (c.energy() || c.is(&["ksdft"])) { Unset("no dispersion correction") } else { Off("Kohn-Sham SCF and the MP2 double hybrids only") }
        }),
        // ---- [mp2]
        key!("mp2", "auxbasis", "RI auxiliary basis of the correlation", |c| s(if c.is(&["tda", "tddft"]) { TDDFT_DEFAULT_AUX } else { DEFAULT_CORRELATION_AUX })),
        key!("mp2", "frozen_core", "frozen occupied orbitals: integer, \"auto\", true (= auto), false (= 0)", |_| n("0")),
        key!("mp2", "omega", "erf/erfc range-separation omega, Angstrom^-1", |c| if c.is(&["att-rimp2", "rs-mp2-rpa"]) { n("0.42") } else { Off("att-rimp2 (erfc) and rs-mp2-rpa only") }),
        key!("mp2", "linlccd_variant", "hh | drivers-only | full", |c| if c.is(&["linlccd"]) { s("hh") } else { Off("linlccd only") }),
        key!("mp2", "kappa", "kappa-regularized MP2, Hartree^-1", |c| if c.is(&["rimp2"]) && c.energy() { Unset("plain MP2") } else { Off("exact rimp2 energies only") }),
        key!("mp2", "c_os", "opposite-spin scale", |c| match c.kind.as_str() {
            "scs-mp2" => n("1.2"),
            "scs-mp2-2terfc" => n("1.27"),
            "laplace-sos-mp2" => n("1.3"),
            _ => Off("scs-mp2, scs-mp2-2terfc and laplace-sos-mp2 only"),
        }),
        key!("mp2", "c_ss", "same-spin scale", |c| match c.kind.as_str() {
            "scs-mp2" => n("0.3333333333333333"),
            "scs-mp2-2terfc" => n("4.05"),
            _ => Off("scs-mp2 and scs-mp2-2terfc only"),
        }),
        key!("mp2", "n_quad", "Laplace quadrature points: 3 | 5 | 7", |c| if c.is(&["laplace-mp2", "laplace-sos-mp2"]) { n("7") } else { Off("laplace-mp2 and laplace-sos-mp2 only") }),
        key!("mp2", "sos_formulation", "mo | ao | ao-sparse | local", |c| if c.is(&["laplace-sos-mp2"]) { s("mo") } else { Off("laplace-sos-mp2 only") }),
        key!("mp2", "domain_cutoff_bohr", "ao-sparse/local radius, Bohr", |_| Off("only with sos_formulation = \"ao-sparse\" or \"local\"")),
        key!("mp2", "formulation", "delta-lr | coupled-rings", |c| if c.is(&["rs-mp2-rpa"]) { s("delta-lr") } else { Off("rs-mp2-rpa only") }),
        key!("mp2", "attenuator", "erf | terf", |c| if c.is(&["rs-mp2-rpa"]) { s("erf") } else { Off("rs-mp2-rpa only") }),
        key!("mp2", "r0", "terf seam radius, Angstrom", |_| Off("rs-mp2-rpa with attenuator = \"terf\" only")),
        key!("mp2", "terf_omega", "terf sharpness, Angstrom^-1", |_| Off("rs-mp2-rpa with attenuator = \"terf\" only")),
        key!("mp2", "r0_sweep", "several terf r0 values on one SCF", |_| Off("rs-mp2-rpa with attenuator = \"terf\" only")),
        key!("mp2", "omega_scan", "several omega values on one SCF", |_| Off("rs-mp2-rpa only, instead of omega")),
        key!("mp2", "formulations", "several formulations in one job", |_| Off("rs-mp2-rpa only, instead of formulation")),
        key!("mp2", "r0_bonded", "bonded terfc r0, Angstrom", |c| if c.is(&["scs-mp2-2terfc"]) { n("0.75") } else { Off("scs-mp2-2terfc only") }),
        key!("mp2", "r0_nonbonded", "non-bonded terfc r0, Angstrom", |c| if c.is(&["scs-mp2-2terfc"]) { n("1.05") } else { Off("scs-mp2-2terfc only") }),
        key!("mp2", "mp2v_r0", "MP2-V r0, Angstrom (also the VV10 damping r0)", |c| if c.is(&["mp2-v"]) { n("1.0") } else { Off("mp2-v only") }),
        key!("mp2", "mp2v_b", "MP2-V VV10 b", |c| if c.is(&["mp2-v"]) { n("11.0") } else { Off("mp2-v only") }),
        key!("mp2", "mp2v_c", "MP2-V VV10 C", |c| if c.is(&["mp2-v"]) { n("0.0089") } else { Off("mp2-v only") }),
        key!("mp2", "mp2v_attenuator", "terfc | erfc", |c| if c.is(&["mp2-v"]) { s("terfc") } else { Off("mp2-v only") }),
        key!("mp2", "mp2v_omega", "MP2-V terfc sharpness, Angstrom^-1", |c| if c.is(&["mp2-v"]) { Unset("linked, omega = 1/(r0 sqrt 2)") } else { Off("mp2-v only") }),
        key!("mp2", "mp2v_vv10_damping", "terfc | none", |c| if c.is(&["mp2-v"]) { s("terfc") } else { Off("mp2-v only") }),
        key!("mp2", "mp2v_nlc_n_radial", "MP2-V VV10 radial points", |c| if c.is(&["mp2-v"]) { n("50") } else { Off("mp2-v only") }),
        key!("mp2", "mp2v_nlc_n_angular", "MP2-V VV10 angular points", |c| if c.is(&["mp2-v"]) { n("50") } else { Off("mp2-v only") }),
        key!("mp2", "oo_max_iter", "orbital-optimization iterations", |c| if c.is(&["oo-rimp2"]) { n("100") } else { Off("oo-rimp2 only") }),
        key!("mp2", "oo_grad_conv", "orbital-gradient norm threshold", |c| if c.is(&["oo-rimp2"]) { n("1e-4") } else { Off("oo-rimp2 only") }),
        key!("mp2", "oo_level_shift", "orbital-Hessian level shift, Hartree", |c| if c.is(&["oo-rimp2"]) { n("0.1") } else { Off("oo-rimp2 only") }),
        key!("mp2", "oo_diis_size", "orbital-rotation DIIS subspace", |c| if c.is(&["oo-rimp2"]) { n("6") } else { Off("oo-rimp2 only") }),
        key!("mp2", "att_operator", "erfc | terfc", |c| if c.is(&["att-rimp2"]) { s("erfc") } else { Off("att-rimp2 only") }),
        key!("mp2", "att_r0", "terfc r0, Angstrom", |_| Off("att-rimp2 with att_operator = \"terfc\" only")),
        key!("mp2", "att_omega", "terfc sharpness, Angstrom^-1", |_| Off("att-rimp2 with att_operator = \"terfc\" only")),
        // ---- [local]
        key!("local", "scheme", "none | amplitude-threshold", |_| s("none")),
        key!("local", "eps", "amplitude threshold (required with amplitude-threshold; no default)", |_| Off("only with scheme = \"amplitude-threshold\", where it is required")),
        key!("local", "eps_sweep", "several eps on one SCF (drpa)", |_| Off("only with scheme = \"amplitude-threshold\"")),
        key!("local", "reference", "also run the exact method", |_| Off("only with scheme = \"amplitude-threshold\"")),
        key!("local", "integral_direct", "integral-direct local MP2", |_| Off("only with scheme = \"amplitude-threshold\"")),
        key!("local", "aux_radius", "", |_| Off("only with integral_direct = true")),
        key!("local", "virt_radius", "", |_| Off("only with integral_direct = true")),
        key!("local", "ao_tail", "", |_| Off("only with integral_direct = true")),
        key!("local", "schwarz_skip", "", |_| Off("only with integral_direct = true")),
        key!("local", "batch_merge", "", |_| Off("only with integral_direct = true")),
        key!("local", "gate_cal", "", |_| Off("only with integral_direct = true")),
        key!("local", "virt_schwarz_kappa", "", |_| Off("only with integral_direct = true")),
        key!("local", "fock_screen", "", |_| Off("only with integral_direct = true")),
        key!("local", "virtual_space", "", |_| Off("only with scheme = \"amplitude-threshold\" (rimp2, att-rimp2)")),
        key!("local", "virtual_tau", "", |_| Off("only with virtual_space = \"local\"")),
        key!("local", "virtual_tau_loose", "", |_| Off("only with virtual_space = \"local\"")),
        key!("local", "virtual_solve_radius", "", |_| Off("only with virtual_space = \"local\"")),
        key!("local", "virtual_boys_tau", "", |_| Off("only with virtual_space = \"local\"")),
        // ---- [cc] / [eom]
        key!("cc", "ladder", "particle-particle ladder: auto | dense | ri", |_| s("auto")),
        key!("eom", "n_roots", "lowest singlet excitations", |_| n("5")),
        key!("eom", "conv_tol", "Davidson residual norm per root", |_| n("1e-6")),
        key!("eom", "max_iter", "Davidson iterations", |_| n("100")),
        key!("eom", "max_space", "subspace columns before a collapse: max(g + 10 n_roots, 50), g = max(2 n_roots, n_roots + 8)", |_| n("63")),
        // ---- [rpa]
        key!("rpa", "auxbasis", "RI auxiliary basis", |c| if c.is(&["rs-mp2-rpa"]) { Off(RS_RPA) } else { s(DEFAULT_CORRELATION_AUX) }),
        key!("rpa", "frozen_core", "as [mp2] frozen_core", |c| if c.is(&["rs-mp2-rpa"]) { Off(RS_RPA) } else { n("0") }),
        key!("rpa", "n_quad", "frequency quadrature points", |c| {
            if c.is(&["rs-mp2-rpa"]) { Off(RS_RPA) } else if c.is(&["gw"]) { n("40") } else if c.energy() { n("20") } else { n("16") }
        }),
        key!("rpa", "quadrature", "gauss-legendre | minimax | chebyshev-tan", |c| if c.is(&["rs-mp2-rpa"]) { Off(RS_RPA) } else { s("gauss-legendre") }),
        key!("rpa", "u0", "quadrature scale (ignored under minimax)", |c| if c.is(&["rs-mp2-rpa"]) { Off(RS_RPA) } else { n("0.5") }),
        key!("rpa", "trunc_thresh", "PDEP truncation; 0 = full rank", |c| n(if c.is(&["rs-mp2-rpa"]) { "0.0" } else { "1e-4" })),
        key!("rpa", "gradient", "fd | analytic", |_| Off("pdep-rpa with task = \"optimize\" or \"frequencies\" only")),
        key!("rpa", "eigensolver_conv_thresh", "PDEP eigensolver convergence (alias davidson_conv_thresh)", |c| if c.is(&["rs-mp2-rpa"]) { Off(RS_RPA) } else { n("1e-6") }),
        key!("rpa", "chi0_sparsity", "dense | boys[:thresh] | auto[:cutoff[:thresh]], optionally @radius", |c| if c.is(&["rs-mp2-rpa"]) { Off(RS_RPA) } else { s("dense") }),
        key!("rpa", "run_diagnostics", "print PDEP diagnostics", |c| if c.is(&["rs-mp2-rpa"]) { Off(RS_RPA) } else { n("false") }),
        key!("rpa", "export_eigpot_prefix", "write <prefix>_eigpot_NNN.cube", |c| if c.is(&["pdep-rpa"]) { Unset("no cube files") } else { Off("pdep-rpa only") }),
        key!("rpa", "export_eigpot_count", "", |_| Off("only with export_eigpot_prefix (default 10)")),
        key!("rpa", "cube_spacing", "", |_| Off("only with export_eigpot_prefix (default 0.2 Bohr)")),
        key!("rpa", "cube_margin", "", |_| Off("only with export_eigpot_prefix (default 4.0 Bohr)")),
        key!("rpa", "export_npz", "path of the NPZ property bundle", |c| if c.is(&["pdep-rpa"]) { Unset("no NPZ bundle") } else { Off("pdep-rpa only") }),
        key!("rpa", "compute_esp", "", |_| Off(NPZ)),
        key!("rpa", "compute_esp_surface", "", |_| Off(NPZ)),
        key!("rpa", "esp_surface_vdw_scale", "", |_| Off(NPZ)),
        key!("rpa", "esp_surface_n_angular", "", |_| Off(NPZ)),
        key!("rpa", "compute_polarizability", "", |_| Off(NPZ)),
        key!("rpa", "compute_alpha_atomic", "", |_| Off(NPZ)),
        key!("rpa", "compute_electric_field", "", |_| Off(NPZ)),
        key!("rpa", "compute_density_matrix", "", |_| Off(NPZ)),
        key!("rpa", "compute_dipole", "", |_| Off(NPZ)),
        key!("rpa", "compute_hirshfeld_charges", "", |_| Off(NPZ)),
        key!("rpa", "compute_lowdin_charges", "", |_| Off(NPZ)),
        key!("rpa", "compute_mulliken_charges", "", |_| Off(NPZ)),
        key!("rpa", "compute_chelpg_charges", "", |_| Off(NPZ)),
        key!("rpa", "compute_resp_charges", "", |_| Off(NPZ)),
        key!("rpa", "compute_c6", "", |_| Off(NPZ)),
        key!("rpa", "allow_partial_npz", "", |_| Off(NPZ)),
        key!("rpa", "c6_source", "", |_| Off("only with export_npz (default \"ts\")")),
        key!("rpa", "c6_partition", "", |_| Off("only with export_npz")),
        key!("rpa", "xc", "Kohn-Sham reference functional", |c| {
            if c.is(&["pdep-rpa", "gw"]) { Unset("Hartree-Fock reference") } else if c.is(&["rs-mp2-rpa"]) { Off(RS_RPA) } else { Off("bse-tda ignores it") }
        }),
        // ---- [gw]
        key!("gw", "method", "g0w0 | cohsex | evgw0 | evgw", |c| if c.is(&["gw"]) { s("g0w0") } else { Off("gw only") }),
        key!("gw", "qp_mos", "[lo, hi) MO indices of the quasiparticles", |c| if c.is(&["gw"]) { Unset("HOMO-2 to LUMO+2") } else { Off("gw only") }),
        key!("gw", "max_ev_iter", "", |_| Off("only with method = \"evgw0\" or \"evgw\" (default 20)")),
        key!("gw", "ev_conv_thresh", "", |_| Off("only with method = \"evgw0\" or \"evgw\" (default 1e-4)")),
        key!("gw", "pade_npts", "Pade nodes; 0 = 18", |c| if c.is(&["gw"]) { n("0") } else { Off("gw only") }),
        key!("gw", "qp_newton_damp", "quasiparticle Newton damping", |c| if c.is(&["gw"]) { n("1.0") } else { Off("gw only") }),
        key!("gw", "frozen_core", "frozen core of W and Sigma", |_| Unset("follows [rpa] frozen_core")),
        key!("gw", "scissor", "scissor shift, Hartree", |_| Off("tdhf-static-polarizability only")),
        key!("gw", "reference", "uhf | rohf", |_| Off("open-shell gw only")),
        // ---- [tddft]
        key!("tddft", "n_roots", "excited states", |_| n("3")),
        key!("tddft", "xc", "reference functional", |_| Unset("Hartree-Fock (CIS / TDHF)")),
        key!("tddft", "c_hf", "exact-exchange fraction", |_| Unset("the functional's own fraction")),
        key!("tddft", "solver", "dense | davidson | auto", |_| s("auto")),
        key!("tddft", "spin", "singlet | triplet", |_| s("singlet")),
        // ---- [optimize] / [frequencies]
        key!("optimize", "max_steps", "geometry steps", |_| n("100")),
        key!("optimize", "g_max_thresh", "max gradient, Hartree/Bohr", |_| n("4.5e-4")),
        key!("optimize", "g_rms_thresh", "rms gradient, Hartree/Bohr", |_| n("3.0e-4")),
        key!("optimize", "e_conv", "energy change, Hartree", |_| n("1e-6")),
        key!("optimize", "trust_radius", "initial step, Bohr", |_| n("0.1")),
        key!("optimize", "coordinates", "cartesian | internal", |_| s("cartesian")),
        key!("optimize", "initial_hessian", "lindh | simple", |_| s("lindh")),
        key!("optimize", "trust_update", "ratio | fixed", |_| s("ratio")),
        key!("optimize", "scf_warm_start", "start each SCF from the previous geometry's orbitals", |_| n("true")),
        key!("frequencies", "hessian", "auto | analytic | analytic-moving-grid | fd", |_| s("auto")),
        key!("frequencies", "delta", "central-difference step, Bohr", |c| if c.scf_kind() { Off("finite-difference Hessians only; auto is analytic here (default 5e-3)") } else { n("5e-3") }),
        key!("frequencies", "scf_warm_start", "warm-start the displaced SCFs", |c| if c.scf_kind() { Off("finite-difference Hessians only") } else { n("false") }),
        key!("frequencies", "temperature", "thermochemistry temperature, K", |_| n("298.15")),
        key!("frequencies", "pressure", "thermochemistry pressure, Pa", |_| n("101325.0")),
        key!("frequencies", "symmetry_number", "rotational symmetry number", |_| n("1")),
        key!("frequencies", "ir_intensities", "IR intensities from the dipole derivative", |c| n(if c.scf_kind() { "true" } else { "false" })),
        // ---- [cas] / [cas.ci] / [opmp3]
        key!("cas", "n_active", "active spatial orbitals (required)", |_| Off("required")),
        key!("cas", "n_elec_active", "[n_alpha, n_beta] active electrons (required)", |_| Off("required")),
        key!("cas", "n_core", "doubly occupied orbitals below the active window", |_| Unset("(N - n_alpha - n_beta) / 2")),
        key!("cas", "reference", "casscf | casci", |c| if c.is(&["nevpt2"]) { s("casscf") } else { Off("nevpt2 only") }),
        key!("cas", "canonicalize", "canonical core/virtual and natural active orbitals", |c| if c.is(&["casscf"]) { n("false") } else { Off("casscf only") }),
        key!("cas", "grad_tol", "orbital-gradient norm threshold", |_| n("1e-7")),
        key!("cas", "energy_tol", "energy-change threshold, Hartree", |_| n("1e-12")),
        key!("cas", "max_iter", "macro iterations", |_| n("300")),
        key!("cas", "max_step", "largest rotation-step element, rad", |_| n("0.2")),
        key!("cas", "optimizer", "ah | bfgs", |_| s("ah")),
        key!("cas", "integrals", "auto | direct | dense | ri", |_| s("auto")),
        key!("cas", "integral_thresh", "", |_| Off("only with integrals = \"direct\" (default 1e-12)")),
        key!("cas", "auxbasis", "", |_| Off("only with integrals = \"ri\", where it is required")),
        key!("cas", "stability", "off | check | follow", |_| s("off")),
        key!("cas", "ci", "the [cas.ci] table below", |_| Off("written as [cas.ci]")),
        key!("cas.ci", "conv_thresh", "CI residual threshold", |_| n("1e-10")),
        key!("cas.ci", "max_subspace", "Davidson subspace before a restart", |_| n("50")),
        key!("cas.ci", "max_iter", "Davidson iterations", |_| n("200")),
        key!("opmp3", "reference", "hf or a functional name", |_| s("hf")),
        key!("opmp3", "x_ss", "same-spin damping (the paper's value for the reference)", |_| n("0.0")),
        key!("opmp3", "integrals", "exact | ri", |_| s("exact")),
        key!("opmp3", "auxbasis", "", |_| Off("only with integrals = \"ri\", where it is required")),
        key!("opmp3", "frozen_core", "frozen occupied orbitals of each spin", |_| n("0")),
        // ---- [memory] / [parallel] / [gpu] / [precision] / [output]
        key!("memory", "budget_gb", "memory budget, GB", |_| Unset("FERRIC_MEM_BUDGET_GB, else 0.8 x available RAM; machine-dependent, so not written")),
        key!("memory", "three_index_budget_gb", "", |_| Off("deprecated alias of budget_gb")),
        key!("parallel", "blas_threads", "OpenBLAS threads of the serial sections", |_| Unset("FERRIC_BLAS_THREADS, else the rayon thread count; machine-dependent, so not written")),
        key!("gpu", "preset", "off | auto | on | mixed | auto-mixed", |_| s("off")),
        key!("gpu", "mode", "off | auto | on", |_| s("off")),
        key!("gpu", "device", "", |_| Off("only with mode = \"auto\" or \"on\" (default 0)")),
        key!("gpu", "memory_gb", "", |_| Off("only with mode = \"auto\" or \"on\"")),
        key!("gpu", "min_flops", "", |_| Off("only with mode = \"auto\" or \"on\"")),
        key!("gpu", "precision", "", |_| Off("deprecated alias of [precision] mode")),
        key!("gpu", "mixed_kernels", "", |_| Off("deprecated alias of [precision] mixed_kernels")),
        key!("precision", "mode", "f64 | mixed", |_| s("f64")),
        key!("precision", "mixed_kernels", "", |_| Off("only with mode = \"mixed\"")),
        key!("output", "json", "run-log path; true = <input-stem>.ferric.jsonl beside the input, false = off", |_| n("true")),
    ]
}

const RS_RPA: &str = "rs-mp2-rpa reads only trunc_thresh from [rpa]";

/// Sections that switch a feature on: their absence is the default, so no
/// full file writes them.
pub(super) const FEATURE_SECTIONS: &[&str] =
    &["cosmo", "pcm", "external_potential", "qmmm", "cell", "cdft"];

/// The sections a run reads, in file order.
fn sections(c: &Ctx) -> Vec<&'static str> {
    let own = match c.kind.as_str() {
        "casscf" | "nevpt2" => Some(crate::cas::READ_SECTIONS),
        "opmp3" => Some(crate::opmp3::READ_SECTIONS),
        _ => None,
    };
    let mut out = vec!["molecule", "basis", "method", "scf"];
    if let Some(own) = own {
        for sec in ["cas", "cas.ci", "opmp3"] {
            let top = sec.split('.').next().unwrap();
            if own.contains(&top) {
                out.push(sec);
            }
        }
        out.extend(["memory", "parallel", "output"]);
        return out;
    }
    let mp2_kinds = [
        "rimp2",
        "mp3",
        "oo-rimp2",
        "att-rimp2",
        "mp2-v",
        "scs-mp2",
        "scs-mp2-2terfc",
        "laplace-mp2",
        "laplace-sos-mp2",
        "rs-mp2-rpa",
        "ccsd",
        "ccd",
        "ccsd(t)",
        "eom-ccsd",
        "linlccd",
        "drpa",
        "wb97x-l-v",
        "b2plyp",
        "dsd-pbep86",
        "tda",
        "tddft",
    ];
    if c.scf_kind() || c.ks_grid() {
        out.push("dft");
    }
    if c.is(&mp2_kinds) {
        out.push("mp2");
    }
    if c.is(LOCAL_KINDS) && c.energy() {
        out.push("local");
    }
    if c.is(&["ccsd", "ccsd(t)", "eom-ccsd"]) {
        out.push("cc");
    }
    if c.is(&["eom-ccsd"]) {
        out.push("eom");
    }
    if c.is(&[
        "pdep-rpa",
        "gw",
        "bse-tda",
        "tdhf-static-polarizability",
        "rs-mp2-rpa",
    ]) {
        out.push("rpa");
    }
    if c.is(&["gw", "bse-tda", "tdhf-static-polarizability"]) {
        out.push("gw");
    }
    if c.is(&["tda", "tddft"]) {
        out.push("tddft");
    }
    match c.task.as_str() {
        "optimize" => out.push("optimize"),
        "frequencies" => out.push("frequencies"),
        _ => {}
    }
    out.extend(["memory", "parallel", "gpu", "precision", "output"]);
    out
}

/// The accepted keys of each table a full file may write (serde's own lists,
/// aliases removed: an alias is documented on the key it aliases).
fn serde_fields(section: &str) -> BTreeSet<&'static str> {
    use super::doc_tests::fields_of;
    let mut f = match section {
        "molecule" => fields_of::<MoleculeCfg>(),
        "basis" => fields_of::<BasisCfg>(),
        "method" => fields_of::<MethodCfg>(),
        "scf" => fields_of::<ScfCfg>(),
        "dft" => fields_of::<DftCfg>(),
        "mp2" => fields_of::<Mp2Cfg>(),
        "cc" => fields_of::<CcCfg>(),
        "eom" => fields_of::<EomCfg>(),
        "cas" => fields_of::<CasCfg>(),
        "cas.ci" => fields_of::<CasCiCfg>(),
        "opmp3" => fields_of::<crate::opmp3::Opmp3Cfg>(),
        "local" => fields_of::<LocalCfg>(),
        "rpa" => fields_of::<RpaCfg>(),
        "gw" => fields_of::<GwCfg>(),
        "tddft" => fields_of::<TddftCfg>(),
        "optimize" => fields_of::<OptimizeCfg>(),
        "frequencies" => fields_of::<FrequenciesCfg>(),
        "memory" => fields_of::<MemoryCfg>(),
        "parallel" => fields_of::<ParallelCfg>(),
        "gpu" => fields_of::<GpuCfg>(),
        "precision" => fields_of::<PrecisionCfg>(),
        "output" => fields_of::<OutputCfg>(),
        other => panic!("no struct mapped for [{other}]"),
    };
    for (table, alias) in super::doc_tests::SERDE_ALIASES {
        if *table == section {
            f.remove(alias);
        }
    }
    f
}

/// The marker lines around the minimal input in a file's header.
const MIN_BEGIN: &str = "# --- minimal input (begin) ---";
const MIN_END: &str = "# --- minimal input (end) ---";

/// A TOML value as one line.
fn inline(v: &toml::Value) -> String {
    match v {
        toml::Value::Table(t) => {
            let parts: Vec<String> = t
                .iter()
                .map(|(k, v)| format!("{k} = {}", inline(v)))
                .collect();
            format!("{{ {} }}", parts.join(", "))
        }
        other => other.to_string(),
    }
}

/// The value `minimal` writes for `section.key`, if any.
fn minimal_value(min: &toml::Table, section: &str, key: &str) -> Option<String> {
    let mut t = min;
    for part in section.split('.') {
        t = t.get(part)?.as_table()?;
    }
    t.get(key).map(inline)
}

/// A `# key: ...` comment line.
fn note(k: &Key, what: &str, why: &str) -> String {
    if k.doc.is_empty() {
        format!("# {}: {what}: {why}", k.key)
    } else {
        format!("# {}: {}; {what}: {why}", k.key, k.doc)
    }
}

/// The full file generated from one target.
fn generate(t: &Target) -> String {
    let min: toml::Table = toml::from_str(&t.minimal).expect("minimal input parses");
    let cfg: Config = toml::from_str(&t.minimal).expect("minimal input is a Config");
    let ctx = Ctx {
        kind: cfg.method.kind.clone(),
        task: cfg.method.task.clone(),
        mult: cfg.molecule.multiplicity,
        basis: cfg.basis.name.clone().unwrap_or_default(),
    };
    let keys = keys();
    let mut out = String::new();
    out.push_str(&format!(
        "# Fully specified input: method.kind = \"{}\", task = \"{}\".\n\
         #\n\
         # Every key of every section this run reads is written at the value the\n\
         # CLI uses when the key is omitted, so this file and the minimal input\n\
         # below describe the same run. A key with no value to write (its default\n\
         # is \"unset\", or it is read only under another key's value or by another\n\
         # kind) is listed in a `# key: ...` comment instead. Not written at all:\n\
         # [{}],\n\
         # each of which turns a feature on when present.\n\
         #\n\
         # Generated by crates/ferric-cli/src/full_examples.rs; regenerate with\n\
         #   FERRIC_WRITE_FULL_EXAMPLES=1 cargo test -p ferric-cli --release full_example\n\
         #\n{MIN_BEGIN}\n",
        ctx.kind,
        ctx.task,
        FEATURE_SECTIONS.join("], [")
    ));
    for l in t.minimal.lines() {
        out.push_str(&format!("#   {l}\n"));
    }
    out.push_str(&format!("{MIN_END}\n"));
    let own_scf: Option<&[&str]> = match ctx.kind.as_str() {
        "casscf" | "nevpt2" => Some(crate::cas::READ_SCF_KEYS),
        "opmp3" => Some(crate::opmp3::READ_SCF_KEYS),
        _ => None,
    };
    for sec in sections(&ctx) {
        out.push_str(&format!("\n[{sec}]\n"));
        let mut active = Vec::new();
        let mut notes = Vec::new();
        for k in keys.iter().filter(|k| k.section == sec) {
            let from_min = minimal_value(&min, sec, k.key);
            let line = match (&from_min, own_scf) {
                (Some(v), _) => Set(v.clone()),
                (None, Some(read)) if sec == "scf" && !read.contains(&k.key) => Off(OWN_START),
                _ => (k.line)(&ctx),
            };
            match line {
                Set(v) => {
                    let mark = if from_min.is_some() {
                        " [minimal input]"
                    } else {
                        ""
                    };
                    active.push((format!("{} = {v}", k.key), format!("{}{mark}", k.doc)));
                }
                Unset(why) => notes.push(note(k, "unset by default", why)),
                Off(why) => notes.push(note(k, "not read here", why)),
            }
        }
        let width = active.iter().map(|(a, _)| a.len()).max().unwrap_or(0);
        for (assign, doc) in &active {
            out.push_str(&format!("{assign:width$}  # {doc}\n"));
        }
        for note in &notes {
            out.push_str(note);
            out.push('\n');
        }
    }
    out
}

fn full_dir() -> PathBuf {
    super::doc_tests::repo_root().join("examples/full")
}

fn file_name(t: &Target) -> String {
    format!("{}.toml", t.name)
}

#[test]
fn full_examples_are_generated() {
    let dir = full_dir();
    let write = std::env::var_os("FERRIC_WRITE_FULL_EXAMPLES").is_some();
    if write {
        std::fs::create_dir_all(&dir).unwrap();
    }
    let mut stale = Vec::new();
    let mut expected = BTreeSet::new();
    for t in targets() {
        let path = dir.join(file_name(&t));
        expected.insert(file_name(&t));
        let text = generate(&t);
        if write {
            std::fs::write(&path, &text).unwrap();
        } else if std::fs::read_to_string(&path).ok().as_deref() != Some(text.as_str()) {
            stale.push(path.display().to_string());
        }
    }
    let present: BTreeSet<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    let extra: Vec<&String> = present.difference(&expected).collect();
    assert!(
        stale.is_empty() && extra.is_empty(),
        "examples/full/ is out of date (stale: {stale:?}; not generated: {extra:?}); run \
         FERRIC_WRITE_FULL_EXAMPLES=1 cargo test -p ferric-cli --release full_example"
    );
}

/// The section of a file each line belongs to, with the `# key:` comment
/// names found in it.
fn comment_keys(text: &str) -> BTreeMap<String, BTreeSet<String>> {
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut sec = String::new();
    for line in text.lines() {
        if let Some(h) = line.strip_prefix('[') {
            sec = h.trim_end_matches(']').to_string();
            out.entry(sec.clone()).or_default();
        } else if let Some(rest) = line.strip_prefix("# ") {
            if sec.is_empty() {
                continue;
            }
            if let Some((name, _)) = rest.split_once(':') {
                if !name.is_empty()
                    && name
                        .chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
                {
                    out.entry(sec.clone()).or_default().insert(name.to_string());
                }
            }
        }
    }
    out
}

/// Check one file's text: every section its kind reads is there, and every
/// serde field of each of its sections is a key or a `# key:` comment.
fn completeness_problems(name: &str, text: &str) -> Vec<String> {
    let mut problems = Vec::new();
    let root: toml::Table = match toml::from_str(text) {
        Ok(t) => t,
        Err(e) => return vec![format!("{name}: does not parse: {e}")],
    };
    let cfg: Config = match toml::from_str(text) {
        Ok(c) => c,
        Err(e) => return vec![format!("{name}: is not a Config: {e}")],
    };
    let ctx = Ctx {
        kind: cfg.method.kind.clone(),
        task: cfg.method.task.clone(),
        mult: cfg.molecule.multiplicity,
        basis: cfg.basis.name.clone().unwrap_or_default(),
    };
    let comments = comment_keys(text);
    for sec in sections(&ctx) {
        let Some(noted) = comments.get(sec) else {
            problems.push(format!(
                "{name}: has no [{sec}] section, which this kind reads"
            ));
            continue;
        };
        let mut table = Some(&root);
        for part in sec.split('.') {
            table = table.and_then(|t| t.get(part)).and_then(|v| v.as_table());
        }
        let written: BTreeSet<&str> = table
            .map(|t| t.keys().map(String::as_str).collect())
            .unwrap_or_default();
        for field in serde_fields(sec) {
            if !written.contains(field) && !noted.contains(field) {
                problems.push(format!(
                    "{name}: [{sec}] `{field}` is neither written nor listed in a `# {field}:` comment"
                ));
            }
        }
    }
    for sec in root.keys() {
        if FEATURE_SECTIONS.contains(&sec.as_str()) {
            problems.push(format!("{name}: writes the feature section [{sec}]"));
        }
    }
    problems
}

#[test]
fn full_examples_name_every_key_of_their_sections() {
    let dir = full_dir();
    let mut problems = Vec::new();
    for t in targets() {
        let path = dir.join(file_name(&t));
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        problems.extend(completeness_problems(t.name, &text));
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// Every key the table names is a real field, so a renamed field cannot leave
/// a stale row that still "documents" it.
#[test]
fn full_example_table_names_only_real_fields() {
    let mut bad = Vec::new();
    for k in keys() {
        if !serde_fields(k.section).contains(k.key) {
            bad.push(format!("[{}] {}", k.section, k.key));
        }
    }
    assert!(bad.is_empty(), "table rows with no serde field: {bad:?}");
}

/// The completeness check fails when a key is dropped from a file: the
/// generated rhf file minus its `[scf] max_iter` line, and minus a comment
/// line, are each reported.
#[test]
fn full_example_completeness_catches_a_deleted_key() {
    let t = targets().into_iter().find(|t| t.name == "rhf").unwrap();
    let text = generate(&t);
    assert!(completeness_problems("rhf", &text).is_empty());
    let drop_line = |pred: &dyn Fn(&str) -> bool| -> String {
        text.lines()
            .filter(|l| !pred(l))
            .map(|l| format!("{l}\n"))
            .collect()
    };
    let no_max_iter = drop_line(&|l| l.starts_with("max_iter = "));
    let p = completeness_problems("rhf", &no_max_iter);
    assert!(p.iter().any(|m| m.contains("[scf] `max_iter`")), "{p:?}");
    let no_note = drop_line(&|l| l.starts_with("# cosx_grid:"));
    let p = completeness_problems("rhf", &no_note);
    assert!(p.iter().any(|m| m.contains("[scf] `cosx_grid`")), "{p:?}");
}

#[test]
fn full_examples_load() {
    let dir = full_dir();
    let root = super::doc_tests::repo_root();
    let mut problems = Vec::new();
    for t in targets() {
        let path = dir.join(file_name(&t));
        match load_config(path.to_str().unwrap()) {
            Ok(cfg) => {
                if !root.join(&cfg.molecule.xyz).is_file() {
                    problems.push(format!("{}: xyz {:?} not found", t.name, cfg.molecule.xyz));
                }
                if let Err(e) = cfg.validate_cli_wired_keys() {
                    problems.push(format!("{}: {e}", t.name));
                }
            }
            Err(e) => problems.push(format!("{}: {e}", t.name)),
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
