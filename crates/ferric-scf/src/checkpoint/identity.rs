//! What a checkpoint must match, and what happens when it does not.

use super::{Checkpoint, CheckpointError, FORMAT_VERSION};
use crate::result::{ScfExit, Spin};
use crate::rhf::RhfConfig;
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::ecp::{EcpDef, EcpShell, EcpTerm};
use ferric_core::mol::{Atom, Molecule};
use ferric_integrals::operator::Operator;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

/// How a mismatch in a field is treated; see the table in the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldClass {
    /// Atoms, basis data, ECP, charge, multiplicity, spin kind: the matrices mean
    /// nothing against a different one of these.
    Structure,
    /// Nuclear positions. Compared bitwise.
    Coordinates,
    /// What the electrons feel besides the nuclei.
    Environment,
    /// Which state and functional the SCF solves for.
    Method,
    /// How J, K and the integrals are built: same state, different numbers.
    Route,
}

/// One compared field. `value` is canonical text, compared exactly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdentityField {
    pub name: String,
    pub class: FieldClass,
    pub value: String,
}

/// Which build produced a file. Passed in by the caller because the
/// `ferric-build-info` crate is deliberately not a dependency of this one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildStamp {
    pub version: String,
    pub commit: Option<String>,
    pub dirty: Option<bool>,
}

impl BuildStamp {
    /// Provably the same source: the same known commit, clean on both sides. Two
    /// dirty trees on one commit can hold different code, so they do not qualify.
    fn same_build(&self, other: &Self) -> bool {
        self.commit.is_some()
            && self.commit == other.commit
            && self.dirty == Some(false)
            && other.dirty == Some(false)
    }
}

impl std::fmt::Display for BuildStamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let commit = self.commit.as_deref().unwrap_or("unknown commit");
        let state = match self.dirty {
            Some(false) => "clean",
            Some(true) => "dirty",
            None => "unknown state",
        };
        write!(f, "{} {commit} {state}", self.version)
    }
}

/// The calculation a file belongs to, or the one about to run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScfIdentity {
    pub fields: Vec<IdentityField>,
    /// Bohr, atom order. Not in `fields` because a guess reports the max shift.
    pub coords_bohr: Vec<[f64; 3]>,
    pub energy_conv: f64,
    pub density_conv: f64,
    pub nao: usize,
    /// For messages only: the data decides, so an alias of the same basis passes.
    pub basis_name: String,
}

fn field(out: &mut Vec<IdentityField>, name: &str, class: FieldClass, value: String) {
    out.push(IdentityField {
        name: name.to_string(),
        class,
        value,
    });
}

/// 64-bit FNV-1a state: dependency-free and stable across Rust releases.
pub(super) struct Fnv(pub(super) u64);

impl Fnv {
    pub(super) fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    pub(super) fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

fn hash_text(text: &str) -> u64 {
    let mut h = Fnv::new();
    h.update(text.as_bytes());
    h.0
}

fn bits(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("{:016x}", v.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

/// One element's shells, in load order, which is AO order.
fn shells_text(shells: &[Shell]) -> String {
    let mut s = String::new();
    for shell in shells {
        let Shell {
            l,
            pure,
            exponents,
            coefficients,
        } = shell;
        let _ = writeln!(
            s,
            "l={l} pure={pure} exp={} coef={}",
            bits(exponents),
            bits(coefficients)
        );
    }
    s
}

fn ecp_text(ecp: &EcpDef) -> String {
    let EcpDef { n_core, shells } = ecp;
    let mut s = format!("n_core={n_core}\n");
    for shell in shells {
        let EcpShell {
            angular_momentum,
            terms,
        } = shell;
        let _ = write!(s, "l={angular_momentum}");
        for term in terms {
            let EcpTerm { coef, r_exp, gexp } = term;
            let _ = write!(s, " ({},{r_exp},{})", bits(&[*coef]), bits(&[*gexp]));
        }
        s.push('\n');
    }
    s
}

fn shell_nao(shell: &Shell) -> usize {
    let l = shell.l as usize;
    if shell.pure {
        2 * l + 1
    } else {
        (l + 1) * (l + 2) / 2
    }
}

/// Elements of the molecule only, ascending: what else the basis covers cannot
/// change this calculation, and neither can its map's iteration order.
fn basis_fields(
    mol: &Molecule,
    basis: &BasisSet,
    out: &mut Vec<IdentityField>,
) -> Result<usize, CheckpointError> {
    let BasisSet {
        name,
        shells: _,
        ecps: _,
    } = basis;
    let elements: BTreeSet<i32> = mol.atoms.iter().map(|a| a.z).collect();
    let mut nao_of = BTreeMap::new();
    for &z in &elements {
        let shells = basis.for_element(z).ok_or_else(|| {
            CheckpointError::Format(format!("basis {name:?} has no shells for Z={z}"))
        })?;
        nao_of.insert(z, shells.iter().map(shell_nao).sum::<usize>());
        let text = shells_text(shells);
        let value = format!("shells={} fnv={:016x}", shells.len(), hash_text(&text));
        field(out, &format!("basis[Z={z}]"), FieldClass::Structure, value);
        if let Some(ecp) = basis.ecp_for_element(z) {
            let value = format!("{:016x}", hash_text(&ecp_text(ecp)));
            field(out, &format!("ecp[Z={z}]"), FieldClass::Structure, value);
        }
    }
    Ok(mol.atoms.iter().map(|a| nao_of[&a.z]).sum())
}

fn molecule_fields(mol: &Molecule, out: &mut Vec<IdentityField>) -> Vec<[f64; 3]> {
    let Molecule {
        atoms,
        charge,
        multiplicity,
    } = mol;
    let mut atom_text = String::new();
    let mut coords = Vec::with_capacity(atoms.len());
    for atom in atoms {
        // `symbol` is a label for `z`, which decides.
        let Atom {
            symbol: _,
            z,
            x,
            y,
            zpos,
            ghost,
            n_core_ecp,
        } = atom;
        let _ = write!(atom_text, "{z}:{ghost}:{n_core_ecp};");
        coords.push([*x, *y, *zpos]);
    }
    field(out, "atoms", FieldClass::Structure, atom_text);
    field(out, "charge", FieldClass::Structure, charge.to_string());
    field(
        out,
        "multiplicity",
        FieldClass::Structure,
        multiplicity.to_string(),
    );
    coords
}

/// Classify every `RhfConfig` field. No `..`: a new field does not compile until
/// it appears here. Values use `{:?}`, which is deterministic for these types
/// because none holds a map (the two-process test populates all of them), and
/// prints an f64 so that it round-trips, sign of zero included.
///
/// Returns `(energy_conv, density_conv)`, which are compared by inequality.
fn config_fields(config: &RhfConfig, out: &mut Vec<IdentityField>) -> (f64, f64) {
    let RhfConfig {
        energy_conv,
        density_conv,
        // What the electrons feel.
        external_potential,
        cosmo,
        pcm,
        polarizable,
        // Which state and functional.
        xc,
        xc_omega,
        dft_grid,
        nlc_grid,
        smearing_sigma,
        symmetry,
        irrep_nelec,
        fractional_occ,
        scf_stability_descent,
        rohf_occupation_guard,
        mom_after_iter,
        constraints,
        cdft_lambda_tol,
        cdft_lambda_init,
        cdft_stability_descent,
        // How J, K and the integrals are built.
        k_builder,
        cosx,
        df_j_aux,
        df_k_aux,
        integral_thresh,
        screening,
        // Unchecked. These steer the iteration or the resources, not what the SCF
        // solves for, so a file from a different setting is still the same answer.
        max_iter: _,
        diis_size: _,
        diis_flavor: _,
        diis_switch_thresh: _,
        level_shift: _,
        newton_trigger: _,
        aurora: _,
        ah_trigger: _,
        trah_trigger: _,
        trah: _,
        stall_window: _,
        divergence_tol: _,
        cdft_max_outer: _,
        three_index_budget_bytes: _,
        verbose: _,
        // Asks for a post-SCF diagnostic, whose absence on a restored result reads
        // "not checked".
        check_stability: _,
        // Starting points: a file replaces them.
        init_guess_density: _,
        guess_basis: _,
        use_sad_guess: _,
    } = config;
    // An empty potential is the same physics as none: every consumer treats it as
    // a no-op, and QM/MM with no MM atoms produces either.
    let external = external_potential.as_ref().filter(|p| !p.is_empty());
    let environment = [
        ("external_potential", format!("{external:?}")),
        ("cosmo", format!("{cosmo:?}")),
        ("pcm", format!("{pcm:?}")),
        ("polarizable", format!("{polarizable:?}")),
    ];
    let method = [
        ("xc", format!("{xc:?}")),
        ("xc_omega", format!("{xc_omega:?}")),
        ("dft_grid", format!("{dft_grid:?}")),
        ("nlc_grid", format!("{nlc_grid:?}")),
        ("smearing_sigma", format!("{smearing_sigma:?}")),
        ("symmetry", format!("{symmetry:?}")),
        ("irrep_nelec", format!("{irrep_nelec:?}")),
        ("fractional_occ", format!("{fractional_occ:?}")),
        (
            "scf_stability_descent",
            format!("{scf_stability_descent:?}"),
        ),
        (
            "rohf_occupation_guard",
            format!("{rohf_occupation_guard:?}"),
        ),
        ("mom_after_iter", format!("{mom_after_iter:?}")),
        ("constraints", format!("{constraints:?}")),
        ("cdft_lambda_tol", format!("{cdft_lambda_tol:?}")),
        ("cdft_lambda_init", format!("{cdft_lambda_init:?}")),
        (
            "cdft_stability_descent",
            format!("{cdft_stability_descent:?}"),
        ),
    ];
    let route = [
        ("k_builder", format!("{k_builder:?}")),
        ("cosx", format!("{cosx:?}")),
        ("df_j_aux", format!("{df_j_aux:?}")),
        ("df_k_aux", format!("{df_k_aux:?}")),
        ("integral_thresh", format!("{integral_thresh:?}")),
        ("screening", format!("{screening:?}")),
    ];
    for (class, group) in [
        (FieldClass::Environment, &environment[..]),
        (FieldClass::Method, &method[..]),
        (FieldClass::Route, &route[..]),
    ] {
        for (name, value) in group {
            field(out, name, class, value.clone());
        }
    }
    (*energy_conv, *density_conv)
}

impl ScfIdentity {
    /// Identity of an SCF on `mol`/`basis` with this spin treatment, two-electron
    /// operator and configuration, under the process's installed precision
    /// settings.
    pub fn new(
        mol: &Molecule,
        basis: &BasisSet,
        spin: Spin,
        op: &Operator,
        config: &RhfConfig,
    ) -> Result<Self, CheckpointError> {
        let settings = ferric_core::gpu::settings();
        let precision = format!("{:?} {:?}", settings.precision, settings.mixed_kernels);
        Self::with_precision(mol, basis, spin, op, config, precision)
    }

    pub(super) fn with_precision(
        mol: &Molecule,
        basis: &BasisSet,
        spin: Spin,
        op: &Operator,
        config: &RhfConfig,
        precision: String,
    ) -> Result<Self, CheckpointError> {
        let mut fields = Vec::new();
        let coords_bohr = molecule_fields(mol, &mut fields);
        field(
            &mut fields,
            "spin",
            FieldClass::Structure,
            format!("{spin:?}"),
        );
        let nao = basis_fields(mol, basis, &mut fields)?;
        field(&mut fields, "nao", FieldClass::Structure, nao.to_string());
        field(
            &mut fields,
            "operator",
            FieldClass::Method,
            format!("{op:?}"),
        );
        field(&mut fields, "precision", FieldClass::Route, precision);
        let (energy_conv, density_conv) = config_fields(config, &mut fields);
        Ok(Self {
            fields,
            coords_bohr,
            energy_conv,
            density_conv,
            nao,
            basis_name: basis.name.clone(),
        })
    }
}

/// Which use a file is being read for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadMode {
    /// Seed an SCF that still iterates to self-consistency.
    Guess,
    /// Replace the SCF. `allow_other_build` is the explicit override; it exists
    /// only here, so it cannot be passed under a guess.
    Skip { allow_other_build: bool },
}

/// One field that differs. `file`/`current` are `None` if the field is absent
/// from that side, which is how a field added by a newer build appears.
#[derive(Debug, Clone, PartialEq)]
pub struct Mismatch {
    pub name: String,
    pub class: FieldClass,
    pub file: Option<String>,
    pub current: Option<String>,
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let side = |v: &Option<String>| v.clone().unwrap_or_else(|| "(absent)".to_string());
        write!(
            f,
            "{}: file {} vs current {}",
            self.name,
            side(&self.file),
            side(&self.current)
        )
    }
}

pub(super) fn describe(mismatches: &[Mismatch]) -> String {
    let lines: Vec<String> = mismatches.iter().map(Mismatch::to_string).collect();
    lines.join("; ")
}

/// What a caller must print and record after an accepted read.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadVerdict {
    /// To print as warnings (guess only; under skip these are errors).
    pub warnings: Vec<Mismatch>,
    /// To log (guess only).
    pub logged: Vec<Mismatch>,
    /// Largest per-axis coordinate difference in Bohr, when positions differ.
    pub max_coord_shift_bohr: Option<f64>,
    /// The file's exit state, to echo. `None` for a mid-run file.
    pub file_exit: Option<ScfExit>,
    /// The file's build, always.
    pub file_build: BuildStamp,
    /// Set when a skip crossed builds on the explicit override. Goes in the result.
    pub build_override_used: bool,
    pub file_hash: Option<u64>,
    pub format_version: u64,
}

fn compare_fields(file: &ScfIdentity, current: &ScfIdentity) -> Vec<Mismatch> {
    let file_by_name: BTreeMap<&str, &IdentityField> =
        file.fields.iter().map(|f| (f.name.as_str(), f)).collect();
    let current_by_name: BTreeMap<&str, &IdentityField> = current
        .fields
        .iter()
        .map(|f| (f.name.as_str(), f))
        .collect();
    let names: BTreeSet<&str> = file_by_name
        .keys()
        .chain(current_by_name.keys())
        .copied()
        .collect();
    names
        .into_iter()
        .filter_map(|name| {
            let (a, b) = (file_by_name.get(name), current_by_name.get(name));
            let class = b.or(a).map(|f| f.class)?;
            let differs = a.map(|f| &f.value) != b.map(|f| &f.value);
            differs.then(|| Mismatch {
                name: name.to_string(),
                class,
                file: a.map(|f| f.value.clone()),
                current: b.map(|f| f.value.clone()),
            })
        })
        .collect()
}

/// `(mismatch if any, max shift)`. A different atom count is a `Structure`
/// mismatch already, so only same-length lists are compared here.
fn compare_coords(file: &ScfIdentity, current: &ScfIdentity) -> (Option<Mismatch>, Option<f64>) {
    let (a, b) = (&file.coords_bohr, &current.coords_bohr);
    if a.len() != b.len() {
        return (None, None);
    }
    let same = a
        .iter()
        .flatten()
        .zip(b.iter().flatten())
        .all(|(x, y)| x.to_bits() == y.to_bits());
    if same {
        return (None, None);
    }
    let shift = a
        .iter()
        .flatten()
        .zip(b.iter().flatten())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    let mismatch = Mismatch {
        name: "coordinates".to_string(),
        class: FieldClass::Coordinates,
        file: Some(format!("max shift {shift:e} Bohr")),
        current: None,
    };
    (Some(mismatch), Some(shift))
}

enum Disposition {
    Error,
    Warn,
    Log,
}

/// The table in the module docs. No wildcard, so a new class must be placed.
fn disposition(class: FieldClass, mode: ReadMode) -> Disposition {
    match (mode, class) {
        (ReadMode::Skip { .. }, _) | (ReadMode::Guess, FieldClass::Structure) => Disposition::Error,
        (
            ReadMode::Guess,
            FieldClass::Coordinates | FieldClass::Environment | FieldClass::Method,
        ) => Disposition::Warn,
        (ReadMode::Guess, FieldClass::Route) => Disposition::Log,
    }
}

/// The exit gate of a skip. No wildcard, so a new exit must be classified.
/// `Plateau` passes because a fresh run accepts it too; the caller echoes it.
fn exit_replaces_scf(exit: ScfExit) -> bool {
    match exit {
        ScfExit::Converged | ScfExit::Plateau => true,
        ScfExit::Stalled | ScfExit::Diverged | ScfExit::MaxIter | ScfExit::NotCertified => false,
    }
}

/// The non-identity gates of a skip. Returns whether the build override was used.
fn check_skip_gates(
    ck: &Checkpoint,
    current: &ScfIdentity,
    current_build: &BuildStamp,
    allow_other_build: bool,
) -> Result<bool, CheckpointError> {
    let state = ck.final_state.as_ref().ok_or(CheckpointError::MidRun)?;
    if !exit_replaces_scf(state.exit) {
        return Err(CheckpointError::ExitRefused {
            exit: format!("{:?}", state.exit),
        });
    }
    let blocking: Vec<String> = state
        .not_stored
        .iter()
        .filter(|name| matches!(name.as_str(), "induced_dipoles" | "cosx_final"))
        .cloned()
        .collect();
    if !blocking.is_empty() {
        return Err(CheckpointError::NotRestorable(blocking));
    }
    for (name, file, requested) in [
        ("energy_conv", ck.identity.energy_conv, current.energy_conv),
        (
            "density_conv",
            ck.identity.density_conv,
            current.density_conv,
        ),
    ] {
        if file > requested {
            return Err(CheckpointError::Looser {
                field: name.to_string(),
                file,
                requested,
            });
        }
    }
    let same = ck.build.same_build(current_build);
    if !same && !allow_other_build {
        return Err(CheckpointError::OtherBuild {
            file: ck.build.to_string(),
            current: current_build.to_string(),
        });
    }
    Ok(!same)
}

pub(super) fn check(
    ck: &Checkpoint,
    current: &ScfIdentity,
    current_build: &BuildStamp,
    mode: ReadMode,
) -> Result<ReadVerdict, CheckpointError> {
    let mut mismatches = compare_fields(&ck.identity, current);
    let (coord_mismatch, max_coord_shift_bohr) = compare_coords(&ck.identity, current);
    mismatches.extend(coord_mismatch);
    let (mut errors, mut warnings, mut logged) = (Vec::new(), Vec::new(), Vec::new());
    for m in mismatches {
        match disposition(m.class, mode) {
            Disposition::Error => errors.push(m),
            Disposition::Warn => warnings.push(m),
            Disposition::Log => logged.push(m),
        }
    }
    if !errors.is_empty() {
        return Err(CheckpointError::Identity(errors));
    }
    let build_override_used = match mode {
        ReadMode::Skip { allow_other_build } => {
            check_skip_gates(ck, current, current_build, allow_other_build)?
        }
        ReadMode::Guess => {
            if !ck.build.same_build(current_build) {
                logged.push(Mismatch {
                    name: "build".to_string(),
                    class: FieldClass::Route,
                    file: Some(ck.build.to_string()),
                    current: Some(current_build.to_string()),
                });
            }
            false
        }
    };
    Ok(ReadVerdict {
        warnings,
        logged,
        max_coord_shift_bohr,
        file_exit: ck.final_state.as_ref().map(|s| s.exit),
        file_build: ck.build.clone(),
        build_override_used,
        file_hash: ck.file_hash,
        format_version: FORMAT_VERSION,
    })
}
