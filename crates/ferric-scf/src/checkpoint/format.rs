//! The NPZ encoding, its checksum, and the atomic write.

use super::identity::{BuildStamp, Fnv, ScfIdentity};
use super::{Checkpoint, CheckpointError, FinalState, OrbitalState, FORMAT_VERSION};
use crate::result::{ConvergenceReport, DfJkRoute, DiisFallback, GradientVetoes, ScfExit, Spin};
use ferric_integrals::operator::{Operator, OperatorKind};
use ndarray::{arr1, Array1, Array2, ArrayD, Ix1, Ix2, IxDyn};
use ndarray_npy::{NpzReader, NpzWriter};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Cursor, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const VERSION_MEMBER: &str = "format_version";
const CHECKSUM_MEMBER: &str = "checksum";
const META_MEMBER: &str = "meta";
const HASH_DOMAIN: &[u8] = b"ferric-scf-checkpoint\0";

/// Points in a write, so the kill tests can stop a real writer there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WriteStage {
    /// Part of the temp file is on disk.
    PartWritten,
    /// The temp file is complete and not yet durable.
    TempWritten,
    /// The temp file is durable; the destination is untouched.
    BeforeRename,
    /// The rename happened; the directory is not yet fsynced.
    Renamed,
}

/// Stored as bit patterns because a `ConvergenceReport` may
/// legitimately hold an infinity (`dp_rms` before a second density exists) and
/// JSON numbers cannot.
mod float_bits {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&format!("{:016x}", v.to_bits()))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        let text = String::deserialize(d)?;
        u64::from_str_radix(&text, 16)
            .map(f64::from_bits)
            .map_err(serde::de::Error::custom)
    }
}

mod opt_float_bits {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &Option<f64>, s: S) -> Result<S::Ok, S::Error> {
        v.map(|x| format!("{:016x}", x.to_bits())).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
        Option::<String>::deserialize(d)?
            .map(|t| u64::from_str_radix(&t, 16).map(f64::from_bits))
            .transpose()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VetoesMeta {
    count: usize,
    last_iteration: usize,
    #[serde(with = "float_bits")]
    last_orbital_gradient: f64,
    #[serde(with = "float_bits")]
    bound: f64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FallbackMeta {
    iteration: usize,
    #[serde(with = "float_bits")]
    err_max: f64,
    #[serde(with = "float_bits")]
    bound: f64,
    flavor: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConvergenceMeta {
    iteration: usize,
    #[serde(with = "float_bits")]
    de: f64,
    #[serde(with = "float_bits")]
    dp_rms: f64,
    #[serde(with = "float_bits")]
    dp_max: f64,
    #[serde(with = "float_bits")]
    err_max: f64,
    #[serde(with = "float_bits")]
    energy_conv: f64,
    #[serde(with = "float_bits")]
    density_conv: f64,
    #[serde(with = "opt_float_bits")]
    de_noise_floor: Option<f64>,
    diis_fallback: Option<FallbackMeta>,
    gradient_vetoes: Option<VetoesMeta>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OperatorMeta {
    kind: String,
    omega: f64,
    distance: f64,
    composite: bool,
    components: Vec<(f64, String, f64)>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteMeta {
    j_aux: Option<String>,
    k_aux: Option<String>,
    rsh_k: Option<(String, f64)>,
    op: OperatorMeta,
    budget_bytes: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FinalMeta {
    energy: f64,
    one_electron_energy: f64,
    nuclear_repulsion: f64,
    df_jk: Option<RouteMeta>,
    exit: String,
    converged: bool,
    iterations: usize,
    computed_quartets: usize,
    convergence: Option<ConvergenceMeta>,
    precision_summary: String,
    precision_detail: String,
    not_stored: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Meta {
    build: BuildStamp,
    identity: ScfIdentity,
    spin: Spin,
    iteration: usize,
    final_state: Option<FinalMeta>,
}

fn kind_name(kind: OperatorKind) -> &'static str {
    match kind {
        OperatorKind::Coulomb => "Coulomb",
        OperatorKind::ErfCoulomb => "ErfCoulomb",
        OperatorKind::ErfcCoulomb => "ErfcCoulomb",
        OperatorKind::Terfc => "Terfc",
        OperatorKind::Terf => "Terf",
        OperatorKind::Yukawa => "Yukawa",
        OperatorKind::SlaterGeminal => "SlaterGeminal",
        OperatorKind::Cgtg => "Cgtg",
        OperatorKind::CgtgCoulomb => "CgtgCoulomb",
        OperatorKind::Delcgtg2 => "Delcgtg2",
    }
}

fn kind_from_name(name: &str) -> Result<OperatorKind, CheckpointError> {
    Ok(match name {
        "Coulomb" => OperatorKind::Coulomb,
        "ErfCoulomb" => OperatorKind::ErfCoulomb,
        "ErfcCoulomb" => OperatorKind::ErfcCoulomb,
        "Terfc" => OperatorKind::Terfc,
        "Terf" => OperatorKind::Terf,
        "Yukawa" => OperatorKind::Yukawa,
        "SlaterGeminal" => OperatorKind::SlaterGeminal,
        "Cgtg" => OperatorKind::Cgtg,
        "CgtgCoulomb" => OperatorKind::CgtgCoulomb,
        "Delcgtg2" => OperatorKind::Delcgtg2,
        other => {
            return Err(CheckpointError::Format(format!(
                "unknown operator kind {other:?}"
            )))
        }
    })
}

fn exit_name(exit: ScfExit) -> &'static str {
    match exit {
        ScfExit::Converged => "Converged",
        ScfExit::Plateau => "Plateau",
        ScfExit::Stalled => "Stalled",
        ScfExit::Diverged => "Diverged",
        ScfExit::MaxIter => "MaxIter",
        ScfExit::NotCertified => "NotCertified",
    }
}

fn exit_from_name(name: &str) -> Result<ScfExit, CheckpointError> {
    Ok(match name {
        "Converged" => ScfExit::Converged,
        "Plateau" => ScfExit::Plateau,
        "Stalled" => ScfExit::Stalled,
        "Diverged" => ScfExit::Diverged,
        "MaxIter" => ScfExit::MaxIter,
        "NotCertified" => ScfExit::NotCertified,
        other => return Err(CheckpointError::Format(format!("unknown exit {other:?}"))),
    })
}

fn operator_to_meta(op: &Operator) -> OperatorMeta {
    OperatorMeta {
        kind: kind_name(op.kind).to_string(),
        omega: op.omega,
        distance: op.distance,
        composite: op.is_composite(),
        components: op
            .components()
            .map(|(c, k, w)| (c, kind_name(k).to_string(), w))
            .collect(),
    }
}

fn operator_from_meta(meta: OperatorMeta) -> Result<Operator, CheckpointError> {
    let kind = kind_from_name(&meta.kind)?;
    let mut op = if meta.composite {
        let components = meta
            .components
            .iter()
            .map(|(c, k, w)| Ok((*c, kind_from_name(k)?, *w)))
            .collect::<Result<Vec<_>, CheckpointError>>()?;
        Operator::composite(&components)
    } else {
        Operator::primitive(kind, meta.omega, meta.distance)
    };
    op.kind = kind;
    op.omega = meta.omega;
    op.distance = meta.distance;
    Ok(op)
}

fn route_to_meta(route: &DfJkRoute) -> RouteMeta {
    let DfJkRoute {
        j_aux,
        k_aux,
        rsh_k,
        op,
        budget_bytes,
    } = route;
    RouteMeta {
        j_aux: j_aux.clone(),
        k_aux: k_aux.clone(),
        rsh_k: rsh_k.clone(),
        op: operator_to_meta(op),
        budget_bytes: *budget_bytes,
    }
}

fn route_from_meta(meta: RouteMeta) -> Result<DfJkRoute, CheckpointError> {
    Ok(DfJkRoute {
        j_aux: meta.j_aux,
        k_aux: meta.k_aux,
        rsh_k: meta.rsh_k,
        op: operator_from_meta(meta.op)?,
        budget_bytes: meta.budget_bytes,
    })
}

fn convergence_to_meta(report: &ConvergenceReport) -> ConvergenceMeta {
    let ConvergenceReport {
        iteration,
        de,
        dp_rms,
        dp_max,
        err_max,
        energy_conv,
        density_conv,
        de_noise_floor,
        diis_fallback,
        gradient_vetoes,
    } = *report;
    ConvergenceMeta {
        iteration,
        de,
        dp_rms,
        dp_max,
        err_max,
        energy_conv,
        density_conv,
        de_noise_floor,
        diis_fallback: diis_fallback.map(|f| {
            let DiisFallback {
                iteration,
                err_max,
                bound,
                flavor,
            } = f;
            FallbackMeta {
                iteration,
                err_max,
                bound,
                flavor: flavor.to_string(),
            }
        }),
        gradient_vetoes: gradient_vetoes.map(|v| {
            let GradientVetoes {
                count,
                last_iteration,
                last_orbital_gradient,
                bound,
            } = v;
            VetoesMeta {
                count,
                last_iteration,
                last_orbital_gradient,
                bound,
            }
        }),
    }
}

/// `DiisFallback::flavor` is `&'static str`, set in one place in `rhf.rs`.
fn fallback_flavor(name: &str) -> Result<&'static str, CheckpointError> {
    match name {
        "ADIIS" => Ok("ADIIS"),
        "EDIIS" => Ok("EDIIS"),
        other => Err(CheckpointError::Format(format!(
            "unknown DIIS fallback flavor {other:?}"
        ))),
    }
}

fn convergence_from_meta(meta: ConvergenceMeta) -> Result<ConvergenceReport, CheckpointError> {
    let diis_fallback = meta
        .diis_fallback
        .map(|f| {
            Ok::<_, CheckpointError>(DiisFallback {
                iteration: f.iteration,
                err_max: f.err_max,
                bound: f.bound,
                flavor: fallback_flavor(&f.flavor)?,
            })
        })
        .transpose()?;
    Ok(ConvergenceReport {
        iteration: meta.iteration,
        de: meta.de,
        dp_rms: meta.dp_rms,
        dp_max: meta.dp_max,
        err_max: meta.err_max,
        energy_conv: meta.energy_conv,
        density_conv: meta.density_conv,
        de_noise_floor: meta.de_noise_floor,
        diis_fallback,
        gradient_vetoes: meta.gradient_vetoes.map(|v| GradientVetoes {
            count: v.count,
            last_iteration: v.last_iteration,
            last_orbital_gradient: v.last_orbital_gradient,
            bound: v.bound,
        }),
    })
}

fn final_to_meta(state: &FinalState) -> FinalMeta {
    FinalMeta {
        energy: state.energy,
        one_electron_energy: state.one_electron_energy,
        nuclear_repulsion: state.nuclear_repulsion,
        df_jk: state.df_jk.as_ref().map(route_to_meta),
        exit: exit_name(state.exit).to_string(),
        converged: state.converged,
        iterations: state.iterations,
        computed_quartets: state.computed_quartets,
        convergence: state.convergence.as_ref().map(convergence_to_meta),
        precision_summary: state.precision_summary.clone(),
        precision_detail: state.precision_detail.clone(),
        not_stored: state.not_stored.clone(),
    }
}

/// An array with the layout the file will reproduce: C or F order kept, anything
/// else (a reversed axis, a stride gap) rewritten to C order. The NPY header can
/// only express those two, and the checksum covers which one it is.
fn stored_layout<D: ndarray::Dimension>(a: &ndarray::Array<f64, D>) -> ArrayD<f64> {
    let dynamic = a.view().into_dyn();
    let contiguous = dynamic.is_standard_layout() || dynamic.t().is_standard_layout();
    if contiguous {
        dynamic.to_owned()
    } else {
        let values: Vec<f64> = dynamic.iter().copied().collect();
        ArrayD::from_shape_vec(IxDyn(dynamic.shape()), values)
            .expect("the shape came from the same array")
    }
}

pub(super) fn encode_arrays(ck: &Checkpoint) -> BTreeMap<String, ArrayD<f64>> {
    let o = &ck.orbitals;
    let mut out = BTreeMap::new();
    let mut put2 = |name: &str, a: &Array2<f64>| {
        out.insert(name.to_string(), stored_layout(a));
    };
    put2("density_total", &o.density_total);
    put2("density_alpha", &o.density_alpha);
    put2("mos_alpha", &o.mos_alpha);
    if let Some(a) = &o.density_beta {
        put2("density_beta", a);
    }
    if let Some(a) = &o.mos_beta {
        put2("mos_beta", a);
    }
    if let Some(state) = &ck.final_state {
        put2("fock_alpha", &state.fock_alpha);
        if let Some(a) = &state.fock_beta {
            put2("fock_beta", a);
        }
        if let Some((a, b)) = &state.rohf_spin_focks {
            put2("rohf_spin_fock_alpha", a);
            put2("rohf_spin_fock_beta", b);
        }
    }
    let mut put1 = |name: &str, v: &[f64]| {
        out.insert(name.to_string(), Array1::from(v.to_vec()).into_dyn());
    };
    put1("eps_alpha", &o.eps_alpha);
    if let Some(v) = &o.eps_beta {
        put1("eps_beta", v);
    }
    out
}

pub(super) fn encode_meta(ck: &Checkpoint) -> Result<Vec<u8>, CheckpointError> {
    let meta = Meta {
        build: ck.build.clone(),
        identity: ck.identity.clone(),
        spin: ck.orbitals.spin,
        iteration: ck.iteration,
        final_state: ck.final_state.as_ref().map(final_to_meta),
    };
    serde_json::to_vec(&meta).map_err(|e| CheckpointError::Format(e.to_string()))
}

/// See the module docs of `checkpoint` for the definition, which the Python test
/// recomputes independently.
pub(super) fn checksum(version: u64, meta: &[u8], arrays: &BTreeMap<String, ArrayD<f64>>) -> u64 {
    let mut h = Fnv::new();
    h.update(HASH_DOMAIN);
    h.update(&version.to_le_bytes());
    h.update(&(meta.len() as u64).to_le_bytes());
    h.update(meta);
    for (name, a) in arrays {
        h.update(&(name.len() as u64).to_le_bytes());
        h.update(name.as_bytes());
        h.update(&(a.ndim() as u64).to_le_bytes());
        for &d in a.shape() {
            h.update(&(d as u64).to_le_bytes());
        }
        h.update(&[u8::from(!a.is_standard_layout())]);
        let values = a
            .as_slice_memory_order()
            .expect("stored_layout and the NPY reader produce contiguous arrays");
        for v in values {
            h.update(&v.to_le_bytes());
        }
    }
    h.0
}

/// The members of one NPZ, in the form both directions share.
pub(super) struct Members {
    pub(super) version: u64,
    pub(super) checksum: u64,
    pub(super) meta: Vec<u8>,
    pub(super) arrays: BTreeMap<String, ArrayD<f64>>,
}

fn npz_error<E: std::fmt::Display>(e: E) -> CheckpointError {
    CheckpointError::Format(e.to_string())
}

pub(super) fn write_members<W: Write + Seek>(
    writer: W,
    members: &Members,
) -> Result<W, CheckpointError> {
    let mut npz = NpzWriter::new(writer);
    npz.add_array(VERSION_MEMBER, &arr1(&[members.version]))
        .map_err(npz_error)?;
    npz.add_array(CHECKSUM_MEMBER, &arr1(&[members.checksum]))
        .map_err(npz_error)?;
    npz.add_array(META_MEMBER, &Array1::from(members.meta.clone()))
        .map_err(npz_error)?;
    for (name, a) in &members.arrays {
        npz.add_array(name.as_str(), a).map_err(npz_error)?;
    }
    npz.finish().map_err(npz_error)
}

/// Reports [`WriteStage::PartWritten`] once, part way through the temp file.
struct Observed<'a> {
    file: File,
    written: u64,
    observer: &'a mut dyn FnMut(WriteStage),
    fired: bool,
}

/// Enough to be past the first members and short of the end of any real file.
const PART_WRITTEN_BYTES: u64 = 1024;

impl Write for Observed<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.file.write(buf)?;
        self.written += n as u64;
        if !self.fired && self.written >= PART_WRITTEN_BYTES {
            self.fired = true;
            (self.observer)(WriteStage::PartWritten);
        }
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

impl Seek for Observed<'_> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.file.seek(pos)
    }
}

fn io_error(path: &Path, what: &str, e: &io::Error) -> CheckpointError {
    CheckpointError::Io {
        path: path.display().to_string(),
        detail: format!("{what}: {e}"),
    }
}

/// Removes the temp file on any early return. A kill runs no destructor, which
/// is why a kill can leave one behind.
struct TempGuard(PathBuf);

impl Drop for TempGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

pub(super) static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// How many taken temp names a write steps past before giving up.
const MAX_TEMP_ATTEMPTS: usize = 4096;

/// Every temp file of `path` starts with this.
fn temp_prefix(path: &Path) -> Result<String, CheckpointError> {
    let name = path
        .file_name()
        .ok_or_else(|| CheckpointError::Io {
            path: path.display().to_string(),
            detail: "the path has no file name".to_string(),
        })?
        .to_string_lossy();
    Ok(format!(".{name}.tmp."))
}

/// A sibling of `path` for this writer, never matching `path` itself. The PID and
/// sequence do not make it unique: a restarted job in a container can have the
/// PID of the job it replaced, and its sequence starts at 0 again.
fn temp_path(path: &Path, prefix: &str) -> PathBuf {
    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    path.with_file_name(format!("{prefix}{}.{seq}", std::process::id()))
}

/// The open file is still what `temp` names, so no cleaner removed it between its
/// creation and its lock.
fn still_named(file: &File, temp: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (file.metadata(), fs::symlink_metadata(temp)) {
        (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
        _ => false,
    }
}

/// Create and lock a new temp file. The lock is what marks a writer as alive: the
/// kernel drops it when the process dies, however it dies.
fn create_temp(path: &Path, prefix: &str) -> Result<(PathBuf, File), CheckpointError> {
    for _ in 0..MAX_TEMP_ATTEMPTS {
        let temp = temp_path(path, prefix);
        let file = match OpenOptions::new().write(true).create_new(true).open(&temp) {
            Ok(file) => file,
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(io_error(path, "cannot create the temp file", &e)),
        };
        if file.try_lock().is_ok() && still_named(&file, &temp) {
            return Ok((temp, file));
        }
    }
    Err(CheckpointError::Io {
        path: path.display().to_string(),
        detail: format!("no free temp file name in {MAX_TEMP_ATTEMPTS} attempts"),
    })
}

/// Remove temp files of `path` that no live writer holds. Best effort: a file that
/// cannot be inspected is left, and failing here must not fail the write.
fn remove_stale_temps(path: &Path, prefix: &str) {
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let Ok(entries) = fs::read_dir(parent) else {
        return;
    };
    for entry in entries.flatten() {
        let is_temp = entry.file_name().to_string_lossy().starts_with(prefix);
        let is_file = entry.file_type().is_ok_and(|t| t.is_file());
        if !is_temp || !is_file {
            continue;
        }
        let Ok(file) = File::open(entry.path()) else {
            continue;
        };
        if file.try_lock().is_ok() {
            let _ = fs::remove_file(entry.path());
        }
    }
}

/// `rename` would replace a link at `path` and leave its target holding the old
/// contents, so a later read through the target would see stale data.
fn refuse_symlink(path: &Path) -> Result<(), CheckpointError> {
    let is_link = fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink());
    if is_link {
        return Err(CheckpointError::Io {
            path: path.display().to_string(),
            detail: "refusing to write through a symlink; name its target instead".to_string(),
        });
    }
    Ok(())
}

pub(super) fn write(
    ck: &Checkpoint,
    path: &Path,
    observer: &mut dyn FnMut(WriteStage),
) -> Result<u64, CheckpointError> {
    ck.validate()?;
    let arrays = encode_arrays(ck);
    let meta = encode_meta(ck)?;
    let hash = checksum(FORMAT_VERSION, &meta, &arrays);
    let members = Members {
        version: FORMAT_VERSION,
        checksum: hash,
        meta,
        arrays,
    };
    write_atomic(path, &members, observer)?;
    Ok(hash)
}

/// Anything that lands at `path` goes through here.
pub(super) fn write_atomic(
    path: &Path,
    members: &Members,
    observer: &mut dyn FnMut(WriteStage),
) -> Result<(), CheckpointError> {
    refuse_symlink(path)?;
    let prefix = temp_prefix(path)?;
    remove_stale_temps(path, &prefix);
    let (temp, file) = create_temp(path, &prefix)?;
    let guard = TempGuard(temp.clone());
    let sink = Observed {
        file,
        written: 0,
        observer: &mut *observer,
        fired: false,
    };
    let sink = write_members(sink, members)?;
    let file = sink.file;
    observer(WriteStage::TempWritten);
    file.sync_all()
        .map_err(|e| io_error(path, "cannot sync the temp file", &e))?;
    observer(WriteStage::BeforeRename);
    fs::rename(&temp, path).map_err(|e| io_error(path, "cannot rename into place", &e))?;
    std::mem::forget(guard);
    // Held until here: unlocked before the rename, a cleaner could remove it.
    drop(file);
    observer(WriteStage::Renamed);
    sync_parent(path)
}

fn sync_parent(path: &Path) -> Result<(), CheckpointError> {
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    File::open(parent)
        .and_then(|dir| dir.sync_all())
        .map_err(|e| io_error(path, "cannot sync the directory", &e))
}

fn read_scalar_u64<R: io::Read + Seek>(
    npz: &mut NpzReader<R>,
    name: &str,
) -> Result<u64, CheckpointError> {
    let a: Array1<u64> = npz
        .by_name(name)
        .map_err(|e| CheckpointError::Format(format!("member {name}: {e}")))?;
    match a.as_slice() {
        Some([v]) => Ok(*v),
        _ => Err(CheckpointError::Format(format!(
            "member {name} must hold exactly one value"
        ))),
    }
}

/// `(bytes per element, shape)` from an NPY header in exactly the form this
/// format's writer emits. Anything else is refused rather than interpreted: the
/// real parser accepts more spellings than this one, and a spelling only it
/// understood could carry a shape that this check never saw.
fn parse_npy_header(header: &str) -> Option<(u64, Vec<u64>)> {
    let rest = header.strip_prefix("{'descr': '")?;
    let (descr, rest) = rest.split_once("', 'fortran_order': ")?;
    let element_bytes = match descr {
        "<f8" | "<u8" => 8,
        "|u1" => 1,
        _ => return None,
    };
    let rest = rest
        .strip_prefix("False, 'shape': (")
        .or_else(|| rest.strip_prefix("True, 'shape': ("))?;
    let (dims, tail) = rest.split_once(")}")?;
    if !tail.trim_end_matches('\n').trim_end_matches(' ').is_empty() {
        return None;
    }
    let shape = dims
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(|d| {
            d.bytes()
                .all(|b| b.is_ascii_digit())
                .then(|| d.parse().ok())?
        })
        .collect::<Option<Vec<u64>>>()?;
    Some((element_bytes, shape))
}

/// The bytes a member's header says its data occupies; `None` on overflow.
fn npy_data_bytes(element_bytes: u64, shape: &[u64]) -> Option<u64> {
    shape
        .iter()
        .try_fold(element_bytes, |total, &d| total.checked_mul(d))
}

/// Check one member's header against its zip size before anything allocates from
/// that header. The NPY reader sizes its buffer from the shape alone, so a 517-byte
/// file claiming 2^63 bytes aborts the process, which no `Result` can report.
fn check_member(member: &mut zip::read::ZipFile<'_>, file_len: u64) -> Result<(), CheckpointError> {
    use std::io::Read as _;
    let name = member.name().to_string();
    let bad = |why: &str| CheckpointError::Format(format!("member {name}: {why}"));
    if member.compression() != zip::CompressionMethod::Stored {
        return Err(bad(
            "is compressed; this format stores members uncompressed",
        ));
    }
    let size = member.size();
    if size > file_len {
        return Err(bad("claims to be larger than the file"));
    }
    let mut preamble = [0_u8; 10];
    member
        .read_exact(&mut preamble)
        .map_err(|_| bad("is too short to be an NPY array"))?;
    if preamble[..8] != *b"\x93NUMPY\x01\x00" {
        return Err(bad("is not an NPY 1.0 array"));
    }
    let header_len = u64::from(u16::from_le_bytes([preamble[8], preamble[9]]));
    let mut header = vec![0_u8; header_len as usize];
    member
        .read_exact(&mut header)
        .map_err(|_| bad("has a truncated NPY header"))?;
    let (element_bytes, shape) = std::str::from_utf8(&header)
        .ok()
        .and_then(parse_npy_header)
        .ok_or_else(|| bad("has an NPY header this format does not write"))?;
    let expected = npy_data_bytes(element_bytes, &shape)
        .and_then(|data| data.checked_add(10 + header_len))
        .ok_or_else(|| bad("has a shape that overflows"))?;
    if expected != size {
        return Err(bad(&format!(
            "header shape {shape:?} needs {expected} bytes but the member holds {size}"
        )));
    }
    Ok(())
}

fn check_all_members(bytes: &[u8]) -> Result<(), CheckpointError> {
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).map_err(npz_error)?;
    for i in 0..zip.len() {
        let mut member = zip.by_index(i).map_err(npz_error)?;
        check_member(&mut member, bytes.len() as u64)?;
    }
    Ok(())
}

/// Version first, so that a future format reports its version rather than
/// whatever else about it this build cannot parse.
fn read_members(bytes: Vec<u8>) -> Result<Members, CheckpointError> {
    check_all_members(&bytes)?;
    let mut npz = NpzReader::new(Cursor::new(bytes)).map_err(npz_error)?;
    let version = read_scalar_u64(&mut npz, VERSION_MEMBER)?;
    if version != FORMAT_VERSION {
        return Err(CheckpointError::Version { found: version });
    }
    let checksum = read_scalar_u64(&mut npz, CHECKSUM_MEMBER)?;
    let meta: Array1<u8> = npz
        .by_name(META_MEMBER)
        .map_err(|e| CheckpointError::Format(format!("member {META_MEMBER}: {e}")))?;
    let mut arrays = BTreeMap::new();
    for name in npz.names().map_err(npz_error)? {
        if matches!(
            name.as_str(),
            VERSION_MEMBER | CHECKSUM_MEMBER | META_MEMBER
        ) {
            continue;
        }
        let a: ArrayD<f64> = npz
            .by_name(&name)
            .map_err(|e| CheckpointError::Format(format!("member {name}: {e}")))?;
        arrays.insert(name, a);
    }
    Ok(Members {
        version,
        checksum,
        meta: meta.to_vec(),
        arrays,
    })
}

fn take2(
    arrays: &mut BTreeMap<String, ArrayD<f64>>,
    name: &str,
) -> Result<Option<Array2<f64>>, CheckpointError> {
    arrays
        .remove(name)
        .map(|a| {
            a.into_dimensionality::<Ix2>()
                .map_err(|_| CheckpointError::Shape {
                    array: name.to_string(),
                    expected: "two dimensions".to_string(),
                    found: "another rank".to_string(),
                })
        })
        .transpose()
}

fn take1(
    arrays: &mut BTreeMap<String, ArrayD<f64>>,
    name: &str,
) -> Result<Option<Vec<f64>>, CheckpointError> {
    arrays
        .remove(name)
        .map(|a| {
            a.into_dimensionality::<Ix1>()
                .map(|v| v.to_vec())
                .map_err(|_| CheckpointError::Shape {
                    array: name.to_string(),
                    expected: "one dimension".to_string(),
                    found: "another rank".to_string(),
                })
        })
        .transpose()
}

fn required<T>(value: Option<T>, name: &str) -> Result<T, CheckpointError> {
    value.ok_or_else(|| CheckpointError::Format(format!("member {name} is missing")))
}

fn decode_final(
    meta: FinalMeta,
    arrays: &mut BTreeMap<String, ArrayD<f64>>,
) -> Result<FinalState, CheckpointError> {
    let spin_focks = (
        take2(arrays, "rohf_spin_fock_alpha")?,
        take2(arrays, "rohf_spin_fock_beta")?,
    );
    let rohf_spin_focks = match spin_focks {
        (Some(a), Some(b)) => Some((a, b)),
        (None, None) => None,
        _ => {
            return Err(CheckpointError::Format(
                "rohf_spin_fock_alpha and rohf_spin_fock_beta must come together".to_string(),
            ))
        }
    };
    Ok(FinalState {
        energy: meta.energy,
        one_electron_energy: meta.one_electron_energy,
        nuclear_repulsion: meta.nuclear_repulsion,
        fock_alpha: required(take2(arrays, "fock_alpha")?, "fock_alpha")?,
        fock_beta: take2(arrays, "fock_beta")?,
        rohf_spin_focks,
        df_jk: meta.df_jk.map(route_from_meta).transpose()?,
        exit: exit_from_name(&meta.exit)?,
        converged: meta.converged,
        iterations: meta.iterations,
        computed_quartets: meta.computed_quartets,
        convergence: meta.convergence.map(convergence_from_meta).transpose()?,
        precision_summary: meta.precision_summary,
        precision_detail: meta.precision_detail,
        not_stored: meta.not_stored,
    })
}

fn decode(members: Members) -> Result<Checkpoint, CheckpointError> {
    let Members {
        checksum: stored,
        meta,
        mut arrays,
        ..
    } = members;
    let meta: Meta =
        serde_json::from_slice(&meta).map_err(|e| CheckpointError::Format(format!("meta: {e}")))?;
    let a = &mut arrays;
    let orbitals = OrbitalState {
        spin: meta.spin,
        density_total: required(take2(a, "density_total")?, "density_total")?,
        density_alpha: required(take2(a, "density_alpha")?, "density_alpha")?,
        density_beta: take2(a, "density_beta")?,
        mos_alpha: required(take2(a, "mos_alpha")?, "mos_alpha")?,
        mos_beta: take2(a, "mos_beta")?,
        eps_alpha: required(take1(a, "eps_alpha")?, "eps_alpha")?,
        eps_beta: take1(a, "eps_beta")?,
    };
    let final_state = meta.final_state.map(|f| decode_final(f, a)).transpose()?;
    if !arrays.is_empty() {
        let extra: BTreeSet<&String> = arrays.keys().collect();
        return Err(CheckpointError::Format(format!(
            "unexpected members {extra:?}"
        )));
    }
    let ck = Checkpoint {
        build: meta.build,
        identity: meta.identity,
        orbitals,
        final_state,
        iteration: meta.iteration,
        file_hash: Some(stored),
    };
    ck.validate()?;
    Ok(ck)
}

pub(super) fn read(path: &Path) -> Result<Checkpoint, CheckpointError> {
    let bytes = fs::read(path).map_err(|e| {
        if e.kind() == io::ErrorKind::NotFound {
            CheckpointError::Missing {
                path: path.display().to_string(),
            }
        } else {
            io_error(path, "cannot read", &e)
        }
    })?;
    let members = read_members(bytes)?;
    let computed = checksum(members.version, &members.meta, &members.arrays);
    if computed != members.checksum {
        return Err(CheckpointError::Checksum {
            stored: members.checksum,
            computed,
        });
    }
    decode(members)
}

fn shape_error(array: &str, expected: String, found: &[usize]) -> CheckpointError {
    CheckpointError::Shape {
        array: array.to_string(),
        expected,
        found: format!("{found:?}"),
    }
}

fn check_square(name: &str, a: &Array2<f64>, nao: usize) -> Result<(), CheckpointError> {
    if a.shape() != [nao, nao] {
        return Err(shape_error(name, format!("[{nao}, {nao}]"), a.shape()));
    }
    check_finite(name, a.iter())
}

fn check_finite<'a>(
    name: &str,
    mut values: impl Iterator<Item = &'a f64>,
) -> Result<(), CheckpointError> {
    if values.all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(CheckpointError::NonFinite {
            array: name.to_string(),
        })
    }
}

/// `mos` is nao x nmo (a linearly dependent basis has nmo < nao), `eps` has nmo.
fn check_channel(
    mos_name: &str,
    mos: &Array2<f64>,
    eps_name: &str,
    eps: &[f64],
    nao: usize,
) -> Result<(), CheckpointError> {
    if mos.nrows() != nao || mos.ncols() > nao {
        return Err(shape_error(
            mos_name,
            format!("[{nao}, <={nao}]"),
            mos.shape(),
        ));
    }
    if eps.len() != mos.ncols() {
        return Err(shape_error(
            eps_name,
            format!("[{}]", mos.ncols()),
            &[eps.len()],
        ));
    }
    check_finite(mos_name, mos.iter())?;
    check_finite(eps_name, eps.iter())
}

fn check_spin_members(o: &OrbitalState) -> Result<(), CheckpointError> {
    let restricted = o.spin == Spin::Restricted;
    if restricted == o.density_beta.is_some() {
        return Err(CheckpointError::Format(format!(
            "density_beta presence does not match spin {:?}",
            o.spin
        )));
    }
    if o.mos_beta.is_some() != o.eps_beta.is_some() || (restricted && o.mos_beta.is_some()) {
        return Err(CheckpointError::Format(format!(
            "mos_beta/eps_beta presence does not match spin {:?}",
            o.spin
        )));
    }
    Ok(())
}

fn validate_orbitals(o: &OrbitalState, nao: usize) -> Result<(), CheckpointError> {
    check_spin_members(o)?;
    check_square("density_total", &o.density_total, nao)?;
    check_square("density_alpha", &o.density_alpha, nao)?;
    if let Some(d) = &o.density_beta {
        check_square("density_beta", d, nao)?;
    }
    check_channel("mos_alpha", &o.mos_alpha, "eps_alpha", &o.eps_alpha, nao)?;
    if let (Some(m), Some(e)) = (&o.mos_beta, &o.eps_beta) {
        check_channel("mos_beta", m, "eps_beta", e, nao)?;
    }
    Ok(())
}

fn validate_final(s: &FinalState, nao: usize) -> Result<(), CheckpointError> {
    for (name, value) in [
        ("energy", s.energy),
        ("one_electron_energy", s.one_electron_energy),
        ("nuclear_repulsion", s.nuclear_repulsion),
    ] {
        check_finite(name, std::iter::once(&value))?;
    }
    check_square("fock_alpha", &s.fock_alpha, nao)?;
    if let Some(f) = &s.fock_beta {
        check_square("fock_beta", f, nao)?;
    }
    if let Some((a, b)) = &s.rohf_spin_focks {
        check_square("rohf_spin_fock_alpha", a, nao)?;
        check_square("rohf_spin_fock_beta", b, nao)?;
    }
    Ok(())
}

pub(super) fn validate(ck: &Checkpoint) -> Result<(), CheckpointError> {
    let id = &ck.identity;
    let spin = format!("{:?}", ck.orbitals.spin);
    if !id
        .fields
        .iter()
        .any(|f| f.name == "spin" && f.value == spin)
    {
        return Err(CheckpointError::Format(format!(
            "the orbitals are {spin} but the identity does not say so"
        )));
    }
    check_finite("coords_bohr", id.coords_bohr.iter().flatten())?;
    check_finite("energy_conv", std::iter::once(&id.energy_conv))?;
    check_finite("density_conv", std::iter::once(&id.density_conv))?;
    validate_orbitals(&ck.orbitals, id.nao)?;
    if let Some(state) = &ck.final_state {
        validate_final(state, id.nao)?;
    }
    Ok(())
}
