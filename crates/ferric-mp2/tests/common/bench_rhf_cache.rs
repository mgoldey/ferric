//! Bitwise on-disk cache of the RHF that `lmp2_direct.rs`'s `setup()` solves, so
//! a bench process can skip the SCF (and the Schwarz bounds only it needs).
//!
//! `FERRIC_LMP2_BENCH_RHF_CACHE=<existing dir>` turns it on; unset, `setup()`
//! solves as it always did. A missing file is solved and written. A file that
//! exists but has another key, a bad checksum or the wrong shape panics: it is
//! never recomputed over, because a silent recompute would hide a wrong cache.
//! The file name holds no SCF config, so a config change is a mismatch, not a miss.
//!
//! The key covers the xyz text, both basis sets (name and data), the operator
//! and `format!("{:?}", RhfConfig)`. It does NOT cover the build or the thread
//! counts: reading a cache from another build asserts that build's SCF gives
//! the same result, which `bench_rhf_cache_matches_fresh_scf` checks.
//!
//! This file depends on nothing else in the tests, so it can be copied into a
//! frozen baseline worktree. Before #603 `ScfResult` has no `precision` field;
//! delete that one line of `read` there.

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::{ScfExit, ScfResult, Spin};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const ENV_DIR: &str = "FERRIC_LMP2_BENCH_RHF_CACHE";
const MAGIC: &[u8; 8] = b"FRHFBIT1";
const CHECKSUM_LEN: usize = 8;

/// FNV-1a 64: detects corruption and keys content. Not collision-resistant
/// against an adversary, which a local bench cache does not need.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The SCF the bench runs.
pub fn bench_scf_config() -> RhfConfig {
    RhfConfig {
        energy_conv: 1e-10,
        ..Default::default()
    }
}

/// Solve the bench's RHF, with no cache involved.
pub fn solve_fresh(mol: &Molecule, obs: &PreparedBasis) -> ScfResult {
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, obs).unwrap();
    solve_rhf(
        &ParallelContext::default(),
        mol,
        obs,
        op,
        &bounds,
        &bench_scf_config(),
    )
    .unwrap()
}

/// A basis set's data, independent of the process. `BasisSet` holds `HashMap`s,
/// whose iteration order (so its derived `Debug`) changes with every process,
/// so the elements are sorted first.
fn basis_hash(bs: &BasisSet) -> u64 {
    let shells: BTreeMap<_, _> = bs.shells.iter().collect();
    let ecps: BTreeMap<_, _> = bs.ecps.iter().collect();
    fnv1a64(format!("{:?}|{shells:?}|{ecps:?}", bs.name).as_bytes())
}

/// Everything the cached RHF depends on, other than the build and thread counts.
pub fn cache_key(
    xyz_text: &str,
    basis: (&str, &BasisSet),
    aux: (&str, &BasisSet),
    scf: &RhfConfig,
) -> String {
    format!(
        "xyz={:016x};basis={};basis_data={:016x};aux={};aux_data={:016x};op=coulomb;scf={:016x}",
        fnv1a64(xyz_text.as_bytes()),
        basis.0,
        basis_hash(basis.1),
        aux.0,
        basis_hash(aux.1),
        fnv1a64(format!("{scf:?}").as_bytes()),
    )
}

pub fn cache_path(dir: &Path, stem: &str, basis: &str, aux: &str) -> PathBuf {
    dir.join(format!("{stem}.{basis}.{aux}.rhfbits"))
}

/// The cache directory from the environment: `None` when unset. A set value
/// that is unusable panics rather than silently running without the cache.
pub fn dir_from_env() -> Option<PathBuf> {
    let raw = match std::env::var(ENV_DIR) {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return None,
        Err(e) => panic!("{ENV_DIR}: {e}"),
    };
    let dir = PathBuf::from(&raw);
    assert!(
        dir.is_dir(),
        "{ENV_DIR}: {raw:?} is not an existing directory"
    );
    Some(dir)
}

/// `setup()`'s RHF: from the cache when [`ENV_DIR`] is set, else solved.
pub fn setup_rhf(
    xyz_path: &str,
    mol: &Molecule,
    obs: &PreparedBasis,
    basis: (&str, &BasisSet),
    aux: (&str, &BasisSet),
) -> ScfResult {
    let Some(dir) = dir_from_env() else {
        return solve_fresh(mol, obs);
    };
    let xyz_text = std::fs::read_to_string(xyz_path).unwrap();
    let stem = Path::new(xyz_path).file_stem().unwrap().to_str().unwrap();
    let key = cache_key(&xyz_text, basis, aux, &bench_scf_config());
    let path = cache_path(&dir, stem, basis.0, aux.0);
    load_or_solve(&path, &key, obs.nbasis(), || solve_fresh(mol, obs))
}

/// Read `path`, or run `solve` and write it when it does not exist. The written
/// result is read back, so a cold and a warm process see the same object. Any
/// unusable file panics and `solve` does not run.
pub fn load_or_solve(
    path: &Path,
    key: &str,
    nbasis: usize,
    solve: impl FnOnce() -> ScfResult,
) -> ScfResult {
    let fail = |e: String| -> ! { panic!("{ENV_DIR}: {}: {e}", path.display()) };
    if path.try_exists().unwrap_or_else(|e| fail(e.to_string())) {
        let rhf = read(path, key, nbasis).unwrap_or_else(|e| fail(e));
        eprintln!(
            "[rhf-cache] loaded {} E={:.12} iters={}",
            path.display(),
            rhf.energy,
            rhf.iterations
        );
        return rhf;
    }
    let fresh = solve();
    write(path, key, &fresh).unwrap_or_else(|e| fail(e));
    let rhf = read(path, key, nbasis).unwrap_or_else(|e| fail(e));
    let diff = rhf_bits_diff(&rhf, &fresh);
    if !diff.is_empty() {
        fail(format!("read-back differs from what was written: {diff:?}"));
    }
    eprintln!(
        "[rhf-cache] wrote {} E={:.12} iters={}",
        path.display(),
        rhf.energy,
        rhf.iterations
    );
    rhf
}

fn exit_code(exit: ScfExit) -> Result<u8, String> {
    match exit {
        ScfExit::Converged => Ok(0),
        ScfExit::Plateau => Ok(1),
        other => Err(format!("refusing an SCF that exited {other:?}")),
    }
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_matrix(out: &mut Vec<u8>, a: &Array2<f64>) -> Result<(), String> {
    // A kernel's bits can depend on the operand layout, and the reader rebuilds
    // row-major, so anything else would not come back as it was solved.
    if !a.is_standard_layout() {
        return Err("refusing a matrix that is not row-major".to_string());
    }
    put_u64(out, a.nrows() as u64);
    put_u64(out, a.ncols() as u64);
    for x in a.iter() {
        put_u64(out, x.to_bits());
    }
    Ok(())
}

/// Serialize a closed-shell converged RHF. Anything else is refused, as are the
/// optional fields a cached result would drop and that could matter: a result
/// with density-fitted J/K or an embedding does not belong in this cache.
pub fn encode(key: &str, rhf: &ScfResult) -> Result<Vec<u8>, String> {
    if rhf.spin != Spin::Restricted || rhf.mos_beta.is_some() || rhf.fock_beta.is_some() {
        return Err("refusing a result that is not closed-shell".to_string());
    }
    if !rhf.converged {
        return Err("refusing an unconverged SCF".to_string());
    }
    if rhf.df_jk.is_some() || rhf.induced_dipoles.is_some() {
        return Err("refusing a density-fitted or polarizable SCF".to_string());
    }
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    put_u64(&mut out, key.len() as u64);
    out.extend_from_slice(key.as_bytes());
    put_u64(&mut out, rhf.energy.to_bits());
    put_u64(&mut out, rhf.iterations as u64);
    put_u64(&mut out, rhf.computed_quartets as u64);
    out.push(exit_code(rhf.exit)?);
    put_u64(&mut out, rhf.eps_alpha.len() as u64);
    for x in &rhf.eps_alpha {
        put_u64(&mut out, x.to_bits());
    }
    for a in [
        &rhf.mos_alpha,
        &rhf.fock_alpha,
        &rhf.density_total,
        &rhf.density_alpha,
    ] {
        put_matrix(&mut out, a)?;
    }
    let checksum = fnv1a64(&out);
    put_u64(&mut out, checksum);
    Ok(out)
}

/// Write atomically, so a killed process cannot leave a truncated cache.
pub fn write(path: &Path, key: &str, rhf: &ScfResult) -> Result<(), String> {
    let bytes = encode(key, rhf)?;
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&tmp, bytes).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        if n > self.0.len() {
            return Err(format!("truncated: need {n} bytes, have {}", self.0.len()));
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn usize(&mut self) -> Result<usize, String> {
        usize::try_from(self.u64()?).map_err(|e| e.to_string())
    }
    fn f64s(&mut self, n: usize) -> Result<Vec<f64>, String> {
        let nbytes = n.checked_mul(8).ok_or("array length overflows")?;
        let raw = self.take(nbytes)?;
        let v: Vec<f64> = raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_bits(u64::from_le_bytes(*c)))
            .collect();
        if v.iter().any(|x| !x.is_finite()) {
            return Err("non-finite value".to_string());
        }
        Ok(v)
    }
    fn matrix(&mut self, name: &str, rows: usize, cols: usize) -> Result<Array2<f64>, String> {
        let (r, c) = (self.usize()?, self.usize()?);
        if (r, c) != (rows, cols) {
            return Err(format!("{name} shape ({r},{c}), expected ({rows},{cols})"));
        }
        let values = self.f64s(r.checked_mul(c).ok_or("matrix size overflows")?)?;
        Array2::from_shape_vec((r, c), values).map_err(|e| e.to_string())
    }
}

/// Parse a cache file. `Err` for a corrupt file, another key or a wrong shape.
pub fn decode(bytes: &[u8], key: &str, nbasis: usize) -> Result<ScfResult, String> {
    if bytes.len() < MAGIC.len() + CHECKSUM_LEN {
        return Err(format!("corrupt: only {} bytes", bytes.len()));
    }
    let (body, stored) = bytes.split_at(bytes.len() - CHECKSUM_LEN);
    if fnv1a64(body).to_le_bytes() != stored {
        return Err("corrupt: checksum mismatch".to_string());
    }
    let mut r = Reader(body);
    if r.take(MAGIC.len())? != MAGIC {
        return Err("corrupt: bad magic".to_string());
    }
    let key_len = r.usize()?;
    let found = String::from_utf8_lossy(r.take(key_len)?).into_owned();
    if found != key {
        return Err(format!("key mismatch: file has {found}, expected {key}"));
    }
    let energy = f64::from_bits(r.u64()?);
    let iterations = r.usize()?;
    let computed_quartets = r.usize()?;
    let exit = match r.u8()? {
        0 => ScfExit::Converged,
        1 => ScfExit::Plateau,
        other => return Err(format!("unknown exit code {other}")),
    };
    let nmo = r.usize()?;
    if nmo == 0 || nmo > nbasis {
        return Err(format!("shape: {nmo} MOs with {nbasis} basis functions"));
    }
    let eps_alpha = r.f64s(nmo)?;
    let mos_alpha = r.matrix("mos_alpha", nbasis, nmo)?;
    let fock_alpha = r.matrix("fock_alpha", nbasis, nbasis)?;
    let density_total = r.matrix("density_total", nbasis, nbasis)?;
    let density_alpha = r.matrix("density_alpha", nbasis, nbasis)?;
    if !r.0.is_empty() {
        return Err(format!("{} trailing bytes", r.0.len()));
    }
    Ok(ScfResult {
        spin: Spin::Restricted,
        energy,
        density_total,
        density_alpha,
        density_beta: None,
        mos_alpha,
        mos_beta: None,
        eps_alpha,
        eps_beta: None,
        fock_alpha,
        fock_beta: None,
        converged: true,
        exit,
        iterations,
        computed_quartets,
        induced_dipoles: None,
        stability: None,
        stability_external: None,
        df_jk: None,
        rohf_spin_focks: None,
        cosx_final: None,
        cosx_schedule: None,
        convergence: None,
        guess_basis: None,
        precision: Default::default(),
    })
}

pub fn read(path: &Path, key: &str, nbasis: usize) -> Result<ScfResult, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    decode(&bytes, key, nbasis)
}

fn matrix_diff(name: &str, a: &Array2<f64>, b: &Array2<f64>, out: &mut Vec<String>) {
    if a.dim() != b.dim() {
        out.push(format!("{name}: shape {:?} vs {:?}", a.dim(), b.dim()));
        return;
    }
    if a.is_standard_layout() != b.is_standard_layout() {
        out.push(format!("{name}: memory layout differs"));
    }
    let mut n = 0usize;
    let mut max_abs = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.to_bits() != y.to_bits() {
            n += 1;
            max_abs = max_abs.max((x - y).abs());
        }
    }
    if n > 0 {
        out.push(format!(
            "{name}: {n} elements differ, max |d| {max_abs:.3e}"
        ));
    }
}

/// Every difference between the fields a cache holds, to the bit. Empty means
/// identical.
pub fn rhf_bits_diff(a: &ScfResult, b: &ScfResult) -> Vec<String> {
    let mut out = Vec::new();
    if a.energy.to_bits() != b.energy.to_bits() {
        out.push(format!("energy: {:e} vs {:e}", a.energy, b.energy));
    }
    if a.iterations != b.iterations {
        out.push(format!("iterations: {} vs {}", a.iterations, b.iterations));
    }
    if a.computed_quartets != b.computed_quartets {
        out.push(format!(
            "computed_quartets: {} vs {}",
            a.computed_quartets, b.computed_quartets
        ));
    }
    if a.exit != b.exit {
        out.push(format!("exit: {:?} vs {:?}", a.exit, b.exit));
    }
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    if bits(&a.eps_alpha) != bits(&b.eps_alpha) {
        out.push("eps_alpha differs".to_string());
    }
    matrix_diff("mos_alpha", &a.mos_alpha, &b.mos_alpha, &mut out);
    matrix_diff("fock_alpha", &a.fock_alpha, &b.fock_alpha, &mut out);
    matrix_diff(
        "density_total",
        &a.density_total,
        &b.density_total,
        &mut out,
    );
    matrix_diff(
        "density_alpha",
        &a.density_alpha,
        &b.density_alpha,
        &mut out,
    );
    out
}
