//! Where the raw RI-J/K `(P|μν)` tensor lives: the `[scf] jk_storage` policy and
//! the cost model that picks between disk and recompute when memory cannot
//! hold it.
//!
//! ```text
//!   tier          resident            per-pass cost
//!   InCore        naux·nao²·8         none
//!   InCorePacked  naux·nao(nao+1)/2·8 none
//!   Spill         one read block      read the packed file
//!   Recompute     one block           recompute the integrals
//! ```
//!
//! `auto` takes the first memory tier that fits the budget. Only when neither
//! fits does it choose between Spill and Recompute, and it does so from two
//! measurements taken on THIS machine at build time (see [`Calibration`]), not
//! from a constant: a laptop NVMe and a SATA SSD at 99% full give opposite
//! answers.

use super::{
    charge_three_index_soft, drop_page_cache, open_direct, pack_lower_triangle, packed_pair_len,
    read_spill_block, spill_block_naux_for, Backend, Operator, PreparedBasis, ThreeIndexSource,
};
use ferric_core::FerricError;
use ndarray::Array3;
use std::io::Write;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// `[scf] jk_storage`: where the raw RI-J/K tensor is kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JkStorage {
    /// Memory when it fits the budget, else the cheaper of disk and recompute
    /// by measured cost (default).
    Auto,
    /// Memory only (unpacked, or packed when only that fits); an error when
    /// neither fits the budget.
    Memory,
    /// Always the disk spill file, even when memory would hold it.
    Disk,
    /// Always recompute each block per pass, even when memory would hold it.
    Direct,
}

impl JkStorage {
    /// Parse the TOML / environment spelling.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "memory" => Ok(Self::Memory),
            "disk" => Ok(Self::Disk),
            "direct" => Ok(Self::Direct),
            other => Err(format!(
                "unknown jk_storage {other:?} (expected \"auto\", \"memory\", \"disk\" or \"direct\")"
            )),
        }
    }

    /// The TOML spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Memory => "memory",
            Self::Disk => "disk",
            Self::Direct => "direct",
        }
    }

    fn code(self) -> u8 {
        self as u8
    }

    fn from_code(c: u8) -> Option<Self> {
        [Self::Auto, Self::Memory, Self::Disk, Self::Direct]
            .into_iter()
            .find(|v| v.code() == c)
    }
}

impl std::fmt::Display for JkStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

const UNSET: u8 = u8::MAX;
static OVERRIDE: AtomicU8 = AtomicU8::new(UNSET);

/// The descriptor behind [`jk_storage`].
pub static JK_STORAGE_VAR: ferric_core::config::ConfigVar<JkStorage> =
    ferric_core::config::ConfigVar {
        env_name: "FERRIC_JK_STORAGE",
        default: JkStorage::Auto,
        parse: JkStorage::parse,
        validate: ferric_core::config::accept_any,
    };

/// The effective policy: an explicit [`set_jk_storage`] value beats the
/// `FERRIC_JK_STORAGE` environment variable, which beats `auto`. A malformed
/// environment value warns and falls back to `auto` (this is read where no
/// `Result` can propagate).
pub fn jk_storage() -> JkStorage {
    let explicit = JkStorage::from_code(OVERRIDE.load(Ordering::Relaxed));
    JK_STORAGE_VAR
        .resolve(explicit, ferric_core::config::env_lookup)
        .map(|r| r.value)
        .unwrap_or_else(|e| {
            eprintln!("[config] FERRIC_JK_STORAGE: {e}; using auto");
            JkStorage::Auto
        })
}

/// Resolve the policy strictly: an invalid `FERRIC_JK_STORAGE` is an error
/// naming the variable, not a warning. The CLI calls this once at start so the
/// environment is refused the way the TOML key is.
pub fn validate_jk_storage() -> Result<JkStorage, String> {
    let explicit = JkStorage::from_code(OVERRIDE.load(Ordering::Relaxed));
    JK_STORAGE_VAR
        .resolve(explicit, ferric_core::config::env_lookup)
        .map(|r| r.value)
        .map_err(|e| format!("FERRIC_JK_STORAGE: {e}"))
}

/// Set (or with `None`, clear) the process-wide explicit policy.
pub fn set_jk_storage(value: Option<JkStorage>) {
    OVERRIDE.store(value.map_or(UNSET, JkStorage::code), Ordering::Relaxed);
}

/// Where a built source lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    InCore,
    InCorePacked,
    Spill,
    Recompute,
}

impl Tier {
    /// Trace spelling.
    pub fn name(self) -> &'static str {
        match self {
            Self::InCore => "InCore",
            Self::InCorePacked => "InCorePacked",
            Self::Spill => "Spill",
            Self::Recompute => "Recompute",
        }
    }
}

/// The memory tier (if any) a band of this size gets under `budget_bytes`.
pub fn memory_tier(nao: usize, band: usize, budget_bytes: usize) -> Option<Tier> {
    let needed = band
        .saturating_mul(nao)
        .saturating_mul(nao)
        .saturating_mul(8);
    let packed = band.saturating_mul(packed_pair_len(nao)).saturating_mul(8);
    if needed <= budget_bytes {
        Some(Tier::InCore)
    } else if super::packed_in_core_fits(packed, nao, budget_bytes) {
        Some(Tier::InCorePacked)
    } else {
        None
    }
}

/// SCF J passes (two per Fock build) times the iterations a converging SCF
/// typically takes. The cost model's only assumed number; a run that needs
/// many more iterations favours Recompute over Spill's one-off write only if
/// recompute is also the cheaper pass, so the choice is insensitive to it
/// except near the break-even.
pub const ASSUMED_PASSES: f64 = 30.0;

/// Bytes written to the calibration file: large enough that the device, not
/// request latency, sets the rate.
const CALIBRATION_IO_BYTES: usize = 64 * 1024 * 1024;

/// Two measurements that decide Spill vs Recompute.
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    /// Seconds to compute (and pack) one aux row, from a timed block in the
    /// middle of the band.
    pub build_s_per_row: f64,
    /// `(write, read)` bytes per second of the spill directory, from a
    /// 64 MiB cold write + sync and a cold direct read. `None` when the
    /// spill file could not be created (spill is then not an option).
    pub io: Option<(f64, f64)>,
}

/// Does recomputing the integrals every pass beat writing the packed tensor
/// once and reading it back every pass?
///
/// ```text
///   spill     = max(T_build, bytes/R_write) + N · bytes/R_read
///   recompute = N · T_build
/// ```
/// `T_build` is the whole-band integral time, `N` is [`ASSUMED_PASSES`]. The
/// spill build overlaps compute with the write, hence the `max`. Ties go to
/// Recompute, which needs no scratch disk.
///
/// The shared RI-JK path streams the raw tensor one more time to dress it for
/// K. That pass costs `T_build` under Recompute and `bytes/R_read` under
/// Spill, one pass in `N + 1`, and is not counted: it cannot move the choice
/// except within a few percent of the break-even.
pub fn recompute_is_cheaper(cal: &Calibration, band: usize, packed_bytes: usize) -> bool {
    let t_build = cal.build_s_per_row * band as f64;
    let Some((w, r)) = cal.io else {
        return true;
    };
    let bytes = packed_bytes as f64;
    let spill = t_build.max(bytes / w.max(1.0)) + ASSUMED_PASSES * bytes / r.max(1.0);
    ASSUMED_PASSES * t_build <= spill
}

/// Pick the tier for `policy`. `calibrate` runs only for `auto` over budget.
pub fn decide(
    policy: JkStorage,
    (nao, band, budget_bytes): (usize, usize, usize),
    calibrate: impl FnOnce() -> Result<Calibration, FerricError>,
) -> Result<(Tier, Option<Calibration>), FerricError> {
    let mem = memory_tier(nao, band, budget_bytes);
    match policy {
        JkStorage::Disk => Ok((Tier::Spill, None)),
        JkStorage::Direct => Ok((Tier::Recompute, None)),
        JkStorage::Memory => mem.map(|t| (t, None)).ok_or_else(|| {
            let packed = band * packed_pair_len(nao) * 8;
            FerricError::General(format!(
                "jk_storage = \"memory\": the 3-index tensor needs {:.2} GB packed \
                 ({:.2} GB unpacked) plus 64 aux rows of scratch, but the memory budget (or what the shared pool has free) is \
                 {:.2} GB. Raise [memory] budget_gb or use jk_storage = \"auto\", \"disk\" \
                 or \"direct\".",
                packed as f64 / 1e9,
                (band * nao * nao * 8) as f64 / 1e9,
                budget_bytes as f64 / 1e9,
            ))
        }),
        JkStorage::Auto => match mem {
            Some(t) => Ok((t, None)),
            None => {
                let cal = calibrate()?;
                let packed = band.saturating_mul(packed_pair_len(nao)).saturating_mul(8);
                let t = if recompute_is_cheaper(&cal, band, packed) {
                    Tier::Recompute
                } else {
                    Tier::Spill
                };
                Ok((t, Some(cal)))
            }
        },
    }
}

/// Measure [`Calibration`] for the band `[p0, p1)`.
///
/// Costs one untimed warm-up block, four timed blocks of ~3% of the band each
/// (at most 128 rows), and a cold write + sync + direct read of up to 64 MiB:
/// about two seconds on a SATA SSD. Blocks use the same rayon-parallel
/// `eri3_block` the real build uses, so the per-row cost includes the parallel
/// efficiency actually obtained. Every allocation here is uncharged, so block
/// rows and the IO buffer are capped by `budget_bytes` the way the spill block
/// is.
///
/// `free_bytes` is the free space of the spill directory (`None` if unknown).
/// When the packed band plus the spill headroom does not fit there, the spill
/// is not an option and no IO is measured (`io = None`, which selects
/// Recompute).
pub fn calibrate(
    op: Operator,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    (p0, p1): (usize, usize),
    (budget_bytes, free_bytes): (usize, Option<u64>),
) -> Result<Calibration, FerricError> {
    let nao = obs.nbasis();
    let pair = packed_pair_len(nao);
    let band = p1 - p0;
    // Four blocks at the centres of the band's quarters: the cost per aux row
    // varies with the aux shell mix (heavy-atom d/f rows versus hydrogen
    // rows), so one block in the middle over- or under-states the average.
    let rows = (band / 32)
        .clamp(1, 128)
        .min(spill_block_naux_for(budget_bytes, nao))
        .min(band);
    // Warm-up with a full-size block, untimed: every rayon worker builds its
    // libint engine on first use, which would otherwise be charged to the rows.
    crate::threeindex::eri3_block(op, obs, dfbs, p0, p0 + rows)?;
    let mut sample = Vec::new();
    let mut seconds = 0.0;
    for q in 0..4 {
        let lo = p0 + ((2 * q + 1) * band / 8).min(band - rows);
        let t = Instant::now();
        let blk = crate::threeindex::eri3_block(op, obs, dfbs, lo, lo + rows)?;
        sample.resize(rows * pair, 0.0);
        pack_lower_triangle(&blk.view(), nao, &mut sample);
        seconds += t.elapsed().as_secs_f64();
    }
    let build_s_per_row = seconds.max(1e-9) / (4 * rows) as f64;
    let packed_bytes = band.saturating_mul(pair).saturating_mul(8);
    let spill_fits = free_bytes.is_none_or(|free| {
        super::check_spill_disk(packed_bytes as u64, free, &super::spill_dir()).is_ok()
    });
    let io = if spill_fits {
        measure_io(&sample, pair, band, budget_bytes / 8)
    } else {
        None
    };
    Ok(Calibration {
        build_s_per_row,
        io,
    })
}

/// Cold `(write, read)` rates of the spill directory, or `None` if it cannot
/// hold a calibration file. Content is the compute sample repeated: the rate of
/// an SSD does not depend on it. The file is at most [`CALIBRATION_IO_BYTES`]
/// and at most `max_bytes` (the read buffer is resident), at least one row.
fn measure_io(sample: &[f64], pair: usize, band: usize, max_bytes: usize) -> Option<(f64, f64)> {
    let io_bytes = CALIBRATION_IO_BYTES.min(max_bytes);
    let io_rows = (io_bytes / (pair * 8)).clamp(1, band);
    let mut file = tempfile::tempfile().ok()?;
    let t = Instant::now();
    let mut written = 0usize;
    while written < io_rows {
        let n = (sample.len() / pair).min(io_rows - written);
        file.write_all(bytemuck::cast_slice(&sample[..n * pair]))
            .ok()?;
        written += n;
    }
    file.sync_data().ok()?;
    let write_s = t.elapsed().as_secs_f64().max(1e-9);
    drop_page_cache(&file);
    let direct = open_direct(&file, io_rows * pair);
    let t = Instant::now();
    read_spill_block(&file, direct.as_ref(), pair, (0, io_rows)).ok()?;
    let read_s = t.elapsed().as_secs_f64().max(1e-9);
    let bytes = (io_rows * pair * 8) as f64;
    Some((bytes / write_s, bytes / read_s))
}

/// What a calibration depends on: the problem size, the budget that caps its
/// buffers, and the directory whose I/O it measures.
type CalibrationKey = (usize, usize, usize, usize, std::path::PathBuf);

static CALIBRATIONS: std::sync::Mutex<Vec<(CalibrationKey, Calibration)>> =
    std::sync::Mutex::new(Vec::new());

/// [`Calibration`] for `key`, measured once per process: every later
/// out-of-memory construction of the same size (the next geometry step, the next
/// SCF) reuses it and does no integral or file I/O. The cache is per process, so
/// each MPI rank measures once; ranks do not exchange the result.
fn cached_calibration(
    key: CalibrationKey,
    measure: impl FnOnce() -> Result<Calibration, FerricError>,
) -> Result<Calibration, FerricError> {
    let hit =
        |t: &Vec<(CalibrationKey, Calibration)>| t.iter().find(|(k, _)| *k == key).map(|(_, c)| *c);
    if let Some(c) = hit(&CALIBRATIONS.lock().unwrap_or_else(|e| e.into_inner())) {
        return Ok(c);
    }
    let c = measure()?;
    let mut table = CALIBRATIONS.lock().unwrap_or_else(|e| e.into_inner());
    if hit(&table).is_none() {
        table.push((key, c));
    }
    Ok(c)
}

impl ThreeIndexSource {
    /// Build the raw tensor for the SCF RI-J/K path under the effective
    /// [`jk_storage`] policy. `mol` is needed only to re-prepare the bases for
    /// the recompute backend, which owns them.
    pub fn build_for_jk(
        op: Operator,
        mol: &ferric_core::mol::Molecule,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        budget_bytes: usize,
        (band_p0, band_p1): (usize, usize),
    ) -> Result<Self, FerricError> {
        let policy = jk_storage();
        let nao = obs.nbasis();
        let band = band_p1 - band_p0;
        let (tier, cal) = decide(policy, (nao, band, budget_bytes), || {
            let dir = super::spill_dir();
            let key = (nao, dfbs.nbasis(), band_p0, budget_bytes, dir.clone());
            cached_calibration(key, || {
                let free = super::free_bytes_at(&dir);
                calibrate(op, obs, dfbs, (band_p0, band_p1), (budget_bytes, free))
            })
        })?;
        if super::ooc_trace() {
            let c = cal.map_or(String::new(), |c| {
                format!(
                    " [calibration: {:.3e} s/row, io {}]",
                    c.build_s_per_row,
                    c.io.map_or("none".into(), |(w, r)| format!(
                        "write {:.0} MB/s read {:.0} MB/s",
                        w / 1e6,
                        r / 1e6
                    ))
                )
            });
            eprintln!(
                "[OOC build] jk_storage={policy} nao={nao} band=[{band_p0},{band_p1}) \
                 budget={:.2}GB -> {}{c}",
                budget_bytes as f64 / 1e9,
                tier.name()
            );
        }
        if tier == Tier::Recompute {
            let dfbs_naux = dfbs.nbasis();
            let obs = Arc::new(PreparedBasis::new(mol, obs.basis_set())?);
            let dfbs = Arc::new(PreparedBasis::new(mol, dfbs.basis_set())?);
            debug_assert_eq!(obs.nbasis(), nao, "re-prepared orbital basis changed size");
            debug_assert_eq!(
                dfbs.nbasis(),
                dfbs_naux,
                "re-prepared auxiliary basis changed size"
            );
            return Self::build_recompute_band(
                op,
                obs,
                dfbs,
                budget_bytes,
                (band_p0, band_p1),
                None,
            );
        }
        Self::build_band_tier(op, obs, dfbs, budget_bytes, (band_p0, band_p1), None, tier)
    }

    /// The recompute backend over the band `[band_p0, band_p1)`, always (it
    /// never falls back to in-core, unlike [`Self::build_recomputing`]).
    ///
    /// `screen` is stored for the source's whole lifetime so a rebuilt block
    /// skips exactly the triples a one-shot build with the same screen skips.
    pub fn build_recompute_band(
        op: Operator,
        obs: Arc<PreparedBasis>,
        dfbs: Arc<PreparedBasis>,
        budget_bytes: usize,
        (band_p0, band_p1): (usize, usize),
        screen: Option<(Arc<crate::qqr3::QqrBounds3>, f64)>,
    ) -> Result<Self, FerricError> {
        let naux = dfbs.nbasis();
        let nao = obs.nbasis();
        assert!(
            band_p0 <= band_p1 && band_p1 <= naux,
            "invalid aux band [{band_p0},{band_p1}) for naux={naux}"
        );
        let band = band_p1 - band_p0;
        let block_naux = spill_block_naux_for(budget_bytes, nao).min(band.max(1));
        let block_bytes = block_naux
            .saturating_mul(nao)
            .saturating_mul(nao)
            .saturating_mul(8);
        let _charge = charge_three_index_soft("DF 3-index (P|mn) recompute block", block_bytes);
        let scratch = Array3::<f64>::zeros((block_naux, nao, nao));
        Ok(Self {
            naux,
            nao,
            block_naux,
            band_p0,
            band_p1,
            backend: Backend::Recompute {
                op,
                obs,
                dfbs,
                scratch,
                screen,
            },
            _charge,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trips_and_rejects_unknown() {
        for v in [
            JkStorage::Auto,
            JkStorage::Memory,
            JkStorage::Disk,
            JkStorage::Direct,
        ] {
            assert_eq!(JkStorage::parse(v.as_str()).unwrap(), v);
            assert_eq!(JkStorage::from_code(v.code()), Some(v));
        }
        assert_eq!(JkStorage::parse(" Direct ").unwrap(), JkStorage::Direct);
        let e = JkStorage::parse("ram").unwrap_err();
        assert!(e.contains("\"ram\"") && e.contains("direct"), "{e}");
    }

    fn cal(build_s_per_row: f64, io: Option<(f64, f64)>) -> Calibration {
        Calibration {
            build_s_per_row,
            io,
        }
    }

    #[test]
    fn cost_model_follows_the_measured_rates() {
        // 2256 rows, 2.17 GB packed (alkane_20 / def2-SVP).
        let (band, bytes) = (2256, 2_170_000_000);
        // Whole-band build 3.5 s; 515 MB/s read, 80 MB/s write: recompute.
        assert!(recompute_is_cheaper(
            &cal(3.5 / band as f64, Some((80e6, 515e6))),
            band,
            bytes
        ));
        // A 3 GB/s NVMe reads the tensor in 0.7 s: spill wins.
        assert!(!recompute_is_cheaper(
            &cal(3.5 / band as f64, Some((2e9, 3e9))),
            band,
            bytes
        ));
        // Slow integrals (60 s) against the SATA disk: spill wins.
        assert!(!recompute_is_cheaper(
            &cal(60.0 / band as f64, Some((80e6, 515e6))),
            band,
            bytes
        ));
        // No usable scratch disk: recompute.
        assert!(recompute_is_cheaper(
            &cal(60.0 / band as f64, None),
            band,
            bytes
        ));
    }

    #[test]
    fn decide_honours_the_policy_and_calibrates_only_when_needed() {
        let nao = 100;
        let band = 400;
        let unpacked = band * nao * nao * 8;
        let packed = band * packed_pair_len(nao) * 8;
        let never = || -> Result<Calibration, FerricError> { panic!("must not calibrate") };
        let sz = |budget| (nao, band, budget);
        assert_eq!(
            decide(JkStorage::Auto, sz(unpacked), never).unwrap().0,
            Tier::InCore
        );
        assert_eq!(
            decide(JkStorage::Auto, sz((unpacked + packed) / 2), never)
                .unwrap()
                .0,
            Tier::InCorePacked
        );
        assert_eq!(
            decide(JkStorage::Memory, sz((unpacked + packed) / 2), never)
                .unwrap()
                .0,
            Tier::InCorePacked
        );
        let e = decide(JkStorage::Memory, sz(packed / 2), never).unwrap_err();
        assert!(format!("{e}").contains("jk_storage = \"memory\""), "{e}");
        assert_eq!(
            decide(JkStorage::Disk, sz(unpacked * 10), never).unwrap().0,
            Tier::Spill
        );
        assert_eq!(
            decide(JkStorage::Direct, sz(unpacked * 10), never)
                .unwrap()
                .0,
            Tier::Recompute
        );
        // Over budget in auto: calibration decides.
        let fast_disk = || Ok(cal(1.0, Some((1e12, 1e12))));
        let (t, c) = decide(JkStorage::Auto, sz(packed / 2), fast_disk).unwrap();
        assert_eq!(t, Tier::Spill);
        assert!(c.is_some());
        let no_disk = || Ok(cal(1.0, None));
        assert_eq!(
            decide(JkStorage::Auto, sz(packed / 2), no_disk).unwrap().0,
            Tier::Recompute
        );
    }

    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    fn dimer() -> Molecule {
        // Two waters 12 A apart: the QQR-3 screen has real work to do.
        Molecule::parse_xyz(
            "6\ndimer\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n\
             O 12 0 0\nH 12 0 0.96\nH 12.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap()
    }

    /// A recompute source built with a screen rebuilds exactly the blocks a
    /// one-shot screened build holds, pass after pass, and the screen is not a
    /// no-op on this geometry (negative control: the unscreened tensor differs).
    #[test]
    fn recompute_with_a_screen_is_bit_identical_to_the_screened_one_shot_build() {
        let mol = dimer();
        let op = Operator::coulomb();
        let obs = Arc::new(PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap());
        let dfbs = Arc::new(
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap(),
        );
        let bounds = Arc::new(crate::qqr3::QqrBounds3::new(op, &mol, &obs, &dfbs).unwrap());
        let thresh = 1e-6;
        let naux = dfbs.nbasis();
        let want = crate::threeindex::eri3_block_screened(
            op,
            &obs,
            &dfbs,
            0,
            naux,
            Some((&bounds, thresh)),
        )
        .unwrap();
        let unscreened = crate::threeindex::eri3_block(op, &obs, &dfbs, 0, naux).unwrap();
        assert!(want != unscreened, "the screen must drop something here");

        let nao = obs.nbasis();
        let tiny = nao * nao * 8 * 9;
        let mut src = ThreeIndexSource::build_recompute_band(
            op,
            obs.clone(),
            dfbs.clone(),
            tiny,
            (0, naux),
            Some((bounds, thresh)),
        )
        .unwrap();
        assert!(src.is_recompute_for_test() && src.n_blocks() > 1);
        for _pass in 0..2 {
            let mut got = ndarray::Array3::<f64>::zeros((naux, nao, nao));
            src.for_each_block(|blk| {
                let b = blk.data.shape()[0];
                got.slice_mut(ndarray::s![blk.p0..blk.p0 + b, .., ..])
                    .assign(&blk.data);
                Ok(())
            })
            .unwrap();
            assert!(
                got == want,
                "recomputed screened blocks != one-shot screened tensor"
            );
        }
    }

    /// The calibration measures something finite and positive, and sees the
    /// spill directory.
    #[test]
    fn calibration_measures_finite_positive_rates() {
        let mol = dimer();
        let op = Operator::coulomb();
        let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let c = calibrate(op, &obs, &dfbs, (0, dfbs.nbasis()), (usize::MAX, None)).unwrap();
        assert!(c.build_s_per_row.is_finite() && c.build_s_per_row > 0.0);
        let (w, r) = c.io.expect("the temp dir is writable in tests");
        assert!(w.is_finite() && w > 0.0 && r.is_finite() && r > 0.0);
    }

    /// `auto` rules the spill out when the spill directory cannot hold the
    /// packed tensor plus headroom: calibration reports no IO, so recompute
    /// wins whatever the build time is.
    #[test]
    fn a_full_spill_directory_rules_out_spill() {
        let mol = dimer();
        let op = Operator::coulomb();
        let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let band = (0, dfbs.nbasis());
        let roomy = calibrate(op, &obs, &dfbs, band, (usize::MAX, Some(u64::MAX / 2))).unwrap();
        assert!(roomy.io.is_some());
        let full = calibrate(op, &obs, &dfbs, band, (usize::MAX, Some(1_000_000))).unwrap();
        assert!(full.io.is_none(), "1 MB free cannot hold the packed tensor");
        let packed = band.1 * packed_pair_len(obs.nbasis()) * 8;
        assert!(recompute_is_cheaper(&full, band.1, packed));
    }

    /// Calibration rows and its IO buffer obey the budget (the allocations are
    /// not charged to the pool).
    #[test]
    fn calibration_respects_a_tiny_budget() {
        let mol = dimer();
        let op = Operator::coulomb();
        let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let nao = obs.nbasis();
        let c = calibrate(
            op,
            &obs,
            &dfbs,
            (0, dfbs.nbasis()),
            (nao * nao * 8 * 2, None),
        )
        .unwrap();
        assert!(c.build_s_per_row > 0.0);
    }

    /// The second call for the same key reuses the first measurement and runs
    /// no measurement closure (no integrals, no file I/O); a different key
    /// measures again.
    #[test]
    fn calibration_is_measured_once_per_key() {
        use std::cell::Cell;
        let dir = std::path::PathBuf::from("/cache-test-dir");
        let first = Calibration {
            build_s_per_row: 1.0,
            io: Some((2.0, 3.0)),
        };
        let runs = Cell::new(0);
        let key = (7001, 7002, 0, 7003, dir.clone());
        let a = cached_calibration(key.clone(), || {
            runs.set(runs.get() + 1);
            Ok(first)
        })
        .unwrap();
        let b = cached_calibration(key, || {
            runs.set(runs.get() + 1);
            Ok(Calibration {
                build_s_per_row: 9.0,
                io: None,
            })
        })
        .unwrap();
        assert_eq!(runs.get(), 1, "the second call re-measured");
        assert_eq!(a.build_s_per_row, b.build_s_per_row);
        assert_eq!(b.io, Some((2.0, 3.0)));
        cached_calibration((7001, 7002, 0, 7004, dir), || {
            runs.set(runs.get() + 1);
            Ok(first)
        })
        .unwrap();
        assert_eq!(runs.get(), 2, "a different budget must measure again");
    }
}
