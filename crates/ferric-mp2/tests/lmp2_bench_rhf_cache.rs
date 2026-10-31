//! The bench RHF cache (`common/bench_rhf_cache.rs`): a cached RHF must leave
//! the direct LMP2 driver's every non-timer output bitwise unchanged, and every
//! kind of unusable file must be refused.

#[allow(dead_code)]
#[path = "common/bench_rhf_cache.rs"]
mod bench_rhf_cache;

use bench_rhf_cache::{
    bench_scf_config, cache_key, decode, encode, fnv1a64, load_or_solve, read, rhf_bits_diff,
    solve_fresh, write,
};
use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::AmplitudeLmp2Config;
use ferric_mp2::lmp2_direct::{amplitude_lmp2_direct, DirectConfig};
use ferric_scf::result::{ScfExit, ScfResult, Spin};
use std::path::PathBuf;

const OBS: &str = "6-31g";
const AUX: &str = "cc-pvdz-ri";

struct Sys {
    xyz_text: String,
    mol: Molecule,
    obs: PreparedBasis,
    obs_bs: BasisSet,
    aux_bs: BasisSet,
    dfbs: PreparedBasis,
}

fn sys(xyz: &str) -> Sys {
    let path = format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    );
    let mol = Molecule::load_xyz(&path).unwrap();
    let obs_bs = basis::bundled(OBS).unwrap();
    let aux_bs = basis::bundled(AUX).unwrap();
    Sys {
        xyz_text: std::fs::read_to_string(&path).unwrap(),
        obs: PreparedBasis::new(&mol, &obs_bs).unwrap(),
        dfbs: PreparedBasis::new(&mol, &aux_bs).unwrap(),
        mol,
        obs_bs,
        aux_bs,
    }
}

impl Sys {
    fn key(&self) -> String {
        cache_key(
            &self.xyz_text,
            (OBS, &self.obs_bs),
            (AUX, &self.aux_bs),
            &bench_scf_config(),
        )
    }
}

/// A fresh, empty directory under cargo's per-target tmpdir.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("rhf_cache_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Every non-timer output of the driver at the bench's production maps, as bits.
fn lmp2_signature(
    s: &Sys,
    rhf: &ScfResult,
    nc: usize,
    op: Operator,
    cal: f64,
    eps: f64,
) -> Vec<(&'static str, u64)> {
    let cfg = AmplitudeLmp2Config {
        eps,
        frozen_core: nc,
        pair_gate_cal: Some(cal),
        eri3_budget_bytes: Some(2usize << 30),
        compute_reference: true,
        ..Default::default()
    };
    let dcfg = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        schwarz_skip: 1e-5,
        batch_merge: 4,
        scratch_budget_bytes: 1usize << 30,
        virt_schwarz_kappa: None,
        split_eri3_timers: false,
    };
    let (r, st) =
        amplitude_lmp2_direct(&s.mol, &s.obs, &s.obs_bs, &s.dfbs, op, rhf, &cfg, &dcfg).unwrap();
    vec![
        ("e_corr", r.e_corr.to_bits()),
        ("e_total", r.e_total.to_bits()),
        ("e_corr_canonical_ri", r.e_corr_canonical_ri.to_bits()),
        ("keep_fraction", r.keep_fraction.to_bits()),
        ("pair_fraction", r.pair_fraction.to_bits()),
        ("dom_mean", r.dom_mean.to_bits()),
        ("dom_max", r.dom_max as u64),
        ("cg_iterations", r.cg_iterations as u64),
        ("cg_relres", r.cg_relres.to_bits()),
        ("cg_true_relres", r.cg_true_relres.to_bits()),
        ("cg_converged", u64::from(r.cg_converged)),
        ("n_valence_virt", r.n_valence_virt as u64),
        ("n_hard_virt", r.n_hard_virt as u64),
        ("ragged_flops", r.ragged_flops_per_matvec),
        ("dense_flops", r.dense_flops_per_matvec),
        ("n_pairs_gated", r.n_pairs_gated as u64),
        ("n_pairs_unique", r.n_pairs_unique as u64),
        ("aux_dom_mean", r.aux_dom_mean.to_bits()),
        ("aux_dom_max", r.aux_dom_max as u64),
        ("strip_rows_mean", st.strip_rows_mean.to_bits()),
        ("strip_rows_max", st.strip_rows_max as u64),
        ("strip_cols_mean", st.strip_cols_mean.to_bits()),
        ("strip_cols_max", st.strip_cols_max as u64),
        ("n_eri3_shell_triples", st.n_eri3_shell_triples),
    ]
}

/// EXACTNESS ANCHOR. Catches: a cache that loses, reorders or re-lays-out
/// anything the driver reads. Also pins the driver's own run-to-run determinism,
/// without which a bitwise comparison of two runs could not mean anything.
#[test]
fn cached_rhf_gives_bitwise_equal_lmp2() {
    let dir = scratch("anchor");
    for nc in [4usize, 8] {
        let s = sys(&format!("alkane_{nc}.xyz"));
        let path = dir.join(format!("alkane_{nc}.rhfbits"));
        let key = s.key();
        let fresh = solve_fresh(&s.mol, &s.obs);
        let cold = load_or_solve(&path, &key, s.obs.nbasis(), || fresh.clone());
        let warm = load_or_solve(&path, &key, s.obs.nbasis(), || panic!("must not solve"));
        assert_eq!(rhf_bits_diff(&cold, &fresh), Vec::<String>::new());
        assert_eq!(rhf_bits_diff(&warm, &fresh), Vec::<String>::new());
        for (op, cal) in [(Operator::coulomb(), 0.7), (Operator::erfc(1.0), 0.02)] {
            for eps in [1e-3, 1e-4] {
                let a = lmp2_signature(&s, &fresh, nc, op, cal, eps);
                let b = lmp2_signature(&s, &warm, nc, op, cal, eps);
                assert_eq!(a, b, "C{nc} cal={cal} eps={eps}");
                println!("C{nc} cal={cal} eps={eps}: {} fields identical", a.len());
            }
        }
    }
}

const CHILD_DIR: &str = "FERRIC_RHF_CACHE_TEST_CHILD_DIR";
const CHILD_XYZ: &str = "alkane_4.xyz";

fn child_signature(s: &Sys, rhf: &ScfResult) -> String {
    let sig = lmp2_signature(s, rhf, 4, Operator::erfc(1.0), 0.02, 1e-3);
    format!("SIGNATURE {sig:?}")
}

/// Run by `warm_read_in_a_new_process_is_bitwise`, in a process of its own.
#[test]
#[ignore = "child process of warm_read_in_a_new_process_is_bitwise"]
fn child_reads_the_cache() {
    let dir = PathBuf::from(std::env::var(CHILD_DIR).unwrap());
    let s = sys(CHILD_XYZ);
    let rhf = load_or_solve(&dir.join("c.rhfbits"), &s.key(), s.obs.nbasis(), || {
        panic!("the child must read the cache, not solve")
    });
    println!("{}", child_signature(&s, &rhf));
}

/// The real use: one process writes, another reads. Catches: a key that is not
/// stable across processes (a `HashMap`'s `Debug` order changes per process, so
/// any test computing the key once per process cannot see this) and anything
/// else in the file or the driver that depends on the process.
#[test]
fn warm_read_in_a_new_process_is_bitwise() {
    let dir = scratch("two_processes");
    let s = sys(CHILD_XYZ);
    let fresh = solve_fresh(&s.mol, &s.obs);
    let path = dir.join("c.rhfbits");
    let _ = load_or_solve(&path, &s.key(), s.obs.nbasis(), || fresh.clone());
    let want = child_signature(&s, &fresh);
    for _ in 0..2 {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["child_reads_the_cache", "--ignored", "--exact"])
            .args(["--nocapture", "--test-threads=1"])
            .env(CHILD_DIR, &dir)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "child failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        // libtest prints "test NAME ... " on the same line under --nocapture.
        assert!(
            stdout.lines().any(|l| l.ends_with(&want)),
            "child printed: {stdout}"
        );
    }
}

fn flip_mo_bit(rhf: &ScfResult, bit: u32) -> ScfResult {
    let mut m = rhf.clone();
    let x = &mut m.mos_alpha[[2, 1]];
    *x = f64::from_bits(x.to_bits() ^ (1u64 << bit));
    m
}

/// MUTANT of the anchor: an orbital bit flipped in a file whose checksum is
/// valid, so only the comparisons stand between it and a result. The low bit
/// must show in the RHF comparison, a higher bit in the LMP2 signature too.
#[test]
fn flipped_orbital_bit_fails_the_anchor() {
    let dir = scratch("mutant");
    let s = sys("alkane_4.xyz");
    let key = s.key();
    let fresh = solve_fresh(&s.mol, &s.obs);
    let path = dir.join("m.rhfbits");

    write(&path, &key, &flip_mo_bit(&fresh, 0)).unwrap();
    let low = read(&path, &key, s.obs.nbasis()).unwrap();
    assert!(rhf_bits_diff(&low, &fresh)
        .iter()
        .any(|d| d.starts_with("mos_alpha: 1 elements")));

    write(&path, &key, &flip_mo_bit(&fresh, 40)).unwrap();
    let high = read(&path, &key, s.obs.nbasis()).unwrap();
    let args = (4, Operator::coulomb(), 0.7, 1e-3);
    let good = lmp2_signature(&s, &fresh, args.0, args.1, args.2, args.3);
    let bad = lmp2_signature(&s, &high, args.0, args.1, args.2, args.3);
    assert_ne!(good[0], bad[0], "e_corr must move");
}

/// Catches: a reader that skips the checksum, at the start, middle and end of
/// the file (magic, orbitals, densities).
#[test]
fn a_flipped_bit_on_disk_is_corrupt() {
    let s = sys("water.xyz");
    let key = s.key();
    let good = encode(&key, &solve_fresh(&s.mol, &s.obs)).unwrap();
    assert!(decode(&good, &key, s.obs.nbasis()).is_ok());
    for at in [0, good.len() / 2, good.len() - 20, good.len() - 1] {
        let mut bad = good.clone();
        bad[at] ^= 1;
        let e = decode(&bad, &key, s.obs.nbasis()).unwrap_err();
        assert!(e.contains("checksum"), "offset {at}: {e}");
    }
}

/// Catches: a key that omits a component, or a reader that does not compare it.
#[test]
fn each_key_component_is_enforced() {
    let s = sys("water.xyz");
    let key = s.key();
    let bytes = encode(&key, &solve_fresh(&s.mol, &s.obs)).unwrap();
    let cfg = bench_scf_config();
    let (o, a) = ((OBS, &s.obs_bs), (AUX, &s.aux_bs));
    let other_bs = basis::bundled("sto-3g").unwrap();
    let mut other_cfg = bench_scf_config();
    other_cfg.energy_conv = 1e-9;
    let moved = s.xyz_text.replacen('.', ",", 1);
    for (what, wrong) in [
        ("xyz", cache_key(&moved, o, a, &cfg)),
        (
            "basis name",
            cache_key(&s.xyz_text, ("x", &s.obs_bs), a, &cfg),
        ),
        (
            "basis data",
            cache_key(&s.xyz_text, (OBS, &other_bs), a, &cfg),
        ),
        (
            "aux name",
            cache_key(&s.xyz_text, o, ("x", &s.aux_bs), &cfg),
        ),
        (
            "aux data",
            cache_key(&s.xyz_text, o, (AUX, &other_bs), &cfg),
        ),
        ("scf config", cache_key(&s.xyz_text, o, a, &other_cfg)),
    ] {
        assert_ne!(wrong, key, "{what}");
        let e = decode(&bytes, &wrong, s.obs.nbasis()).unwrap_err();
        assert!(e.contains("key mismatch"), "{what}: {e}");
    }
}

/// A mismatch must stop the bench, not fall back to solving.
#[test]
#[should_panic(expected = "key mismatch")]
fn load_or_solve_panics_on_a_wrong_key_instead_of_solving() {
    let dir = scratch("wrong_key");
    let s = sys("water.xyz");
    let path = dir.join("w.rhfbits");
    write(&path, &s.key(), &solve_fresh(&s.mol, &s.obs)).unwrap();
    let _ = load_or_solve(&path, "another key", s.obs.nbasis(), || {
        panic!("solve must not run")
    });
}

/// Catches: a checksum-valid file of the wrong shape or length being accepted.
#[test]
fn wrong_shape_and_bad_length_are_refused() {
    let s = sys("water.xyz");
    let key = s.key();
    let n = s.obs.nbasis();
    let good = encode(&key, &solve_fresh(&s.mol, &s.obs)).unwrap();
    for wrong_n in [n - 1, n + 1] {
        let e = decode(&good, &key, wrong_n).unwrap_err();
        assert!(e.contains("shape"), "{wrong_n}: {e}");
    }
    let rechecksummed = |mut body: Vec<u8>| {
        body.extend_from_slice(&fnv1a64(&body).to_le_bytes());
        body
    };
    let body = &good[..good.len() - 8];
    let short = rechecksummed(body[..body.len() - 8].to_vec());
    assert!(decode(&short, &key, n).unwrap_err().contains("truncated"));
    let mut long = body.to_vec();
    long.push(0);
    let e = decode(&rechecksummed(long), &key, n).unwrap_err();
    assert!(e.contains("trailing"), "{e}");
    assert!(decode(&good[..4], &key, n).unwrap_err().contains("corrupt"));
}

/// Catches: caching a failed SCF, which later runs would take for a real one,
/// or a matrix whose layout the reader would not restore.
#[test]
fn an_uncacheable_scf_is_refused() {
    let s = sys("water.xyz");
    let key = s.key();
    let fresh = solve_fresh(&s.mol, &s.obs);
    let mut unconverged = fresh.clone();
    unconverged.converged = false;
    assert!(encode(&key, &unconverged)
        .unwrap_err()
        .contains("unconverged"));
    let mut stalled = fresh.clone();
    stalled.exit = ScfExit::Stalled;
    assert!(encode(&key, &stalled).unwrap_err().contains("Stalled"));
    let mut column_major = fresh.clone();
    column_major.fock_alpha = fresh.fock_alpha.clone().reversed_axes();
    assert!(encode(&key, &column_major)
        .unwrap_err()
        .contains("row-major"));
    let mut open = fresh;
    open.spin = Spin::Unrestricted;
    assert!(encode(&key, &open).unwrap_err().contains("closed-shell"));
}
