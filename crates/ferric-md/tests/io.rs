//! XYZ trajectory and energy CSV writers round-trip the recorded frames.

mod common;

use common::morse_dimer;
use ferric_core::mol::Molecule;
use ferric_md::{run_md, EnergyCsvWriter, MdConfig, Thermostat, XyzTrajectoryWriter};
use std::path::PathBuf;

/// A per-process path under the runtime temp dir. Not `CARGO_TARGET_TMPDIR`:
/// that path is fixed at compile time, and CI runs test binaries on machines
/// where the build's target directory does not exist.
fn out_path(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ferric-md-io-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

#[test]
fn trajectory_and_energies_round_trip() {
    let (xyz_path, csv_path) = (out_path("traj.xyz"), out_path("energies.csv"));
    let (sys, mut p) = morse_dimer();
    let cfg = MdConfig {
        n_steps: 12,
        stride: 4,
        // Thermostatted, so the conserved column differs from e_tot.
        thermostat: Thermostat::Berendsen { tau_fs: 2.0 },
        temperature_k: Some(600.0),
        init_temperature_k: Some(300.0),
        ..MdConfig::default()
    };
    let res = {
        let mut xyz = XyzTrajectoryWriter::create(&xyz_path).unwrap();
        let mut csv = EnergyCsvWriter::create(&csv_path).unwrap();
        run_md(&sys, &mut p, &cfg, &mut [&mut xyz, &mut csv]).unwrap()
    };
    assert_eq!(res.frames.len(), 4);
    let last = res.frames.last().unwrap();
    assert!(
        (last.total - last.conserved).abs() > 1e-5,
        "Q should be non-trivial"
    );

    let text = std::fs::read_to_string(&xyz_path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 4 * 4);
    for (k, chunk) in lines.chunks(4).enumerate() {
        let f = &res.frames[k];
        assert!(
            chunk[1].starts_with(&format!("step={} time_fs=", f.step)),
            "{}",
            chunk[1]
        );
        assert!(chunk[1].contains(" e_pot=") && chunk[1].contains(" e_tot="));
        let mol = Molecule::parse_xyz(&chunk.join("\n"), 0, 1).unwrap();
        for (a, r) in mol.atoms.iter().zip(res.positions[k].rows()) {
            assert_eq!(a.symbol, "H");
            let d = [a.x - r[0], a.y - r[1], a.zpos - r[2]];
            assert!(d.iter().all(|x| x.abs() < 1e-10), "frame {k}: {d:?}");
        }
    }

    let csv = std::fs::read_to_string(&csv_path).unwrap();
    let mut rows = csv.lines();
    assert_eq!(
        rows.next(),
        Some("step,time_fs,e_pot,e_kin,e_tot,conserved,temperature_k")
    );
    let rows: Vec<Vec<f64>> = rows
        .map(|l| l.split(',').map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(rows.len(), res.frames.len());
    for (row, f) in rows.iter().zip(&res.frames) {
        let want = [
            f.step as f64,
            f.time_fs,
            f.potential,
            f.kinetic,
            f.total,
            f.conserved,
            f.temperature_k,
        ];
        for (got, want) in row.iter().zip(want) {
            assert!(
                (got - want).abs() < 1e-6 * want.abs().max(1e-6),
                "{got} vs {want}"
            );
        }
    }
}

#[test]
fn create_reports_unwritable_paths() {
    let bad = out_path("no-such-dir").join("x.xyz");
    assert!(XyzTrajectoryWriter::create(&bad).is_err());
    assert!(EnergyCsvWriter::create(&bad).is_err());
}
