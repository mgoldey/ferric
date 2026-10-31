//! Through the real binary: a budget that holds the raw 3-index tensor but not
//! also the XC working set must run, and say that XC is not covered. Before XC
//! was committed last, this budget exited 1 at the raw tensor's charge. Derived
//! from the tensor's bytes so it stays in that band if basis data changes.
//!
//! Water on the default grid at one thread has `L` = 11.05 `T`, so 11.5 `T` is
//! inside the old `[L, L + T)` band, and XC cannot fit beside raw's `T`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn a_budget_that_holds_raw_but_not_xc_runs_and_says_so() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mol = Molecule::load_xyz(root.join("testdata/molecules/water.xyz").to_str().unwrap());
    let mol = mol.unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = basis::bundled("def2-universal-jkfit").unwrap();
    let aux = PreparedBasis::new(&mol, &aux).unwrap();
    let tensor = aux.nbasis() * obs.nbasis() * obs.nbasis() * 8;
    let budget_gb = (23 * tensor / 2) as f64 / (1u64 << 30) as f64;

    let dir = root.join("target");
    std::fs::create_dir_all(&dir).unwrap();
    let toml = dir.join("ks_xc_uncharged.toml");
    std::fs::write(
        &toml,
        format!(
            "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n\
             [basis]\nname = \"cc-pvdz\"\n\
             [method]\nkind = \"ksdft\"\n\
             [dft]\nfunctional = \"PBE\"\n\
             [memory]\nbudget_gb = {budget_gb:.9}\n"
        ),
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_ferric-cli"))
        .arg(&toml)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(
        stderr.contains("KS-DFT grid AO blocks recomputed"),
        "{stderr}"
    );
    assert!(stderr.contains("runs uncharged"), "{stderr}");
    assert!(stderr.contains("can exceed its budget"), "{stderr}");
}
