//! Point-group-blocked closed-shell Kohn-Sham (RKS) (issue #294).
//!
//! The blocked RKS is the blocked RHF with `RhfConfig::xc` set and NO Fock or
//! density symmetrisation. That is exact because the KS grid is built from the
//! reoriented molecule: every Lebedev rule is an Oh orbit placed at the atoms
//! without rotation, and the canonical-frame D2h operations `diag(±1, ±1, ±1)`
//! lie in Oh, so the grid (points and Becke weights) maps onto itself under
//! every detected operation and V_xc of a symmetric density commutes with them
//! to roundoff.
//!
//! Anchors, written before any measurement:
//! (a) C1 trivial limit: `symmetry = true` on a C1 molecule is BIT-identical to
//!     `symmetry = false` under KS;
//! (b) blocked RKS == unblocked RKS in the SAME frame to 1e-10 Ha, and the
//!     converged KS Fock (V_xc included) and density commute with every
//!     operation to 1e-10, under both RI-J (the KS default) and exact J;
//! (c) a partially filled degenerate shell is refused unless `irrep_nelec`
//!     lists one of its irreps, and so is a grid too coarse for that check;
//! (d) stretched H2 reaches sigma_g^2 with symmetry and misses it without
//!     (negative control).
//!
//! The rotated-input case DOCUMENTS that the grid is not rotation invariant:
//! the blocked energy (symmetry frame) differs from the unblocked input-frame
//! energy at the grid-error level, so it is bounded loosely, not to 1e-9.
//! The PySCF comparison is `validation_scf_symmetry.rs` (ignored per-commit).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::symmetry::{detect, AoRepresentation, DEFAULT_SYMMETRY_TOL};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::symmetric::{solve_rhf_symmetric, SymmetricRhf};
use ferric_scf::ScfResult;
use ndarray::Array2;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn load(name: &str) -> Molecule {
    Molecule::load_xyz(
        root()
            .join("testdata/molecules")
            .join(name)
            .to_str()
            .unwrap(),
    )
    .unwrap()
}

fn run_mol(mol: &Molecule, basis_name: &str, cfg: &RhfConfig) -> Result<ScfResult, String> {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    solve_rhf(&ParallelContext::default(), mol, &prep, op, &bounds, cfg).map_err(|e| e.to_string())
}

fn run_sym(mol: &Molecule, basis_name: &str, cfg: &RhfConfig) -> Result<SymmetricRhf, String> {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    solve_rhf_symmetric(&ParallelContext::default(), mol, &prep, op, &bounds, cfg)
        .map_err(|e| e.to_string())
}

/// PBE with the KS defaults (75x110 grid, RI-J with def2-universal-jkfit).
/// `energy_conv` is only a sanity bound under RI-J; the density gate decides.
fn ks(symmetry: bool) -> RhfConfig {
    RhfConfig {
        symmetry,
        xc: Some("PBE".into()),
        density_conv: 1e-9,
        energy_conv: 1e-8,
        max_iter: 200,
        ..RhfConfig::default()
    }
}

fn exact_j(mut c: RhfConfig) -> RhfConfig {
    c.df_j_aux = Some(String::new());
    c
}

fn nocc(mol: &Molecule) -> usize {
    (mol.nelec() / 2) as usize
}

fn occupation(r: &SymmetricRhf, nocc: usize) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for l in &r.orbital_irreps()[..nocc] {
        *out.entry(l.clone()).or_insert(0) += 1;
    }
    out
}

/// `max |R X R^T - X|` over the operations, for an AO matrix X.
fn max_noncommutation(rep: &AoRepresentation, nops: usize, x: &Array2<f64>) -> f64 {
    (0..nops)
        .map(|k| {
            let r = rep.dense(k);
            (&r.dot(x).dot(&r.t()) - x)
                .iter()
                .fold(0.0f64, |m, v| m.max(v.abs()))
        })
        .fold(0.0, f64::max)
}

/// Blocked vs unblocked (same frame) energy difference and the largest
/// commutator of the converged KS Fock and density.
struct Anchor {
    de: f64,
    fcomm: f64,
    dcomm: f64,
}

fn anchor(mol: &Molecule, basis_name: &str, on: &RhfConfig, off: &RhfConfig) -> Anchor {
    let sym = run_sym(mol, basis_name, on).unwrap();
    let blocked = sym.scf_in_frame();
    assert!(blocked.converged, "blocked did not converge");
    let plain = run_mol(sym.molecule(), basis_name, off).unwrap();
    assert!(plain.converged, "unblocked did not converge");
    for (a, b) in blocked
        .eps_alpha
        .iter()
        .zip(&plain.eps_alpha)
        .take(nocc(mol))
    {
        assert!((a - b).abs() < 1e-7, "occupied eps {a} vs {b}");
    }
    let det = detect(mol, DEFAULT_SYMMETRY_TOL).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let rep = AoRepresentation::new(&det, &det.molecule, &bs).unwrap();
    let nops = det.ops.len();
    Anchor {
        de: (blocked.energy - plain.energy).abs(),
        fcomm: max_noncommutation(&rep, nops, &blocked.fock_alpha),
        dcomm: max_noncommutation(&rep, nops, &blocked.density_total),
    }
}

// ---------------------------------------------------------------- anchor (a)

#[test]
fn ks_c1_molecule_is_bit_identical_with_symmetry_on() {
    let mut mol = load("nh3.xyz");
    mol.atoms[1].x += 0.11;
    mol.atoms[1].y -= 0.07;
    mol.atoms[1].zpos += 0.05;
    let off = run_mol(&mol, "sto-3g", &ks(false)).unwrap();
    let sym = run_sym(&mol, "sto-3g", &ks(true)).unwrap();
    assert_eq!(sym.group(), "C1");
    let on = sym.scf_in_frame();
    assert_eq!(off.energy.to_bits(), on.energy.to_bits());
    assert_eq!(off.mos_alpha, on.mos_alpha);
    assert_eq!(off.eps_alpha, on.eps_alpha);
    assert_eq!(off.iterations, on.iterations);
}

// ---------------------------------------------------------------- anchor (b)

#[test]
fn ks_blocked_matches_unblocked_and_ks_fock_commutes() {
    for (xyz, group) in [
        ("water.xyz", "C2v"),
        ("nh3.xyz", "Cs"),
        ("c2h4.xyz", "D2h"),
        ("benzene.xyz", "D2h"),
    ] {
        let mol = load(xyz);
        assert_eq!(
            detect(&mol, DEFAULT_SYMMETRY_TOL).unwrap().group.name(),
            group
        );
        let a = anchor(&mol, "sto-3g", &ks(true), &ks(false));
        eprintln!(
            "{xyz} PBE/STO-3G RI-J: |dE| = {:.1e}, max [R,F_KS] = {:.1e}, max [R,D] = {:.1e}",
            a.de, a.fcomm, a.dcomm
        );
        assert!(a.de < 1e-10, "{xyz}: blocked vs unblocked {:e}", a.de);
        assert!(a.fcomm < 1e-10, "{xyz}: [R,F_KS] = {:e}", a.fcomm);
        assert!(a.dcomm < 1e-10, "{xyz}: [R,D] = {:e}", a.dcomm);
    }
}

/// The blocking is independent of how J is built: RI-J (the KS default) and
/// exact four-centre J both satisfy the anchor, and they are different
/// energies (the fitting error), so the test can tell them apart.
#[test]
fn ks_blocking_holds_for_exact_j_and_ri_j() {
    let mol = load("water.xyz");
    let ri = anchor(&mol, "6-31g", &ks(true), &ks(false));
    let ex = anchor(&mol, "6-31g", &exact_j(ks(true)), &exact_j(ks(false)));
    let e_ri = run_sym(&mol, "6-31g", &ks(true)).unwrap().energy();
    let e_ex = run_sym(&mol, "6-31g", &exact_j(ks(true))).unwrap().energy();
    eprintln!(
        "water PBE/6-31G: RI-J |dE| {:.1e} [R,F] {:.1e}; exact J |dE| {:.1e} [R,F] {:.1e}; \
         E(RI-J) - E(exact J) = {:.3e}",
        ri.de,
        ri.fcomm,
        ex.de,
        ex.fcomm,
        e_ri - e_ex
    );
    for (name, a) in [("RI-J", &ri), ("exact J", &ex)] {
        assert!(a.de < 1e-10, "{name}: {:e}", a.de);
        assert!(a.fcomm < 1e-10, "{name}: [R,F] {:e}", a.fcomm);
    }
    assert!(
        (e_ri - e_ex).abs() > 1e-7,
        "RI-J and exact J did not differ"
    );
}

/// A rotated, translated input: the anchor holds in the symmetry frame, but
/// the blocked energy differs from the unblocked INPUT-frame energy because
/// the Becke-Lebedev grid is not rotation invariant. Documented, bounded
/// loosely; the size printed here is the grid-orientation error.
#[test]
fn ks_rotated_input_documents_the_grid_orientation_change() {
    let (ax, ang) = ([1.0f64, 2.0, 3.0], 0.7f64);
    let n = (ax.iter().map(|v| v * v).sum::<f64>()).sqrt();
    let k = [ax[0] / n, ax[1] / n, ax[2] / n];
    let (c, s) = (ang.cos(), ang.sin());
    let mut mol = load("water.xyz");
    for a in &mut mol.atoms {
        let v = [a.x, a.y, a.zpos];
        let kv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
        let cr = [
            k[1] * v[2] - k[2] * v[1],
            k[2] * v[0] - k[0] * v[2],
            k[0] * v[1] - k[1] * v[0],
        ];
        let r: Vec<f64> = (0..3)
            .map(|i| v[i] * c + cr[i] * s + k[i] * kv * (1.0 - c))
            .collect();
        a.x = r[0] + 1.3;
        a.y = r[1] - 0.8;
        a.zpos = r[2] + 2.1;
    }
    let a = anchor(&mol, "sto-3g", &ks(true), &ks(false));
    assert!(a.de < 1e-10, "same-frame anchor {:e}", a.de);
    assert!(a.fcomm < 1e-10, "same-frame [R,F_KS] {:e}", a.fcomm);
    let sym = run_sym(&mol, "sto-3g", &ks(true)).unwrap();
    let input_frame = run_mol(&mol, "sto-3g", &ks(false)).unwrap();
    let d = (sym.energy() - input_frame.energy).abs();
    eprintln!(
        "water PBE/STO-3G rotated input: |E_blocked(symmetry frame) - E_unblocked(input frame)| \
         = {d:.2e} Ha (grid orientation), symmetrization shift {:.1e} Bohr",
        sym.max_symmetrization_shift()
    );
    assert!(d < 1e-3, "grid-orientation change {d:e} is not grid-sized");
}

// ---------------------------------------------------------------- anchor (c)

/// Singlet O2 puts one electron pair into the pi_g pair (B2g + B3g in D2h):
/// a partially filled degenerate shell. Under KS the grid may split such a
/// shell (C3/C6 partners by up to ~6e-5 Ha), so aufbau would choose by grid
/// noise; it is refused unless `irrep_nelec` decides.
#[test]
fn ks_partially_filled_degenerate_shell_needs_irrep_nelec() {
    let mol = load("o2.xyz");
    let err = run_sym(&mol, "sto-3g", &ks(true)).unwrap_err();
    assert!(err.contains("near-degenerate"), "{err}");
    assert!(err.contains("irrep_nelec"), "{err}");
    // A PARTIAL request that leaves both pi_g irreps unlisted is still
    // refused: Ag takes its three pairs by request, and the remaining pair
    // would again be placed in pi_g by aufbau over the grid split.
    let mut partial = ks(true);
    partial.irrep_nelec = Some(vec![("Ag".into(), 6)]);
    let err = run_sym(&mol, "sto-3g", &partial).unwrap_err();
    assert!(
        err.contains("near-degenerate"),
        "partial irrep_nelec: {err}"
    );
    // The same shell under HF is allowed (documented cluster ordering).
    let mut hf = ks(true);
    hf.xc = None;
    assert!(run_sym(&mol, "sto-3g", &hf).is_ok());
    // With irrep_nelec the KS solve runs and honours the request.
    let probe = run_sym(&mol, "sto-3g", &hf).unwrap();
    let mut req: Vec<(String, usize)> = occupation(&probe, nocc(&mol))
        .into_iter()
        .map(|(l, n)| (l, 2 * n))
        .collect();
    let all: Vec<String> = probe
        .orbital_irreps()
        .iter()
        .filter(|l| *l != "-")
        .cloned()
        .collect();
    for l in all {
        if !req.iter().any(|(k, _)| *k == l) {
            req.push((l, 0));
        }
    }
    let mut c = ks(true);
    c.irrep_nelec = Some(req.clone());
    let r = run_sym(&mol, "sto-3g", &c).unwrap();
    let got: BTreeMap<String, usize> = occupation(&r, nocc(&mol))
        .into_iter()
        .map(|(l, n)| (l, 2 * n))
        .collect();
    let want: BTreeMap<String, usize> = req.into_iter().filter(|(_, n)| *n > 0).collect();
    assert_eq!(got, want);
}

// ---------------------------------------------------------------- anchor (d)

fn h2(r_angstrom: f64) -> Molecule {
    let h = r_angstrom / 2.0;
    Molecule::parse_xyz(&format!("2\nh2\nH 0 0 {h}\nH 0 0 -{h}\n"), 0, 1).unwrap()
}

fn reference(name: &str) -> serde_json::Value {
    let p = root()
        .join("testdata/reference/validation/scf_symmetry")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// sigma_g^2 is reached with `irrep_nelec` (the bare blocked solve refuses the
/// degenerate sigma_g/sigma_u shell, anchor (c)); the unblocked KS SCF misses
/// it at 10 000 Å. Reference: PySCF `dft.RKS` PBE with symmetry on ferric's
/// grid and RI-J (`gen_scf_symmetry.py`).
#[test]
fn ks_stretched_h2_needs_symmetry_to_reach_sigma_g_squared() {
    for (r, file) in [
        (50.0, "h2_r50_cc-pvdz.json"),
        (10000.0, "h2_r10000_cc-pvdz.json"),
    ] {
        let mol = h2(r);
        let want = reference(file)["rks_pbe"]["sigma_g2"]["energy"]
            .as_f64()
            .unwrap();
        let bare = run_sym(&mol, "cc-pvdz", &ks(true)).unwrap_err();
        assert!(bare.contains("near-degenerate"), "R={r}: {bare}");
        let mut on = ks(true);
        on.irrep_nelec = Some(vec![("Ag".into(), 2)]);
        let blocked = run_sym(&mol, "cc-pvdz", &on).unwrap();
        assert_eq!(blocked.orbital_irreps()[0], "Ag");
        let plain = run_mol(&mol, "cc-pvdz", &ks(false));
        eprintln!(
            "H2 PBE R={r}: blocked {:.12} ref {want:.12} (diff {:.2e}); unblocked {:?}",
            blocked.energy(),
            blocked.energy() - want,
            plain.as_ref().map(|p| (p.energy, p.converged))
        );
        assert!(
            (blocked.energy() - want).abs() < 1e-9,
            "R={r}: {} vs {want}",
            blocked.energy()
        );
        // Non-convergence is Ok(converged = false), not an Err: an Err here
        // is a failure of the control itself, not a miss.
        let p = plain.unwrap();
        assert!(
            (p.energy - want).abs() > 1e-3,
            "R={r}: unblocked KS reached sigma_g^2 ({} vs {want}); the negative \
             control fails",
            p.energy
        );
    }
}

/// Below 110 angular points the grid splits degenerate partners by more than
/// the split-shell tolerance (benzene: 7e-4 Ha at 50, 7e-3 Ha at 26), so
/// blocked KS refuses such a grid; 110 (also pruned) is accepted.
#[test]
fn ks_coarse_grid_is_refused() {
    let mol = load("water.xyz");
    for (ang, prune, ok) in [
        (50, None, false),
        (26, None, false),
        (111, None, false),
        (110, Some(ferric_dft::prune::PruneScheme::NwchemLike), true),
    ] {
        let mut c = ks(true);
        c.dft_grid = Some(ferric_dft::grid::AtomicGridConfig {
            n_radial: 50,
            n_angular: ang,
            prune,
        });
        let r = run_sym(&mol, "sto-3g", &c);
        match (ok, r) {
            (true, r) => assert!(r.is_ok(), "{ang}: {:?}", r.err()),
            (false, Err(e)) => assert!(e.contains("angular points"), "{ang}: {e}"),
            (false, Ok(_)) => panic!("{ang} angular points accepted"),
        }
    }
}

/// Beyond PBE: a meta-GGA (tau term, auto level shift) and a range-separated
/// hybrid with VV10 (SR/LR fitted exchange, nonlocal grid) block exactly too.
#[test]
fn ks_blocking_holds_for_scan_and_wb97x_v() {
    let mol = load("water.xyz");
    for xc in ["SCAN", "wB97X-V"] {
        let (mut on, mut off) = (ks(true), ks(false));
        on.xc = Some(xc.into());
        off.xc = Some(xc.into());
        let a = anchor(&mol, "sto-3g", &on, &off);
        eprintln!(
            "water {xc}/STO-3G: |dE| = {:.1e}, max [R,F_KS] = {:.1e}",
            a.de, a.fcomm
        );
        assert!(a.de < 1e-10, "{xc}: {:e}", a.de);
        assert!(a.fcomm < 1e-10, "{xc}: [R,F] {:e}", a.fcomm);
    }
}
