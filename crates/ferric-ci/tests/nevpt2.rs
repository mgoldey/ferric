//! SC-NEVPT2 (`ferric_ci::nevpt2`) anchors.
//!
//! # Anchors (exactness first)
//!
//! * Empty external spaces: H2/6-31G CAS(2,4) has no core and no virtual
//!   orbitals, so all eight classes are exactly 0; H2/6-31G CAS(2,2) has no
//!   core, so the six classes with a core label are exactly 0 while `Sr` and
//!   `Srs` are not; LiH/STO-3G with one core orbital and CAS(2,5) has no
//!   virtuals, so the six classes with a virtual label are exactly 0 while
//!   `Si` and `Sij` are not.
//! * Single-determinant active space vs MP2 (prototype anchor 2): H2O/6-31G
//!   on RHF orbitals, the HOMO as the one active orbital (CI vector (1)).
//!   Expected, from an independent MO transform and the RHF orbital
//!   energies: `Sijrs` = the MP2 pair sum over core pairs, `Srsi` = twice the
//!   core-HOMO pairs, `Srs = −Σ_ab (ta|tb)² / (ε_a + ε_b − 2ε_t + (tt|tt))`
//!   (Dyall: `2F^I_tt + (tt|tt)` is `E_0`, `F^I_tt = ε_t − (tt|tt)`), and
//!   every other class zero (no active holes: zero norm; `Sr`, `Sir`: second
//!   order in RHF Fock elements). This sees the orbital-energy partition and
//!   the external denominators; it cannot see cumulant terms.
//! * Same inputs as the numpy prototype (anchor 1): ferric's canonical CASSCF
//!   orbitals and CI vector from `testdata/reference/nevpt2_refs.json`
//!   (`scripts/gen_nevpt2_refs.py`), evaluated by `sc_nevpt2` with
//!   `canonicalize = false` and by the prototype's Wick construction with no
//!   further canonicalization. H2O CAS(4,4), N2 CAS(6,6) at 1.0977 and
//!   2.2 Å (the last with large 3- and 4-RDM cumulants), 6-31G.
//! * PySCF (anchor 2): ferric's own CASSCF here, end to end, against
//!   `pyscf.mrpt.NEVPT` on PySCF's own CASSCF (on ferric's AO integrals).
//!
//! # Artifact vs physics
//!
//! A wrong sign or index in one class formula leaves the trivial and MP2
//! anchors intact for the classes they pin to zero; it moves that class
//! away from the prototype by far more than rounding. A wrong orbital-energy
//! partition moves `Sijrs` off MP2. A dropped `f3ca`/`f3ac` term changes
//! `Sr` or `Si` only on correlated references (prototype comparison).
//!
//! # Bars
//!
//! * Prototype: both sides contract the same tensors in different orders.
//!   Per class the error is at most `L · M · ε · B`: `L` labels, `M = n^6`
//!   products in the longest per-label sum (`Sr`/`Si` norm and K, n active
//!   orbitals), `B = 4` bounding one product (integrals below 1 Ha, RDM
//!   elements below 2, the factor 2 of the formulas); the label energy
//!   `N/(Δ + K/N)` adds no amplification while `Δ + K/N` stays above 0.1 Ha.
//!   `L` is bounded by `n_v n_c + n_v² + n_c²`.
//! * MP2: the RHF orbitals are canonical to the SCF convergence, the
//!   residual off-diagonal Fock elements enter at first order; 1e-10 Ha.
//! * PySCF: `|dE_class/dκ| (‖g_ferric‖ + ‖g_PySCF‖) / λ_min + 2 spread +
//!   1e-10` (the first-order orbital error of two independently converged
//!   CASSCF references, plus the spread under rotations inside degenerate
//!   external groups, which each code fixes differently; the ingredients
//!   come from the generator, see `scripts/prototypes/nevpt2_derivation.md`).

mod casscf_common;

use casscf_common::{f64s, molecule_text, num, setup_text, workspace_root, EPS};
use ferric_ci::nevpt2::{sc_nevpt2, Nevpt2Classes, Nevpt2Config, Nevpt2Reference};
use ferric_ci::{run_cas_ci, run_casscf, CasCiConfig, CasScfConfig};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::Array2;
use serde_json::Value;

const H2: &str = "2\n\nH 0 0 0\nH 0 0 0.74\n";
const LIH: &str = "2\n\nLi 0 0 0\nH 0 0 1.6\n";

fn refs() -> Value {
    let path = workspace_root().join("testdata/reference/nevpt2_refs.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

fn system(label: &str) -> Value {
    refs()["systems"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["system"] == label)
        .unwrap_or_else(|| panic!("no system {label}"))
        .clone()
}

fn class(v: &Value, name: &str) -> f64 {
    v[name]
        .as_f64()
        .unwrap_or_else(|| panic!("missing class {name}"))
}

fn casci_nevpt2(xyz: &str, basis: &str, n_core: usize, n_active: usize) -> Nevpt2Classes {
    let (mol, prep, rhf) = setup_text(xyz, basis);
    let n_el = (mol.nelec() as usize - 2 * n_core) / 2;
    let cfg = CasCiConfig {
        n_active,
        n_elec_active: (n_el, n_el),
        active_start: n_core,
        conv_thresh: 1e-10,
        ..Default::default()
    };
    let ci = run_cas_ci(&mol, &prep, &rhf, cfg).unwrap();
    let reference = Nevpt2Reference::from_casci(&ci, rhf.mos_r(), &cfg).unwrap();
    sc_nevpt2(&mol, &prep, &reference, &Nevpt2Config::default())
        .unwrap()
        .energies
}

/// Classes whose external space is empty are exactly zero; the others are
/// not (so the zeros are not an empty run).
#[test]
fn empty_external_spaces_give_exact_zeros() {
    let full = casci_nevpt2(H2, "6-31g", 0, 4);
    assert_eq!(full.to_array(), [0.0; 8], "H2 CAS(2,4): {full:?}");

    let no_core = casci_nevpt2(H2, "6-31g", 0, 2);
    eprintln!("H2/6-31G CAS(2,2): {no_core:?}");
    for (v, name) in [
        (no_core.si, "Si"),
        (no_core.sijrs, "Sijrs"),
        (no_core.sijr, "Sijr"),
        (no_core.srsi, "Srsi"),
        (no_core.sij, "Sij"),
        (no_core.sir, "Sir"),
    ] {
        assert_eq!(v, 0.0, "{name} with no core orbitals");
    }
    assert!(no_core.sr < -1e-6 && no_core.srs < -1e-6, "{no_core:?}");

    let no_virt = casci_nevpt2(LIH, "sto-3g", 1, 5);
    eprintln!("LiH/STO-3G core 1 CAS(2,5): {no_virt:?}");
    for (v, name) in [
        (no_virt.sr, "Sr"),
        (no_virt.sijrs, "Sijrs"),
        (no_virt.sijr, "Sijr"),
        (no_virt.srsi, "Srsi"),
        (no_virt.srs, "Srs"),
        (no_virt.sir, "Sir"),
    ] {
        assert_eq!(v, 0.0, "{name} with no virtual orbitals");
    }
    assert!(no_virt.si < -1e-9 && no_virt.sij < -1e-9, "{no_virt:?}");
}

/// `(C_0 C_1|C_2 C_3)` by four naive quarter transforms of the dense AO block.
fn mo_eri(eri: &[f64], nb: usize, c: &Array2<f64>, idx: [&[usize]; 4]) -> Vec<f64> {
    let mut cur = eri.to_vec();
    let mut dims = [nb; 4];
    // Transform axis 3, 2, 1, 0 in turn (row-major layout kept).
    for ax in (0..4).rev() {
        let w = idx[ax].len();
        let mut nd = dims;
        nd[ax] = w;
        let mut out = vec![0.0; nd.iter().product()];
        let st = |d: &[usize; 4], a: usize| d[a + 1..].iter().product::<usize>();
        let (so, si) = (st(&nd, ax), st(&dims, ax));
        let outer: usize = dims[..ax].iter().product();
        let inner = si;
        for o in 0..outer {
            for (k, &p) in idx[ax].iter().enumerate() {
                for mu in 0..nb {
                    let cm = c[(mu, p)];
                    let src = o * dims[ax] * si + mu * si;
                    let dst = o * w * so + k * so;
                    for t in 0..inner {
                        out[dst + t] += cm * cur[src + t];
                    }
                }
            }
        }
        cur = out;
        dims = nd;
    }
    cur
}

/// Single-determinant active space on RHF orbitals vs MP2 pair sums.
#[test]
fn single_determinant_active_space_matches_mp2_h2o() {
    let (mol, prep, rhf) = setup_text(&molecule_text("water.xyz"), "6-31g");
    let nocc = mol.nelec() as usize / 2;
    let (nc, t) = (nocc - 1, nocc - 1);
    let c = rhf.mos_r().clone();
    let eps = rhf.eps_r().to_vec();
    let nmo = c.ncols();
    let reference = Nevpt2Reference {
        mo_coeff: &c,
        n_core: nc,
        n_active: 1,
        n_elec_active: (1, 1),
        ci_vector: &[1.0],
    };
    let r = sc_nevpt2(&mol, &prep, &reference, &Nevpt2Config::default()).unwrap();

    let nb = prep.nbasis();
    let eri = ferric_mp2::canonical::dense_ao_eri(&prep, Operator::coulomb()).unwrap();
    let occ: Vec<usize> = (0..nocc).collect();
    let vir: Vec<usize> = (nocc..nmo).collect();
    let nv = vir.len();
    let ovov = mo_eri(&eri, nb, &c, [&occ, &vir, &occ, &vir]);
    let g = |i: usize, a: usize, j: usize, b: usize| ovov[((i * nv + a) * nocc + j) * nv + b];
    let mut pair = Array2::<f64>::zeros((nocc, nocc));
    for i in 0..nocc {
        for j in 0..nocc {
            for a in 0..nv {
                for b in 0..nv {
                    let den = eps[i] + eps[j] - eps[nocc + a] - eps[nocc + b];
                    pair[(i, j)] += g(i, a, j, b) * (2.0 * g(i, a, j, b) - g(i, b, j, a)) / den;
                }
            }
        }
    }
    let tt = mo_eri(&eri, nb, &c, [&[t], &[t], &[t], &[t]])[0];
    let mut e_rs = 0.0;
    for a in 0..nv {
        for b in 0..nv {
            let v = g(t, a, t, b);
            e_rs -= v * v / (eps[nocc + a] + eps[nocc + b] - 2.0 * eps[t] + tt);
        }
    }
    let e = r.energies;
    let want = Nevpt2Classes {
        sijrs: pair.slice(ndarray::s![0..nc, 0..nc]).sum(),
        srsi: 2.0 * pair.slice(ndarray::s![0..nc, t]).sum(),
        srs: e_rs,
        ..Default::default()
    };
    let bar = 1e-10;
    for ((name, got), w) in Nevpt2Classes::NAMES
        .iter()
        .zip(e.to_array())
        .zip(want.to_array())
    {
        eprintln!("{name:6} {got:.14} expected {w:.14} diff {:.1e}", got - w);
        assert!(
            (got - w).abs() < bar,
            "{name}: {got} vs MP2 {w} (bar {bar:e})"
        );
    }
}

struct Stored {
    mol: Molecule,
    prep: PreparedBasis,
    c: Array2<f64>,
    ci: Vec<f64>,
    n_core: usize,
    n_active: usize,
    n_elec: (usize, usize),
    v: Value,
}

fn stored(label: &str) -> Stored {
    let v = system(label);
    let mol = Molecule::parse_xyz(v["xyz"].as_str().unwrap(), 0, 1).unwrap();
    let bs = ferric_core::basis::bundled(v["basis"].as_str().unwrap()).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let fc = &v["ferric_casscf"];
    let shape: Vec<usize> = fc["mo_coeff_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let c = Array2::from_shape_vec((shape[0], shape[1]), f64s(fc, "mo_coeff")).unwrap();
    let ne = v["n_elec_active"].as_array().unwrap();
    Stored {
        mol,
        prep,
        c,
        ci: f64s(fc, "ci_vector"),
        n_core: v["n_core"].as_u64().unwrap() as usize,
        n_active: v["n_active"].as_u64().unwrap() as usize,
        n_elec: (
            ne[0].as_u64().unwrap() as usize,
            ne[1].as_u64().unwrap() as usize,
        ),
        v,
    }
}

/// Anchor 1: the prototype's numbers on the same orbitals and CI vector.
fn vs_prototype(label: &str, canonicalize: bool) {
    let s = stored(label);
    let reference = Nevpt2Reference {
        mo_coeff: &s.c,
        n_core: s.n_core,
        n_active: s.n_active,
        n_elec_active: s.n_elec,
        ci_vector: &s.ci,
    };
    let cfg = Nevpt2Config {
        canonicalize,
        ..Default::default()
    };
    let r = sc_nevpt2(&s.mol, &s.prep, &reference, &cfg).unwrap();
    let e_ref = num(&s.v["ferric_casscf"], "e_total");
    assert!(
        (r.e_reference - e_ref).abs() < 1e-10,
        "{label}: reference energy {} vs stored CASSCF {e_ref}",
        r.e_reference
    );
    let (n, nc, nv) = (s.n_active as f64, r.n_core as f64, r.n_virtual as f64);
    let labels = nv * nc + nv * nv + nc * nc;
    let bar = labels * n.powi(6) * EPS * 4.0;
    let mut worst = 0.0f64;
    for (name, got) in Nevpt2Classes::NAMES.iter().zip(r.energies.to_array()) {
        let want = class(&s.v["prototype"], name);
        let d = got - want;
        worst = worst.max(d.abs());
        eprintln!("{label} {name:6} rust {got:.14} prototype {want:.14} diff {d:.1e}");
        assert!(
            d.abs() < bar,
            "{label} {name}: {got} vs prototype {want} (bar {bar:.1e})"
        );
    }
    eprintln!("{label} canonicalize={canonicalize}: max diff {worst:.1e}, bar {bar:.1e}");
}

#[test]
fn matches_prototype_h2o_cas44() {
    vs_prototype("h2o", false);
}

/// H2O has no degenerate core or virtual orbitals, so diagonalizing the
/// (already diagonal) blocks again reproduces the same numbers.
#[test]
fn matches_prototype_h2o_cas44_recanonicalized() {
    vs_prototype("h2o", true);
}

#[test]
fn matches_prototype_n2_cas66() {
    vs_prototype("n2", false);
}

#[test]
fn matches_prototype_n2_stretched_cas66() {
    vs_prototype("n2-stretched", false);
}

/// Anchor 2: ferric CASSCF + SC-NEVPT2 end to end vs PySCF.
fn vs_pyscf(label: &str) {
    let s = stored(label);
    let (mol, prep) = (s.mol, s.prep);
    let (_, _, rhf) = setup_text(s.v["xyz"].as_str().unwrap(), s.v["basis"].as_str().unwrap());
    let cfg = CasScfConfig {
        n_active: s.n_active,
        n_elec_active: s.n_elec,
        n_core: s.n_core,
        grad_tol: num(&s.v["ferric_casscf"], "casscf_grad_tol"),
        canonicalize: true,
        ..Default::default()
    };
    let cas = run_casscf(&mol, &prep, &rhf, &cfg).unwrap();
    let reference = Nevpt2Reference::from_casscf(&cas, &cfg).unwrap();
    let r = sc_nevpt2(&mol, &prep, &reference, &Nevpt2Config::default()).unwrap();
    let p = &s.v["pyscf"];
    let dk = (cas.grad_norm + num(p, "grad_norm")) / num(p, "lambda_min");
    eprintln!(
        "{label}: E_CASSCF ferric {:.12} pyscf {:.12}; |g| ferric {:.1e} pyscf {:.1e}",
        cas.e_total,
        num(p, "e_casscf"),
        cas.grad_norm,
        num(p, "grad_norm")
    );
    for (name, got) in Nevpt2Classes::NAMES.iter().zip(r.energies.to_array()) {
        let want = class(&p["classes"], name);
        let bar =
            class(&p["dE_dkappa"], name) * dk + 2.0 * class(&p["degenerate_spread"], name) + 1e-10;
        let d = got - want;
        eprintln!("{label} {name:6} rust {got:.12} pyscf {want:.12} diff {d:.1e} bar {bar:.1e}");
        assert!(
            d.abs() < bar,
            "{label} {name}: {got} vs PySCF {want} (bar {bar:.1e})"
        );
    }
}

#[test]
fn matches_pyscf_h2o_cas44() {
    vs_pyscf("h2o");
}

#[test]
fn matches_pyscf_n2_cas66() {
    vs_pyscf("n2");
}

#[test]
fn matches_pyscf_n2_stretched_cas66() {
    vs_pyscf("n2-stretched");
}

/// Unsupported references are refused: an open-shell active space, a CI
/// vector that is not an eigenvector, an unconverged CASSCF, non-canonical
/// orbitals with canonicalize off, and a starved memory budget.
#[test]
fn unsupported_references_are_refused() {
    let s = stored("h2o");
    let base = Nevpt2Reference {
        mo_coeff: &s.c,
        n_core: s.n_core,
        n_active: s.n_active,
        n_elec_active: s.n_elec,
        ci_vector: &s.ci,
    };
    let cfg = Nevpt2Config::default();
    let run = |r: &Nevpt2Reference<'_>, c: &Nevpt2Config| sc_nevpt2(&s.mol, &s.prep, r, c);

    let open = Nevpt2Reference {
        n_core: s.n_core - 1,
        n_elec_active: (s.n_elec.0 + 2, s.n_elec.1),
        ..base
    };
    let e = run(&open, &cfg).unwrap_err().to_string();
    assert!(e.contains("open-shell"), "{e}");

    // A normalized vector that is not an eigenvector: mix in a second
    // determinant.
    let mut mixed = s.ci.clone();
    mixed[0] += 0.3;
    mixed[1] -= 0.2;
    let nrm = mixed.iter().map(|x| x * x).sum::<f64>().sqrt();
    mixed.iter_mut().for_each(|x| *x /= nrm);
    let e = run(
        &Nevpt2Reference {
            ci_vector: &mixed,
            ..base
        },
        &cfg,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("not an eigenvector"), "{e}");

    // Rotate two virtual orbitals: the virtual block of F is no longer
    // diagonal.
    let mut c2 = s.c.clone();
    let (a, b) = (s.n_core + s.n_active, s.n_core + s.n_active + 1);
    let (ca, cb) = (c2.column(a).to_owned(), c2.column(b).to_owned());
    let (co, si) = (0.3f64.cos(), 0.3f64.sin());
    c2.column_mut(a).assign(&(&ca * co + &cb * si));
    c2.column_mut(b).assign(&(&cb * co - &ca * si));
    let rot = Nevpt2Reference {
        mo_coeff: &c2,
        ..base
    };
    let off = Nevpt2Config {
        canonicalize: false,
        ..cfg
    };
    let e = run(&rot, &off).unwrap_err().to_string();
    assert!(e.contains("off-diagonal"), "{e}");
    // With canonicalize on the same orbitals give the stored numbers back.
    let r = run(&rot, &cfg).unwrap();
    let want = class(&s.v["prototype"], "Sir");
    assert!(
        (r.energies.sir - want).abs() < 1e-9,
        "{} vs {want}",
        r.energies.sir
    );

    let starved = Nevpt2Config {
        memory_budget_bytes: Some(1 << 16),
        ..cfg
    };
    let e = run(&base, &starved).unwrap_err();
    assert!(
        matches!(e, ferric_core::FerricError::MemoryExhausted(_)),
        "{e}"
    );

    let unconverged = ferric_ci::CasScfResult {
        e_total: 0.0,
        e_core: 0.0,
        mo_coeff: s.c.clone(),
        mo_energy: None,
        ci_vector: s.ci.clone(),
        rdms: ferric_ci::CasCiRdms {
            dm1: Array2::zeros((s.n_active, s.n_active)),
            dm2: ndarray::Array4::zeros((s.n_active, s.n_active, s.n_active, s.n_active)),
        },
        n_determinants: s.ci.len(),
        iterations: 300,
        grad_norm: 1e-3,
        converged: false,
        ci_residual_norm: 0.0,
    };
    let cas_cfg = CasScfConfig {
        n_active: s.n_active,
        n_elec_active: s.n_elec,
        n_core: s.n_core,
        ..Default::default()
    };
    assert!(Nevpt2Reference::from_casscf(&unconverged, &cas_cfg).is_err());
}
