//! One-time active-space MO integral transform for CAS-CI.
//!
//! Given converged RHF MO coefficients, this builds the active-space
//! one-electron Hamiltonian `h_pq`, the active-space two-electron integrals
//! `(pq|rs)` (chemist's notation), and the *core* (inactive + nuclear) energy
//! constant `E_core`. The CAS-CI Hamiltonian acts only within the active space;
//! the fully-occupied inactive (frozen-core) orbitals contribute a constant
//! plus an effective one-electron potential folded into `h_pq`.
//!
//! The AO ERIs are built densely (`nbas^4`, every shell quartet, no
//! permutational symmetry), then transformed to MOs `0 .. active_start +
//! n_active` (the inactive block is needed for the core energy and the
//! effective potential) by four BLAS3 quarter transforms
//! ([`transform_eri_quarter_gemm`]), O(n_needed·nbas^4) flops for the first.
//! The dense AO block limits this to small bases.

use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::oneelectron::hcore;
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;
use ndarray::Array2;

/// Active-space integrals in the MO basis, plus the closed-shell core energy.
#[derive(Debug, Clone)]
pub struct ActiveSpaceIntegrals {
    /// Number of active spatial orbitals.
    pub n_active: usize,
    /// Effective one-electron integrals `h_pq` (n_active x n_active), including
    /// the mean-field potential of the inactive (doubly-occupied core)
    /// orbitals. Symmetric.
    pub h: Array2<f64>,
    /// Two-electron integrals `(pq|rs)` in chemist's notation over the active
    /// orbitals, flat-indexed `[((p*n+q)*n+r)*n+s]`, length `n_active^4`.
    pub eri: Vec<f64>,
    /// Constant energy: nuclear repulsion + inactive-core electronic energy.
    pub e_core: f64,
}

impl ActiveSpaceIntegrals {
    /// `(pq|rs)` chemist-notation active two-electron integral.
    #[inline]
    pub fn g(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let n = self.n_active;
        self.eri[((p * n + q) * n + r) * n + s]
    }
}

/// Peak elements held by [`build_active_space_integrals`].
///
/// The dense AO block and the first quarter-transform output are the largest
/// pair ever co-resident:
///
/// ```text
///   ao_eri               nbas^4
///   first quarter out    n_needed·nbas^3
/// ```
///
/// [`transform_eri_quarter_gemm`] consumes `ao_eri` and drops each quarter's
/// input once its output exists, so the later pairs
/// (`n·nbas^3 + n²·nbas²`, `n²·nbas² + n³·nbas`, `n³·nbas + n^4`) are never
/// larger than the first when `n_needed <= nbas`. `n_needed = active_start +
/// n_active` includes the inactive core: 1.21x the AO block at
/// benzene/cc-pVDZ CAS(6,6) with an 18-orbital core, and exactly 2x in the
/// limit `n_needed == nbas`.
pub fn active_space_integral_elems(nbas: usize, n_needed: usize) -> usize {
    nbas.saturating_pow(4)
        .saturating_add(n_needed.saturating_mul(nbas.saturating_pow(3)))
}

/// Dense chemist-notation AO ERIs `(mu nu|la sg)`, flat-indexed
/// `[((mu*nbas+nu)*nbas+la)*nbas+sg]`, from the full shell-quartet loop.
///
/// Every quartet is computed (no permutational symmetry); the caller charges
/// the `nbas^4` allocation to a memory plan.
fn dense_ao_eri(prep: &PreparedBasis) -> Result<Vec<f64>, FerricError> {
    let nbas = prep.nbasis();
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut eng = Engine::new_2e(Operator::coulomb(), prep, 1e-14)?;
    let mut ao_eri = vec![0.0f64; nbas * nbas * nbas * nbas];
    let idx4 = |mu: usize, nu: usize, la: usize, sg: usize| -> usize {
        ((mu * nbas + nu) * nbas + la) * nbas + sg
    };
    for s1 in 0..nsh {
        for s2 in 0..nsh {
            for s3 in 0..nsh {
                for s4 in 0..nsh {
                    if let Some(q) = eng.compute_quartet(prep, s1, s2, s3, s4) {
                        let (n1, n2, n3, nn4) = (dims[s1], dims[s2], dims[s3], dims[s4]);
                        let (o1, o2, o3, o4) = (offs[s1], offs[s2], offs[s3], offs[s4]);
                        for a in 0..n1 {
                            let mu = o1 + a;
                            for b in 0..n2 {
                                let nu = o2 + b;
                                for cc in 0..n3 {
                                    let la = o3 + cc;
                                    for dd in 0..nn4 {
                                        let sg = o4 + dd;
                                        ao_eri[idx4(mu, nu, la, sg)] =
                                            q[((a * n2 + b) * n3 + cc) * nn4 + dd];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(ao_eri)
}

/// One quarter transform: contract the LAST index of `x` (viewed as an
/// `(m, nbas)` row-major matrix) with `C`, and put the new MO index FIRST.
///
/// `out[s, m] = Σ_k C[k, s] x[m, k]`, i.e. `out = Cᵀ · xᵀ`, one dgemm.
/// Index order `[a, b, c, k]` becomes `[s, a, b, c]`, so four calls rotate
/// `[mu, nu, la, sg]` round to `[p, q, r, s]`. `x` is dropped as soon as the
/// product exists.
fn quarter_transform_rotate(
    x: Vec<f64>,
    nbas: usize,
    ct: &Array2<f64>,
) -> Result<Vec<f64>, FerricError> {
    let m = x.len() / nbas;
    let xm = Array2::from_shape_vec((m, nbas), x)
        .map_err(|e| FerricError::General(format!("CAS-CI quarter transform shape: {e}")))?;
    let out = ct.dot(&xm.t());
    drop(xm);
    // `dot` can return a column-major result for degenerate shapes
    // (CLAUDE.md, "ndarray dot layout"); never flat-read it unnormalized.
    let out = if out.is_standard_layout() {
        out
    } else {
        out.as_standard_layout().into_owned()
    };
    let (v, off) = out.into_raw_vec_and_offset();
    if off.unwrap_or(0) != 0 {
        return Err(FerricError::General(
            "CAS-CI quarter transform: unexpected non-zero array offset".to_string(),
        ));
    }
    Ok(v)
}

/// AO→MO four-index transform
/// `(pq|rs) = Σ C[mu,p] C[nu,q] C[la,r] C[sg,s] (mu nu|la sg)` as four BLAS3
/// quarter transforms: O(n·nbas^4) for the first, less for each later one.
///
/// * `ao_eri` – dense AO ERIs laid out as by [`dense_ao_eri`] (consumed).
/// * `c` – `nbas x n` MO coefficient columns to transform to.
///
/// Returns `(pq|rs)` flat-indexed `[((p*n+q)*n+r)*n+s]`. At most two of the
/// chain's buffers are alive at once (see [`active_space_integral_elems`]).
pub(crate) fn transform_eri_quarter_gemm(
    ao_eri: Vec<f64>,
    nbas: usize,
    c: ndarray::ArrayView2<'_, f64>,
) -> Result<Vec<f64>, FerricError> {
    if ao_eri.len() != nbas.pow(4) || c.nrows() != nbas {
        return Err(FerricError::General(format!(
            "CAS-CI transform: ao_eri len {} / C rows {} inconsistent with nbas {nbas}",
            ao_eri.len(),
            c.nrows()
        )));
    }
    let ct = c.t().as_standard_layout().into_owned();
    let mut x = ao_eri;
    for _ in 0..4 {
        x = quarter_transform_rotate(x, nbas, &ct)?;
    }
    Ok(x)
}

/// Build the active-space MO integrals from a converged restricted RHF result.
///
/// * `mol`  – molecule (for nuclear repulsion).
/// * `prep` – prepared AO basis (for the AO integrals).
/// * `rhf`  – converged restricted SCF result (MO coefficients).
/// * `active_start` – index of the first active MO. MOs `0..active_start` are
///   the doubly-occupied *inactive* (frozen-core) orbitals.
/// * `n_active` – number of active spatial orbitals.
///
/// Validates the active window against the total MO count up front and returns
/// a clean `Err` (never panics deep inside the transform).
pub fn build_active_space_integrals(
    mol: &Molecule,
    prep: &PreparedBasis,
    rhf: &ScfResult,
    active_start: usize,
    n_active: usize,
    memory_budget_bytes: Option<usize>,
) -> Result<ActiveSpaceIntegrals, FerricError> {
    let nbas = prep.nbasis();
    let c = rhf.mos_r();
    let nmo = c.ncols();

    // ---- Up-front validation (no panics past this point) ---------------
    if n_active == 0 {
        return Err(FerricError::General(
            "CAS-CI: n_active must be >= 1".to_string(),
        ));
    }
    if active_start + n_active > nmo {
        return Err(FerricError::General(format!(
            "CAS-CI: active window active_start({active_start}) + n_active({n_active}) \
             = {} exceeds the number of MOs ({nmo})",
            active_start + n_active
        )));
    }

    let n_inactive = active_start; // doubly-occupied core orbitals

    // ---- AO core Hamiltonian ------------------------------------------
    let h_ao = hcore(prep);

    // ---- Dense AO ERI tensor + transform chain, charged up front -------
    // `memory_budget_bytes` is the caller's `CasCiConfig::memory_budget_bytes`
    // (None resolves the env / cgroup / RAM ceiling). The charge is the
    // co-resident peak of the AO block plus the transform chain, ao_eri + the
    // first quarter output (see `active_space_integral_elems`).
    let n4 = nbas * nbas * nbas * nbas;
    let peak_elems = active_space_integral_elems(nbas, active_start + n_active);
    let mut plan = MemoryPlan::from_global_pool(
        memory_budget_bytes,
        format!(
            "CAS-CI dense AO ERIs + MO transform chain (nbas={nbas}, \
             n_needed={}; ao nbas^4 = {n4} f64, peak = {peak_elems} f64)",
            active_start + n_active,
        ),
    );
    plan.reserve(
        "ao_eri + first quarter-transform output (co-resident peak)",
        peak_elems,
        Lifetime::Resident,
    );
    // Held for the rest of this function; the transform's buffers are all
    // allocated and released inside it.
    let _mem = plan.commit()?;
    let ao_eri = dense_ao_eri(prep)?;

    // ---- Transform to the MO subset we need ----------------------------
    // We need MOs 0..(active_start + n_active): the inactive block (for the
    // core energy + effective potential) and the active block.
    let n_needed = active_start + n_active;
    let c_needed = c.slice(ndarray::s![.., 0..n_needed]);
    // Four BLAS3 quarter transforms; `ao_eri` is consumed (dropped after the
    // first quarter) so the chain never holds more than two buffers.
    let mo_eri = transform_eri_quarter_gemm(ao_eri, nbas, c_needed)?;
    let midx = |p: usize, q: usize, r: usize, s: usize| -> usize {
        ((p * n_needed + q) * n_needed + r) * n_needed + s
    };

    // ---- One-electron h in the MO basis (full needed block) ------------
    // h_mo[p,q] = sum_{mu,nu} C[mu,p] h_ao[mu,nu] C[nu,q]
    let mut h_mo = Array2::<f64>::zeros((n_needed, n_needed));
    for p in 0..n_needed {
        for q in 0..n_needed {
            let mut acc = 0.0;
            for mu in 0..nbas {
                let cmp = c[(mu, p)];
                if cmp == 0.0 {
                    continue;
                }
                for nu in 0..nbas {
                    acc += cmp * h_ao[(mu, nu)] * c[(nu, q)];
                }
            }
            h_mo[(p, q)] = acc;
        }
    }

    // ---- Core energy: nuclear repulsion + inactive electronic energy ---
    // Inactive orbitals i in 0..n_inactive are doubly occupied. The closed-shell
    // core energy is
    //   E_core = E_nuc + sum_i 2 h_ii + sum_{ij} [ 2 (ii|jj) - (ij|ji) ].
    let e_nuc = mol.nuclear_repulsion();
    let mut e_core = e_nuc;
    for i in 0..n_inactive {
        e_core += 2.0 * h_mo[(i, i)];
    }
    for i in 0..n_inactive {
        for j in 0..n_inactive {
            e_core += 2.0 * mo_eri[midx(i, i, j, j)] - mo_eri[midx(i, j, j, i)];
        }
    }

    // ---- Effective one-electron h in the active space -------------------
    // The doubly-occupied inactive orbitals contribute an effective potential
    //   h_eff[p,q] = h[p,q] + sum_i [ 2 (pq|ii) - (pi|iq) ]
    // for active p,q (indices shifted by active_start).
    let mut h_act = Array2::<f64>::zeros((n_active, n_active));
    for pa in 0..n_active {
        let p = active_start + pa;
        for qa in 0..n_active {
            let q = active_start + qa;
            let mut val = h_mo[(p, q)];
            for i in 0..n_inactive {
                val += 2.0 * mo_eri[midx(p, q, i, i)] - mo_eri[midx(p, i, i, q)];
            }
            h_act[(pa, qa)] = val;
        }
    }

    // ---- Active two-electron integrals (pq|rs) -------------------------
    let mut eri_act = vec![0.0f64; n_active * n_active * n_active * n_active];
    for pa in 0..n_active {
        let p = active_start + pa;
        for qa in 0..n_active {
            let q = active_start + qa;
            for ra in 0..n_active {
                let r = active_start + ra;
                for sa in 0..n_active {
                    let s = active_start + sa;
                    eri_act[((pa * n_active + qa) * n_active + ra) * n_active + sa] =
                        mo_eri[midx(p, q, r, s)];
                }
            }
        }
    }

    Ok(ActiveSpaceIntegrals {
        n_active,
        h: h_act,
        eri: eri_act,
        e_core,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    use std::time::Instant;

    /// The scalar-loop quarter transform this module used before the GEMM
    /// version, kept verbatim (only lifted into a function) as the
    /// independent construction the anchor compares against.
    fn transform_eri_loops(
        ao_eri: &[f64],
        nbas: usize,
        c: &Array2<f64>,
        n_needed: usize,
    ) -> Vec<f64> {
        let idx4 = |mu: usize, nu: usize, la: usize, sg: usize| -> usize {
            ((mu * nbas + nu) * nbas + la) * nbas + sg
        };
        let mut mo_eri = vec![0.0f64; n_needed * n_needed * n_needed * n_needed];
        let midx = |p: usize, q: usize, r: usize, s: usize| -> usize {
            ((p * n_needed + q) * n_needed + r) * n_needed + s
        };
        let mut t1 = vec![0.0f64; n_needed * nbas * nbas * nbas];
        let t1idx =
            |p: usize, nu: usize, la: usize, sg: usize| ((p * nbas + nu) * nbas + la) * nbas + sg;
        for mu in 0..nbas {
            for nu in 0..nbas {
                for la in 0..nbas {
                    for sg in 0..nbas {
                        let v = ao_eri[idx4(mu, nu, la, sg)];
                        if v == 0.0 {
                            continue;
                        }
                        for p in 0..n_needed {
                            let cmp = c[(mu, p)];
                            if cmp != 0.0 {
                                t1[t1idx(p, nu, la, sg)] += cmp * v;
                            }
                        }
                    }
                }
            }
        }
        let mut t2 = vec![0.0f64; n_needed * n_needed * nbas * nbas];
        let t2idx =
            |p: usize, q: usize, la: usize, sg: usize| ((p * n_needed + q) * nbas + la) * nbas + sg;
        for p in 0..n_needed {
            for nu in 0..nbas {
                for la in 0..nbas {
                    for sg in 0..nbas {
                        let v = t1[t1idx(p, nu, la, sg)];
                        if v == 0.0 {
                            continue;
                        }
                        for q in 0..n_needed {
                            let cnq = c[(nu, q)];
                            if cnq != 0.0 {
                                t2[t2idx(p, q, la, sg)] += cnq * v;
                            }
                        }
                    }
                }
            }
        }
        let mut t3 = vec![0.0f64; n_needed * n_needed * n_needed * nbas];
        let t3idx = |p: usize, q: usize, r: usize, sg: usize| {
            ((p * n_needed + q) * n_needed + r) * nbas + sg
        };
        for p in 0..n_needed {
            for q in 0..n_needed {
                for la in 0..nbas {
                    for sg in 0..nbas {
                        let v = t2[t2idx(p, q, la, sg)];
                        if v == 0.0 {
                            continue;
                        }
                        for r in 0..n_needed {
                            let clr = c[(la, r)];
                            if clr != 0.0 {
                                t3[t3idx(p, q, r, sg)] += clr * v;
                            }
                        }
                    }
                }
            }
        }
        for p in 0..n_needed {
            for q in 0..n_needed {
                for r in 0..n_needed {
                    for sg in 0..nbas {
                        let v = t3[t3idx(p, q, r, sg)];
                        if v == 0.0 {
                            continue;
                        }
                        for s in 0..n_needed {
                            let csg = c[(sg, s)];
                            if csg != 0.0 {
                                mo_eri[midx(p, q, r, s)] += csg * v;
                            }
                        }
                    }
                }
            }
        }
        mo_eri
    }

    fn rhf_on(mol: Molecule, basis_name: &str) -> (Molecule, PreparedBasis, ScfResult) {
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &prep,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                density_conv: 1e-9,
                ..Default::default()
            },
        )
        .unwrap();
        (mol, prep, rhf)
    }

    const WATER: &str = "3\nH2O\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";

    fn water(basis_name: &str) -> (Molecule, PreparedBasis, ScfResult) {
        rhf_on(Molecule::parse_xyz(WATER, 0, 1).unwrap(), basis_name)
    }

    fn benzene(basis_name: &str) -> (Molecule, PreparedBasis, ScfResult) {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/molecules/benzene.xyz"
        );
        rhf_on(Molecule::load_xyz(path).unwrap(), basis_name)
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    /// Old loop transform vs new GEMM transform on the same AO ERIs and C.
    fn gemm_vs_loops(prep: &PreparedBasis, rhf: &ScfResult, n_needed: usize) -> (f64, f64) {
        let nbas = prep.nbasis();
        let c = rhf.mos_r();
        let ao = dense_ao_eri(prep).unwrap();
        let old = transform_eri_loops(&ao, nbas, c, n_needed);
        let new =
            transform_eri_quarter_gemm(ao, nbas, c.slice(ndarray::s![.., 0..n_needed])).unwrap();
        let scale = old.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        (max_abs_diff(&old, &new), scale)
    }

    /// EXACTNESS ANCHOR: the GEMM transform reproduces the scalar-loop
    /// transform element for element. Cases cover the full MO space
    /// (n_needed == nbas), a strict subset in a larger basis, and the
    /// degenerate n_needed == 1 shape (where `dot` layout can flip).
    #[test]
    fn gemm_transform_matches_loop_transform() {
        let (_, prep, rhf) = water("sto-3g");
        let nbas = prep.nbasis();
        for n_needed in [1, 5, nbas] {
            let (d, scale) = gemm_vs_loops(&prep, &rhf, n_needed);
            eprintln!("H2O/STO-3G nbas={nbas} n={n_needed}: max|d|={d:.2e} (max|g|={scale:.2})");
            assert!(d < 1e-12, "H2O/STO-3G n={n_needed}: max|d| = {d:.3e}");
        }
        let (_, prep, rhf) = water("cc-pvdz");
        let nbas = prep.nbasis();
        for n_needed in [9, 17] {
            let (d, scale) = gemm_vs_loops(&prep, &rhf, n_needed);
            eprintln!("H2O/cc-pVDZ nbas={nbas} n={n_needed}: max|d|={d:.2e} (max|g|={scale:.2})");
            assert!(d < 1e-12, "H2O/cc-pVDZ n={n_needed}: max|d| = {d:.3e}");
        }
    }

    /// Old vs new transform on the same AO ERIs, `reps` times each, per case.
    /// The dense AO ERI build is timed once per case for context.
    ///
    /// Run on a quiet box (release, OPENBLAS_NUM_THREADS=1):
    /// `cargo test --release -p ferric-ci --lib timing_transform -- --ignored --nocapture`
    #[test]
    #[ignore = "quiet box: timing"]
    fn timing_transform_old_vs_new() {
        let reps = 3;
        type Fixture = fn(&str) -> (Molecule, PreparedBasis, ScfResult);
        let cases: [(&str, Fixture, &str, usize); 4] = [
            ("H2O", water, "cc-pvdz", 9),
            ("benzene", benzene, "sto-3g", 24),
            ("benzene", benzene, "6-31g", 24),
            ("benzene", benzene, "cc-pvdz", 24),
        ];
        for (name, fixture, basis_name, n_needed) in cases {
            let (_, prep, rhf) = fixture(basis_name);
            let nbas = prep.nbasis();
            let c = rhf.mos_r();
            let t = Instant::now();
            let ao = dense_ao_eri(&prep).unwrap();
            let t_ao = t.elapsed().as_secs_f64();
            let mut t_old = Vec::new();
            let mut t_new = Vec::new();
            let mut d = 0.0f64;
            for _ in 0..reps {
                let t = Instant::now();
                let old = transform_eri_loops(&ao, nbas, c, n_needed);
                t_old.push(t.elapsed().as_secs_f64());
                let ao_copy = ao.clone();
                let t = Instant::now();
                let new = transform_eri_quarter_gemm(
                    ao_copy,
                    nbas,
                    c.slice(ndarray::s![.., 0..n_needed]),
                )
                .unwrap();
                t_new.push(t.elapsed().as_secs_f64());
                d = d.max(max_abs_diff(&old, &new));
            }
            eprintln!(
                "CITX {name}/{basis_name} nbas={nbas} n_needed={n_needed} \
                 ao_eri_build={t_ao:.3}s old={t_old:.4?} new={t_new:.4?} max|d|={d:.2e}"
            );
        }
    }

    fn vm_hwm_kib() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find_map(|l| l.strip_prefix("VmHWM:"))
                    .and_then(|r| r.trim().trim_end_matches("kB").trim().parse().ok())
            })
            .unwrap_or(0)
    }

    /// Peak-RSS rise over the resident AO block for ONE path (VmHWM never
    /// resets, so one path per process): `CITX_PATH=old|new`,
    /// benzene/cc-pVDZ, n_needed = 24.
    #[test]
    #[ignore = "quiet box: timing"]
    fn memory_transform_one_path() {
        let path = std::env::var("CITX_PATH").unwrap_or_else(|_| "new".to_string());
        let (_, prep, rhf) = benzene("cc-pvdz");
        let nbas = prep.nbasis();
        let c = rhf.mos_r();
        let n_needed = 24;
        let ao = dense_ao_eri(&prep).unwrap();
        let before = vm_hwm_kib();
        let out = if path == "old" {
            transform_eri_loops(&ao, nbas, c, n_needed)
        } else {
            transform_eri_quarter_gemm(ao, nbas, c.slice(ndarray::s![.., 0..n_needed])).unwrap()
        };
        let after = vm_hwm_kib();
        eprintln!(
            "CITX-MEM path={path} nbas={nbas} n_needed={n_needed} VmHWM before={before} kB \
             after={after} kB rise={} kB (out len {})",
            after.saturating_sub(before),
            out.len()
        );
    }
}
