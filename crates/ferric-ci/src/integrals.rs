//! One-time active-space MO integral transform for CAS-CI.
//!
//! Given converged RHF MO coefficients, this builds the active-space
//! one-electron Hamiltonian `h_pq`, the active-space two-electron integrals
//! `(pq|rs)` (chemist's notation), and the *core* (inactive + nuclear) energy
//! constant `E_core`. The CAS-CI Hamiltonian acts only within the active space;
//! the fully-occupied inactive (frozen-core) orbitals contribute a constant
//! plus an effective one-electron potential folded into `h_pq`.
//!
//! The AO→MO transform mirrors `ferric_mp2::canonical`'s dense shell-quartet
//! loop, but restricted to the active-orbital subset
//! `active_start .. active_start + n_active` (plus, for the effective potential,
//! contractions over the inactive orbitals). This is O(N^4) in the *full* AO
//! basis for the transform and is intended for spike-scale systems only.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::oneelectron::hcore;
use ndarray::Array2;
use ferric_scf::ScfResult;

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

    // ---- Dense AO ERI tensor (chemist (mu nu | la sg)) -----------------
    // Spike-scale: full nbas^4 dense build via the shell-quartet loop, mirroring
    // ferric_mp2::canonical. O(nbas^4) memory — fine for STO-3G test systems.
    let op = Operator::coulomb();
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut eng = Engine::new_2e(op, prep, 1e-14)?;

    let n4 = nbas * nbas * nbas * nbas;
    // Guard the dense AO-ERI allocation against the memory budget.
    //
    // `memory_budget_bytes` used to be a hardcoded `None` here, which resolves
    // the env / cgroup / RAM-auto-detected ceiling and so silently discarded
    // `CasCiConfig::memory_budget_bytes` — the very field whose doc describes
    // it as bounding the CAS-CI peak. The nbas^4 AO-ERI block is allocated
    // before the Davidson bases that field was written for, so a caller who
    // set a budget got it honoured on the later, smaller allocation and
    // ignored on the earlier, larger one.
    ferric_core::memory::check_alloc(
        &format!("CAS-CI dense AO ERIs (nbas={nbas}; nbas^4 = {n4} f64)"),
        n4.saturating_mul(8),
        ferric_core::memory::resolve_budget_bytes(memory_budget_bytes),
    )?;
    let mut ao_eri = vec![0.0f64; n4];
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

    // ---- Transform to the MO subset we need ----------------------------
    // We need MOs 0..(active_start + n_active): the inactive block (for the
    // core energy + effective potential) and the active block.
    let n_needed = active_start + n_active;

    // (pq|rs) in MO basis for p,q,r,s in 0..n_needed. Quarter-transform chain.
    // Step 1: half-transform first two indices  (p q | mu nu) ... we do the
    // standard 4x quarter transforms restricted to the needed MO columns.
    // For spike sizes this straightforward O(nbas^4 * n_needed) approach is fine.
    let mut mo_eri = vec![0.0f64; n_needed * n_needed * n_needed * n_needed];
    let midx = |p: usize, q: usize, r: usize, s: usize| -> usize {
        ((p * n_needed + q) * n_needed + r) * n_needed + s
    };

    // t1[p, nu, la, sg] = sum_mu C[mu,p] (mu nu|la sg)
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
    // t2[p, q, la, sg] = sum_nu C[nu,q] t1[p, nu, la, sg]
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
    // t3[p, q, r, sg] = sum_la C[la,r] t2[p, q, la, sg]
    let mut t3 = vec![0.0f64; n_needed * n_needed * n_needed * nbas];
    let t3idx =
        |p: usize, q: usize, r: usize, sg: usize| ((p * n_needed + q) * n_needed + r) * nbas + sg;
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
    // mo_eri[p,q,r,s] = sum_sg C[sg,s] t3[p, q, r, sg]
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
