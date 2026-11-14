//! Solute electrostatic potential at cavity tesserae, and the reaction-field
//! one-electron AO operator built from the resulting apparent surface
//! charges.
//!
//! Mirrors `ferric_rpa::properties::esp_at_atoms` exactly, just evaluated at
//! cavity tesserae instead of nuclear positions — see that module's doc
//! comment for the libint sign-convention derivation this reuses
//! (`⟨μ| −1/|r−R| |ν⟩` from a Z=+1 probe charge).

use ferric_core::special::erf;
use std::os::raw::c_int;

use ferric_core::external_potential::{PointCharge, SmearedCharge};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::cosx_a::CosxScreen;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi::{self, CAtom};
use ferric_integrals::md3c1e::Md3c1e;
use ferric_integrals::oneelectron::smeared_attraction;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ndarray::Array2;

use crate::cavity::Tessera;

/// Tesserae per batch of the McMurchie-Davidson route. Fixed (never derived
/// from the thread count), so the order of every floating-point sum is the
/// same for every pool size: potentials are independent per tessera, and the
/// operator sums its tessera axis in chunks of this size in order.
const TESS_CHUNK: usize = 256;

/// Runs `f` under a `FERRIC_TIMING` stage named `label` (inert otherwise).
fn timed<T>(label: &'static str, f: impl FnOnce() -> T) -> T {
    let stage = ferric_core::timing::Stage::start(label);
    let out = f();
    stage.end();
    out
}

fn check_density_shape(what: &str, density: &Array2<f64>, nbas: usize) -> Result<(), FerricError> {
    if density.shape() != [nbas, nbas] {
        return Err(FerricError::General(format!(
            "{what}: density shape {:?} != ({nbas},{nbas})",
            density.shape()
        )));
    }
    Ok(())
}

/// `Σ_B Z_B / |r_k − R_B|` at tessera `t`.
fn nuclear_potential_at(what: &str, mol: &Molecule, t: &Tessera) -> Result<f64, FerricError> {
    let mut v_nuc = 0.0_f64;
    for atom in &mol.atoms {
        if atom.ghost {
            continue;
        }
        let dx = t.position[0] - atom.x;
        let dy = t.position[1] - atom.y;
        let dz = t.position[2] - atom.zpos;
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r < 1e-8 {
            return Err(FerricError::General(format!(
                "{what}: tessera coincides with a nucleus (degenerate cavity)"
            )));
        }
        v_nuc += atom.effective_z() as f64 / r;
    }
    Ok(v_nuc)
}

/// Evaluate the total (nuclear + electronic) solute electrostatic potential
/// at each cavity tessera, in Hartree.
///
/// ```text
///     v_k = Σ_B Z_B / |r_k − R_B|  −  Σ_{μν} D_{μν} ⟨μ| 1/|r − r_k| |ν⟩
/// ```
///
/// No self-interaction subtlety here (unlike `esp_at_atoms`'s atom-centered
/// case): every tessera sits strictly outside every real nucleus by
/// cavity construction, so the nuclear sum has no `A == k` singularity to
/// skip.
///
/// The electronic term runs through the batched McMurchie-Davidson kernel
/// ([`ferric_integrals::md3c1e`]) over fixed-size batches of tesserae,
/// parallel over batches; the result is bit-identical for every pool size.
/// A basis the kernel refuses (pure p, `l` above `MAX_L`) takes
/// [`solute_potential_at_tesserae_libint`].
pub fn solute_potential_at_tesserae(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
    tess: &[Tessera],
) -> Result<Vec<f64>, FerricError> {
    timed("pcm:potential_at_tesserae", || match Md3c1e::new(prep) {
        Ok(kernel) => solute_potential_md3c1e(&kernel, mol, prep, density, tess),
        Err(_) => solute_potential_at_tesserae_libint(mol, prep, density, tess),
    })
}

/// [`solute_potential_at_tesserae`] through the batched kernel. The kernel
/// returns the repulsive `+1/|r − r_k|`, so the electronic term is
/// subtracted (libint's attraction sign is the opposite).
pub fn solute_potential_md3c1e(
    kernel: &Md3c1e,
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
    tess: &[Tessera],
) -> Result<Vec<f64>, FerricError> {
    use rayon::prelude::*;
    const WHAT: &str = "solute_potential_at_tesserae";
    check_density_shape(WHAT, density, prep.nbasis())?;
    let points: Vec<[f64; 3]> = tess.iter().map(|t| t.position).collect();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let none = CosxScreen::none();

    let chunks: Vec<Vec<f64>> = points
        .par_chunks(TESS_CHUNK)
        .map_init(
            || kernel.scratch(),
            |scr, pts| -> Result<Vec<f64>, FerricError> {
                let g = pts.len();
                let mut acc = vec![0.0_f64; g];
                kernel.for_each_pair(pts, None, none, scr, |s1, s2, blk| {
                    let (n1, n2) = (dims[s1], dims[s2]);
                    let (o1, o2) = (offs[s1], offs[s2]);
                    let fac = if s1 == s2 { 1.0 } else { 2.0 };
                    for i in 0..n1 {
                        for j in 0..n2 {
                            let w = fac * density[(o1 + i, o2 + j)];
                            let row = &blk[(i * n2 + j) * g..(i * n2 + j + 1) * g];
                            for (a, &x) in acc.iter_mut().zip(row) {
                                *a += w * x;
                            }
                        }
                    }
                })?;
                Ok(acc)
            },
        )
        .collect::<Result<_, _>>()?;

    let mut out = Vec::with_capacity(tess.len());
    for (t, v_rep) in tess.iter().zip(chunks.iter().flatten()) {
        out.push(nuclear_potential_at(WHAT, mol, t)? - v_rep);
    }
    Ok(out)
}

/// [`solute_potential_at_tesserae`] through libint2's nuclear-attraction
/// engine, one tessera at a time (parallel over tesserae, one engine per
/// worker). The route for a basis the batched kernel refuses, and the
/// independent oracle the kernel is tested against.
pub fn solute_potential_at_tesserae_libint(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
    tess: &[Tessera],
) -> Result<Vec<f64>, FerricError> {
    use rayon::prelude::*;
    const WHAT: &str = "solute_potential_at_tesserae";
    check_density_shape(WHAT, density, prep.nbasis())?;
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let nsh = prep.nshells();

    tess.par_iter()
        .map_init(
            || Engine::new_1e(ffi::OP_NUCLEAR, prep, 1e-14),
            |eng, t| -> Result<f64, FerricError> {
                let eng = eng.as_mut().map_err(|e| {
                    FerricError::General(format!("{WHAT}: engine init failed: {e}"))
                })?;
                let probe = [CAtom {
                    atomic_number: 1.0,
                    x: t.position[0],
                    y: t.position[1],
                    z: t.position[2],
                }];
                // SAFETY: probe is a stack-local [CAtom; 1] that outlives the FFI call;
                // eng is a valid Engine handle; len=1 matches the array.
                let rc = unsafe {
                    ffi::scf_engine_set_point_charges(
                        eng.handle_mut(),
                        probe.as_ptr(),
                        probe.len() as c_int,
                    )
                };
                if rc < 0 {
                    return Err(FerricError::General(format!(
                        "{WHAT}: set_point_charges failed (rc={rc})"
                    )));
                }
                let mut v_elec = 0.0_f64;
                for s1 in 0..nsh {
                    for s2 in 0..=s1 {
                        let block = eng.compute_1e_block(prep, s1, s2);
                        let (n1, n2) = (dims[s1], dims[s2]);
                        let (o1, o2) = (offs[s1], offs[s2]);
                        let fac = if s1 == s2 { 1.0 } else { 2.0 };
                        for i in 0..n1 {
                            for j in 0..n2 {
                                v_elec += fac * density[(o1 + i, o2 + j)] * block[i * n2 + j];
                            }
                        }
                    }
                }
                Ok(nuclear_potential_at(WHAT, mol, t)? + v_elec)
            },
        )
        .collect()
}

/// Build the one-electron reaction-field AO operator `V_pcm` from the
/// apparent surface charges `q` at cavity tesserae:
///
/// ```text
///     V_pcm_{μν} = Σ_k q_k · ⟨μ| −1/|r − r_k| |ν⟩
/// ```
///
/// This is exactly the same sign convention `nuclear_with_external` uses
/// for a generic classical `PointCharge` (treating each apparent surface
/// charge as a fixed point charge for the purpose of the one-electron
/// integral) — see the module doc for the derivation of why this is the
/// correct one-electron potential energy operator for an electron
/// interacting with the reaction field.
///
/// Runs through the batched kernel ([`ferric_integrals::md3c1e`]), parallel
/// over shell pairs (each output block is independent, so the result is
/// bit-identical for every pool size); a basis the kernel refuses takes
/// [`build_reaction_field_operator_libint`].
pub fn build_reaction_field_operator(
    prep: &PreparedBasis,
    tess: &[Tessera],
    q: &[f64],
) -> Result<Array2<f64>, FerricError> {
    timed("pcm:reaction_field_operator", || match Md3c1e::new(prep) {
        Ok(kernel) => build_reaction_field_operator_md3c1e(&kernel, prep, tess, q),
        Err(_) => build_reaction_field_operator_libint(prep, tess, q),
    })
}

fn check_charges(what: &str, tess: &[Tessera], q: &[f64]) -> Result<(), FerricError> {
    if tess.len() != q.len() {
        return Err(FerricError::General(format!(
            "{what}: {} tesserae but {} charges",
            tess.len(),
            q.len()
        )));
    }
    Ok(())
}

/// Shell pairs `s1 >= s2`.
fn shell_pairs(nsh: usize) -> Vec<(usize, usize)> {
    (0..nsh)
        .flat_map(|s1| (0..=s1).map(move |s2| (s1, s2)))
        .collect()
}

/// Scatter per-pair blocks (`[n1][n2]` row-major) into a symmetric matrix.
fn scatter_symmetric(prep: &PreparedBasis, blocks: &[((usize, usize), Vec<f64>)]) -> Array2<f64> {
    let n = prep.nbasis();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut out = Array2::<f64>::zeros((n, n));
    for &((s1, s2), ref blk) in blocks {
        let (n1, n2) = (dims[s1], dims[s2]);
        for i in 0..n1 {
            for j in 0..n2 {
                let v = blk[i * n2 + j];
                out[(offs[s1] + i, offs[s2] + j)] = v;
                out[(offs[s2] + j, offs[s1] + i)] = v;
            }
        }
    }
    out
}

/// [`build_reaction_field_operator`] through the batched kernel:
/// `V_{μν} = −Σ_k q_k A^k_{μν}` (the kernel's `A` is the repulsive
/// `+1/|r − r_k|`).
pub fn build_reaction_field_operator_md3c1e(
    kernel: &Md3c1e,
    prep: &PreparedBasis,
    tess: &[Tessera],
    q: &[f64],
) -> Result<Array2<f64>, FerricError> {
    use rayon::prelude::*;
    check_charges("build_reaction_field_operator", tess, q)?;
    let points: Vec<[f64; 3]> = tess.iter().map(|t| t.position).collect();
    let dims = prep.shell_dims();
    let pairs = shell_pairs(prep.nshells());
    let blocks: Vec<((usize, usize), Vec<f64>)> = pairs
        .par_iter()
        .map_init(
            || (kernel.scratch(), Vec::<f64>::new()),
            |(scr, buf), &(s1, s2)| -> Result<_, FerricError> {
                let nn = dims[s1] * dims[s2];
                let mut acc = vec![0.0_f64; nn];
                for (pts, qc) in points.chunks(TESS_CHUNK).zip(q.chunks(TESS_CHUNK)) {
                    let g = pts.len();
                    buf.resize(nn * g, 0.0);
                    kernel.pair_block(s1, s2, pts, scr, buf)?;
                    for (ij, a) in acc.iter_mut().enumerate() {
                        let row = &buf[ij * g..(ij + 1) * g];
                        let mut sum = 0.0_f64;
                        for (&x, &qk) in row.iter().zip(qc) {
                            sum += qk * x;
                        }
                        *a -= sum;
                    }
                }
                Ok(((s1, s2), acc))
            },
        )
        .collect::<Result<_, _>>()?;
    Ok(scatter_symmetric(prep, &blocks))
}

/// [`build_reaction_field_operator`] through libint2's nuclear-attraction
/// engine with the tessera charges as point charges (parallel over shell
/// pairs, one engine per worker). The route for a basis the batched kernel
/// refuses, and the independent oracle the kernel is tested against.
pub fn build_reaction_field_operator_libint(
    prep: &PreparedBasis,
    tess: &[Tessera],
    q: &[f64],
) -> Result<Array2<f64>, FerricError> {
    use rayon::prelude::*;
    const WHAT: &str = "build_reaction_field_operator";
    check_charges(WHAT, tess, q)?;
    let point_charges: Vec<PointCharge> = tess
        .iter()
        .zip(q.iter())
        .map(|(t, &qk)| PointCharge {
            q: qk,
            x: t.position[0],
            y: t.position[1],
            z: t.position[2],
        })
        .collect();
    // Real atoms zeroed (position kept, charge 0), tessera charges appended:
    // only the reaction-field charges contribute (the real nuclear attraction
    // is in hcore separately, so `set_point_charges_extra` would double-count it).
    let mut all_atoms: Vec<CAtom> = prep
        .atoms()
        .iter()
        .map(|a| CAtom {
            atomic_number: 0.0,
            x: a.x,
            y: a.y,
            z: a.z,
        })
        .collect();
    all_atoms.extend(point_charges.iter().map(|pc| CAtom {
        atomic_number: pc.q,
        x: pc.x,
        y: pc.y,
        z: pc.z,
    }));
    let make_engine = || -> Result<Engine, FerricError> {
        let mut eng = Engine::new_1e(ffi::OP_NUCLEAR, prep, 1e-14)
            .map_err(|e| FerricError::General(format!("{WHAT}: engine init failed: {e}")))?;
        eng.set_point_charges_extra(prep, &point_charges)?;
        // SAFETY: all_atoms is a Vec<CAtom> that outlives the FFI call;
        // eng is a valid Engine handle; len matches the vec length.
        let rc = unsafe {
            ffi::scf_engine_set_point_charges(
                eng.handle_mut(),
                all_atoms.as_ptr(),
                all_atoms.len() as c_int,
            )
        };
        if rc < 0 {
            return Err(FerricError::General(format!(
                "{WHAT}: set_point_charges failed (rc={rc})"
            )));
        }
        Ok(eng)
    };
    let pairs = shell_pairs(prep.nshells());
    let blocks: Vec<((usize, usize), Vec<f64>)> = pairs
        .par_iter()
        .map_init(make_engine, |eng, &(s1, s2)| -> Result<_, FerricError> {
            let eng = eng
                .as_mut()
                .map_err(|e| FerricError::General(e.to_string()))?;
            Ok(((s1, s2), eng.compute_1e_block(prep, s1, s2).to_vec()))
        })
        .collect::<Result<_, _>>()?;
    Ok(scatter_symmetric(prep, &blocks))
}

/// [`solute_potential_at_tesserae`] with each tessera a normalized Gaussian
/// charge of exponent `ξ_k²` (`ξ_k` = [`Tessera::charge_exp`]) instead of a
/// point — [`crate::config::ProbeKind::GaussianSmeared`]:
///
/// ```text
///     v_k = Σ_B Z_B erf(ξ_k |r_k − R_B|)/|r_k − R_B|
///           − Σ_{μν} D_{μν} ⟨μ| erf(ξ_k |r − r_k|)/|r − r_k| |ν⟩
/// ```
///
/// The electronic term is the 3-centre Coulomb integral `(μν|g_k)` against
/// a unit-self-overlap s-Gaussian divided by its integral
/// ([`ferric_integrals::site_basis::SiteBasis`], whose `ζ → ∞` limit is
/// pinned against the point-charge path there).
pub fn solute_potential_at_tesserae_smeared(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
    tess: &[Tessera],
) -> Result<Vec<f64>, FerricError> {
    let nbas = prep.nbasis();
    if density.shape() != [nbas, nbas] {
        return Err(FerricError::General(format!(
            "solute_potential_at_tesserae_smeared: density shape {:?} != ({nbas},{nbas})",
            density.shape()
        )));
    }
    let sites: Vec<[f64; 4]> = tess
        .iter()
        .map(|t| {
            [
                t.position[0],
                t.position[1],
                t.position[2],
                t.charge_exp * t.charge_exp,
            ]
        })
        .collect();
    let site_basis = SiteBasis::new(&sites, 0)?;
    let mut eng = Engine::new_3center(Operator::coulomb(), prep, &site_basis.prep, 1e-14)?;

    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let nsh = prep.nshells();

    let mut out = Vec::with_capacity(tess.len());
    for (k, t) in tess.iter().enumerate() {
        let sh_p = site_basis.site_shell[k];
        let mut v_elec = 0.0_f64;
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                let Some(block) = eng.compute_eri3(prep, &site_basis.prep, sh_p, s1, s2) else {
                    continue;
                };
                let n1 = dims[s1];
                let n2 = dims[s2];
                let o1 = offs[s1];
                let o2 = offs[s2];
                let fac = if s1 == s2 { 1.0 } else { 2.0 };
                let mut acc = 0.0_f64;
                for i in 0..n1 {
                    for j in 0..n2 {
                        acc += density[(o1 + i, o2 + j)] * block[i * n2 + j];
                    }
                }
                v_elec += fac * acc;
            }
        }
        v_elec /= site_basis.norm_int[k];

        let xi = t.charge_exp;
        let mut v_nuc = 0.0_f64;
        for atom in &mol.atoms {
            if atom.ghost {
                continue;
            }
            let dx = t.position[0] - atom.x;
            let dy = t.position[1] - atom.y;
            let dz = t.position[2] - atom.zpos;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-8 {
                return Err(FerricError::General(
                    "solute_potential_at_tesserae_smeared: tessera coincides with a nucleus \
                     (degenerate cavity)"
                        .into(),
                ));
            }
            // SAFETY: erf is the C standard math library function; argument is finite.
            v_nuc += atom.effective_z() as f64 * erf(xi * r) / r;
        }

        out.push(v_nuc - v_elec);
    }
    Ok(out)
}

/// [`build_reaction_field_operator`] with each tessera charge a normalized
/// Gaussian of exponent `ξ_k²` — [`crate::config::ProbeKind::GaussianSmeared`]:
///
/// ```text
///     V_pcm_{μν} = Σ_k q_k · ⟨μ| −erf(ξ_k |r − r_k|)/|r − r_k| |ν⟩
/// ```
///
/// Delegates to [`ferric_integrals::oneelectron::smeared_attraction`] with
/// `width = 1/ξ_k` (that function's `ζ = 1/width²`).
pub fn build_reaction_field_operator_smeared(
    prep: &PreparedBasis,
    tess: &[Tessera],
    q: &[f64],
) -> Result<Array2<f64>, FerricError> {
    if tess.len() != q.len() {
        return Err(FerricError::General(format!(
            "build_reaction_field_operator_smeared: {} tesserae but {} charges",
            tess.len(),
            q.len()
        )));
    }
    let smeared: Vec<SmearedCharge> = tess
        .iter()
        .zip(q.iter())
        .map(|(t, &qk)| SmearedCharge {
            q: qk,
            x: t.position[0],
            y: t.position[1],
            z: t.position[2],
            width: 1.0 / t.charge_exp,
        })
        .collect();
    smeared_attraction(prep, &smeared)
}

impl crate::config::ProbeKind {
    /// Solute electrostatic potential at each tessera for this probe kind.
    pub fn potential_at_tesserae(
        self,
        mol: &Molecule,
        prep: &PreparedBasis,
        density: &Array2<f64>,
        tess: &[Tessera],
    ) -> Result<Vec<f64>, FerricError> {
        match self {
            Self::Point => solute_potential_at_tesserae(mol, prep, density, tess),
            Self::GaussianSmeared => solute_potential_at_tesserae_smeared(mol, prep, density, tess),
        }
    }

    /// Reaction-field one-electron AO operator from the tessera charges `q`.
    pub fn reaction_field_operator(
        self,
        prep: &PreparedBasis,
        tess: &[Tessera],
        q: &[f64],
    ) -> Result<Array2<f64>, FerricError> {
        match self {
            Self::Point => build_reaction_field_operator(prep, tess, q),
            Self::GaussianSmeared => build_reaction_field_operator_smeared(prep, tess, q),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cavity::{build_cavity, CavityConfig};
    use ferric_core::basis;

    fn water_sto3g() -> (Molecule, PreparedBasis) {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        (mol, prep)
    }

    #[test]
    fn potential_at_tesserae_matches_hand_calc_for_zero_density() {
        // With D=0, v_k should equal the pure nuclear-sum ESP, hand-computable.
        let (mol, prep) = water_sto3g();
        let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
        let d0 = Array2::<f64>::zeros((prep.nbasis(), prep.nbasis()));
        let v = solute_potential_at_tesserae(&mol, &prep, &d0, &tess).unwrap();
        assert_eq!(v.len(), tess.len());
        // Hand check tessera 0 against a direct sum.
        let mut expected = 0.0;
        for atom in &mol.atoms {
            let dx = tess[0].position[0] - atom.x;
            let dy = tess[0].position[1] - atom.y;
            let dz = tess[0].position[2] - atom.zpos;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            expected += atom.z as f64 / r;
        }
        assert!(
            (v[0] - expected).abs() < 1e-10,
            "got {}, expected {}",
            v[0],
            expected
        );
    }

    #[test]
    fn reaction_field_operator_is_symmetric_and_zero_for_zero_charges() {
        let (mol, prep) = water_sto3g();
        let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
        let q = vec![0.0; tess.len()];
        let vmat = build_reaction_field_operator(&prep, &tess, &q).unwrap();
        assert!(vmat.iter().all(|&x| x == 0.0));

        let mut q_nonzero = vec![0.0; tess.len()];
        q_nonzero[0] = 0.01;
        let vmat2 = build_reaction_field_operator(&prep, &tess, &q_nonzero).unwrap();
        let n = prep.nbasis();
        for i in 0..n {
            for j in 0..n {
                assert!((vmat2[(i, j)] - vmat2[(j, i)]).abs() < 1e-12);
            }
        }
        assert!(vmat2.iter().any(|&x| x != 0.0));
    }

    /// Exactness anchor for the Gaussian-probe path: as `ξ → ∞` a normalized
    /// Gaussian charge is a point charge, so both smeared functions must
    /// reproduce the independent point-probe functions (which share no
    /// integral code with them: 1e nuclear-attraction engine vs 3-centre
    /// ERIs over a `SiteBasis`).
    #[test]
    fn smeared_probes_reduce_to_point_probes_for_tight_gaussians() {
        let (mol, prep) = water_sto3g();
        let mut tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
        for t in tess.iter_mut() {
            t.charge_exp = 1.0e3;
        }
        let n = prep.nbasis();
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            d[(i, i)] = 0.5;
            if i + 1 < n {
                d[(i, i + 1)] = 0.1;
                d[(i + 1, i)] = 0.1;
            }
        }
        let vp = solute_potential_at_tesserae(&mol, &prep, &d, &tess).unwrap();
        let vs = solute_potential_at_tesserae_smeared(&mol, &prep, &d, &tess).unwrap();
        let max_dv = vp
            .iter()
            .zip(vs.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_dv < 1e-8,
            "point vs tight-Gaussian ESP differ by {max_dv:e}"
        );

        let q: Vec<f64> = (0..tess.len())
            .map(|k| 1e-3 * ((k % 7) as f64 - 3.0))
            .collect();
        let op = build_reaction_field_operator(&prep, &tess, &q).unwrap();
        let os = build_reaction_field_operator_smeared(&prep, &tess, &q).unwrap();
        let max_dm = (&op - &os).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        assert!(
            max_dm < 1e-8,
            "point vs tight-Gaussian V_pcm differ by {max_dm:e}"
        );
        assert!(
            op.iter().any(|&x| x.abs() > 1e-6),
            "V_pcm is trivially zero"
        );
    }

    #[test]
    fn mismatched_lengths_error() {
        let (mol, prep) = water_sto3g();
        let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
        let q = vec![0.0; tess.len() + 1];
        assert!(build_reaction_field_operator(&prep, &tess, &q).is_err());
    }
}
