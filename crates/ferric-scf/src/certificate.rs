//! The [`Certificate`](ferric_core::certificate::Certificate) of an SCF / KS-DFT result: capability and grade from
//! `ferric_core::certificate::METHOD_KIND_GRADES`, the approximations the
//! solve actually ran with, and warnings for conditions the solver already
//! detected (non-convergence, plateau exits, stability verdicts).
//!
//! Built from the result, the config it was solved with and the molecule,
//! after the solve; nothing here changes an energy. The build stamp is left
//! `None`: only `ferric-cli` and `ferric-python` link `ferric-build-info`
//! and stamp it (see the `ferric_core::certificate` module doc).

use ferric_core::certificate::{Certificate, Grade};
use ferric_core::mol::Molecule;

use crate::result::{ScfExit, ScfResult, Spin};
use crate::rhf::RhfConfig;
use crate::stability::{StabilityResult, StabilityVerdict};

/// The CLI `method.kind` a result is graded under, as `Config::dispatch_kind`
/// routes it: RHF and RKS (`xc` set) are `rhf` / `ksdft`; UHF and UKS run
/// under `uhf`; ROHF and ROKS under `rohf`.
pub fn scf_capability(result: &ScfResult, config: &RhfConfig) -> &'static str {
    match (result.spin, config.xc.is_some()) {
        (Spin::Restricted, false) => "rhf",
        (Spin::Restricted, true) => "ksdft",
        (Spin::Unrestricted, _) => "uhf",
        (Spin::RestrictedOpen, _) => "rohf",
    }
}

/// The certificate of `result`, solved for `mol` with `config` (for a
/// laddered solve, the config of the rung that produced `result`).
///
/// A result with `converged = false` is capped to [`Grade::Ungraded`]: the
/// validation rows grade converged energies only.
pub fn scf_certificate(mol: &Molecule, result: &ScfResult, config: &RhfConfig) -> Certificate {
    let mut c = Certificate::for_method_kind(scf_capability(result, config))
        .with_precision(&result.precision);
    for (name, detail) in approximations(mol, result, config) {
        c = c.with_approximation(name, detail);
    }
    if result.exit == ScfExit::Plateau {
        c.push_warning(
            "SCF exit Plateau: accepted on a near-degeneracy plateau (orbital gradient \
             stalled below the 1e-4 floor), not the standard convergence test",
        );
    }
    stability_warnings(&mut c, result);
    if !result.converged {
        let why = match result.exit {
            ScfExit::NotCertified => "SCF converged but the ROHF occupation witness found a \
                                      lower one-electron move (exit NotCertified)"
                .to_string(),
            e => format!(
                "SCF did not converge (exit {e:?}, {} iterations)",
                result.iterations
            ),
        };
        c = c.cap_grade(Grade::Ungraded, why);
    }
    c
}

/// Whether exact (HF-type) exchange enters the energy, and whether it is
/// range-separated. HF: `(true, false)`. Unresolvable names count as HF-like,
/// as `ferric-cli`'s `log_jk_path` does.
fn exchange_used(config: &RhfConfig) -> (bool, bool) {
    let k_mix = config
        .xc
        .as_deref()
        .and_then(|name| ferric_dft::libxc::xc_def_from_name(name).ok())
        .map(|d| ferric_dft::libxc::k_mix_from_xc_def(&d));
    match k_mix {
        None => (true, false),
        Some(m) => (m.sr > 0.0 || m.omega > 0.0, m.omega > 0.0),
    }
}

fn grid_text(g: &ferric_dft::grid::AtomicGridConfig) -> String {
    let prune = g
        .prune
        .map_or("flat".to_string(), |p| format!("{p:?}").to_lowercase());
    let cut = g
        .weight_cut
        .map_or("no weight cut".to_string(), |w| format!("weight cut {w:e}"));
    format!("({}, {}), {prune}, {cut}", g.n_radial, g.n_angular)
}

/// `(name, detail)` of every approximation in effect, in a fixed order.
fn approximations(
    mol: &Molecule,
    result: &ScfResult,
    config: &RhfConfig,
) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    // Density fitting: what the solver recorded it ran, not what was asked.
    let route = result.df_jk.as_ref();
    if let Some(a) = route.and_then(|r| r.j_aux.as_deref()) {
        out.push(("ri-j", format!("aux {a}")));
    }
    if let Some(a) = route.and_then(|r| r.k_aux.as_deref()) {
        out.push(("ri-k", format!("aux {a}")));
    }
    if let Some((a, w)) = route.and_then(|r| r.rsh_k.as_ref()) {
        out.push((
            "ri-k",
            format!("range-separated SR/LR pair, aux {a}, omega {w}"),
        ));
    }
    let (k_used, rsh) = exchange_used(config);
    let k_fitted = route.is_some_and(|r| r.k_aux.is_some() || r.rsh_k.is_some());
    if k_used && !rsh && !k_fitted {
        match config.k_builder.as_deref() {
            Some("link") => out.push(("link", "LinK screened exchange".to_string())),
            Some("cosx") => {
                let fin = result.cosx_final.map_or_else(
                    || "no final pass".to_string(),
                    |f| format!("final pass on {} points reported", f.npts_final),
                );
                out.push((
                    "cosx",
                    format!(
                        "seminumerical exchange, grid {}, {fin}",
                        grid_text(&config.cosx.grid)
                    ),
                ));
            }
            _ => {}
        }
    }
    if let Some(xc) = config.xc.as_deref() {
        let grids = crate::ks_gradient::KsGradGrids::from_config(config);
        out.push(("xc-grid", format!("{xc} on {}", grid_text(&grids.main))));
        let vv10 = ferric_dft::libxc::xc_def_from_name(xc).is_ok_and(|d| d.vv10.is_some());
        if vv10 {
            out.push(("nlc-grid", grid_text(&grids.nlc)));
        }
    }
    let shift = crate::driver::effective_level_shift(config);
    if shift > 0.0 {
        let who = if config.level_shift == 0.0 {
            "auto-applied for meta-GGA"
        } else {
            "user-set"
        };
        out.push((
            "level-shift",
            format!("{shift} Ha virtual-block shift, {who} (ramped to 0 at convergence)"),
        ));
    }
    if let Some(s) = config.smearing_sigma {
        out.push(("smearing", format!("sigma {s} Ha, fractional occupations")));
    }
    let n_ecp = mol.atoms.iter().filter(|a| a.n_core_ecp > 0).count();
    if n_ecp > 0 {
        let core: i32 = mol.atoms.iter().map(|a| a.n_core_ecp.max(0)).sum();
        out.push((
            "ecp",
            format!("{n_ecp} atom(s) with an ECP, {core} core electrons replaced"),
        ));
    }
    if let Some(ext) = config.external_potential.as_ref() {
        let field = ext
            .field
            .map_or(String::new(), |f| format!(", field {f:?} a.u."));
        out.push((
            "external-potential",
            format!(
                "{} point, {} smeared charge(s){field}",
                ext.point_charges.len(),
                ext.smeared_charges.len()
            ),
        ));
    }
    if let Some(p) = config.polarizable.as_ref() {
        out.push((
            "polarizable-embedding",
            format!("{} Thole site(s)", p.sites.len()),
        ));
    }
    if let Some(cm) = config.cosmo.as_ref() {
        out.push(("cosmo", format!("epsilon {}", cm.epsilon)));
    }
    if let Some(p) = config.pcm.as_ref() {
        out.push(("pcm", format!("IEF-PCM, epsilon {}", p.epsilon)));
    }
    out
}

/// Warnings from the stability verdicts on the result. Open-shell
/// unrestricted results that were not checked get the saddle caveat; a
/// checked result warns unless the verdict is `Stable`.
fn stability_warnings(c: &mut Certificate, result: &ScfResult) {
    let mut checked = |r: &StabilityResult| {
        if r.verdict() != StabilityVerdict::Stable {
            c.push_warning(format!("stability {}", r.summary()));
        }
    };
    if let Some(r) = result.stability.as_ref() {
        checked(r);
    }
    if let Some(r) = result.stability_external.as_ref() {
        checked(r);
    }
    if result.spin == Spin::Unrestricted && result.stability.is_none() {
        c.push_warning(
            "internal stability not checked (check_stability = false): a converged UHF/UKS \
             can be a saddle point of the orbital Hessian",
        );
    }
}
