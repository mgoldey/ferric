//! [`Certificate`](ferric_core::certificate::Certificate)s of periodic results. No periodic capability has a row in
//! the `method.kind` matrix of `site/src/reference/validation.md`, so every
//! periodic certificate is [`Grade::Ungraded`](ferric_core::certificate::Grade::Ungraded)
//! with no grade source; it records the J/K route, the auxiliary basis, the
//! exchange-divergence treatment, the k-point mesh and non-convergence.

use ferric_core::certificate::Certificate;

/// The certificate of a periodic result named `capability`
/// (`"rhf-gamma"`, `"mp2-kpts"`, ...). `jk` is the J/K route
/// (`"dense"` or `"rsgdf"`), `mesh` the k-point mesh and its centring
/// (`None` for a Γ-point run).
pub fn pbc_certificate(
    capability: &str,
    jk: &str,
    auxbasis: Option<&str>,
    exxdiv: &str,
    mesh: Option<((usize, usize, usize), &str)>,
    converged: bool,
    iterations: usize,
) -> Certificate {
    let jk_detail = match auxbasis {
        Some(a) => format!("{jk}, aux {a}"),
        None => jk.to_string(),
    };
    let mut c = Certificate::ungraded(capability)
        .with_approximation("jk", jk_detail)
        .with_approximation("exxdiv", exxdiv);
    c = match mesh {
        Some(((a, b, d), centring)) => {
            c.with_approximation("kpoints", format!("{a}x{b}x{d} mesh, {centring}"))
        }
        None => c.with_approximation("kpoints", "Gamma point only"),
    };
    if !converged {
        c.push_warning(format!("did not converge ({iterations} iterations)"));
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::certificate::Grade;

    #[test]
    fn periodic_certificates_are_ungraded_and_record_the_route() {
        let g = pbc_certificate(
            "rhf-gamma",
            "rsgdf",
            Some("def2-svp-ri"),
            "ewald",
            None,
            true,
            9,
        );
        assert_eq!(g.capability, "rhf-gamma");
        assert_eq!(g.grade, Grade::Ungraded);
        assert_eq!(g.grade_source, None);
        let d: Vec<&str> = g.approximations.iter().map(|a| a.detail.as_str()).collect();
        assert_eq!(d, ["rsgdf, aux def2-svp-ri", "ewald", "Gamma point only"]);
        assert!(g.warnings.is_empty());
        let k = pbc_certificate(
            "mp2-kpts",
            "dense",
            None,
            "none",
            Some(((2, 2, 1), "gamma-centred")),
            false,
            50,
        );
        assert_eq!(k.approximations[2].detail, "2x2x1 mesh, gamma-centred");
        assert_eq!(k.warnings, ["did not converge (50 iterations)"]);
    }
}
