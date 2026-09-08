//! QQR distance-dependent integral screening bounds.
//!
//! Refines Schwarz estimates with inverse-distance decay for well-separated
//! shell pairs. Reference: Maurer, Lambrecht, Ochsenfeld, JCP 136, 144107 (2012).

use crate::screening::{Bound, SchwarzBounds};
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;

/// QQR (distance-dependent) integral screening bounds.
///
/// For each shell pair (i,j), stores:
/// - **pair center**: weighted average of shell origins, using the most diffuse exponent
/// - **pair extent**: spatial width `1/sqrt(alpha_i_min + alpha_j_min)`
///
/// The bound is
///
/// ```text
///   |(ij|kl)|  <=  Q(i,j) * Q(k,l) * min(1, SAFETY * ext_sum / R_eff)
/// ```
///
/// where `ext_sum = ext_ij + ext_kl` and `R_eff = max(0, R - ext_sum)` is the
/// EDGE-TO-EDGE separation between the two pair charge clouds (`R` being the
/// distance between their charge centers).
///
/// # Why this form (the earlier one was an INVALID bound)
///
/// `(ij|kl)` is the Coulomb interaction of two Gaussian charge clouds, so its
/// leading (monopole) term decays as `1/R`. The distance factor must therefore
/// scale as `1/R_eff` with an ext_**SUM** numerator — NOT the `ext_ij * ext_kl / R`
/// product-over-center-to-center form used here previously, which under-states
/// the cloud charges and collapses far too fast. The numerator being `ext_sum`
/// makes the factor continuous and equal to 1 at contact (`R = ext_sum`), so it
/// reduces smoothly to plain Schwarz for overlapping/penetrating clouds where
/// the multipole expansion is not valid at all.
///
/// This mirrors [`ferric_integrals::qqr3::QqrBounds3`], the validated 3-index
/// sibling, whose module docs record the same correction for `(P|mn)`.
///
/// MEASURED against true PySCF shell-quartet integrals at benzene/cc-pVDZ over
/// 30_000 quartets (`scripts/qqr4_bound_validity.py`), worst `|true| / bound`
/// (a valid bound keeps this <= 1):
///
/// ```text
///   form                            Coulomb          erfc(omega=1.0)
///   ext*ext/R + exp(-w^2 R^2)  2.05 (13499 viol)  6.0e11 (30000 viol)
///   ext_sum/R_eff (this)       0.92 (    0 viol)  0.77   (    0 viol)
/// ```
///
/// # erfc attenuation is carried by the Schwarz factors, not a distance factor
///
/// For an `ErfcCoulomb` operator the per-operator Schwarz factors `Q(i,j)` are
/// computed with the erfc kernel and are already smaller than their Coulomb
/// counterparts — that is where the attenuation enters, and it is sufficient.
/// The previous code multiplied in an ADDITIONAL `exp(-omega^2 * R^2)`, which
/// over-suppresses genuine long-range quartets: at benzene/cc-pVDZ with
/// omega = 1.0 that factor made EVERY sampled quartet violate the bound, by up
/// to 12 orders of magnitude. The same Coulomb envelope is used for both
/// operators. (`ferric_integrals::qqr3` reached the identical conclusion
/// independently, measuring worst ratio 1.7-5.3 for its own erfc factor.)
///
/// # Not currently wired into `solve_rhf` — one measurement, narrow scope
///
/// `QqrBounds` is correct (see the anchors in
/// `crates/ferric-scf/tests/screening_exactness.rs`, which run the full
/// threshold sweep and the trivial limit against it). It is simply not used by
/// `solve_rhf` today, which builds plain [`SchwarzBounds`].
///
/// ## SCOPE OF THE MEASUREMENT BELOW — read this before citing it
///
/// The 2026-09-07 measurement was taken on LINEAR ALKANES at cc-pVDZ, and that
/// choice was not neutral: a 1-D gapped hydrocarbon chain is the FRIENDLIEST
/// case for LinK's density-pair screen, and therefore the WORST case for
/// showing QQR's marginal value on top of it. The systems were picked because
/// they were the benchmark family already in hand, NOT because they represent
/// the systems this code is used on.
///
/// So what follows is one data point about ONE composition (QQR layered under
/// LinK) on ONE molecular topology and ONE basis family. It is not a verdict on
/// QQR, and it is not evidence about:
///
/// * small-gap / slow-density-decay systems, where `dp.partners` stays widest
///   and a geometric envelope would retain the most territory. UNMEASURED.
///   (3-D extended topology and diffuse bases WERE subsequently measured — see
///   the scope-control table below; the dilution holds there.)
/// * any builder that is not LinK. The literature's QQR (Maurer, Lambrecht,
///   Ochsenfeld, JCP 136, 144107 (2012)) is applied where no LinK-style
///   pair-list prune precedes it, and in that regime the benefit REPRODUCES
///   here — see the full-population column below (3.83%). Nothing in this note
///   contradicts the published method.
/// * [`ferric_integrals::qqr3::QqrBounds3`], which screens `(P|mn)` with no
///   pair-list prune ahead of it. This finding does not transfer to it.
///
/// If you are here because you want QQR in the exchange path for a system that
/// is not a linear alkane, the measurement below does NOT answer your question
/// and should not be used to close it.
///
/// ## The measurement
///
/// alkane_16 / cc-pVDZ (50 atoms, 198 shells, 38.7 Bohr — past the ~30 Bohr
/// locality onset), one fixed converged density, PSI-clean, best of 3:
///
/// ```text
///   thresh   bound     K build   quartets computed   extra screened
///   1e-8     Schwarz    7.574s        16_184_971
///   1e-8     QQR        7.585s        16_183_454          0.009%
///   1e-10    Schwarz   10.270s        24_228_150
///   1e-10    QQR       10.640s        24_227_868          0.001%
/// ```
///
/// QQR was never faster, at any size or threshold tested (alkane_8 and
/// alkane_16, 1e-8 and 1e-10). The extra table costs ~0.4-0.5 ms to build,
/// which is negligible — the point is that it buys nothing to offset even that.
///
/// ## Why the 2-5% benefit does not survive contact with LinK
///
/// The benefit IS real over the full quartet population, and reproduces:
/// at alkane_16/1e-8 QQR screens 3.83% more of all 195_368_250 unique quartets.
/// But that is not the population LinK walks. Restricted to quartets LinK's
/// pair lists admit, the same comparison gives 0.273% — and only 0.009% survives
/// to the innermost test (`tests/qqr_population_gap.rs`).
///
/// Two independent reasons, both structural:
///
/// 1. **The pair lists cannot see QQR at all.** Both `SignificantPairs::build`
///    and `DensityPairs::build` screen on `estimate(i, j, i, j)` — a DIAGONAL
///    quartet, bra pair == ket pair, so the separation between pair centers is
///    zero and the envelope is identically 1. QQR is bit-identical to Schwarz
///    there, proven over all 102² pairs by `tests/qqr_diagonal_noop.rs`. So
///    swapping the bound cannot shrink either pair list; it acts only on LinK's
///    innermost quartet test.
/// 2. **LinK already screens on distance, by another mechanism.** The ket loop
///    is restricted to `sp.partners(ish)` ∩ `dp.partners(jsh)`. A distant
///    bra/ket pairing survives that intersection only if both pairs are locally
///    significant AND the density couples them. That is the same long-range
///    population QQR's envelope targets, so the second screen finds almost
///    nothing the first has not already removed. The two are redundant.
///
/// ## Where QQR would still pay
///
/// The dilution is caused by LinK's pair-list prune specifically, so a builder
/// that walks the quartet space without one would see far more of the 2-4%.
/// The dense `build_jk` is such a builder, but it is NOT a drop-in candidate:
/// it takes a concrete `&SchwarzBounds` and screens against `bounds.q`
/// directly, rather than through the `Bound` trait, so pointing QQR at it means
/// refactoring its signature first. It is also not the production exchange
/// path, so that refactor was not attempted — and note the benefit there is
/// unmeasured: the 3.83% figure above is a bound-vs-bound quartet count, not a
/// measured wall-clock win against `build_jk`'s own screening.
///
/// The 3-index sibling [`ferric_integrals::qqr3::QqrBounds3`] is a different
/// case again: it screens `(P|mn)` where no LinK-style pair-list prune
/// precedes it, so this finding does not transfer to it.
///
/// ## What would actually settle this
///
/// The dilution above is a statement about how much territory LinK's
/// density-pair screen has ALREADY taken. That fraction is system-dependent, so
/// the open question is not "does QQR work" but "where is `dp.partners` wide
/// enough to leave QQR something to do". The axes to vary, none of them tested:
///
/// * topology: 3-D / globular (water clusters, benzene dimer) vs the 1-D chain
///   measured here;
/// * basis diffuseness: aug-cc-pVDZ and larger, where significant pairs reach
///   much further and both LinK pair lists widen;
/// * HOMO-LUMO gap: slow density-matrix decay keeps `dp.partners` wide, which
///   is precisely the regime where a geometric envelope still has value.
///
/// ## Scope control — MEASURED, and the dilution holds off the alkane series
///
/// The above was written as an open question; it has since been measured, so
/// the alkane caveat is narrowed rather than left standing. Quartet counts
/// only (deterministic bound evaluations, unaffected by machine load):
///
/// ```text
///   system / basis                    thresh   full popn   LinK popn   dilution
///   alkane_16 / cc-pVDZ                1e-8      3.830%      0.273%       14x
///   benzene_dimer_T / cc-pVDZ          1e-8      0.985%      0.059%       17x
///   benzene_dimer_T / cc-pVDZ          1e-10     0.819%      0.015%       55x
///   benzene_dimer_T / aug-cc-pVDZ      1e-8      0.387%      0.023%       17x
/// ```
///
/// The T-shaped benzene dimer is 3-D, extended and non-chain, and aug-cc-pVDZ
/// widens both LinK pair lists — the conditions under which LinK's density
/// screen should be WEAKEST and QQR should retain the most territory. The
/// dilution persists at 17-55x. So the redundancy is a property of the
/// composition (a distance envelope layered under a pair-list prune that
/// already cuts on distance), NOT of linear alkanes.
///
/// Also recorded, because it is a trap: BARE BENZENE IS NOT A VALID CONTROL
/// here. It is compact enough (~9.4 Bohr) to have essentially no far field —
/// QQR screens only 0.030% of even the FULL population at cc-pVDZ — so it
/// cannot distinguish "LinK already screened the far field" from "there was no
/// far field". A control system must HAVE long range before the question is
/// meaningful.
///
/// Still unmeasured, and still not covered by any of this: small-gap /
/// slow-density-decay systems, and any builder other than LinK.
#[derive(Debug, Clone)]
pub struct QqrBounds {
    schwarz: SchwarzBounds,
    /// Pair centers, indexed as `pair_centers[i * nshells + j]`.
    pair_centers: Vec<[f64; 3]>,
    /// Pair extents, indexed as `pair_extents[i * nshells + j]`.
    pair_extents: Vec<f64>,
    /// The two-electron operator, used for operator-aware decay.
    op: Operator,
    nshells: usize,
}

/// Multiplier on the bare `ext_sum / R_eff` monopole envelope.
///
/// The bare envelope is a MODEL, not a rigorous bound: it can decay slightly
/// faster than the true integral in the near-intermediate zone, where the
/// dipole/quadrupole terms the monopole model omits are still appreciable.
/// [`ferric_integrals::qqr3`] measured its 3-index analogue under-estimating by
/// up to 5.9% there and adopted a flat 1.10 multiplier.
///
/// The 4-center prototype sweep (`scripts/qqr4_bound_validity.py`, water +
/// benzene at cc-pVDZ, Coulomb and erfc(1.0)) found the bare form ALREADY valid
/// on that sample — minimum sufficient factor 1.000, worst |true|/bound 0.9194.
/// We nonetheless keep 1.10, matching the sibling: those systems do not probe
/// the near-intermediate zone as densely as qqr3's 797_568-triple sweep did,
/// and an unjustified 1.0 would make validity depend on the sample happening to
/// miss the worst case. The cost is small — bounds inflate ~10% only where the
/// envelope is actually engaged (mean bound/Schwarz 0.87 at 1.10 vs 0.79 bare,
/// on benzene's separated subset).
///
/// Re-derive with `scripts/qqr4_bound_validity.py` if the extent/center
/// definitions change.
const SAFETY_FACTOR: f64 = 1.10;

impl QqrBounds {
    /// Compute QQR bounds from an existing Schwarz bound, molecule, basis, and prepared basis.
    ///
    /// Needs the raw `BasisSet` to access per-shell exponents (the minimum exponent of
    /// each shell determines the pair center and extent). The `Molecule` and `PreparedBasis`
    /// provide atom coordinates and the shell-to-atom mapping.
    /// Infallible wrapper over [`Self::try_new`], for call sites that have
    /// already established the system fits (the tests in this crate). Panics
    /// with the budget breakdown if the dense pair tables do not fit — prefer
    /// [`Self::try_new`] anywhere the size is not known in advance.
    pub fn new(
        schwarz: SchwarzBounds,
        mol: &Molecule,
        bs: &BasisSet,
        prep: &PreparedBasis,
        op: Operator,
    ) -> Self {
        Self::try_new(schwarz, mol, bs, prep, op)
            .unwrap_or_else(|e| panic!("QqrBounds::new: {e}"))
    }

    /// Fallible constructor: refuses up front when the dense `nshells²` pair
    /// tables would not fit the memory budget (defect E).
    ///
    /// The tables are DENSE over ordered shell pairs by design — the screening
    /// predicate's value comes from an O(1) `i * nshells + j` lookup, and that
    /// layout is deliberately NOT changed here. What was missing is any check:
    /// `vec![[0.0; 3]; nsh * nsh]` and `vec![0.0; nsh * nsh]` were allocated
    /// with no reference to the budget, so an oversized system met the OOM
    /// killer instead of an error naming the term.
    ///
    /// Per ordered pair: `[f64; 3]` center (24 B) + `f64` extent (8 B) = 32 B.
    pub fn try_new(
        schwarz: SchwarzBounds,
        mol: &Molecule,
        bs: &BasisSet,
        prep: &PreparedBasis,
        op: Operator,
    ) -> Result<Self, ferric_core::FerricError> {
        let nsh = prep.nshells();
        let pair_bytes = nsh
            .saturating_mul(nsh)
            .saturating_mul(std::mem::size_of::<[f64; 3]>() + std::mem::size_of::<f64>());
        ferric_core::memory::check_alloc(
            &format!("QQR pair tables (nshells={nsh} ordered pairs, dense by design)"),
            pair_bytes,
            ferric_core::memory::resolve_budget_bytes(None),
        )?;

        // Collect the minimum exponent per shell and the atom coordinates per shell.
        // We iterate atoms in order, collecting shells per atom, mirroring PreparedBasis::new.
        let mut min_exponents: Vec<f64> = Vec::with_capacity(nsh);
        let mut shell_origins: Vec<[f64; 3]> = Vec::with_capacity(nsh);

        for atom in &mol.atoms {
            let shells = bs.for_element(atom.z).unwrap();
            for sh in shells {
                let alpha_min = sh.exponents.iter().cloned().fold(f64::INFINITY, f64::min);
                min_exponents.push(alpha_min);
                shell_origins.push([atom.x, atom.y, atom.zpos]);
            }
        }
        assert_eq!(min_exponents.len(), nsh, "shell count mismatch");

        // Build pair centers and extents.
        let mut pair_centers = vec![[0.0; 3]; nsh * nsh];
        let mut pair_extents = vec![0.0; nsh * nsh];

        for i in 0..nsh {
            let ai = min_exponents[i];
            let ri = shell_origins[i];
            for j in 0..nsh {
                let aj = min_exponents[j];
                let rj = shell_origins[j];
                let sum_alpha = ai + aj;
                let idx = i * nsh + j;

                // Weighted pair center: R_ij = (alpha_i * R_i + alpha_j * R_j) / (alpha_i + alpha_j)
                pair_centers[idx] = [
                    (ai * ri[0] + aj * rj[0]) / sum_alpha,
                    (ai * ri[1] + aj * rj[1]) / sum_alpha,
                    (ai * ri[2] + aj * rj[2]) / sum_alpha,
                ];
                // Pair extent: epsilon_ij = 1 / sqrt(alpha_i + alpha_j)
                pair_extents[idx] = 1.0 / sum_alpha.sqrt();
            }
        }

        Ok(QqrBounds {
            schwarz,
            pair_centers,
            pair_extents,
            op,
            nshells: nsh,
        })
    }

    /// Access the underlying Schwarz bounds.
    pub fn schwarz(&self) -> &SchwarzBounds {
        &self.schwarz
    }

    /// The operator associated with these bounds.
    pub fn op(&self) -> Operator {
        self.op
    }

    /// Number of shells.
    pub fn nshells(&self) -> usize {
        self.nshells
    }

    /// Pair center for shell pair (i,j).
    pub fn pair_center(&self, i: usize, j: usize) -> [f64; 3] {
        self.pair_centers[i * self.nshells + j]
    }

    /// Pair extent for shell pair (i,j).
    pub fn pair_extent(&self, i: usize, j: usize) -> f64 {
        self.pair_extents[i * self.nshells + j]
    }
}

impl Bound for QqrBounds {
    fn estimate(&self, sh1: usize, sh2: usize, sh3: usize, sh4: usize) -> f64 {
        let schwarz_est = self.schwarz.estimate(sh1, sh2, sh3, sh4);
        let nsh = self.nshells;
        let idx_bra = sh1 * nsh + sh2;
        let idx_ket = sh3 * nsh + sh4;
        let c_bra = &self.pair_centers[idx_bra];
        let c_ket = &self.pair_centers[idx_ket];

        let dx = c_bra[0] - c_ket[0];
        let dy = c_bra[1] - c_ket[1];
        let dz = c_bra[2] - c_ket[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();

        if r < 1e-14 {
            // Overlapping pair centers: QQR reduces to Schwarz.
            return schwarz_est;
        }

        // QQR multipole distance envelope; see the type-level docs for why this
        // is `ext_sum / R_eff` and not the old `ext_ij * ext_kl / R`.
        let ext_sum = self.pair_extents[idx_bra] + self.pair_extents[idx_ket];
        // Edge-to-edge separation: while the clouds overlap (r <= ext_sum) the
        // integral is dominated by the penetration region, where the 1/R decay
        // has not set in, so the factor stays 1 and the bound is plain Schwarz.
        let r_eff = (r - ext_sum).max(0.0);
        let decay = if r_eff > 0.0 {
            (SAFETY_FACTOR * ext_sum / r_eff).min(1.0)
        } else {
            1.0
        };

        // NOTE: deliberately NO operator-specific distance factor. For an
        // ErfcCoulomb operator the attenuation already lives in the erfc Schwarz
        // factors `Q(i,j)`; an extra exp(-omega^2 R^2) here made the bound
        // invalid at every sampled quartet (see the type-level docs).
        schwarz_est * decay
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screening::{Bound, SchwarzBounds};
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;

    fn water_qqr() -> (QqrBounds, usize) {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let schwarz = SchwarzBounds::compute(op, &prep).unwrap();
        let nsh = prep.nshells();
        let qqr = QqrBounds::new(schwarz, &mol, &bs, &prep, op);
        (qqr, nsh)
    }

    /// THE validity anchor: `estimate` must be a TRUE UPPER BOUND on the actual
    /// integral, for every shell quartet.
    ///
    /// This replaces a `QQR <= Schwarz` assertion that could never have caught
    /// the defect it was nominally guarding. That test only checked TIGHTNESS,
    /// and under-estimation — the one failure mode that makes a screening bound
    /// unsound — is exactly what it rewarded: the more severely the bound
    /// collapsed, the more comfortably it passed. The old `ext*ext/R` form
    /// under-estimated the true integral by up to 2.05x under Coulomb and 6e11x
    /// under erfc(1.0) (measured at benzene/cc-pVDZ) while passing that test at
    /// every one of those quartets.
    ///
    /// `QQR <= Schwarz` is still asserted below, but as a secondary tightness
    /// property — never as the validity criterion.
    fn assert_estimate_bounds_true_integral(path: &str, basis: &str, op: Operator) {
        let mol = Molecule::load_xyz(path).unwrap();
        let bs = basis::bundled(basis).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let schwarz = SchwarzBounds::compute(op, &prep).unwrap();
        let qqr = QqrBounds::new(schwarz, &mol, &bs, &prep, op);
        let nsh = prep.nshells();

        // Tight precision so libint does not prescreen a small-but-real quartet
        // to nothing and hand us a spuriously "satisfied" bound.
        let mut eng = ferric_integrals::engine::Engine::new_2e(op, &prep, 1e-30).unwrap();

        let mut worst_ratio = 0.0f64; // |true| / bound; must stay <= 1
        let mut worst_at = (0, 0, 0, 0);
        for i in 0..nsh {
            for j in 0..=i {
                for k in 0..=i {
                    for l in 0..=k {
                        let bound = qqr.estimate(i, j, k, l);
                        let tru = match eng.compute_quartet(&prep, i, j, k, l) {
                            Some(block) => block.iter().fold(0.0f64, |m, v| m.max(v.abs())),
                            None => 0.0,
                        };
                        // The quantity under test is the DISTANCE ENVELOPE, so
                        // the reference is the Schwarz bound this build actually
                        // has. Schwarz itself is a separate (and separately
                        // fixed) concern: ferric builds its table at libint
                        // precision 1e-14, which rounds small self-integrals a
                        // few ULP low, so `Q(i,j)*Q(k,l)` can sit a whisker under
                        // |(ij|kl)| on diagonal quartets where Cauchy-Schwarz is
                        // an equality. MEASURED at benzene/cc-pVDZ (12,0|12,0):
                        // ferric 2.213197e-9 vs true 2.214871e-9, while the SAME
                        // bound computed at full precision gives ratio
                        // 0.9999999999999998 — i.e. valid. That deficit is the
                        // `fix/schwarz-bound-validity` defect (Schwarz table
                        // precision), NOT the envelope, and is out of scope here.
                        //
                        // So we require the envelope never to push the bound
                        // below the true integral by more than whatever slack
                        // Schwarz already gave away. Any genuine envelope defect
                        // is orders of magnitude larger than this ULP-scale
                        // effect (the old form under-estimated by 2.05x under
                        // Coulomb and 6e11x under erfc).
                        let schwarz_ref = qqr.schwarz().estimate(i, j, k, l);
                        let floor = schwarz_ref.min(tru);
                        assert!(
                            bound >= floor - 1e-12 * floor.max(1.0),
                            "QQR({i},{j},{k},{l}) = {bound:.6e} UNDER-estimates \
                             min(true, Schwarz) = {floor:.6e} (true {tru:.6e}, \
                             Schwarz {schwarz_ref:.6e}) — the distance envelope \
                             is not a valid bound"
                        );
                        // Guard against a vacuous pass: a bound that is huge
                        // everywhere would satisfy the assert above trivially.
                        // Ratio is against min(true, Schwarz) for the reason
                        // given above, so the Schwarz-precision slack does not
                        // masquerade as an envelope violation.
                        if floor > 1e-14 && bound > 0.0 {
                            let ratio = floor / bound;
                            if ratio > worst_ratio {
                                worst_ratio = ratio;
                                worst_at = (i, j, k, l);
                            }
                        }
                    }
                }
            }
        }
        eprintln!(
            "{path}/{basis} {:?}: worst |true|/bound = {worst_ratio:.4} at {worst_at:?}",
            op.kind
        );
        assert!(
            worst_ratio <= 1.0 + 1e-9,
            "bound is invalid: worst |true|/bound = {worst_ratio} > 1"
        );
        // REACHABILITY: the bound must be attained closely somewhere, otherwise
        // "valid" would just mean "enormous" and the test would prove nothing.
        assert!(
            worst_ratio > 0.1,
            "worst ratio {worst_ratio} is suspiciously loose — the bound may be \
             vacuously large rather than genuinely tight"
        );
    }

    #[test]
    fn test_estimate_is_valid_upper_bound_water_coulomb() {
        assert_estimate_bounds_true_integral(
            "../../testdata/molecules/water.xyz",
            "cc-pvdz",
            Operator::coulomb(),
        );
    }

    #[test]
    fn test_estimate_is_valid_upper_bound_water_erfc() {
        // The erfc case is the one the old exp(-omega^2 R^2) factor destroyed.
        assert_estimate_bounds_true_integral(
            "../../testdata/molecules/water.xyz",
            "cc-pvdz",
            Operator::erfc(1.0),
        );
    }

    #[test]
    fn test_estimate_is_valid_upper_bound_benzene_coulomb() {
        // Benzene is the smallest system here where pair clouds actually
        // separate (~69% of pairs have R > ext_sum), so the distance envelope
        // is genuinely exercised rather than clamped to 1 by overlap.
        assert_estimate_bounds_true_integral(
            "../../testdata/molecules/benzene.xyz",
            "cc-pvdz",
            Operator::coulomb(),
        );
    }

    #[test]
    fn test_estimate_is_valid_upper_bound_benzene_erfc() {
        assert_estimate_bounds_true_integral(
            "../../testdata/molecules/benzene.xyz",
            "cc-pvdz",
            Operator::erfc(1.0),
        );
    }

    #[test]
    fn test_qqr_le_schwarz() {
        // Secondary TIGHTNESS property (not a validity check — see
        // assert_estimate_bounds_true_integral for why this cannot be one).
        let (qqr, nsh) = water_qqr();
        for i in 0..nsh {
            for j in 0..nsh {
                for k in 0..nsh {
                    for l in 0..nsh {
                        let s = qqr.schwarz().estimate(i, j, k, l);
                        let q = qqr.estimate(i, j, k, l);
                        assert!(
                            q <= s + 1e-15,
                            "QQR({i},{j},{k},{l}) = {q} > Schwarz = {s}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_qqr_tighter_for_distant_pairs() {
        // The envelope must actually ENGAGE somewhere, or the "fix" would be a
        // bound that is merely Schwarz under another name.
        //
        // This uses BENZENE, not water. Under the corrected edge-to-edge form
        // the factor is 1 unless R > ext_ij + ext_kl, and water is too small for
        // that to ever happen: at cc-pVDZ its extents are ~0.65-2.0 Bohr so
        // ext_sum is ~1.8, while its largest pair-center separation is 2.86 Bohr
        // — MEASURED 0/2211 quartets separated. Its clouds always penetrate, so
        // falling back to Schwarz there is correct behaviour, not a regression.
        // Benzene has ~69% of pair combinations separated and is the smallest
        // molecule here that probes the regime the bound exists for.
        //
        // (The old center-to-center ext*ext/R form "passed" this on water only
        // because it decayed even for overlapping clouds — the very behaviour
        // that made it an invalid bound.)
        let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let schwarz = SchwarzBounds::compute(op, &prep).unwrap();
        let nsh = prep.nshells();
        let qqr = QqrBounds::new(schwarz, &mol, &bs, &prep, op);
        let mut found_tighter = false;
        for i in 0..nsh {
            for j in 0..nsh {
                for k in 0..nsh {
                    for l in 0..nsh {
                        let s = qqr.schwarz().estimate(i, j, k, l);
                        let q = qqr.estimate(i, j, k, l);
                        if s > 1e-10 && (q / s) < 0.99 {
                            found_tighter = true;
                        }
                    }
                }
            }
        }
        assert!(found_tighter, "QQR should be strictly tighter than Schwarz for some distant pairs");
    }

    #[test]
    fn test_qqr_overlapping_equals_schwarz() {
        // Same shell on same atom: pair centers overlap, so QQR == Schwarz.
        let (qqr, _nsh) = water_qqr();
        let s = qqr.schwarz().estimate(0, 0, 0, 0);
        let q = qqr.estimate(0, 0, 0, 0);
        assert!(
            (q - s).abs() < 1e-15,
            "overlapping pairs: QQR={q} != Schwarz={s}"
        );
    }

    #[test]
    fn test_pair_center_self_pair() {
        // Self-pair center should be the atom center.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let schwarz = SchwarzBounds::compute(op, &prep).unwrap();
        let qqr = QqrBounds::new(schwarz, &mol, &bs, &prep, op);
        // Shell 0 is on atom 0 (oxygen)
        let c = qqr.pair_center(0, 0);
        let ox = mol.atoms[0].x;
        let oy = mol.atoms[0].y;
        let oz = mol.atoms[0].zpos;
        assert!((c[0] - ox).abs() < 1e-12);
        assert!((c[1] - oy).abs() < 1e-12);
        assert!((c[2] - oz).abs() < 1e-12);
    }

    #[test]
    fn test_erfc_attenuation_comes_from_schwarz_factors_not_a_distance_factor() {
        // erfc screening benefit must come ENTIRELY from the smaller erfc
        // Schwarz factors, with the distance envelope identical to Coulomb's.
        //
        // The previous version of this test built both bounds from the SAME
        // (Coulomb) Schwarz table and asserted the erfc bound was strictly
        // smaller — which could only be satisfied by an extra distance factor,
        // i.e. it actively pinned in place the exp(-omega^2 R^2) term that made
        // the bound invalid (worst |true|/bound 6.0e11 at benzene/cc-pVDZ).
        //
        // So we assert the opposite structure: (a) with each operator's own
        // Schwarz factors the erfc bound is genuinely tighter, and (b) the
        // distance envelope itself is operator-INDEPENDENT.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();

        let op_c = Operator::coulomb();
        let op_e = Operator::erfc(0.5);
        let qqr_c = QqrBounds::new(
            SchwarzBounds::compute(op_c, &prep).unwrap(), &mol, &bs, &prep, op_c);
        let qqr_e = QqrBounds::new(
            SchwarzBounds::compute(op_e, &prep).unwrap(), &mol, &bs, &prep, op_e);

        // (b) Same Schwarz table, differing only in operator => identical
        // bounds, proving no operator-dependent distance factor survives.
        let qqr_e_on_c_schwarz = QqrBounds::new(
            SchwarzBounds::compute(op_c, &prep).unwrap(), &mol, &bs, &prep, op_e);

        let nsh = prep.nshells();
        let mut found_tighter = false;
        for i in 0..nsh {
            for j in 0..nsh {
                for k in 0..nsh {
                    for l in 0..nsh {
                        let c = qqr_c.estimate(i, j, k, l);
                        let e = qqr_e.estimate(i, j, k, l);
                        assert!(
                            e <= c + 1e-15,
                            "erfc QQR({i},{j},{k},{l}) = {e} > Coulomb QQR = {c}"
                        );
                        if c > 1e-10 && (e / c) < 0.99 {
                            found_tighter = true;
                        }
                        let same = qqr_e_on_c_schwarz.estimate(i, j, k, l);
                        assert!(
                            (same - c).abs() <= 1e-15 * c.max(1.0),
                            "distance envelope is operator-dependent at \
                             ({i},{j},{k},{l}): erfc-op {same} vs Coulomb-op {c} \
                             on identical Schwarz factors"
                        );
                    }
                }
            }
        }
        assert!(
            found_tighter,
            "erfc QQR should be strictly tighter than Coulomb QQR somewhere, \
             via its smaller Schwarz factors"
        );
    }
}
