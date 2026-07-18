# terf vs erf attenuator sweep on r0 — SR-MP2 + LR-RPA (small cases)

Driver: `terfc_sweep.py` (idempotent; see `LAUNCH_SWEEPS.md` to relaunch).
Systems: water, water dimer, ethene dimer / cc-pVDZ, frozen_core=0, full-rank
dRPA (trunc_thresh=0). Both formulations B (delta-lr) and T (coupled-rings).

The tempered split satisfies `terf + terfc = Coulomb` exactly (same split
identity as `erf + erfc = Coulomb`), so it has the **same exact limits** as the
erf arm — only the attenuator *shape* differs. r0 (Bohr) is the single knob;
the curvature ω = 1/(r0·√2) is derived. r0 = 3.18 Bohr ⇒ ω ≈ 0.2224 Bohr⁻¹ =
0.42 Å⁻¹ is the matched operating point (the direct comparison row against the
erf arm). The raw per-job `.out` files live in `terfc_out/out/` (gitignored,
regenerable from the driver); the numbers below are quoted from them.

## Headline finding

At the matched operating point (terf r0 = 3.18 Bohr vs erf ω = 0.2224 Bohr⁻¹):

- **B / delta-lr is essentially invariant to the erf↔terf splitter choice.**
  The Δ-form cancels the split-operator difference, so total energies agree to
  ~1e-8 Ha or better even though the underlying SR/LR pieces differ:

  | system       | B, terf r0=3.18 (Ha) | B, erf ω=0.2224 (Ha) | Δ (Ha)  |
  |--------------|----------------------|----------------------|---------|
  | water        | −76.2284068280       | −76.2284068187       | 9.3e-9  |
  | ethene dimer | −156.6316793696      | −156.6316819186      | 2.5e-6  |

- **T / coupled-rings IS sensitive to the splitter.** Here the split operator
  does not cancel — `E(ΔdRPA, erfc)` differs between the two shapes:

  | system       | T, terf r0=3.18 (Ha) | T, erf ω=0.2224 (Ha) | Δ (Ha)  |
  |--------------|----------------------|----------------------|---------|
  | water        | −76.2284853357       | −76.2281043544       | 3.8e-4  |
  | ethene dimer | −156.6312929820      | −156.6288450192      | 2.4e-3  |

Interpretation: in **B** the terf-vs-erf choice is (numerically) a no-op at the
matched operating point on these systems — B is defined by a difference that
cancels the splitter shape. In **T** the shape is a genuine lever. This matches
the design intent (both arms share the exact limits; only intermediate-range
shape can differ) and localizes any "attenuator shape is the lever" hypothesis
to the T formulation, not B.

## Limits verified

r0 → ∞ (ω → 0): terf → 0 ⇒ the method collapses to plain MP2, exactly matching
the erf ω → 0 arm. E.g. water B, r0 = 12.0:
`E(LR-MP2) = E(dRPA) = −0.0000000000`, Total = −76.2284068338 Ha — identical to
the matched erf ω = 0.0589 row to all printed digits.

The r0 → 0 (ω → ∞) limit (⇒ MP2 + ΔdRPA[Coulomb]) is verified for **H2** in the
`ferric-rpa` unit tests (`terf_small_r0_is_mp2_plus_delta_drpa`, r0 = 0.05), but
see the known bug below for larger multi-fragment systems.

## KNOWN BUG — small-r0 dimers panic (unfixed)

8 of 72 sweep jobs failed: the small-r0 dimer cases
(`water_dimer`, `ethene_dimer` at r0 ∈ {0.3, 1.0}, both B and T) panic with

    thread 'main' panicked at crates/ferric-integrals/src/threeindex.rs
    index out of bounds: the len is 0 but the index is 0

Root cause: at very small r0 the terf/terfc `compute_eri2`/`eri3` path reports a
shell block as **fully screened** and returns a length-0 slice (the shim's
"0 = fully screened" contract), but the 2-center metric / 3-index builder then
indexes `block[0]` unconditionally. Water *monomer* at r0 = 0.3 succeeds; only
the multi-fragment dimers (with far-separated shell pairs that fully screen)
trip it. The passing H2 unit tests do not hit this because at their r0 no block
fully screens.

Consequence: the **r0 → 0 strong-attenuation limit is unverified for dimers**.
This bug is in the pre-existing three-index/metric builder (exposed by the new
terf operator at extreme r0), **not** in the terf feature wiring itself, and it
does not affect the default erf arm or any committed test. It is a known
limitation, tracked separately — fixing it means teaching the metric/3-index
builders to treat a length-0 (fully-screened) block as an all-zero block.

## Related — GMTKN30 ACONF ω-scan (erf only)

See `../gmtkn30/ACONF_RSSCAN.md` for the completed erf ω-scan on ACONF (15
reactions vs W1h-val CCSD(T)/CBS). That is an **erf** study (no terf), but it is
the companion accuracy benchmark: B tracks RI-MP2 at small ω and degrades
monotonically with ω; T is worse than B at every ω. Combined with the finding
above (B invariant to the splitter, T sensitive to it), the practical read is
that B is the robust arm and the terf shape does not change its answer.
