# Hypotheses — open-shell Z-vector near-null mode on ²Π radicals (written BEFORE any measurement)

Date: 2026-10-06. Branch `feat/degenerate-somo-zvector`, off origin/main 90f123f8.
Issue #265.

## The observation (from the issue, not re-measured yet)

UKS-PBE/STO-3G OH, MBD@rsSCS exact gradient: the unrestricted Z-vector PCG met
p·Hp = −8.4e-19 with p·Mp = 1.1e-16 at relative residual 1.4e-4. Since #261,
`pcg_curvature_ok` turns that into an error.

## A STRUCTURAL OBSERVATION MADE BEFORE MEASURING (source-read)

The issue's suggested mechanism — "a rotation within the degenerate OCCUPIED
pair" — cannot be the mode: the Z-vector space holds only occ→virt rotations
(UKS: per spin `κ_vo`; ROKS: `κ_vc, κ_vo, κ_oc`). Occ–occ rotations are not
parameters of H at all. For OH ²Π (9 electrons) the α set holds both π
orbitals (their rotation is occ–occ, absent), while the β set holds ONE π and
leaves its partner virtual. A rotation about the molecular axis turns the
occupied β π_x into the virtual β π_y: an occ→virt rotation that maps the
UKS determinant onto its rotated copy, which has the SAME energy because the
nuclei lie on the axis.

## Competing hypotheses (pre-registered, distinguishable)

**H-GOLDSTONE (physics).** The near-null mode is the orbital rotation generated
by a rotation of the electrons about the molecular axis,
`κ_L = (Cᵀ S Gᵀ C)` restricted to the Z-vector blocks, with `G` the AO
representation of the axis-rotation generator (each shell maps onto itself
because every centre is on the axis). A ²Π UKS/ROKS state breaks the
cylindrical symmetry, so `κ_L` is an exact zero mode of the orbital Hessian
in the continuum; only the anisotropy of ferric's numerical grids (the Becke–
Lebedev XC grid; COSX is not involved here) lifts it.
  PREDICTIONS
  * P1: the smallest-|eigenvalue| eigenvector of the dense H (OH/STO-3G, 13
    rotation parameters) is `κ_L` to |cos| > 0.999.
  * P2: the gap-metric Rayleigh quotient λ_L = κ_L·Hκ_L / κ_L·Mκ_L is at the
    rounding floor (≲ 1e-10) for UHF (exact K, no grid) and at the XC-grid
    anisotropy level for UKS-PBE, shrinking when the angular grid is refined
    (110 → 302). Either sign is possible for UKS.
  * P3: the remaining eigenvalues of H are O(gap) and positive.
  * P4: for a NON-linear radical (NH2) no generator exists (no projection,
    bit-identical solve); for a linear Σ / both-π-filled state (O2 ³Σg⁻,
    linear CN ²Σ⁺) κ_L vanishes to rounding (no mode to remove).
  * P5: the right-hand side of a rotationally invariant property (E_MBD) is
    orthogonal to κ_L up to the anisotropy of its own grid (MBD volume
    lattice), so the deflated solve is consistent and the dropped component's
    contribution to the gradient, Z_L·∂F_L/∂R = −Z_L·(κ_L·H dκ*/dR), is at
    the same small order.

**H-ARTIFACT (implementation).** The zero curvature is PCG noise at the
rounding floor of a well-conditioned H (A1), or the null mode is unrelated to
the symmetry (A2: e.g. a sign/convention error in the Hessian product that
makes some ordinary direction look null).
  PREDICTIONS
  * A1: the dense H's smallest eigenvalue is O(gap) ≫ 1e-6, so a PCG floor
    without a null mode; κ_L is NOT near-null.
  * A2: a near-null eigenvector exists but has |cos| ≪ 1 with κ_L, or it
    persists (same size) on NH2 / for exact-K UHF where no symmetry is broken
    by a grid.

P1+P2 vs A1/A2 is decisive: the two families predict different eigenvalue
magnitudes AND different eigenvectors.

## The fix this would justify (only if H-GOLDSTONE holds)

Detect: molecule linear (all centres on one line), `κ_L` non-vanishing, and
|λ_L| below a bar placed between the measured null (P2) and the measured
smallest non-null eigenvalue (P3). Project: deflated PCG on the complement of
`κ_L` (b, iterates and preconditioned residuals projected), Z_L := 0. Refuse:
when the right-hand side is NOT orthogonal to κ_L (a property that is not
axis-symmetric, P5 violated), and everywhere the detection does not apply
(non-linear molecules keep today's `pcg_curvature_ok` verdict).

## Anchors and validation planned

* Non-linear radical (NH2): the solver path is unchanged → bit-identical
  relaxation gradient (asserted with `==`).
* Linear molecule with no null mode (O2 ³Σg⁻): projection skipped (κ_L = 0).
* Exact gradient vs central FD of the full SCF + MBD pipeline on OH, NO, CH
  (²Π), UKS and ROKS where they converge.
* Negative control: the same solve WITHOUT projection must either be refused
  (curvature error) or miss FD — it must never pass silently.
