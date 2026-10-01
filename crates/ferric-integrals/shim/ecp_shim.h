#ifndef FERRIC_ECP_SHIM_H
#define FERRIC_ECP_SHIM_H

/* C-ABI wrapper around libecpint's ECPIntegrator.
 *
 * Computes the dense scalar ECP matrix V_ECP[ncart][ncart] for a molecule with
 * one or more ECP centers, over all Gaussian shell pairs. Output is in
 * libecpint's *Cartesian* ordering (per shell, canonical Cartesian order
 * {x^2, xy, xz, y^2, yz, z^2} for L=2, etc.), row-major M(i,j) = i*ncart + j.
 *
 * This file is intentionally independent of the libint2 shim (shim.h/shim.cc).
 */

#ifdef __cplusplus
extern "C" {
#endif

#define FERRIC_ECP_OK         0
#define FERRIC_ECP_EINVAL    -1
#define FERRIC_ECP_EINTERNAL -3

/* One Gaussian basis shell, flattened. Mirrors the data ferric already holds:
 * a contracted shell of `nprim` primitives at center (x,y,z) in Bohr.
 * Always Cartesian on the libecpint side; ferric handles cart<->sph itself. */
typedef struct {
    int     l;            /* angular momentum */
    int     nprim;        /* number of primitives */
    double  x, y, z;      /* shell center in Bohr */
    const double *exponents;    /* length nprim */
    const double *coefficients; /* length nprim (contraction coefs as ferric stores them) */
} ferric_ecp_gshell;

/* One ECP center. The semilocal expansion is a flat list of `nterm` primitives,
 * each tagged with angular momentum `ams[k]`, r-power `ns[k]`, exponent
 * `exponents[k]`, coefficient `coefs[k]`. The channel with the maximum `am` is
 * the local term (libecpint determines this internally). */
typedef struct {
    double  x, y, z;      /* ECP center in Bohr */
    int     nterm;        /* number of primitives across all channels */
    const int    *ams;    /* length nterm: angular momentum per primitive */
    const int    *ns;     /* length nterm: r-power per primitive (BSE r_exponents) */
    const double *exponents; /* length nterm */
    const double *coefficients; /* length nterm */
} ferric_ecp_center;

/* Compute the dense Cartesian ECP matrix.
 *   shells   : nshell Gaussian shells (Cartesian)
 *   ecps     : necp ECP centers
 *   out_vecp : caller-allocated ncart*ncart doubles, row-major.
 * ncart is the total number of Cartesian functions = sum_s (l_s+1)(l_s+2)/2.
 * Returns FERRIC_ECP_OK on success, a negative error code otherwise. The caller
 * is responsible for passing a correctly-sized `out_vecp` (use
 * ferric_ecp_ncart to size it). */
int ferric_ecp_matrix(const ferric_ecp_gshell *shells, int nshell,
                      const ferric_ecp_center *ecps, int necp,
                      double *out_vecp);

/* Total number of Cartesian functions for the given shells. */
int ferric_ecp_ncart(const ferric_ecp_gshell *shells, int nshell);

/* Number of distinct atomic centers libecpint will infer from the given shells
 * and ECPs. libecpint does NOT take an atom list: it derives atom ids by
 * deduplicating shell/ECP centers with a 1e-4 Bohr tolerance (see
 * ECPIntegrator::init in libecpint/src/lib/api.cpp), assigning ids in order of
 * first appearance -- shells first, then any ECP center not already seen.
 *
 * The caller MUST use this to size the derivative buffer and to map libecpint's
 * atom ids back onto its own atom list; assuming a 1:1 correspondence with the
 * caller's atom ordering is wrong whenever an atom carries no basis shells.
 * Returns a negative error code on invalid input. */
int ferric_ecp_natoms(const ferric_ecp_gshell *shells, int nshell,
                      const ferric_ecp_center *ecps, int necp);

/* Compute the first derivatives of the Cartesian ECP matrix with respect to
 * every atomic coordinate.
 *
 *   out_derivs : caller-allocated 3*natoms*ncart*ncart doubles, where natoms is
 *                ferric_ecp_natoms(...) and ncart is ferric_ecp_ncart(...).
 *                Layout is [3*natoms][ncart][ncart], row-major throughout, in
 *                the order {A_x, A_y, A_z, B_x, B_y, B_z, ...} over libecpint's
 *                inferred atom ids.
 *   out_natoms : if non-NULL, receives the inferred natoms (cross-check).
 *
 * Each matrix is the full (symmetric) derivative of V_ECP with respect to that
 * one coordinate; the A/B/C (bra center, ket center, ECP center) contributions
 * are already summed per atom by libecpint.
 *
 * Returns FERRIC_ECP_OK on success, a negative error code otherwise. */
int ferric_ecp_matrix_deriv(const ferric_ecp_gshell *shells, int nshell,
                            const ferric_ecp_center *ecps, int necp,
                            double *out_derivs, int *out_natoms);

/* Rectangular ECP block between two INDEPENDENT shell lists at arbitrary
 * (e.g. lattice-translated) centres -- the periodic-ECP kernel. Unlike
 * ferric_ecp_matrix there is no bra/ket symmetry, no centre deduplication and
 * NO internal distance screening: every (bra shell a, ket shell b, ECP u)
 * triple enabled by `mask` is evaluated with libecpint's per-shell-pair kernel
 * (ECPIntegral::compute_shell_pair) and summed.
 *
 *   bra, nbra  : row shells (Cartesian, bare-Cartesian coefficients)
 *   ket, nket  : column shells
 *   ecps, necp : ECP centres (any positions; images allowed)
 *   mask       : NULL = every triple; else nbra*nket*necp bytes, index
 *                (a*nket + b)*necp + u, nonzero = evaluate. The caller's
 *                screen lives here, so the truncation is the caller's to
 *                report.
 *   out        : caller-allocated, out_len doubles, row-major
 *                [ncart(bra)][ncart(ket)]; overwritten (zeroed first).
 *   out_len    : MUST equal ferric_ecp_ncart(bra) * ferric_ecp_ncart(ket)
 *                (size cross-check: a mismatch returns FERRIC_ECP_EINVAL and
 *                writes nothing).
 *
 * Validates l, nprim, exponents and ECP angular momenta against
 * LIBECPINT_MAX_L BEFORE constructing the libecpint engine (whose own checks
 * are assert()s, i.e. aborts). Never lets a C++ exception cross the ABI.
 * Returns FERRIC_ECP_OK or a negative error code. */
int ferric_ecp_block(const ferric_ecp_gshell *bra, int nbra,
                     const ferric_ecp_gshell *ket, int nket,
                     const ferric_ecp_center *ecps, int necp,
                     const unsigned char *mask,
                     double *out, long long out_len);

/* First derivatives of the rectangular block of ferric_ecp_block (periodic-ECP
 * forces). For every enabled triple (bra shell a at A, ket shell b at B, ECP
 * centre u at C) the integral <a|U_u|b> depends on A, B and C; this returns
 * the TRUE partial derivatives with respect to each of the three centres:
 *
 *   out_bra   [3][ncart(bra)][ncart(ket)] += d/dA_x <a|U_u|b>       (x = 0,1,2)
 *   out_ket   [3][ncart(bra)][ncart(ket)] += d/dB_x <a|U_u|b>
 *   out_centre[ngroup][3][ncart(bra)][ncart(ket)]
 *             [centre_group[u]]           += d/dC_x <a|U_u|b> = -(d/dA_x + d/dB_x)
 *
 * summed over the enabled triples (row-major; all three outputs zeroed first).
 * The caller decides which atom each slot belongs to: every bra shell is one
 * function of its atom, every ket shell (possibly a lattice image) one
 * function of its atom, and `centre_group[u]` names the caller's group (e.g.
 * the cell atom of an ECP image) the centre derivative is accumulated under.
 * No libecpint atom inference / 1e-4 Bohr deduplication is involved.
 *
 * ON-CENTRE SHELLS (the libecpint quirk this function removes):
 * ECPIntegral::compute_shell_pair_derivative, for a shell within 1e-6 Bohr
 * (L1) of the ECP centre, skips that shell's derivative and reports
 * A = -B, C = 0 (or B = -A, C = 0). Only the per-ATOM totals are right, and
 * only because a coincident shell and centre belong to the same atom. That is
 * wrong for any caller that uses the slots separately (e.g. mutation tests
 * that drop the centre term, or a caller whose "same atom" assumption fails).
 * This function therefore does NOT call compute_shell_pair_derivative; it
 * calls the same kernel that function uses off-centre,
 * ECPIntegral::left_shell_derivative, for BOTH shells unconditionally:
 *   bra = left_shell_derivative(U, a, b), ket = left_shell_derivative(U, b, a)^T,
 *   centre = -(bra + ket)   (translation invariance of each triple).
 * Off-centre this is bitwise what compute_shell_pair_derivative returns; on
 * the centre the three slots are the true partial derivatives (and the per-
 * atom totals agree with libecpint's to roundoff).
 *
 *   bra, nbra / ket, nket / ecps, necp / mask : as ferric_ecp_block.
 *   centre_group : necp ints, each in [0, ngroup) (checked; EINVAL otherwise).
 *   ngroup       : > 0.
 *   out_bra, out_ket : 3 * out_len doubles each.
 *   out_centre   : ngroup * 3 * out_len doubles.
 *   out_len      : MUST equal ncart(bra) * ncart(ket) (checked; nothing is
 *                  written on a mismatch).
 *
 * libecpint's engine is built with deriv = 1, so max l(bra, ket) + 1 must not
 * exceed LIBECPINT_MAX_L (checked before construction: its own guards are
 * assert()s). Never lets a C++ exception cross the ABI.
 * Returns FERRIC_ECP_OK or a negative error code. */
int ferric_ecp_block_deriv(const ferric_ecp_gshell *bra, int nbra,
                           const ferric_ecp_gshell *ket, int nket,
                           const ferric_ecp_center *ecps, int necp,
                           const unsigned char *mask,
                           const int *centre_group, int ngroup,
                           double *out_bra, double *out_ket, double *out_centre,
                           long long out_len);

#ifdef __cplusplus
}
#endif
#endif
