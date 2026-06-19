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

#ifdef __cplusplus
}
#endif
#endif
