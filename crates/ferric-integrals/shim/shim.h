#ifndef GOSCF_LIBINTSHIM_H
#define GOSCF_LIBINTSHIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes returned by shim functions. */
#define GOSCF_OK         0
#define GOSCF_EINVAL    -1
#define GOSCF_ENOTIMPL  -2
#define GOSCF_EINTERNAL -3

/* Opaque handles. */
typedef struct goscf_engine goscf_engine;
typedef struct goscf_basis  goscf_basis;

/* Shell description passed from Go to C when building a basis set.
 * Contraction coefficients are libint-normalized by the shim before use. */
typedef struct {
    int     L;            /* angular momentum */
    int     nprim;        /* number of primitives */
    int     atom_index;   /* index into the goscf_atom array (which atomic center) */
    int     pure;         /* 0 = Cartesian, 1 = spherical harmonics */
    double *exponents;    /* length nprim, owned by caller for the duration of the create call */
    double *coefficients; /* length nprim, owned by caller for the duration of the create call */
} goscf_shell;

typedef struct {
    int    Z;             /* atomic number */
    double x, y, z;       /* Bohr */
} goscf_atom;

/* libint init/finalize must be called once per process. Idempotent. */
void goscf_libint_init(void);
void goscf_libint_finalize(void);

/* Build a basis set from caller-owned arrays. Returns NULL on error. */
goscf_basis *goscf_basis_create(const goscf_shell *shells, int nshells,
                                const goscf_atom *atoms, int natoms);
void         goscf_basis_destroy(goscf_basis *bs);
int          goscf_basis_nbasis(const goscf_basis *bs);
int          goscf_basis_nshells(const goscf_basis *bs);
/* Writes nfunc per shell into out_nfunc_per_shell, length goscf_basis_nshells.
 * nfunc = (2L+1) for pure/spherical shells, (L+1)(L+2)/2 for Cartesian. */
void         goscf_basis_shell_dims(const goscf_basis *bs, int *out_nfunc_per_shell);
/* Writes the maximum primitive count and maximum L across all shells. */
void         goscf_basis_max_dims(const goscf_basis *bs, int *out_max_nprim, int *out_max_L);

/* Engine creation. op_kind:
 *   0 = Coulomb            (1/r12)              -- v1
 *   1 = ErfCoulomb         (erf(omega r)/r)     -- v1
 *   2 = ErfcCoulomb                              -- post-v1, returns NULL
 *   3 = Yukawa                                   -- post-v1, returns NULL
 * For one-electron integrals, op_kind is one of:
 *   100 = Overlap, 101 = Kinetic, 102 = Nuclear
 * (Two-electron and one-electron engines are constructed via the same factory.)
 */
goscf_engine *goscf_engine_create(int op_kind, double omega,
                                  int max_nprim, int max_L, double precision);
/* deriv_order=1 derivative engine. Returns NULL if libint2 was built without derivative support. */
goscf_engine *goscf_engine_create_deriv(int op_kind, double omega,
                                        int max_nprim, int max_L, double precision);
void          goscf_engine_destroy(goscf_engine *eng);

/* For Nuclear-attraction engines, set the array of point charges.
 * Centers and Z values are caller-owned. */
int  goscf_engine_set_point_charges(goscf_engine *eng,
                                    const goscf_atom *atoms, int natoms);

/* Compute one shell-pair block of a one-electron operator. Writes into out
 * (caller-allocated, sized n1*n2 doubles where n1, n2 are the shells'
 * function counts). Returns the number of doubles written. */
int goscf_compute_1e_block(goscf_engine *eng, const goscf_basis *bs,
                           int sh1, int sh2, double *out);

/* Compute one shell-quartet (sh1 sh2 | sh3 sh4). Writes n1*n2*n3*n4 doubles
 * into out in row-major (i j k l). Returns n_written, or 0 if libint screened. */
int goscf_compute_eri_quartet(goscf_engine *eng, const goscf_basis *bs,
                              int sh1, int sh2, int sh3, int sh4,
                              double *out);

/* Compute the shell-pair Schwarz Q matrix, Q[i,j] = sqrt(max |(ij|ij)|).
 * out is caller-allocated row-major (nshells, nshells). */
void goscf_compute_schwarz(goscf_engine *eng, const goscf_basis *bs, double *qmat);

/* --- First derivative integrals (requires libint2 with LIBINT2_MAX_DERIV_ORDER >= 1) --- */

/* Compute first derivative of a 1e shell-pair block. The engine must have been
 * created with goscf_engine_create_deriv(). Writes 6 blocks (dx1,dy1,dz1,dx2,dy2,dz2)
 * of n1*n2 doubles each into out (total 6*n1*n2). Returns 6*n1*n2 on success, 0 if screened. */
int goscf_compute_1e_deriv_block(goscf_engine *eng, const goscf_basis *bs,
                                 int sh1, int sh2, double *out);

/* Compute first derivative of a 2e shell quartet. Writes 12 blocks
 * (dx1,dy1,dz1,dx2,dy2,dz2,dx3,dy3,dz3,dx4,dy4,dz4) of n1*n2*n3*n4 doubles each.
 * Returns 12*n1*n2*n3*n4 on success, 0 if screened. */
int goscf_compute_eri_deriv_quartet(goscf_engine *eng, const goscf_basis *bs,
                                    int sh1, int sh2, int sh3, int sh4, double *out);

#ifdef __cplusplus
}
#endif
#endif
