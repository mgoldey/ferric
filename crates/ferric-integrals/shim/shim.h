#ifndef SCF_LIBINTSHIM_H
#define SCF_LIBINTSHIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes returned by shim functions. */
#define SCF_OK         0
#define SCF_EINVAL    -1
#define SCF_ENOTIMPL  -2
#define SCF_EINTERNAL -3

/* Opaque handles. */
typedef struct scf_engine scf_engine;
typedef struct scf_basis  scf_basis;

/* Shell description passed from Go to C when building a basis set.
 * Contraction coefficients are libint-normalized by the shim before use. */
typedef struct {
    int     L;            /* angular momentum */
    int     nprim;        /* number of primitives */
    int     atom_index;   /* index into the scf_atom array (which atomic center) */
    int     pure;         /* 0 = Cartesian, 1 = spherical harmonics */
    double *exponents;    /* length nprim, owned by caller for the duration of the create call */
    double *coefficients; /* length nprim, owned by caller for the duration of the create call */
} scf_shell;

typedef struct {
    double Z;             /* atomic number (fractional for external point charges) */
    double x, y, z;       /* Bohr */
} scf_atom;

/* libint init/finalize must be called once per process. Idempotent. */
void scf_libint_init(void);
void scf_libint_finalize(void);

/* Build a basis set from caller-owned arrays. Returns NULL on error. */
scf_basis *scf_basis_create(const scf_shell *shells, int nshells,
                                const scf_atom *atoms, int natoms);
void         scf_basis_destroy(scf_basis *bs);
int          scf_basis_nbasis(const scf_basis *bs);
int          scf_basis_nshells(const scf_basis *bs);
/* Writes nfunc per shell into out_nfunc_per_shell, length scf_basis_nshells.
 * nfunc = (2L+1) for pure/spherical shells, (L+1)(L+2)/2 for Cartesian. */
void         scf_basis_shell_dims(const scf_basis *bs, int *out_nfunc_per_shell);
/* Writes the maximum primitive count and maximum L across all shells. */
void         scf_basis_max_dims(const scf_basis *bs, int *out_max_nprim, int *out_max_L);

/* Engine creation. op_kind:
 *   0 = Coulomb            (1/r12)              -- v1
 *   1 = ErfCoulomb         (erf(omega r)/r)     -- v1
 *   2 = ErfcCoulomb                              -- post-v1, returns NULL
 *   3 = Yukawa                                   -- post-v1, returns NULL
 * For one-electron integrals, op_kind is one of:
 *   100 = Overlap, 101 = Kinetic, 102 = Nuclear
 * (Two-electron and one-electron engines are constructed via the same factory.)
 */
scf_engine *scf_engine_create(int op_kind, double omega,
                                  int max_nprim, int max_L, double precision);
/* deriv_order=1 derivative engine. Returns NULL if libint2 was built without derivative support. */
scf_engine *scf_engine_create_deriv(int op_kind, double omega,
                                        int max_nprim, int max_L, double precision);

/* Geminal (F12 / MP2-F12) two-electron engine. op_kind:
 *   200 = cgtg            (contracted Gaussian geminal  f12)
 *   201 = cgtg_x_coulomb  (f12 / r12)
 *   202 = delcgtg2        ([Ti,f12] kinetic commutator: |∇f12|^2)
 * The geminal is supplied as `ngauss` (exponent, coefficient) pairs
 * approximating the Slater geminal exp(-gamma r12). Arrays are caller-owned.
 * Returns NULL if libint2 was built without the G12 integral class
 * (config.h: G12_MAX_AM undefined) or op_kind is unknown. */
scf_engine *scf_engine_create_geminal(int op_kind, int ngauss,
                                          const double *exps, const double *coefs,
                                          int max_nprim, int max_L, double precision);
void          scf_engine_destroy(scf_engine *eng);

/* For Nuclear-attraction engines, set the array of point charges.
 * Centers and Z values are caller-owned. */
int  scf_engine_set_point_charges(scf_engine *eng,
                                    const scf_atom *atoms, int natoms);

/* All scf_compute_* functions catch C++ exceptions from libint (an exception
 * must never unwind across the C ABI) and return SCF_EINTERNAL (negative). */

/* Compute one shell-pair block of a one-electron operator. Writes into out
 * (caller-allocated, sized n1*n2 doubles where n1, n2 are the shells'
 * function counts). Returns the number of doubles written, or SCF_EINTERNAL. */
int scf_compute_1e_block(scf_engine *eng, const scf_basis *bs,
                           int sh1, int sh2, double *out);

/* Compute one shell-quartet (sh1 sh2 | sh3 sh4). Writes n1*n2*n3*n4 doubles
 * into out in row-major (i j k l). Returns n_written, 0 if libint screened,
 * or SCF_EINTERNAL. */
int scf_compute_eri_quartet(scf_engine *eng, const scf_basis *bs,
                              int sh1, int sh2, int sh3, int sh4,
                              double *out);

/* Compute the shell-pair Schwarz Q matrix, Q[i,j] = sqrt(max |(ij|ij)|).
 * out is caller-allocated row-major (nshells, nshells).
 * Returns SCF_OK or SCF_EINTERNAL. */
int scf_compute_schwarz(scf_engine *eng, const scf_basis *bs, double *qmat);

/* --- 3-center and 2-center ERIs for density fitting / RI --- */

scf_engine *scf_engine_create_3center(int op_kind, double omega,
                                          int max_nprim, int max_L, double precision);
scf_engine *scf_engine_create_2center(int op_kind, double omega,
                                          int max_nprim, int max_L, double precision);

/* Compute (shP | sh1 sh2) 3-center ERI. obs = orbital basis, dfbs = auxiliary basis.
 * Writes nP*n1*n2 doubles. Returns n_written or 0 if screened. */
int scf_compute_eri3(scf_engine *eng, const scf_basis *obs,
                       const scf_basis *dfbs,
                       int shP, int sh1, int sh2, double *out);

/* Compute (shP | shQ) 2-center ERI. Returns nP*nQ. */
int scf_compute_eri2(scf_engine *eng, const scf_basis *dfbs,
                       int shP, int shQ, double *out);

/* --- 3-center and 2-center ERI derivative engines (deriv_order=1) --- */

scf_engine *scf_engine_create_3center_deriv(int op_kind, double omega,
                                                int max_nprim, int max_L, double precision);
scf_engine *scf_engine_create_2center_deriv(int op_kind, double omega,
                                                int max_nprim, int max_L, double precision);

/* Compute first derivative of (shP | sh1 sh2) 3-center ERI. Writes 9 blocks
 * (3 centers × 3 coords) of nP*n1*n2 doubles each. Returns 9*nP*n1*n2 on success, 0 if screened. */
int scf_compute_eri3_deriv(scf_engine *eng, const scf_basis *obs,
                             const scf_basis *dfbs,
                             int shP, int sh1, int sh2, double *out);

/* Compute first derivative of (shP | shQ) 2-center ERI. Writes 6 blocks
 * (2 centers × 3 coords) of nP*nQ doubles each. Returns 6*nP*nQ on success, 0 if screened. */
int scf_compute_eri2_deriv(scf_engine *eng, const scf_basis *dfbs,
                             int shP, int shQ, double *out);

/* --- Electric dipole integrals via emultipole1 --- */

/* Compute electric dipole integrals ⟨μ|(r - origin)|ν⟩ for all shell pairs.
 * Returns the 3 dipole matrices (x, y, z) each of size nbas×nbas,
 * packed as out[0..nbas*nbas] = x-component, out[nbas*nbas..2*nbas*nbas] = y, etc.
 * origin[3] = {ox, oy, oz} in Bohr. Returns nbas*nbas*3 on success, -1 on error. */
int scf_compute_dipole(const scf_basis *bs, const double *origin,
                         int nbas, double *out);

/* --- First derivative integrals (requires libint2 with LIBINT2_MAX_DERIV_ORDER >= 1) --- */

/* Compute first derivative of a 1e shell-pair block. The engine must have been
 * created with scf_engine_create_deriv(). Writes 6 blocks (dx1,dy1,dz1,dx2,dy2,dz2)
 * of n1*n2 doubles each into out (total 6*n1*n2). Returns 6*n1*n2 on success, 0 if screened. */
int scf_compute_1e_deriv_block(scf_engine *eng, const scf_basis *bs,
                                 int sh1, int sh2, double *out);

/* Compute first derivative of a 2e shell quartet. Writes 12 blocks
 * (dx1,dy1,dz1,dx2,dy2,dz2,dx3,dy3,dz3,dx4,dy4,dz4) of n1*n2*n3*n4 doubles each.
 * Returns 12*n1*n2*n3*n4 on success, 0 if screened. */
int scf_compute_eri_deriv_quartet(scf_engine *eng, const scf_basis *bs,
                                    int sh1, int sh2, int sh3, int sh4, double *out);

/* --- terfc(r,r0)/r attenuated integrals via 2D interpolation tables --- *
 *
 * Exact terfc(r,r0)/r integrals using precomputed G_{m,n}(S,s) auxiliary function
 * tables and Obara-Saika recurrences. No fitting; same approach as Q-Chem.
 * (Dutoi & Head-Gordon, JPCA 2008; Goldey PhD thesis 2014.)
 *
 * table_dir: path to directory with binary table files generated by
 *            terf-tables/generate_tables.py
 *            (16_4_2.bin, 8_10_5.bin, 4_20_20.bin, 2_20_80.bin)
 *
 * omega must satisfy the curvature constraint: r0 * omega = 1/sqrt(2).
 */

scf_engine *scf_engine_create_terfc_3center(double r0, double omega,
                                               int max_nprim, int max_L,
                                               double precision,
                                               const char *table_dir);

scf_engine *scf_engine_create_terfc_2center(double r0, double omega,
                                               int max_nprim, int max_L,
                                               double precision,
                                               const char *table_dir);

/* Compute (shP|terfc|sh1 sh2). Returns nP*n1*n2, 0 if screened, SCF_EINTERNAL on error. */
int scf_compute_terfc_eri3(scf_engine *eng, const scf_basis *obs,
                              const scf_basis *dfbs,
                              int shP, int sh1, int sh2, double *out);

/* Compute (shP|terfc|shQ). Returns nP*nQ or SCF_EINTERNAL on error. */
int scf_compute_terfc_eri2(scf_engine *eng, const scf_basis *dfbs,
                              int shP, int shQ, double *out);

/* --- terf(r,r0)/r = tempered LONG-RANGE complement of terfc, via the SAME
 * 2D interpolation tables --- *
 *
 * Exact identity: terf(r,r0)/r + terfc(r,r0)/r = 1/r (Coulomb), verified to
 * machine precision because both share the identical table lookup / OS
 * recurrence / cart->pure transform; only the final combine step differs
 * (terfc subtracts this same block from Coulomb, terf returns it directly).
 *
 * Same r0/omega curvature constraint as terfc: r0 * omega = 1/sqrt(2).
 */

scf_engine *scf_engine_create_terf_3center(double r0, double omega,
                                              int max_nprim, int max_L,
                                              double precision,
                                              const char *table_dir);

scf_engine *scf_engine_create_terf_2center(double r0, double omega,
                                              int max_nprim, int max_L,
                                              double precision,
                                              const char *table_dir);

/* Compute (shP|terf|sh1 sh2). Returns nP*n1*n2, 0 if screened, SCF_EINTERNAL on error. */
int scf_compute_terf_eri3(scf_engine *eng, const scf_basis *obs,
                             const scf_basis *dfbs,
                             int shP, int sh1, int sh2, double *out);

/* Compute (shP|terf|shQ). Returns nP*nQ or SCF_EINTERNAL on error. */
int scf_compute_terf_eri2(scf_engine *eng, const scf_basis *dfbs,
                             int shP, int shQ, double *out);

#ifdef __cplusplus
}
#endif
#endif
