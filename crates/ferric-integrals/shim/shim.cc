// Implementation of the scf libint2 shim. See shim.h for the contract.
#include "shim.h"

#include <libint2.hpp>
#include <libint2/solidharmonics.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <atomic>
#include <mutex>
#include <memory>
#include <string>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <cstdio>
#include <algorithm>

using libint2::Engine;
using libint2::Operator;
using libint2::Shell;
using libint2::BasisSet;

struct scf_basis {
    BasisSet           bs;
    std::vector<int>   nfunc;       // nfunc per shell: (2L+1) if pure, (L+1)(L+2)/2 if Cartesian
    int                max_nprim;
    int                max_L;
};

// Forward declaration; the terfc table set is defined further down.
struct TerfcTableSet;

struct scf_engine {
    Engine engine;
    // --- terfc extension (unused/default for ordinary libint engines) ---
    bool                            is_terfc = false;
    // When true, the compute functions below return the "terf" complement
    // (the tempered LR piece) instead of "terfc" (coulomb - terf). Both reuse
    // the identical table set / OS machinery; only the final combine step
    // differs (see scf_compute_terf_eri3/2 vs scf_compute_terfc_eri3/2).
    bool                            is_terf_complement = false;
    double                          r0 = 0.0;
    double                          omega = 0.0;
    double                          precision = 1e-14;
    int                             max_L = 0;
    std::shared_ptr<TerfcTableSet>  terfc_tables;
};

static std::atomic<int> libint_init_count{0};

// libint2's Engine and Shell/BasisSet constructors touch process-global
// state (libint2::initialize() tables, normalization scratch) and are NOT
// reentrant. Compute on a fully-constructed Engine is thread-safe (each
// thread owns its own Engine), but *construction* must be serialized.
// Without this, many threads building engines at once (e.g. the parallel
// test binary, or several SCFs in flight) corrupt the heap. Production runs
// one SCF at a time so it rarely trips, but it is the same latent bug.
static std::mutex libint_ctor_mutex;

void scf_libint_init(void) {
    if (libint_init_count.fetch_add(1) == 0) {
        libint2::initialize();
    }
}

void scf_libint_finalize(void) {
    if (libint_init_count.fetch_sub(1) == 1) {
        libint2::finalize();
    }
}

scf_basis *scf_basis_create(const scf_shell *shells, int nshells,
                                const scf_atom *atoms, int natoms) {
    std::lock_guard<std::mutex> lock(libint_ctor_mutex);
    try {
        // Build per-atom Atom records (libint type) for nuclear positions.
        std::vector<libint2::Atom> li_atoms(natoms);
        for (int a = 0; a < natoms; ++a) {
            li_atoms[a].atomic_number = static_cast<int>(atoms[a].Z);
            li_atoms[a].x = atoms[a].x;
            li_atoms[a].y = atoms[a].y;
            li_atoms[a].z = atoms[a].z;
        }
        // Build the libint shell list, one libint::Shell per scf_shell.
        std::vector<Shell> li_shells;
        li_shells.reserve(nshells);
        for (int s = 0; s < nshells; ++s) {
            const scf_shell &g = shells[s];
            libint2::svector<double> exps(g.exponents, g.exponents + g.nprim);
            libint2::svector<double> coefs(g.coefficients, g.coefficients + g.nprim);
            libint2::svector<libint2::Shell::Contraction> contr;
            contr.push_back({g.L, g.pure != 0, coefs});
            std::array<double, 3> c{
                atoms[g.atom_index].x,
                atoms[g.atom_index].y,
                atoms[g.atom_index].z,
            };
            li_shells.emplace_back(exps, contr, c);
        }
        BasisSet bs(std::move(li_shells));
        auto *out = new (std::nothrow) scf_basis{std::move(bs), {}, 0, 0};
        if (!out) return nullptr;
        out->nfunc.reserve(out->bs.size());
        for (const auto &sh : out->bs) {
            int L = sh.contr[0].l;
            bool pure = sh.contr[0].pure;
            out->nfunc.push_back(pure ? (2 * L + 1) : ((L + 1) * (L + 2)) / 2);
            int nprim = static_cast<int>(sh.alpha.size());
            if (nprim > out->max_nprim) out->max_nprim = nprim;
            if (L > out->max_L) out->max_L = L;
        }
        return out;
    } catch (...) {
        return nullptr;
    }
}

void scf_basis_destroy(scf_basis *bs) {
    delete bs;
}

int scf_basis_nbasis(const scf_basis *bs) {
    return static_cast<int>(bs->bs.nbf());
}

int scf_basis_nshells(const scf_basis *bs) {
    return static_cast<int>(bs->bs.size());
}

void scf_basis_shell_dims(const scf_basis *bs, int *out) {
    for (size_t i = 0; i < bs->nfunc.size(); ++i) out[i] = bs->nfunc[i];
}

void scf_basis_max_dims(const scf_basis *bs, int *max_nprim, int *max_L) {
    *max_nprim = bs->max_nprim;
    *max_L = bs->max_L;
}

static Operator op_for_kind(int kind, bool *ok) {
    *ok = true;
    switch (kind) {
        case 0:   return Operator::coulomb;
        case 1:   return Operator::erf_coulomb;
        case 2:   return Operator::erfc_coulomb;
        case 100: return Operator::overlap;
        case 101: return Operator::kinetic;
        case 102: return Operator::nuclear;
        default:  *ok = false; return Operator::coulomb;
    }
}

/* Geminal engine: cgtg / cgtg_x_coulomb / delcgtg2. Requires libint2 built
 * with the G12 integral class (G12_MAX_AM defined). */
scf_engine *scf_engine_create_geminal(int op_kind, int ngauss,
                                          const double *exps, const double *coefs,
                                          int max_nprim, int max_L, double precision) {
#ifdef G12_MAX_AM
    Operator op;
    switch (op_kind) {
        case 200: op = Operator::cgtg;           break;
        case 201: op = Operator::cgtg_x_coulomb; break;
        case 202: op = Operator::delcgtg2;       break;
        default:  return nullptr;
    }
    libint2::ContractedGaussianGeminal cgg;
    cgg.reserve(ngauss);
    for (int i = 0; i < ngauss; ++i) {
        cgg.emplace_back(exps[i], coefs[i]);
    }
    std::lock_guard<std::mutex> lock(libint_ctor_mutex);
    try {
        // Pass the geminal via the constructor's params argument (6th arg), as
        // libint's own HF++ test does. The set_params() path throws bad_any_cast
        // for delcgtg2 (its K=2 core-eval params are derived differently when
        // set post-construction); the ctor routes through enforce_params_type.
        Engine eng(op, max_nprim, max_L, 0, precision, cgg);
        auto *out = new (std::nothrow) scf_engine{std::move(eng)};
        return out;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "scf_engine_create_geminal(op_kind=%d): %s\n", op_kind, e.what());
        return nullptr;
    } catch (...) {
        std::fprintf(stderr, "scf_engine_create_geminal(op_kind=%d): unknown exception\n", op_kind);
        return nullptr;
    }
#else
    (void)op_kind; (void)ngauss; (void)exps; (void)coefs;
    (void)max_nprim; (void)max_L; (void)precision;
    return nullptr;
#endif
}

scf_engine *scf_engine_create(int op_kind, double omega,
                                  int max_nprim, int max_L, double precision) {
    bool ok = false;
    Operator op = op_for_kind(op_kind, &ok);
    if (!ok) return nullptr;
    std::lock_guard<std::mutex> lock(libint_ctor_mutex);
    try {
        Engine eng(op, max_nprim, max_L, 0, precision);
        if (op_kind == 1 || op_kind == 2) {
            // ErfCoulomb / ErfcCoulomb attenuation parameter.
            eng.set_params(omega);
        }
        auto *out = new (std::nothrow) scf_engine{std::move(eng)};
        return out;
    } catch (...) {
        return nullptr;
    }
}

scf_engine *scf_engine_create_deriv(int op_kind, double omega,
                                        int max_nprim, int max_L, double precision) {
#if LIBINT2_MAX_DERIV_ORDER >= 1
    bool ok = false;
    Operator op = op_for_kind(op_kind, &ok);
    if (!ok) return nullptr;
    std::lock_guard<std::mutex> lock(libint_ctor_mutex);
    try {
        Engine eng(op, max_nprim, max_L, 1, precision);
        if (op_kind == 1 || op_kind == 2) {
            eng.set_params(omega);
        }
        auto *out = new (std::nothrow) scf_engine{std::move(eng)};
        return out;
    } catch (...) {
        return nullptr;
    }
#else
    return nullptr;
#endif
}

void scf_engine_destroy(scf_engine *eng) {
    delete eng;
}

int scf_engine_set_point_charges(scf_engine *eng,
                                   const scf_atom *atoms, int natoms) {
    try {
        std::vector<std::pair<double, std::array<double, 3>>> q(natoms);
        for (int a = 0; a < natoms; ++a) {
            q[a].first = static_cast<double>(atoms[a].Z);
            q[a].second = {atoms[a].x, atoms[a].y, atoms[a].z};
        }
        eng->engine.set_params(q);
        return SCF_OK;
    } catch (...) {
        return SCF_EINTERNAL;
    }
}

/* Compute functions: libint2's Engine::compute can throw (std::bad_alloc under
 * memory pressure, or invalid engine/shell combinations). A C++ exception must
 * never unwind across the C ABI into Rust (undefined behavior), so every
 * compute path catches and returns SCF_EINTERNAL. */

int scf_compute_1e_block(scf_engine *eng, const scf_basis *bs,
                           int sh1, int sh2, double *out) {
  try {
    const auto &shells = bs->bs;
    eng->engine.compute(shells[sh1], shells[sh2]);
    const auto &result = eng->engine.results();
    int n = bs->nfunc[sh1] * bs->nfunc[sh2];
    if (result[0] == nullptr) {
        // All zero.
        for (int i = 0; i < n; ++i) out[i] = 0.0;
    } else {
        for (int i = 0; i < n; ++i) out[i] = result[0][i];
    }
    return n;
  } catch (...) {
    return SCF_EINTERNAL;
  }
}

int scf_compute_eri_quartet(scf_engine *eng, const scf_basis *bs,
                              int sh1, int sh2, int sh3, int sh4, double *out) {
  try {
    const auto &shells = bs->bs;
    eng->engine.compute(shells[sh1], shells[sh2], shells[sh3], shells[sh4]);
    const auto &result = eng->engine.results();
    if (result[0] == nullptr) {
        return 0;  // libint screened the quartet (all zero).
    }
    int n = bs->nfunc[sh1] * bs->nfunc[sh2] * bs->nfunc[sh3] * bs->nfunc[sh4];
    for (int i = 0; i < n; ++i) out[i] = result[0][i];
    return n;
  } catch (...) {
    return SCF_EINTERNAL;
  }
}

int scf_compute_schwarz(scf_engine *eng, const scf_basis *bs, double *qmat) {
  try {
    const int nsh = static_cast<int>(bs->bs.size());
    for (int i = 0; i < nsh; ++i) {
        for (int j = 0; j <= i; ++j) {
            int n1 = bs->nfunc[i];
            int n2 = bs->nfunc[j];
            // Compute (ij|ij).
            eng->engine.compute(bs->bs[i], bs->bs[j], bs->bs[i], bs->bs[j]);
            const auto &r = eng->engine.results();
            double maxv = 0.0;
            if (r[0] != nullptr) {
                // Self-contracted entries: indices a,b in [0, n1)x[0, n2).
                // The full (ab|ab) magnitude is |result[a*n2+b, a*n2+b]|.
                for (int a = 0; a < n1; ++a) {
                    for (int b = 0; b < n2; ++b) {
                        int idx = ((a * n2 + b) * n1 + a) * n2 + b;
                        double v = std::fabs(r[0][idx]);
                        if (v > maxv) maxv = v;
                    }
                }
            }
            double q = std::sqrt(maxv);
            qmat[i * nsh + j] = q;
            qmat[j * nsh + i] = q;
        }
    }
    return SCF_OK;
  } catch (...) {
    return SCF_EINTERNAL;
  }
}

int scf_compute_1e_deriv_block(scf_engine *eng, const scf_basis *bs,
                                 int sh1, int sh2, double *out) {
#if LIBINT2_MAX_DERIV_ORDER >= 1
  try {
    const auto &shells = bs->bs;
    eng->engine.compute(shells[sh1], shells[sh2]);
    const auto &result = eng->engine.results();
    int n = bs->nfunc[sh1] * bs->nfunc[sh2];
    // For overlap/kinetic: 2 centers × 3 coords = 6 derivative blocks
    // For nuclear: 2 shell centers + natoms nuclear centers = 3*(2+natoms) blocks
    int nderiv = static_cast<int>(result.size());
    if (result[0] == nullptr) {
        for (int i = 0; i < nderiv * n; ++i) out[i] = 0.0;
        return 0;
    }
    for (int d = 0; d < nderiv; ++d) {
        const double *src = result[d];
        double *dst = out + d * n;
        if (src) {
            for (int i = 0; i < n; ++i) dst[i] = src[i];
        } else {
            for (int i = 0; i < n; ++i) dst[i] = 0.0;
        }
    }
    return nderiv * n;
  } catch (...) {
    return SCF_EINTERNAL;
  }
#else
    (void)eng; (void)bs; (void)sh1; (void)sh2; (void)out;
    return 0;
#endif
}

int scf_compute_eri_deriv_quartet(scf_engine *eng, const scf_basis *bs,
                                    int sh1, int sh2, int sh3, int sh4, double *out) {
#if LIBINT2_MAX_DERIV_ORDER >= 1
  try {
    const auto &shells = bs->bs;
    eng->engine.compute(shells[sh1], shells[sh2], shells[sh3], shells[sh4]);
    const auto &result = eng->engine.results();
    int n = bs->nfunc[sh1] * bs->nfunc[sh2] * bs->nfunc[sh3] * bs->nfunc[sh4];
    // 4 centers × 3 coords = 12 derivative blocks
    int nderiv = 12;
    if (result[0] == nullptr) {
        return 0;
    }
    for (int d = 0; d < nderiv; ++d) {
        const double *src = result[d];
        double *dst = out + d * n;
        if (src) {
            for (int i = 0; i < n; ++i) dst[i] = src[i];
        } else {
            for (int i = 0; i < n; ++i) dst[i] = 0.0;
        }
    }
    return nderiv * n;
  } catch (...) {
    return SCF_EINTERNAL;
  }
#else
    (void)eng; (void)bs; (void)sh1; (void)sh2; (void)sh3; (void)sh4; (void)out;
    return 0;
#endif
}

/* --- Electric dipole integrals via emultipole1 --- */

int scf_compute_dipole(const scf_basis *bs, const double *origin,
                         int nbas, double *out) {
    try {
        // Zero output: 3 matrices of size nbas*nbas
        int total = 3 * nbas * nbas;
        for (int i = 0; i < total; ++i) out[i] = 0.0;

        // Create emultipole1 engine: results are [overlap, x, y, z]
        Engine eng(Operator::emultipole1, bs->max_nprim, bs->max_L, 0, 1e-14);
        std::array<double, 3> orig{origin[0], origin[1], origin[2]};
        eng.set_params(orig);

        const int nsh = static_cast<int>(bs->bs.size());
        // Compute shell offsets
        std::vector<int> sh_off(nsh + 1, 0);
        for (int s = 0; s < nsh; ++s) sh_off[s + 1] = sh_off[s] + bs->nfunc[s];

        double *ox = out;
        double *oy = out + nbas * nbas;
        double *oz = out + 2 * nbas * nbas;

        for (int s1 = 0; s1 < nsh; ++s1) {
            for (int s2 = 0; s2 <= s1; ++s2) {
                eng.compute(bs->bs[s1], bs->bs[s2]);
                const auto &result = eng.results();
                int n1 = bs->nfunc[s1];
                int n2 = bs->nfunc[s2];
                int o1 = sh_off[s1];
                int o2 = sh_off[s2];
                // result[0] = overlap, result[1] = x, result[2] = y, result[3] = z
                for (int c = 0; c < 3; ++c) {
                    double *mat = (c == 0) ? ox : (c == 1) ? oy : oz;
                    const double *src = result[c + 1];
                    if (src == nullptr) continue;
                    for (int i = 0; i < n1; ++i) {
                        for (int j = 0; j < n2; ++j) {
                            double v = src[i * n2 + j];
                            mat[(o1 + i) * nbas + (o2 + j)] = v;
                            mat[(o2 + j) * nbas + (o1 + i)] = v;
                        }
                    }
                }
            }
        }
        return total;
    } catch (...) {
        return -1;
    }
}

/* --- 3-center and 2-center ERI engines for density fitting / RI --- */

scf_engine *scf_engine_create_3center(int op_kind, double omega,
                                          int max_nprim, int max_L, double precision) {
#if LIBINT2_SUPPORT_ERI3
    bool ok = false;
    Operator op = op_for_kind(op_kind, &ok);
    if (!ok) return nullptr;
    std::lock_guard<std::mutex> lock(libint_ctor_mutex);
    try {
        Engine eng(op, max_nprim, max_L, 0, precision);
        eng.set(libint2::BraKet::xs_xx);
        if (op_kind == 1 || op_kind == 2) eng.set_params(omega);
        return new (std::nothrow) scf_engine{std::move(eng)};
    } catch (...) {
        return nullptr;
    }
#else
    (void)op_kind; (void)omega; (void)max_nprim; (void)max_L; (void)precision;
    return nullptr;
#endif
}

scf_engine *scf_engine_create_2center(int op_kind, double omega,
                                          int max_nprim, int max_L, double precision) {
#if LIBINT2_SUPPORT_ERI2
    bool ok = false;
    Operator op = op_for_kind(op_kind, &ok);
    if (!ok) return nullptr;
    std::lock_guard<std::mutex> lock(libint_ctor_mutex);
    try {
        Engine eng(op, max_nprim, max_L, 0, precision);
        eng.set(libint2::BraKet::xs_xs);
        if (op_kind == 1 || op_kind == 2) eng.set_params(omega);
        return new (std::nothrow) scf_engine{std::move(eng)};
    } catch (...) {
        return nullptr;
    }
#else
    (void)op_kind; (void)omega; (void)max_nprim; (void)max_L; (void)precision;
    return nullptr;
#endif
}

int scf_compute_eri3(scf_engine *eng, const scf_basis *obs,
                       const scf_basis *dfbs,
                       int shP, int sh1, int sh2, double *out) {
#if LIBINT2_SUPPORT_ERI3
  try {
    // BraKet::xs_xx rank=3: compute(aux_shell, obs_shell1, obs_shell2)
    eng->engine.compute(dfbs->bs[shP], obs->bs[sh1], obs->bs[sh2]);
    const auto &result = eng->engine.results();
    if (result[0] == nullptr) return 0;
    int nP = dfbs->nfunc[shP];
    int n1 = obs->nfunc[sh1];
    int n2 = obs->nfunc[sh2];
    int n = nP * n1 * n2;
    for (int i = 0; i < n; ++i) out[i] = result[0][i];
    return n;
  } catch (...) {
    return SCF_EINTERNAL;
  }
#else
    (void)eng; (void)obs; (void)dfbs; (void)shP; (void)sh1; (void)sh2; (void)out;
    return 0;
#endif
}

int scf_compute_eri2(scf_engine *eng, const scf_basis *dfbs,
                       int shP, int shQ, double *out) {
#if LIBINT2_SUPPORT_ERI2
  try {
    // BraKet::xs_xs rank=2: compute(aux_shell_P, aux_shell_Q)
    eng->engine.compute(dfbs->bs[shP], dfbs->bs[shQ]);
    const auto &result = eng->engine.results();
    int n = dfbs->nfunc[shP] * dfbs->nfunc[shQ];
    if (result[0] == nullptr) {
        for (int i = 0; i < n; ++i) out[i] = 0.0;
    } else {
        for (int i = 0; i < n; ++i) out[i] = result[0][i];
    }
    return n;
  } catch (...) {
    return SCF_EINTERNAL;
  }
#else
    (void)eng; (void)dfbs; (void)shP; (void)shQ; (void)out;
    return 0;
#endif
}

/* --- 3-center and 2-center ERI derivative engines --- */

scf_engine *scf_engine_create_3center_deriv(int op_kind, double omega,
                                                int max_nprim, int max_L, double precision) {
#if LIBINT2_SUPPORT_ERI3 && LIBINT2_MAX_DERIV_ORDER >= 1
    bool ok = false;
    Operator op = op_for_kind(op_kind, &ok);
    if (!ok) return nullptr;
    std::lock_guard<std::mutex> lock(libint_ctor_mutex);
    try {
        Engine eng(op, max_nprim, max_L, 1, precision);
        eng.set(libint2::BraKet::xs_xx);
        if (op_kind == 1 || op_kind == 2) eng.set_params(omega);
        return new (std::nothrow) scf_engine{std::move(eng)};
    } catch (...) {
        return nullptr;
    }
#else
    (void)op_kind; (void)omega; (void)max_nprim; (void)max_L; (void)precision;
    return nullptr;
#endif
}

scf_engine *scf_engine_create_2center_deriv(int op_kind, double omega,
                                                int max_nprim, int max_L, double precision) {
#if LIBINT2_SUPPORT_ERI2 && LIBINT2_MAX_DERIV_ORDER >= 1
    bool ok = false;
    Operator op = op_for_kind(op_kind, &ok);
    if (!ok) return nullptr;
    std::lock_guard<std::mutex> lock(libint_ctor_mutex);
    try {
        Engine eng(op, max_nprim, max_L, 1, precision);
        eng.set(libint2::BraKet::xs_xs);
        if (op_kind == 1 || op_kind == 2) eng.set_params(omega);
        return new (std::nothrow) scf_engine{std::move(eng)};
    } catch (...) {
        return nullptr;
    }
#else
    (void)op_kind; (void)omega; (void)max_nprim; (void)max_L; (void)precision;
    return nullptr;
#endif
}

int scf_compute_eri3_deriv(scf_engine *eng, const scf_basis *obs,
                             const scf_basis *dfbs,
                             int shP, int sh1, int sh2, double *out) {
#if LIBINT2_SUPPORT_ERI3 && LIBINT2_MAX_DERIV_ORDER >= 1
  try {
    eng->engine.compute(dfbs->bs[shP], obs->bs[sh1], obs->bs[sh2]);
    const auto &result = eng->engine.results();
    if (result[0] == nullptr) return 0;
    int nP = dfbs->nfunc[shP];
    int n1 = obs->nfunc[sh1];
    int n2 = obs->nfunc[sh2];
    int n = nP * n1 * n2;
    int nderiv = (int)result.size();
    for (int d = 0; d < nderiv; ++d) {
        const double *src = result[d];
        double *dst = out + d * n;
        if (src) {
            for (int i = 0; i < n; ++i) dst[i] = src[i];
        } else {
            for (int i = 0; i < n; ++i) dst[i] = 0.0;
        }
    }
    return nderiv * n;
  } catch (...) {
    return SCF_EINTERNAL;
  }
#else
    (void)eng; (void)obs; (void)dfbs; (void)shP; (void)sh1; (void)sh2; (void)out;
    return 0;
#endif
}

int scf_compute_eri2_deriv(scf_engine *eng, const scf_basis *dfbs,
                             int shP, int shQ, double *out) {
#if LIBINT2_SUPPORT_ERI2 && LIBINT2_MAX_DERIV_ORDER >= 1
  try {
    eng->engine.compute(dfbs->bs[shP], dfbs->bs[shQ]);
    const auto &result = eng->engine.results();
    if (result[0] == nullptr) return 0;
    int nP = dfbs->nfunc[shP];
    int nQ = dfbs->nfunc[shQ];
    int n = nP * nQ;
    int nderiv = (int)result.size();
    for (int d = 0; d < nderiv; ++d) {
        const double *src = result[d];
        double *dst = out + d * n;
        if (src) {
            for (int i = 0; i < n; ++i) dst[i] = src[i];
        } else {
            for (int i = 0; i < n; ++i) dst[i] = 0.0;
        }
    }
    return nderiv * n;
  } catch (...) {
    return SCF_EINTERNAL;
  }
#else
    (void)eng; (void)dfbs; (void)shP; (void)shQ; (void)out;
    return 0;
#endif
}

/* ==========================================================================
 *  terfc(r,r0)/r attenuated 3-center / 2-center integral engine
 *
 *  Clean "Coulomb - terf" decomposition (machine-precision verified against a
 *  1e-60 closed-form oracle; see terf-tables/terfc_lookup_reference.py and the
 *  Rust harness tests/terfc_base_validation.rs):
 *
 *    terfc(r,r0)/r = 1/r  -  terf(r,r0)/r
 *    terf(r,r0)/r  = (erf(w(r-r0)) + erf(w(r+r0))) / (2 r),   w = 1/(r0 sqrt2).
 *
 *  Both pieces are built by the SAME McMurchie-Davidson Cartesian pass
 *  (compute_cart_eri3 / _eri2) so the ordering + normalization are byte-
 *  compatible with libint's Coulomb eri3/eri2 (verified: MD Coulomb == libint to
 *  ~1e-12 for s/p/d/f). The output goes through libint2::solidharmonics so the
 *  spherical component ordering matches libint (p is m=-1,0,+1 = y,z,x).
 *
 *  Per primitive (p = aux exp, q = obs-pair combined exp, R = |P-Q|):
 *    theta^2 = p q/(p+q)                       # Coulomb reduced exponent
 *    phi^2   = 1/(1/p + 1/q + 1/omega^2)       # folds in 1/omega^2  (the crux)
 *    T = theta^2 R^2,   S = phi^2 R^2,   s = phi^2 r0^2
 *  The Coulomb pass uses reduced exponent theta^2 and Boys F_m(T); the terf pass
 *  uses reduced exponent phi^2 and the tabulated replacement
 *    A_m = (phi/theta) * G_{m,0}(S,s)          # (phi/theta): Dutoi Eq 9 vs 6
 *  as the drop-in Boys vector (n-index = 0 for energies). See terf_aux().
 * ========================================================================== */

constexpr int TERFC_DIMM = 24;  // m-index depth stored in tables
constexpr int TERFC_DIMN = 12;  // n-index depth stored in tables

// One loaded G_{m,n}(S,s) table. Defined at global scope (not the anonymous
// namespace) so it matches the forward declaration used by scf_engine.
struct TerfcTable {
    int    nS = 0, ns = 0, dimm = 0, dimn = 0;
    double delta_S = 0.0, delta_s = 0.0;   // grid spacing = 1/pts
    double S_max = 0.0, s_max = 0.0;
    std::vector<double> data;              // [nS][ns][dimm][dimn], C-order

    inline double at(int iS, int is, int m, int n) const {
        return data[(((size_t)iS * ns + is) * dimm + m) * dimn + n];
    }
    bool covers(double S, double s) const { return S <= S_max && s <= s_max; }
};

// The four tables, ordered finest-first for query-time selection.
struct TerfcTableSet {
    std::vector<TerfcTable> tables;  // sorted so tables[0] is finest
};

namespace {

// Load one binary table file (little-endian: 4x int32 header + f64 data).
// pts is used to set the grid spacing (delta = 1/pts) and S_max/s_max.
bool load_terfc_table(const std::string &path, int pts, double S_max, double s_max,
                      TerfcTable &out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t hdr[4];
    f.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
    if (!f) return false;
    out.nS = hdr[0];
    out.ns = hdr[1];
    out.dimm = hdr[2];
    out.dimn = hdr[3];
    if (out.nS <= 1 || out.ns <= 1 || out.dimm < TERFC_DIMM || out.dimn < TERFC_DIMN)
        return false;
    size_t count = (size_t)out.nS * out.ns * out.dimm * out.dimn;
    out.data.resize(count);
    f.read(reinterpret_cast<char *>(out.data.data()), count * sizeof(double));
    if (!f) return false;
    out.delta_S = 1.0 / pts;
    out.delta_s = 1.0 / pts;
    out.S_max = S_max;
    out.s_max = s_max;
    return true;
}

// Table directory resolution: explicit arg > env FERRIC_TERF_TABLE_DIR.
std::string resolve_table_dir(const char *table_dir) {
    if (table_dir && table_dir[0] != '\0') return std::string(table_dir);
    const char *env = std::getenv("FERRIC_TERF_TABLE_DIR");
    if (env && env[0] != '\0') return std::string(env);
    return std::string();
}

// Process-global cache of the loaded table set, keyed by directory.
std::mutex terfc_tables_mutex;
std::shared_ptr<TerfcTableSet> g_terfc_tables;   // last loaded set
std::string                    g_terfc_dir;      // its directory

std::shared_ptr<TerfcTableSet> get_terfc_tables(const std::string &dir) {
    std::lock_guard<std::mutex> lock(terfc_tables_mutex);
    if (g_terfc_tables && g_terfc_dir == dir) return g_terfc_tables;
    auto set = std::make_shared<TerfcTableSet>();
    // (pts, S_max, s_max, filename) finest-first.
    struct Spec { int pts; double S_max; double s_max; const char *name; };
    const Spec specs[] = {
        {16, 4.0,  2.0,  "16_4_2.bin"},
        {8,  10.0, 5.0,  "8_10_5.bin"},
        {4,  20.0, 20.0, "4_20_20.bin"},
        {2,  20.0, 80.0, "2_20_80.bin"},
    };
    for (const auto &sp : specs) {
        TerfcTable t;
        std::string path = dir;
        if (!path.empty() && path.back() != '/') path.push_back('/');
        path += sp.name;
        if (!load_terfc_table(path, sp.pts, sp.S_max, sp.s_max, t)) {
            return nullptr;  // missing/corrupt table => fail engine creation
        }
        set->tables.push_back(std::move(t));
    }
    g_terfc_tables = set;
    g_terfc_dir = dir;
    return set;
}

// 1D Lagrange interpolation on consecutive-integer nodes {node0, node0+1, ...}.
// vals[j] is the tabulated value at integer node (node0 + j); x is the continuous
// grid coordinate (S*pts or s*pts). K nodes. This is the exact analogue of the
// Python reference's _lagrange_1d.
inline double lagrange_1d(int node0, const double *vals, int K, double x) {
    double tot = 0.0;
    for (int j = 0; j < K; ++j) {
        double xj = node0 + j;
        double term = vals[j];
        for (int k = 0; k < K; ++k) {
            if (k == j) continue;
            double xk = node0 + k;
            term *= (x - xk) / (xj - xk);
        }
        tot += term;
    }
    return tot;
}

// 10x10-term polynomial (Lagrange) interpolation of G_{m,n}(S,s) from the finest
// covering table. Matches terfc_lookup_reference.py interp_G to ~1e-12 (bilinear
// was ~1e-4). Returns false only if no table covers (S,s) (caller: negligible).
inline bool interp_G(const TerfcTableSet &set, double S, double s,
                     int m, int n, double &out, int K = 10) {
    const TerfcTable *tbl = nullptr;
    for (const auto &t : set.tables) {
        if (t.covers(S, s)) { tbl = &t; break; }
    }
    if (!tbl) return false;
    const int nS = tbl->nS, ns = tbl->ns;
    const double fS = S / tbl->delta_S;  // == S * pts
    const double fs = s / tbl->delta_s;  // == s * pts

    // Window: K nodes centered on floor(f), clamped to [0, N-K] (mirrors the
    // reference's `window`: i0 = floor(f) - K/2 + 1, clamp to [0, N-K]).
    auto window = [](double f, int N, int Kw) -> int {
        if (N < Kw) return 0;
        int i0 = (int)std::floor(f) - Kw / 2 + 1;
        if (i0 < 0) i0 = 0;
        if (i0 > N - Kw) i0 = N - Kw;
        return i0;
    };
    const int K_S = std::min(K, nS);
    const int K_s = std::min(K, ns);
    const int iS0 = window(fS, nS, K_S);
    const int is0 = window(fs, ns, K_s);

    // Interpolate along s for each S node, then along S. K<=10 so stack scratch.
    double col[16];
    double row[16];
    for (int a = 0; a < K_S; ++a) {
        const int iS = iS0 + a;
        for (int c = 0; c < K_s; ++c)
            row[c] = tbl->at(iS, is0 + c, m, n);
        col[a] = lagrange_1d(is0, row, K_s, fs);
    }
    out = lagrange_1d(iS0, col, K_S, fS);
    return true;
}

// Boys function F_m(T) via upward recursion from F_0, downward-stable eval of F_0.
// Only used inside A_m (S is the Boys arg); we need F_0..F_Lmax.
void boys_upto(int mmax, double T, double *F) {
    // F_0(T) = sqrt(pi/(4T)) erf(sqrt T); series near T=0.
    if (T < 1e-13) {
        for (int m = 0; m <= mmax; ++m) F[m] = 1.0 / (2 * m + 1);
        return;
    }
    // F_0 accurate; then upward recursion F_{m+1}=((2m+1)F_m - e^{-T})/(2T)
    // Upward recursion is stable for T >= ~ mmax; for small T use downward.
    double eT = std::exp(-T);
    if (T > (double)mmax) {
        F[0] = std::sqrt(M_PI / (4.0 * T)) * std::erf(std::sqrt(T));
        for (int m = 0; m < mmax; ++m)
            F[m + 1] = ((2 * m + 1) * F[m] - eT) / (2.0 * T);
    } else {
        // Downward recursion from a high starting order for stability.
        const int mtop = mmax + 20;
        double f = 0.0;  // asymptotic seed for F_mtop ~ small
        // series for F_mtop(T): F_m = e^{-T} sum_{k>=0} (2m-1)!!... use Kummer series:
        // F_m(T) = e^{-T} sum_{k=0}^inf (2T)^k (2m-1)!!/(2m+2k+1)!! ; approximate a few terms.
        double term = 1.0 / (2 * mtop + 1);
        double sum = term;
        for (int k = 1; k < 200; ++k) {
            term *= (2.0 * T) / (2 * mtop + 2 * k + 1);
            sum += term;
            if (term < 1e-17 * sum) break;
        }
        f = eT * sum;
        double Fm = f;
        for (int m = mtop; m > 0; --m) {
            double Fm1 = (2.0 * T * Fm + eT) / (2 * m - 1);
            if (m - 1 <= mmax) F[m - 1] = Fm1;
            Fm = Fm1;
        }
        // F[mmax] .. F[0] filled for m-1<=mmax; also set F[mmax] if not.
        // The loop above fills F[m-1] for m-1 in [0,mtop-1]; covers [0,mmax].
    }
}

// Exact evaluation of the Dutoi auxiliary G_{m,0}(S,s) by its DEFINING Poisson
// series (generate_tables.py `_compute_Gmn`):
//   pmf_S[i] = e^{-S} S^i / i!                        (k=1 row)
//   gS[k][i] = gS[k-1][i] - gS[k-1][i-1], k >= 2      (forward differences)
//   cdf_s[i] = sum_{j<=i} e^{-s} s^j / j!             (n=0 row)
//   df(2i)   = (2i)!! / (2i+1)!!
//   G_{m,0}(S,s) = sum_i df(2i) * gS[m+1][i] * cdf_s[i]
//
// Used when (S,s) lies outside every interpolation table. The only reachable
// out-of-table region is S > 20 with s < 1/2: the curvature constraint
// r0*omega = 1/sqrt2 gives s = phi^2 r0^2 <= omega^2 r0^2 = 1/2, so s is always
// deep inside every table's s-range and coverage reduces to S <= 20.
//
// float64 series validated against the 256-bit mpmath reference: <= 6e-12
// relative for S in [20.25, 200], s in [0, 0.5], m <= 16 (worst 5e-5 at m=16,
// S=200 where |G| ~ 1e-26). At s=0 the series reduces exactly to the Boys
// function, G_m(S,0) = F_m(S) (verified to 1e-77). For S > 600 the
// s-dependence is < e^{-580} relative, so Boys is used directly (also avoids
// e^{-S} underflow in the PMF recurrence).
void terf_G_series(double S, double s, int mmax, double *G) {
    if (S > 600.0) {
        boys_upto(mmax, S, G);
        return;
    }
    const int N = (int)(S + 12.0 * std::sqrt(S) + 60.0);
    std::vector<double> pmf(N), cdf(N), df(N), next(N);
    pmf[0] = std::exp(-S);
    for (int i = 1; i < N; ++i) pmf[i] = pmf[i - 1] * S / i;
    double t = std::exp(-s), acc = t;
    cdf[0] = acc;
    for (int i = 1; i < N; ++i) {
        t *= s / i;
        acc += t;
        cdf[i] = acc < 1.0 ? acc : 1.0;
    }
    df[0] = 1.0;
    for (int i = 1; i < N; ++i) df[i] = df[i - 1] * (2.0 * i) / (2.0 * i + 1.0);
    std::vector<double> row = pmf;  // k=1 row; order m uses row k=m+1
    for (int m = 0; m <= mmax; ++m) {
        if (m > 0) {  // advance k=m -> k=m+1 by one forward difference
            next[0] = row[0];
            for (int i = 1; i < N; ++i) next[i] = row[i] - row[i - 1];
            row.swap(next);
        }
        double total = 0.0;
        for (int i = 0; i < N; ++i) total += df[i] * row[i] * cdf[i];
        G[m] = total;
    }
}

// -------------------------------------------------------------------------
//  terf Boys-replacement vector  A[m] = (phi/theta) * G_{m,0}(S,s).
//
//  The clean decomposition (verified to 1e-16 vs the 1e-60 oracle, see
//  terfc_lookup_reference.py + verify_terf_pref.py):
//    I[terfc] = I[coulomb] - I[terf]
//  where the terf piece is a STANDARD Coulomb-form McMurchie-Davidson integral
//  with the reduced exponent theta^2 replaced by phi^2 = 1/(1/p+1/q+1/omega^2)
//  and the Boys vector F_m(T) replaced by A[m]:
//    S = phi^2 * |P-Q|^2,   s = phi^2 * r0^2,   theta^2 = p q/(p+q),
//    A[m] = (phi/theta) * G_{m,0}(S,s).
//  The overall MD prefactor (2 pi^2.5 K)/(p q sqrt(p+q)) is IDENTICAL to the
//  Coulomb pass; the (phi/theta) factor is the ratio of the two operators'
//  fundamental normalisations (Dutoi Eq 9 vs Eq 6) and is m-independent.
//
//  For ENERGY integrals the n-index is a fixed spectator at 0 (s already carries
//  r0 through phi).
//
//  Out-of-table (S,s) — i.e. far-field S > 20 — falls back to the exact
//  Poisson-series evaluation (terf_G_series). This is NOT optional: at large
//  separation terf -> full Coulomb (it is terfc that is negligible), so
//  SKIPPING the terf subtraction leaves the full Coulomb value in the terfc
//  result and inflates far-field integrals from ~0 to 1/R magnitude. That bug
//  made (P|Q)_terfc spuriously INDEFINITE on larger systems (alkane_4+/
//  cc-pVDZ-RI at r0=0.75 A; alkane_12 at r0=1.05 A) and blew up downstream
//  RI-MP2 energies. Always returns true.
// -------------------------------------------------------------------------
inline bool terf_aux(const TerfcTableSet &set, double S, double s,
                     double phi_over_theta, int mmax, double *A) {
    double g;
    if (interp_G(set, S, s, /*m=*/0, /*n=*/0, g)) {
        A[0] = phi_over_theta * g;
        for (int m = 1; m <= mmax; ++m) {
            // coverage is (S,s)-only, so these cannot fail after m=0 succeeded
            if (!interp_G(set, S, s, m, /*n=*/0, g)) return false;
            A[m] = phi_over_theta * g;
        }
        return true;
    }
    // Outside all tables: exact series (reachable only for S > 20, s < 1/2).
    double gser[TERFC_DIMM];
    terf_G_series(S, s, mmax, gser);
    for (int m = 0; m <= mmax; ++m) A[m] = phi_over_theta * gser[m];
    return true;
}

} // anonymous namespace

/* --------------------------------------------------------------------------
 *  Cartesian Obara-Saika 3-center engine for the terfc operator.
 *
 *  We compute the Cartesian block (P_cart | a_cart b_cart) for one primitive
 *  combination via McMurchie-Davidson. For Coulomb (use_boys) the Boys vector is
 *  F_m(T) with reduced exponent theta^2; for terf the Boys vector is the
 *  tabulated A_m = (phi/theta) G_{m,0}(S,s) with reduced exponent phi^2 (see
 *  terf_aux). Cartesian component ordering follows libint's FOR_CART macro so
 *  that libint2::solidharmonics::tform_* produces byte-compatible spherical
 *  output.
 *
 *  Reference scheme (aux P is one "electron", the obs pair a,b the other):
 *    Treat as a 3-center (P | a b). Build [P|ab] by:
 *      (1) vertical recurrence on P (the aux/bra) building [e0|s s] with the
 *          A_m auxiliaries, then
 *      (2) electron-transfer / horizontal recurrence to move angular momentum
 *          onto a and b.
 *  For clarity and correctness we use the McMurchie-Davidson-free OS form.
 * -------------------------------------------------------------------------- */

#include <libint2/cgshell_ordering.h>

namespace {

// Enumerate Cartesian components (lx,ly,lz) for angular momentum L in libint's
// FOR_CART order. Returns list of (lx,ly,lz).
void cart_components(int L, std::vector<std::array<int,3>> &out) {
    out.clear();
    int i, j, k;
    FOR_CART(i, j, k, L)
        out.push_back({i, j, k});
    END_FOR_CART
}

// --------------------------------------------------------------------------
//  McMurchie-Davidson Hermite-expansion 3-center / 2-center engine.
//
//  We compute the Cartesian block over *unnormalized* Cartesian monomials
//  x^lx y^ly z^lz exp(-a r^2) -- i.e. the same "unnormalized Cartesian" that
//  libint2::solidharmonics feeds its cart->pure transform. Each primitive
//  carries the libint-renormalized radial coefficient contr[0].coeff[p]
//  (identical radial factor for every Cartesian component of a shell), so
//  applying libint2's solidharmonics::coeff() reproduces the pure integrals
//  byte-for-byte.
//
//  The ONLY operator-specific input is the Boys-order vector: Coulomb uses
//  F_m(T); terfc uses A[m] = G_{m,0}(S,s). Both plug into the identical
//  Hermite-Coulomb R_{tuv} recurrence below.
// --------------------------------------------------------------------------

// Hermite expansion coefficients E^{i,j}_t for one Cartesian direction.
// Product of two 1D Gaussians (exp a at Ax, exp b at Bx). p=a+b, mu=a*b/p.
// E[i][j][t], i in 0..li, j in 0..lj, t in 0..(li+lj). Recurrence (Helgaker).
struct HermiteE {
    int li, lj;
    std::vector<double> e;  // (li+1)*(lj+1)*(li+lj+1)
    inline int idx(int i, int j, int t) const {
        return (i * (lj + 1) + j) * (li + lj + 1) + t;
    }
    inline double at(int i, int j, int t) const {
        if (t < 0 || t > i + j) return 0.0;
        return e[idx(i, j, t)];
    }
};

void build_hermite_E(double a, double b, double Ax, double Bx, int li, int lj,
                     HermiteE &H) {
    H.li = li;
    H.lj = lj;
    const double p = a + b;
    const double mu = a * b / p;
    const double AB = Ax - Bx;
    const double Px = (a * Ax + b * Bx) / p;
    const double PA = Px - Ax;
    const double PB = Px - Bx;
    const int tmax = li + lj;
    H.e.assign((size_t)(li + 1) * (lj + 1) * (tmax + 1), 0.0);
    auto set = [&](int i, int j, int t, double v) { H.e[H.idx(i, j, t)] = v; };
    auto get = [&](int i, int j, int t) -> double {
        if (t < 0 || t > i + j) return 0.0;
        return H.e[H.idx(i, j, t)];
    };
    set(0, 0, 0, std::exp(-mu * AB * AB));
    // Increment i first (j=0), then increment j.
    for (int i = 0; i <= li; ++i) {
        for (int j = 0; j <= lj; ++j) {
            if (i == 0 && j == 0) continue;
            for (int t = 0; t <= i + j; ++t) {
                double v = 0.0;
                if (i > 0) {
                    // lower i by one
                    v += (1.0 / (2.0 * p)) * get(i - 1, j, t - 1);
                    v += PA * get(i - 1, j, t);
                    v += (t + 1) * get(i - 1, j, t + 1);
                } else {
                    // lower j by one
                    v += (1.0 / (2.0 * p)) * get(i, j - 1, t - 1);
                    v += PB * get(i, j - 1, t);
                    v += (t + 1) * get(i, j - 1, t + 1);
                }
                set(i, j, t, v);
            }
        }
    }
}

// Hermite Coulomb integrals R_{tuv} built from a Boys-order vector Fn[0..tot].
// PQ = P - Q; alpha = reduced exponent theta; Rpc = |PQ|^2 already folded into Fn.
// Standard downward recurrence (Helgaker 9.9.18-20).
struct HermiteR {
    int tmax;
    std::vector<double> r;  // (tmax+1)^3, index (t,u,v)
    inline int idx(int t, int u, int v) const {
        return (t * (tmax + 1) + u) * (tmax + 1) + v;
    }
    inline double at(int t, int u, int v) const {
        if (t < 0 || u < 0 || v < 0) return 0.0;
        return r[idx(t, u, v)];
    }
};

// Build R_{tuv} for t+u+v <= L. Fn must hold the (already prefactor-free)
// Boys-order auxiliary A_n = (-2 theta)^n * <base>_n; here we pass the vector
// scaled so that R^{0}_{000..n} = Fn[n]. We use the auxiliary-index recurrence.
void build_hermite_R(int L, double theta, double PQx, double PQy, double PQz,
                     const double *Fn, HermiteR &R) {
    R.tmax = L;
    const int dim = (L + 1) * (L + 1) * (L + 1);
    // Auxiliary R^{n}_{tuv}; we only need R^{0}. Store per-n scratch.
    // R^{n}_{000} = (-2 theta)^n Fn[n].
    std::vector<double> Rn((size_t)(L + 1) * dim, 0.0);
    auto RIDX = [&](int n, int t, int u, int v) {
        return (size_t)n * dim + (size_t)(t * (L + 1) + u) * (L + 1) + v;
    };
    double fac = 1.0;
    for (int n = 0; n <= L; ++n) {
        Rn[RIDX(n, 0, 0, 0)] = fac * Fn[n];
        fac *= (-2.0 * theta);
    }
    // Build up t+u+v from 1..L. R^{n}_{t+1,u,v} = t*R^{n+1}_{t-1,u,v} + PQx*R^{n+1}_{t,u,v}
    for (int tot = 1; tot <= L; ++tot) {
        for (int t = 0; t <= tot; ++t) {
            for (int u = 0; u <= tot - t; ++u) {
                int v = tot - t - u;
                for (int n = 0; n <= L - tot; ++n) {
                    double val = 0.0;
                    if (t > 0) {
                        val = PQx * Rn[RIDX(n + 1, t - 1, u, v)];
                        if (t > 1) val += (t - 1) * Rn[RIDX(n + 1, t - 2, u, v)];
                    } else if (u > 0) {
                        val = PQy * Rn[RIDX(n + 1, t, u - 1, v)];
                        if (u > 1) val += (u - 1) * Rn[RIDX(n + 1, t, u - 2, v)];
                    } else { // v > 0
                        val = PQz * Rn[RIDX(n + 1, t, u, v - 1)];
                        if (v > 1) val += (v - 1) * Rn[RIDX(n + 1, t, u, v - 2)];
                    }
                    Rn[RIDX(n, t, u, v)] = val;
                }
            }
        }
    }
    R.r.assign(dim, 0.0);
    for (int t = 0; t <= L; ++t)
        for (int u = 0; u <= L; ++u)
            for (int v = 0; v <= L; ++v)
                if (t + u + v <= L)
                    R.r[R.idx(t, u, v)] = Rn[RIDX(0, t, u, v)];
}

// Solid-harmonic (cart->pure) transform matching libint2 for one shell axis.
// Given a Cartesian block of dimension (ncart_row x rest), transform the row
// axis to pure if pure_row; layout is row-major [cart_row][rest].
// We provide simple explicit transforms for the three shells.

// Number of cartesian / pure functions.
inline int ncart_of(int L) { return (L + 1) * (L + 2) / 2; }
inline int npure_of(int L) { return 2 * L + 1; }

// Compute the terfc (or Coulomb, if use_boys) Cartesian 3-center block
// (P_cart | a_cart b_cart), contracted over primitives, into out_cart
// (row-major [ncartP][ncartA][ncartB]).
//
// tables/r0/omega used only when use_boys==false (terfc path).
// Returns false if every primitive contribution was screened out of the tables.
bool compute_cart_eri3(const Shell &shP, const Shell &shA, const Shell &shB,
                       const TerfcTableSet *tables, double omega, double r0,
                       bool use_boys, std::vector<double> &out_cart) {
    const int lP = shP.contr[0].l;
    const int lA = shA.contr[0].l;
    const int lB = shB.contr[0].l;
    const int Ltot = lP + lA + lB;
    const int ncP = ncart_of(lP), ncA = ncart_of(lA), ncB = ncart_of(lB);
    out_cart.assign((size_t)ncP * ncA * ncB, 0.0);

    std::vector<std::array<int, 3>> compP, compA, compB;
    cart_components(lP, compP);
    cart_components(lA, compA);
    cart_components(lB, compB);

    const auto &AO = shA.O;
    const auto &BO = shB.O;
    const auto &PO = shP.O;

    const double omega2 = omega * omega;
    const double r02 = r0 * r0;
    bool any = false;

    // Loop obs pair primitives (a,b) then aux primitive P.
    for (size_t pa = 0; pa < shA.alpha.size(); ++pa) {
        const double a = shA.alpha[pa];
        const double ca = shA.contr[0].coeff[pa];
        for (size_t pb = 0; pb < shB.alpha.size(); ++pb) {
            const double b = shB.alpha[pb];
            const double cb = shB.contr[0].coeff[pb];
            const double q = a + b;
            const double Qx = (a * AO[0] + b * BO[0]) / q;
            const double Qy = (a * AO[1] + b * BO[1]) / q;
            const double Qz = (a * AO[2] + b * BO[2]) / q;
            // NOTE: the Gaussian-product factor exp(-(ab/q)|A-B|^2) is already
            // carried by the Hermite E-coefficients below (E_x(0,0,0)*E_y*E_z =
            // exp(-mu|AB|^2)); do NOT multiply it in again here.
            const double cab = ca * cb;

            // Hermite E-coefficients for the obs pair, per direction.
            HermiteE Ex, Ey, Ez;
            build_hermite_E(a, b, AO[0], BO[0], lA, lB, Ex);
            build_hermite_E(a, b, AO[1], BO[1], lA, lB, Ey);
            build_hermite_E(a, b, AO[2], BO[2], lA, lB, Ez);

            for (size_t pp = 0; pp < shP.alpha.size(); ++pp) {
                const double p = shP.alpha[pp];
                const double cP = shP.contr[0].coeff[pp];
                // aux P is a single Gaussian: Hermite expansion of a monomial at
                // its own center (partner = phantom s at same center).
                HermiteE EPx, EPy, EPz;
                build_hermite_E(p, 0.0, PO[0], PO[0], lP, 0, EPx);
                build_hermite_E(p, 0.0, PO[1], PO[1], lP, 0, EPy);
                build_hermite_E(p, 0.0, PO[2], PO[2], lP, 0, EPz);

                // theta2 = Coulomb reduced exponent (p q/(p+q)); phi2 folds in
                // 1/omega^2 for the terf piece. The MD R-recurrence uses whichever
                // reduced exponent matches the operator (theta2 for Coulomb F_m,
                // phi2 for terf G_m). See terf_aux() for the decomposition.
                const double theta2 = p * q / (p + q);
                const double PQx = PO[0] - Qx;
                const double PQy = PO[1] - Qy;
                const double PQz = PO[2] - Qz;
                const double PQ2 = PQx * PQx + PQy * PQy + PQz * PQz;

                double alpha_R;             // reduced exponent for build_hermite_R
                double Fn[TERFC_DIMM];      // Boys / terf-aux vector
                if (use_boys) {
                    alpha_R = theta2;
                    boys_upto(Ltot, theta2 * PQ2 /* Boys T */, Fn);
                } else {
                    // phi^2 = 1/(1/p + 1/q + 1/omega^2) = theta2*omega2/(theta2+omega2)
                    const double phi2 = theta2 * omega2 / (theta2 + omega2);
                    const double S = phi2 * PQ2;
                    const double s = phi2 * r02;
                    const double phi_over_theta = std::sqrt(phi2 / theta2);
                    // terf_aux always succeeds (table interp, or exact series
                    // for far-field S > 20); the guard is defensive only.
                    // Skipping here would leave the full Coulomb value
                    // un-subtracted (terf -> 1/r at large r, NOT negligible).
                    if (!terf_aux(*tables, S, s, phi_over_theta, Ltot, Fn)) {
                        continue;
                    }
                    alpha_R = phi2;
                }

                const double pref =
                    2.0 * std::pow(M_PI, 2.5) / (p * q * std::sqrt(p + q));
                const double scale = pref * cab * cP;

                HermiteR R;
                build_hermite_R(Ltot, alpha_R, PQx, PQy, PQz, Fn, R);
                any = true;

                // Assemble Cartesian integrals.
                // (P_cart | a_cart b_cart) = scale *
                //   sum_{t'u'v'} EP_{t'} * sum_{tuv} Eab_{tuv} * (-1)^{t+u+v}
                //     * R_{t'+t, u'+u, v'+v}
                for (int ip = 0; ip < ncP; ++ip) {
                    const int px = compP[ip][0], py = compP[ip][1], pz = compP[ip][2];
                    for (int ia = 0; ia < ncA; ++ia) {
                        const int ax = compA[ia][0], ay = compA[ia][1], az = compA[ia][2];
                        for (int ib = 0; ib < ncB; ++ib) {
                            const int bx = compB[ib][0], by = compB[ib][1], bz = compB[ib][2];
                            double sum = 0.0;
                            for (int tp = 0; tp <= px; ++tp) {
                                const double eptx = EPx.at(px, 0, tp);
                                if (eptx == 0.0) continue;
                                for (int up = 0; up <= py; ++up) {
                                    const double epty = EPy.at(py, 0, up);
                                    if (epty == 0.0) continue;
                                    for (int vp = 0; vp <= pz; ++vp) {
                                        const double eptz = EPz.at(pz, 0, vp);
                                        if (eptz == 0.0) continue;
                                        const double eP = eptx * epty * eptz;
                                        for (int t = 0; t <= ax + bx; ++t) {
                                            const double ex = Ex.at(ax, bx, t);
                                            if (ex == 0.0) continue;
                                            for (int u = 0; u <= ay + by; ++u) {
                                                const double ey = Ey.at(ay, by, u);
                                                if (ey == 0.0) continue;
                                                for (int v = 0; v <= az + bz; ++v) {
                                                    const double ez = Ez.at(az, bz, v);
                                                    if (ez == 0.0) continue;
                                                    const double eab = ex * ey * ez;
                                                    const double sgn =
                                                        ((t + u + v) & 1) ? -1.0 : 1.0;
                                                    sum += eP * eab * sgn *
                                                           R.at(tp + t, up + u, vp + v);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            out_cart[((size_t)ip * ncA + ia) * ncB + ib] += scale * sum;
                        }
                    }
                }
            }
        }
    }
    return any;
}

// 2-center metric (P | Q): both are single aux Gaussians (phantom s partner).
// Same MD scheme with the "ket" being a single Gaussian Q.
bool compute_cart_eri2(const Shell &shP, const Shell &shQ,
                       const TerfcTableSet *tables, double omega, double r0,
                       bool use_boys, std::vector<double> &out_cart) {
    const int lP = shP.contr[0].l;
    const int lQ = shQ.contr[0].l;
    const int Ltot = lP + lQ;
    const int ncP = ncart_of(lP), ncQ = ncart_of(lQ);
    out_cart.assign((size_t)ncP * ncQ, 0.0);

    std::vector<std::array<int, 3>> compP, compQ;
    cart_components(lP, compP);
    cart_components(lQ, compQ);

    const auto &PO = shP.O;
    const auto &QO = shQ.O;
    const double omega2 = omega * omega;
    const double r02 = r0 * r0;
    bool any = false;

    for (size_t pp = 0; pp < shP.alpha.size(); ++pp) {
        const double p = shP.alpha[pp];
        const double cP = shP.contr[0].coeff[pp];
        HermiteE EPx, EPy, EPz;
        build_hermite_E(p, 0.0, PO[0], PO[0], lP, 0, EPx);
        build_hermite_E(p, 0.0, PO[1], PO[1], lP, 0, EPy);
        build_hermite_E(p, 0.0, PO[2], PO[2], lP, 0, EPz);
        for (size_t pq = 0; pq < shQ.alpha.size(); ++pq) {
            const double q = shQ.alpha[pq];
            const double cQ = shQ.contr[0].coeff[pq];
            HermiteE EQx, EQy, EQz;
            build_hermite_E(q, 0.0, QO[0], QO[0], lQ, 0, EQx);
            build_hermite_E(q, 0.0, QO[1], QO[1], lQ, 0, EQy);
            build_hermite_E(q, 0.0, QO[2], QO[2], lQ, 0, EQz);

            const double theta2 = p * q / (p + q);
            const double PQx = PO[0] - QO[0];
            const double PQy = PO[1] - QO[1];
            const double PQz = PO[2] - QO[2];
            const double PQ2 = PQx * PQx + PQy * PQy + PQz * PQz;

            double alpha_R;
            double Fn[TERFC_DIMM];
            if (use_boys) {
                alpha_R = theta2;
                boys_upto(Ltot, theta2 * PQ2 /* Boys T */, Fn);
            } else {
                const double phi2 = theta2 * omega2 / (theta2 + omega2);
                const double S = phi2 * PQ2;
                const double s = phi2 * r02;
                const double phi_over_theta = std::sqrt(phi2 / theta2);
                // Always succeeds (table or exact far-field series); defensive.
                if (!terf_aux(*tables, S, s, phi_over_theta, Ltot, Fn)) continue;
                alpha_R = phi2;
            }
            const double pref =
                2.0 * std::pow(M_PI, 2.5) / (p * q * std::sqrt(p + q));
            const double scale = pref * cP * cQ;

            HermiteR R;
            build_hermite_R(Ltot, alpha_R, PQx, PQy, PQz, Fn, R);
            any = true;

            for (int ip = 0; ip < ncP; ++ip) {
                const int px = compP[ip][0], py = compP[ip][1], pz = compP[ip][2];
                for (int iq = 0; iq < ncQ; ++iq) {
                    const int qx = compQ[iq][0], qy = compQ[iq][1], qz = compQ[iq][2];
                    double sum = 0.0;
                    for (int tp = 0; tp <= px; ++tp) {
                        const double eptx = EPx.at(px, 0, tp);
                        if (eptx == 0.0) continue;
                        for (int up = 0; up <= py; ++up) {
                            const double epty = EPy.at(py, 0, up);
                            if (epty == 0.0) continue;
                            for (int vp = 0; vp <= pz; ++vp) {
                                const double eptz = EPz.at(pz, 0, vp);
                                if (eptz == 0.0) continue;
                                const double eP = eptx * epty * eptz;
                                for (int tq = 0; tq <= qx; ++tq) {
                                    const double eqx = EQx.at(qx, 0, tq);
                                    if (eqx == 0.0) continue;
                                    for (int uq = 0; uq <= qy; ++uq) {
                                        const double eqy = EQy.at(qy, 0, uq);
                                        if (eqy == 0.0) continue;
                                        for (int vq = 0; vq <= qz; ++vq) {
                                            const double eqz = EQz.at(qz, 0, vq);
                                            if (eqz == 0.0) continue;
                                            const double eQ = eqx * eqy * eqz;
                                            const double sgn =
                                                ((tq + uq + vq) & 1) ? -1.0 : 1.0;
                                            sum += eP * eQ * sgn *
                                                   R.at(tp + tq, up + uq, vp + vq);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    out_cart[(size_t)ip * ncQ + iq] += scale * sum;
                }
            }
        }
    }
    return any;
}

// Apply libint2 cart->pure transform on one axis of a 3-index block.
// in_block: row-major [ncart_axis][rest]; out_block: [npure_axis][rest].
inline void tform_axis(int L, int rest, const std::vector<double> &in,
                       std::vector<double> &out) {
    if (L < 1) {  // s: pure == cartesian (identity). p and up: solid harmonics
                  // (libint orders p as m=-1,0,+1 = y,z,x, NOT the cart x,y,z).
        out = in;
        return;
    }
    const int npure = npure_of(L);
    out.assign((size_t)npure * rest, 0.0);
    const auto &coefs =
        libint2::solidharmonics::SolidHarmonicsCoefficients<double>::instance(L);
    for (int s = 0; s < npure; ++s) {
        const auto nc = coefs.nnz(s);
        const auto *cidx = coefs.row_idx(s);
        const auto *cval = coefs.row_values(s);
        for (int ic = 0; ic < nc; ++ic) {
            const int c = cidx[ic];
            const double w = cval[ic];
            const double *src = &in[(size_t)c * rest];
            double *dst = &out[(size_t)s * rest];
            for (int r = 0; r < rest; ++r) dst[r] += w * src[r];
        }
    }
}

// Transform a cartesian [ncP][ncA][ncB] block to pure [nP][nA][nB] matching
// libint's per-shell pure/cartesian flags and outer=P, then A, then B layout.
void transform_cart_to_pure3(const Shell &shP, const Shell &shA, const Shell &shB,
                             const std::vector<double> &cart,
                             std::vector<double> &pureout) {
    const int lP = shP.contr[0].l, lA = shA.contr[0].l, lB = shB.contr[0].l;
    const bool puP = shP.contr[0].pure, puA = shA.contr[0].pure,
               puB = shB.contr[0].pure;
    const int ncP = ncart_of(lP), ncA = ncart_of(lA), ncB = ncart_of(lB);
    const int nP = puP ? npure_of(lP) : ncP;
    const int nA = puA ? npure_of(lA) : ncA;
    const int nB = puB ? npure_of(lB) : ncB;

    // Transform B (last axis): treat [ncP*ncA][ncB] -> need per-row transform.
    // We transform axis-by-axis by reshaping. Do P (outer) first via tform_axis
    // with rest=ncA*ncB, then A with a strided pass, then B.
    std::vector<double> tmp1;  // after P transform: [nP][ncA][ncB]
    if (puP && lP >= 1) {
        tform_axis(lP, ncA * ncB, cart, tmp1);
    } else {
        tmp1 = cart;
    }
    // Transform A: block layout [nP][ncA][ncB]; for each p, transform middle.
    std::vector<double> tmp2;  // [nP][nA][ncB]
    if (puA && lA >= 1) {
        tmp2.assign((size_t)nP * nA * ncB, 0.0);
        std::vector<double> sub_in((size_t)ncA * ncB), sub_out;
        for (int ipp = 0; ipp < nP; ++ipp) {
            for (int i = 0; i < ncA * ncB; ++i)
                sub_in[i] = tmp1[((size_t)ipp * ncA * ncB) + i];
            tform_axis(lA, ncB, sub_in, sub_out);
            for (int i = 0; i < nA * ncB; ++i)
                tmp2[((size_t)ipp * nA * ncB) + i] = sub_out[i];
        }
    } else {
        tmp2 = tmp1;
    }
    // Transform B (last axis): [nP*nA][ncB] -> [nP*nA][nB].
    if (puB && lB >= 1) {
        pureout.assign((size_t)nP * nA * nB, 0.0);
        const int nrows = nP * nA;
        const auto &coefs =
            libint2::solidharmonics::SolidHarmonicsCoefficients<double>::instance(lB);
        for (int row = 0; row < nrows; ++row) {
            const double *src = &tmp2[(size_t)row * ncB];
            double *dst = &pureout[(size_t)row * nB];
            for (int s = 0; s < nB; ++s) {
                const auto nc = coefs.nnz(s);
                const auto *cidx = coefs.row_idx(s);
                const auto *cval = coefs.row_values(s);
                double acc = 0.0;
                for (int ic = 0; ic < nc; ++ic) acc += cval[ic] * src[cidx[ic]];
                dst[s] = acc;
            }
        }
    } else {
        pureout = tmp2;
    }
}

// 2-center: transform [ncP][ncQ] -> [nP][nQ].
void transform_cart_to_pure2(const Shell &shP, const Shell &shQ,
                             const std::vector<double> &cart,
                             std::vector<double> &pureout) {
    const int lP = shP.contr[0].l, lQ = shQ.contr[0].l;
    const bool puP = shP.contr[0].pure, puQ = shQ.contr[0].pure;
    const int ncP = ncart_of(lP), ncQ = ncart_of(lQ);
    const int nP = puP ? npure_of(lP) : ncP;
    const int nQ = puQ ? npure_of(lQ) : ncQ;
    std::vector<double> tmp1;  // [nP][ncQ]
    if (puP && lP >= 1) {
        tform_axis(lP, ncQ, cart, tmp1);
    } else {
        tmp1 = cart;
    }
    if (puQ && lQ >= 1) {
        pureout.assign((size_t)nP * nQ, 0.0);
        const auto &coefs =
            libint2::solidharmonics::SolidHarmonicsCoefficients<double>::instance(lQ);
        for (int row = 0; row < nP; ++row) {
            const double *src = &tmp1[(size_t)row * ncQ];
            double *dst = &pureout[(size_t)row * nQ];
            for (int s = 0; s < nQ; ++s) {
                const auto nc = coefs.nnz(s);
                const auto *cidx = coefs.row_idx(s);
                const auto *cval = coefs.row_values(s);
                double acc = 0.0;
                for (int ic = 0; ic < nc; ++ic) acc += cval[ic] * src[cidx[ic]];
                dst[s] = acc;
            }
        }
    } else {
        pureout = tmp1;
    }
}

} // anonymous namespace

/* TEMP DEBUG (milestone validation only; remove before Task 2). Loads tables
 * from `dir` and returns poly-10 interp_G(S,s,m,n), or NaN if uncovered/failed. */
extern "C" double scf_terfc_debug_interp_G(const char *dir, double S, double s,
                                           int m, int n) {
    try {
        auto set = get_terfc_tables(resolve_table_dir(dir));
        if (!set) return std::nan("");
        double out = 0.0;
        if (!interp_G(*set, S, s, m, n, out)) return std::nan("");
        return out;
    } catch (...) {
        return std::nan("");
    }
}

/* TEST-ONLY hook: evaluate G_{m,0}(S,s) via the SHIPPED far-field Poisson
 * series (terf_G_series — the exact code path terf_aux uses for out-of-table
 * (S,s)), so the Rust oracle-anchor tests exercise production code, not a
 * reimplementation. Returns NaN on invalid m or internal error (never
 * unwinds across the C ABI). */
extern "C" double scf_terfc_debug_series_G(double S, double s, int m) {
    try {
        if (m < 0 || m >= TERFC_DIMM) return std::nan("");
        double G[TERFC_DIMM];
        terf_G_series(S, s, m, G);
        return G[m];
    } catch (...) {
        return std::nan("");
    }
}

/* TEMP DEBUG (milestone validation only; remove before Task 2). Computes the
 * Coulomb 3-center block (shP|sh1 sh2) via the SAME MD machinery the terfc path
 * uses (use_boys=true), so a comparison against libint's scf_compute_eri3
 * validates normalisation + cart->spherical ordering independently of the tables.
 * Writes nP*n1*n2 spherical doubles; returns n or negative on error. */
extern "C" int scf_terfc_debug_coulomb_eri3(const scf_basis *obs,
                                            const scf_basis *dfbs,
                                            int shP, int sh1, int sh2,
                                            double *out) {
    try {
        if (!obs || !dfbs || !out) return SCF_EINVAL;
        const Shell &shPsh = dfbs->bs[shP];
        const Shell &shAsh = obs->bs[sh1];
        const Shell &shBsh = obs->bs[sh2];
        std::vector<double> cart;
        compute_cart_eri3(shPsh, shAsh, shBsh, nullptr, 0.0, 0.0,
                          /*use_boys=*/true, cart);
        int n = dfbs->nfunc[shP] * obs->nfunc[sh1] * obs->nfunc[sh2];
        std::vector<double> pureout;
        transform_cart_to_pure3(shPsh, shAsh, shBsh, cart, pureout);
        if ((int)pureout.size() != n) return SCF_EINTERNAL;
        for (int i = 0; i < n; ++i) out[i] = pureout[i];
        return n;
    } catch (...) {
        return SCF_EINTERNAL;
    }
}

/* TEMP DEBUG (milestone validation only; remove before Task 2). Returns the raw
 * pre-transform Cartesian terf block (use_boys=false) for (shP|sh1 sh2), so the
 * per-cart-component terf vs coulomb structure can be inspected. Writes
 * ncartP*ncartA*ncartB doubles; returns that count, or negative on error. The
 * `which`=0 -> coulomb cart, `which`=1 -> terf cart. r0/omega from args. */
/* --------------------------------------------------------------------------
 *  C ABI: terfc engine creation and compute.
 *
 *  Engine creation loads the four G_{m,n}(S,s) tables (process-global cache)
 *  and stashes r0/omega/precision/max_L + the table shared_ptr on scf_engine.
 *  The compute functions build the Cartesian OS integrals and apply libint2's
 *  solid-harmonic transform for byte-compatible spherical output.
 *
 *  Every function wraps its body in try/catch(...) and returns SCF_EINTERNAL on
 *  any C++ exception -- a throw must never unwind across the C ABI (UB).
 * -------------------------------------------------------------------------- */

extern "C" scf_engine *scf_engine_create_terfc_3center(double r0, double omega,
                                                       int max_nprim, int max_L,
                                                       double precision,
                                                       const char *table_dir) {
    (void)max_nprim;
    std::lock_guard<std::mutex> lock(libint_ctor_mutex);
    try {
        std::string dir = resolve_table_dir(table_dir);
        if (dir.empty()) return nullptr;
        auto tables = get_terfc_tables(dir);
        if (!tables) return nullptr;
        auto *out = new (std::nothrow) scf_engine{Engine()};
        if (!out) return nullptr;
        out->is_terfc = true;
        out->r0 = r0;
        out->omega = omega;
        out->precision = precision;
        out->max_L = max_L;
        out->terfc_tables = std::move(tables);
        return out;
    } catch (...) {
        return nullptr;
    }
}

extern "C" scf_engine *scf_engine_create_terfc_2center(double r0, double omega,
                                                       int max_nprim, int max_L,
                                                       double precision,
                                                       const char *table_dir) {
    // Same engine payload as the 3-center variant; the 2-center metric reuses
    // the identical table set and OS base (aux-aux instead of aux-obs pair).
    return scf_engine_create_terfc_3center(r0, omega, max_nprim, max_L, precision,
                                           table_dir);
}

extern "C" int scf_compute_terfc_eri3(scf_engine *eng, const scf_basis *obs,
                                      const scf_basis *dfbs,
                                      int shP, int sh1, int sh2, double *out) {
    try {
        if (!eng || !eng->is_terfc || !eng->terfc_tables) return SCF_EINVAL;
        if (!obs || !dfbs || !out) return SCF_EINVAL;
        const Shell &shPsh = dfbs->bs[shP];
        const Shell &shAsh = obs->bs[sh1];
        const Shell &shBsh = obs->bs[sh2];

        // terfc = coulomb - terf. Both use the identical MD machinery (same
        // ordering, prefactor, normalisation, cart->pure transform), so the
        // subtraction is valid element-by-element in the Cartesian basis.
        std::vector<double> cart_coul, cart_terf;
        bool any_c = compute_cart_eri3(shPsh, shAsh, shBsh, nullptr,
                                       eng->omega, eng->r0, /*use_boys=*/true,
                                       cart_coul);
        bool any_t = compute_cart_eri3(shPsh, shAsh, shBsh, eng->terfc_tables.get(),
                                       eng->omega, eng->r0, /*use_boys=*/false,
                                       cart_terf);
        int nP = dfbs->nfunc[shP];
        int n1 = obs->nfunc[sh1];
        int n2 = obs->nfunc[sh2];
        int n = nP * n1 * n2;
        if (!any_c && !any_t) {
            for (int i = 0; i < n; ++i) out[i] = 0.0;
            return 0;  // fully screened
        }
        // Form the Cartesian difference (terf may screen where Coulomb doesn't;
        // treat a screened piece as an all-zero block of the right size).
        std::vector<double> cart(cart_coul.size(), 0.0);
        if (any_c) cart = cart_coul;
        if (any_t) {
            if (cart_terf.size() != cart.size()) return SCF_EINTERNAL;
            for (size_t i = 0; i < cart.size(); ++i) cart[i] -= cart_terf[i];
        }
        std::vector<double> pureout;
        transform_cart_to_pure3(shPsh, shAsh, shBsh, cart, pureout);
        if ((int)pureout.size() != n) return SCF_EINTERNAL;
        for (int i = 0; i < n; ++i) out[i] = pureout[i];
        return n;
    } catch (...) {
        return SCF_EINTERNAL;
    }
}

extern "C" int scf_compute_terfc_eri2(scf_engine *eng, const scf_basis *dfbs,
                                      int shP, int shQ, double *out) {
    try {
        if (!eng || !eng->is_terfc || !eng->terfc_tables) return SCF_EINVAL;
        if (!dfbs || !out) return SCF_EINVAL;
        const Shell &shPsh = dfbs->bs[shP];
        const Shell &shQsh = dfbs->bs[shQ];

        std::vector<double> cart_coul, cart_terf;
        bool any_c = compute_cart_eri2(shPsh, shQsh, nullptr,
                                       eng->omega, eng->r0, /*use_boys=*/true,
                                       cart_coul);
        bool any_t = compute_cart_eri2(shPsh, shQsh, eng->terfc_tables.get(),
                                       eng->omega, eng->r0, /*use_boys=*/false,
                                       cart_terf);
        int nP = dfbs->nfunc[shP];
        int nQ = dfbs->nfunc[shQ];
        int n = nP * nQ;
        std::vector<double> cart(cart_coul.size(), 0.0);
        if (any_c) cart = cart_coul;
        if (any_t) {
            if (cart_terf.size() != cart.size()) return SCF_EINTERNAL;
            for (size_t i = 0; i < cart.size(); ++i) cart[i] -= cart_terf[i];
        }
        std::vector<double> pureout;
        transform_cart_to_pure2(shPsh, shQsh, cart, pureout);
        if ((int)pureout.size() != n) return SCF_EINTERNAL;
        for (int i = 0; i < n; ++i) out[i] = pureout[i];
        return n;
    } catch (...) {
        return SCF_EINTERNAL;
    }
}

/* --------------------------------------------------------------------------
 *  C ABI: terf (tempered LR complement) engine creation and compute.
 *
 *  terf(r,r0)/r = erf-like LONG-RANGE piece of the exact tempered kernel:
 *      terf(r,r0)/r = (erf(w(r-r0)) + erf(w(r+r0))) / (2 r),  w = 1/(r0 sqrt2)
 *  identically the "cart_terf" Cartesian block already computed inside
 *  scf_compute_terfc_eri3/2 (terfc = coulomb - terf). This entry point
 *  returns that SAME block directly instead of subtracting it from Coulomb,
 *  so terf + terfc = coulomb holds at machine precision by construction
 *  (both share the identical table lookup / OS recurrence / cart->pure
 *  transform code path -- only the final combine differs).
 *
 *  Engine creation reuses scf_engine_create_terfc_3center's table-loading
 *  logic verbatim; only the is_terf_complement tag differs, so the SAME
 *  process-global table cache (get_terfc_tables) is shared between the terf
 *  and terfc engines for a given table_dir.
 * -------------------------------------------------------------------------- */

extern "C" scf_engine *scf_engine_create_terf_3center(double r0, double omega,
                                                      int max_nprim, int max_L,
                                                      double precision,
                                                      const char *table_dir) {
    scf_engine *eng = scf_engine_create_terfc_3center(r0, omega, max_nprim, max_L,
                                                       precision, table_dir);
    if (eng) eng->is_terf_complement = true;
    return eng;
}

extern "C" scf_engine *scf_engine_create_terf_2center(double r0, double omega,
                                                      int max_nprim, int max_L,
                                                      double precision,
                                                      const char *table_dir) {
    scf_engine *eng = scf_engine_create_terfc_2center(r0, omega, max_nprim, max_L,
                                                       precision, table_dir);
    if (eng) eng->is_terf_complement = true;
    return eng;
}

extern "C" int scf_compute_terf_eri3(scf_engine *eng, const scf_basis *obs,
                                     const scf_basis *dfbs,
                                     int shP, int sh1, int sh2, double *out) {
    try {
        if (!eng || !eng->is_terfc || !eng->is_terf_complement || !eng->terfc_tables) {
            return SCF_EINVAL;
        }
        if (!obs || !dfbs || !out) return SCF_EINVAL;
        const Shell &shPsh = dfbs->bs[shP];
        const Shell &shAsh = obs->bs[sh1];
        const Shell &shBsh = obs->bs[sh2];

        // terf is the SAME Cartesian block terfc subtracts from Coulomb --
        // return it directly (no combine), so terf + terfc = coulomb exactly.
        std::vector<double> cart_terf;
        bool any_t = compute_cart_eri3(shPsh, shAsh, shBsh, eng->terfc_tables.get(),
                                       eng->omega, eng->r0, /*use_boys=*/false,
                                       cart_terf);
        int nP = dfbs->nfunc[shP];
        int n1 = obs->nfunc[sh1];
        int n2 = obs->nfunc[sh2];
        int n = nP * n1 * n2;
        if (!any_t) {
            for (int i = 0; i < n; ++i) out[i] = 0.0;
            return 0;  // fully screened
        }
        std::vector<double> pureout;
        transform_cart_to_pure3(shPsh, shAsh, shBsh, cart_terf, pureout);
        if ((int)pureout.size() != n) return SCF_EINTERNAL;
        for (int i = 0; i < n; ++i) out[i] = pureout[i];
        return n;
    } catch (...) {
        return SCF_EINTERNAL;
    }
}

extern "C" int scf_compute_terf_eri2(scf_engine *eng, const scf_basis *dfbs,
                                     int shP, int shQ, double *out) {
    try {
        if (!eng || !eng->is_terfc || !eng->is_terf_complement || !eng->terfc_tables) {
            return SCF_EINVAL;
        }
        if (!dfbs || !out) return SCF_EINVAL;
        const Shell &shPsh = dfbs->bs[shP];
        const Shell &shQsh = dfbs->bs[shQ];

        std::vector<double> cart_terf;
        bool any_t = compute_cart_eri2(shPsh, shQsh, eng->terfc_tables.get(),
                                       eng->omega, eng->r0, /*use_boys=*/false,
                                       cart_terf);
        int nP = dfbs->nfunc[shP];
        int nQ = dfbs->nfunc[shQ];
        int n = nP * nQ;
        if (!any_t) {
            for (int i = 0; i < n; ++i) out[i] = 0.0;
            return 0;
        }
        std::vector<double> pureout;
        transform_cart_to_pure2(shPsh, shQsh, cart_terf, pureout);
        if ((int)pureout.size() != n) return SCF_EINTERNAL;
        for (int i = 0; i < n; ++i) out[i] = pureout[i];
        return n;
    } catch (...) {
        return SCF_EINTERNAL;
    }
}
