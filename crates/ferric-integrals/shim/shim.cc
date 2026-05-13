// Implementation of the goscf libint2 shim. See shim.h for the contract.
#include "shim.h"

#include <libint2.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <atomic>
#include <new>

using libint2::Engine;
using libint2::Operator;
using libint2::Shell;
using libint2::BasisSet;

struct goscf_basis {
    BasisSet           bs;
    std::vector<int>   nfunc;       // nfunc per shell: (2L+1) if pure, (L+1)(L+2)/2 if Cartesian
    int                max_nprim;
    int                max_L;
};

struct goscf_engine {
    Engine engine;
};

static std::atomic<int> libint_init_count{0};

void goscf_libint_init(void) {
    if (libint_init_count.fetch_add(1) == 0) {
        libint2::initialize();
    }
}

void goscf_libint_finalize(void) {
    if (libint_init_count.fetch_sub(1) == 1) {
        libint2::finalize();
    }
}

goscf_basis *goscf_basis_create(const goscf_shell *shells, int nshells,
                                const goscf_atom *atoms, int natoms) {
    try {
        // Build per-atom Atom records (libint type) for nuclear positions.
        std::vector<libint2::Atom> li_atoms(natoms);
        for (int a = 0; a < natoms; ++a) {
            li_atoms[a].atomic_number = atoms[a].Z;
            li_atoms[a].x = atoms[a].x;
            li_atoms[a].y = atoms[a].y;
            li_atoms[a].z = atoms[a].z;
        }
        // Build the libint shell list, one libint::Shell per goscf_shell.
        std::vector<Shell> li_shells;
        li_shells.reserve(nshells);
        for (int s = 0; s < nshells; ++s) {
            const goscf_shell &g = shells[s];
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
        auto *out = new (std::nothrow) goscf_basis{std::move(bs), {}, 0, 0};
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

void goscf_basis_destroy(goscf_basis *bs) {
    delete bs;
}

int goscf_basis_nbasis(const goscf_basis *bs) {
    return static_cast<int>(bs->bs.nbf());
}

int goscf_basis_nshells(const goscf_basis *bs) {
    return static_cast<int>(bs->bs.size());
}

void goscf_basis_shell_dims(const goscf_basis *bs, int *out) {
    for (size_t i = 0; i < bs->nfunc.size(); ++i) out[i] = bs->nfunc[i];
}

void goscf_basis_max_dims(const goscf_basis *bs, int *max_nprim, int *max_L) {
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

goscf_engine *goscf_engine_create(int op_kind, double omega,
                                  int max_nprim, int max_L, double precision) {
    bool ok = false;
    Operator op = op_for_kind(op_kind, &ok);
    if (!ok) return nullptr;
    try {
        Engine eng(op, max_nprim, max_L, 0, precision);
        if (op_kind == 1 || op_kind == 2) {
            // ErfCoulomb / ErfcCoulomb attenuation parameter.
            eng.set_params(omega);
        }
        auto *out = new (std::nothrow) goscf_engine{std::move(eng)};
        return out;
    } catch (...) {
        return nullptr;
    }
}

goscf_engine *goscf_engine_create_deriv(int op_kind, double omega,
                                        int max_nprim, int max_L, double precision) {
#if LIBINT2_MAX_DERIV_ORDER >= 1
    bool ok = false;
    Operator op = op_for_kind(op_kind, &ok);
    if (!ok) return nullptr;
    try {
        Engine eng(op, max_nprim, max_L, 1, precision);
        if (op_kind == 1 || op_kind == 2) {
            eng.set_params(omega);
        }
        auto *out = new (std::nothrow) goscf_engine{std::move(eng)};
        return out;
    } catch (...) {
        return nullptr;
    }
#else
    return nullptr;
#endif
}

void goscf_engine_destroy(goscf_engine *eng) {
    delete eng;
}

int goscf_engine_set_point_charges(goscf_engine *eng,
                                   const goscf_atom *atoms, int natoms) {
    try {
        std::vector<std::pair<double, std::array<double, 3>>> q(natoms);
        for (int a = 0; a < natoms; ++a) {
            q[a].first = static_cast<double>(atoms[a].Z);
            q[a].second = {atoms[a].x, atoms[a].y, atoms[a].z};
        }
        eng->engine.set_params(q);
        return GOSCF_OK;
    } catch (...) {
        return GOSCF_EINTERNAL;
    }
}

int goscf_compute_1e_block(goscf_engine *eng, const goscf_basis *bs,
                           int sh1, int sh2, double *out) {
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
}

int goscf_compute_eri_quartet(goscf_engine *eng, const goscf_basis *bs,
                              int sh1, int sh2, int sh3, int sh4, double *out) {
    const auto &shells = bs->bs;
    eng->engine.compute(shells[sh1], shells[sh2], shells[sh3], shells[sh4]);
    const auto &result = eng->engine.results();
    if (result[0] == nullptr) {
        return 0;  // libint screened the quartet (all zero).
    }
    int n = bs->nfunc[sh1] * bs->nfunc[sh2] * bs->nfunc[sh3] * bs->nfunc[sh4];
    for (int i = 0; i < n; ++i) out[i] = result[0][i];
    return n;
}

void goscf_compute_schwarz(goscf_engine *eng, const goscf_basis *bs, double *qmat) {
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
}

int goscf_compute_1e_deriv_block(goscf_engine *eng, const goscf_basis *bs,
                                 int sh1, int sh2, double *out) {
#if LIBINT2_MAX_DERIV_ORDER >= 1
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
#else
    (void)eng; (void)bs; (void)sh1; (void)sh2; (void)out;
    return 0;
#endif
}

int goscf_compute_eri_deriv_quartet(goscf_engine *eng, const goscf_basis *bs,
                                    int sh1, int sh2, int sh3, int sh4, double *out) {
#if LIBINT2_MAX_DERIV_ORDER >= 1
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
#else
    (void)eng; (void)bs; (void)sh1; (void)sh2; (void)sh3; (void)sh4; (void)out;
    return 0;
#endif
}

/* --- 3-center and 2-center ERI engines for density fitting / RI --- */

goscf_engine *goscf_engine_create_3center(int op_kind, double omega,
                                          int max_nprim, int max_L, double precision) {
#if LIBINT2_SUPPORT_ERI3
    bool ok = false;
    Operator op = op_for_kind(op_kind, &ok);
    if (!ok) return nullptr;
    try {
        Engine eng(op, max_nprim, max_L, 0, precision);
        eng.set(libint2::BraKet::xs_xx);
        if (op_kind == 1 || op_kind == 2) eng.set_params(omega);
        return new (std::nothrow) goscf_engine{std::move(eng)};
    } catch (...) {
        return nullptr;
    }
#else
    (void)op_kind; (void)omega; (void)max_nprim; (void)max_L; (void)precision;
    return nullptr;
#endif
}

goscf_engine *goscf_engine_create_2center(int op_kind, double omega,
                                          int max_nprim, int max_L, double precision) {
#if LIBINT2_SUPPORT_ERI2
    bool ok = false;
    Operator op = op_for_kind(op_kind, &ok);
    if (!ok) return nullptr;
    try {
        Engine eng(op, max_nprim, max_L, 0, precision);
        eng.set(libint2::BraKet::xs_xs);
        if (op_kind == 1 || op_kind == 2) eng.set_params(omega);
        return new (std::nothrow) goscf_engine{std::move(eng)};
    } catch (...) {
        return nullptr;
    }
#else
    (void)op_kind; (void)omega; (void)max_nprim; (void)max_L; (void)precision;
    return nullptr;
#endif
}

int goscf_compute_eri3(goscf_engine *eng, const goscf_basis *obs,
                       const goscf_basis *dfbs,
                       int shP, int sh1, int sh2, double *out) {
#if LIBINT2_SUPPORT_ERI3
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
#else
    (void)eng; (void)obs; (void)dfbs; (void)shP; (void)sh1; (void)sh2; (void)out;
    return 0;
#endif
}

int goscf_compute_eri2(goscf_engine *eng, const goscf_basis *dfbs,
                       int shP, int shQ, double *out) {
#if LIBINT2_SUPPORT_ERI2
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
#else
    (void)eng; (void)dfbs; (void)shP; (void)shQ; (void)out;
    return 0;
#endif
}
