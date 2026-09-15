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
#include <limits>

using libint2::Engine;
using libint2::Operator;
using libint2::Shell;
using libint2::BasisSet;
using libint2::ShellPair;
using libint2::BraKet;

struct scf_basis {
    BasisSet           bs;
    std::vector<int>   nfunc;       // nfunc per shell: (2L+1) if pure, (L+1)(L+2)/2 if Cartesian
    int                max_nprim;
    int                max_L;
};

// Forward declaration; the terfc table set is defined further down.
struct TerfcTableSet;

/* ==========================================================================
 *  Per-engine ShellPair cache.
 *
 *  libint2's Engine::compute(sh1,sh2,sh3,sh4) 4-argument overload forwards
 *  to compute2<...>(sh1,sh2,sh3,sh4,nullptr,nullptr): passing null for the
 *  precomputed bra/ket ShellPair forces libint2 to call ShellPair::init(...)
 *  from scratch for BOTH pairs on EVERY quartet (engine.impl.h,
 *  Engine::compute2: `spbra_.init(bra1, bra2, target_shellpair_ln_precision,
 *  screening_method_)` when no precomputed pair is supplied or the supplied
 *  one was built at a looser precision). A `perf` profile of benzene/
 *  aug-cc-pVDZ RHF showed ShellPair::init + its internal exp() calls at
 *  ~7.5% of total runtime, entirely redundant: shell pairs recur constantly
 *  across a quartet sweep (scatter_bra_pair fixes the bra pair for an entire
 *  ket loop; ket pairs themselves recur across bra iterations too).
 *
 *  This cache stores one ShellPair per unique ORDERED shell index pair (i,j)
 *  requested (see "Why NOT triangular" below), built once per engine and
 *  reused for every quartet that touches it.
 *
 *  --- Precision matching (the whole correctness question) ---
 *  ShellPair::init's `ln_prec` argument is compared against a per-primitive-
 *  pair log-screening factor; get it wrong and primitive pairs are
 *  included/excluded differently than the uncached path, silently changing
 *  integrals (see schwarz.rs's SCHWARZ_TABLE_PRECISION comment for the same
 *  class of bug in a different table). libint2 itself computes:
 *      ln_precision_ = (precision_ > 0) ? log(precision_)
 *                                       : numeric_limits<scalar_type>::lowest();
 *  (engine.h, Engine::set_precision) and then uses exactly that value as
 *  `target_shellpair_ln_precision` for BOTH bra and ket in the nullptr path
 *  (engine.impl.h, Engine::compute2). `ln_precision_` has no public accessor,
 *  but `Engine::precision()` (the pre-log value passed to set_precision) does
 *  -- so we reproduce the identical formula from it. A cached pair built at
 *  ln_prec == target_shellpair_ln_precision makes libint2's own
 *  `recompute_spbra = pair->ln_prec > target_shellpair_ln_precision` check
 *  evaluate false (not `>`), so the precomputed pair is accepted AS-IS: this
 *  is not an approximation of the uncached path, it is the same computation
 *  memoized. `screening_method_` must also match (engine.impl.h asserts
 *  `spbra.screening_method_ == screening_method_`); scf_engine_create never
 *  overrides it, so it is always `Engine::screening_method()`'s value (a
 *  public accessor), read at cache-build time.
 *
 *  --- Ownership / lifetime ---
 *  Owned by scf_engine (one cache per libint2::Engine), not scf_basis. Engines
 *  are per-thread (ferric's EnginePool: one Engine per rayon worker via a
 *  Mutex<Engine>, never shared live across threads), so a per-engine cache
 *  needs no locking. A composite ferric Engine (linear combination of several
 *  scf_engine handles, e.g. range-separated operators) gets one cache PER
 *  scf_engine handle, which is correct: ShellPair data depends only on
 *  geometry + ln_prec + screening_method, all identical across the
 *  composite's component engines in practice, but there is no cross-engine
 *  sharing here to keep the invariant trivially local (each scf_engine is
 *  self-contained).
 *
 *  --- Basis invalidation ---
 *  Keyed by the `scf_basis*` pointer last used to populate it: a compute call
 *  against a different basis clears and rebuilds. This is a pointer-identity
 *  check only (cheap), sufficient because scf_basis is immutable after
 *  scf_basis_create (see basis_bridge.rs: PreparedBasis has no mutation API)
 *  -- the shell array a given scf_basis* denotes never changes underneath it.
 *
 *  --- Memory ---
 *  Storage is a flat nshells*nshells matrix of lazily-built slots (NOT
 *  triangular -- see below for why), each a `ShellPair` plus a `built` flag.
 *  Slots are allocated up front as empty (`ShellPair()`'s default ctor is a
 *  few small members, no heap use until `init` runs -- see the byte estimate
 *  below), and each ShellPair's internal `primpairs` vector is populated only
 *  when that exact ordered pair is first requested. No unbounded growth: slot
 *  count is fixed at cache-build time from nshells, and each populated
 *  ShellPair holds at most max_nprim^2 PrimPairData entries (its primitive-
 *  pair count for that specific shell pair, typically far below max_nprim^2
 *  after screening).
 *
 *  --- Why NOT triangular ---
 *  `ShellPair::init(s1, s2, ...)` stores a DIRECTED `AB = s1.O - s2.O` vector.
 *  Engine::compute2 accepts a precomputed pair only for the EXACT (tbra1,
 *  tbra2) / (tket1,tket2) order the shells were passed to compute() in (it
 *  internally permutes the pair's AB sign via `swap_bra`/`swap_ket` ONLY
 *  relative to libint2's own canonical angular-momentum ordering, not
 *  relative to whatever order the pair was cached under) -- see
 *  engine.impl.h: `BA[xyz] = -spbra_precomputed->AB[xyz]` is applied only
 *  when `spbra_is_swapped` (i.e. when the SUPPLIED pair is precomputed AND
 *  the shells needed permuting for libint2's internal convention), not as a
 *  general "we'll sort it out" step. A pair built as `init(shells[j],
 *  shells[i], ...)` and handed back for a call passing `(shells[i],
 *  shells[j])` in that order would silently carry a sign-flipped `AB` (and,
 *  in general, swap primitive-pair `P`, `nonsph_screen_fac`, etc.) relative
 *  to what the uncached path would have built -- exactly the class of
 *  "compiles, computes, silently wrong" bug this whole project's
 *  Experimental Protocol warns about. Rather than re-deriving which specific
 *  fields are safe to swap post hoc, this cache keys strictly on the ORDERED
 *  pair (i,j) as requested and never reuses a slot across (i,j) and (j,i).
 *  ferric's own call sites (scatter_bra_pair, qqr.rs, schwarz.rs) already
 *  request shells in a canonical i>=j order for one of bra/ket, so in
 *  practice only close to half the matrix is ever populated -- this is a
 *  safety-over-density tradeoff, not a missed optimization.
 * ========================================================================== */
struct ShellPairCacheEntry {
    ShellPair pair;
    bool      built = false;
};

struct ShellPairCache {
    // Flat nshells x nshells matrix, ordered-pair-keyed (see doc above).
    std::vector<ShellPairCacheEntry> slots;
    int                              nshells = 0;
    const void                      *basis_key = nullptr;  // scf_basis* last used
    bool                             enabled = true;
    bool                             env_checked = false;

    inline size_t index(int i, int j) const {
        return (size_t)i * (size_t)nshells + (size_t)j;
    }

    void ensure_sized_for(const scf_basis *bs) {
        if (basis_key != bs) {
            // Basis changed (or first use): drop everything and resize fresh.
            // A stale ShellPair computed from another basis's shells would be
            // a use-after-free hazard (Shell references live inside the old
            // BasisSet) as well as numerically wrong, so a full rebuild (not a
            // partial invalidation) is the only safe response here.
            nshells = static_cast<int>(bs->bs.size());
            slots.clear();
            slots.resize((size_t)nshells * (size_t)nshells);
            basis_key = bs;
        }
    }

    // Read the FERRIC_SHELLPAIR_CACHE kill switch exactly once (lazily, on
    // first real use) so a normal run pays one getenv call, and tests that
    // mutate the env var before constructing/using a fresh engine still see
    // the intended value (each new scf_engine gets its own ShellPairCache
    // with env_checked=false).
    void check_env_once() {
        if (env_checked) return;
        env_checked = true;
        const char *v = std::getenv("FERRIC_SHELLPAIR_CACHE");
        if (v && (std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0 ||
                  std::strcmp(v, "false") == 0 || std::strcmp(v, "OFF") == 0 ||
                  std::strcmp(v, "FALSE") == 0)) {
            enabled = false;
        }
    }

    // Look up (or lazily build) the ShellPair for the ORDERED pair (i,j) at
    // the given engine precision/screening method. Returns nullptr if the
    // cache is disabled (caller falls back to the uncached nullptr,nullptr
    // path). (i,j) and (j,i) are DISTINCT slots -- see the class doc above
    // for why this must not be canonicalized.
    const ShellPair *get(const scf_basis *bs, int i, int j, double ln_prec,
                        libint2::ScreeningMethod screening_method) {
        check_env_once();
        if (!enabled) return nullptr;
        ensure_sized_for(bs);
        ShellPairCacheEntry &slot = slots[index(i, j)];
        if (!slot.built) {
            slot.pair.init(bs->bs[i], bs->bs[j], ln_prec, screening_method);
            slot.built = true;
        }
        return &slot.pair;
    }

    void set_enabled(bool e) {
        env_checked = true;  // explicit override wins over the env var
        enabled = e;
    }
};

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
    // Explicit null default so aggregate-init sites (scf_engine{...}) that
    // omit trailing members compile clean under -Wmissing-field-initializers
    // (value-init already produced null; this only silences the warning).
    std::shared_ptr<TerfcTableSet>  terfc_tables = nullptr;
    // Lazily-populated shell-pair cache for scf_compute_eri_quartet. Declared
    // last so every existing aggregate-init site (scf_engine{std::move(eng)},
    // scf_engine{std::move(eng), ...}) keeps compiling: default-constructed,
    // trailing member, no positional initializer needed.
    // Explicit default-init for the same reason `terfc_tables` carries one
    // above: the eight `scf_engine{...}` aggregate-init sites in this file all
    // omit the trailing members, which is correct (value-init already runs
    // ShellPairCache's own default ctor) but trips
    // -Wmissing-field-initializers on every one of them. This only silences
    // the warning; it changes no behavior.
    ShellPairCache                  shellpair_cache = {};
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
        // Yukawa / screened Coulomb exp(-zeta r)/r. libint2 aliases this as
        // Operator::yukawa == Operator::stg_x_coulomb (TennoGmEval core).
        case 3:   return Operator::stg_x_coulomb;
        // Exact (unfitted) Slater-type geminal exp(-zeta r) (TennoGmEval core).
        case 4:   return Operator::stg;
        case 100: return Operator::overlap;
        case 101: return Operator::kinetic;
        case 102: return Operator::nuclear;
        default:  *ok = false; return Operator::coulomb;
    }
}

// True for op_kinds whose libint2 operator takes a single scalar attenuation/
// decay parameter set post-construction via Engine::set_params(double):
//   1 = erf_coulomb (omega), 2 = erfc_coulomb (omega),
//   3 = stg_x_coulomb/yukawa (zeta), 4 = stg (zeta).
// The geminal ops (200/202) instead pass a ContractedGaussianGeminal through the
// constructor and are handled in scf_engine_create_geminal, not here.
static bool op_needs_scalar_param(int kind) {
    return kind == 1 || kind == 2 || kind == 3 || kind == 4;
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
        if (op_needs_scalar_param(op_kind)) {
            // Scalar attenuation/decay parameter: erf/erfc omega, or stg/yukawa zeta.
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
        if (op_needs_scalar_param(op_kind)) {
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

// Reproduces libint2's Engine::set_precision formula (engine.h) from the
// public Engine::precision() accessor, since ln_precision_ itself has no
// public getter. MUST stay byte-identical to that formula: it is what makes
// a cached ShellPair's ln_prec compare equal (not just close) to the
// target_shellpair_ln_precision libint2 computes internally for the
// nullptr,nullptr path (engine.impl.h, Engine::compute2), which is the whole
// basis for claiming the cached and uncached paths compute the same thing.
static inline double ln_precision_of(const Engine &engine) {
    const double prec = engine.precision();
    if (prec > 0.0) {
        return std::log(prec);
    }
    return std::numeric_limits<double>::lowest();
}

// Looks up the compute2 function pointer for `engine`'s current
// (operator, braket, deriv_order) via the SAME public dispatch table
// Engine::compute()'s 4-shell overload uses internally (engine.impl.h lines
// ~146-163), but exposes the spbra/spket parameters that overload hardcodes
// to nullptr. All symbols used here (Engine::oper(), Engine::braket(),
// Engine::deriv_order(), Engine::compute2_ptrs(), Engine::compute2_ptr_type,
// libint2::nbrakets_2body, libint2::nderivorders_2body, Operator::
// first_2body_oper, BraKet::first_2body_braket) are public libint2 API --
// this is not a private-member reach-around, it is the identical arithmetic
// Engine::compute() performs, made externally callable so we can pass real
// ShellPair pointers instead of null ones.
static inline Engine::compute2_ptr_type quartet_compute_ptr(const Engine &engine) {
    // Signed arithmetic throughout (matches the *intent* of engine.impl.h's
    // own `Engine::compute`, which computes this same index as `auto` --
    // deduced size_t there only because it mixes with the size_t constants
    // below, making its own `compute_ptr_idx >= 0` assert vacuously true;
    // done in `long` here so the not-a-2-body-operator guard below is a real
    // check rather than dead code).
    const long oper_off = static_cast<long>(engine.oper()) -
                         static_cast<long>(Operator::first_2body_oper);
    const long braket_off = static_cast<long>(engine.braket()) -
                           static_cast<long>(BraKet::first_2body_braket);
    const long compute_ptr_idx =
        (oper_off * static_cast<long>(libint2::nbrakets_2body) + braket_off) *
            static_cast<long>(libint2::nderivorders_2body) +
        static_cast<long>(engine.deriv_order());
    const auto &ptrs = engine.compute2_ptrs();
    if (compute_ptr_idx < 0 ||
        static_cast<size_t>(compute_ptr_idx) >= ptrs.size()) {
        return nullptr;
    }
    return ptrs[static_cast<size_t>(compute_ptr_idx)];
}

int scf_compute_eri_quartet(scf_engine *eng, const scf_basis *bs,
                              int sh1, int sh2, int sh3, int sh4, double *out) {
  try {
    const auto &shells = bs->bs;
    Engine &engine = eng->engine;

    // Try the cached-ShellPair fast path first. Falls through to the
    // uncached libint2 default (compute(), which passes nullptr,nullptr and
    // re-runs ShellPair::init on every call) whenever: the cache is disabled
    // (FERRIC_SHELLPAIR_CACHE=0), or this engine's operator/braket/deriv
    // combination has no entry in compute2_ptrs() (e.g. an operator variant
    // outside the standard 2-body dispatch table -- defensive only, every
    // op_kind scf_engine_create supports is a standard 2-body operator).
    auto compute_ptr = quartet_compute_ptr(engine);
    const ShellPair *spbra = nullptr;
    const ShellPair *spket = nullptr;
    if (compute_ptr != nullptr) {
        const double ln_prec = ln_precision_of(engine);
        const auto screening_method = engine.screening_method();
        spbra = eng->shellpair_cache.get(bs, sh1, sh2, ln_prec, screening_method);
        if (spbra != nullptr) {
            spket = eng->shellpair_cache.get(bs, sh3, sh4, ln_prec, screening_method);
        }
    }

    if (compute_ptr != nullptr && spbra != nullptr && spket != nullptr) {
        (engine.*compute_ptr)(shells[sh1], shells[sh2], shells[sh3], shells[sh4],
                              spbra, spket);
    } else {
        engine.compute(shells[sh1], shells[sh2], shells[sh3], shells[sh4]);
    }
    const auto &result = engine.results();
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

void scf_engine_set_shellpair_cache_enabled(scf_engine *eng, int enabled) {
    eng->shellpair_cache.set_enabled(enabled != 0);
}

int scf_engine_shellpair_cache_enabled(const scf_engine *eng) {
    // check_env_once() is non-const (lazily latches the env-var read), so
    // mirror its logic read-only here rather than const_cast: if the env
    // check hasn't happened yet, report what it WOULD report (this getter is
    // test/diagnostic-only, never on the hot compute path).
    if (eng->shellpair_cache.env_checked) {
        return eng->shellpair_cache.enabled ? 1 : 0;
    }
    const char *v = std::getenv("FERRIC_SHELLPAIR_CACHE");
    if (v && (std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0 ||
              std::strcmp(v, "false") == 0 || std::strcmp(v, "OFF") == 0 ||
              std::strcmp(v, "FALSE") == 0)) {
        return 0;
    }
    return 1;
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

/* --- Cartesian second-moment integrals via emultipole2 --- */

int scf_compute_second_moment(const scf_basis *bs, const double *origin,
                              int nbas, double *out) {
    try {
        // Zero output: 6 matrices (xx, xy, xz, yy, yz, zz) of size nbas*nbas
        int total = 6 * nbas * nbas;
        for (int i = 0; i < total; ++i) out[i] = 0.0;

        // emultipole2 results: [overlap, x, y, z, xx, xy, xz, yy, yz, zz]
        Engine eng(Operator::emultipole2, bs->max_nprim, bs->max_L, 0, 1e-14);
        std::array<double, 3> orig{origin[0], origin[1], origin[2]};
        eng.set_params(orig);

        const int nsh = static_cast<int>(bs->bs.size());
        std::vector<int> sh_off(nsh + 1, 0);
        for (int s = 0; s < nsh; ++s) sh_off[s + 1] = sh_off[s] + bs->nfunc[s];

        for (int s1 = 0; s1 < nsh; ++s1) {
            for (int s2 = 0; s2 <= s1; ++s2) {
                eng.compute(bs->bs[s1], bs->bs[s2]);
                const auto &result = eng.results();
                int n1 = bs->nfunc[s1];
                int n2 = bs->nfunc[s2];
                int o1 = sh_off[s1];
                int o2 = sh_off[s2];
                for (int c = 0; c < 6; ++c) {
                    double *mat = out + c * nbas * nbas;
                    const double *src = result[c + 4];
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
        if (op_needs_scalar_param(op_kind)) eng.set_params(omega);
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
        if (op_needs_scalar_param(op_kind)) eng.set_params(omega);
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
        if (op_needs_scalar_param(op_kind)) eng.set_params(omega);
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
        if (op_needs_scalar_param(op_kind)) eng.set_params(omega);
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
    // EXPERIMENT (2026-09-15): only n=0 is ever read by the live kernel, so
    // drop the n axis here. Footprint falls 12x and consecutive m become
    // contiguous (m-stride 12 doubles -> 1). Values are bit-identical: this
    // is a gather of the same entries, no arithmetic.
    {
        std::vector<double> packed((size_t)out.nS * out.ns * out.dimm);
        for (size_t iS = 0; iS < (size_t)out.nS; ++iS)
            for (size_t is = 0; is < (size_t)out.ns; ++is)
                for (size_t m = 0; m < (size_t)out.dimm; ++m)
                    packed[(iS * out.ns + is) * out.dimm + m] =
                        out.data[((iS * out.ns + is) * out.dimm + m) * out.dimn];
        out.data.swap(packed);
        out.dimn = 1;
    }
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
    // pts/unit is normally the shipped 16/8/4/2. FERRIC_TERF_PTS_DIV=N divides
    // every pts by N so a COARSER regenerated table set (same S/s coverage,
    // fewer points per unit) can be loaded with the correct delta_S/delta_s.
    // Experiment knob only -- the default path is unchanged.
    int ptsdiv = 1;
    if (const char *e = std::getenv("FERRIC_TERF_PTS_DIV")) {
        // Only 1 and 2 are valid: the shipped pts are 16/8/4/2, so any larger
        // divisor drives the last table's pts to 0 and delta = 1.0/0 = inf,
        // silently corrupting every lookup rather than failing.
        int v = atoi(e);
        if (v == 1 || v == 2) ptsdiv = v;
    }
    const Spec specs[] = {
        {16 / ptsdiv, 4.0,  2.0,  "16_4_2.bin"},
        {8  / ptsdiv, 10.0, 5.0,  "8_10_5.bin"},
        {4  / ptsdiv, 20.0, 20.0, "4_20_20.bin"},
        {2  / ptsdiv, 20.0, 80.0, "2_20_80.bin"},
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
//
//  FREQUENCY, measured — the series path is NOT rare, despite reading like an
//  edge case. Instrumented on decane / cc-pVDZ + cc-pVDZ-RI (terf 3-index
//  block): 40 785 656 table hits vs 8 330 944 series fallbacks = 17.0% of all
//  terf_aux calls, and `perf` puts terf_G_series at 18.4% of total runtime
//  (second only to terf_aux itself at 29.6%). The coverage ceiling is
//  S_max = 20 across every registered table, and tight primitives on separated
//  centers clear it routinely.
//
//  So the obvious next optimization on this kernel is NOT micro-tuning the
//  interpolation further — it is extending table coverage past S = 20 (or
//  giving the series a cheaper large-S asymptotic form), which would convert
//  ~18% of runtime into ~10x-cheaper table lookups. Not attempted here: it
//  changes numerical output in the far field, so it needs its own accuracy
//  gate against the series, which is the exact reference.
// -------------------------------------------------------------------------
inline bool terf_aux(const TerfcTableSet &set, double S, double s,
                     double phi_over_theta, int mmax, double *A) {
    // Fast path: every m shares the SAME (S, s), so the table selection, the
    // node windows and the Lagrange WEIGHTS are identical across m -- only the
    // tabulated values differ. The old loop called interp_G once per m, which
    // recomputed all of that mmax+1 times; lagrange_1d is O(K^2) with a divide
    // in its inner loop and runs 11 times per interp_G, so that redundancy
    // dominated the terf kernel (measured: terf eri3 31.5 s vs Coulomb 0.54 s
    // on decane/cc-pVDZ+RI).
    //
    // Hoist everything (S, s)-dependent, then loop m innermost over a plain
    // weighted sum of table reads. Numerically this is the SAME interpolation:
    // same nodes, same weights, same summation order over (a, c) -- the weights
    // are merely computed once instead of mmax+1 times. Pinned bit-identical
    // against the reference path by tests/terf_table_interp_identity.rs.
    {
        const TerfcTable *tbl = nullptr;
        for (const auto &t : set.tables) {
            if (t.covers(S, s)) { tbl = &t; break; }
        }
        if (tbl) {
            const int K = 10;
            const int nS = tbl->nS, ns = tbl->ns;
            const double fS = S / tbl->delta_S;
            const double fs = s / tbl->delta_s;
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

            // Lagrange cardinal weights on consecutive-integer nodes. Computed
            // exactly as lagrange_1d does (same factor order), so the products
            // below reproduce its arithmetic term by term.
            double wS[16], ws[16];
            for (int j = 0; j < K_S; ++j) {
                const double xj = iS0 + j;
                double term = 1.0;
                for (int k = 0; k < K_S; ++k) {
                    if (k == j) continue;
                    const double xk = iS0 + k;
                    term *= (fS - xk) / (xj - xk);
                }
                wS[j] = term;
            }
            for (int j = 0; j < K_s; ++j) {
                const double xj = is0 + j;
                double term = 1.0;
                for (int k = 0; k < K_s; ++k) {
                    if (k == j) continue;
                    const double xk = is0 + k;
                    term *= (fs - xk) / (xj - xk);
                }
                ws[j] = term;
            }

            // n is ALWAYS 0 at every call site, so index the row directly and
            // walk m with the table's m-stride instead of re-deriving the
            // offset per lookup.
            const int dimn = tbl->dimn;
            for (int m = 0; m <= mmax; ++m) {
                double acc = 0.0;
                for (int a = 0; a < K_S; ++a) {
                    const int iS = iS0 + a;
                    double rowacc = 0.0;
                    for (int c = 0; c < K_s; ++c) {
                        rowacc += ws[c] * tbl->data[((((size_t)iS * ns) + (is0 + c)) * tbl->dimm + m) * dimn];
                    }
                    acc += wS[a] * rowacc;
                }
                A[m] = phi_over_theta * acc;
            }
            return true;
        }
    }
    // Outside all tables: exact series (reachable only for S > 20, s < 1/2).
    double gser[TERFC_DIMM];
    terf_G_series(S, s, mmax, gser);
    for (int m = 0; m <= mmax; ++m) A[m] = phi_over_theta * gser[m];
    return true;
}


/* ==========================================================================
 *  STEP 1 of the libint2 core-eval port (wiki/perf-tasks/terf-as-libint2-core-eval.md)
 *
 *  terf_gm_eval_impl is the argument-mapping core of a libint2
 *  `os_core_ints::terf_gm_eval<Real>`. libint2 hands a core evaluator
 *  (Gm, rho, T, mmax, <oper params>); terf_aux needs (S, s, phi_over_theta).
 *  The map, with theta2 == libint2's rho and T == theta2 * PQ2:
 *
 *      phi2           = rho * omega^2 / (rho + omega^2)
 *      S              = phi2 * PQ2   = T * (phi2 / rho)
 *      s              = phi2 * r0^2
 *      phi_over_theta = sqrt(phi2 / rho)
 *
 *  Nothing here is new numerics -- it is the SAME terf_aux on arguments
 *  recovered from libint2's convention. The gate
 *  scf_terf_gm_eval_matches_terf_aux() proves that claim bit-for-bit before a
 *  single libint2 header is patched. If it ever fails, the mapping is wrong
 *  and the port must stop.
 * ========================================================================== */
inline bool terf_gm_eval_impl(const TerfcTableSet &set, double rho, double T,
                              int mmax, double omega, double r0, double *Gm) {
    if (!(rho > 0.0) || !(omega > 0.0)) return false;
    const double omega2 = omega * omega;
    const double phi2 = rho * omega2 / (rho + omega2);
    // Recover PQ2 and form S exactly as compute_cart_eri3 does (S = phi2*PQ2),
    // NOT as T*(phi2/rho). The two are algebraically identical but round
    // differently: measured 32/336 sample points differ by up to 7.1e-15,
    // which propagated to 242/3024 mismatches in the gate. Matching the
    // reference's expression keeps the port bit-identical, so the gate tests
    // the MAPPING rather than a choice of floating-point association.
    const double PQ2 = T / rho;
    const double S = phi2 * PQ2;
    const double s = phi2 * r0 * r0;
    const double phi_over_theta = std::sqrt(phi2 / rho);
    return terf_aux(set, S, s, phi_over_theta, mmax, Gm);
}


#ifdef FERRIC_LIBINT2_TERF
/* Host hook for libint2's Operator::terf core evaluator (STEP 4).
 *
 * libint2 must not own the multi-MB interpolation tables, so terf_gm_eval calls
 * back here. Signature is fixed by libint2::os_core_ints::terf_gm_eval::hook_type.
 * Returns false only if the tables are unavailable.
 *
 * Thread-safety: get_terfc_tables caches a shared_ptr behind the process-global
 * g_terfc_tables; the tables are immutable once loaded and terf_gm_eval_impl
 * only reads them, so concurrent rayon workers are safe. The FIRST call must
 * happen before threads fan out -- scf_engine_create_terf_libint2 forces the
 * load at engine-construction time for exactly that reason.
 */
bool ferric_terf_libint2_hook(double rho, double T, int mmax, double omega,
                              double r0, double *Gm) {
    auto set = g_terfc_tables;          // shared_ptr copy: no torn read
    if (!set) return false;
    if (!terf_gm_eval_impl(*set, rho, T, mmax, omega, r0, Gm)) return false;

    /* PREFACTOR CONVENTION -- the reason the first cut was ~200x off.
     *
     * Two independent mismatches had to be undone here:
     *
     * 1. terf_aux returns `sqrt(phi2/rho) * G_m`, a CONSTANT-in-m factor, and
     *    libint2 separately applies its own `pfac = K12*sqrt(gammapq)/gammapq`
     *    after this hook returns. Returning the baked-in factor double-counts.
     *
     * 2. Deeper: the MD driver feeds `alpha_R = phi2` (NOT rho) into its own
     *    Hermite-R recursion, so its reduced exponent differs from libint2's.
     *    libint2's generated recurrences are hardwired to rho, and the way
     *    every other attenuated operator reconciles that is an M-DEPENDENT
     *    power -- erf_coulomb_gm_eval scales by (w2/(w2+rho))^(m+1/2).
     *
     * So convert to libint2's convention: strip terf_aux's constant factor and
     * apply (phi2/rho)^(m+1/2) instead. At m=3 the two differ by 27x-3.6e4x
     * over the rho/omega range in play, which brackets the 1.989e2 the gate saw.
     */
    const double omega2 = omega * omega;
    const double phi2 = rho * omega2 / (rho + omega2);
    const double ratio = phi2 / rho;
    const double inv_const = 1.0 / std::sqrt(ratio);   // undo terf_aux's factor
    double pow_m = std::sqrt(ratio);                   // (ratio)^(m+1/2), m=0
    for (int m = 0; m <= mmax; ++m, pow_m *= ratio) {
        Gm[m] *= inv_const * pow_m;
    }
    return true;
}
#endif

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

// 4-center quartet (a b | c d): both sides are GENUINE Gaussian pairs.
//
// compute_cart_eri3 is the special case of this routine in which the bra side
// is a degenerate pair (exponent 0.0 partner, same center twice, lB=0) -- the
// "phantom" aux Gaussian. Here both sides call the full 4-argument
// build_hermite_E, so the bra pair (A,B) has exponent p = a+b and Gaussian
// product center P, and the ket pair (C,D) has q = c+d and center Q.
//
// build_hermite_R is already general in Ltot -- it is called here with
// Ltot = lA+lB+lC+lD instead of lP+lA+lB, nothing else changes.
//
// Output layout row-major [ncA][ncB][ncC][ncD] (A slowest, D fastest), matching
// the eri3 convention of "bra index slowest".
//
// tables/r0/omega used only when use_boys==false (terf path).
// Returns false if every primitive contribution was screened out of the tables.
// True when a quartet's total angular momentum exceeds the m-depth stored in
// the terf tables (and the Fn stack buffer sized to match). Entry points call
// this BEFORE computing so they can return a negative status; compute_cart_eri4
// itself can only answer `false`, which means "screened" to its callers.
inline bool eri4_Ltot_exceeds_tables(const Shell &shA, const Shell &shB,
                                     const Shell &shC, const Shell &shD) {
    return shA.contr[0].l + shB.contr[0].l + shC.contr[0].l + shD.contr[0].l >=
           TERFC_DIMM;
}

bool compute_cart_eri4(const Shell &shA, const Shell &shB,
                       const Shell &shC, const Shell &shD,
                       const TerfcTableSet *tables, double omega, double r0,
                       bool use_boys, std::vector<double> &out_cart) {
    const int lA = shA.contr[0].l;
    const int lB = shB.contr[0].l;
    const int lC = shC.contr[0].l;
    const int lD = shD.contr[0].l;
    const int Ltot = lA + lB + lC + lD;
    const int ncA = ncart_of(lA), ncB = ncart_of(lB);
    const int ncC = ncart_of(lC), ncD = ncart_of(lD);
    out_cart.assign((size_t)ncA * ncB * ncC * ncD, 0.0);

    // Fn is a fixed-size stack buffer of TERFC_DIMM entries; a quartet whose
    // total angular momentum exceeds it would overrun. Callers MUST pre-check
    // with eri4_Ltot_exceeds_tables() and return a negative status, because the
    // `false` return here is indistinguishable from "fully screened" and would
    // otherwise hand back zeros for a quartet that is merely too high-L -- a
    // silently wrong Schwarz bound rather than an error. This is a belt-and-
    // braces stop, not the reporting path. (Ltot <= 23 covers l <= 5 on all
    // four shells; eri3/eri2 need no such guard as lP+lA+lB cannot reach 24.)
    if (Ltot >= TERFC_DIMM) return false;

    std::vector<std::array<int, 3>> compA, compB, compC, compD;
    cart_components(lA, compA);
    cart_components(lB, compB);
    cart_components(lC, compC);
    cart_components(lD, compD);

    const auto &AO = shA.O;
    const auto &BO = shB.O;
    const auto &CO = shC.O;
    const auto &DO = shD.O;

    const double omega2 = omega * omega;
    const double r02 = r0 * r0;
    bool any = false;

    // Bra pair (a,b) -> p, P; ket pair (c,d) -> q, Q.
    for (size_t pa = 0; pa < shA.alpha.size(); ++pa) {
        const double a = shA.alpha[pa];
        const double ca = shA.contr[0].coeff[pa];
        for (size_t pb = 0; pb < shB.alpha.size(); ++pb) {
            const double b = shB.alpha[pb];
            const double cb = shB.contr[0].coeff[pb];
            const double p = a + b;
            const double Px = (a * AO[0] + b * BO[0]) / p;
            const double Py = (a * AO[1] + b * BO[1]) / p;
            const double Pz = (a * AO[2] + b * BO[2]) / p;
            // NOTE: the Gaussian-product factor exp(-(ab/p)|A-B|^2) is already
            // carried by the Hermite E-coefficients below; do NOT multiply it
            // in again here (same caveat as compute_cart_eri3).
            const double cab = ca * cb;

            // Hermite E-coefficients for the bra pair, hoisted out of the ket
            // loops exactly as eri3 hoists its obs-pair E's out of the aux loop.
            HermiteE Ex, Ey, Ez;
            build_hermite_E(a, b, AO[0], BO[0], lA, lB, Ex);
            build_hermite_E(a, b, AO[1], BO[1], lA, lB, Ey);
            build_hermite_E(a, b, AO[2], BO[2], lA, lB, Ez);

            for (size_t pc = 0; pc < shC.alpha.size(); ++pc) {
                const double c = shC.alpha[pc];
                const double cc = shC.contr[0].coeff[pc];
                for (size_t pd = 0; pd < shD.alpha.size(); ++pd) {
                    const double d = shD.alpha[pd];
                    const double cd = shD.contr[0].coeff[pd];
                    const double q = c + d;
                    const double Qx = (c * CO[0] + d * DO[0]) / q;
                    const double Qy = (c * CO[1] + d * DO[1]) / q;
                    const double Qz = (c * CO[2] + d * DO[2]) / q;
                    const double ccd = cc * cd;

                    HermiteE ECx, ECy, ECz;
                    build_hermite_E(c, d, CO[0], DO[0], lC, lD, ECx);
                    build_hermite_E(c, d, CO[1], DO[1], lC, lD, ECy);
                    build_hermite_E(c, d, CO[2], DO[2], lC, lD, ECz);

                    // theta2 = Coulomb reduced exponent (p q/(p+q)); phi2 folds
                    // in 1/omega^2 for the terf piece. Identical operator
                    // decomposition as eri3/eri2 -- see terf_aux().
                    const double theta2 = p * q / (p + q);
                    const double PQx = Px - Qx;
                    const double PQy = Py - Qy;
                    const double PQz = Pz - Qz;
                    const double PQ2 = PQx * PQx + PQy * PQy + PQz * PQz;

                    double alpha_R;             // reduced exponent for build_hermite_R
                    double Fn[TERFC_DIMM];      // Boys / terf-aux vector
                    if (use_boys) {
                        alpha_R = theta2;
                        boys_upto(Ltot, theta2 * PQ2 /* Boys T */, Fn);
                    } else {
                        // phi^2 = 1/(1/p + 1/q + 1/omega^2)
                        //       = theta2*omega2/(theta2+omega2)
                        const double phi2 = theta2 * omega2 / (theta2 + omega2);
                        const double S = phi2 * PQ2;
                        const double s = phi2 * r02;
                        const double phi_over_theta = std::sqrt(phi2 / theta2);
                        // terf_aux always succeeds (table interp, or exact
                        // series for far-field S > 20); the guard is defensive
                        // only. Skipping here would leave the full Coulomb
                        // value un-subtracted (terf -> 1/r at large r, NOT
                        // negligible).
                        if (!terf_aux(*tables, S, s, phi_over_theta, Ltot, Fn)) {
                            continue;
                        }
                        alpha_R = phi2;
                    }

                    // Standard MD two-pair prefactor. eri3 uses the identical
                    // expression with its aux exponent playing the role of the
                    // bra pair exponent -- it is the p<->aux special case of
                    // this, so it carries over unchanged.
                    const double pref =
                        2.0 * std::pow(M_PI, 2.5) / (p * q * std::sqrt(p + q));
                    const double scale = pref * cab * ccd;

                    HermiteR R;
                    build_hermite_R(Ltot, alpha_R, PQx, PQy, PQz, Fn, R);
                    any = true;

                    // (a b | c d) = scale *
                    //   sum_{tuv} Eab_{tuv} * sum_{t'u'v'} Ecd_{t'u'v'}
                    //     * (-1)^{t'+u'+v'} * R_{t+t', u+u', v+v'}
                    //
                    // SIGN CONVENTION: the (-1)^{t'+u'+v'} belongs to the KET
                    // (C,D) side, i.e. the side that is SUBTRACTED in
                    // PQ = P - Q. Evidence: compute_cart_eri2 has genuine
                    // (phantom-pair) Hermite sets on both sides with the same
                    // PQ = P_bra - Q_ket convention, and puts the sign on the
                    // (tq,uq,vq) = Q indices; compute_cart_eri3 uses
                    // PQ = P_aux - Q_obspair and puts the sign on the obs-pair
                    // (t,u,v) indices -- again the subtracted side. Both agree,
                    // so the ket carries the sign here.
                    for (int ia = 0; ia < ncA; ++ia) {
                        const int ax = compA[ia][0], ay = compA[ia][1], az = compA[ia][2];
                        for (int ib = 0; ib < ncB; ++ib) {
                            const int bx = compB[ib][0], by = compB[ib][1], bz = compB[ib][2];
                            for (int ic = 0; ic < ncC; ++ic) {
                                const int cx = compC[ic][0], cy = compC[ic][1],
                                          cz = compC[ic][2];
                                for (int id = 0; id < ncD; ++id) {
                                    const int dx = compD[id][0], dy = compD[id][1],
                                              dz = compD[id][2];
                                    double sum = 0.0;
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
                                                for (int tc = 0; tc <= cx + dx; ++tc) {
                                                    const double ecx = ECx.at(cx, dx, tc);
                                                    if (ecx == 0.0) continue;
                                                    for (int uc = 0; uc <= cy + dy; ++uc) {
                                                        const double ecy =
                                                            ECy.at(cy, dy, uc);
                                                        if (ecy == 0.0) continue;
                                                        for (int vc = 0; vc <= cz + dz;
                                                             ++vc) {
                                                            const double ecz =
                                                                ECz.at(cz, dz, vc);
                                                            if (ecz == 0.0) continue;
                                                            const double ecd =
                                                                ecx * ecy * ecz;
                                                            const double sgn =
                                                                ((tc + uc + vc) & 1)
                                                                    ? -1.0
                                                                    : 1.0;
                                                            sum += eab * ecd * sgn *
                                                                   R.at(t + tc, u + uc,
                                                                        v + vc);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    out_cart[(((size_t)ia * ncB + ib) * ncC + ic) * ncD +
                                             id] += scale * sum;
                                }
                            }
                        }
                    }
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

// 4-center: transform [ncA][ncB][ncC][ncD] -> [nA][nB][nC][nD], axis by axis,
// matching each shell's own pure/cartesian flag. Same strategy as
// transform_cart_to_pure3, with one more middle axis.
void transform_cart_to_pure4(const Shell &shA, const Shell &shB,
                             const Shell &shC, const Shell &shD,
                             const std::vector<double> &cart,
                             std::vector<double> &pureout) {
    const int lA = shA.contr[0].l, lB = shB.contr[0].l;
    const int lC = shC.contr[0].l, lD = shD.contr[0].l;
    const bool puA = shA.contr[0].pure, puB = shB.contr[0].pure;
    const bool puC = shC.contr[0].pure, puD = shD.contr[0].pure;
    const int ncA = ncart_of(lA), ncB = ncart_of(lB);
    const int ncC = ncart_of(lC), ncD = ncart_of(lD);
    const int nA = puA ? npure_of(lA) : ncA;
    const int nB = puB ? npure_of(lB) : ncB;
    const int nC = puC ? npure_of(lC) : ncC;
    const int nD = puD ? npure_of(lD) : ncD;

    // Axis A (outermost): one tform_axis pass with rest = ncB*ncC*ncD.
    std::vector<double> tmp1;  // [nA][ncB][ncC][ncD]
    if (puA && lA >= 1) {
        tform_axis(lA, ncB * ncC * ncD, cart, tmp1);
    } else {
        tmp1 = cart;
    }

    // Axis B: for each of the nA outer blocks, transform [ncB][ncC*ncD].
    std::vector<double> tmp2;  // [nA][nB][ncC][ncD]
    if (puB && lB >= 1) {
        tmp2.assign((size_t)nA * nB * ncC * ncD, 0.0);
        std::vector<double> sub_in((size_t)ncB * ncC * ncD), sub_out;
        for (int ia = 0; ia < nA; ++ia) {
            for (int i = 0; i < ncB * ncC * ncD; ++i)
                sub_in[i] = tmp1[((size_t)ia * ncB * ncC * ncD) + i];
            tform_axis(lB, ncC * ncD, sub_in, sub_out);
            for (int i = 0; i < nB * ncC * ncD; ++i)
                tmp2[((size_t)ia * nB * ncC * ncD) + i] = sub_out[i];
        }
    } else {
        tmp2 = tmp1;
    }

    // Axis C: for each of the nA*nB outer blocks, transform [ncC][ncD].
    std::vector<double> tmp3;  // [nA][nB][nC][ncD]
    if (puC && lC >= 1) {
        tmp3.assign((size_t)nA * nB * nC * ncD, 0.0);
        std::vector<double> sub_in((size_t)ncC * ncD), sub_out;
        const int nouter = nA * nB;
        for (int ob = 0; ob < nouter; ++ob) {
            for (int i = 0; i < ncC * ncD; ++i)
                sub_in[i] = tmp2[((size_t)ob * ncC * ncD) + i];
            tform_axis(lC, ncD, sub_in, sub_out);
            for (int i = 0; i < nC * ncD; ++i)
                tmp3[((size_t)ob * nC * ncD) + i] = sub_out[i];
        }
    } else {
        tmp3 = tmp2;
    }

    // Axis D (innermost): [nA*nB*nC][ncD] -> [nA*nB*nC][nD].
    if (puD && lD >= 1) {
        pureout.assign((size_t)nA * nB * nC * nD, 0.0);
        const int nrows = nA * nB * nC;
        const auto &coefs =
            libint2::solidharmonics::SolidHarmonicsCoefficients<double>::instance(lD);
        for (int row = 0; row < nrows; ++row) {
            const double *src = &tmp3[(size_t)row * ncD];
            double *dst = &pureout[(size_t)row * nD];
            for (int s = 0; s < nD; ++s) {
                const auto nc = coefs.nnz(s);
                const auto *cidx = coefs.row_idx(s);
                const auto *cval = coefs.row_values(s);
                double acc = 0.0;
                for (int ic = 0; ic < nc; ++ic) acc += cval[ic] * src[cidx[ic]];
                dst[s] = acc;
            }
        }
    } else {
        pureout = tmp3;
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

/* Validation hook: the 4-center MD path run with the plain Coulomb kernel
 * (use_boys=true, no tables), so it can be compared quartet-for-quartet against
 * libint2's own scf_compute_eri_quartet. This is THE check that validates the
 * new contraction -- the ket-side (-1)^(t+u+v) sign, the 2*pi^2.5/(p q sqrt(p+q))
 * prefactor, the [n1][n2][n3][n4] layout and the four-axis solid-harmonic
 * transform -- against an independent implementation. terf+terfc==coulomb cannot
 * do that job: both sides share this machinery, so an error cancels.
 * Writes n1*n2*n3*n4 doubles; returns that count, or a negative SCF_E* code. */
extern "C" int scf_debug_coulomb_eri4(const scf_basis *obs, int sh1, int sh2,
                                      int sh3, int sh4, double *out) {
    try {
        if (!obs || !out) return SCF_EINVAL;
        const Shell &shAsh = obs->bs[sh1];
        const Shell &shBsh = obs->bs[sh2];
        const Shell &shCsh = obs->bs[sh3];
        const Shell &shDsh = obs->bs[sh4];
        if (eri4_Ltot_exceeds_tables(shAsh, shBsh, shCsh, shDsh)) {
            return SCF_EINVAL;
        }
        const int n = obs->nfunc[sh1] * obs->nfunc[sh2] * obs->nfunc[sh3] *
                      obs->nfunc[sh4];
        std::vector<double> cart;
        const bool any = compute_cart_eri4(shAsh, shBsh, shCsh, shDsh, nullptr,
                                           0.0, 0.0, /*use_boys=*/true, cart);
        if (!any) {
            for (int i = 0; i < n; ++i) out[i] = 0.0;
            return n;  // genuinely screened; libint2 should agree it is ~0
        }
        std::vector<double> pureout;
        transform_cart_to_pure4(shAsh, shBsh, shCsh, shDsh, cart, pureout);
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

extern "C" int scf_compute_terfc_eri4(scf_engine *eng, const scf_basis *obs,
                                      int sh1, int sh2, int sh3, int sh4,
                                      double *out) {
    try {
        if (!eng || !eng->is_terfc || !eng->terfc_tables) return SCF_EINVAL;
        if (!obs || !out) return SCF_EINVAL;
        // All four shells come from the orbital basis: a (PQ|PQ) Schwarz /
        // CSB quartet is drawn from ONE basis, so there is no dfbs argument.
        const Shell &shAsh = obs->bs[sh1];
        const Shell &shBsh = obs->bs[sh2];
        const Shell &shCsh = obs->bs[sh3];
        const Shell &shDsh = obs->bs[sh4];
        // Too-high total L is a refusal, not a screened block: returning zeros
        // here would understate a Schwarz/CSB bound instead of reporting a gap.
        if (eri4_Ltot_exceeds_tables(shAsh, shBsh, shCsh, shDsh)) {
            return SCF_EINVAL;
        }

        // terfc = coulomb - terf. Both use the identical MD machinery (same
        // ordering, prefactor, normalisation, cart->pure transform), so the
        // subtraction is valid element-by-element in the Cartesian basis.
        std::vector<double> cart_coul, cart_terf;
        bool any_c = compute_cart_eri4(shAsh, shBsh, shCsh, shDsh, nullptr,
                                       eng->omega, eng->r0, /*use_boys=*/true,
                                       cart_coul);
        bool any_t = compute_cart_eri4(shAsh, shBsh, shCsh, shDsh,
                                       eng->terfc_tables.get(), eng->omega,
                                       eng->r0, /*use_boys=*/false, cart_terf);
        int n1 = obs->nfunc[sh1];
        int n2 = obs->nfunc[sh2];
        int n3 = obs->nfunc[sh3];
        int n4 = obs->nfunc[sh4];
        int n = n1 * n2 * n3 * n4;
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
        transform_cart_to_pure4(shAsh, shBsh, shCsh, shDsh, cart, pureout);
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

extern "C" int scf_compute_terf_eri4(scf_engine *eng, const scf_basis *obs,
                                     int sh1, int sh2, int sh3, int sh4,
                                     double *out) {
    try {
        if (!eng || !eng->is_terfc || !eng->is_terf_complement || !eng->terfc_tables) {
            return SCF_EINVAL;
        }
        if (!obs || !out) return SCF_EINVAL;
        const Shell &shAsh = obs->bs[sh1];
        const Shell &shBsh = obs->bs[sh2];
        const Shell &shCsh = obs->bs[sh3];
        const Shell &shDsh = obs->bs[sh4];
        // Too-high total L is a refusal, not a screened block: returning zeros
        // here would understate a Schwarz/CSB bound instead of reporting a gap.
        if (eri4_Ltot_exceeds_tables(shAsh, shBsh, shCsh, shDsh)) {
            return SCF_EINVAL;
        }

        // terf is the SAME Cartesian block terfc subtracts from Coulomb --
        // return it directly (no combine), so terf + terfc = coulomb exactly.
        std::vector<double> cart_terf;
        bool any_t = compute_cart_eri4(shAsh, shBsh, shCsh, shDsh,
                                       eng->terfc_tables.get(), eng->omega,
                                       eng->r0, /*use_boys=*/false, cart_terf);
        int n1 = obs->nfunc[sh1];
        int n2 = obs->nfunc[sh2];
        int n3 = obs->nfunc[sh3];
        int n4 = obs->nfunc[sh4];
        int n = n1 * n2 * n3 * n4;
        if (!any_t) {
            for (int i = 0; i < n; ++i) out[i] = 0.0;
            return 0;  // fully screened
        }
        std::vector<double> pureout;
        transform_cart_to_pure4(shAsh, shBsh, shCsh, shDsh, cart_terf, pureout);
        if ((int)pureout.size() != n) return SCF_EINTERNAL;
        for (int i = 0; i < n; ++i) out[i] = pureout[i];
        return n;
    } catch (...) {
        return SCF_EINTERNAL;
    }
}

/* Gate for STEP 1: terf_gm_eval_impl must reproduce terf_aux BIT-FOR-BIT when
 * fed libint2-convention arguments. Returns the number of (rho,T,omega,r0,m)
 * samples compared, or a negative SCF_E* code. `*mismatches` receives the count
 * of non-bit-identical values -- it MUST be 0. */
extern "C" int scf_terf_gm_eval_matches_terf_aux(const char *table_dir,
                                                 int *mismatches,
                                                 double *worst_abs_diff) {
    try {
        auto set = get_terfc_tables(resolve_table_dir(table_dir));
        if (!set) return SCF_EINVAL;
        int compared = 0, bad = 0;
        double worst = 0.0;
        const int mmax = 8;
        double A[TERFC_DIMM], B[TERFC_DIMM];
        // Span the regimes that matter: table-covered and series-fallback, tight
        // and diffuse primitives, both curvature-linked and decoupled omega.
        for (double rho : {0.05, 0.25, 1.0, 4.0, 20.0, 100.0}) {
            for (double pq2 : {0.0, 0.01, 0.5, 2.0, 10.0, 50.0, 200.0}) {
                for (double r0 : {0.75, 1.0, 2.0, 4.0}) {
                    for (double wmul : {1.0, 2.0}) {
                        const double omega = wmul / (r0 * std::sqrt(2.0));
                        const double T = rho * pq2;
                        // reference: the existing production path
                        const double omega2 = omega * omega;
                        const double phi2 = rho * omega2 / (rho + omega2);
                        const double S = phi2 * pq2;
                        const double sarg = phi2 * r0 * r0;
                        const double pot = std::sqrt(phi2 / rho);
                        const bool ok_ref = terf_aux(*set, S, sarg, pot, mmax, A);
                        const bool ok_new =
                            terf_gm_eval_impl(*set, rho, T, mmax, omega, r0, B);
                        if (ok_ref != ok_new) { ++bad; continue; }
                        if (!ok_ref) continue;
                        for (int m = 0; m <= mmax; ++m) {
                            ++compared;
                            unsigned long long ba, bb;
                            std::memcpy(&ba, &A[m], 8);
                            std::memcpy(&bb, &B[m], 8);
                            if (ba != bb) {
                                ++bad;
                                const double d = std::fabs(A[m] - B[m]);
                                if (d > worst) worst = d;
                            }
                        }
                    }
                }
            }
        }
        if (mismatches) *mismatches = bad;
        if (worst_abs_diff) *worst_abs_diff = worst;
        return compared;
    } catch (...) {
        return SCF_EINTERNAL;
    }
}


#ifdef FERRIC_LIBINT2_TERF
/* STEP 4: create a libint2 engine driving Operator::terf, so terf rides the
 * SAME generated recurrences as Coulomb/erfc instead of the hand-rolled MD
 * driver (which measured 25.4x slower than libint2 with no tables at all).
 *
 * `braket` selects 2-, 3- or 4-center: 2 -> xs_xs, 3 -> xs_xx, 4 -> xx_xx.
 * Loads the tables and installs the core-eval hook BEFORE the engine exists,
 * so no worker thread can race the first table load.
 */
extern "C" scf_engine *scf_engine_create_terf_libint2(double r0, double omega,
                                                      int braket, int max_nprim,
                                                      int max_L, double precision,
                                                      const char *table_dir) {
    std::lock_guard<std::mutex> lock(libint_ctor_mutex);
    try {
        auto tables = get_terfc_tables(resolve_table_dir(table_dir));
        if (!tables) return nullptr;      // tables missing => fail loudly
        libint2::os_core_ints::terf_gm_eval<double>::hook() =
            &ferric_terf_libint2_hook;

        libint2::BraKet bk;
        switch (braket) {
            case 2: bk = libint2::BraKet::xs_xs; break;
            case 3: bk = libint2::BraKet::xs_xx; break;
            case 4: bk = libint2::BraKet::xx_xx; break;
            default: return nullptr;
        }
        // NOTE: the 6th ctor arg is Params, NOT BraKet -- passing the BraKet
        // there silently mis-constructs and the engine fails. Follow the
        // working pattern used by scf_engine_create_3center: construct, then
        // .set(BraKet), then .set_params().
        auto *out = new (std::nothrow) scf_engine{
            Engine(libint2::Operator::terf, max_nprim, max_L, 0, precision)};
        if (!out) return nullptr;
        out->engine.set(bk);
        out->engine.set_params(std::array<double, 2>{{omega, r0}});
        // Both false: this engine is NOT the table-MD path. The compute
        // functions that branch on is_terfc must never see this handle --
        // it is driven through scf_compute_eri{2,3,_quartet} instead.
        out->is_terfc = false;
        out->is_terf_complement = false;
        out->r0 = r0;
        out->omega = omega;
        out->terfc_tables = std::move(tables);
        return out;
    } catch (...) {
        return nullptr;
    }
}
#endif

#ifdef FERRIC_LIBINT2_TERF
/* STEP 4 accuracy gate: the libint2-native terf path vs the hand-rolled MD
 * path, on the SAME 3-index block. These will NOT be bit-identical (different
 * recurrence order), so the caller compares against a stated tolerance.
 * Writes max|libint2 - md| to *max_abs and max|.|/max|md| to *max_rel.
 * Returns the number of elements compared, or a negative SCF_E* code. */
extern "C" int scf_terf_libint2_vs_md_eri3(const scf_basis *obs,
                                           const scf_basis *dfbs,
                                           int shP, int sh1, int sh2,
                                           double r0, double omega,
                                           const char *table_dir,
                                           double *max_abs, double *max_rel) {
    try {
        if (!obs || !dfbs) return -100;
        const int nP = dfbs->nfunc[shP], n1 = obs->nfunc[sh1], n2 = obs->nfunc[sh2];
        const int n = nP * n1 * n2;
        std::vector<double> md(n, 0.0), li(n, 0.0);

        // --- MD path (existing production kernel) ---
        scf_engine *emd = scf_engine_create_terf_3center(
            r0, omega, std::max(obs->max_nprim, dfbs->max_nprim),
            std::max(obs->max_L, dfbs->max_L), 0.0, table_dir);
        if (!emd) return -101;   // MD engine ctor failed
        const int wmd = scf_compute_terf_eri3(emd, obs, dfbs, shP, sh1, sh2, md.data());
        scf_engine_destroy(emd);
        if (wmd < 0) return wmd;

        // --- libint2-native path ---
        scf_engine *eli = scf_engine_create_terf_libint2(
            r0, omega, 3, std::max(obs->max_nprim, dfbs->max_nprim),
            std::max(obs->max_L, dfbs->max_L), 0.0, table_dir);
        if (!eli) return -102;   // libint2 terf engine ctor failed
        const int wli = scf_compute_eri3(eli, obs, dfbs, shP, sh1, sh2, li.data());
        scf_engine_destroy(eli);
        if (wli < 0) return wli;

        // Normalize by the block magnitude taken from BOTH sides, and skip
        // blocks that are entirely numerical noise. Without the floor, a block
        // whose largest element is ~1e-21 yields a meaningless ratio: that is
        // what made this gate report 2.287e1 while every physically
        // significant element agreed to 11 digits.
        double mabs = 0.0, scale = 0.0;
        for (int i = 0; i < n; ++i) {
            mabs = std::max(mabs, std::fabs(li[i] - md[i]));
            scale = std::max(scale, std::max(std::fabs(md[i]), std::fabs(li[i])));
        }
        if (scale < 1e-12) {   // nothing resolvable in this block
            if (max_abs) *max_abs = 0.0;
            if (max_rel) *max_rel = 0.0;
            return n;
        }
        if (std::getenv("FERRIC_TERF_DUMP")) {
            std::fprintf(stderr, "block (%d,%d,%d) n=%d  nP=%d n1=%d n2=%d\n",
                         shP, sh1, sh2, n, nP, n1, n2);
            for (int i = 0; i < n && i < 24; ++i) {
                const double r = (md[i] != 0.0) ? li[i] / md[i] : 0.0;
                std::fprintf(stderr, "  [%3d] md=% .10e  li=% .10e  li/md=% .6f\n",
                             i, md[i], li[i], r);
            }
        }
        if (max_abs) *max_abs = mabs;
        if (max_rel) *max_rel = mabs / scale;
        return n;
    } catch (...) {
        return SCF_EINTERNAL;
    }
}
#endif
