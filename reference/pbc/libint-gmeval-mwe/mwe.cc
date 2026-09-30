// Minimal reproducer: libint2 erf/erfc two-electron integrals scale poorly
// across threads because GenericGmEval::eval copies the Gm evaluator per
// primitive quartet (heap scratch for erfc + shared_ptr copy of the one
// process-wide Boys table => malloc/free + atomic refcount on a shared cache
// line). Plain Coulomb calls the table by reference and scales normally.
//
// NOT YET COMPILED OR RUN. Written from the libint 2.7.2 headers
// (~/.local/include/libint2); build/run lines are in run.sh.
//
// Each worker thread owns its own Engine (the documented libint2 usage) and
// runs the same fixed list of shell quartets `calls` times. Nothing is shared
// between threads except what libint2 itself shares. The program prints the
// wall ns per call of every thread and a checksum of all integrals.
//
// Usage: mwe <coulomb|erf|erfc> <nthreads> <calls>
//
// Expected on the machine where the ferric measurement was made (6-core
// Broadwell-E, libint 2.7.2, the production erfc 3-centre path: 3.28x
// per-call slowdown at 6 threads vs 1.05x for 6 separate 1-thread processes):
//   erfc/erf, 6 threads : per-call time several x the 1-thread time
//   erfc/erf, 6 processes of 1 thread each : ~1x
//   coulomb, 6 threads  : ~1x (no evaluator copy on that path)

#include <libint2.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace {

using libint2::BraKet;
using libint2::Engine;
using libint2::Operator;
using libint2::Shell;

// Carbon-like contracted s and p shells (cc-pVDZ-style exponents, several
// primitives each => many primitive quartets per call, so the per-quartet
// evaluator copy dominates).
std::vector<Shell> make_shells() {
  const libint2::svector<double> e_s = {6665.0, 1000.0, 228.0, 64.71, 21.06, 7.495, 2.797, 0.5215, 0.1596};
  const libint2::svector<double> c_s = {0.000692, 0.005329, 0.027077, 0.101718, 0.27474,
                                   0.448564, 0.285074, 0.015204, -0.003191};
  const libint2::svector<double> e_p = {9.439, 2.002, 0.5456, 0.1517};
  const libint2::svector<double> c_p = {0.038109, 0.20948, 0.508557, 0.468842};
  std::vector<Shell> shells;
  const double centers[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 2.2}, {1.9, 0.4, -0.8}};
  for (const auto& c : centers) {
    shells.push_back(Shell{e_s, {{0, false, c_s}}, {{c[0], c[1], c[2]}}});
    shells.push_back(Shell{e_p, {{1, false, c_p}}, {{c[0], c[1], c[2]}}});
  }
  return shells;
}

Engine make_engine(const char* op, size_t max_nprim, int max_l) {
  const double omega = 1.0;  // bohr^-1, as in ferric's RS-GDF default
  const double eps = std::numeric_limits<double>::epsilon();
  if (std::strcmp(op, "erfc") == 0)
    return Engine(Operator::erfc_coulomb, max_nprim, max_l, 0, eps, omega);
  if (std::strcmp(op, "erf") == 0)
    return Engine(Operator::erf_coulomb, max_nprim, max_l, 0, eps, omega);
  return Engine(Operator::coulomb, max_nprim, max_l, 0, eps);
}

uint64_t mix(uint64_t h, double v) {
  uint64_t b;
  std::memcpy(&b, &v, sizeof b);
  h ^= b + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  return h;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <coulomb|erf|erfc> <nthreads> <calls>\n", argv[0]);
    return 2;
  }
  const char* op = argv[1];
  const int nthreads = std::atoi(argv[2]);
  const long calls = std::atol(argv[3]);

  libint2::initialize();
  const auto shells = make_shells();
  size_t max_nprim = 0;
  int max_l = 0;
  for (const auto& s : shells) {
    max_nprim = std::max(max_nprim, s.nprim());
    max_l = std::max(max_l, s.contr[0].l);
  }
  // A fixed quartet list: every (i j | k l) over the 6 shells.
  std::vector<std::array<int, 4>> quartets;
  const int n = static_cast<int>(shells.size());
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k)
        for (int l = 0; l < n; ++l) quartets.push_back({i, j, k, l});

  std::vector<double> ns_per_call(nthreads, 0.0);
  std::vector<uint64_t> hashes(nthreads, 0);
  std::atomic<int> ready{0};

  auto worker = [&](int t) {
    Engine engine = make_engine(op, max_nprim, max_l);  // one engine per thread
    const auto& buf = engine.results();
    ready.fetch_add(1);
    while (ready.load() < nthreads) {
    }  // start together
    uint64_t h = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (long c = 0; c < calls; ++c) {
      const auto& q = quartets[c % quartets.size()];
      engine.compute(shells[q[0]], shells[q[1]], shells[q[2]], shells[q[3]]);
      if (c < static_cast<long>(quartets.size()) && buf[0] != nullptr) {
        const size_t len = shells[q[0]].size() * shells[q[1]].size() * shells[q[2]].size() *
                           shells[q[3]].size();
        for (size_t x = 0; x < len; ++x) h = mix(h, buf[0][x]);
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    ns_per_call[t] =
        std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(calls);
    hashes[t] = h;
  };

  std::vector<std::thread> pool;
  for (int t = 0; t < nthreads; ++t) pool.emplace_back(worker, t);
  for (auto& th : pool) th.join();

  double mean = 0.0;
  for (double v : ns_per_call) mean += v / nthreads;
  std::printf("op %s threads %d calls/thread %ld quartets %zu: mean %.1f ns/call", op, nthreads,
              calls, quartets.size(), mean);
  for (double v : ns_per_call) std::printf(" %.1f", v);
  std::printf("  checksum %016llx\n", static_cast<unsigned long long>(hashes[0]));
  for (int t = 1; t < nthreads; ++t)
    if (hashes[t] != hashes[0]) std::printf("WARNING: thread %d checksum differs\n", t);

  libint2::finalize();
  return 0;
}
