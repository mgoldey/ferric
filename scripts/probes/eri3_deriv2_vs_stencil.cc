// Compares libint's ANALYTIC second derivatives of the 3-centre integral
// (P|mu nu) with the atom-folded fourth-order stencil of
// ferric_scf::hessian_df_stencil, shell triple by shell triple.
//
// Needs a libint2 built with ERI3 derivative order 2 (the source build under
// ~/.local/libint2-2.13.1; the PyPI wheel's 2.7.2 "small" export aborts on a
// deriv-2 3-centre engine). Effective limit: max l over (P, mu, nu) <= 3.
//
//   FERRIC_DERIV2_DUMP=/tmp/d2.txt OPENBLAS_NUM_THREADS=1 cargo test \
//     -p ferric-scf --test df_hessian_stencil dump_for_libint_deriv2_probe \
//     -- --ignored
//   P=$HOME/.local/libint2-2.13.1
//   g++ -O2 -std=c++17 -I$P/include -I$P/include/libint2 -I/usr/include/eigen3 \
//     scripts/probes/eri3_deriv2_vs_stencil.cc -L$P/lib -Wl,-rpath,$P/lib \
//     -lint2 -o /tmp/eri3_deriv2_probe
//   /tmp/eri3_deriv2_probe /tmp/d2.txt
//
// Per case it prints max |analytic(folded to atoms) - stencil| and the max
// analytic element (to show the case is not vacuous).
#include <libint2.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

using namespace libint2;

static Shell read_shell(std::istream &in, int &l) {
  int pure, nprim;
  double c[3];
  in >> l >> pure >> nprim >> c[0] >> c[1] >> c[2];
  svector<double> e(nprim), co(nprim);
  for (auto &v : e) in >> v;
  for (auto &v : co) in >> v;
  return Shell{e, {{l, pure != 0, co}}, {c[0], c[1], c[2]}};
}

int main(int argc, char **argv) {
  initialize();
  std::ifstream in(argv[1]);
  int ncase;
  in >> ncase;
  double worst_all = 0, scale_all = 0;
  for (int cs = 0; cs < ncase; ++cs) {
    std::string tag;
    int lp, l1, l2, at[3], n;
    in >> tag >> lp >> l1 >> l2 >> at[0] >> at[1] >> at[2] >> n;
    int l;
    Shell sp = read_shell(in, l), s1 = read_shell(in, l), s2 = read_shell(in, l);
    int nd;
    in >> nd;
    std::vector<int> dist(nd);
    for (auto &d : dist) in >> d;
    std::vector<double> st((size_t)nd * 3 * nd * 3 * n);
    for (auto &v : st) in >> v;

    int L = std::max({lp, l1, l2});
    Engine e(Operator::coulomb, 3, L, 2);
    e.set(BraKet::xs_xx);
    e.compute(sp, s1, s2);
    auto idx2 = [](int i, int j) { return i * 9 - i * (i - 1) / 2 + (j - i); };
    auto pos = [&](int atom) {
      for (int i = 0; i < nd; ++i)
        if (dist[i] == atom) return i;
      return -1;
    };
    double worst = 0, scale = 0;
    for (int ia = 0; ia < nd; ++ia)
      for (int x = 0; x < 3; ++x)
        for (int ib = 0; ib < nd; ++ib)
          for (int y = 0; y < 3; ++y) {
            // fold centre-level analytic second derivatives onto the atoms
            std::vector<double> an(n, 0.0);
            for (int k = 0; k < 3; ++k)
              for (int m = 0; m < 3; ++m) {
                if (pos(at[k]) != ia || pos(at[m]) != ib) continue;
                int ci = 3 * k + x, cj = 3 * m + y;
                int idx = idx2(std::min(ci, cj), std::max(ci, cj));
                const double *r = e.results()[idx];
                if (!r) continue;
                for (int i = 0; i < n; ++i) an[i] += r[i];
              }
            const double *sv = &st[((((size_t)ia * 3 + x) * nd + ib) * 3 + y) * n];
            for (int i = 0; i < n; ++i) {
              worst = std::max(worst, std::fabs(an[i] - sv[i]));
              scale = std::max(scale, std::fabs(an[i]));
            }
          }
    std::printf("l=(%d|%d,%d) atoms=(%d,%d,%d) distinct=%d  max|analytic|=%.3e  max|an-stencil|=%.3e\n", lp,
                l1, l2, at[0], at[1], at[2], nd, scale, worst);
    worst_all = std::max(worst_all, worst);
    scale_all = std::max(scale_all, scale);
  }
  std::printf("ALL %d cases: max |an - stencil| = %.3e (max |an| = %.3e)\n", ncase, worst_all, scale_all);
  finalize();
}
