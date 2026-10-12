#include <libecpint/api.hpp>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include "ecp_data.h"
using namespace libecpint;

// atom 0 = I (with ECP), atom 1 = H
static double GEOM[2][3];

struct Result { std::vector<std::vector<double>> d1, d2; int ncart; };

static Result run(int deriv) {
  std::vector<double> gc, ge, gcoef; std::vector<int> gam, glen;
  for (auto &s : SHELLS) {
    for (int k = 0; k < 3; k++) gc.push_back(GEOM[s.atom][k]);
    gam.push_back(s.l); glen.push_back((int)s.e.size());
    for (size_t p = 0; p < s.e.size(); p++) { ge.push_back(s.e[p]); gcoef.push_back(s.c[p]); }
  }
  std::vector<double> uc, ue, ucoef; std::vector<int> uam, un, ulen;
  for (int k = 0; k < 3; k++) uc.push_back(GEOM[0][k]);
  ulen.push_back((int)ECPT.size());
  for (auto &t : ECPT) { uam.push_back(t.l); un.push_back(t.n); ue.push_back(t.e); ucoef.push_back(t.c); }
  ECPIntegrator ig;
  ig.set_gaussian_basis((int)SHELLS.size(), gc.data(), ge.data(), gcoef.data(), gam.data(), glen.data());
  ig.set_ecp_basis(1, uc.data(), ue.data(), ucoef.data(), uam.data(), un.data(), ulen.data());
  ig.init(deriv);
  ig.compute_first_derivs();
  if (deriv >= 2) ig.compute_second_derivs();
  Result r;
  for (auto &p : ig.get_first_derivs()) r.d1.push_back(*p);
  if (deriv >= 2) for (auto &p : ig.get_second_derivs()) r.d2.push_back(*p);
  r.ncart = (int)std::sqrt((double)r.d1[0].size() + 0.5);
  return r;
}

int main(int argc, char **argv) {
  double h = argc > 1 ? atof(argv[1]) : 2e-3;
  double base[2][3] = {{0.0, 0.0, 0.0}, {0.3, -0.4, 3.1}};
  for (int a = 0; a < 2; a++) for (int k = 0; k < 3; k++) GEOM[a][k] = base[a][k];
  Result an = run(2);
  // packing index for pair of coordinates (ia,ax),(ja,ay), 2 atoms
  auto pk = [&](int ia, int ax, int ja, int ay) {
    int q1 = 3 * ia + ax, q2 = 3 * ja + ay;
    if (q1 > q2) std::swap(q1, q2), std::swap(ia, ja), std::swap(ax, ay);
    if (ia == 0 && ja == 0) { static const int m[3][3] = {{0,1,2},{1,3,4},{2,4,5}}; return m[ax][ay]; }
    if (ia == 0 && ja == 1) return 6 + 3 * ax + ay;
    static const int m[3][3] = {{0,1,2},{1,3,4},{2,4,5}}; return 15 + m[ax][ay];
  };
  double w[4] = {1, -8, 8, -1}, off[4] = {-2, -1, 1, 2};
  double worstAll = 0, scale = 0;
  for (int ia = 0; ia < 2; ia++) for (int ax = 0; ax < 3; ax++)
  for (int ja = ia; ja < 2; ja++) for (int ay = 0; ay < 3; ay++) {
    if (ja == ia && ay < ax) continue;
    // FD: d/d(ja,ay) of first_derivs[3*ia+ax]
    std::vector<double> fd;
    for (int p = 0; p < 4; p++) {
      GEOM[ja][ay] = base[ja][ay] + off[p] * h;
      Result r = run(1);
      auto &blk = r.d1[3 * ia + ax];
      if (fd.empty()) fd.assign(blk.size(), 0.0);
      for (size_t i = 0; i < blk.size(); i++) fd[i] += w[p] * blk[i] / (12 * h);
    }
    GEOM[ja][ay] = base[ja][ay];
    auto &a2 = an.d2[pk(ia, ax, ja, ay)];
    double md = 0, mx = 0;
    for (size_t i = 0; i < fd.size(); i++) { md = std::max(md, std::fabs(fd[i] - a2[i])); mx = std::max(mx, std::fabs(a2[i])); }
    printf("atoms(%d,%d) axes(%d,%d) pack=%2d  max|an|=%.3e max|an-fd|=%.3e rel=%.2e\n", ia, ja, ax, ay, pk(ia, ax, ja, ay), mx, md, md / (mx > 0 ? mx : 1));
    worstAll = std::max(worstAll, md); scale = std::max(scale, mx);
  }
  printf("ncart=%d worst abs %.3e (scale %.3e)\n", an.ncart, worstAll, scale);
}
