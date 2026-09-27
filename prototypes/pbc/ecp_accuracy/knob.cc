#include <libecpint.hpp>
#include <array>
#include <vector>
using namespace libecpint;
// value <a|U|b> for single shells with engine knobs; out ncart_a*ncart_b
extern "C" int knob_value(int la, int na, const double* pa, const double* ea, const double* ca,
                          int lb, int nb, const double* pb, const double* eb, const double* cb,
                          const double* pc, int nt, const int* ams, const int* ns, const double* ez, const double* dz,
                          double thresh, int small, int big, double* out) {
  try {
    std::array<double,3> A{pa[0],pa[1],pa[2]}, B{pb[0],pb[1],pb[2]};
    GaussianShell sa(A, la), sb(B, lb);
    for (int i=0;i<na;i++) sa.addPrim(ea[i], ca[i]);
    for (int i=0;i<nb;i++) sb.addPrim(eb[i], cb[i]);
    ECP U(pc);
    for (int t=0;t<nt;t++) U.addPrimitive(ns[t], ams[t], ez[t], dz[t]);
    U.sort();
    ECPIntegral eng(std::max(la,lb), U.getL(), 0, thresh, small, big);
    TwoIndex<double> v;
    eng.compute_shell_pair(U, sa, sb, v);
    for (int i=0;i<v.dims[0];i++) for (int j=0;j<v.dims[1];j++) out[i*v.dims[1]+j]=v(i,j);
    return 0;
  } catch (...) { return -1; }
}
