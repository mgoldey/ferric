// Timing of the hot loop of the proposed Rust type-2 kernel (a C++ stand-in; NOT production code):
// per window (primitive pair x ECP term): GL nodes on [max(0, r0 - T/sqrt p), rpk + T/sqrt p], two exp-scaled Bessel
// sets (series + downward / upward recurrence, as ecpq.bessel_ktil), the Gaussian envelope, and the accumulation of
// R[K][lam][lam'] += w g r^K ktil_lam(zA) ktil_lam'(zB).
// Input: windows.txt from cost.py (la lb l alpha A beta B n zeta coef per line).
//   g++ -O2 -march=native -o radial_kernel radial_kernel.cc && ./radial_kernel windows.txt [N]   (N: fixed GL-N; omit = production rule)
// Prints wall time per window set and a checksum (sum of all R entries) to compare with the Python reference.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

static void gauleg(int n, std::vector<double>& x, std::vector<double>& w) {
  x.resize(n); w.resize(n);
  for (int i = 0; i < n; ++i) {
    double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), pp = 0;
    for (int it = 0; it < 100; ++it) {
      double p1 = 1, p2 = 0;
      for (int j = 0; j < n; ++j) { double p3 = p2; p2 = p1; p1 = ((2 * j + 1) * z * p2 - j * p3) / (j + 1); }
      pp = n * (z * p1 - p2) / (z * z - 1);
      double z1 = z; z = z1 - p1 / pp;
      if (std::fabs(z - z1) < 1e-16) break;
    }
    x[i] = -z; w[i] = 2 / ((1 - z * z) * pp * pp);
  }
}

// Gauss-Hermite nodes/weights (Numerical Recipes gauher), weights returned multiplied by exp(x^2).
static void gauher(int n, std::vector<double>& x, std::vector<double>& w) {
  x.assign(n, 0); w.assign(n, 0);
  const double PIM4 = 0.7511255444649425;
  double z = 0, pp = 0;
  int m = (n + 1) / 2;
  for (int i = 0; i < m; ++i) {
    if (i == 0) z = std::sqrt(2.0 * n + 1) - 1.85575 * std::pow(2.0 * n + 1, -0.16667);
    else if (i == 1) z -= 1.14 * std::pow((double)n, 0.426) / z;
    else if (i == 2) z = 1.86 * z - 0.86 * x[0];
    else if (i == 3) z = 1.91 * z - 0.91 * x[1];
    else z = 2.0 * z - x[i - 2];
    for (int it = 0; it < 100; ++it) {
      double p1 = PIM4, p2 = 0;
      for (int j = 0; j < n; ++j) { double p3 = p2; p2 = p1; p1 = z * std::sqrt(2.0 / (j + 1)) * p2 - std::sqrt((double)j / (j + 1)) * p3; }
      pp = std::sqrt(2.0 * n) * p2;
      double z1 = z; z = z1 - p1 / pp;
      if (std::fabs(z - z1) < 1e-15) break;
    }
    x[i] = z; x[n - 1 - i] = -z;
    w[i] = 2.0 / (pp * pp) * std::exp(z * z); w[n - 1 - i] = w[i];
  }
}

// Reciprocal tables (built once): RECIP[n][k] = 1/(k (2n+2k+1)), IDF[n] = 1/(2n+1)!!  -> no divisions in the series.
static double RECIP[16][64], IDF[16];
static void init_tables() {
  for (int n = 0; n < 16; ++n) {
    double d = 1; for (int j = 1; j <= 2 * n + 1; j += 2) d *= j; IDF[n] = 1 / d;
    for (int k = 1; k < 64; ++k) RECIP[n][k] = 1.0 / (k * (2.0 * n + 2 * k + 1));
  }
}

// exp(-z) i_n(z) / exp(-z) for the top orders (caller multiplies by exp(-z) once).
static inline double series_noexp(int n, double z, double zn) {
  double h = 0.5 * z * z, t = 1, s = 1;
  for (int k = 1; k < 64; ++k) { t *= h * RECIP[n][k]; s += t; if (t <= 1e-17 * s) break; }
  return zn * IDF[n] * s;
}

// Same regimes as ecpq.bessel_ktil.
static inline void ktil(int nmax, double z, double* out) {
  if (z == 0.0) { out[0] = 1; for (int n = 1; n <= nmax; ++n) out[n] = 0; return; }
  double zs = nmax > 4 ? 4.0 * nmax : 16.0;
  if (z < zs) {
    double ez = std::exp(-z);
    if (z < 1e-8) {
      double zn = ez;
      for (int n = 0; n <= nmax; ++n) { out[n] = zn * IDF[n] * (1 + z * z / (2 * (2 * n + 3))); zn *= z; }
      return;
    }
    double zn = ez; for (int n = 0; n < nmax; ++n) zn *= z;          // e^{-z} z^nmax
    double kn = series_noexp(nmax, z, zn), kp1 = series_noexp(nmax + 1, z, zn * z);
    double iz = 1 / z;
    out[nmax] = kn;
    for (int n = nmax; n > 0; --n) { double km1 = kp1 + (2 * n + 1) * iz * kn; out[n - 1] = km1; kp1 = kn; kn = km1; }
    return;
  }
  double e2 = std::exp(-2 * z), iz = 1 / z;
  out[0] = 0.5 * (1 - e2) * iz;
  if (nmax >= 1) out[1] = 0.5 * (1 + e2) * iz - 0.5 * (1 - e2) * iz * iz;
  for (int n = 1; n < nmax; ++n) out[n + 1] = out[n - 1] - (2 * n + 1) * iz * out[n];
}

struct Win { int la, lb, l; double al, A, be, B; int n; double zeta, c; };

int main(int argc, char** argv) {
  const char* fn = argc > 1 ? argv[1] : "windows.txt";
  int fixedN = argc > 2 ? atoi(argv[2]) : 0;   // 0 = production rule; N > 0 = fixed GL-N on T = 6 windows
  double T = 6.0;
  init_tables();
  std::vector<Win> ws;
  FILE* f = fopen(fn, "r");
  Win w;
  while (fscanf(f, "%d %d %d %lf %lf %lf %lf %d %lf %lf", &w.la, &w.lb, &w.l, &w.al, &w.A, &w.be, &w.B, &w.n, &w.zeta,
                &w.c) == 10) ws.push_back(w);
  fclose(f);
  std::vector<std::vector<double>> glx(41), glw(41);
  for (int n = 1; n <= 40; ++n) gauleg(n, glx[n], glw[n]);
  std::vector<double> hx, hw; gauher(20, hx, hw);
  const double FP2 = 16 * M_PI * M_PI;
  double checksum = 0; long nodes = 0;
  double R[16 * 16 * 16];
  auto t0 = std::chrono::steady_clock::now();
  for (int rep = 0; rep < 3; ++rep) {
    checksum = 0; nodes = 0;
    for (const Win& v : ws) {
      int na = v.l + v.la + 1, nb = v.l + v.lb + 1, nK = v.la + v.lb + 1;
      double p = v.al + v.be + v.zeta, r0 = (v.al * v.A + v.be * v.B) / p;
      double K = v.al * v.A * v.A + v.be * v.B * v.B - p * r0 * r0;
      int deg = v.n + v.la + v.lb;
      double sp = std::sqrt(p), rpk = 0.5 * (r0 + std::sqrt(r0 * r0 + 2.0 * deg / p));
      // production rule (ecpq.radial_nodes): GH-20 interior, else adaptive GL on [0 or r0 - 6/sqrt p, rpk + 6/sqrt p]
      const double *X, *Wt; double a0, a1; int nn;
      if (fixedN == 0 && r0 * sp >= 6.5) { X = hx.data(); Wt = hw.data(); nn = 20; a0 = r0; a1 = 1 / sp; }
      else {
        double lo = std::max(0.0, r0 - T / sp), hi = std::max(r0, rpk) + T / sp;
        nn = fixedN ? fixedN : std::min(40, std::max(16, (int)std::ceil(3.4 * (hi - lo) * sp)));
        X = glx[nn].data(); Wt = glw[nn].data(); a0 = 0.5 * (hi + lo); a1 = 0.5 * (hi - lo);
      }
      for (int i = 0; i < nK * na * nb; ++i) R[i] = 0;
      double ka[16], kb[16];
      for (int i = 0; i < nn; ++i) {
        double r = a0 + a1 * X[i], wt = a1 * Wt[i];
        ktil(na - 1, 2 * v.al * v.A * r, ka);
        ktil(nb - 1, 2 * v.be * v.B * r, kb);
        double rn = v.n == 0 ? 1.0 : (v.n == 1 ? r : r * r);
        double g = v.c * FP2 * wt * rn * std::exp(-p * (r - r0) * (r - r0) - K);
        double rk = g;
        for (int k = 0; k < nK; ++k) {
          for (int a = 0; a < na; ++a) {
            double t = rk * ka[a];
            double* Rrow = R + (k * na + a) * nb;
            for (int b = 0; b < nb; ++b) Rrow[b] += t * kb[b];
          }
          rk *= r;
        }
        ++nodes;
      }
      for (int i = 0; i < nK * na * nb; ++i) checksum += R[i];
    }
  }
  double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 3;
  printf("rule %s: %zu windows, %ld nodes: %.4f s per pass (%.1f ns/node), checksum %.15e\n", fixedN ? "fixed GL" : "production", ws.size(), nodes, dt,
         1e9 * dt / nodes, checksum);
  (void)fixedN;
  return 0;
}
