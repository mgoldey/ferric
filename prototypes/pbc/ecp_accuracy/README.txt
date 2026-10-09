FINDINGS "ECP derivative clean-band discrepancy — 2026-09-27" scripts.
Build the two ctypes libraries from a built ferric-integrals (no cargo needed if target/ is warm):
  B=<ferric-pbc>/target/release/build/ferric-integrals-*/out
  g++ -shared -o libecpshim.so -Wl,--whole-archive $B/libferric_ecp_shim.a -Wl,--no-whole-archive $B/libecpint-lib/libecpint.a $B/libecpint-lib/libFaddeeva.a
  g++ -O2 -fPIC -shared -std=c++17 -o libknob.so knob.cc -I<ferric-pbc>/crates/ferric-integrals/shim/libecpint/include -I$B/libecpint-build/include -I$B/libecpint-build/include/libecpint $B/libecpint-lib/libecpint.a $B/libecpint-lib/libFaddeeva.a
Run (PySCF venv, OPENBLAS_NUM_THREADS=1): ecp_clean.py (bands + worst 10 vs PySCF), quadd.py (quadrature oracle),
scan.py (value smoothness), knobscan.py (engine thresh/grids), split.py (type 1 / type 2 by channel),
stripped.py / worst_stripped.py / ch0.py (d projector removed; s/p residuals vs quadrature), conv.py (quadrature convergence).
quad.py = independent radial Gauss-Legendre x Lebedev quadrature, semi-local projectors via the Legendre addition theorem.
