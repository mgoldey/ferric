# PRE-REGISTERED go/no-go threshold -- 3c1e batched kernel (Phase 1)
# Written BEFORE any measurement. Timestamp below is the write time.

DECIDING NUMBER: f_a = (per-point setup cost) / (per-point total cost)
  (a) = engine parameter setup per grid point: scf_engine_set_point_charges ->
        libint2 Engine::set_params -> the pset/compute_primdata path.
  (b) = integral arithmetic: Boys evaluation + AM recursion + contraction,
        i.e. the shell-pair sweep of compute_1e_block after params are set.

A batched kernel amortizes (a) across a point batch. It CANNOT reduce (b).
So the ceiling on batching speedup is 1/(1-f_a).

PRE-REGISTERED DECISION RULE:
  f_a >= 0.85  -> GO. Ceiling >= 6.7x. Build the kernel (Phase 2).
  0.50 <= f_a < 0.85 -> MARGINAL. Ceiling 2x-6.7x. Report; build ONLY if
        f_a >= 0.70 at the largest basis (TZ), else NO-GO.
  f_a < 0.50  -> NO-GO. Ceiling < 2x. STOP. Do not write recursions.

Even a GO does NOT flip the COSX kill gate on its own: deficit is ~150x,
batching ceiling ~1/(1-f_a), screening ~2x already counted. State that plainly.

ARTIFACT HYPOTHESIS (stated before measuring):
  If (a) genuinely dominates, I expect: setup-only microbench time per call to
  be a large fraction of full-compute time per point, AND that fraction to RISE
  with primitive count (TZ > DZ) because compute_primdata is O(n_primitive_pairs)
  while (b) is O(n_prim_pairs * angular work) -- so (b) also grows. If instead my
  microbench is BROKEN (e.g. set_point_charges is lazy and defers primdata to the
  first compute call), I expect setup-only time to be ~0 and independent of basis
  size. Those predictions DIFFER, so the experiment can distinguish them.
  A near-zero, basis-independent setup time means the split measurement is
  INVALID, not that f_a=0 -- I must then measure by a different route
  (perf symbol attribution, or N-points-per-setup scaling).

SECONDARY (more robust) MEASUREMENT -- the scaling route:
  Since libint2 accumulates charges, I cannot get per-charge output, but I CAN
  time: T(k) = one set_point_charges with k charges + one full shell-pair sweep.
  The sweep cost (b) is independent of k for k>=1 ONLY if libint2's inner loop
  does not iterate charges -- it DOES (pset loop), so T(k) = a + k*b_prime.
  Therefore: linear fit of T(k) vs k gives intercept a (setup) and slope b_prime
  (per-charge arithmetic). f_a = a/(a+b_prime) is the k=1 split. This uses the
  SAME code path the real COSX build uses and needs no instrumentation of C++.
  This is the PRIMARY estimator; the microbench is the cross-check.
2026-09-07T13:12:59-04:00

================================================================================
RESULTS (release build, PSI full avg10=0.00 before AND after every run)
================================================================================

PRIMARY estimator -- linear fit T(k) = a + k*b, R^2 >= 0.9997 on all four cases:

  system / basis      nbf   nsh    a(setup)     b(arith/pt)   f_a     ceiling
  water  / cc-pVDZ     24    12   1.730e-5 s    7.315e-5 s   0.1912   1.24x
  water  / cc-pVTZ     58    22   2.118e-5 s    2.597e-4 s   0.0754   1.08x
  butane / cc-pVDZ    106    54   1.373e-4 s    1.374e-3 s   0.0909   1.10x
  butane / cc-pVTZ    260   100   1.698e-4 s    5.353e-3 s   0.0307   1.03x

CROSS-CHECK -- direct measurement, no model (K=32 points):
  probe1  = set_params alone (no sweep)
  probe2a = the real COSX pattern: K x (set_params + full sweep)
  probe2b = perfect-batch floor: 1 x set_params(K charges) + 1 sweep

  system / basis     probe1       f_a      MEASURED ceiling (2a/2b)
  water  / cc-pVDZ   1.84e-7 s   0.0019          1.36x
  water  / cc-pVTZ   1.26e-7 s   0.0004          1.03x
  butane / cc-pVDZ   1.27e-7 s   0.0001          1.06x
  butane / cc-pVTZ   1.23e-7 s   0.0000          1.09x

VERDICT: NO-GO. f_a is far below the 0.50 NO-GO line by BOTH estimators, and
falls with basis size (opposite of the GO artifact hypothesis). Do not write
recursions.

WHY THE TASK PREMISE WAS WRONG (verified in the headers, not inferred):
The brief assumed set_params -> compute_primdata, so that setup dominates.
It does not. ~/.local/include/libint2/engine.h:650 shows set_params is only
  params_ = params; init_core_ints_params(params_); reset_scratch();
-- a cheap assignment. The expensive compute_primdata call is at
engine.impl.h:268, INSIDE compute(), nested in the `for (pset ...)` loop
(engine.impl.h:261) which iterates over the charges. So per-charge primitive
work is already inside the shell-pair sweep and is charged to (b), not (a).
Probe1 measures set_params at ~1.2e-7 s -- 4-5 ORDERS OF MAGNITUDE below a
point's cost -- which is what a cheap assignment looks like, and is consistent
with the header rather than with the brief.

This means the accumulation behaviour that makes libint2 unusable for COSX
(no per-charge output) is NOT the same thing as a per-point setup penalty.
Batching the grid axis has almost nothing to amortize: probe2b, which is the
ABSOLUTE FLOOR for any batched kernel on this engine (it does the strictly
smaller job of computing only the SUM over the 32 points), still costs 0.91x
of the fully serial path at butane/cc-pVTZ.

The remaining 1.03-1.36x is scratch-buffer zeroing and shell-loop overhead,
not primitive setup -- and it SHRINKS as the basis grows, i.e. it vanishes
exactly where COSX would need it most.

WHAT THIS DOES NOT SAY: it does not say a purpose-built 3c1e kernel (ORCA's)
is worthless -- such a kernel wins by replacing libint2's ARITHMETIC (b), via
a rolled-out Rys/HGP scheme with the grid axis as the inner vectorized
dimension. That is a different and far larger project than "amortize setup",
and Phase 1 as scoped (amortize (a)) cannot deliver it. See the report.
