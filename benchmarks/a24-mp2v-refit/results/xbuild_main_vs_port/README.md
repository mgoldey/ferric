# Default-path bit-identity: origin/main vs the MP2-V refit port (2026-10-24)

`xbuild.py` prints 29 energies (`float.hex`) reachable through main's Python API:
`run_mp2_v` (published lockstep, omega = 2, r0 = 1.35/b = 6, undamped), `run_mp2_v_scan`
(2 arms x 3 b; its VV10 half now runs through the new tuple scan), `run_vv10_variants`
(4 weights), and wB97X-V `run_dft`. Water, cc-pVDZ / cc-pVDZ-ri, frozen core 1.

| file | build (`ferric.__build__`) |
|---|---|
| `energies_main.txt` | 6e7e7e44a (origin/main), clean |
| `energies_port.txt` | 6e7e7e44a + the port's working tree, dirty (Rust library code identical to the branch's; later commits changed only tests, docs and benchmark files) |

`diff energies_main.txt energies_port.txt` is empty: all 29 values are bit-identical.
Each run loaded its own `.so` through a PYTHONPATH shim (a `ferric/` package directory
holding `__init__.py` and a symlinked extension); the shared venv symlink was not touched.
