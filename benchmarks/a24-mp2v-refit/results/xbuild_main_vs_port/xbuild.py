"""Print every MP2-V / VV10 energy reachable from main's API as float.hex, for a main-vs-port bit diff."""

import sys

import ferric

print("BUILD", ferric.__build__, file=sys.stderr)
W = "3\nw\nO 0 0 0.118\nH 0 0.755 -0.471\nH 0 -0.755 -0.471\n"
mol = ferric.Molecule.from_xyz_string(W)
obs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")
out = []


def rec(tag, x):
    out.append(f"{tag} {float(x).hex()}")


# 1. production lockstep MP2-V, published params, and decoupled-omega lockstep
for kw in [{}, {"omega": 2.0}, {"r0": 1.35, "b": 6.0}, {"vv10_damping": "none"}]:
    r = ferric.run_mp2_v(mol, obs, aux, frozen_core=1, **kw)
    for f in ("total_energy", "rhf_energy", "att_mp2_corr", "vv10_e_nl"):
        rec(f"mp2v{kw}.{f}", getattr(r, f))
# 2. main's scan (compute_vv10_weighted_energy_scan is now routed via the tuple scan)
s = ferric.run_mp2_v_scan(
    mol,
    obs,
    aux,
    [(1.0, None), (1.2, 2.0)],
    [6.0, 11.0, 14.5],
    frozen_core=1,
    nlc_grid=(20, 26),
)
for a, arm in enumerate(s["arms"]):
    rec(f"scan{a}.att", arm["att_mp2_corr"])
    for k, e in enumerate(arm["vv10_e_nl"]):
        rec(f"scan{a}.vv10[{k}]", e)
# 3. research VV10 variants
specs = [
    ("none", 1.0, None, 1),
    ("eq11", 1.0, None, 1),
    ("eq11", 1.0, 4.0, 1),
    ("terf", 1.0, None, 2),
]
for i, e in enumerate(ferric.run_vv10_variants(mol, obs, specs, nlc_grid=(20, 26))):
    rec(f"variant{i}", e)
# 4. wB97X-V KS-DFT (production VV10 in the SCF)
d = ferric.run_dft(mol, obs, functional="wB97X-V")
rec("wb97xv.total", d.total_energy)
print("\n".join(out))
