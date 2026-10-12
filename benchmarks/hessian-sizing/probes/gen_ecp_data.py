"""Emit ecp_data.h (I/H def2-SVP shells + the I def2 ECP) for libecpint_second_derivs.cc.

Usage: python gen_ecp_data.py > ecp_data.h
PySCF's local ECP channel (l = -1) is relabelled l = 3 (= max semi-local l + 1),
the libecpint convention; every term carries r-power index n = 2 (r^0).
"""

from pyscf import gto

mol = gto.M(
    atom=[("I", (0, 0, 0)), ("H", (0, 0, 1.6))],
    basis="def2-svp",
    ecp={"I": "def2-svp"},
    unit="Bohr",
    verbose=0,
    cart=True,
)


def fmt(xs):
    return ",".join(repr(float(x)) for x in xs)


out = [
    "struct Sh{int atom;int l;std::vector<double> e,c;};",
    "static std::vector<Sh> SHELLS={",
]
for ib in range(mol.nbas):
    e = mol.bas_exp(ib)
    c = mol.bas_ctr_coeff(ib)[:, 0]
    out.append(
        f"  {{{mol.bas_atom(ib)},{mol.bas_angular(ib)},{{{fmt(e)}}},{{{fmt(c)}}}}},"
    )
out.append("};")

ncore, chans = gto.basis.load_ecp("def2-svp", "I")
out.append(f"// ncore={ncore}")
out.append("struct EcpT{int l;int n;double e;double c;};")
out.append("static std::vector<EcpT> ECPT={")
for l, terms in chans:
    for pw, lst in enumerate(terms):
        for e, c in lst:
            out.append(f"  {{{3 if l == -1 else l},{pw},{e!r},{c!r}}},")
out.append("};")
print("\n".join(out))
