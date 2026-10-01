import numpy as np, ctypes as C, sys
import ecp_clean as E

D = C.POINTER(C.c_double)
I = C.POINTER(C.c_int)
K = C.CDLL("./libknob.so")
K.knob_value.restype = C.c_int
K.knob_value.argtypes = [
    C.c_int,
    C.c_int,
    D,
    D,
    D,
    C.c_int,
    C.c_int,
    D,
    D,
    D,
    D,
    C.c_int,
    I,
    I,
    D,
    D,
    C.c_double,
    C.c_int,
    C.c_int,
    D,
]


def arr(t, v):
    return (t * len(v))(*v)


ams, ns, ez, dz = [], [], [], []
for l, ts in E.ECP_I:
    for n, z, d in ts:
        ams.append(l)
        ns.append(n)
        ez.append(z)
        dz.append(d)


def val(sa, pa, sb, pb, pc, thresh, small, big):
    la, ea, ca = sa
    lb, eb, cb = sb
    out = (C.c_double * (E.nc(la) * E.nc(lb)))()
    st = K.knob_value(
        la,
        len(ea),
        arr(C.c_double, list(pa)),
        arr(C.c_double, ea),
        arr(C.c_double, E.bare(la, ea, ca)),
        lb,
        len(eb),
        arr(C.c_double, list(pb)),
        arr(C.c_double, eb),
        arr(C.c_double, E.bare(lb, eb, cb)),
        arr(C.c_double, list(pc)),
        len(ams),
        arr(C.c_int, ams),
        arr(C.c_int, ns),
        arr(C.c_double, ez),
        arr(C.c_double, dz),
        thresh,
        small,
        big,
        out,
    )
    assert st == 0
    return np.array(out[:]).reshape(E.nc(la), E.nc(lb)) * E.C2S[la] * E.C2S[lb]


sa = E.I_SH[3]
sb = E.I_SH[3]
pa = E.I0
pb = E.I0 + E.LV
pc = E.I0 + np.array([1.5, -1.2, 0.8])
e = np.eye(3)[0]
hs = np.arange(-20, 21) * 2.5e-5
vp = np.array([E.pyscf_value(sa, pa, sb, pb, pc + h * e)[0, 2] for h in hs])
for th, sm, bg in [
    (1e-15, 256, 1024),
    (1e-15, 1024, 4096),
    (1e-15, 2048, 8192),
    (1e-17, 256, 1024),
    (1e-12, 256, 1024),
    (1e-17, 2048, 8192),
]:
    v = np.array([val(sa, pa, sb, pb, pc + h * e, th, sm, bg)[0, 2] for h in hs])
    c = np.polyfit(hs, v, 3)
    r = v - np.polyval(c, hs)
    print(
        f"thresh {th:.0e} grids {sm}/{bg}: jitter (cubic resid) {abs(r).max():.2e}, |v-pyscf| max {abs(v - vp).max():.2e}, slope {c[2]:.9e} (pyscf slope {np.polyfit(hs, vp, 3)[2]:.9e})",
        flush=True,
    )
