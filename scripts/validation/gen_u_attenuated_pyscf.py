"""PySCF references for open-shell attenuated MP2, SCS-MP2 and RS-MP2 + LR-RPA.

Consumer: crates/ferric-rpa/tests/validation_u_attenuated_pyscf.rs.
Output:   testdata/reference/validation/u_attenuated_pyscf/<system>_<basis>.json

WHAT IS BUILT
-------------
On a stable UHF (PySCF `scf.UHF`, exact integrals, `common.run_open_shell`),
for each operator kind in {coulomb, erf, erfc} at the production
omega = 0.222254 Bohr^-1, with B_s = M (Q|ia)_s per spin s:

  e_aa, e_bb   same-spin U-MP2 blocks, 1/2 sum (ia|jb)[(ia|jb) - (ib|ja)] / D
  e_ab         opposite-spin block, sum (ia|JB)^2 / D
  d_aa, d_bb   direct same-spin sums, sum (ia|jb)^2 / D (no exchange)
  e_drpa       spin-summed dRPA, eps = I + Pi_a + Pi_b, prefactor 2 per spin,
               on the Gauss-Legendre grid nw = 20, x0 = 0.5 (ferric's default
               QuadratureConfig: the same map and weights)

and the fields ferric reports (crates/ferric-mp2/theory.md section 5.1,
crates/ferric-rpa/theory.md "Open-shell SR-MP2 + LR-RPA"):

  att_erfc   = E_MP2[erfc]           att_erf = E_MP2[erf]
  scs        = 6/5 e_ab[C] + 1/3 (e_aa + e_bb)[C]
  dmp2[op]   = e_ab + (d_aa + d_bb) / 2
  B          = E_MP2[C] + E_dRPA[erf] - dmp2[erf]
  T          = E_MP2[C] + (E_dRPA[C] - dmp2[C]) - (E_dRPA[erfc] - dmp2[erfc])

FOUR ROUTES PER KIND
--------------------
  pyscf        PySCF's own code: `mp.dfump2.DFUMP2` (e_corr_os = e_ab,
               e_corr_ss = e_aa + e_bb) and `gw.urpa.URPA`, both on a
               `df.DF` built inside `mol.with_range_coulomb(w)` (w > 0 erf,
               w < 0 erfc; PySCF's convention, checked by
               gen_rs_mp2_rpa_pyscf.sign_convention on a closed form).
  numpy_pyscf_tensor  numpy on PySCF's own _cderi (checks the numpy code
               against PySCF's: `code_residual`).
  numpy_pyscf_ints    numpy on PySCF's int3c2e/int2c2e with ferric's metric
               factorization (erf: eigh dropping eigenvalues < 1e-10;
               Coulomb and erfc: Cholesky).
  numpy_ferric_ints   the same numpy on ferric's own integrals
               (`ferric.AoIntegrals.eri3/eri2`, libint2): the reference the
               Rust test compares to first.

The Coulomb and erfc metrics are Cholesky-factored in both codes, so on the
fields that use only those (att_erfc, scs, T) PySCF's code value is a direct
reference. The erf metric is not: PySCF falls back to eigh with lindep 1e-7,
ferric cuts at 1e-10, so for att_erf and B only the numpy routes match
ferric's construction.

`fp_floor` is the largest change of each field when every PySCF integral is
perturbed elementwise by a relative +-64 eps (two seeds), with ferric's
factorization: the floating-point floor of the comparison.
`integral_shift` = field(numpy_ferric_ints) - field(numpy_pyscf_ints).

The UHF orbitals of both spins are written in ferric's AO order so the Rust
test runs ferric's correlation code on the same orbitals and energies.

Run (light; well under a minute per system):
    OPENBLAS_NUM_THREADS=1 box-lease exec -- \\
        uv run --no-sync python scripts/validation/gen_u_attenuated_pyscf.py [system ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_attenuated_rpa as gar  # noqa: E402
import gen_properties  # noqa: E402
import gen_rs_mp2_rpa_pyscf as grp  # noqa: E402
import gen_urpa  # noqa: E402

ROW = "u_attenuated_pyscf"
OMEGA = gar.OMEGA_DEFAULT_BOHR
NW = 20
X0 = 0.5
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9
FERRIC_LINDEP = 1e-10
C_OS, C_SS = 6.0 / 5.0, 1.0 / 3.0
CODE_RESIDUAL_MAX = 1e-10
INT_REL_ERR = 64 * 2.220446049250313e-16
FLOOR_SEEDS = (1, 2)
KINDS = ("coulomb", "erf", "erfc")
SYSTEMS = {  # system -> (xyz, multiplicity, basis, aux)
    "oh": ("oh.xyz", 2, "cc-pvdz", "cc-pvdz-ri"),
    "nh2": ("nh2.xyz", 2, "cc-pvdz", "cc-pvdz-ri"),
    "o2": ("o2.xyz", 3, "cc-pvdz", "cc-pvdz-ri"),
}


def spin_blocks(Bs, es, nos):
    """U-MP2 blocks and direct sums from per-spin B (nov, naux)."""
    import numpy as np

    out = {}
    eo = [e[:n] for e, n in zip(es, nos)]
    ev = [e[n:] for e, n in zip(es, nos)]
    B3 = [B.reshape(n, e.size - n, -1) for B, e, n in zip(Bs, es, nos)]
    for s, tag in ((0, "aa"), (1, "bb")):
        e_ss = 0.0
        d_ss = 0.0
        for i in range(nos[s]):
            for j in range(nos[s]):
                v = B3[s][i] @ B3[s][j].T
                d = eo[s][i] + eo[s][j] - ev[s][:, None] - ev[s][None, :]
                e_ss += float(np.sum(v * (v - v.T) / d))
                d_ss += float(np.sum(v * v / d))
        out[f"e_{tag}"] = 0.5 * e_ss
        out[f"d_{tag}"] = d_ss
    e_ab = 0.0
    for i in range(nos[0]):
        for j in range(nos[1]):
            v = B3[0][i] @ B3[1][j].T
            d = eo[0][i] + eo[1][j] - ev[0][:, None] - ev[1][None, :]
            e_ab += float(np.sum(v * v / d))
    out["e_ab"] = e_ab
    return out


def udrpa(Bs, ds):
    import numpy as np

    channels = [(B, d, np.ones_like(d)) for B, d in zip(Bs, ds)]
    return gen_urpa.numpy_urpa(channels, NW)[0]


def numpy_kind(Bs, es, nos):
    ds = [(e[None, n:] - e[:n, None]).ravel() for e, n in zip(es, nos)]
    rec = spin_blocks(Bs, es, nos)
    rec["e_drpa"] = udrpa(Bs, ds)
    rec["e_mp2"] = rec["e_aa"] + rec["e_bb"] + rec["e_ab"]
    rec["dmp2"] = rec["e_ab"] + 0.5 * (rec["d_aa"] + rec["d_bb"])
    return rec


def ferric_ints(xyz, mult, basis_name, aux_name, perm, aux_perm, kind):
    """ferric's own (mu nu|P), (P|Q) for `kind` in PySCF's AO order and
    (nao, nao, naux) layout (gen_rs_mp2_rpa_pyscf.ferric_ints at a given
    multiplicity; the integrals do not depend on it)."""
    import ferric
    import numpy as np

    ao = ferric.AoIntegrals(
        ferric.Molecule.from_xyz(str(xyz), 0, mult), ferric.BasisSet.bundled(basis_name)
    )
    aux = ferric.BasisSet.bundled(aux_name)
    op = {"coulomb": ("coulomb", None), "erf": ("erf", OMEGA), "erfc": ("erfc", OMEGA)}
    name, w = op[kind]
    v3 = np.asarray(ao.eri3(aux, operator=name, omega=w))  # (P, mu, nu), ferric order
    v2 = np.asarray(ao.eri2(aux, operator=name, omega=w))
    inv = np.argsort(np.asarray(perm))
    ainv = np.argsort(np.asarray(aux_perm))
    return v3.transpose(1, 2, 0)[np.ix_(inv, inv, ainv)], v2[np.ix_(ainv, ainv)]


def b_spin(v3, m, c, no):
    import numpy as np

    iap = np.einsum("mnP,mi,na->iaP", v3, c[:, :no], c[:, no:], optimize=True)
    return iap.reshape(-1, v3.shape[2]) @ m.T


def from_ints(v3, v2, kind, cs, es, nos):
    m, info = gar.metric_inv_sqrt(v2, kind, FERRIC_LINDEP)
    Bs = [b_spin(v3, m, c, n) for c, n in zip(cs, nos)]
    return numpy_kind(Bs, es, nos), info


def assemble(k):
    c, lr, sr = k["coulomb"], k["erf"], k["erfc"]
    return {
        "att_erfc": sr["e_mp2"],
        "att_erf": lr["e_mp2"],
        "att_erfc_e_os": sr["e_ab"],
        "att_erfc_e_ss": sr["e_aa"] + sr["e_bb"],
        "scs": C_OS * c["e_ab"] + C_SS * (c["e_aa"] + c["e_bb"]),
        "scs_e_os": c["e_ab"],
        "scs_e_ss": c["e_aa"] + c["e_bb"],
        "e_mp2_full": c["e_mp2"],
        "e_dmp2_lr": lr["dmp2"],
        "B_e_drpa_lr": lr["e_drpa"],
        "B_e_corr": c["e_mp2"] + lr["e_drpa"] - lr["dmp2"],
        "T_e_delta_drpa_full": c["e_drpa"] - c["dmp2"],
        "T_e_delta_drpa_sr": sr["e_drpa"] - sr["dmp2"],
        "T_e_corr": c["e_mp2"] + (c["e_drpa"] - c["dmp2"]) - (sr["e_drpa"] - sr["dmp2"]),
        # Controls: the closed-shell rule carried over, and a dropped beta ring.
        "B_control_2eab": c["e_mp2"] + lr["e_drpa"] - 2.0 * lr["e_ab"],
        "T_control_2eab": c["e_mp2"]
        + (c["e_drpa"] - 2.0 * c["e_ab"])
        - (sr["e_drpa"] - 2.0 * sr["e_ab"]),
    }


PYSCF_FIELDS = (
    "att_erfc",
    "att_erfc_e_os",
    "att_erfc_e_ss",
    "scs",
    "scs_e_os",
    "scs_e_ss",
    "e_mp2_full",
    "T_e_delta_drpa_full",
    "T_e_delta_drpa_sr",
    "T_e_corr",
)


def pyscf_code(mf, mol, aux, kind):
    """DFUMP2 + URPA on PySCF's own fit for `kind`; numpy on the same tensor."""
    import numpy as np
    from pyscf import df
    from pyscf.gw.urpa import URPA
    from pyscf.mp import dfump2

    w = gar.pyscf_omega(kind, OMEGA)
    with mol.with_range_coulomb(w):
        dfo = df.DF(mol, auxbasis=aux)
        dfo.verbose = 0
        dfo.build()
    assert mol.omega == 0.0 and dfo.auxmol.omega == w
    pt = dfump2.DFUMP2(mf)
    pt.with_df = dfo
    pt.verbose = 0
    pt.kernel()
    r = URPA(mf)
    r.with_df = dfo
    r.verbose = 0
    r.e_hf = float(mf.e_tot)
    r.kernel(nw=NW, x0=X0)
    code = {
        "e_ab": float(pt.e_corr_os),
        "e_ss": float(pt.e_corr_ss),
        "e_mp2": float(pt.e_corr),
        "e_drpa": float(r.e_corr),
    }
    v3p = grp.unpack_cderi(dfo, mol.nao_nr())
    cs = [np.asarray(c) for c in mf.mo_coeff]
    es = [np.asarray(e) for e in mf.mo_energy]
    nos = [int(np.count_nonzero(o > 0)) for o in mf.mo_occ]
    eye = np.eye(v3p.shape[2])
    num = numpy_kind([b_spin(v3p, eye, c, n) for c, n in zip(cs, nos)], es, nos)
    resid = {
        "e_ab": abs(num["e_ab"] - code["e_ab"]),
        "e_ss": abs(num["e_aa"] + num["e_bb"] - code["e_ss"]),
        "e_mp2": abs(num["e_mp2"] - code["e_mp2"]),
        "e_drpa": abs(num["e_drpa"] - code["e_drpa"]),
    }
    return code, num, resid, grp.decomposition_pyscf_used(dfo)


def gen(system):
    import ferric
    import numpy as np
    import pyscf
    from pyscf import gto

    xyz_rel, mult, basis_name, aux_name = SYSTEMS[system]
    t0 = time.perf_counter()
    xyz = common.MOL_DIR / xyz_rel
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name, multiplicity=mult)
    ll = common.check_basis_like_for_like(mol, basis_name, symbols)
    aux = gen_urpa.aux_dict(aux_name, symbols)
    auxmol = gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords),
        unit="Bohr",
        basis=aux,
        cart=False,
        spin=mult - 1,
        verbose=0,
    )
    naux = int(auxmol.nao_nr())
    assert naux == common.ferric_nao(aux_name, symbols)
    perm = gen_properties.ferric_ao_permutation(mol, basis_name, symbols)
    aux_perm = gen_properties.ferric_ao_permutation(auxmol, aux_name, symbols)
    sign = grp.sign_convention(OMEGA)

    uhf, mf = common.run_open_shell(
        mol, "uhf", conv_tol=CONV_TOL, conv_tol_grad=CONV_TOL_GRAD, return_mf=True
    )
    cs = [np.asarray(c) for c in mf.mo_coeff]
    es = [np.asarray(e) for e in mf.mo_energy]
    nos = [int(np.count_nonzero(o > 0)) for o in mf.mo_occ]

    kinds = {}
    routes = {"numpy_pyscf_ints": {}, "numpy_ferric_ints": {}, "numpy_pyscf_tensor": {}}
    pert = [dict() for _ in FLOOR_SEEDS]
    for kind in KINDS:
        code, num_t, resid, decomp = pyscf_code(mf, mol, aux, kind)
        if kind != "erf":
            worst = max(resid.values())
            if worst > CODE_RESIDUAL_MAX:
                raise RuntimeError(f"{system} {kind}: numpy vs PySCF code {worst:.2e}")
        v3, v2 = gar.ints(mol, auxmol, gar.pyscf_omega(kind, OMEGA))
        rec_p, info = from_ints(v3, v2, kind, cs, es, nos)
        v3f, v2f = ferric_ints(xyz, mult, basis_name, aux_name, perm, aux_perm, kind)
        rec_f, info_f = from_ints(v3f, v2f, kind, cs, es, nos)
        for i, seed in enumerate(FLOOR_SEEDS):
            rng = np.random.default_rng(seed)
            s3 = rng.choice((-1.0, 1.0), size=v3.shape)
            s3 = 0.5 * (s3 + s3.transpose(1, 0, 2))
            s2 = rng.choice((-1.0, 1.0), size=v2.shape)
            s2 = np.triu(s2) + np.triu(s2, 1).T
            pert[i][kind] = from_ints(
                v3 * (1.0 + INT_REL_ERR * s3),
                v2 * (1.0 + INT_REL_ERR * s2),
                kind,
                cs,
                es,
                nos,
            )[0]
        routes["numpy_pyscf_ints"][kind] = rec_p
        routes["numpy_ferric_ints"][kind] = rec_f
        routes["numpy_pyscf_tensor"][kind] = num_t
        kinds[kind] = {
            "pyscf_code": code,
            "pyscf_decomposition": decomp,
            "code_residual": resid,
            "metric_pyscf_ints": info,
            "metric_ferric_ints": info_f,
            "integral_difference": {
                "max_abs_3c": float(np.max(np.abs(v3f - v3))),
                "max_abs_2c": float(np.max(np.abs(v2f - v2))),
            },
        }

    fields = {r: assemble(v) for r, v in routes.items()}
    code_fields = assemble(
        {
            k: {
                "e_ab": kinds[k]["pyscf_code"]["e_ab"],
                "e_aa": kinds[k]["pyscf_code"]["e_ss"],
                "e_bb": 0.0,
                "e_mp2": kinds[k]["pyscf_code"]["e_mp2"],
                "e_drpa": kinds[k]["pyscf_code"]["e_drpa"],
                # dmp2 is not a PySCF output; numpy on PySCF's own tensor.
                "dmp2": routes["numpy_pyscf_tensor"][k]["dmp2"],
            }
            for k in KINDS
        }
    )
    fields["pyscf"] = {k: code_fields[k] for k in PYSCF_FIELDS}
    pf = [assemble(p) for p in pert]
    base = fields["numpy_pyscf_ints"]
    fp_floor = {k: max(abs(p[k] - base[k]) for p in pf) for k in base}
    integral_shift = {k: fields["numpy_ferric_ints"][k] - base[k] for k in base}

    def spin(s):
        c = cs[s][np.asarray(perm), :]
        return {
            "mo_energy": [float(x) for x in es[s]],
            "nocc": nos[s],
            "mo_coeff_ferric_order": [[float(x) for x in row] for row in c],
        }

    payload = {
        "system": system,
        "basis": basis_name,
        "aux": aux_name,
        "charge": 0,
        "multiplicity": mult,
        "nao": int(mol.nao_nr()),
        "naux": naux,
        "nelectron": int(mol.nelectron),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "like_for_like": ll,
        "omega_bohr_inv": OMEGA,
        "nw": NW,
        "x0": X0,
        "frozen_core": 0,
        "c_os": C_OS,
        "c_ss": C_SS,
        "uhf": uhf,
        "uhf_orbitals": {"alpha": spin(0), "beta": spin(1)},
        "sign_convention": sign,
        "kernels": kinds,
        "fields": fields,
        "fp_floor": fp_floor,
        "integral_shift": integral_shift,
        "generator_seconds": time.perf_counter() - t0,
    }
    payload["provenance"] = common.provenance(
        code="PySCF (UHF; df.DF under Mole.with_range_coulomb; mp.dfump2.DFUMP2; "
        "gw.urpa.URPA) + numpy (U-MP2 blocks and U-dRPA on PySCF's and ferric's integrals)",
        version=pyscf.__version__,
        keywords={
            "uhf": f"scf.UHF, exact integrals, conv_tol {CONV_TOL}, stability-followed",
            "operator": "w=0 Coulomb, w>0 erf, w<0 erfc",
            "ferric_metric": "erf: eigh dropping eigenvalues < 1e-10; Coulomb/erfc: Cholesky",
            "rpa": f"nw {NW}, x0 {X0}, prefactor 2 per spin",
            "ferric_integrals": f"ferric.AoIntegrals eri3/eri2 from build {ferric.__build__}",
        },
        basis_name=basis_name,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid=None,
        aux=aux_name,
        frozen_core=0,
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        stability={"uhf": uhf["stability"]},
        generator="scripts/validation/gen_u_attenuated_pyscf.py",
        extra={},
    )
    path = common.write_reference(ROW, system, basis_name, payload)
    f = fields["numpy_ferric_ints"]
    print(
        f"{system}: E_UHF {uhf['energy']:.10f} att_erfc {f['att_erfc']:.12f} "
        f"scs {f['scs']:.12f} B {f['B_e_corr']:.12f} T {f['T_e_corr']:.12f}; "
        f"pyscf-numpy T {fields['pyscf']['T_e_corr'] - base['T_e_corr']:+.2e}; "
        f"int shift B {integral_shift['B_e_corr']:+.2e} T {integral_shift['T_e_corr']:+.2e}; "
        f"fp floor B {fp_floor['B_e_corr']:.1e} T {fp_floor['T_e_corr']:.1e}; "
        f"2eab miss B {abs(f['B_control_2eab'] - f['B_e_corr']):.1e} "
        f"({payload['generator_seconds']:.1f} s) -> {path.relative_to(common.ROOT)}"
    )


def main(argv):
    want = list(argv) or list(SYSTEMS)
    unknown = [k for k in want if k not in SYSTEMS]
    if unknown:
        raise SystemExit(f"unknown systems: {unknown} (known: {sorted(SYSTEMS)})")
    for k in want:
        gen(k)
    print(f"GEN_U_ATTENUATED_PYSCF_DONE written={len(want)}")


if __name__ == "__main__":
    main(sys.argv[1:])
