#!/usr/bin/env python3
"""In-memory MP2-V parameter session (research helper; EXPERIMENTAL).

Holds many fragment sessions (`ferric.Mp2vSession`: SCF run once, orbitals,
density and NLC grid state kept alive) so parameters can be iterated without
redoing the SCF, and drives the pre-registered nested search (see
PREREGISTRATION.md, Addendum F):

  expensive : (r0, s = r0*omega) MP2 seam  -> one attenuated MP2 per fragment
  cheap     : b, C, damping form, (r0_vv, s_vv) -> stored grid state only

Parameter vector (a plain dict, JSON-able; lengths in Angstrom):

  {"r0": 1.0, "s": None,        # MP2 seam; s=None is the Dutoi-linked width
   "b": 11.0, "c": 0.0089,
   "vv": None}                  # None = published LOCKSTEP (Eq. 11 shares the seam)
                                # or {"form": "eq11"|"terf"|"bare", "r0": .., "s": ..}

Interactive:   python -i session.py  ->  pool = Pool("adz"); pool.add("a24", [2, 5])
               pool.eint(P(r0=1.0, s=None, b=11.0), "a24", 2, cp=True)
Long-running:  python session.py serve --dir jobs/ --basis adz  (polls jobs/*.json,
               writes *.out.json; sessions persist across jobs; see `serve`)
One-shot:      python session.py search --set a24 --systems 2,5 --basis adz --cp

Every expensive/cheap result is cached to JSON keyed by (fragment, kind, params),
so a restarted process re-uses prior numbers and only builds a fragment session
(one SCF) on its first cache MISS.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("FERRIC_TERF_TABLE_DIR", str(HERE.parents[1] / "terf-tables"))

from harness import BASES, K, fragments, load_sets, n_frozen_core, xyz_text  # noqa: E402

LINKED_S = 1.0 / math.sqrt(2.0)  # r0*omega of the Dutoi-linked width
FRAG_CP = ("dimer", "mA_cp", "mB_cp")
FRAG_NCP = ("dimer", "mA", "mB")


def P(r0=1.0, s=None, b=11.0, c=0.0089, vv=None):
    return {"r0": r0, "s": s, "b": b, "c": c, "vv": vv}


def rss_bytes():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def _omega(r0, s):
    return None if s is None else s / r0


def _tuple(p):
    """(b, C, form, r0_vv, omega_vv) for the binding; lockstep when p['vv'] is None."""
    vv = p["vv"]
    if vv is None:
        return (p["b"], p["c"], "eq11", p["r0"], _omega(p["r0"], p["s"]))
    return (
        p["b"],
        p["c"],
        vv["form"],
        vv.get("r0", 1.0),
        _omega(vv.get("r0", 1.0), vv.get("s")),
    )


def _seam_key(r0, s):
    return f"{r0:.8g}|{'L' if s is None else format(s, '.8g')}"


def _nl_key(t):
    b, c, form, r0v, wv = t
    return f"{b:.10g}|{c:.10g}|{form}|{r0v:.8g}|{'L' if wv is None else format(wv, '.10g')}"


class Cache:
    """JSON cache: {context: {...}, entries: {key: value}}; context mismatch refuses."""

    def __init__(self, path, context):
        self.path, self.context = Path(path) if path else None, context
        self.data = {}
        self.dirty = False
        if self.path and self.path.exists():
            db = json.loads(self.path.read_text())
            if db["context"] != context:
                sys.exit(
                    f"{self.path}: cache context differs ({db['context']} vs {context})"
                )
            self.data = db["entries"]

    def get(self, k):
        return self.data.get(k)

    def put(self, k, v):
        self.data[k] = v
        self.dirty = True

    def save(self):
        if self.path and self.dirty:
            tmp = str(self.path) + f".tmp{os.getpid()}"
            Path(tmp).write_text(
                json.dumps({"context": self.context, "entries": self.data})
            )
            os.replace(tmp, self.path)
            self.dirty = False


class Fragment:
    """One fragment: lazy `ferric.Mp2vSession`, cached att-MP2 and E_nl."""

    def __init__(self, pool, name, atoms):
        self.pool, self.name, self.atoms = pool, name, atoms
        self.sess = None
        self.rss_delta = None
        self.n_mp2 = 0
        self.n_nl_batches = 0

    def session(self):
        if self.sess is None:
            self.sess = self.pool._build(self)
        return self.sess

    def hf(self):
        k = f"{self.name}|hf"
        v = self.pool.cache.get(k)
        if v is None:
            v = self.session().rhf_energy
            self.pool.cache.put(k, v)
        return v

    def att(self, r0, s):
        k = f"{self.name}|att|{_seam_key(r0, s)}"
        v = self.pool.cache.get(k)
        if v is None:
            v = self.session().att_mp2(r0, _omega(r0, s))["att_mp2_corr"]
            self.n_mp2 += 1
            self.pool.n_mp2 += 1
            self.pool.cache.put(k, v)
        return v

    def coulomb(self):
        k = f"{self.name}|coul"
        v = self.pool.cache.get(k)
        if v is None:
            v = self.session().coulomb_mp2()
            self.pool.cache.put(k, v)
        return v

    def nl(self, tuples):
        keys = [f"{self.name}|nl|{_nl_key(t)}" for t in tuples]
        miss = [i for i, k in enumerate(keys) if self.pool.cache.get(k) is None]
        if miss:
            vals = self.session().vv10([tuples[i] for i in miss])
            self.n_nl_batches += 1
            for i, v in zip(miss, vals):
                self.pool.cache.put(keys[i], v)
        return np.array([self.pool.cache.get(k) for k in keys])


class Pool:
    """Many systems x fragments, one basis/NLC grid, an honest memory budget."""

    def __init__(
        self,
        basis="adz",
        nlc=(50, 50),
        cache=None,
        budget_gb=4.0,
        sets=None,
        coulomb=False,
    ):
        self.basis, self.nlc, self.budget = basis, tuple(nlc), budget_gb * 2**30
        self.sets = sets or load_sets()
        self.cache = Cache(
            cache, {"basis": basis, "nlc": list(nlc), "fc": "H0/Li-Ne1/Na-Ar5"}
        )
        self.frags = {}  # (set, id, tag) -> Fragment
        self.systems = {}  # (set, id) -> ref
        self.n_mp2 = 0
        self.coulomb = coulomb

    # -- membership / memory -------------------------------------------------
    def add(self, setname, ids, factors=None):
        """Register systems. `factors` (e.g. [0.9, 1.3]) adds rigid-monomer
        COM-stretched copies (stretch.py G1); a stretched system has no
        CCSD(T) reference (ref None) -- planted-parameter tests supply one.
        Non-CP monomers (mA/mB) are shared between stretches of one system."""
        import stretch

        for i in ids:
            rec = self.sets[setname]["systems"][str(i)]
            for f in factors or [None]:
                sid = i if f is None else f"{i}@{f:g}"
                self.systems[(setname, sid)] = rec["ref"] if f is None else None
                frs = fragments(rec) if f is None else stretch.frags_for(rec, f, "G1")
                for tag, atoms in frs.items():
                    base = f"{setname}-{i:02d}" + ("" if f is None else f"@{f:g}")
                    if tag in ("mA", "mB"):
                        base = f"{setname}-{i:02d}"
                    key = (setname, sid, tag)
                    shared = (
                        self.frags.get((setname, i, tag))
                        if tag in ("mA", "mB")
                        else None
                    )
                    self.frags[key] = shared or Fragment(self, f"{base}|{tag}", atoms)

    def drop(self, setname=None, i=None):
        """Release fragment sessions (and forget the systems) -- memory comes back."""
        for key in [
            k
            for k in self.frags
            if (setname is None or k[0] == setname) and (i is None or k[1] == i)
        ]:
            self.frags[key].sess = None
            if setname is not None and i is not None:
                del self.frags[key]
        if i is not None:
            self.systems.pop((setname, i), None)
        self.cache.save()

    def live_bytes(self):
        uniq = {id(f): f for f in self.frags.values()}.values()
        return sum(f.rss_delta or 0 for f in uniq if f.sess is not None)

    def _build(self, frag):
        import ferric

        natoms = sum(not a[0].startswith("@") for a in frag.atoms)
        projected = rss_bytes() + max(
            [f.rss_delta for f in self.frags.values() if f.rss_delta] + [natoms * 25e6]
        )
        if self.live_bytes() + (projected - rss_bytes()) > self.budget:
            raise MemoryError(
                f"session budget {self.budget / 2**30:.1f} GiB would be exceeded building "
                f"{frag.name} (live {self.live_bytes() / 2**30:.2f} GiB); drop() some systems"
            )
        obs_name, aux_name = BASES[self.basis]
        mol = ferric.Molecule.from_xyz_string(xyz_text(frag.atoms), 0, 1)
        r0 = rss_bytes()
        s = ferric.Mp2vSession(
            mol,
            ferric.BasisSet.bundled(obs_name),
            ferric.BasisSet.bundled(aux_name),
            frozen_core=n_frozen_core(frag.atoms),
            nlc_grid=self.nlc,
        )
        frag.rss_delta = max(rss_bytes() - r0, s.memory_bytes_estimate)
        return s

    # -- energies ------------------------------------------------------------
    def _tags(self, cp):
        return FRAG_CP if cp else FRAG_NCP

    def eint_b(self, setname, i, r0, s, bs, cp, c=0.0089, vv=None):
        """Interaction energy (kcal/mol) at each b in `bs` for one MP2 seam."""
        out = 0.0
        for sign, tag in zip((1.0, -1.0, -1.0), self._tags(cp)):
            f = self.frags[(setname, i, tag)]
            tuples = [_tuple(P(r0, s, b, c, vv)) for b in bs]
            tot = f.hf() + f.att(r0, s) + f.nl(tuples)
            out = out + sign * tot
        return out * K

    def eint(self, p, setname, i, cp=True):
        return float(
            self.eint_b(setname, i, p["r0"], p["s"], [p["b"]], cp, p["c"], p["vv"])[0]
        )

    def eint_coulomb(self, setname, i, cp=True):
        tot = 0.0
        for sign, tag in zip((1.0, -1.0, -1.0), self._tags(cp)):
            f = self.frags[(setname, i, tag)]
            tot += sign * (f.hf() + f.coulomb())
        return tot * K

    def errors_b(self, ids, r0, s, bs, cp, c=0.0089, vv=None):
        """(n_sys, n_b) signed errors vs the CCSD(T)/CBS reference, kcal/mol."""
        rows = []
        for sn, i in ids:
            rows.append(
                self.eint_b(sn, i, r0, s, bs, cp, c, vv) - self.systems[(sn, i)]
            )
        return np.array(rows)

    def report(self, p, ids=None, cp=True):
        ids = ids or sorted(self.systems)
        e = np.array([self.eint(p, sn, i, cp) - self.systems[(sn, i)] for sn, i in ids])
        return {
            "rmsd": float(np.sqrt(np.mean(e**2))),
            "mse": float(e.mean()),
            "mae": float(np.abs(e).mean()),
            "max": float(np.abs(e).max()),
            "n": len(e),
        }

    def stats(self):
        return {
            "live_sessions": sum(f.sess is not None for f in self.frags.values()),
            "live_bytes": self.live_bytes(),
            "budget_bytes": self.budget,
            "n_mp2": self.n_mp2,
            "rss": rss_bytes(),
        }


# ---------------------------------------------------------------------------
# Adaptive nested search (pre-registered rule; PREREGISTRATION.md Addendum F)
# ---------------------------------------------------------------------------
RULES = {
    # coarse grid
    "r0_coarse": [round(x, 4) for x in np.geomspace(0.6, 2.0, 7)],
    "s_coarse": [None, 1.0, 2.0, 4.0, 8.0],  # None = linked (1/sqrt2)
    # hard box (edge test); s capped at 8 until integral-validity anchors pass at 16/32
    "box": {"r0": (0.4, 3.0), "s": (0.5, 8.0)},
    # b rule
    "b_window": (2.0, 64.0),
    "n_cheb": 9,
    "b_verify_tol": 0.002,  # kcal/mol: spline vs exact RMSD at b*
    "b_limits": (0.4, 400.0),
    # local refinement
    "n_basins": 2,
    "step_floor": {"u": 0.02, "v": 0.04},  # ln units
    "min_gain": 0.005,  # kcal/mol RMSD per round
    "max_rounds": 4,
    "max_extensions": 2,
    "max_nodes": 80,
    "hess_h": 0.10,  # ln-unit FD step for the Hessian
}


def _sval(s):
    return LINKED_S if s is None else s


def optimal_b(pool, ids, r0, s, cp, c=0.0089, vv=None, rules=RULES):
    """1-D b minimisation on stored state. The per-system error is smooth in
    ln b, so it is evaluated on N_CHEB Chebyshev nodes in ln b over a window,
    spline-interpolated per system, and the pooled RMSD minimised on a fine
    ln b grid. The optimum is then VERIFIED by an exact evaluation (that point
    joins the node set); disagreement > `b_verify_tol` kcal/mol refits with the
    extra node. A minimum on the window edge shifts the window by x4 (up to
    `b_limits`); on the limit it is reported as an edge. Returns
    dict(b, rmsd, edge, n_kernels)."""
    from scipy.interpolate import CubicSpline

    lo_lim, hi_lim = rules["b_limits"]
    lo, hi = rules["b_window"]
    nk = 0
    for _ in range(6):
        k = np.arange(rules["n_cheb"])
        x = 0.5 * (math.log(lo) + math.log(hi)) + 0.5 * (
            math.log(hi) - math.log(lo)
        ) * -np.cos(math.pi * (k + 0.5) / rules["n_cheb"])
        bs = list(np.exp(x))
        err = pool.errors_b(ids, r0, s, bs, cp, c, vv)  # (n_sys, n_cheb)
        nk += len(bs)
        for _v in range(3):
            spl = CubicSpline(x, err, axis=1)
            xf = np.linspace(x[0], x[-1], 4001)
            r = np.sqrt((spl(xf) ** 2).mean(axis=0))
            j = int(np.argmin(r))
            bstar = math.exp(xf[j])
            ex = pool.errors_b(ids, r0, s, [bstar], cp, c, vv)
            nk += 1
            rtrue = float(np.sqrt((ex**2).mean()))
            if abs(rtrue - r[j]) <= rules["b_verify_tol"] or _v == 2:
                break
            x = np.append(x, xf[j])
            err = np.concatenate([err, ex], axis=1)
            o = np.argsort(x)
            x, err = x[o], err[:, o]
        if 0 < j < len(xf) - 1 and xf[j] not in (x[0], x[-1]):
            return {"b": bstar, "rmsd": rtrue, "edge": None, "n_kernels": nk}
        if j == 0 and lo / 4 >= lo_lim:
            lo, hi = lo / 4, lo * 2
        elif j == len(xf) - 1 and hi * 4 <= hi_lim:
            lo, hi = hi / 2, hi * 4
        else:
            return {
                "b": bstar,
                "rmsd": rtrue,
                "edge": "low" if j == 0 else "high",
                "n_kernels": nk,
            }
    return {"b": bstar, "rmsd": rtrue, "edge": "unresolved", "n_kernels": nk}


def node(pool, ids, r0, s, cp, c=0.0089, vv=None, rules=RULES):
    d = optimal_b(pool, ids, r0, s, cp, c, vv, rules)
    d.update(r0=r0, s=s)
    return d


def adaptive_search(pool, ids, cp, rules=RULES, c=0.0089, log=print):
    """Coarse (r0, s) grid -> extension if on the box edge -> local 3x3 refinement
    around the best `n_basins` nodes with halving steps. Returns map + optimum."""
    nodes = {}  # (round(u,6), round(v,6)) -> dict

    def key(r0, s):
        return (round(math.log(r0), 6), round(math.log(_sval(s)), 6))

    def ev(r0, s):
        k = key(r0, s)
        if k not in nodes:
            if len(nodes) >= rules["max_nodes"]:
                return None
            lo, hi = rules["box"]["r0"], rules["box"]["s"]
            if not (lo[0] <= r0 <= lo[1] and hi[0] <= _sval(s) <= hi[1]):
                return None
            nodes[k] = node(pool, ids, r0, s, cp, c, None, rules)
            log(
                f"node r0={r0:.3f} s={_sval(s):.3f} b*={nodes[k]['b']:.2f} rmsd={nodes[k]['rmsd']:.4f}"
            )
        return nodes[k]

    for r0 in rules["r0_coarse"]:
        for s in rules["s_coarse"]:
            ev(r0, s)
    pool.cache.save()
    r0s, ss = rules["r0_coarse"], rules["s_coarse"]
    # extension of the coarse box in r0 (s is capped by the hard box)
    for _ in range(rules["max_extensions"]):
        best = min(nodes.values(), key=lambda d: d["rmsd"])
        if best["r0"] == r0s[0] and best["r0"] / 1.22 >= rules["box"]["r0"][0]:
            r0s = [round(r0s[0] / 1.22, 4)] + r0s
            for s in ss:
                ev(r0s[0], s)
        elif best["r0"] == r0s[-1] and best["r0"] * 1.22 <= rules["box"]["r0"][1]:
            r0s = r0s + [round(r0s[-1] * 1.22, 4)]
            for s in ss:
                ev(r0s[-1], s)
        else:
            break
    du0 = float(np.mean(np.diff(np.log(r0s))))
    dv0 = float(np.mean(np.diff(np.log([_sval(x) for x in ss]))))
    centers = []
    for d in sorted(nodes.values(), key=lambda d: d["rmsd"]):
        u, v = math.log(d["r0"]), math.log(_sval(d["s"]))
        if all(
            abs(u - cu) > 1.5 * du0 or abs(v - cv) > 1.5 * dv0 for cu, cv in centers
        ):
            centers.append((u, v))
        if len(centers) == rules["n_basins"]:
            break
    rounds = []
    for cu, cv in centers:
        du, dv = du0 / 2, dv0 / 2
        prev = min(
            (
                d["rmsd"]
                for d in nodes.values()
                if (round(cu, 6), round(cv, 6)) == key(d["r0"], d["s"])
            ),
            default=math.inf,
        )
        for rd in range(rules["max_rounds"]):
            if du < rules["step_floor"]["u"] and dv < rules["step_floor"]["v"]:
                break
            for a in (-1, 0, 1):
                for b_ in (-1, 0, 1):
                    ev(math.exp(cu + a * du), math.exp(cv + b_ * dv))
            near = [
                d
                for d in nodes.values()
                if abs(math.log(d["r0"]) - cu) <= du * 1.01
                and abs(math.log(_sval(d["s"])) - cv) <= dv * 1.01
            ]
            best = min(near, key=lambda d: d["rmsd"])
            gain = prev - best["rmsd"]
            rounds.append(
                {
                    "center": [cu, cv],
                    "du": du,
                    "dv": dv,
                    "gain": gain,
                    "best": best["rmsd"],
                }
            )
            cu, cv = math.log(best["r0"]), math.log(_sval(best["s"]))
            prev = best["rmsd"]
            if gain < rules["min_gain"] and rd > 0:
                break
            du, dv = du / 2, dv / 2
        pool.cache.save()
    best = min(nodes.values(), key=lambda d: d["rmsd"])
    box = rules["box"]
    edge = {
        "r0_low": best["r0"] <= box["r0"][0] * 1.0001,
        "r0_high": best["r0"] >= box["r0"][1] * 0.9999,
        "s_high": _sval(best["s"]) >= box["s"][1] * 0.9999,
        "s_low": _sval(best["s"]) <= box["s"][0] * 1.0001,
        "b_edge": best["edge"],
        "on_coarse_boundary": best["r0"] in (r0s[0], r0s[-1]),
    }
    return {
        "nodes": list(nodes.values()),
        "best": best,
        "edge": edge,
        "rounds": rounds,
        "n_nodes": len(nodes),
    }


def hessian(pool, ids, best, cp, c=0.0089, h=0.10, sigma=0.05):
    """FD Hessian of F = RMSD^2 (kcal^2/mol^2) in w = (ln r0, ln s, ln b) at the
    optimum; step h in ln units (stated), 9 MP2 nodes (3x3 in (u, v)) with b varied
    cheaply. Returns eigenvalues/eigenvectors and the implied 1-sigma ranges at
    sigma_noise = sigma kcal/mol."""
    u0, v0, w0 = math.log(best["r0"]), math.log(_sval(best["s"])), math.log(best["b"])

    def F(du, dv, dw):
        r0, sv, b = math.exp(u0 + du), math.exp(v0 + dv), math.exp(w0 + dw)
        s = None if (best["s"] is None and du == 0 and dv == 0) else sv
        e = pool.errors_b(ids, r0, s, [b], cp, c, None)[:, 0]
        return float((e**2).mean())

    d = [-h, 0.0, h]
    f = {
        (a, b_, g): F(a, b_, g)
        for a in d
        for b_ in d
        for g in d
        if (a != 0 or b_ != 0 or g in d)
    }
    # full 3D stencil cost is cheap in b; MP2 nodes are the 9 (a, b_) pairs
    H = np.zeros((3, 3))
    for i in range(3):
        e = [0, 0, 0]
        e[i] = h
        e_m = [0, 0, 0]
        e_m[i] = -h
        H[i, i] = (f[tuple(e)] - 2 * f[(0.0, 0.0, 0.0)] + f[tuple(e_m)]) / h**2
    for i in range(3):
        for j in range(i + 1, 3):

            def at(si, sj):
                v = [0.0, 0.0, 0.0]
                v[i], v[j] = si * h, sj * h
                return f[tuple(v)]

            H[i, j] = H[j, i] = (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1)) / (
                4 * h * h
            )
    evals, evecs = np.linalg.eigh(H)
    n = len(ids)
    sig = sigma
    # chi2 = n F / sig^2 ; Delta chi2 = 1 along eigvec i => delta = sqrt(2 sig^2 / (n * lam_i))
    one_sigma = [
        math.sqrt(2 * sig**2 / (n * lam)) if lam > 0 else math.inf for lam in evals
    ]
    return {
        "h": h,
        "vars": ["ln r0", "ln s", "ln b"],
        "F0": f[(0.0, 0.0, 0.0)],
        "hessian": H.tolist(),
        "eigenvalues": evals.tolist(),
        "eigenvectors": evecs.T.tolist(),
        "one_sigma_ln_units": one_sigma,
        "condition": (evals[-1] / evals[0]) if evals[0] > 0 else math.inf,
    }


# ---------------------------------------------------------------------------
# long-running job loop
# ---------------------------------------------------------------------------
def serve(pool, jobdir, poll=2.0):
    """Poll jobdir/*.json; each job {"op": ..., ...} -> jobdir/<name>.out.json.
    ops: add {set, ids} | drop {set, id} | eval {params, ids?, cp} | profile_b
    {r0, s, bs, ids?, cp} | search {ids?, cp} | stats | quit. Sessions persist."""
    jobdir = Path(jobdir)
    jobdir.mkdir(exist_ok=True)
    while True:
        for jf in sorted(jobdir.glob("*.json")):
            if jf.name.endswith(".out.json"):
                continue
            out = jf.with_name(jf.stem + ".out.json")
            if out.exists():
                continue
            job = json.loads(jf.read_text())
            try:
                res = _run_job(pool, job)
            except Exception as ex:  # report, keep serving
                res = {"error": f"{type(ex).__name__}: {ex}"}
            out.write_text(json.dumps(res, default=float))
            if job.get("op") == "quit":
                return
        time.sleep(poll)


def _ids(pool, job):
    return [tuple(x) for x in job["ids"]] if "ids" in job else sorted(pool.systems)


def _run_job(pool, job):
    op = job["op"]
    if op == "add":
        pool.add(job["set"], job["ids"])
        return pool.stats()
    if op == "drop":
        pool.drop(job.get("set"), job.get("id"))
        return pool.stats()
    if op == "eval":
        return pool.report(job["params"], _ids(pool, job), job.get("cp", True))
    if op == "profile_b":
        e = pool.errors_b(
            _ids(pool, job), job["r0"], job.get("s"), job["bs"], job.get("cp", True)
        )
        return {"bs": job["bs"], "rmsd": np.sqrt((e**2).mean(axis=0)).tolist()}
    if op == "search":
        r = adaptive_search(
            pool, _ids(pool, job), job.get("cp", True), log=lambda *_: None
        )
        return {k: r[k] for k in ("best", "edge", "n_nodes")}
    if op in ("stats", "quit"):
        return pool.stats()
    raise ValueError(f"unknown op {op}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "serve", "stats"])
    ap.add_argument("--set", default="a24")
    ap.add_argument("--systems", default="")
    ap.add_argument("--basis", default="adz", choices=list(BASES))
    ap.add_argument("--nlc", default="50,50")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--budget-gb", type=float, default=4.0)
    ap.add_argument("--cp", action="store_true")
    ap.add_argument("--dir", default="jobs")
    a = ap.parse_args(argv)
    pool = Pool(a.basis, [int(x) for x in a.nlc.split(",")], a.cache, a.budget_gb)
    if a.cmd == "serve":
        return serve(pool, a.dir)
    sets = pool.sets[a.set]["systems"]
    ids = (
        [int(x) for x in a.systems.split(",")] if a.systems else sorted(map(int, sets))
    )
    pool.add(a.set, ids)
    res = adaptive_search(pool, [(a.set, i) for i in ids], a.cp)
    print(
        json.dumps(
            {k: res[k] for k in ("best", "edge", "n_nodes")}, indent=1, default=float
        )
    )
    pool.cache.save()


if __name__ == "__main__" and not hasattr(sys, "ps1"):
    sys.exit(main())
