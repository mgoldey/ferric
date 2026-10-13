"""Parse probe logs into markdown tables (printed to stdout)."""

import re
import os

L = os.path.join(os.path.dirname(__file__), "..", "logs")


def rd(name):
    return open(os.path.join(L, name)).read().splitlines()


def kv(line):
    return dict(re.findall(r"(\w[\w\-\[\],|()]*)=([^\s]+)", line))


def validity(name, title):
    print(f"\n#### {title}\n")
    print(
        "| op | set | bound | n(true>1e-12) | violations | zeroQ | vacuous(inf) | min bound/true | median | q90 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for ln in rd(name):
        m = re.match(
            r"VALID (\S+) (triple|quartet) (.*?) n=(\d+) violations=(\d+)(?: zeroQ=(\d+))? vacuous\(inf\)=(\d+) minratio=(\S+) q10=(\S+) q50=(\S+) q90=(\S+)",
            ln,
        )
        if not m:
            continue
        op, st, b, n, v, z, vac, mn, q10, q50, q90 = m.groups()
        print(
            f"| {op} | {st} | {b} | {n} | {v} | {z or 'n/a'} | {vac} | {float(mn):.4g} | {float(q50):.3g} | {float(q90):.3g} |"
        )


def count3(name):
    print(
        "\n#### C12 global shell-triple survivors (fraction of ALL 2.354M triples; weighted = x primitive-triples x nfn)\n"
    )
    print(
        "| operator | tau | oracle (true>=tau) | Schwarz | B1 (delta=1e-4) | min(S,B1) | B2 (sample 20k) | min(S,B2) | wfrac oracle / S / S&B1 / S&B2 | S false-pos share | S&B1 false-pos share |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    lines = rd(name)
    b2 = {}
    for ln in lines:
        m = re.match(
            r"COUNT3B2 (\S+) tau=(\S+) sample=\d+ frac\[oracle,schwarz,B2,min\(S,B2\)\]=\[(.*?)\] wfrac=\[(.*?)\]",
            ln,
        )
        if m:
            b2[(m.group(1), m.group(2))] = (
                [float(x) for x in m.group(3).split(",")],
                [float(x) for x in m.group(4).split(",")],
            )
    for ln in lines:
        m = re.match(
            r"COUNT3 (\S+) delta=1e-4 tau=(\S+) total=(\d+) frac\[.*?\]=\[(.*?)\] wfrac=\[(.*?)\] miss.*?falsepos\[S,B1,cmb\]=\[(.*?)\]",
            ln,
        )
        if not m:
            continue
        op, tau, tot, fr, wf, fp = m.groups()
        fr = [float(x) for x in fr.split(",")]
        wf = [float(x) for x in wf.split(",")]
        fp = [int(x) for x in fp.split(",")]
        tot = int(tot)
        bb, bw = b2[(op, tau)]
        print(
            f"| {op} | {tau} | {fr[0]:.3f} | {fr[1]:.3f} | {fr[2]:.3f} | {fr[3]:.3f} | {bb[2]:.3f} | {bb[3]:.3f} | {wf[0]:.3f} / {wf[1]:.3f} / {wf[3]:.3f} / {bw[3]:.3f} | {fp[0] / (fr[1] * tot):.2f} | {fp[2] / (fr[3] * tot):.2f} |"
        )


def prim(name):
    print(
        "\n#### C12 primitive-level screening inside the shell triples (3000-triple sample; fraction of ALL primitive work kept)\n"
    )
    print(
        "| operator | tau | shell-Schwarz only | +primitive bound (Coulomb-majorant, kernel-independent) | +primitive bound (kernel-aware, 6-eps grid) | primitive bound alone (no shell screen), coul / kernel |"
    )
    print("|---|---|---|---|---|---|")
    for ln in rd(name):
        m = re.match(
            r"PRIM3 (\S+) tau=(\S+) sample=\d+ work.*?coul-prim-bound=(\S+) kernel-prim-bound=(\S+) \| after shell-Schwarz: shell-frac=(\S+) \+coul-prim=(\S+) \+kernel-prim=(\S+)",
            ln,
        )
        if m:
            op, tau, c, k, s, sc, sk = m.groups()
            print(f"| {op} | {tau} | {s} | {sc} | {sk} | {c} / {k} |")


def lmp2(name, title):
    print(f"\n#### {title}\n")
    print(
        "| operator | tau | evaluations seen | skippable: Schwarz | B1 | S or B1 | weighted S / S&B1 | B2-sample: S / B2 / S or B2 |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for ln in rd(name):
        m = re.match(
            r"LMP2 (\S+) (\S+) maps.*? tau=(\S+) apply.*?\| map-survivor triples seen=(\d+) skippable frac S=(\S+) B1=(\S+) cmb=(\S+) \(weighted S=(\S+) B1=(\S+) cmb=(\S+)\) \| B2 sample n=(\d+): skippable S=(\S+) B2=(\S+) S\|B2=(\S+)",
            ln,
        )
        if m and m.group(3) != "0e0":
            g = m.groups()
            print(
                f"| {g[1]} {g[0]} | {g[2]} | {g[3]} | {g[4]} | {g[5]} | {g[6]} | {g[7]} / {g[9]} | {g[11]} / {g[12]} / {g[13]} |"
            )


def apply(name, title):
    print(f"\n#### {title}\n")
    print(
        "| tau | rule | e_corr | dE vs unscreened | rel | triples evaluated | wall s | t_eri3 s |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for ln in rd(name):
        m = re.match(
            r"LMP2 (\S+) (\S+) maps.*? tau=(\S+) apply=(\S+) rule=(\d) e_corr=(\S+) dE_vs_base=(\S+) rel=(\S+) triples_eval=(\d+) skipped=(\d+) wall=(\S+) t_eri3=(\S+)",
            ln,
        )
        if m:
            g = m.groups()
            print(
                f"| {g[2]} | {g[4]} | {g[5]} | {g[6]} | {g[7]} | {g[8]} | {g[10]} | {g[11]} |"
            )


if __name__ == "__main__":
    validity(
        "validity_c4_631g.log",
        "Validity: C4H10 6-31G / cc-pVDZ-RI, ALL triples (zeroQ = libint2 None-diagonal artifact, not a violation)",
    )
    validity("validity_c4_631g_quartets.log", "Validity: C4H10 6-31G, ALL quartets")
    count3("count_c12_631g.log")
    prim("count_c12_631g_v2.log")
    lmp2(
        "lmp2_overlay_c12_linked.log",
        "Production-map overlay C12 (linked terfc): skippable fraction of the map-surviving triple EVALUATIONS",
    )
    lmp2("lmp2_countonly_c20.log", "Production-map overlay C20 (count only)")
    lmp2("lmp2_countonly_c32.log", "Production-map overlay C32 (count only)")
    apply("lmp2_apply_c12_linked_schwarz.log", "C12 linked, APPLY Schwarz-only skip")
    apply("lmp2_apply_c12_linked_cmb.log", "C12 linked, APPLY S|B1 skip")


def count4(name):
    print(
        "\n#### C8 quartets (ab|cd): 4.28M shell quartets, ORACLE = true max|(ab|cd)| >= tau\n"
    )
    print(
        "| operator | tau | oracle | Schwarz | B1 (delta=1e-4) | min(S,B1) | weighted oracle / S / S&B1 | oracle misses by S,B1,cmb |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for ln in rd(name):
        m = re.match(
            r"COUNT4 (\S+) delta=1e-4 tau=(\S+) total=\d+ frac\[.*?\]=\[(.*?)\] wfrac=\[(.*?)\] miss=\[(.*?)\]",
            ln,
        )
        if m:
            op, tau, fr, wf, ms = m.groups()
            fr = [float(x) for x in fr.split(",")]
            wf = [float(x) for x in wf.split(",")]
            print(
                f"| {op} | {tau} | {fr[0]:.3f} | {fr[1]:.3f} | {fr[2]:.3f} | {fr[3]:.3f} | {wf[0]:.3f} / {wf[1]:.3f} / {wf[3]:.3f} | {ms} |"
            )


if __name__ == "__main__":
    count4("count_c8_quartets.log")
    apply(
        "lmp2_apply_c16_linked.log",
        "C16 linked (r0=1 A, s=0.7071), APPLY S|B1 skip; tau=1e-4/1e-3 are the deliberately loose mutation rows",
    )
    apply("lmp2_apply_c16_ncp.log", "C16 NCP (r0=0.991 A, s=1.354), APPLY S|B1 skip")
