import json,sys
sys.path.insert(0,'benchmarks/a24-mp2v-refit')
import numpy as np, stretch, att_scan
K=att_scan.K
d=json.load(open('benchmarks/a24-mp2v-refit/results/att_scan_f1.json'))
arms=att_scan.ARMS
print([ (a['r0'],a['r0omega']) for a in arms])
for sid in (2,4,5,8,19):
    D,A,B=d[f"{sid}|1.0|dimer"],d[f"{sid}|mono|mA"],d[f"{sid}|mono|mB"]
    e=[((D['rhf']+D['att'][k])-(A['rhf']+A['att'][k])-(B['rhf']+B['att'][k]))*K for k in range(len(arms))]
    p17=stretch.paper('17',sid)[1]
    lin=[(a['r0'],x) for a,x in zip(arms,e) if a['r0omega'] is None and a['r0']>=1.25]
    xs,ys=zip(*lin); 
    r0eff=float(np.interp(p17,[y for y in ys][::-1] if ys[0]>ys[-1] else ys,[x for x in xs][::-1] if ys[0]>ys[-1] else xs))
    print(sid,'paper',p17,'E(r0)',[round(x,4) for x in e[:6]],'r0eff',round(r0eff,4),'| r0*omega arms',[round(x,4) for x in e[6:]])
