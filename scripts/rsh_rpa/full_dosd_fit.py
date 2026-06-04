#!/usr/bin/env python3
"""Full-DOSD fit of omega-from-dielectric (data from full_dosd_calibration_dump).
VERDICT: lambda_max r drops 0.97 (7-mol) -> 0.72 (full 13). Single-descriptor ceiling
~0.72-0.80 (top3 best). 2-feature lambda_max+alpha/V LOO-MAE(omega) 0.047 but 4 params
at n~11. hf,h2 unfittable (C6 overshoots DOSD at omega->0). The clean 7-mol law was a
small-sample artifact; the concept survives only as a weak/modest correlation."""
import numpy as np
rows={'h2o':(2.5948,5.7748,5.9237,0.2182,0.1759),'ch4':(2.2337,5.5154,7.4051,0.3036,0.1255),
'nh3':(2.4809,5.7978,6.7583,0.2630,0.1035),'co':(3.0979,6.6318,8.0631,0.1801,0.3651),
'n2':(3.2815,6.8053,8.2230,0.1760,0.4640),'co2':(3.1830,7.7430,12.0077,0.1911,0.3712),
'c2h4':(2.6899,6.5319,11.2474,0.2767,0.2267),'h2':(1.6429,4.3508,1.8698,0.3762,0.0200),
'hf':(2.5325,5.5337,4.9410,0.1615,0.0200),'hcl':(2.0585,5.7306,6.1748,0.1541,0.2580),
'h2s':(2.0546,5.8525,7.1712,0.1908,0.2624),'c2h2':(3.0418,6.6727,9.2609,0.2367,0.2886),
'c2h6':(2.2659,6.2564,12.9889,0.2971,0.1574)}
A=np.array(list(rows.values())); lmax,top3,tln,aov,wopt=[A[:,i] for i in range(5)]
print("full-13 r(lmax)=",round(np.corrcoef(lmax,wopt)[0,1],3),"r(top3)=",round(np.corrcoef(top3,wopt)[0,1],3))
