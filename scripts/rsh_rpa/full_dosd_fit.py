#!/usr/bin/env python3
"""Full-DOSD fit of omega-from-dielectric (14 closed-shell mols; O2 open-shell skipped).
Data from full_dosd_calibration_dump (ferric, LC-wPBE/aug-cc-pVDZ).

VERDICT: the clean 7-molecule law (lambda_max r=0.97, parameter-free (1/3)(lambda-2),
C6 MAE 1.74%) was a SMALL-SAMPLE ARTIFACT. On the full set:
  r(lambda_max) = 0.72,  r(top3) = 0.75  (single-descriptor ceiling)
  descriptors NOT stable as molecules are added (top3 0.80@13 -> 0.75@14).
  hf, h2 unfittable by any omega (C6 overshoots DOSD at omega->0).
  c6h6 (lambda_max 2.60 ~= h2o 2.60, but omega_opt 0.26 vs 0.18) shows the leading
  eigenvalue under-ranks screening for large multi-mode systems: screening for C6 is a
  COLLECTIVE spectral property, not the single top mode -- but trace_log (spectral sum)
  correlates WORSE (0.32, conflates screening with size).

CONCLUSION: omega-from-computed-dielectric is a real-but-weak correlation, no transferable
law. Strong claim dead. Not a method; at most an open puzzle (right size-intensive spectral
descriptor unknown). Needs a much bigger set to pursue without overfitting."""
import numpy as np
rows = {
 'h2o':(2.5948,5.7748,5.9237,0.2182,0.1759),'ch4':(2.2337,5.5154,7.4051,0.3036,0.1255),
 'nh3':(2.4809,5.7978,6.7583,0.2630,0.1035),'co':(3.0979,6.6318,8.0631,0.1801,0.3651),
 'n2':(3.2815,6.8053,8.2230,0.1760,0.4640),'co2':(3.1830,7.7430,12.0077,0.1911,0.3712),
 'c2h4':(2.6899,6.5319,11.2474,0.2767,0.2267),'h2':(1.6429,4.3508,1.8698,0.3762,0.0200),
 'hf':(2.5325,5.5337,4.9410,0.1615,0.0200),'hcl':(2.0585,5.7306,6.1748,0.1541,0.2580),
 'h2s':(2.0546,5.8525,7.1712,0.1908,0.2624),'c2h2':(3.0418,6.6727,9.2609,0.2367,0.2886),
 'c2h6':(2.2659,6.2564,12.9889,0.2971,0.1574),'c6h6':(2.5996,7.6652,28.2314,0.2927,0.2615),
}
A=np.array(list(rows.values())); lmax,top3,tln,aov,wopt=[A[:,i] for i in range(5)]
p=lambda x,y:np.corrcoef(x,y)[0,1]
print(f"full-14  r(lmax)={p(lmax,wopt):+.3f}  r(top3)={p(top3,wopt):+.3f}  r(aoV)={p(aov,wopt):+.3f}  r(trace_ln)={p(tln,wopt):+.3f}")
