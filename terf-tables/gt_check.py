#!/usr/bin/env python3
"""Establish an independent ground-truth for [0|f|0] over s-Gaussians and
anchor it against the Coulomb Boys value F_0(S)."""
import mpmath as mp
mp.mp.prec = 200
SQRT_PI = mp.sqrt(mp.pi)

def boys(m,T):
    T=mp.mpf(T)
    if T==0: return mp.mpf(1)/(2*m+1)
    aa=mp.mpf(m)+mp.mpf('0.5')
    return mp.gammainc(aa,0,T)/(2*T**aa)

def setup(p,a,b,Ra,Rb,Rp,r0):
    q=a+b
    Q=[(a*Ra[i]+b*Rb[i])/q for i in range(3)]
    AB2=sum((Ra[i]-Rb[i])**2 for i in range(3))
    K=mp.exp(-a*b/q*AB2)
    rho=p*q/(p+q)
    D2=sum((Rp[i]-Q[i])**2 for i in range(3))
    D=mp.sqrt(D2); S=rho*D2
    pref=2*mp.pi**mp.mpf('2.5')/(p*q*mp.sqrt(p+q))*K
    return dict(p=p,q=q,rho=rho,D=D,S=S,pref=pref,r0=r0)

# Independent GT: two spherical Gaussian charges, exponent p at P and q at Q.
# The two-electron integral  int d3r1 d3r2 e^{-p|r1-P|^2} e^{-q|r2-Q|^2} f(|r1-r2|)
# The standard reduction: define the difference density; integrating out the
# center of mass gives a single Gaussian in the separation vector s=r1-r2 with
# exponent rho=pq/(p+q), centered at D=P-Q, times normalisation (pi/(p+q))^{3/2}
# ... Actually integral of two gaussians of separations reduces to:
#   = (pi/(p+q))^{3/2} * (pi/rho)^{?}...
# Let's just do it fully numerically as a 1D integral over the separation r,
# with the correct radial weight, and CALIBRATE the constant against Boys.
#
# For s=r1-r2, the product-density integrated over COM yields weight
#   N(s) = (pi/(p+q))^{3/2} * exp(-rho |s-D|^2) ... times overall gaussian norms.
# Then value = norm * int d3s exp(-rho|s-D|^2) f(|s|).
# Do the 3D s-integral -> reduces to 1D over |s|=r with angular avg over D:
#   int d3s e^{-rho|s-D|^2} f(|s|)
#     = 2pi int_0^inf r^2 f(r) e^{-rho(r^2+D^2)} * [int_{-1}^1 e^{2 rho r D x} dx] dr
#     = 2pi int_0^inf r^2 f(r) e^{-rho(r^2+D^2)} * sinh(2 rho r D)/(rho r D) dr
#     = (2pi/(rho D)) e^{-rho D^2} int_0^inf r f(r) e^{-rho r^2} sinh(2 rho r D) dr
def gt_phi_over_pref(par, fop):
    """Return Phi = [0|f|0]/pref via the 1D separation integral, CALIBRATED so
    that coulomb gives F_0(S). We fix the multiplicative constant by matching
    the coulomb anchor once, then reuse it."""
    rho=par['rho']; D=par['D']
    if D==0:
        core = 2*mp.pi*mp.quad(lambda r: r*r*fop(r)*mp.exp(-rho*r*r), [0,mp.inf])
    else:
        core = (2*mp.pi/(rho*D))*mp.exp(-rho*D*D)*mp.quad(
            lambda r: r*fop(r)*mp.exp(-rho*r*r)*mp.sinh(2*rho*r*D),[0,mp.inf])
    return core

def gt_phi(par, fop):
    """Calibrated independent GT for Phi=[0|f|0]/pref. Calibration constant is
    fixed by matching coulomb->F_0(S) at this rho (geometry-independent)."""
    C = gt_phi_over_pref(par, lambda r:1/r)/boys(0,par['S'])
    return gt_phi_over_pref(par, fop)/C

if __name__=="__main__":
    par=setup(mp.mpf('1.3'),mp.mpf('0.9'),mp.mpf('0.7'),
              [mp.mpf(0),mp.mpf(0),mp.mpf(0)],
              [mp.mpf(0),mp.mpf(0),mp.mpf('0.8')],
              [mp.mpf('0.3'),mp.mpf(0),mp.mpf('1.5')], mp.mpf('2.0'))
    print("rho=",par['rho'],"D=",par['D'],"S=",par['S'])
    coul = gt_phi_over_pref(par, lambda r: 1/r)
    F0 = boys(0, par['S'])
    print("coulomb core =", coul)
    print("F_0(S)       =", F0)
    print("ratio core/F0=", coul/F0)
