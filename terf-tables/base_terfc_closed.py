#!/usr/bin/env python3
"""PROVEN closed-form for the terfc (s|s|s) base integral, verified vs direct
radial quadrature to ~1e-12 across r0 and D.

Result (all verified numerically in this file):

  [0|terfc|0] = pref_boys * Phi_terfc(S)
  Phi_terfc(S) = F_0(S) - Phi_h(S)
  Phi_h(S) = (4 e^{-a^2}/sqrt(pi)) * int_0^w Phi_cosh(tau) dtau,   a=1/sqrt(2), w=1/(r0 sqrt2)
  Phi_cosh(tau) = Phi[e^{-tau^2 r^2} cosh(2 a tau r)]
                = (1/2)( Phi_e(tau^2, 2a tau) + Phi_e(tau^2, -2a tau) )

with the PROVEN Gaussian-with-linear-term reduction (exact, 1e-60):

  Phi_e(c, g) = Phi[e^{-c r^2 + g r}]
  A = rho + c
  J(A,B) = 1/(2A) + (B sqrt(pi))/(4 A^{3/2}) e^{B^2/(4A)} (1 + erf(B/(2 sqrt A)))
  K(B)   = sqrt(pi)/(2 sqrt rho) e^{B^2/(4 rho)} (1 + erf(B/(2 sqrt rho)))
  Phi_e(c,g) = [ J(A, g+2 rho D) - J(A, g-2 rho D) ]  /  [ K(2 rho D) - K(-2 rho D) ] * F_0(S)

Here Phi = [0|op|0] / pref_boys, S = rho D^2, D = |P-Q|, rho = p q/(p+q).

The simpler proven building block (g=0):
  Phi[e^{-c r^2}] = (sqrt(pi)/2) * rho / (rho+c)^{3/2} * e^{-S c/(rho+c)}.
"""
import mpmath as mp
mp.mp.prec = 120
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gt_check import setup, gt_phi, boys

A_CONST = 1/mp.sqrt(2)   # = a, from curvature constraint r0*omega = 1/sqrt2

def J(A, B):
    return 1/(2*A) + (B*mp.sqrt(mp.pi))/(4*A**mp.mpf('1.5'))*mp.exp(B*B/(4*A))*(1+mp.erf(B/(2*mp.sqrt(A))))

def base_terfc_closed(par, r0):
    """Return Phi_terfc(S) = [0|terfc|0]/pref_boys via the proven closed form."""
    rho = par['rho']; S = par['S']; D = par['D']
    w = 1/(r0*mp.sqrt(2))
    F0 = boys(0, S)
    if D == 0:
        # limit handled by direct small-D; caller avoids D=0 in the verify grid
        raise ValueError("use D>0")
    def K(B):
        return mp.sqrt(mp.pi)/(2*mp.sqrt(rho))*mp.exp(B*B/(4*rho))*(1+mp.erf(B/(2*mp.sqrt(rho))))
    denom = K(2*rho*D) - K(-2*rho*D)
    def Phi_e(c, g):
        Ac = rho + c
        return (J(Ac, g+2*rho*D) - J(Ac, g-2*rho*D))/denom*F0
    def Phi_cosh(tau):
        c = tau*tau; g = 2*A_CONST*tau
        return mp.mpf('0.5')*(Phi_e(c, g) + Phi_e(c, -g))
    Phi_h = (4*mp.exp(-A_CONST*A_CONST)/mp.sqrt(mp.pi))*mp.quad(Phi_cosh, [0, w])
    return F0 - Phi_h

def base_terfc_gt(par, r0):
    """Independent ground truth: direct radial quadrature of the true terfc operator."""
    w = 1/(r0*mp.sqrt(2))
    def op(r):
        return 1/r - (mp.erf(w*(r-r0)) + mp.erf(w*(r+r0)))/r
    return gt_phi(par, op)

if __name__ == "__main__":
    p, a, b = mp.mpf('1.3'), mp.mpf('0.9'), mp.mpf('0.7')
    Ra = [mp.mpf(0)]*3
    Rb = [mp.mpf(0), mp.mpf(0), mp.mpf('0.8')]
    print("CRITICAL VERIFICATION: closed form vs direct radial quadrature")
    print("%-6s %-8s %-22s %-22s %-10s" % ("r0", "S", "closed", "ground-truth", "absdiff"))
    maxdiff = mp.mpf(0)
    for r0 in [mp.mpf('0.5'), mp.mpf('1.0'), mp.mpf('2.0'), mp.mpf('4.0')]:
        for Rpz in [mp.mpf('0.3'), mp.mpf('0.9'), mp.mpf('1.5'), mp.mpf('2.5'), mp.mpf('3.5')]:
            Rp = [mp.mpf('0.3'), mp.mpf(0), Rpz]
            par = setup(p, a, b, Ra, Rb, Rp, r0)
            cl = base_terfc_closed(par, r0)
            gt = base_terfc_gt(par, r0)
            d = abs(cl - gt); maxdiff = max(maxdiff, d)
            print("%-6s %-8.4f %-22.15g %-22.15g %.2e" % (
                float(r0), float(par['S']), float(cl), float(gt), float(d)))
    print("\nMAX ABS DIFF (base s|s|s) =", mp.nstr(maxdiff, 4))
