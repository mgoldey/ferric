import sys

sys.path.insert(0, ".")
import time
from run_kpts_anchor import H2_A, H2_ATOMS
from pyscf.pbc import gto as pgto, scf as pscf, df as pdf, tools as ptools

pc = pgto.Cell(a=H2_A, atom=H2_ATOMS, basis="sto-3g", unit="B", cart=True, verbose=0)
pc.precision = 1e-12
pc.max_memory = 1200
pc.build()
kp = pc.make_kpts([1, 1, 2], with_gamma_point=False)
print("kpts", repr(kp), flush=True)
print("madelung", repr(ptools.pbc.madelung(pc, kp)), flush=True)
kp3 = pc.make_kpts([2, 1, 3], with_gamma_point=False)
print("kpts213", repr(kp3), flush=True)
for ex in (None, "ewald"):
    t = time.time()
    mf = pscf.KRHF(pc, kp, exxdiv=ex)
    mf.with_df = pdf.AFTDF(pc, kp)
    mf.with_df.mesh = [61] * 3
    mf.conv_tol = 1e-11
    e = mf.kernel()
    print(
        f"exxdiv={ex} E={e!r} conv={mf.converged} eps={[list(x) for x in mf.mo_energy]} {time.time() - t:.0f}s",
        flush=True,
    )
