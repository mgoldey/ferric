# OP-MP3 / SSD-OPMP3: working equations

Source: K. Bystrom, T. C. Berkelbach, D. Hait, "Near Chemical Accuracy From
Size-Consistent Padé Resummed Møller-Plesset Perturbation Theory",
arXiv:2609.25330v2 (22 Oct 2026). Data and code: Zenodo 10.5281/zenodo.23482348
(`SI_pade_paper.tar.gz`). Equation numbers below are the paper's. The
prototype is `opmp3_proto.py`.

## 1. MP amplitudes (Eqs. 4-10)

The spin orbitals are real and canonical (or semicanonical, Sec. 5). Occupied
orbitals are i, j, k, l; virtual orbitals are a, b, c, d.
Δ_ij^ab = ε_a + ε_b − ε_i − ε_j.

- Eq. 6: E_c^(n) = ¼ Σ_ijab c_ijab^(n−1) ⟨ij‖ab⟩. Singles drop out by Brillouin's
  theorem. Triples drop out by Slater-Condon.
- Eq. 7: c_ijab^(1) = −⟨ij‖ab⟩ / Δ_ij^ab.
- Eq. 8: E_c^(2) = −¼ Σ |⟨ij‖ab⟩|² / Δ.
- Eq. 9: c_ijab^(2) = −(1/Δ_ij^ab) [ ½ Σ_cd ⟨ab‖cd⟩ c_ijcd^(1) + ½ Σ_kl ⟨kl‖ij⟩ c_klab^(1)
  + Σ_kc ( ⟨kb‖cj⟩ c_ikac − ⟨kb‖ci⟩ c_jkac − ⟨ka‖cj⟩ c_ikbc + ⟨ka‖ci⟩ c_jkbc ) ].
  The last bracket is R_ijab − R_jiab − R_ijba + R_jiba with
  R_ijab = Σ_kc ⟨kb‖cj⟩ c_ikac^(1).
- Eq. 10 is E_c^(3) in amplitude form. The prototype evaluates E_c^(3) as Eq. 6
  with n = 3 and checks the result against ferric's and PySCF's MP3.

Cost is O(N_o² N_v⁴), from the ⟨ab‖cd⟩ ladder in c^(2), the same as MP3.
The prototype contracts the ladder in its spatial form,
½ Σ_cd ⟨ab‖cd⟩ c_ijcd = Σ_cd (ac|bd) c_ijcd (c is antisymmetric in cd), with
the spatial (ac|bd) of each spin pair. Every other block is held as a
spin-orbital tensor.

## 2. The self-energy matrix W (Eqs. 11-12)

- Eq. 11a: W_ij^(n) = ⅛ Σ_kab ( c_ikab^(n−1) ⟨jk‖ab⟩ + c_jkab^(n−1) ⟨ik‖ab⟩ ).
  This is an N_o × N_o occupied-space matrix: an orbital-resolved partition of
  E^(n). The paper relates it to the 2p1h block of the symmetrized static
  self-energy.
- Eq. 11b: E_c^(n) = Tr W^(n).
- Eq. 12: for infinitely separated fragments, W^(n) is block diagonal, because
  c_ikab vanishes when i and k sit on different fragments. Under an occupied
  rotation U, W → U W Uᵀ. W is invariant to virtual rotations.

W^(2) comes from c^(1) (it is "the MP2 self-energy"). W^(3) comes from c^(2)
(the third-order increment, "the MP3 piece"). The resummation acts on W^(2)
and W^(3). It never acts on W^(2) + W^(3).

## 3. Spin channels (Eqs. 16-18)

W is block diagonal in spin: W_{iσ jσ̄} = 0, which follows from Eq. 11a. Each
spin block splits by the spin of the summed occupied index k:

- Eq. 17 (SS): W^SS(n)_{iσ jσ} = ⅛ Σ_{kσ aσ bσ} ( c_ikab ⟨jk‖ab⟩ + (i↔j) ).
- Eq. 18 (OS): W^OS(n)_{iσ jσ} = ¼ Σ_{k σ̄, a σ, b σ̄} ( c_ikab ⟨jk|ab⟩ + (i↔j) ), with
  plain, not antisymmetrized, integrals.

Eq. 18 equals the k ∈ σ̄ part of Eq. 11a. In the spin-orbital sum, (a σ, b σ̄)
and (a σ̄, b σ) give equal terms, so the two ⅛ terms add to ¼, and
⟨jk‖ab⟩ = ⟨jk|ab⟩ for those spins. The prototype builds SS and OS as the
k-spin split of Eq. 11a. A test checks the split against Eq. 18 written out
literally (`--anchor eq18`).

The paper's code (`wmatrix_code/pyscf_w_matrices.py`) uses the same
prefactors and symmetrizes in (i, j). It takes PySCF ADC(3) amplitudes
(t2[0] = c^(1), t2[1] = c^(2)). For closed shells it doubles the RADC
same-spin blocks.

## 4. Resummation (Eqs. 1-3, 13-15, 19-22)

- Eq. 1 / 2 (scalar, energy-level Padé):
  E^[2,3] = E2 (E2 − E3)^{−1} E2 = E2 / (1 − E3/E2) = E2 + E3 + E3²/E2 + ….
- Eq. 3: E^[2,3] is NOT size-consistent. For A + B it is a ratio of sums. It is
  size-extensive only for identical copies, where E3/E2 is the same for each.
- Eq. 13: P(A, B) = A (A − B)^{−1} A, a matrix [1/1] Padé. Its series is
  A + B + B A^{−1} B + …, so Tr P(W2, xW3) = E2 + x E3 + x² Tr(W3 W2^{−1} W3) + ….
- Eq. 14 (OP-MP3): E_c = Tr P(W^(2), W^(3)). A is the second-order matrix in the
  numerator and the denominator. B is the third-order matrix, in the
  denominator only.
- Eq. 15 (OP-MP3(x)): E_c = Tr P(W^(2), x W^(3)). The scale x multiplies W^(3)
  inside the Padé, in the B slot. The paper's optimum x = 0.7890.
- Eq. 19: E^OP-MP3(SS)(x_ss) = Σ_σ Tr P(W^SS(2)_σσ, x_ss W^SS(3)_σσ).
- Eq. 20: E^OP-MP3(OS)(x_os) is the same sum over the OS blocks.
- Eq. 21 (SOP-MP3): E = Eq. 19 + Eq. 20. Each spin block and each channel is
  resummed separately. OP-MP3 instead resums the sum SS + OS of each spin
  block. The default is x_ss = x_os = 1.
- Eq. 22 (SSD-OPMP3): E_c = E^OP-MP3(SS)(x_ss) + E^OP-MP3(OS)(1.00).
  **Same-spin damping** scales only the third-order same-spin matrix,
  W^SS(3) → x_ss W^SS(3), inside the SS Padé. W^(2) is never scaled, so the
  second-order (dispersion) asymptote is kept. The OS channel is resummed
  unscaled.
- Size consistency follows from Eq. 12. Matrix products, inverses and traces
  of block-diagonal matrices are the sums of the blocks. The model is
  invariant to occ-occ and vir-vir rotations.
- Eqs. 23-25 give a heuristic justification for x_ss: an approximate W^SS(4)
  for which the [2/3] Padé of the SS channel collapses to P(W2, x_ss W3). The
  paper never evaluates W^(4).

Total energy (Computational methods): E = E_HF[ρ_ref] + E_NBS + E_c^model.
E_NBS is the MP2-level non-Brillouin singles energy, added without
resummation or scaling.

### Zero blocks

A one-electron spin channel (the SS block of H, or the β SS block of any
doublet with one β electron) is zero. W matrices also carry frozen-core
orbitals as zero rows and columns (Zenodo README §6). P(0, 0) is taken as 0.
The prototype drops the rows and columns where W^(2) vanishes before the
solve. The paper's `analysis_code/resummation.py` does the same thing a
different way. It evaluates Tr P = Σ_i c_i / (1 − x ζ_i), where ζ are the
eigenvalues of pinvh(W2) W3, and gives a sector with Tr W2 = 0 the value 0.
The scalar Padé of a one-electron system (E2 = 0) is also 0.

## 5. Reference determinant and parameters

- References: κOOMP2 (κ = 1.45, unrestricted), or RO-HF / RO-DFT (HF, SPW92,
  PBE, PBE0, r2SCAN, ωB97X-V, ωB97M-V). From the reference density they build
  the UHF Fock matrix and diagonalize its oo and vv blocks separately
  ("semicanonicalization"). The MP denominators use that Fock diagonal, so DFT
  orbital energies never enter. F_ia enters only through
  E_NBS = Σ_ia |F_ia|² / (ε_i − ε_a), over both spins. No third-order singles
  terms are included.
- x_ss for SSD-OPMP3 (Fig. 3; W4-17 nonMR TAE RMSE fit, rounded to 0.05):
  κOOMP2 0.45 (unrounded fit 0.4279), PBE0 0.50, ωB97X-V 0.50 (fit 0.4702).
  HF uses 0 (fit −0.1643, clipped to 0). The other references are in the SI.
- The other fitted values (Fig. 2, κOOMP2): MP2.X x = 0.8365; spin-scaled MP3
  without resummation x_ss = 0.6499, x_os = 0.9876; OP-MP3(x) x = 0.7890;
  SOP-MP3 x_ss = 0.3809, x_os = 1.0253.
- Basis: aug-cc-pCVTZ, all electrons correlated, exact integrals. H and He take
  aug-cc-pVTZ. aug-cc-pwCVTZ is used for K, Ca and the transition metals.
  S22 is the exception: aug-cc-pVTZ, frozen core, aug-cc-pVQZ-RIFIT density
  fitting, counterpoise corrected. The density fitting covers only the
  amplitude and W stage.
- CCSD(T) benchmark: on HF orbitals, in the same basis.

## 6. The paper's own consistency checks

1. Fig. 1(b): water dimer dissociation energy with n infinitely separated He
   atoms. The explicit supersystem calculations (markers) match curves built
   from isolated-fragment results for Eq. 14. Eq. 1 drifts, and the
   dissociation energy eventually turns negative.
2. Eq. 3 / text: scalar Padé is size-extensive for identical copies only.
3. Eq. 12 and text: block diagonality, and invariance of Eq. 14 to occ-occ and
   vir-vir rotations.
4. Eq. 11b: Tr W^(n) = E^(n).
5. Eq. 18 remark: W_{iσ jσ̄} = 0, so the spin blocks never couple.
6. Zenodo `wmatrix_code` tests (2253): E_HF and E_NBS, Tr W^(2) to 1e-5 Eh and
   Tr W^(3) to 1e-4 Eh against Q-Chem MP2/MP3 (cc-pVDZ; H, He, H2O, CN, CH2,
   CH3, HCO; HF and PBE0; U, RO and closed shell; frozen core, DF, field).
   Each closed-shell run is also checked against its unrestricted counterpart.
7. Fit transparency: rounded x_ss vs the unrounded fit (RMSE change
   < 0.01 kcal/mol). They also correct the CCSD(T) reference by post-CCSD(T)
   increments: RMSE 1.33, MAE 0.95 kcal/mol.

## 7. Prototype construction choices (and why)

- The prototype uses ferric RHF/UHF references only. For HF, the
  semicanonicalization is the identity and E_NBS = 0. κOOMP2 and RO-DFT
  references are reproduced from the paper's stored W matrices, not by
  ferric.
- Integrals are exact (`AoIntegrals.ao2mo`). RI (`df_mo`) is used only for the
  trivial-limit comparison with ferric's RI `run_mp3` / `run_rimp2`, which
  need an auxiliary basis.
- The spin-orbital form is the literal form of Eqs. 7, 9 and 11a. Its only
  rewrite is the spatial ladder, and the MP3 anchors check that piece.
