# References and citing

## Citing ferric

If you publish a number computed with ferric, cite three things:

1. **The software.** The repository carries a `CITATION.cff`, which GitHub
   turns into a "Cite this repository" entry:

   > Goldey, M. *ferric: a Rust-native quantum chemistry engine.*
   > <https://github.com/mgoldey/ferric>. Licensed MIT OR Apache-2.0.

   Give the commit hash you ran; the crates are at version 0.1.0 and there are
   no tagged releases yet.

2. **The libraries every calculation goes through.**
   - libint2 (all Gaussian integrals): E. F. Valeev, *Libint: a library for the
     evaluation of molecular integrals of many-body operators over Gaussian
     functions*, <https://github.com/evaleev/libint>. Cite the version you
     built against (ferric builds on libint 2.7+).
   - libxc (every DFT functional): S. Lehtola, C. Steigemann, M. J. T.
     Oliveira & M. A. L. Marques, *SoftwareX* **7**, 1 (2018),
     [doi:10.1016/j.softx.2017.11.002](https://doi.org/10.1016/j.softx.2017.11.002).
     Only needed if you ran a DFT functional or a DFT reference.

3. **The method papers for what you ran**, from the list below. Each method
   page also ends with a short *Cite* line naming the entries it relies on.

## Methods implemented in ferric

Entries are limited to methods the code implements, and most are the
references the source code itself cites.

### SCF, convergence and integrals

- Szabo & Ostlund, *Modern Quantum Chemistry* (Dover, 1996)
- Pulay, *Chem. Phys. Lett.* **73**, 393 (1980): DIIS
- Kudin, Scuseria & Cancès, *J. Chem. Phys.* **116**, 8255 (2002): EDIIS
- Gilbert, Besley & Gill, *J. Phys. Chem. A* **112**, 13164 (2008): maximum overlap method (MOM)
- Ochsenfeld, White & Head-Gordon, *J. Chem. Phys.* **109**, 1663 (1998): LinK exchange
- Maurer, Lambrecht & Ochsenfeld, *J. Chem. Phys.* **136**, 144107 (2012): QQR screening
- Thompson & Ochsenfeld, *J. Chem. Phys.* **147**, 144101 (2017): CSB and CSAM integral screening
- White & Head-Gordon, *J. Chem. Phys.* **101**, 6593 (1994): continuous fast multipole method
- Neese, Wennmohs, Hansen & Becker, *Chem. Phys.* **356**, 98 (2009): COSX seminumerical exchange
- Izsák & Neese, *J. Chem. Phys.* **135**, 144105 (2011): COSX overlap fitting
- Weigend, *Phys. Chem. Chem. Phys.* **4**, 4285 (2002): RI-JK (fully direct RI-HF) and its auxiliary sets
- Dunlap, *J. Mol. Struct. (THEOCHEM)* **529**, 37 (2000): robust density fitting

### DFT, grids and dispersion corrections

- Becke, *J. Chem. Phys.* **88**, 2547 (1988): multicentre integration (Becke partitioning)
- Treutler & Ahlrichs, *J. Chem. Phys.* **102**, 346 (1995): radial grids (M4 mapping)
- Lebedev & Laikov, *Dokl. Math.* **59**, 477 (1999): angular grids
- Perdew, Burke & Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996): PBE
- Becke, *J. Chem. Phys.* **98**, 5648 (1993); Stephens, Devlin, Chabalowski & Frisch, *J. Phys. Chem.* **98**, 11623 (1994): B3LYP
- Vydrov & Van Voorhis, *J. Chem. Phys.* **133**, 244103 (2010): VV10 nonlocal correlation
- Mardirossian & Head-Gordon, *Phys. Chem. Chem. Phys.* **16**, 9904 (2014): ωB97X-V
- Sun, Ruzsinszky & Perdew, *Phys. Rev. Lett.* **115**, 036402 (2015): SCAN
- Furness, Kaplan, Ning, Perdew & Sun, *J. Phys. Chem. Lett.* **11**, 8208 (2020): r2SCAN
- Grimme, Antony, Ehrlich & Krieg, *J. Chem. Phys.* **132**, 154104 (2010): DFT-D3
- Grimme, Ehrlich & Goerigk, *J. Comput. Chem.* **32**, 1456 (2011): Becke–Johnson damping for D3
- Tkatchenko & Scheffler, *Phys. Rev. Lett.* **102**, 073005 (2009): TS dispersion and free-atom reference data
- Gould & Bučko, *J. Chem. Theory Comput.* **12**, 3603 (2016): free-atom reference data for Z = 19–54
- Tkatchenko, DiStasio, Car & Scheffler, *Phys. Rev. Lett.* **108**, 236402 (2012): many-body dispersion (MBD)

### Solvation and embedding

- Cancès, Mennucci & Tomasi, *J. Chem. Phys.* **107**, 3032 (1997): IEF-PCM
- Klamt & Schüürmann, *J. Chem. Soc., Perkin Trans. 2*, 799 (1993): COSMO
- Lin & Truhlar, *J. Phys. Chem. A* **109**, 3991 (2005): redistributed-charge QM/MM boundary schemes

### Geometry, frequencies and reaction paths

- Pulay & Fogarasi, *J. Chem. Phys.* **96**, 2856 (1992): redundant internal coordinates
- Banerjee, Adams, Simons & Shepard, *J. Phys. Chem.* **89**, 52 (1985): P-RFO saddle search
- Bofill, *J. Comput. Chem.* **15**, 1 (1994): Hessian update for saddle searches

### Charges and properties

- Breneman & Wiberg, *J. Comput. Chem.* **11**, 361 (1990): CHELPG
- Bayly, Cieplak, Cornell & Kollman, *J. Phys. Chem.* **97**, 10269 (1993): RESP
- Foster & Boys, *Rev. Mod. Phys.* **32**, 300 (1960): Boys localization

### MP2 family

- Weigend, Häser, Patzelt & Ahlrichs, *Chem. Phys. Lett.* **294**, 143 (1998): RI-MP2 auxiliary basis sets
- Grimme, *J. Chem. Phys.* **118**, 9095 (2003): SCS-MP2
- Jung, Lochan, Dutoi & Head-Gordon, *J. Chem. Phys.* **121**, 9793 (2004): SOS-MP2
- Häser & Almlöf, *J. Chem. Phys.* **96**, 489 (1992): Laplace-transform MP2
- Takatsuka, Ten-no & Hackbusch, *J. Chem. Phys.* **129**, 044112 (2008): minimax Laplace quadrature
- Lochan & Head-Gordon, *J. Chem. Phys.* **126**, 164101 (2007): orbital-optimized (opposite-spin) MP2
- Bozkaya, Turney, Yamaguchi, Schaefer & Sherrill, *J. Chem. Phys.* **135**, 104103 (2011): orbital-optimized MP2 algorithm
- Lee & Head-Gordon, *J. Chem. Theory Comput.* **14**, 5203 (2018): κ-regularized MP2
- Dutoi & Head-Gordon, *J. Phys. Chem. A* **112**, 2110 (2008): the terfc attenuator
- Goldey & Head-Gordon, *J. Phys. Chem. Lett.* **3**, 3592 (2012): attenuated MP2 (erfc, aug-cc-pVDZ)
- Goldey, Dutoi & Head-Gordon, *Phys. Chem. Chem. Phys.* **15**, 15869 (2013): attenuated MP2 in aug-cc-pVTZ (terfc)
- Goldey & Head-Gordon, *J. Phys. Chem. B* **118**, 6519 (2014): SCS-MP2(2terfc), separate attenuation of the two spin components
- Goldey, Belzunces & Head-Gordon, *J. Chem. Theory Comput.* **11**, 4159 (2015): MP2-V
- Wang, Aldossary, Shi, Liu, Li & Head-Gordon, *J. Chem. Theory Comput.* **19**, 7577 (2023): single-threshold local MP2 (the "WSHG23" scheme in the code)

### Coupled cluster

- Scuseria, Janssen & Schaefer, *J. Chem. Phys.* **89**, 7382 (1988): CCSD
- Hirata, Podeszwa, Tobita & Bartlett, *J. Chem. Phys.* **120**, 2581 (2004): spin-adapted closed-shell CCSD equations
- Raghavachari, Trucks, Pople & Head-Gordon, *Chem. Phys. Lett.* **157**, 479 (1989): CCSD(T)
- Rendell, Lee & Komornicki, *Chem. Phys. Lett.* **178**, 462 (1991): closed-shell (T) algorithm
- Carter-Fenk, *J. Phys. Chem. A* **129**, 7251 (2025): LinLCCD(hh)
- Ransford & Carter-Fenk, *Phys. Chem. Chem. Phys.* **28**, 14428 (2026): ωB97X-L-V
- Bartlett & Musiał, *Rev. Mod. Phys.* **79**, 291 (2007): coupled-cluster theory (review)

### Double hybrids

- Grimme, *J. Chem. Phys.* **124**, 034108 (2006): B2PLYP
- Kozuch & Martin, *Phys. Chem. Chem. Phys.* **13**, 20104 (2011): DSD-PBEP86

### RPA, GW and excited states

- Wilson, Gygi & Galli, *Phys. Rev. B* **78**, 113303 (2008): iterative dielectric eigenpotentials (PDEP)
- Eshuis, Yarkony & Furche, *J. Chem. Phys.* **132**, 234114 (2010): RI-RPA and its frequency quadrature
- Kaltak, Klimeš & Kresse, *J. Chem. Theory Comput.* **10**, 2498 (2014): minimax imaginary-frequency grids
- Hedin, *Phys. Rev.* **139**, A796 (1965): the GW approximation
- van Setten et al., *J. Chem. Theory Comput.* **11**, 5665 (2015): GW100 benchmark (the MOLGW reference values)
- Dreuw & Head-Gordon, *Chem. Rev.* **105**, 4009 (2005): TDDFT and CIS (review)

### Constrained DFT

- Wu & Van Voorhis, *J. Chem. Phys.* **125**, 164105 (2006): electron-transfer couplings from constrained DFT

### Background on MP2's dispersion error

Not implemented in ferric; cited on [Electronic response](../idea/response.md):

- Cybulski & Lytle, *J. Chem. Phys.* **127**, 141102 (2007)
- Heßelmann, *J. Chem. Phys.* **128**, 144112 (2008)
- Pitoňák & Heßelmann, *J. Chem. Theory Comput.* **6**, 168 (2010)

## Software dependencies

- [libint2](https://github.com/evaleev/libint): Gaussian integral engine
- [libxc](https://libxc.gitlab.io/): exchange–correlation functionals
- [pyo3](https://pyo3.rs/): Rust/Python interop
- [ndarray](https://docs.rs/ndarray) and [ndarray-linalg](https://docs.rs/ndarray-linalg): arrays and LAPACK bindings
- D3 reference tables are generated from [simple-dftd3](https://github.com/dftd3/simple-dftd3) (LGPL-3.0-or-later)

## License

Dual-licensed under either

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/mgoldey/ferric/blob/main/LICENSE-APACHE))
- MIT License ([LICENSE-MIT](https://github.com/mgoldey/ferric/blob/main/LICENSE-MIT))

at your option.
