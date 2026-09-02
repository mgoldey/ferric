# References

## Dependencies

- [libint2](https://github.com/evaleev/libint) — Obara–Saika integral engine
- [pyo3](https://pyo3.rs/) — Rust/Python interop
- [ndarray](https://docs.rs/ndarray) — N-dimensional arrays for Rust
- [ndarray-linalg](https://docs.rs/ndarray-linalg) — LAPACK bindings for ndarray
- [libxc](https://libxc.gitlab.io/) — exchange–correlation functionals

## Methods

**General**

- Szabo & Ostlund, *Modern Quantum Chemistry* (1996)
- Pulay, *Chem. Phys. Lett.* **73**, 393 (1980) — DIIS convergence acceleration

**MP2 family**

- Weigend, *Phys. Chem. Chem. Phys.* **4**, 4285 (2002) — RI-MP2 auxiliary basis sets
- Grimme, *J. Chem. Phys.* **118**, 9095 (2003) — SCS-MP2
- Bozkaya & Sherrill, *J. Chem. Phys.* **135**, 104103 (2011) — orbital-optimized MP2
- Goldey & Head-Gordon, *J. Phys. Chem. Lett.* **3**, 3592 (2012) — attenuated MP2
- Goldey, Dutoi & Head-Gordon, *Phys. Chem. Chem. Phys.* **15**, 15869 (2013) — SCS-MP2(2terfc)

**Coupled cluster**

- Scuseria, Janssen & Schaefer, *J. Chem. Phys.* **89**, 7382 (1988) — CCSD
- Raghavachari et al., *Chem. Phys. Lett.* **157**, 479 (1989) — CCSD(T) triples
- Bartlett & Musiał, *Rev. Mod. Phys.* **79**, 291 (2007) — coupled-cluster theory

**Screening and scaling**

- Ochsenfeld, White & Head-Gordon, *J. Chem. Phys.* **109**, 1663 (1998) — LinK exchange
- Maurer, Lambrecht & Ochsenfeld, *J. Chem. Phys.* **136**, 144107 (2012) — QQR screening

**Constrained DFT and nonlocal correlation**

- Wu & Van Voorhis, *J. Chem. Phys.* **125**, 164105 (2006) — cDFT electron-transfer coupling \\( H_{ab} \\)
- Vydrov & Van Voorhis, *J. Chem. Phys.* **133**, 244103 (2010) — VV10 nonlocal correlation

## License

Dual-licensed under either

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/mgoldey/ferric/blob/main/LICENSE-APACHE))
- MIT License ([LICENSE-MIT](https://github.com/mgoldey/ferric/blob/main/LICENSE-MIT))

at your option.
