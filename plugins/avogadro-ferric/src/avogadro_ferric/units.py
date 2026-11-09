"""Unit conversions between ferric (Hartree, Bohr) and Avogadro (kJ/mol, Angstrom).

The Bohr radius and the Hartree are the CODATA 2010 values ferric itself uses
(crates/ferric-core/src/units.rs), so a length converted here lands exactly
where ferric puts the same Angstrom input.
"""

# ferric_core::units::BOHR_RADIUS_ANGSTROM
BOHR_TO_ANGSTROM = 0.52917721092
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# ferric_core::units::HARTREE_JOULE x N_A (CODATA 2010, 6.02214129e23) / 1000.
HARTREE_TO_KJ_PER_MOL = 4.35974434e-18 * 6.02214129e23 / 1000.0
HARTREE_TO_EV = 27.21138505  # CODATA 2010
HARTREE_TO_KCAL_PER_MOL = HARTREE_TO_KJ_PER_MOL / 4.184

# dE/dR: Hartree/Bohr -> kJ/mol/Angstrom.
HARTREE_PER_BOHR_TO_KJ_PER_MOL_ANGSTROM = HARTREE_TO_KJ_PER_MOL / BOHR_TO_ANGSTROM
