//! `ferric-mm`: an explicit-parameter AMBER-form molecular-mechanics force
//! field.
//!
//! This crate **assigns no parameters of its own**. It is arithmetic only:
//! given a [`topology::MmTopology`] (charges, Lennard-Jones sigma/epsilon per
//! atom, and harmonic bond/angle/periodic-torsion lists, all supplied by the
//! caller), it computes the AMBER-form bonded + nonbonded energy and its
//! analytic gradient. Parameter *assignment* — mapping a PDB structure to a
//! real force field like `amber14-all.xml` — is a separate, external step
//! (see `tools/active_site/mm_topology.py::topology_from_openmm`, which reads
//! parameters out of an actual OpenMM `System` built with a real force
//! field).
//!
//! Everything here is atomic units internally (Hartree, Bohr, radians);
//! [`topology::MmTopology::from_amber_units`] converts once from the
//! kcal/mol/Å/degree convention most force fields are published in.
//!
//! Validated against OpenMM's Reference platform on toy topologies built from
//! the *same* explicit parameters (`tests/vs_openmm.rs`) — see the crate's
//! parent plan doc for the measured per-term agreement.

pub mod energy;
pub mod topology;
pub mod units;

pub use energy::{energy, gradient, qm_mm_lj_energy_gradient, MmEnergy};
pub use topology::{Angle, Bond, LjParams, MmTopology, Torsion};
