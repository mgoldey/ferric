//! Reader for the packed bundled basis sets written by `build.rs`.
//!
//! The blob is deflated JSON holding a pool of distinct BSE shells and ECPs
//! plus, per bundled file stem, each element's list of pool indices. It is
//! inflated, parsed and converted to ferric [`Shell`]s once per process, on
//! the first [`super::bundled`] call; later calls only clone shells out of
//! the pool. The shell conversion is the same [`super::convert_bse_shell`]
//! that [`super::load_bse_json`] uses, so a bundled set and the same JSON
//! loaded from disk produce identical [`BasisSet`]s (tested).

use super::{convert_bse_shell, parse_ecp_block, BasisSet, BseEcpPotential, BseShell, Shell};
use crate::FerricError;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

static BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bundled_bases.bin"));

#[derive(Deserialize)]
struct PackedEcp {
    ecp_electrons: i32,
    ecp_potentials: Vec<BseEcpPotential>,
}

#[derive(Deserialize)]
struct PackedElement {
    /// Indices into the shell pool, in the source file's order.
    s: Vec<usize>,
    /// Index into the ECP pool, if the element carries an ECP.
    e: Option<usize>,
}

#[derive(Deserialize)]
struct PackedSet {
    elements: HashMap<String, PackedElement>,
}

#[derive(Deserialize)]
struct PackedFile {
    shells: Vec<BseShell>,
    ecps: Vec<PackedEcp>,
    sets: HashMap<String, PackedSet>,
}

struct Library {
    /// Pool shell i converted to ferric shells (an SP shell gives two).
    shells: Vec<Vec<Shell>>,
    ecps: Vec<PackedEcp>,
    sets: HashMap<String, PackedSet>,
}

fn library() -> Result<&'static Library, FerricError> {
    static LIB: OnceLock<Result<Library, String>> = OnceLock::new();
    LIB.get_or_init(|| load().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| FerricError::Basis(e.clone()))
}

fn load() -> Result<Library, FerricError> {
    let raw = miniz_oxide::inflate::decompress_to_vec(BLOB)
        .map_err(|e| FerricError::Basis(format!("bundled basis blob: inflate failed: {e:?}")))?;
    let file: PackedFile = serde_json::from_slice(&raw)
        .map_err(|e| FerricError::Basis(format!("bundled basis blob: {e}")))?;
    let shells = file
        .shells
        .iter()
        .map(|sh| {
            let mut out = Vec::with_capacity(1);
            convert_bse_shell(sh, &mut out)?;
            Ok(out)
        })
        .collect::<Result<Vec<_>, FerricError>>()?;
    Ok(Library {
        shells,
        ecps: file.ecps,
        sets: file.sets,
    })
}

/// Stems of every packed file (every bundled JSON but the ones build.rs skips).
#[cfg(test)]
pub(super) fn packed_stems() -> Vec<String> {
    let mut v: Vec<String> = library().unwrap().sets.keys().cloned().collect();
    v.sort();
    v
}

/// Build the [`BasisSet`] for bundled file `stem`, named `name`.
pub(super) fn basis_set(stem: &str, name: &str) -> Result<BasisSet, FerricError> {
    let lib = library()?;
    let set = lib.sets.get(stem).ok_or_else(|| {
        FerricError::Basis(format!(
            "bundled basis file {stem}.json is not in the packed library"
        ))
    })?;
    let corrupt = |what: &str, i: usize| {
        FerricError::Basis(format!(
            "bundled basis {stem}: {what} index {i} out of range"
        ))
    };
    let mut shells: HashMap<i32, Vec<Shell>> = HashMap::new();
    let mut ecps = HashMap::new();
    for (z_str, el) in &set.elements {
        let z: i32 = z_str
            .parse()
            .map_err(|e| FerricError::Basis(format!("bad element key {z_str:?}: {e}")))?;
        for &i in &el.s {
            let pooled = lib.shells.get(i).ok_or_else(|| corrupt("shell", i))?;
            shells.entry(z).or_default().extend(pooled.iter().cloned());
        }
        if let Some(i) = el.e {
            let ecp = lib.ecps.get(i).ok_or_else(|| corrupt("ECP", i))?;
            ecps.insert(
                z,
                parse_ecp_block(z, ecp.ecp_electrons, &ecp.ecp_potentials)?,
            );
        }
    }
    Ok(BasisSet {
        name: name.to_string(),
        shells,
        ecps,
    })
}
