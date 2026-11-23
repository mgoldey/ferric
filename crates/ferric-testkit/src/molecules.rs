//! Standard molecules, loaded from `testdata/molecules` so a test and the
//! reference generator read the same coordinates.

use ferric_core::mol::Molecule;

use crate::paths::testdata;

/// `testdata/molecules/water.xyz`, neutral singlet.
pub fn water_xyz() -> Molecule {
    Molecule::load_xyz(
        testdata("molecules/water.xyz")
            .to_str()
            .expect("utf-8 path"),
    )
    .expect("water.xyz")
}
