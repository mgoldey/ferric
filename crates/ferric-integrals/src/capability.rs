//! Per-integral-class capability checks against the linked libint2.
//!
//! libint2 generates each integral class (4-centre, 3-centre, 2-centre,
//! one-body) separately, per derivative order, up to its own angular-momentum
//! limit. An engine for a class the library did NOT generate can still be
//! constructed (libint2 falls back to its "default" task), and its first
//! compute then hits `assert(buildfnptrs_[...] && "null build function ptr")`
//! — which aborts the whole process, Python interpreter included, because
//! ferric's shim is compiled without `NDEBUG`. conda-forge's win-64 libint2
//! 2.13.1 has no 4-centre second-derivative class at all, so an analytic
//! Hessian aborted there. The same happens for a ket (orbital) shell of a
//! centre-dependent 3-centre class above libint2's default-class limit.
//!
//! [`require_libint_class`](crate::capability::require_libint_class)
//! decides, from the shim's compile-time table (`scf_libint_max_am`), whether
//! a class at a derivative order covers the angular momenta an engine will be
//! asked for, and returns a typed
//! [`FerricError::Libint`](ferric_core::FerricError::Libint) naming the class
//! and the libint2 build otherwise.
//! Every derivative [`crate::engine::Engine`] constructor calls it before
//! touching libint2, so no derivative engine that could assert is ever built.

use crate::ffi;
use ferric_core::FerricError;
use std::sync::RwLock;

/// Rows of the shim's capability table (`scf_libint_max_am`'s `cls`).
const N_ROWS: usize = 6;

/// The shim's capability table: `table[row][deriv]` is the largest angular
/// momentum the linked libint2 generated (`-1`: not generated). Rows: 0
/// 4-centre ERI, 1 3-centre bra (auxiliary) shell, 2 2-centre bra shell,
/// 3 one-body, 4 3-centre ket (orbital) shells, 5 2-centre ket shell.
pub type LibintAmTable = [[i32; 3]; N_ROWS];

/// Substring every refusal of [`require_libint_class`] carries; see
/// [`is_libint_class_unavailable`].
pub const LIBINT_CLASS_UNAVAILABLE: &str = "libint2 integral class unavailable";

static AM_OVERRIDE: RwLock<Option<LibintAmTable>> = RwLock::new(None);

/// An integral class libint2 generates on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LibintClass {
    /// 4-centre two-electron integrals `(μν|λσ)`.
    FourCentre,
    /// 3-centre `(P|μν)`: bra = auxiliary shell, ket = the orbital shells.
    ThreeCentre,
    /// 2-centre `(P|Q)` (both shells auxiliary).
    TwoCentre,
    /// One-body integrals (overlap, kinetic, nuclear attraction).
    OneBody,
}

impl LibintClass {
    /// `(bra row, ket row)` of the capability table.
    fn rows(self) -> (usize, usize) {
        match self {
            LibintClass::FourCentre => (0, 0),
            LibintClass::ThreeCentre => (1, 4),
            LibintClass::TwoCentre => (2, 5),
            LibintClass::OneBody => (3, 3),
        }
    }

    fn describe(self) -> &'static str {
        match self {
            LibintClass::FourCentre => "4-centre ERI",
            LibintClass::ThreeCentre => "3-centre ERI",
            LibintClass::TwoCentre => "2-centre ERI",
            LibintClass::OneBody => "one-body",
        }
    }
}

/// The capability table compiled into the shim (ignores any test override).
pub fn linked_libint_am_table() -> LibintAmTable {
    let mut t = [[-1; 3]; N_ROWS];
    for (row, entry) in t.iter_mut().enumerate() {
        for (deriv, am) in entry.iter_mut().enumerate() {
            // SAFETY: returns a compile-time constant; any argument is valid.
            *am = unsafe { ffi::scf_libint_max_am(row as i32, deriv as i32) };
        }
    }
    t
}

/// The table [`require_libint_class`] consults: the linked one, or the test
/// override when one is installed.
pub fn libint_am_table() -> LibintAmTable {
    let guard = AM_OVERRIDE.read().unwrap_or_else(|p| p.into_inner());
    guard.unwrap_or_else(linked_libint_am_table)
}

/// Test hook: replace the capability table [`require_libint_class`] consults
/// (`None` restores the linked library's). Process-wide, so tests that use it
/// must run in their own test binary and serialise. It only changes ferric's
/// decision; libint2 itself is untouched.
#[doc(hidden)]
pub fn set_libint_max_am_for_test(table: Option<LibintAmTable>) {
    *AM_OVERRIDE.write().unwrap_or_else(|p| p.into_inner()) = table;
}

fn overridden() -> bool {
    AM_OVERRIDE
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .is_some()
}

fn limit_text(limit: i32) -> String {
    if limit < 0 {
        "not generated at all".to_string()
    } else {
        format!("generated only up to l = {limit}")
    }
}

/// Refuse, with a [`FerricError::Libint`] carrying
/// [`LIBINT_CLASS_UNAVAILABLE`], unless the linked libint2 generated `class`
/// at derivative order `deriv` for bra angular momentum up to `bra_l` and ket
/// angular momentum up to `ket_l` (for 3-centre: bra = auxiliary basis, ket =
/// orbital basis; otherwise pass the basis' max `l` for both).
pub fn require_libint_class(
    class: LibintClass,
    deriv: usize,
    bra_l: i32,
    ket_l: i32,
) -> Result<(), FerricError> {
    let table = libint_am_table();
    let (bra_row, ket_row) = class.rows();
    let at = |row: usize| table[row].get(deriv).copied().unwrap_or(-1);
    let (bra_lim, ket_lim) = (at(bra_row), at(ket_row));
    let split = bra_row != ket_row;
    let (need, lim, which) = if bra_l > bra_lim {
        (bra_l, bra_lim, " (bra/auxiliary shell)")
    } else if ket_l > ket_lim {
        (ket_l, ket_lim, " (ket shells)")
    } else {
        return Ok(());
    };
    let which = if split { which } else { "" };
    let order = match deriv {
        0 => "underived".to_string(),
        1 => "first-derivative".to_string(),
        2 => "second-derivative".to_string(),
        d => format!("order-{d} derivative"),
    };
    let note = if overridden() {
        " [capability table overridden for testing]"
    } else {
        ""
    };
    Err(FerricError::Libint(format!(
        "{LIBINT_CLASS_UNAVAILABLE}: {order} {} integrals{which} are needed up to l = {need}, \
         but the linked libint2 {} ({} library) has that class {}{note}; see \
         ferric.build_info()['libint_max_am'], or link a libint2 generated with it",
        class.describe(),
        crate::libint_version(),
        crate::libint_linkage(),
        limit_text(lim),
    )))
}

/// Whether `e` (possibly wrapped in another error's message) is a refusal of
/// [`require_libint_class`].
pub fn is_libint_class_unavailable(e: &FerricError) -> bool {
    e.to_string().contains(LIBINT_CLASS_UNAVAILABLE)
}
