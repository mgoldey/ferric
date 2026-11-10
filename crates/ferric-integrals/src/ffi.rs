//! Raw FFI declarations for the libint2 C++ shim (`shim/shim.cc`).
//!
//! These are unsafe C-linkage functions. Prefer the safe wrappers in
//! [`crate::engine`] and [`crate::basis_bridge`].

use std::os::raw::{c_char, c_double, c_int, c_void};

/// C-compatible shell descriptor passed to the libint2 shim.
#[repr(C)]
pub struct CShell {
    /// Angular momentum of the shell.
    pub l: c_int,
    /// Number of primitives: the length of `exponents` and `coefficients`.
    pub nprim: c_int,
    /// Index into the [`CAtom`] array passed to `scf_basis_create`; the shell is
    /// centred on that atom. The shim does not bounds-check it.
    pub atom_index: c_int,
    /// 1 for spherical-harmonic (pure) functions, 0 for Cartesian.
    pub pure: c_int,
    /// Primitive exponents, `nprim` doubles. Read (copied) only during
    /// `scf_basis_create`.
    pub exponents: *const c_double,
    /// Contraction coefficients, `nprim` doubles, copied during
    /// `scf_basis_create`; libint2's `Shell` constructor normalizes them.
    pub coefficients: *const c_double,
}

impl std::fmt::Debug for CShell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CShell")
            .field("l", &self.l)
            .field("nprim", &self.nprim)
            .field("atom_index", &self.atom_index)
            .field("pure", &self.pure)
            .finish_non_exhaustive()
    }
}

/// C-compatible atom descriptor (atomic number + Cartesian position in Bohr).
///
/// `atomic_number` is `f64` (not an integer type) so that external point
/// charges (e.g. QM/MM partial charges) can carry a fractional charge;
/// libint2's own `Engine::set_params` already takes `double` internally.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct CAtom {
    /// Nuclear or point charge. `scf_basis_create` truncates it to an integer
    /// (only the position matters there); `scf_engine_set_point_charges` uses
    /// the value as given.
    pub atomic_number: c_double,
    /// Cartesian x coordinate, Bohr.
    pub x: c_double,
    /// Cartesian y coordinate, Bohr.
    pub y: c_double,
    /// Cartesian z coordinate, Bohr.
    pub z: c_double,
}

extern "C" {
    /// Reference-counted `libint2::initialize()`: the first call initializes
    /// libint2, later calls only increment the count. Must run before any engine
    /// is created.
    pub fn scf_libint_init();
    /// Decrements the [`scf_libint_init`] count and calls `libint2::finalize()`
    /// when it reaches zero.
    pub fn scf_libint_finalize();
    /// Static, NUL-terminated `LIBINT_VERSION` of the compiled-against
    /// headers, or `"unknown"`. Never NULL.
    pub fn scf_libint_version() -> *const c_char;
    /// Boys function `F_0..F_mmax(x)` through libint2's `FmEval_Chebyshev7`;
    /// `out` holds `mmax + 1` doubles. Returns `mmax + 1` or a negative status.
    pub fn scf_boys_fm(x: f64, mmax: c_int, out: *mut f64) -> c_int;
    /// Yukawa (`kind` 0) / Slater-geminal (`kind` 1) core integrals through
    /// libint2's `TennoGmEval`; `out` holds `mmax + 1` doubles. Returns
    /// `mmax + 1` or a negative status.
    pub fn scf_tenno_gm(
        kind: c_int,
        one_over_rho: f64,
        t: f64,
        mmax: c_int,
        zeta: f64,
        out: *mut f64,
    ) -> c_int;
    /// Builds an opaque libint2 basis from `nshells` [`CShell`]s centred on
    /// `natoms` [`CAtom`]s (both arrays are copied during the call). Returns NULL
    /// if libint2 throws. Free with [`scf_basis_destroy`].
    pub fn scf_basis_create(
        shells: *const CShell,
        nshells: c_int,
        atoms: *const CAtom,
        natoms: c_int,
    ) -> *mut c_void;
    /// Frees a basis returned by [`scf_basis_create`].
    pub fn scf_basis_destroy(bs: *mut c_void);
    /// Total number of basis functions in the basis.
    pub fn scf_basis_nbasis(bs: *const c_void) -> c_int;
    /// Number of shells in the basis.
    pub fn scf_basis_nshells(bs: *const c_void) -> c_int;
    /// Writes each shell's function count (`2l+1` pure, `(l+1)(l+2)/2`
    /// Cartesian) into `out`, which must hold [`scf_basis_nshells`] ints.
    pub fn scf_basis_shell_dims(bs: *const c_void, out: *mut c_int);
    /// Writes the largest primitive count and the largest angular momentum over all shells.
    pub fn scf_basis_max_dims(bs: *const c_void, max_nprim: *mut c_int, max_l: *mut c_int);
    /// Creates a derivative-order-0 libint2 engine for `op_kind`: one of
    /// [`OP_COULOMB`], [`OP_ERF_COULOMB`], [`OP_ERFC_COULOMB`], [`OP_YUKAWA`],
    /// [`OP_SLATER_GEMINAL`] (two-electron, braket `xx_xx`) or [`OP_OVERLAP`],
    /// [`OP_KINETIC`], [`OP_NUCLEAR`] (one-electron). For kinds 1-4 `omega` is
    /// passed to `Engine::set_params` (ω for erf/erfc, ζ for Yukawa / Slater
    /// geminal); otherwise it is ignored. `max_nprim` / `max_l` bound the shells
    /// the engine accepts and `precision` is libint2's screening precision.
    /// Returns NULL for any other `op_kind` or if libint2 throws. Free with
    /// [`scf_engine_destroy`].
    pub fn scf_engine_create(
        op_kind: c_int,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
    ) -> *mut c_void;
    /// Frees an engine returned by any `scf_engine_create*` function.
    pub fn scf_engine_destroy(eng: *mut c_void);
    /// Replaces the point charges of a nuclear-attraction engine with `natoms`
    /// [`CAtom`]s (charge = `atomic_number`, position in Bohr; copied). Returns
    /// `SCF_OK` (0) or [`SCF_EINTERNAL`].
    pub fn scf_engine_set_point_charges(
        eng: *mut c_void,
        atoms: *const CAtom,
        natoms: c_int,
    ) -> c_int;
    /// One-electron block ⟨sh1|op|sh2⟩ for a 1e engine: writes `n1*n2` doubles
    /// row-major (sh1 index slowest) into `out`, all zeros if libint2 screened
    /// the pair. Returns `n1*n2` or [`SCF_EINTERNAL`]. Shell indices are not
    /// bounds-checked.
    pub fn scf_compute_1e_block(
        eng: *mut c_void,
        bs: *const c_void,
        sh1: c_int,
        sh2: c_int,
        out: *mut c_double,
    ) -> c_int;
    /// `scf_compute_1e_block` with shell `sh2` translated by `shift` (3
    /// doubles, Bohr). Returns n1*n2, `SCF_EINVAL` (-1) or `SCF_EINTERNAL` (-3).
    pub fn scf_compute_1e_block_shifted(
        eng: *mut c_void,
        bs: *const c_void,
        sh1: c_int,
        sh2: c_int,
        shift: *const c_double,
        out: *mut c_double,
    ) -> c_int;
    /// Two-electron quartet (sh1 sh2|sh3 sh4) from one basis: writes
    /// `n1*n2*n3*n4` doubles row-major (sh1 slowest, sh4 fastest). Returns that
    /// count, 0 if libint2 screened the quartet (nothing written), or
    /// [`SCF_EINTERNAL`]. Uses the engine's shell-pair cache when enabled (see
    /// [`scf_engine_set_shellpair_cache_enabled`]).
    pub fn scf_compute_eri_quartet(
        eng: *mut c_void,
        bs: *const c_void,
        sh1: c_int,
        sh2: c_int,
        sh3: c_int,
        sh4: c_int,
        out: *mut c_double,
    ) -> c_int;
    /// Programmatic override of the per-engine shell-pair cache used by
    /// `scf_compute_eri_quartet`, independent of the `FERRIC_SHELLPAIR_CACHE`
    /// env var (so tests can A/B within one process without mutating global
    /// env state). `enabled != 0` enables.
    pub fn scf_engine_set_shellpair_cache_enabled(eng: *mut c_void, enabled: c_int);
    /// Returns 1 if the shell-pair cache is (or would be, if unset) enabled
    /// for this engine, 0 otherwise. Diagnostic/test use only.
    pub fn scf_engine_shellpair_cache_enabled(eng: *const c_void) -> c_int;
    /// Fills `qmat` (row-major `nshells x nshells`, symmetric) with
    /// `Q[i,j] = sqrt(max_ab |(ab|ab)|)` over functions `a` in shell `i`, `b` in
    /// shell `j`, using the engine's operator. Returns `SCF_OK` (0) or
    /// [`SCF_EINTERNAL`].
    pub fn scf_compute_schwarz(eng: *mut c_void, bs: *const c_void, qmat: *mut c_double) -> c_int;
    /// As [`scf_engine_create`] with derivative order 1. Also returns NULL
    /// when libint2 was built without first derivatives.
    pub fn scf_engine_create_deriv(
        op_kind: c_int,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
    ) -> *mut c_void;
    /// Creates an F12 geminal engine for [`OP_CGTG`], [`OP_CGTG_X_COULOMB`] or
    /// [`OP_DELCGTG2`]; the geminal is the `ngauss` (exponent, coefficient) pairs
    /// in `exps` / `coefs` (copied). Returns NULL when libint2 lacks the G12
    /// integral class, for any other `op_kind`, or if libint2 throws (message on
    /// stderr).
    pub fn scf_engine_create_geminal(
        op_kind: c_int,
        ngauss: c_int,
        exps: *const c_double,
        coefs: *const c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
    ) -> *mut c_void;
    /// First derivatives of a 1e block, for an engine from
    /// [`scf_engine_create_deriv`]: writes `nderiv` blocks of `n1*n2` doubles,
    /// where `nderiv` is libint2's result count: 6 for overlap/kinetic (x, y, z
    /// of sh1, then of sh2) and `3*(2 + ncharges)` for nuclear (followed by x, y,
    /// z of each point charge in [`scf_engine_set_point_charges`] order). The
    /// shim does not check the size of `out`; the caller must size it for the
    /// nuclear case. Returns `nderiv*n1*n2`, 0 if screened (zeros written) or
    /// without derivative support (nothing written), or [`SCF_EINTERNAL`].
    pub fn scf_compute_1e_deriv_block(
        eng: *mut c_void,
        bs: *const c_void,
        sh1: c_int,
        sh2: c_int,
        out: *mut c_double,
    ) -> c_int;
    /// `scf_compute_1e_deriv_block` with shell `sh2` translated by `shift`
    /// (3 doubles, Bohr). `out_len` = capacity of `out` in doubles (checked by
    /// the shim before writing). Returns nderiv*n1*n2, 0 if screened,
    /// `SCF_EINVAL` (-1) or `SCF_EINTERNAL` (-3).
    pub fn scf_compute_1e_deriv_block_shifted(
        eng: *mut c_void,
        bs: *const c_void,
        sh1: c_int,
        sh2: c_int,
        shift: *const c_double,
        out: *mut c_double,
        out_len: c_int,
    ) -> c_int;
    /// First derivatives of (sh1 sh2|sh3 sh4), for an engine from
    /// [`scf_engine_create_deriv`]: 12 blocks (x, y, z of sh1, sh2, sh3, sh4 in
    /// that order) of `n1*n2*n3*n4` doubles each. Returns `12*n1*n2*n3*n4`, 0 if
    /// screened or without derivative support (nothing written), or
    /// [`SCF_EINTERNAL`].
    pub fn scf_compute_eri_deriv_quartet(
        eng: *mut c_void,
        bs: *const c_void,
        sh1: c_int,
        sh2: c_int,
        sh3: c_int,
        sh4: c_int,
        out: *mut c_double,
    ) -> c_int;
    /// `LIBINT2_MAX_DERIV_ORDER` of the libint2 the shim was compiled against.
    pub fn scf_libint_max_deriv_order() -> c_int;
    /// Max AM of the linked libint2 for integral class `cls` (0 eri, 1 eri3
    /// bra/aux, 2 eri2 bra, 3 one-body, 4 eri3 ket/orbital, 5 eri2 ket) at
    /// derivative order `deriv` (0..=2); -1 when the build does not define
    /// it, did not generate the class, or the arguments are out of range.
    pub fn scf_libint_max_am(cls: c_int, deriv: c_int) -> c_int;
    /// `LIBINT2_MAX_AM_R12kG12` of the linked libint2, or -1.
    pub fn scf_libint_g12_max_am() -> c_int;
    /// deriv_order=2 engine; NULL without second-derivative support or when
    /// `max_l` exceeds the linked library's second-derivative AM limit.
    pub fn scf_engine_create_deriv2(
        op_kind: c_int,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
    ) -> *mut c_void;
    /// Unique second-derivative blocks of a 1e shell pair (see shim.h for the
    /// ordering). `out_len` is the capacity of `out` in doubles.
    pub fn scf_compute_1e_deriv2_block(
        eng: *mut c_void,
        bs: *const c_void,
        sh1: c_int,
        sh2: c_int,
        out: *mut c_double,
        out_len: c_int,
    ) -> c_int;
    /// The 78 unique second-derivative blocks of a 2e shell quartet.
    #[allow(clippy::too_many_arguments)]
    pub fn scf_compute_eri_deriv2_quartet(
        eng: *mut c_void,
        bs: *const c_void,
        sh1: c_int,
        sh2: c_int,
        sh3: c_int,
        sh4: c_int,
        out: *mut c_double,
        out_len: c_int,
    ) -> c_int;
    /// As [`scf_engine_create`] with braket `xs_xx`, for 3-centre (P|μν)
    /// integrals. Returns NULL without libint2 3-centre ERI support.
    pub fn scf_engine_create_3center(
        op_kind: c_int,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
    ) -> *mut c_void;
    /// As [`scf_engine_create`] with braket `xs_xs`, for 2-centre (P|Q)
    /// integrals. Returns NULL without libint2 2-centre ERI support.
    pub fn scf_engine_create_2center(
        op_kind: c_int,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
    ) -> *mut c_void;
    /// 3-centre block (shP|sh1 sh2) with `shP` from the auxiliary basis `dfbs`
    /// and `sh1`, `sh2` from `obs`: writes `nP*n1*n2` doubles row-major (P
    /// slowest). Returns that count, 0 if screened or without 3-centre support
    /// (nothing written), or [`SCF_EINTERNAL`].
    pub fn scf_compute_eri3(
        eng: *mut c_void,
        obs: *const c_void,
        dfbs: *const c_void,
        shP: c_int,
        sh1: c_int,
        sh2: c_int,
        out: *mut c_double,
    ) -> c_int;
    /// `scf_compute_eri3` with the shells translated by `shifts` =
    /// `[sP; s1; s2]` (9 doubles, Bohr). Returns nP*n1*n2, 0 if screened,
    /// `SCF_EINVAL` (-1) or `SCF_EINTERNAL` (-3).
    pub fn scf_compute_eri3_shifted(
        eng: *mut c_void,
        obs: *const c_void,
        dfbs: *const c_void,
        shP: c_int,
        sh1: c_int,
        sh2: c_int,
        shifts: *const c_double,
        out: *mut c_double,
    ) -> c_int;
    /// 2-centre block (shP|shQ), both shells from `dfbs`: writes `nP*nQ`
    /// doubles row-major, zeros if screened. Returns `nP*nQ`, 0 without
    /// 2-centre support, or [`SCF_EINTERNAL`].
    pub fn scf_compute_eri2(
        eng: *mut c_void,
        dfbs: *const c_void,
        shP: c_int,
        shQ: c_int,
        out: *mut c_double,
    ) -> c_int;
    /// `scf_compute_eri2` with the ket shell translated by `shift_q`
    /// (3 doubles, Bohr): `(P | Q(r − s_Q))`. Returns nP*nQ (zeros written if
    /// screened), `SCF_EINVAL` (-1) or `SCF_EINTERNAL` (-3).
    pub fn scf_compute_eri2_shifted(
        eng: *mut c_void,
        dfbs: *const c_void,
        shP: c_int,
        shQ: c_int,
        shift_q: *const c_double,
        out: *mut c_double,
    ) -> c_int;
    /// As [`scf_engine_create_3center`] with derivative order 1. Returns NULL
    /// without 3-centre or first-derivative support.
    pub fn scf_engine_create_3center_deriv(
        op_kind: c_int,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
    ) -> *mut c_void;
    /// As [`scf_engine_create_2center`] with derivative order 1. Returns NULL
    /// without 2-centre or first-derivative support.
    pub fn scf_engine_create_2center_deriv(
        op_kind: c_int,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
    ) -> *mut c_void;
    /// First derivatives of (shP|sh1 sh2): `nderiv` blocks of `nP*n1*n2`
    /// doubles, laid out `[d/dP, d/d1, d/d2] x [x, y, z]` (9 blocks per
    /// `shim.h`; the shim copies libint2's full result count and does not check
    /// the size of `out`). Returns `nderiv*nP*n1*n2`, 0 if screened or without
    /// support (nothing written), or [`SCF_EINTERNAL`].
    pub fn scf_compute_eri3_deriv(
        eng: *mut c_void,
        obs: *const c_void,
        dfbs: *const c_void,
        shP: c_int,
        sh1: c_int,
        sh2: c_int,
        out: *mut c_double,
    ) -> c_int;
    /// `scf_compute_eri3_deriv` with the shells translated by `shifts` =
    /// `[sP; s1; s2]` (9 doubles, Bohr). `out_len` = capacity of `out` in
    /// doubles. Returns nderiv*nP*n1*n2, 0 if screened, `SCF_EINVAL` (-1) or
    /// `SCF_EINTERNAL` (-3).
    pub fn scf_compute_eri3_deriv_shifted(
        eng: *mut c_void,
        obs: *const c_void,
        dfbs: *const c_void,
        shP: c_int,
        sh1: c_int,
        sh2: c_int,
        shifts: *const c_double,
        out: *mut c_double,
        out_len: c_int,
    ) -> c_int;
    /// First derivatives of (shP|shQ): `nderiv` blocks of `nP*nQ` doubles,
    /// laid out `[d/dP, d/dQ] x [x, y, z]` (6 blocks per `shim.h`; the size of
    /// `out` is not checked). Returns `nderiv*nP*nQ`, 0 if screened or without
    /// support (nothing written), or [`SCF_EINTERNAL`].
    pub fn scf_compute_eri2_deriv(
        eng: *mut c_void,
        dfbs: *const c_void,
        shP: c_int,
        shQ: c_int,
        out: *mut c_double,
    ) -> c_int;
    /// `scf_compute_eri2_deriv` with the ket shell translated by `shift_q`
    /// (3 doubles, Bohr): derivatives of `(P | Q(r − s_Q))`, layout
    /// `[d/dP, d/dQ] × [x, y, z]`. `out_len` = capacity of `out` in doubles.
    /// Returns nderiv*nP*nQ, 0 if screened, `SCF_EINVAL` (-1) or
    /// `SCF_EINTERNAL` (-3).
    pub fn scf_compute_eri2_deriv_shifted(
        eng: *mut c_void,
        dfbs: *const c_void,
        shP: c_int,
        shQ: c_int,
        shift_q: *const c_double,
        out: *mut c_double,
        out_len: c_int,
    ) -> c_int;
    /// Creates an engine for exact terfc(r, r0)/r 3-centre integrals, evaluated
    /// by the shim's own McMurchie-Davidson code from the G_{m,n}(S,s)
    /// interpolation tables (Dutoi/Goldey). `table_dir` is a NUL-terminated
    /// directory, or null/empty to use `FERRIC_TERF_TABLE_DIR`; the loaded table
    /// set is cached per process. `max_nprim` is unused. `shim.h` states the
    /// curvature constraint `r0 * omega = 1/sqrt(2)`. Returns NULL if no table
    /// directory resolves, the tables fail to load, or on an exception.
    pub fn scf_engine_create_terfc_3center(
        r0: c_double,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
        table_dir: *const c_char,
    ) -> *mut c_void;
    /// Creates a terfc engine for the 2-centre (P|Q) metric; identical to
    /// [`scf_engine_create_terfc_3center`] (same tables and payload).
    pub fn scf_engine_create_terfc_2center(
        r0: c_double,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
        table_dir: *const c_char,
    ) -> *mut c_void;
    /// terfc 3-centre block (shP|sh1 sh2), `shP` from `dfbs`: writes `nP*n1*n2`
    /// doubles row-major (P slowest; spherical where a shell is pure), computed
    /// as Coulomb minus terf. Returns that count, 0 if fully screened (zeros written),
    /// [`SCF_EINVAL`] for a non-terfc engine, a null pointer or a total angular
    /// momentum beyond the tables, or [`SCF_EINTERNAL`].
    pub fn scf_compute_terfc_eri3(
        eng: *mut c_void,
        obs: *const c_void,
        dfbs: *const c_void,
        shP: c_int,
        sh1: c_int,
        sh2: c_int,
        out: *mut c_double,
    ) -> c_int;
    /// terfc 2-centre block (shP|shQ), both from `dfbs`: `nP*nQ` doubles
    /// row-major, computed as Coulomb minus terf. Returns `nP*nQ`,
    /// [`SCF_EINVAL`] (as for [`scf_compute_terfc_eri3`]) or [`SCF_EINTERNAL`].
    pub fn scf_compute_terfc_eri2(
        eng: *mut c_void,
        dfbs: *const c_void,
        shP: c_int,
        shQ: c_int,
        out: *mut c_double,
    ) -> c_int;
    /// Creates an engine for terf(r, r0)/r, the long-range complement of terfc
    /// (terf + terfc = 1/r), 3-centre; same tables and arguments as
    /// [`scf_engine_create_terfc_3center`].
    pub fn scf_engine_create_terf_3center(
        r0: c_double,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
        table_dir: *const c_char,
    ) -> *mut c_void;
    /// terf counterpart of [`scf_engine_create_terfc_2center`], for the 2-centre
    /// metric.
    pub fn scf_engine_create_terf_2center(
        r0: c_double,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
        table_dir: *const c_char,
    ) -> *mut c_void;
    /// terf 3-centre block (shP|sh1 sh2); same layout and return contract as
    /// [`scf_compute_terfc_eri3`]. Requires an engine from
    /// [`scf_engine_create_terf_3center`] (else [`SCF_EINVAL`]).
    pub fn scf_compute_terf_eri3(
        eng: *mut c_void,
        obs: *const c_void,
        dfbs: *const c_void,
        shP: c_int,
        sh1: c_int,
        sh2: c_int,
        out: *mut c_double,
    ) -> c_int;
    /// terf 2-centre block (shP|shQ); same layout as [`scf_compute_terfc_eri2`].
    /// Returns `nP*nQ`, 0 if fully screened (zeros written), [`SCF_EINVAL`] or
    /// [`SCF_EINTERNAL`].
    pub fn scf_compute_terf_eri2(
        eng: *mut c_void,
        dfbs: *const c_void,
        shP: c_int,
        shQ: c_int,
        out: *mut c_double,
    ) -> c_int;
    /// terfc 4-centre quartet (sh1 sh2|sh3 sh4) with all four shells from `obs`
    /// (for Schwarz / CSB (PQ|PQ) bounds): `n1*n2*n3*n4` doubles row-major (sh1
    /// slowest). Returns that count, 0 if fully screened, [`SCF_EINVAL`] (bad
    /// engine, null pointer, or total angular momentum beyond the tables) or
    /// [`SCF_EINTERNAL`].
    pub fn scf_compute_terfc_eri4(
        eng: *mut c_void,
        obs: *const c_void,
        sh1: c_int,
        sh2: c_int,
        sh3: c_int,
        sh4: c_int,
        out: *mut c_double,
    ) -> c_int;
    /// terf counterpart of [`scf_compute_terfc_eri4`]; element-wise
    /// `terf_eri4 + terfc_eri4` equals the Coulomb quartet.
    pub fn scf_compute_terf_eri4(
        eng: *mut c_void,
        obs: *const c_void,
        sh1: c_int,
        sh2: c_int,
        sh3: c_int,
        sh4: c_int,
        out: *mut c_double,
    ) -> c_int;
    /// Validation hook: the plain-Coulomb (sh1 sh2|sh3 sh4) quartet through the
    /// shim's McMurchie-Davidson path (no engine, no tables), for comparison with
    /// [`scf_compute_eri_quartet`]. Writes `n1*n2*n3*n4` doubles in the same
    /// layout and returns that count (zeros written when screened), or a
    /// negative status.
    pub fn scf_debug_coulomb_eri4(
        obs: *const c_void,
        sh1: c_int,
        sh2: c_int,
        sh3: c_int,
        sh4: c_int,
        out: *mut c_double,
    ) -> c_int;
    /// Test probe comparing the exact terf series with the Boys-function
    /// asymptotic `G_m(S,s) -> F_m(S)` on fixed out-of-table points (S = 25..400,
    /// s = 0.05..0.5) for `m = 0..=mmax`. Writes the worst relative error to
    /// `worst_rel` and the mean ns per call of each path over `reps` repetitions
    /// to `ser_ns` / `asym_ns` (each output may be null). Returns 1, or
    /// [`SCF_EINTERNAL`].
    pub fn scf_terf_asym_probe(
        mmax: c_int,
        reps: c_int,
        ser_ns: *mut c_double,
        asym_ns: *mut c_double,
        worst_rel: *mut c_double,
    ) -> c_int;
    /// Test probe for the tail form `G_m(S,s) = F_m(S) - Delta_m(S,s)`, an
    /// exact rearrangement whose sum length is set by s, not S. `worst_rel` is
    /// measured on G over the curvature-constrained regime (s <= 0.5);
    /// `worst_rel_delta` on Delta (the terfc auxiliary F - G that production
    /// consumes) over the same sweep plus a large-s point exercising the
    /// adaptive index, which `worst_i` reports. Delta stays accurate where G
    /// cannot: at large s, Delta converges onto F and G = F - Delta is pure
    /// cancellation. `tail_ns` / `series_ns` receive
    /// per-call timings over `reps` repetitions. Returns the number of samples
    /// compared, [`SCF_EINVAL`] for `mmax` outside `[0, 24)`, or
    /// [`SCF_EINTERNAL`].
    pub fn scf_terf_tail_probe(
        mmax: c_int,
        reps: c_int,
        tail_ns: *mut c_double,
        series_ns: *mut c_double,
        worst_rel: *mut c_double,
        worst_i: *mut c_int,
        worst_rel_delta: *mut c_double,
    ) -> c_int;
    /// Reads the process-global counters of terf auxiliary evaluations served
    /// by table interpolation (`tab`) and by the exact out-of-table series
    /// (`ser`). Either pointer may be null.
    pub fn scf_terf_series_counters(tab: *mut u64, ser: *mut u64);
    /// Resets both counters read by [`scf_terf_series_counters`] to zero.
    pub fn scf_terf_series_reset();
    /// Test probe: worst relative error of the table interpolation against the
    /// exact Poisson series over an off-node (S, s) sweep inside the finest
    /// tables (S < 4, s < 2), for `m = 0..=mmax`; written to `worst_rel`.
    /// Returns the number of samples compared, [`SCF_EINVAL`] if the tables do
    /// not load, or [`SCF_EINTERNAL`].
    pub fn scf_terf_interp_accuracy(
        table_dir: *const c_char,
        mmax: c_int,
        worst_rel: *mut c_double,
    ) -> c_int;
    /// Test gate: checks that the libint2-convention terf core evaluator
    /// reproduces the table-based auxiliary bit for bit over a fixed sweep of
    /// (rho, T, omega, r0) with `mmax = 8`. Writes the number of non-identical
    /// values to `mismatches` (expected 0) and the largest absolute difference
    /// to `worst`. Returns the number of values compared, [`SCF_EINVAL`] if the
    /// tables do not load, or [`SCF_EINTERNAL`].
    pub fn scf_terf_gm_eval_matches_terf_aux(
        table_dir: *const c_char,
        mismatches: *mut c_int,
        worst: *mut c_double,
    ) -> c_int;
    /// Creates a libint2 engine driving `Operator::terf` (feature-gated build
    /// with `FERRIC_LIBINT2_TERF` only; the symbol is absent otherwise).
    /// `braket`: 2 -> `xs_xs`, 3 -> `xs_xx`, 4 -> `xx_xx`; params are
    /// `{omega, r0}`. `status` (nullable) receives `SCF_OK`; [`SCF_EINVAL`] for a
    /// bad braket, `max_nprim < 1`, `omega` not finite and positive, `r0` not
    /// finite, or `max_l` outside `[0, scf_terf_libint2_max_l(braket)]`;
    /// [`SCF_ENOTIMPL`] for a `max_l` past [`scf_libint2_eri_max_l`];
    /// [`SCF_EUNSUPPORTED`] if the tables cannot be loaded; [`SCF_EINTERNAL`] on
    /// a C++ exception. A non-OK status always comes with a null return.
    pub fn scf_engine_create_terf_libint2(
        r0: c_double,
        omega: c_double,
        braket: c_int,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
        table_dir: *const c_char,
        status: *mut c_int,
    ) -> *mut c_void;
    /// Largest `max_l` the libint2 terf engine accepts for `braket` (2/3/4):
    /// `(TERFC_DIMM - 1) / braket` with table depth `TERFC_DIMM = 24`.
    /// [`SCF_EINVAL`] for any other braket. Feature-gated build only.
    pub fn scf_terf_libint2_max_l(braket: c_int) -> c_int;
    /// Largest `max_l` the linked libint2 supports for the terf engine at
    /// `braket`, derivative order 0: `LIBINT2_MAX_AM_eri` for 4, and its minimum
    /// with the uniform 2eri / 3eri limit (capped at `LIBINT2_MAX_AM_default`
    /// when libint2 uses center-dependent AM) for 2 / 3. [`SCF_EINVAL`] for any
    /// other braket. Feature-gated build only.
    pub fn scf_libint2_eri_max_l(braket: c_int) -> c_int;
    /// Fault injection for tests (feature-gated build only): while `on != 0`
    /// the libint2 terf hook throws on every call, so a test can drive the
    /// engine's refusal path. Process-global; never set it in production.
    pub fn scf_terf_libint2_inject_hook_fault(on: c_int);
    /// Calls the libint2 terf core-evaluation hook directly (test probe,
    /// feature-gated build only). `gm` holds `gm_len >= mmax + 1` doubles
    /// (checked). Returns `SCF_OK` if evaluated, [`SCF_EINVAL`] for bad probe
    /// arguments or if the hook refused `mmax`, `rho` or `omega`,
    /// [`SCF_EUNSUPPORTED`] if the tables are missing or the hook returned
    /// false, [`SCF_EINTERNAL`] on any other exception.
    pub fn scf_terf_libint2_hook_probe(
        table_dir: *const c_char,
        rho: c_double,
        t: c_double,
        mmax: c_int,
        omega: c_double,
        r0: c_double,
        gm: *mut c_double,
        gm_len: c_int,
    ) -> c_int;
    /// Accuracy gate (feature-gated build only): computes the terf block
    /// (shP|sh1 sh2) through both the libint2-native engine and the shim's
    /// McMurchie-Davidson path and writes `max|libint2 - md|` to `max_abs` and
    /// that difference divided by the block's largest magnitude to `max_rel`
    /// (both 0 when the block's magnitude is below 1e-12). Returns the number of
    /// elements compared, a negative status from either compute call, -100 for
    /// a null basis, -101 / -102 if the MD / libint2 engine cannot be created,
    /// or [`SCF_EINTERNAL`].
    pub fn scf_terf_libint2_vs_md_eri3(
        obs: *const c_void,
        dfbs: *const c_void,
        shP: c_int,
        sh1: c_int,
        sh2: c_int,
        r0: c_double,
        omega: c_double,
        table_dir: *const c_char,
        max_abs: *mut c_double,
        max_rel: *mut c_double,
    ) -> c_int;
    /// Dipole integrals ⟨μ|(r - origin)|ν⟩ over the whole basis (`origin` = 3
    /// doubles, Bohr) through a libint2 `emultipole1` engine: writes the x, y, z
    /// matrices (`nbas x nbas` row-major, symmetric) consecutively into `out`
    /// (`3*nbas*nbas` doubles). `nbas` must equal [`scf_basis_nbasis`]. Returns
    /// `3*nbas*nbas`, or -1 if libint2 throws.
    pub fn scf_compute_dipole(
        bs: *const c_void,
        origin: *const c_double,
        nbas: c_int,
        out: *mut c_double,
    ) -> c_int;
    /// Second-moment integrals ⟨μ|(r - origin)_p (r - origin)_q|ν⟩ (`origin` in
    /// Bohr) through a libint2 `emultipole2` engine: writes the xx, xy, xz, yy,
    /// yz, zz matrices (`nbas x nbas` row-major, symmetric) consecutively into
    /// `out` (`6*nbas*nbas` doubles). Returns `6*nbas*nbas`, or -1 if libint2
    /// throws.
    pub fn scf_compute_second_moment(
        bs: *const c_void,
        origin: *const c_double,
        nbas: c_int,
        out: *mut c_double,
    ) -> c_int;
}

/// Standard 1/r Coulomb operator.
pub const OP_COULOMB: c_int = 0;
/// Long-range attenuated Coulomb: erf(omega * r) / r.
pub const OP_ERF_COULOMB: c_int = 1;
/// Short-range attenuated Coulomb: erfc(omega * r) / r.
pub const OP_ERFC_COULOMB: c_int = 2;
/// Yukawa / screened Coulomb exp(-zeta r)/r -- libint2 `Operator::stg_x_coulomb`.
pub const OP_YUKAWA: c_int = 3;
/// Exact Slater-type geminal exp(-zeta r) -- libint2 `Operator::stg`.
pub const OP_SLATER_GEMINAL: c_int = 4;
/// Overlap operator S = ⟨μ|ν⟩.
pub const OP_OVERLAP: c_int = 100;
/// Kinetic energy operator T = -½⟨μ|∇²|ν⟩.
pub const OP_KINETIC: c_int = 101;
/// Nuclear attraction operator V = -Σ_A Z_A/|r-R_A|.
pub const OP_NUCLEAR: c_int = 102;
/// Electric dipole operator ⟨μ|(r-O)|ν⟩ (libint2 `emultipole1`). The engine
/// factories have no case for this kind and return NULL for it; dipole
/// integrals come from [`scf_compute_dipole`].
pub const OP_EMULTIPOLE1: c_int = 103;
/// Contracted Gaussian geminal f12 (Cgtg) -- see `scf_engine_create_geminal`.
pub const OP_CGTG: c_int = 200;
/// f12/r12 (geminal times Coulomb) -- see `scf_engine_create_geminal`.
pub const OP_CGTG_X_COULOMB: c_int = 201;
/// |∇f12|² kinetic commutator integrand -- see `scf_engine_create_geminal`.
pub const OP_DELCGTG2: c_int = 202;

/// Shim status: invalid argument (e.g. an output buffer too small for the result).
pub const SCF_EINVAL: c_int = -1;
/// Shim status: not available in this build (e.g. an angular momentum the
/// linked libint2 was not generated for).
pub const SCF_ENOTIMPL: c_int = -2;
/// Shim status: libint2 threw; the call's output is undefined.
pub const SCF_EINTERNAL: c_int = -3;
/// Shim status: the linked libint2 lacks the requested derivative order.
pub const SCF_EUNSUPPORTED: c_int = -4;
