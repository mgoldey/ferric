//! Raw FFI declarations for the libint2 C++ shim (`shim/shim.cc`).
//!
//! These are unsafe C-linkage functions. Prefer the safe wrappers in
//! [`crate::engine`] and [`crate::basis_bridge`].

use std::os::raw::{c_char, c_double, c_int, c_void};

/// C-compatible shell descriptor passed to the libint2 shim.
#[repr(C)]
pub struct CShell {
    pub l: c_int,
    pub nprim: c_int,
    pub atom_index: c_int,
    pub pure: c_int,
    pub exponents: *const c_double,
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
    pub atomic_number: c_double,
    pub x: c_double,
    pub y: c_double,
    pub z: c_double,
}

extern "C" {
    pub fn scf_libint_init();
    pub fn scf_libint_finalize();
    pub fn scf_basis_create(
        shells: *const CShell,
        nshells: c_int,
        atoms: *const CAtom,
        natoms: c_int,
    ) -> *mut c_void;
    pub fn scf_basis_destroy(bs: *mut c_void);
    pub fn scf_basis_nbasis(bs: *const c_void) -> c_int;
    pub fn scf_basis_nshells(bs: *const c_void) -> c_int;
    pub fn scf_basis_shell_dims(bs: *const c_void, out: *mut c_int);
    pub fn scf_basis_max_dims(bs: *const c_void, max_nprim: *mut c_int, max_l: *mut c_int);
    pub fn scf_engine_create(
        op_kind: c_int,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
    ) -> *mut c_void;
    pub fn scf_engine_destroy(eng: *mut c_void);
    pub fn scf_engine_set_point_charges(
        eng: *mut c_void,
        atoms: *const CAtom,
        natoms: c_int,
    ) -> c_int;
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
    pub fn scf_compute_schwarz(eng: *mut c_void, bs: *const c_void, qmat: *mut c_double) -> c_int;
    pub fn scf_engine_create_deriv(
        op_kind: c_int,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
    ) -> *mut c_void;
    pub fn scf_engine_create_geminal(
        op_kind: c_int,
        ngauss: c_int,
        exps: *const c_double,
        coefs: *const c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
    ) -> *mut c_void;
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
    pub fn scf_compute_eri_deriv_quartet(
        eng: *mut c_void,
        bs: *const c_void,
        sh1: c_int,
        sh2: c_int,
        sh3: c_int,
        sh4: c_int,
        out: *mut c_double,
    ) -> c_int;
    pub fn scf_engine_create_3center(
        op_kind: c_int,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
    ) -> *mut c_void;
    pub fn scf_engine_create_2center(
        op_kind: c_int,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
    ) -> *mut c_void;
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
    pub fn scf_engine_create_3center_deriv(
        op_kind: c_int,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
    ) -> *mut c_void;
    pub fn scf_engine_create_2center_deriv(
        op_kind: c_int,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
    ) -> *mut c_void;
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
    // Exact terfc(r,r0)/r via 2D interpolation tables (Dutoi/Goldey). table_dir may be
    // null (falls back to FERRIC_TERF_TABLE_DIR). See shim.h / terf-tables/terf_plan.md.
    pub fn scf_engine_create_terfc_3center(
        r0: c_double,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
        table_dir: *const c_char,
    ) -> *mut c_void;
    pub fn scf_engine_create_terfc_2center(
        r0: c_double,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
        table_dir: *const c_char,
    ) -> *mut c_void;
    pub fn scf_compute_terfc_eri3(
        eng: *mut c_void,
        obs: *const c_void,
        dfbs: *const c_void,
        shP: c_int,
        sh1: c_int,
        sh2: c_int,
        out: *mut c_double,
    ) -> c_int;
    pub fn scf_compute_terfc_eri2(
        eng: *mut c_void,
        dfbs: *const c_void,
        shP: c_int,
        shQ: c_int,
        out: *mut c_double,
    ) -> c_int;
    // terf(r,r0)/r = tempered LONG-RANGE complement of terfc (terf + terfc = coulomb),
    // same tables/curvature constraint. See shim.h.
    pub fn scf_engine_create_terf_3center(
        r0: c_double,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
        table_dir: *const c_char,
    ) -> *mut c_void;
    pub fn scf_engine_create_terf_2center(
        r0: c_double,
        omega: c_double,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
        table_dir: *const c_char,
    ) -> *mut c_void;
    pub fn scf_compute_terf_eri3(
        eng: *mut c_void,
        obs: *const c_void,
        dfbs: *const c_void,
        shP: c_int,
        sh1: c_int,
        sh2: c_int,
        out: *mut c_double,
    ) -> c_int;
    pub fn scf_compute_terf_eri2(
        eng: *mut c_void,
        dfbs: *const c_void,
        shP: c_int,
        shQ: c_int,
        out: *mut c_double,
    ) -> c_int;
    // 4-center terfc/terf quartets (sh1 sh2|op|sh3 sh4), all four shells from `obs`.
    // These are what Schwarz/CSB need to build a (PQ|PQ) table for terfc.
    pub fn scf_compute_terfc_eri4(
        eng: *mut c_void,
        obs: *const c_void,
        sh1: c_int,
        sh2: c_int,
        sh3: c_int,
        sh4: c_int,
        out: *mut c_double,
    ) -> c_int;
    pub fn scf_compute_terf_eri4(
        eng: *mut c_void,
        obs: *const c_void,
        sh1: c_int,
        sh2: c_int,
        sh3: c_int,
        sh4: c_int,
        out: *mut c_double,
    ) -> c_int;
    // Validation hook: the same 4-center MD path with the plain Coulomb kernel,
    // for cross-checking the new contraction against libint2's own quartet.
    pub fn scf_debug_coulomb_eri4(
        obs: *const c_void,
        sh1: c_int,
        sh2: c_int,
        sh3: c_int,
        sh4: c_int,
        out: *mut c_double,
    ) -> c_int;
    // STEP 1 gate for the libint2 core-eval port: terf_gm_eval_impl vs terf_aux.
    pub fn scf_terf_asym_probe(
        mmax: c_int,
        reps: c_int,
        ser_ns: *mut c_double,
        asym_ns: *mut c_double,
        worst_rel: *mut c_double,
    ) -> c_int;
    // Tail form G_m(S,s) = F_m(S) - Delta_m(S,s): exact rearrangement whose sum
    // length is set by s, not S. `worst_rel` is measured on G over the
    // curvature-constrained regime (s <= 0.5); `worst_rel_delta` is measured on
    // Delta (== the terfc auxiliary F - G, the quantity production consumes)
    // over the same sweep plus a large-s point exercising the adaptive index,
    // which `worst_i` reports. Delta stays accurate where G cannot: at large s,
    // Delta converges onto F and G = F - Delta is pure cancellation.
    pub fn scf_terf_tail_probe(
        mmax: c_int,
        reps: c_int,
        tail_ns: *mut c_double,
        series_ns: *mut c_double,
        worst_rel: *mut c_double,
        worst_i: *mut c_int,
        worst_rel_delta: *mut c_double,
    ) -> c_int;
    pub fn scf_terf_series_counters(tab: *mut u64, ser: *mut u64);
    pub fn scf_terf_series_reset();
    pub fn scf_terf_interp_accuracy(
        table_dir: *const c_char,
        mmax: c_int,
        worst_rel: *mut c_double,
    ) -> c_int;
    pub fn scf_terf_gm_eval_matches_terf_aux(
        table_dir: *const c_char,
        mismatches: *mut c_int,
        worst: *mut c_double,
    ) -> c_int;
    // STEP 4 gate: libint2-native terf vs the hand-rolled MD path (feature-gated
    // build only; the symbol is absent without FERRIC_LIBINT2_TERF).
    // STEP 4/5: libint2-native terf engine (feature-gated build only).
    // braket: 2 -> xs_xs, 3 -> xs_xx, 4 -> xx_xx. Params are {omega, r0}.
    pub fn scf_engine_create_terf_libint2(
        r0: c_double,
        omega: c_double,
        braket: c_int,
        max_nprim: c_int,
        max_l: c_int,
        precision: c_double,
        table_dir: *const c_char,
    ) -> *mut c_void;
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
    pub fn scf_compute_dipole(
        bs: *const c_void,
        origin: *const c_double,
        nbas: c_int,
        out: *mut c_double,
    ) -> c_int;
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
/// Electric multipole operator (dipole): ⟨μ|(r-O)|ν⟩.
pub const OP_EMULTIPOLE1: c_int = 103;
/// Contracted Gaussian geminal f12 (Cgtg) -- see `scf_engine_create_geminal`.
pub const OP_CGTG: c_int = 200;
/// f12/r12 (geminal times Coulomb) -- see `scf_engine_create_geminal`.
pub const OP_CGTG_X_COULOMB: c_int = 201;
/// |∇f12|² kinetic commutator integrand -- see `scf_engine_create_geminal`.
pub const OP_DELCGTG2: c_int = 202;
