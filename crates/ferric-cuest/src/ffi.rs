//! Raw FFI declarations for the subset of the cuEST 0.3.x C API that
//! ferric uses, plus the few CUDA runtime calls needed to move buffers.
//!
//! Written by hand from the cuEST 0.3.0 documentation. Every prototype and
//! every enum value here is restated in `csrc/abi_check.c`, which build.rs
//! compiles against the installed `cuest.h`; a drift between this file and
//! the header is therefore a build error. Keep the two files in step.

#![allow(non_camel_case_types, non_snake_case)]

use std::os::raw::{c_char, c_int, c_void};

pub type cuestStatus_t = c_int;
pub type cuestHandle_t = *mut c_void;
pub type cuestAOShell_t = *mut c_void;
pub type cuestAOBasis_t = *mut c_void;
pub type cuestAOPairList_t = *mut c_void;
pub type cuestOEIntPlan_t = *mut c_void;
/// Every `cuest*Parameters_t` is `void*` in the header.
pub type cuestParams_t = *mut c_void;

// cuestStatus_t
pub const CUEST_STATUS_SUCCESS: cuestStatus_t = 0;
pub const CUEST_STATUS_UNSUPPORTED_ARCHITECTURE: cuestStatus_t = 11;

/// `cuestStatus_t` names, indexed by value (0..=16).
pub const STATUS_NAMES: [&str; 17] = [
    "SUCCESS",
    "EXCEPTION",
    "NULL_POINTER",
    "INVALID_ARGUMENT",
    "INVALID_SIZE",
    "INVALID_TYPE",
    "INVALID_PARAMETER",
    "INVALID_ATTRIBUTE",
    "INVALID_HANDLE",
    "UNKNOWN_ERROR",
    "UNSUPPORTED_ARGUMENT",
    "UNSUPPORTED_ARCHITECTURE",
    "INVALID_PLAN",
    "HOME_NOT_FOUND",
    "UNRECOVERABLE_ERROR",
    "UNRECOVERABLE_ON_ENTRY",
    "EXCEEDS_MAX_L",
];

// cuestType_t (for cuestQuery)
pub const CUEST_AOBASIS: c_int = 2;
// cuestAOBasisAttributes_t
pub const CUEST_AOBASIS_NUM_AO: c_int = 5;

// cuestParametersType_t
pub const CUEST_HANDLE_PARAMETERS: c_int = 0;
pub const CUEST_AOSHELL_PARAMETERS: c_int = 1;
pub const CUEST_AOBASIS_PARAMETERS: c_int = 2;
pub const CUEST_AOPAIRLIST_PARAMETERS: c_int = 3;
pub const CUEST_OEINTPLAN_PARAMETERS: c_int = 4;
pub const CUEST_OVERLAPCOMPUTE_PARAMETERS: c_int = 9;
pub const CUEST_KINETICCOMPUTE_PARAMETERS: c_int = 11;
pub const CUEST_POTENTIALCOMPUTE_PARAMETERS: c_int = 13;

// cuestPotentialComputeParametersAttributes_t / cuestFfloatUsageMode_t
pub const CUEST_POTENTIALCOMPUTE_PARAMETERS_FFLOAT_USAGE_MODE: c_int = 0;
pub const CUEST_FFLOAT_USAGE_MODE_OFF: c_int = 2;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct cuestWorkspace_t {
    pub hostBuffer: usize,
    pub hostBufferSizeInBytes: usize,
    pub deviceBuffer: usize,
    pub deviceBufferSizeInBytes: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct cuestWorkspaceDescriptor_t {
    pub hostBufferSizeInBytes: usize,
    pub deviceBufferSizeInBytes: usize,
}

extern "C" {
    pub fn cuestParametersCreate(
        parameterType: c_int,
        outParameters: *mut cuestParams_t,
    ) -> cuestStatus_t;
    pub fn cuestParametersDestroy(parameterType: c_int, parameters: cuestParams_t)
        -> cuestStatus_t;
    pub fn cuestParametersConfigure(
        parameterType: c_int,
        parameters: cuestParams_t,
        attribute: c_int,
        attributeValue: *const c_void,
        attributeValueSize: usize,
    ) -> cuestStatus_t;

    pub fn cuestCreate(parameters: cuestParams_t, handle: *mut cuestHandle_t) -> cuestStatus_t;
    pub fn cuestDestroy(handle: cuestHandle_t) -> cuestStatus_t;
    pub fn cuestGetMajorVersion(handle: cuestHandle_t, v: *mut u32) -> cuestStatus_t;
    pub fn cuestGetMinorVersion(handle: cuestHandle_t, v: *mut u32) -> cuestStatus_t;
    pub fn cuestGetPatchVersion(handle: cuestHandle_t, v: *mut u32) -> cuestStatus_t;

    pub fn cuestQuery(
        handle: cuestHandle_t,
        r#type: c_int,
        object: *const c_void,
        attribute: c_int,
        attributeValue: *mut c_void,
        attributeValueSize: usize,
    ) -> cuestStatus_t;

    pub fn cuestAOShellCreate(
        handle: cuestHandle_t,
        isPure: i32,
        L: u64,
        numPrimitive: u64,
        exponents: *const f64,
        coefficients: *const f64,
        parameters: cuestParams_t,
        outShell: *mut cuestAOShell_t,
    ) -> cuestStatus_t;
    pub fn cuestAOShellDestroy(shell: cuestAOShell_t) -> cuestStatus_t;

    pub fn cuestAOBasisCreateWorkspaceQuery(
        handle: cuestHandle_t,
        numAtoms: u64,
        numShellsPerAtom: *const u64,
        shells: *const cuestAOShell_t,
        parameters: cuestParams_t,
        persistent: *mut cuestWorkspaceDescriptor_t,
        temporary: *mut cuestWorkspaceDescriptor_t,
        outBasis: *mut cuestAOBasis_t,
    ) -> cuestStatus_t;
    pub fn cuestAOBasisCreate(
        handle: cuestHandle_t,
        numAtoms: u64,
        numShellsPerAtom: *const u64,
        shells: *const cuestAOShell_t,
        parameters: cuestParams_t,
        persistent: *mut cuestWorkspace_t,
        temporary: *mut cuestWorkspace_t,
        outBasis: *mut cuestAOBasis_t,
    ) -> cuestStatus_t;
    pub fn cuestAOBasisDestroy(basis: cuestAOBasis_t) -> cuestStatus_t;

    pub fn cuestAOPairListCreateWorkspaceQuery(
        handle: cuestHandle_t,
        basis: cuestAOBasis_t,
        numAtoms: u64,
        xyz: *const f64,
        thresholdPQ: f64,
        parameters: cuestParams_t,
        persistent: *mut cuestWorkspaceDescriptor_t,
        temporary: *mut cuestWorkspaceDescriptor_t,
        outPairList: *mut cuestAOPairList_t,
    ) -> cuestStatus_t;
    pub fn cuestAOPairListCreate(
        handle: cuestHandle_t,
        basis: cuestAOBasis_t,
        numAtoms: u64,
        xyz: *const f64,
        thresholdPQ: f64,
        parameters: cuestParams_t,
        persistent: *mut cuestWorkspace_t,
        temporary: *mut cuestWorkspace_t,
        outPairList: *mut cuestAOPairList_t,
    ) -> cuestStatus_t;
    pub fn cuestAOPairListDestroy(pairList: cuestAOPairList_t) -> cuestStatus_t;

    pub fn cuestOEIntPlanCreateWorkspaceQuery(
        handle: cuestHandle_t,
        basis: cuestAOBasis_t,
        pairList: cuestAOPairList_t,
        parameters: cuestParams_t,
        persistent: *mut cuestWorkspaceDescriptor_t,
        temporary: *mut cuestWorkspaceDescriptor_t,
        outPlan: *mut cuestOEIntPlan_t,
    ) -> cuestStatus_t;
    pub fn cuestOEIntPlanCreate(
        handle: cuestHandle_t,
        basis: cuestAOBasis_t,
        pairList: cuestAOPairList_t,
        parameters: cuestParams_t,
        persistent: *mut cuestWorkspace_t,
        temporary: *mut cuestWorkspace_t,
        outPlan: *mut cuestOEIntPlan_t,
    ) -> cuestStatus_t;
    pub fn cuestOEIntPlanDestroy(plan: cuestOEIntPlan_t) -> cuestStatus_t;

    pub fn cuestOverlapComputeWorkspaceQuery(
        handle: cuestHandle_t,
        plan: cuestOEIntPlan_t,
        parameters: cuestParams_t,
        temporary: *mut cuestWorkspaceDescriptor_t,
        outSMatrix: *mut f64,
    ) -> cuestStatus_t;
    pub fn cuestOverlapCompute(
        handle: cuestHandle_t,
        plan: cuestOEIntPlan_t,
        parameters: cuestParams_t,
        temporary: *mut cuestWorkspace_t,
        outSMatrix: *mut f64,
    ) -> cuestStatus_t;

    pub fn cuestKineticComputeWorkspaceQuery(
        handle: cuestHandle_t,
        plan: cuestOEIntPlan_t,
        parameters: cuestParams_t,
        temporary: *mut cuestWorkspaceDescriptor_t,
        outTMatrix: *mut f64,
    ) -> cuestStatus_t;
    pub fn cuestKineticCompute(
        handle: cuestHandle_t,
        plan: cuestOEIntPlan_t,
        parameters: cuestParams_t,
        temporary: *mut cuestWorkspace_t,
        outTMatrix: *mut f64,
    ) -> cuestStatus_t;

    pub fn cuestPotentialComputeWorkspaceQuery(
        handle: cuestHandle_t,
        plan: cuestOEIntPlan_t,
        parameters: cuestParams_t,
        temporary: *mut cuestWorkspaceDescriptor_t,
        numCharges: u64,
        xyz: *const f64,
        q: *const f64,
        outVMatrix: *mut f64,
    ) -> cuestStatus_t;
    pub fn cuestPotentialCompute(
        handle: cuestHandle_t,
        plan: cuestOEIntPlan_t,
        parameters: cuestParams_t,
        temporary: *mut cuestWorkspace_t,
        numCharges: u64,
        xyz: *const f64,
        q: *const f64,
        outVMatrix: *mut f64,
    ) -> cuestStatus_t;
}

// ---- CUDA runtime (libcudart) -------------------------------------------

pub type cudaError_t = c_int;
pub const CUDA_SUCCESS: cudaError_t = 0;
pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR: c_int = 76;

extern "C" {
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
    ) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
    pub fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> cudaError_t;
    pub fn cudaGetErrorString(error: cudaError_t) -> *const c_char;
    pub fn cudaRuntimeGetVersion(runtimeVersion: *mut c_int) -> cudaError_t;
    pub fn cudaDriverGetVersion(driverVersion: *mut c_int) -> cudaError_t;
}
