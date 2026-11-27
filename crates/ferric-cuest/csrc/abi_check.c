/*
 * Compile-time agreement check between src/ffi.rs and the installed cuest.h.
 *
 * Each line below restates, in C, a prototype or constant that ffi.rs
 * declares by hand. Assigning a cuEST function to a pointer of the type we
 * EXPECT fails to compile (-Werror=incompatible-pointer-types) if the header
 * disagrees; _Static_assert does the same for enum values and struct sizes.
 * build.rs compiles this file whenever the `cuest` feature is on.
 *
 * Keep this file and ffi.rs in step: a new extern in ffi.rs needs a line here.
 */
#include <stddef.h>
#include <stdint.h>
#include <cuest.h>

typedef void *P; /* every cuest*Parameters_t / handle typedef is void* */

#define EXPECT_FN(name, ret, ...) ret (*const ferric_abi_##name)(__VA_ARGS__) = name;

EXPECT_FN(cuestParametersCreate, cuestStatus_t, cuestParametersType_t, void **)
EXPECT_FN(cuestParametersDestroy, cuestStatus_t, cuestParametersType_t, void *)
EXPECT_FN(cuestParametersConfigure, cuestStatus_t, cuestParametersType_t, void *, int, const void *, size_t)
EXPECT_FN(cuestCreate, cuestStatus_t, P, P *)
EXPECT_FN(cuestDestroy, cuestStatus_t, P)
EXPECT_FN(cuestGetMajorVersion, cuestStatus_t, P, uint32_t *)
EXPECT_FN(cuestGetMinorVersion, cuestStatus_t, P, uint32_t *)
EXPECT_FN(cuestGetPatchVersion, cuestStatus_t, P, uint32_t *)
EXPECT_FN(cuestQuery, cuestStatus_t, P, cuestType_t, const void *, int, void *, size_t)
EXPECT_FN(cuestAOShellCreate, cuestStatus_t, P, int32_t, uint64_t, uint64_t, const double *, const double *, const P, P *)
EXPECT_FN(cuestAOShellDestroy, cuestStatus_t, P)
EXPECT_FN(cuestAOBasisCreateWorkspaceQuery, cuestStatus_t, P, uint64_t, const uint64_t *, const P *, const P,
          cuestWorkspaceDescriptor_t *, cuestWorkspaceDescriptor_t *, P *)
EXPECT_FN(cuestAOBasisCreate, cuestStatus_t, P, uint64_t, const uint64_t *, const P *, const P,
          cuestWorkspace_t *, cuestWorkspace_t *, P *)
EXPECT_FN(cuestAOBasisDestroy, cuestStatus_t, P)
EXPECT_FN(cuestAOPairListCreateWorkspaceQuery, cuestStatus_t, P, const P, uint64_t, const double *, double, const P,
          cuestWorkspaceDescriptor_t *, cuestWorkspaceDescriptor_t *, P *)
EXPECT_FN(cuestAOPairListCreate, cuestStatus_t, P, const P, uint64_t, const double *, double, const P,
          cuestWorkspace_t *, cuestWorkspace_t *, P *)
EXPECT_FN(cuestAOPairListDestroy, cuestStatus_t, P)
EXPECT_FN(cuestOEIntPlanCreateWorkspaceQuery, cuestStatus_t, P, const P, const P, const P,
          cuestWorkspaceDescriptor_t *, cuestWorkspaceDescriptor_t *, P *)
EXPECT_FN(cuestOEIntPlanCreate, cuestStatus_t, P, const P, const P, const P,
          cuestWorkspace_t *, cuestWorkspace_t *, P *)
EXPECT_FN(cuestOEIntPlanDestroy, cuestStatus_t, P)
EXPECT_FN(cuestOverlapComputeWorkspaceQuery, cuestStatus_t, P, const P, const P, cuestWorkspaceDescriptor_t *, double *)
EXPECT_FN(cuestOverlapCompute, cuestStatus_t, P, const P, const P, cuestWorkspace_t *, double *)
EXPECT_FN(cuestKineticComputeWorkspaceQuery, cuestStatus_t, P, const P, const P, cuestWorkspaceDescriptor_t *, double *)
EXPECT_FN(cuestKineticCompute, cuestStatus_t, P, const P, const P, cuestWorkspace_t *, double *)
EXPECT_FN(cuestPotentialComputeWorkspaceQuery, cuestStatus_t, P, const P, const P, cuestWorkspaceDescriptor_t *,
          uint64_t, const double *, const double *, double *)
EXPECT_FN(cuestPotentialCompute, cuestStatus_t, P, const P, const P, cuestWorkspace_t *,
          uint64_t, const double *, const double *, double *)

EXPECT_FN(cuestDirectIntPlanCreateWorkspaceQuery, cuestStatus_t, P, const P, const P, const P,
          cuestWorkspaceDescriptor_t *, cuestWorkspaceDescriptor_t *, P *)
EXPECT_FN(cuestDirectIntPlanCreate, cuestStatus_t, P, const P, const P, const P,
          cuestWorkspace_t *, cuestWorkspace_t *, P *)
EXPECT_FN(cuestDirectIntPlanDestroy, cuestStatus_t, P)
EXPECT_FN(cuestDirectCoulombComputeWorkspaceQuery, cuestStatus_t, P, const P, const P, cuestWorkspaceDescriptor_t *,
          uint64_t, const double *, double *)
EXPECT_FN(cuestDirectCoulombCompute, cuestStatus_t, P, const P, const P, cuestWorkspace_t *,
          uint64_t, const double *, double *)
EXPECT_FN(cuestDirectSymmetricExchangeComputeWorkspaceQuery, cuestStatus_t, P, const P, const P,
          cuestWorkspaceDescriptor_t *, uint64_t, const double *, double *)
EXPECT_FN(cuestDirectSymmetricExchangeCompute, cuestStatus_t, P, const P, const P, cuestWorkspace_t *,
          uint64_t, const double *, double *)

/* Enum values ffi.rs hard-codes. */
_Static_assert(sizeof(cuestStatus_t) == sizeof(int), "cuestStatus_t is not int-sized");
_Static_assert(sizeof(cuestParametersType_t) == sizeof(int), "cuestParametersType_t is not int-sized");
_Static_assert(CUEST_STATUS_SUCCESS == 0, "status");
_Static_assert(CUEST_STATUS_EXCEPTION == 1, "status");
_Static_assert(CUEST_STATUS_INVALID_ARGUMENT == 3, "status");
_Static_assert(CUEST_STATUS_UNSUPPORTED_ARGUMENT == 10, "status");
_Static_assert(CUEST_STATUS_UNSUPPORTED_ARCHITECTURE == 11, "status");
_Static_assert(CUEST_STATUS_EXCEEDS_MAX_L == 16, "status: STATUS_NAMES in ffi.rs ends at 16");
_Static_assert(CUEST_AOBASIS == 2, "cuestType_t");
_Static_assert(CUEST_AOBASIS_NUM_AO == 5, "cuestAOBasisAttributes_t");
_Static_assert(CUEST_HANDLE_PARAMETERS == 0, "parameters type");
_Static_assert(CUEST_AOSHELL_PARAMETERS == 1, "parameters type");
_Static_assert(CUEST_AOBASIS_PARAMETERS == 2, "parameters type");
_Static_assert(CUEST_AOPAIRLIST_PARAMETERS == 3, "parameters type");
_Static_assert(CUEST_OEINTPLAN_PARAMETERS == 4, "parameters type");
_Static_assert(CUEST_OVERLAPCOMPUTE_PARAMETERS == 9, "parameters type");
_Static_assert(CUEST_KINETICCOMPUTE_PARAMETERS == 11, "parameters type");
_Static_assert(CUEST_POTENTIALCOMPUTE_PARAMETERS == 13, "parameters type");
_Static_assert(CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS == 62, "parameters type");
_Static_assert(CUEST_DIRECTINTPLAN_PARAMETERS == 63, "parameters type");
_Static_assert(CUEST_DIRECTSYMMETRICEXCHANGECOMPUTE_PARAMETERS == 64, "parameters type");
_Static_assert(CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS_DENSITY_SCHWARZ_THRESHOLD == 1, "direct J attribute");
_Static_assert(CUEST_DIRECTSYMMETRICEXCHANGECOMPUTE_PARAMETERS_DENSITY_SCHWARZ_THRESHOLD == 1, "direct K attribute");
_Static_assert(CUEST_POTENTIALCOMPUTE_PARAMETERS_FFLOAT_USAGE_MODE == 0, "potential attribute");
_Static_assert(CUEST_FFLOAT_USAGE_MODE_OFF == 2, "ffloat mode");
_Static_assert(sizeof(cuestFfloatUsageMode_t) == sizeof(int), "ffloat mode enum is not int-sized");

/* Workspace structs: four / two pointer-sized fields, in this order. */
_Static_assert(sizeof(cuestWorkspace_t) == 4 * sizeof(size_t), "cuestWorkspace_t size");
_Static_assert(offsetof(cuestWorkspace_t, hostBuffer) == 0, "cuestWorkspace_t layout");
_Static_assert(offsetof(cuestWorkspace_t, hostBufferSizeInBytes) == 8, "cuestWorkspace_t layout");
_Static_assert(offsetof(cuestWorkspace_t, deviceBuffer) == 16, "cuestWorkspace_t layout");
_Static_assert(offsetof(cuestWorkspace_t, deviceBufferSizeInBytes) == 24, "cuestWorkspace_t layout");
_Static_assert(sizeof(cuestWorkspaceDescriptor_t) == 2 * sizeof(size_t), "descriptor size");
_Static_assert(offsetof(cuestWorkspaceDescriptor_t, hostBufferSizeInBytes) == 0, "descriptor layout");
_Static_assert(offsetof(cuestWorkspaceDescriptor_t, deviceBufferSizeInBytes) == 8, "descriptor layout");
