// C-ABI wrapper around libecpint's ECPIntegrator. See ecp_shim.h for the contract.
//
// libecpint applies NO internal normalization to the Gaussian contraction
// coefficients (it uses shell.coef(a) verbatim). Callers must therefore pass
// fully-normalized coefficients (primitive normalization N(alpha,l) folded in,
// contraction normalized to unit self-overlap). ferric's basis_bridge handles
// that on the Rust side; this shim is a thin pass-through.

#include "ecp_shim.h"

#include <libecpint.hpp>
#include <vector>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>

using libecpint::ECPIntegrator;

static int ncart_for_l(int l) { return ((l + 1) * (l + 2)) / 2; }

extern "C" int ferric_ecp_ncart(const ferric_ecp_gshell *shells, int nshell) {
    int n = 0;
    for (int s = 0; s < nshell; ++s) n += ncart_for_l(shells[s].l);
    return n;
}

/* Flatten the caller's shells + ECP centers into the streams libecpint's
 * ECPIntegrator wants, and initialise it to the requested derivative order.
 * Shared by the energy and derivative entry points so the two can never drift
 * apart in how they set up the basis. Throws on libecpint failure; every caller
 * runs inside a try/catch (the C ABI must never see an exception unwind). */
static void setup_integrator(ECPIntegrator &integrator,
                             const ferric_ecp_gshell *shells, int nshell,
                             const ferric_ecp_center *ecps, int necp,
                             int deriv) {
    // --- Flatten the Gaussian basis into the streams set_gaussian_basis wants ---
    std::vector<double> g_coords;   // 3 per shell
    std::vector<double> g_exps;     // sum of nprim
    std::vector<double> g_coefs;    // sum of nprim
    std::vector<int>    g_ams;      // 1 per shell
    std::vector<int>    g_lengths;  // 1 per shell (nprim)
    g_coords.reserve(3 * nshell);
    g_ams.reserve(nshell);
    g_lengths.reserve(nshell);
    for (int s = 0; s < nshell; ++s) {
        const ferric_ecp_gshell &sh = shells[s];
        g_coords.push_back(sh.x);
        g_coords.push_back(sh.y);
        g_coords.push_back(sh.z);
        g_ams.push_back(sh.l);
        g_lengths.push_back(sh.nprim);
        for (int p = 0; p < sh.nprim; ++p) {
            g_exps.push_back(sh.exponents[p]);
            g_coefs.push_back(sh.coefficients[p]);
        }
    }

    // --- Flatten the ECP centers into the streams set_ecp_basis wants ---
    std::vector<double> u_coords;   // 3 per ECP
    std::vector<double> u_exps;     // sum of nterm
    std::vector<double> u_coefs;    // sum of nterm
    std::vector<int>    u_ams;      // 1 per term
    std::vector<int>    u_ns;       // 1 per term (r-power)
    std::vector<int>    u_lengths;  // 1 per ECP (nterm)
    u_coords.reserve(3 * necp);
    u_lengths.reserve(necp);
    for (int e = 0; e < necp; ++e) {
        const ferric_ecp_center &u = ecps[e];
        u_coords.push_back(u.x);
        u_coords.push_back(u.y);
        u_coords.push_back(u.z);
        u_lengths.push_back(u.nterm);
        for (int t = 0; t < u.nterm; ++t) {
            u_ams.push_back(u.ams[t]);
            u_ns.push_back(u.ns[t]);
            u_exps.push_back(u.exponents[t]);
            u_coefs.push_back(u.coefficients[t]);
        }
    }

    integrator.set_gaussian_basis(nshell, g_coords.data(), g_exps.data(),
                                  g_coefs.data(), g_ams.data(), g_lengths.data());
    integrator.set_ecp_basis(necp, u_coords.data(), u_exps.data(),
                             u_coefs.data(), u_ams.data(), u_ns.data(),
                             u_lengths.data());
    integrator.init(deriv);
}

extern "C" int ferric_ecp_matrix(const ferric_ecp_gshell *shells, int nshell,
                                 const ferric_ecp_center *ecps, int necp,
                                 double *out_vecp) {
    if (nshell <= 0 || necp <= 0 || shells == nullptr || ecps == nullptr ||
        out_vecp == nullptr) {
        return FERRIC_ECP_EINVAL;
    }
    try {
        ECPIntegrator integrator;
        setup_integrator(integrator, shells, nshell, ecps, necp, /*deriv=*/0);
        integrator.compute_integrals();

        auto data = integrator.get_integrals();  // shared_ptr<vector<double>>, M(i,j)=i*ncart+j
        const int ncart = ferric_ecp_ncart(shells, nshell);
        if (static_cast<int>(data->size()) != ncart * ncart) {
            std::fprintf(stderr,
                "ferric_ecp_matrix: size mismatch %zu vs %d^2\n",
                data->size(), ncart);
            return FERRIC_ECP_EINTERNAL;
        }
        for (int i = 0; i < ncart * ncart; ++i) out_vecp[i] = (*data)[i];
        return FERRIC_ECP_OK;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "ferric_ecp_matrix: %s\n", ex.what());
        return FERRIC_ECP_EINTERNAL;
    } catch (...) {
        return FERRIC_ECP_EINTERNAL;
    }
}

extern "C" int ferric_ecp_natoms(const ferric_ecp_gshell *shells, int nshell,
                                 const ferric_ecp_center *ecps, int necp) {
    if (nshell <= 0 || necp <= 0 || shells == nullptr || ecps == nullptr) {
        return FERRIC_ECP_EINVAL;
    }
    try {
        // Replicate ECPIntegrator::init's center deduplication exactly (same
        // 1e-4 Bohr L1 tolerance, same order: shells first, then ECP centers).
        // Doing it here rather than constructing an integrator keeps this cheap
        // enough to call for buffer sizing.
        std::vector<std::array<double, 3>> centers;
        auto intern = [&centers](double x, double y, double z) {
            for (const auto &c : centers) {
                const double diff = std::abs(c[0] - x) + std::abs(c[1] - y) +
                                    std::abs(c[2] - z);
                if (diff < 1e-4) return;
            }
            centers.push_back({x, y, z});
        };
        for (int s = 0; s < nshell; ++s) intern(shells[s].x, shells[s].y, shells[s].z);
        for (int e = 0; e < necp; ++e) intern(ecps[e].x, ecps[e].y, ecps[e].z);
        return static_cast<int>(centers.size());
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "ferric_ecp_natoms: %s\n", ex.what());
        return FERRIC_ECP_EINTERNAL;
    } catch (...) {
        return FERRIC_ECP_EINTERNAL;
    }
}

extern "C" int ferric_ecp_matrix_deriv(const ferric_ecp_gshell *shells, int nshell,
                                       const ferric_ecp_center *ecps, int necp,
                                       double *out_derivs, int *out_natoms) {
    if (nshell <= 0 || necp <= 0 || shells == nullptr || ecps == nullptr ||
        out_derivs == nullptr) {
        return FERRIC_ECP_EINVAL;
    }
    try {
        ECPIntegrator integrator;
        setup_integrator(integrator, shells, nshell, ecps, necp, /*deriv=*/1);
        integrator.compute_first_derivs();

        const int ncart = ferric_ecp_ncart(shells, nshell);
        const int natoms = integrator.natoms;
        if (out_natoms != nullptr) *out_natoms = natoms;

        // Cross-check against the standalone predictor the caller sized its
        // buffer with. A mismatch means the dedup logic here and in libecpint
        // have diverged, which would mean a buffer overrun -- fail, never write.
        const int predicted = ferric_ecp_natoms(shells, nshell, ecps, necp);
        if (predicted != natoms) {
            std::fprintf(stderr,
                "ferric_ecp_matrix_deriv: natoms mismatch (predicted %d, "
                "libecpint %d)\n", predicted, natoms);
            return FERRIC_ECP_EINTERNAL;
        }

        auto data = integrator.get_first_derivs();
        if (static_cast<int>(data.size()) != 3 * natoms) {
            std::fprintf(stderr,
                "ferric_ecp_matrix_deriv: got %zu deriv matrices, expected %d\n",
                data.size(), 3 * natoms);
            return FERRIC_ECP_EINTERNAL;
        }
        for (int c = 0; c < 3 * natoms; ++c) {
            if (!data[c] || static_cast<int>(data[c]->size()) != ncart * ncart) {
                std::fprintf(stderr,
                    "ferric_ecp_matrix_deriv: deriv %d size mismatch vs %d^2\n",
                    c, ncart);
                return FERRIC_ECP_EINTERNAL;
            }
            const std::vector<double> &m = *data[c];
            double *dst = out_derivs + static_cast<size_t>(c) * ncart * ncart;
            for (int i = 0; i < ncart * ncart; ++i) dst[i] = m[i];
        }
        return FERRIC_ECP_OK;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "ferric_ecp_matrix_deriv: %s\n", ex.what());
        return FERRIC_ECP_EINTERNAL;
    } catch (...) {
        return FERRIC_ECP_EINTERNAL;
    }
}
