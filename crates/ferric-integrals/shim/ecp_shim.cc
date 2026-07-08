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
#include <cstdio>

using libecpint::ECPIntegrator;

static int ncart_for_l(int l) { return ((l + 1) * (l + 2)) / 2; }

extern "C" int ferric_ecp_ncart(const ferric_ecp_gshell *shells, int nshell) {
    int n = 0;
    for (int s = 0; s < nshell; ++s) n += ncart_for_l(shells[s].l);
    return n;
}

extern "C" int ferric_ecp_matrix(const ferric_ecp_gshell *shells, int nshell,
                                 const ferric_ecp_center *ecps, int necp,
                                 double *out_vecp) {
    if (nshell <= 0 || necp <= 0 || shells == nullptr || ecps == nullptr ||
        out_vecp == nullptr) {
        return FERRIC_ECP_EINVAL;
    }
    try {
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

        ECPIntegrator integrator;
        integrator.set_gaussian_basis(nshell, g_coords.data(), g_exps.data(),
                                      g_coefs.data(), g_ams.data(), g_lengths.data());
        integrator.set_ecp_basis(necp, u_coords.data(), u_exps.data(),
                                 u_coefs.data(), u_ams.data(), u_ns.data(),
                                 u_lengths.data());
        integrator.init(0);
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
