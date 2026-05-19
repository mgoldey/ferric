import json
import numpy as np

def get_tau_grid(n_points, e_min, e_max):
    # Using an exponential mapping for tau
    t_max = 15.0 / e_min
    u = np.linspace(0, 1, n_points)
    tau = t_max * (np.exp(3*u) - 1) / (np.exp(3) - 1)
    return tau

def compute_w_lk_svd(tau_points, omega_points, e_min, e_max):
    """
    Implements the SVD approach from the greenX library.
    transformation_type = cosine_wt: cos(tau) -> cos(omega)
    mat_A = cos(tau * omega) * psi(omega, x)
    Wait, the exact GreenX formulation for W_lk (which replaces cos) is:
    W_lk is the transformation from tau to omega.
    We want: sum_l W_lk exp(-x tau_l) \approx x / (omega_k^2 + x^2)
    In GreenX, the target is psi(x) = 2x / (x^2 + omega_k^2).
    mat_A(x, tau_l) = exp(-x tau_l)
    """
    n_points = len(tau_points)
    
    # 200 nodes per decade
    n_x_nodes = int(np.log10(e_max / e_min) + 1) * 200
    n_x_nodes = max(n_x_nodes, n_points)
    
    # log spacing
    x_factor = (e_max / e_min) ** (1.0 / (n_x_nodes - 1))
    x_mu = e_min * x_factor ** np.arange(n_x_nodes)
    
    # mat_A: shape (n_x_nodes, n_points)
    mat_A = np.zeros((n_x_nodes, n_points))
    for j in range(n_points):
        mat_A[:, j] = np.exp(-x_mu * tau_points[j])
        
    U, S, VT = np.linalg.svd(mat_A, full_matrices=False)
    
    W = np.zeros((len(omega_points), n_points))
    
    for k, omega in enumerate(omega_points):
        # target
        psi = (x_mu) / (x_mu**2 + omega**2)
        
        # solve: mat_A * w = psi
        # w = V S^-1 U^T psi
        UT_psi = U.T @ psi
        S_inv = np.zeros_like(S)
        # Use a small regularization
        reg = 1e-14
        for idx in range(len(S)):
            S_inv[idx] = S[idx] / (S[idx]**2 + reg**2)
            
        w = VT.T @ (S_inv * UT_psi)
        W[k, :] = w
        
    return W

def main():
    e_min = 0.1
    e_max = 100.0
    n_points = 12
    
    with open('/home/matt/qc/ferric/minimax_freq_grids.json', 'r') as f:
        freq_grids = json.load(f)
        
    grid_data = freq_grids[str(n_points)][0]
    omega_points = np.array(grid_data['nodes'])
    omega_weights = np.array(grid_data['weights'])
    
    tau_points = get_tau_grid(n_points, e_min, e_max)
    
    W = compute_w_lk_svd(tau_points, omega_points, e_min, e_max)
    
    joint = {
        "tau_points": tau_points.tolist(),
        "omega_points": omega_points.tolist(),
        "omega_weights": omega_weights.tolist(),
        "w_transform": W.flatten().tolist()
    }
    
    with open('/home/matt/qc/ferric/joint_minimax_N12.json', 'w') as f:
        json.dump(joint, f, indent=2)
        
    print("Generated joint_minimax_N12.json")
    
if __name__ == '__main__':
    main()
