import numpy as np

def generate_exponential_grid(n_points, t_max):
    u = np.linspace(0, 1, n_points)
    return t_max * (np.exp(3*u) - 1) / (np.exp(3) - 1)

def compute_W_lk(tau_nodes, omega_nodes, eps_grid):
    n_tau = len(tau_nodes)
    n_omega = len(omega_nodes)
    n_eps = len(eps_grid)
    
    A = np.zeros((n_eps, n_tau))
    for m, eps in enumerate(eps_grid):
        for l, tau in enumerate(tau_nodes):
            A[m, l] = np.exp(-eps * tau)
            
    W = np.zeros((n_omega, n_tau))
    
    for k, omega in enumerate(omega_nodes):
        b = np.zeros(n_eps)
        for m, eps in enumerate(eps_grid):
            b[m] = eps / (omega**2 + eps**2)
            
        res, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        W[k, :] = res
        
    return W

def run_spike():
    n_tau = 12
    n_omega = 12
    t_max = 20.0
    
    # Mock grids
    tau_nodes = generate_exponential_grid(n_tau, t_max)
    # Omega nodes: log spaced
    omega_nodes = np.logspace(-2, 2, n_omega)
    
    # Dense epsilon grid for the fitting
    eps_grid = np.logspace(-2, 2, 500)
    
    print("Computing optimized W_{lk} weights via Least Squares...")
    W = compute_W_lk(tau_nodes, omega_nodes, eps_grid)
    
    # Test accuracy on a set of validation epsilons
    val_eps = [0.1, 1.0, 5.0, 20.0]
    
    print(f"\n{'Omega':>8} | {'Epsilon':>8} | {'Exact':>15} | {'Optimized W':>15} | {'Error':>15}")
    print("-" * 65)
    
    max_err = 0.0
    for k, omega in enumerate([omega_nodes[0], omega_nodes[n_omega//2], omega_nodes[-1]]):
        for eps in val_eps:
            exact = eps / (omega**2 + eps**2)
            
            # Use the optimized weights instead of cos(omega*tau)
            # sum_l W_{lk} * exp(-eps * tau_l)
            approx = 0.0
            for l, tau in enumerate(tau_nodes):
                approx += W[k, l] * np.exp(-eps * tau)
                
            err = abs(approx - exact)
            max_err = max(max_err, err)
            print(f"{omega:8.2f} | {eps:8.2f} | {exact:15.5e} | {approx:15.5e} | {err:15.5e}")
            
    print(f"\nMaximum validation error across sampled points: {max_err:.5e}")

if __name__ == '__main__':
    run_spike()
