//! DIIS (Direct Inversion in the Iterative Subspace) convergence accelerator.
//!
//! Implements Pulay's DIIS extrapolation for accelerating SCF convergence.
//! The error vectors (FDS - SDF) are used to construct a least-squares
//! extrapolation of the Fock matrix.

use ndarray::Array2;

/// DIIS extrapolator with a fixed-size rolling subspace.
pub struct Diis {
    max_subspace: usize,
    fock_hist: Vec<Array2<f64>>,
    err_hist: Vec<Array2<f64>>,
}

impl Diis {
    /// Create a DIIS accelerator with the given maximum subspace size.
    pub fn new(max_subspace: usize) -> Self {
        Diis {
            max_subspace,
            fock_hist: Vec::new(),
            err_hist: Vec::new(),
        }
    }

    /// Clear the DIIS history.
    pub fn reset(&mut self) {
        self.fock_hist.clear();
        self.err_hist.clear();
    }

    /// Add a Fock matrix and error vector to the history, return the extrapolated Fock matrix.
    pub fn step(&mut self, f: &Array2<f64>, err: &Array2<f64>) -> Array2<f64> {
        self.fock_hist.push(f.clone());
        self.err_hist.push(err.clone());
        if self.fock_hist.len() > self.max_subspace {
            self.fock_hist.remove(0);
            self.err_hist.remove(0);
        }
        let m = self.fock_hist.len();
        if m < 2 {
            return f.clone();
        }
        let dim = m + 1;
        let mut a = vec![0.0f64; dim * dim];
        let mut rhs = vec![0.0f64; dim];
        for i in 0..m {
            for j in 0..m {
                a[i * dim + j] = dot(&self.err_hist[i], &self.err_hist[j]);
            }
            a[i * dim + m] = 1.0;
            a[m * dim + i] = 1.0;
        }
        rhs[m] = 1.0;
        let c = match solve_linear(a, rhs, dim) {
            Some(c) => c,
            None => return self.fock_hist.last().unwrap().clone(),
        };
        let shape = f.dim();
        let mut out = Array2::zeros(shape);
        for i in 0..m {
            out.scaled_add(c[i], &self.fock_hist[i]);
        }
        out
    }
}

fn dot(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn solve_linear(mut a: Vec<f64>, mut x: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut pivot = col;
        let mut max_val = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > max_val {
                max_val = v;
                pivot = r;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            x.swap(col, pivot);
        }
        for r in (col + 1)..n {
            let factor = a[r * n + col] / a[col * n + col];
            for k in col..n {
                a[r * n + k] -= factor * a[col * n + k];
            }
            x[r] -= factor * x[col];
        }
    }
    for r in (0..n).rev() {
        let mut s = x[r];
        for k in (r + 1)..n {
            s -= a[r * n + k] * x[k];
        }
        x[r] = s / a[r * n + r];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diis_one_iteration() {
        let mut d = Diis::new(8);
        let f = Array2::eye(3);
        let err = Array2::zeros((3, 3));
        let out = d.step(&f, &err);
        assert_eq!(out, f);
    }

    #[test]
    fn test_diis_two_iterations() {
        let mut d = Diis::new(8);
        let f1 = Array2::eye(3);
        let err1 = Array2::zeros((3, 3));
        d.step(&f1, &err1);
        let f2 = Array2::eye(3) * 2.0;
        let err2 = Array2::zeros((3, 3));
        let out = d.step(&f2, &err2);
        assert_eq!(out.dim(), (3, 3));
    }
}
