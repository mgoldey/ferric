//! Shared by every U-RI-MP2 device test binary: an owned spin channel and the
//! CPU reference energies.
use ferric_mp2::u_rimp2::{opposite_spin_pair_kernel, same_spin_pair_kernel, SpinChannel};
use ndarray::Array2;

/// An owned spin channel.
pub struct Chan {
    pub b: Array2<f64>,
    pub eps: Vec<f64>,
    pub nocc: usize,
    pub nvir: usize,
    pub first_occ: usize,
    pub nocc_total: usize,
}

impl Chan {
    pub fn ch(&self) -> SpinChannel<'_> {
        SpinChannel {
            b: &self.b,
            eps: &self.eps,
            nocc: self.nocc,
            nvir: self.nvir,
            first_occ: self.first_occ,
            nocc_total: self.nocc_total,
        }
    }
}

fn one_thread() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
}

/// The CPU energy-only kernels inside a 1-thread rayon pool, where the device
/// dispatcher declines (it never runs inside a rayon worker): the CPU energy
/// whatever the GPU mode is.
pub fn cpu_same(c: &Chan) -> f64 {
    one_thread().install(|| same_spin_pair_kernel(c.ch(), false).0)
}

pub fn cpu_opp(a: &Chan, b: &Chan) -> f64 {
    one_thread().install(|| opposite_spin_pair_kernel(a.ch(), b.ch(), false).0)
}
