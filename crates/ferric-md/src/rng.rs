//! Self-contained xoshiro256++ generator seeded via SplitMix64.

/// xoshiro256++ pseudo-random number generator.
#[derive(Debug, Clone)]
pub struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    /// Seed the four state words from `seed` with SplitMix64.
    pub fn seed_from_u64(seed: u64) -> Self {
        let _ = seed;
        todo!()
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let _ = &self.s;
        todo!()
    }

    /// Uniform double in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        todo!()
    }

    /// Standard normal deviate (Box-Muller).
    pub fn next_gaussian(&mut self) -> f64 {
        todo!()
    }
}
