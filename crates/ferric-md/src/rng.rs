//! Self-contained xoshiro256++ generator seeded via SplitMix64.
//!
//! Reference: D. Blackman and S. Vigna, "Scrambled linear pseudorandom number
//! generators", ACM TOMS 47 (2021); <https://prng.di.unimi.it/>. Gaussian
//! deviates use the Box-Muller transform, caching the second deviate.

/// One SplitMix64 step: advance `state` and return the next output.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// xoshiro256++ pseudo-random number generator.
#[derive(Debug, Clone)]
pub struct Xoshiro256pp {
    s: [u64; 4],
    spare_gaussian: Option<f64>,
}

impl Xoshiro256pp {
    /// Seed the four state words from `seed` with SplitMix64.
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Self::from_state(s)
    }

    /// Start from an explicit state (must not be all zero).
    pub fn from_state(s: [u64; 4]) -> Self {
        Self {
            s,
            spare_gaussian: None,
        }
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform double in `[0, 1)` from the top 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
        (self.next_u64() >> 11) as f64 * SCALE
    }

    /// Standard normal deviate (Box-Muller).
    pub fn next_gaussian(&mut self) -> f64 {
        if let Some(z) = self.spare_gaussian.take() {
            return z;
        }
        // u1 in (0, 1] keeps the logarithm finite.
        let u1 = 1.0 - self.next_f64();
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let phi = 2.0 * std::f64::consts::PI * u2;
        self.spare_gaussian = Some(r * phi.sin());
        r * phi.cos()
    }
}
