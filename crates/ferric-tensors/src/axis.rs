//! Axis labels carried by [`crate::tensor::Tensor`] so contractions can be
//! checked for orbital-space consistency (e.g. catching an o<->v swap).

/// The orbital/index space an axis lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    /// Atomic-orbital basis index.
    Ao,
    /// Occupied (spin)orbital index.
    O,
    /// Virtual (spin)orbital index.
    V,
    /// Auxiliary (density-fitting) basis index.
    Aux,
    /// Generic / unlabeled axis (used when a space is not tracked).
    Any,
}

impl Axis {
    /// Short single-token name for debug/log messages.
    pub fn short(self) -> &'static str {
        match self {
            Axis::Ao => "ao",
            Axis::O => "o",
            Axis::V => "v",
            Axis::Aux => "P",
            Axis::Any => "?",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_is_copy_and_eq() {
        let a = Axis::V;
        let b = a; // Copy
        assert_eq!(a, b);
        assert_ne!(Axis::O, Axis::V);
    }

    #[test]
    fn axis_short_name() {
        assert_eq!(Axis::O.short(), "o");
        assert_eq!(Axis::Aux.short(), "P");
    }
}
