//! [`Tensor`]: a thin labeled wrapper over `ndarray::ArrayD<f64>`.
//!
//! The only addition over a bare ndarray is a small `[Axis; N]` of `Copy`
//! labels. `Tensor` `Deref`s to the inner `ArrayD` so all existing ndarray code
//! keeps working. Label checks are debug-only and compile out of release builds.

use crate::axis::Axis;
use ndarray::ArrayD;
use std::ops::Deref;

/// A dense f64 tensor of `N` axes, each tagged with an [`Axis`] label.
///
/// `N` is the label count and MUST equal the runtime ndim of `data`; the
/// constructor debug-asserts this.
#[derive(Debug, Clone)]
pub struct Tensor<const N: usize> {
    data: ArrayD<f64>,
    labels: [Axis; N],
}

impl<const N: usize> Tensor<N> {
    /// Wrap an `ArrayD` with axis labels. Debug-panics if `data.ndim() != N`.
    pub fn new(data: ArrayD<f64>, labels: [Axis; N]) -> Self {
        debug_assert_eq!(
            data.ndim(),
            N,
            "Tensor label count ({}) must equal ndim ({})",
            N,
            data.ndim()
        );
        Self { data, labels }
    }

    /// The axis labels.
    pub fn labels(&self) -> &[Axis; N] {
        &self.labels
    }

    /// The axis label at position `pos` (panics if out of range).
    pub fn axis_at(&self, pos: usize) -> Axis {
        self.labels[pos]
    }

    /// Borrow the inner ndarray view.
    pub fn view(&self) -> ndarray::ArrayViewD<'_, f64> {
        self.data.view()
    }

    /// Consume and return the inner `ArrayD`.
    pub fn into_inner(self) -> ArrayD<f64> {
        self.data
    }
}

impl<const N: usize> Deref for Tensor<N> {
    type Target = ArrayD<f64>;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Lets `einsum!` query an operand's axis label in debug builds. `Tensor`
/// returns `Some(label)`; anything else (bare ndarray) returns `None` and the
/// check is skipped.
pub trait MaybeLabeled {
    /// Axis label at `pos`, if this operand carries labels.
    fn axis_label(&self, pos: usize) -> Option<Axis>;
}

impl<const N: usize> MaybeLabeled for Tensor<N> {
    fn axis_label(&self, pos: usize) -> Option<Axis> { self.labels.get(pos).copied() }
}
impl<const N: usize> MaybeLabeled for &Tensor<N> {
    fn axis_label(&self, pos: usize) -> Option<Axis> { (**self).axis_label(pos) }
}
impl MaybeLabeled for ndarray::ArrayD<f64> {
    fn axis_label(&self, _pos: usize) -> Option<Axis> { None }
}
impl MaybeLabeled for &ndarray::ArrayD<f64> {
    fn axis_label(&self, _pos: usize) -> Option<Axis> { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axis::Axis;
    use ndarray::array;

    #[test]
    fn construct_and_label() {
        let a = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let t = Tensor::new(a, [Axis::O, Axis::V]);
        assert_eq!(t.labels(), &[Axis::O, Axis::V]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t[[0, 1]], 2.0);
    }

    #[test]
    #[should_panic(expected = "label count")]
    fn label_count_must_match_ndim() {
        let a = array![[1.0, 2.0]].into_dyn(); // 2D
        let _t = Tensor::new(a, [Axis::O]); // only 1 label -> panic in debug
    }

    #[test]
    fn into_inner_roundtrip() {
        let a = array![1.0, 2.0, 3.0].into_dyn();
        let t = Tensor::new(a.clone(), [Axis::V]);
        let back = t.into_inner();
        assert_eq!(back, a);
    }
}
