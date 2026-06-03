use ferric_tensors::{einsum, Axis, Tensor};
use ndarray::{Array, IxDyn};

fn t2(shape: &[usize], labels2: [Axis; 2]) -> Tensor<2> {
    let n: usize = shape.iter().product();
    let a = Array::from_shape_vec(IxDyn(shape), (0..n).map(|x| x as f64).collect()).unwrap();
    Tensor::new(a, labels2)
}

#[test]
fn macro_matmul() {
    let a = t2(&[2, 3], [Axis::O, Axis::Aux]); // "ik"
    let b = t2(&[3, 2], [Axis::Aux, Axis::V]); // "kj"
    let c: ndarray::ArrayD<f64> = einsum!("ik,kj->ij", &a, &b);
    let av = a.view(); let bv = b.view();
    let mut want = Array::zeros(IxDyn(&[2, 2]));
    for i in 0..2 { for j in 0..2 { let mut s=0.0; for k in 0..3 { s += av[[i,k]]*bv[[k,j]]; } want[[i,j]]=s; } }
    assert!((&c - &want).iter().all(|x| x.abs() < 1e-12));
}

#[test]
fn macro_scalar() {
    let a = t2(&[2, 2], [Axis::O, Axis::V]);
    let b = t2(&[2, 2], [Axis::O, Axis::V]);
    let s: f64 = einsum!("ij,ij->", &a, &b);
    let av=a.view(); let bv=b.view();
    let mut want=0.0; for i in 0..2 { for j in 0..2 { want += av[[i,j]]*bv[[i,j]]; } }
    assert!((s - want).abs() < 1e-12);
}

#[test]
fn macro_four_index_from_three() {
    use ndarray::Array;
    let p = 2; let n = 2;
    let arr = Array::from_shape_vec(IxDyn(&[p, n, n]), (0..p*n*n).map(|x| x as f64).collect()).unwrap();
    let b = Tensor::new(arr, [Axis::Aux, Axis::O, Axis::V]);
    let out: ndarray::ArrayD<f64> = einsum!("Pia,Pjb->iajb", &b, &b);
    let bv = b.view();
    let mut want = Array::zeros(IxDyn(&[n,n,n,n]));
    for i in 0..n {for a in 0..n {for j in 0..n {for bb in 0..n {
        let mut s=0.0; for pp in 0..p { s += bv[[pp,i,a]]*bv[[pp,j,bb]]; }
        want[[i,a,j,bb]] = s;
    }}}}
    assert!((&out - &want).iter().all(|x| x.abs() < 1e-12));
}

#[test]
#[should_panic(expected = "axis mismatch")]
fn macro_axis_mismatch_panics_in_debug() {
    use ndarray::Array;
    // 'k' contracted: Aux in `a` but V in `b` -> mismatch.
    let a = Tensor::new(
        Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).map(|x| x as f64).collect()).unwrap(),
        [Axis::O, Axis::Aux]); // "ik", k=Aux
    let b = Tensor::new(
        Array::from_shape_vec(IxDyn(&[3, 2]), (0..6).map(|x| x as f64).collect()).unwrap(),
        [Axis::V, Axis::O]);   // "kj", k=V -> mismatch
    let _c: ndarray::ArrayD<f64> = einsum!("ik,kj->ij", &a, &b);
}
