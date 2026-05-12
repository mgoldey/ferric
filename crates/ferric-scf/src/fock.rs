use ferric_core::FerricError;
use ndarray::Array2;

pub trait JBuilder {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<(), FerricError>;
    fn reset(&mut self);
}

pub trait KBuilder {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<(), FerricError>;
    fn update_density(&mut self, d: &Array2<f64>);
    fn reset(&mut self);
}

pub struct FockBuilder {
    pub hcore: Array2<f64>,
    pub j: Box<dyn JBuilder>,
    pub k: Box<dyn KBuilder>,
}

impl FockBuilder {
    pub fn build(&mut self, d: &Array2<f64>, f: &mut Array2<f64>) -> Result<(), FerricError> {
        let n = d.nrows();
        let mut j = Array2::zeros((n, n));
        let mut k = Array2::zeros((n, n));
        self.j.build(d, &mut j)?;
        self.k.build(d, &mut k)?;
        f.assign(&(&self.hcore + &j - &(0.5 * &k)));
        Ok(())
    }
}
