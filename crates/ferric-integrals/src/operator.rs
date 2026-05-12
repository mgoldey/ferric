#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OperatorKind {
    Coulomb,
    ErfCoulomb,
    ErfcCoulomb,
    Yukawa,
}

#[derive(Debug, Clone, Copy)]
pub struct Operator {
    pub kind: OperatorKind,
    pub omega: f64,
    pub distance: f64,
}

impl Operator {
    pub fn coulomb() -> Self {
        Self { kind: OperatorKind::Coulomb, omega: 0.0, distance: 0.0 }
    }
    pub fn erf(omega: f64) -> Self {
        Self { kind: OperatorKind::ErfCoulomb, omega, distance: 0.0 }
    }
}
