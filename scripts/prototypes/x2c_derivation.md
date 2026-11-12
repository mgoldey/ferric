# Spin-free X2C-1e: equations of the prototype

`x2c_proto.py` builds the spin-free exact-two-component core Hamiltonian
(Dyall, JCP 106, 9618 (1997); Kutzelnigg and Liu, JCP 123, 241102 (2005);
Liu and Peng, JCP 131, 031104 (2009)) from ferric's one-electron blocks and
runs it through `run_rhf(fock_hook=)`.

## Blocks

In a basis {m} of real Gaussians, with V(r) = -sum_C Z_C / |r - R_C|:

| Block | Definition | ferric | PySCF |
|---|---|---|---|
| S | <m\|n> | `AoIntegrals.overlap()` | `int1e_ovlp` |
| T | <m\|-1/2 nabla^2\|n> | `AoIntegrals.kinetic()` | `int1e_kin` |
| V | <m\|V\|n> | `AoIntegrals.nuclear()` | `int1e_nuc` |
| W | sum_a <d_a m\|V\|d_a n> | `AoIntegrals.pvp()` | `int1e_pnucp` |

W is the spin-free part of <sigma.p m| V |sigma.p n>. The Rust function
`ferric_integrals::pvp::pvp` sums the three mixed bra-centre/ket-centre
blocks of the nuclear-attraction second derivative: a function centred at A
satisfies d m / d A_a = -d_a m, so d^2 <m|V|n> / dA_a dB_a = <d_a m|V|d_a n>.

## Modified Dirac equation

With the small component written as (sigma.p / 2c) times a pseudo-large
function, the one-electron Dirac equation in the basis {m} for both
components is the generalized eigenproblem

    [ V    T            ] [C_L]   [ S   0        ] [C_L]
    [ T    W/(4c^2) - T ] [C_S] = [ 0   T/(2c^2) ] [C_S] E

of dimension 2n. Its n highest eigenvalues are the electronic states, the n
lowest the positronic ones (below -2c^2). The electronic columns define the
decoupling matrix X = C_S C_L^-1.

## Decoupling and renormalization

The electronic block of the block-diagonalized Hamiltonian, in the large
component basis, is

    L = V + T X + X^T T - X^T T X + X^T W X / (4c^2),

with metric S~ = S + X^T T X / (2c^2). The renormalization

    R = S^-1/2 (S^-1/2 S~ S^-1/2)^-1/2 S^1/2

satisfies R^T S~ R = S, so h_X2C = R^T L R is a Hamiltonian in the original
metric S, and its generalized eigenvalues with S are the electronic Dirac
eigenvalues of the matrix problem above.

## Picture change

Only the one-electron operator is transformed (the 1e approximation). The
two-electron interaction and every property operator stay in the
nonrelativistic picture.

## Uncontracted basis and contraction back

The decoupling needs a basis flexible enough to represent the small
component, so X, L and R are built in the uncontracted basis: one shell per
unique exponent and angular momentum of the orbital basis (PySCF's
`decontract_basis`, the `xuncontract = True` default). The contracted basis
functions are exact linear combinations of the uncontracted ones,
mu = sum_k C_k,mu chi_k, where C carries the contraction coefficients
(which multiply unit-normalized primitives) divided by the norm of the
contraction, sqrt(c^T S_l c), and S_l(a, b) = (2 sqrt(ab)/(a + b))^(l + 3/2)
is the same-centre primitive overlap. Then

    h_X2C(contracted) = C^T h_X2C(uncontracted) C.

The `contract` anchor checks C^T S C, C^T T C and C^T V C against ferric's
contracted S, T, V.

## Feeding the SCF through a constant hook

`ferric.run_rhf(fock_hook=hook)` calls `hook(D)` with the total density
every iteration, adds the returned V to the Fock matrix, and adds the
returned E to the energy. The energy expression in
`ferric_scf::rhf` is

    E = 1/2 Tr[D (h + F)] + E_xc + E_hook + ... + E_nn,

where F is the Fock matrix before the hook's V is added (the trace is
evaluated before `apply_fock_hook`). For a constant one-electron term
dH = h_X2C - (T + V) the hooked run must be the plain run with h replaced
by h + dH:

    1/2 Tr[D (h + dH + F + dH)] = 1/2 Tr[D (h + F)] + Tr[D dH],

so the hook returns `(Tr[D dH], dH)`. Returning `1/2 Tr[D dH]` would count
half the correction.

The SCF guess is built without the hook (the core or SAD guess of the
nonrelativistic Hamiltonian), and the hooked solve converges from it.

## The c -> infinity limit, and what it cannot detect

As c -> infinity, X -> 1 + O(c^-2) and h_X2C -> T + V with a leading term
of order c^-2, so ||h_X2C(c) - (T + V)|| falls by 100 per decade of c. The
`cscan` anchor checks this. The limit is blind to the sign of W: with -W
the leading correction still scales as c^-2 (only its coefficient changes),
so `--mutate flip-w` passes `cscan` and only the comparison with PySCF's
`int1e_pnucp`, its X2C core Hamiltonian and its energy correction detects it.

## Matched settings for the PySCF comparison

`scf.RHF(mol).sfx2c1e()` with `approx = '1e'`, `xuncontract = True`,
`basis = None`, `c = lib.param.LIGHT_SPEED = 137.03599967994`. The PySCF
molecule is built from ferric's bundled cc-pVDZ JSON (the same exponents and
coefficients), spherical d, Cartesian p as in ferric. Both SCFs use
`energy_conv` (ferric) and `conv_tol` (PySCF) of 1e-10.

## Measured (2026-11-12, cc-pVDZ, ferric commit 48b0151ad + binding)

Uncontracted sizes: Ne 26, Ar 41, Kr 77, HBr 84 AOs. HBr at 1.414 Angstrom.

| System | pVp eig, max \|d\| (largest \|eig\|) | X2C hcore eig, max \|d\| | E_X2C - E_NR (ferric) | ddE vs PySCF |
|---|---|---|---|---|
| Ne | 1.5e-8 (1.06e8) | 2.4e-11 | -0.1269415624 | 3.6e-11 |
| Ar | 1.9e-6 (4.42e9) | 2.6e-11 | -1.5849415706 | 1.0e-10 |
| Kr | 7.6e-6 (8.94e10) | 2.2e-11 | -27.6482999025 | 2.9e-11 |
| HBr | 1.3e-5 (7.91e10) | 4.4e-11 | -24.6567514471 | 1.6e-10 |

The pVp eigenvalues agree to 1.4e-16 to 1.7e-16 relative to the largest
eigenvalue. The eigenvalue bars (`gen_eig_bar`, first-order perturbation
bound with 1/lambda_min(S)) are loose: the measured deviations sit at 1e-8
to 3e-3 of them. The energy bar is the sum of the four SCF energy_conv
values, 4e-10 Ha.

`cscan` ratios ||dH(c f)|| / ||dH(10 c f)|| for f = 1, 10, 100: Ne 92.9,
99.84, 99.998; Kr 76.5, 98.15, 99.970. With -W (`--mutate flip-w`) the ratios
are also 100 to 2e-4 (Kr 98.2, 100.32, 100.007), so the c-limit passes,
while the X2C hcore eigenvalues miss PySCF's by 0.24 (Ne) to 40 Ha (Kr) and
E_X2C - E_NR changes sign (Kr +90.45 Ha against -27.65 Ha).

A hook returning (0, 0) gives the plain run bitwise (energy, iteration count,
density) on all four systems.
