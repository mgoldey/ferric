//! Intrinsic reaction coordinate: which two minima does this saddle connect?
//!
//! A transition-state search plus `n_imaginary == 1` establishes that a
//! geometry IS a first-order saddle. It does not establish that it is YOUR
//! saddle. A molecule of any size has many; the one the search found is
//! whichever was uphill from where you started, and a methyl rotor gives one
//! imaginary mode just as a bond-breaking coordinate does.
//!
//! The IRC answers the remaining question by construction: displace along the
//! imaginary mode in each direction and follow the gradient downhill. Where
//! you land IS the pair of minima the barrier separates.
//!
//! # What this implements, and what it does not
//!
//! **Mass-weighted steepest descent**, which is the definition of the IRC
//! (Fukui). Steps are taken in mass-weighted Cartesians `q = sqrt(m) x`, so
//! the path is the one a classical trajectory with infinitesimal kinetic
//! energy would follow. Unweighted steepest descent gives a different path and
//! is not an IRC, so the weighting is not an optional refinement.
//!
//! NOT implemented: higher-order path integrators (IRC/Gonzalez-Schlegel,
//! LQA). Those take larger steps for the same path accuracy; this takes small
//! ones. For the question "which minima does it connect?" the endpoint is what
//! matters and the path shape between them is not used, so a simple integrator
//! is honest rather than limiting. If a future caller needs the path LENGTH or
//! the reaction-coordinate profile with quantitative spacing, it needs a
//! better integrator and this docstring should stop being reassuring.
//!
//! NOT implemented: automatic detection that an endpoint IS a minimum. The
//! walk stops when the gradient falls below a threshold or the step budget
//! runs out, and `IrcBranch::converged` reports which. Confirming the endpoint
//! is a minimum needs a Hessian there, which costs another 6N+1 gradients per
//! side -- the caller decides whether that is worth it.

use crate::frequencies::atom_masses;
use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ndarray::Array1;

/// Step-length cap as a multiple of the mass-weighted gradient norm.
///
/// Makes the step shrink with the gradient so the walk terminates in a basin
/// rather than oscillating across it. Large enough that it does not bind on
/// the steep part of the path, where `IrcConfig::step` should govern.
const DAMPING: f64 = 2.0;

/// How far to step, and when to stop.
#[derive(Debug, Clone)]
pub struct IrcConfig {
    /// Step length along the mass-weighted path, in sqrt(amu)*Bohr.
    ///
    /// Smaller is a more faithful path and more gradient evaluations. 0.1 is
    /// small enough that the endpoint is insensitive to it on the systems
    /// tested; see `the_endpoints_do_not_depend_on_the_step_size`.
    pub step: f64,
    /// Maximum steps per direction.
    pub max_steps: usize,
    /// Stop when the maximum gradient component falls below this
    /// (Hartree/Bohr). The default is looser than a geometry optimizer's
    /// because an IRC only has to identify WHICH basin it fell into, not
    /// locate the minimum precisely.
    pub g_max_thresh: f64,
    /// Initial displacement from the saddle along the imaginary mode, in
    /// mass-weighted units. Must be non-zero: AT the saddle the gradient is
    /// zero by definition, so a walk started exactly there never moves.
    pub initial_displacement: f64,
}

impl Default for IrcConfig {
    fn default() -> Self {
        Self {
            step: 0.1,
            max_steps: 200,
            g_max_thresh: 1e-4,
            initial_displacement: 0.2,
        }
    }
}

/// One direction of the walk.
#[derive(Debug, Clone)]
pub struct IrcBranch {
    /// Where the walk ended.
    pub mol: Molecule,
    /// Energy there.
    pub energy: f64,
    /// Steps taken.
    pub steps: usize,
    /// `true` if the gradient threshold was met, `false` if the step budget ran
    /// out first. A `false` here means the endpoint is NOT a minimum and the
    /// branch identifies no basin -- it is a partial path.
    pub converged: bool,
}

/// Both directions, plus the saddle they came from.
#[derive(Debug, Clone)]
pub struct IrcResult {
    /// The `+mode` direction.
    pub forward: IrcBranch,
    /// The `-mode` direction.
    pub reverse: IrcBranch,
    /// Energy at the saddle, for barrier heights against each endpoint.
    pub saddle_energy: f64,
}

impl IrcResult {
    /// Barrier from the forward endpoint up to the saddle, in Hartree.
    ///
    /// Negative would mean the "endpoint" is higher than the saddle, which is
    /// impossible on a correct downhill walk and indicates the path left the
    /// intended surface.
    #[must_use]
    pub fn forward_barrier(&self) -> f64 {
        self.saddle_energy - self.forward.energy
    }

    /// Barrier from the reverse endpoint up to the saddle, in Hartree.
    #[must_use]
    pub fn reverse_barrier(&self) -> f64 {
        self.saddle_energy - self.reverse.energy
    }

    /// Did both directions reach a basin?
    ///
    /// Only then does the pair of endpoints answer "which two minima does this
    /// saddle connect?". One unconverged branch means one side is unidentified,
    /// and the result must not be read as a reaction.
    #[must_use]
    pub fn both_converged(&self) -> bool {
        self.forward.converged && self.reverse.converged
    }
}

/// Follow the IRC downhill from a saddle in both directions.
///
/// `mode` is the imaginary-mode eigenvector in CARTESIAN coordinates, as
/// `SaddleResult::imaginary_mode` provides it -- a flat `3N` vector. It is
/// mass-weighted and normalised internally, so its input scale does not
/// matter, but its DIRECTION does: `forward` is `+mode`.
///
/// `energy_gradient` is the same closure `find_saddle` takes.
pub fn follow_irc(
    saddle: &Molecule,
    mode: &Array1<f64>,
    config: &IrcConfig,
    mut energy_gradient: impl FnMut(&Molecule) -> Result<(f64, Array1<f64>), FerricError>,
) -> Result<IrcResult, FerricError> {
    let n3 = saddle.atoms.len() * 3;
    if mode.len() != n3 {
        return Err(FerricError::General(format!(
            "IRC: the imaginary mode has {} components but the molecule has {} \
             atoms ({n3} Cartesian coordinates)",
            mode.len(),
            saddle.atoms.len()
        )));
    }
    if !(config.step > 0.0 && config.initial_displacement > 0.0) {
        return Err(FerricError::General(format!(
            "IRC: step ({}) and initial_displacement ({}) must both be positive. \
             At the saddle the gradient is ZERO by definition, so a walk started \
             with no displacement never moves and would report the saddle itself \
             as both endpoints.",
            config.step, config.initial_displacement
        )));
    }

    let masses = atom_masses(saddle)?;
    // sqrt(m) per Cartesian component, so q = sqrt(m) x.
    let sm: Vec<f64> = masses
        .iter()
        .flat_map(|m| std::iter::repeat_n(m.sqrt(), 3))
        .collect();

    let (e_saddle, _) = energy_gradient(saddle)?;

    let forward = walk(saddle, mode, &sm, config, 1.0, &mut energy_gradient)?;
    let reverse = walk(saddle, mode, &sm, config, -1.0, &mut energy_gradient)?;

    Ok(IrcResult {
        forward,
        reverse,
        saddle_energy: e_saddle,
    })
}

/// One direction. `sign` is +1 or -1 on the mode.
fn walk(
    saddle: &Molecule,
    mode: &Array1<f64>,
    sm: &[f64],
    config: &IrcConfig,
    sign: f64,
    energy_gradient: &mut impl FnMut(&Molecule) -> Result<(f64, Array1<f64>), FerricError>,
) -> Result<IrcBranch, FerricError> {
    let n3 = sm.len();

    // Mass-weight the mode and normalise it there. Normalising in CARTESIANS
    // instead would make the initial displacement depend on which atoms the
    // mode happens to move -- a mode on hydrogens would step much further in
    // mass-weighted space than one on carbons, for the same nominal length.
    let mut q_mode: Vec<f64> = (0..n3).map(|i| mode[i] * sm[i]).collect();
    let norm = q_mode.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= 1e-12 {
        return Err(FerricError::General(
            "IRC: the imaginary mode is zero after mass weighting; there is no \
             direction to follow"
                .to_string(),
        ));
    }
    for v in &mut q_mode {
        *v = *v / norm * sign;
    }

    // Start displaced OFF the saddle: the gradient there is zero.
    let mut q: Vec<f64> = (0..n3)
        .map(|i| {
            let x = cart(saddle, i);
            x * sm[i] + config.initial_displacement * q_mode[i]
        })
        .collect();

    let mut mol = saddle.clone();
    let mut energy = 0.0;
    let mut converged = false;
    let mut steps = 0;

    for _ in 0..config.max_steps {
        set_cart_from_mass_weighted(&mut mol, &q, sm);
        let (e, g) = energy_gradient(&mol)?;
        energy = e;
        steps += 1;

        let g_max = g.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        if g_max < config.g_max_thresh {
            converged = true;
            break;
        }

        // Mass-weighted gradient: dE/dq_i = (dE/dx_i)/sqrt(m_i).
        let gq: Vec<f64> = (0..n3).map(|i| g[i] / sm[i]).collect();
        let gn = gq.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gn <= 1e-14 {
            converged = true;
            break;
        }
        // Step downhill, DAMPED as the gradient falls.
        //
        // A fixed-length step cannot settle into a basin: near the minimum it
        // overshoots and the walk oscillates across it forever. MEASURED on
        // NH3 inversion at `g_max_thresh = 1e-4`: step 0.15 and step 0.05 both
        // ran to the step budget (400 and 800) without converging, while step
        // 0.02 converged in 63. That is not a threshold to loosen -- it is the
        // integrator failing to terminate.
        //
        // Capping the step at a fraction of |g| makes the displacement shrink
        // with the gradient, so the walk decelerates into the basin instead of
        // orbiting it. Far from the minimum `config.step` still governs and
        // the path is unchanged; the cap only binds at the end, where the path
        // SHAPE no longer matters because the endpoint is what is being
        // identified.
        let len = config.step.min(DAMPING * gn);
        for i in 0..n3 {
            q[i] -= len * gq[i] / gn;
        }
    }

    set_cart_from_mass_weighted(&mut mol, &q, sm);
    Ok(IrcBranch {
        mol,
        energy,
        steps,
        converged,
    })
}

fn cart(mol: &Molecule, i: usize) -> f64 {
    let a = &mol.atoms[i / 3];
    match i % 3 {
        0 => a.x,
        1 => a.y,
        _ => a.zpos,
    }
}

fn set_cart_from_mass_weighted(mol: &mut Molecule, q: &[f64], sm: &[f64]) {
    for (k, atom) in mol.atoms.iter_mut().enumerate() {
        atom.x = q[3 * k] / sm[3 * k];
        atom.y = q[3 * k + 1] / sm[3 * k + 1];
        atom.zpos = q[3 * k + 2] / sm[3 * k + 2];
    }
}
