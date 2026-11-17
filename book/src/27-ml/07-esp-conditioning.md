# A worked consumer: esp-conditioning {#esp-conditioning}

<!-- status: edit; registry: pdep-rpa; theory: none -->

The earlier sections describe the interface from ferric's side. This one
reads it from the other side, through one real consumer. esp-conditioning is
a separate project that builds training data for ESP-conditioned molecular
models from ferric's NPZ bundle: the model is given the electrostatic
potential at the nuclei, charges and per-atom polarizabilities as
conditioning inputs. It is described here as an example of the interface,
read from its code. Its results are not part of this book.

## The pipeline {#esp-conditioning-pipeline}

| Step | What the builder does | ferric side |
|---|---|---|
| 1 | SMILES → RDKit ETKDGv3 conformer → MMFF94 relaxation → centre of mass at the origin, positions in Å | none |
| 2 | writes a TOML: `kind = "pdep-rpa"`, `[scf]` `df_j_aux` and `df_k_aux` = `def2-universal-jkfit`, `[rpa]` `auxbasis`, `n_quad = 20`, Gauss–Legendre, `trunc_thresh = 1e-4`, `export_npz`, and `compute_esp`, `compute_polarizability`, `compute_alpha_atomic`, `compute_electric_field`, `compute_density_matrix`, `compute_hirshfeld_charges`, `compute_lowdin_charges` on | the CLI of section @sec:npz-bundle-turning-keys-off |
| 3 | runs `ferric-cli` as a subprocess with `OPENBLAS_NUM_THREADS=1` | section @sec:reproducibility-and-grades-bitwise-reproducibility |
| 4 | loads the NPZ, renames `esp_atoms` → `esp_at_nuclei` and `alpha_atomic` → `polarizability_tensor`, casts arrays to float32 | the keys of @sec:npz-bundle-keys |
| 5 | parses the total energy from the CLI's stdout line `Total (RHF+RPA):` | the NPZ has no energy (@sec:npz-bundle-layout) |
| 6 | computes CM5 charges from the **Löwdin** charges, with $\alpha = 2.474$ Å⁻¹, Cordero radii and the standard $D_Z$ and $D_{ZZ'}$ tables | `lowdin_charges` |
| 7 | writes HDF5 shards, one group `sample_<id>` per molecule | — |

Step 6 is a choice between baselines. CM5 adds a pairwise correction
$\sum_{B\ne A}D_{Z_AZ_B}\exp\!\big(-\alpha(r_{AB}-R_{Z_A}-R_{Z_B})\big)$ to a
population-analysis charge. CM5 was parameterized on Hirshfeld charges, but
the builder's docstring gives three reasons for Löwdin instead. Löwdin needs
no proatom shape, and the single-exponential Slater proatoms (ferric's
fallback Hirshfeld model, section @sec:electrostatics-and-charges-five-charge-schemes) misallocate density in C–O bonds
and similar heteropolar pairs. Löwdin conserves the electron count exactly,
$\mathrm{Tr}(S^{1/2}DS^{1/2}) = N_e$. And it is more stable under a change of
basis than Mulliken.

## What the example shows about the interface {#esp-conditioning-example-shows-about-interface}

**The energy is scraped from text.** Step 5 exists because the NPZ carries no
energy, and for `pdep-rpa` the CLI's JSON run log has no `result` record
either: its `run_end` holds the SCF reference energy only
(`site/src/using/run-logs.md`). Scraping stdout is therefore the only CLI
route today. Calling `run_pdep_rpa` from Python instead returns the energy
and the arrays together.

**The certificate is dropped.** The HDF5 schema keeps no certificate. The
shard-level provenance records a ferric git hash, but it is the `HEAD` of the
checkout that holds the binary at packing time, with no dirty flag, so it
need not be the commit the binary was built from. No sample records the
reference J/K route or the approximations that produced it. Keeping the
`certificate` bytes per sample costs little, and its `build` field is the
commit baked in at compile time (section @sec:python-cube-provenance-certificate-provenance-label).

**Settings outside the anchor.** `trunc_thresh = 1e-4` and the 20-point grid
put these runs outside the `pdep-rpa` anchor scope, so their certificates
would carry no `measured_error` (section @sec:python-cube-provenance-certificate-provenance-label). This does not affect the
reference-density features (ESP, field, charges), which do not depend on the
RPA settings. It does affect the energy.

**Names that drifted.** The renamed `polarizability_tensor` is ferric's Becke
Krishtal per-atom α, and the CM5 baseline is Löwdin in the code. Comments in
the builder call the first "Hirshfeld" and the second "Hirshfeld baseline".
Read the code, not the comments.

**Float32 is applied after the fact.** The arrays are computed in f64 and
cast on write, which is safe for the features; the precision loss is the
consumer's choice, not ferric's.

**Geometry.** The builder recentres the conformer before ferric sees it, and
ferric does not reorient (@sec:npz-bundle-keys). The stored geometry is therefore
centre-of-mass-centred in RDKit's arbitrary orientation, which is the frame
the lab-frame vectors and tensors (`electric_field`, α) refer to. An
equivariant model can use them as is; an invariant one needs only the
scalars.

## Sources

| Source | Used for |
|---|---|
| the esp-conditioning builder code (TOML generation, NPZ loading, CM5 post-processing, HDF5 schema, shard provenance) | every step in @sec:esp-conditioning-pipeline and the provenance hash |
| `crates/ferric-export/src/ml.rs`, `crates/ferric-rpa/src/property_bundle.rs` | the ferric side of each step |
| `site/src/using/run-logs.md`, `crates/ferric-cli/src/lib.rs` | why step 5 has no structured alternative in the CLI |

Changes from the sources: none; the project's comments are corrected in the
text where they disagree with its code.
