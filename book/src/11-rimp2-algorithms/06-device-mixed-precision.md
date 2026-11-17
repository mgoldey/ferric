<!-- section: 11.6; status: ready; registry: ri-mp2;
     theory: none -->

# 11.6 Device and mixed-precision kernels

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/rimp2_gpu.rs::try_spin_components_on_device`, `ferric-mp2/src/rimp2.rs::mixed_energy_operator` |
| Build | the `gpu` cargo feature; a default build has no device code |
| CLI | `[gpu] mode` or `preset`; `[precision] mode = "mixed"` with `rimp2-energy` in `mixed_kernels` (`site/src/reference/input.md`) |
| Grade | f64 device: within a summation-order bound of the CPU. Mixed: a measured error map, not Proven (`site/src/reference/validation.md`, "Mixed-precision GPU kernels") |

Section 11.3 put the $O(N^5)$ work into one GEMM per occupied orbital. A GPU
does large GEMMs faster than the CPU, and in single precision faster still.
The device path replaces exactly that GEMM and nothing else, so the
question for each variant is how far its $G^{(i)}$ can move from the CPU's.
The plan: when the device runs, the f64 path and its summation-order bound,
the mixed path and its element bound, then the measured errors.

The device replaces one operation: the GEMM that forms $G^{(i)}$
(equation 11.7). Everything else is shared with the CPU path of section 11.3:
the pair arithmetic is `pair_energy`, the $j$ fold is in ascending order, and
the $i$ fold is in ascending order. The device and CPU energies therefore
differ only by the rounding of the GEMM.

## 11.6.1 When the device runs

`ri_mp2_spin_components` calls the dispatcher first. It returns `None`, and
the CPU path runs, when any of these holds:

| Condition | Counted reason |
|---|---|
| `[gpu] mode = "off"` (the default) | none |
| no usable device | none |
| called from inside a rayon worker | `InsideRayonWorker` |
| no device memory pool | none |
| the device cannot be opened | `CudaError` |
| a device error other than those below (pool full, layout, CUDA) | counted, then the CPU path |

$B_{ov}$ is resident on the device for the whole stage. For each $i$ the
device forms $G^{(i)}$ on the calling thread and copies it to the host. The
host pair loop for that $i$ runs in parallel over $j$ with disjoint outputs
and is then folded in ascending $j$, which is bit-identical to the serial
fold for any worker count
(`pair_fold_is_bit_identical_for_every_worker_count_and_run_to_run` in
`crates/ferric-mp2/tests/gpu_rimp2_resident.rs`).

The device holds, from `Stage::new` in `crates/ferric-mp2/src/rimp2_gpu.rs`:

| Buffer | f64 path | Mixed path |
|---|---|---|
| resident $B_{ov}$ | $8 N_{\mathrm{aux}} n_o n_v$ | $4 N_{\mathrm{aux}} n_o n_v$ |
| $G^{(i)}$ scratch, sized for $i = 0$ | $8 n_v \cdot n_o n_v$ | $8 n_v \cdot n_o n_v$ |
| f32 panel scratch | none | $4 n_v \cdot n_o n_v$ |

For benzene in aug-cc-pVTZ with a frozen core ($N_{\mathrm{aux}} = 912$,
$n_o = 15$, $n_v = 393$) that is 43.0 MB of $B_{ov}$ in f64 or 21.5 MB in
f32, beside 18.5 MB of $G^{(i)}$ scratch. The scratch is reserved first, so
a pool too small for it refuses before the larger upload.

## 11.6.2 The f64 device path

The device and the CPU compute the same sums $\sum_P B^P_{ia}B^P_{jb}$ in a
different order. With $S_{ab} = \sum_P |B^P_{ia}|\,|B^P_{jb}|$, the
difference of two summation orders of $N_{\mathrm{aux}}$ terms is bounded per
element by

$$ \big| G^{\mathrm{dev}}_{ab} - G^{\mathrm{cpu}}_{ab} \big| \le 2\,\gamma_{N_{\mathrm{aux}}}(u_{64})\, S_{ab}, \qquad \gamma_n(u) = \frac{n u}{1 - n u}, \qquad u_{64} = 2^{-53} . \tag{11.10} $$

> **Unverified equation 11.10** · the summation-order bound
>
> Each of the two dot products is within $\gamma_{N_{\mathrm{aux}}}S_{ab}$ of
> the exact sum (Higham, *Accuracy and Stability of Numerical Algorithms*,
> section 3.1), so their difference is within twice that.
>
> | | |
> |---|---|
> | Proof | none committed. The bound is the standard dot-product error bound. To add: a numpy claim that sums random products in two orders at extended precision and checks the bound, with a negative control that sums in f32. |
> | Code tests | `per_element_g_blocks_match_the_cpu_within_two_gamma_s` and `device_f64_energy_is_within_the_derived_bound_and_the_gate_has_teeth` in `crates/ferric-mp2/tests/gpu_rimp2_resident.rs` (features `gpu` and `test-seams`): measured difference at most $1.7 \times 10^{-4}$ (OS) and $1.9 \times 10^{-4}$ (SS) of the bound; an f32 product on the CPU exceeds it, and the mutations offset and `fac = 1` fail |
>
> <!-- unverified: id=rimp2alg.device-f64-bound; todo=add summation-order bound claim to a scripts/ proof file -->

The bound is the triangle inequality applied to the exact sum: each side is
within $\gamma_{N_{\mathrm{aux}}}S_{ab}$ of it. At
$N_{\mathrm{aux}} = 912$ the factor is $2\gamma_{912}(u_{64}) =
2.0 \times 10^{-13}$. It scales with $S_{ab}$, not with $|G_{ab}|$, because
$G_{ab}$ can be much smaller than its terms when they cancel.

`kappa_path_takes_the_device_within_its_derived_bound` in the same file pins
that a $\kappa$-regularized run also takes the f64 device path.

## 11.6.3 The mixed-precision path

The mixed path runs only when all of these hold:

- `[precision] mode = "mixed"` with `rimp2-energy` in `mixed_kernels`;
- `kappa` is unset;
- `mixed_energy_operator(op)` is true: the Coulomb, erfc and terfc
  operators, primitive only (erf and fitted composite operators run in f64).

The scheme:

| Part | Precision |
|---|---|
| $B_{ov}$ on the device | f32, each element rounded to nearest once |
| products inside a panel of $b = 64$ auxiliary rows | f32 |
| sum over the $\lceil N_{\mathrm{aux}}/b\rceil$ panels | f64 |
| pair arithmetic, denominators, all energy sums | f64 |

In `ferric-core/src/gpu/mixed.rs::gemm_f32_f64acc_dev_into` each panel is one
`cublasSgemm` with $\beta = 0$ into an f32 scratch, followed by a kernel that
adds the scratch into the f64 result. The f32 rounding errors therefore
accumulate over $b$ terms at most; the panels are added in f64.

The per-element bound, from the validation page:

$$ \big| G^{\mathrm{mix}}_{ab} - G_{ab} \big| \le \varepsilon_G\, S_{ab} + \eta, \qquad \varepsilon_G = (1+u_{32})^2\big(1+\gamma_b(u_{32})\big)\big(1+\gamma_{\lceil N_{\mathrm{aux}}/b\rceil}(u_{64})\big) - 1 + \gamma_{N_{\mathrm{aux}}}(u_{64}), \tag{11.11} $$

with $u_{32} = 2^{-24}$ and $\eta$ the f32 underflow term. The factor
$(1+u_{32})^2$ is the rounding of the two operands, $\gamma_b(u_{32})$ the f32
panel sum, $\gamma_{\lceil N_{\mathrm{aux}}/b\rceil}(u_{64})$ the f64 sum over
panels, and the last term the rounding of the f64 reference itself.

> **Unverified equation 11.11** · the mixed-precision element bound
>
> | | |
> |---|---|
> | Proof | none committed; the derivation is in the text of `site/src/reference/validation.md`, "Mixed-precision GPU kernels". To add: a numpy claim that emulates f32 panels and f64 panel sums and checks (11.11), with a negative control that accumulates across panels in f32. |
> | Code tests | `the_shipped_kernel_is_pinned_by_a_two_sided_per_element_measurement` and `the_f32_upload_is_round_to_nearest_even_bit_for_bit` in `crates/ferric-mp2/tests/gpu_rimp2_mixed.rs` |
>
> <!-- unverified: id=rimp2alg.mixed-element-bound; todo=add f32-panel emulation claim to a scripts/ proof file -->

**Which term dominates.** Evaluate the four terms at $b = 64$ and
$N_{\mathrm{aux}} = 912$:

| Term | Value |
|---|---|
| operand rounding, $\approx 2u_{32}$ | $1.2 \times 10^{-7}$ |
| f32 panel sum, $\gamma_{64}(u_{32})$ | $3.8 \times 10^{-6}$ |
| f64 panel sum, $\gamma_{15}(u_{64})$ | $1.7 \times 10^{-15}$ |
| f64 reference, $\gamma_{912}(u_{64})$ | $1.0 \times 10^{-13}$ |
| $\varepsilon_G$ | $3.93 \times 10^{-6}$ |

The f32 panel sum is 97% of $\varepsilon_G$, and it depends on $b$ only. The
terms that grow with $N_{\mathrm{aux}}$ are all in $u_{64}$. So
$\varepsilon_G$ is the same, $3.93 \times 10^{-6}$ to three digits, at
$N_{\mathrm{aux}} = 84$ and at 912. A plain SGEMM that sums all
$N_{\mathrm{aux}}$ terms in f32 would carry $\gamma_{N_{\mathrm{aux}}}(u_{32})$
instead: $5.0 \times 10^{-6}$ at 84 and $5.4 \times 10^{-5}$ at 912, 14
times $\varepsilon_G$. The f64 sum over panels is what keeps the bound from
growing with the auxiliary basis. A smaller $b$ shrinks the dominant term
and slows the kernel; the doc comment of `MIXED_K_PANEL_DEFAULT` in
`crates/ferric-core/src/gpu/mixed.rs` records the sweep that chose 64.

**Fallbacks.** Before upload, every element of $B_{ov}$ is checked against
$\sqrt{f32_{\max}/b}$. The reason: if every $|B| \le m$, a panel sum of $b$
products is at most $b\,m^2$, which stays below $f32_{\max}$ exactly when
$m \le \sqrt{f32_{\max}/b}$, $2.3 \times 10^{18}$ at $b = 64$. Here is the
check up to the comparison, which returns `GpuError::F32Range` when
`max > limit`:

<!-- excerpt: ferric-mp2/src/rimp2_gpu.rs::check_f32_arithmetic_range -->
```rust
let b = effective_k_panel().clamp(1, b_ov.nrows().max(1));
let limit = (f64::from(f32::MAX) / b as f64).sqrt();
let max = b_ov
    .iter()
    .filter(|x| x.is_finite())
    .fold(0.0f64, |m, x| m.max(x.abs()));
```

An element above the limit, an element outside the f32 range, or a flush
kernel that will not load, runs the f64 device path instead; the switch is
counted (`mixed_fallback_f64`) and traced. The filter skips non-finite
values, so a NaN in $B_{ov}$ passes through to a NaN energy, as on the CPU.
A device pool too small for the f32 tensor runs the CPU path.

## 11.6.4 Measured error of the mixed path

The validation page measures the mixed energy against the f64 CPU energy of
the same $B_{ov}$, at $b = 64$, frozen core, on an RI-JK RHF reference. Two
rows of its table, the smallest and the largest auxiliary basis:

| System | $n_o / n_v / N_{\mathrm{aux}}$ | $\Delta E$ total (Eh) | bound OS / SS (Eh) |
|---|---|---|---|
| H2O/cc-pVDZ | 5 / 19 / 84 | $-7.0 \times 10^{-9}$ | $1.6 \times 10^{-6}$ / $2.8 \times 10^{-6}$ |
| benzene/aug-cc-pVTZ | 15 / 393 / 912 | $-9.7 \times 10^{-10}$ | $1.0 \times 10^{-5}$ / $2.0 \times 10^{-5}$ |

<!-- figures: site/src/reference/validation.md "Mixed-precision GPU kernels" -->

On every system of the page each component is inside its bound. The bound is
a worst case over rounding signs and sits three to six decades above the
measured errors. The measured errors are smaller because rounding errors of
random sign partly cancel, while the bound adds their magnitudes. The bound
catches gross defects: a wrong offset, a dropped panel, a dropped pair
weight. It does not catch subtle ones: a plain SGEMM without the f64 sum
over panels, an f32 $B_{ov}$ with an f64 GEMM, a truncating upload, or f32
accumulation across panels all stay inside it. The validation page pins
those by three other checks:

| Defect class | Check |
|---|---|
| rounding mode of the upload | bit-exact: the device f32 $B_{ov}$ equals `as f32` element for element |
| f64 sum over panels | a constructed case whose panel sums are $+2^{20}$, $2^{-9}$, $-2^{20}$: the device gives exactly $2^{-9}$, an f32 sum gives 0 |
| panel structure | the per-element RMS of err$/S$ over every $G$ block of benzene/aug-cc-pVTZ, device against a host twin of the same algorithm, accepted between 0.49 and 1.13 times the twin |

The second check works because an f32 sum keeps 24 significant bits:
$2^{20} + 2^{-9}$ needs 30, so the small panel is lost before the third
panel cancels the first. In f64 it survives. The third check is device and
system dependent; the validation page states that its margin is of the size
of the spread between systems and is not a guarantee on another device.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/rimp2_gpu.rs`, module doc, dispatcher, `Stage::new` and `check_f32_arithmetic_range` | sections 11.6.1 to 11.6.3 |
| `crates/ferric-core/src/gpu/mixed.rs`, module doc and `MIXED_K_PANEL_DEFAULT` | the panel scheme |
| `site/src/reference/validation.md`, the `rimp2` row and "Mixed-precision GPU kernels" | equation 11.11, the scheme, the error table |
| `crates/ferric-mp2/tests/gpu_rimp2_resident.rs`, `crates/ferric-mp2/tests/gpu_rimp2_mixed.rs` | the pinned bounds and the measured ratios |

Changes from the sources: $\varepsilon_G$ and $\eta$ are as on the validation
page; the per-element sum $S_{ab}$ is written with the book's index order.
The device buffer sizes, the term values of $\varepsilon_G$, the plain-SGEMM
comparison and the overflow limit are computed for this section.
