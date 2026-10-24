# 289 curvature moments, all 81 controls

The surface remains degree eight in both axes, with 9×9 controls per channel.
For exact Bernstein functions,

```
b_i^8(x) b_j^8(x) = C(8,i) C(8,j) / C(16,i+j) * b_(i+j)^16(x).
```

The 45 unordered products span only 17 independent functions. Tensor products
therefore need 17×17=289 moment coordinates instead of 45×45=2025. Arbitrary
pixel masks do not invalidate this pointwise identity. It is not a lower-degree
surface or a sevenfold end-to-end speed claim.

Our Q20 recurrence rounds transfers at each degree. Its sampled pair products
have full rank 45 on the tested 64-, 193-, 512- and 1024-sample axes. Exact
modular-rank results are retained in `basis-product-rank.json`; this is not a
floating-point numerical-rank judgment. Consequently the reduced curvature is
an approximation to this implementation's quantized basis.

The new solver uses degree-16 moments for its quadratic search model, including
a consistent full polynomial Gram matrix and the existing ridge. Initializing
the fit, evaluating the objective, computing its gradient, checking the objective
gap, accepting steps and rendering all retain the Q20 basis. A proposed step
must improve the actual Q20 objective. The old iteration and line-search limits
remain. This does not prove identical convergence or controls; it is a bounded
approximate-Newton method with exact-model objective checks.

Identical input channels also share a single fit. This second optimization is
exact. Work counters count operations actually performed; objective totals
include all three channels. No user-selectable alternate model is introduced.

## Measured complete CPU jobs

Compared with commit f4a77b5's numerical implementation on Apple A18 Pro,
macOS 26.7.1, LLVM 22.1.8. Three alternating batches, two warmups and seven
measured jobs per batch: 21 samples per implementation/image. Times include
input copying, fitting, integer rendering, RGBA packing and worker publication;
they exclude OBS graphics operations and frame/frequency scheduling.

| Image | Previous median | New median | CPU capacity, fits/s |
|---|---:|---:|---:|
| Combined 1024×512 | 71.396 ms | 57.028 ms | 14.01 → 17.54 |
| Combined 640×320 | 25.745 ms | 24.641 ms | 38.84 → 40.58 |
| Smooth 512×512 | 67.377 ms | 58.149 ms | 14.84 → 17.20 |

All three worker outputs were byte-identical. On the ten-image fit corpus,
rendered outputs were also byte-identical; 12 controls changed in the axis-aligned
step case. Raw results are in `moment17-worker.json` and `moment17-corpus.json`.
Some timing variation remains. Large grayscale speedups include channel sharing,
not only moment reduction. The unshared first trial regressed on the axis-aligned
step because approximate curvature took additional late iterations.

At 1024×512, solver payload falls from 4,337,232 to 3,917,280 bytes, about 9.7%.
Small images need not save memory: the additional fixed curvature Gram matrix
can outweigh the smaller per-axis tables. Caller buffers and graphics storage
are excluded.

## Numerical differences and checks

Twenty additional deterministic cases include one-pixel axes, near-flat patterns,
ramps, varied patterns and constants. Three differed from the preceding output
by one channel level: 5 samples on the 63×65 ramp, 706 on the 63×65 near-flat
pattern, and 2 on the 128×96 ramp. The near-flat pattern's unquantized interval
cost increased from 0.008260929 to 0.008287521 within the same iteration budget;
the two ramps' costs decreased. Other cases had identical rendered pixels.
See `moment17-quality.json`. These observations are not a universal one-level
error guarantee or a guarantee of non-worsening final cost versus the old solver.

The research implementation passed focused ASan/UBSan candidate/worker checks
and actual OBS Metal integration checks. Publication-package checks are recorded
in PERFORMANCE.md. The installed OBS effect retains the same two views.

The Bernstein product identity follows directly from the binomial definition;
see J. Sánchez-Reyes, [Algebraic manipulation in the Bernstein form made simple
via convolutions](https://www.cise.ufl.edu/research/SurfLab/seminar/algebraicmanipulation.pdf),
Computer-Aided Design 35 (2003), section 2.1.
