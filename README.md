# stx_bernstein

**Fast, bounded Bernstein surface fitting for images.** C++20 implementation,
C interface, MIT license, no runtime dependencies beyond the standard library.

Fit an image with a smooth degree-eight surface: **81 controls per channel**.
Reconstruct that surface, subtract it from the image, and retain the residual.
The aim is to lift broad structure without introducing an oscillatory basis.
The nonnegative Bernstein basis does not, by itself, guarantee monotonicity or
an absence of all spatial oscillation in the fitted surface.

The implementation grew out of the fast Meyer–Bregman decomposition work.
It minimizes squared distance to pixel quantization intervals, using a bounded
active-set quadratic solve and an objective-gap stopping estimate. It is a
specialized image-surface fitter, not a general Bregman solver.

The speed comes from doing less work: solve control-bound changes in coefficient
space, reuse prepared basis data, accumulate independent sums contiguously,
reject losing trials early, and build curvature only when another step is needed.
No platform intrinsics, image subsampling, or blanket fast-math.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix /your/install/prefix
```

Consumers use `find_package(stx_bernstein CONFIG REQUIRED)` and link
`stx::bernstein`. Include [`stx_bernstein.h`](include/stx_bernstein.h).

## Interface

1. Create a workspace with `stx_bernstein_create(width, height)`.
2. Fit tightly packed RGB8 with `stx_bernstein_fit` into 243 interleaved Q8 controls.
3. Reconstruct with `stx_bernstein_render_signed`, widening controls to `int32_t`.
4. Reuse the workspace; destroy it with `stx_bernstein_destroy`.

Signed residual fitting is also available. Dimensions are limited to 2,048 per
axis and 524,288 pixels total. A workspace admits one caller at a time; repeated
fits allocate no storage. The header specifies counts, bounds and failure behavior.

On the measured 1024×512 workload, solver storage is **4.14 MiB**. The latest
optimization reduced complete CPU-job time on an A18 Pro from **70.0 to 65.7 ms**;
the smooth 512×512 field improved from **88.7 to 64.6 ms**. These are paired
comparisons with the preceding implementation, not claims against other libraries.
See [algorithm](docs/ALGORITHM.md) and [measurements](docs/PERFORMANCE.md).

This is an experimental library. It is the numerical core used by the separate
Bernstein OBS effect; OBS and codec machinery are not dependencies or part of
this package.
