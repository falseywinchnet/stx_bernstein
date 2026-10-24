# Fitting objective

For each channel, let `B` be the tensor-product Bernstein basis, `c` the 81
controls, and `y[p]` a pixel. Minimize

```
sum_p distance(B[p] c, [y[p] - 0.499, y[p] + 0.499])²
```

subject to `0 ≤ c ≤ 255` (or `−255 ≤ c ≤ 255` for signed residual input).
This is convex and piecewise quadratic. It is not a minimum encoded-size
objective, and lower objective value does not guarantee more exact rounded pixels.

The implementation prepares mass-preserving Q20 axis weights, initializes the
controls, and computes the interval objective and half-gradient. A primal-feasible
active-set solve handles control bounds within the 81-dimensional quadratic
model. The true image objective accepts or rejects each line-search trial.

A tangent minimized over the control box estimates the remaining objective gap:
`2 sum_i g[i] (c[i] - bound[i])`, where `g` is the stored half-gradient and
`bound[i]` minimizes that tangent. It stops at `max(1e-8, 1e-10 * cost)`.
This is a floating-point estimate, not an interval-arithmetic certificate.
The implementation bounds outer iterations, line-search trials and active-set
transitions. Hitting those limits need not yield the global optimum.

Fitting uses binary64 arithmetic and returns Q8 controls. Reconstruction uses
integer arithmetic with half-up rounding, including signed values. The final
control quantization can change interval agreement. No promise of identical
floating-point results across compilers or architectures is made.

The connection to the Meyer–Bregman research is methodological: preserve useful
state, solve coupled updates before returning to the image, and remove redundant
passes. The solver here is an active-set quadratic method, not the cartoon
algorithm's Krylov transport map. See the related research in
[falseywinchnet/papers_please](https://github.com/falseywinchnet/papers_please).

The basis is nonnegative and sums to one, so reconstruction stays within the
control range. This convex-hull property does not guarantee a monotone surface,
perfect edge preservation, or universal freedom from visible oscillation.

The library is extracted from the independent Bernstein research fork. Original
file hashes are recorded in `provenance.json`. The historical renderer under
`tests/reference/` is an independent test oracle and is not linked into the library.
