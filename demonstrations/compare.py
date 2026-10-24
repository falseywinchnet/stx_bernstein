#!/usr/bin/env python3
"""Deterministic same-space fitting and signed-residual entropy demonstrations."""
import argparse
import ctypes as ct
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import lsq_linear


class Stats(ct.Structure):
    _fields_ = [(name, ct.c_size_t) for name in
                ("workspace_bytes", "accepted_steps", "bound_solves", "fallback_directions")]
    _fields_ += [("initial_cost", ct.c_double), ("final_cost", ct.c_double)]


def axis(length):
    """The library's mass-preserving Q20 degree-eight basis, as binary64."""
    position = np.arange(length, dtype=np.int64)
    weights = np.zeros((length, 9), dtype=np.int64)
    weights[:, 0] = 1 << 20
    for degree in range(1, 9):
        following = np.zeros_like(weights)
        for i in range(degree):
            right = weights[:, i] * position // max(length - 1, 1)
            following[:, i] += weights[:, i] - right
            following[:, i + 1] += right
        weights = following
    return weights.astype(np.float64) / (1 << 20)


def entropy(samples):
    """Mean of three channel-wise empirical symbol entropies, bits/sample."""
    values = []
    for channel in range(3):
        _, counts = np.unique(samples[..., channel], return_counts=True)
        probability = counts / counts.sum()
        values.append(float(-np.sum(probability * np.log2(probability))))
    return float(np.mean(values))


def fixtures(height, width):
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    x /= width - 1
    y /= height - 1
    base = np.stack([35 + 130*x + 35*y + 18*x*y,
                     45 + 85*x + 70*y - 20*x*y,
                     65 + 50*x + 90*y + 12*x*y], axis=-1)
    rng = np.random.default_rng(20261024)
    foreground = base.copy()
    mask = ((x - .48)**2 + (y - .48)**2 < .12**2)
    foreground[mask] = (240, 35, 65)
    foreground[(x > .69) & (x < .88) & (y > .15) & (y < .78)] = (30, 45, 60)
    texture = base + 13*np.sin(2*np.pi*(19*x+7*y))[..., None]
    noise = base + rng.normal(0, 10, base.shape)
    yield "smooth", base, base
    yield "foreground", foreground, base
    yield "texture", texture, base
    yield "noise", noise, base
    yield "unstructured", rng.integers(0, 256, base.shape), None


def configure_library(path):
    library = ct.CDLL(str(path.resolve()))
    library.stx_bernstein_create.argtypes = [ct.c_size_t, ct.c_size_t]
    library.stx_bernstein_create.restype = ct.c_void_p
    library.stx_bernstein_destroy.argtypes = [ct.c_void_p]
    library.stx_bernstein_destroy.restype = None
    library.stx_bernstein_fit.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint8), ct.c_size_t,
                                        ct.POINTER(ct.c_uint16), ct.c_size_t, ct.POINTER(Stats)]
    library.stx_bernstein_fit.restype = ct.c_int
    library.stx_bernstein_render_signed.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int32),
                                                   ct.c_size_t, ct.POINTER(ct.c_int16), ct.c_size_t]
    library.stx_bernstein_render_signed.restype = ct.c_int
    return library


def native_fit(library, workspace, source):
    controls = np.empty(243, dtype=np.uint16)
    stats = Stats()
    status = library.stx_bernstein_fit(workspace, source.ctypes.data_as(ct.POINTER(ct.c_uint8)),
                                     source.size, controls.ctypes.data_as(ct.POINTER(ct.c_uint16)),
                                     controls.size, ct.byref(stats))
    if status != 0:
        raise RuntimeError(f"fit returned {status}")
    widened = controls.astype(np.int32)
    rendered = np.empty(source.shape, dtype=np.int16)
    status = library.stx_bernstein_render_signed(
        workspace, widened.ctypes.data_as(ct.POINTER(ct.c_int32)), widened.size,
        rendered.ctypes.data_as(ct.POINTER(ct.c_int16)), rendered.size)
    if status != 0:
        raise RuntimeError(f"render returned {status}")
    return rendered, controls.reshape(9, 9, 3).astype(float)/256, stats


def metrics(source, rendered, continuous, truth):
    residual = source.astype(np.int32) - rendered.astype(np.int32)
    assert np.array_equal(rendered.astype(np.int32) + residual, source)
    excess = np.maximum(np.abs(source.astype(float) - continuous) - .499, 0)
    result = {"residual_entropy_bits_per_sample": entropy(residual),
              "zero_residual_percent": float(np.mean(residual == 0)*100),
              "rounded_rmse": float(np.sqrt(np.mean(residual.astype(float)**2))),
              "interval_mean_squared_error": float(np.mean(excess**2)),
              "continuous_rmse": float(np.sqrt(np.mean((source-continuous)**2))),
              "rendered_min": int(rendered.min()), "rendered_max": int(rendered.max())}
    if truth is not None:
        result["known_background_rmse"] = float(np.sqrt(np.mean((continuous-truth)**2)))
    return result, residual


def run(args):
    # Separate QR factors avoid a pixel-count by 81 design matrix. The reduced
    # 81 by 81 problem preserves the full-grid least-squares minimizer.
    height, width = 128, 256
    by, bx = axis(height), axis(width)
    qy, ry = np.linalg.qr(by, mode="reduced")
    qx, rx = np.linalg.qr(bx, mode="reduced")
    reduced = np.kron(ry, rx)
    library = configure_library(args.library)
    workspace = library.stx_bernstein_create(width, height)
    if not workspace:
        raise RuntimeError("workspace creation failed")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"dimensions": [width, height], "seed": 20261024,
              "python": platform.python_version(), "numpy": np.__version__,
              "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
              "platform": platform.platform(),
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cases": {}}
    rows = []
    methods = ["Least squares", "Bounded least squares", "stx_bernstein"]
    try:
        for name, raw, truth in fixtures(height, width):
            source = np.floor(np.clip(raw, 0, 255)+.5).astype(np.uint8)
            target = np.stack([qy.T @ source[..., c] @ qx for c in range(3)], axis=-1)
            ols = np.linalg.solve(reduced, target.reshape(81, 3)).reshape(9, 9, 3)
            bounded = np.empty_like(ols)
            for c in range(3):
                solution = lsq_linear(reduced, target[..., c].ravel(), bounds=(0, 255),
                                      method="bvls", tol=1e-12, max_iter=1000)
                if not solution.success:
                    raise RuntimeError(solution.message)
                bounded[..., c] = solution.x.reshape(9, 9)
            native, controls, stats = native_fit(library, workspace, source)
            case = {"input_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
                    "source_entropy_bits_per_sample": entropy(source),
                    "workspace_bytes": stats.workspace_bytes, "methods": {}}
            panels = []
            for label, coeff in zip(methods, [ols, bounded, controls]):
                continuous = np.stack([by @ coeff[..., c] @ bx.T for c in range(3)], axis=-1)
                # No clipping for numerical comparisons: overshoot must survive
                # in the residual. Only the displayed RGB preview is clipped.
                rendered = native if label == methods[-1] else np.floor(continuous+.5).astype(np.int32)
                if label == methods[-1]:
                    assert np.array_equal(rendered, np.floor(continuous+.5).astype(np.int32))
                result, residual = metrics(source, rendered, continuous, truth)
                result["control_min"] = float(coeff.min())
                result["control_max"] = float(coeff.max())
                case["methods"][label] = result
                panels.append((rendered, residual, result))
            report["cases"][name] = case
            rows.append((name, source, panels))
            print(name, json.dumps(case["methods"]), flush=True)
    finally:
        library.stx_bernstein_destroy(workspace)
    (args.output / "results.json").write_text(json.dumps(report, indent=2)+"\n")
    draw(rows, methods, args.output)


def draw(rows, methods, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 10), layout="constrained")
    for row, (name, source, panels) in enumerate(rows):
        axes[row, 0].imshow(source)
        axes[row, 0].set_ylabel(name.capitalize())
        for column, (rendered, residual, result) in enumerate(panels, 1):
            axes[row, column].imshow(np.clip(rendered, 0, 255).astype(np.uint8))
            axes[row, column].set_xlabel(f"Residual H₀ {result['residual_entropy_bits_per_sample']:.3f} bits/sample")
        for axis_plot in axes[row]:
            axis_plot.set_xticks([])
            axis_plot.set_yticks([])
    for axis_plot, title in zip(axes[0], ["Input"]+methods):
        axis_plot.set_title(title, fontweight="bold")
    fig.suptitle("Same 9 × 9 sampled Bernstein basis. Different fitting objectives.\n"
                 "All pixels fitted; no foreground mask. RGB previews clipped for display only.", fontsize=15)
    fig.savefig(output / "backgrounds.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 10), layout="constrained")
    for row, (name, source, panels) in enumerate(rows):
        for col, (_, residual, result) in enumerate(panels):
            # A signed channel, not a clipped RGB residual: one scale throughout.
            artist = axes[row, col].imshow(residual[..., 0], cmap="RdBu_r", vmin=-32, vmax=32)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(name.capitalize())
            if row == 0:
                axes[row, col].set_title(methods[col], fontweight="bold")
    fig.colorbar(artist, ax=axes, shrink=.7, label="Signed red-channel residual (display saturates at ±32)")
    fig.suptitle("What remains after fitting\nEntropy measurements retain every signed residual value.", fontsize=15)
    fig.savefig(output / "residuals.png", dpi=150)
    plt.close(fig)

    fig, (ax, inset) = plt.subplots(1, 2, figsize=(14, 4.5),
                                    width_ratios=[4, 1], layout="constrained")
    positions = np.arange(len(rows))
    for index, method in enumerate(["Source RGB"]+methods):
        values = [entropy(source) if index == 0 else panels[index-1][2]["residual_entropy_bits_per_sample"]
                  for _, source, panels in rows]
        ax.bar(positions+(index-1.5)*.20, values, width=.19, label=method)
    ax.set_xticks(positions, [name.capitalize() for name, _, _ in rows])
    ax.set_ylabel("Empirical H₀ (bits/sample, mean across RGB)")
    ax.set_title("Residual entropy: measured, not optimized")
    ax.legend(ncol=2)
    ax.set_ylim(0, 9)
    smooth = rows[0][2]
    inset.bar([0, 1, 2], [panel[2]["residual_entropy_bits_per_sample"] for panel in smooth],
              color=["tab:orange", "tab:green", "tab:red"])
    inset.set_xticks([0, 1, 2], ["LS", "Bounded LS", "stx"])
    inset.set_title("Smooth residuals, magnified", fontsize=9)
    inset.tick_params(labelsize=8)
    fig.savefig(output / "entropy.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("demonstrations/results"))
    arguments = parser.parse_args()
    # Entropy sanity checks independent of fitting.
    assert entropy(np.zeros((2, 2, 3), dtype=int)) == 0
    assert entropy(np.tile(np.array([0, 1])[:, None, None], (1, 2, 3))) == 1
    run(arguments)
