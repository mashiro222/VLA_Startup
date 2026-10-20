"""Analytic conditional flow matching in 1D; no neural model or robot required.

Run from the repository root:
    python docs/learning/examples/flow_matching_demo.py

Dependencies: numpy, matplotlib.
The convention matches OpenPI: t=1 is noise, t=0 is data.
The exact target is an equal mixture of N(-2, 0.35^2) and N(2, 0.35^2).
"""

from __future__ import annotations

import argparse
from math import erf, sqrt
from pathlib import Path

import numpy as np

MEANS = np.array([-2.0, 2.0])
SIGMA = 0.35


def velocity(x: np.ndarray, t: float) -> np.ndarray:
    """E[noise - action | X_t=x] for X_t=(1-t)*action+t*noise.

    Within each mixture component, joint Gaussian conditioning gives:
      mean(X_t) = (1-t)*mu
      var(X_t) = (1-t)^2*sigma^2 + t^2
      E[noise-action | x, component]
        = -mu + (t-(1-t)*sigma^2)/var(X_t) * (x-mean(X_t)).
    Average these velocities with posterior component probabilities.
    """
    variance = (1 - t) ** 2 * SIGMA**2 + t**2
    residual = x[..., None] - (1 - t) * MEANS
    log_weight = -0.5 * residual**2 / variance
    log_weight -= log_weight.max(axis=-1, keepdims=True)
    weight = np.exp(log_weight)
    weight /= weight.sum(axis=-1, keepdims=True)
    component_velocity = -MEANS + (
        (t - (1 - t) * SIGMA**2) / variance
    ) * residual
    return np.sum(weight * component_velocity, axis=-1)


def euler(initial: np.ndarray, steps: int, *, history: bool = False) -> np.ndarray:
    x = np.array(initial, dtype=np.float64, copy=True)
    trajectory = [x.copy()] if history else None
    dt = -1.0 / steps
    for n in range(steps):
        t = 1.0 - n / steps
        x += dt * velocity(x, t)
        if trajectory is not None:
            trajectory.append(x.copy())
    return np.asarray(trajectory) if trajectory is not None else x


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def target_cdf(x: float) -> float:
    return sum(normal_cdf((x - mu) / SIGMA) for mu in MEANS) / 2


def exact_endpoint(initial: float) -> float:
    """A 1D smooth flow preserves quantiles: F_data(x_0)=Phi(x_1)."""
    probability = normal_cdf(initial)
    left, right = -8.0, 8.0
    for _ in range(80):
        middle = (left + right) / 2
        if target_cdf(middle) < probability:
            left = middle
        else:
            right = middle
    return (left + right) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Compare the actual solver against exact endpoint quantiles, rather than
    # comparing it only with another discretization of the same ODE.
    initial_grid = np.linspace(-2.7, 2.7, 100)
    exact = np.array([exact_endpoint(float(x)) for x in initial_grid])
    for steps in (10, 100, 1000):
        endpoint = euler(initial_grid, steps)
        error = np.abs(endpoint - exact)
        print(
            f"Euler N={steps:4d}: mean endpoint error={error.mean():.6f}, "
            f"max error={error.max():.6f}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), layout="constrained")
    rng = np.random.default_rng(20261020)
    progress = np.linspace(0, 1, 251)
    initial_lines = np.linspace(-2.4, 2.4, 24)
    target_lines = rng.choice(MEANS, size=len(initial_lines)) + rng.normal(
        0, SIGMA, len(initial_lines)
    )
    pair_paths = (
        (1 - progress[:, None]) * initial_lines
        + progress[:, None] * target_lines
    )
    colors = ["#25818e" if x < 0 else "#d47737" for x in initial_lines]
    for i, color in enumerate(colors):
        axes[0].plot(progress, pair_paths[:, i], color=color, alpha=0.55, lw=1)
    axes[0].set_title("1. Training pairs: straight interpolation", loc="left")

    paths = euler(initial_lines, 250, history=True)
    for i, color in enumerate(colors):
        axes[1].plot(progress, paths[:, i], color=color, alpha=0.8, lw=1.1)
    axes[1].set_title("2. Marginal ODE: curved sample paths", loc="left")
    for ax in axes[:2]:
        ax.set_xlabel("Generation progress s = 1 - t")
        ax.set_ylabel("Action-space coordinate")
        ax.set_xlim(0, 1)
        ax.set_ylim(-3.3, 3.3)
        ax.grid(alpha=0.15)
        ax.set_xticks([0, 0.5, 1], ["0 (noise)", "0.5", "1 (data)"])

    samples = rng.standard_normal(40_000)
    bins = np.linspace(-3.5, 3.5, 120)
    axes[2].hist(
        euler(samples, 10), bins=bins, density=True, histtype="step",
        color="#d47737", lw=1.2, label="Euler: 10 steps",
    )
    axes[2].hist(
        euler(samples, 100), bins=bins, density=True, histtype="step",
        color="#25818e", lw=1.2, label="Euler: 100 steps",
    )
    x = np.linspace(-3.5, 3.5, 600)
    pdf = np.exp(-0.5 * ((x[:, None] - MEANS) / SIGMA) ** 2).mean(axis=1)
    pdf /= SIGMA * np.sqrt(2 * np.pi)
    axes[2].plot(x, pdf, "--", color="#242c3a", lw=1.7, label="Exact target density")
    axes[2].set_title("3. Generated endpoints: two modes", loc="left")
    axes[2].set_xlabel("Action-space coordinate")
    axes[2].set_ylabel("Probability density")
    axes[2].legend(frameon=False, fontsize=9)
    axes[2].grid(alpha=0.15)
    fig.suptitle(
        "Flow matching with an analytic velocity field (not a pi0.5 rollout)",
        fontsize=14,
    )
    for extension in ("png", "svg"):
        path = args.output_dir / f"flow-matching-demo.{extension}"
        fig.savefig(path, dpi=160)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
