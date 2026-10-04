import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from supplementary_suite import fit_and_eval, make_envs


def empty_env_leaf_rate(model, X_list):
    m = len(X_list)
    leaf_to_counts = {}
    for env_idx, X in enumerate(X_list):
        for row in X:
            key = model._route_leaf_key(row)
            if key not in leaf_to_counts:
                leaf_to_counts[key] = [0 for _ in range(m)]
            leaf_to_counts[key][env_idx] += 1
    if not leaf_to_counts:
        return 0.0, 0
    total_pairs = len(leaf_to_counts) * m
    empty_pairs = sum(count == 0 for counts in leaf_to_counts.values() for count in counts)
    return float(empty_pairs / total_pairs), int(len(leaf_to_counts))


def run_study():
    os.makedirs("experiment/results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    seeds = list(range(42, 50))
    sample_sizes = [25, 50, 100, 200]
    alpha_grid = [0.0, 0.5, 1.0, 2.0]
    max_depth = 4
    lambda_inv = 10.0
    rows = []

    for n_samples in sample_sizes:
        for alpha in alpha_grid:
            print(f"[sparse] n={n_samples}, alpha={alpha}")
            for seed in seeds:
                train_envs, test_envs = make_envs(
                    seed=seed,
                    train_corrs=(0.95, 0.80, 0.70),
                    test_corrs=(0.20, 0.10),
                    n_samples=n_samples,
                )
                res = fit_and_eval(
                    train_envs,
                    test_envs,
                    max_depth=max_depth,
                    lambda_inv=lambda_inv,
                    laplace_alpha=alpha,
                )
                X_train_list = [x for x, _ in train_envs]
                empty_rate, n_leaves = empty_env_leaf_rate(res["model"], X_train_list)
                rows.append(
                    {
                        "seed": seed,
                        "n_samples": int(n_samples),
                        "laplace_alpha": float(alpha),
                        "max_depth": max_depth,
                        "lambda_inv": lambda_inv,
                        "train_acc": res["train_acc"],
                        "avg_ood_acc": res["avg_ood_acc"],
                        "worst_ood_acc": res["worst_ood_acc"],
                        "train_invariance": res["train_invariance"],
                        "runtime_s": res["runtime_s"],
                        "empty_env_leaf_rate": empty_rate,
                        "n_leaves": n_leaves,
                    }
                )

    runs_df = pd.DataFrame(rows)
    summary_df = (
        runs_df.groupby(["n_samples", "laplace_alpha"], as_index=False)
        .agg(
            avg_ood_mean=("avg_ood_acc", "mean"),
            avg_ood_std=("avg_ood_acc", "std"),
            worst_ood_mean=("worst_ood_acc", "mean"),
            inv_mean=("train_invariance", "mean"),
            empty_rate_mean=("empty_env_leaf_rate", "mean"),
            n_leaves_mean=("n_leaves", "mean"),
        )
    )
    runs_df.to_csv("experiment/results/sparse_smoothing_runs.csv", index=False)
    summary_df.to_csv("experiment/results/sparse_smoothing_summary.csv", index=False)

    pivot_ood = summary_df.pivot(index="n_samples", columns="laplace_alpha", values="avg_ood_mean")
    pivot_empty = summary_df.pivot(index="n_samples", columns="laplace_alpha", values="empty_rate_mean")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    im0 = axes[0].imshow(pivot_ood.to_numpy(), aspect="auto", cmap="viridis")
    axes[0].set_title("Sparse-leaf study: average OOD")
    axes[0].set_xlabel("Laplace alpha")
    axes[0].set_ylabel("Samples per environment")
    axes[0].set_xticks(range(len(pivot_ood.columns)))
    axes[0].set_xticklabels([str(x) for x in pivot_ood.columns])
    axes[0].set_yticks(range(len(pivot_ood.index)))
    axes[0].set_yticklabels([str(x) for x in pivot_ood.index])
    for i in range(pivot_ood.shape[0]):
        for j in range(pivot_ood.shape[1]):
            axes[0].text(j, i, f"{pivot_ood.iat[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pivot_empty.to_numpy(), aspect="auto", cmap="magma_r", vmin=0.0, vmax=1.0)
    axes[1].set_title("Sparse-leaf study: empty env-leaf rate")
    axes[1].set_xlabel("Laplace alpha")
    axes[1].set_ylabel("Samples per environment")
    axes[1].set_xticks(range(len(pivot_empty.columns)))
    axes[1].set_xticklabels([str(x) for x in pivot_empty.columns])
    axes[1].set_yticks(range(len(pivot_empty.index)))
    axes[1].set_yticklabels([str(x) for x in pivot_empty.index])
    for i in range(pivot_empty.shape[0]):
        for j in range(pivot_empty.shape[1]):
            axes[1].text(j, i, f"{pivot_empty.iat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig("figures/sparse_smoothing_heatmaps.png", dpi=150)
    plt.close(fig)
    print("Saved sparse smoothing outputs.")


if __name__ == "__main__":
    run_study()