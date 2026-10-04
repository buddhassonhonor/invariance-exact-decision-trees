import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ciedt import CIEDTClassifier
from data_gen import generate_spurious_data
from experiment_utils import accuracy_score


def make_envs(seed, train_corrs, test_corrs, n_samples=500, core_accuracy=0.85):
    train_envs = []
    test_envs = []
    for i, corr in enumerate(train_corrs):
        X, y, _ = generate_spurious_data(
            n_samples,
            corr,
            core_accuracy=core_accuracy,
            seed=seed + 101 * (i + 1),
        )
        train_envs.append((X, y))
    for i, corr in enumerate(test_corrs):
        X, y, _ = generate_spurious_data(
            n_samples,
            corr,
            core_accuracy=core_accuracy,
            seed=seed + 1001 * (i + 1),
        )
        test_envs.append((X, y))
    return train_envs, test_envs


def pool_envs(envs):
    X = np.vstack([x for x, _ in envs])
    y = np.concatenate([y for _, y in envs])
    return X, y


def evaluate_on_envs(model, envs):
    scores = []
    for X, y in envs:
        scores.append(accuracy_score(y, model.predict(X)))
    return float(np.mean(scores)), float(np.min(scores))


def run_supplement():
    print("Running synthetic depth-lambda supplement...")
    os.makedirs("experiment/results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    seeds = list(range(42, 52))
    depth_grid = [1, 2, 3, 4]
    lambda_grid = [0.0, 1.0, 5.0, 10.0, 20.0, 40.0]
    train_corrs = (0.95, 0.80, 0.70)
    test_corrs = (0.20, 0.10)
    spurious_feature_idx = 2

    rows = []
    for depth in depth_grid:
        for lam in lambda_grid:
            print(f"  depth={depth}, lambda={lam}")
            for seed in seeds:
                train_envs, test_envs = make_envs(
                    seed=seed,
                    train_corrs=train_corrs,
                    test_corrs=test_corrs,
                )
                X_train_list = [x for x, _ in train_envs]
                y_train_list = [y for _, y in train_envs]
                X_train, y_train = pool_envs(train_envs)

                t0 = time.time()
                model = CIEDTClassifier(
                    max_depth=depth,
                    lambda_inv=lam,
                    laplace_alpha=1.0,
                )
                model.fit(X_train_list, y_train_list)
                runtime_s = time.time() - t0

                train_acc = accuracy_score(y_train, model.predict(X_train))
                avg_ood, worst_ood = evaluate_on_envs(model, test_envs)
                train_inv = model.compute_invariance(X_train_list, y_train_list)
                root_feature = model.get_root_feature()
                rows.append(
                    {
                        "seed": seed,
                        "max_depth": depth,
                        "lambda_inv": lam,
                        "train_acc": train_acc,
                        "avg_ood_acc": avg_ood,
                        "worst_ood_acc": worst_ood,
                        "train_invariance": train_inv,
                        "runtime_s": runtime_s,
                        "root_feature": -1 if root_feature is None else int(root_feature),
                        "uses_spurious_root": int(root_feature == spurious_feature_idx)
                        if root_feature is not None
                        else 0,
                    }
                )

    runs_df = pd.DataFrame(rows)
    runs_df.to_csv("experiment/results/synthetic_grid_runs.csv", index=False)

    summary_df = (
        runs_df.groupby(["max_depth", "lambda_inv"], as_index=False)
        .agg(
            train_acc_mean=("train_acc", "mean"),
            avg_ood_mean=("avg_ood_acc", "mean"),
            avg_ood_std=("avg_ood_acc", "std"),
            worst_ood_mean=("worst_ood_acc", "mean"),
            train_inv_mean=("train_invariance", "mean"),
            runtime_mean=("runtime_s", "mean"),
            spurious_root_rate=("uses_spurious_root", "mean"),
        )
    )
    summary_df.to_csv("experiment/results/synthetic_grid_summary.csv", index=False)

    pivot_ood = summary_df.pivot(index="max_depth", columns="lambda_inv", values="avg_ood_mean")
    pivot_sp = summary_df.pivot(index="max_depth", columns="lambda_inv", values="spurious_root_rate")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(pivot_ood.to_numpy(), aspect="auto", cmap="viridis")
    axes[0].set_title("Synthetic XOR: mean average OOD")
    axes[0].set_xlabel("lambda")
    axes[0].set_ylabel("max depth")
    axes[0].set_xticks(range(len(pivot_ood.columns)))
    axes[0].set_xticklabels([str(x) for x in pivot_ood.columns])
    axes[0].set_yticks(range(len(pivot_ood.index)))
    axes[0].set_yticklabels([str(x) for x in pivot_ood.index])
    for i in range(pivot_ood.shape[0]):
        for j in range(pivot_ood.shape[1]):
            axes[0].text(
                j,
                i,
                f"{pivot_ood.iat[i, j]:.3f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pivot_sp.to_numpy(), aspect="auto", cmap="magma_r", vmin=0.0, vmax=1.0)
    axes[1].set_title("Synthetic XOR: spurious root frequency")
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("max depth")
    axes[1].set_xticks(range(len(pivot_sp.columns)))
    axes[1].set_xticklabels([str(x) for x in pivot_sp.columns])
    axes[1].set_yticks(range(len(pivot_sp.index)))
    axes[1].set_yticklabels([str(x) for x in pivot_sp.index])
    for i in range(pivot_sp.shape[0]):
        for j in range(pivot_sp.shape[1]):
            axes[1].text(
                j,
                i,
                f"{pivot_sp.iat[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig("figures/synthetic_lambda_depth_grid.png", dpi=150)
    plt.close(fig)

    best_row = summary_df.sort_values(
        ["avg_ood_mean", "worst_ood_mean", "train_inv_mean"],
        ascending=[False, False, True],
    ).iloc[0]
    print("Saved:")
    print("  experiment/results/synthetic_grid_runs.csv")
    print("  experiment/results/synthetic_grid_summary.csv")
    print("  figures/synthetic_lambda_depth_grid.png")
    print(
        "Best config by avg OOD: "
        f"depth={int(best_row['max_depth'])}, lambda={best_row['lambda_inv']}, "
        f"avgOOD={best_row['avg_ood_mean']:.3f}, worstOOD={best_row['worst_ood_mean']:.3f}, "
        f"spurious_root_rate={best_row['spurious_root_rate']:.2f}"
    )


if __name__ == "__main__":
    run_supplement()