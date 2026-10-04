import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(values, n_boot=5000, alpha=0.05, seed=0):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boots.append(np.mean(sample))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1.0 - alpha / 2))
    return lo, hi


def cohens_d_paired(delta):
    delta = np.asarray(delta, dtype=float)
    if delta.size <= 1:
        return np.nan
    std = np.std(delta, ddof=1)
    if std == 0:
        return np.inf if np.mean(delta) > 0 else -np.inf if np.mean(delta) < 0 else 0.0
    return float(np.mean(delta) / std)


def run_analysis():
    os.makedirs("experiment/results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    runs_df = pd.read_csv("experiment/results/revision_runs.csv")

    pair_rows = []
    for dataset_name in sorted(runs_df["dataset"].unique()):
        ciedt_df = runs_df[(runs_df["dataset"] == dataset_name) & (runs_df["method"] == "CIEDT")].sort_values("seed")
        for baseline in ["CART", "TunedCART", "ODT", "LR"]:
            base_df = runs_df[(runs_df["dataset"] == dataset_name) & (runs_df["method"] == baseline)].sort_values("seed")
            if len(ciedt_df) != len(base_df) or len(base_df) == 0:
                continue
            for metric in ["avg_ood_acc", "worst_ood_acc"]:
                delta = ciedt_df[metric].to_numpy() - base_df[metric].to_numpy()
                t_res = stats.ttest_rel(ciedt_df[metric].to_numpy(), base_df[metric].to_numpy())
                try:
                    w_res = stats.wilcoxon(delta)
                    w_p = float(w_res.pvalue)
                except ValueError:
                    w_p = np.nan
                ci_lo, ci_hi = bootstrap_ci(delta, seed=17)
                pair_rows.append(
                    {
                        "dataset": dataset_name,
                        "metric": metric,
                        "comparison": f"CIEDT-{baseline}",
                        "delta_mean": float(np.mean(delta)),
                        "ci95_low": ci_lo,
                        "ci95_high": ci_hi,
                        "paired_t_p": float(t_res.pvalue),
                        "wilcoxon_p": w_p,
                        "cohens_d_paired": cohens_d_paired(delta),
                        "win_rate": float(np.mean(delta > 0)),
                    }
                )
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv("experiment/results/revision_pairwise_tests.csv", index=False)

    selection_df = (
        runs_df[runs_df["method"] == "CIEDT"]
        .groupby(["dataset", "selected_depth", "selected_lambda"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    selection_df.to_csv("experiment/results/revision_selection_summary.csv", index=False)

    worst_df = runs_df.groupby(["dataset", "method"], as_index=False).agg(
        worst_ood_mean=("worst_ood_acc", "mean"),
        worst_ood_std=("worst_ood_acc", "std"),
    )
    worst_df.to_csv("experiment/results/revision_worst_ood_summary.csv", index=False)

    tree_df = runs_df[runs_df["method"].isin(["CART", "TunedCART", "ODT", "CIEDT"])]
    runtime_df = []
    for dataset_name in sorted(runs_df["dataset"].unique()):
        dataset_sub = runs_df[runs_df["dataset"] == dataset_name]
        cart_time = float(dataset_sub[dataset_sub["method"] == "CART"]["runtime_s"].mean())
        ciedt_time = float(dataset_sub[dataset_sub["method"] == "CIEDT"]["runtime_s"].mean())
        odt_time = float(dataset_sub[dataset_sub["method"] == "ODT"]["runtime_s"].mean())
        runtime_df.append(
            {
                "dataset": dataset_name,
                "ciedt_over_cart": ciedt_time / cart_time if cart_time > 0 else np.nan,
                "ciedt_over_odt": ciedt_time / odt_time if odt_time > 0 else np.nan,
            }
        )
    pd.DataFrame(runtime_df).to_csv("experiment/results/revision_runtime_ratios.csv", index=False)

    dataset_order = ["synthetic_xor", "cmnist_tabular", "breast_cancer", "wine_binary"]
    method_order = ["CART", "TunedCART", "ODT", "CIEDT", "LR"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, dataset_name in zip(axes.flatten(), dataset_order):
        sub = runs_df[runs_df["dataset"] == dataset_name]
        data = [sub[sub["method"] == m]["avg_ood_acc"].to_numpy() for m in method_order]
        ax.boxplot(data, tick_labels=method_order)
        ax.set_title(dataset_name)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Seed-wise average OOD distributions")
    fig.tight_layout()
    fig.savefig("figures/revision_ood_boxplots.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    color_map = {"CART": "tab:blue", "TunedCART": "tab:orange", "ODT": "tab:green", "CIEDT": "tab:red"}
    for method in ["CART", "TunedCART", "ODT", "CIEDT"]:
        sub = tree_df[tree_df["method"] == method]
        ax.scatter(sub["train_invariance"], sub["avg_ood_acc"], label=method, alpha=0.8, color=color_map[method])
    ax.set_xlabel("Training invariance proxy")
    ax.set_ylabel("Average OOD accuracy")
    ax.set_title("Invariance--OOD trade-off across tree runs")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/revision_invariance_tradeoff.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, dataset_name in zip(axes.flatten(), dataset_order):
        sub = selection_df[selection_df["dataset"] == dataset_name].copy()
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        pivot = sub.pivot(index="selected_depth", columns="selected_lambda", values="count").fillna(0)
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues")
        ax.set_title(dataset_name)
        ax.set_xlabel("lambda")
        ax.set_ylabel("depth")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(x) for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(int(x)) for x in pivot.index])
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, int(pivot.iat[i, j]), ha="center", va="center", color="black")
    fig.suptitle("CIEDT selection frequency across seeds")
    fig.tight_layout()
    fig.savefig("figures/revision_ciedt_selection.png", dpi=150)
    plt.close(fig)

    print("Saved revision analysis outputs.")


if __name__ == "__main__":
    run_analysis()