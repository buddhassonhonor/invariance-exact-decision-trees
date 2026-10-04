import contextlib
import io
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ciedt import CIEDTClassifier
from data_gen import generate_spurious_data
from experiment_utils import accuracy_score


def pool_envs(envs):
    X = np.vstack([x for x, _ in envs])
    y = np.concatenate([y for _, y in envs])
    return X, y


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
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


def make_envs(
    seed,
    train_corrs,
    test_corrs,
    n_samples=500,
    core_accuracy=0.85,
):
    train_envs = []
    test_envs = []
    for idx, corr in enumerate(train_corrs):
        X, y, _ = generate_spurious_data(
            n_samples,
            corr,
            core_accuracy=core_accuracy,
            seed=seed + 101 * (idx + 1),
        )
        train_envs.append((X, y))
    for idx, corr in enumerate(test_corrs):
        X, y, _ = generate_spurious_data(
            n_samples,
            corr,
            core_accuracy=core_accuracy,
            seed=seed + 1001 * (idx + 1),
        )
        test_envs.append((X, y))
    return train_envs, test_envs


def evaluate_on_envs(model, envs):
    scores = []
    for X, y in envs:
        scores.append(accuracy_score(y, model.predict(X)))
    return float(np.mean(scores)), float(np.min(scores))


def fit_and_eval(train_envs, test_envs, max_depth, lambda_inv, laplace_alpha=1.0):
    X_train_list = [x for x, _ in train_envs]
    y_train_list = [y for _, y in train_envs]
    X_train, y_train = pool_envs(train_envs)
    t0 = time.time()
    model = CIEDTClassifier(
        max_depth=max_depth,
        lambda_inv=lambda_inv,
        laplace_alpha=laplace_alpha,
    )
    model.fit(X_train_list, y_train_list)
    runtime_s = time.time() - t0
    avg_ood, worst_ood = evaluate_on_envs(model, test_envs)
    return {
        "model": model,
        "train_acc": accuracy_score(y_train, model.predict(X_train)),
        "avg_ood_acc": avg_ood,
        "worst_ood_acc": worst_ood,
        "train_invariance": model.compute_invariance(X_train_list, y_train_list),
        "runtime_s": runtime_s,
        "root_feature": model.get_root_feature(),
    }


def loeo_select_ciedt(train_envs, depth_grid, lambda_grid, laplace_alpha=1.0):
    m = len(train_envs)
    best = None
    for depth in depth_grid:
        for lam in lambda_grid:
            fold_scores = []
            for val_idx in range(m):
                tr_envs = [train_envs[i] for i in range(m) if i != val_idx]
                X_list = [x for x, _ in tr_envs]
                y_list = [y for _, y in tr_envs]
                X_val, y_val = train_envs[val_idx]
                clf = CIEDTClassifier(
                    max_depth=depth,
                    lambda_inv=lam,
                    laplace_alpha=laplace_alpha,
                )
                clf.fit(X_list, y_list)
                fold_scores.append(accuracy_score(y_val, clf.predict(X_val)))
            score_worst = float(np.min(fold_scores))
            score_avg = float(np.mean(fold_scores))
            key = (score_worst, score_avg, -depth, -lam)
            if best is None or key > best["key"]:
                best = {
                    "max_depth": int(depth),
                    "lambda_inv": float(lam),
                    "val_worst": score_worst,
                    "val_avg": score_avg,
                    "key": key,
                }
    return best


def evaluate_grid(train_envs, test_envs, depth_grid, lambda_grid, laplace_alpha=1.0):
    rows = []
    m = len(train_envs)
    for depth in depth_grid:
        for lam in lambda_grid:
            fold_scores = []
            for val_idx in range(m):
                tr_envs = [train_envs[i] for i in range(m) if i != val_idx]
                X_list = [x for x, _ in tr_envs]
                y_list = [y for _, y in tr_envs]
                X_val, y_val = train_envs[val_idx]
                clf = CIEDTClassifier(
                    max_depth=depth,
                    lambda_inv=lam,
                    laplace_alpha=laplace_alpha,
                )
                clf.fit(X_list, y_list)
                fold_scores.append(accuracy_score(y_val, clf.predict(X_val)))
            res = fit_and_eval(
                train_envs,
                test_envs,
                max_depth=depth,
                lambda_inv=lam,
                laplace_alpha=laplace_alpha,
            )
            rows.append(
                {
                    "max_depth": int(depth),
                    "lambda_inv": float(lam),
                    "laplace_alpha": float(laplace_alpha),
                    "val_worst": float(np.min(fold_scores)),
                    "val_avg": float(np.mean(fold_scores)),
                    "train_acc": res["train_acc"],
                    "avg_ood_acc": res["avg_ood_acc"],
                    "worst_ood_acc": res["worst_ood_acc"],
                    "train_invariance": res["train_invariance"],
                    "runtime_s": res["runtime_s"],
                    "root_feature": -1 if res["root_feature"] is None else int(res["root_feature"]),
                }
            )
    return pd.DataFrame(rows)


def run_selection_gap_study():
    seeds = list(range(42, 52))
    depth_grid = [1, 2, 3, 4]
    lambda_grid = [0.0, 1.0, 5.0, 10.0, 20.0, 40.0]
    chosen_rows = []
    grid_rows = []

    for seed in seeds:
        train_envs, test_envs = make_envs(
            seed=seed,
            train_corrs=(0.95, 0.80, 0.70),
            test_corrs=(0.20, 0.10),
            n_samples=500,
        )
        grid_df = evaluate_grid(train_envs, test_envs, depth_grid, lambda_grid, laplace_alpha=1.0)
        grid_df.insert(0, "seed", seed)
        grid_rows.append(grid_df)

        loeo = grid_df.sort_values(
            ["val_worst", "val_avg", "max_depth", "lambda_inv"],
            ascending=[False, False, True, True],
        ).iloc[0]
        oracle = grid_df.sort_values(
            ["avg_ood_acc", "worst_ood_acc", "train_invariance", "max_depth", "lambda_inv"],
            ascending=[False, False, True, True, True],
        ).iloc[0]

        chosen_rows.append(
            {
                "seed": seed,
                "selector": "LOEO",
                "max_depth": int(loeo["max_depth"]),
                "lambda_inv": float(loeo["lambda_inv"]),
                "val_worst": float(loeo["val_worst"]),
                "val_avg": float(loeo["val_avg"]),
                "avg_ood_acc": float(loeo["avg_ood_acc"]),
                "worst_ood_acc": float(loeo["worst_ood_acc"]),
                "train_invariance": float(loeo["train_invariance"]),
            }
        )
        chosen_rows.append(
            {
                "seed": seed,
                "selector": "Oracle",
                "max_depth": int(oracle["max_depth"]),
                "lambda_inv": float(oracle["lambda_inv"]),
                "val_worst": float(oracle["val_worst"]),
                "val_avg": float(oracle["val_avg"]),
                "avg_ood_acc": float(oracle["avg_ood_acc"]),
                "worst_ood_acc": float(oracle["worst_ood_acc"]),
                "train_invariance": float(oracle["train_invariance"]),
            }
        )

    grid_all = pd.concat(grid_rows, ignore_index=True)
    chosen_df = pd.DataFrame(chosen_rows)
    loeo_df = chosen_df[chosen_df["selector"] == "LOEO"].sort_values("seed")
    oracle_df = chosen_df[chosen_df["selector"] == "Oracle"].sort_values("seed")
    gap_df = loeo_df[["seed"]].copy()
    gap_df["loeo_avg_ood"] = loeo_df["avg_ood_acc"].to_numpy()
    gap_df["oracle_avg_ood"] = oracle_df["avg_ood_acc"].to_numpy()
    gap_df["avg_ood_gap"] = gap_df["oracle_avg_ood"] - gap_df["loeo_avg_ood"]
    gap_df["loeo_depth"] = loeo_df["max_depth"].to_numpy()
    gap_df["loeo_lambda"] = loeo_df["lambda_inv"].to_numpy()
    gap_df["oracle_depth"] = oracle_df["max_depth"].to_numpy()
    gap_df["oracle_lambda"] = oracle_df["lambda_inv"].to_numpy()
    ci_lo, ci_hi = bootstrap_ci(gap_df["avg_ood_gap"].to_numpy(), seed=123)
    summary_df = pd.DataFrame(
        [
            {
                "mean_loeo_avg_ood": float(gap_df["loeo_avg_ood"].mean()),
                "mean_oracle_avg_ood": float(gap_df["oracle_avg_ood"].mean()),
                "mean_gap": float(gap_df["avg_ood_gap"].mean()),
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "same_config_rate": float(
                    np.mean(
                        (gap_df["loeo_depth"] == gap_df["oracle_depth"])
                        & (gap_df["loeo_lambda"] == gap_df["oracle_lambda"])
                    )
                ),
            }
        ]
    )

    grid_all.to_csv("experiment/results/selection_gap_grid.csv", index=False)
    chosen_df.to_csv("experiment/results/selection_gap_chosen.csv", index=False)
    gap_df.to_csv("experiment/results/selection_gap_summary.csv", index=False)
    summary_df.to_csv("experiment/results/selection_gap_overview.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(gap_df["oracle_avg_ood"], gap_df["loeo_avg_ood"], s=60)
    lo = min(gap_df["oracle_avg_ood"].min(), gap_df["loeo_avg_ood"].min()) - 0.01
    hi = max(gap_df["oracle_avg_ood"].max(), gap_df["loeo_avg_ood"].max()) + 0.01
    axes[0].plot([lo, hi], [lo, hi], linestyle="--", color="gray")
    axes[0].set_xlabel("Oracle-selected average OOD")
    axes[0].set_ylabel("LOEO-selected average OOD")
    axes[0].set_title("LOEO vs Oracle selection")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(gap_df["seed"].astype(str), gap_df["avg_ood_gap"])
    axes[1].axhline(gap_df["avg_ood_gap"].mean(), color="red", linestyle="--", label="mean gap")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Oracle - LOEO average OOD")
    axes[1].set_title("Selection gap by seed")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("figures/selection_gap.png", dpi=150)
    plt.close(fig)


def run_alpha_size_study():
    seeds = list(range(42, 48))
    depth_grid = [2, 3]
    lambda_grid = [0.0, 1.0, 5.0, 10.0, 20.0, 40.0]
    sample_sizes = [50, 100, 200, 500]
    alpha_grid = [0.0, 0.5, 1.0, 2.0]
    rows = []

    for n_samples in sample_sizes:
        for alpha in alpha_grid:
            print(f"[alpha-size] n={n_samples}, alpha={alpha}")
            for seed in seeds:
                train_envs, test_envs = make_envs(
                    seed=seed,
                    train_corrs=(0.95, 0.80, 0.70),
                    test_corrs=(0.20, 0.10),
                    n_samples=n_samples,
                )
                cfg = loeo_select_ciedt(
                    train_envs,
                    depth_grid=depth_grid,
                    lambda_grid=lambda_grid,
                    laplace_alpha=alpha,
                )
                res = fit_and_eval(
                    train_envs,
                    test_envs,
                    max_depth=cfg["max_depth"],
                    lambda_inv=cfg["lambda_inv"],
                    laplace_alpha=alpha,
                )
                rows.append(
                    {
                        "seed": seed,
                        "n_samples": int(n_samples),
                        "laplace_alpha": float(alpha),
                        "selected_depth": int(cfg["max_depth"]),
                        "selected_lambda": float(cfg["lambda_inv"]),
                        "val_worst": float(cfg["val_worst"]),
                        "val_avg": float(cfg["val_avg"]),
                        "train_acc": res["train_acc"],
                        "avg_ood_acc": res["avg_ood_acc"],
                        "worst_ood_acc": res["worst_ood_acc"],
                        "train_invariance": res["train_invariance"],
                        "runtime_s": res["runtime_s"],
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
            runtime_mean=("runtime_s", "mean"),
        )
    )
    runs_df.to_csv("experiment/results/alpha_size_runs.csv", index=False)
    summary_df.to_csv("experiment/results/alpha_size_summary.csv", index=False)

    pivot_ood = summary_df.pivot(index="n_samples", columns="laplace_alpha", values="avg_ood_mean")
    pivot_inv = summary_df.pivot(index="n_samples", columns="laplace_alpha", values="inv_mean")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    im0 = axes[0].imshow(pivot_ood.to_numpy(), aspect="auto", cmap="viridis")
    axes[0].set_title("Average OOD vs sample size and Laplace alpha")
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

    im1 = axes[1].imshow(pivot_inv.to_numpy(), aspect="auto", cmap="magma_r")
    axes[1].set_title("Training invariance proxy vs sample size and alpha")
    axes[1].set_xlabel("Laplace alpha")
    axes[1].set_ylabel("Samples per environment")
    axes[1].set_xticks(range(len(pivot_inv.columns)))
    axes[1].set_xticklabels([str(x) for x in pivot_inv.columns])
    axes[1].set_yticks(range(len(pivot_inv.index)))
    axes[1].set_yticklabels([str(x) for x in pivot_inv.index])
    for i in range(pivot_inv.shape[0]):
        for j in range(pivot_inv.shape[1]):
            axes[1].text(j, i, f"{pivot_inv.iat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig("figures/alpha_size_heatmaps.png", dpi=150)
    plt.close(fig)


def run_envcount_study():
    seeds = list(range(42, 52))
    depth_grid = [2, 3, 4]
    lambda_grid = [1.0, 5.0, 10.0, 20.0, 40.0]
    env_count_grid = [2, 3, 4, 5]
    rows = []

    for env_count in env_count_grid:
        train_corrs = tuple(np.round(np.linspace(0.95, 0.55, env_count), 2).tolist())
        for seed in seeds:
            train_envs, test_envs = make_envs(
                seed=seed,
                train_corrs=train_corrs,
                test_corrs=(0.20, 0.10),
                n_samples=500,
            )
            ciedt_cfg = loeo_select_ciedt(train_envs, depth_grid, lambda_grid, laplace_alpha=1.0)
            ciedt_res = fit_and_eval(
                train_envs,
                test_envs,
                max_depth=ciedt_cfg["max_depth"],
                lambda_inv=ciedt_cfg["lambda_inv"],
                laplace_alpha=1.0,
            )
            rows.append(
                {
                    "seed": seed,
                    "n_train_envs": env_count,
                    "method": "CIEDT",
                    "selected_depth": int(ciedt_cfg["max_depth"]),
                    "selected_lambda": float(ciedt_cfg["lambda_inv"]),
                    "avg_ood_acc": ciedt_res["avg_ood_acc"],
                    "worst_ood_acc": ciedt_res["worst_ood_acc"],
                    "train_invariance": ciedt_res["train_invariance"],
                    "runtime_s": ciedt_res["runtime_s"],
                }
            )

            odt_cfg = loeo_select_ciedt(train_envs, depth_grid, [0.0], laplace_alpha=1.0)
            odt_res = fit_and_eval(
                train_envs,
                test_envs,
                max_depth=odt_cfg["max_depth"],
                lambda_inv=0.0,
                laplace_alpha=1.0,
            )
            rows.append(
                {
                    "seed": seed,
                    "n_train_envs": env_count,
                    "method": "ODT",
                    "selected_depth": int(odt_cfg["max_depth"]),
                    "selected_lambda": 0.0,
                    "avg_ood_acc": odt_res["avg_ood_acc"],
                    "worst_ood_acc": odt_res["worst_ood_acc"],
                    "train_invariance": odt_res["train_invariance"],
                    "runtime_s": odt_res["runtime_s"],
                }
            )

    runs_df = pd.DataFrame(rows)
    summary_df = (
        runs_df.groupby(["n_train_envs", "method"], as_index=False)
        .agg(
            avg_ood_mean=("avg_ood_acc", "mean"),
            avg_ood_std=("avg_ood_acc", "std"),
            worst_ood_mean=("worst_ood_acc", "mean"),
            inv_mean=("train_invariance", "mean"),
            runtime_mean=("runtime_s", "mean"),
        )
    )

    delta_rows = []
    for env_count in env_count_grid:
        ciedt_scores = runs_df[
            (runs_df["n_train_envs"] == env_count) & (runs_df["method"] == "CIEDT")
        ].sort_values("seed")["avg_ood_acc"].to_numpy()
        odt_scores = runs_df[
            (runs_df["n_train_envs"] == env_count) & (runs_df["method"] == "ODT")
        ].sort_values("seed")["avg_ood_acc"].to_numpy()
        delta = ciedt_scores - odt_scores
        ci_lo, ci_hi = bootstrap_ci(delta, seed=env_count)
        delta_rows.append(
            {
                "n_train_envs": env_count,
                "delta_mean": float(np.mean(delta)),
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "win_rate": float(np.mean(delta > 0)),
            }
        )
    delta_df = pd.DataFrame(delta_rows)

    runs_df.to_csv("experiment/results/envcount_runs.csv", index=False)
    summary_df.to_csv("experiment/results/envcount_summary.csv", index=False)
    delta_df.to_csv("experiment/results/envcount_deltas.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method in ["ODT", "CIEDT"]:
        sub = summary_df[summary_df["method"] == method].sort_values("n_train_envs")
        axes[0].plot(sub["n_train_envs"], sub["avg_ood_mean"], marker="o", label=method)
        axes[0].fill_between(
            sub["n_train_envs"],
            sub["avg_ood_mean"] - sub["avg_ood_std"],
            sub["avg_ood_mean"] + sub["avg_ood_std"],
            alpha=0.15,
        )
    axes[0].set_xlabel("Number of training environments")
    axes[0].set_ylabel("Average OOD accuracy")
    axes[0].set_title("Environment-count sensitivity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(delta_df["n_train_envs"], delta_df["delta_mean"], marker="o", color="tab:red")
    axes[1].fill_between(
        delta_df["n_train_envs"],
        delta_df["ci95_low"],
        delta_df["ci95_high"],
        alpha=0.2,
        color="tab:red",
    )
    axes[1].axhline(0.0, color="gray", linestyle="--")
    axes[1].set_xlabel("Number of training environments")
    axes[1].set_ylabel("CIEDT - ODT average OOD")
    axes[1].set_title("Gain from invariance regularization")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("figures/envcount_sensitivity.png", dpi=150)
    plt.close(fig)


def run_case_study():
    feature_names = ["X1", "X2", "X_spurious", "N1", "N2"]
    seed = 42
    train_envs, test_envs = make_envs(
        seed=seed,
        train_corrs=(0.95, 0.80, 0.70),
        test_corrs=(0.20, 0.10),
        n_samples=500,
    )
    ciedt_cfg = loeo_select_ciedt(train_envs, [2, 3, 4], [1.0, 5.0, 10.0, 20.0, 40.0], laplace_alpha=1.0)
    odt_cfg = loeo_select_ciedt(train_envs, [2, 3, 4], [0.0], laplace_alpha=1.0)
    ciedt_res = fit_and_eval(train_envs, test_envs, ciedt_cfg["max_depth"], ciedt_cfg["lambda_inv"], laplace_alpha=1.0)
    odt_res = fit_and_eval(train_envs, test_envs, odt_cfg["max_depth"], 0.0, laplace_alpha=1.0)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print("Representative synthetic case study")
        print(f"seed={seed}")
        print(f"CIEDT config: depth={ciedt_cfg['max_depth']}, lambda={ciedt_cfg['lambda_inv']}")
        print(
            f"CIEDT avgOOD={ciedt_res['avg_ood_acc']:.3f}, worstOOD={ciedt_res['worst_ood_acc']:.3f}, "
            f"trainInv={ciedt_res['train_invariance']:.3f}, root={feature_names[ciedt_res['root_feature']]}"
        )
        ciedt_res["model"].print_tree()
        print()
        print(f"ODT config: depth={odt_cfg['max_depth']}, lambda=0.0")
        print(
            f"ODT avgOOD={odt_res['avg_ood_acc']:.3f}, worstOOD={odt_res['worst_ood_acc']:.3f}, "
            f"trainInv={odt_res['train_invariance']:.3f}, root={feature_names[odt_res['root_feature']]}"
        )
        odt_res["model"].print_tree()

    with open("experiment/results/synthetic_case_study.txt", "w", encoding="utf-8") as handle:
        handle.write(buffer.getvalue())


def run_all():
    os.makedirs("experiment/results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    run_selection_gap_study()
    run_alpha_size_study()
    run_envcount_study()
    run_case_study()
    print("Saved supplementary outputs under experiment/results and figures.")


if __name__ == "__main__":
    run_all()