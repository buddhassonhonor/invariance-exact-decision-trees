import itertools
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from ciedt import CIEDTClassifier
from data_cmnist import generate_cmnist_tabular
from data_gen import generate_spurious_data
from data_tabular_shift import generate_shifted_tabular_envs


BASE_FONT = 18
plt.rcParams.update(
    {
        "font.size": BASE_FONT,
        "axes.titlesize": BASE_FONT,
        "axes.labelsize": BASE_FONT,
        "xtick.labelsize": BASE_FONT,
        "ytick.labelsize": BASE_FONT,
        "legend.fontsize": BASE_FONT,
    }
)


def pool_envs(envs):
    X = np.vstack([x for x, _ in envs])
    y = np.concatenate([y for _, y in envs])
    return X, y


def bootstrap_ci(values, n_boot=5000, alpha=0.05, seed=0):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(np.mean(sample))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1.0 - alpha / 2))
    return lo, hi


def pairwise_jaccard_mean(feature_sets):
    if len(feature_sets) <= 1:
        return 1.0
    vals = []
    for a, b in itertools.combinations(feature_sets, 2):
        union = len(a | b)
        inter = len(a & b)
        vals.append(1.0 if union == 0 else inter / union)
    return float(np.mean(vals))


def make_synthetic_envs(
    seed,
    train_corrs=(0.95, 0.80, 0.70),
    test_corrs=(0.20, 0.10),
    n_samples=500,
    core_accuracy=0.85,
):
    train_envs = []
    test_envs = []
    for i, corr in enumerate(train_corrs):
        X, y, _ = generate_spurious_data(
            n_samples, corr, core_accuracy=core_accuracy, seed=seed + 101 * (i + 1)
        )
        train_envs.append((X, y))
    for i, corr in enumerate(test_corrs):
        X, y, _ = generate_spurious_data(
            n_samples, corr, core_accuracy=core_accuracy, seed=seed + 1001 * (i + 1)
        )
        test_envs.append((X, y))
    metadata = {
        "dataset_name": "synthetic_xor",
        "train_corrs": list(train_corrs),
        "test_corrs": list(test_corrs),
        "n_samples_per_env": int(n_samples),
        "core_features": [0, 1],
        "spurious_feature_index": 2,
    }
    return train_envs, test_envs, metadata


def evaluate_on_envs(model, test_envs):
    scores = []
    for X, y in test_envs:
        scores.append(float(accuracy_score(y, model.predict(X))))
    return float(np.mean(scores)), float(np.min(scores)), scores


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
                fold_scores.append(float(accuracy_score(y_val, clf.predict(X_val))))
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


def loeo_select_cart(train_envs):
    depth_grid = [2, 3, 4]
    min_leaf_grid = [1, 10, 20]
    ccp_grid = [0.0, 0.001, 0.005]
    class_weight_grid = [None, "balanced"]
    m = len(train_envs)
    best = None

    for depth in depth_grid:
        for min_leaf in min_leaf_grid:
            for ccp in ccp_grid:
                for cw in class_weight_grid:
                    fold_scores = []
                    for val_idx in range(m):
                        tr_envs = [train_envs[i] for i in range(m) if i != val_idx]
                        X_tr, y_tr = pool_envs(tr_envs)
                        X_val, y_val = train_envs[val_idx]
                        clf = DecisionTreeClassifier(
                            max_depth=depth,
                            min_samples_leaf=min_leaf,
                            ccp_alpha=ccp,
                            class_weight=cw,
                            random_state=0,
                        )
                        clf.fit(X_tr, y_tr)
                        fold_scores.append(float(accuracy_score(y_val, clf.predict(X_val))))
                    score_worst = float(np.min(fold_scores))
                    score_avg = float(np.mean(fold_scores))
                    cw_flag = 0 if cw is None else -1
                    key = (score_worst, score_avg, -depth, -min_leaf, -ccp, cw_flag)
                    if best is None or key > best["key"]:
                        best = {
                            "max_depth": int(depth),
                            "min_samples_leaf": int(min_leaf),
                            "ccp_alpha": float(ccp),
                            "class_weight": cw,
                            "val_worst": score_worst,
                            "val_avg": score_avg,
                            "key": key,
                        }
    return best


def sklearn_tree_invariance_proxy(model, X_list, y_list, alpha=1.0):
    leaf_to_env = {}
    m = len(X_list)
    for e in range(m):
        leaf_ids = model.apply(X_list[e])
        y_e = y_list[e]
        for leaf in np.unique(leaf_ids):
            mask = leaf_ids == leaf
            n_el = int(np.sum(mask))
            pos_el = float(np.sum(y_e[mask] == 1))
            if leaf not in leaf_to_env:
                leaf_to_env[leaf] = [{"n": 0, "pos": 0.0} for _ in range(m)]
            leaf_to_env[leaf][e]["n"] = n_el
            leaf_to_env[leaf][e]["pos"] = pos_el

    total = 0.0
    for leaf, stats in leaf_to_env.items():
        probs = []
        N = 0
        for e in range(m):
            n_el = stats[e]["n"]
            pos = stats[e]["pos"]
            N += n_el
            if n_el == 0 and alpha == 0:
                continue
            denom = n_el + 2.0 * alpha
            p = 0.5 if denom == 0 else (pos + alpha) / denom
            probs.append(p)
        if len(probs) > 1:
            total += N * np.var(np.asarray(probs))
    return float(total)


def sklearn_tree_structure_stats(model, spurious_feature_idx):
    feats = np.asarray(model.tree_.feature)
    split_feats = feats[feats >= 0]
    if len(split_feats) == 0:
        return None, 0.0, set()
    root = int(split_feats[0])
    spurious_ratio = float(np.mean(split_feats == spurious_feature_idx))
    used = set(int(v) for v in split_feats.tolist())
    return root, spurious_ratio, used


def ciedt_structure_stats(model, spurious_feature_idx):
    root = model.get_root_feature()
    counts, total_splits = model.get_split_feature_counts()
    spurious_ratio = (
        0.0
        if total_splits == 0
        else float(counts.get(spurious_feature_idx, 0) / total_splits)
    )
    used = model.get_used_features()
    return root, spurious_ratio, used


def run_revision_suite():
    os.makedirs("experiment/results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    seeds = [42, 43, 44, 45, 46]
    depth_grid = [2, 3]
    lambda_grid = [1.0, 5.0, 10.0, 20.0, 40.0]

    all_runs = []
    env_cfg_rows = []
    shift_rows = []
    stability_rows = []

    datasets = ["synthetic_xor", "cmnist_tabular", "breast_cancer", "wine_binary"]

    for dataset_name in datasets:
        print(f"[Dataset] {dataset_name}")
        ciedt_feature_sets = []
        for seed in seeds:
            if dataset_name == "synthetic_xor":
                train_envs, test_envs, metadata = make_synthetic_envs(seed=seed)
            elif dataset_name == "cmnist_tabular":
                envs, metadata = generate_cmnist_tabular(seed=seed, top_k_pixels=12)
                train_envs = envs[:3]
                test_envs = envs[3:]
                metadata["dataset_name"] = dataset_name
                metadata["train_corrs"] = metadata["env_corrs"][:3]
                metadata["test_corrs"] = metadata["env_corrs"][3:]
                metadata["n_samples_per_env"] = 400
            elif dataset_name in {"breast_cancer", "wine_binary"}:
                train_envs, test_envs, metadata = generate_shifted_tabular_envs(
                    dataset_name=dataset_name,
                    seed=seed,
                    n_samples_per_env=350,
                    top_k_features=12,
                )
            else:
                raise ValueError(dataset_name)

            if seed == seeds[0]:
                env_cfg_rows.append(
                    {
                        "dataset": dataset_name,
                        "n_train_envs": len(train_envs),
                        "n_test_envs": len(test_envs),
                        "train_corrs": json.dumps(metadata.get("train_corrs", [])),
                        "test_corrs": json.dumps(metadata.get("test_corrs", [])),
                        "n_samples_per_env": metadata.get("n_samples_per_env"),
                        "spurious_feature_index": metadata.get("spurious_feature_index"),
                    }
                )

            spurious_idx = int(metadata["spurious_feature_index"])
            X_train, y_train = pool_envs(train_envs)
            X_test_all, y_test_all = pool_envs(test_envs)
            X_train_list = [x for x, _ in train_envs]
            y_train_list = [y for _, y in train_envs]

            # CART
            t0 = time.time()
            cart = DecisionTreeClassifier(max_depth=3, random_state=seed)
            cart.fit(X_train, y_train)
            cart_time = time.time() - t0
            cart_avg_ood, cart_worst_ood, _ = evaluate_on_envs(cart, test_envs)
            cart_train = float(accuracy_score(y_train, cart.predict(X_train)))
            cart_inv = sklearn_tree_invariance_proxy(
                cart, X_train_list, y_train_list, alpha=1.0
            )
            cart_root, cart_sp_ratio, cart_used = sklearn_tree_structure_stats(
                cart, spurious_idx
            )
            all_runs.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": "CART",
                    "train_acc": cart_train,
                    "avg_ood_acc": cart_avg_ood,
                    "worst_ood_acc": cart_worst_ood,
                    "runtime_s": cart_time,
                    "selected_depth": 3,
                    "selected_lambda": np.nan,
                    "train_invariance": cart_inv,
                }
            )
            stability_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": "CART",
                    "root_feature": cart_root,
                    "spurious_split_ratio": cart_sp_ratio,
                    "uses_spurious_root": int(cart_root == spurious_idx)
                    if cart_root is not None
                    else 0,
                    "used_features_json": json.dumps(sorted(list(cart_used))),
                }
            )

            # Tuned CART
            cart_cfg = loeo_select_cart(train_envs)
            t0 = time.time()
            tuned_cart = DecisionTreeClassifier(
                max_depth=cart_cfg["max_depth"],
                min_samples_leaf=cart_cfg["min_samples_leaf"],
                ccp_alpha=cart_cfg["ccp_alpha"],
                class_weight=cart_cfg["class_weight"],
                random_state=seed,
            )
            tuned_cart.fit(X_train, y_train)
            tuned_time = time.time() - t0
            tuned_avg_ood, tuned_worst_ood, _ = evaluate_on_envs(tuned_cart, test_envs)
            tuned_train = float(accuracy_score(y_train, tuned_cart.predict(X_train)))
            tuned_inv = sklearn_tree_invariance_proxy(
                tuned_cart, X_train_list, y_train_list, alpha=1.0
            )
            tuned_root, tuned_sp_ratio, tuned_used = sklearn_tree_structure_stats(
                tuned_cart, spurious_idx
            )
            all_runs.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": "TunedCART",
                    "train_acc": tuned_train,
                    "avg_ood_acc": tuned_avg_ood,
                    "worst_ood_acc": tuned_worst_ood,
                    "runtime_s": tuned_time,
                    "selected_depth": cart_cfg["max_depth"],
                    "selected_lambda": np.nan,
                    "train_invariance": tuned_inv,
                }
            )
            stability_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": "TunedCART",
                    "root_feature": tuned_root,
                    "spurious_split_ratio": tuned_sp_ratio,
                    "uses_spurious_root": int(tuned_root == spurious_idx)
                    if tuned_root is not None
                    else 0,
                    "used_features_json": json.dumps(sorted(list(tuned_used))),
                }
            )

            # ODT (lambda=0, depth from LOEO)
            odt_cfg = loeo_select_ciedt(
                train_envs, depth_grid=depth_grid, lambda_grid=[0.0], laplace_alpha=1.0
            )
            t0 = time.time()
            odt = CIEDTClassifier(
                max_depth=odt_cfg["max_depth"], lambda_inv=0.0, laplace_alpha=1.0
            )
            odt.fit(X_train_list, y_train_list)
            odt_time = time.time() - t0
            odt_avg_ood, odt_worst_ood, _ = evaluate_on_envs(odt, test_envs)
            odt_train = float(accuracy_score(y_train, odt.predict(X_train)))
            odt_inv = odt.compute_invariance(X_train_list, y_train_list)
            odt_root, odt_sp_ratio, odt_used = ciedt_structure_stats(odt, spurious_idx)
            all_runs.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": "ODT",
                    "train_acc": odt_train,
                    "avg_ood_acc": odt_avg_ood,
                    "worst_ood_acc": odt_worst_ood,
                    "runtime_s": odt_time,
                    "selected_depth": odt_cfg["max_depth"],
                    "selected_lambda": 0.0,
                    "train_invariance": odt_inv,
                }
            )
            stability_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": "ODT",
                    "root_feature": odt_root,
                    "spurious_split_ratio": odt_sp_ratio,
                    "uses_spurious_root": int(odt_root == spurious_idx)
                    if odt_root is not None
                    else 0,
                    "used_features_json": json.dumps(sorted(list(odt_used))),
                }
            )

            # CIEDT (LOEO for lambda and depth)
            ciedt_cfg = loeo_select_ciedt(
                train_envs,
                depth_grid=depth_grid,
                lambda_grid=lambda_grid,
                laplace_alpha=1.0,
            )
            t0 = time.time()
            ciedt = CIEDTClassifier(
                max_depth=ciedt_cfg["max_depth"],
                lambda_inv=ciedt_cfg["lambda_inv"],
                laplace_alpha=1.0,
            )
            ciedt.fit(X_train_list, y_train_list)
            ciedt_time = time.time() - t0
            ciedt_avg_ood, ciedt_worst_ood, _ = evaluate_on_envs(ciedt, test_envs)
            ciedt_train = float(accuracy_score(y_train, ciedt.predict(X_train)))
            ciedt_inv = ciedt.compute_invariance(X_train_list, y_train_list)
            ciedt_root, ciedt_sp_ratio, ciedt_used = ciedt_structure_stats(
                ciedt, spurious_idx
            )
            ciedt_feature_sets.append(ciedt_used)
            all_runs.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": "CIEDT",
                    "train_acc": ciedt_train,
                    "avg_ood_acc": ciedt_avg_ood,
                    "worst_ood_acc": ciedt_worst_ood,
                    "runtime_s": ciedt_time,
                    "selected_depth": ciedt_cfg["max_depth"],
                    "selected_lambda": ciedt_cfg["lambda_inv"],
                    "train_invariance": ciedt_inv,
                }
            )
            stability_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": "CIEDT",
                    "root_feature": ciedt_root,
                    "spurious_split_ratio": ciedt_sp_ratio,
                    "uses_spurious_root": int(ciedt_root == spurious_idx)
                    if ciedt_root is not None
                    else 0,
                    "used_features_json": json.dumps(sorted(list(ciedt_used))),
                }
            )

            # Logistic Regression
            t0 = time.time()
            lr = LogisticRegression(max_iter=1000, random_state=seed)
            lr.fit(X_train, y_train)
            lr_time = time.time() - t0
            lr_avg_ood, lr_worst_ood, _ = evaluate_on_envs(lr, test_envs)
            lr_train = float(accuracy_score(y_train, lr.predict(X_train)))
            all_runs.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": "LR",
                    "train_acc": lr_train,
                    "avg_ood_acc": lr_avg_ood,
                    "worst_ood_acc": lr_worst_ood,
                    "runtime_s": lr_time,
                    "selected_depth": np.nan,
                    "selected_lambda": np.nan,
                    "train_invariance": np.nan,
                }
            )

            # Shift-intensity curve (synthetic only)
            if dataset_name == "synthetic_xor":
                corr_grid = [0.95, 0.80, 0.60, 0.40, 0.20, 0.10]
                for corr in corr_grid:
                    X_shift, y_shift, _ = generate_spurious_data(
                        500,
                        corr,
                        core_accuracy=0.85,
                        seed=seed + int(10000 * corr) + 77,
                    )
                    shift_rows.append(
                        {
                            "seed": seed,
                            "corr": corr,
                            "method": "CART",
                            "acc": float(accuracy_score(y_shift, cart.predict(X_shift))),
                        }
                    )
                    shift_rows.append(
                        {
                            "seed": seed,
                            "corr": corr,
                            "method": "ODT",
                            "acc": float(accuracy_score(y_shift, odt.predict(X_shift))),
                        }
                    )
                    shift_rows.append(
                        {
                            "seed": seed,
                            "corr": corr,
                            "method": "CIEDT",
                            "acc": float(accuracy_score(y_shift, ciedt.predict(X_shift))),
                        }
                    )

            print(
                f"  seed={seed}: CIEDT avgOOD={ciedt_avg_ood:.3f}, "
                f"CART avgOOD={cart_avg_ood:.3f}, ODT avgOOD={odt_avg_ood:.3f}"
            )

        stability_rows.append(
            {
                "dataset": dataset_name,
                "seed": -1,
                "method": "CIEDT",
                "root_feature": np.nan,
                "spurious_split_ratio": np.nan,
                "uses_spurious_root": np.nan,
                "used_features_json": json.dumps({"mean_pairwise_jaccard": pairwise_jaccard_mean(ciedt_feature_sets)}),
            }
        )

    runs_df = pd.DataFrame(all_runs)
    runs_df.to_csv("experiment/results/revision_runs.csv", index=False)

    summary_df = (
        runs_df.groupby(["dataset", "method"], as_index=False)
        .agg(
            train_acc_mean=("train_acc", "mean"),
            train_acc_std=("train_acc", "std"),
            avg_ood_mean=("avg_ood_acc", "mean"),
            avg_ood_std=("avg_ood_acc", "std"),
            worst_ood_mean=("worst_ood_acc", "mean"),
            worst_ood_std=("worst_ood_acc", "std"),
            runtime_mean=("runtime_s", "mean"),
            runtime_std=("runtime_s", "std"),
            train_inv_mean=("train_invariance", "mean"),
            train_inv_std=("train_invariance", "std"),
        )
    )
    summary_df.to_csv("experiment/results/revision_summary.csv", index=False)

    env_cfg_df = pd.DataFrame(env_cfg_rows)
    env_cfg_df.to_csv("experiment/results/environment_config.csv", index=False)

    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv("experiment/results/stability_summary.csv", index=False)

    shift_df = pd.DataFrame(shift_rows)
    shift_df.to_csv("experiment/results/shift_curve.csv", index=False)

    # Pairwise statistics: CIEDT vs baselines
    stats_rows = []
    for dataset_name in runs_df["dataset"].unique():
        ciedt_scores = (
            runs_df[(runs_df["dataset"] == dataset_name) & (runs_df["method"] == "CIEDT")]
            .sort_values("seed")["avg_ood_acc"]
            .to_numpy()
        )
        for baseline in ["CART", "TunedCART", "ODT", "LR"]:
            base_scores = (
                runs_df[
                    (runs_df["dataset"] == dataset_name) & (runs_df["method"] == baseline)
                ]
                .sort_values("seed")["avg_ood_acc"]
                .to_numpy()
            )
            if len(ciedt_scores) == len(base_scores) and len(base_scores) > 0:
                delta = ciedt_scores - base_scores
                ci_lo, ci_hi = bootstrap_ci(delta, n_boot=5000, seed=123)
                stats_rows.append(
                    {
                        "dataset": dataset_name,
                        "comparison": f"CIEDT-{baseline}",
                        "delta_mean": float(np.mean(delta)),
                        "ci95_low": ci_lo,
                        "ci95_high": ci_hi,
                        "win_rate": float(np.mean(delta > 0)),
                    }
                )
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv("experiment/results/revision_stats.csv", index=False)

    # Figure 1: main OOD summary
    plot_df = summary_df.copy()
    methods = ["CART", "TunedCART", "ODT", "CIEDT", "LR"]
    dataset_order = ["synthetic_xor", "cmnist_tabular", "breast_cancer", "wine_binary"]
    x = np.arange(len(dataset_order))
    width = 0.16
    plt.figure(figsize=(12, 6))
    for i, method in enumerate(methods):
        y = []
        yerr = []
        for d in dataset_order:
            sub = plot_df[(plot_df["dataset"] == d) & (plot_df["method"] == method)]
            if len(sub) == 0:
                y.append(np.nan)
                yerr.append(0.0)
            else:
                y.append(float(sub["avg_ood_mean"].iloc[0]))
                yerr.append(float(sub["avg_ood_std"].iloc[0]))
        plt.bar(x + (i - 2) * width, y, width=width, label=method, yerr=yerr, capsize=4)
    plt.xticks(x, ["Synthetic", "CMNIST-tab", "Breast", "Wine"])
    plt.ylabel("Average OOD Accuracy")
    plt.title("Main OOD Results Across Datasets (5 seeds)")
    plt.ylim(0.0, 1.05)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig("figures/revision_main_ood.png", dpi=100)
    plt.close()

    # Figure 2: shift-intensity curve
    if len(shift_df) > 0:
        curve = (
            shift_df.groupby(["corr", "method"], as_index=False)
            .agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"))
            .sort_values("corr", ascending=False)
        )
        plt.figure(figsize=(10, 6))
        for method in ["CART", "ODT", "CIEDT"]:
            sub = curve[curve["method"] == method]
            plt.plot(sub["corr"], sub["acc_mean"], marker="o", label=method)
            plt.fill_between(
                sub["corr"],
                sub["acc_mean"] - sub["acc_std"],
                sub["acc_mean"] + sub["acc_std"],
                alpha=0.15,
            )
        plt.xlabel("Spurious match probability in test environment")
        plt.ylabel("OOD Accuracy")
        plt.title("Shift Intensity Curve on Synthetic XOR (5 seeds)")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/shift_curve.png", dpi=100)
        plt.close()

    # Figure 3: total spurious-feature usage across internal split nodes.
    stab_plot = stability_df[
        (stability_df["seed"] >= 0) & (stability_df["method"].isin(["CART", "ODT", "CIEDT"]))
    ].copy()
    stab_plot["spurious_split_ratio"] = stab_plot["spurious_split_ratio"].astype(float)
    stab_group = (
        stab_plot.groupby(["dataset", "method"], as_index=False)
        .agg(spurious_usage=("spurious_split_ratio", "mean"))
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, method in enumerate(["CART", "ODT", "CIEDT"]):
        vals = []
        for d in dataset_order:
            row = stab_group[(stab_group["dataset"] == d) & (stab_group["method"] == method)]
            vals.append(float(row["spurious_usage"].iloc[0]) if len(row) else np.nan)
        bars = ax.bar(x + (i - 1) * 0.22, vals, width=0.22, label=method)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.008,
                    f"{100 * val:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(["Synthetic", "CMNIST-tab", "Breast", "Wine"], fontsize=13)
    ax.set_ylabel("Spurious split usage", fontsize=14)
    ax.set_ylim(0.0, 0.36)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig("figures/stability_spurious_usage.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:")
    print("  experiment/results/revision_runs.csv")
    print("  experiment/results/revision_summary.csv")
    print("  experiment/results/revision_stats.csv")
    print("  experiment/results/environment_config.csv")
    print("  experiment/results/stability_summary.csv")
    print("  experiment/results/shift_curve.csv")
    print("  figures/revision_main_ood.png")
    print("  figures/shift_curve.png")
    print("  figures/stability_spurious_usage.png")


if __name__ == "__main__":
    run_revision_suite()
