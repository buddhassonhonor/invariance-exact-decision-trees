import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine


def _to_binary_features(X_cont):
    med = np.median(X_cont, axis=0)
    return (X_cont > med).astype(int), med


def _top_variance_feature_idx(X_bin, top_k):
    top_k = min(top_k, X_bin.shape[1])
    variances = np.var(X_bin, axis=0)
    order = np.argsort(-variances)
    return np.sort(order[:top_k])


def _inject_spurious(y, match_prob, rng):
    flips = rng.random(len(y)) >= match_prob
    s = y.copy()
    s[flips] = 1 - s[flips]
    return s.astype(int)


def _load_base_dataset(dataset_name):
    if dataset_name == "breast_cancer":
        ds = load_breast_cancer()
        X_cont = ds.data.astype(float)
        y = ds.target.astype(int)
        target_desc = "malignant vs benign"
    elif dataset_name == "wine_binary":
        ds = load_wine()
        X_cont = ds.data.astype(float)
        y = (ds.target == 0).astype(int)
        target_desc = "class-0 vs others"
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    return X_cont, y, target_desc


def generate_shifted_tabular_envs(
    dataset_name,
    seed=42,
    n_samples_per_env=350,
    train_corrs=None,
    test_corrs=None,
    top_k_features=12,
    flip_rate=0.03,
):
    if train_corrs is None:
        train_corrs = [0.95, 0.80, 0.70]
    if test_corrs is None:
        test_corrs = [0.10, 0.30]

    X_cont, y, target_desc = _load_base_dataset(dataset_name)
    X_bin_full, medians = _to_binary_features(X_cont)
    feat_idx = _top_variance_feature_idx(X_bin_full, top_k_features)
    X_bin = X_bin_full[:, feat_idx]

    env_corrs = list(train_corrs) + list(test_corrs)
    envs = []
    for env_id, corr in enumerate(env_corrs):
        rng = np.random.default_rng(seed + 3571 * (env_id + 1))
        sample_idx = rng.choice(len(X_bin), size=n_samples_per_env, replace=True)
        X_sub = X_bin[sample_idx].copy()
        y_sub = y[sample_idx].copy()

        if flip_rate > 0:
            noise_mask = rng.random(X_sub.shape) < flip_rate
            X_sub[noise_mask] = 1 - X_sub[noise_mask]

        spurious = _inject_spurious(y_sub, match_prob=corr, rng=rng)
        X_final = np.column_stack([X_sub, spurious]).astype(int)
        envs.append((X_final, y_sub.astype(int)))

    n_train = len(train_corrs)
    train_envs = envs[:n_train]
    test_envs = envs[n_train:]

    metadata = {
        "dataset_name": dataset_name,
        "target_desc": target_desc,
        "train_corrs": list(train_corrs),
        "test_corrs": list(test_corrs),
        "n_samples_per_env": int(n_samples_per_env),
        "top_k_features": int(top_k_features),
        "selected_feature_indices": feat_idx.tolist(),
        "median_thresholds": medians[feat_idx].tolist(),
        "flip_rate": float(flip_rate),
        "spurious_feature_index": int(top_k_features),
    }
    return train_envs, test_envs, metadata
