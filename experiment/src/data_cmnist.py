from sklearn.datasets import load_digits
import numpy as np


def _select_top_variance_features(X, top_k):
    variances = np.var(X, axis=0)
    order = np.argsort(-variances)
    chosen = order[:top_k]
    return np.sort(chosen)


def _inject_spurious(y, match_prob, rng):
    flips = rng.random(len(y)) >= match_prob
    spurious = y.copy()
    spurious[flips] = 1 - spurious[flips]
    return spurious.astype(int)


def generate_cmnist_tabular(
    n_samples_per_env=400,
    env_corrs=None,
    seed=42,
    top_k_pixels=12,
    pixel_threshold=8,
    pixel_flip_rate=0.10,
):
    if env_corrs is None:
        env_corrs = [0.95, 0.80, 0.70, 0.10, 0.30]

    digits = load_digits()
    mask = (digits.target == 0) | (digits.target == 1)
    X_all = digits.data[mask]
    y_all = digits.target[mask].astype(int)
    X_all = (X_all > pixel_threshold).astype(int)

    feature_idx = _select_top_variance_features(X_all, top_k_pixels)
    X_core = X_all[:, feature_idx]

    envs = []
    for env_id, corr in enumerate(env_corrs):
        rng = np.random.default_rng(seed + 7919 * (env_id + 1))
        sample_idx = rng.choice(len(X_core), size=n_samples_per_env, replace=True)

        X_sub = X_core[sample_idx].copy()
        y_sub = y_all[sample_idx].copy()

        if pixel_flip_rate > 0:
            noise_mask = rng.random(X_sub.shape) < pixel_flip_rate
            X_sub[noise_mask] = 1 - X_sub[noise_mask]

        color_like = _inject_spurious(y_sub, match_prob=corr, rng=rng)
        X_final = np.column_stack([X_sub, color_like])
        envs.append((X_final.astype(int), y_sub.astype(int)))

    metadata = {
        "task": "digits-0-vs-1",
        "pixel_threshold": pixel_threshold,
        "selected_pixels": feature_idx.tolist(),
        "top_k_pixels": int(top_k_pixels),
        "pixel_flip_rate": float(pixel_flip_rate),
        "env_corrs": list(env_corrs),
        "spurious_feature_index": int(top_k_pixels),
    }
    return envs, metadata
