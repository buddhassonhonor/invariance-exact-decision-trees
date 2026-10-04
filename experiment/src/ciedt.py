import numpy as np


class CIEDTClassifier:
    def __init__(self, max_depth=3, lambda_inv=0.1, laplace_alpha=1.0):
        self.max_depth = max_depth
        self.lambda_inv = lambda_inv
        self.laplace_alpha = laplace_alpha
        self.tree = None
        self.feature_names = None
        self.n_features = 0
        self.envs = 0

    def fit(self, X_list, y_list):
        self.envs = len(X_list)
        if self.envs == 0:
            self.tree = None
            self.n_features = 0
            return self

        self.n_features = X_list[0].shape[1]
        indices_list = [np.arange(len(X)) for X in X_list]
        self.tree = self._build_tree(X_list, y_list, indices_list, depth=0)
        return self

    def _build_tree(self, X_list, y_list, indices_list, depth):
        leaf_cost, prediction, leaf_invariance = self._calculate_leaf_stats(y_list, indices_list)
        total_samples = sum(len(idx) for idx in indices_list)
        node = {
            "type": "leaf",
            "prediction": prediction,
            "cost": leaf_cost,
            "invariance": leaf_invariance,
            "samples": total_samples,
        }

        if depth >= self.max_depth or total_samples == 0:
            return node

        best_split_cost = float("inf")
        best_split_node = None

        for feat_idx in range(self.n_features):
            left_indices = []
            right_indices = []

            for e in range(self.envs):
                idx = indices_list[e]
                if len(idx) == 0:
                    left_indices.append(np.array([], dtype=int))
                    right_indices.append(np.array([], dtype=int))
                    continue

                feat_vals = X_list[e][idx, feat_idx]
                left_indices.append(idx[feat_vals == 0])
                right_indices.append(idx[feat_vals == 1])

            total_left = sum(len(left) for left in left_indices)
            total_right = sum(len(right) for right in right_indices)
            if total_left == 0 or total_right == 0:
                continue

            left_node = self._build_tree(X_list, y_list, left_indices, depth + 1)
            right_node = self._build_tree(X_list, y_list, right_indices, depth + 1)
            split_cost = left_node["cost"] + right_node["cost"]

            if split_cost < best_split_cost:
                best_split_cost = split_cost
                best_split_node = {
                    "type": "split",
                    "feature": feat_idx,
                    "left": left_node,
                    "right": right_node,
                    "cost": split_cost,
                    "prediction": prediction,
                    "samples": total_samples,
                }

        if best_split_node is not None and best_split_node["cost"] + 1e-12 < leaf_cost:
            return best_split_node
        return node

    def _calculate_leaf_stats(self, y_list, indices_list):
        pooled_chunks = [
            y_list[e][indices_list[e]]
            for e in range(self.envs)
            if len(indices_list[e]) > 0
        ]
        if not pooled_chunks:
            return 0.0, 0, 0.0

        pooled_y = np.concatenate(pooled_chunks)
        counts = np.bincount(pooled_y, minlength=2)
        prediction = int(np.argmax(counts))
        misclass = float(np.sum(pooled_y != prediction))
        total_samples = len(pooled_y)

        probs = []
        for e in range(self.envs):
            idx = indices_list[e]
            n_el = len(idx)
            if n_el == 0 and self.laplace_alpha == 0:
                continue
            y_sub = y_list[e][idx]
            pos = float(np.sum(y_sub == 1))
            alpha = self.laplace_alpha
            denom = n_el + 2.0 * alpha
            p_el = 0.5 if denom == 0 else (pos + alpha) / denom
            probs.append(p_el)

        invariance = 0.0
        if len(probs) > 1:
            invariance = float(total_samples * np.var(np.asarray(probs)))
        total_cost = misclass + self.lambda_inv * invariance
        return total_cost, prediction, invariance

    def predict(self, X):
        if self.tree is None:
            return np.zeros(len(X), dtype=int)

        predictions = []
        for i in range(len(X)):
            node = self.tree
            while node["type"] == "split":
                if X[i, node["feature"]] == 0:
                    node = node["left"]
                else:
                    node = node["right"]
            predictions.append(node["prediction"])
        return np.asarray(predictions, dtype=int)

    def _route_leaf_key(self, x_row):
        node = self.tree
        key = []
        while node["type"] == "split":
            if x_row[node["feature"]] == 0:
                key.append("L")
                node = node["left"]
            else:
                key.append("R")
                node = node["right"]
        return "".join(key)

    def compute_invariance(self, X_list, y_list):
        if self.tree is None:
            return 0.0

        leaf_to_indices = {}
        for e, X in enumerate(X_list):
            for i in range(len(X)):
                key = self._route_leaf_key(X[i])
                if key not in leaf_to_indices:
                    leaf_to_indices[key] = [list() for _ in range(len(X_list))]
                leaf_to_indices[key][e].append(i)

        total_invariance = 0.0
        for key in leaf_to_indices:
            indices_list = [
                np.asarray(idx, dtype=int) for idx in leaf_to_indices[key]
            ]
            _, _, leaf_invariance = self._calculate_leaf_stats(y_list, indices_list)
            total_invariance += leaf_invariance
        return float(total_invariance)

    def get_root_feature(self):
        if self.tree is None or self.tree["type"] != "split":
            return None
        return int(self.tree["feature"])

    def get_split_feature_counts(self):
        counts = {}

        def _walk(node):
            if node["type"] != "split":
                return
            feat = int(node["feature"])
            counts[feat] = counts.get(feat, 0) + 1
            _walk(node["left"])
            _walk(node["right"])

        if self.tree is not None:
            _walk(self.tree)
        total_splits = int(sum(counts.values()))
        return counts, total_splits

    def get_used_features(self):
        counts, _ = self.get_split_feature_counts()
        return set(counts.keys())

    def print_tree(self, node=None, indent=""):
        if self.tree is None:
            print("Empty tree")
            return

        if node is None:
            node = self.tree

        if node["type"] == "leaf":
            print(
                f"{indent}Leaf: Pred={node['prediction']}, Cost={node['cost']:.4f}, "
                f"Inv={node.get('invariance', 0.0):.4f}, Samples={node['samples']}"
            )
            return

        print(
            f"{indent}Split: Feat={node['feature']}, Cost={node['cost']:.4f}, "
            f"Samples={node['samples']}"
        )
        print(f"{indent}  Left (0):")
        self.print_tree(node["left"], indent + "    ")
        print(f"{indent}  Right (1):")
        self.print_tree(node["right"], indent + "    ")
