"""Models for the ESWA experiments; the historical solver remains untouched."""
from functools import lru_cache
import numpy as np


class ExactInvariantTree:
    """Exhaustive Bellman search with optional state memoization.

    Counts are aggregated by identical binary vectors, retaining separate
    environment/class counts. This changes storage, not the leaf objective.
    """
    def __init__(self, depth=3, lam=0., alpha=1., greedy=False, memo=True,
                 support_weight=True):
        self.depth, self.lam, self.alpha = depth, lam, alpha
        self.greedy, self.memo, self.support_weight = greedy, memo, support_weight

    def fit(self, envs):
        self.p = envs[0][0].shape[1]
        X = np.vstack([x for x, y in envs]).astype(np.uint64)
        assert self.p < 63 and np.isin(X, [0, 1]).all()
        codes = (X * (np.uint64(1) << np.arange(self.p, dtype=np.uint64))).sum(axis=1)
        self.codes, inverse = np.unique(codes, return_inverse=True)
        self.counts = np.zeros((len(self.codes), len(envs), 2), dtype=np.int64)
        pos = 0
        for e, (x, y) in enumerate(envs):
            np.add.at(self.counts, (inverse[pos:pos+len(y)], e, y.astype(int)), 1)
            pos += len(y)
        self.calls = 0
        self.solve = lru_cache(None)(self._solve) if self.memo else self._solve
        self.tree = self.solve(0, 0, self.depth)
        self.cache_info = self.solve.cache_info()._asdict() if self.memo else {}
        return self

    def _leaf(self, mask, values):
        counts = self.counts[(self.codes & np.uint64(mask)) == values].sum(axis=0)
        n = counts.sum(axis=1)
        positives = counts[:, 1]
        pooled = counts.sum(axis=0)
        denom = n + 2 * self.alpha
        valid = denom > 0
        probs = (positives[valid] + self.alpha) / denom[valid]
        inv = float(np.var(probs)) if len(probs) > 1 else 0.
        inv *= int(n.sum()) if self.support_weight else 1
        cost = float(pooled.min()) + self.lam * inv
        return dict(type='leaf', prediction=int(pooled[1] > pooled[0]),
                    cost=cost, invariance=inv, samples=int(n.sum()),
                    counts=counts.tolist())

    def _solve(self, mask, values, remaining):
        self.calls += 1
        node = self._leaf(mask, values)
        if remaining == 0 or not node['samples']:
            return node
        best, choice = float('inf'), None
        for j in range(self.p):
            bit = 1 << j
            if mask & bit:
                continue
            newmask = mask | bit
            if self.greedy:
                left, right = self._leaf(newmask, values), self._leaf(newmask, values | bit)
            else:
                # Reject pooled-trivial splits before searching successors.
                leftleaf, rightleaf = self._leaf(newmask, values), self._leaf(newmask, values | bit)
                if not leftleaf['samples'] or not rightleaf['samples']:
                    continue
                left, right = self.solve(newmask, values, remaining-1), self.solve(newmask, values | bit, remaining-1)
            if not left['samples'] or not right['samples']:
                continue
            cost = left['cost'] + right['cost']
            if cost < best:
                best, choice = cost, (j, left, right)
        if choice is None or best + 1e-12 >= node['cost']:
            return node
        j, left, right = choice
        if self.greedy:
            bit = 1 << j
            left = self.solve(mask | bit, values, remaining-1)
            right = self.solve(mask | bit, values | bit, remaining-1)
        return dict(type='split', feature=j, left=left, right=right,
                    prediction=node['prediction'], samples=node['samples'],
                    cost=left['cost']+right['cost'])

    def predict(self, X):
        return predict_tree(self.tree, X)


def predict_tree(tree, X):
    out = []
    for x in X:
        n = tree
        while n['type'] == 'split':
            n = n['left'] if x[n['feature']] == 0 else n['right']
        out.append(n['prediction'])
    return np.asarray(out, dtype=int)


def era_split_score(envs, j, boltzmann=0., ridge=1., directional=False):
    """Delise equations (2),(4),(5); g=y, h=1, fixed ridge.

    The author's ERAMSE proxy is twice the equation-(2) gain. A corresponding
    half-scaled Boltzmann parameter gives identical feature rankings.
    Directional scores use equation (8); an unsupported direction contributes 0.
    """
    gains, directions = [], []
    for X, y in envs:
        left, right = y[X[:, j] == 0], y[X[:, j] == 1]
        gain = .5 * (left.sum()**2/(len(left)+ridge)
                     + right.sum()**2/(len(right)+ridge)
                     - y.sum()**2/(len(y)+ridge))
        gains.append(float(gain))
        directions.append(0. if not len(left) or not len(right)
                          else float(np.sign(left.mean()-right.mean())))
    gains = np.asarray(gains)
    z = boltzmann * gains
    weights = np.exp(z-z.max())
    score = float(np.dot(gains, weights)/weights.sum())
    return (float(abs(np.mean(directions))), score) if directional else (score,)


class EraTree:
    """Single-tree binary-regression adaptation of published splitting rules.

    Leaf outputs are pooled proportions, thresholded at .5 (ties -> class 0).
    This is a disclosed adaptation, not the author's full GBDT implementation.
    """
    def __init__(self, depth=3, boltzmann=0., directional=False):
        self.depth, self.boltzmann, self.directional = depth, boltzmann, directional

    def fit(self, envs):
        self.tree = self._build(envs, self.depth)
        return self

    def _build(self, envs, remaining):
        y = np.concatenate([y for x, y in envs])
        node = dict(type='leaf', prediction=int(y.mean() > .5), samples=len(y))
        if not remaining or not len(y) or y.min() == y.max():
            return node
        best, chosen = None, None
        for j in range(envs[0][0].shape[1]):
            left = [(X[X[:,j] == 0], yy[X[:,j] == 0]) for X, yy in envs]
            right = [(X[X[:,j] == 1], yy[X[:,j] == 1]) for X, yy in envs]
            if not sum(len(yy) for X,yy in left) or not sum(len(yy) for X,yy in right):
                continue
            score = era_split_score(envs,j,self.boltzmann,directional=self.directional)
            if best is None or score > best:
                best, chosen = score, (j,left,right)
        if chosen is None or best[0] <= 1e-12:
            return node
        j,left,right = chosen
        return dict(type='split', feature=j, prediction=node['prediction'],
                    samples=len(y), left=self._build(left,remaining-1),
                    right=self._build(right,remaining-1))

    def predict(self, X):
        return predict_tree(self.tree,X)


def sklearn_tree(model):
    t = model.tree_
    def walk(i):
        node = dict(type='leaf', prediction=int(model.classes_[np.argmax(t.value[i])]), samples=int(t.n_node_samples[i]))
        if t.children_left[i] != t.children_right[i]:
            node.update(type='split',feature=int(t.feature[i]),left=walk(t.children_left[i]),right=walk(t.children_right[i]))
        return node
    return walk(0)


def robust_tree(model):
    def walk(n):
        status = model._get_node_status(model.b_value,model.w_value,n)
        pruned,branching,feature,cutoff,leaf,value = status
        if leaf:
            return dict(type='leaf',prediction=int(value))
        if not branching:
            raise ValueError('Missing incumbent tree decision')
        j = list(model._X_col_labels).index(feature)
        assert cutoff == 0, (feature,cutoff)
        return dict(type='split',feature=j,left=walk(2*n),right=walk(2*n+1))
    return walk(1)


def tree_stats(tree, envs, spurious=None, alpha=1.):
    features, depths, leaves = [], [], []
    def walk(n,depth,rows):
        if n['type'] == 'leaf':
            depths.append(depth); leaves.append(rows)
        else:
            j=n['feature'];features.append(j)
            walk(n['left'],depth+1,[(x[x[:,j]==0],y[x[:,j]==0]) for x,y in rows])
            walk(n['right'],depth+1,[(x[x[:,j]==1],y[x[:,j]==1]) for x,y in rows])
    walk(tree,0,envs)
    inv=0.
    for rows in leaves:
        probs=[(y.sum()+alpha)/(len(y)+2*alpha) for x,y in rows if len(y)+2*alpha>0]
        inv+=sum(len(y) for x,y in rows)*np.var(probs)
    return dict(actual_depth=max(depths),leaves=len(leaves),splits=len(features),
                features=sorted(set(features)),train_invariance=float(inv),
                spurious_fraction=None if spurious is None else (features.count(spurious)/len(features) if features else 0.),
                spurious_root=None if spurious is None else int(tree['type']=='split' and tree['feature']==spurious))
